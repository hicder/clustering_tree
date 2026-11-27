//! FTS invariant verification.

use std::collections::{BTreeMap, HashMap, HashSet};
use roaring::RoaringBitmap;
use izzydb_core::config::{
    AttrId, DocId, RowBlock, RowId, SegmentId, SkipChunk, TermId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{CollectionSchema, ColumnStorage};
use izzydb_fts::{
    AttrFtsStats, DocLenBlock, FtsAnalyzer, FtsChunkView, FtsMeta, INLINE_TERM_MAX_DF,
    PackedPositionBlock, PackedPostingBlock, ReusableBuffer, TermRecord,
    dl_over_tf_lower_bound,
};
use izzydb_storage::codec::{
    FTS_DOC_LEN_TAG,
    decode_doc_len_key, RowView,
};
use izzydb_storage::{
    CF_FTS_POSITIONS,
    CF_FTS_POSTINGS, CF_FTS_SKIPS, Storage,
};


pub(super) fn verify_fts_invariants(
    storage: &Storage,
    schema: &CollectionSchema,
    seg_snap: &izzydb_storage::SegmentTableSnapshot,
    dropped: &RoaringBitmap,
    live_by_seg: &HashMap<SegmentId, RoaringBitmap>,
    live_rows: &HashMap<RowId, (DocId, Vec<u8>)>,
) -> Result<(usize, usize, usize, usize)> {
    if !schema.has_fts() {
        if storage.get_fts_meta_bytes()?.is_some() {
            return Err(TreeError::InvariantViolation(
                "unexpected \\x01fts_meta for schema without FTS".into(),
            ));
        }
        return Ok((0, 0, 0, 0));
    }

    let fts_meta = match storage.get_fts_meta_bytes()? {
        Some(b) => FtsMeta::decode(&b)?,
        None => FtsMeta::default(),
    };

    let fts_entries: Vec<_> = schema
        .fts_entries()
        .filter(|e| e.storage == ColumnStorage::Inline)
        .collect();
    let analyzers: Vec<FtsAnalyzer> = fts_entries
        .iter()
        .map(|e| FtsAnalyzer::new(e.full_text_search.clone().unwrap()))
        .collect();
    let mut buf = ReusableBuffer::new();

    let mut exp_attr_stats: BTreeMap<AttrId, AttrFtsStats> = BTreeMap::new();
    let mut exp_dls: HashMap<(AttrId, RowBlock), HashMap<u8, u32>> = HashMap::new();
    // term_key -> (df, max_tf, min_dl_over_tf, RowBlock -> [(slot, tf)])
    type ExpTerm = (u64, u32, f32, BTreeMap<RowBlock, Vec<(u8, u32)>>);
    let mut exp_terms: HashMap<Vec<u8>, ExpTerm> = HashMap::new();
    // term_key -> RowBlock -> [(slot, positions)]
    type ExpPosMap = HashMap<Vec<u8>, BTreeMap<RowBlock, Vec<(u8, Vec<u32>)>>>;
    let mut exp_positions: ExpPosMap = HashMap::new();
    let mut fts_doc_count = 0usize;

    for (&row, (_, raw_bytes)) in live_rows {
        let view = RowView::parse(raw_bytes)?;
        let rblk = RowBlock::new(row.seg(), row.block_idx());
        let slot = row.slot();
        let mut doc_had_fts = false;

        for (i, entry) in fts_entries.iter().enumerate() {
            let Some(val) = view.decode_field(entry.id)? else {
                continue;
            };
            let Some(analyzed) = analyzers[i].analyze_attribute_value(&val, &mut buf)? else {
                continue;
            };
            doc_had_fts = true;
            let st = exp_attr_stats.entry(entry.id).or_default();
            st.doc_count += 1;
            st.total_tokens += analyzed.doc_len as u64;

            if analyzed.doc_len > 0 {
                exp_dls
                    .entry((entry.id, rblk))
                    .or_default()
                    .insert(slot, analyzed.doc_len);
                let idx_pos = entry.full_text_search.as_ref().unwrap().index_positions;
                for (tok, term) in analyzed.terms {
                    let tkey = izzydb_storage::codec::encode_fts_term_key(entry.id, &tok);
                    let te = exp_terms
                        .entry(tkey.clone())
                        .or_insert_with(|| (0, 0, f32::INFINITY, BTreeMap::new()));
                    te.0 += 1;
                    te.1 = te.1.max(term.tf);
                    te.2 = te.2.min(dl_over_tf_lower_bound(analyzed.doc_len, term.tf));
                    te.3.entry(rblk).or_default().push((slot, term.tf));
                    if idx_pos {
                        exp_positions
                            .entry(tkey)
                            .or_default()
                            .entry(rblk)
                            .or_default()
                            .push((slot, term.positions));
                    }
                }
            }
        }
        if doc_had_fts {
            fts_doc_count += 1;
        }
    }

    for entry in &fts_entries {
        let disk_st = fts_meta
            .per_attr_stats
            .get(&entry.id)
            .copied()
            .unwrap_or_default();
        let exp_st = exp_attr_stats.get(&entry.id).copied().unwrap_or_default();
        if disk_st != exp_st {
            return Err(TreeError::InvariantViolation(format!(
                "FTS attr_stats mismatch for '{}': disk {disk_st:?} != expected {exp_st:?}",
                entry.name
            )));
        }
    }

    // Check doc lens (0x05).
    let mut disk_dls: HashMap<(AttrId, RowBlock), DocLenBlock> = HashMap::new();
    storage.visit_default(&[FTS_DOC_LEN_TAG], |k, v| {
        let (aid, rblk) = decode_doc_len_key(k)?;
        if dropped.contains(rblk.seg()) {
            return Ok(());
        }
        if seg_snap.segments.get(&rblk.seg()).is_some_and(|e| e.is_visible()) {
            disk_dls.insert((aid, rblk), DocLenBlock::decode(v)?);
        }
        Ok(())
    })?;
    for ((aid, rblk), slot_map) in &exp_dls {
        let Some(disk_blk) = disk_dls.get(&(*aid, *rblk)) else {
            return Err(TreeError::InvariantViolation(format!(
                "missing 0x05 DocLenBlock for attr={aid} {rblk:?}"
            )));
        };
        for (&slot, &exp_dl) in slot_map {
            if disk_blk.get(slot) != exp_dl {
                return Err(TreeError::InvariantViolation(format!(
                    "DocLenBlock mismatch at attr={aid} {rblk:?} slot={slot}"
                )));
            }
        }
    }

    // Check FTS terms, unified chunks in cf_fts_skips, and positions (filtered to live slots in visible segments).
    let disk_terms: HashMap<Vec<u8>, Vec<u8>> = storage.all_fts_terms()?.into_iter().collect();
    let mut disk_chunks_by_tid: HashMap<TermId, BTreeMap<SkipChunk, Vec<u8>>> = HashMap::new();
    let mut total_disk_postings = 0usize;
    storage.visit_cf_raw(CF_FTS_SKIPS, |k, v| {
        if k.len() != 12 {
            return Err(TreeError::InvariantViolation("12B skip chunk key expected".into()));
        }
        let tid = u32::from_be_bytes(k[0..4].try_into().unwrap());
        let seg = SegmentId::from_be_bytes(k[4..8].try_into().unwrap());
        let chunk = u32::from_be_bytes(k[8..12].try_into().unwrap());
        if !dropped.contains(seg) && seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            let cv = FtsChunkView::parse(v)?;
            total_disk_postings += cv.count;
            disk_chunks_by_tid
                .entry(tid)
                .or_default()
                .insert(SkipChunk::new(seg, chunk), v.to_vec());
        }
        Ok(())
    })?;

    storage.visit_cf_raw(CF_FTS_POSTINGS, |k, _| {
        if k.len() != 12 {
            return Err(TreeError::InvariantViolation("12B posting key expected".into()));
        }
        let seg = SegmentId::from_be_bytes(k[4..8].try_into().unwrap());
        if !dropped.contains(seg) && seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            return Err(TreeError::InvariantViolation(
                "unexpected visible key in deprecated cf_fts_postings".into(),
            ));
        }
        Ok(())
    })?;

    let mut disk_positions_by_tid: HashMap<TermId, BTreeMap<RowBlock, Vec<u8>>> = HashMap::new();
    let mut total_disk_positions = 0usize;
    storage.visit_cf_raw(CF_FTS_POSITIONS, |k, v| {
        if k.len() != 12 {
            return Err(TreeError::InvariantViolation("12B position key expected".into()));
        }
        let tid = u32::from_be_bytes(k[0..4].try_into().unwrap());
        let seg = SegmentId::from_be_bytes(k[4..8].try_into().unwrap());
        let blk = u32::from_be_bytes(k[8..12].try_into().unwrap());
        if !dropped.contains(seg) && seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            total_disk_positions += 1;
            disk_positions_by_tid
                .entry(tid)
                .or_default()
                .insert(RowBlock::new(seg, blk), v.to_vec());
        }
        Ok(())
    })?;

    let live_slots_for_block = |rblk: RowBlock| -> Vec<u8> {
        let Some(bm) = live_by_seg.get(&rblk.seg()) else {
            return Vec::new();
        };
        let base = rblk.block() * 256;
        bm.range(base..base + 256)
            .map(|l| (l - base) as u8)
            .collect()
    };

    let mut seen_tids = HashSet::new();
    for (tkey, (exp_df, exp_max_tf, exp_min_ratio, exp_blocks)) in &exp_terms {
        let Some(raw_rec) = disk_terms.get(tkey) else {
            return Err(TreeError::InvariantViolation(
                "missing live FTS term in cf_fts_terms".into(),
            ));
        };
        let rec = TermRecord::decode(raw_rec)?;
        if rec.term_id >= fts_meta.next_term_id || !seen_tids.insert(rec.term_id) {
            return Err(TreeError::InvariantViolation(format!(
                "invalid or duplicate term_id {}",
                rec.term_id
            )));
        }
        if rec.stats.doc_freq != *exp_df
            || rec.stats.max_tf < *exp_max_tf
            || rec.stats.min_dl_over_tf > *exp_min_ratio
        {
            return Err(TreeError::InvariantViolation(format!(
                "FTS TermRecord stats mismatch for tid={}: disk {:?} vs expected (df={exp_df}, max_tf>={exp_max_tf}, min_ratio<={exp_min_ratio})",
                rec.term_id, rec.stats
            )));
        }

        // Collect all visible posting blocks for this term (inline or out-of-line).
        let mut live_actual_blocks: BTreeMap<RowBlock, Vec<(u8, u32)>> = BTreeMap::new();
        if let Some((ib_rblk, ref ib_bytes)) = rec.inline_block {
            if *exp_df > INLINE_TERM_MAX_DF as u64 {
                return Err(TreeError::InvariantViolation(format!(
                    "inlined term {} has doc_freq={} > INLINE_TERM_MAX_DF",
                    rec.term_id, exp_df
                )));
            }
            if seg_snap.segments.get(&ib_rblk.seg()).is_some_and(|e| e.is_visible()) {
                let pblk = PackedPostingBlock::decode(ib_bytes)?;
                let base = ib_rblk.block() * 256;
                let live_seg = live_by_seg.get(&ib_rblk.seg());
                let live_entries: Vec<(u8, u32)> = pblk
                    .slots
                    .iter()
                    .zip(pblk.tfs.iter())
                    .filter(|&(&s, _)| live_seg.is_some_and(|bm| bm.contains(base + s as u32)))
                    .map(|(&s, &tf)| (s, tf))
                    .collect();
                if !live_entries.is_empty() {
                    live_actual_blocks.insert(ib_rblk, live_entries);
                }
            }
        } else if let Some(chunks) = disk_chunks_by_tid.get(&rec.term_id) {
            for (&chunk, raw_chunk) in chunks {
                let cv = FtsChunkView::parse(raw_chunk)?;
                let mut exp_chunk_max_tf = 0u32;
                let mut exp_chunk_min_ratio = f32::INFINITY;
                for i in 0..cv.count {
                    let rel = cv.rel_block_id(i);
                    let rblk = chunk.block(rel);
                    let block_max_tf = cv.max_tf(i);
                    let block_min_ratio = cv.min_dl_over_tf(i);
                    exp_chunk_max_tf = exp_chunk_max_tf.max(block_max_tf);
                    exp_chunk_min_ratio = exp_chunk_min_ratio.min(block_min_ratio);

                    let mut pblk = PackedPostingBlock::decode_compact(
                        cv.block_bytes(i),
                        block_max_tf,
                        block_min_ratio,
                    )?;
                    let stored_max_tf = pblk.max_tf;
                    let stored_min_ratio = pblk.min_dl_over_tf;
                    pblk.recompute_bounds_with(|slot| {
                        disk_dls
                            .get(&(
                                u16::from_be_bytes([tkey[0], tkey[1]]),
                                rblk,
                            ))
                            .map_or(0, |d| d.get(slot))
                    });
                    if stored_max_tf < pblk.max_tf || stored_min_ratio > pblk.min_dl_over_tf {
                        return Err(TreeError::InvariantViolation(format!(
                            "FtsChunk block bounds looser/invalid for tid={} {rblk:?}",
                            rec.term_id
                        )));
                    }

                    let base = rblk.block() * 256;
                    let live_seg = live_by_seg.get(&rblk.seg());
                    let live_entries: Vec<(u8, u32)> = pblk
                        .slots
                        .iter()
                        .zip(pblk.tfs.iter())
                        .filter(|&(&s, _)| live_seg.is_some_and(|bm| bm.contains(base + s as u32)))
                        .map(|(&s, &tf)| (s, tf))
                        .collect();
                    if !live_entries.is_empty() {
                        live_actual_blocks.insert(rblk, live_entries);
                    }
                }
                if cv.chunk_max_tf < exp_chunk_max_tf || cv.chunk_min_dl_over_tf > exp_chunk_min_ratio
                {
                    return Err(TreeError::InvariantViolation(format!(
                        "FtsChunk header bounds mismatch for tid={} {chunk:?}",
                        rec.term_id
                    )));
                }
            }
        }

        let mut exp_sorted = exp_blocks.clone();
        for v in exp_sorted.values_mut() {
            v.sort_unstable_by_key(|&(s, _)| s);
        }
        if live_actual_blocks != exp_sorted {
            return Err(TreeError::InvariantViolation(format!(
                "FTS postings (∩ live) mismatch for tid={}",
                rec.term_id
            )));
        }

        if let Some(exp_pos_blocks) = exp_positions.get(tkey) {
            let disk_pos_blocks = disk_positions_by_tid.get(&rec.term_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!(
                    "missing cf_fts_positions for tid={}",
                    rec.term_id
                ))
            })?;
            for (rblk, exp_slots_pos) in exp_pos_blocks {
                let raw_pos = disk_pos_blocks.get(rblk).ok_or_else(|| {
                    TreeError::InvariantViolation(format!(
                        "missing cf_fts_positions block {rblk:?} for tid={}",
                        rec.term_id
                    ))
                })?;
                let pos_blk = PackedPositionBlock::decode(raw_pos)?;
                let mut exp_sorted = exp_slots_pos.clone();
                exp_sorted.sort_unstable_by_key(|(s, _)| *s);
                let live_set: HashSet<u8> = live_slots_for_block(*rblk).into_iter().collect();
                let mut disk_live: Vec<(u8, Vec<u32>)> = pos_blk
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|&(_, s)| live_set.contains(s))
                    .map(|(idx, &s)| (s, pos_blk.positions[idx].clone()))
                    .collect();
                disk_live.sort_unstable_by_key(|(s, _)| *s);
                if disk_live != exp_sorted {
                    return Err(TreeError::InvariantViolation(format!(
                        "cf_fts_positions (∩ live) mismatch at {rblk:?} for tid={}",
                        rec.term_id
                    )));
                }
            }
        }
    }

    Ok((
        fts_doc_count,
        disk_terms.len(),
        total_disk_postings,
        total_disk_positions,
    ))
}
