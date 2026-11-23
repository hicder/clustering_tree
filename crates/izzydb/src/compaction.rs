//! Phase 9: Segment compaction (`Building -> Sealed -> Forwarding -> Retired`), rank-based
//! forwarding remap, ANN vacuum coordination, FTS block consolidation, and range-delete +
//! compaction-filter reclamation.

use std::collections::{BTreeMap, HashMap, HashSet};

use roaring::RoaringBitmap;

use izzydb_core::config::{
    AttrId, ColumnGen, DocId, LocalId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentId, SkipChunk,
    TermId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rowset::Liveness;
use izzydb_core::schema::{ColumnStorage, SchemaEntry};
use izzydb_fts::{
    DocLenBlock, FtsAnalyzer, INLINE_TERM_MAX_DF, PackedPositionBlock, PackedPostingBlock,
    PostingBlockHeader, ReusableBuffer, SkipBlock, TermRecord,
};
use izzydb_storage::codec::{
    RowView, SEG_FLAG_HAS_COLUMNS, SEG_FLAG_HAS_VECTORS, SegPosting, SegmentMeta, SegmentState,
    TAG_FLOAT, TAG_INT, TAG_STRING, TAG_UINT, ZoneEntry, decode_row_entries, detached_key,
    detached_posting_key, encode_fts_term_key, extract_index_keys_for_attr,
    extract_index_keys_from_entries, make_posting_key,
};
use izzydb_storage::column::ColumnBlockBuilder;
use izzydb_storage::{ColumnBlockView, SegmentTableSnapshot, TxContext};

use crate::collection::Collection;
use crate::columns::columnar_inline_attrs;
use crate::detached::all_detached_gens;
use crate::segments::RowResolver;
use crate::tree::{VacuumBudget, VacuumStats, vacuum_ann_storage};

const WRITE_CHUNK_ROWS: usize = 2048;

/// Policy thresholds determining which sealed segments are eligible for compaction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionPolicy {
    /// Minimum ratio `deleted_count / row_count` (`>= 0.30` by default) to trigger compaction.
    pub tombstone_ratio: f64,
    /// Minimum live ratio relative to `seal_rows` (`< 0.25`, i.e., `live < seal_rows / 4`).
    pub small_segment_ratio: f64,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            tombstone_ratio: 0.30,
            small_segment_ratio: 0.25,
        }
    }
}

/// Bounded budget and policy for [`Collection::compact`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CompactionOptions {
    pub max_groups: Option<usize>,
    pub max_bytes: Option<usize>,
    pub policy: CompactionPolicy,
}

impl CompactionOptions {
    pub fn with_max_groups(mut self, max_groups: usize) -> Self {
        self.max_groups = Some(max_groups);
        self
    }

    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = Some(max_bytes);
        self
    }

    pub fn with_policy(mut self, policy: CompactionPolicy) -> Self {
        self.policy = policy;
        self
    }
}

/// Outcome statistics of [`Collection::compact`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactionStats {
    pub groups_merged: usize,
    pub segments_forwarded: usize,
    pub rows_copied: u64,
    pub bytes_copied: u64,
    pub segments_retired: usize,
    pub vacuum_stats: Option<VacuumStats>,
}

/// Crash/fault-injection stages exposed for tests verifying crash recovery between compaction steps.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionHookStage {
    AfterStepABuild(SegmentId),
    AfterStepBCommit(SegmentId),
    AfterStepCVacuum,
}

#[derive(Debug)]
struct CandidateSeg {
    seg: SegmentId,
    live_rows: u32,
    deleted_count: u32,
}

fn select_merge_groups(
    snap: &SegmentTableSnapshot,
    seal_rows: u32,
    policy: CompactionPolicy,
) -> Vec<Vec<CandidateSeg>> {
    let small_limit = ((seal_rows as f64) * policy.small_segment_ratio).ceil() as u32;
    let mut candidates: Vec<CandidateSeg> = Vec::new();
    for (&seg, entry) in &snap.segments {
        if entry.meta.state != SegmentState::Sealed {
            continue;
        }
        let rc = entry.meta.row_count;
        let del = entry.meta.deleted_count;
        let live = rc.saturating_sub(del);
        let high_tombstones = rc > 0 && (del as f64) / (rc as f64) >= policy.tombstone_ratio;
        let small = live < small_limit;
        if high_tombstones || small {
            candidates.push(CandidateSeg {
                seg,
                live_rows: live,
                deleted_count: del,
            });
        }
    }
    candidates.sort_unstable_by_key(|c| c.seg);

    let mut groups: Vec<Vec<CandidateSeg>> = Vec::new();
    let mut cur_group: Vec<CandidateSeg> = Vec::new();
    let mut cur_live: u32 = 0;

    let flush_group = |grp: Vec<CandidateSeg>, out: &mut Vec<Vec<CandidateSeg>>| {
        if grp.is_empty() {
            return;
        }
        if grp.len() == 1 && grp[0].deleted_count == 0 {
            return;
        }
        out.push(grp);
    };

    for c in candidates {
        if !cur_group.is_empty() && cur_live.saturating_add(c.live_rows) > seal_rows {
            flush_group(std::mem::take(&mut cur_group), &mut groups);
            cur_live = 0;
        }
        cur_live = cur_live.saturating_add(c.live_rows);
        cur_group.push(c);
    }
    flush_group(cur_group, &mut groups);
    groups
}

/// Run bounded segment compaction across sealed candidates, forwarding source segments, updating
/// `cf_id_map`, vacuuming ANN leaves (`vacuum_epoch`), and retiring dead forwarding segments.
pub(crate) fn run_compact<F>(
    coll: &mut Collection,
    opts: CompactionOptions,
    hook: &mut F,
) -> Result<CompactionStats>
where
    F: FnMut(CompactionHookStage) -> Result<()>,
{
    if coll
        .storage
        .all_detached_metas()?
        .into_iter()
        .any(|(_, m)| m.pending_gen != 0)
    {
        return Err(TreeError::InvalidConfig(
            "cannot compact while a detached column ingest has pending_gen != 0".into(),
        ));
    }

    let mut stats = CompactionStats::default();
    let snap = coll.storage.segment_snapshot();
    let groups = select_merge_groups(&snap, coll.manifest.seal_rows, opts.policy);

    for group in groups {
        if opts.max_groups.is_some_and(|m| stats.groups_merged >= m) {
            break;
        }
        if opts.max_bytes.is_some_and(|m| stats.bytes_copied >= m as u64) {
            break;
        }

        let total_live: u32 = group.iter().map(|c| c.live_rows).sum();
        if total_live == 0 {
            // All-dead segments need no target segment `T`: mark Forwarding directly so ANN cleanup
            // and retire can drop them cleanly.
            let vacuum_epoch = coll.ann_meta.as_ref().map_or(0, |m| m.vacuum_epoch);
            let mut tx = TxContext::new(
                &coll.storage,
                &mut coll.internal_cache,
                &mut coll.leaf_parent,
                coll.manifest.clone(),
                coll.ann_meta.clone(),
            );
            for c in &group {
                let mut sm = tx.get_seg_meta(c.seg).ok_or_else(|| {
                    TreeError::CorruptedData(format!("missing SegmentMeta for {}", c.seg))
                })?;
                sm.state = SegmentState::Forwarding;
                sm.forward_target = 0;
                sm.forward_base = 0;
                sm.forwarded_at_epoch = vacuum_epoch;
                tx.put_seg_meta(c.seg, sm);
                stats.segments_forwarded += 1;
            }
            let (m, am) = tx.commit()?;
            coll.manifest = m;
            coll.ann_meta = am;
            stats.groups_merged += 1;
            continue;
        }

        let (rows, bytes, target_seg) = compact_one_group(coll, &group, hook)?;
        stats.groups_merged += 1;
        stats.segments_forwarded += group.len();
        stats.rows_copied += rows;
        stats.bytes_copied += bytes;
        let _ = target_seg;
    }

    if stats.segments_forwarded > 0 && coll.ann.is_some() {
        let vac = vacuum_ann_storage(
            &coll.storage,
            coll.ann.as_ref().unwrap(),
            &mut coll.manifest,
            &mut coll.ann_meta,
            &mut coll.internal_cache,
            &mut coll.leaf_parent,
            VacuumBudget::unlimited(),
        )?;
        stats.vacuum_stats = Some(vac);
        hook(CompactionHookStage::AfterStepCVacuum)?;
    }

    stats.segments_retired = retire_eligible_forwarding_segments(coll)?;
    Ok(stats)
}

/// Execute Step A (`Building` target segment `T`) and Step B (atomic commit) for one merge group.
fn compact_one_group<F>(
    coll: &mut Collection,
    group: &[CandidateSeg],
    hook: &mut F,
) -> Result<(u64, u64, SegmentId)>
where
    F: FnMut(CompactionHookStage) -> Result<()>,
{
    let total_live: u32 = group.iter().map(|c| c.live_rows).sum();
    let source_set: HashSet<SegmentId> = group.iter().map(|c| c.seg).collect();

    // Allocate T and persist SegmentState::Building synchronously BEFORE streaming rows.
    let target_seg = coll.manifest.next_segment_id;
    coll.manifest.next_segment_id = target_seg
        .checked_add(1)
        .ok_or_else(|| TreeError::Storage("SegmentId u32 overflow during compaction".into()))?;
    let created_seq = coll.storage.latest_sequence_number();
    {
        let mut tx = TxContext::new(
            &coll.storage,
            &mut coll.internal_cache,
            &mut coll.leaf_parent,
            coll.manifest.clone(),
            coll.ann_meta.clone(),
        );
        let mut building_meta = SegmentMeta::new_open(created_seq);
        building_meta.state = SegmentState::Building;
        building_meta.row_count = total_live;
        tx.manifest_mut().next_segment_id = coll.manifest.next_segment_id;
        tx.put_seg_meta(target_seg, building_meta);
        let (m, am) = tx.commit()?;
        coll.manifest = m;
        coll.ann_meta = am;
    }

    let snap = coll.storage.segment_snapshot();
    let schema = coll.schema.clone();
    let col_attrs = if coll.config.segment.defer_column_build {
        Vec::new()
    } else {
        columnar_inline_attrs(&schema)
    };
    let mut inline_col_slot: HashMap<AttrId, usize> = HashMap::new();
    for (i, &(aid, _)) in col_attrs.iter().enumerate() {
        inline_col_slot.insert(aid, i);
    }

    let detached_entries: Vec<SchemaEntry> = schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
        .cloned()
        .collect();
    let detached_gens = all_detached_gens(&coll.storage)?;

    let fts_entries: Vec<SchemaEntry> = schema.fts_entries().cloned().collect();
    let fts_analyzers: Vec<FtsAnalyzer> = fts_entries
        .iter()
        .map(|e| FtsAnalyzer::new(e.full_text_search.clone().unwrap()))
        .collect();
    let mut fts_buf = ReusableBuffer::new();

    let mut dst_local: LocalId = 0;
    let mut base_by_seg: Vec<(SegmentId, LocalId)> = Vec::with_capacity(group.len());
    let mut id_map_updates: Vec<(DocId, RowId)> = Vec::with_capacity(total_live as usize);
    let mut bytes_copied = 0u64;
    let mut has_any_vector = false;

    // Accumulators for T
    let mut inline_postings: HashMap<Vec<u8>, RoaringBitmap> = HashMap::new();
    let mut detached_postings: Vec<HashMap<Vec<u8>, RoaringBitmap>> =
        vec![HashMap::new(); detached_entries.len()];
    let mut zones: HashMap<(AttrId, u8), ZoneEntry> = HashMap::new();

    // Per-block builders for current 256-row block in T
    let mut cur_blk: u32 = 0;
    let mut doc_id_blk = [0u64; ROW_BLOCK_SIZE];
    let mut col_builders: Vec<ColumnBlockBuilder> = col_attrs
        .iter()
        .map(|&(_, ty)| ColumnBlockBuilder::new(ty))
        .collect();
    let mut det_builders: Vec<ColumnBlockBuilder> = detached_entries
        .iter()
        .map(|e| ColumnBlockBuilder::new(e.attr_type))
        .collect();
    let mut dl_blocks: Vec<(DocLenBlock, bool)> =
        vec![(DocLenBlock::default(), false); fts_entries.len()];

    // Accumulated FTS posting & position blocks keyed by term_key -> block_idx -> ...
    let mut fts_term_postings: BTreeMap<Vec<u8>, BTreeMap<u32, PackedPostingBlock>> =
        BTreeMap::new();
    let mut fts_term_positions: HashMap<(Vec<u8>, u32), PackedPositionBlock> = HashMap::new();
    let mut fts_dl_written: Vec<(AttrId, RowBlock, Vec<u8>)> = Vec::new();

    // Cached source detached blocks for the active source segment
    let mut src_det_cache: Vec<Option<(u32, Option<Vec<u8>>)>> = vec![None; detached_entries.len()];

    let mut stage_batch = coll.storage.new_batch();
    let mut pending_batch_rows = 0usize;

    let flush_current_block = |blk: u32,
                               row_count_in_blk: usize,
                               doc_id_blk: &mut [DocId; ROW_BLOCK_SIZE],
                               col_builders: &mut Vec<ColumnBlockBuilder>,
                               det_builders: &mut Vec<ColumnBlockBuilder>,
                               dl_blocks: &mut [(DocLenBlock, bool)],
                               fts_dl_written: &mut Vec<(AttrId, RowBlock, Vec<u8>)>,
                               stage_batch: &mut izzydb_storage::StorageBatch<'_>| {
        if row_count_in_blk == 0 {
            return;
        }
        stage_batch.put_doc_id_block(target_seg, blk, doc_id_blk);
        *doc_id_blk = [0u64; ROW_BLOCK_SIZE];

        for (&(attr_id, ty), b) in col_attrs.iter().zip(col_builders.iter_mut()) {
            let old_b = std::mem::replace(b, ColumnBlockBuilder::new(ty));
            if old_b.present_count() > 0 {
                stage_batch.put_column_block(
                    &izzydb_storage::codec::column_key(target_seg, attr_id, blk),
                    &old_b.encode(),
                );
            }
        }
        for (entry, b) in detached_entries.iter().zip(det_builders.iter_mut()) {
            let old_b = std::mem::replace(b, ColumnBlockBuilder::new(entry.attr_type));
            if old_b.present_count() > 0 {
                let g = detached_gens.get(&entry.id).copied().unwrap_or(0);
                stage_batch.put_detached_block(
                    &detached_key(entry.id, g, target_seg, blk),
                    &old_b.encode(),
                );
            }
        }
        for (fi, (dl_blk, dirty)) in dl_blocks.iter_mut().enumerate() {
            let old_dl = std::mem::take(dl_blk);
            if *dirty {
                *dirty = false;
                fts_dl_written.push((
                    fts_entries[fi].id,
                    RowBlock::new(target_seg, blk),
                    old_dl.encode(),
                ));
            }
        }
    };

    for cand in group {
        base_by_seg.push((cand.seg, dst_local));
        if cand.live_rows == 0 {
            continue;
        }
        src_det_cache.fill(None);

        let mut src_live_rows: Vec<RowId> = Vec::with_capacity(cand.live_rows as usize);
        let mut src_row_bytes: Vec<Vec<u8>> = Vec::with_capacity(cand.live_rows as usize);
        coll.storage.visit_segment_rows(cand.seg, |row, bytes| {
            if snap.is_live(row) {
                src_live_rows.push(row);
                src_row_bytes.push(bytes.to_vec());
            }
            Ok(())
        })?;
        let src_doc_ids = RowResolver::resolve_all(&coll.storage, &src_live_rows)?;

        let mut vec_src_rows: Vec<(usize, RowId)> = Vec::new();
        for (idx, (&src_row, raw)) in src_live_rows.iter().zip(&src_row_bytes).enumerate() {
            let v = RowView::parse(raw)?;
            if v.has_vector() {
                vec_src_rows.push((idx, src_row));
            }
        }
        let mut vec_bytes_by_idx: Vec<Option<Vec<u8>>> = vec![None; src_live_rows.len()];
        if !vec_src_rows.is_empty() {
            has_any_vector = true;
            let vrows: Vec<RowId> = vec_src_rows.iter().map(|&(_, r)| r).collect();
            let vslices = coll.storage.multi_get_vectors_pinned(&vrows)?;
            for ((idx, row), opt_slice) in vec_src_rows.into_iter().zip(vslices) {
                let s = opt_slice.ok_or_else(|| {
                    TreeError::InvariantViolation(format!(
                        "row {row:?} has_vector=1 missing in cf_vectors during compaction"
                    ))
                })?;
                vec_bytes_by_idx[idx] = Some(s.to_vec());
            }
        }

        for (idx, ((&src_row, doc_id), raw_bytes)) in src_live_rows
            .iter()
            .zip(src_doc_ids)
            .zip(src_row_bytes.iter())
            .enumerate()
        {
            let blk = dst_local >> 8;
            let slot = (dst_local & 0xFF) as u8;
            if blk != cur_blk {
                flush_current_block(
                    cur_blk,
                    ROW_BLOCK_SIZE,
                    &mut doc_id_blk,
                    &mut col_builders,
                    &mut det_builders,
                    &mut dl_blocks,
                    &mut fts_dl_written,
                    &mut stage_batch,
                );
                cur_blk = blk;
                if pending_batch_rows >= WRITE_CHUNK_ROWS {
                    let b = std::mem::replace(&mut stage_batch, coll.storage.new_batch());
                    coll.storage.write_batch(b)?;
                    pending_batch_rows = 0;
                }
            }

            let dst_row = RowId::new(target_seg, dst_local);
            doc_id_blk[slot as usize] = doc_id;
            id_map_updates.push((doc_id, dst_row));
            stage_batch.put_row_bytes(dst_row, raw_bytes);
            bytes_copied += raw_bytes.len() as u64;
            if let Some(ref vbytes) = vec_bytes_by_idx[idx] {
                stage_batch.put_vector_bytes(dst_row, vbytes);
                bytes_copied += vbytes.len() as u64;
            }

            let view = RowView::parse(raw_bytes)?;
            let entries = decode_row_entries(raw_bytes)?;

            // Inline attribute postings + zones
            let keys = extract_index_keys_from_entries(&schema, &entries)?;
            for ak in keys {
                let aid = u16::from_be_bytes([ak[0], ak[1]]);
                let tag = ak[2];
                if ak.len() > 3
                    && matches!(tag, TAG_INT | TAG_UINT | TAG_FLOAT | TAG_STRING)
                    && schema.get_by_id(aid).is_some_and(|e| e.attr_type.is_ordered())
                {
                    let z = zones
                        .entry((aid, tag))
                        .or_insert_with(|| ZoneEntry::singleton(&ak[3..]));
                    z.widen(&ak[3..]);
                }
                inline_postings.entry(ak).or_default().insert(dst_local);
            }

            // Inline cf_columns
            if !col_attrs.is_empty() {
                for i in 0..view.len() {
                    let (aid, _, _, _) = view.dir_entry(i);
                    if let Some(&pos) = inline_col_slot.get(&aid)
                        && let Some(field) = view.get_raw_field(aid)?
                    {
                        col_builders[pos].set_row_field(slot, field)?;
                    }
                }
            }

            // Detached columns carry-over
            for (di, entry) in detached_entries.iter().enumerate() {
                let col_gen = detached_gens.get(&entry.id).copied().unwrap_or(0);
                let src_blk = src_row.block_idx();
                let need_load = src_det_cache[di]
                    .as_ref()
                    .is_none_or(|(b, _)| *b != src_blk);
                if need_load {
                    let loaded = coll
                        .storage
                        .get_detached_block_pinned(entry.id, col_gen, cand.seg, src_blk)?
                        .map(|s| s.to_vec());
                    src_det_cache[di] = Some((src_blk, loaded));
                }
                if let Some((_, Some(ref blk_bytes))) = src_det_cache[di] {
                    let bv = ColumnBlockView::parse(blk_bytes)?;
                    if bv.is_present(src_row.slot()) {
                        let val = bv.decode_attr_value(src_row.slot())?;
                        det_builders[di].set_attr_value(slot, &val)?;
                        if entry.filterable {
                            for ak in extract_index_keys_for_attr(entry.id, &val)? {
                                let tag = ak[2];
                                if ak.len() > 3
                                    && matches!(tag, TAG_INT | TAG_UINT | TAG_FLOAT | TAG_STRING)
                                    && entry.attr_type.is_ordered()
                                {
                                    let z = zones
                                        .entry((entry.id, tag))
                                        .or_insert_with(|| ZoneEntry::singleton(&ak[3..]));
                                    z.widen(&ak[3..]);
                                }
                                detached_postings[di]
                                    .entry(ak)
                                    .or_default()
                                    .insert(dst_local);
                            }
                        }
                    }
                }
            }

            // FTS re-encoding into exact T blocks
            for (fi, entry) in fts_entries.iter().enumerate() {
                if entry.storage != ColumnStorage::Inline {
                    continue;
                }
                let Some(val) = view.decode_field(entry.id)? else {
                    continue;
                };
                let Some(analyzed) = fts_analyzers[fi].analyze_attribute_value(&val, &mut fts_buf)?
                else {
                    continue;
                };
                dl_blocks[fi].0.set(slot, analyzed.doc_len);
                dl_blocks[fi].1 = true;

                let index_pos = entry
                    .full_text_search
                    .as_ref()
                    .is_some_and(|c| c.index_positions);
                for (tok, term) in &analyzed.terms {
                    let tk = encode_fts_term_key(entry.id, tok);
                    let pblk = fts_term_postings
                        .entry(tk.clone())
                        .or_default()
                        .entry(blk)
                        .or_default();
                    pblk.upsert(slot, term.tf, &dl_blocks[fi].0);
                    if index_pos {
                        let pos_blk = fts_term_positions.entry((tk, blk)).or_default();
                        pos_blk.upsert(slot, term.positions.clone());
                    }
                }
            }

            dst_local += 1;
            pending_batch_rows += 1;
        }
    }

    // Flush final partial block and any buffered postings before Step A hook.
    let rem_in_blk = (dst_local as usize) % ROW_BLOCK_SIZE;
    let last_count = if dst_local > 0 && rem_in_blk == 0 {
        ROW_BLOCK_SIZE
    } else {
        rem_in_blk
    };
    flush_current_block(
        cur_blk,
        last_count,
        &mut doc_id_blk,
        &mut col_builders,
        &mut det_builders,
        &mut dl_blocks,
        &mut fts_dl_written,
        &mut stage_batch,
    );
    for (ak, bm) in inline_postings {
        stage_batch.put_attr_posting(&make_posting_key(&ak, target_seg), &SegPosting::from_bitmap(bm).encode()?);
    }
    for (entry, pmap) in detached_entries.iter().zip(detached_postings) {
        let col_gen = detached_gens.get(&entry.id).copied().unwrap_or(0);
        for (ak, bm) in pmap {
            let pkey = detached_posting_key(entry.id, col_gen, ak[2], &ak[3..], target_seg);
            stage_batch.put_detached_posting(&pkey, &SegPosting::from_bitmap(bm).encode()?);
        }
    }
    coll.storage.write_batch_sync(stage_batch)?;

    hook(CompactionHookStage::AfterStepABuild(target_seg))?;

    // Step B — short atomic commit flipping T -> Sealed, S -> Forwarding, cf_id_map, zones, FTS terms.
    let vacuum_epoch = coll.ann_meta.as_ref().map_or(0, |m| m.vacuum_epoch);
    let fts_keys: Vec<Vec<u8>> = fts_term_postings.keys().cloned().collect();
    let existing_term_bytes: Vec<Option<Vec<u8>>> = if fts_keys.is_empty() {
        Vec::new()
    } else {
        let mut out = Vec::with_capacity(fts_keys.len());
        coll.storage
            .multi_get_fts_term_bytes_sorted(&fts_keys, |opt| {
                out.push(opt.map(|s| s.to_vec()));
                Ok(())
            })?;
        out
    };

    let mut tx = TxContext::new(
        &coll.storage,
        &mut coll.internal_cache,
        &mut coll.leaf_parent,
        coll.manifest.clone(),
        coll.ann_meta.clone(),
    );
    let mut sealed_meta = SegmentMeta::new_open(created_seq);
    sealed_meta.state = SegmentState::Sealed;
    sealed_meta.sealed_seq = coll.storage.latest_sequence_number();
    sealed_meta.row_count = dst_local;
    sealed_meta.flags = if has_any_vector {
        SEG_FLAG_HAS_VECTORS
    } else {
        0
    } | if !coll.config.segment.defer_column_build {
        SEG_FLAG_HAS_COLUMNS
    } else {
        0
    };
    tx.put_seg_meta(target_seg, sealed_meta);

    for ((cand_idx, cand), &(_, base)) in group.iter().enumerate().zip(base_by_seg.iter()) {
        let _ = cand_idx;
        let mut sm = tx.get_seg_meta(cand.seg).ok_or_else(|| {
            TreeError::CorruptedData(format!("missing SegmentMeta for {}", cand.seg))
        })?;
        sm.state = SegmentState::Forwarding;
        sm.forward_target = target_seg;
        sm.forward_base = base;
        sm.forwarded_at_epoch = vacuum_epoch;
        tx.put_seg_meta(cand.seg, sm);
    }

    for (doc_id, new_row) in id_map_updates {
        tx.put_id_map(doc_id, new_row);
    }
    for ((aid, tag), z) in zones {
        tx.set_zone_entry(target_seg, aid, tag, z);
    }
    for (aid, rblk, bytes) in fts_dl_written {
        tx.put_fts_doc_len_bytes(aid, rblk, bytes);
    }

    for ((tk, blk_map), opt_rec) in fts_term_postings.into_iter().zip(existing_term_bytes) {
        let Some(rec_bytes) = opt_rec else {
            return Err(TreeError::InvariantViolation(format!(
                "missing TermRecord for compacted term key {tk:?}"
            )));
        };
        let mut rec = TermRecord::decode(&rec_bytes)?;
        let tid: TermId = rec.term_id;

        let mut encoded_blocks: Vec<(u32, Vec<u8>)> = Vec::with_capacity(blk_map.len());
        for (blk, pblk) in blk_map {
            encoded_blocks.push((blk, pblk.encode()));
        }

        let mut keep_inline = false;
        if let Some((ib_id, ref ib_bytes)) = rec.inline_block {
            if source_set.contains(&ib_id.seg()) {
                if encoded_blocks.len() == 1 {
                    let (only_blk, ref only_bytes) = encoded_blocks[0];
                    let hdr = PostingBlockHeader::parse(only_bytes)?;
                    if (hdr.count as usize) <= INLINE_TERM_MAX_DF {
                        rec.inline_block =
                            Some((RowBlock::new(target_seg, only_blk), only_bytes.clone()));
                        tx.put_fts_term_bytes(tk.clone(), rec.encode());
                        keep_inline = true;
                    }
                }
                if !keep_inline {
                    rec.inline_block = None;
                    tx.put_fts_term_bytes(tk.clone(), rec.encode());
                }
            } else {
                // Inline block lives in a non-compacted segment: materialize it to cf_fts_postings + skips.
                let hdr = PostingBlockHeader::parse(ib_bytes)?;
                tx.put_fts_posting_bytes(tid, ib_id, ib_bytes.clone());
                let mut sb = match tx.get_fts_skip_bytes(tid, ib_id.chunk())? {
                    Some(b) => SkipBlock::decode(&b)?,
                    None => SkipBlock::default(),
                };
                sb.upsert(ib_id.rel_block(), hdr.max_tf, hdr.min_dl_over_tf);
                tx.put_fts_skip_bytes(tid, ib_id.chunk(), sb.encode());
                rec.inline_block = None;
                tx.put_fts_term_bytes(tk.clone(), rec.encode());
            }
        }

        if !keep_inline {
            let mut by_chunk: BTreeMap<SkipChunk, Vec<(u8, u32, f32)>> = BTreeMap::new();
            for &(blk, ref bytes) in &encoded_blocks {
                let rb = RowBlock::new(target_seg, blk);
                let hdr = PostingBlockHeader::parse(bytes)?;
                tx.put_fts_posting_bytes(tid, rb, bytes.clone());
                by_chunk.entry(rb.chunk()).or_default().push((
                    rb.rel_block(),
                    hdr.max_tf,
                    hdr.min_dl_over_tf,
                ));
            }
            for (chunk, entries) in by_chunk {
                let mut sb = match tx.get_fts_skip_bytes(tid, chunk)? {
                    Some(b) => SkipBlock::decode(&b)?,
                    None => SkipBlock::default(),
                };
                for (rel, mtf, min_r) in entries {
                    sb.upsert(rel, mtf, min_r);
                }
                tx.put_fts_skip_bytes(tid, chunk, sb.encode());
            }
        }

        for (blk, _) in encoded_blocks {
            if let Some(pblk) = fts_term_positions.remove(&(tk.clone(), blk)) {
                tx.put_fts_position_bytes(tid, RowBlock::new(target_seg, blk), pblk.encode());
            }
        }
    }

    let (m, am) = tx.commit()?;
    coll.manifest = m;
    coll.ann_meta = am;

    hook(CompactionHookStage::AfterStepBCommit(target_seg))?;

    Ok((dst_local as u64, bytes_copied, target_seg))
}

/// Retire all `Forwarding` segments whose forwarding epoch is strictly older than
/// `ann_meta.vacuum_epoch` (or any `Forwarding` segment if the collection has no ANN index)
/// and that are not referenced as `forward_target` by another remaining `Forwarding` segment.
pub(crate) fn retire_eligible_forwarding_segments(coll: &mut Collection) -> Result<usize> {
    let snap = coll.storage.segment_snapshot();
    let cur_epoch = coll.ann_meta.as_ref().map(|m| m.vacuum_epoch);

    let forwarding_targets: HashSet<SegmentId> = snap
        .segments
        .values()
        .filter(|e| e.meta.state == SegmentState::Forwarding)
        .map(|e| e.meta.forward_target)
        .collect();

    let eligible: Vec<SegmentId> = snap
        .segments
        .iter()
        .filter_map(|(&seg, e)| {
            if e.meta.state != SegmentState::Forwarding {
                return None;
            }
            if forwarding_targets.contains(&seg) {
                return None;
            }
            let epoch_ok = match cur_epoch {
                Some(ep) => ep > e.meta.forwarded_at_epoch,
                None => true,
            };
            epoch_ok.then_some(seg)
        })
        .collect();

    if eligible.is_empty() {
        return Ok(0);
    }

    let det_gens: Vec<(AttrId, ColumnGen)> = all_detached_gens(&coll.storage)?
        .into_iter()
        .collect();
    let mut all_det: Vec<(AttrId, ColumnGen)> = Vec::new();
    for entry in coll
        .schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
    {
        let g = det_gens
            .iter()
            .find(|&&(id, _)| id == entry.id)
            .map_or(0, |&(_, g)| g);
        all_det.push((entry.id, g));
    }
    let fts_attrs: Vec<AttrId> = coll.schema.fts_entries().map(|e| e.id).collect();

    let mut tx = TxContext::new(
        &coll.storage,
        &mut coll.internal_cache,
        &mut coll.leaf_parent,
        coll.manifest.clone(),
        coll.ann_meta.clone(),
    );
    for &seg in &eligible {
        tx.retire_segment(seg, all_det.clone(), fts_attrs.clone());
    }
    let (m, am) = tx.commit()?;
    coll.manifest = m;
    coll.ann_meta = am;
    Ok(eligible.len())
}
