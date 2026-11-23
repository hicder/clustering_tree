//! Token and phrase predicates evaluated to `RowSet`s.

use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, RowBlock, RowId, SegmentId, TermId, for_each_set_bit_256};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_fts::{
    FtsPostingSource, PositionBlockView,
    PostingBlockView, QueryToken, TermRecord,
};
use izzydb_storage::codec::fts_block_key_coord;
use izzydb_storage::{
    RocksSkipCursor, SegmentTableSnapshot, Storage,
};


use super::*;

pub(super) fn exact_query_tokens(qtokens: &[QueryToken], last_as_prefix: bool) -> Vec<String> {
    let slice = if last_as_prefix {
        &qtokens[..qtokens.len() - 1]
    } else {
        qtokens
    };
    slice.iter().map(|t| t.text.clone()).collect()
}

pub(super) fn eval_any_token_predicate(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    qtokens: &[QueryToken],
    prefix_expansions: Option<Vec<(String, TermRecord)>>,
) -> Result<RowSet> {
    let mut exact_tokens = exact_query_tokens(qtokens, prefix_expansions.is_some());
    exact_tokens.sort_unstable();
    exact_tokens.dedup();
    let mut records: Vec<TermRecord> = source
        .multi_get_term_records(attr_id, &exact_tokens)?
        .into_iter()
        .flatten()
        .collect();
    if let Some(exp) = prefix_expansions {
        records.extend(exp.into_iter().map(|(_, rec)| rec));
    }
    eval_any_token_records(storage, snap, &records)
}

pub(super) fn eval_all_tokens_predicate(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    qtokens: &[QueryToken],
    prefix_expansions: Option<Vec<(String, TermRecord)>>,
) -> Result<RowSet> {
    let last_as_prefix = prefix_expansions.is_some();
    let exact_tokens = exact_query_tokens(qtokens, last_as_prefix);
    let prefix_records = prefix_expansions.unwrap_or_default();
    if last_as_prefix && prefix_records.is_empty() {
        return Ok(RowSet::empty());
    }
    let Some(exact_map) = fetch_exact_term_map(source, attr_id, &exact_tokens)? else {
        return Ok(RowSet::empty());
    };
    let exact_recs: Vec<TermRecord> = exact_map.into_values().collect();
    let prefix_recs: Vec<TermRecord> = prefix_records.into_iter().map(|(_, r)| r).collect();
    let blocks = eval_all_tokens_blocks(
        storage,
        snap,
        &exact_recs,
        last_as_prefix.then_some(prefix_recs.as_slice()),
    )?;
    Ok(blocks_to_rowset(&blocks))
}

pub(super) fn eval_token_sequence_predicate(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    qtokens: &[QueryToken],
    prefix_expansions: Option<Vec<(String, TermRecord)>>,
) -> Result<RowSet> {
    if qtokens.len() == 1 {
        let records: Vec<TermRecord> = if let Some(exp) = prefix_expansions {
            exp.into_iter().map(|(_, r)| r).collect()
        } else {
            source
                .multi_get_term_records(attr_id, std::slice::from_ref(&qtokens[0].text))?
                .into_iter()
                .flatten()
                .collect()
        };
        return eval_any_token_records(storage, snap, &records);
    }

    let last_as_prefix = prefix_expansions.is_some();
    let exact_tokens = exact_query_tokens(qtokens, last_as_prefix);
    let prefix_records = prefix_expansions.unwrap_or_default();
    if last_as_prefix && prefix_records.is_empty() {
        return Ok(RowSet::empty());
    }

    let Some(exact_map) = fetch_exact_term_map(source, attr_id, &exact_tokens)? else {
        return Ok(RowSet::empty());
    };
    let exact_recs: Vec<TermRecord> = exact_map.values().cloned().collect();
    let prefix_recs: Vec<TermRecord> = prefix_records.into_iter().map(|(_, r)| r).collect();

    let candidate_blocks = eval_all_tokens_blocks(
        storage,
        snap,
        &exact_recs,
        last_as_prefix.then_some(prefix_recs.as_slice()),
    )?;
    if candidate_blocks.is_empty() {
        return Ok(RowSet::empty());
    }

    let base_pos = qtokens[0].position;
    let rel_positions: Vec<u32> = qtokens.iter().map(|t| t.position - base_pos).collect();
    let exact_end = if last_as_prefix {
        qtokens.len() - 1
    } else {
        qtokens.len()
    };
    let query_term_ids: Vec<TermId> = qtokens[..exact_end]
        .iter()
        .map(|qt| exact_map[&qt.text].term_id)
        .collect();

    let mut unique_term_ids: Vec<TermId> = Vec::with_capacity(exact_end + prefix_recs.len());
    unique_term_ids.extend_from_slice(&query_term_ids);
    unique_term_ids.extend(prefix_recs.iter().map(|pr| pr.term_id));
    unique_term_ids.sort_unstable();
    unique_term_ids.dedup();

    let mut pos_keys: Vec<[u8; 12]> = Vec::with_capacity(unique_term_ids.len());
    let mut matched_rows: Vec<RowId> = Vec::new();
    for (block_id, cand_mask) in candidate_blocks {
        match_block_positions(
            storage,
            block_id,
            &cand_mask,
            &unique_term_ids,
            &query_term_ids,
            &prefix_recs,
            &rel_positions,
            last_as_prefix,
            &mut pos_keys,
            &mut matched_rows,
        )?;
    }
    Ok(RowSet::from_rows(matched_rows))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn match_block_positions(
    storage: &Storage,
    block_id: RowBlock,
    cand_mask: &[u64; 4],
    unique_term_ids: &[TermId],
    query_term_ids: &[TermId],
    prefix_recs: &[TermRecord],
    rel_positions: &[u32],
    last_as_prefix: bool,
    pos_keys: &mut Vec<[u8; 12]>,
    out_rows: &mut Vec<RowId>,
) -> Result<()> {
    pos_keys.clear();
    for &tid in unique_term_ids {
        pos_keys.push(fts_block_key_coord(tid, block_id));
    }
    let pos_slices = storage.multi_get_fts_positions_pinned(pos_keys)?;

    let mut views: Vec<Option<PositionBlockView<'_>>> = vec![None; unique_term_ids.len()];
    for (idx, opt_slice) in pos_slices.iter().enumerate() {
        if let Some(slice) = opt_slice {
            views[idx] = Some(PositionBlockView::parse(slice)?);
        }
    }

    for &tid in query_term_ids {
        let idx = unique_term_ids.binary_search(&tid).unwrap();
        if views[idx].is_none() {
            return Ok(());
        }
    }

    let prefix_view_indices: Vec<usize> = if last_as_prefix {
        prefix_recs
            .iter()
            .filter_map(|pr| {
                let idx = unique_term_ids.binary_search(&pr.term_id).unwrap();
                views[idx].as_ref().map(|_| idx)
            })
            .collect()
    } else {
        Vec::new()
    };
    if last_as_prefix && prefix_view_indices.is_empty() {
        return Ok(());
    }

    let first_view = views[unique_term_ids.binary_search(&query_term_ids[0]).unwrap()]
        .as_ref()
        .unwrap();
    let follow_view_indices: Vec<usize> = query_term_ids[1..]
        .iter()
        .map(|tid| unique_term_ids.binary_search(tid).unwrap())
        .collect();

    let mut match_err: Option<TreeError> = None;
    for_each_set_bit_256(cand_mask, |slot| {
        if match_err.is_some() {
            return;
        }
        let first_positions = match first_view.slot_positions(slot) {
            Ok(Some(p)) => p,
            Ok(None) => return,
            Err(e) => {
                match_err = Some(e);
                return;
            }
        };

        'start: for &p0 in &first_positions {
            for (offset_i, &v_idx) in follow_view_indices.iter().enumerate() {
                let target_pos = p0 + rel_positions[offset_i + 1];
                let pv = views[v_idx].as_ref().unwrap();
                match pv.slot_contains_pos(slot, target_pos) {
                    Ok(true) => {}
                    Ok(false) => continue 'start,
                    Err(e) => {
                        match_err = Some(e);
                        return;
                    }
                }
            }
            if last_as_prefix {
                let target_pos = p0 + *rel_positions.last().unwrap();
                let mut any_prefix_match = false;
                for &p_idx in &prefix_view_indices {
                    let pv = views[p_idx].as_ref().unwrap();
                    match pv.slot_contains_pos(slot, target_pos) {
                        Ok(true) => {
                            any_prefix_match = true;
                            break;
                        }
                        Ok(false) => {}
                        Err(e) => {
                            match_err = Some(e);
                            return;
                        }
                    }
                }
                if !any_prefix_match {
                    continue 'start;
                }
            }
            out_rows.push(block_id.row(slot));
            return;
        }
    });

    if let Some(e) = match_err {
        return Err(e);
    }
    Ok(())
}

pub(super) fn eval_any_token_records(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    records: &[TermRecord],
) -> Result<RowSet> {
    let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    let mut add_block = |block_id: RowBlock, bytes: &[u8]| -> Result<()> {
        let Some(live_mask) = snap.block_live_mask(block_id.seg(), block_id.block()) else {
            let view = PostingBlockView::parse(bytes)?;
            let bm = by_seg.entry(block_id.seg()).or_default();
            let base = block_id.block() << 8;
            view.for_each_slot(|slot| {
                bm.insert(base | (slot as u32));
            });
            return Ok(());
        };
        if live_mask == [0u64; 4] {
            return Ok(());
        }
        let mut mask = [0u64; 4];
        PostingBlockView::parse(bytes)?.or_slots_into_mask(&mut mask);
        for w in 0..4 {
            mask[w] &= live_mask[w];
        }
        if mask != [0u64; 4] {
            let bm = by_seg.entry(block_id.seg()).or_default();
            let base = block_id.block() << 8;
            for_each_set_bit_256(&mask, |slot| {
                bm.insert(base | (slot as u32));
            });
        }
        Ok(())
    };

    for rec in records {
        if rec.stats.doc_freq == 0 {
            continue;
        }
        if let Some((block_id, ref ib_bytes)) = rec.inline_block {
            add_block(block_id, ib_bytes)?;
        } else {
            storage.visit_fts_postings_for_term(rec.term_id, |block_id, bytes| {
                add_block(block_id, bytes)
            })?;
        }
    }
    Ok(RowSet::from_sorted_segs(by_seg.into_iter().collect()))
}

pub(super) fn eval_all_tokens_blocks(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    exact_records: &[TermRecord],
    prefix_records: Option<&[TermRecord]>,
) -> Result<Vec<(RowBlock, [u64; 4])>> {
    if exact_records.is_empty() {
        let Some(precs) = prefix_records else {
            return Ok(Vec::new());
        };
        let mut by_block: BTreeMap<RowBlock, [u64; 4]> = BTreeMap::new();
        for rec in precs {
            if rec.stats.doc_freq == 0 {
                continue;
            }
            if let Some((block_id, ref ib_bytes)) = rec.inline_block {
                let view = PostingBlockView::parse(ib_bytes)?;
                let mask = by_block.entry(block_id).or_insert([0u64; 4]);
                view.or_slots_into_mask(mask);
            } else {
                storage.visit_fts_postings_for_term(rec.term_id, |block_id, bytes| {
                    let view = PostingBlockView::parse(bytes)?;
                    let mask = by_block.entry(block_id).or_insert([0u64; 4]);
                    view.or_slots_into_mask(mask);
                    Ok(())
                })?;
            }
        }
        let mut out = Vec::with_capacity(by_block.len());
        for (bid, mut mask) in by_block {
            if let Some(live) = snap.block_live_mask(bid.seg(), bid.block()) {
                for w in 0..4 {
                    mask[w] &= live[w];
                }
            }
            if mask != [0u64; 4] {
                out.push((bid, mask));
            }
        }
        return Ok(out);
    }

    let mut ordered: Vec<TermRecord> = exact_records.to_vec();
    ordered.sort_unstable_by_key(|rec| rec.stats.doc_freq);

    let raw_cursors = storage.open_fts_skip_cursors(&ordered)?;
    let mut cursors: Vec<(TermId, RocksSkipCursor<'_>)> = Vec::with_capacity(ordered.len());
    for (rec, cur) in ordered.iter().zip(raw_cursors) {
        if cur.current_block_id().is_none() {
            return Ok(Vec::new());
        }
        cursors.push((rec.term_id, cur));
    }

    let mut prefix_cursors: Vec<(TermId, RocksSkipCursor<'_>)> = match prefix_records {
        Some(precs) => {
            let p_raw = storage.open_fts_skip_cursors(precs)?;
            let mut pcs = Vec::with_capacity(precs.len());
            for (rec, cur) in precs.iter().zip(p_raw) {
                if cur.current_block_id().is_some() {
                    pcs.push((rec.term_id, cur));
                }
            }
            if pcs.is_empty() {
                return Ok(Vec::new());
            }
            pcs
        }
        None => Vec::new(),
    };

    let mut results = Vec::new();
    'block_loop: while let Some(mut target) = cursors[0].1.current_block_id() {

        loop {
            let mut all_match = true;
            for (_, cur) in &mut cursors {
                match cur.current_block_id() {
                    None => break 'block_loop,
                    Some(bid) if bid < target => {
                        cur.seek_block(target)?;
                        match cur.current_block_id() {
                            None => break 'block_loop,
                            Some(nb) if nb > target => {
                                target = nb;
                                all_match = false;
                            }
                            _ => {}
                        }
                    }
                    Some(bid) if bid > target => {
                        target = bid;
                        all_match = false;
                    }
                    _ => {}
                }
            }
            if all_match {
                break;
            }
        }

        let mut mask = snap
            .block_live_mask(target.seg(), target.block())
            .unwrap_or([u64::MAX; 4]);
        if mask != [0u64; 4] {
            for (_, cur) in &mut cursors {
                let mut cur_mask = [0u64; 4];
                if let Some(bytes) = cur.current_posting_bytes()? {
                    PostingBlockView::parse(bytes)?.or_slots_into_mask(&mut cur_mask);
                }
                for w in 0..4 {
                    mask[w] &= cur_mask[w];
                }
                if mask == [0u64; 4] {
                    break;
                }
            }
        }

        if mask != [0u64; 4] && prefix_records.is_some() {
            let mut prefix_mask = [0u64; 4];
            for (_, pcur) in &mut prefix_cursors {
                if let Some(bid) = pcur.current_block_id()
                    && bid < target
                {
                    pcur.seek_block(target)?;
                }
                if pcur.current_block_id() == Some(target)
                    && let Some(bytes) = pcur.current_posting_bytes()?
                {
                    PostingBlockView::parse(bytes)?.or_slots_into_mask(&mut prefix_mask);
                }
            }
            for w in 0..4 {
                mask[w] &= prefix_mask[w];
            }
        }

        if mask != [0u64; 4] {
            results.push((target, mask));
        }

        for (_, cur) in &mut cursors {
            cur.next_block()?;
        }
    }

    Ok(results)
}

pub(super) fn blocks_to_rowset(blocks: &[(RowBlock, [u64; 4])]) -> RowSet {
    let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    for &(bid, ref mask) in blocks {
        let bm = by_seg.entry(bid.seg()).or_default();
        let base = bid.block() << 8;
        for_each_set_bit_256(mask, |slot| {
            bm.insert(base | (slot as u32));
        });
    }
    RowSet::from_sorted_segs(by_seg.into_iter().collect())
}

pub(super) fn fetch_exact_term_map(
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    exact_tokens: &[String],
) -> Result<Option<HashMap<String, TermRecord>>> {
    let mut unique_exact: Vec<String> = exact_tokens.to_vec();
    unique_exact.sort_unstable();
    unique_exact.dedup();
    if unique_exact.is_empty() {
        return Ok(Some(HashMap::new()));
    }
    let recs = source.multi_get_term_records(attr_id, &unique_exact)?;
    let mut map = HashMap::with_capacity(unique_exact.len());
    for (tok, opt_rec) in unique_exact.into_iter().zip(recs) {
        let Some(rec) = opt_rec else {
            return Ok(None);
        };
        if rec.stats.doc_freq == 0 {
            return Ok(None);
        }
        map.insert(tok, rec);
    }
    Ok(Some(map))
}
