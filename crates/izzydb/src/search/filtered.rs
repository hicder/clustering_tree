//! Filtered ANN search regimes.

use std::collections::{BinaryHeap, HashMap};
use izzydb_core::config::{DistanceMetric, NodeId, RowId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::metric::{prepare_vector_cow, search_distance, search_distance_bytes};
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::CollectionSchema;
use izzydb_rabitq::{
    QueryScratch, RabitqQuantizer,
    pack_query_bits_into,
};
use izzydb_storage::codec::RowView;
use izzydb_storage::node::{AnnMeta, LeafNodeView, PackedInternalNode};
use izzydb_storage::{SegmentTableSnapshot, Storage};
use crate::filter_plan::{self, ColumnFilter, FilterPlan, FilterPlanTrace, FilteredSearchRegime};
use crate::fts_index::FtsIndex;
use crate::simd::LeafCandidate;

use super::postings::PostingMembership;
use super::*;

/// Filtered approximate nearest neighbor search using the v3 `RowSet` planner:
/// - **Exact scan**: materialized sets with `<= exact_scan_threshold` rows
/// - **Masked traversal**: materialized sets with selectivity `<= 0.5`
/// - **Post-filter**: `Lazy` plans or high-selectivity (`> 0.5`) filters
#[allow(clippy::too_many_arguments)]
pub fn search_filtered(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &HashMap<NodeId, NodeId>,
    meta: &AnnMeta,
    quantizer: &RabitqQuantizer,
    schema: &CollectionSchema,
    fts_index: &FtsIndex,
    query: &[f32],
    opts: SearchOptions,
    filter: &Filter,
    config: &TreeConfig,
) -> Result<Vec<(RowId, f32)>> {
    let mut trace = FilterPlanTrace::default();
    search_filtered_with_trace(
        storage,
        internal_cache,
        leaf_parent,
        meta,
        quantizer,
        schema,
        fts_index,
        query,
        opts,
        filter,
        config,
        &mut trace,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn search_filtered_with_trace(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &HashMap<NodeId, NodeId>,
    meta: &AnnMeta,
    quantizer: &RabitqQuantizer,
    schema: &CollectionSchema,
    fts_index: &FtsIndex,
    query: &[f32],
    opts: SearchOptions,
    filter: &Filter,
    config: &TreeConfig,
    trace: &mut FilterPlanTrace,
) -> Result<Vec<(RowId, f32)>> {
    if opts.beam_width == 0 {
        return Err(TreeError::InvalidConfig(
            "beam_width must be >= 1".into(),
        ));
    }
    filter.validate(schema)?;
    let q = prepare_vector_cow(query, config.dim, config.metric)?;
    if opts.k == 0 || meta.num_vectors == 0 {
        trace.regime = Some(FilteredSearchRegime::Empty);
        return Ok(Vec::new());
    }

    let plan = filter_plan::plan_filter_with_trace(storage, schema, filter, fts_index, trace)?;
    let total_leaves = leaf_parent.len().max(1);

    match plan {
        FilterPlan::All => {
            trace.regime = Some(FilteredSearchRegime::Unfiltered);
            search(storage, internal_cache, meta, quantizer, query, opts, config)
        }
        FilterPlan::Empty => {
            trace.regime = Some(FilteredSearchRegime::Empty);
            Ok(Vec::new())
        }
        FilterPlan::Materialized(rowset) => {
            if rowset.is_empty() {
                trace.regime = Some(FilteredSearchRegime::Empty);
                return Ok(Vec::new());
            }
            if rowset.len() as usize <= opts.exact_scan_threshold {
                trace.regime = Some(FilteredSearchRegime::ExactScan);
                return search_exact_rowset(storage, &q, &rowset, opts.k, config.dim, config.metric);
            }
            let s = (rowset.len() as f64 / meta.num_vectors.max(1) as f64).clamp(1e-6, 1.0);
            if s <= 0.5 {
                trace.regime = Some(FilteredSearchRegime::MaskedTraversal);
                search_masked_traversal(
                    storage,
                    internal_cache,
                    meta,
                    quantizer,
                    &q,
                    opts,
                    &rowset,
                    s,
                    total_leaves,
                    config,
                )
            } else {
                trace.regime = Some(FilteredSearchRegime::PostFilter);
                search_post_filter(
                    storage,
                    internal_cache,
                    meta,
                    quantizer,
                    schema,
                    fts_index,
                    &q,
                    opts,
                    None,
                    Some(&rowset),
                    s,
                    total_leaves,
                    config,
                )
            }
        }
        FilterPlan::Lazy { residual, seed } => {
            if let Some(ref seed_rs) = seed
                && seed_rs.len() as usize <= opts.exact_scan_threshold
            {
                let snap = storage.segment_snapshot();
                let exact_rs = filter_plan::materialize_lazy(
                    storage,
                    schema,
                    fts_index,
                    &snap,
                    &residual,
                    Some(seed_rs),
                )?;
                if exact_rs.is_empty() {
                    trace.regime = Some(FilteredSearchRegime::Empty);
                    return Ok(Vec::new());
                }
                trace.regime = Some(FilteredSearchRegime::ExactScan);
                return search_exact_rowset(
                    storage,
                    &q,
                    &exact_rs,
                    opts.k,
                    config.dim,
                    config.metric,
                );
            }
            let est = trace.estimated_cardinality.max(1) as f64;
            let s = (est / meta.num_vectors.max(1) as f64).clamp(1e-4, 1.0);
            trace.regime = Some(FilteredSearchRegime::PostFilter);
            search_post_filter(
                storage,
                internal_cache,
                meta,
                quantizer,
                schema,
                fts_index,
                &q,
                opts,
                Some(&residual),
                seed.as_ref(),
                s,
                total_leaves,
                config,
            )
        }
    }
}

pub(super) fn search_exact_rowset(
    storage: &Storage,
    q: &[f32],
    rowset: &RowSet,
    k: usize,
    dim: usize,
    metric: DistanceMetric,
) -> Result<Vec<(RowId, f32)>> {
    let sorted_rows: Vec<RowId> = rowset.iter().collect();
    if sorted_rows.is_empty() {
        return Ok(Vec::new());
    }
    let vec_slices = storage.multi_get_vectors_pinned(&sorted_rows)?;
    let expected_bytes = dim * 4;
    let mut exact = Vec::with_capacity(sorted_rows.len());
    let mut missing_rows: Vec<RowId> = Vec::new();
    for (&row_id, opt_slice) in sorted_rows.iter().zip(vec_slices.iter()) {
        match opt_slice.as_ref() {
            Some(slice) => {
                if slice.len() != expected_bytes {
                    return Err(TreeError::CorruptedData(format!(
                        "vector {row_id:?} expected {expected_bytes} bytes, got {}",
                        slice.len()
                    )));
                }
                exact.push((row_id, search_distance_bytes(metric, q, slice)));
            }
            None => missing_rows.push(row_id),
        }
    }
    if !missing_rows.is_empty() {
        let row_slices = storage.multi_get_row_bytes_pinned(&missing_rows)?;
        for (&row_id, opt_row) in missing_rows.iter().zip(row_slices.iter()) {
            let slice = opt_row.as_ref().ok_or_else(|| {
                TreeError::InvariantViolation(format!("row {row_id:?} missing in cf_rows"))
            })?;
            let view = RowView::parse(slice)?;
            if view.has_vector() {
                return Err(TreeError::CorruptedData(format!(
                    "missing vector {row_id:?} in cf_vectors"
                )));
            }
        }
    }
    select_top_k(&mut exact, k, cmp_dist_row);
    Ok(exact)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn search_masked_traversal(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    meta: &AnnMeta,
    quantizer: &RabitqQuantizer,
    q: &[f32],
    opts: SearchOptions,
    rowset: &RowSet,
    selectivity: f64,
    total_leaves: usize,
    config: &TreeConfig,
) -> Result<Vec<(RowId, f32)>> {
    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();
    let rabitq_words = quantizer.num_words();
    let inv_sqrt_padded_dim = quantizer.inv_sqrt_padded_dim();

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    heap.push(HeapEntry {
        dist: 0.0,
        node_id: meta.root_id,
        level: meta.root_level,
    });

    let mut scratch = QueryScratch::new(padded_dim, rabitq_words);
    quantizer.rotate_into(q, &mut scratch.rot_q);
    let dot_q_meta = if config.metric == DistanceMetric::DotProduct {
        Some(pack_query_bits_into(&scratch.rot_q, &mut scratch.planes))
    } else {
        None
    };
    let mut score_ctx = LeafScoreContext {
        q,
        scratch: &mut scratch,
        dot_q_meta,
        inv_sqrt_padded_dim,
        rabitq_words,
        metric: config.metric,
    };

    let target = opts.rerank_candidates.unwrap_or(opts.k).max(opts.k);
    let default_max = ((opts.beam_width as f64 / selectivity * 1.5).ceil() as usize)
        .clamp(opts.beam_width, total_leaves.max(opts.beam_width));
    let max_expanded = opts
        .max_expanded_leaves
        .unwrap_or(default_max)
        .max(opts.beam_width);

    let seg_snap = storage.segment_snapshot();
    let dropped = storage.dropped_segments_snapshot();
    let need_liveness_resolve = !seg_snap.has_deletes_or_forwarding.is_empty() || !dropped.is_empty();

    let mut probed_leaves = 0usize;
    let mut batch_target = opts.beam_width;
    let mut candidates: Vec<LeafCandidate> = Vec::with_capacity(target.max(opts.beam_width * 16));
    let mut batch_leaves: Vec<NodeId> = Vec::new();
    let mut mask_buf: Vec<u64> = Vec::new();
    let mut any_forwarded = false;

    while probed_leaves < max_expanded && !heap.is_empty() {
        let take_limit = batch_target.min(max_expanded - probed_leaves);
        batch_leaves.clear();
        while batch_leaves.len() < take_limit
            && let Some(item) = heap.pop()
        {
            if item.level == 0 {
                batch_leaves.push(item.node_id);
            } else {
                expand_internal_node_unpruned(internal_cache, &item, q, dim, config.metric, &mut heap)?;
            }
        }
        if batch_leaves.is_empty() {
            break;
        }
        probed_leaves += batch_leaves.len();
        batch_leaves.sort_unstable();
        batch_leaves.dedup();

        let leaf_slices = storage.multi_get_leaf_nodes_pinned(&batch_leaves)?;
        for (leaf_id, opt_slice) in batch_leaves.iter().copied().zip(leaf_slices.iter()) {
            let slice = opt_slice.as_ref().ok_or_else(|| {
                TreeError::CorruptedData(format!("missing leaf blob for leaf {leaf_id}"))
            })?;
            let view = LeafNodeView::parse(slice, dim, padded_dim, rabitq_words)?;
            if view.live_count == 0 {
                continue;
            }
            let (any_live, leaf_forwarded) = build_effective_mask_with_rowset(
                &view,
                &seg_snap,
                &dropped,
                need_liveness_resolve,
                Some(rowset),
                &mut mask_buf,
            );
            if !any_live {
                continue;
            }
            any_forwarded |= leaf_forwarded;
            score_leaf_view(&view, Some(&mask_buf), &mut score_ctx, &mut candidates);
        }

        if (candidates.len() >= target && probed_leaves >= opts.beam_width)
            || heap.is_empty()
            || probed_leaves >= max_expanded
        {
            break;
        }
        batch_target = batch_target.saturating_mul(2);
    }

    if any_forwarded {
        resolve_and_dedup_forwarded(&seg_snap, &dropped, &mut candidates);
    }
    finalize_candidates(
        storage,
        q,
        candidates,
        opts.k,
        opts.rerank_candidates,
        dim,
        config.metric,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn search_post_filter(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    meta: &AnnMeta,
    quantizer: &RabitqQuantizer,
    schema: &CollectionSchema,
    fts_index: &FtsIndex,
    q: &[f32],
    opts: SearchOptions,
    residual: Option<&Filter>,
    seed: Option<&RowSet>,
    selectivity: f64,
    total_leaves: usize,
    config: &TreeConfig,
) -> Result<Vec<(RowId, f32)>> {
    let mut posting_membership = match residual {
        Some(filter) => PostingMembership::for_filter(schema, filter)?,
        None => None,
    };
    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();
    let rabitq_words = quantizer.num_words();
    let inv_sqrt_padded_dim = quantizer.inv_sqrt_padded_dim();

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    heap.push(HeapEntry {
        dist: 0.0,
        node_id: meta.root_id,
        level: meta.root_level,
    });

    let mut scratch = QueryScratch::new(padded_dim, rabitq_words);
    quantizer.rotate_into(q, &mut scratch.rot_q);
    let dot_q_meta = if config.metric == DistanceMetric::DotProduct {
        Some(pack_query_bits_into(&scratch.rot_q, &mut scratch.planes))
    } else {
        None
    };
    let mut score_ctx = LeafScoreContext {
        q,
        scratch: &mut scratch,
        dot_q_meta,
        inv_sqrt_padded_dim,
        rabitq_words,
        metric: config.metric,
    };

    let target = opts.rerank_candidates.unwrap_or(opts.k).max(opts.k);
    let default_max = ((opts.beam_width as f64 / selectivity * 1.5).ceil() as usize)
        .clamp(opts.beam_width, total_leaves.max(opts.beam_width));
    let max_expanded = opts
        .max_expanded_leaves
        .unwrap_or(default_max)
        .max(opts.beam_width);

    let seg_snap = storage.segment_snapshot();
    let dropped = storage.dropped_segments_snapshot();
    let need_liveness = !seg_snap.has_deletes_or_forwarding.is_empty() || !dropped.is_empty();

    let mut probed_leaves = 0usize;
    let mut batch_target = opts.beam_width;
    let mut verified: Vec<LeafCandidate> = Vec::with_capacity(target.max(opts.beam_width * 16));
    let mut batch_leaves: Vec<NodeId> = Vec::new();
    let mut batch_cands: Vec<LeafCandidate> = Vec::new();
    let mut mask_buf: Vec<u64> = Vec::new();
    let column_filter = if posting_membership.is_none() {
        residual.and_then(|filter| ColumnFilter::new(schema, filter))
    } else {
        None
    };

    while probed_leaves < max_expanded && !heap.is_empty() {
        let take_limit = batch_target.min(max_expanded - probed_leaves);
        batch_leaves.clear();
        while batch_leaves.len() < take_limit
            && let Some(item) = heap.pop()
        {
            if item.level == 0 {
                batch_leaves.push(item.node_id);
            } else {
                expand_internal_node_unpruned(internal_cache, &item, q, dim, config.metric, &mut heap)?;
            }
        }
        if batch_leaves.is_empty() {
            break;
        }
        probed_leaves += batch_leaves.len();
        batch_leaves.sort_unstable();
        batch_leaves.dedup();

        batch_cands.clear();
        let leaf_slices = storage.multi_get_leaf_nodes_pinned(&batch_leaves)?;
        let mut batch_forwarded = false;

        for (leaf_id, opt_slice) in batch_leaves.iter().copied().zip(leaf_slices.iter()) {
            let slice = opt_slice.as_ref().ok_or_else(|| {
                TreeError::CorruptedData(format!("missing leaf blob for leaf {leaf_id}"))
            })?;
            let view = LeafNodeView::parse(slice, dim, padded_dim, rabitq_words)?;
            if view.live_count == 0 {
                continue;
            }
            if !need_liveness && seed.is_none() {
                let allowed_slots = if view.has_tombstones() {
                    view.copy_live_mask_into(&mut mask_buf);
                    Some(mask_buf.as_slice())
                } else {
                    None
                };
                score_leaf_view(&view, allowed_slots, &mut score_ctx, &mut batch_cands);
            } else {
                let (any_live, leaf_forwarded) = build_effective_mask_with_rowset(
                    &view,
                    &seg_snap,
                    &dropped,
                    need_liveness,
                    seed,
                    &mut mask_buf,
                );
                if !any_live {
                    continue;
                }
                batch_forwarded |= leaf_forwarded;
                score_leaf_view(&view, Some(&mask_buf), &mut score_ctx, &mut batch_cands);
            }
        }

        if batch_forwarded {
            resolve_and_dedup_forwarded(&seg_snap, &dropped, &mut batch_cands);
        }

        if let Some(postings) = &mut posting_membership {
            postings.retain_matches(storage, &mut batch_cands)?;
            verified.extend_from_slice(&batch_cands);
        } else if let Some(res_filter) = residual {
            batch_cands.sort_unstable_by(cmp_cand_dist_id);
            let mut chunk_buf: Vec<(RowId, LeafCandidate)> = Vec::new();
            let mut rows_buf: Vec<RowId> = Vec::new();
            let mut i = 0usize;
            while i < batch_cands.len() {
                let remaining_end = if verified.len() >= target {
                    select_top_k(&mut verified, target, cmp_cand_dist_id);
                    let cut = verified[target - 1];
                    if cmp_cand_dist_id(&batch_cands[i], &cut).is_gt() {
                        break;
                    }
                    i + batch_cands[i..].partition_point(|c| cmp_cand_dist_id(c, &cut).is_le())
                } else {
                    batch_cands.len()
                };

                let need = target.saturating_sub(verified.len()).max(1);
                let chunk_len = ((need as f64 / selectivity) * 1.2).ceil() as usize;
                let chunk_len = chunk_len.clamp(32, 512).min(remaining_end - i);

                chunk_buf.clear();
                chunk_buf.extend(
                    batch_cands[i..i + chunk_len]
                        .iter()
                        .map(|&c| (RowId(c.doc_id), c)),
                );
                chunk_buf.sort_unstable_by_key(|&(r, _)| r);
                chunk_buf.dedup_by_key(|(r, _)| *r);
                rows_buf.clear();
                rows_buf.extend(chunk_buf.iter().map(|&(r, _)| r));

                if let Some(filter) = &column_filter
                    && let Some(matches) = filter.match_candidates(storage, &seg_snap, &rows_buf)?
                {
                    let mut matches = matches.into_iter();
                    chunk_buf.retain(|&(_, candidate)| match matches.next().unwrap() {
                        Some(true) => {
                            verified.push(candidate);
                            false
                        }
                        Some(false) => false,
                        None => true,
                    });
                    rows_buf.clear();
                    rows_buf.extend(chunk_buf.iter().map(|&(r, _)| r));
                }
                if !chunk_buf.is_empty() {
                    verify_chunk(
                        storage,
                        schema,
                        fts_index,
                        res_filter,
                        &chunk_buf,
                        &rows_buf,
                        &mut verified,
                    )?;
                }
                i += chunk_len;
            }
        } else {
            verified.extend_from_slice(&batch_cands);
        }

        if (verified.len() >= target && probed_leaves >= opts.beam_width)
            || heap.is_empty()
            || probed_leaves >= max_expanded
        {
            break;
        }
        batch_target = batch_target.saturating_mul(2);
    }

    verified.sort_unstable_by(|a, b| a.doc_id.cmp(&b.doc_id).then_with(|| a.dist.total_cmp(&b.dist)));
    verified.dedup_by_key(|c| c.doc_id);
    finalize_candidates(
        storage,
        q,
        verified,
        opts.k,
        opts.rerank_candidates,
        dim,
        config.metric,
    )
}

fn verify_chunk(
    storage: &Storage,
    schema: &CollectionSchema,
    fts_index: &FtsIndex,
    res_filter: &Filter,
    chunk: &[(RowId, LeafCandidate)],
    rows: &[RowId],
    verified: &mut Vec<LeafCandidate>,
) -> Result<()> {
    if crate::detached::filter_mentions_detached(schema, res_filter) {
        let flags = crate::detached::filter_rows_mixed_batch(
            storage, schema, fts_index, res_filter, rows,
        )?;
        for (&(_, cand), ok) in chunk.iter().zip(flags) {
            if ok {
                verified.push(cand);
            }
        }
    } else {
        let slices = storage.multi_get_row_bytes_pinned(rows)?;
        for (&(row, cand), opt_slice) in chunk.iter().zip(slices.iter()) {
            let slice = opt_slice.as_ref().ok_or_else(|| {
                TreeError::InvariantViolation(format!("row {row:?} missing in cf_rows"))
            })?;
            let view = RowView::parse(slice)?;
            if view.matches_filter(
                schema,
                res_filter,
                &mut |kind, attr, query, last_as_prefix, val| {
                    fts_index.matches_fts_attr(schema, kind, attr, query, last_as_prefix, val)
                },
            )? {
                verified.push(cand);
            }
        }
    }
    Ok(())
}

pub(super) fn expand_internal_node_unpruned(
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    item: &HeapEntry,
    q: &[f32],
    dim: usize,
    metric: DistanceMetric,
    heap: &mut BinaryHeap<HeapEntry>,
) -> Result<()> {
    let node = internal_cache.get(&item.node_id).ok_or_else(|| {
        TreeError::CorruptedData(format!(
            "missing internal node {} in cache",
            item.node_id
        ))
    })?;
    let child_level = item.level - 1;
    for (i, &child_id) in node.child_ids.iter().enumerate() {
        let d = search_distance(metric, q, node.child_centroid(i, dim));
        heap.push(HeapEntry {
            dist: d,
            node_id: child_id,
            level: child_level,
        });
    }
    Ok(())
}

#[inline]
pub(super) fn build_effective_mask_with_rowset(
    view: &LeafNodeView<'_>,
    seg_snap: &SegmentTableSnapshot,
    dropped: &roaring::RoaringBitmap,
    need_liveness_resolve: bool,
    rowset: Option<&RowSet>,
    mask_buf: &mut Vec<u64>,
) -> (bool, bool) {
    view.copy_live_mask_into(mask_buf);
    let mut any_live = false;
    let mut saw_forwarded = false;
    for (w_idx, word_ref) in mask_buf.iter_mut().enumerate() {
        let mut bits = *word_ref;
        let mut out = bits;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let slot = (w_idx << 6) | b;
            if slot >= view.slot_cap {
                out &= !(1u64 << b);
                continue;
            }
            let row = view.row_id(slot);
            let resolved_opt = if need_liveness_resolve {
                seg_snap
                    .resolve_row(row)
                    .filter(|r| !dropped.contains(r.seg()))
            } else {
                Some(row)
            };
            match resolved_opt {
                Some(resolved) if rowset.is_none_or(|rs| rs.contains(resolved)) => {
                    any_live = true;
                    if resolved != row {
                        saw_forwarded = true;
                    }
                }
                _ => {
                    out &= !(1u64 << b);
                }
            }
        }
        *word_ref = out;
    }
    (any_live, saw_forwarded)
}

/// Build `live_mask & liveness_mask(row_ids)` for `view` against `seg_snap`.
/// Returns `(any_live_slots, saw_forwarded_row)`.
#[inline]
pub(crate) fn build_effective_mask(
    view: &LeafNodeView<'_>,
    seg_snap: &SegmentTableSnapshot,
    dropped: &roaring::RoaringBitmap,
    mask_buf: &mut Vec<u64>,
) -> (bool, bool) {
    view.copy_live_mask_into(mask_buf);
    let mut any_live = false;
    let mut saw_forwarded = false;
    for (w_idx, word_ref) in mask_buf.iter_mut().enumerate() {
        let mut bits = *word_ref;
        let mut out = bits;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let slot = (w_idx << 6) | b;
            if slot >= view.slot_cap {
                out &= !(1u64 << b);
                continue;
            }
            let row = view.row_id(slot);
            match seg_snap.resolve_row(row) {
                Some(resolved) if !dropped.contains(resolved.seg()) => {
                    any_live = true;
                    if resolved != row {
                        saw_forwarded = true;
                    }
                }
                _ => {
                    out &= !(1u64 << b);
                }
            }
        }
        *word_ref = out;
    }
    (any_live, saw_forwarded)
}

pub(super) fn resolve_and_dedup_forwarded(
    seg_snap: &SegmentTableSnapshot,
    dropped: &roaring::RoaringBitmap,
    candidates: &mut Vec<LeafCandidate>,
) {
    let mut write = 0usize;
    for read in 0..candidates.len() {
        let mut c = candidates[read];
        if let Some(resolved) = seg_snap.resolve_row(RowId(c.doc_id))
            && !dropped.contains(resolved.seg())
        {
            c.doc_id = resolved.raw();
            candidates[write] = c;
            write += 1;
        }
    }
    candidates.truncate(write);
    candidates.sort_unstable_by(|a, b| a.doc_id.cmp(&b.doc_id).then_with(|| a.dist.total_cmp(&b.dist)));
    candidates.dedup_by_key(|c| c.doc_id);
}
