//! Beam-search traversal, RaBitQ leaf scanning, effective-slot masking, and exact reranking.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use izzydb_core::config::{DistanceMetric, NodeId, RowId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::metric::{cluster_distance, prepare_vector_cow, search_distance, search_distance_bytes};
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::CollectionSchema;
use izzydb_rabitq::{
    QuantizedQueryMeta, QueryScratch, RabitqQuantizer, pack_query_bits_from_diff_bytes_into,
    pack_query_bits_into,
};
use izzydb_storage::codec::RowView;
use izzydb_storage::node::{AnnMeta, LeafNodeView, PackedInternalNode};
use izzydb_storage::{DBPinnableSlice, SegmentTableSnapshot, Storage, TxContext};

use crate::filter_plan::{self, FilterPlan, FilterPlanTrace, FilteredSearchRegime};
use crate::fts_index::FtsIndex;
use crate::simd::{self, LeafCandidate};

/// Parameters controlling quantized vector search, optional full-vector reranking, and filtered expansion.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    pub k: usize,
    pub beam_width: usize,
    pub rerank_candidates: Option<usize>,
    /// Materialized filtered sets with `<= exact_scan_threshold` vector rows use exact scan.
    pub exact_scan_threshold: usize,
    /// Maximum number of leaves probed during filtered dynamic expansion (defaults to `beam_width * 16`).
    pub max_expanded_leaves: Option<usize>,
}

impl SearchOptions {
    pub fn new(k: usize, beam_width: usize) -> Self {
        Self {
            k,
            beam_width,
            rerank_candidates: None,
            exact_scan_threshold: 4_096,
            max_expanded_leaves: None,
        }
    }

    pub fn with_rerank(mut self, rerank_candidates: usize) -> Self {
        self.rerank_candidates = Some(rerank_candidates);
        self
    }

    pub fn with_exact_scan_threshold(mut self, threshold: usize) -> Self {
        self.exact_scan_threshold = threshold;
        self
    }

    pub fn with_max_expanded_leaves(mut self, max_leaves: usize) -> Self {
        self.max_expanded_leaves = Some(max_leaves);
        self
    }
}

/// Route a prepared (validated and normalized if Cosine) vector to the single closest leaf.
pub fn route_to_leaf(
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    meta: &AnnMeta,
    prepared_vec: &[f32],
    beam_width: usize,
    config: &TreeConfig,
) -> Result<NodeId> {
    let leaves = search_leaves(internal_cache, meta, prepared_vec, 1, beam_width, config)?;
    leaves
        .into_iter()
        .next()
        .map(|(id, _)| id)
        .ok_or_else(|| TreeError::CorruptedData("route_to_leaf reached empty frontier".into()))
}

/// Route a prepared vector to the single closest leaf, reading through `TxContext` overlays.
pub(crate) fn route_to_leaf_in_tx(
    tx: &TxContext<'_>,
    prepared_vec: &[f32],
    beam_width: usize,
    config: &TreeConfig,
) -> Result<NodeId> {
    let meta = tx
        .ann_meta()
        .ok_or_else(|| TreeError::InvalidConfig("collection has no vector index".into()))?;
    let leaves = search_leaves_with(
        |id| tx.get_internal_node_ref(id),
        meta,
        prepared_vec,
        1,
        beam_width,
        config,
    )?;
    leaves
        .into_iter()
        .next()
        .map(|(id, _)| id)
        .ok_or_else(|| TreeError::CorruptedData("route_to_leaf_in_tx reached empty frontier".into()))
}

/// Route a prepared vector to the single closest leaf via an arbitrary internal-node lookup closure.
pub fn route_to_leaf_with<'a, F>(
    get_internal: F,
    meta: &AnnMeta,
    prepared_vec: &[f32],
    beam_width: usize,
    config: &TreeConfig,
) -> Result<NodeId>
where
    F: Fn(NodeId) -> Result<&'a PackedInternalNode>,
{
    let leaves = search_leaves_with(get_internal, meta, prepared_vec, 1, beam_width, config)?;
    leaves
        .into_iter()
        .next()
        .map(|(id, _)| id)
        .ok_or_else(|| TreeError::CorruptedData("route_to_leaf reached empty frontier".into()))
}

/// Find the top `count` closest leaves to `prepared_vec` using in-memory `cluster_distance` beam search.
pub fn search_leaves(
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    meta: &AnnMeta,
    prepared_vec: &[f32],
    count: usize,
    beam_width: usize,
    config: &TreeConfig,
) -> Result<Vec<(NodeId, f32)>> {
    search_leaves_with(
        |node_id| {
            internal_cache.get(&node_id).ok_or_else(|| {
                TreeError::CorruptedData(format!("missing internal node {node_id} in cache"))
            })
        },
        meta,
        prepared_vec,
        count,
        beam_width,
        config,
    )
}

pub fn search_leaves_with<'a, F>(
    get_internal: F,
    meta: &AnnMeta,
    prepared_vec: &[f32],
    count: usize,
    beam_width: usize,
    config: &TreeConfig,
) -> Result<Vec<(NodeId, f32)>>
where
    F: Fn(NodeId) -> Result<&'a PackedInternalNode>,
{
    let beam = beam_width.max(count).max(1);
    let mut frontier: Vec<(NodeId, f32)> = vec![(meta.root_id, 0.0)];
    let mut next: Vec<(NodeId, f32)> = Vec::new();
    let mut level = meta.root_level;
    let dim = config.dim;

    while level > 0 {
        next.clear();
        for &(node_id, _) in &frontier {
            let node = get_internal(node_id)?;
            for (i, &child_id) in node.child_ids.iter().enumerate() {
                let d = cluster_distance(config.metric, prepared_vec, node.child_centroid(i, dim));
                next.push((child_id, d));
            }
        }
        select_top_k(&mut next, beam, cmp_node_dist_id);
        if next.is_empty() {
            return Err(TreeError::CorruptedData(format!(
                "internal frontier empty at level {level}"
            )));
        }
        std::mem::swap(&mut frontier, &mut next);
        level -= 1;
    }

    if frontier.len() > count {
        frontier.truncate(count);
    }
    Ok(frontier)
}

/// Unfiltered approximate nearest neighbor search using in-memory beam traversal,
/// batched zero-copy RaBitQ leaf scanning with effective-slot liveness masking,
/// and optional batched full-vector reranking over `RowId`s.
pub fn search(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    meta: &AnnMeta,
    quantizer: &RabitqQuantizer,
    query: &[f32],
    opts: SearchOptions,
    config: &TreeConfig,
) -> Result<Vec<(RowId, f32)>> {
    if opts.beam_width == 0 {
        return Err(TreeError::InvalidConfig(
            "beam_width must be >= 1".into(),
        ));
    }
    let q = prepare_vector_cow(query, config.dim, config.metric)?;
    if opts.k == 0 || meta.num_vectors == 0 {
        return Ok(Vec::new());
    }

    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();
    let rabitq_words = quantizer.num_words();
    let inv_sqrt_padded_dim = quantizer.inv_sqrt_padded_dim();

    let mut frontier: Vec<(NodeId, f32)> = Vec::with_capacity(opts.beam_width);
    frontier.push((meta.root_id, 0.0));
    let mut next: Vec<(NodeId, f32)> = Vec::new();
    let mut level = meta.root_level;

    while level > 0 {
        next.clear();
        for &(node_id, _) in &frontier {
            let node = internal_cache.get(&node_id).ok_or_else(|| {
                TreeError::CorruptedData(format!("missing internal node {node_id} in cache"))
            })?;
            for (i, &child_id) in node.child_ids.iter().enumerate() {
                let d = search_distance(config.metric, &q, node.child_centroid(i, dim));
                next.push((child_id, d));
            }
        }
        select_top_k(&mut next, opts.beam_width, cmp_node_dist_id);
        if next.is_empty() {
            return Err(TreeError::CorruptedData(format!(
                "internal frontier empty during search at level {level}"
            )));
        }
        std::mem::swap(&mut frontier, &mut next);
        level -= 1;
    }

    let mut leaf_ids: Vec<NodeId> = frontier.iter().map(|&(id, _)| id).collect();
    leaf_ids.sort_unstable();
    leaf_ids.dedup();

    let leaf_slices = storage.multi_get_leaf_nodes_pinned(&leaf_ids)?;
    let seg_snap = storage.segment_snapshot();
    let dropped = storage.dropped_segments_snapshot();
    let need_liveness_mask = !seg_snap.has_deletes_or_forwarding.is_empty() || !dropped.is_empty();

    let mut scratch = QueryScratch::new(padded_dim, rabitq_words);
    quantizer.rotate_into(&q, &mut scratch.rot_q);

    let dot_q_meta = if config.metric == DistanceMetric::DotProduct {
        Some(pack_query_bits_into(&scratch.rot_q, &mut scratch.planes))
    } else {
        None
    };

    let mut score_ctx = LeafScoreContext {
        q: &q,
        scratch: &mut scratch,
        dot_q_meta,
        inv_sqrt_padded_dim,
        rabitq_words,
        metric: config.metric,
    };
    let mut candidates: Vec<LeafCandidate> =
        Vec::with_capacity(leaf_ids.len() * config.max_leaf_size);
    let mut live_mask_buf: Vec<u64> = Vec::new();
    let mut any_forwarded = false;

    for (leaf_id, opt_slice) in leaf_ids.iter().copied().zip(leaf_slices.iter()) {
        let slice = opt_slice.as_ref().ok_or_else(|| {
            TreeError::CorruptedData(format!("missing leaf blob for leaf {leaf_id}"))
        })?;
        let view = LeafNodeView::parse(slice, dim, padded_dim, rabitq_words)?;
        if view.live_count == 0 {
            continue;
        }
        if !need_liveness_mask {
            let allowed_slots = if view.has_tombstones() {
                view.copy_live_mask_into(&mut live_mask_buf);
                Some(live_mask_buf.as_slice())
            } else {
                None
            };
            score_leaf_view(&view, allowed_slots, &mut score_ctx, &mut candidates);
        } else {
            let (any_live, leaf_forwarded) =
                build_effective_mask(&view, &seg_snap, &dropped, &mut live_mask_buf);
            if !any_live {
                continue;
            }
            any_forwarded |= leaf_forwarded;
            score_leaf_view(&view, Some(&live_mask_buf), &mut score_ctx, &mut candidates);
        }
    }

    if any_forwarded {
        resolve_and_dedup_forwarded(&seg_snap, &dropped, &mut candidates);
    }

    finalize_candidates(
        storage,
        &q,
        candidates,
        opts.k,
        opts.rerank_candidates,
        dim,
        config.metric,
    )
}

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

fn search_exact_rowset(
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
fn search_masked_traversal(
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
fn search_post_filter(
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

        if let Some(res_filter) = residual {
            batch_cands.sort_unstable_by(cmp_cand_dist_id);
            let kth_cutoff = if verified.len() >= target {
                select_top_k(&mut verified, target, cmp_cand_dist_id);
                Some(verified[target - 1].dist)
            } else {
                None
            };
            let check_limit = match kth_cutoff {
                Some(cut) => batch_cands.partition_point(|c| c.dist <= cut),
                None => batch_cands.len(),
            };
            let to_check = &batch_cands[..check_limit];
            let mut row_pairs: Vec<(RowId, LeafCandidate)> =
                to_check.iter().map(|&c| (RowId(c.doc_id), c)).collect();
            row_pairs.sort_unstable_by_key(|&(r, _)| r);
            row_pairs.dedup_by_key(|(r, _)| *r);
            for chunk in row_pairs.chunks(128) {
                let rows: Vec<RowId> = chunk.iter().map(|&(r, _)| r).collect();
                if crate::detached::filter_mentions_detached(schema, res_filter) {
                    let flags = crate::detached::filter_rows_mixed_batch(
                        storage, schema, fts_index, res_filter, &rows,
                    )?;
                    for (&(_, cand), ok) in chunk.iter().zip(flags) {
                        if ok {
                            verified.push(cand);
                        }
                    }
                } else {
                    let slices = storage.multi_get_row_bytes_pinned(&rows)?;
                    for (&(row, cand), opt_slice) in chunk.iter().zip(slices.iter()) {
                        let slice = opt_slice.as_ref().ok_or_else(|| {
                            TreeError::InvariantViolation(format!("row {row:?} missing in cf_rows"))
                        })?;
                        let view = RowView::parse(slice)?;
                        if view.matches_filter(
                            schema,
                            res_filter,
                            &mut |kind, attr, query, last_as_prefix, val| {
                                fts_index
                                    .matches_fts_attr(schema, kind, attr, query, last_as_prefix, val)
                            },
                        )? {
                            verified.push(cand);
                        }
                    }
                }
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

fn expand_internal_node_unpruned(
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
fn build_effective_mask_with_rowset(
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

fn resolve_and_dedup_forwarded(
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

#[allow(dead_code)]
pub(crate) fn parse_leaf_views<'a>(
    leaf_ids: &[NodeId],
    leaf_slices: &'a [Option<DBPinnableSlice<'a>>],
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<Vec<Option<LeafNodeView<'a>>>> {
    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();
    let rabitq_words = quantizer.num_words();
    let mut views = Vec::with_capacity(leaf_ids.len());
    for (&leaf_id, opt_slice) in leaf_ids.iter().zip(leaf_slices.iter()) {
        let slice = opt_slice.as_ref().ok_or_else(|| {
            TreeError::CorruptedData(format!("missing leaf blob for leaf {leaf_id}"))
        })?;
        views.push(Some(LeafNodeView::parse(slice, dim, padded_dim, rabitq_words)?));
    }
    Ok(views)
}

pub(crate) struct LeafScoreContext<'a> {
    pub q: &'a [f32],
    pub scratch: &'a mut QueryScratch,
    pub dot_q_meta: Option<QuantizedQueryMeta>,
    pub inv_sqrt_padded_dim: f32,
    pub rabitq_words: usize,
    pub metric: DistanceMetric,
}

pub(crate) fn score_leaf_view(
    view: &LeafNodeView<'_>,
    allowed_slots: Option<&[u64]>,
    ctx: &mut LeafScoreContext<'_>,
    candidates: &mut Vec<LeafCandidate>,
) {
    let (q_res_norm_sq, zero_res_dist, q_meta) = match ctx.metric {
        DistanceMetric::L2Squared => {
            let d2 = simd::l2_squared_bytes(ctx.q, view.centroid_bytes);
            let meta = pack_query_bits_from_diff_bytes_into(
                &ctx.scratch.rot_q,
                view.rotated_centroid_bytes,
                &mut ctx.scratch.rot_diff,
                &mut ctx.scratch.planes,
            );
            (d2, d2, meta)
        }
        DistanceMetric::Cosine => {
            let d2 = simd::l2_squared_bytes(ctx.q, view.centroid_bytes);
            let zero = (1.0 - simd::dot_product_bytes(ctx.q, view.centroid_bytes)).max(0.0);
            let meta = pack_query_bits_from_diff_bytes_into(
                &ctx.scratch.rot_q,
                view.rotated_centroid_bytes,
                &mut ctx.scratch.rot_diff,
                &mut ctx.scratch.planes,
            );
            (d2, zero, meta)
        }
        DistanceMetric::DotProduct => {
            let zero = -simd::dot_product_bytes(ctx.q, view.centroid_bytes);
            (0.0, zero, ctx.dot_q_meta.unwrap())
        }
    };

    simd::score_leaf_rabitq_masked(
        &simd::RabitqLeafScoreParams {
            slot_cap: view.slot_cap,
            rabitq_words: ctx.rabitq_words,
            popcount_u32: view.popcount_u32,
            doc_ids_bytes: view.row_ids_bytes,
            codes_bytes: view.codes_bytes,
            code_popcounts_bytes: view.code_popcounts_bytes,
            res_norm_sqs_bytes: view.res_norm_sqs_bytes,
            scales_bytes: view.scales_bytes,
            q0: &ctx.scratch.planes[0],
            q1: &ctx.scratch.planes[1],
            q2: &ctx.scratch.planes[2],
            q3: &ctx.scratch.planes[3],
            vl: q_meta.vl,
            delta: q_meta.delta,
            sum_q: q_meta.sum_q,
            inv_sqrt_padded_dim: ctx.inv_sqrt_padded_dim,
            q_res_norm_sq,
            zero_res_dist,
            metric: ctx.metric,
        },
        allowed_slots,
        candidates,
    );
}

pub(crate) fn score_exact_vectors(
    storage: &Storage,
    q: &[f32],
    sorted_rows: &[RowId],
    k: usize,
    dim: usize,
    metric: DistanceMetric,
) -> Result<Vec<(RowId, f32)>> {
    let vec_slices = storage.multi_get_vectors_pinned(sorted_rows)?;
    let expected_bytes = dim * 4;
    let mut exact = Vec::with_capacity(sorted_rows.len());
    for (&row_id, opt_slice) in sorted_rows.iter().zip(vec_slices.iter()) {
        let slice = opt_slice.as_ref().ok_or_else(|| {
            TreeError::CorruptedData(format!("missing vector {row_id:?} in cf_vectors"))
        })?;
        if slice.len() != expected_bytes {
            return Err(TreeError::CorruptedData(format!(
                "vector {row_id:?} expected {expected_bytes} bytes, got {}",
                slice.len()
            )));
        }
        exact.push((row_id, search_distance_bytes(metric, q, slice)));
    }
    select_top_k(&mut exact, k, cmp_dist_row);
    Ok(exact)
}

pub(crate) fn finalize_candidates(
    storage: &Storage,
    q: &[f32],
    mut candidates: Vec<LeafCandidate>,
    k: usize,
    rerank_candidates: Option<usize>,
    dim: usize,
    metric: DistanceMetric,
) -> Result<Vec<(RowId, f32)>> {
    let Some(rerank_k) = rerank_candidates else {
        select_top_k(&mut candidates, k, cmp_cand_dist_id);
        return Ok(candidates
            .into_iter()
            .map(|c| (RowId(c.doc_id), c.dist))
            .collect());
    };

    let shortlist_len = rerank_k.max(k);
    select_top_k(&mut candidates, shortlist_len, cmp_cand_dist_id);
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    candidates.sort_unstable_by_key(|c| c.doc_id);
    let sorted_rows: Vec<RowId> = candidates.iter().map(|c| RowId(c.doc_id)).collect();
    score_exact_vectors(storage, q, &sorted_rows, k, dim, metric)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapEntry {
    pub dist: f32,
    pub node_id: NodeId,
    pub level: u8,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id && self.dist.to_bits() == other.dist.to_bits()
    }
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior on BinaryHeap.
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.node_id.cmp(&self.node_id))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[allow(dead_code)]
#[inline]
pub(crate) fn make_min_heap() -> BinaryHeap<HeapEntry> {
    BinaryHeap::new()
}

#[inline]
pub(crate) fn select_top_k<T, F>(items: &mut Vec<T>, limit: usize, cmp: F)
where
    F: Fn(&T, &T) -> Ordering,
{
    if limit == 0 {
        items.clear();
        return;
    }
    if items.len() > limit {
        items.select_nth_unstable_by(limit - 1, &cmp);
        items.truncate(limit);
    }
    items.sort_unstable_by(cmp);
}

#[inline]
fn cmp_dist_row(a: &(RowId, f32), b: &(RowId, f32)) -> Ordering {
    a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
}

#[inline]
fn cmp_node_dist_id(a: &(NodeId, f32), b: &(NodeId, f32)) -> Ordering {
    a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
}

#[inline]
fn cmp_cand_dist_id(a: &LeafCandidate, b: &LeafCandidate) -> Ordering {
    a.dist.total_cmp(&b.dist).then_with(|| a.doc_id.cmp(&b.doc_id))
}
