use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use roaring::RoaringBitmap;

use crate::attr_index::{AttributeIndex, ClusterCandidates};
use crate::config::{DistanceMetric, DocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::filter::Filter;
use crate::metric::{cluster_distance, prepare_vector_cow, search_distance, search_distance_bytes};
use crate::node::{LeafNodeView, PackedInternalNode, TreeMeta};
use crate::rabitq::{
    QueryScratch, RabitqQuantizer, pack_query_bits_from_diff_bytes_into, pack_query_bits_into,
};
use crate::schema::CollectionSchema;
use crate::simd::{self, LeafCandidate};
use crate::storage::{Storage, TxContext};

/// Route a prepared (validated and normalized if Cosine) vector to the single closest leaf.
pub fn route_to_leaf(
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    meta: &TreeMeta,
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
    let leaves = search_leaves_with(
        |id| tx.get_internal_node_ref(id),
        tx.meta(),
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

/// Find the top `count` closest leaves to `prepared_vec` using in-memory `cluster_distance` beam search.
pub fn search_leaves(
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    meta: &TreeMeta,
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

fn search_leaves_with<'a, F>(
    get_internal: F,
    meta: &TreeMeta,
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

/// Parameters controlling quantized search, optional full-vector reranking, and filtered expansion.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    pub k: usize,
    pub beam_width: usize,
    pub rerank_candidates: Option<usize>,
    /// When cluster pruning yields `<= exhaustive_cluster_threshold` leaves, bypass tree traversal
    /// and RaBitQ scoring and evaluate exact vectors directly.
    pub exhaustive_cluster_threshold: usize,
    /// Maximum number of leaves probed during filtered dynamic expansion (defaults to `beam_width * 16`).
    pub max_expanded_leaves: Option<usize>,
}

impl SearchOptions {
    pub fn new(k: usize, beam_width: usize) -> Self {
        Self {
            k,
            beam_width,
            rerank_candidates: None,
            exhaustive_cluster_threshold: 8,
            max_expanded_leaves: None,
        }
    }

    pub fn with_rerank(mut self, rerank_candidates: usize) -> Self {
        self.rerank_candidates = Some(rerank_candidates);
        self
    }

    pub fn with_exhaustive_cluster_threshold(mut self, threshold: usize) -> Self {
        self.exhaustive_cluster_threshold = threshold;
        self
    }

    pub fn with_max_expanded_leaves(mut self, max_leaves: usize) -> Self {
        self.max_expanded_leaves = Some(max_leaves);
        self
    }
}

/// Approximate nearest neighbor search using in-memory beam traversal,
/// batched zero-copy RaBitQ leaf scanning, and optional batched full-vector reranking.
pub fn search(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    meta: &TreeMeta,
    quantizer: &RabitqQuantizer,
    query: &[f32],
    opts: SearchOptions,
    config: &TreeConfig,
) -> Result<Vec<(DocId, f32)>> {
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

    let mut scratch = QueryScratch::new(padded_dim, rabitq_words);
    quantizer.rotate_into(&q, &mut scratch.rot_q);

    let dot_q_meta = if config.metric == DistanceMetric::DotProduct {
        Some(pack_query_bits_into(&scratch.rot_q, &mut scratch.planes))
    } else {
        None
    };

    let mut candidates: Vec<LeafCandidate> =
        Vec::with_capacity(leaf_ids.len() * config.max_leaf_size);

    for (leaf_id, opt_slice) in leaf_ids.iter().copied().zip(leaf_slices.iter()) {
        let slice = opt_slice.as_ref().ok_or_else(|| {
            TreeError::CorruptedData(format!("missing leaf blob for leaf {leaf_id}"))
        })?;
        let view = LeafNodeView::parse(slice, dim, padded_dim, rabitq_words)?;
        if view.entry_count == 0 {
            continue;
        }
        score_leaf_view(
            &view,
            None,
            &q,
            &mut scratch,
            dot_q_meta,
            inv_sqrt_padded_dim,
            rabitq_words,
            config.metric,
            &mut candidates,
        );
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

/// Filtered approximate nearest neighbor search with cluster-level pruning,
/// exhaustive-scan crossover for highly selective filters, best-first dynamic expansion,
/// and masked RaBitQ scoring.
#[allow(clippy::too_many_arguments)]
pub fn search_filtered(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &HashMap<NodeId, NodeId>,
    meta: &TreeMeta,
    quantizer: &RabitqQuantizer,
    schema: &CollectionSchema,
    attr_index: &AttributeIndex,
    query: &[f32],
    opts: SearchOptions,
    filter: &Filter,
    config: &TreeConfig,
) -> Result<Vec<(DocId, f32)>> {
    if opts.beam_width == 0 {
        return Err(TreeError::InvalidConfig(
            "beam_width must be >= 1".into(),
        ));
    }
    filter.validate(schema)?;
    let q = prepare_vector_cow(query, config.dim, config.metric)?;
    if opts.k == 0 || meta.num_vectors == 0 {
        return Ok(Vec::new());
    }

    // Step 1: Cluster-level pruning via cf_attr_clusters.
    let prepared = attr_index.prepare_filter(storage, schema, filter)?;
    if prepared.cluster_candidates.is_empty() {
        return Ok(Vec::new());
    }

    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();
    let rabitq_words = quantizer.num_words();

    // Step 2: Exhaustive-scan crossover for highly selective filters.
    if let ClusterCandidates::Subset(ref bm) = prepared.cluster_candidates
        && bm.len() as usize <= opts.exhaustive_cluster_threshold
    {
        let leaf_ids: Vec<NodeId> = bm.iter().collect();
        let leaf_slices = storage.multi_get_leaf_nodes_pinned(&leaf_ids)?;
        let mut views: Vec<Option<LeafNodeView<'_>>> = Vec::with_capacity(leaf_ids.len());
        for (leaf_id, opt_slice) in leaf_ids.iter().copied().zip(leaf_slices.iter()) {
            let slice = opt_slice.as_ref().ok_or_else(|| {
                TreeError::CorruptedData(format!("missing leaf blob for leaf {leaf_id}"))
            })?;
            views.push(Some(LeafNodeView::parse(slice, dim, padded_dim, rabitq_words)?));
        }

        let leaf_pairs: Vec<(NodeId, Option<&LeafNodeView<'_>>)> = leaf_ids
            .iter()
            .copied()
            .zip(views.iter().map(|v| v.as_ref()))
            .collect();
        let masks = attr_index.eval_leaves_masks(storage, &prepared, &leaf_pairs)?;

        let mut matching_doc_ids: Vec<DocId> = Vec::new();
        for (view_opt, mask) in views.iter().zip(masks.iter()) {
            if mask.is_empty() {
                continue;
            }
            if let Some(view) = view_opt {
                for slot in 0..view.entry_count {
                    if mask.contains(view.local_id(slot)) {
                        matching_doc_ids.push(view.doc_id(slot));
                    }
                }
            }
        }
        if matching_doc_ids.is_empty() {
            return Ok(Vec::new());
        }

        matching_doc_ids.sort_unstable();
        matching_doc_ids.dedup();
        let vec_slices = storage.multi_get_vectors_pinned(&matching_doc_ids)?;
        let expected_bytes = dim * 4;
        let mut exact_results: Vec<(DocId, f32)> = Vec::with_capacity(matching_doc_ids.len());
        for (doc_id, opt_slice) in matching_doc_ids.iter().copied().zip(vec_slices.iter()) {
            let slice = opt_slice.as_ref().ok_or_else(|| {
                TreeError::CorruptedData(format!("missing vector {doc_id} in cf_vectors"))
            })?;
            if slice.len() != expected_bytes {
                return Err(TreeError::CorruptedData(format!(
                    "vector {doc_id} expected {expected_bytes} bytes, got {}",
                    slice.len()
                )));
            }
            exact_results.push((doc_id, search_distance_bytes(config.metric, &q, slice)));
        }
        select_top_k(&mut exact_results, opts.k, cmp_dist_id);
        return Ok(exact_results);
    }

    // Step 3: Pruned best-first traversal + dynamic cluster expansion.
    let allowed_leaves: Option<&RoaringBitmap> = match prepared.cluster_candidates {
        ClusterCandidates::All => None,
        ClusterCandidates::Subset(ref bm) => Some(bm),
    };

    let total_leaves = leaf_parent.len().max(1);
    let allowed_internal: Option<RoaringBitmap> = match allowed_leaves {
        Some(bm) if (bm.len() as usize) * 2 < total_leaves && meta.root_level > 0 => {
            let mut internal_bm = RoaringBitmap::new();
            for leaf_id in bm.iter() {
                if let Some(&p1) = leaf_parent.get(&leaf_id)
                    && internal_bm.insert(p1)
                {
                    let mut cur = p1;
                    while let Some(node) = internal_cache.get(&cur)
                        && let Some(pid) = node.parent_id
                    {
                        if !internal_bm.insert(pid) {
                            break;
                        }
                        cur = pid;
                    }
                }
            }
            Some(internal_bm)
        }
        _ => None,
    };

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let root_allowed = if meta.root_level == 0 {
        allowed_leaves.is_none_or(|bm| bm.contains(meta.root_id))
    } else {
        allowed_internal
            .as_ref()
            .is_none_or(|bm| bm.contains(meta.root_id))
    };
    if root_allowed {
        heap.push(HeapEntry {
            dist: 0.0,
            node_id: meta.root_id,
            level: meta.root_level,
        });
    }

    let inv_sqrt_padded_dim = quantizer.inv_sqrt_padded_dim();
    let mut scratch = QueryScratch::new(padded_dim, rabitq_words);
    quantizer.rotate_into(&q, &mut scratch.rot_q);
    let dot_q_meta = if config.metric == DistanceMetric::DotProduct {
        Some(pack_query_bits_into(&scratch.rot_q, &mut scratch.planes))
    } else {
        None
    };

    let target = opts.rerank_candidates.unwrap_or(opts.k).max(opts.k);
    let max_expanded = opts
        .max_expanded_leaves
        .unwrap_or(opts.beam_width.saturating_mul(16))
        .max(opts.beam_width);

    let mut probed_leaves = 0usize;
    let mut batch_target = opts.beam_width;
    let mut candidates: Vec<LeafCandidate> = Vec::with_capacity(target.max(opts.beam_width * 16));
    let mut batch_leaves: Vec<NodeId> = Vec::new();
    let mut slot_mask_words: Vec<u64> = Vec::new();

    while probed_leaves < max_expanded && !heap.is_empty() {
        let take_limit = batch_target.min(max_expanded - probed_leaves);
        batch_leaves.clear();
        while batch_leaves.len() < take_limit
            && let Some(item) = heap.pop()
        {
            if item.level == 0 {
                batch_leaves.push(item.node_id);
            } else {
                let node = internal_cache.get(&item.node_id).ok_or_else(|| {
                    TreeError::CorruptedData(format!(
                        "missing internal node {} in cache",
                        item.node_id
                    ))
                })?;
                let child_level = item.level - 1;
                for (i, &child_id) in node.child_ids.iter().enumerate() {
                    let allowed = if child_level == 0 {
                        allowed_leaves.is_none_or(|bm| bm.contains(child_id))
                    } else {
                        allowed_internal
                            .as_ref()
                            .is_none_or(|bm| bm.contains(child_id))
                    };
                    if allowed {
                        let d = search_distance(config.metric, &q, node.child_centroid(i, dim));
                        heap.push(HeapEntry {
                            dist: d,
                            node_id: child_id,
                            level: child_level,
                        });
                    }
                }
            }
        }

        if batch_leaves.is_empty() {
            break;
        }
        probed_leaves += batch_leaves.len();

        batch_leaves.sort_unstable();
        batch_leaves.dedup();
        let leaf_slices = storage.multi_get_leaf_nodes_pinned(&batch_leaves)?;

        let mut parsed_views: Vec<Option<LeafNodeView<'_>>> = Vec::with_capacity(batch_leaves.len());
        for (&leaf_id, opt_slice) in batch_leaves.iter().zip(leaf_slices.iter()) {
            let slice = opt_slice.as_ref().ok_or_else(|| {
                TreeError::CorruptedData(format!("missing leaf blob for leaf {leaf_id}"))
            })?;
            parsed_views.push(Some(LeafNodeView::parse(slice, dim, padded_dim, rabitq_words)?));
        }

        let leaf_pairs: Vec<(NodeId, Option<&LeafNodeView<'_>>)> = batch_leaves
            .iter()
            .copied()
            .zip(parsed_views.iter().map(|v| v.as_ref()))
            .collect();
        let masks = attr_index.eval_leaves_masks(storage, &prepared, &leaf_pairs)?;

        for (view_opt, local_mask) in parsed_views.iter().zip(masks.iter()) {
            if local_mask.is_empty() {
                continue;
            }
            let Some(view) = view_opt else {
                continue;
            };
            if view.entry_count == 0 {
                continue;
            }
            let word_count = view.entry_count.div_ceil(64);
            slot_mask_words.clear();
            slot_mask_words.resize(word_count, 0);
            let mut any_match = false;
            for slot in 0..view.entry_count {
                if local_mask.contains(view.local_id(slot)) {
                    slot_mask_words[slot / 64] |= 1u64 << (slot % 64);
                    any_match = true;
                }
            }
            if !any_match {
                continue;
            }

            score_leaf_view(
                view,
                Some(&slot_mask_words),
                &q,
                &mut scratch,
                dot_q_meta,
                inv_sqrt_padded_dim,
                rabitq_words,
                config.metric,
                &mut candidates,
            );
        }

        if (candidates.len() >= target && probed_leaves >= opts.beam_width)
            || heap.is_empty()
            || probed_leaves >= max_expanded
        {
            break;
        }
        batch_target = batch_target.saturating_mul(2);
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

#[allow(clippy::too_many_arguments)]
fn score_leaf_view(
    view: &LeafNodeView<'_>,
    allowed_slots: Option<&[u64]>,
    q: &[f32],
    scratch: &mut QueryScratch,
    dot_q_meta: Option<crate::rabitq::QuantizedQueryMeta>,
    inv_sqrt_padded_dim: f32,
    rabitq_words: usize,
    metric: DistanceMetric,
    candidates: &mut Vec<LeafCandidate>,
) {
    let (q_res_norm_sq, zero_res_dist, q_meta) = match metric {
        DistanceMetric::L2Squared => {
            let d2 = simd::l2_squared_bytes(q, view.centroid_bytes);
            let meta = pack_query_bits_from_diff_bytes_into(
                &scratch.rot_q,
                view.rotated_centroid_bytes,
                &mut scratch.rot_diff,
                &mut scratch.planes,
            );
            (d2, d2, meta)
        }
        DistanceMetric::Cosine => {
            let d2 = simd::l2_squared_bytes(q, view.centroid_bytes);
            let zero = (1.0 - simd::dot_product_bytes(q, view.centroid_bytes)).max(0.0);
            let meta = pack_query_bits_from_diff_bytes_into(
                &scratch.rot_q,
                view.rotated_centroid_bytes,
                &mut scratch.rot_diff,
                &mut scratch.planes,
            );
            (d2, zero, meta)
        }
        DistanceMetric::DotProduct => {
            let zero = -simd::dot_product_bytes(q, view.centroid_bytes);
            (0.0, zero, dot_q_meta.unwrap())
        }
    };

    simd::score_leaf_rabitq_masked(
        &simd::RabitqLeafScoreParams {
            entry_count: view.entry_count,
            rabitq_words,
            doc_ids_bytes: view.doc_ids_bytes,
            codes_bytes: view.codes_bytes,
            res_norm_sqs_bytes: view.res_norm_sqs_bytes,
            scales_bytes: view.scales_bytes,
            q0: &scratch.planes[0],
            q1: &scratch.planes[1],
            q2: &scratch.planes[2],
            q3: &scratch.planes[3],
            vl: q_meta.vl,
            delta: q_meta.delta,
            sum_q: q_meta.sum_q,
            inv_sqrt_padded_dim,
            q_res_norm_sq,
            zero_res_dist,
            metric,
        },
        allowed_slots,
        candidates,
    );
}

fn finalize_candidates(
    storage: &Storage,
    q: &[f32],
    mut candidates: Vec<LeafCandidate>,
    k: usize,
    rerank_candidates: Option<usize>,
    dim: usize,
    metric: DistanceMetric,
) -> Result<Vec<(DocId, f32)>> {
    let Some(rerank_k) = rerank_candidates else {
        select_top_k(&mut candidates, k, cmp_cand_dist_id);
        return Ok(candidates.into_iter().map(|c| (c.doc_id, c.dist)).collect());
    };

    let shortlist_len = rerank_k.max(k);
    select_top_k(&mut candidates, shortlist_len, cmp_cand_dist_id);
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    candidates.sort_unstable_by_key(|c| c.doc_id);
    let sorted_doc_ids: Vec<DocId> = candidates.iter().map(|c| c.doc_id).collect();
    let vec_slices = storage.multi_get_vectors_pinned(&sorted_doc_ids)?;

    let expected_bytes = dim * 4;
    let mut reranked: Vec<(DocId, f32)> = Vec::with_capacity(candidates.len());
    for (doc_id, opt_slice) in sorted_doc_ids.iter().copied().zip(vec_slices.iter()) {
        let slice = opt_slice.as_ref().ok_or_else(|| {
            TreeError::CorruptedData(format!("missing vector {doc_id} in cf_vectors"))
        })?;
        if slice.len() != expected_bytes {
            return Err(TreeError::CorruptedData(format!(
                "vector {doc_id} expected {expected_bytes} bytes, got {}",
                slice.len()
            )));
        }
        reranked.push((doc_id, search_distance_bytes(metric, q, slice)));
    }
    select_top_k(&mut reranked, k, cmp_dist_id);
    Ok(reranked)
}

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    dist: f32,
    node_id: NodeId,
    level: u8,
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

#[inline]
fn select_top_k<T, F>(items: &mut Vec<T>, limit: usize, cmp: F)
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
fn cmp_dist_id(a: &(DocId, f32), b: &(DocId, f32)) -> Ordering {
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
