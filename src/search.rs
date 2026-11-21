use std::cmp::Ordering;
use std::collections::HashMap;

use crate::config::{DistanceMetric, DocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::metric::{cluster_distance, prepare_vector_cow, search_distance, search_distance_bytes};
use crate::node::{LeafNodeView, LeafVectorsView, PackedInternalNode, TreeMeta};
use crate::rabitq::{
    QueryScratch, RabitqQuantizer, pack_query_bits_from_diff_bytes_into, pack_query_bits_into,
};
use crate::simd::{self, LeafCandidate};
use crate::storage::Storage;

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

/// Find the top `count` closest leaves to `prepared_vec` using in-memory `cluster_distance` beam search.
pub fn search_leaves(
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    meta: &TreeMeta,
    prepared_vec: &[f32],
    count: usize,
    beam_width: usize,
    config: &TreeConfig,
) -> Result<Vec<(NodeId, f32)>> {
    let beam = beam_width.max(count).max(1);
    let mut frontier: Vec<(NodeId, f32)> = vec![(meta.root_id, 0.0)];
    let mut next: Vec<(NodeId, f32)> = Vec::new();
    let mut level = meta.root_level;
    let dim = config.dim;

    while level > 0 {
        next.clear();
        for &(node_id, _) in &frontier {
            let node = internal_cache.get(&node_id).ok_or_else(|| {
                TreeError::CorruptedData(format!("missing internal node {node_id} in cache"))
            })?;
            for (i, &child_id) in node.child_ids.iter().enumerate() {
                let d = cluster_distance(config.metric, prepared_vec, node.child_centroid(i, dim));
                next.push((child_id, d));
            }
        }
        select_top_k(&mut next, beam, cmp_dist_id);
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

/// Parameters controlling quantized search and optional full-vector reranking.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    pub k: usize,
    pub beam_width: usize,
    pub rerank_candidates: Option<usize>,
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
    let SearchOptions {
        k,
        beam_width,
        rerank_candidates,
    } = opts;
    if beam_width == 0 {
        return Err(TreeError::InvalidConfig(
            "beam_width must be >= 1".into(),
        ));
    }
    let q = prepare_vector_cow(query, config.dim, config.metric)?;
    if k == 0 || meta.num_vectors == 0 {
        return Ok(Vec::new());
    }

    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();
    let rabitq_words = quantizer.num_words();
    let inv_sqrt_padded_dim = quantizer.inv_sqrt_padded_dim();

    let mut frontier: Vec<(NodeId, f32)> = Vec::with_capacity(beam_width);
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
        select_top_k(&mut next, beam_width, cmp_dist_id);
        if next.is_empty() {
            return Err(TreeError::CorruptedData(format!(
                "internal frontier empty during search at level {level}"
            )));
        }
        std::mem::swap(&mut frontier, &mut next);
        level -= 1;
    }

    // Sort leaf IDs once so RocksDB `batched_multi_get_cf` receives sorted keys.
    let mut leaf_ids: Vec<NodeId> = frontier.iter().map(|&(id, _)| id).collect();
    leaf_ids.sort_unstable();
    leaf_ids.dedup();

    let leaf_slices = storage.multi_get_leaf_nodes_pinned(&leaf_ids)?;

    // Rotate query vector once per search (O(D log D)), instead of once per leaf.
    let mut scratch = QueryScratch::new(padded_dim, rabitq_words);
    quantizer.rotate_into(&q, &mut scratch.rot_q);

    let dot_q_meta = if config.metric == DistanceMetric::DotProduct {
        Some(pack_query_bits_into(&scratch.rot_q, &mut scratch.planes))
    } else {
        None
    };

    let mut candidates: Vec<LeafCandidate> = Vec::with_capacity(leaf_ids.len() * config.max_leaf_size);

    for (leaf_id, opt_slice) in leaf_ids.iter().copied().zip(leaf_slices.iter()) {
        let slice = opt_slice.as_ref().ok_or_else(|| {
            TreeError::CorruptedData(format!("missing leaf blob for leaf {leaf_id}"))
        })?;
        let view = LeafNodeView::parse(slice, dim, padded_dim, rabitq_words)?;
        if view.entry_count == 0 {
            continue;
        }

        let (q_res_norm_sq, zero_res_dist, q_meta) = match config.metric {
            DistanceMetric::L2Squared => {
                let d2 = simd::l2_squared_bytes(&q, view.centroid_bytes);
                let meta = pack_query_bits_from_diff_bytes_into(
                    &scratch.rot_q,
                    view.rotated_centroid_bytes,
                    &mut scratch.rot_diff,
                    &mut scratch.planes,
                );
                (d2, d2, meta)
            }
            DistanceMetric::Cosine => {
                let d2 = simd::l2_squared_bytes(&q, view.centroid_bytes);
                let zero = (1.0 - simd::dot_product_bytes(&q, view.centroid_bytes)).max(0.0);
                let meta = pack_query_bits_from_diff_bytes_into(
                    &scratch.rot_q,
                    view.rotated_centroid_bytes,
                    &mut scratch.rot_diff,
                    &mut scratch.planes,
                );
                (d2, zero, meta)
            }
            DistanceMetric::DotProduct => {
                let zero = -simd::dot_product_bytes(&q, view.centroid_bytes);
                (0.0, zero, dot_q_meta.unwrap())
            }
        };

        simd::score_leaf_rabitq(
            &simd::RabitqLeafScoreParams {
                leaf_id,
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
                metric: config.metric,
            },
            &mut candidates,
        );
    }

    let Some(rerank_k) = rerank_candidates else {
        select_top_k(&mut candidates, k, cmp_cand_dist_id);
        return Ok(candidates.into_iter().map(|c| (c.doc_id, c.dist)).collect());
    };

    let shortlist_len = rerank_k.max(k);
    select_top_k(&mut candidates, shortlist_len, cmp_cand_dist_id);
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Fetch only the distinct leaves present in the shortlist (at most `beam_width`).
    let mut vec_leaf_ids: Vec<NodeId> = Vec::with_capacity(leaf_ids.len());
    for c in &candidates {
        if !vec_leaf_ids.contains(&c.leaf_id) {
            vec_leaf_ids.push(c.leaf_id);
        }
    }
    vec_leaf_ids.sort_unstable();

    let vec_slices = storage.multi_get_leaf_vectors_pinned(&vec_leaf_ids)?;
    let mut vec_views: Vec<(NodeId, LeafVectorsView<'_>)> = Vec::with_capacity(vec_leaf_ids.len());
    for (lid, opt_slice) in vec_leaf_ids.iter().copied().zip(vec_slices.iter()) {
        let slice = opt_slice.as_ref().ok_or_else(|| {
            TreeError::CorruptedData(format!("missing leaf vectors blob for leaf {lid}"))
        })?;
        vec_views.push((lid, LeafVectorsView::parse(slice, dim)?));
    }

    let mut reranked: Vec<(DocId, f32)> = Vec::with_capacity(candidates.len());
    for c in &candidates {
        let (_, view) = vec_views
            .iter()
            .find(|&&(lid, _)| lid == c.leaf_id)
            .expect("shortlist leaf must exist in vec_views");
        let vec_bytes = view.vec_bytes_at(c.slot as usize, dim);
        reranked.push((c.doc_id, search_distance_bytes(config.metric, &q, vec_bytes)));
    }
    select_top_k(&mut reranked, k, cmp_dist_id);
    Ok(reranked)
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
fn cmp_dist_id(a: &(u64, f32), b: &(u64, f32)) -> Ordering {
    a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
}

#[inline]
fn cmp_cand_dist_id(a: &LeafCandidate, b: &LeafCandidate) -> Ordering {
    a.dist.total_cmp(&b.dist).then_with(|| a.doc_id.cmp(&b.doc_id))
}
