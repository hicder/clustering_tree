//! Unfiltered ANN search.

use std::collections::HashMap;
use izzydb_core::config::{DistanceMetric, NodeId, RowId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::metric::{prepare_vector_cow, search_distance};
use izzydb_rabitq::{
    QueryScratch, RabitqQuantizer,
    pack_query_bits_into,
};
use izzydb_storage::node::{AnnMeta, LeafNodeView, PackedInternalNode};
use izzydb_storage::Storage;
use crate::simd::LeafCandidate;

use super::*;

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
