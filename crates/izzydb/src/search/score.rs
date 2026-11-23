//! Leaf scoring, candidate finalization, and top-k selection.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use izzydb_core::config::{DistanceMetric, NodeId, RowId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::metric::search_distance_bytes;
use izzydb_rabitq::{
    QuantizedQueryMeta, QueryScratch, RabitqQuantizer, pack_query_bits_from_diff_bytes_into,
};
use izzydb_storage::node::LeafNodeView;
use izzydb_storage::{DBPinnableSlice, Storage};
use crate::simd::{self, LeafCandidate};


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
pub(super) fn cmp_dist_row(a: &(RowId, f32), b: &(RowId, f32)) -> Ordering {
    a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
}

#[inline]
pub(super) fn cmp_node_dist_id(a: &(NodeId, f32), b: &(NodeId, f32)) -> Ordering {
    a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
}

#[inline]
pub(super) fn cmp_cand_dist_id(a: &LeafCandidate, b: &LeafCandidate) -> Ordering {
    a.dist.total_cmp(&b.dist).then_with(|| a.doc_id.cmp(&b.doc_id))
}
