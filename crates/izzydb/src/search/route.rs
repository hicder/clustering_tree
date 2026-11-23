//! Leaf routing and leaf search traversal.

use std::collections::HashMap;
use izzydb_core::config::{NodeId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::metric::cluster_distance;
use izzydb_storage::node::{AnnMeta, PackedInternalNode};
use izzydb_storage::TxContext;

use super::*;

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
