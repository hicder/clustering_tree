use std::cmp::Ordering;

use crate::config::{DocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::metric::{cluster_distance, prepare_vector, search_distance};
use crate::node::TreeMeta;
use crate::storage::Storage;

/// Route a prepared (validated and normalized if Cosine) vector to the single closest leaf.
pub fn route_to_leaf(
    storage: &Storage,
    meta: &TreeMeta,
    prepared_vec: &[f32],
    beam_width: usize,
    config: &TreeConfig,
) -> Result<NodeId> {
    let leaves = search_leaves(storage, meta, prepared_vec, 1, beam_width, config)?;
    leaves
        .into_iter()
        .next()
        .map(|(id, _)| id)
        .ok_or_else(|| TreeError::CorruptedData("route_to_leaf reached empty frontier".into()))
}

/// Find the top `count` closest leaves to `prepared_vec` using `cluster_distance` beam search.
pub fn search_leaves(
    storage: &Storage,
    meta: &TreeMeta,
    prepared_vec: &[f32],
    count: usize,
    beam_width: usize,
    config: &TreeConfig,
) -> Result<Vec<(NodeId, f32)>> {
    let beam = beam_width.max(count).max(1);
    let mut frontier: Vec<(NodeId, f32)> = vec![(meta.root_id, 0.0)];
    let mut level = meta.root_level;

    while level > 0 {
        let mut next: Vec<(NodeId, f32)> = Vec::new();
        for &(node_id, _) in &frontier {
            storage.scan_internal_with(node_id, |child_id, _subtree_count, centroid| {
                let d = cluster_distance(config.metric, prepared_vec, centroid);
                next.push((child_id, d));
            })?;
        }
        sort_and_truncate(&mut next, beam);
        if next.is_empty() {
            return Err(TreeError::CorruptedData(format!(
                "internal frontier empty at level {level}"
            )));
        }
        frontier = next;
        level -= 1;
    }

    if frontier.len() > count {
        frontier.truncate(count);
    }
    Ok(frontier)
}

/// Approximate nearest neighbor search returning up to `k` `(DocId, distance)` pairs
/// sorted by ascending distance.
pub fn search(
    storage: &Storage,
    meta: &TreeMeta,
    query: &[f32],
    k: usize,
    beam_width: usize,
    config: &TreeConfig,
) -> Result<Vec<(DocId, f32)>> {
    if beam_width == 0 {
        return Err(TreeError::InvalidConfig(
            "beam_width must be >= 1".into(),
        ));
    }
    let q = prepare_vector(query, config.dim, config.metric)?;
    if k == 0 || meta.num_vectors == 0 {
        return Ok(Vec::new());
    }

    let mut frontier: Vec<(NodeId, f32)> = vec![(meta.root_id, 0.0)];
    let mut level = meta.root_level;

    while level > 0 {
        let mut next: Vec<(NodeId, f32)> = Vec::new();
        for &(node_id, _) in &frontier {
            storage.scan_internal_with(node_id, |child_id, _subtree_count, centroid| {
                let d = search_distance(config.metric, &q, centroid);
                next.push((child_id, d));
            })?;
        }
        sort_and_truncate(&mut next, beam_width);
        if next.is_empty() {
            return Err(TreeError::CorruptedData(format!(
                "internal frontier empty during search at level {level}"
            )));
        }
        frontier = next;
        level -= 1;
    }

    let mut candidates: Vec<(DocId, f32)> = Vec::new();
    for &(leaf_id, _) in &frontier {
        storage.scan_leaf_with(leaf_id, |doc_id, vec| {
            let d = search_distance(config.metric, &q, vec);
            candidates.push((doc_id, d));
        })?;
    }

    sort_and_truncate(&mut candidates, k);
    Ok(candidates)
}

#[inline]
fn sort_and_truncate(items: &mut Vec<(u64, f32)>, limit: usize) {
    items.sort_by(cmp_dist_id);
    if items.len() > limit {
        items.truncate(limit);
    }
}

#[inline]
fn cmp_dist_id(a: &(u64, f32), b: &(u64, f32)) -> Ordering {
    a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
}
