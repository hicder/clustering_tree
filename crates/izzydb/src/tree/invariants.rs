//! Tree invariant verification.


use std::collections::{HashMap, HashSet};

use izzydb_core::config::{NodeId, RowId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_storage::node::{AnnMeta, PackedInternalNode, PackedLeafNode};
use izzydb_storage::Storage;

use crate::TreeStats;



/// Exhaustively verify ANN structural invariants over `RowId`s and `SegmentTableSnapshot`.
pub fn verify_tree_invariants(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &HashMap<NodeId, NodeId>,
    mem_meta: &AnnMeta,
    config: &TreeConfig,
) -> Result<TreeStats> {
    let disk_meta = storage
        .get_ann_meta()?
        .ok_or_else(|| TreeError::InvariantViolation("missing \\x01ann_meta on disk".into()))?;
    if &disk_meta != mem_meta {
        return Err(TreeError::InvariantViolation(
            "in-memory AnnMeta differs from disk".into(),
        ));
    }
    if disk_meta.dim as usize != config.dim {
        return Err(TreeError::InvariantViolation(format!(
            "AnnMeta dim {} != config.dim {}",
            disk_meta.dim, config.dim
        )));
    }
    if disk_meta.metric != config.metric {
        return Err(TreeError::InvariantViolation(format!(
            "AnnMeta metric {:?} != config.metric {:?}",
            disk_meta.metric, config.metric
        )));
    }
    if disk_meta.root_id >= disk_meta.next_node_id {
        return Err(TreeError::InvariantViolation(format!(
            "root_id {} >= next_node_id {}",
            disk_meta.root_id, disk_meta.next_node_id
        )));
    }

    let disk_internals: HashMap<NodeId, PackedInternalNode> =
        storage.all_internal_nodes()?.into_iter().collect();
    if &disk_internals != internal_cache {
        return Err(TreeError::InvariantViolation(
            "in-memory internal_cache differs from cf_internal_nodes on disk".into(),
        ));
    }

    let disk_leaves: HashMap<NodeId, PackedLeafNode> =
        storage.all_leaf_nodes()?.into_iter().collect();
    let disk_vectors: HashMap<RowId, Vec<f32>> = storage.all_vectors()?.into_iter().collect();
    let seg_snap = storage.segment_snapshot();
    let dropped = storage.dropped_segments_snapshot();

    let mut visited_internal = HashSet::new();
    let mut visited_leaves = HashSet::new();
    let mut expected_leaf_parent = HashMap::new();
    let mut seen_live_rows: HashSet<RowId> = HashSet::new();
    let mut stale_count = 0u64;
    let mut min_leaf = usize::MAX;
    let mut max_leaf = 0usize;
    let mut min_internal = usize::MAX;
    let mut max_internal = 0usize;

    let mut stack = vec![(disk_meta.root_id, disk_meta.root_level, None)];
    while let Some((node_id, expected_level, expected_parent)) = stack.pop() {
        if visited_internal.contains(&node_id) || visited_leaves.contains(&node_id) {
            return Err(TreeError::InvariantViolation(format!(
                "cycle or shared node detected at {node_id}"
            )));
        }
        let is_root = node_id == disk_meta.root_id;
        if expected_level == 0 {
            visited_leaves.insert(node_id);
            if let Some(pid) = expected_parent {
                expected_leaf_parent.insert(node_id, pid);
            }
            if disk_internals.contains_key(&node_id) {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} present in internal nodes"
                )));
            }
            let leaf = disk_leaves.get(&node_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!("missing leaf {node_id}"))
            })?;
            if leaf.parent_id != expected_parent {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} parent_id {:?} != expected {expected_parent:?}",
                    leaf.parent_id
                )));
            }
            if leaf.centroid.len() != config.dim || leaf.centroid.iter().any(|x| !x.is_finite()) {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} invalid centroid"
                )));
            }
            let count = leaf.entry_count();
            if leaf.slot_cap > config.max_leaf_size {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} slot_cap {} > max_leaf_size {}",
                    leaf.slot_cap, config.max_leaf_size
                )));
            }
            if is_root {
                if count > config.max_leaf_size {
                    return Err(TreeError::InvariantViolation(format!(
                        "root leaf {node_id} count {count} > max_leaf_size {}",
                        config.max_leaf_size
                    )));
                }
            } else if count < config.min_leaf_size || count > config.max_leaf_size {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} count {count} not in {}..={}",
                    config.min_leaf_size, config.max_leaf_size
                )));
            }
            min_leaf = min_leaf.min(count);
            max_leaf = max_leaf.max(count);

            let mut slot_live = 0usize;
            for slot in 0..leaf.slot_cap {
                if !leaf.is_live(slot) {
                    continue;
                }
                slot_live += 1;
                let raw_row = leaf.row_id(slot);
                match seg_snap.resolve_row(raw_row) {
                    Some(resolved) if !dropped.contains(resolved.seg()) => {
                        let vec = disk_vectors.get(&resolved).ok_or_else(|| {
                            TreeError::InvariantViolation(format!(
                                "missing vector for live {resolved:?} in cf_vectors"
                            ))
                        })?;
                        if vec.len() != config.dim || vec.iter().any(|x| !x.is_finite()) {
                            return Err(TreeError::InvariantViolation(format!(
                                "invalid vector for {resolved:?}"
                            )));
                        }
                        if !seen_live_rows.insert(resolved) {
                            return Err(TreeError::InvariantViolation(format!(
                                "duplicate live row {resolved:?} in ANN leaves"
                            )));
                        }
                    }
                    _ => {
                        stale_count += 1;
                    }
                }
                let res_norm_sq = leaf.res_norm_sqs[slot];
                let scale = leaf.scales[slot];
                if !res_norm_sq.is_finite()
                    || res_norm_sq < 0.0
                    || !scale.is_finite()
                    || scale < 0.0
                    || (res_norm_sq > 0.0 && scale <= 0.0)
                {
                    return Err(TreeError::InvariantViolation(format!(
                        "row {raw_row:?} in leaf {node_id} has invalid quantized entry"
                    )));
                }
            }
            if slot_live != count {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} live_mask popcount {slot_live} != live_count {count}"
                )));
            }
        } else {
            visited_internal.insert(node_id);
            if disk_leaves.contains_key(&node_id) {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} present in cf_leaf_entries"
                )));
            }
            let node = disk_internals.get(&node_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!("missing internal node {node_id}"))
            })?;
            if node.level != expected_level || node.parent_id != expected_parent {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} level/parent mismatch"
                )));
            }
            if node.centroid.len() != config.dim || node.centroid.iter().any(|x| !x.is_finite()) {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} invalid centroid"
                )));
            }
            let count = node.entry_count();
            if is_root {
                if count < 2 || count > config.max_internal_fanout {
                    return Err(TreeError::InvariantViolation(format!(
                        "internal root {node_id} has {count} children"
                    )));
                }
            } else if count < config.min_internal_fanout || count > config.max_internal_fanout {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} has {count} children"
                )));
            }
            min_internal = min_internal.min(count);
            max_internal = max_internal.max(count);
            let sub_sum: u64 = node.child_subtree_counts.iter().sum();
            if node.subtree_count != sub_sum {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} subtree_count {} != child sum {sub_sum}",
                    node.subtree_count
                )));
            }
            for &cid in &node.child_ids {
                stack.push((cid, expected_level - 1, Some(node_id)));
            }
        }
    }

    if leaf_parent != &expected_leaf_parent {
        return Err(TreeError::InvariantViolation(
            "in-memory leaf_parent cache differs from disk".into(),
        ));
    }
    if visited_internals_or_leaves_orphaned(&disk_internals, &disk_leaves, &visited_internal, &visited_leaves) {
        return Err(TreeError::InvariantViolation(
            "orphaned tree nodes unreachable from root".into(),
        ));
    }
    if seen_live_rows.len() as u64 != disk_meta.num_vectors {
        return Err(TreeError::InvariantViolation(format!(
            "live ANN rows {} != AnnMeta.num_vectors {}",
            seen_live_rows.len(),
            disk_meta.num_vectors
        )));
    }
    if stale_count != disk_meta.stale_rows {
        return Err(TreeError::InvariantViolation(format!(
            "stale ANN slots {stale_count} != AnnMeta.stale_rows {}",
            disk_meta.stale_rows
        )));
    }

    Ok(TreeStats {
        num_vectors: disk_meta.num_vectors,
        stale_rows: disk_meta.stale_rows,
        root_id: disk_meta.root_id,
        root_level: disk_meta.root_level,
        total_nodes: visited_internal.len() + visited_leaves.len(),
        internal_nodes: visited_internal.len(),
        leaf_nodes: visited_leaves.len(),
        min_leaf_entries: if min_leaf == usize::MAX { 0 } else { min_leaf },
        max_leaf_entries: max_leaf,
        min_internal_fanout: if min_internal == usize::MAX { 0 } else { min_internal },
        max_internal_fanout: max_internal,
    })
}

pub(super) fn visited_internals_or_leaves_orphaned(
    disk_internals: &HashMap<NodeId, PackedInternalNode>,
    disk_leaves: &HashMap<NodeId, PackedLeafNode>,
    visited_internal: &HashSet<NodeId>,
    visited_leaves: &HashSet<NodeId>,
) -> bool {
    disk_internals.keys().any(|id| !visited_internal.contains(id))
        || disk_leaves.keys().any(|id| !visited_leaves.contains(id))
}
