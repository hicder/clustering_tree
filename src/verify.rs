use std::collections::{HashMap, HashSet};

use crate::config::{DocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::node::{PackedInternalNode, TreeMeta};
use crate::storage::Storage;

/// Summary statistics collected by [`crate::ClusteringTree::verify_invariants`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeStats {
    pub num_vectors: u64,
    pub root_id: NodeId,
    pub root_level: u8,
    pub total_nodes: usize,
    pub internal_nodes: usize,
    pub leaf_nodes: usize,
    pub min_leaf_entries: usize,
    pub max_leaf_entries: usize,
    pub min_internal_fanout: usize,
    pub max_internal_fanout: usize,
}

/// Exhaustively verify all structural invariants of the tree in RocksDB and in-memory cache.
pub fn verify_invariants(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    mem_meta: &TreeMeta,
    config: &TreeConfig,
) -> Result<TreeStats> {
    let disk_meta = storage
        .get_meta()?
        .ok_or_else(|| TreeError::InvariantViolation("missing TreeMeta in cf_meta".into()))?;
    if &disk_meta != mem_meta {
        return Err(TreeError::InvariantViolation(format!(
            "in-memory TreeMeta {mem_meta:?} differs from disk {disk_meta:?}"
        )));
    }
    if disk_meta.dim as usize != config.dim {
        return Err(TreeError::InvariantViolation(format!(
            "TreeMeta dim {} != config.dim {}",
            disk_meta.dim, config.dim
        )));
    }
    if disk_meta.metric != config.metric {
        return Err(TreeError::InvariantViolation(format!(
            "TreeMeta metric {:?} != config.metric {:?}",
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

    let disk_leaves: HashMap<NodeId, _> = storage.all_leaf_nodes()?.into_iter().collect();
    let disk_leaf_vecs: HashMap<NodeId, _> = storage.all_leaf_vectors()?.into_iter().collect();
    if disk_leaves.len() != disk_leaf_vecs.len() {
        return Err(TreeError::InvariantViolation(format!(
            "cf_leaf_entries count {} != cf_leaf_vectors count {}",
            disk_leaves.len(),
            disk_leaf_vecs.len()
        )));
    }

    let mut visited_internal = HashSet::new();
    let mut visited_leaves = HashSet::new();
    let mut seen_docs: HashMap<DocId, NodeId> = HashMap::new();
    let mut min_leaf = usize::MAX;
    let mut max_leaf = 0usize;
    let mut min_internal = usize::MAX;
    let mut max_internal = 0usize;

    let mut stack: Vec<(NodeId, u8, Option<NodeId>)> =
        vec![(disk_meta.root_id, disk_meta.root_level, None)];

    while let Some((node_id, expected_level, expected_parent)) = stack.pop() {
        if visited_internal.contains(&node_id) || visited_leaves.contains(&node_id) {
            return Err(TreeError::InvariantViolation(format!(
                "cycle or shared node detected at node {node_id}"
            )));
        }

        let is_root = node_id == disk_meta.root_id;

        if expected_level == 0 {
            visited_leaves.insert(node_id);
            if disk_internals.contains_key(&node_id) {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} unexpectedly present in cf_internal_nodes"
                )));
            }
            let leaf = disk_leaves.get(&node_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!("missing leaf {node_id} in cf_leaf_entries"))
            })?;
            let leaf_vecs = disk_leaf_vecs.get(&node_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!(
                    "missing leaf {node_id} in cf_leaf_vectors"
                ))
            })?;

            if leaf.parent_id != expected_parent {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} has parent_id {:?}, expected {expected_parent:?}",
                    leaf.parent_id
                )));
            }
            if leaf.centroid.len() != config.dim || leaf.centroid.iter().any(|x| !x.is_finite()) {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} has invalid centroid"
                )));
            }
            if leaf.doc_ids != leaf_vecs.doc_ids {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} doc_ids mismatch between cf_leaf_entries and cf_leaf_vectors"
                )));
            }
            if leaf_vecs.vectors.iter().any(|x| !x.is_finite()) {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} contains non-finite vector component"
                )));
            }

            let count = leaf.entry_count();
            if is_root {
                if count > config.max_leaf_size {
                    return Err(TreeError::InvariantViolation(format!(
                        "root leaf {node_id} has {count} > max_leaf_size {}",
                        config.max_leaf_size
                    )));
                }
            } else if count < config.min_leaf_size || count > config.max_leaf_size {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} has {count} entries, expected {}..={}",
                    config.min_leaf_size, config.max_leaf_size
                )));
            }

            min_leaf = min_leaf.min(count);
            max_leaf = max_leaf.max(count);

            for (i, &doc_id) in leaf.doc_ids.iter().enumerate() {
                let res_norm_sq = leaf.res_norm_sqs[i];
                let scale = leaf.scales[i];
                if !res_norm_sq.is_finite()
                    || res_norm_sq < 0.0
                    || !scale.is_finite()
                    || scale < 0.0
                    || (res_norm_sq > 0.0 && scale <= 0.0)
                {
                    return Err(TreeError::InvariantViolation(format!(
                        "doc {doc_id} in leaf {node_id} has invalid quantized entry"
                    )));
                }
                if let Some(prev_leaf) = seen_docs.insert(doc_id, node_id) {
                    return Err(TreeError::InvariantViolation(format!(
                        "duplicate doc_id {doc_id} in leaves {prev_leaf} and {node_id}"
                    )));
                }
            }
        } else {
            visited_internal.insert(node_id);
            if disk_leaves.contains_key(&node_id) || disk_leaf_vecs.contains_key(&node_id) {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} unexpectedly present in leaf CFs"
                )));
            }
            let node = disk_internals.get(&node_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!(
                    "missing internal node {node_id} in cf_internal_nodes"
                ))
            })?;

            if node.level != expected_level {
                return Err(TreeError::InvariantViolation(format!(
                    "node {node_id} has level {}, expected {expected_level}",
                    node.level
                )));
            }
            if node.parent_id != expected_parent {
                return Err(TreeError::InvariantViolation(format!(
                    "node {node_id} has parent_id {:?}, expected {expected_parent:?}",
                    node.parent_id
                )));
            }
            if node.centroid.len() != config.dim || node.centroid.iter().any(|x| !x.is_finite()) {
                return Err(TreeError::InvariantViolation(format!(
                    "node {node_id} has invalid centroid"
                )));
            }

            let count = node.entry_count();
            if is_root {
                if count < 2 || count > config.max_internal_fanout {
                    return Err(TreeError::InvariantViolation(format!(
                        "internal root {node_id} has {count} children, expected 2..={}",
                        config.max_internal_fanout
                    )));
                }
            } else if count < config.min_internal_fanout || count > config.max_internal_fanout {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} has {count} children, expected {}..={}",
                    config.min_internal_fanout, config.max_internal_fanout
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

            for &child_id in &node.child_ids {
                stack.push((child_id, expected_level - 1, Some(node_id)));
            }
        }
    }

    if seen_docs.len() as u64 != disk_meta.num_vectors {
        return Err(TreeError::InvariantViolation(format!(
            "meta.num_vectors {} != actual leaf entries {}",
            disk_meta.num_vectors,
            seen_docs.len()
        )));
    }

    // Check cf_doc_loc is a 1-to-1 bijection with leaf entries.
    let doc_locs = storage.all_doc_locs()?;
    if doc_locs.len() != seen_docs.len() {
        return Err(TreeError::InvariantViolation(format!(
            "cf_doc_loc count {} != leaf entries count {}",
            doc_locs.len(),
            seen_docs.len()
        )));
    }
    for (doc_id, leaf_id) in doc_locs {
        match seen_docs.get(&doc_id) {
            Some(&actual_leaf) if actual_leaf == leaf_id => {}
            Some(&actual_leaf) => {
                return Err(TreeError::InvariantViolation(format!(
                    "cf_doc_loc[{doc_id}] = {leaf_id}, but doc resides in {actual_leaf}"
                )));
            }
            None => {
                return Err(TreeError::InvariantViolation(format!(
                    "orphaned cf_doc_loc entry for doc {doc_id} -> {leaf_id}"
                )));
            }
        }
    }

    // Check for orphaned internal or leaf nodes.
    let total_visited = visited_internal.len() + visited_leaves.len();
    for nid in disk_internals.keys() {
        if !visited_internal.contains(nid) {
            return Err(TreeError::InvariantViolation(format!(
                "orphaned internal node {nid} not reachable from root"
            )));
        }
    }
    for lid in disk_leaves.keys() {
        if !visited_leaves.contains(lid) {
            return Err(TreeError::InvariantViolation(format!(
                "orphaned leaf {lid} not reachable from root"
            )));
        }
    }
    for lid in disk_leaf_vecs.keys() {
        if !visited_leaves.contains(lid) {
            return Err(TreeError::InvariantViolation(format!(
                "orphaned leaf vector blob {lid} not reachable from root"
            )));
        }
    }

    Ok(TreeStats {
        num_vectors: disk_meta.num_vectors,
        root_id: disk_meta.root_id,
        root_level: disk_meta.root_level,
        total_nodes: total_visited,
        internal_nodes: visited_internal.len(),
        leaf_nodes: visited_leaves.len(),
        min_leaf_entries: if visited_leaves.is_empty() { 0 } else { min_leaf },
        max_leaf_entries: max_leaf,
        min_internal_fanout: if visited_internal.is_empty() {
            0
        } else {
            min_internal
        },
        max_internal_fanout: max_internal,
    })
}
