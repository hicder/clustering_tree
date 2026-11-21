use std::collections::{HashMap, HashSet};

use crate::config::{DocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::node::TreeMeta;
use crate::storage::Storage;

/// Summary statistics collected by [`ClusteringTree::verify_invariants`].
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

/// Exhaustively verify all structural invariants of the tree in RocksDB.
pub fn verify_invariants(
    storage: &Storage,
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

        let header = storage.get_node_header(node_id)?.ok_or_else(|| {
            TreeError::InvariantViolation(format!("missing NodeHeader for reachable node {node_id}"))
        })?;

        if header.level != expected_level {
            return Err(TreeError::InvariantViolation(format!(
                "node {node_id} has level {}, expected {expected_level}",
                header.level
            )));
        }
        if header.parent_id != expected_parent {
            return Err(TreeError::InvariantViolation(format!(
                "node {node_id} has parent_id {:?}, expected {expected_parent:?}",
                header.parent_id
            )));
        }
        if header.centroid.len() != config.dim || header.centroid.iter().any(|x| !x.is_finite()) {
            return Err(TreeError::InvariantViolation(format!(
                "node {node_id} has invalid centroid"
            )));
        }

        let is_root = node_id == disk_meta.root_id;
        let count = header.entry_count as usize;

        if expected_level == 0 {
            visited_leaves.insert(node_id);
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

            let internal_entries = storage.scan_internal_entries(node_id)?;
            if !internal_entries.is_empty() {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} has {} unexpected internal entries",
                    internal_entries.len()
                )));
            }

            let leaf_entries = storage.scan_leaf_entries(node_id)?;
            if leaf_entries.len() != count {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} header.entry_count {count} != actual {}",
                    leaf_entries.len()
                )));
            }
            if header.subtree_count != count as u64 {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} subtree_count {} != entry_count {count}",
                    header.subtree_count
                )));
            }

            for (doc_id, q_entry) in leaf_entries {
                if !q_entry.residual_norm.is_finite()
                    || q_entry.residual_norm < 0.0
                    || !q_entry.o_bar_dot_o.is_finite()
                    || q_entry.o_bar_dot_o <= 0.0
                    || q_entry.code.len() != storage.rabitq_words()
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

            let leaf_entries = storage.scan_leaf_entries(node_id)?;
            if !leaf_entries.is_empty() {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} has {} unexpected leaf entries",
                    leaf_entries.len()
                )));
            }

            let internal_entries = storage.scan_internal_entries(node_id)?;
            if internal_entries.len() != count {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} header.entry_count {count} != actual {}",
                    internal_entries.len()
                )));
            }

            let sub_sum: u64 = internal_entries.iter().map(|(_, e)| e.subtree_count).sum();
            if header.subtree_count != sub_sum {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} subtree_count {} != child sum {sub_sum}",
                    header.subtree_count
                )));
            }

            for (child_id, _) in internal_entries {
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

    // Check cf_doc_loc is a 1-to-1 bijection with cf_leaf_entries.
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

    // Check cf_vectors is a 1-to-1 bijection with seen_docs and all vectors are valid.
    let stored_vecs = storage.all_vectors()?;
    if stored_vecs.len() != seen_docs.len() {
        return Err(TreeError::InvariantViolation(format!(
            "cf_vectors count {} != leaf entries count {}",
            stored_vecs.len(),
            seen_docs.len()
        )));
    }
    for (doc_id, vec) in stored_vecs {
        if !seen_docs.contains_key(&doc_id) {
            return Err(TreeError::InvariantViolation(format!(
                "orphaned cf_vectors entry for doc {doc_id}"
            )));
        }
        if vec.len() != config.dim || vec.iter().any(|x| !x.is_finite()) {
            return Err(TreeError::InvariantViolation(format!(
                "doc {doc_id} in cf_vectors has invalid vector"
            )));
        }
    }

    // Check for orphaned node headers or orphaned entries.
    let total_visited = visited_internal.len() + visited_leaves.len();
    for (nid, _) in storage.all_node_headers()? {
        if !visited_internal.contains(&nid) && !visited_leaves.contains(&nid) {
            return Err(TreeError::InvariantViolation(format!(
                "orphaned NodeHeader {nid} not reachable from root"
            )));
        }
    }
    for (nid, cid) in storage.all_internal_entry_keys()? {
        if !visited_internal.contains(&nid) {
            return Err(TreeError::InvariantViolation(format!(
                "orphaned internal entry ({nid}, {cid})"
            )));
        }
    }
    for (lid, did) in storage.all_leaf_entry_keys()? {
        if !visited_leaves.contains(&lid) {
            return Err(TreeError::InvariantViolation(format!(
                "orphaned leaf entry ({lid}, {did})"
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
