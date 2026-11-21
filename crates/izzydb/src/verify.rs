use std::collections::{HashMap, HashSet};

use roaring::RoaringBitmap;

use crate::config::{DocId, LocalId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::node::{PackedInternalNode, TreeMeta};
use crate::schema::CollectionSchema;
use crate::codec::{
    decode_attribute_entries, extract_index_keys_from_entries, make_posting_key,
};
use crate::storage::{DocLoc, Storage};

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

/// Summary statistics collected by [`crate::Collection::verify_invariants`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionStats {
    pub tree: TreeStats,
    pub num_docs: usize,
    pub num_cluster_bitmaps: usize,
    pub num_posting_bitmaps: usize,
}

/// Exhaustively verify all structural invariants of the tree in RocksDB and in-memory caches.
pub fn verify_invariants(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &HashMap<NodeId, NodeId>,
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
    let disk_vectors: HashMap<DocId, Vec<f32>> = storage.all_vectors()?.into_iter().collect();
    if disk_vectors.len() != disk_meta.num_vectors as usize {
        return Err(TreeError::InvariantViolation(format!(
            "cf_vectors count {} != meta.num_vectors {}",
            disk_vectors.len(),
            disk_meta.num_vectors
        )));
    }

    let mut visited_internal = HashSet::new();
    let mut visited_leaves = HashSet::new();
    let mut expected_leaf_parent: HashMap<NodeId, NodeId> = HashMap::new();
    let mut seen_docs: HashMap<DocId, DocLoc> = HashMap::new();
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
            if let Some(pid) = expected_parent {
                expected_leaf_parent.insert(node_id, pid);
            }
            if disk_internals.contains_key(&node_id) {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} unexpectedly present in cf_internal_nodes"
                )));
            }
            let leaf = disk_leaves.get(&node_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!("missing leaf {node_id} in cf_leaf_entries"))
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

            let count = leaf.entry_count();
            if leaf.local_ids.len() != count {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} local_ids.len() {} != entry_count {count}",
                    leaf.local_ids.len()
                )));
            }
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

            let mut seen_locals: HashSet<LocalId> = HashSet::with_capacity(count);
            for (i, (&doc_id, &local_id)) in
                leaf.doc_ids.iter().zip(leaf.local_ids.iter()).enumerate()
            {
                if local_id >= leaf.next_local_id {
                    return Err(TreeError::InvariantViolation(format!(
                        "leaf {node_id} doc {doc_id} has local_id {local_id} >= next_local_id {}",
                        leaf.next_local_id
                    )));
                }
                if !seen_locals.insert(local_id) {
                    return Err(TreeError::InvariantViolation(format!(
                        "duplicate local_id {local_id} in leaf {node_id}"
                    )));
                }
                let vec = disk_vectors.get(&doc_id).ok_or_else(|| {
                    TreeError::InvariantViolation(format!(
                        "missing vector for doc {doc_id} (leaf {node_id}) in cf_vectors"
                    ))
                })?;
                if vec.len() != config.dim || vec.iter().any(|x| !x.is_finite()) {
                    return Err(TreeError::InvariantViolation(format!(
                        "doc {doc_id} in cf_vectors has invalid vector"
                    )));
                }
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
                let loc = DocLoc {
                    leaf_id: node_id,
                    local_id,
                };
                if let Some(prev_loc) = seen_docs.insert(doc_id, loc) {
                    return Err(TreeError::InvariantViolation(format!(
                        "duplicate doc_id {doc_id} in leaves {} and {node_id}",
                        prev_loc.leaf_id
                    )));
                }
            }
        } else {
            visited_internal.insert(node_id);
            if disk_leaves.contains_key(&node_id) {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} unexpectedly present in cf_leaf_entries"
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

    if leaf_parent != &expected_leaf_parent {
        return Err(TreeError::InvariantViolation(
            "in-memory leaf_parent cache differs from disk tree structure".into(),
        ));
    }

    if seen_docs.len() as u64 != disk_meta.num_vectors {
        return Err(TreeError::InvariantViolation(format!(
            "meta.num_vectors {} != actual leaf entries {}",
            disk_meta.num_vectors,
            seen_docs.len()
        )));
    }

    // Check cf_doc_loc is a 1-to-1 bijection with leaf (leaf_id, local_id) entries.
    let doc_locs = storage.all_doc_locs()?;
    if doc_locs.len() != seen_docs.len() {
        return Err(TreeError::InvariantViolation(format!(
            "cf_doc_loc count {} != leaf entries count {}",
            doc_locs.len(),
            seen_docs.len()
        )));
    }
    for (doc_id, loc) in doc_locs {
        match seen_docs.get(&doc_id) {
            Some(&actual_loc) if actual_loc == loc => {}
            Some(&actual_loc) => {
                return Err(TreeError::InvariantViolation(format!(
                    "cf_doc_loc[{doc_id}] = {loc:?}, but doc resides at {actual_loc:?}"
                )));
            }
            None => {
                return Err(TreeError::InvariantViolation(format!(
                    "orphaned cf_doc_loc entry for doc {doc_id} -> {loc:?}"
                )));
            }
        }
    }

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
    if disk_vectors.len() != seen_docs.len() {
        return Err(TreeError::InvariantViolation(format!(
            "cf_vectors count {} != reachable doc count {}",
            disk_vectors.len(),
            seen_docs.len()
        )));
    }
    for doc_id in disk_vectors.keys() {
        if !seen_docs.contains_key(doc_id) {
            return Err(TreeError::InvariantViolation(format!(
                "orphaned vector {doc_id} in cf_vectors not reachable from root"
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

/// Exhaustively verify tree invariants plus `cf_docs`, `cf_attr_postings`, and `cf_attr_clusters`.
pub fn verify_collection_invariants(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &HashMap<NodeId, NodeId>,
    mem_meta: &TreeMeta,
    config: &TreeConfig,
    mem_schema: &CollectionSchema,
) -> Result<CollectionStats> {
    let tree = verify_invariants(storage, internal_cache, leaf_parent, mem_meta, config)?;

    let disk_schema = storage.get_schema()?.ok_or_else(|| {
        TreeError::InvariantViolation("missing CollectionSchema in cf_meta".into())
    })?;
    if &disk_schema != mem_schema {
        return Err(TreeError::InvariantViolation(
            "in-memory CollectionSchema differs from disk".into(),
        ));
    }

    let doc_locs: HashMap<DocId, DocLoc> = storage.all_doc_locs()?.into_iter().collect();
    let docs_bytes: HashMap<DocId, Vec<u8>> = storage.all_docs_bytes()?.into_iter().collect();

    if docs_bytes.len() != tree.num_vectors as usize {
        return Err(TreeError::InvariantViolation(format!(
            "cf_docs count {} != meta.num_vectors {}",
            docs_bytes.len(),
            tree.num_vectors
        )));
    }
    for &doc_id in doc_locs.keys() {
        if !docs_bytes.contains_key(&doc_id) {
            return Err(TreeError::InvariantViolation(format!(
                "missing doc {doc_id} in cf_docs"
            )));
        }
    }

    let mut expected_clusters: HashMap<Vec<u8>, RoaringBitmap> = HashMap::new();
    let mut expected_postings: HashMap<Vec<u8>, RoaringBitmap> = HashMap::new();

    for (&doc_id, raw_bytes) in &docs_bytes {
        let loc = doc_locs.get(&doc_id).ok_or_else(|| {
            TreeError::InvariantViolation(format!("orphaned doc {doc_id} in cf_docs"))
        })?;
        let entries = decode_attribute_entries(raw_bytes)?;
        let keys = extract_index_keys_from_entries(mem_schema, &entries)?;
        for cluster_key in keys {
            expected_clusters
                .entry(cluster_key.clone())
                .or_default()
                .insert(loc.leaf_id);
            let posting_key = make_posting_key(&cluster_key, loc.leaf_id);
            expected_postings
                .entry(posting_key)
                .or_default()
                .insert(loc.local_id);
        }
    }

    let disk_postings: HashMap<Vec<u8>, RoaringBitmap> =
        storage.all_attr_postings()?.into_iter().collect();
    for (k, bm) in &disk_postings {
        if bm.is_empty() {
            return Err(TreeError::InvariantViolation(format!(
                "empty bitmap persisted in cf_attr_postings for key {k:?}"
            )));
        }
    }
    if disk_postings != expected_postings {
        return Err(TreeError::InvariantViolation(format!(
            "cf_attr_postings mismatch: disk has {} entries, expected {}",
            disk_postings.len(),
            expected_postings.len()
        )));
    }

    let disk_clusters: HashMap<Vec<u8>, RoaringBitmap> =
        storage.all_attr_clusters()?.into_iter().collect();
    for (k, bm) in &disk_clusters {
        if bm.is_empty() {
            return Err(TreeError::InvariantViolation(format!(
                "empty bitmap persisted in cf_attr_clusters for key {k:?}"
            )));
        }
    }
    if disk_clusters != expected_clusters {
        return Err(TreeError::InvariantViolation(format!(
            "cf_attr_clusters mismatch: disk has {} entries, expected {}",
            disk_clusters.len(),
            expected_clusters.len()
        )));
    }

    Ok(CollectionStats {
        tree,
        num_docs: docs_bytes.len(),
        num_cluster_bitmaps: disk_clusters.len(),
        num_posting_bitmaps: disk_postings.len(),
    })
}
