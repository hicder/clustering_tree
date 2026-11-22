//! Exhaustive structural, attribute-index, and FTS invariant verification.

use std::collections::{BTreeMap, HashMap, HashSet};

use roaring::RoaringBitmap;

use crate::codec::{
    LeafPosting, decode_attribute_entries, encode_fts_doc_len_key, encode_fts_term_id_key,
    encode_fts_term_key, extract_index_keys_from_entries, make_posting_key, split_posting_key,
};
use crate::config::{
    AttrId, DocId, LocalId, NodeId, TermId, TreeConfig, fts_block_id, fts_skip_chunk_id,
    fts_skip_slot, fts_slot,
};
use crate::error::{Result, TreeError};
use crate::fts_index::INLINE_TERM_STATS_RECOMPUTE_MAX_BLOCKS;
use crate::node::{PackedInternalNode, PackedLeafNode, TreeMeta};
use crate::schema::{AttributeValue, CollectionSchema};
use crate::storage::{AttrZoneMap, DocLoc, FtsRevBlock, Storage};
use izzydb_fts::{
    AttrFtsStats, DocLenBlock, FtsAnalyzer, FtsMeta, INLINE_TERM_MAX_DF, PackedPositionBlock,
    PackedPostingBlock, PostingBlockHeader, ReusableBuffer, SkipBlock, TermRecord,
    dl_over_tf_lower_bound,
};

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
    pub num_fts_docs: usize,
    pub num_fts_terms: usize,
    pub num_fts_postings: usize,
    pub num_fts_positions: usize,
}

struct TreeWalkState {
    visited_internal: HashSet<NodeId>,
    visited_leaves: HashSet<NodeId>,
    expected_leaf_parent: HashMap<NodeId, NodeId>,
    seen_docs: HashMap<DocId, (NodeId, LocalId)>,
    min_leaf: usize,
    max_leaf: usize,
    min_internal: usize,
    max_internal: usize,
}

impl TreeWalkState {
    fn new() -> Self {
        Self {
            visited_internal: HashSet::new(),
            visited_leaves: HashSet::new(),
            expected_leaf_parent: HashMap::new(),
            seen_docs: HashMap::new(),
            min_leaf: usize::MAX,
            max_leaf: 0,
            min_internal: usize::MAX,
            max_internal: 0,
        }
    }
}

/// Exhaustively verify all structural invariants of the tree in RocksDB and in-memory caches.
pub fn verify_invariants(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &HashMap<NodeId, NodeId>,
    mem_meta: &TreeMeta,
    config: &TreeConfig,
) -> Result<TreeStats> {
    let disk_meta = verify_tree_meta(storage, mem_meta, config)?;

    let disk_internals: HashMap<NodeId, PackedInternalNode> =
        storage.all_internal_nodes()?.into_iter().collect();
    if &disk_internals != internal_cache {
        return Err(TreeError::InvariantViolation(
            "in-memory internal_cache differs from cf_internal_nodes on disk".into(),
        ));
    }

    let disk_leaves: HashMap<NodeId, PackedLeafNode> =
        storage.all_leaf_nodes()?.into_iter().collect();
    let disk_vectors: HashMap<DocId, Vec<f32>> = storage.all_vectors()?.into_iter().collect();
    if disk_vectors.len() != disk_meta.num_vectors as usize {
        return Err(TreeError::InvariantViolation(format!(
            "cf_vectors count {} != meta.num_vectors {}",
            disk_vectors.len(),
            disk_meta.num_vectors
        )));
    }

    let mut state = TreeWalkState::new();
    let mut stack: Vec<(NodeId, u8, Option<NodeId>)> =
        vec![(disk_meta.root_id, disk_meta.root_level, None)];

    while let Some((node_id, expected_level, expected_parent)) = stack.pop() {
        if state.visited_internal.contains(&node_id) || state.visited_leaves.contains(&node_id) {
            return Err(TreeError::InvariantViolation(format!(
                "cycle or shared node detected at node {node_id}"
            )));
        }

        let is_root = node_id == disk_meta.root_id;
        if expected_level == 0 {
            verify_leaf_node(
                node_id,
                expected_parent,
                is_root,
                config,
                &disk_internals,
                &disk_leaves,
                &disk_vectors,
                &mut state,
            )?;
        } else {
            verify_internal_node(
                node_id,
                expected_level,
                expected_parent,
                is_root,
                config,
                &disk_internals,
                &disk_leaves,
                &mut state,
                &mut stack,
            )?;
        }
    }

    if leaf_parent != &state.expected_leaf_parent {
        return Err(TreeError::InvariantViolation(
            "in-memory leaf_parent cache differs from disk tree structure".into(),
        ));
    }

    if state.seen_docs.len() as u64 != disk_meta.num_vectors {
        return Err(TreeError::InvariantViolation(format!(
            "meta.num_vectors {} != actual leaf entries {}",
            disk_meta.num_vectors,
            state.seen_docs.len()
        )));
    }

    verify_doc_loc_bijection(storage, &state.seen_docs)?;
    verify_no_orphans(
        &disk_internals,
        &disk_leaves,
        &disk_vectors,
        &state.visited_internal,
        &state.visited_leaves,
        &state.seen_docs,
    )?;

    Ok(TreeStats {
        num_vectors: disk_meta.num_vectors,
        root_id: disk_meta.root_id,
        root_level: disk_meta.root_level,
        total_nodes: state.visited_internal.len() + state.visited_leaves.len(),
        internal_nodes: state.visited_internal.len(),
        leaf_nodes: state.visited_leaves.len(),
        min_leaf_entries: if state.visited_leaves.is_empty() {
            0
        } else {
            state.min_leaf
        },
        max_leaf_entries: state.max_leaf,
        min_internal_fanout: if state.visited_internal.is_empty() {
            0
        } else {
            state.min_internal
        },
        max_internal_fanout: state.max_internal,
    })
}

fn verify_tree_meta(
    storage: &Storage,
    mem_meta: &TreeMeta,
    config: &TreeConfig,
) -> Result<TreeMeta> {
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
    Ok(disk_meta)
}

#[allow(clippy::too_many_arguments)]
fn verify_leaf_node(
    node_id: NodeId,
    expected_parent: Option<NodeId>,
    is_root: bool,
    config: &TreeConfig,
    disk_internals: &HashMap<NodeId, PackedInternalNode>,
    disk_leaves: &HashMap<NodeId, PackedLeafNode>,
    disk_vectors: &HashMap<DocId, Vec<f32>>,
    state: &mut TreeWalkState,
) -> Result<()> {
    state.visited_leaves.insert(node_id);
    if let Some(pid) = expected_parent {
        state.expected_leaf_parent.insert(node_id, pid);
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
    if leaf.slot_cap > config.max_leaf_size {
        return Err(TreeError::InvariantViolation(format!(
            "leaf {node_id} slot_cap {} > max_leaf_size {}",
            leaf.slot_cap, config.max_leaf_size
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

    state.min_leaf = state.min_leaf.min(count);
    state.max_leaf = state.max_leaf.max(count);

    let mut live_seen = 0usize;
    for slot in 0..leaf.slot_cap {
        if !leaf.is_live(slot) {
            continue;
        }
        live_seen += 1;
        let doc_id = leaf.doc_ids[slot];
        let local_id = slot as LocalId;
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
        let res_norm_sq = leaf.res_norm_sqs[slot];
        let scale = leaf.scales[slot];
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
        if let Some((prev_leaf, _)) = state.seen_docs.insert(doc_id, (node_id, local_id)) {
            return Err(TreeError::InvariantViolation(format!(
                "duplicate doc_id {doc_id} in leaves {prev_leaf} and {node_id}",
            )));
        }
    }
    if live_seen != count {
        return Err(TreeError::InvariantViolation(format!(
            "leaf {node_id} live_mask count {live_seen} != live_count {count}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_internal_node(
    node_id: NodeId,
    expected_level: u8,
    expected_parent: Option<NodeId>,
    is_root: bool,
    config: &TreeConfig,
    disk_internals: &HashMap<NodeId, PackedInternalNode>,
    disk_leaves: &HashMap<NodeId, PackedLeafNode>,
    state: &mut TreeWalkState,
    stack: &mut Vec<(NodeId, u8, Option<NodeId>)>,
) -> Result<()> {
    state.visited_internal.insert(node_id);
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

    state.min_internal = state.min_internal.min(count);
    state.max_internal = state.max_internal.max(count);

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
    Ok(())
}

/// Check `cf_doc_loc` is a 1-to-1 bijection with leaf `(leaf_id, local_id)` entries.
fn verify_doc_loc_bijection(
    storage: &Storage,
    seen_docs: &HashMap<DocId, (NodeId, LocalId)>,
) -> Result<()> {
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
            Some(&actual_pos) if actual_pos == (loc.leaf_id, loc.local_id) => {}
            Some(&actual_pos) => {
                return Err(TreeError::InvariantViolation(format!(
                    "cf_doc_loc[{doc_id}] = {loc:?}, but doc resides at {actual_pos:?}"
                )));
            }
            None => {
                return Err(TreeError::InvariantViolation(format!(
                    "orphaned cf_doc_loc entry for doc {doc_id} -> {loc:?}"
                )));
            }
        }
    }
    Ok(())
}

fn verify_no_orphans(
    disk_internals: &HashMap<NodeId, PackedInternalNode>,
    disk_leaves: &HashMap<NodeId, PackedLeafNode>,
    disk_vectors: &HashMap<DocId, Vec<f32>>,
    visited_internal: &HashSet<NodeId>,
    visited_leaves: &HashSet<NodeId>,
    seen_docs: &HashMap<DocId, (NodeId, LocalId)>,
) -> Result<()> {
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
    Ok(())
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

    let (num_cluster_bitmaps, num_posting_bitmaps) =
        verify_attr_indexes(storage, mem_schema, &doc_locs, &docs_bytes)?;
    let (num_fts_docs, num_fts_terms, num_fts_postings, num_fts_positions) =
        verify_fts_invariants(storage, mem_schema, &doc_locs, &docs_bytes)?;

    Ok(CollectionStats {
        tree,
        num_docs: docs_bytes.len(),
        num_cluster_bitmaps,
        num_posting_bitmaps,
        num_fts_docs,
        num_fts_terms,
        num_fts_postings,
        num_fts_positions,
    })
}

fn verify_attr_indexes(
    storage: &Storage,
    schema: &CollectionSchema,
    doc_locs: &HashMap<DocId, DocLoc>,
    docs_bytes: &HashMap<DocId, Vec<u8>>,
) -> Result<(usize, usize)> {
    let mut expected_clusters: HashMap<Vec<u8>, RoaringBitmap> = HashMap::new();
    let mut expected_postings: HashMap<Vec<u8>, LeafPosting> = HashMap::new();
    let mut expected_zones: HashMap<Vec<u8>, AttrZoneMap> = HashMap::new();
    let mut expected_counts: HashMap<Vec<u8>, u64> = HashMap::new();

    for (&doc_id, raw_bytes) in docs_bytes {
        let loc = doc_locs.get(&doc_id).ok_or_else(|| {
            TreeError::InvariantViolation(format!("orphaned doc {doc_id} in cf_docs"))
        })?;
        let entries = decode_attribute_entries(raw_bytes)?;
        let keys = extract_index_keys_from_entries(schema, &entries)?;
        for cluster_key in keys {
            *expected_counts.entry(cluster_key.clone()).or_insert(0) += 1;
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

    for (pkey, posting) in &expected_postings {
        if posting.is_empty() {
            continue;
        }
        let (attr_id, type_tag, leaf_id, val_bytes) = split_posting_key(pkey)?;
        if type_tag == crate::codec::TAG_PRESENT || type_tag == crate::codec::TAG_BOOL {
            continue;
        }
        let zkey = crate::codec::encode_attr_prefix(attr_id, type_tag);
        let zmap = expected_zones.entry(zkey.to_vec()).or_default();
        match zmap.get_mut(&leaf_id) {
            Some((min_b, max_b)) => {
                if val_bytes < min_b.as_slice() {
                    *min_b = val_bytes.to_vec();
                }
                if val_bytes > max_b.as_slice() {
                    *max_b = val_bytes.to_vec();
                }
            }
            None => {
                zmap.insert(leaf_id, (val_bytes.to_vec(), val_bytes.to_vec()));
            }
        }
    }

    let disk_postings: HashMap<Vec<u8>, LeafPosting> =
        storage.all_attr_postings()?.into_iter().collect();
    for (k, lp) in &disk_postings {
        if lp.is_empty() {
            return Err(TreeError::InvariantViolation(format!(
                "empty posting persisted in cf_attr_postings for key {k:?}"
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

    let disk_zones: HashMap<Vec<u8>, AttrZoneMap> = storage.all_attr_zones()?.into_iter().collect();
    if disk_zones != expected_zones {
        return Err(TreeError::InvariantViolation(format!(
            "cf_attr_zones mismatch: disk has {} entries, expected {}",
            disk_zones.len(),
            expected_zones.len()
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

    if !storage.has_attr_counts_marker()? {
        return Err(TreeError::InvariantViolation(
            "missing default[0x06] attr counts marker".into(),
        ));
    }
    let disk_counts: HashMap<Vec<u8>, u64> = storage.all_attr_counts()?.into_iter().collect();
    for (k, &c) in &disk_counts {
        if c == 0 {
            return Err(TreeError::InvariantViolation(format!(
                "zero attr count persisted in default[0x06] for key {k:?}"
            )));
        }
    }
    if disk_counts != expected_counts {
        let mismatch = expected_counts
            .iter()
            .find(|(k, v)| disk_counts.get(*k) != Some(v))
            .map(|(k, v)| format!("key {k:?}: expected {v}, disk {:?}", disk_counts.get(k)))
            .or_else(|| {
                disk_counts
                    .keys()
                    .find(|k| !expected_counts.contains_key(*k))
                    .map(|k| format!("unexpected disk key {k:?}"))
            })
            .unwrap_or_default();
        return Err(TreeError::InvariantViolation(format!(
            "default[0x06] attr counts mismatch: disk has {} entries, expected {} ({mismatch})",
            disk_counts.len(),
            expected_counts.len()
        )));
    }

    Ok((disk_clusters.len(), disk_postings.len()))
}

struct ExpectedFtsState {
    attr_stats: BTreeMap<AttrId, AttrFtsStats>,
    terms: HashMap<Vec<u8>, (u64, u32, f32)>,
    term_postings: HashMap<Vec<u8>, BTreeMap<u32, PackedPostingBlock>>,
    term_positions: HashMap<Vec<u8>, BTreeMap<u32, PackedPositionBlock>>,
    doc_lens: HashMap<[u8; 6], DocLenBlock>,
}

fn verify_fts_invariants(
    storage: &Storage,
    schema: &CollectionSchema,
    doc_locs: &HashMap<DocId, DocLoc>,
    docs_bytes: &HashMap<DocId, Vec<u8>>,
) -> Result<(usize, usize, usize, usize)> {
    let mut disk_doc_map: Vec<(DocId, u32)> = Vec::new();
    for (&doc_id, loc) in doc_locs {
        if let Some(fid) = loc.fts_doc_id {
            disk_doc_map.push((doc_id, fid));
        }
    }
    let disk_rev_blocks = storage.all_fts_rev_blocks()?;
    let disk_terms = storage.all_fts_terms()?;
    let disk_skips = storage.all_fts_skips()?;
    let disk_postings = storage.all_fts_postings()?;
    let disk_positions = storage.all_fts_positions()?;
    let disk_doc_lens = storage.all_fts_doc_lens()?;
    let free_ids = storage.get_fts_free_ids()?;

    if !schema.has_fts() {
        if !disk_doc_map.is_empty()
            || !disk_rev_blocks.is_empty()
            || !disk_terms.is_empty()
            || !disk_skips.is_empty()
            || !disk_postings.is_empty()
            || !disk_positions.is_empty()
            || !disk_doc_lens.is_empty()
            || !free_ids.is_empty()
        {
            return Err(TreeError::InvariantViolation(
                "schema has no FTS attributes, but FTS column families are non-empty".into(),
            ));
        }
        return Ok((0, 0, 0, 0));
    }

    if docs_bytes.is_empty()
        && disk_doc_map.is_empty()
        && disk_rev_blocks.is_empty()
        && disk_terms.is_empty()
        && disk_skips.is_empty()
        && disk_postings.is_empty()
        && disk_positions.is_empty()
        && disk_doc_lens.is_empty()
    {
        if !free_ids.is_empty() {
            return Err(TreeError::InvariantViolation(
                "fts_free_ids non-empty in empty FTS collection".into(),
            ));
        }
        if let Some(bytes) = storage.get_fts_meta_bytes()? {
            let fts_meta = FtsMeta::decode(&bytes)?;
            if fts_meta.next_fts_doc_id != 0 {
                return Err(TreeError::InvariantViolation(format!(
                    "empty FTS collection has non-zero next_fts_doc_id {}",
                    fts_meta.next_fts_doc_id
                )));
            }
        }
        return Ok((0, 0, 0, 0));
    }

    let fts_meta = match storage.get_fts_meta_bytes()? {
        Some(bytes) => FtsMeta::decode(&bytes)?,
        None => {
            return Err(TreeError::InvariantViolation(
                "missing fts_meta in cf_meta for non-empty FTS collection".into(),
            ));
        }
    };

    let doc_to_fts = verify_fts_doc_mapping(
        doc_locs,
        docs_bytes,
        disk_doc_map,
        &disk_rev_blocks,
        &free_ids,
        fts_meta.next_fts_doc_id,
    )?;

    let fts_entries: Vec<_> = schema.fts_entries().collect();
    let expected = reconstruct_expected_fts_state(&fts_entries, docs_bytes, &doc_to_fts)?;

    // Compare per_attr_stats ignoring zero-doc entries.
    for entry in &fts_entries {
        let disk_st = fts_meta
            .per_attr_stats
            .get(&entry.id)
            .copied()
            .unwrap_or_default();
        let exp_st = expected.attr_stats.get(&entry.id).copied().unwrap_or_default();
        if disk_st != exp_st {
            return Err(TreeError::InvariantViolation(format!(
                "FTS attr_stats mismatch for attr {}: disk {:?} != expected {:?}",
                entry.name, disk_st, exp_st
            )));
        }
    }

    verify_fts_doc_lens(disk_doc_lens, &expected.doc_lens)?;
    let (num_terms, num_postings, num_positions) = verify_fts_terms_and_blocks(
        &fts_meta,
        &expected,
        disk_terms,
        disk_skips,
        disk_postings,
        disk_positions,
    )?;

    Ok((doc_to_fts.len(), num_terms, num_postings, num_positions))
}

fn verify_fts_doc_mapping(
    doc_locs: &HashMap<DocId, DocLoc>,
    docs_bytes: &HashMap<DocId, Vec<u8>>,
    disk_doc_map: Vec<(DocId, u32)>,
    disk_rev_blocks: &[(u32, FtsRevBlock)],
    free_ids: &RoaringBitmap,
    next_fts_doc_id: u32,
) -> Result<HashMap<DocId, u32>> {
    let mut rev_to_entry: HashMap<u32, (DocId, NodeId, LocalId)> = HashMap::new();
    for (bid, blk) in disk_rev_blocks {
        if blk.is_empty() {
            return Err(TreeError::InvariantViolation(format!(
                "empty FtsRevBlock {bid} persisted"
            )));
        }
        for slot in 0..=u8::MAX {
            if let Some(entry) = blk.get(slot) {
                let fts_id = crate::config::make_fts_doc_id(*bid, slot);
                if rev_to_entry.insert(fts_id, entry).is_some() {
                    return Err(TreeError::InvariantViolation(format!(
                        "duplicate fts_doc_id {fts_id} in cf_fts_rev_doc_map"
                    )));
                }
            }
        }
    }

    if disk_doc_map.len() != docs_bytes.len() || rev_to_entry.len() != docs_bytes.len() {
        return Err(TreeError::InvariantViolation(format!(
            "FTS doc map counts ({}, {}) != cf_docs count ({})",
            disk_doc_map.len(),
            rev_to_entry.len(),
            docs_bytes.len()
        )));
    }

    let doc_to_fts: HashMap<DocId, u32> = disk_doc_map.into_iter().collect();

    for free_id in free_ids.iter() {
        if free_id >= next_fts_doc_id {
            return Err(TreeError::InvariantViolation(format!(
                "fts_free_ids contains {free_id} >= next_fts_doc_id {next_fts_doc_id}"
            )));
        }
        if rev_to_entry.contains_key(&free_id) {
            return Err(TreeError::InvariantViolation(format!(
                "fts_free_ids contains live fts_doc_id {free_id}"
            )));
        }
    }
    if next_fts_doc_id > 0 && free_ids.contains(next_fts_doc_id - 1) {
        return Err(TreeError::InvariantViolation(format!(
            "fts_free_ids contains tail ID {}, next_fts_doc_id should have shrunk",
            next_fts_doc_id - 1
        )));
    }

    for (&doc_id, &fts_id) in &doc_to_fts {
        if !docs_bytes.contains_key(&doc_id) {
            return Err(TreeError::InvariantViolation(format!(
                "orphaned cf_doc_loc fts_doc_id for doc {doc_id}"
            )));
        }
        if fts_id >= next_fts_doc_id {
            return Err(TreeError::InvariantViolation(format!(
                "fts_doc_id {fts_id} >= next_fts_doc_id {next_fts_doc_id}"
            )));
        }
        if free_ids.contains(fts_id) {
            return Err(TreeError::InvariantViolation(format!(
                "live fts_doc_id {fts_id} also present in fts_free_ids"
            )));
        }
        let loc = doc_locs[&doc_id];
        if rev_to_entry.get(&fts_id) != Some(&(doc_id, loc.leaf_id, loc.local_id)) {
            return Err(TreeError::InvariantViolation(format!(
                "FTS doc map bijection broken for doc {doc_id} <-> fts_doc_id {fts_id}"
            )));
        }
    }

    Ok(doc_to_fts)
}

fn reconstruct_expected_fts_state(
    fts_entries: &[&crate::schema::SchemaEntry],
    docs_bytes: &HashMap<DocId, Vec<u8>>,
    doc_to_fts: &HashMap<DocId, u32>,
) -> Result<ExpectedFtsState> {
    let mut state = ExpectedFtsState {
        attr_stats: BTreeMap::new(),
        terms: HashMap::new(),
        term_postings: HashMap::new(),
        term_positions: HashMap::new(),
        doc_lens: HashMap::new(),
    };
    let mut buf = ReusableBuffer::new();

    for (&doc_id, raw_bytes) in docs_bytes {
        let fts_id = *doc_to_fts.get(&doc_id).ok_or_else(|| {
            TreeError::InvariantViolation(format!("missing doc {doc_id} fts_doc_id in cf_doc_loc"))
        })?;
        let block_id = fts_block_id(fts_id);
        let slot = fts_slot(fts_id);

        let entries = decode_attribute_entries(raw_bytes)?;
        let by_id: HashMap<AttrId, AttributeValue> = entries.into_iter().collect();

        for &schema_entry in fts_entries {
            let fts_cfg = schema_entry.full_text_search.as_ref().unwrap();
            let Some(val) = by_id.get(&schema_entry.id) else {
                continue;
            };
            let analyzer = FtsAnalyzer::new(fts_cfg.clone());
            let Some(analyzed) = analyzer.analyze_attribute_value(val, &mut buf)? else {
                continue;
            };
            let st = state.attr_stats.entry(schema_entry.id).or_default();
            st.doc_count += 1;
            st.total_tokens += analyzed.doc_len as u64;

            if analyzed.doc_len > 0 {
                let dl_key = encode_fts_doc_len_key(schema_entry.id, block_id);
                let dl_blk = state.doc_lens.entry(dl_key).or_default();
                dl_blk.set(slot, analyzed.doc_len);
                let dl_blk_ref = &*dl_blk;

                for (tok, term) in analyzed.terms {
                    let tkey = encode_fts_term_key(schema_entry.id, &tok);
                    let te = state
                        .terms
                        .entry(tkey.clone())
                        .or_insert((0, 0, f32::INFINITY));
                    te.0 += 1;
                    te.1 = te.1.max(term.tf);
                    te.2 = te.2.min(dl_over_tf_lower_bound(analyzed.doc_len, term.tf));

                    state
                        .term_postings
                        .entry(tkey.clone())
                        .or_default()
                        .entry(block_id)
                        .or_default()
                        .upsert(slot, term.tf, dl_blk_ref);
                    if fts_cfg.index_positions {
                        state
                            .term_positions
                            .entry(tkey)
                            .or_default()
                            .entry(block_id)
                            .or_default()
                            .upsert(slot, term.positions);
                    }
                }
            }
        }
    }

    Ok(state)
}

fn verify_fts_doc_lens(
    disk_doc_lens: Vec<(Vec<u8>, Vec<u8>)>,
    expected_doc_lens: &HashMap<[u8; 6], DocLenBlock>,
) -> Result<()> {
    let disk_doc_lens_map: HashMap<Vec<u8>, Vec<u8>> = disk_doc_lens.into_iter().collect();
    if disk_doc_lens_map.len() != expected_doc_lens.len() {
        return Err(TreeError::InvariantViolation(format!(
            "cf_fts_doc_lens count {} != expected {}",
            disk_doc_lens_map.len(),
            expected_doc_lens.len()
        )));
    }
    for (k, exp_dl_blk) in expected_doc_lens {
        let Some(raw) = disk_doc_lens_map.get(k.as_slice()) else {
            return Err(TreeError::InvariantViolation(
                "missing expected key in cf_fts_doc_lens".into(),
            ));
        };
        let disk_dl_blk = DocLenBlock::decode(raw)?;
        if &disk_dl_blk != exp_dl_blk {
            return Err(TreeError::InvariantViolation(
                "cf_fts_doc_lens block mismatch against reconstructed documents".into(),
            ));
        }
    }
    Ok(())
}

fn verify_fts_terms_and_blocks(
    fts_meta: &FtsMeta,
    expected: &ExpectedFtsState,
    disk_terms: Vec<(Vec<u8>, Vec<u8>)>,
    disk_skips: Vec<(Vec<u8>, Vec<u8>)>,
    disk_postings: Vec<(Vec<u8>, Vec<u8>)>,
    disk_positions: Vec<(Vec<u8>, Vec<u8>)>,
) -> Result<(usize, usize, usize)> {
    let disk_terms_map: HashMap<Vec<u8>, Vec<u8>> = disk_terms.into_iter().collect();
    if disk_terms_map.len() != expected.terms.len() {
        return Err(TreeError::InvariantViolation(format!(
            "cf_fts_terms count {} != expected {}",
            disk_terms_map.len(),
            expected.terms.len()
        )));
    }

    let disk_skips_map: HashMap<[u8; 8], Vec<u8>> = disk_skips
        .into_iter()
        .map(|(k, v)| (k.try_into().expect("8B skip key"), v))
        .collect();
    let disk_postings_map: HashMap<[u8; 8], Vec<u8>> = disk_postings
        .into_iter()
        .map(|(k, v)| (k.try_into().expect("8B posting key"), v))
        .collect();
    let disk_positions_map: HashMap<[u8; 8], Vec<u8>> = disk_positions
        .into_iter()
        .map(|(k, v)| (k.try_into().expect("8B position key"), v))
        .collect();

    let mut expected_posting_count = 0usize;
    let mut expected_skip_count = 0usize;
    let mut expected_position_count = 0usize;
    let mut seen_term_ids: HashSet<TermId> = HashSet::with_capacity(expected.terms.len());

    for (tkey, (exp_df, exp_max_tf, exp_min_dl_over_tf)) in &expected.terms {
        let Some(raw) = disk_terms_map.get(tkey) else {
            return Err(TreeError::InvariantViolation(
                "missing expected key in cf_fts_terms".into(),
            ));
        };
        let rec = TermRecord::decode(raw)?;
        if rec.term_id >= fts_meta.next_term_id || !seen_term_ids.insert(rec.term_id) {
            return Err(TreeError::InvariantViolation(format!(
                "invalid or duplicate term_id {} in cf_fts_terms",
                rec.term_id
            )));
        }
        let ts = rec.stats;
        if ts.doc_freq != *exp_df
            || ts.max_tf < *exp_max_tf
            || ts.min_dl_over_tf > *exp_min_dl_over_tf
        {
            return Err(TreeError::InvariantViolation(format!(
                "cf_fts_terms stats mismatch: disk {ts:?} vs expected (df={exp_df}, max_tf>={exp_max_tf}, min_dl_over_tf<={exp_min_dl_over_tf})"
            )));
        }

        let exp_blocks = &expected.term_postings[tkey];
        let block_count = exp_blocks.len();
        if block_count <= INLINE_TERM_STATS_RECOMPUTE_MAX_BLOCKS
            && (ts.max_tf != *exp_max_tf || ts.min_dl_over_tf != *exp_min_dl_over_tf)
        {
            return Err(TreeError::InvariantViolation(format!(
                "cf_fts_terms stats not tight for {block_count}-block term: disk {ts:?} vs exact (max_tf={exp_max_tf}, min_dl_over_tf={exp_min_dl_over_tf})"
            )));
        }

        if let Some((ib_id, ref ib_bytes)) = rec.inline_block {
            if exp_blocks.len() != 1 || *exp_df > INLINE_TERM_MAX_DF as u64 {
                return Err(TreeError::InvariantViolation(format!(
                    "term_id {} unexpectedly inlined with {} blocks, df={}",
                    rec.term_id,
                    exp_blocks.len(),
                    exp_df
                )));
            }
            let Some(exp_blk) = exp_blocks.get(&ib_id) else {
                return Err(TreeError::InvariantViolation(format!(
                    "inline TermRecord block_id {ib_id} missing from expected blocks"
                )));
            };
            let disk_blk = PackedPostingBlock::decode(ib_bytes)?;
            if &disk_blk != exp_blk {
                return Err(TreeError::InvariantViolation(
                    "inline TermRecord posting block mismatch against reconstructed documents"
                        .into(),
                ));
            }
        } else {
            let mut expected_skips_for_term: BTreeMap<u32, SkipBlock> = BTreeMap::new();
            for (&block_id, exp_blk) in exp_blocks {
                expected_posting_count += 1;
                let pkey = encode_fts_term_id_key(rec.term_id, block_id);
                let Some(raw_p) = disk_postings_map.get(&pkey) else {
                    return Err(TreeError::InvariantViolation(
                        "missing expected key in cf_fts_postings".into(),
                    ));
                };
                let disk_blk = PackedPostingBlock::decode(raw_p)?;
                if &disk_blk != exp_blk {
                    return Err(TreeError::InvariantViolation(
                        "cf_fts_postings block mismatch against reconstructed documents".into(),
                    ));
                }
                let hdr = PostingBlockHeader::parse(raw_p)?;
                expected_skips_for_term
                    .entry(fts_skip_chunk_id(block_id))
                    .or_default()
                    .upsert(fts_skip_slot(block_id), hdr.max_tf, hdr.min_dl_over_tf);
            }
            for (chunk_id, exp_skip) in expected_skips_for_term {
                expected_skip_count += 1;
                let skey = encode_fts_term_id_key(rec.term_id, chunk_id);
                let Some(raw_s) = disk_skips_map.get(&skey) else {
                    return Err(TreeError::InvariantViolation(
                        "missing expected key in cf_fts_skips".into(),
                    ));
                };
                let disk_skip = SkipBlock::decode(raw_s)?;
                if disk_skip != exp_skip {
                    return Err(TreeError::InvariantViolation(
                        "cf_fts_skips block mismatch against posting headers".into(),
                    ));
                }
            }
        }

        if let Some(exp_pos_blocks) = expected.term_positions.get(tkey) {
            for (&block_id, exp_pblk) in exp_pos_blocks {
                expected_position_count += 1;
                let pkey = encode_fts_term_id_key(rec.term_id, block_id);
                let Some(raw_pos) = disk_positions_map.get(&pkey) else {
                    return Err(TreeError::InvariantViolation(
                        "missing expected key in cf_fts_positions".into(),
                    ));
                };
                let disk_pblk = PackedPositionBlock::decode(raw_pos)?;
                if &disk_pblk != exp_pblk {
                    return Err(TreeError::InvariantViolation(
                        "cf_fts_positions block mismatch against reconstructed documents".into(),
                    ));
                }
            }
        }
    }

    if disk_postings_map.len() != expected_posting_count {
        return Err(TreeError::InvariantViolation(format!(
            "cf_fts_postings count {} != expected {expected_posting_count}",
            disk_postings_map.len()
        )));
    }
    if disk_skips_map.len() != expected_skip_count {
        return Err(TreeError::InvariantViolation(format!(
            "cf_fts_skips count {} != expected {expected_skip_count}",
            disk_skips_map.len()
        )));
    }
    if disk_positions_map.len() != expected_position_count {
        return Err(TreeError::InvariantViolation(format!(
            "cf_fts_positions count {} != expected {expected_position_count}",
            disk_positions_map.len()
        )));
    }

    Ok((
        disk_terms_map.len(),
        disk_postings_map.len(),
        disk_positions_map.len(),
    ))
}
