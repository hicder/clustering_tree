//! Mutable, RocksDB-backed hierarchical clustering tree (~100x fanout) with
//! RaBitQ-quantized leaf posting lists and full-precision vector storage.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use crate::config::{DocId, LocalId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::kmeans::balanced_two_means;
use crate::lire;
use crate::metric::{
    cluster_distance, compute_mean_centroid_flat, compute_weighted_centroid, incremental_mean,
    prepare_vector, validate_vector,
};
use crate::node::{PackedInternalNode, PackedLeafNode, TreeMeta};
use crate::rabitq::{EncodeScratch, RabitqQuantizer};
use crate::search::{self, SearchOptions};
use crate::storage::{DocLoc, Storage, TxContext};
use crate::verify::{self, TreeStats};

const SMALL_LEAF_REENCODE_THRESHOLD: usize = 16;
const MIN_COMPACT_SLOT_CAP: usize = 32;

/// Summary of a child node passed during root creation, parent attachment, and redistribution.
#[derive(Clone, Copy)]
struct ChildSummary<'a> {
    id: NodeId,
    subtree_count: u64,
    centroid: &'a [f32],
}

/// Metadata about a completed leaf split passed to SPFresh LIRE.
#[derive(Debug, Clone)]
pub(crate) struct SplitPair {
    pub left_id: NodeId,
    pub right_id: NodeId,
    pub old_centroid: Vec<f32>,
}

/// Returns true when a leaf should compact its slots to `0..live_count`.
#[inline]
pub(crate) fn needs_leaf_compaction(leaf: &PackedLeafNode, config: &TreeConfig) -> bool {
    leaf.slot_cap > config.max_leaf_size
        || (leaf.slot_cap > MIN_COMPACT_SLOT_CAP
            && leaf.live_count.saturating_mul(2) <= leaf.slot_cap)
}

/// Outcome of updating an existing document's vector inside `TxContext`.
#[derive(Debug)]
pub(crate) enum VectorUpdateResult {
    SameLeaf {
        delta: ClusterDelta,
    },
    Moved {
        old_loc: DocLoc,
        new_loc: DocLoc,
        rem_delta: ClusterDelta,
        ins_delta: ClusterDelta,
        split: Option<SplitPair>,
    },
}

/// A `(DocId, leaf_id, local_id)` entry in a [`ClusterDelta`].
pub type ClusterEntry = (DocId, NodeId, LocalId);

/// Documents whose `(leaf_id, local_id)` physical assignment changed during a tree mutation.
#[derive(Debug, Clone, Default)]
pub struct ClusterDelta {
    /// Docs whose `(leaf_id, local_id)` was removed (deleted, moved out, or renumbered).
    pub removed: Vec<ClusterEntry>,
    /// Docs whose `(leaf_id, local_id)` was added (inserted, moved in, or renumbered).
    pub added: Vec<ClusterEntry>,
}

impl ClusterDelta {
    pub fn is_empty(&self) -> bool {
        self.removed.is_empty() && self.added.is_empty()
    }

    pub fn merge(&mut self, other: Self) {
        self.removed.extend(other.removed);
        self.added.extend(other.added);
    }

    /// Cancel out identical `(doc_id, leaf_id, local_id)` pairs appearing in both `removed`
    /// and `added` (e.g. multi-step merges or compactions within a single transaction), returning
    /// `(net_removed, net_added)`.
    pub fn net_changes(&self) -> (Vec<ClusterEntry>, Vec<ClusterEntry>) {
        let mut rem = self.removed.clone();
        let mut add = self.added.clone();
        rem.sort_unstable();
        add.sort_unstable();

        let mut net_rem = Vec::with_capacity(rem.len());
        let mut net_add = Vec::with_capacity(add.len());
        let (mut i, mut j) = (0, 0);
        while i < rem.len() && j < add.len() {
            match rem[i].cmp(&add[j]) {
                std::cmp::Ordering::Less => {
                    net_rem.push(rem[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    net_add.push(add[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        net_rem.extend_from_slice(&rem[i..]);
        net_add.extend_from_slice(&add[j..]);
        (net_rem, net_add)
    }
}

/// RocksDB-backed mutable hierarchical clustering tree (~100x fanout)
/// with in-memory internal nodes, packed RaBitQ posting lists, and global per-document full vectors.
pub struct ClusteringTree {
    pub(crate) storage: Arc<Storage>,
    pub(crate) config: TreeConfig,
    pub(crate) meta: TreeMeta,
    pub(crate) quantizer: RabitqQuantizer,
    pub(crate) internal_cache: HashMap<NodeId, PackedInternalNode>,
    pub(crate) leaf_parent: HashMap<NodeId, NodeId>,
}

impl ClusteringTree {
    /// Open or create a [`ClusteringTree`] at `path` with the given `config`.
    pub fn open(path: impl AsRef<Path>, config: TreeConfig) -> Result<Self> {
        config.validate()?;
        let (storage, meta) = Storage::open(path, &config)?;
        Self::from_storage(Arc::new(storage), meta, config)
    }

    /// Construct a [`ClusteringTree`] over a shared [`Storage`] instance (used by `Collection`).
    pub fn from_storage(
        storage: Arc<Storage>,
        meta: TreeMeta,
        config: TreeConfig,
    ) -> Result<Self> {
        config.validate()?;
        let quantizer = RabitqQuantizer::new(config.dim, meta.rabitq_seed);
        let internal_cache: HashMap<NodeId, PackedInternalNode> =
            storage.all_internal_nodes()?.into_iter().collect();
        let mut leaf_parent = HashMap::new();
        for (&nid, node) in &internal_cache {
            if node.level == 1 {
                for &child_id in &node.child_ids {
                    leaf_parent.insert(child_id, nid);
                }
            }
        }
        Ok(Self {
            storage,
            config,
            meta,
            quantizer,
            internal_cache,
            leaf_parent,
        })
    }

    /// Return the tree configuration.
    pub fn config(&self) -> &TreeConfig {
        &self.config
    }

    /// Return the number of vectors currently stored in the tree.
    pub fn len(&self) -> u64 {
        self.meta.num_vectors
    }

    /// Return `true` if the tree contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.meta.num_vectors == 0
    }

    /// Retrieve the stored full-precision vector for `id`.
    pub fn get_vector(&self, id: DocId) -> Result<Option<Vec<f32>>> {
        self.storage.get_vector(id)
    }

    /// Open a read-your-own-writes transaction alongside immutable config and quantizer refs.
    pub(crate) fn begin_tx_parts(
        &mut self,
    ) -> (&TreeConfig, &RabitqQuantizer, TxContext<'_>) {
        (
            &self.config,
            &self.quantizer,
            TxContext::new(
                &self.storage,
                &mut self.internal_cache,
                &mut self.leaf_parent,
                self.meta.clone(),
            ),
        )
    }

    /// Execute `f` inside an atomic [`TxContext`], committing `self.meta` on success.
    pub(crate) fn in_tx<T>(
        &mut self,
        f: impl FnOnce(&mut TxContext<'_>, &TreeConfig, &RabitqQuantizer) -> Result<T>,
    ) -> Result<T> {
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.meta.clone(),
        );
        let result = f(&mut tx, &self.config, &self.quantizer)?;
        self.meta = tx.commit()?;
        Ok(result)
    }

    /// Insert or update a vector in the tree, quantizing it for the leaf posting list
    /// while persisting the full-precision vector in `cf_vectors`.
    pub fn add_leaf(&mut self, id: DocId, embedding: &[f32]) -> Result<()> {
        let vec = prepare_vector(embedding, self.config.dim, self.config.metric)?;
        let split_opt = self.in_tx(|tx, config, quantizer| {
            if let Some(old_loc) = tx.get_doc_loc(id)? {
                match update_vector_in_tx(tx, id, old_loc, &vec, config, quantizer)? {
                    VectorUpdateResult::SameLeaf { .. } => Ok(None),
                    VectorUpdateResult::Moved { split, .. } => Ok(split),
                }
            } else {
                let (_loc, _delta, split_opt) = insert_in_tx(tx, id, &vec, config, quantizer)?;
                Ok(split_opt)
            }
        })?;

        if let Some(split_pair) = split_opt
            && self.config.enable_lire
            && self.meta.root_level > 0
        {
            lire::run_lire_with_hook(self, split_pair, |_, _| Ok(()))?;
        }
        Ok(())
    }

    /// Remove a vector from the tree by `DocId`, triggering merges, redistributions, or root collapse as needed.
    ///
    /// Returns `true` if `id` existed and was removed, or `false` if `id` was absent.
    pub fn remove(&mut self, id: DocId) -> Result<bool> {
        let (config, quantizer, mut tx) = self.begin_tx_parts();
        let removed = remove_existing_in_tx(&mut tx, id, config, quantizer)?.is_some();
        if removed {
            self.meta = tx.commit()?;
        }
        Ok(removed)
    }

    /// Remove a vector from the tree, triggering merges, redistributions, or root collapse as needed.
    ///
    /// If `id` is not present in the tree, returns `Ok(())`.
    pub fn remove_leaf(&mut self, id: DocId, embedding: &[f32]) -> Result<()> {
        validate_vector(embedding, self.config.dim)?;
        let _ = self.remove(id)?;
        Ok(())
    }

    /// Approximate nearest neighbor search using RaBitQ-quantized posting lists.
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Result<Vec<(DocId, f32)>> {
        search::search(
            &self.storage,
            &self.internal_cache,
            &self.meta,
            &self.quantizer,
            query,
            SearchOptions::new(k, beam_width),
            &self.config,
        )
    }

    /// Approximate nearest neighbor search using RaBitQ posting lists for candidate selection,
    /// followed by exact reranking of the top `rerank_candidates` against stored full vectors.
    pub fn search_with_rerank(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        rerank_candidates: usize,
    ) -> Result<Vec<(DocId, f32)>> {
        search::search(
            &self.storage,
            &self.internal_cache,
            &self.meta,
            &self.quantizer,
            query,
            SearchOptions::new(k, beam_width).with_rerank(rerank_candidates),
            &self.config,
        )
    }

    /// Exhaustively verify all structural invariants and return tree statistics.
    pub fn verify_invariants(&self) -> Result<TreeStats> {
        verify::verify_invariants(
            &self.storage,
            &self.internal_cache,
            &self.leaf_parent,
            &self.meta,
            &self.config,
        )
    }
}

struct CompactArgs<'a, 'b> {
    tx: &'a mut TxContext<'b>,
    leaf_id: NodeId,
    leaf: &'a mut PackedLeafNode,
    flat_vecs: &'a [f32],
    quantizer: &'a RabitqQuantizer,
    dim: usize,
    scratch: &'a mut EncodeScratch,
    delta: &'a mut ClusterDelta,
}

fn compact_leaf_in_tx(args: CompactArgs<'_, '_>) {
    let pairs = args.leaf.live_slot_doc_pairs();
    let mut doc_ids = Vec::with_capacity(pairs.len());
    for &(lid, did) in &pairs {
        args.delta.removed.push((did, args.leaf_id, lid));
        doc_ids.push(did);
    }
    args.leaf
        .reset_and_encode(&doc_ids, args.flat_vecs, args.quantizer, args.dim, args.scratch);
    for (lid, did) in args.leaf.live_slot_doc_pairs() {
        args.tx.put_doc_pos(did, args.leaf_id, lid);
        args.delta.added.push((did, args.leaf_id, lid));
    }
}

/// Recompute a leaf's centroid and either re-encode all live slots or rotate the centroid.
fn refresh_leaf_after_shrink(
    tx: &TxContext<'_>,
    leaf: &mut PackedLeafNode,
    old_centroid: &[f32],
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    scratch: &mut EncodeScratch,
) -> Result<(Vec<f32>, bool)> {
    let count = leaf.entry_count();
    let live_docs = leaf.live_doc_ids();
    let flat_vecs = tx.get_vectors_flat(&live_docs)?;
    leaf.centroid = compute_mean_centroid_flat(&flat_vecs, count, config.dim, config.metric);
    let drifted =
        cluster_distance(config.metric, old_centroid, &leaf.centroid) > config.centroid_drift_eps;
    if count <= SMALL_LEAF_REENCODE_THRESHOLD || drifted {
        leaf.re_encode_existing(&flat_vecs, quantizer, config.dim, scratch);
    } else {
        quantizer.rotate_into(&leaf.centroid, &mut leaf.rotated_centroid);
    }
    Ok((flat_vecs, drifted))
}

/// Update an existing document's vector inside `tx`, using an in-place same-leaf path
/// whenever the new vector routes back to `old_loc.leaf_id`.
pub(crate) fn update_vector_in_tx(
    tx: &mut TxContext<'_>,
    id: DocId,
    old_loc: DocLoc,
    new_vec: &[f32],
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<VectorUpdateResult> {
    let target_leaf = if tx.meta().root_level == 0 {
        tx.meta().root_id
    } else {
        search::route_to_leaf_in_tx(tx, new_vec, config.insert_beam_width, config)?
    };

    if target_leaf == old_loc.leaf_id {
        tx.put_vector(id, new_vec.to_vec());
        let mut leaf = tx.get_leaf_node(old_loc.leaf_id)?;
        let old_centroid = leaf.centroid.clone();
        let count = leaf.entry_count();
        let live_docs = leaf.live_doc_ids();
        let flat_vecs = tx.get_vectors_flat(&live_docs)?;
        leaf.centroid = compute_mean_centroid_flat(&flat_vecs, count, config.dim, config.metric);
        let drifted = cluster_distance(config.metric, &old_centroid, &leaf.centroid)
            > config.centroid_drift_eps;

        let mut delta = ClusterDelta::default();
        let mut scratch = EncodeScratch::default();
        if needs_leaf_compaction(&leaf, config) {
            compact_leaf_in_tx(CompactArgs {
                tx,
                leaf_id: old_loc.leaf_id,
                leaf: &mut leaf,
                flat_vecs: &flat_vecs,
                quantizer,
                dim: config.dim,
                scratch: &mut scratch,
                delta: &mut delta,
            });
        } else if count <= SMALL_LEAF_REENCODE_THRESHOLD || drifted {
            leaf.re_encode_existing(&flat_vecs, quantizer, config.dim, &mut scratch);
        } else {
            quantizer.rotate_into(&leaf.centroid, &mut leaf.rotated_centroid);
            let words = quantizer.num_words();
            let mut code = vec![0u64; words];
            let (res_norm_sq, scale, popcnt) =
                quantizer.encode_into_with_popcnt(new_vec, &leaf.centroid, &mut scratch, &mut code);
            let local_id = leaf.upsert_entry(
                id,
                Some(old_loc.local_id),
                res_norm_sq,
                scale,
                &code,
                popcnt,
                words,
            );
            debug_assert_eq!(local_id, old_loc.local_id);
        }

        let parent_id = leaf.parent_id;
        let new_centroid = leaf.centroid.clone();
        tx.put_leaf_node(old_loc.leaf_id, leaf);

        if drifted && let Some(pid) = parent_id {
            propagate_centroid_upward(
                tx,
                old_loc.leaf_id,
                pid,
                count as u64,
                &old_centroid,
                &new_centroid,
                config,
            )?;
        }

        Ok(VectorUpdateResult::SameLeaf { delta })
    } else {
        let (rem_loc, rem_delta) =
            remove_existing_in_tx(tx, id, config, quantizer)?.ok_or_else(|| {
                TreeError::InvariantViolation(format!("doc {id} missing during vector update"))
            })?;
        let (new_loc, ins_delta, split) = insert_in_tx(tx, id, new_vec, config, quantizer)?;
        Ok(VectorUpdateResult::Moved {
            old_loc: rem_loc,
            new_loc,
            rem_delta,
            ins_delta,
            split,
        })
    }
}

/// Remove `id` inside `tx` if it exists, returning `(removed_doc_loc, other_docs_delta)`.
pub(crate) fn remove_existing_in_tx(
    tx: &mut TxContext<'_>,
    id: DocId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<Option<(DocLoc, ClusterDelta)>> {
    let Some(loc) = tx.get_doc_loc(id)? else {
        return Ok(None);
    };

    let mut leaf = tx.get_leaf_node(loc.leaf_id)?;
    let old_centroid = leaf.centroid.clone();

    leaf.remove_entry(id, Some(loc.local_id), quantizer.num_words());
    tx.delete_vector(id);
    tx.delete_doc_loc(id);
    tx.meta_mut().num_vectors = tx.meta().num_vectors.saturating_sub(1);

    let mut delta = ClusterDelta::default();
    let new_count = leaf.entry_count();
    let will_underflow = loc.leaf_id != tx.meta().root_id && new_count < config.min_leaf_size;
    if new_count == 0 {
        leaf.centroid = vec![0.0; config.dim];
        leaf.rotated_centroid.fill(0.0);
    } else if !will_underflow {
        let mut scratch = EncodeScratch::default();
        if needs_leaf_compaction(&leaf, config) {
            let live_docs = leaf.live_doc_ids();
            let flat_vecs = tx.get_vectors_flat(&live_docs)?;
            leaf.centroid = compute_mean_centroid_flat(
                &flat_vecs,
                new_count,
                config.dim,
                config.metric,
            );
            compact_leaf_in_tx(CompactArgs {
                tx,
                leaf_id: loc.leaf_id,
                leaf: &mut leaf,
                flat_vecs: &flat_vecs,
                quantizer,
                dim: config.dim,
                scratch: &mut scratch,
                delta: &mut delta,
            });
        } else {
            refresh_leaf_after_shrink(tx, &mut leaf, &old_centroid, config, quantizer, &mut scratch)?;
        }
    }

    let parent_id = leaf.parent_id;
    let new_centroid = leaf.centroid.clone();
    tx.put_leaf_node(loc.leaf_id, leaf);

    if loc.leaf_id == tx.meta().root_id {
        return Ok(Some((loc, delta)));
    }

    if will_underflow {
        handle_underflow(tx, loc.leaf_id, config, quantizer, &mut delta)?;
    } else if let Some(pid) = parent_id {
        propagate_centroid_upward(
            tx,
            loc.leaf_id,
            pid,
            new_count as u64,
            &old_centroid,
            &new_centroid,
            config,
        )?;
    }

    Ok(Some((loc, delta)))
}

/// Remove a batch of `DocId`s inside `tx` (deduplicating and ignoring missing IDs),
/// updating each affected leaf once before running underflow merges/redistributions
/// and per-leaf `LocalId` compaction.
pub(crate) fn remove_many_in_tx(
    tx: &mut TxContext<'_>,
    ids: &[DocId],
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<(Vec<(DocId, DocLoc)>, ClusterDelta)> {
    if ids.is_empty() {
        return Ok((Vec::new(), ClusterDelta::default()));
    }
    let mut unique_ids = ids.to_vec();
    unique_ids.sort_unstable();
    unique_ids.dedup();

    let locs = tx.multi_get_doc_locs(&unique_ids)?;
    let mut removed_docs: Vec<(DocId, DocLoc)> = Vec::with_capacity(unique_ids.len());
    let mut by_leaf: BTreeMap<NodeId, Vec<(DocId, LocalId)>> = BTreeMap::new();
    for (doc_id, opt_loc) in unique_ids.into_iter().zip(locs) {
        if let Some(loc) = opt_loc {
            removed_docs.push((doc_id, loc));
            by_leaf.entry(loc.leaf_id).or_default().push((doc_id, loc.local_id));
        }
    }
    if removed_docs.is_empty() {
        return Ok((Vec::new(), ClusterDelta::default()));
    }

    let words = quantizer.num_words();
    let mut underflow_leaves: Vec<NodeId> = Vec::new();
    let mut compaction_candidates: Vec<NodeId> = Vec::new();
    let mut scratch = EncodeScratch::default();

    for (leaf_id, doc_entries) in by_leaf {
        let mut leaf = tx.get_leaf_node(leaf_id)?;
        let old_centroid = leaf.centroid.clone();

        for (doc_id, local_id) in doc_entries {
            leaf.remove_entry(doc_id, Some(local_id), words);
            tx.delete_vector(doc_id);
            tx.delete_doc_loc(doc_id);
            tx.meta_mut().num_vectors = tx.meta().num_vectors.saturating_sub(1);
        }

        let new_count = leaf.entry_count();
        let will_underflow = leaf_id != tx.meta().root_id && new_count < config.min_leaf_size;
        if new_count == 0 {
            leaf.centroid = vec![0.0; config.dim];
            leaf.rotated_centroid.fill(0.0);
        } else if !will_underflow {
            refresh_leaf_after_shrink(tx, &mut leaf, &old_centroid, config, quantizer, &mut scratch)?;
            if needs_leaf_compaction(&leaf, config) {
                compaction_candidates.push(leaf_id);
            }
        }

        let parent_id = leaf.parent_id;
        let new_centroid = leaf.centroid.clone();
        tx.put_leaf_node(leaf_id, leaf);

        if leaf_id != tx.meta().root_id {
            if will_underflow {
                underflow_leaves.push(leaf_id);
            } else if let Some(pid) = parent_id {
                propagate_centroid_upward(
                    tx,
                    leaf_id,
                    pid,
                    new_count as u64,
                    &old_centroid,
                    &new_centroid,
                    config,
                )?;
            }
        }
    }

    let mut delta = ClusterDelta::default();
    for leaf_id in underflow_leaves {
        if leaf_id == tx.meta().root_id {
            continue;
        }
        let Some(leaf) = tx.try_get_leaf_node(leaf_id)? else {
            continue;
        };
        if leaf.entry_count() < config.min_leaf_size {
            handle_underflow(tx, leaf_id, config, quantizer, &mut delta)?;
        }
    }

    for leaf_id in compaction_candidates {
        let Some(mut leaf) = tx.try_get_leaf_node(leaf_id)? else {
            continue;
        };
        if leaf.entry_count() > 0 && needs_leaf_compaction(&leaf, config) {
            let live_docs = leaf.live_doc_ids();
            let flat_vecs = tx.get_vectors_flat(&live_docs)?;
            compact_leaf_in_tx(CompactArgs {
                tx,
                leaf_id,
                leaf: &mut leaf,
                flat_vecs: &flat_vecs,
                quantizer,
                dim: config.dim,
                scratch: &mut scratch,
                delta: &mut delta,
            });
            tx.put_leaf_node(leaf_id, leaf);
        }
    }

    Ok((removed_docs, delta))
}

/// Insert a prepared vector for `id` (which must not currently be present in `tx`),
/// returning `(doc_loc, other_docs_delta, optional_split_pair)`.
pub(crate) fn insert_in_tx(
    tx: &mut TxContext<'_>,
    id: DocId,
    vec: &[f32],
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<(DocLoc, ClusterDelta, Option<SplitPair>)> {
    let target_leaf = if tx.meta().root_level == 0 {
        tx.meta().root_id
    } else {
        search::route_to_leaf_in_tx(tx, vec, config.insert_beam_width, config)?
    };

    let mut leaf = tx.get_leaf_node(target_leaf)?;
    let old_centroid = leaf.centroid.clone();

    tx.put_vector(id, vec.to_vec());
    tx.meta_mut().num_vectors += 1;

    let new_count = leaf.entry_count() + 1;
    let words = quantizer.num_words();
    if new_count <= config.max_leaf_size {
        let mut delta = ClusterDelta::default();
        let needs_compact = leaf.slot_cap > config.max_leaf_size;
        let local_id = if new_count == 1 {
            leaf.centroid = vec.to_vec();
            quantizer.rotate_into(&leaf.centroid, &mut leaf.rotated_centroid);
            let mut scratch = EncodeScratch::default();
            let mut code = vec![0u64; words];
            let (res_norm_sq, scale, popcnt) =
                quantizer.encode_into_with_popcnt(vec, &leaf.centroid, &mut scratch, &mut code);
            leaf.upsert_entry(id, None, res_norm_sq, scale, &code, popcnt, words)
        } else if needs_compact {
            for (lid, did) in leaf.live_slot_doc_pairs() {
                delta.removed.push((did, target_leaf, lid));
            }
            leaf.insert_slot_for_reencode(id, words);
            let doc_ids = leaf.live_doc_ids();
            let flat_vecs = tx.get_vectors_flat(&doc_ids)?;
            leaf.centroid = compute_mean_centroid_flat(
                &flat_vecs,
                new_count,
                config.dim,
                config.metric,
            );
            let mut scratch = EncodeScratch::default();
            leaf.reset_and_encode(&doc_ids, &flat_vecs, quantizer, config.dim, &mut scratch);
            let mut new_id_local = 0;
            for (lid, did) in leaf.live_slot_doc_pairs() {
                if did == id {
                    new_id_local = lid;
                } else {
                    tx.put_doc_pos(did, target_leaf, lid);
                    delta.added.push((did, target_leaf, lid));
                }
            }
            new_id_local
        } else if new_count.is_multiple_of(SMALL_LEAF_REENCODE_THRESHOLD) {
            let lid = leaf.insert_slot_for_reencode(id, words);
            let live_docs = leaf.live_doc_ids();
            let flat_vecs = tx.get_vectors_flat(&live_docs)?;
            leaf.centroid = compute_mean_centroid_flat(
                &flat_vecs,
                new_count,
                config.dim,
                config.metric,
            );
            let mut scratch = EncodeScratch::default();
            leaf.re_encode_existing(&flat_vecs, quantizer, config.dim, &mut scratch);
            lid
        } else {
            leaf.centroid = incremental_mean(&leaf.centroid, vec, new_count, config.metric);
            let drifted = cluster_distance(config.metric, &old_centroid, &leaf.centroid)
                > config.centroid_drift_eps;
            if new_count <= SMALL_LEAF_REENCODE_THRESHOLD || drifted {
                let lid = leaf.insert_slot_for_reencode(id, words);
                let live_docs = leaf.live_doc_ids();
                let flat_vecs = tx.get_vectors_flat(&live_docs)?;
                let mut scratch = EncodeScratch::default();
                leaf.re_encode_existing(&flat_vecs, quantizer, config.dim, &mut scratch);
                lid
            } else {
                quantizer.rotate_into(&leaf.centroid, &mut leaf.rotated_centroid);
                let mut scratch = EncodeScratch::default();
                let mut code = vec![0u64; words];
                let (res_norm_sq, scale, popcnt) =
                    quantizer.encode_into_with_popcnt(vec, &leaf.centroid, &mut scratch, &mut code);
                leaf.upsert_entry(id, None, res_norm_sq, scale, &code, popcnt, words)
            }
        };

        let doc_loc = DocLoc {
            leaf_id: target_leaf,
            local_id,
            fts_doc_id: None,
        };
        tx.put_doc_pos(id, target_leaf, local_id);

        let parent_id = leaf.parent_id;
        let new_centroid = leaf.centroid.clone();
        tx.put_leaf_node(target_leaf, leaf);

        if let Some(pid) = parent_id {
            propagate_centroid_upward(
                tx,
                target_leaf,
                pid,
                new_count as u64,
                &old_centroid,
                &new_centroid,
                config,
            )?;
        }
        Ok((doc_loc, delta, None))
    } else {
        leaf.insert_slot_for_reencode(id, words);
        tx.put_leaf_node(target_leaf, leaf);
        let mut delta = ClusterDelta::default();
        let split_pair = split_node(tx, target_leaf, config, quantizer, &mut delta)?;

        delta.removed.retain(|&(doc_id, _, _)| doc_id != id);
        let mut doc_loc = None;
        delta.added.retain(|&(doc_id, leaf_id, local_id)| {
            if doc_id == id {
                doc_loc = Some(DocLoc {
                    leaf_id,
                    local_id,
                    fts_doc_id: None,
                });
                false
            } else {
                true
            }
        });
        let doc_loc = doc_loc.ok_or_else(|| {
            TreeError::InvariantViolation(format!("split_node lost newly inserted doc {id}"))
        })?;
        Ok((doc_loc, delta, Some(split_pair)))
    }
}

/// Propagate centroid and `subtree_count` changes upward while centroid drift exceeds `centroid_drift_eps`.
pub(crate) fn propagate_centroid_upward(
    tx: &mut TxContext<'_>,
    start_node_id: NodeId,
    start_parent_id: NodeId,
    start_subtree_count: u64,
    old_start_centroid: &[f32],
    new_start_centroid: &[f32],
    config: &TreeConfig,
) -> Result<()> {
    if cluster_distance(config.metric, old_start_centroid, new_start_centroid)
        <= config.centroid_drift_eps
    {
        return Ok(());
    }

    let mut child_id = start_node_id;
    let mut parent_id_opt = Some(start_parent_id);
    let mut child_sub = start_subtree_count;
    let mut child_centroid = new_start_centroid.to_vec();

    while let Some(parent_id) = parent_id_opt {
        let mut parent = tx.get_internal_node(parent_id)?;
        let old_parent_centroid = parent.centroid.clone();
        parent.upsert_child(child_id, child_sub, &child_centroid, config.dim);
        parent.recompute_stats(config.dim, config.metric);

        let next_parent = parent.parent_id;
        let next_sub = parent.subtree_count;
        let next_centroid = parent.centroid.clone();
        tx.put_internal_node(parent_id, parent);

        if cluster_distance(config.metric, &old_parent_centroid, &next_centroid)
            <= config.centroid_drift_eps
        {
            break;
        }

        child_id = parent_id;
        parent_id_opt = next_parent;
        child_sub = next_sub;
        child_centroid = next_centroid;
    }

    Ok(())
}

/// Split an overflowing leaf or internal node into two balanced nodes, cascading upward as needed.
pub(crate) fn split_node(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    delta: &mut ClusterDelta,
) -> Result<SplitPair> {
    let right_id = tx.alloc_node_id();
    if !tx.is_internal_node(node_id) {
        split_leaf(tx, node_id, right_id, config, quantizer, delta)
    } else {
        split_internal(tx, node_id, right_id, config, quantizer, delta)
    }
}

fn split_leaf(
    tx: &mut TxContext<'_>,
    left_id: NodeId,
    right_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    delta: &mut ClusterDelta,
) -> Result<SplitPair> {
    let leaf = tx.get_leaf_node(left_id)?;
    let live_pairs = leaf.live_slot_doc_pairs();
    let mut live_doc_ids = Vec::with_capacity(live_pairs.len());
    for &(lid, did) in &live_pairs {
        delta.removed.push((did, left_id, lid));
        live_doc_ids.push(did);
    }

    let flat_vecs = tx.get_vectors_flat(&live_doc_ids)?;
    let parent_id = leaf.parent_id;
    let old_centroid = leaf.centroid;
    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();

    let weighted: Vec<(&[f32], u64)> = flat_vecs
        .chunks_exact(dim)
        .map(|v| (v, 1u64))
        .collect();
    let mut split = balanced_two_means(
        &weighted,
        config.min_leaf_size,
        config.max_leaf_size,
        dim,
        config.metric,
    );
    split.part1.sort_unstable();
    split.part2.sort_unstable();

    let build_half = |indices: &[usize]| -> (Vec<DocId>, Vec<f32>) {
        let mut ids = Vec::with_capacity(indices.len());
        let mut vecs = Vec::with_capacity(indices.len() * dim);
        for &idx in indices {
            ids.push(live_doc_ids[idx]);
            vecs.extend_from_slice(&flat_vecs[idx * dim..(idx + 1) * dim]);
        }
        (ids, vecs)
    };
    let (left_doc_ids, left_vecs) = build_half(&split.part1);
    let (right_doc_ids, right_vecs) = build_half(&split.part2);

    let mut scratch = EncodeScratch::default();
    let mut encode_leaf = |nid: NodeId, centroid: Vec<f32>, ids: &[DocId], vecs: &[f32]| {
        let mut half = PackedLeafNode::empty(parent_id, dim, padded_dim);
        half.centroid = centroid;
        half.reset_and_encode(ids, vecs, quantizer, dim, &mut scratch);
        for (lid, did) in half.live_slot_doc_pairs() {
            tx.put_doc_pos(did, nid, lid);
            delta.added.push((did, nid, lid));
        }
        half
    };
    let mut left_leaf = encode_leaf(left_id, split.centroid1, &left_doc_ids, &left_vecs);
    let mut right_leaf = encode_leaf(right_id, split.centroid2, &right_doc_ids, &right_vecs);

    let left_c = left_leaf.centroid.clone();
    let right_c = right_leaf.centroid.clone();
    let left_summary = ChildSummary {
        id: left_id,
        subtree_count: left_leaf.entry_count() as u64,
        centroid: &left_c,
    };
    let right_summary = ChildSummary {
        id: right_id,
        subtree_count: right_leaf.entry_count() as u64,
        centroid: &right_c,
    };

    match parent_id {
        None => {
            let new_root_id = tx.alloc_node_id();
            left_leaf.parent_id = Some(new_root_id);
            right_leaf.parent_id = Some(new_root_id);
            tx.put_leaf_node(left_id, left_leaf);
            tx.put_leaf_node(right_id, right_leaf);
            create_new_root(tx, new_root_id, 1, left_summary, right_summary, config);
        }
        Some(pid) => {
            tx.put_leaf_node(left_id, left_leaf);
            tx.put_leaf_node(right_id, right_leaf);
            attach_to_parent(
                tx,
                pid,
                left_summary,
                right_summary,
                config,
                quantizer,
                delta,
            )?;
        }
    }

    Ok(SplitPair {
        left_id,
        right_id,
        old_centroid,
    })
}

fn split_internal(
    tx: &mut TxContext<'_>,
    left_id: NodeId,
    right_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    delta: &mut ClusterDelta,
) -> Result<SplitPair> {
    let node = tx.get_internal_node(left_id)?;
    let level = node.level;
    let parent_id = node.parent_id;
    let old_centroid = node.centroid.clone();
    let dim = config.dim;

    let weighted: Vec<(&[f32], u64)> = node
        .child_centroids
        .chunks_exact(dim)
        .zip(node.child_subtree_counts.iter().copied())
        .collect();
    let mut split = balanced_two_means(
        &weighted,
        config.min_internal_fanout,
        config.max_internal_fanout,
        dim,
        config.metric,
    );
    split.part1.sort_unstable();
    split.part2.sort_unstable();

    let build_half = |indices: &[usize], centroid: Vec<f32>| -> PackedInternalNode {
        let mut half = PackedInternalNode {
            level,
            parent_id,
            subtree_count: 0,
            centroid,
            child_ids: Vec::with_capacity(indices.len()),
            child_subtree_counts: Vec::with_capacity(indices.len()),
            child_centroids: Vec::with_capacity(indices.len() * dim),
        };
        for &idx in indices {
            let sub = node.child_subtree_counts[idx];
            half.subtree_count += sub;
            half.child_ids.push(node.child_ids[idx]);
            half.child_subtree_counts.push(sub);
            half.child_centroids
                .extend_from_slice(node.child_centroid(idx, dim));
        }
        half
    };

    let mut left_node = build_half(&split.part1, split.centroid1);
    let mut right_node = build_half(&split.part2, split.centroid2);
    for &child_id in &right_node.child_ids {
        set_child_parent(tx, child_id, Some(right_id))?;
    }

    let left_c = left_node.centroid.clone();
    let right_c = right_node.centroid.clone();
    let left_summary = ChildSummary {
        id: left_id,
        subtree_count: left_node.subtree_count,
        centroid: &left_c,
    };
    let right_summary = ChildSummary {
        id: right_id,
        subtree_count: right_node.subtree_count,
        centroid: &right_c,
    };

    match parent_id {
        None => {
            let new_root_id = tx.alloc_node_id();
            left_node.parent_id = Some(new_root_id);
            right_node.parent_id = Some(new_root_id);
            tx.put_internal_node(left_id, left_node);
            tx.put_internal_node(right_id, right_node);
            create_new_root(
                tx,
                new_root_id,
                level + 1,
                left_summary,
                right_summary,
                config,
            );
        }
        Some(pid) => {
            tx.put_internal_node(left_id, left_node);
            tx.put_internal_node(right_id, right_node);
            attach_to_parent(
                tx,
                pid,
                left_summary,
                right_summary,
                config,
                quantizer,
                delta,
            )?;
        }
    }

    Ok(SplitPair {
        left_id,
        right_id,
        old_centroid,
    })
}

fn create_new_root(
    tx: &mut TxContext<'_>,
    new_root_id: NodeId,
    new_root_level: u8,
    left: ChildSummary<'_>,
    right: ChildSummary<'_>,
    config: &TreeConfig,
) {
    let root_centroid = compute_weighted_centroid(
        &[
            (left.centroid, left.subtree_count),
            (right.centroid, right.subtree_count),
        ],
        config.dim,
        config.metric,
    );
    let mut root_node = PackedInternalNode {
        level: new_root_level,
        parent_id: None,
        subtree_count: left.subtree_count + right.subtree_count,
        centroid: root_centroid,
        child_ids: Vec::with_capacity(2),
        child_subtree_counts: Vec::with_capacity(2),
        child_centroids: Vec::with_capacity(2 * config.dim),
    };
    root_node.upsert_child(left.id, left.subtree_count, left.centroid, config.dim);
    root_node.upsert_child(right.id, right.subtree_count, right.centroid, config.dim);
    tx.put_internal_node(new_root_id, root_node);
    tx.meta_mut().root_id = new_root_id;
    tx.meta_mut().root_level = new_root_level;
}

fn attach_to_parent(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    left: ChildSummary<'_>,
    right: ChildSummary<'_>,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    delta: &mut ClusterDelta,
) -> Result<()> {
    let mut parent = tx.get_internal_node(parent_id)?;
    let old_parent_centroid = parent.centroid.clone();
    parent.upsert_child(left.id, left.subtree_count, left.centroid, config.dim);
    parent.upsert_child(right.id, right.subtree_count, right.centroid, config.dim);
    parent.recompute_stats(config.dim, config.metric);
    let fanout = parent.entry_count();
    let parent_pid = parent.parent_id;
    let parent_sub = parent.subtree_count;
    let new_parent_centroid = parent.centroid.clone();
    tx.put_internal_node(parent_id, parent);

    if fanout > config.max_internal_fanout {
        split_node(tx, parent_id, config, quantizer, delta)?;
    } else if let Some(gp_id) = parent_pid {
        propagate_centroid_upward(
            tx,
            parent_id,
            gp_id,
            parent_sub,
            &old_parent_centroid,
            &new_parent_centroid,
            config,
        )?;
    }
    Ok(())
}

fn set_child_parent(
    tx: &mut TxContext<'_>,
    child_id: NodeId,
    new_parent: Option<NodeId>,
) -> Result<()> {
    if tx.is_internal_node(child_id) {
        let mut ch = tx.get_internal_node(child_id)?;
        ch.parent_id = new_parent;
        tx.put_internal_node(child_id, ch);
    } else {
        let mut leaf = tx.get_leaf_node(child_id)?;
        leaf.parent_id = new_parent;
        tx.put_leaf_node(child_id, leaf);
    }
    Ok(())
}

/// Resolve an underflowing non-root node via merge, redistribution, or empty-node deletion.
pub(crate) fn handle_underflow(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    delta: &mut ClusterDelta,
) -> Result<()> {
    if node_id == tx.meta().root_id {
        collapse_root(tx)?;
        return Ok(());
    }

    let is_internal = tx.is_internal_node(node_id);
    let (entry_count, parent_id_opt, node_centroid) = if is_internal {
        let n = tx.get_internal_node(node_id)?;
        (n.entry_count(), n.parent_id, n.centroid)
    } else {
        let l = tx.get_leaf_node(node_id)?;
        (l.entry_count(), l.parent_id, l.centroid)
    };

    let min_size = if is_internal {
        config.min_internal_fanout
    } else {
        config.min_leaf_size
    };
    if entry_count >= min_size {
        return Ok(());
    }

    let parent_id = parent_id_opt.ok_or_else(|| {
        TreeError::InvariantViolation(format!("non-root node {node_id} missing parent_id"))
    })?;

    if entry_count == 0 {
        return delete_empty_node(tx, node_id, parent_id, is_internal, config, quantizer, delta);
    }

    let Some(sib_id) = find_closest_sibling(tx, parent_id, node_id, &node_centroid, config)? else {
        if parent_id == tx.meta().root_id {
            collapse_root(tx)?;
            return Ok(());
        }
        return Err(TreeError::InvariantViolation(format!(
            "non-root internal node {parent_id} has only 1 child {node_id}"
        )));
    };

    if !is_internal {
        rebalance_underflowing_leaf(tx, node_id, sib_id, parent_id, config, quantizer, delta)
    } else {
        rebalance_underflowing_internal(tx, node_id, sib_id, parent_id, config, quantizer, delta)
    }
}

fn delete_empty_node(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    parent_id: NodeId,
    is_internal: bool,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    delta: &mut ClusterDelta,
) -> Result<()> {
    if is_internal {
        tx.delete_internal_node(node_id);
    } else {
        tx.delete_leaf_node(node_id);
    }
    let mut parent = tx.get_internal_node(parent_id)?;
    parent.remove_child(node_id, config.dim);
    tx.put_internal_node(parent_id, parent);
    refresh_parent_after_removal(tx, parent_id, config, quantizer, delta)
}

fn find_closest_sibling(
    tx: &TxContext<'_>,
    parent_id: NodeId,
    node_id: NodeId,
    node_centroid: &[f32],
    config: &TreeConfig,
) -> Result<Option<NodeId>> {
    let parent = tx.get_internal_node_ref(parent_id)?;
    let dim = config.dim;
    Ok(parent
        .child_ids
        .iter()
        .enumerate()
        .filter(|&(_, &cid)| cid != node_id)
        .min_by(|&(ia, &id_a), &(ib, &id_b)| {
            let da = cluster_distance(config.metric, node_centroid, parent.child_centroid(ia, dim));
            let db = cluster_distance(config.metric, node_centroid, parent.child_centroid(ib, dim));
            da.total_cmp(&db).then_with(|| id_a.cmp(&id_b))
        })
        .map(|(_, &cid)| cid))
}

fn rebalance_underflowing_leaf(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    sib_id: NodeId,
    parent_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    delta: &mut ClusterDelta,
) -> Result<()> {
    let dim = config.dim;
    let mut left_leaf = tx.get_leaf_node(node_id)?;
    let mut sib_leaf = tx.get_leaf_node(sib_id)?;

    let left_pairs = left_leaf.live_slot_doc_pairs();
    let sib_pairs = sib_leaf.live_slot_doc_pairs();
    for &(lid, did) in &left_pairs {
        delta.removed.push((did, node_id, lid));
    }
    for &(lid, did) in &sib_pairs {
        delta.removed.push((did, sib_id, lid));
    }

    let left_ids: Vec<DocId> = left_pairs.iter().map(|&(_, d)| d).collect();
    let sib_ids: Vec<DocId> = sib_pairs.iter().map(|&(_, d)| d).collect();
    let left_vecs = tx.get_vectors_flat(&left_ids)?;
    let sib_vecs = tx.get_vectors_flat(&sib_ids)?;
    let total = left_ids.len() + sib_ids.len();

    let mut combined: Vec<(DocId, &[f32])> = Vec::with_capacity(total);
    for (i, &did) in left_ids.iter().enumerate() {
        combined.push((did, &left_vecs[i * dim..(i + 1) * dim]));
    }
    for (i, &did) in sib_ids.iter().enumerate() {
        combined.push((did, &sib_vecs[i * dim..(i + 1) * dim]));
    }
    combined.sort_unstable_by_key(|&(did, _)| did);

    let mut scratch = EncodeScratch::default();
    if total <= config.max_leaf_size {
        let mut merged_doc_ids = Vec::with_capacity(total);
        let mut merged_vecs = Vec::with_capacity(total * dim);
        for (did, v) in combined {
            merged_doc_ids.push(did);
            merged_vecs.extend_from_slice(v);
        }

        sib_leaf.centroid = compute_mean_centroid_flat(&merged_vecs, total, dim, config.metric);
        sib_leaf.reset_and_encode(&merged_doc_ids, &merged_vecs, quantizer, dim, &mut scratch);
        for (lid, did) in sib_leaf.live_slot_doc_pairs() {
            tx.put_doc_pos(did, sib_id, lid);
            delta.added.push((did, sib_id, lid));
        }

        let sib_sub = sib_leaf.entry_count() as u64;
        let sib_c = sib_leaf.centroid.clone();
        tx.put_leaf_node(sib_id, sib_leaf);
        tx.delete_leaf_node(node_id);

        let mut parent = tx.get_internal_node(parent_id)?;
        parent.remove_child(node_id, dim);
        parent.upsert_child(sib_id, sib_sub, &sib_c, dim);
        tx.put_internal_node(parent_id, parent);

        refresh_parent_after_removal(tx, parent_id, config, quantizer, delta)
    } else {
        let weighted: Vec<(&[f32], u64)> = combined.iter().map(|&(_, v)| (v, 1u64)).collect();
        let mut split = balanced_two_means(
            &weighted,
            config.min_leaf_size,
            config.max_leaf_size,
            dim,
            config.metric,
        );
        split.part1.sort_unstable();
        split.part2.sort_unstable();

        let mut reencode_half = |nid: NodeId, leaf: &mut PackedLeafNode, indices: &[usize], centroid: Vec<f32>| {
            let mut ids = Vec::with_capacity(indices.len());
            let mut vecs = Vec::with_capacity(indices.len() * dim);
            for &idx in indices {
                let (did, v) = combined[idx];
                ids.push(did);
                vecs.extend_from_slice(v);
            }
            leaf.centroid = centroid;
            leaf.reset_and_encode(&ids, &vecs, quantizer, dim, &mut scratch);
            for (lid, did) in leaf.live_slot_doc_pairs() {
                tx.put_doc_pos(did, nid, lid);
                delta.added.push((did, nid, lid));
            }
        };
        reencode_half(node_id, &mut left_leaf, &split.part1, split.centroid1);
        reencode_half(sib_id, &mut sib_leaf, &split.part2, split.centroid2);

        let left_c = left_leaf.centroid.clone();
        let sib_c = sib_leaf.centroid.clone();
        let left_summary = ChildSummary {
            id: node_id,
            subtree_count: left_leaf.entry_count() as u64,
            centroid: &left_c,
        };
        let sib_summary = ChildSummary {
            id: sib_id,
            subtree_count: sib_leaf.entry_count() as u64,
            centroid: &sib_c,
        };
        tx.put_leaf_node(node_id, left_leaf);
        tx.put_leaf_node(sib_id, sib_leaf);

        update_parent_pair(tx, parent_id, left_summary, sib_summary, config)
    }
}

fn rebalance_underflowing_internal(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    sib_id: NodeId,
    parent_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    delta: &mut ClusterDelta,
) -> Result<()> {
    let dim = config.dim;
    let mut left_node = tx.get_internal_node(node_id)?;
    let mut sib_node = tx.get_internal_node(sib_id)?;
    let total = left_node.entry_count() + sib_node.entry_count();

    if total <= config.max_internal_fanout {
        for (i, &child_id) in left_node.child_ids.iter().enumerate() {
            sib_node.upsert_child(
                child_id,
                left_node.child_subtree_counts[i],
                left_node.child_centroid(i, dim),
                dim,
            );
            set_child_parent(tx, child_id, Some(sib_id))?;
        }
        sib_node.recompute_stats(dim, config.metric);

        let sib_sub = sib_node.subtree_count;
        let sib_c = sib_node.centroid.clone();
        tx.put_internal_node(sib_id, sib_node);
        tx.delete_internal_node(node_id);

        let mut parent = tx.get_internal_node(parent_id)?;
        parent.remove_child(node_id, dim);
        parent.upsert_child(sib_id, sib_sub, &sib_c, dim);
        tx.put_internal_node(parent_id, parent);

        refresh_parent_after_removal(tx, parent_id, config, quantizer, delta)
    } else {
        struct ChildItem {
            id: NodeId,
            sub: u64,
            centroid: Vec<f32>,
            was_left: bool,
        }
        let mut combined: Vec<ChildItem> = Vec::with_capacity(total);
        for (i, &cid) in left_node.child_ids.iter().enumerate() {
            combined.push(ChildItem {
                id: cid,
                sub: left_node.child_subtree_counts[i],
                centroid: left_node.child_centroid(i, dim).to_vec(),
                was_left: true,
            });
        }
        for (i, &cid) in sib_node.child_ids.iter().enumerate() {
            combined.push(ChildItem {
                id: cid,
                sub: sib_node.child_subtree_counts[i],
                centroid: sib_node.child_centroid(i, dim).to_vec(),
                was_left: false,
            });
        }
        combined.sort_unstable_by_key(|item| item.id);

        let weighted: Vec<(&[f32], u64)> = combined
            .iter()
            .map(|c| (c.centroid.as_slice(), c.sub))
            .collect();
        let mut split = balanced_two_means(
            &weighted,
            config.min_internal_fanout,
            config.max_internal_fanout,
            dim,
            config.metric,
        );
        split.part1.sort_unstable();
        split.part2.sort_unstable();

        let mut refill =
            |target_id: NodeId, node: &mut PackedInternalNode, indices: &[usize], centroid: Vec<f32>, expect_left: bool| -> Result<()> {
                node.child_ids.clear();
                node.child_subtree_counts.clear();
                node.child_centroids.clear();
                node.subtree_count = 0;
                node.centroid = centroid;
                for &idx in indices {
                    let item = &combined[idx];
                    node.subtree_count += item.sub;
                    node.child_ids.push(item.id);
                    node.child_subtree_counts.push(item.sub);
                    node.child_centroids.extend_from_slice(&item.centroid);
                    if item.was_left != expect_left {
                        set_child_parent(tx, item.id, Some(target_id))?;
                    }
                }
                Ok(())
            };
        refill(node_id, &mut left_node, &split.part1, split.centroid1, true)?;
        refill(sib_id, &mut sib_node, &split.part2, split.centroid2, false)?;

        let left_c = left_node.centroid.clone();
        let sib_c = sib_node.centroid.clone();
        let left_summary = ChildSummary {
            id: node_id,
            subtree_count: left_node.subtree_count,
            centroid: &left_c,
        };
        let sib_summary = ChildSummary {
            id: sib_id,
            subtree_count: sib_node.subtree_count,
            centroid: &sib_c,
        };
        tx.put_internal_node(node_id, left_node);
        tx.put_internal_node(sib_id, sib_node);

        update_parent_pair(tx, parent_id, left_summary, sib_summary, config)
    }
}

fn update_parent_pair(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    left: ChildSummary<'_>,
    right: ChildSummary<'_>,
    config: &TreeConfig,
) -> Result<()> {
    let mut parent = tx.get_internal_node(parent_id)?;
    let old_parent_centroid = parent.centroid.clone();
    parent.upsert_child(left.id, left.subtree_count, left.centroid, config.dim);
    parent.upsert_child(right.id, right.subtree_count, right.centroid, config.dim);
    parent.recompute_stats(config.dim, config.metric);

    let gp_id_opt = parent.parent_id;
    let parent_sub = parent.subtree_count;
    let new_parent_centroid = parent.centroid.clone();
    tx.put_internal_node(parent_id, parent);

    if let Some(gp_id) = gp_id_opt {
        propagate_centroid_upward(
            tx,
            parent_id,
            gp_id,
            parent_sub,
            &old_parent_centroid,
            &new_parent_centroid,
            config,
        )?;
    }
    Ok(())
}

fn refresh_parent_after_removal(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    delta: &mut ClusterDelta,
) -> Result<()> {
    let mut parent = tx.get_internal_node(parent_id)?;
    let old_parent_centroid = parent.centroid.clone();
    parent.recompute_stats(config.dim, config.metric);
    let fanout = parent.entry_count();
    let gp_id_opt = parent.parent_id;
    let parent_sub = parent.subtree_count;
    let new_parent_centroid = parent.centroid.clone();
    tx.put_internal_node(parent_id, parent);

    if parent_id == tx.meta().root_id {
        if fanout == 1 {
            collapse_root(tx)?;
        }
    } else if fanout < config.min_internal_fanout {
        handle_underflow(tx, parent_id, config, quantizer, delta)?;
    } else if let Some(gp_id) = gp_id_opt {
        propagate_centroid_upward(
            tx,
            parent_id,
            gp_id,
            parent_sub,
            &old_parent_centroid,
            &new_parent_centroid,
            config,
        )?;
    }
    Ok(())
}

fn collapse_root(tx: &mut TxContext<'_>) -> Result<()> {
    while tx.meta().root_level > 0 {
        let root_id = tx.meta().root_id;
        let root = tx.get_internal_node(root_id)?;
        if root.entry_count() != 1 {
            break;
        }
        let only_child_id = root.child_ids[0];
        let child_level = if tx.is_internal_node(only_child_id) {
            let mut ch = tx.get_internal_node(only_child_id)?;
            ch.parent_id = None;
            let lvl = ch.level;
            tx.put_internal_node(only_child_id, ch);
            lvl
        } else {
            let mut leaf = tx.get_leaf_node(only_child_id)?;
            leaf.parent_id = None;
            tx.put_leaf_node(only_child_id, leaf);
            0
        };
        tx.delete_internal_node(root_id);
        tx.meta_mut().root_id = only_child_id;
        tx.meta_mut().root_level = child_level;
    }
    Ok(())
}
