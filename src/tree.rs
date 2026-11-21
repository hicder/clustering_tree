use std::collections::HashMap;
use std::path::Path;

use crate::config::{DocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::kmeans::balanced_two_means;
use crate::lire;
use crate::metric::{
    cluster_distance, compute_mean_centroid_flat, compute_weighted_centroid, incremental_mean,
    prepare_vector, validate_vector,
};
use crate::node::{PackedInternalNode, PackedLeafNode, PackedLeafVectors, TreeMeta};
use crate::rabitq::{EncodeScratch, RabitqQuantizer};
use crate::search;
use crate::storage::{Storage, TxContext};
use crate::verify::{self, TreeStats};

/// Metadata about a completed leaf split passed to SPFresh LIRE.
#[derive(Debug, Clone)]
pub(crate) struct SplitPair {
    pub left_id: NodeId,
    pub right_id: NodeId,
    pub old_centroid: Vec<f32>,
}

/// RocksDB-backed mutable hierarchical clustering tree (~100x fanout)
/// with in-memory internal nodes, packed RaBitQ posting lists, and leaf-grouped full vectors.
pub struct ClusteringTree {
    pub(crate) storage: Storage,
    pub(crate) config: TreeConfig,
    pub(crate) meta: TreeMeta,
    pub(crate) quantizer: RabitqQuantizer,
    pub(crate) internal_cache: HashMap<NodeId, PackedInternalNode>,
}

impl ClusteringTree {
    /// Open or create a [`ClusteringTree`] at `path` with the given `config`.
    pub fn open(path: impl AsRef<Path>, config: TreeConfig) -> Result<Self> {
        config.validate()?;
        let (storage, meta) = Storage::open(path, &config)?;
        let quantizer = RabitqQuantizer::new(config.dim, meta.rabitq_seed);
        let internal_cache: HashMap<NodeId, PackedInternalNode> =
            storage.all_internal_nodes()?.into_iter().collect();
        Ok(Self {
            storage,
            config,
            meta,
            quantizer,
            internal_cache,
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

    /// Insert or update a vector in the tree, quantizing it for the leaf posting list
    /// while persisting the full-precision vector in `cf_leaf_vectors`.
    pub fn add_leaf(&mut self, id: DocId, embedding: &[f32]) -> Result<()> {
        let vec = prepare_vector(embedding, self.config.dim, self.config.metric)?;

        // Idempotency / upsert: remove existing location first if present.
        if self.storage.get_doc_loc(id)?.is_some() {
            self.remove_doc_internal(id)?;
        }

        let target_leaf = if self.meta.root_level == 0 {
            self.meta.root_id
        } else {
            search::route_to_leaf(
                &self.internal_cache,
                &self.meta,
                &vec,
                self.config.insert_beam_width,
                &self.config,
            )?
        };

        let mut tx = TxContext::new(&self.storage, &mut self.internal_cache, self.meta.clone());
        let mut leaf = tx.get_leaf_node(target_leaf)?;
        let mut leaf_vecs = tx.get_leaf_vectors(target_leaf)?;
        let old_centroid = leaf.centroid.clone();

        leaf_vecs.upsert(id, &vec, self.config.dim);
        tx.put_doc_loc(id, target_leaf);
        tx.meta_mut().num_vectors += 1;

        let new_count = leaf_vecs.doc_ids.len();
        if new_count <= self.config.max_leaf_size {
            if new_count == 1 {
                leaf.centroid = vec.clone();
                self.quantizer
                    .rotate_into(&leaf.centroid, &mut leaf.rotated_centroid);
                let q = self.quantizer.encode(&vec, &leaf.centroid);
                leaf.upsert_entry(
                    id,
                    q.res_norm_sq,
                    q.scale,
                    &q.code,
                    self.quantizer.num_words(),
                );
            } else if new_count.is_multiple_of(16) {
                leaf.centroid = compute_mean_centroid_flat(
                    &leaf_vecs.vectors,
                    new_count,
                    self.config.dim,
                    self.config.metric,
                );
                let mut scratch = EncodeScratch::default();
                leaf.re_encode_all(&leaf_vecs, &self.quantizer, self.config.dim, &mut scratch);
            } else {
                leaf.centroid =
                    incremental_mean(&leaf.centroid, &vec, new_count, self.config.metric);
                let drifted = cluster_distance(self.config.metric, &old_centroid, &leaf.centroid)
                    > self.config.centroid_drift_eps;
                if new_count <= 16 || drifted {
                    let mut scratch = EncodeScratch::default();
                    leaf.re_encode_all(&leaf_vecs, &self.quantizer, self.config.dim, &mut scratch);
                } else {
                    self.quantizer
                        .rotate_into(&leaf.centroid, &mut leaf.rotated_centroid);
                    let q = self.quantizer.encode(&vec, &leaf.centroid);
                    leaf.upsert_entry(
                        id,
                        q.res_norm_sq,
                        q.scale,
                        &q.code,
                        self.quantizer.num_words(),
                    );
                }
            }

            let parent_id = leaf.parent_id;
            let new_centroid = leaf.centroid.clone();
            tx.put_leaf_node(target_leaf, leaf);
            tx.put_leaf_vectors(target_leaf, leaf_vecs);

            if let Some(pid) = parent_id {
                propagate_centroid_upward(
                    &mut tx,
                    target_leaf,
                    pid,
                    new_count as u64,
                    &old_centroid,
                    &new_centroid,
                    &self.config,
                )?;
            }
            self.meta = tx.commit()?;
            Ok(())
        } else {
            tx.put_leaf_node(target_leaf, leaf);
            tx.put_leaf_vectors(target_leaf, leaf_vecs);
            let split_pair = split_node(&mut tx, target_leaf, &self.config, &self.quantizer)?;
            self.meta = tx.commit()?;

            if self.config.enable_lire && self.meta.root_level > 0 {
                lire::run_lire(self, split_pair)?;
            }
            Ok(())
        }
    }

    /// Remove a vector from the tree, triggering merges, redistributions, or root collapse as needed.
    ///
    /// If `id` is not present in the tree, returns `Ok(())`.
    pub fn remove_leaf(&mut self, id: DocId, embedding: &[f32]) -> Result<()> {
        validate_vector(embedding, self.config.dim)?;
        self.remove_doc_internal(id)
    }

    /// Approximate nearest neighbor search using RaBitQ-quantized posting lists.
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Result<Vec<(DocId, f32)>> {
        search::search(
            &self.storage,
            &self.internal_cache,
            &self.meta,
            &self.quantizer,
            query,
            search::SearchOptions {
                k,
                beam_width,
                rerank_candidates: None,
            },
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
            search::SearchOptions {
                k,
                beam_width,
                rerank_candidates: Some(rerank_candidates),
            },
            &self.config,
        )
    }

    /// Exhaustively verify all structural invariants and return tree statistics.
    pub fn verify_invariants(&self) -> Result<TreeStats> {
        verify::verify_invariants(&self.storage, &self.internal_cache, &self.meta, &self.config)
    }

    fn remove_doc_internal(&mut self, id: DocId) -> Result<()> {
        let Some(leaf_id) = self.storage.get_doc_loc(id)? else {
            return Ok(());
        };

        let mut tx = TxContext::new(&self.storage, &mut self.internal_cache, self.meta.clone());
        let mut leaf = tx.get_leaf_node(leaf_id)?;
        let mut leaf_vecs = tx.get_leaf_vectors(leaf_id)?;
        let old_centroid = leaf.centroid.clone();

        leaf.remove_entry(id, self.quantizer.num_words());
        leaf_vecs.remove(id, self.config.dim);
        tx.delete_doc_loc(id);
        tx.meta_mut().num_vectors = tx.meta().num_vectors.saturating_sub(1);

        let new_count = leaf_vecs.doc_ids.len();
        let will_underflow = leaf_id != tx.meta().root_id && new_count < self.config.min_leaf_size;
        if new_count == 0 {
            leaf.centroid = vec![0.0; self.config.dim];
            leaf.rotated_centroid.fill(0.0);
        } else {
            leaf.centroid = compute_mean_centroid_flat(
                &leaf_vecs.vectors,
                new_count,
                self.config.dim,
                self.config.metric,
            );
            if !will_underflow
                && (new_count <= 16
                    || cluster_distance(self.config.metric, &old_centroid, &leaf.centroid)
                        > self.config.centroid_drift_eps)
            {
                let mut scratch = EncodeScratch::default();
                leaf.re_encode_all(&leaf_vecs, &self.quantizer, self.config.dim, &mut scratch);
            } else if !will_underflow {
                self.quantizer
                    .rotate_into(&leaf.centroid, &mut leaf.rotated_centroid);
            }
        }

        let parent_id = leaf.parent_id;
        let new_centroid = leaf.centroid.clone();
        tx.put_leaf_node(leaf_id, leaf);
        tx.put_leaf_vectors(leaf_id, leaf_vecs);

        if leaf_id == tx.meta().root_id {
            self.meta = tx.commit()?;
            return Ok(());
        }

        if will_underflow {
            handle_underflow(&mut tx, leaf_id, &self.config, &self.quantizer)?;
        } else if let Some(pid) = parent_id {
            propagate_centroid_upward(
                &mut tx,
                leaf_id,
                pid,
                new_count as u64,
                &old_centroid,
                &new_centroid,
                &self.config,
            )?;
        }

        self.meta = tx.commit()?;
        Ok(())
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
) -> Result<SplitPair> {
    let is_internal = tx.is_internal_node(node_id);
    let right_id = tx.alloc_node_id();

    if !is_internal {
        let leaf = tx.get_leaf_node(node_id)?;
        let vecs = tx.get_leaf_vectors(node_id)?;
        let parent_id = leaf.parent_id;
        let old_centroid = leaf.centroid;
        let dim = config.dim;
        let padded_dim = quantizer.padded_dim();

        let weighted: Vec<(&[f32], u64)> = vecs
            .vectors
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
        // Preserve sorted doc_id order in both halves.
        split.part1.sort_unstable();
        split.part2.sort_unstable();

        let mut left_vecs = PackedLeafVectors {
            doc_ids: Vec::with_capacity(split.part1.len()),
            vectors: Vec::with_capacity(split.part1.len() * dim),
        };
        for &idx in &split.part1 {
            left_vecs.doc_ids.push(vecs.doc_ids[idx]);
            left_vecs.vectors.extend_from_slice(vecs.vec_at(idx, dim));
        }

        let mut right_vecs = PackedLeafVectors {
            doc_ids: Vec::with_capacity(split.part2.len()),
            vectors: Vec::with_capacity(split.part2.len() * dim),
        };
        for &idx in &split.part2 {
            let doc_id = vecs.doc_ids[idx];
            right_vecs.doc_ids.push(doc_id);
            right_vecs.vectors.extend_from_slice(vecs.vec_at(idx, dim));
            tx.put_doc_loc(doc_id, right_id);
        }

        let mut scratch = EncodeScratch::default();
        let mut left_leaf = PackedLeafNode::empty(parent_id, dim, padded_dim);
        left_leaf.centroid = split.centroid1;
        left_leaf.re_encode_all(&left_vecs, quantizer, dim, &mut scratch);

        let mut right_leaf = PackedLeafNode::empty(parent_id, dim, padded_dim);
        right_leaf.centroid = split.centroid2;
        right_leaf.re_encode_all(&right_vecs, quantizer, dim, &mut scratch);

        let left_sub = left_leaf.entry_count() as u64;
        let right_sub = right_leaf.entry_count() as u64;
        let left_c = left_leaf.centroid.clone();
        let right_c = right_leaf.centroid.clone();

        tx.put_leaf_vectors(node_id, left_vecs);
        tx.put_leaf_vectors(right_id, right_vecs);

        match parent_id {
            None => {
                let new_root_id = tx.alloc_node_id();
                left_leaf.parent_id = Some(new_root_id);
                right_leaf.parent_id = Some(new_root_id);
                tx.put_leaf_node(node_id, left_leaf);
                tx.put_leaf_node(right_id, right_leaf);

                create_new_root(
                    tx,
                    new_root_id,
                    1,
                    node_id,
                    left_sub,
                    &left_c,
                    right_id,
                    right_sub,
                    &right_c,
                    config,
                );
            }
            Some(pid) => {
                tx.put_leaf_node(node_id, left_leaf);
                tx.put_leaf_node(right_id, right_leaf);
                attach_to_parent(
                    tx,
                    pid,
                    node_id,
                    left_sub,
                    &left_c,
                    right_id,
                    right_sub,
                    &right_c,
                    config,
                    quantizer,
                )?;
            }
        }

        Ok(SplitPair {
            left_id: node_id,
            right_id,
            old_centroid,
        })
    } else {
        let node = tx.get_internal_node(node_id)?;
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

        let mut left_node = PackedInternalNode {
            level,
            parent_id,
            subtree_count: 0,
            centroid: split.centroid1,
            child_ids: Vec::with_capacity(split.part1.len()),
            child_subtree_counts: Vec::with_capacity(split.part1.len()),
            child_centroids: Vec::with_capacity(split.part1.len() * dim),
        };
        for &idx in &split.part1 {
            let sub = node.child_subtree_counts[idx];
            left_node.subtree_count += sub;
            left_node.child_ids.push(node.child_ids[idx]);
            left_node.child_subtree_counts.push(sub);
            left_node
                .child_centroids
                .extend_from_slice(node.child_centroid(idx, dim));
        }

        let mut right_node = PackedInternalNode {
            level,
            parent_id,
            subtree_count: 0,
            centroid: split.centroid2,
            child_ids: Vec::with_capacity(split.part2.len()),
            child_subtree_counts: Vec::with_capacity(split.part2.len()),
            child_centroids: Vec::with_capacity(split.part2.len() * dim),
        };
        for &idx in &split.part2 {
            let child_id = node.child_ids[idx];
            let sub = node.child_subtree_counts[idx];
            right_node.subtree_count += sub;
            right_node.child_ids.push(child_id);
            right_node.child_subtree_counts.push(sub);
            right_node
                .child_centroids
                .extend_from_slice(node.child_centroid(idx, dim));
            set_child_parent(tx, child_id, Some(right_id))?;
        }

        let left_sub = left_node.subtree_count;
        let right_sub = right_node.subtree_count;
        let left_c = left_node.centroid.clone();
        let right_c = right_node.centroid.clone();

        match parent_id {
            None => {
                let new_root_id = tx.alloc_node_id();
                left_node.parent_id = Some(new_root_id);
                right_node.parent_id = Some(new_root_id);
                tx.put_internal_node(node_id, left_node);
                tx.put_internal_node(right_id, right_node);

                create_new_root(
                    tx,
                    new_root_id,
                    level + 1,
                    node_id,
                    left_sub,
                    &left_c,
                    right_id,
                    right_sub,
                    &right_c,
                    config,
                );
            }
            Some(pid) => {
                tx.put_internal_node(node_id, left_node);
                tx.put_internal_node(right_id, right_node);
                attach_to_parent(
                    tx,
                    pid,
                    node_id,
                    left_sub,
                    &left_c,
                    right_id,
                    right_sub,
                    &right_c,
                    config,
                    quantizer,
                )?;
            }
        }

        Ok(SplitPair {
            left_id: node_id,
            right_id,
            old_centroid,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn create_new_root(
    tx: &mut TxContext<'_>,
    new_root_id: NodeId,
    new_root_level: u8,
    left_id: NodeId,
    left_sub: u64,
    left_centroid: &[f32],
    right_id: NodeId,
    right_sub: u64,
    right_centroid: &[f32],
    config: &TreeConfig,
) {
    let root_centroid = compute_weighted_centroid(
        &[(left_centroid, left_sub), (right_centroid, right_sub)],
        config.dim,
        config.metric,
    );
    let mut root_node = PackedInternalNode {
        level: new_root_level,
        parent_id: None,
        subtree_count: left_sub + right_sub,
        centroid: root_centroid,
        child_ids: Vec::with_capacity(2),
        child_subtree_counts: Vec::with_capacity(2),
        child_centroids: Vec::with_capacity(2 * config.dim),
    };
    root_node.upsert_child(left_id, left_sub, left_centroid, config.dim);
    root_node.upsert_child(right_id, right_sub, right_centroid, config.dim);
    tx.put_internal_node(new_root_id, root_node);
    tx.meta_mut().root_id = new_root_id;
    tx.meta_mut().root_level = new_root_level;
}

#[allow(clippy::too_many_arguments)]
fn attach_to_parent(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    left_id: NodeId,
    left_sub: u64,
    left_centroid: &[f32],
    right_id: NodeId,
    right_sub: u64,
    right_centroid: &[f32],
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let mut parent = tx.get_internal_node(parent_id)?;
    let old_parent_centroid = parent.centroid.clone();
    parent.upsert_child(left_id, left_sub, left_centroid, config.dim);
    parent.upsert_child(right_id, right_sub, right_centroid, config.dim);
    parent.recompute_stats(config.dim, config.metric);
    let fanout = parent.entry_count();
    let parent_pid = parent.parent_id;
    let parent_sub = parent.subtree_count;
    let new_parent_centroid = parent.centroid.clone();
    tx.put_internal_node(parent_id, parent);

    if fanout > config.max_internal_fanout {
        split_node(tx, parent_id, config, quantizer)?;
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

    let (min_size, max_size) = if is_internal {
        (config.min_internal_fanout, config.max_internal_fanout)
    } else {
        (config.min_leaf_size, config.max_leaf_size)
    };

    if entry_count >= min_size {
        return Ok(());
    }

    let parent_id = parent_id_opt.ok_or_else(|| {
        TreeError::InvariantViolation(format!("non-root node {node_id} missing parent_id"))
    })?;

    if entry_count == 0 {
        if is_internal {
            tx.delete_internal_node(node_id);
        } else {
            tx.delete_leaf_node(node_id);
            tx.delete_leaf_vectors(node_id);
        }
        let mut parent = tx.get_internal_node(parent_id)?;
        parent.remove_child(node_id, config.dim);
        tx.put_internal_node(parent_id, parent);
        refresh_parent_after_removal(tx, parent_id, config, quantizer)?;
        return Ok(());
    }

    let parent = tx.get_internal_node(parent_id)?;
    let dim = config.dim;
    let sib_id = parent
        .child_ids
        .iter()
        .enumerate()
        .filter(|&(_, &cid)| cid != node_id)
        .min_by(|&(ia, &id_a), &(ib, &id_b)| {
            let da = cluster_distance(config.metric, &node_centroid, parent.child_centroid(ia, dim));
            let db = cluster_distance(config.metric, &node_centroid, parent.child_centroid(ib, dim));
            da.total_cmp(&db).then_with(|| id_a.cmp(&id_b))
        })
        .map(|(_, &cid)| cid);

    let Some(sib_id) = sib_id else {
        if parent_id == tx.meta().root_id {
            collapse_root(tx)?;
            return Ok(());
        }
        return Err(TreeError::InvariantViolation(format!(
            "non-root internal node {parent_id} has only 1 child {node_id}"
        )));
    };

    if !is_internal {
        let mut left_leaf = tx.get_leaf_node(node_id)?;
        let mut sib_leaf = tx.get_leaf_node(sib_id)?;
        let left_vecs = tx.get_leaf_vectors(node_id)?;
        let mut sib_vecs = tx.get_leaf_vectors(sib_id)?;
        let total = left_vecs.doc_ids.len() + sib_vecs.doc_ids.len();

        if total <= max_size {
            for (i, &doc_id) in left_vecs.doc_ids.iter().enumerate() {
                sib_vecs.upsert(doc_id, left_vecs.vec_at(i, dim), dim);
                tx.put_doc_loc(doc_id, sib_id);
            }
            sib_leaf.centroid = compute_mean_centroid_flat(
                &sib_vecs.vectors,
                sib_vecs.doc_ids.len(),
                dim,
                config.metric,
            );
            let mut scratch = EncodeScratch::default();
            sib_leaf.re_encode_all(&sib_vecs, quantizer, dim, &mut scratch);

            let sib_sub = sib_leaf.entry_count() as u64;
            let sib_c = sib_leaf.centroid.clone();
            tx.put_leaf_node(sib_id, sib_leaf);
            tx.put_leaf_vectors(sib_id, sib_vecs);
            tx.delete_leaf_node(node_id);
            tx.delete_leaf_vectors(node_id);

            let mut parent = tx.get_internal_node(parent_id)?;
            parent.remove_child(node_id, dim);
            parent.upsert_child(sib_id, sib_sub, &sib_c, dim);
            tx.put_internal_node(parent_id, parent);

            refresh_parent_after_removal(tx, parent_id, config, quantizer)?;
        } else {
            let mut combined: Vec<(DocId, &[f32], bool)> = Vec::with_capacity(total);
            for (i, &did) in left_vecs.doc_ids.iter().enumerate() {
                combined.push((did, left_vecs.vec_at(i, dim), true));
            }
            for (i, &did) in sib_vecs.doc_ids.iter().enumerate() {
                combined.push((did, sib_vecs.vec_at(i, dim), false));
            }
            combined.sort_unstable_by_key(|&(did, _, _)| did);

            let weighted: Vec<(&[f32], u64)> = combined.iter().map(|&(_, v, _)| (v, 1u64)).collect();
            let mut split = balanced_two_means(&weighted, min_size, max_size, dim, config.metric);
            split.part1.sort_unstable();
            split.part2.sort_unstable();

            let mut new_left_vecs = PackedLeafVectors {
                doc_ids: Vec::with_capacity(split.part1.len()),
                vectors: Vec::with_capacity(split.part1.len() * dim),
            };
            for &idx in &split.part1 {
                let (did, v, was_left) = combined[idx];
                new_left_vecs.doc_ids.push(did);
                new_left_vecs.vectors.extend_from_slice(v);
                if !was_left {
                    tx.put_doc_loc(did, node_id);
                }
            }

            let mut new_right_vecs = PackedLeafVectors {
                doc_ids: Vec::with_capacity(split.part2.len()),
                vectors: Vec::with_capacity(split.part2.len() * dim),
            };
            for &idx in &split.part2 {
                let (did, v, was_left) = combined[idx];
                new_right_vecs.doc_ids.push(did);
                new_right_vecs.vectors.extend_from_slice(v);
                if was_left {
                    tx.put_doc_loc(did, sib_id);
                }
            }

            let mut scratch = EncodeScratch::default();
            left_leaf.centroid = split.centroid1;
            left_leaf.re_encode_all(&new_left_vecs, quantizer, dim, &mut scratch);
            sib_leaf.centroid = split.centroid2;
            sib_leaf.re_encode_all(&new_right_vecs, quantizer, dim, &mut scratch);

            let left_sub = left_leaf.entry_count() as u64;
            let sib_sub = sib_leaf.entry_count() as u64;
            let left_c = left_leaf.centroid.clone();
            let sib_c = sib_leaf.centroid.clone();

            tx.put_leaf_node(node_id, left_leaf);
            tx.put_leaf_vectors(node_id, new_left_vecs);
            tx.put_leaf_node(sib_id, sib_leaf);
            tx.put_leaf_vectors(sib_id, new_right_vecs);

            update_parent_pair(
                tx, parent_id, node_id, left_sub, &left_c, sib_id, sib_sub, &sib_c, config,
            )?;
        }
    } else {
        let mut left_node = tx.get_internal_node(node_id)?;
        let mut sib_node = tx.get_internal_node(sib_id)?;
        let total = left_node.entry_count() + sib_node.entry_count();

        if total <= max_size {
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

            refresh_parent_after_removal(tx, parent_id, config, quantizer)?;
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
            let mut split = balanced_two_means(&weighted, min_size, max_size, dim, config.metric);
            split.part1.sort_unstable();
            split.part2.sort_unstable();

            left_node.child_ids.clear();
            left_node.child_subtree_counts.clear();
            left_node.child_centroids.clear();
            left_node.subtree_count = 0;
            left_node.centroid = split.centroid1;
            for &idx in &split.part1 {
                let item = &combined[idx];
                left_node.subtree_count += item.sub;
                left_node.child_ids.push(item.id);
                left_node.child_subtree_counts.push(item.sub);
                left_node.child_centroids.extend_from_slice(&item.centroid);
                if !item.was_left {
                    set_child_parent(tx, item.id, Some(node_id))?;
                }
            }

            sib_node.child_ids.clear();
            sib_node.child_subtree_counts.clear();
            sib_node.child_centroids.clear();
            sib_node.subtree_count = 0;
            sib_node.centroid = split.centroid2;
            for &idx in &split.part2 {
                let item = &combined[idx];
                sib_node.subtree_count += item.sub;
                sib_node.child_ids.push(item.id);
                sib_node.child_subtree_counts.push(item.sub);
                sib_node.child_centroids.extend_from_slice(&item.centroid);
                if item.was_left {
                    set_child_parent(tx, item.id, Some(sib_id))?;
                }
            }

            let left_sub = left_node.subtree_count;
            let sib_sub = sib_node.subtree_count;
            let left_c = left_node.centroid.clone();
            let sib_c = sib_node.centroid.clone();

            tx.put_internal_node(node_id, left_node);
            tx.put_internal_node(sib_id, sib_node);

            update_parent_pair(
                tx, parent_id, node_id, left_sub, &left_c, sib_id, sib_sub, &sib_c, config,
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_parent_pair(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    left_id: NodeId,
    left_sub: u64,
    left_c: &[f32],
    right_id: NodeId,
    right_sub: u64,
    right_c: &[f32],
    config: &TreeConfig,
) -> Result<()> {
    let mut parent = tx.get_internal_node(parent_id)?;
    let old_parent_centroid = parent.centroid.clone();
    parent.upsert_child(left_id, left_sub, left_c, config.dim);
    parent.upsert_child(right_id, right_sub, right_c, config.dim);
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
        handle_underflow(tx, parent_id, config, quantizer)?;
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
