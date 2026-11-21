use std::path::Path;

use crate::config::{DocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::kmeans::balanced_two_means;
use crate::lire;
use crate::metric::{
    cluster_distance, compute_mean_centroid, compute_weighted_centroid, incremental_mean,
    prepare_vector, validate_vector,
};
use crate::node::{InternalEntry, NodeHeader, TreeMeta};
use crate::rabitq::RabitqQuantizer;
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
/// with RaBitQ-quantized posting lists and full-precision vector storage.
pub struct ClusteringTree {
    pub(crate) storage: Storage,
    pub(crate) config: TreeConfig,
    pub(crate) meta: TreeMeta,
    pub(crate) quantizer: RabitqQuantizer,
}

impl ClusteringTree {
    /// Open or create a [`ClusteringTree`] at `path` with the given `config`.
    pub fn open(path: impl AsRef<Path>, config: TreeConfig) -> Result<Self> {
        config.validate()?;
        let (storage, meta) = Storage::open(path, &config)?;
        let quantizer = RabitqQuantizer::new(config.dim, meta.rabitq_seed);
        Ok(Self {
            storage,
            config,
            meta,
            quantizer,
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

    /// Retrieve the stored full-precision vector for `id` from `cf_vectors`.
    pub fn get_vector(&self, id: DocId) -> Result<Option<Vec<f32>>> {
        self.storage.get_vector(id)
    }

    /// Insert or update a vector in the tree, quantizing it for the leaf posting list
    /// while persisting the full-precision vector in `cf_vectors`.
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
                &self.storage,
                &self.meta,
                &vec,
                self.config.insert_beam_width,
                &self.config,
            )?
        };

        let mut tx = TxContext::new(&self.storage, self.meta.clone());
        let mut header = tx.get_header(target_leaf)?;
        let old_centroid = header.centroid.clone();

        tx.put_vector(id, vec.clone());
        tx.put_doc_loc(id, target_leaf);
        tx.meta_mut().num_vectors += 1;

        let new_count = header.entry_count as usize + 1;
        header.entry_count = new_count as u32;
        header.subtree_count = new_count as u64;

        if new_count <= self.config.max_leaf_size {
            if new_count == 1 {
                header.centroid = vec.clone();
                let q_entry = self.quantizer.encode(&vec, &header.centroid);
                tx.put_leaf_entry(target_leaf, id, q_entry);
            } else if new_count.is_multiple_of(16) {
                // Periodically recompute exact centroid and re-quantize the leaf to prevent drift.
                let mut doc_ids = tx.get_leaf_doc_ids(target_leaf)?;
                doc_ids.push(id);
                let items = tx.get_vectors_batch(&doc_ids)?;
                let refs: Vec<&[f32]> = items.iter().map(|(_, v)| v.as_slice()).collect();
                header.centroid = compute_mean_centroid(&refs, self.config.dim, self.config.metric);
                for (doc_id, v) in items {
                    let q_entry = self.quantizer.encode(&v, &header.centroid);
                    tx.put_leaf_entry(target_leaf, doc_id, q_entry);
                }
            } else {
                header.centroid =
                    incremental_mean(&header.centroid, &vec, new_count, self.config.metric);
                let drifted = cluster_distance(self.config.metric, &old_centroid, &header.centroid)
                    > self.config.centroid_drift_eps;
                if new_count <= 16 || drifted {
                    let mut doc_ids = tx.get_leaf_doc_ids(target_leaf)?;
                    doc_ids.push(id);
                    let items = tx.get_vectors_batch(&doc_ids)?;
                    for (doc_id, v) in items {
                        let q_entry = self.quantizer.encode(&v, &header.centroid);
                        tx.put_leaf_entry(target_leaf, doc_id, q_entry);
                    }
                } else {
                    let q_entry = self.quantizer.encode(&vec, &header.centroid);
                    tx.put_leaf_entry(target_leaf, id, q_entry);
                }
            }

            tx.put_header(target_leaf, header);
            if target_leaf != tx.meta().root_id {
                propagate_centroid_upward(&mut tx, target_leaf, &old_centroid, &self.config)?;
            }
            self.meta = tx.commit()?;
            Ok(())
        } else {
            // Add the new quantized entry so split_node sees all doc_ids in target_leaf.
            let q_entry = self.quantizer.encode(&vec, &header.centroid);
            tx.put_leaf_entry(target_leaf, id, q_entry);
            tx.put_header(target_leaf, header);
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
        verify::verify_invariants(&self.storage, &self.meta, &self.config)
    }

    fn remove_doc_internal(&mut self, id: DocId) -> Result<()> {
        let Some(leaf_id) = self.storage.get_doc_loc(id)? else {
            return Ok(());
        };

        let mut tx = TxContext::new(&self.storage, self.meta.clone());
        let mut header = tx.get_header(leaf_id)?;
        let old_centroid = header.centroid.clone();

        tx.delete_leaf_entry(leaf_id, id);
        tx.delete_doc_loc(id);
        tx.delete_vector(id);
        tx.meta_mut().num_vectors = tx.meta().num_vectors.saturating_sub(1);

        let remaining_ids = tx.get_leaf_doc_ids(leaf_id)?;
        let new_count = remaining_ids.len();
        header.entry_count = new_count as u32;
        header.subtree_count = new_count as u64;

        let will_underflow = leaf_id != tx.meta().root_id && new_count < self.config.min_leaf_size;
        if new_count == 0 {
            header.centroid = vec![0.0; self.config.dim];
        } else {
            let items = tx.get_vectors_batch(&remaining_ids)?;
            let refs: Vec<&[f32]> = items.iter().map(|(_, v)| v.as_slice()).collect();
            header.centroid = compute_mean_centroid(&refs, self.config.dim, self.config.metric);
            if !will_underflow
                && (new_count <= 16
                    || cluster_distance(self.config.metric, &old_centroid, &header.centroid)
                        > self.config.centroid_drift_eps)
            {
                for (doc_id, v) in items {
                    let q_entry = self.quantizer.encode(&v, &header.centroid);
                    tx.put_leaf_entry(leaf_id, doc_id, q_entry);
                }
            }
        }
        tx.put_header(leaf_id, header);

        if leaf_id == tx.meta().root_id {
            self.meta = tx.commit()?;
            return Ok(());
        }

        if will_underflow {
            handle_underflow(&mut tx, leaf_id, &self.config, &self.quantizer)?;
        } else {
            propagate_centroid_upward(&mut tx, leaf_id, &old_centroid, &self.config)?;
        }

        self.meta = tx.commit()?;
        Ok(())
    }
}

/// Propagate centroid and `subtree_count` changes upward while centroid drift exceeds `centroid_drift_eps`.
pub(crate) fn propagate_centroid_upward(
    tx: &mut TxContext<'_>,
    start_node_id: NodeId,
    old_start_centroid: &[f32],
    config: &TreeConfig,
) -> Result<()> {
    let mut current_id = start_node_id;
    let mut current_header = tx.get_header(current_id)?;
    let mut prev_centroid = old_start_centroid.to_vec();

    while let Some(parent_id) = current_header.parent_id {
        if cluster_distance(config.metric, &prev_centroid, &current_header.centroid)
            <= config.centroid_drift_eps
        {
            break;
        }

        tx.put_internal_entry(
            parent_id,
            current_id,
            InternalEntry {
                subtree_count: current_header.subtree_count,
                centroid: current_header.centroid.clone(),
            },
        );

        let mut parent_header = tx.get_header(parent_id)?;
        prev_centroid = parent_header.centroid.clone();
        let parent_entries = tx.get_internal_entries(parent_id)?;
        let (total_sub, new_centroid) = recompute_internal_stats(&parent_entries, config);
        parent_header.entry_count = parent_entries.len() as u32;
        parent_header.subtree_count = total_sub;
        parent_header.centroid = new_centroid;
        tx.put_header(parent_id, parent_header.clone());

        current_id = parent_id;
        current_header = parent_header;
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
    let header = tx.get_header(node_id)?;
    let level = header.level;
    let parent_id = header.parent_id;
    let old_centroid = header.centroid.clone();

    let right_id = tx.alloc_node_id();
    let (left_header, right_header) = if level == 0 {
        let doc_ids = tx.get_leaf_doc_ids(node_id)?;
        let items = tx.get_vectors_batch(&doc_ids)?;
        let weighted: Vec<(&[f32], u64)> =
            items.iter().map(|(_, v)| (v.as_slice(), 1u64)).collect();
        let split = balanced_two_means(
            &weighted,
            config.min_leaf_size,
            config.max_leaf_size,
            config.dim,
            config.metric,
        );

        for &idx in &split.part1 {
            let (doc_id, ref vec) = items[idx];
            tx.put_leaf_entry(node_id, doc_id, quantizer.encode(vec, &split.centroid1));
        }
        for &idx in &split.part2 {
            let (doc_id, ref vec) = items[idx];
            tx.delete_leaf_entry(node_id, doc_id);
            tx.put_leaf_entry(right_id, doc_id, quantizer.encode(vec, &split.centroid2));
            tx.put_doc_loc(doc_id, right_id);
        }

        let left_h = NodeHeader {
            level: 0,
            parent_id,
            entry_count: split.part1.len() as u32,
            subtree_count: split.part1.len() as u64,
            centroid: split.centroid1,
        };
        let right_h = NodeHeader {
            level: 0,
            parent_id,
            entry_count: split.part2.len() as u32,
            subtree_count: split.part2.len() as u64,
            centroid: split.centroid2,
        };
        (left_h, right_h)
    } else {
        let entries = tx.get_internal_entries(node_id)?;
        let items: Vec<(NodeId, InternalEntry)> = entries.into_iter().collect();
        let weighted: Vec<(&[f32], u64)> = items
            .iter()
            .map(|(_, e)| (e.centroid.as_slice(), e.subtree_count))
            .collect();
        let split = balanced_two_means(
            &weighted,
            config.min_internal_fanout,
            config.max_internal_fanout,
            config.dim,
            config.metric,
        );

        let left_sub: u64 = split.part1.iter().map(|&i| items[i].1.subtree_count).sum();
        let right_sub: u64 = split.part2.iter().map(|&i| items[i].1.subtree_count).sum();

        for &idx in &split.part2 {
            let (child_id, ref entry) = items[idx];
            tx.delete_internal_entry(node_id, child_id);
            tx.put_internal_entry(right_id, child_id, entry.clone());
            let mut child_h = tx.get_header(child_id)?;
            child_h.parent_id = Some(right_id);
            tx.put_header(child_id, child_h);
        }

        let left_h = NodeHeader {
            level,
            parent_id,
            entry_count: split.part1.len() as u32,
            subtree_count: left_sub,
            centroid: split.centroid1,
        };
        let right_h = NodeHeader {
            level,
            parent_id,
            entry_count: split.part2.len() as u32,
            subtree_count: right_sub,
            centroid: split.centroid2,
        };
        (left_h, right_h)
    };

    attach_split_pair(
        tx,
        node_id,
        left_header,
        right_id,
        right_header,
        config,
        quantizer,
    )?;

    Ok(SplitPair {
        left_id: node_id,
        right_id,
        old_centroid,
    })
}

fn attach_split_pair(
    tx: &mut TxContext<'_>,
    left_id: NodeId,
    mut left_header: NodeHeader,
    right_id: NodeId,
    mut right_header: NodeHeader,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let level = left_header.level;
    match left_header.parent_id {
        None => {
            // Root split: grow a new root level above left_id and right_id.
            let new_root_id = tx.alloc_node_id();
            left_header.parent_id = Some(new_root_id);
            right_header.parent_id = Some(new_root_id);

            let root_sub = left_header.subtree_count + right_header.subtree_count;
            let root_centroid = compute_weighted_centroid(
                &[
                    (&left_header.centroid, left_header.subtree_count),
                    (&right_header.centroid, right_header.subtree_count),
                ],
                config.dim,
                config.metric,
            );
            let root_header = NodeHeader {
                level: level + 1,
                parent_id: None,
                entry_count: 2,
                subtree_count: root_sub,
                centroid: root_centroid,
            };

            tx.put_internal_entry(
                new_root_id,
                left_id,
                InternalEntry {
                    subtree_count: left_header.subtree_count,
                    centroid: left_header.centroid.clone(),
                },
            );
            tx.put_internal_entry(
                new_root_id,
                right_id,
                InternalEntry {
                    subtree_count: right_header.subtree_count,
                    centroid: right_header.centroid.clone(),
                },
            );
            tx.put_header(left_id, left_header);
            tx.put_header(right_id, right_header);
            tx.put_header(new_root_id, root_header);

            tx.meta_mut().root_id = new_root_id;
            tx.meta_mut().root_level = level + 1;
        }
        Some(parent_id) => {
            tx.put_internal_entry(
                parent_id,
                left_id,
                InternalEntry {
                    subtree_count: left_header.subtree_count,
                    centroid: left_header.centroid.clone(),
                },
            );
            tx.put_internal_entry(
                parent_id,
                right_id,
                InternalEntry {
                    subtree_count: right_header.subtree_count,
                    centroid: right_header.centroid.clone(),
                },
            );
            tx.put_header(left_id, left_header);
            tx.put_header(right_id, right_header);

            let mut parent_header = tx.get_header(parent_id)?;
            let old_parent_centroid = parent_header.centroid.clone();
            let parent_entries = tx.get_internal_entries(parent_id)?;
            let (total_sub, new_centroid) = recompute_internal_stats(&parent_entries, config);
            parent_header.entry_count = parent_entries.len() as u32;
            parent_header.subtree_count = total_sub;
            parent_header.centroid = new_centroid;
            tx.put_header(parent_id, parent_header);

            if parent_entries.len() > config.max_internal_fanout {
                split_node(tx, parent_id, config, quantizer)?;
            } else {
                propagate_centroid_upward(tx, parent_id, &old_parent_centroid, config)?;
            }
        }
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

    let header = tx.get_header(node_id)?;
    let level = header.level;
    let min_size = if level == 0 {
        config.min_leaf_size
    } else {
        config.min_internal_fanout
    };
    let max_size = if level == 0 {
        config.max_leaf_size
    } else {
        config.max_internal_fanout
    };

    if (header.entry_count as usize) >= min_size {
        return Ok(());
    }

    let parent_id = header.parent_id.ok_or_else(|| {
        TreeError::InvariantViolation(format!("non-root node {node_id} missing parent_id"))
    })?;

    // Empty non-root node (can happen if min_leaf_size == 1 or after LIRE drains a leaf).
    if header.entry_count == 0 {
        tx.delete_header(node_id);
        tx.delete_internal_entry(parent_id, node_id);
        refresh_parent_after_removal(tx, parent_id, config, quantizer)?;
        return Ok(());
    }

    let parent_entries = tx.get_internal_entries(parent_id)?;
    let sib_id = parent_entries
        .iter()
        .filter(|&(&cid, _)| cid != node_id)
        .min_by(|&(id_a, e_a), &(id_b, e_b)| {
            let da = cluster_distance(config.metric, &header.centroid, &e_a.centroid);
            let db = cluster_distance(config.metric, &header.centroid, &e_b.centroid);
            da.total_cmp(&db).then_with(|| id_a.cmp(id_b))
        })
        .map(|(&cid, _)| cid);

    let Some(sib_id) = sib_id else {
        if parent_id == tx.meta().root_id {
            collapse_root(tx)?;
            return Ok(());
        }
        return Err(TreeError::InvariantViolation(format!(
            "non-root internal node {parent_id} has only 1 child {node_id}"
        )));
    };

    let mut sib_header = tx.get_header(sib_id)?;
    let total = header.entry_count as usize + sib_header.entry_count as usize;

    if total <= max_size {
        // Case 1: Merge `node_id` into `sib_id`.
        if level == 0 {
            let moved_ids = tx.get_leaf_doc_ids(node_id)?;
            let mut sib_ids = tx.get_leaf_doc_ids(sib_id)?;
            for &doc_id in &moved_ids {
                tx.delete_leaf_entry(node_id, doc_id);
                tx.put_doc_loc(doc_id, sib_id);
            }
            sib_ids.extend(moved_ids);

            let sib_vecs = tx.get_vectors_batch(&sib_ids)?;
            let refs: Vec<&[f32]> = sib_vecs.iter().map(|(_, v)| v.as_slice()).collect();
            sib_header.entry_count = sib_vecs.len() as u32;
            sib_header.subtree_count = sib_vecs.len() as u64;
            sib_header.centroid = compute_mean_centroid(&refs, config.dim, config.metric);

            for (doc_id, vec) in sib_vecs {
                tx.put_leaf_entry(sib_id, doc_id, quantizer.encode(&vec, &sib_header.centroid));
            }
        } else {
            let entries = tx.get_internal_entries(node_id)?;
            for (child_id, entry) in entries {
                tx.delete_internal_entry(node_id, child_id);
                tx.put_internal_entry(sib_id, child_id, entry);
                let mut child_h = tx.get_header(child_id)?;
                child_h.parent_id = Some(sib_id);
                tx.put_header(child_id, child_h);
            }
            let sib_entries = tx.get_internal_entries(sib_id)?;
            let (total_sub, new_centroid) = recompute_internal_stats(&sib_entries, config);
            sib_header.entry_count = sib_entries.len() as u32;
            sib_header.subtree_count = total_sub;
            sib_header.centroid = new_centroid;
        }

        tx.put_header(sib_id, sib_header.clone());
        tx.delete_header(node_id);
        tx.delete_internal_entry(parent_id, node_id);
        tx.put_internal_entry(
            parent_id,
            sib_id,
            InternalEntry {
                subtree_count: sib_header.subtree_count,
                centroid: sib_header.centroid,
            },
        );

        refresh_parent_after_removal(tx, parent_id, config, quantizer)?;
    } else {
        // Case 2: Redistribute `node_id` and `sib_id` via balanced 2-means.
        let mut left_header = header;
        if level == 0 {
            let left_ids = tx.get_leaf_doc_ids(node_id)?;
            let right_ids = tx.get_leaf_doc_ids(sib_id)?;
            let left_vecs = tx.get_vectors_batch(&left_ids)?;
            let right_vecs = tx.get_vectors_batch(&right_ids)?;

            let mut combined: Vec<(DocId, Vec<f32>, bool)> =
                Vec::with_capacity(left_vecs.len() + right_vecs.len());
            for (did, v) in left_vecs {
                combined.push((did, v, true));
            }
            for (did, v) in right_vecs {
                combined.push((did, v, false));
            }

            let weighted: Vec<(&[f32], u64)> =
                combined.iter().map(|(_, v, _)| (v.as_slice(), 1u64)).collect();
            let split = balanced_two_means(&weighted, min_size, max_size, config.dim, config.metric);

            for &idx in &split.part1 {
                let (did, ref v, was_left) = combined[idx];
                if !was_left {
                    tx.delete_leaf_entry(sib_id, did);
                    tx.put_doc_loc(did, node_id);
                }
                tx.put_leaf_entry(node_id, did, quantizer.encode(v, &split.centroid1));
            }
            for &idx in &split.part2 {
                let (did, ref v, was_left) = combined[idx];
                if was_left {
                    tx.delete_leaf_entry(node_id, did);
                    tx.put_doc_loc(did, sib_id);
                }
                tx.put_leaf_entry(sib_id, did, quantizer.encode(v, &split.centroid2));
            }

            left_header.entry_count = split.part1.len() as u32;
            left_header.subtree_count = split.part1.len() as u64;
            left_header.centroid = split.centroid1;
            sib_header.entry_count = split.part2.len() as u32;
            sib_header.subtree_count = split.part2.len() as u64;
            sib_header.centroid = split.centroid2;
        } else {
            let left_entries = tx.get_internal_entries(node_id)?;
            let right_entries = tx.get_internal_entries(sib_id)?;
            let mut combined: Vec<(NodeId, InternalEntry, bool)> =
                Vec::with_capacity(left_entries.len() + right_entries.len());
            for (cid, e) in left_entries {
                combined.push((cid, e, true));
            }
            for (cid, e) in right_entries {
                combined.push((cid, e, false));
            }

            let weighted: Vec<(&[f32], u64)> = combined
                .iter()
                .map(|(_, e, _)| (e.centroid.as_slice(), e.subtree_count))
                .collect();
            let split = balanced_two_means(&weighted, min_size, max_size, config.dim, config.metric);

            let mut left_sub = 0u64;
            for &idx in &split.part1 {
                let (cid, ref e, was_left) = combined[idx];
                left_sub += e.subtree_count;
                if !was_left {
                    tx.delete_internal_entry(sib_id, cid);
                    tx.put_internal_entry(node_id, cid, e.clone());
                    let mut ch = tx.get_header(cid)?;
                    ch.parent_id = Some(node_id);
                    tx.put_header(cid, ch);
                }
            }
            let mut right_sub = 0u64;
            for &idx in &split.part2 {
                let (cid, ref e, was_left) = combined[idx];
                right_sub += e.subtree_count;
                if was_left {
                    tx.delete_internal_entry(node_id, cid);
                    tx.put_internal_entry(sib_id, cid, e.clone());
                    let mut ch = tx.get_header(cid)?;
                    ch.parent_id = Some(sib_id);
                    tx.put_header(cid, ch);
                }
            }

            left_header.entry_count = split.part1.len() as u32;
            left_header.subtree_count = left_sub;
            left_header.centroid = split.centroid1;
            sib_header.entry_count = split.part2.len() as u32;
            sib_header.subtree_count = right_sub;
            sib_header.centroid = split.centroid2;
        }

        tx.put_internal_entry(
            parent_id,
            node_id,
            InternalEntry {
                subtree_count: left_header.subtree_count,
                centroid: left_header.centroid.clone(),
            },
        );
        tx.put_internal_entry(
            parent_id,
            sib_id,
            InternalEntry {
                subtree_count: sib_header.subtree_count,
                centroid: sib_header.centroid.clone(),
            },
        );
        tx.put_header(node_id, left_header);
        tx.put_header(sib_id, sib_header);

        let mut parent_header = tx.get_header(parent_id)?;
        let old_parent_centroid = parent_header.centroid.clone();
        let updated_parent_entries = tx.get_internal_entries(parent_id)?;
        let (total_sub, new_centroid) = recompute_internal_stats(&updated_parent_entries, config);
        parent_header.entry_count = updated_parent_entries.len() as u32;
        parent_header.subtree_count = total_sub;
        parent_header.centroid = new_centroid;
        tx.put_header(parent_id, parent_header);

        propagate_centroid_upward(tx, parent_id, &old_parent_centroid, config)?;
    }

    Ok(())
}

fn refresh_parent_after_removal(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let mut parent_header = tx.get_header(parent_id)?;
    let old_parent_centroid = parent_header.centroid.clone();
    let parent_entries = tx.get_internal_entries(parent_id)?;
    let (total_sub, new_centroid) = recompute_internal_stats(&parent_entries, config);
    parent_header.entry_count = parent_entries.len() as u32;
    parent_header.subtree_count = total_sub;
    parent_header.centroid = new_centroid;
    tx.put_header(parent_id, parent_header);

    if parent_id == tx.meta().root_id {
        if parent_entries.len() == 1 {
            collapse_root(tx)?;
        }
    } else if parent_entries.len() < config.min_internal_fanout {
        handle_underflow(tx, parent_id, config, quantizer)?;
    } else {
        propagate_centroid_upward(tx, parent_id, &old_parent_centroid, config)?;
    }
    Ok(())
}

fn collapse_root(tx: &mut TxContext<'_>) -> Result<()> {
    loop {
        let root_id = tx.meta().root_id;
        let root_header = tx.get_header(root_id)?;
        if root_header.level == 0 || root_header.entry_count != 1 {
            break;
        }
        let entries = tx.get_internal_entries(root_id)?;
        if entries.len() != 1 {
            break;
        }
        let (&only_child_id, _) = entries.iter().next().unwrap();
        let mut child_header = tx.get_header(only_child_id)?;
        child_header.parent_id = None;
        tx.put_header(only_child_id, child_header.clone());
        tx.delete_internal_entry(root_id, only_child_id);
        tx.delete_header(root_id);
        tx.meta_mut().root_id = only_child_id;
        tx.meta_mut().root_level = child_header.level;
    }
    Ok(())
}

pub(crate) fn recompute_internal_stats(
    entries: &std::collections::BTreeMap<NodeId, InternalEntry>,
    config: &TreeConfig,
) -> (u64, Vec<f32>) {
    let total_sub: u64 = entries.values().map(|e| e.subtree_count).sum();
    let items: Vec<(&[f32], u64)> = entries
        .values()
        .map(|e| (e.centroid.as_slice(), e.subtree_count))
        .collect();
    let centroid = compute_weighted_centroid(&items, config.dim, config.metric);
    (total_sub, centroid)
}
