//! Vacuum budget, maintenance options, and resumable ANN vacuum.


use std::collections::HashMap;

use izzydb_core::config::NodeId;
use izzydb_core::error::Result;
use izzydb_core::metric::compute_mean_centroid_flat;
use izzydb_rabitq::EncodeScratch;
use izzydb_storage::codec::Manifest;
use izzydb_storage::node::{AnnMeta, LeafNodeView, PackedInternalNode};
use izzydb_storage::{LeafVisitAction, Storage, TxContext};



use super::*;

/// Resource budget for [`Collection::vacuum_ann`] / [`ClusteringTree::vacuum_ann`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VacuumBudget {
    /// Maximum number of leaf blobs to scan (`None` = scan until full pass finishes).
    pub max_leaves: Option<usize>,
    /// Maximum number of encoded leaf bytes to scan (`None` = unlimited).
    pub max_bytes: Option<usize>,
}

impl VacuumBudget {
    pub fn unlimited() -> Self {
        Self::default()
    }

    pub fn leaves(max_leaves: usize) -> Self {
        Self {
            max_leaves: Some(max_leaves),
            max_bytes: None,
        }
    }

    pub fn bytes(max_bytes: usize) -> Self {
        Self {
            max_leaves: None,
            max_bytes: Some(max_bytes),
        }
    }
}

/// Statistics returned by `vacuum_ann`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VacuumStats {
    pub leaves_scanned: usize,
    pub leaves_rewritten: usize,
    pub leaves_merged_or_deleted: usize,
    pub stale_slots_dropped: u64,
    pub rows_remapped: u64,
    pub completed_pass: bool,
}

/// Policy knobs for [`Collection::maintenance`].
#[derive(Debug, Clone, Copy)]
pub struct MaintenanceOptions {
    /// Trigger `vacuum_ann` when `stale_rows / (num_vectors + stale_rows) > vacuum_stale_ratio`.
    pub vacuum_stale_ratio: f64,
    pub vacuum_budget: VacuumBudget,
    pub compaction: crate::compaction::CompactionOptions,
}

impl Default for MaintenanceOptions {
    fn default() -> Self {
        Self {
            vacuum_stale_ratio: 0.2,
            vacuum_budget: VacuumBudget::unlimited(),
            compaction: crate::compaction::CompactionOptions::default(),
        }
    }
}

/// Outcome of a [`Collection::maintenance`] pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaintenanceStats {
    pub vacuum: Option<VacuumStats>,
    pub columns_built: usize,
    pub detached_rebuilt: usize,
    pub compaction: crate::compaction::CompactionStats,
}

/// Incrementally vacuum stale slots and handle underflowing leaves across bounded transactions.
pub(crate) fn vacuum_ann_storage(
    storage: &Storage,
    ann: &AnnIndex,
    manifest: &mut Manifest,
    ann_meta: &mut Option<AnnMeta>,
    internal_cache: &mut HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &mut HashMap<NodeId, NodeId>,
    budget: VacuumBudget,
) -> Result<VacuumStats> {
    let Some(ref mut meta_ref) = *ann_meta else {
        return Ok(VacuumStats::default());
    };

    let config = &ann.config;
    let quantizer = &ann.quantizer;
    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();
    let words = quantizer.num_words();

    let mut stats = VacuumStats::default();
    let start_cursor = meta_ref.vacuum_cursor;
    let mut cursor = start_cursor;
    let mut wrapped = false;
    let mut bytes_scanned = 0usize;

    loop {
        if budget
            .max_leaves
            .is_some_and(|max_l| stats.leaves_scanned >= max_l)
            || budget.max_bytes.is_some_and(|max_b| bytes_scanned >= max_b)
        {
            break;
        }

        let seg_snap = storage.segment_snapshot();
        let dropped = storage.dropped_segments_snapshot();
        let has_any_stale_segs =
            !seg_snap.has_deletes_or_forwarding.is_empty() || !dropped.is_empty();

        let mut dirty_batch: Vec<NodeId> = Vec::new();
        let mut hit_wrap_boundary = false;
        let root_id = meta_ref.root_id;
        let is_unlimited = budget.max_leaves.is_none() && budget.max_bytes.is_none();

        let next_cursor_opt = storage.visit_leaf_nodes_from(cursor, |leaf_id, bytes| {
            if wrapped && leaf_id >= start_cursor {
                hit_wrap_boundary = true;
                return Ok(LeafVisitAction::StopBefore);
            }
            if budget
                .max_leaves
                .is_some_and(|max_l| stats.leaves_scanned >= max_l)
                || budget
                    .max_bytes
                    .is_some_and(|max_b| bytes_scanned > 0 && bytes_scanned >= max_b)
            {
                return Ok(LeafVisitAction::StopBefore);
            }
            stats.leaves_scanned += 1;
            bytes_scanned += bytes.len();

            let view = LeafNodeView::parse(bytes, dim, padded_dim, words)?;
            let needs_work = view.stale_hint > 0
                || needs_leaf_compaction_view(&view, config)
                || (leaf_id != root_id && view.live_count < config.min_leaf_size)
                || (has_any_stale_segs && leaf_view_has_stale_or_forwarded(&view, &seg_snap, &dropped));
            let hit_budget_now = budget
                .max_leaves
                .is_some_and(|max_l| stats.leaves_scanned >= max_l)
                || budget.max_bytes.is_some_and(|max_b| bytes_scanned >= max_b);
            if needs_work {
                dirty_batch.push(leaf_id);
                if dirty_batch.len() >= VACUUM_TX_BATCH || hit_budget_now {
                    return Ok(LeafVisitAction::StopAfter);
                }
            } else if hit_budget_now {
                return Ok(LeafVisitAction::StopAfter);
            }
            Ok(LeafVisitAction::Continue)
        })?;

        let (next_cursor, pass_finished_here) = if hit_wrap_boundary {
            (0, true)
        } else {
            match next_cursor_opt {
                Some(next_id) => (next_id, false),
                None => {
                    if is_unlimited && start_cursor > 0 && !wrapped {
                        wrapped = true;
                        (0, false)
                    } else {
                        (0, true)
                    }
                }
            }
        };

        if !dirty_batch.is_empty() || next_cursor != meta_ref.vacuum_cursor || pass_finished_here {
            let mut tx = TxContext::new(
                storage,
                internal_cache,
                leaf_parent,
                manifest.clone(),
                Some(meta_ref.clone()),
            );
            let mut scratch = EncodeScratch::default();
            for leaf_id in dirty_batch {
                let Some(mut leaf) = tx.try_get_leaf_node(leaf_id)? else {
                    continue;
                };
                let (dropped_slots, remapped_slots) = prune_and_remap_leaf(&mut tx, &mut leaf, words)?;
                stats.stale_slots_dropped += dropped_slots;
                stats.rows_remapped += remapped_slots;

                let cur_root_id = tx.ann_meta().unwrap().root_id;
                let count = leaf.entry_count();
                if count == 0 {
                    if leaf_id == cur_root_id {
                        leaf.centroid = vec![0.0; dim];
                        leaf.rotated_centroid.fill(0.0);
                        leaf.reset_and_encode(&[], &[], quantizer, dim, &mut scratch);
                        tx.put_leaf_node(leaf_id, leaf);
                        stats.leaves_rewritten += 1;
                    } else {
                        tx.put_leaf_node(leaf_id, leaf);
                        handle_underflow(&mut tx, leaf_id, config, quantizer)?;
                        stats.leaves_merged_or_deleted += 1;
                    }
                } else if leaf_id != cur_root_id && count < config.min_leaf_size {
                    tx.put_leaf_node(leaf_id, leaf);
                    handle_underflow(&mut tx, leaf_id, config, quantizer)?;
                    stats.leaves_merged_or_deleted += 1;
                } else if dropped_slots > 0 || remapped_slots > 0 || needs_leaf_compaction(&leaf, config) {
                    let old_c = leaf.centroid.clone();
                    if dropped_slots > 0 || needs_leaf_compaction(&leaf, config) {
                        let live_rows = leaf.live_row_ids();
                        let flat_vecs = tx.get_vectors_flat(&live_rows)?;
                        leaf.centroid = compute_mean_centroid_flat(
                            &flat_vecs,
                            count,
                            dim,
                            config.metric,
                        );
                        leaf.reset_and_encode(&live_rows, &flat_vecs, quantizer, dim, &mut scratch);
                    }
                    let parent_id = leaf.parent_id;
                    let new_c = leaf.centroid.clone();
                    tx.put_leaf_node(leaf_id, leaf);
                    if let Some(pid) = parent_id {
                        propagate_centroid_upward(
                            &mut tx,
                            leaf_id,
                            pid,
                            count as u64,
                            &old_c,
                            &new_c,
                            config,
                        )?;
                    }
                    stats.leaves_rewritten += 1;
                }
            }

            {
                let meta = tx.ann_meta_mut()?;
                meta.vacuum_cursor = next_cursor;
                if pass_finished_here {
                    meta.vacuum_epoch += 1;
                }
            }
            let (new_manifest, new_ann_meta) = tx.commit()?;
            *manifest = new_manifest;
            if let Some(m) = new_ann_meta {
                *meta_ref = m;
            }
        }

        cursor = next_cursor;
        if pass_finished_here {
            stats.completed_pass = true;
            break;
        }
    }

    Ok(stats)
}
