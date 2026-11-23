//! Underflow handling: merge, redistribute, and root collapse.



use izzydb_core::config::{NodeId, RowId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::kmeans::balanced_two_means;
use izzydb_core::metric::{
    cluster_distance, compute_mean_centroid_flat,
};
use izzydb_rabitq::{EncodeScratch, RabitqQuantizer};
use izzydb_storage::node::{PackedInternalNode, PackedLeafNode};
use izzydb_storage::TxContext;



use super::*;

/// Resolve an underflowing non-root node via merge, redistribution, or empty-node deletion.
pub(crate) fn handle_underflow(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let root_id = tx
        .ann_meta()
        .ok_or_else(|| TreeError::InvalidConfig("collection has no vector index".into()))?
        .root_id;
    if node_id == root_id {
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
        return delete_empty_node(tx, node_id, parent_id, is_internal, config, quantizer);
    }

    let Some(sib_id) = find_closest_sibling(tx, parent_id, node_id, &node_centroid, config)? else {
        if parent_id == root_id {
            collapse_root(tx)?;
            return Ok(());
        }
        return Err(TreeError::InvariantViolation(format!(
            "non-root internal node {parent_id} has only 1 child {node_id}"
        )));
    };

    if !is_internal {
        rebalance_underflowing_leaf(tx, node_id, sib_id, parent_id, config, quantizer)
    } else {
        rebalance_underflowing_internal(tx, node_id, sib_id, parent_id, config, quantizer)
    }
}

pub(super) fn delete_empty_node(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    parent_id: NodeId,
    is_internal: bool,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    if is_internal {
        tx.delete_internal_node(node_id);
    } else {
        tx.delete_leaf_node(node_id);
    }
    let mut parent = tx.get_internal_node(parent_id)?;
    parent.remove_child(node_id, config.dim);
    tx.put_internal_node(parent_id, parent);
    refresh_parent_after_removal(tx, parent_id, config, quantizer)
}

pub(super) fn find_closest_sibling(
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

pub(super) fn rebalance_underflowing_leaf(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    sib_id: NodeId,
    parent_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let dim = config.dim;
    let words = quantizer.num_words();
    let mut left_leaf = tx.get_leaf_node(node_id)?;
    let mut sib_leaf = tx.get_leaf_node(sib_id)?;
    let _ = prune_and_remap_leaf(tx, &mut left_leaf, words)?;
    let _ = prune_and_remap_leaf(tx, &mut sib_leaf, words)?;

    let left_ids = left_leaf.live_row_ids();
    let sib_ids = sib_leaf.live_row_ids();
    let total = left_ids.len() + sib_ids.len();

    if total == 0 {
        tx.put_leaf_node(sib_id, sib_leaf);
        delete_empty_node(tx, node_id, parent_id, false, config, quantizer)?;
        let cur_root = tx.ann_meta().unwrap().root_id;
        if sib_id != cur_root && tx.try_get_leaf_node(sib_id)?.is_some() {
            handle_underflow(tx, sib_id, config, quantizer)?;
        }
        return Ok(());
    }

    let left_vecs = tx.get_vectors_flat(&left_ids)?;
    let sib_vecs = tx.get_vectors_flat(&sib_ids)?;

    let mut combined: Vec<(RowId, &[f32])> = Vec::with_capacity(total);
    for (i, &rid) in left_ids.iter().enumerate() {
        combined.push((rid, &left_vecs[i * dim..(i + 1) * dim]));
    }
    for (i, &rid) in sib_ids.iter().enumerate() {
        combined.push((rid, &sib_vecs[i * dim..(i + 1) * dim]));
    }
    combined.sort_unstable_by_key(|&(rid, _)| rid);
    combined.dedup_by_key(|(rid, _)| *rid);
    let total = combined.len();

    let mut scratch = EncodeScratch::default();
    if total <= config.max_leaf_size {
        let mut merged_ids = Vec::with_capacity(total);
        let mut merged_vecs = Vec::with_capacity(total * dim);
        for (rid, v) in combined {
            merged_ids.push(rid);
            merged_vecs.extend_from_slice(v);
        }

        sib_leaf.centroid = compute_mean_centroid_flat(&merged_vecs, total, dim, config.metric);
        sib_leaf.reset_and_encode(&merged_ids, &merged_vecs, quantizer, dim, &mut scratch);

        let sib_sub = sib_leaf.entry_count() as u64;
        let sib_c = sib_leaf.centroid.clone();
        tx.put_leaf_node(sib_id, sib_leaf);
        tx.delete_leaf_node(node_id);

        let mut parent = tx.get_internal_node(parent_id)?;
        parent.remove_child(node_id, dim);
        parent.upsert_child(sib_id, sib_sub, &sib_c, dim);
        tx.put_internal_node(parent_id, parent);

        refresh_parent_after_removal(tx, parent_id, config, quantizer)?;

        let cur_root = tx.ann_meta().unwrap().root_id;
        if sib_id != cur_root
            && total < config.min_leaf_size
            && tx.try_get_leaf_node(sib_id)?.is_some()
        {
            handle_underflow(tx, sib_id, config, quantizer)?;
        }
        Ok(())
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

        let mut reencode_half = |leaf: &mut PackedLeafNode, indices: &[usize], centroid: Vec<f32>| {
            let mut ids = Vec::with_capacity(indices.len());
            let mut vecs = Vec::with_capacity(indices.len() * dim);
            for &idx in indices {
                let (rid, v) = combined[idx];
                ids.push(rid);
                vecs.extend_from_slice(v);
            }
            leaf.centroid = centroid;
            leaf.reset_and_encode(&ids, &vecs, quantizer, dim, &mut scratch);
        };
        reencode_half(&mut left_leaf, &split.part1, split.centroid1);
        reencode_half(&mut sib_leaf, &split.part2, split.centroid2);

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

pub(super) fn rebalance_underflowing_internal(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    sib_id: NodeId,
    parent_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
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

        refresh_parent_after_removal(tx, parent_id, config, quantizer)
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

        let mut refill = |target_id: NodeId,
                          node: &mut PackedInternalNode,
                          indices: &[usize],
                          centroid: Vec<f32>,
                          expect_left: bool|
         -> Result<()> {
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

pub(super) fn update_parent_pair(
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

pub(super) fn refresh_parent_after_removal(
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

    let root_id = tx.ann_meta().unwrap().root_id;
    if parent_id == root_id {
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

pub(super) fn collapse_root(tx: &mut TxContext<'_>) -> Result<()> {
    while tx.ann_meta().is_some_and(|m| m.root_level > 0) {
        let root_id = tx.ann_meta().unwrap().root_id;
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
        let meta = tx.ann_meta_mut()?;
        meta.root_id = only_child_id;
        meta.root_level = child_level;
    }
    Ok(())
}
