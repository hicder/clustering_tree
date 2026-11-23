//! Node splits, root creation, and centroid propagation.



use izzydb_core::config::{NodeId, RowId, TreeConfig};
use izzydb_core::error::Result;
use izzydb_core::kmeans::balanced_two_means;
use izzydb_core::metric::{
    cluster_distance, compute_weighted_centroid,
};
use izzydb_rabitq::{EncodeScratch, RabitqQuantizer};
use izzydb_storage::node::{PackedInternalNode, PackedLeafNode};
use izzydb_storage::TxContext;



use super::*;

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
    let right_id = tx.alloc_node_id()?;
    if !tx.is_internal_node(node_id) {
        split_leaf(tx, node_id, right_id, config, quantizer)
    } else {
        split_internal(tx, node_id, right_id, config, quantizer)
    }
}

pub(super) fn split_leaf(
    tx: &mut TxContext<'_>,
    left_id: NodeId,
    right_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<SplitPair> {
    let mut leaf = tx.get_leaf_node(left_id)?;
    let _ = prune_and_remap_leaf(tx, &mut leaf, quantizer.num_words())?;
    let live_row_ids = leaf.live_row_ids();

    let flat_vecs = tx.get_vectors_flat(&live_row_ids)?;
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

    let build_half = |indices: &[usize]| -> (Vec<RowId>, Vec<f32>) {
        let mut ids = Vec::with_capacity(indices.len());
        let mut vecs = Vec::with_capacity(indices.len() * dim);
        for &idx in indices {
            ids.push(live_row_ids[idx]);
            vecs.extend_from_slice(&flat_vecs[idx * dim..(idx + 1) * dim]);
        }
        (ids, vecs)
    };
    let (left_rows, left_vecs) = build_half(&split.part1);
    let (right_rows, right_vecs) = build_half(&split.part2);

    let mut scratch = EncodeScratch::default();
    let mut encode_leaf = |centroid: Vec<f32>, ids: &[RowId], vecs: &[f32]| {
        let mut half = PackedLeafNode::empty(parent_id, dim, padded_dim);
        half.centroid = centroid;
        half.reset_and_encode(ids, vecs, quantizer, dim, &mut scratch);
        half
    };
    let mut left_leaf = encode_leaf(split.centroid1, &left_rows, &left_vecs);
    let mut right_leaf = encode_leaf(split.centroid2, &right_rows, &right_vecs);

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
            let new_root_id = tx.alloc_node_id()?;
            left_leaf.parent_id = Some(new_root_id);
            right_leaf.parent_id = Some(new_root_id);
            tx.put_leaf_node(left_id, left_leaf);
            tx.put_leaf_node(right_id, right_leaf);
            create_new_root(tx, new_root_id, 1, left_summary, right_summary, config)?;
        }
        Some(pid) => {
            tx.put_leaf_node(left_id, left_leaf);
            tx.put_leaf_node(right_id, right_leaf);
            attach_to_parent(tx, pid, left_summary, right_summary, config, quantizer)?;
        }
    }

    Ok(SplitPair {
        left_id,
        right_id,
        old_centroid,
    })
}

pub(super) fn split_internal(
    tx: &mut TxContext<'_>,
    left_id: NodeId,
    right_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
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
            let new_root_id = tx.alloc_node_id()?;
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
            )?;
        }
        Some(pid) => {
            tx.put_internal_node(left_id, left_node);
            tx.put_internal_node(right_id, right_node);
            attach_to_parent(tx, pid, left_summary, right_summary, config, quantizer)?;
        }
    }

    Ok(SplitPair {
        left_id,
        right_id,
        old_centroid,
    })
}

pub(super) fn create_new_root(
    tx: &mut TxContext<'_>,
    new_root_id: NodeId,
    new_root_level: u8,
    left: ChildSummary<'_>,
    right: ChildSummary<'_>,
    config: &TreeConfig,
) -> Result<()> {
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
    let meta = tx.ann_meta_mut()?;
    meta.root_id = new_root_id;
    meta.root_level = new_root_level;
    Ok(())
}

pub(super) fn attach_to_parent(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    left: ChildSummary<'_>,
    right: ChildSummary<'_>,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
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

pub(super) fn set_child_parent(
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
