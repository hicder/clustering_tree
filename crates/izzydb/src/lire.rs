//! SPFresh LIRE boundary-vector reassignment after leaf splits (pure `RowId` leaf operations).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use izzydb_core::config::{DistanceMetric, NodeId, RowId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::metric::{cluster_distance, compute_mean_centroid_flat};
use izzydb_rabitq::{EncodeScratch, RabitqQuantizer};
use izzydb_storage::node::PackedLeafNode;
use izzydb_storage::TxContext;

use crate::search;
use crate::tree::{
    SplitPair, handle_underflow, propagate_centroid_upward, prune_and_remap_leaf, split_node,
};

const MAX_LIRE_ROUNDS: usize = 4;

struct NeighborLeaves {
    centroids: HashMap<NodeId, Vec<f32>>,
    row_ids: HashMap<NodeId, Vec<RowId>>,
}

/// Run SPFresh LIRE boundary reassignment inside `tx` after a leaf split.
/// Uses `tx` read-your-own-writes overlays and filters out stale/tombstoned `RowId`s.
pub(crate) fn run_lire_in_tx(
    tx: &mut TxContext<'_>,
    initial_split: SplitPair,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    if config.lire_neighbor_count == 0 {
        return Ok(());
    }

    let words = quantizer.num_words();
    let mut lire_rounds = 0usize;
    let mut pending_lire: VecDeque<SplitPair> = VecDeque::from([initial_split]);
    let mut maintenance: VecDeque<NodeId> = VecDeque::new();

    while !pending_lire.is_empty() || !maintenance.is_empty() {
        while let Some(pair) = pending_lire.pop_front() {
            if lire_rounds >= MAX_LIRE_ROUNDS {
                pending_lire.clear();
                break;
            }
            lire_rounds += 1;
            reassign_split_pair_in_tx(tx, pair, config, quantizer, &mut maintenance)?;
        }

        while let Some(leaf_id) = maintenance.pop_front() {
            let Some(mut leaf) = tx.try_get_leaf_node(leaf_id)? else {
                continue;
            };
            let (dropped, remapped) = prune_and_remap_leaf(tx, &mut leaf, words)?;
            let count = leaf.entry_count();
            let root_id = tx.ann_meta().map(|m| m.root_id).unwrap_or(0);
            let root_level = tx.ann_meta().map(|m| m.root_level).unwrap_or(0);

            if count > config.max_leaf_size {
                if dropped > 0 || remapped > 0 {
                    tx.put_leaf_node(leaf_id, leaf);
                }
                let split_pair = split_node(tx, leaf_id, config, quantizer)?;
                for child_id in [split_pair.left_id, split_pair.right_id] {
                    if let Some(ch) = tx.try_get_leaf_node(child_id)?
                        && ch.entry_count() > config.max_leaf_size
                    {
                        maintenance.push_back(child_id);
                    }
                }
                if lire_rounds < MAX_LIRE_ROUNDS && root_level > 0 {
                    pending_lire.push_back(split_pair);
                    break;
                }
            } else if leaf_id != root_id && count < config.min_leaf_size {
                if dropped > 0 || remapped > 0 {
                    tx.put_leaf_node(leaf_id, leaf);
                }
                handle_underflow(tx, leaf_id, config, quantizer)?;
            } else if dropped > 0 || remapped > 0 {
                let old_c = leaf.centroid.clone();
                if count == 0 {
                    leaf.centroid = vec![0.0; config.dim];
                    leaf.rotated_centroid.fill(0.0);
                    let mut scratch = EncodeScratch::default();
                    leaf.reset_and_encode(&[], &[], quantizer, config.dim, &mut scratch);
                } else {
                    let live_rows = leaf.live_row_ids();
                    let flat_vecs = tx.get_vectors_flat(&live_rows)?;
                    leaf.centroid = compute_mean_centroid_flat(
                        &flat_vecs,
                        count,
                        config.dim,
                        config.metric,
                    );
                    let mut scratch = EncodeScratch::default();
                    leaf.reset_and_encode(&live_rows, &flat_vecs, quantizer, config.dim, &mut scratch);
                }
                let parent_id = leaf.parent_id;
                let new_c = leaf.centroid.clone();
                tx.put_leaf_node(leaf_id, leaf);
                if count > 0
                    && let Some(pid) = parent_id
                {
                    propagate_centroid_upward(tx, leaf_id, pid, count as u64, &old_c, &new_c, config)?;
                }
            }
        }
    }

    Ok(())
}

fn reassign_split_pair_in_tx(
    tx: &mut TxContext<'_>,
    pair: SplitPair,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    maintenance: &mut VecDeque<NodeId>,
) -> Result<()> {
    let Some(u_leaf) = tx.try_get_leaf_node(pair.left_id)? else {
        return Ok(());
    };
    let Some(v_leaf) = tx.try_get_leaf_node(pair.right_id)? else {
        return Ok(());
    };

    let Some(neighbors) = load_neighbor_leaves_in_tx(tx, &pair, config)? else {
        return Ok(());
    };

    let plan = collect_npa_moves_in_tx(tx, &pair, &u_leaf, &v_leaf, &neighbors, config)?;
    if plan.moves.is_empty() {
        return Ok(());
    }

    apply_leaf_moves_in_tx(tx, &plan.moves, &plan.vecs_by_id, config, quantizer, maintenance)
}

fn resolve_leaf_live_rows(tx: &TxContext<'_>, leaf: &PackedLeafNode) -> Vec<RowId> {
    let mut out = Vec::with_capacity(leaf.live_count);
    for raw in leaf.live_row_ids() {
        if let Some(resolved) = tx.resolve_row(raw) {
            out.push(resolved);
        }
    }
    out
}

fn load_neighbor_leaves_in_tx(
    tx: &mut TxContext<'_>,
    pair: &SplitPair,
    config: &TreeConfig,
) -> Result<Option<NeighborLeaves>> {
    let meta = tx
        .ann_meta()
        .ok_or_else(|| TreeError::InvalidConfig("collection has no vector index".into()))?;
    let m = config.lire_neighbor_count;
    let beam = (m + 2).max(config.insert_beam_width);
    let candidate_leaves = search::search_leaves_with(
        |id| tx.get_internal_node_ref(id),
        meta,
        &pair.old_centroid,
        m + 2,
        beam,
        config,
    )?;

    let mut neighbor_ids: Vec<NodeId> = candidate_leaves
        .into_iter()
        .map(|(id, _)| id)
        .filter(|&id| id != pair.left_id && id != pair.right_id)
        .take(m)
        .collect();
    if neighbor_ids.is_empty() {
        return Ok(None);
    }
    neighbor_ids.sort_unstable();

    let mut centroids: HashMap<NodeId, Vec<f32>> = HashMap::new();
    let mut row_ids: HashMap<NodeId, Vec<RowId>> = HashMap::new();
    for nid in neighbor_ids {
        if let Some(leaf) = tx.try_get_leaf_node(nid)? {
            centroids.insert(nid, leaf.centroid.clone());
            row_ids.insert(nid, resolve_leaf_live_rows(tx, &leaf));
        }
    }
    if centroids.is_empty() {
        return Ok(None);
    }
    Ok(Some(NeighborLeaves { centroids, row_ids }))
}

struct NpaMovePlan {
    moves: Vec<(RowId, NodeId, NodeId)>,
    vecs_by_id: HashMap<RowId, Vec<f32>>,
}

fn collect_npa_moves_in_tx(
    tx: &TxContext<'_>,
    pair: &SplitPair,
    u_leaf: &PackedLeafNode,
    v_leaf: &PackedLeafNode,
    neighbors: &NeighborLeaves,
    config: &TreeConfig,
) -> Result<NpaMovePlan> {
    let metric = config.metric;
    let dim = config.dim;
    let c_u = &u_leaf.centroid;
    let c_v = &v_leaf.centroid;

    let mut npa_neighbors: Vec<NodeId> = neighbors
        .centroids
        .iter()
        .filter(|&(_, c_n)| {
            let d_old = cluster_distance(metric, c_n, &pair.old_centroid);
            let d_new_min =
                cluster_distance(metric, c_n, c_u).min(cluster_distance(metric, c_n, c_v));
            d_new_min < d_old
        })
        .map(|(&nid, _)| nid)
        .collect();
    npa_neighbors.sort_unstable();

    let u_rows = resolve_leaf_live_rows(tx, u_leaf);
    let v_rows = resolve_leaf_live_rows(tx, v_leaf);
    let mut required_rows: Vec<RowId> = Vec::with_capacity(u_rows.len() + v_rows.len());
    required_rows.extend_from_slice(&u_rows);
    required_rows.extend_from_slice(&v_rows);
    for &nid in &npa_neighbors {
        if let Some(ids) = neighbors.row_ids.get(&nid) {
            required_rows.extend_from_slice(ids);
        }
    }
    required_rows.sort_unstable();
    required_rows.dedup();

    let flat = tx.get_vectors_flat(&required_rows)?;
    let mut vecs_by_id: HashMap<RowId, Vec<f32>> = HashMap::with_capacity(required_rows.len());
    for (i, &row_id) in required_rows.iter().enumerate() {
        vecs_by_id.insert(row_id, flat[i * dim..(i + 1) * dim].to_vec());
    }

    let mut moves: Vec<(RowId, NodeId, NodeId)> = Vec::new();

    // 1. Outward check: vectors in U and V that are closer to a neighbor leaf's centroid.
    let mut uv_leaves = [(pair.left_id, u_leaf, u_rows), (pair.right_id, v_leaf, v_rows)];
    uv_leaves.sort_unstable_by_key(|&(id, _, _)| id);
    for (src_id, src_leaf, src_rows) in uv_leaves {
        let src_centroid = &src_leaf.centroid;
        for row_id in src_rows {
            let vec = &vecs_by_id[&row_id];
            let d_self = cluster_distance(metric, vec, src_centroid);
            if let Some(to_id) = closest_closer_neighbor(vec, d_self, &neighbors.centroids, metric) {
                moves.push((row_id, src_id, to_id));
            }
        }
    }

    // 2. Inward check: neighbor leaves filtered by NPA condition.
    for &nid in &npa_neighbors {
        let c_n = &neighbors.centroids[&nid];
        let Some(row_ids) = neighbors.row_ids.get(&nid) else {
            continue;
        };
        for &row_id in row_ids {
            let vec = &vecs_by_id[&row_id];
            let d_n = cluster_distance(metric, vec, c_n);
            let d_u = cluster_distance(metric, vec, c_u);
            let d_v = cluster_distance(metric, vec, c_v);
            if d_u < d_n || d_v < d_n {
                let to_id = if d_u <= d_v {
                    pair.left_id
                } else {
                    pair.right_id
                };
                moves.push((row_id, nid, to_id));
            }
        }
    }

    Ok(NpaMovePlan { moves, vecs_by_id })
}

#[inline]
fn closest_closer_neighbor(
    vec: &[f32],
    d_self: f32,
    centroids: &HashMap<NodeId, Vec<f32>>,
    metric: DistanceMetric,
) -> Option<NodeId> {
    let mut best: Option<(NodeId, f32)> = None;
    for (&nid, c_n) in centroids {
        let d_n = cluster_distance(metric, vec, c_n);
        if d_n < d_self
            && best.is_none_or(|(best_id, best_d)| d_n < best_d || (d_n == best_d && nid < best_id))
        {
            best = Some((nid, d_n));
        }
    }
    best.map(|(id, _)| id)
}

fn apply_leaf_moves_in_tx(
    tx: &mut TxContext<'_>,
    moves: &[(RowId, NodeId, NodeId)],
    vecs_by_id: &HashMap<RowId, Vec<f32>>,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
    maintenance: &mut VecDeque<NodeId>,
) -> Result<()> {
    let dim = config.dim;
    let metric = config.metric;
    let words = quantizer.num_words();

    let mut touched_leaves = BTreeSet::new();
    for &(_, from_leaf_id, to_leaf_id) in moves {
        touched_leaves.insert(from_leaf_id);
        touched_leaves.insert(to_leaf_id);
    }

    // Prune stale/forwarded rows in each touched leaf first so moves operate over live RowIds.
    let mut leaf_objs: BTreeMap<NodeId, PackedLeafNode> = BTreeMap::new();
    let mut leaf_row_ids: BTreeMap<NodeId, Vec<RowId>> = BTreeMap::new();
    for &leaf_id in &touched_leaves {
        let mut leaf = tx.get_leaf_node(leaf_id)?;
        let _ = prune_and_remap_leaf(tx, &mut leaf, words)?;
        let mut rows = leaf.live_row_ids();
        rows.sort_unstable();
        rows.dedup();
        leaf_objs.insert(leaf_id, leaf);
        leaf_row_ids.insert(leaf_id, rows);
    }

    for &(row_id, from_leaf_id, to_leaf_id) in moves {
        if let Some(from_rows) = leaf_row_ids.get_mut(&from_leaf_id)
            && let Ok(pos) = from_rows.binary_search(&row_id)
        {
            from_rows.remove(pos);
        }
        let to_rows = leaf_row_ids.get_mut(&to_leaf_id).unwrap();
        if let Err(pos) = to_rows.binary_search(&row_id) {
            to_rows.insert(pos, row_id);
        }
    }

    let root_id = tx.ann_meta().map(|m| m.root_id).unwrap_or(0);
    let mut scratch = EncodeScratch::default();
    for leaf_id in touched_leaves {
        let mut leaf = leaf_objs.remove(&leaf_id).unwrap();
        let row_ids = leaf_row_ids.remove(&leaf_id).unwrap();
        let old_c = leaf.centroid.clone();
        let count = row_ids.len();

        if count == 0 {
            leaf.centroid = vec![0.0; dim];
            leaf.rotated_centroid.fill(0.0);
            leaf.reset_and_encode(&[], &[], quantizer, dim, &mut scratch);
        } else {
            let mut flat_vecs = vec![0.0f32; count * dim];
            let mut missing: Vec<(RowId, usize)> = Vec::new();
            for (slot, &id) in row_ids.iter().enumerate() {
                if let Some(v) = vecs_by_id.get(&id) {
                    flat_vecs[slot * dim..(slot + 1) * dim].copy_from_slice(v);
                } else {
                    missing.push((id, slot));
                }
            }
            if !missing.is_empty() {
                let missing_ids: Vec<RowId> = missing.iter().map(|&(id, _)| id).collect();
                let fetched = tx.get_vectors_flat(&missing_ids)?;
                for (i, &(_, slot)) in missing.iter().enumerate() {
                    flat_vecs[slot * dim..(slot + 1) * dim]
                        .copy_from_slice(&fetched[i * dim..(i + 1) * dim]);
                }
            }
            leaf.centroid = compute_mean_centroid_flat(&flat_vecs, count, dim, metric);
            leaf.reset_and_encode(&row_ids, &flat_vecs, quantizer, dim, &mut scratch);
        }

        let parent_id = leaf.parent_id;
        let new_c = leaf.centroid.clone();
        tx.put_leaf_node(leaf_id, leaf);

        if count > 0
            && let Some(pid) = parent_id
        {
            propagate_centroid_upward(tx, leaf_id, pid, count as u64, &old_c, &new_c, config)?;
        }

        if count > config.max_leaf_size || (leaf_id != root_id && count < config.min_leaf_size) {
            maintenance.push_back(leaf_id);
        }
    }

    Ok(())
}
