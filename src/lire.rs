use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::config::{DocId, NodeId};
use crate::error::Result;
use crate::metric::{cluster_distance, compute_mean_centroid_flat};
use crate::node::LeafNodeView;
use crate::rabitq::EncodeScratch;
use crate::search;
use crate::storage::TxContext;
use crate::tree::{
    ClusteringTree, SplitPair, handle_underflow, propagate_centroid_upward, split_node,
};

const MAX_LIRE_ROUNDS: usize = 4;

/// Run SPFresh LIRE boundary reassignment after a leaf split, using a bounded
/// maintenance queue to resolve any resulting overflows or underflows.
pub(crate) fn run_lire(tree: &mut ClusteringTree, initial_split: SplitPair) -> Result<()> {
    if tree.config.lire_neighbor_count == 0 {
        return Ok(());
    }

    let mut lire_rounds = 0usize;
    let mut pending_lire: VecDeque<SplitPair> = VecDeque::from([initial_split]);
    let mut maintenance: VecDeque<NodeId> = VecDeque::new();

    while !pending_lire.is_empty() || !maintenance.is_empty() {
        while let Some(pair) = pending_lire.pop_front() {
            if lire_rounds >= MAX_LIRE_ROUNDS {
                break;
            }
            lire_rounds += 1;
            reassign_split_pair(tree, pair, &mut maintenance)?;
        }

        while let Some(leaf_id) = maintenance.pop_front() {
            let Some(leaf) = tree.storage.get_leaf_node(leaf_id)? else {
                continue;
            };
            let count = leaf.entry_count();
            if count > tree.config.max_leaf_size {
                let mut tx =
                    TxContext::new(&tree.storage, &mut tree.internal_cache, tree.meta.clone());
                let split_pair = split_node(&mut tx, leaf_id, &tree.config, &tree.quantizer)?;
                tree.meta = tx.commit()?;

                for child_id in [split_pair.left_id, split_pair.right_id] {
                    if let Some(ch) = tree.storage.get_leaf_node(child_id)?
                        && ch.entry_count() > tree.config.max_leaf_size
                    {
                        maintenance.push_back(child_id);
                    }
                }

                if lire_rounds < MAX_LIRE_ROUNDS && tree.meta.root_level > 0 {
                    pending_lire.push_back(split_pair);
                    break;
                }
            } else if leaf_id != tree.meta.root_id && count < tree.config.min_leaf_size {
                let mut tx =
                    TxContext::new(&tree.storage, &mut tree.internal_cache, tree.meta.clone());
                handle_underflow(&mut tx, leaf_id, &tree.config, &tree.quantizer)?;
                tree.meta = tx.commit()?;
            }
        }
    }

    Ok(())
}

fn reassign_split_pair(
    tree: &mut ClusteringTree,
    pair: SplitPair,
    maintenance: &mut VecDeque<NodeId>,
) -> Result<()> {
    let Some(u_leaf) = tree.storage.get_leaf_node(pair.left_id)? else {
        return Ok(());
    };
    let Some(v_leaf) = tree.storage.get_leaf_node(pair.right_id)? else {
        return Ok(());
    };

    let m = tree.config.lire_neighbor_count;
    let beam = (m + 2).max(tree.config.insert_beam_width);
    let candidate_leaves = search::search_leaves(
        &tree.internal_cache,
        &tree.meta,
        &pair.old_centroid,
        m + 2,
        beam,
        &tree.config,
    )?;

    let mut neighbor_ids: Vec<NodeId> = candidate_leaves
        .into_iter()
        .map(|(id, _)| id)
        .filter(|&id| id != pair.left_id && id != pair.right_id)
        .take(m)
        .collect();
    if neighbor_ids.is_empty() {
        return Ok(());
    }
    neighbor_ids.sort_unstable();

    let dim = tree.config.dim;
    let padded_dim = tree.quantizer.padded_dim();
    let words = tree.quantizer.num_words();

    let neighbor_slices = tree.storage.multi_get_leaf_nodes_pinned(&neighbor_ids)?;
    let mut neighbor_centroids: HashMap<NodeId, Vec<f32>> = HashMap::new();
    for (&nid, opt_slice) in neighbor_ids.iter().zip(neighbor_slices.iter()) {
        if let Some(slice) = opt_slice {
            let view = LeafNodeView::parse(slice, dim, padded_dim, words)?;
            neighbor_centroids.insert(nid, view.centroid_cow(dim)?.into_owned());
        }
    }
    if neighbor_centroids.is_empty() {
        return Ok(());
    }

    let metric = tree.config.metric;
    let c_u = &u_leaf.centroid;
    let c_v = &v_leaf.centroid;

    let mut moves: Vec<(DocId, NodeId, NodeId)> = Vec::new();

    // 1. Outward check: vectors in U and V that are closer to a neighbor leaf's centroid.
    let mut uv_ids = [pair.left_id, pair.right_id];
    uv_ids.sort_unstable();
    let uv_vec_slices = tree.storage.multi_get_leaf_vectors_pinned(&uv_ids)?;
    for (src_id, opt_slice) in uv_ids.iter().copied().zip(uv_vec_slices.iter()) {
        let Some(slice) = opt_slice else {
            continue;
        };
        let src_centroid = if src_id == pair.left_id { c_u } else { c_v };
        let vecs = crate::node::PackedLeafVectors::decode(slice, dim)?;
        for (i, &doc_id) in vecs.doc_ids.iter().enumerate() {
            let vec = vecs.vec_at(i, dim);
            let d_self = cluster_distance(metric, vec, src_centroid);
            let mut best_neighbor: Option<(NodeId, f32)> = None;
            for (&nid, c_n) in &neighbor_centroids {
                let d_n = cluster_distance(metric, vec, c_n);
                if d_n < d_self
                    && best_neighbor.is_none_or(|(best_id, best_d)| {
                        d_n < best_d || (d_n == best_d && nid < best_id)
                    })
                {
                    best_neighbor = Some((nid, d_n));
                }
            }
            if let Some((to_id, _)) = best_neighbor {
                moves.push((doc_id, src_id, to_id));
            }
        }
    }

    // 2. Inward check: neighbor leaves filtered by NPA condition.
    let mut npa_neighbors: Vec<NodeId> = Vec::new();
    for (&nid, c_n) in &neighbor_centroids {
        let d_old = cluster_distance(metric, c_n, &pair.old_centroid);
        let d_new_min = cluster_distance(metric, c_n, c_u).min(cluster_distance(metric, c_n, c_v));
        if d_new_min < d_old {
            npa_neighbors.push(nid);
        }
    }
    npa_neighbors.sort_unstable();

    if !npa_neighbors.is_empty() {
        let npa_slices = tree.storage.multi_get_leaf_vectors_pinned(&npa_neighbors)?;
        for (nid, opt_slice) in npa_neighbors.iter().copied().zip(npa_slices.iter()) {
            let Some(slice) = opt_slice else {
                continue;
            };
            let c_n = &neighbor_centroids[&nid];
            let vecs = crate::node::PackedLeafVectors::decode(slice, dim)?;
            for (i, &doc_id) in vecs.doc_ids.iter().enumerate() {
                let vec = vecs.vec_at(i, dim);
                let d_n = cluster_distance(metric, vec, c_n);
                let d_u = cluster_distance(metric, vec, c_u);
                let d_v = cluster_distance(metric, vec, c_v);
                if d_u < d_n || d_v < d_n {
                    let to_id = if d_u <= d_v {
                        pair.left_id
                    } else {
                        pair.right_id
                    };
                    moves.push((doc_id, nid, to_id));
                }
            }
        }
    }

    if moves.is_empty() {
        return Ok(());
    }

    let mut tx = TxContext::new(&tree.storage, &mut tree.internal_cache, tree.meta.clone());
    let mut touched_leaves = BTreeSet::new();

    for &(doc_id, from_leaf, to_leaf) in &moves {
        let mut from_vecs = tx.get_leaf_vectors(from_leaf)?;
        let vec = from_vecs
            .get(doc_id, dim)
            .expect("moved doc must exist in source leaf")
            .to_vec();
        from_vecs.remove(doc_id, dim);
        tx.put_leaf_vectors(from_leaf, from_vecs);

        let mut to_vecs = tx.get_leaf_vectors(to_leaf)?;
        to_vecs.upsert(doc_id, &vec, dim);
        tx.put_leaf_vectors(to_leaf, to_vecs);

        tx.put_doc_loc(doc_id, to_leaf);
        touched_leaves.insert(from_leaf);
        touched_leaves.insert(to_leaf);
    }

    let mut scratch = EncodeScratch::default();
    for leaf_id in touched_leaves {
        let vecs = tx.get_leaf_vectors(leaf_id)?;
        let mut leaf = tx.get_leaf_node(leaf_id)?;
        let old_c = leaf.centroid.clone();
        let count = vecs.doc_ids.len();

        if count == 0 {
            leaf.centroid = vec![0.0; dim];
            leaf.rotated_centroid.fill(0.0);
            leaf.doc_ids.clear();
            leaf.res_norm_sqs.clear();
            leaf.scales.clear();
            leaf.codes.clear();
        } else {
            leaf.centroid = compute_mean_centroid_flat(&vecs.vectors, count, dim, metric);
            leaf.re_encode_all(&vecs, &tree.quantizer, dim, &mut scratch);
        }

        let parent_id = leaf.parent_id;
        let new_c = leaf.centroid.clone();
        tx.put_leaf_node(leaf_id, leaf);

        if count > 0
            && let Some(pid) = parent_id
        {
            propagate_centroid_upward(&mut tx, leaf_id, pid, count as u64, &old_c, &new_c, &tree.config)?;
        }

        if count > tree.config.max_leaf_size
            || (leaf_id != tx.meta().root_id && count < tree.config.min_leaf_size)
        {
            maintenance.push_back(leaf_id);
        }
    }

    tree.meta = tx.commit()?;
    Ok(())
}
