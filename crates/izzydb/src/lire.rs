use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::config::{DocId, NodeId};
use crate::error::{Result, TreeError};
use crate::metric::{cluster_distance, compute_mean_centroid_flat};
use crate::node::{LeafNodeView, decode_f32_slice};
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
    let mut neighbor_doc_ids: HashMap<NodeId, Vec<DocId>> = HashMap::new();
    for (&nid, opt_slice) in neighbor_ids.iter().zip(neighbor_slices.iter()) {
        if let Some(slice) = opt_slice {
            let view = LeafNodeView::parse(slice, dim, padded_dim, words)?;
            neighbor_centroids.insert(nid, view.centroid_cow(dim)?.into_owned());
            let mut doc_ids = Vec::with_capacity(view.entry_count);
            for &chunk in view.doc_ids_bytes.as_chunks::<8>().0 {
                doc_ids.push(u64::from_le_bytes(chunk));
            }
            neighbor_doc_ids.insert(nid, doc_ids);
        }
    }
    if neighbor_centroids.is_empty() {
        return Ok(());
    }

    let metric = tree.config.metric;
    let c_u = &u_leaf.centroid;
    let c_v = &v_leaf.centroid;

    // Identify NPA-qualifying neighbors up front so we can batch-fetch all required vectors once.
    let mut npa_neighbors: Vec<NodeId> = Vec::new();
    for (&nid, c_n) in &neighbor_centroids {
        let d_old = cluster_distance(metric, c_n, &pair.old_centroid);
        let d_new_min = cluster_distance(metric, c_n, c_u).min(cluster_distance(metric, c_n, c_v));
        if d_new_min < d_old {
            npa_neighbors.push(nid);
        }
    }
    npa_neighbors.sort_unstable();

    let mut required_doc_ids: Vec<DocId> =
        Vec::with_capacity(u_leaf.doc_ids.len() + v_leaf.doc_ids.len());
    required_doc_ids.extend_from_slice(&u_leaf.doc_ids);
    required_doc_ids.extend_from_slice(&v_leaf.doc_ids);
    for &nid in &npa_neighbors {
        if let Some(ids) = neighbor_doc_ids.get(&nid) {
            required_doc_ids.extend_from_slice(ids);
        }
    }
    required_doc_ids.sort_unstable();
    required_doc_ids.dedup();

    let vec_slices = tree.storage.multi_get_vectors_pinned(&required_doc_ids)?;
    let mut vecs_by_id: HashMap<DocId, Vec<f32>> = HashMap::with_capacity(required_doc_ids.len());
    for (&doc_id, opt_slice) in required_doc_ids.iter().zip(vec_slices.iter()) {
        let slice = opt_slice.as_ref().ok_or_else(|| {
            TreeError::CorruptedData(format!("missing vector {doc_id} in cf_vectors"))
        })?;
        vecs_by_id.insert(doc_id, decode_f32_slice(slice, dim)?);
    }

    let mut moves: Vec<(DocId, NodeId, NodeId)> = Vec::new();

    // 1. Outward check: vectors in U and V that are closer to a neighbor leaf's centroid.
    let mut uv_leaves = [(pair.left_id, &u_leaf), (pair.right_id, &v_leaf)];
    uv_leaves.sort_unstable_by_key(|&(id, _)| id);
    for (src_id, src_leaf) in uv_leaves {
        let src_centroid = &src_leaf.centroid;
        for &doc_id in &src_leaf.doc_ids {
            let vec = &vecs_by_id[&doc_id];
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
    for &nid in &npa_neighbors {
        let c_n = &neighbor_centroids[&nid];
        let Some(doc_ids) = neighbor_doc_ids.get(&nid) else {
            continue;
        };
        for &doc_id in doc_ids {
            let vec = &vecs_by_id[&doc_id];
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

    if moves.is_empty() {
        return Ok(());
    }

    let mut tx = TxContext::new(&tree.storage, &mut tree.internal_cache, tree.meta.clone());
    let mut touched_leaves = BTreeSet::new();

    for &(doc_id, from_leaf_id, to_leaf_id) in &moves {
        let mut from_leaf = tx.get_leaf_node(from_leaf_id)?;
        let pos = from_leaf
            .doc_ids
            .binary_search(&doc_id)
            .expect("moved doc must exist in source leaf");
        from_leaf.doc_ids.remove(pos);
        tx.put_leaf_node(from_leaf_id, from_leaf);

        let mut to_leaf = tx.get_leaf_node(to_leaf_id)?;
        if let Err(pos) = to_leaf.doc_ids.binary_search(&doc_id) {
            to_leaf.doc_ids.insert(pos, doc_id);
        }
        tx.put_leaf_node(to_leaf_id, to_leaf);

        tx.put_doc_loc(doc_id, to_leaf_id);
        touched_leaves.insert(from_leaf_id);
        touched_leaves.insert(to_leaf_id);
    }

    let mut scratch = EncodeScratch::default();
    for leaf_id in touched_leaves {
        let mut leaf = tx.get_leaf_node(leaf_id)?;
        let doc_ids = leaf.doc_ids.clone();
        let old_c = leaf.centroid.clone();
        let count = doc_ids.len();

        if count == 0 {
            leaf.centroid = vec![0.0; dim];
            leaf.rotated_centroid.fill(0.0);
            leaf.res_norm_sqs.clear();
            leaf.scales.clear();
            leaf.codes.clear();
        } else {
            let mut flat_vecs = vec![0.0f32; count * dim];
            let mut missing: Vec<(DocId, usize)> = Vec::new();
            for (slot, &id) in doc_ids.iter().enumerate() {
                if let Some(v) = vecs_by_id.get(&id) {
                    flat_vecs[slot * dim..(slot + 1) * dim].copy_from_slice(v);
                } else {
                    missing.push((id, slot));
                }
            }
            if !missing.is_empty() {
                let missing_ids: Vec<DocId> = missing.iter().map(|&(id, _)| id).collect();
                let fetched = tx.get_vectors_flat(&missing_ids)?;
                for (i, &(_, slot)) in missing.iter().enumerate() {
                    flat_vecs[slot * dim..(slot + 1) * dim]
                        .copy_from_slice(&fetched[i * dim..(i + 1) * dim]);
                }
            }
            leaf.centroid = compute_mean_centroid_flat(&flat_vecs, count, dim, metric);
            leaf.re_encode_all(&doc_ids, &flat_vecs, &tree.quantizer, dim, &mut scratch);
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
