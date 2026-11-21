use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::config::{DocId, NodeId};
use crate::error::{Result, TreeError};
use crate::metric::{cluster_distance, compute_mean_centroid};
use crate::search;
use crate::storage::TxContext;
use crate::tree::{
    handle_underflow, propagate_centroid_upward, split_node, ClusteringTree, SplitPair,
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
            let Some(header) = tree.storage.get_node_header(leaf_id)? else {
                continue;
            };
            if header.level != 0 {
                continue;
            }
            let count = header.entry_count as usize;
            if count > tree.config.max_leaf_size {
                let mut tx = TxContext::new(&tree.storage, tree.meta.clone());
                let split_pair = split_node(&mut tx, leaf_id, &tree.config, &tree.quantizer)?;
                tree.meta = tx.commit()?;

                // If the leaf had > 2 * max_leaf_size entries, either half could still exceed max.
                for child_id in [split_pair.left_id, split_pair.right_id] {
                    if let Some(ch) = tree.storage.get_node_header(child_id)?
                        && ch.entry_count as usize > tree.config.max_leaf_size
                    {
                        maintenance.push_back(child_id);
                    }
                }

                if lire_rounds < MAX_LIRE_ROUNDS && tree.meta.root_level > 0 {
                    pending_lire.push_back(split_pair);
                    break;
                }
            } else if leaf_id != tree.meta.root_id && count < tree.config.min_leaf_size {
                let mut tx = TxContext::new(&tree.storage, tree.meta.clone());
                handle_underflow(&mut tx, leaf_id, &tree.config, &tree.quantizer)?;
                tree.meta = tx.commit()?;
            }
        }
    }

    Ok(())
}

fn fetch_leaf_vectors(tree: &ClusteringTree, leaf_id: NodeId) -> Result<Vec<(DocId, Vec<f32>)>> {
    let doc_ids = tree.storage.scan_leaf_doc_ids(leaf_id)?;
    let vecs = tree.storage.multi_get_vectors(&doc_ids)?;
    let mut out = Vec::with_capacity(doc_ids.len());
    for (doc_id, opt_vec) in doc_ids.into_iter().zip(vecs) {
        let vec = opt_vec.ok_or_else(|| {
            TreeError::CorruptedData(format!("missing full vector for doc {doc_id}"))
        })?;
        out.push((doc_id, vec));
    }
    Ok(out)
}

fn reassign_split_pair(
    tree: &mut ClusteringTree,
    pair: SplitPair,
    maintenance: &mut VecDeque<NodeId>,
) -> Result<()> {
    let Some(u_header) = tree.storage.get_node_header(pair.left_id)? else {
        return Ok(());
    };
    let Some(v_header) = tree.storage.get_node_header(pair.right_id)? else {
        return Ok(());
    };
    if u_header.level != 0 || v_header.level != 0 {
        return Ok(());
    }

    let m = tree.config.lire_neighbor_count;
    let beam = (m + 2).max(tree.config.insert_beam_width);
    let candidate_leaves = search::search_leaves(
        &tree.storage,
        &tree.meta,
        &pair.old_centroid,
        m + 2,
        beam,
        &tree.config,
    )?;

    let neighbor_ids: Vec<NodeId> = candidate_leaves
        .into_iter()
        .map(|(id, _)| id)
        .filter(|&id| id != pair.left_id && id != pair.right_id)
        .take(m)
        .collect();

    if neighbor_ids.is_empty() {
        return Ok(());
    }

    let mut neighbor_centroids: HashMap<NodeId, Vec<f32>> = HashMap::new();
    for &nid in &neighbor_ids {
        if let Some(nh) = tree.storage.get_node_header(nid)?
            && nh.level == 0
        {
            neighbor_centroids.insert(nid, nh.centroid);
        }
    }
    if neighbor_centroids.is_empty() {
        return Ok(());
    }

    let metric = tree.config.metric;
    let c_u = &u_header.centroid;
    let c_v = &v_header.centroid;

    // Collect `(doc_id, from_leaf, to_leaf)` across outward and inward checks.
    let mut moves: Vec<(DocId, NodeId, NodeId)> = Vec::new();

    // 1. Outward check: vectors in U and V that are closer to a neighbor leaf's centroid.
    for (src_id, src_centroid) in [(pair.left_id, c_u), (pair.right_id, c_v)] {
        let entries = fetch_leaf_vectors(tree, src_id)?;
        for (doc_id, vec) in entries {
            let d_self = cluster_distance(metric, &vec, src_centroid);
            let mut best_neighbor: Option<(NodeId, f32)> = None;
            for (&nid, c_n) in &neighbor_centroids {
                let d_n = cluster_distance(metric, &vec, c_n);
                if d_n < d_self
                    && best_neighbor
                        .is_none_or(|(best_id, best_d)| {
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
    for (&nid, c_n) in &neighbor_centroids {
        let d_old = cluster_distance(metric, c_n, &pair.old_centroid);
        let d_new_min = cluster_distance(metric, c_n, c_u).min(cluster_distance(metric, c_n, c_v));
        if d_new_min >= d_old {
            continue;
        }

        let entries = fetch_leaf_vectors(tree, nid)?;
        for (doc_id, vec) in entries {
            let d_n = cluster_distance(metric, &vec, c_n);
            let d_u = cluster_distance(metric, &vec, c_u);
            let d_v = cluster_distance(metric, &vec, c_v);
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

    let mut tx = TxContext::new(&tree.storage, tree.meta.clone());
    let mut touched_leaves = BTreeSet::new();
    let mut leaf_docs: HashMap<NodeId, BTreeSet<DocId>> = HashMap::new();

    for &(doc_id, from_leaf, to_leaf) in &moves {
        if let std::collections::hash_map::Entry::Vacant(e) = leaf_docs.entry(from_leaf) {
            let ids: BTreeSet<DocId> = tx.get_leaf_doc_ids(from_leaf)?.into_iter().collect();
            e.insert(ids);
        }
        if let std::collections::hash_map::Entry::Vacant(e) = leaf_docs.entry(to_leaf) {
            let ids: BTreeSet<DocId> = tx.get_leaf_doc_ids(to_leaf)?.into_iter().collect();
            e.insert(ids);
        }
        leaf_docs.get_mut(&from_leaf).unwrap().remove(&doc_id);
        leaf_docs.get_mut(&to_leaf).unwrap().insert(doc_id);

        tx.delete_leaf_entry(from_leaf, doc_id);
        tx.put_doc_loc(doc_id, to_leaf);
        touched_leaves.insert(from_leaf);
        touched_leaves.insert(to_leaf);
    }

    for leaf_id in touched_leaves {
        let doc_ids: Vec<DocId> = leaf_docs[&leaf_id].iter().copied().collect();
        let mut header = tx.get_header(leaf_id)?;
        let old_c = header.centroid.clone();
        let count = doc_ids.len();
        header.entry_count = count as u32;
        header.subtree_count = count as u64;
        if count == 0 {
            header.centroid = vec![0.0; tree.config.dim];
        } else {
            let vecs = tx.get_vectors_batch(&doc_ids)?;
            let refs: Vec<&[f32]> = vecs.iter().map(|(_, v)| v.as_slice()).collect();
            header.centroid = compute_mean_centroid(&refs, tree.config.dim, metric);
            for (doc_id, vec) in vecs {
                let q_entry = tree.quantizer.encode(&vec, &header.centroid);
                tx.put_leaf_entry(leaf_id, doc_id, q_entry);
            }
        }
        tx.put_header(leaf_id, header);

        if count > 0 && leaf_id != tx.meta().root_id {
            propagate_centroid_upward(&mut tx, leaf_id, &old_c, &tree.config)?;
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
