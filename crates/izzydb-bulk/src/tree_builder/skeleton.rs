//! Skeleton tree construction: balanced weighted splits and hierarchy building.

use std::collections::HashMap;


use izzydb::config::{DistanceMetric, NodeId, TreeConfig};
use izzydb::error::Result;
use izzydb::kmeans::balanced_two_means;
use izzydb::metric::{cluster_distance, compute_mean_centroid, compute_weighted_centroid};
use izzydb::node::{AnnMeta, PackedInternalNode};

use super::*;

pub(super) struct SkelLeaf {
    pub(super) centroid: Vec<f32>,
    pub(super) weight: u64,
    pub(super) parent: Option<usize>,
    pub(super) entries: Vec<VecEntry>,
    pub(super) alive: bool,
}

pub(super) struct SkelInternal {
    pub(super) level: u8,
    pub(super) parent: Option<usize>,
    pub(super) children: Vec<usize>,
    pub(super) subtree_count: u64,
    pub(super) centroid: Vec<f32>,
    pub(super) alive: bool,
}

pub(super) fn build_skeleton_leaves(
    sample_refs: &[&[f32]],
    total_n: usize,
    sample_n: usize,
    target_leaf_size: usize,
    min_leaf_size: usize,
    dim: usize,
    metric: DistanceMetric,
) -> Vec<SkelLeaf> {
    let initial_indices: Vec<usize> = (0..sample_refs.len()).collect();
    let mut out = Vec::new();
    let mut stack = vec![initial_indices];

    while let Some(indices) = stack.pop() {
        let projected = (indices.len() as f64) * (total_n as f64) / (sample_n as f64);
        let force_split = out.is_empty() && stack.is_empty() && indices.len() >= 2;
        if (projected > target_leaf_size as f64 || force_split) && indices.len() >= 2 {
            let min_split = ((min_leaf_size as f64 * (sample_n as f64) / (total_n as f64)).floor()
                as usize)
                .max(1)
                .min(indices.len() / 2);
            let items: Vec<(&[f32], u64)> = indices.iter().map(|&i| (sample_refs[i], 1)).collect();
            let split = balanced_two_means(&items, min_split, indices.len(), dim, metric);
            let left: Vec<usize> = split.part1.into_iter().map(|p| indices[p]).collect();
            let right: Vec<usize> = split.part2.into_iter().map(|p| indices[p]).collect();
            stack.push(right);
            stack.push(left);
        } else {
            let refs: Vec<&[f32]> = indices.iter().map(|&i| sample_refs[i]).collect();
            let centroid = compute_mean_centroid(&refs, dim, metric);
            out.push(SkelLeaf {
                centroid,
                weight: projected.round().max(1.0) as u64,
                parent: None,
                entries: Vec::new(),
                alive: true,
            });
        }
    }
    out
}

pub(super) struct SkelTree {
    pub(super) leaves: Vec<SkelLeaf>,
    pub(super) internals: Vec<SkelInternal>,
    pub(super) root_idx: usize,
    pub(super) root_level: u8,
}

pub(super) fn build_skeleton_hierarchy(mut leaves: Vec<SkelLeaf>, config: &TreeConfig) -> SkelTree {
    let dim = config.dim;
    let metric = config.metric;
    let min_f = config.min_internal_fanout;
    let max_f = config.max_internal_fanout;
    let target_f = ((0.75 * max_f as f32).round() as usize).clamp(min_f, max_f);

    let mut internals: Vec<SkelInternal> = Vec::new();
    let mut cur_level_is_leaf = true;
    let mut cur_nodes: Vec<(usize, u64, Vec<f32>)> = leaves
        .iter()
        .enumerate()
        .map(|(i, l)| (i, l.weight, l.centroid.clone()))
        .collect();
    let mut level = 1u8;

    loop {
        let c = cur_nodes.len();
        if c <= max_f {
            let root_idx = internals.len();
            let children: Vec<usize> = cur_nodes.iter().map(|&(idx, _, _)| idx).collect();
            let weighted: Vec<(&[f32], u64)> = cur_nodes
                .iter()
                .map(|(_, w, cent)| (cent.as_slice(), *w))
                .collect();
            let subtree_count: u64 = cur_nodes.iter().map(|(_, w, _)| *w).sum();
            let centroid = compute_weighted_centroid(&weighted, dim, metric);
            for &cid in &children {
                if cur_level_is_leaf {
                    leaves[cid].parent = Some(root_idx);
                } else {
                    internals[cid].parent = Some(root_idx);
                }
            }
            internals.push(SkelInternal {
                level,
                parent: None,
                children,
                subtree_count,
                centroid,
                alive: true,
            });
            return SkelTree {
                leaves,
                internals,
                root_idx,
                root_level: level,
            };
        }

        let mut p = c.div_ceil(target_f).max(2);
        while c / p < min_f {
            p -= 1;
        }
        p = p.max(c.div_ceil(max_f));

        let items: Vec<(&[f32], u64)> = cur_nodes
            .iter()
            .map(|(_, w, cent)| (cent.as_slice(), *w))
            .collect();
        let groups = split_weighted_into_k(&items, p, min_f, max_f, dim, metric);

        let mut next_nodes = Vec::with_capacity(groups.len());
        for group in groups {
            let parent_idx = internals.len();
            let children: Vec<usize> = group.iter().map(|&g| cur_nodes[g].0).collect();
            let weighted: Vec<(&[f32], u64)> = group
                .iter()
                .map(|&g| (cur_nodes[g].2.as_slice(), cur_nodes[g].1))
                .collect();
            let subtree_count: u64 = group.iter().map(|&g| cur_nodes[g].1).sum();
            let centroid = compute_weighted_centroid(&weighted, dim, metric);

            for &cid in &children {
                if cur_level_is_leaf {
                    leaves[cid].parent = Some(parent_idx);
                } else {
                    internals[cid].parent = Some(parent_idx);
                }
            }
            internals.push(SkelInternal {
                level,
                parent: None,
                children,
                subtree_count,
                centroid: centroid.clone(),
                alive: true,
            });
            next_nodes.push((parent_idx, subtree_count, centroid));
        }

        cur_nodes = next_nodes;
        cur_level_is_leaf = false;
        level += 1;
    }
}

pub(super) fn split_weighted_into_k(
    items: &[(&[f32], u64)],
    k: usize,
    min_f: usize,
    max_f: usize,
    dim: usize,
    metric: DistanceMetric,
) -> Vec<Vec<usize>> {
    let indices: Vec<usize> = (0..items.len()).collect();
    split_weighted_indices(items, &indices, k, min_f, max_f, dim, metric)
}

pub(super) fn split_weighted_indices(
    all_items: &[(&[f32], u64)],
    indices: &[usize],
    k: usize,
    min_f: usize,
    max_f: usize,
    dim: usize,
    metric: DistanceMetric,
) -> Vec<Vec<usize>> {
    if k <= 1 {
        return vec![indices.to_vec()];
    }
    let n = indices.len();
    let k_left = k / 2;
    let k_right = k - k_left;
    let min_left = (k_left * min_f).max(n.saturating_sub(k_right * max_f));
    let max_left = (k_left * max_f).min(n.saturating_sub(k_right * min_f));
    debug_assert!(min_left <= max_left);

    let sub_items: Vec<(&[f32], u64)> = indices.iter().map(|&i| all_items[i]).collect();
    let mut split = balanced_two_means(
        &sub_items,
        min_left.min(n / 2),
        max_left.max(n - min_left),
        dim,
        metric,
    );

    if split.part1.len() < min_left {
        let need = min_left - split.part1.len();
        move_by_margin(
            &mut split.part1,
            &mut split.part2,
            &split.centroid1,
            &split.centroid2,
            need,
            &sub_items,
            metric,
        );
    } else if split.part1.len() > max_left {
        let excess = split.part1.len() - max_left;
        move_by_margin(
            &mut split.part2,
            &mut split.part1,
            &split.centroid2,
            &split.centroid1,
            excess,
            &sub_items,
            metric,
        );
    }

    let left_global: Vec<usize> = split.part1.into_iter().map(|p| indices[p]).collect();
    let right_global: Vec<usize> = split.part2.into_iter().map(|p| indices[p]).collect();

    let mut out = split_weighted_indices(all_items, &left_global, k_left, min_f, max_f, dim, metric);
    out.extend(split_weighted_indices(
        all_items,
        &right_global,
        k_right,
        min_f,
        max_f,
        dim,
        metric,
    ));
    out
}

pub(super) fn move_by_margin(
    dst: &mut Vec<usize>,
    src: &mut Vec<usize>,
    c_dst: &[f32],
    c_src: &[f32],
    count: usize,
    items: &[(&[f32], u64)],
    metric: DistanceMetric,
) {
    if count == 0 {
        return;
    }
    let mut scored: Vec<(usize, f32)> = src
        .iter()
        .map(|&idx| {
            let v = items[idx].0;
            let margin = cluster_distance(metric, v, c_dst) - cluster_distance(metric, v, c_src);
            (idx, margin)
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let move_n = count.min(scored.len());
    let mut moved: Vec<usize> = scored.into_iter().take(move_n).map(|(idx, _)| idx).collect();
    dst.extend_from_slice(&moved);
    moved.sort_unstable();
    src.retain(|idx| moved.binary_search(idx).is_err());
}

impl SkelTree {
    pub(super) fn build_routing_cache(
        &self,
        dim: usize,
        metric: DistanceMetric,
    ) -> HashMap<NodeId, PackedInternalNode> {
        let mut cache = HashMap::with_capacity(self.internals.len());
        for (idx, node) in self.internals.iter().enumerate() {
            let children: Vec<(NodeId, u64, &[f32])> = node
                .children
                .iter()
                .map(|&cid| {
                    if node.level == 1 {
                        let l = &self.leaves[cid];
                        (cid as NodeId, l.weight, l.centroid.as_slice())
                    } else {
                        let c = &self.internals[cid];
                        (cid as NodeId, c.subtree_count, c.centroid.as_slice())
                    }
                })
                .collect();
            let packed = PackedInternalNode::from_children(
                node.level,
                node.parent.map(|p| p as NodeId),
                &children,
                dim,
                metric,
            );
            cache.insert(idx as NodeId, packed);
        }
        cache
    }

    pub(super) fn repair(
        &mut self,
        vectors_bytes: &[u8],
        config: &TreeConfig,
        target_leaf_size: usize,
        pool: &rayon::ThreadPool,
    ) -> Result<()> {
        let dim = config.dim;
        let metric = config.metric;
        let min_leaf = config.min_leaf_size;
        let max_leaf = config.max_leaf_size;

        let oversize_ids: Vec<usize> = (0..self.leaves.len())
            .filter(|&i| self.leaves[i].alive && self.leaves[i].entries.len() > max_leaf)
            .collect();

        type LeafSplits = (usize, Vec<(Vec<VecEntry>, Vec<f32>)>);
        let splits: Vec<LeafSplits> = pool.install(|| {
            oversize_ids
                .par_iter()
                .map(|&lid| {
                    let entries = &self.leaves[lid].entries;
                    let n = entries.len();
                    let min_k = n.div_ceil(max_leaf);
                    let max_k = (n / min_leaf).max(min_k);
                    let k = n.div_ceil(target_leaf_size).clamp(min_k, max_k).max(2);
                    let items: Vec<(&[f32], u64)> = entries
                        .iter()
                        .map(|e| (read_vector_at(vectors_bytes, e.vec_idx, dim), 1u64))
                        .collect();
                    let groups = split_weighted_into_k(&items, k, min_leaf, max_leaf, dim, metric);
                    let parts = groups
                        .into_iter()
                        .map(|g| {
                            let part_entries: Vec<VecEntry> =
                                g.into_iter().map(|i| entries[i]).collect();
                            let refs: Vec<&[f32]> = part_entries
                                .iter()
                                .map(|e| read_vector_at(vectors_bytes, e.vec_idx, dim))
                                .collect();
                            let cent = compute_mean_centroid(&refs, dim, metric);
                            (part_entries, cent)
                        })
                        .collect();
                    (lid, parts)
                })
                .collect()
        });

        for (lid, mut parts) in splits {
            let parent = self.leaves[lid].parent;
            let (first_entries, first_cent) = parts.remove(0);
            self.leaves[lid].entries = first_entries;
            self.leaves[lid].centroid = first_cent;

            for (part_entries, part_cent) in parts {
                let new_lid = self.leaves.len();
                self.leaves.push(SkelLeaf {
                    centroid: part_cent,
                    weight: part_entries.len() as u64,
                    parent,
                    entries: part_entries,
                    alive: true,
                });
                if let Some(pid) = parent {
                    self.internals[pid].children.push(new_lid);
                }
            }
        }

        let leaf_count = self.leaves.len();
        for lid in 0..leaf_count {
            if !self.leaves[lid].alive {
                continue;
            }
            if self.leaves[lid].entries.is_empty() {
                self.detach_leaf(lid);
                continue;
            }
            while self.leaves[lid].alive && self.leaves[lid].entries.len() < min_leaf {
                let Some(sib_id) = self.find_nearest_alive_sibling_leaf(lid, metric) else {
                    break;
                };
                let mut combined = std::mem::take(&mut self.leaves[lid].entries);
                combined.append(&mut self.leaves[sib_id].entries);
                if combined.len() <= max_leaf {
                    let refs: Vec<&[f32]> = combined
                        .iter()
                        .map(|e| read_vector_at(vectors_bytes, e.vec_idx, dim))
                        .collect();
                    self.leaves[sib_id].centroid = compute_mean_centroid(&refs, dim, metric);
                    self.leaves[sib_id].entries = combined;
                    self.detach_leaf(lid);
                    break;
                } else {
                    let items: Vec<(&[f32], u64)> = combined
                        .iter()
                        .map(|e| (read_vector_at(vectors_bytes, e.vec_idx, dim), 1u64))
                        .collect();
                    let split = balanced_two_means(&items, min_leaf, max_leaf, dim, metric);
                    let e1: Vec<VecEntry> = split.part1.into_iter().map(|i| combined[i]).collect();
                    let e2: Vec<VecEntry> = split.part2.into_iter().map(|i| combined[i]).collect();
                    self.leaves[lid].entries = e1;
                    self.leaves[lid].centroid = split.centroid1;
                    self.leaves[sib_id].entries = e2;
                    self.leaves[sib_id].centroid = split.centroid2;
                }
            }
        }

        let alive_leaf_ids: Vec<usize> = (0..self.leaves.len())
            .filter(|&i| self.leaves[i].alive)
            .collect();
        let leaf_centroids: Vec<(usize, Vec<f32>)> = pool.install(|| {
            alive_leaf_ids
                .par_iter()
                .map(|&lid| {
                    let refs: Vec<&[f32]> = self.leaves[lid]
                        .entries
                        .iter()
                        .map(|e| read_vector_at(vectors_bytes, e.vec_idx, dim))
                        .collect();
                    (lid, compute_mean_centroid(&refs, dim, metric))
                })
                .collect()
        });
        for (lid, cent) in leaf_centroids {
            self.leaves[lid].centroid = cent;
            self.leaves[lid].weight = self.leaves[lid].entries.len() as u64;
        }

        self.repair_internals(config)?;
        Ok(())
    }

    pub(super) fn detach_leaf(&mut self, lid: usize) {
        self.leaves[lid].alive = false;
        self.leaves[lid].entries.clear();
        if let Some(pid) = self.leaves[lid].parent.take() {
            self.internals[pid].children.retain(|&c| c != lid);
        }
    }

    pub(super) fn find_nearest_alive_sibling_leaf(&self, lid: usize, metric: DistanceMetric) -> Option<usize> {
        let cent = &self.leaves[lid].centroid;
        if let Some(pid) = self.leaves[lid].parent {
            let best_same_parent = self.internals[pid]
                .children
                .iter()
                .copied()
                .filter(|&cid| cid != lid && self.leaves[cid].alive)
                .min_by(|&a, &b| {
                    let da = cluster_distance(metric, cent, &self.leaves[a].centroid);
                    let db = cluster_distance(metric, cent, &self.leaves[b].centroid);
                    da.total_cmp(&db).then_with(|| a.cmp(&b))
                });
            if best_same_parent.is_some() {
                return best_same_parent;
            }
        }
        (0..self.leaves.len())
            .filter(|&cid| cid != lid && self.leaves[cid].alive)
            .min_by(|&a, &b| {
                let da = cluster_distance(metric, cent, &self.leaves[a].centroid);
                let db = cluster_distance(metric, cent, &self.leaves[b].centroid);
                da.total_cmp(&db).then_with(|| a.cmp(&b))
            })
    }

    pub(super) fn repair_internals(&mut self, config: &TreeConfig) -> Result<()> {
        let dim = config.dim;
        let metric = config.metric;
        let min_f = config.min_internal_fanout;
        let max_f = config.max_internal_fanout;
        let target_f = ((0.75 * max_f as f32).round() as usize).clamp(min_f, max_f);

        let mut level = 1u8;
        while level <= self.root_level {
            let level_nodes: Vec<usize> = (0..self.internals.len())
                .filter(|&i| self.internals[i].alive && self.internals[i].level == level)
                .collect();

            for &nid in &level_nodes {
                self.recompute_internal_stats(nid, dim, metric);
                if self.internals[nid].children.is_empty() {
                    self.detach_internal(nid);
                }
            }

            for &nid in &level_nodes {
                if !self.internals[nid].alive {
                    continue;
                }
                if self.internals[nid].children.len() > max_f {
                    self.split_oversize_internal(nid, target_f, min_f, max_f, dim, metric);
                }
            }

            let current_level_nodes: Vec<usize> = (0..self.internals.len())
                .filter(|&i| self.internals[i].alive && self.internals[i].level == level)
                .collect();
            for &nid in &current_level_nodes {
                if !self.internals[nid].alive || nid == self.root_idx {
                    continue;
                }
                while self.internals[nid].alive
                    && nid != self.root_idx
                    && self.internals[nid].children.len() < min_f
                {
                    let Some(sib_id) = self.find_nearest_alive_sibling_internal(nid, level, metric)
                    else {
                        break;
                    };
                    let mut combined = std::mem::take(&mut self.internals[nid].children);
                    combined.extend_from_slice(&self.internals[sib_id].children);
                    if combined.len() <= max_f {
                        for &cid in &combined {
                            self.set_child_parent(cid, level, Some(sib_id));
                        }
                        self.internals[sib_id].children = combined;
                        self.recompute_internal_stats(sib_id, dim, metric);
                        self.detach_internal(nid);
                        break;
                    } else {
                        let items: Vec<(Vec<f32>, u64)> = combined
                            .iter()
                            .map(|&cid| self.child_summary(cid, level))
                            .collect();
                        let item_refs: Vec<(&[f32], u64)> =
                            items.iter().map(|(c, w)| (c.as_slice(), *w)).collect();
                        let split = balanced_two_means(&item_refs, min_f, max_f, dim, metric);
                        let ch1: Vec<usize> = split.part1.into_iter().map(|i| combined[i]).collect();
                        let ch2: Vec<usize> = split.part2.into_iter().map(|i| combined[i]).collect();
                        for &cid in &ch1 {
                            self.set_child_parent(cid, level, Some(nid));
                        }
                        for &cid in &ch2 {
                            self.set_child_parent(cid, level, Some(sib_id));
                        }
                        self.internals[nid].children = ch1;
                        self.internals[sib_id].children = ch2;
                        self.recompute_internal_stats(nid, dim, metric);
                        self.recompute_internal_stats(sib_id, dim, metric);
                    }
                }
            }

            level += 1;
        }

        while self.internals[self.root_idx].children.len() == 1 && self.root_level > 1 {
            let only_child = self.internals[self.root_idx].children[0];
            self.internals[self.root_idx].alive = false;
            self.internals[only_child].parent = None;
            self.root_idx = only_child;
            self.root_level -= 1;
        }

        for lv in 1..=self.root_level {
            let ids: Vec<usize> = (0..self.internals.len())
                .filter(|&i| self.internals[i].alive && self.internals[i].level == lv)
                .collect();
            for nid in ids {
                self.recompute_internal_stats(nid, dim, metric);
            }
        }

        Ok(())
    }

    pub(super) fn child_summary(&self, cid: usize, parent_level: u8) -> (Vec<f32>, u64) {
        if parent_level == 1 {
            let l = &self.leaves[cid];
            (l.centroid.clone(), l.entries.len() as u64)
        } else {
            let c = &self.internals[cid];
            (c.centroid.clone(), c.subtree_count)
        }
    }

    pub(super) fn set_child_parent(&mut self, cid: usize, parent_level: u8, parent: Option<usize>) {
        if parent_level == 1 {
            self.leaves[cid].parent = parent;
        } else {
            self.internals[cid].parent = parent;
        }
    }

    pub(super) fn recompute_internal_stats(&mut self, nid: usize, dim: usize, metric: DistanceMetric) {
        let level = self.internals[nid].level;
        let summaries: Vec<(Vec<f32>, u64)> = self.internals[nid]
            .children
            .iter()
            .map(|&cid| self.child_summary(cid, level))
            .collect();
        let refs: Vec<(&[f32], u64)> = summaries
            .iter()
            .map(|(c, w)| (c.as_slice(), *w))
            .collect();
        self.internals[nid].subtree_count = summaries.iter().map(|(_, w)| *w).sum();
        self.internals[nid].centroid = compute_weighted_centroid(&refs, dim, metric);
    }

    pub(super) fn detach_internal(&mut self, nid: usize) {
        self.internals[nid].alive = false;
        self.internals[nid].children.clear();
        if let Some(pid) = self.internals[nid].parent.take() {
            self.internals[pid].children.retain(|&c| c != nid);
        }
    }

    pub(super) fn split_oversize_internal(
        &mut self,
        nid: usize,
        target_f: usize,
        min_f: usize,
        max_f: usize,
        dim: usize,
        metric: DistanceMetric,
    ) {
        let level = self.internals[nid].level;
        let parent = self.internals[nid].parent;
        let children = std::mem::take(&mut self.internals[nid].children);
        let n = children.len();
        let min_k = n.div_ceil(max_f);
        let max_k = (n / min_f).max(min_k);
        let k = n.div_ceil(target_f).clamp(min_k, max_k).max(2);

        let summaries: Vec<(Vec<f32>, u64)> = children
            .iter()
            .map(|&cid| self.child_summary(cid, level))
            .collect();
        let refs: Vec<(&[f32], u64)> = summaries
            .iter()
            .map(|(c, w)| (c.as_slice(), *w))
            .collect();
        let mut groups = split_weighted_into_k(&refs, k, min_f, max_f, dim, metric);

        let first_group: Vec<usize> = groups.remove(0).into_iter().map(|i| children[i]).collect();
        for &cid in &first_group {
            self.set_child_parent(cid, level, Some(nid));
        }
        self.internals[nid].children = first_group;
        self.recompute_internal_stats(nid, dim, metric);

        let mut created_ids = vec![nid];
        for g in groups {
            let group_children: Vec<usize> = g.into_iter().map(|i| children[i]).collect();
            let new_id = self.internals.len();
            for &cid in &group_children {
                self.set_child_parent(cid, level, Some(new_id));
            }
            self.internals.push(SkelInternal {
                level,
                parent,
                children: group_children,
                subtree_count: 0,
                centroid: Vec::new(),
                alive: true,
            });
            self.recompute_internal_stats(new_id, dim, metric);
            created_ids.push(new_id);
        }

        if let Some(pid) = parent {
            for &new_id in &created_ids[1..] {
                self.internals[pid].children.push(new_id);
            }
        } else {
            let new_root_idx = self.internals.len();
            for &cid in &created_ids {
                self.internals[cid].parent = Some(new_root_idx);
            }
            self.internals.push(SkelInternal {
                level: level + 1,
                parent: None,
                children: created_ids,
                subtree_count: 0,
                centroid: Vec::new(),
                alive: true,
            });
            self.recompute_internal_stats(new_root_idx, dim, metric);
            self.root_idx = new_root_idx;
            self.root_level = level + 1;
        }
    }

    pub(super) fn find_nearest_alive_sibling_internal(
        &self,
        nid: usize,
        level: u8,
        metric: DistanceMetric,
    ) -> Option<usize> {
        let cent = &self.internals[nid].centroid;
        if let Some(pid) = self.internals[nid].parent {
            let best_same_parent = self.internals[pid]
                .children
                .iter()
                .copied()
                .filter(|&cid| cid != nid && self.internals[cid].alive)
                .min_by(|&a, &b| {
                    let da = cluster_distance(metric, cent, &self.internals[a].centroid);
                    let db = cluster_distance(metric, cent, &self.internals[b].centroid);
                    da.total_cmp(&db).then_with(|| a.cmp(&b))
                });
            if best_same_parent.is_some() {
                return best_same_parent;
            }
        }
        (0..self.internals.len())
            .filter(|&cid| cid != nid && self.internals[cid].alive && self.internals[cid].level == level)
            .min_by(|&a, &b| {
                let da = cluster_distance(metric, cent, &self.internals[a].centroid);
                let db = cluster_distance(metric, cent, &self.internals[b].centroid);
                da.total_cmp(&db).then_with(|| a.cmp(&b))
            })
    }

    pub(super) fn finalize(mut self, dim: usize, metric: DistanceMetric, rabitq_seed: u64) -> BuiltTree {
        let mut internal_order: Vec<usize> = Vec::new();
        let mut leaf_order: Vec<usize> = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.root_idx);

        while let Some(nid) = queue.pop_front() {
            internal_order.push(nid);
            let node = &self.internals[nid];
            if node.level == 1 {
                for &lid in &node.children {
                    leaf_order.push(lid);
                }
            } else {
                for &cid in &node.children {
                    queue.push_back(cid);
                }
            }
        }

        let mut internal_id_map: HashMap<usize, NodeId> =
            HashMap::with_capacity(internal_order.len());
        let mut next_id: NodeId = 1;
        for &idx in &internal_order {
            internal_id_map.insert(idx, next_id);
            next_id += 1;
        }
        let mut leaf_id_map: HashMap<usize, NodeId> = HashMap::with_capacity(leaf_order.len());
        for &idx in &leaf_order {
            leaf_id_map.insert(idx, next_id);
            next_id += 1;
        }

        let mut packed_internals = Vec::with_capacity(internal_order.len());
        for &idx in &internal_order {
            let skel = &self.internals[idx];
            let node_id = internal_id_map[&idx];
            let parent_id = skel.parent.map(|p| internal_id_map[&p]);
            let children: Vec<(NodeId, u64, &[f32])> = skel
                .children
                .iter()
                .map(|&cid| {
                    if skel.level == 1 {
                        let l = &self.leaves[cid];
                        (leaf_id_map[&cid], l.entries.len() as u64, l.centroid.as_slice())
                    } else {
                        let c = &self.internals[cid];
                        (internal_id_map[&cid], c.subtree_count, c.centroid.as_slice())
                    }
                })
                .collect();
            let packed = PackedInternalNode::from_children(
                skel.level,
                parent_id,
                &children,
                dim,
                metric,
            );
            packed_internals.push((node_id, packed));
        }

        let mut assigned_leaves = Vec::with_capacity(leaf_order.len());
        let mut total_vectors: u64 = 0;

        for &idx in &leaf_order {
            let leaf_id = leaf_id_map[&idx];
            let skel = &mut self.leaves[idx];
            let parent_id = skel.parent.map(|p| internal_id_map[&p]);
            let mut entries = std::mem::take(&mut skel.entries);
            entries.sort_unstable_by_key(|e| e.row_id);
            total_vectors += entries.len() as u64;
            assigned_leaves.push(AssignedLeaf {
                leaf_id,
                parent_id,
                centroid: std::mem::take(&mut skel.centroid),
                entries,
            });
        }

        let meta = AnnMeta {
            root_id: 1,
            root_level: self.root_level,
            next_node_id: next_id,
            dim: dim as u32,
            metric,
            num_vectors: total_vectors,
            rabitq_seed,
            rabitq_bits: 1,
            stale_rows: 0,
            vacuum_epoch: 0,
            vacuum_cursor: 0,
        };

        BuiltTree {
            meta,
            internal_nodes: packed_internals,
            leaves: assigned_leaves,
        }
    }
}
