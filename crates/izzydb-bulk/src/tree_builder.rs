use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use izzydb::config::{DistanceMetric, DocId, FtsDocId, NodeId, TreeConfig};
use izzydb::error::{Result, TreeError};
use izzydb::kmeans::balanced_two_means;
use izzydb::metric::{cluster_distance, compute_mean_centroid, compute_weighted_centroid};
use izzydb::node::{PackedInternalNode, TreeMeta};
use izzydb::search::route_to_leaf_with;

#[derive(Debug, Clone)]
pub struct AssignedLeaf {
    pub leaf_id: NodeId,
    pub parent_id: Option<NodeId>,
    pub centroid: Vec<f32>,
    pub rows: Vec<u32>,
    pub first_fts_doc_id: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub leaf_id: NodeId,
    pub local_id: u32,
    pub fts_doc_id: u32,
}

impl Placement {
    pub const SIZE: usize = 12;

    #[inline]
    pub fn encode(self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.leaf_id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.local_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.fts_doc_id.to_le_bytes());
        buf
    }

    #[inline]
    pub fn decode(bytes: &[u8]) -> Self {
        Self {
            leaf_id: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            local_id: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            fts_doc_id: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        }
    }
}

pub struct BuiltTree {
    pub meta: TreeMeta,
    pub internal_nodes: Vec<(NodeId, PackedInternalNode)>,
    pub leaves: Vec<AssignedLeaf>,
    pub placement_by_row_path: PathBuf,
}

#[inline]
pub fn read_vector_at(vectors_bytes: &[u8], row: u32, dim: usize) -> &[f32] {
    let start = (row as usize) * dim * 4;
    let slice = &vectors_bytes[start..start + dim * 4];
    #[cfg(target_endian = "little")]
    if let Ok(typed) = bytemuck::try_cast_slice::<u8, f32>(slice) {
        return typed;
    }
    panic!("vectors.bin must be 4-byte aligned on little-endian targets");
}

#[inline]
pub fn read_doc_id_at(doc_ids_bytes: &[u8], row: u32) -> DocId {
    let off = (row as usize) * 8;
    u64::from_le_bytes(doc_ids_bytes[off..off + 8].try_into().unwrap())
}

#[allow(clippy::too_many_arguments)]
pub fn build_tree_and_assign(
    live_rows: &[u32],
    total_rows_read: u64,
    vectors_bytes: &[u8],
    doc_ids_bytes: &[u8],
    config: &TreeConfig,
    fill_factor: f32,
    train_sample: usize,
    memory_budget_bytes: usize,
    rabitq_seed: u64,
    has_fts: bool,
    stage_dir: &Path,
    pool: &rayon::ThreadPool,
) -> Result<BuiltTree> {
    let n = live_rows.len();
    let dim = config.dim;
    let metric = config.metric;
    let placement_path = stage_dir.join("placement.bin");

    if has_fts && n >= FtsDocId::MAX as usize {
        return Err(TreeError::Storage("FtsDocId u32 space exhausted".into()));
    }

    // Trivial case: 0 vectors.
    if n == 0 {
        File::create(&placement_path)
            .map_err(|e| TreeError::Storage(format!("create placement.bin: {e}")))?;
        let meta = TreeMeta {
            root_id: 1,
            root_level: 0,
            next_node_id: 2,
            dim: dim as u32,
            metric,
            num_vectors: 0,
            rabitq_seed,
            rabitq_bits: 1,
        };
        let root_leaf = AssignedLeaf {
            leaf_id: 1,
            parent_id: None,
            centroid: vec![0.0; dim],
            rows: Vec::new(),
            first_fts_doc_id: 0,
        };
        return Ok(BuiltTree {
            meta,
            internal_nodes: Vec::new(),
            leaves: vec![root_leaf],
            placement_by_row_path: placement_path,
        });
    }

    // Single root-leaf case: N <= max_leaf_size.
    if n <= config.max_leaf_size {
        let mut rows = live_rows.to_vec();
        rows.sort_unstable_by_key(|&r| read_doc_id_at(doc_ids_bytes, r));
        let vec_refs: Vec<&[f32]> = rows
            .iter()
            .map(|&r| read_vector_at(vectors_bytes, r, dim))
            .collect();
        let centroid = compute_mean_centroid(&vec_refs, dim, metric);
        let leaf = AssignedLeaf {
            leaf_id: 1,
            parent_id: None,
            centroid,
            rows,
            first_fts_doc_id: 0,
        };
        write_placement_file(
            &placement_path,
            total_rows_read as usize,
            std::slice::from_ref(&leaf),
            has_fts,
        )?;
        let meta = TreeMeta {
            root_id: 1,
            root_level: 0,
            next_node_id: 2,
            dim: dim as u32,
            metric,
            num_vectors: n as u64,
            rabitq_seed,
            rabitq_bits: 1,
        };
        return Ok(BuiltTree {
            meta,
            internal_nodes: Vec::new(),
            leaves: vec![leaf],
            placement_by_row_path: placement_path,
        });
    }

    // Phase 2: Sample and train skeleton.
    let target_leaf_size = ((fill_factor * config.max_leaf_size as f32).round() as usize)
        .clamp(config.min_leaf_size, config.max_leaf_size);

    let max_sample_by_mem = (memory_budget_bytes / 2)
        .max(16 * 1024 * 1024)
        / (dim * 4).max(4);
    let sample_cap = train_sample.max(config.max_leaf_size * 4).min(max_sample_by_mem).max(2);
    let sample_n = n.min(sample_cap);

    let sample_rows = sample_live_rows(live_rows, sample_n, rabitq_seed);
    let sample_refs: Vec<&[f32]> = sample_rows
        .iter()
        .map(|&r| read_vector_at(vectors_bytes, r, dim))
        .collect();

    let skel_leaves = build_skeleton_leaves(
        &sample_refs,
        n,
        sample_n,
        target_leaf_size,
        config.min_leaf_size,
        dim,
        metric,
    );
    debug_assert!(skel_leaves.len() >= 2);

    let mut tree = build_skeleton_hierarchy(skel_leaves, config);

    // Phase 3a: Route every live vector through the skeleton in parallel.
    let route_meta = TreeMeta {
        root_id: tree.root_idx as NodeId,
        root_level: tree.root_level,
        next_node_id: 0,
        dim: dim as u32,
        metric,
        num_vectors: n as u64,
        rabitq_seed,
        rabitq_bits: 1,
    };
    let route_cache: HashMap<NodeId, PackedInternalNode> = tree.build_routing_cache(dim, metric);

    let mut assigned_leaf_for_live = vec![0usize; n];
    pool.install(|| -> Result<()> {
        assigned_leaf_for_live
            .par_chunks_mut(4096)
            .zip(live_rows.par_chunks(4096))
            .try_for_each(|(out_chunk, row_chunk)| -> Result<()> {
                for (out, &row) in out_chunk.iter_mut().zip(row_chunk.iter()) {
                    let v = read_vector_at(vectors_bytes, row, dim);
                    let lid = route_to_leaf_with(
                        |nid| {
                            route_cache.get(&nid).ok_or_else(|| {
                                TreeError::CorruptedData(format!("missing skeleton node {nid}"))
                            })
                        },
                        &route_meta,
                        v,
                        config.insert_beam_width,
                        config,
                    )?;
                    *out = lid as usize;
                }
                Ok(())
            })
    })?;
    drop(route_cache);

    // Counting sort live rows into skeleton leaves.
    let num_skel_leaves = tree.leaves.len();
    let mut counts = vec![0usize; num_skel_leaves];
    for &leaf_idx in &assigned_leaf_for_live {
        counts[leaf_idx] += 1;
    }
    for (idx, leaf) in tree.leaves.iter_mut().enumerate() {
        leaf.rows = Vec::with_capacity(counts[idx]);
    }
    for (&row, &leaf_idx) in live_rows.iter().zip(assigned_leaf_for_live.iter()) {
        tree.leaves[leaf_idx].rows.push(row);
    }
    drop(assigned_leaf_for_live);

    // Phase 3b: Repair leaf and internal node sizes bottom-up.
    tree.repair(
        vectors_bytes,
        config,
        target_leaf_size,
        pool,
    )?;

    // Phase 3c: Assign BFS NodeIds, sort leaf rows by doc_id, assign slots and fts_doc_id.
    let built = tree.finalize(
        doc_ids_bytes,
        total_rows_read as usize,
        dim,
        metric,
        rabitq_seed,
        has_fts,
        &placement_path,
    )?;

    Ok(built)
}

fn sample_live_rows(live_rows: &[u32], sample_n: usize, seed: u64) -> Vec<u32> {
    let n = live_rows.len();
    if sample_n >= n {
        return live_rows.to_vec();
    }
    // Floyd's algorithm / reservoir sampling for uniform sample without replacement.
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut reservoir: Vec<u32> = live_rows[..sample_n].to_vec();
    for (i, &r) in live_rows.iter().enumerate().skip(sample_n) {
        let j = rng.random_range(0..=i);
        if j < sample_n {
            reservoir[j] = r;
        }
    }
    reservoir
}

struct SkelLeaf {
    centroid: Vec<f32>,
    weight: u64,
    parent: Option<usize>,
    rows: Vec<u32>,
    alive: bool,
}

struct SkelInternal {
    level: u8,
    parent: Option<usize>,
    children: Vec<usize>, // at level == 1: indices into leaves; at level > 1: indices into internals
    subtree_count: u64,
    centroid: Vec<f32>,
    alive: bool,
}

fn build_skeleton_leaves(
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

    // Enforce at least 2 leaves since total_n > max_leaf_size.
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
                rows: Vec::new(),
                alive: true,
            });
        }
    }
    out
}

struct SkelTree {
    leaves: Vec<SkelLeaf>,
    internals: Vec<SkelInternal>,
    root_idx: usize,
    root_level: u8,
}

fn build_skeleton_hierarchy(mut leaves: Vec<SkelLeaf>, config: &TreeConfig) -> SkelTree {
    let dim = config.dim;
    let metric = config.metric;
    let min_f = config.min_internal_fanout;
    let max_f = config.max_internal_fanout;
    let target_f = ((0.75 * max_f as f32).round() as usize).clamp(min_f, max_f);

    let mut internals: Vec<SkelInternal> = Vec::new();
    // Current level nodes: (node_idx, weight, centroid)
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
            // Single root internal node at `level`.
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

        // Partition `c` nodes into `p` balanced parent groups, each in `[min_f, max_f]`.
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

/// Partition `items` (length `n`, with `k * min_f <= n <= k * max_f`) into `k` groups
/// each of size in `[min_f, max_f]`.
fn split_weighted_into_k(
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

fn split_weighted_indices(
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

fn move_by_margin(
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
    fn build_routing_cache(&self, dim: usize, metric: DistanceMetric) -> HashMap<NodeId, PackedInternalNode> {
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

    fn repair(
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

        // 1. Split oversize leaves in parallel to compute partitions, then update tree.
        let oversize_ids: Vec<usize> = (0..self.leaves.len())
            .filter(|&i| self.leaves[i].alive && self.leaves[i].rows.len() > max_leaf)
            .collect();

        type LeafSplits = (usize, Vec<(Vec<u32>, Vec<f32>)>);
        let splits: Vec<LeafSplits> = pool.install(|| {
            oversize_ids
                .par_iter()
                .map(|&lid| {
                    let rows = &self.leaves[lid].rows;
                    let n = rows.len();
                    let min_k = n.div_ceil(max_leaf);
                    let max_k = (n / min_leaf).max(min_k);
                    let k = n.div_ceil(target_leaf_size).clamp(min_k, max_k).max(2);
                    let items: Vec<(&[f32], u64)> = rows
                        .iter()
                        .map(|&r| (read_vector_at(vectors_bytes, r, dim), 1u64))
                        .collect();
                    let groups = split_weighted_into_k(&items, k, min_leaf, max_leaf, dim, metric);
                    let parts = groups
                        .into_iter()
                        .map(|g| {
                            let part_rows: Vec<u32> = g.into_iter().map(|i| rows[i]).collect();
                            let refs: Vec<&[f32]> = part_rows
                                .iter()
                                .map(|&r| read_vector_at(vectors_bytes, r, dim))
                                .collect();
                            let cent = compute_mean_centroid(&refs, dim, metric);
                            (part_rows, cent)
                        })
                        .collect();
                    (lid, parts)
                })
                .collect()
        });

        for (lid, mut parts) in splits {
            let parent = self.leaves[lid].parent;
            let (first_rows, first_cent) = parts.remove(0);
            self.leaves[lid].rows = first_rows;
            self.leaves[lid].centroid = first_cent;

            for (part_rows, part_cent) in parts {
                let new_lid = self.leaves.len();
                self.leaves.push(SkelLeaf {
                    centroid: part_cent,
                    weight: part_rows.len() as u64,
                    parent,
                    rows: part_rows,
                    alive: true,
                });
                if let Some(pid) = parent {
                    self.internals[pid].children.push(new_lid);
                }
            }
        }

        // 2. Remove empty leaves and merge/rebalance undersize leaves.
        let leaf_count = self.leaves.len();
        for lid in 0..leaf_count {
            if !self.leaves[lid].alive {
                continue;
            }
            if self.leaves[lid].rows.is_empty() {
                self.detach_leaf(lid);
                continue;
            }
            while self.leaves[lid].alive && self.leaves[lid].rows.len() < min_leaf {
                let Some(sib_id) = self.find_nearest_alive_sibling_leaf(lid, metric) else {
                    break;
                };
                let mut combined = std::mem::take(&mut self.leaves[lid].rows);
                combined.append(&mut self.leaves[sib_id].rows);
                if combined.len() <= max_leaf {
                    let refs: Vec<&[f32]> = combined
                        .iter()
                        .map(|&r| read_vector_at(vectors_bytes, r, dim))
                        .collect();
                    self.leaves[sib_id].centroid = compute_mean_centroid(&refs, dim, metric);
                    self.leaves[sib_id].rows = combined;
                    self.detach_leaf(lid);
                    break;
                } else {
                    let items: Vec<(&[f32], u64)> = combined
                        .iter()
                        .map(|&r| (read_vector_at(vectors_bytes, r, dim), 1u64))
                        .collect();
                    let split = balanced_two_means(&items, min_leaf, max_leaf, dim, metric);
                    let rows1: Vec<u32> = split.part1.into_iter().map(|i| combined[i]).collect();
                    let rows2: Vec<u32> = split.part2.into_iter().map(|i| combined[i]).collect();
                    self.leaves[lid].rows = rows1;
                    self.leaves[lid].centroid = split.centroid1;
                    self.leaves[sib_id].rows = rows2;
                    self.leaves[sib_id].centroid = split.centroid2;
                }
            }
        }

        // 3. Recompute leaf centroids in parallel from exact assigned vectors.
        let alive_leaf_ids: Vec<usize> = (0..self.leaves.len())
            .filter(|&i| self.leaves[i].alive)
            .collect();
        let leaf_centroids: Vec<(usize, Vec<f32>)> = pool.install(|| {
            alive_leaf_ids
                .par_iter()
                .map(|&lid| {
                    let refs: Vec<&[f32]> = self.leaves[lid]
                        .rows
                        .iter()
                        .map(|&r| read_vector_at(vectors_bytes, r, dim))
                        .collect();
                    (lid, compute_mean_centroid(&refs, dim, metric))
                })
                .collect()
        });
        for (lid, cent) in leaf_centroids {
            self.leaves[lid].centroid = cent;
            self.leaves[lid].weight = self.leaves[lid].rows.len() as u64;
        }

        // 4. Repair internal nodes bottom-up so all fanout and depth invariants hold.
        self.repair_internals(config)?;

        Ok(())
    }

    fn detach_leaf(&mut self, lid: usize) {
        self.leaves[lid].alive = false;
        self.leaves[lid].rows.clear();
        if let Some(pid) = self.leaves[lid].parent.take() {
            self.internals[pid].children.retain(|&c| c != lid);
        }
    }

    fn find_nearest_alive_sibling_leaf(&self, lid: usize, metric: DistanceMetric) -> Option<usize> {
        let cent = &self.leaves[lid].centroid;
        // Prefer siblings under the same parent first.
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
        // Fallback: any other alive leaf in the tree.
        (0..self.leaves.len())
            .filter(|&cid| cid != lid && self.leaves[cid].alive)
            .min_by(|&a, &b| {
                let da = cluster_distance(metric, cent, &self.leaves[a].centroid);
                let db = cluster_distance(metric, cent, &self.leaves[b].centroid);
                da.total_cmp(&db).then_with(|| a.cmp(&b))
            })
    }

    fn repair_internals(&mut self, config: &TreeConfig) -> Result<()> {
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

            // Recompute stats & prune empty nodes.
            for &nid in &level_nodes {
                self.recompute_internal_stats(nid, dim, metric);
                if self.internals[nid].children.is_empty() {
                    self.detach_internal(nid);
                }
            }

            // Split any oversize nodes at this level.
            for &nid in &level_nodes {
                if !self.internals[nid].alive {
                    continue;
                }
                if self.internals[nid].children.len() > max_f {
                    self.split_oversize_internal(nid, target_f, min_f, max_f, dim, metric);
                }
            }

            // Merge/rebalance any undersize non-root nodes at this level.
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

        // Collapse root if it shrank to a single child (cannot become level 0 because N > max_leaf_size).
        while self.internals[self.root_idx].children.len() == 1 && self.root_level > 1 {
            let only_child = self.internals[self.root_idx].children[0];
            self.internals[self.root_idx].alive = false;
            self.internals[only_child].parent = None;
            self.root_idx = only_child;
            self.root_level -= 1;
        }

        // Final bottom-up stats recomputation.
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

    fn child_summary(&self, cid: usize, parent_level: u8) -> (Vec<f32>, u64) {
        if parent_level == 1 {
            let l = &self.leaves[cid];
            (l.centroid.clone(), l.rows.len() as u64)
        } else {
            let c = &self.internals[cid];
            (c.centroid.clone(), c.subtree_count)
        }
    }

    fn set_child_parent(&mut self, cid: usize, parent_level: u8, parent: Option<usize>) {
        if parent_level == 1 {
            self.leaves[cid].parent = parent;
        } else {
            self.internals[cid].parent = parent;
        }
    }

    fn recompute_internal_stats(&mut self, nid: usize, dim: usize, metric: DistanceMetric) {
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

    fn detach_internal(&mut self, nid: usize) {
        self.internals[nid].alive = false;
        self.internals[nid].children.clear();
        if let Some(pid) = self.internals[nid].parent.take() {
            self.internals[pid].children.retain(|&c| c != nid);
        }
    }

    fn split_oversize_internal(
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
            // Split root: create a new root above `created_ids` (splitting further if > max_f).
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

    fn find_nearest_alive_sibling_internal(
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

    #[allow(clippy::too_many_arguments)]
    fn finalize(
        mut self,
        doc_ids_bytes: &[u8],
        total_rows_read: usize,
        dim: usize,
        metric: DistanceMetric,
        rabitq_seed: u64,
        has_fts: bool,
        placement_path: &Path,
    ) -> Result<BuiltTree> {
        // BFS from root_idx to assign NodeIds: root = 1, internals contiguous, then leaves contiguous.
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

        let mut internal_id_map: HashMap<usize, NodeId> = HashMap::with_capacity(internal_order.len());
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
                        (leaf_id_map[&cid], l.rows.len() as u64, l.centroid.as_slice())
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
        let mut next_fts_doc_id: u32 = 0;
        let mut total_vectors: u64 = 0;

        for &idx in &leaf_order {
            let leaf_id = leaf_id_map[&idx];
            let skel = &mut self.leaves[idx];
            let parent_id = skel.parent.map(|p| internal_id_map[&p]);
            let mut rows = std::mem::take(&mut skel.rows);
            rows.sort_unstable_by_key(|&r| read_doc_id_at(doc_ids_bytes, r));

            let first_fts_doc_id = next_fts_doc_id;
            if has_fts {
                next_fts_doc_id = next_fts_doc_id
                    .checked_add(rows.len() as u32)
                    .ok_or_else(|| TreeError::Storage("FtsDocId u32 space exhausted".into()))?;
            }
            total_vectors += rows.len() as u64;
            assigned_leaves.push(AssignedLeaf {
                leaf_id,
                parent_id,
                centroid: std::mem::take(&mut skel.centroid),
                rows,
                first_fts_doc_id,
            });
        }

        write_placement_file(placement_path, total_rows_read, &assigned_leaves, has_fts)?;

        let meta = TreeMeta {
            root_id: 1,
            root_level: self.root_level,
            next_node_id: next_id,
            dim: dim as u32,
            metric,
            num_vectors: total_vectors,
            rabitq_seed,
            rabitq_bits: 1,
        };

        Ok(BuiltTree {
            meta,
            internal_nodes: packed_internals,
            leaves: assigned_leaves,
            placement_by_row_path: placement_path.to_path_buf(),
        })
    }
}

fn write_placement_file(
    placement_path: &Path,
    total_rows_read: usize,
    leaves: &[AssignedLeaf],
    has_fts: bool,
) -> Result<()> {
    let mut placements = vec![0u8; total_rows_read * Placement::SIZE];
    for leaf in leaves {
        for (slot, &row) in leaf.rows.iter().enumerate() {
            let fts_doc_id = if has_fts {
                leaf.first_fts_doc_id + slot as u32
            } else {
                FtsDocId::MAX
            };
            let enc = Placement {
                leaf_id: leaf.leaf_id,
                local_id: slot as u32,
                fts_doc_id,
            }
            .encode();
            let off = (row as usize) * Placement::SIZE;
            placements[off..off + Placement::SIZE].copy_from_slice(&enc);
        }
    }
    let file = File::create(placement_path)
        .map_err(|e| TreeError::Storage(format!("create placement.bin: {e}")))?;
    let mut writer = BufWriter::with_capacity(256 * 1024, file);
    writer
        .write_all(&placements)
        .map_err(|e| TreeError::Storage(format!("write placement.bin: {e}")))?;
    writer
        .flush()
        .map_err(|e| TreeError::Storage(format!("flush placement.bin: {e}")))?;
    Ok(())
}
