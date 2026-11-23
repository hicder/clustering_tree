
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use izzydb::config::{NodeId, RowId, TreeConfig};
use izzydb::error::{Result, TreeError};
use izzydb::metric::compute_mean_centroid;
use izzydb::node::{AnnMeta, PackedInternalNode};
use izzydb::search::route_to_leaf_with;

mod skeleton;
use skeleton::*;

#[derive(Debug, Clone, Copy)]
pub struct VecEntry {
    pub row_id: RowId,
    pub vec_idx: u32,
}

pub struct AssignedLeaf {
    pub leaf_id: NodeId,
    pub parent_id: Option<NodeId>,
    pub centroid: Vec<f32>,
    pub entries: Vec<VecEntry>,
}

pub struct BuiltTree {
    pub meta: AnnMeta,
    pub internal_nodes: Vec<(NodeId, PackedInternalNode)>,
    pub leaves: Vec<AssignedLeaf>,
}

#[inline]
pub fn read_vector_at(vectors_bytes: &[u8], vec_idx: u32, dim: usize) -> &[f32] {
    let start = (vec_idx as usize) * dim * 4;
    let slice = &vectors_bytes[start..start + dim * 4];
    bytemuck::cast_slice(slice)
}

#[allow(clippy::too_many_arguments)]
pub fn build_tree_and_assign(
    entries: &[VecEntry],
    vectors_bytes: &[u8],
    config: &TreeConfig,
    fill_factor: f32,
    train_sample: usize,
    memory_budget_bytes: usize,
    rabitq_seed: u64,
    pool: &rayon::ThreadPool,
) -> Result<BuiltTree> {
    let dim = config.dim;
    let metric = config.metric;
    let n = entries.len();

    if n == 0 {
        let meta = AnnMeta {
            root_id: 1,
            root_level: 0,
            next_node_id: 2,
            dim: dim as u32,
            metric,
            num_vectors: 0,
            rabitq_seed,
            rabitq_bits: 1,
            stale_rows: 0,
            vacuum_epoch: 0,
            vacuum_cursor: 0,
        };
        let root_leaf = AssignedLeaf {
            leaf_id: 1,
            parent_id: None,
            centroid: vec![0.0; dim],
            entries: Vec::new(),
        };
        return Ok(BuiltTree {
            meta,
            internal_nodes: Vec::new(),
            leaves: vec![root_leaf],
        });
    }

    if n <= config.max_leaf_size {
        let mut sorted = entries.to_vec();
        sorted.sort_unstable_by_key(|e| e.row_id);
        let vec_refs: Vec<&[f32]> = sorted
            .iter()
            .map(|e| read_vector_at(vectors_bytes, e.vec_idx, dim))
            .collect();
        let centroid = compute_mean_centroid(&vec_refs, dim, metric);
        let leaf = AssignedLeaf {
            leaf_id: 1,
            parent_id: None,
            centroid,
            entries: sorted,
        };
        let meta = AnnMeta {
            root_id: 1,
            root_level: 0,
            next_node_id: 2,
            dim: dim as u32,
            metric,
            num_vectors: n as u64,
            rabitq_seed,
            rabitq_bits: 1,
            stale_rows: 0,
            vacuum_epoch: 0,
            vacuum_cursor: 0,
        };
        return Ok(BuiltTree {
            meta,
            internal_nodes: Vec::new(),
            leaves: vec![leaf],
        });
    }

    let target_leaf_size = ((fill_factor * config.max_leaf_size as f32).round() as usize)
        .clamp(config.min_leaf_size, config.max_leaf_size);

    let max_sample_by_mem = (memory_budget_bytes / 2)
        .max(16 * 1024 * 1024)
        / (dim * 4).max(4);
    let sample_cap = train_sample
        .max(config.max_leaf_size * 4)
        .min(max_sample_by_mem)
        .max(2);
    let sample_n = n.min(sample_cap);

    let sample_entries = sample_vec_entries(entries, sample_n, rabitq_seed);
    let sample_refs: Vec<&[f32]> = sample_entries
        .iter()
        .map(|e| read_vector_at(vectors_bytes, e.vec_idx, dim))
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

    let route_meta = AnnMeta {
        root_id: tree.root_idx as NodeId,
        root_level: tree.root_level,
        next_node_id: 0,
        dim: dim as u32,
        metric,
        num_vectors: n as u64,
        rabitq_seed,
        rabitq_bits: 1,
        stale_rows: 0,
        vacuum_epoch: 0,
        vacuum_cursor: 0,
    };
    let route_cache = tree.build_routing_cache(dim, metric);

    let mut assigned_leaf_for_entry = vec![0usize; n];
    pool.install(|| -> Result<()> {
        assigned_leaf_for_entry
            .par_chunks_mut(4096)
            .zip(entries.par_chunks(4096))
            .try_for_each(|(out_chunk, entry_chunk)| -> Result<()> {
                for (out, entry) in out_chunk.iter_mut().zip(entry_chunk.iter()) {
                    let v = read_vector_at(vectors_bytes, entry.vec_idx, dim);
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

    let num_skel_leaves = tree.leaves.len();
    let mut counts = vec![0usize; num_skel_leaves];
    for &leaf_idx in &assigned_leaf_for_entry {
        counts[leaf_idx] += 1;
    }
    for (idx, leaf) in tree.leaves.iter_mut().enumerate() {
        leaf.entries = Vec::with_capacity(counts[idx]);
    }
    for (&entry, &leaf_idx) in entries.iter().zip(assigned_leaf_for_entry.iter()) {
        tree.leaves[leaf_idx].entries.push(entry);
    }
    drop(assigned_leaf_for_entry);

    tree.repair(vectors_bytes, config, target_leaf_size, pool)?;
    Ok(tree.finalize(dim, metric, rabitq_seed))
}

fn sample_vec_entries(entries: &[VecEntry], sample_n: usize, seed: u64) -> Vec<VecEntry> {
    let n = entries.len();
    if sample_n >= n {
        return entries.to_vec();
    }
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut reservoir: Vec<VecEntry> = entries[..sample_n].to_vec();
    for (i, &e) in entries.iter().enumerate().skip(sample_n) {
        let j = rng.random_range(0..=i);
        if j < sample_n {
            reservoir[j] = e;
        }
    }
    reservoir
}
