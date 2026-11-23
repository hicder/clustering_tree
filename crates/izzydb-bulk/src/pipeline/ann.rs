//! ANN leaf SST emission.

use std::path::Path;

use rayon::prelude::*;

use izzydb::error::Result;
use izzydb::node::PackedLeafNode;
use izzydb::rabitq::RabitqQuantizer;
use izzydb::storage::{
    CF_LEAF_ENTRIES, cf_options_for, default_sst_block_cache,
    node_key,
};

use crate::common::sst::{SstFileInfo, SstSink};
use crate::tree_builder::{BuiltTree, read_vector_at};


pub(super) fn emit_ann_leaf_ssts(
    bundle_dir: &Path,
    tree: &BuiltTree,
    dim: usize,
    quantizer: &RabitqQuantizer,
    vectors_mmap: &[u8],
    target_sst_bytes: u64,
    pool: &rayon::ThreadPool,
) -> Result<Vec<SstFileInfo>> {
    let block_cache = default_sst_block_cache();
    let leaf_opts = cf_options_for(CF_LEAF_ENTRIES, &block_cache)?;
    let padded_dim = quantizer.padded_dim();
    let rabitq_words = quantizer.num_words();

    let num_workers = pool.current_num_threads().min(tree.leaves.len()).max(1);
    let per_worker = tree.leaves.len().div_ceil(num_workers);
    let ranges: Vec<(usize, usize)> = (0..tree.leaves.len())
        .step_by(per_worker)
        .map(|s| (s, (s + per_worker).min(tree.leaves.len())))
        .collect();

    let results: Vec<Vec<SstFileInfo>> = pool.install(|| -> Result<Vec<_>> {
        ranges
            .par_iter()
            .enumerate()
            .map(|(w_idx, &(start, end))| {
                let mut sink = SstSink::new(
                    bundle_dir.to_path_buf(),
                    CF_LEAF_ENTRIES,
                    &leaf_opts,
                    target_sst_bytes,
                    format!("{CF_LEAF_ENTRIES}/part-{w_idx:02}"),
                );
                let mut scratch = izzydb::rabitq::EncodeScratch::default();
                let mut row_ids = Vec::new();
                let mut flat_vecs = Vec::new();
                for leaf in &tree.leaves[start..end] {
                    row_ids.clear();
                    flat_vecs.clear();
                    row_ids.reserve(leaf.entries.len());
                    flat_vecs.reserve(leaf.entries.len() * dim);
                    for entry in &leaf.entries {
                        row_ids.push(entry.row_id);
                        flat_vecs.extend_from_slice(read_vector_at(vectors_mmap, entry.vec_idx, dim));
                    }
                    let packed = PackedLeafNode::from_entries(
                        leaf.parent_id,
                        leaf.centroid.clone(),
                        &row_ids,
                        &flat_vecs,
                        quantizer,
                        dim,
                        &mut scratch,
                    );
                    sink.put(
                        &node_key(leaf.leaf_id),
                        &packed.encode(dim, padded_dim, rabitq_words),
                    )?;
                }
                sink.finish()
            })
            .collect::<Result<Vec<_>>>()
    })?;

    Ok(results.into_iter().flatten().collect())
}
