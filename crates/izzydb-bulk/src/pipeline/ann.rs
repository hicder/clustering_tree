//! ANN leaf SST emission.

use std::collections::HashMap;
use std::path::Path;

use rayon::prelude::*;

use izzydb::codec::RowView;
use izzydb::error::Result;
use izzydb::node::PackedLeafNode;
use izzydb::rabitq::RabitqQuantizer;
use izzydb::schema::{CollectionSchema, ColumnStorage};
use izzydb::leaf_attrs::{LeafAttrBlockBuilder, LeafAttrSpec, LeafAttrView, NodeSummary};
use izzydb::storage::{
    CF_LEAF_ENTRIES, CF_NODE_SUMMARIES, cf_options_for, default_sst_block_cache, node_key,
};

use crate::common::parquet_reader::{MappedFile, decode_staged_doc};
use crate::common::sst::{SstFileInfo, SstSink};
use super::items::read_doc_bytes;
use crate::tree_builder::{BuiltTree, read_vector_at};

/// SST files emitted for leaves and node summaries.
///
/// ```text
/// LeafEmitOutput { leaf_ssts, summary_ssts }
/// ```
pub(super) struct LeafEmitOutput {
    /// `cf_leaf_entries` SST parts.
    pub leaf_ssts: Vec<SstFileInfo>,
    /// `cf_node_summaries` SST parts.
    pub summary_ssts: Vec<SstFileInfo>,
}

fn leaf_attr_specs(schema: &CollectionSchema) -> Vec<LeafAttrSpec> {
    schema
        .entries()
        .iter()
        .filter(|e| {
            e.storage == ColumnStorage::Inline
                && e.filterable
                && LeafAttrSpec::is_supported(e.attr_type)
        })
        .map(|e| LeafAttrSpec {
            attr_id: e.id,
            attr_type: e.attr_type,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_ann_leaf_ssts(
    bundle_dir: &Path,
    tree: &BuiltTree,
    dim: usize,
    quantizer: &RabitqQuantizer,
    vectors_mmap: &[u8],
    docs_idx_mmap: &[u8],
    docs_shards: &[MappedFile],
    has_detached_staging: bool,
    schema: &CollectionSchema,
    target_sst_bytes: u64,
    pool: &rayon::ThreadPool,
) -> Result<LeafEmitOutput> {
    let block_cache = default_sst_block_cache();
    let leaf_opts = cf_options_for(CF_LEAF_ENTRIES, &block_cache)?;
    let padded_dim = quantizer.padded_dim();
    let rabitq_words = quantizer.num_words();
    let specs = leaf_attr_specs(schema);

    let num_workers = pool.current_num_threads().min(tree.leaves.len()).max(1);
    let per_worker = tree.leaves.len().div_ceil(num_workers);
    let ranges: Vec<(usize, usize)> = (0..tree.leaves.len())
        .step_by(per_worker)
        .map(|s| (s, (s + per_worker).min(tree.leaves.len())))
        .collect();

    type WorkerLeafOutput = (Vec<SstFileInfo>, Vec<(u32, NodeSummary)>);
    let results: Vec<WorkerLeafOutput> = pool
        .install(|| -> Result<Vec<_>> {
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
                    let mut summaries = Vec::with_capacity(end - start);
                    for leaf in &tree.leaves[start..end] {
                        row_ids.clear();
                        flat_vecs.clear();
                        row_ids.reserve(leaf.entries.len());
                        flat_vecs.reserve(leaf.entries.len() * dim);
                        for entry in &leaf.entries {
                            row_ids.push(entry.row_id);
                            flat_vecs
                                .extend_from_slice(read_vector_at(vectors_mmap, entry.vec_idx, dim));
                        }
                        let mut packed = PackedLeafNode::from_entries(
                            leaf.parent_id,
                            leaf.centroid.clone(),
                            &row_ids,
                            &flat_vecs,
                            quantizer,
                            dim,
                            &mut scratch,
                        );

                        if !specs.is_empty() {
                            let slot_cap = leaf.entries.len();
                            let mut builder = LeafAttrBlockBuilder::new(&specs, slot_cap);
                            for (slot, entry) in leaf.entries.iter().enumerate() {
                                let staged_bytes = read_doc_bytes(
                                    docs_idx_mmap,
                                    docs_shards,
                                    entry.vec_idx,
                                );
                                let (inline_bytes, _) = decode_staged_doc(
                                    staged_bytes,
                                    has_detached_staging,
                                )?;
                                let view = RowView::parse(inline_bytes)?;
                                builder.set_from_row_view(slot, &view)?;
                            }
                            let block = builder.finish();
                            let bytes = block.encode();
                            let view = LeafAttrView::parse(&bytes)?;
                            summaries.push((
                                leaf.leaf_id,
                                NodeSummary::from_leaf(&view, slot_cap as u32),
                            ));
                            packed.attr_block = bytes;
                        }
                        sink.put(
                            &node_key(leaf.leaf_id),
                            &packed.encode(dim, padded_dim, rabitq_words),
                        )?;
                    }
                    Ok((sink.finish()?, summaries))
                })
                .collect::<Result<Vec<_>>>()
        })?;

    let mut leaf_ssts = Vec::new();
    let mut summaries = HashMap::new();
    for (leaf, leaf_summaries) in results {
        leaf_ssts.extend(leaf);
        for (id, summary) in leaf_summaries {
            summaries.insert(id, summary);
        }
    }

    let mut summary_ssts = Vec::new();
    if !summaries.is_empty() {
        let mut internals: Vec<&(izzydb::config::NodeId, izzydb::node::PackedInternalNode)> =
            tree.internal_nodes.iter().collect();
        internals.sort_unstable_by_key(|&&(nid, ref node)| (node.level, nid));
        for &(nid, node) in &internals {
            let mut summary = NodeSummary {
                row_count: 0,
                attrs: Vec::new(),
                complete: true,
            };
            for &child in &node.child_ids {
                match summaries.get(&child) {
                    Some(child_summary) => summary.merge_child(child_summary),
                    None => summary.complete = false,
                }
            }
            summaries.insert(*nid, summary);
        }

        let summary_opts = cf_options_for(CF_NODE_SUMMARIES, &block_cache)?;
        let mut summary_sink = SstSink::new(
            bundle_dir.to_path_buf(),
            CF_NODE_SUMMARIES,
            &summary_opts,
            target_sst_bytes,
            format!("{CF_NODE_SUMMARIES}/part-00"),
        );
        let mut ids: Vec<izzydb::config::NodeId> = summaries.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            summary_sink.put(&node_key(id), &summaries[&id].encode())?;
        }
        summary_ssts = summary_sink.finish()?;
    }

    Ok(LeafEmitOutput {
        leaf_ssts,
        summary_ssts,
    })
}
