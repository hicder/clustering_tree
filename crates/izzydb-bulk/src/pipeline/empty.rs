//! Empty-import SST emission.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;


use izzydb::codec::{
    ATTR_COUNT_TAG, DETACHED_COUNT_TAG, FTS_DOC_LEN_TAG, INTERNAL_NODE_TAG, Manifest, doc_len_key_coord, seg_meta_key, zone_key,
};
use izzydb::config::{
    NodeId, SegmentConfig, TreeConfig,
};
use izzydb::error::Result;
use izzydb::fts::FtsMeta;
use izzydb::node::{PackedInternalNode, PackedLeafNode};
use izzydb::rabitq::RabitqQuantizer;
use izzydb::schema::CollectionSchema;
use izzydb::storage::{
    ANN_META_KEY, ATTR_COUNTS_VERSION_KEY, CF_LEAF_ENTRIES, CF_NODE_SUMMARIES, DEFAULT_CF_NAME,
    FTS_META_KEY, MANIFEST_KEY, SCHEMA_KEY, cf_options_for, default_sst_block_cache, node_key,
    tagged_vec,
};

use crate::common::parquet_reader::{MappedFile, StagedData};
use crate::common::sst::{SstFileInfo, SstSink};
use crate::tree_builder::BuiltTree;

use super::*;

/// Emit all RocksDB SSTs for an empty-target `bulk_import`.
#[allow(clippy::too_many_arguments)]
pub fn emit_empty_import_ssts(
    bundle_dir: &Path,
    stage_dir: &Path,
    staged: &StagedData,
    built_tree: Option<&BuiltTree>,
    vec_cfg: Option<&TreeConfig>,
    quantizer: Option<&RabitqQuantizer>,
    schema: &CollectionSchema,
    seg_cfg: &SegmentConfig,
    vectors_mmap: &[u8],
    doc_ids_mmap: &[u8],
    docs_idx_mmap: &[u8],
    docs_shards: &[MappedFile],
    target_sst_bytes: u64,
    memory_budget_bytes: usize,
    bit_sliced_index: bool,
    pool: &rayon::ThreadPool,
) -> Result<EmptyImportEmitResult> {
    let block_cache = default_sst_block_cache();
    let n_docs = staged.docs_imported as usize;
    let seal_rows = seg_cfg.seal_rows as usize;
    let has_vectors = vec_cfg.is_some();
    let has_fts = schema.has_fts();

    let mut cf_files: BTreeMap<String, Vec<SstFileInfo>> = BTreeMap::new();

    if n_docs == 0 {
        let manifest = Manifest::new(seg_cfg.seal_rows);
        if let (Some(vcfg), Some(q), Some(tree)) = (vec_cfg, quantizer, built_tree) {
            let leaf_opts = cf_options_for(CF_LEAF_ENTRIES, &block_cache)?;
            let mut leaf_sink = SstSink::new(
                bundle_dir.to_path_buf(),
                CF_LEAF_ENTRIES,
                &leaf_opts,
                target_sst_bytes,
                format!("{CF_LEAF_ENTRIES}/part-00"),
            );
            let empty_leaf = PackedLeafNode::empty(None, vcfg.dim, q.padded_dim());
            leaf_sink.put(
                &node_key(1),
                &empty_leaf.encode(vcfg.dim, q.padded_dim(), q.num_words()),
            )?;
            cf_files.insert(CF_LEAF_ENTRIES.to_owned(), leaf_sink.finish()?);
            let def_opts = cf_options_for(DEFAULT_CF_NAME, &block_cache)?;
            let mut def_sink = SstSink::new(
                bundle_dir.to_path_buf(),
                DEFAULT_CF_NAME,
                &def_opts,
                target_sst_bytes,
                "default/part-00",
            );
            def_sink.put(ANN_META_KEY, &tree.meta.encode())?;
            def_sink.put(ATTR_COUNTS_VERSION_KEY, &[1u8])?;
            if has_fts {
                def_sink.put(FTS_META_KEY, &FtsMeta::default().encode())?;
            }
            def_sink.put(MANIFEST_KEY, &manifest.encode())?;
            def_sink.put(SCHEMA_KEY, &schema.encode())?;
            cf_files.insert(DEFAULT_CF_NAME.to_owned(), def_sink.finish()?);
            return Ok((cf_files, manifest, Some(tree.meta.clone()), 0, 0));
        } else {
            let def_opts = cf_options_for(DEFAULT_CF_NAME, &block_cache)?;
            let mut def_sink = SstSink::new(
                bundle_dir.to_path_buf(),
                DEFAULT_CF_NAME,
                &def_opts,
                target_sst_bytes,
                "default/part-00",
            );
            def_sink.put(ATTR_COUNTS_VERSION_KEY, &[1u8])?;
            if has_fts {
                def_sink.put(FTS_META_KEY, &FtsMeta::default().encode())?;
            }
            def_sink.put(MANIFEST_KEY, &manifest.encode())?;
            def_sink.put(SCHEMA_KEY, &schema.encode())?;
            cf_files.insert(DEFAULT_CF_NAME.to_owned(), def_sink.finish()?);
            return Ok((cf_files, manifest, None, 0, 0));
        }
    }

    // Emit CF_LEAF_ENTRIES in parallel if vectors are enabled.
    if let (Some(tree), Some(vcfg), Some(q)) = (built_tree, vec_cfg, quantizer) {
        let out = emit_ann_leaf_ssts(
            bundle_dir,
            tree,
            vcfg.dim,
            q,
            vectors_mmap,
            docs_idx_mmap,
            docs_shards,
            staged.has_detached_staging,
            schema,
            target_sst_bytes,
            pool,
        )?;
        if !out.leaf_ssts.is_empty() {
            cf_files.insert(CF_LEAF_ENTRIES.to_owned(), out.leaf_ssts);
        }
        if !out.summary_ssts.is_empty() {
            cf_files.insert(CF_NODE_SUMMARIES.to_owned(), out.summary_ssts);
        }
    }

    let num_segs = n_docs.div_ceil(seal_rows) as u32;
    let empty_gen_map = HashMap::new();
    let empty_existing_terms = HashMap::new();

    let mut seg_out = emit_segment_range_ssts(
        bundle_dir,
        stage_dir,
        staged,
        schema,
        seg_cfg,
        0,
        num_segs,
        has_vectors,
        vec_cfg.map_or(0, |v| v.dim),
        &empty_gen_map,
        &empty_existing_terms,
        0,
        true,
        false,
        vectors_mmap,
        doc_ids_mmap,
        docs_idx_mmap,
        docs_shards,
        target_sst_bytes,
        memory_budget_bytes,
        bit_sliced_index,
        pool,
    )?;

    for (cf, files) in seg_out.cf_files {
        cf_files.insert(cf, files);
    }

    let manifest = Manifest {
        open_segment: None,
        next_segment_id: num_segs,
        seal_rows: seg_cfg.seal_rows,
        live_rows: n_docs as u64,
        total_rows: n_docs as u64,
    };
    let ann_meta = built_tree.map(|t| t.meta.clone());

    // Build ordered default CF SST.
    let def_opts = cf_options_for(DEFAULT_CF_NAME, &block_cache)?;
    let mut def_sink = SstSink::new(
        bundle_dir.to_path_buf(),
        DEFAULT_CF_NAME,
        &def_opts,
        target_sst_bytes,
        "default/part-00",
    );

    if let Some(ref am) = ann_meta {
        def_sink.put(ANN_META_KEY, &am.encode())?;
    }
    def_sink.put(ATTR_COUNTS_VERSION_KEY, &[1u8])?;
    if has_fts {
        let fts_meta = FtsMeta::new(seg_out.next_fts_term_id, seg_out.fts_attr_stats.clone());
        def_sink.put(FTS_META_KEY, &fts_meta.encode())?;
    }
    def_sink.put(MANIFEST_KEY, &manifest.encode())?;
    def_sink.put(SCHEMA_KEY, &schema.encode())?;

    // 0x02 Internal nodes sorted by node_id ascending.
    if let (Some(tree), Some(vcfg)) = (built_tree, vec_cfg) {
        let mut internals: Vec<&(NodeId, PackedInternalNode)> =
            tree.internal_nodes.iter().collect();
        internals.sort_unstable_by_key(|&&(nid, _)| nid);
        for &(nid, ref node) in internals {
            let key = tagged_vec(INTERNAL_NODE_TAG, &node_key(nid));
            def_sink.put(&key, &node.encode(vcfg.dim))?;
        }
    }

    // 0x05 FTS doc length blocks sorted by (attr_id, seg, blk).
    seg_out.doc_len_blocks.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.seg().cmp(&b.1.seg()))
            .then_with(|| a.1.block().cmp(&b.1.block()))
    });
    for (attr_id, rblk, bytes) in seg_out.doc_len_blocks {
        let key = tagged_vec(FTS_DOC_LEN_TAG, &doc_len_key_coord(attr_id, rblk));
        def_sink.put(&key, &bytes)?;
    }

    // 0x06 Attribute counts sorted by cluster_key.
    for (cluster_key, count) in seg_out.attr_count_deltas {
        let key = tagged_vec(ATTR_COUNT_TAG, &cluster_key);
        def_sink.put(&key, &count.to_le_bytes())?;
    }

    // 0x07 Detached counts sorted by count_suffix.
    for (count_suffix, count) in seg_out.detached_count_deltas {
        let key = tagged_vec(DETACHED_COUNT_TAG, &count_suffix);
        def_sink.put(&key, &count.to_le_bytes())?;
    }

    // 0x11 SegmentMeta sorted by seg ascending.
    seg_out.seg_metas.sort_unstable_by_key(|&(seg, _)| seg);
    for (seg, meta) in seg_out.seg_metas {
        def_sink.put(&seg_meta_key(seg), &meta.encode())?;
    }

    // 0x12 Zones sorted by (seg, attr_id, tag).
    seg_out.zones.sort_unstable_by(|a, b| {
        a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2))
    });
    for (seg, attr_id, tag, zone) in seg_out.zones {
        def_sink.put(&zone_key(seg, attr_id, tag), &zone.encode())?;
    }

    cf_files.insert(DEFAULT_CF_NAME.to_owned(), def_sink.finish()?);

    Ok((
        cf_files,
        manifest,
        ann_meta,
        num_segs,
        seg_out.next_fts_term_id as u64,
    ))
}
