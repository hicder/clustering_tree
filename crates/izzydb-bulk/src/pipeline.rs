use std::collections::{BTreeMap, HashMap};
use std::io::{self, Read, Write};
use std::path::Path;

use rayon::prelude::*;
use roaring::RoaringBitmap;

use izzydb::codec::{
    ATTR_COUNT_TAG, DETACHED_COUNT_TAG, FTS_DOC_LEN_TAG, INTERNAL_NODE_TAG, Manifest,
    SEG_FLAG_HAS_COLUMNS, SEG_FLAG_HAS_VECTORS, SegPosting, SegmentMeta, SegmentState, TAG_BOOL,
    TAG_PRESENT, ZoneEntry, column_key, decode_row_entries, detached_count_suffix, detached_key,
    detached_posting_key, doc_id_block_key, doc_len_key_coord, encode_doc_id_block,
    encode_fts_term_key, extract_index_keys_for_attr, extract_index_keys_from_entries,
    fts_block_key_coord, fts_chunk_key_coord, make_posting_key, row_key, seg_meta_key, zone_key,
    RowView,
};
use izzydb::config::{
    AttrId, ColumnGen, DocId, LocalId, NodeId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentConfig,
    SegmentId, TermId, TreeConfig,
};
use izzydb::error::{Result, TreeError};
use izzydb::fts::{
    AttrFtsStats, DocLenBlock, FtsAnalyzer, FtsMeta, INLINE_TERM_MAX_DF, PackedPositionBlock,
    PackedPostingBlock, PostingBlockHeader, ReusableBuffer, SkipBlock, TermRecord, TermStats,
    dl_over_tf_lower_bound,
};
use izzydb::node::{AnnMeta, PackedInternalNode, PackedLeafNode};
use izzydb::rabitq::RabitqQuantizer;
use izzydb::schema::{
    AttributeType, AttributeValue, CollectionSchema, ColumnStorage, FullTextSearchConfig,
    SchemaEntry,
};
use izzydb::storage::{
    ANN_META_KEY, ATTR_COUNTS_VERSION_KEY, CF_ATTR_POSTINGS, CF_COLUMNS, CF_DETACHED,
    CF_DETACHED_POSTINGS, CF_FTS_POSITIONS, CF_FTS_POSTINGS, CF_FTS_SKIPS, CF_FTS_TERMS,
    CF_ID_MAP, CF_LEAF_ENTRIES, CF_ROW_DOC_IDS, CF_ROWS, CF_VECTORS, DEFAULT_CF_NAME,
    FTS_META_KEY, MANIFEST_KEY, SCHEMA_KEY, cf_options_for, default_sst_block_cache, doc_key,
    node_key, tagged_vec,
};
use izzydb_storage::column::ColumnBlockBuilder;

use crate::common::parquet_reader::{MappedFile, StagedData, decode_staged_doc, read_doc_id_at};
use crate::common::sorter::{
    ExternalSorter, KvItem, SortItem, read_bytes_lp, read_bytes_req, read_u16_opt, read_u32_req,
    write_bytes_lp, write_u16, write_u32, write_u64,
};
use crate::common::sst::{SstFileInfo, SstSink};
use crate::tree_builder::{BuiltTree, read_vector_at};

struct CountDeltaItem {
    key: Vec<u8>,
    count: u64,
}

impl SortItem for CountDeltaItem {
    #[inline]
    fn cmp_key(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.key.cmp(&b.key)
    }

    #[inline]
    fn estimated_bytes(&self) -> usize {
        self.key.len() + 24
    }

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        write_bytes_lp(w, &self.key)?;
        write_u64(w, self.count)
    }

    fn read_from(r: &mut impl Read) -> io::Result<Option<Self>> {
        let Some(key) = read_bytes_lp(r)? else {
            return Ok(None);
        };
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        Ok(Some(Self {
            key,
            count: u64::from_le_bytes(b),
        }))
    }
}

struct FtsPostingItem {
    attr_id: AttrId,
    token: String,
    seg: SegmentId,
    blk: u32,
    slot: u8,
    tf: u32,
    dl: u32,
    positions: Vec<u32>,
}

impl SortItem for FtsPostingItem {
    #[inline]
    fn cmp_key(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.attr_id
            .cmp(&b.attr_id)
            .then_with(|| a.token.as_bytes().cmp(b.token.as_bytes()))
            .then_with(|| a.seg.cmp(&b.seg))
            .then_with(|| a.blk.cmp(&b.blk))
            .then_with(|| a.slot.cmp(&b.slot))
    }

    #[inline]
    fn estimated_bytes(&self) -> usize {
        self.token.len() + self.positions.len() * 4 + 56
    }

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        write_u16(w, self.attr_id)?;
        write_bytes_lp(w, self.token.as_bytes())?;
        write_u32(w, self.seg)?;
        write_u32(w, self.blk)?;
        w.write_all(&[self.slot])?;
        write_u32(w, self.tf)?;
        write_u32(w, self.dl)?;
        write_u32(w, self.positions.len() as u32)?;
        for &p in &self.positions {
            write_u32(w, p)?;
        }
        Ok(())
    }

    fn read_from(r: &mut impl Read) -> io::Result<Option<Self>> {
        let Some(attr_id) = read_u16_opt(r)? else {
            return Ok(None);
        };
        let tok_bytes = read_bytes_req(r)?;
        let token = String::from_utf8(tok_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let seg = read_u32_req(r)?;
        let blk = read_u32_req(r)?;
        let mut s = [0u8; 1];
        r.read_exact(&mut s)?;
        let slot = s[0];
        let tf = read_u32_req(r)?;
        let dl = read_u32_req(r)?;
        let pos_len = read_u32_req(r)? as usize;
        let mut positions = Vec::with_capacity(pos_len);
        for _ in 0..pos_len {
            positions.push(read_u32_req(r)?);
        }
        Ok(Some(Self {
            attr_id,
            token,
            seg,
            blk,
            slot,
            tf,
            dl,
            positions,
        }))
    }
}

pub struct SegmentPassOutputs {
    pub cf_files: BTreeMap<String, Vec<SstFileInfo>>,
    pub seg_metas: Vec<(SegmentId, SegmentMeta)>,
    pub zones: Vec<(SegmentId, AttrId, u8, ZoneEntry)>,
    pub doc_len_blocks: Vec<(AttrId, RowBlock, Vec<u8>)>,
    pub attr_count_deltas: Vec<(Vec<u8>, u64)>,
    pub detached_count_deltas: Vec<(Vec<u8>, u64)>,
    pub fts_attr_stats: BTreeMap<AttrId, AttrFtsStats>,
    pub new_fts_terms: Vec<(Vec<u8>, Vec<u8>)>,
    pub updated_fts_terms: Vec<FtsTermAppendDelta>,
    pub next_fts_term_id: u32,
}

pub struct FtsTermAppendDelta {
    pub term_key: Vec<u8>,
    pub add_doc_freq: u64,
    pub max_tf: u32,
    pub min_dl_over_tf: f32,
}

#[derive(Default)]
struct WorkerSegAccum {
    id_map_ssts: Vec<SstFileInfo>,
    rows_ssts: Vec<SstFileInfo>,
    vecs_ssts: Vec<SstFileInfo>,
    doc_ids_ssts: Vec<SstFileInfo>,
    cols_ssts: Vec<SstFileInfo>,
    seg_metas: Vec<(SegmentId, SegmentMeta)>,
    zones: Vec<(SegmentId, AttrId, u8, ZoneEntry)>,
    doc_len_blocks: Vec<(AttrId, RowBlock, Vec<u8>)>,
    fts_stats: Vec<AttrFtsStats>,
}

#[inline]
fn read_doc_bytes<'a>(docs_idx: &[u8], docs_shards: &'a [MappedFile], row: u32) -> &'a [u8] {
    let off = (row as usize) * crate::common::parquet_reader::DocIndexEntry::SIZE;
    let entry = crate::common::parquet_reader::DocIndexEntry::decode(
        &docs_idx[off..off + crate::common::parquet_reader::DocIndexEntry::SIZE],
    );
    if entry.len == 0 {
        return &[];
    }
    let shard = docs_shards[entry.shard as usize].as_bytes();
    &shard[entry.offset as usize..entry.offset as usize + entry.len as usize]
}

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
    pool: &rayon::ThreadPool,
) -> Result<(BTreeMap<String, Vec<SstFileInfo>>, Manifest, Option<AnnMeta>, u32, u64)> {
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
        let leaf_ssts = emit_ann_leaf_ssts(
            bundle_dir,
            tree,
            vcfg.dim,
            q,
            vectors_mmap,
            target_sst_bytes,
            pool,
        )?;
        cf_files.insert(CF_LEAF_ENTRIES.to_owned(), leaf_ssts);
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

fn emit_ann_leaf_ssts(
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

/// Emit per-segment immutable SSTs for segments `[start_seg .. start_seg + num_segs)`.
#[allow(clippy::too_many_arguments)]
pub fn emit_segment_range_ssts(
    bundle_dir: &Path,
    stage_dir: &Path,
    staged: &StagedData,
    schema: &CollectionSchema,
    seg_cfg: &SegmentConfig,
    start_seg: SegmentId,
    num_segs: u32,
    has_vectors: bool,
    dim: usize,
    detached_gens: &HashMap<AttrId, ColumnGen>,
    existing_fts_terms: &HashMap<(AttrId, String), TermRecord>,
    fts_next_term_id_init: u32,
    emit_id_map_sst: bool,
    is_append_mode: bool,
    vectors_mmap: &[u8],
    doc_ids_mmap: &[u8],
    docs_idx_mmap: &[u8],
    docs_shards: &[MappedFile],
    target_sst_bytes: u64,
    memory_budget_bytes: usize,
    pool: &rayon::ThreadPool,
) -> Result<SegmentPassOutputs> {
    let block_cache = default_sst_block_cache();
    let n_docs = staged.docs_imported as usize;
    let seal_rows = seg_cfg.seal_rows as usize;
    let rows_by_doc_id_mmap = MappedFile::open(&staged.rows_by_doc_id_path)?;

    let has_attr_index = schema.entries().iter().any(|e| {
        e.storage == ColumnStorage::Inline && e.filterable
    });
    let has_detached_filter = schema.entries().iter().any(|e| {
        e.storage == ColumnStorage::Detached && e.filterable
    });
    let has_fts = schema.has_fts();

    let mut col_attrs: Vec<(AttrId, AttributeType)> = if seg_cfg.defer_column_build {
        Vec::new()
    } else {
        schema
            .entries()
            .iter()
            .filter(|e| e.storage == ColumnStorage::Inline && e.is_columnar())
            .map(|e| (e.id, e.attr_type))
            .collect()
    };
    col_attrs.sort_unstable_by_key(|&(id, _)| id);

    let mut detached_entries: Vec<SchemaEntry> = schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
        .cloned()
        .collect();
    detached_entries.sort_unstable_by_key(|e| e.id);

    let mut fts_attrs: Vec<(AttrId, &FullTextSearchConfig)> = schema
        .fts_entries()
        .filter(|e| e.storage == ColumnStorage::Inline)
        .map(|e| (e.id, e.full_text_search.as_ref().unwrap()))
        .collect();
    fts_attrs.sort_unstable_by_key(|&(id, _)| id);

    let num_workers = pool.current_num_threads().min(num_segs as usize).max(1);
    let segs_per_worker = (num_segs as usize).div_ceil(num_workers);
    let seg_ranges: Vec<(u32, u32)> = (0..num_segs)
        .step_by(segs_per_worker)
        .map(|s| {
            let s_u32 = s as u32;
            let e_u32 = (s as u32 + segs_per_worker as u32).min(num_segs);
            (s_u32, e_u32)
        })
        .collect();

    let mut active_sorters = 0usize;
    if has_attr_index {
        active_sorters += 2;
    }
    if staged.has_detached_staging && !detached_entries.is_empty() {
        active_sorters += 1;
        if has_detached_filter {
            active_sorters += 2;
        }
    }
    if has_fts {
        active_sorters += 1;
    }
    let per_sorter_budget = memory_budget_bytes
        .checked_div(active_sorters)
        .map_or(memory_budget_bytes, |b| b.max(1024 * 1024));

    let attr_postings_sorter = if has_attr_index {
        Some(ExternalSorter::<KvItem>::new(
            "attr_postings",
            stage_dir.join("runs_attr_postings"),
            per_sorter_budget,
            seg_ranges.len(),
        )?)
    } else {
        None
    };
    let attr_counts_sorter = if has_attr_index {
        Some(ExternalSorter::<CountDeltaItem>::new(
            "attr_counts",
            stage_dir.join("runs_attr_counts"),
            per_sorter_budget,
            seg_ranges.len(),
        )?)
    } else {
        None
    };

    let det_blocks_sorter = if staged.has_detached_staging && !detached_entries.is_empty() {
        Some(ExternalSorter::<KvItem>::new(
            "det_blocks",
            stage_dir.join("runs_det_blocks"),
            per_sorter_budget,
            seg_ranges.len(),
        )?)
    } else {
        None
    };
    let det_postings_sorter = if staged.has_detached_staging && has_detached_filter {
        Some(ExternalSorter::<KvItem>::new(
            "det_postings",
            stage_dir.join("runs_det_postings"),
            per_sorter_budget,
            seg_ranges.len(),
        )?)
    } else {
        None
    };
    let det_counts_sorter = if staged.has_detached_staging && has_detached_filter {
        Some(ExternalSorter::<CountDeltaItem>::new(
            "det_counts",
            stage_dir.join("runs_det_counts"),
            per_sorter_budget,
            seg_ranges.len(),
        )?)
    } else {
        None
    };

    let fts_sorter = if has_fts && !fts_attrs.is_empty() {
        Some(ExternalSorter::<FtsPostingItem>::new(
            "fts_postings",
            stage_dir.join("runs_fts_postings"),
            per_sorter_budget,
            seg_ranges.len(),
        )?)
    } else {
        None
    };

    let id_map_opts = cf_options_for(CF_ID_MAP, &block_cache)?;
    let rows_opts = cf_options_for(CF_ROWS, &block_cache)?;
    let vecs_opts = cf_options_for(CF_VECTORS, &block_cache)?;
    let doc_ids_opts = cf_options_for(CF_ROW_DOC_IDS, &block_cache)?;
    let cols_opts = cf_options_for(CF_COLUMNS, &block_cache)?;

    let worker_results: Vec<WorkerSegAccum> = pool.install(|| -> Result<Vec<_>> {
        seg_ranges
            .par_iter()
            .enumerate()
            .map(|(w_idx, &(rel_s, rel_e))| {
                let mut accum = WorkerSegAccum {
                    fts_stats: vec![AttrFtsStats::default(); fts_attrs.len()],
                    ..Default::default()
                };

                let mut id_map_sink = if emit_id_map_sst {
                    Some(SstSink::new(
                        bundle_dir.to_path_buf(),
                        CF_ID_MAP,
                        &id_map_opts,
                        target_sst_bytes,
                        format!("{CF_ID_MAP}/part-{w_idx:02}"),
                    ))
                } else {
                    None
                };
                let mut rows_sink = SstSink::new(
                    bundle_dir.to_path_buf(),
                    CF_ROWS,
                    &rows_opts,
                    target_sst_bytes,
                    format!("{CF_ROWS}/part-{w_idx:02}"),
                );
                let mut vecs_sink = if has_vectors {
                    Some(SstSink::new(
                        bundle_dir.to_path_buf(),
                        CF_VECTORS,
                        &vecs_opts,
                        target_sst_bytes,
                        format!("{CF_VECTORS}/part-{w_idx:02}"),
                    ))
                } else {
                    None
                };
                let mut doc_ids_sink = SstSink::new(
                    bundle_dir.to_path_buf(),
                    CF_ROW_DOC_IDS,
                    &doc_ids_opts,
                    target_sst_bytes,
                    format!("{CF_ROW_DOC_IDS}/part-{w_idx:02}"),
                );
                let mut cols_sink = if !col_attrs.is_empty() {
                    Some(SstSink::new(
                        bundle_dir.to_path_buf(),
                        CF_COLUMNS,
                        &cols_opts,
                        target_sst_bytes,
                        format!("{CF_COLUMNS}/part-{w_idx:02}"),
                    ))
                } else {
                    None
                };

                let mut attr_p_shard = attr_postings_sorter.as_ref().map(|s| s.shard());
                let mut attr_c_shard = attr_counts_sorter.as_ref().map(|s| s.shard());
                let mut det_b_shard = det_blocks_sorter.as_ref().map(|s| s.shard());
                let mut det_p_shard = det_postings_sorter.as_ref().map(|s| s.shard());
                let mut det_c_shard = det_counts_sorter.as_ref().map(|s| s.shard());
                let mut fts_shard = fts_sorter.as_ref().map(|s| s.shard());

                let col_attr_slot: HashMap<AttrId, usize> = col_attrs
                    .iter()
                    .enumerate()
                    .map(|(i, &(id, _))| (id, i))
                    .collect();
                let det_attr_slot: HashMap<AttrId, usize> = detached_entries
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (e.id, i))
                    .collect();
                let fts_attr_slot: HashMap<AttrId, usize> = fts_attrs
                    .iter()
                    .enumerate()
                    .map(|(i, &(id, _))| (id, i))
                    .collect();
                let fts_analyzers: Vec<FtsAnalyzer> = fts_attrs
                    .iter()
                    .map(|&(_, cfg)| FtsAnalyzer::new(cfg.clone()))
                    .collect();
                let mut fts_buf = ReusableBuffer::new();

                let rows_bytes = rows_by_doc_id_mmap.as_bytes();
                let seg_flags = (if has_vectors { SEG_FLAG_HAS_VECTORS } else { 0 })
                    | (if !seg_cfg.defer_column_build {
                        SEG_FLAG_HAS_COLUMNS
                    } else {
                        0
                    });
                let vec_byte_len = dim * 4;

                for rel_seg in rel_s..rel_e {
                    let seg = start_seg + rel_seg;
                    let ord_start = (rel_seg as usize) * seal_rows;
                    let ord_end = ((rel_seg as usize + 1) * seal_rows).min(n_docs);
                    let seg_rows = ord_end - ord_start;

                    let mut seg_inline_postings: BTreeMap<Vec<u8>, RoaringBitmap> = BTreeMap::new();
                    let mut seg_det_postings: Vec<BTreeMap<Vec<u8>, RoaringBitmap>> =
                        vec![BTreeMap::new(); detached_entries.len()];
                    let mut seg_zones: HashMap<(AttrId, u8), ZoneEntry> = HashMap::new();

                    let mut seg_col_blocks: Vec<Vec<(u32, Vec<u8>)>> =
                        vec![Vec::new(); col_attrs.len()];

                    let mut cur_blk = u32::MAX;
                    let mut doc_id_blk = [0u64; ROW_BLOCK_SIZE];
                    let mut col_builders: Vec<ColumnBlockBuilder> = col_attrs
                        .iter()
                        .map(|&(_, ty)| ColumnBlockBuilder::new(ty))
                        .collect();
                    let mut det_builders: Vec<ColumnBlockBuilder> = detached_entries
                        .iter()
                        .map(|e| ColumnBlockBuilder::new(e.attr_type))
                        .collect();
                    let mut dl_arrs: Vec<[u32; ROW_BLOCK_SIZE]> =
                        vec![[0u32; ROW_BLOCK_SIZE]; fts_attrs.len()];
                    let mut blk_row_count = 0usize;

                    let flush_block = |blk: u32,
                                       row_cnt: usize,
                                       doc_id_blk: &mut [DocId; ROW_BLOCK_SIZE],
                                       col_builders: &mut Vec<ColumnBlockBuilder>,
                                       det_builders: &mut Vec<ColumnBlockBuilder>,
                                       dl_arrs: &mut Vec<[u32; ROW_BLOCK_SIZE]>,
                                       seg_col_blocks: &mut [Vec<(u32, Vec<u8>)>],
                                       doc_ids_sink: &mut SstSink<'_>,
                                       det_b_shard: Option<
                        &mut crate::common::sorter::SorterShard<KvItem>,
                    >,
                                       doc_len_blocks: &mut Vec<(AttrId, RowBlock, Vec<u8>)>|
                     -> Result<()> {
                        if blk == u32::MAX {
                            return Ok(());
                        }
                        doc_ids_sink.put(
                            &doc_id_block_key(seg, blk),
                            &encode_doc_id_block(doc_id_blk),
                        )?;
                        doc_id_blk.fill(0);

                        for (idx, &(_, ty)) in col_attrs.iter().enumerate() {
                            let b = std::mem::replace(
                                &mut col_builders[idx],
                                ColumnBlockBuilder::new(ty),
                            );
                            if b.present_count() > 0 {
                                seg_col_blocks[idx].push((blk, b.encode()));
                            }
                        }

                        if let Some(ds) = det_b_shard {
                            for (idx, entry) in detached_entries.iter().enumerate() {
                                let b = std::mem::replace(
                                    &mut det_builders[idx],
                                    ColumnBlockBuilder::new(entry.attr_type),
                                );
                                if b.present_count() > 0 {
                                    let col_gen =
                                        detached_gens.get(&entry.id).copied().unwrap_or(0);
                                    ds.push(KvItem {
                                        key: detached_key(entry.id, col_gen, seg, blk).to_vec(),
                                        val: b.encode(),
                                    })?;
                                }
                            }
                        }

                        for (f_idx, &(aid, _)) in fts_attrs.iter().enumerate() {
                            let dl_blk = DocLenBlock::from_dls(&dl_arrs[f_idx][..row_cnt]);
                            if !dl_blk.is_empty() {
                                doc_len_blocks.push((
                                    aid,
                                    RowBlock::new(seg, blk),
                                    dl_blk.encode(),
                                ));
                            }
                            dl_arrs[f_idx].fill(0);
                        }
                        Ok(())
                    };

                    for (local_usize, ord) in (ord_start..ord_end).enumerate() {
                        let local = local_usize as LocalId;
                        let blk = local >> 8;
                        let slot = (local & 0xFF) as u8;
                        if cur_blk != blk {
                            flush_block(
                                cur_blk,
                                blk_row_count,
                                &mut doc_id_blk,
                                &mut col_builders,
                                &mut det_builders,
                                &mut dl_arrs,
                                &mut seg_col_blocks,
                                &mut doc_ids_sink,
                                det_b_shard.as_mut(),
                                &mut accum.doc_len_blocks,
                            )?;
                            cur_blk = blk;
                            blk_row_count = 0;
                        }
                        blk_row_count += 1;

                        let staged_row = u32::from_le_bytes(
                            rows_bytes[ord * 4..(ord + 1) * 4].try_into().unwrap(),
                        );
                        let doc_id = read_doc_id_at(doc_ids_mmap, staged_row);
                        let row_id = RowId::new(seg, local);

                        doc_id_blk[slot as usize] = doc_id;
                        if let Some(ref mut sink) = id_map_sink {
                            sink.put(&doc_key(doc_id), &row_key(row_id))?;
                        }

                        let staged_bytes = read_doc_bytes(docs_idx_mmap, docs_shards, staged_row);
                        let (inline_bytes, det_pairs) =
                            decode_staged_doc(staged_bytes, staged.has_detached_staging)?;
                        rows_sink.put(&row_key(row_id), inline_bytes)?;

                        if let Some(ref mut sink) = vecs_sink {
                            let off = (staged_row as usize) * vec_byte_len;
                            sink.put(&row_key(row_id), &vectors_mmap[off..off + vec_byte_len])?;
                        }

                        let view = RowView::parse(inline_bytes)?;
                        if !col_attrs.is_empty() {
                            for i in 0..view.len() {
                                let (aid, _, _, _) = view.dir_entry(i);
                                if let Some(&pos) = col_attr_slot.get(&aid)
                                    && let Some(field) = view.get_raw_field(aid)?
                                {
                                    col_builders[pos].set_row_field(slot, field)?;
                                }
                            }
                        }

                        if has_attr_index || !fts_attrs.is_empty() {
                            let entries = decode_row_entries(inline_bytes)?;
                            if has_attr_index {
                                for ak in extract_index_keys_from_entries(schema, &entries)? {
                                    let aid = u16::from_be_bytes([ak[0], ak[1]]);
                                    let tag = ak[2];
                                    if ak.len() > 3
                                        && tag != TAG_PRESENT
                                        && tag != TAG_BOOL
                                        && schema.get_by_id(aid).is_some_and(|e| e.attr_type.is_ordered())
                                    {
                                        let z = seg_zones
                                            .entry((aid, tag))
                                            .or_insert_with(|| ZoneEntry::singleton(&ak[3..]));
                                        z.widen(&ak[3..]);
                                    }
                                    seg_inline_postings.entry(ak).or_default().insert(local);
                                }
                            }

                            if let Some(ref mut fs) = fts_shard {
                                for (aid, val) in &entries {
                                    let Some(&f_idx) = fts_attr_slot.get(aid) else {
                                        continue;
                                    };
                                    if matches!(val, AttributeValue::Null) {
                                        continue;
                                    }
                                    if let Some(analyzed) = fts_analyzers[f_idx]
                                        .analyze_attribute_value(&val, &mut fts_buf)?
                                    {
                                        accum.fts_stats[f_idx].doc_count += 1;
                                        accum.fts_stats[f_idx].total_tokens +=
                                            analyzed.doc_len as u64;
                                        if analyzed.doc_len > 0 {
                                            dl_arrs[f_idx][slot as usize] = analyzed.doc_len;
                                            for (token, term) in analyzed.terms {
                                                fs.push(FtsPostingItem {
                                                    attr_id: *aid,
                                                    token,
                                                    seg,
                                                    blk,
                                                    slot,
                                                    tf: term.tf,
                                                    dl: analyzed.doc_len,
                                                    positions: term.positions,
                                                })?;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        for (aid, val) in det_pairs {
                            let Some(&di) = det_attr_slot.get(&aid) else {
                                continue;
                            };
                            det_builders[di].set_attr_value(slot, &val)?;
                            let entry = &detached_entries[di];
                            if entry.filterable {
                                for ak in extract_index_keys_for_attr(entry.id, &val)? {
                                    let tag = ak[2];
                                    if ak.len() > 3
                                        && tag != TAG_PRESENT
                                        && tag != TAG_BOOL
                                        && entry.attr_type.is_ordered()
                                    {
                                        let z = seg_zones
                                            .entry((entry.id, tag))
                                            .or_insert_with(|| ZoneEntry::singleton(&ak[3..]));
                                        z.widen(&ak[3..]);
                                    }
                                    seg_det_postings[di].entry(ak).or_default().insert(local);
                                }
                            }
                        }
                    }

                    flush_block(
                        cur_blk,
                        blk_row_count,
                        &mut doc_id_blk,
                        &mut col_builders,
                        &mut det_builders,
                        &mut dl_arrs,
                        &mut seg_col_blocks,
                        &mut doc_ids_sink,
                        det_b_shard.as_mut(),
                        &mut accum.doc_len_blocks,
                    )?;

                    if let Some(ref mut sink) = cols_sink {
                        for (c_idx, &(aid, _)) in col_attrs.iter().enumerate() {
                            for (blk, bytes) in &seg_col_blocks[c_idx] {
                                sink.put(&column_key(seg, aid, *blk), bytes)?;
                            }
                        }
                    }

                    if let (Some(ps), Some(cs)) = (attr_p_shard.as_mut(), attr_c_shard.as_mut()) {
                        for (ak, bm) in seg_inline_postings {
                            let cnt = bm.len();
                            ps.push(KvItem {
                                key: make_posting_key(&ak, seg),
                                val: SegPosting::from_bitmap(bm).encode()?,
                            })?;
                            cs.push(CountDeltaItem {
                                key: ak,
                                count: cnt,
                            })?;
                        }
                    }

                    if let (Some(dps), Some(dcs)) = (det_p_shard.as_mut(), det_c_shard.as_mut()) {
                        for (di, pmap) in seg_det_postings.into_iter().enumerate() {
                            let entry = &detached_entries[di];
                            let col_gen = detached_gens.get(&entry.id).copied().unwrap_or(0);
                            for (ak, bm) in pmap {
                                let cnt = bm.len();
                                dps.push(KvItem {
                                    key: detached_posting_key(
                                        entry.id,
                                        col_gen,
                                        ak[2],
                                        &ak[3..],
                                        seg,
                                    ),
                                    val: SegPosting::from_bitmap(bm).encode()?,
                                })?;
                                dcs.push(CountDeltaItem {
                                    key: detached_count_suffix(col_gen, &ak),
                                    count: cnt,
                                })?;
                            }
                        }
                    }

                    for ((aid, tag), z) in seg_zones {
                        accum.zones.push((seg, aid, tag, z));
                    }

                    accum.seg_metas.push((
                        seg,
                        SegmentMeta {
                            state: SegmentState::Sealed,
                            flags: seg_flags,
                            row_count: seg_rows as u32,
                            deleted_count: 0,
                            forward_target: 0,
                            forward_base: 0,
                            forwarded_at_epoch: 0,
                            created_seq: 1,
                            sealed_seq: 1,
                        },
                    ));
                }

                if let Some(s) = attr_p_shard {
                    s.finish()?;
                }
                if let Some(s) = attr_c_shard {
                    s.finish()?;
                }
                if let Some(s) = det_b_shard {
                    s.finish()?;
                }
                if let Some(s) = det_p_shard {
                    s.finish()?;
                }
                if let Some(s) = det_c_shard {
                    s.finish()?;
                }
                if let Some(s) = fts_shard {
                    s.finish()?;
                }

                if let Some(sink) = id_map_sink {
                    accum.id_map_ssts = sink.finish()?;
                }
                accum.rows_ssts = rows_sink.finish()?;
                if let Some(sink) = vecs_sink {
                    accum.vecs_ssts = sink.finish()?;
                }
                accum.doc_ids_ssts = doc_ids_sink.finish()?;
                if let Some(sink) = cols_sink {
                    accum.cols_ssts = sink.finish()?;
                }
                Ok(accum)
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let mut cf_files: BTreeMap<String, Vec<SstFileInfo>> = BTreeMap::new();
    let mut seg_metas = Vec::with_capacity(num_segs as usize);
    let mut zones = Vec::new();
    let mut doc_len_blocks = Vec::new();
    let mut merged_fts_stats: BTreeMap<AttrId, AttrFtsStats> = BTreeMap::new();

    for w in worker_results {
        if !w.id_map_ssts.is_empty() {
            cf_files
                .entry(CF_ID_MAP.to_owned())
                .or_default()
                .extend(w.id_map_ssts);
        }
        if !w.rows_ssts.is_empty() {
            cf_files
                .entry(CF_ROWS.to_owned())
                .or_default()
                .extend(w.rows_ssts);
        }
        if !w.vecs_ssts.is_empty() {
            cf_files
                .entry(CF_VECTORS.to_owned())
                .or_default()
                .extend(w.vecs_ssts);
        }
        if !w.doc_ids_ssts.is_empty() {
            cf_files
                .entry(CF_ROW_DOC_IDS.to_owned())
                .or_default()
                .extend(w.doc_ids_ssts);
        }
        if !w.cols_ssts.is_empty() {
            cf_files
                .entry(CF_COLUMNS.to_owned())
                .or_default()
                .extend(w.cols_ssts);
        }
        seg_metas.extend(w.seg_metas);
        zones.extend(w.zones);
        doc_len_blocks.extend(w.doc_len_blocks);
        for (i, st) in w.fts_stats.into_iter().enumerate() {
            let entry = merged_fts_stats.entry(fts_attrs[i].0).or_default();
            entry.doc_count += st.doc_count;
            entry.total_tokens += st.total_tokens;
        }
    }

    if let Some(sorter) = attr_postings_sorter {
        let files = emit_sorted_kv_sst(bundle_dir, CF_ATTR_POSTINGS, target_sst_bytes, sorter)?;
        if !files.is_empty() {
            cf_files.insert(CF_ATTR_POSTINGS.to_owned(), files);
        }
    }
    let attr_count_deltas = if let Some(sorter) = attr_counts_sorter {
        collect_summed_counts(sorter)?
    } else {
        Vec::new()
    };

    if let Some(sorter) = det_blocks_sorter {
        let files = emit_sorted_kv_sst(bundle_dir, CF_DETACHED, target_sst_bytes, sorter)?;
        if !files.is_empty() {
            cf_files.insert(CF_DETACHED.to_owned(), files);
        }
    }
    if let Some(sorter) = det_postings_sorter {
        let files = emit_sorted_kv_sst(bundle_dir, CF_DETACHED_POSTINGS, target_sst_bytes, sorter)?;
        if !files.is_empty() {
            cf_files.insert(CF_DETACHED_POSTINGS.to_owned(), files);
        }
    }
    let detached_count_deltas = if let Some(sorter) = det_counts_sorter {
        collect_summed_counts(sorter)?
    } else {
        Vec::new()
    };

    let (new_fts_terms, updated_fts_terms, next_fts_term_id) = if let Some(sorter) = fts_sorter {
        let attr_lookup: HashMap<AttrId, bool> = fts_attrs
            .iter()
            .map(|&(id, cfg)| (id, cfg.index_positions))
            .collect();
        let mut iter = sorter.into_sorted_iter()?;

        if is_append_mode {
            let skips_sorter = ExternalSorter::<KvItem>::new(
                "fts_skips_append",
                stage_dir.join("runs_fts_skips"),
                memory_budget_bytes / 4,
                1,
            )?;
            let postings_sorter = ExternalSorter::<KvItem>::new(
                "fts_postings_append",
                stage_dir.join("runs_fts_postings_kv"),
                memory_budget_bytes / 4,
                1,
            )?;
            let positions_sorter = ExternalSorter::<KvItem>::new(
                "fts_positions_append",
                stage_dir.join("runs_fts_positions"),
                memory_budget_bytes / 4,
                1,
            )?;
            let mut skips_shard = skips_sorter.shard();
            let mut postings_shard = postings_sorter.shard();
            let mut positions_shard = positions_sorter.shard();
            let mut new_terms = Vec::new();
            let mut upd_terms = Vec::new();

            let next_tid = emit_fts_streams_inner(
                &mut iter,
                &attr_lookup,
                existing_fts_terms,
                fts_next_term_id_init,
                is_append_mode,
                &mut |k, v| {
                    new_terms.push((k.to_vec(), v.to_vec()));
                    Ok(())
                },
                &mut |d| {
                    upd_terms.push(d);
                    Ok(())
                },
                &mut |k, v| {
                    skips_shard.push(KvItem {
                        key: k.to_vec(),
                        val: v.to_vec(),
                    })
                },
                &mut |k, v| {
                    postings_shard.push(KvItem {
                        key: k.to_vec(),
                        val: v.to_vec(),
                    })
                },
                &mut |k, v| {
                    positions_shard.push(KvItem {
                        key: k.to_vec(),
                        val: v.to_vec(),
                    })
                },
            )?;
            skips_shard.finish()?;
            postings_shard.finish()?;
            positions_shard.finish()?;

            let skips_files =
                emit_sorted_kv_sst(bundle_dir, CF_FTS_SKIPS, target_sst_bytes, skips_sorter)?;
            let postings_files =
                emit_sorted_kv_sst(bundle_dir, CF_FTS_POSTINGS, target_sst_bytes, postings_sorter)?;
            let positions_files =
                emit_sorted_kv_sst(bundle_dir, CF_FTS_POSITIONS, target_sst_bytes, positions_sorter)?;
            if !skips_files.is_empty() {
                cf_files.insert(CF_FTS_SKIPS.to_owned(), skips_files);
            }
            if !postings_files.is_empty() {
                cf_files.insert(CF_FTS_POSTINGS.to_owned(), postings_files);
            }
            if !positions_files.is_empty() {
                cf_files.insert(CF_FTS_POSITIONS.to_owned(), positions_files);
            }
            (new_terms, upd_terms, next_tid)
        } else {
            let terms_opts = cf_options_for(CF_FTS_TERMS, &block_cache)?;
            let skips_opts = cf_options_for(CF_FTS_SKIPS, &block_cache)?;
            let postings_opts = cf_options_for(CF_FTS_POSTINGS, &block_cache)?;
            let positions_opts = cf_options_for(CF_FTS_POSITIONS, &block_cache)?;

            let mut terms_sink = SstSink::new(
                bundle_dir.to_path_buf(),
                CF_FTS_TERMS,
                &terms_opts,
                target_sst_bytes,
                format!("{CF_FTS_TERMS}/part-00"),
            );
            let mut skips_sink = SstSink::new(
                bundle_dir.to_path_buf(),
                CF_FTS_SKIPS,
                &skips_opts,
                target_sst_bytes,
                format!("{CF_FTS_SKIPS}/part-00"),
            );
            let mut postings_sink = SstSink::new(
                bundle_dir.to_path_buf(),
                CF_FTS_POSTINGS,
                &postings_opts,
                target_sst_bytes,
                format!("{CF_FTS_POSTINGS}/part-00"),
            );
            let mut positions_sink = SstSink::new(
                bundle_dir.to_path_buf(),
                CF_FTS_POSITIONS,
                &positions_opts,
                target_sst_bytes,
                format!("{CF_FTS_POSITIONS}/part-00"),
            );

            let next_tid = emit_fts_streams_inner(
                &mut iter,
                &attr_lookup,
                existing_fts_terms,
                fts_next_term_id_init,
                false,
                &mut |k, v| terms_sink.put(k, v),
                &mut |_| Ok(()),
                &mut |k, v| skips_sink.put(k, v),
                &mut |k, v| postings_sink.put(k, v),
                &mut |k, v| positions_sink.put(k, v),
            )?;

            let tf = terms_sink.finish()?;
            let sf = skips_sink.finish()?;
            let pf = postings_sink.finish()?;
            let posf = positions_sink.finish()?;
            if !tf.is_empty() {
                cf_files.insert(CF_FTS_TERMS.to_owned(), tf);
            }
            if !sf.is_empty() {
                cf_files.insert(CF_FTS_SKIPS.to_owned(), sf);
            }
            if !pf.is_empty() {
                cf_files.insert(CF_FTS_POSTINGS.to_owned(), pf);
            }
            if !posf.is_empty() {
                cf_files.insert(CF_FTS_POSITIONS.to_owned(), posf);
            }
            (Vec::new(), Vec::new(), next_tid)
        }
    } else {
        (Vec::new(), Vec::new(), fts_next_term_id_init)
    };

    Ok(SegmentPassOutputs {
        cf_files,
        seg_metas,
        zones,
        doc_len_blocks,
        attr_count_deltas,
        detached_count_deltas,
        fts_attr_stats: merged_fts_stats,
        new_fts_terms,
        updated_fts_terms,
        next_fts_term_id,
    })
}

fn emit_sorted_kv_sst(
    bundle_dir: &Path,
    cf_name: &str,
    target_sst_bytes: u64,
    sorter: ExternalSorter<KvItem>,
) -> Result<Vec<SstFileInfo>> {
    let block_cache = default_sst_block_cache();
    let opts = cf_options_for(cf_name, &block_cache)?;
    let mut sink = SstSink::new(
        bundle_dir.to_path_buf(),
        cf_name,
        &opts,
        target_sst_bytes,
        format!("{cf_name}/part-00"),
    );
    let mut iter = sorter.into_sorted_iter()?;
    while let Some(item) = iter.next_item()? {
        sink.put(&item.key, &item.val)?;
    }
    sink.finish()
}

fn collect_summed_counts(sorter: ExternalSorter<CountDeltaItem>) -> Result<Vec<(Vec<u8>, u64)>> {
    let mut iter = sorter.into_sorted_iter()?;
    let mut out = Vec::new();
    let mut cur_key: Option<Vec<u8>> = None;
    let mut cur_sum: u64 = 0;
    while let Some(item) = iter.next_item()? {
        match &cur_key {
            Some(k) if k == &item.key => cur_sum += item.count,
            _ => {
                if let Some(prev) = cur_key.take() {
                    out.push((prev, cur_sum));
                }
                cur_key = Some(item.key);
                cur_sum = item.count;
            }
        }
    }
    if let Some(prev) = cur_key.take() {
        out.push((prev, cur_sum));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn emit_fts_streams_inner(
    iter: &mut crate::common::sorter::SortedIter<FtsPostingItem>,
    attr_lookup: &HashMap<AttrId, bool>,
    existing_terms: &HashMap<(AttrId, String), TermRecord>,
    next_term_id_init: u32,
    is_append_mode: bool,
    emit_new_term: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
    emit_updated_term: &mut dyn FnMut(FtsTermAppendDelta) -> Result<()>,
    emit_skip: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
    emit_posting: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
    emit_position: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
) -> Result<u32> {
    let mut next_term_id = next_term_id_init;

    let mut has_term = false;
    let mut cur_attr: AttrId = 0;
    let mut cur_token = String::new();
    let mut cur_term_id: TermId = 0;
    let mut cur_is_existing = false;
    let mut cur_index_pos = false;
    let mut cur_stats = TermStats::default();

    let mut cur_rblk: Option<RowBlock> = None;
    let mut blocks_flushed_for_term: u32 = 0;
    let mut block_tfs: Vec<(u8, u32)> = Vec::new();
    let mut block_dls = [0u32; ROW_BLOCK_SIZE];
    let mut block_pos: Vec<(u8, Vec<u32>)> = Vec::new();

    let mut deferred_first_block: Option<(RowBlock, Vec<u8>, u32, f32)> = None;
    let mut cur_skip_chunk: Option<izzydb::config::SkipChunk> = None;
    let mut cur_skip_bounds: Vec<(u8, u32, f32)> = Vec::new();

    let flush_skip_chunk = |term_id: TermId,
                            chunk_opt: &mut Option<izzydb::config::SkipChunk>,
                            bounds: &mut Vec<(u8, u32, f32)>,
                            emit_skip: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>|
     -> Result<()> {
        if let Some(chunk) = chunk_opt.take()
            && !bounds.is_empty()
        {
            let sb = SkipBlock::from_sorted_bounds(bounds);
            let skey = fts_chunk_key_coord(term_id, chunk);
            emit_skip(&skey, &sb.encode())?;
            bounds.clear();
        }
        Ok(())
    };

    #[allow(clippy::too_many_arguments)]
    let flush_block = |term_id: TermId,
                       is_existing_term: bool,
                       index_pos: bool,
                       rblk_opt: &mut Option<RowBlock>,
                       blocks_flushed: &mut u32,
                       block_tfs: &mut Vec<(u8, u32)>,
                       block_dls: &mut [u32; ROW_BLOCK_SIZE],
                       block_pos: &mut Vec<(u8, Vec<u32>)>,
                       deferred: &mut Option<(RowBlock, Vec<u8>, u32, f32)>,
                       cur_skip_chunk: &mut Option<izzydb::config::SkipChunk>,
                       cur_skip_bounds: &mut Vec<(u8, u32, f32)>,
                       total_df: u64,
                       is_last_block: bool,
                       emit_skip: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
                       emit_posting: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
                       emit_position: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>|
     -> Result<()> {
        let Some(rblk) = rblk_opt.take() else {
            return Ok(());
        };
        let is_first_block = *blocks_flushed == 0;
        *blocks_flushed += 1;

        let pblk = PackedPostingBlock::from_sorted_with_dl(block_tfs, |slot| {
            block_dls[slot as usize]
        });
        let encoded_p = pblk.encode();
        let hdr = PostingBlockHeader::parse(&encoded_p)?;

        if index_pos {
            let pos_slices: Vec<(u8, &[u32])> = block_pos
                .iter()
                .map(|(s, p)| (*s, p.as_slice()))
                .collect();
            let encoded_pos = PackedPositionBlock::encode_from_sorted_slices(&pos_slices);
            emit_position(&fts_block_key_coord(term_id, rblk), &encoded_pos)?;
        }

        block_tfs.clear();
        block_dls.fill(0);
        block_pos.clear();

        if !is_existing_term
            && is_first_block
            && is_last_block
            && total_df <= INLINE_TERM_MAX_DF as u64
        {
            *deferred = Some((rblk, encoded_p, hdr.max_tf, hdr.min_dl_over_tf));
            return Ok(());
        }

        if let Some((d_rblk, d_bytes, d_max_tf, d_min_ratio)) = deferred.take() {
            emit_posting(&fts_block_key_coord(term_id, d_rblk), &d_bytes)?;
            let chunk = d_rblk.chunk();
            *cur_skip_chunk = Some(chunk);
            cur_skip_bounds.push((d_rblk.rel_block(), d_max_tf, d_min_ratio));
        }

        emit_posting(&fts_block_key_coord(term_id, rblk), &encoded_p)?;
        let chunk = rblk.chunk();
        if *cur_skip_chunk != Some(chunk) {
            flush_skip_chunk(term_id, cur_skip_chunk, cur_skip_bounds, emit_skip)?;
            *cur_skip_chunk = Some(chunk);
        }
        cur_skip_bounds.push((rblk.rel_block(), hdr.max_tf, hdr.min_dl_over_tf));
        Ok(())
    };

    let finish_term = |attr_id: AttrId,
                       token: &str,
                       term_id: TermId,
                       is_existing: bool,
                       stats: TermStats,
                       deferred: &mut Option<(RowBlock, Vec<u8>, u32, f32)>,
                       emit_new_term: &mut dyn FnMut(&[u8], &[u8]) -> Result<()>,
                       emit_updated_term: &mut dyn FnMut(FtsTermAppendDelta) -> Result<()>|
     -> Result<()> {
        let tk = encode_fts_term_key(attr_id, token);
        if is_existing {
            emit_updated_term(FtsTermAppendDelta {
                term_key: tk,
                add_doc_freq: stats.doc_freq,
                max_tf: stats.max_tf,
                min_dl_over_tf: stats.min_dl_over_tf,
            })?;
        } else {
            let rec = match deferred.take() {
                Some((rblk, bytes, _, _)) => TermRecord::with_inline(term_id, stats, rblk, bytes),
                None => TermRecord::out_of_line(term_id, stats),
            };
            emit_new_term(&tk, &rec.encode())?;
        }
        Ok(())
    };

    let _ = is_append_mode;
    while let Some(item) = iter.next_item()? {
        if has_term && (item.attr_id != cur_attr || item.token != cur_token) {
            flush_block(
                cur_term_id,
                cur_is_existing,
                cur_index_pos,
                &mut cur_rblk,
                &mut blocks_flushed_for_term,
                &mut block_tfs,
                &mut block_dls,
                &mut block_pos,
                &mut deferred_first_block,
                &mut cur_skip_chunk,
                &mut cur_skip_bounds,
                cur_stats.doc_freq,
                true,
                emit_skip,
                emit_posting,
                emit_position,
            )?;
            flush_skip_chunk(
                cur_term_id,
                &mut cur_skip_chunk,
                &mut cur_skip_bounds,
                emit_skip,
            )?;
            finish_term(
                cur_attr,
                &cur_token,
                cur_term_id,
                cur_is_existing,
                cur_stats,
                &mut deferred_first_block,
                emit_new_term,
                emit_updated_term,
            )?;
            has_term = false;
        }

        if !has_term {
            has_term = true;
            cur_attr = item.attr_id;
            cur_token = item.token;
            cur_index_pos = *attr_lookup.get(&cur_attr).unwrap();
            if let Some(existing) = existing_terms.get(&(cur_attr, cur_token.clone())) {
                cur_term_id = existing.term_id;
                cur_is_existing = true;
            } else {
                cur_term_id = next_term_id;
                next_term_id = next_term_id
                    .checked_add(1)
                    .ok_or_else(|| TreeError::Storage("TermId u32 space exhausted".into()))?;
                cur_is_existing = false;
            }
            cur_stats = TermStats::default();
            cur_rblk = None;
            blocks_flushed_for_term = 0;
            deferred_first_block = None;
            cur_skip_chunk = None;
            cur_skip_bounds.clear();
        }

        let rblk = RowBlock::new(item.seg, item.blk);
        if cur_rblk != Some(rblk) && cur_rblk.is_some() {
            flush_block(
                cur_term_id,
                cur_is_existing,
                cur_index_pos,
                &mut cur_rblk,
                &mut blocks_flushed_for_term,
                &mut block_tfs,
                &mut block_dls,
                &mut block_pos,
                &mut deferred_first_block,
                &mut cur_skip_chunk,
                &mut cur_skip_bounds,
                cur_stats.doc_freq,
                false,
                emit_skip,
                emit_posting,
                emit_position,
            )?;
        }
        cur_rblk = Some(rblk);

        cur_stats.doc_freq += 1;
        cur_stats.max_tf = cur_stats.max_tf.max(item.tf);
        cur_stats.min_dl_over_tf = cur_stats
            .min_dl_over_tf
            .min(dl_over_tf_lower_bound(item.dl, item.tf));
        block_tfs.push((item.slot, item.tf));
        block_dls[item.slot as usize] = item.dl;
        if cur_index_pos {
            block_pos.push((item.slot, item.positions));
        }
    }

    if has_term {
        flush_block(
            cur_term_id,
            cur_is_existing,
            cur_index_pos,
            &mut cur_rblk,
            &mut blocks_flushed_for_term,
            &mut block_tfs,
            &mut block_dls,
            &mut block_pos,
            &mut deferred_first_block,
            &mut cur_skip_chunk,
            &mut cur_skip_bounds,
            cur_stats.doc_freq,
            true,
            emit_skip,
            emit_posting,
            emit_position,
        )?;
        flush_skip_chunk(
            cur_term_id,
            &mut cur_skip_chunk,
            &mut cur_skip_bounds,
            emit_skip,
        )?;
        finish_term(
            cur_attr,
            &cur_token,
            cur_term_id,
            cur_is_existing,
            cur_stats,
            &mut deferred_first_block,
            emit_new_term,
            emit_updated_term,
        )?;
    }

    Ok(next_term_id)
}
