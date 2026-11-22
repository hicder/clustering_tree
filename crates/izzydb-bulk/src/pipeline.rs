use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rayon::prelude::*;
use roaring::RoaringBitmap;

use izzydb::codec::{
    LeafPosting, decode_attribute_entries, encode_fts_doc_len_key, encode_fts_term_id_key,
    encode_fts_term_key, encode_roaring_bitmap, encode_zone_map, extract_index_keys_from_entries,
    make_posting_key, split_posting_key, TAG_BOOL, TAG_PRESENT,
};
use izzydb::config::{
    AttrId, FTS_BLOCK_SIZE, FtsBlockId, LocalId, NodeId, fts_block_id, fts_skip_chunk_id,
    fts_skip_slot, fts_slot,
};
use izzydb::error::{Result, TreeError};
use izzydb::fts::{
    AttrFtsStats, DocLenBlock, FtsAnalyzer, FtsMeta, INLINE_TERM_MAX_DF, PackedPositionBlock,
    PackedPostingBlock, PostingBlockHeader, ReusableBuffer, SkipBlock, TermRecord, TermStats,
    dl_over_tf_lower_bound,
};
use izzydb::node::PackedLeafNode;
use izzydb::rabitq::{EncodeScratch, RabitqQuantizer};
use izzydb::schema::{AttributeValue, CollectionSchema, FullTextSearchConfig};
use izzydb::storage::{
    ATTR_COUNT_TAG, ATTR_COUNTS_VERSION_KEY, ATTR_ZONE_TAG, AttrZoneMap, CF_ATTR_CLUSTERS,
    CF_ATTR_POSTINGS, CF_DOC_LOC, CF_DOCS, CF_FTS_POSITIONS, CF_FTS_POSTINGS, CF_FTS_SKIPS,
    CF_FTS_TERMS, CF_LEAF_ENTRIES, CF_VECTORS, DEFAULT_CF_NAME, DocLoc, FTS_DOC_LEN_TAG,
    FTS_META_KEY, FTS_REV_BLOCK_TAG, FtsRevBlock, INTERNAL_NODE_TAG, SCHEMA_KEY, META_KEY,
    cf_options_for, default_sst_block_cache, doc_key, encode_count, node_key, tagged_key,
    tagged_vec,
};

use crate::parquet_reader::{DocIndexEntry, MappedFile, StagedData};
use crate::sorter::{
    ExternalSorter, SortItem, read_bytes_lp, read_u16_opt, read_u32_req, write_bytes_lp, write_u16,
    write_u32,
};
use crate::sst::{SstFileInfo, SstSink};
use crate::tree_builder::{AssignedLeaf, BuiltTree, Placement, read_doc_id_at, read_vector_at};

struct AttrPostingItem {
    posting_key: Vec<u8>,
    posting_val: Vec<u8>,
}

impl SortItem for AttrPostingItem {
    #[inline]
    fn cmp_key(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.posting_key.cmp(&b.posting_key)
    }

    #[inline]
    fn estimated_bytes(&self) -> usize {
        self.posting_key.len() + self.posting_val.len() + 48
    }

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        write_bytes_lp(w, &self.posting_key)?;
        write_bytes_lp(w, &self.posting_val)
    }

    fn read_from(r: &mut impl Read) -> io::Result<Option<Self>> {
        let Some(posting_key) = read_bytes_lp(r)? else {
            return Ok(None);
        };
        let posting_val = crate::sorter::read_bytes_req(r)?;
        Ok(Some(Self {
            posting_key,
            posting_val,
        }))
    }
}

struct AttrClusterCountItem {
    cluster_key: Vec<u8>,
    leaf_id: NodeId,
    leaf_doc_count: u32,
}

impl SortItem for AttrClusterCountItem {
    #[inline]
    fn cmp_key(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.cluster_key
            .cmp(&b.cluster_key)
            .then_with(|| a.leaf_id.cmp(&b.leaf_id))
    }

    #[inline]
    fn estimated_bytes(&self) -> usize {
        self.cluster_key.len() + 32
    }

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        write_bytes_lp(w, &self.cluster_key)?;
        write_u32(w, self.leaf_id)?;
        write_u32(w, self.leaf_doc_count)
    }

    fn read_from(r: &mut impl Read) -> io::Result<Option<Self>> {
        let Some(cluster_key) = read_bytes_lp(r)? else {
            return Ok(None);
        };
        let leaf_id = read_u32_req(r)?;
        let leaf_doc_count = read_u32_req(r)?;
        Ok(Some(Self {
            cluster_key,
            leaf_id,
            leaf_doc_count,
        }))
    }
}

struct FtsPostingItem {
    attr_id: AttrId,
    token: String,
    fts_doc_id: u32,
    tf: u32,
    positions: Vec<u32>,
}

impl SortItem for FtsPostingItem {
    #[inline]
    fn cmp_key(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.attr_id
            .cmp(&b.attr_id)
            .then_with(|| a.token.as_bytes().cmp(b.token.as_bytes()))
            .then_with(|| a.fts_doc_id.cmp(&b.fts_doc_id))
    }

    #[inline]
    fn estimated_bytes(&self) -> usize {
        self.token.len() + self.positions.len() * 4 + 48
    }

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        write_u16(w, self.attr_id)?;
        write_bytes_lp(w, self.token.as_bytes())?;
        write_u32(w, self.fts_doc_id)?;
        write_u32(w, self.tf)?;
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
        let tok_bytes = crate::sorter::read_bytes_req(r)?;
        let token = String::from_utf8(tok_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let fts_doc_id = read_u32_req(r)?;
        let tf = read_u32_req(r)?;
        let pos_len = read_u32_req(r)? as usize;
        let mut positions = Vec::with_capacity(pos_len);
        for _ in 0..pos_len {
            positions.push(read_u32_req(r)?);
        }
        Ok(Some(Self {
            attr_id,
            token,
            fts_doc_id,
            tf,
            positions,
        }))
    }
}

type PassAWorkerOutput = (
    Vec<SstFileInfo>,
    HashMap<(AttrId, u8), AttrZoneMap>,
    Vec<AttrFtsStats>,
);

pub struct EmitOutputs {
    pub cf_files: BTreeMap<String, Vec<SstFileInfo>>,
    pub terms_count: u64,
}

#[inline]
fn read_doc_bytes<'a>(docs_idx: &[u8], docs_shards: &'a [MappedFile], row: u32) -> &'a [u8] {
    let off = (row as usize) * DocIndexEntry::SIZE;
    let entry = DocIndexEntry::decode(&docs_idx[off..off + DocIndexEntry::SIZE]);
    if entry.len == 0 {
        return &[];
    }
    let shard = docs_shards[entry.shard as usize].as_bytes();
    &shard[entry.offset as usize..entry.offset as usize + entry.len as usize]
}

#[allow(clippy::too_many_arguments)]
pub fn emit_all_ssts(
    bundle_dir: &Path,
    stage_dir: &Path,
    staged: &StagedData,
    tree: &BuiltTree,
    schema: &CollectionSchema,
    quantizer: &RabitqQuantizer,
    vectors_mmap: &[u8],
    doc_ids_mmap: &[u8],
    docs_idx_mmap: &[u8],
    docs_shards: &[MappedFile],
    target_sst_bytes: u64,
    memory_budget_bytes: usize,
    pool: &rayon::ThreadPool,
) -> Result<EmitOutputs> {
    let block_cache = default_sst_block_cache();
    let dim = tree.meta.dim as usize;
    let padded_dim = quantizer.padded_dim();
    let rabitq_words = quantizer.num_words();
    let n_docs = tree.meta.num_vectors as usize;

    let mut cf_files: BTreeMap<String, Vec<SstFileInfo>> = BTreeMap::new();

    // Empty collection fast-path.
    if n_docs == 0 {
        let leaf_opts = cf_options_for(CF_LEAF_ENTRIES, &block_cache)?;
        let mut leaf_sink = SstSink::new(
            bundle_dir.to_path_buf(),
            CF_LEAF_ENTRIES,
            &leaf_opts,
            target_sst_bytes,
            format!("{CF_LEAF_ENTRIES}/part-00"),
        );
        let empty_leaf = PackedLeafNode::empty(None, dim, padded_dim);
        leaf_sink.put(
            &node_key(1),
            &empty_leaf.encode(dim, padded_dim, rabitq_words),
        )?;
        cf_files.insert(CF_LEAF_ENTRIES.to_owned(), leaf_sink.finish()?);

        let def_opts = cf_options_for(DEFAULT_CF_NAME, &block_cache)?;
        let mut def_sink = SstSink::new(
            bundle_dir.to_path_buf(),
            DEFAULT_CF_NAME,
            &def_opts,
            target_sst_bytes,
            "default/part-00".to_owned(),
        );
        def_sink.put(ATTR_COUNTS_VERSION_KEY, &[1u8])?;
        def_sink.put(SCHEMA_KEY, &schema.encode())?;
        def_sink.put(META_KEY, &tree.meta.encode())?;
        cf_files.insert(DEFAULT_CF_NAME.to_owned(), def_sink.finish()?);

        return Ok(EmitOutputs {
            cf_files,
            terms_count: 0,
        });
    }

    let has_attr_index = schema.entries().iter().any(|e| e.filterable);
    let has_fts = schema.has_fts();

    let mut fts_attrs: Vec<(AttrId, &FullTextSearchConfig)> = schema
        .fts_entries()
        .map(|e| (e.id, e.full_text_search.as_ref().unwrap()))
        .collect();
    fts_attrs.sort_unstable_by_key(|&(id, _)| id);

    // Split fts_dls storage per attr into disjoint worker chunks aligned with leaf ranges.
    let num_workers = pool
        .current_num_threads()
        .min(tree.leaves.len())
        .max(1);
    let leaves_per_worker = tree.leaves.len().div_ceil(num_workers);
    let leaf_ranges: Vec<(usize, usize)> = (0..tree.leaves.len())
        .step_by(leaves_per_worker)
        .enumerate()
        .map(|(w_idx, start)| {
            let end = (start + leaves_per_worker).min(tree.leaves.len());
            let _ = w_idx;
            (start, end)
        })
        .collect();

    let active_sorters = (if has_attr_index { 2 } else { 0 }) + (if has_fts { 1 } else { 0 });
    let per_sorter_budget = memory_budget_bytes
        .checked_div(active_sorters)
        .map_or(memory_budget_bytes, |b| b.max(1024 * 1024));

    let attr_postings_sorter = if has_attr_index {
        Some(ExternalSorter::<AttrPostingItem>::new(
            "attr_postings",
            stage_dir.join("runs_attr_postings"),
            per_sorter_budget,
            leaf_ranges.len(),
        )?)
    } else {
        None
    };
    let attr_clusters_sorter = if has_attr_index {
        Some(ExternalSorter::<AttrClusterCountItem>::new(
            "attr_clusters",
            stage_dir.join("runs_attr_clusters"),
            per_sorter_budget,
            leaf_ranges.len(),
        )?)
    } else {
        None
    };
    let fts_sorter = if has_fts {
        Some(ExternalSorter::<FtsPostingItem>::new(
            "fts_postings",
            stage_dir.join("runs_fts_postings"),
            per_sorter_budget,
            leaf_ranges.len(),
        )?)
    } else {
        None
    };

    // Per-worker FTS doc length buffers (each worker owns a disjoint contiguous range of fts_doc_id).
    let mut fts_worker_dls: Vec<Vec<Vec<u32>>> = if has_fts {
        leaf_ranges
            .iter()
            .map(|&(s, e)| {
                let docs_in_range: usize = tree.leaves[s..e].iter().map(|l| l.rows.len()).sum();
                vec![vec![0u32; docs_in_range]; fts_attrs.len()]
            })
            .collect()
    } else {
        Vec::new()
    };

    // Phase 4: Pass A (leaf-major, parallel over contiguous leaf ranges).
    let leaf_opts = cf_options_for(CF_LEAF_ENTRIES, &block_cache)?;
    let pass_a_results: Vec<PassAWorkerOutput> = pool.install(|| -> Result<Vec<_>> {
        if has_fts {
            leaf_ranges
                .par_iter()
                .zip(fts_worker_dls.par_iter_mut())
                .enumerate()
                .map(|(w_idx, (&(l_start, l_end), w_dls))| {
                    run_pass_a_worker(
                        w_idx,
                        bundle_dir,
                        &leaf_opts,
                        target_sst_bytes,
                        &tree.leaves[l_start..l_end],
                        schema,
                        quantizer,
                        dim,
                        padded_dim,
                        rabitq_words,
                        vectors_mmap,
                        doc_ids_mmap,
                        docs_idx_mmap,
                        docs_shards,
                        has_attr_index,
                        &fts_attrs,
                        Some(w_dls),
                        attr_postings_sorter.as_ref().map(|s| s.shard()),
                        attr_clusters_sorter.as_ref().map(|s| s.shard()),
                        fts_sorter.as_ref().map(|s| s.shard()),
                    )
                })
                .collect()
        } else {
            leaf_ranges
                .par_iter()
                .enumerate()
                .map(|(w_idx, &(l_start, l_end))| {
                    run_pass_a_worker(
                        w_idx,
                        bundle_dir,
                        &leaf_opts,
                        target_sst_bytes,
                        &tree.leaves[l_start..l_end],
                        schema,
                        quantizer,
                        dim,
                        padded_dim,
                        rabitq_words,
                        vectors_mmap,
                        doc_ids_mmap,
                        docs_idx_mmap,
                        docs_shards,
                        has_attr_index,
                        &fts_attrs,
                        None,
                        attr_postings_sorter.as_ref().map(|s| s.shard()),
                        attr_clusters_sorter.as_ref().map(|s| s.shard()),
                        None,
                    )
                })
                .collect()
        }
    })?;

    let mut leaf_ssts = Vec::new();
    let mut merged_zones: BTreeMap<(AttrId, u8), AttrZoneMap> = BTreeMap::new();
    let mut merged_fts_stats: BTreeMap<AttrId, AttrFtsStats> = BTreeMap::new();

    for (ssts, zones, fts_stats) in pass_a_results {
        leaf_ssts.extend(ssts);
        for (k, zmap) in zones {
            let dst = merged_zones.entry(k).or_default();
            for (lid, bounds) in zmap {
                dst.insert(lid, bounds);
            }
        }
        for (idx, st) in fts_stats.into_iter().enumerate() {
            let entry = merged_fts_stats.entry(fts_attrs[idx].0).or_default();
            entry.doc_count += st.doc_count;
            entry.total_tokens += st.total_tokens;
        }
    }
    cf_files.insert(CF_LEAF_ENTRIES.to_owned(), leaf_ssts);

    // Flatten worker FTS doc lengths into one contiguous array per FTS attribute (in 0..N order).
    let mut fts_dls: Vec<Vec<u32>> = Vec::with_capacity(fts_attrs.len());
    if has_fts {
        for f_idx in 0..fts_attrs.len() {
            let mut full = Vec::with_capacity(n_docs);
            for w_dls in &mut fts_worker_dls {
                full.extend_from_slice(&w_dls[f_idx]);
            }
            debug_assert_eq!(full.len(), n_docs);
            fts_dls.push(full);
        }
    }
    drop(fts_worker_dls);

    // Phase 5: Pass B (doc-id order, parallel over disjoint doc_id ranges).
    let rows_by_doc_id_mmap = MappedFile::open(&staged.rows_by_doc_id_path)?;
    let placement_mmap = MappedFile::open(&tree.placement_by_row_path)?;
    let pass_b_workers = pool.current_num_threads().min(n_docs).max(1);
    let docs_per_worker = n_docs.div_ceil(pass_b_workers);
    let doc_ranges: Vec<(usize, usize)> = (0..n_docs)
        .step_by(docs_per_worker)
        .map(|s| (s, (s + docs_per_worker).min(n_docs)))
        .collect();

    let docs_opts = cf_options_for(CF_DOCS, &block_cache)?;
    let vecs_opts = cf_options_for(CF_VECTORS, &block_cache)?;
    let loc_opts = cf_options_for(CF_DOC_LOC, &block_cache)?;

    let pass_b_results: Vec<(Vec<SstFileInfo>, Vec<SstFileInfo>, Vec<SstFileInfo>)> =
        pool.install(|| -> Result<Vec<_>> {
            doc_ranges
                .par_iter()
                .enumerate()
                .map(|(w_idx, &(start, end))| {
                    run_pass_b_worker(
                        w_idx,
                        bundle_dir,
                        &docs_opts,
                        &vecs_opts,
                        &loc_opts,
                        target_sst_bytes,
                        dim,
                        start,
                        end,
                        rows_by_doc_id_mmap.as_bytes(),
                        placement_mmap.as_bytes(),
                        doc_ids_mmap,
                        vectors_mmap,
                        docs_idx_mmap,
                        docs_shards,
                    )
                })
                .collect()
        })?;

    let mut docs_ssts = Vec::new();
    let mut vecs_ssts = Vec::new();
    let mut loc_ssts = Vec::new();
    for (d, v, l) in pass_b_results {
        docs_ssts.extend(d);
        vecs_ssts.extend(v);
        loc_ssts.extend(l);
    }
    cf_files.insert(CF_DOCS.to_owned(), docs_ssts);
    cf_files.insert(CF_VECTORS.to_owned(), vecs_ssts);
    cf_files.insert(CF_DOC_LOC.to_owned(), loc_ssts);

    // Phase 6: Finish sorted CFs (attribute postings, attribute clusters + counts, FTS, and default).
    let counts_spill_path = stage_dir.join("attr_counts_sorted.bin");

    let attr_postings_res: Result<Vec<SstFileInfo>> = if let Some(sorter) = attr_postings_sorter {
        let mut iter = sorter.into_sorted_iter()?;
        let opts = cf_options_for(CF_ATTR_POSTINGS, &block_cache)?;
        let mut sink = SstSink::new(
            bundle_dir.to_path_buf(),
            CF_ATTR_POSTINGS,
            &opts,
            target_sst_bytes,
            format!("{CF_ATTR_POSTINGS}/part-00"),
        );
        while let Some(item) = iter.next_item()? {
            sink.put(&item.posting_key, &item.posting_val)?;
        }
        sink.finish()
    } else {
        Ok(Vec::new())
    };

    let attr_clusters_res: Result<Vec<SstFileInfo>> = if let Some(sorter) = attr_clusters_sorter {
        let mut iter = sorter.into_sorted_iter()?;
        let opts = cf_options_for(CF_ATTR_CLUSTERS, &block_cache)?;
        let mut sink = SstSink::new(
            bundle_dir.to_path_buf(),
            CF_ATTR_CLUSTERS,
            &opts,
            target_sst_bytes,
            format!("{CF_ATTR_CLUSTERS}/part-00"),
        );
        let counts_file = File::create(&counts_spill_path)
            .map_err(|e| TreeError::Storage(format!("create counts spill: {e}")))?;
        let mut counts_writer = BufWriter::with_capacity(256 * 1024, counts_file);

        let mut cur_key: Option<Vec<u8>> = None;
        let mut cur_leaves: Vec<NodeId> = Vec::new();
        let mut cur_count: u64 = 0;

        while let Some(item) = iter.next_item()? {
            match &cur_key {
                Some(k) if k == &item.cluster_key => {
                    if cur_leaves.last().copied() != Some(item.leaf_id) {
                        cur_leaves.push(item.leaf_id);
                    }
                    cur_count += item.leaf_doc_count as u64;
                }
                _ => {
                    if let Some(prev_key) = cur_key.take() {
                        let bm = RoaringBitmap::from_sorted_iter(cur_leaves.drain(..))
                            .map_err(|e| TreeError::Storage(format!("roaring error: {e}")))?;
                        sink.put(&prev_key, &encode_roaring_bitmap(&bm)?)?;
                        write_count_record(&mut counts_writer, &prev_key, cur_count)?;
                    }
                    cur_key = Some(item.cluster_key);
                    cur_leaves.push(item.leaf_id);
                    cur_count = item.leaf_doc_count as u64;
                }
            }
        }
        if let Some(prev_key) = cur_key.take() {
            let bm = RoaringBitmap::from_sorted_iter(cur_leaves.drain(..))
                .map_err(|e| TreeError::Storage(format!("roaring error: {e}")))?;
            sink.put(&prev_key, &encode_roaring_bitmap(&bm)?)?;
            write_count_record(&mut counts_writer, &prev_key, cur_count)?;
        }
        counts_writer
            .flush()
            .map_err(|e| TreeError::Storage(format!("flush counts spill: {e}")))?;
        sink.finish()
    } else {
        Ok(Vec::new())
    };

    let attr_postings_ssts = attr_postings_res?;
    if !attr_postings_ssts.is_empty() {
        cf_files.insert(CF_ATTR_POSTINGS.to_owned(), attr_postings_ssts);
    }
    let attr_clusters_ssts = attr_clusters_res?;
    if !attr_clusters_ssts.is_empty() {
        cf_files.insert(CF_ATTR_CLUSTERS.to_owned(), attr_clusters_ssts);
    }

    // Finish FTS CFs if enabled.
    let mut fts_meta_opt: Option<FtsMeta> = None;
    let mut terms_count = 0u64;
    if let Some(sorter) = fts_sorter {
        let mut iter = sorter.into_sorted_iter()?;
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

        let attr_lookup: HashMap<AttrId, (usize, bool)> = fts_attrs
            .iter()
            .enumerate()
            .map(|(i, &(id, cfg))| (id, (i, cfg.index_positions)))
            .collect();

        let next_term_id = emit_fts_streams(
            &mut iter,
            &attr_lookup,
            &fts_dls,
            &mut terms_sink,
            &mut skips_sink,
            &mut postings_sink,
            &mut positions_sink,
        )?;
        terms_count = next_term_id as u64;

        let terms_ssts = terms_sink.finish()?;
        let skips_ssts = skips_sink.finish()?;
        let postings_ssts = postings_sink.finish()?;
        let positions_ssts = positions_sink.finish()?;
        if !terms_ssts.is_empty() {
            cf_files.insert(CF_FTS_TERMS.to_owned(), terms_ssts);
        }
        if !skips_ssts.is_empty() {
            cf_files.insert(CF_FTS_SKIPS.to_owned(), skips_ssts);
        }
        if !postings_ssts.is_empty() {
            cf_files.insert(CF_FTS_POSTINGS.to_owned(), postings_ssts);
        }
        if !positions_ssts.is_empty() {
            cf_files.insert(CF_FTS_POSITIONS.to_owned(), positions_ssts);
        }

        fts_meta_opt = Some(FtsMeta::new(n_docs as u32, next_term_id, merged_fts_stats));
    }

    // Build `default` CF SST in strict byte order:
    // 0x01 attr_counts_v1 < 0x01 fts_meta < 0x01 schema < 0x01 tree_meta
    // < 0x02 internal_nodes < 0x03 zones < 0x04 rev_blocks < 0x05 doc_lens < 0x06 counts
    let def_opts = cf_options_for(DEFAULT_CF_NAME, &block_cache)?;
    let mut def_sink = SstSink::new(
        bundle_dir.to_path_buf(),
        DEFAULT_CF_NAME,
        &def_opts,
        target_sst_bytes,
        "default/part-00".to_owned(),
    );

    def_sink.put(ATTR_COUNTS_VERSION_KEY, &[1u8])?;
    if let Some(ref fts_meta) = fts_meta_opt {
        def_sink.put(FTS_META_KEY, &fts_meta.encode())?;
    }
    def_sink.put(SCHEMA_KEY, &schema.encode())?;
    def_sink.put(META_KEY, &tree.meta.encode())?;

    // 0x02 internal nodes (already sorted by NodeId ascending).
    for (nid, node) in &tree.internal_nodes {
        let key = tagged_key::<5>(INTERNAL_NODE_TAG, &node_key(*nid));
        def_sink.put(&key, &node.encode(dim))?;
    }

    // 0x03 attribute zones (BTreeMap ensures ascending (attr_id, type_tag)).
    for (&(attr_id, type_tag), zmap) in &merged_zones {
        if zmap.is_empty() {
            continue;
        }
        let zkey = izzydb::codec::encode_attr_prefix(attr_id, type_tag);
        let key = tagged_key::<4>(ATTR_ZONE_TAG, &zkey);
        def_sink.put(&key, &encode_zone_map(zmap))?;
    }

    // 0x04 FTS rev blocks (fts_doc_id assigned in ascending leaf-major order 0..N).
    if has_fts {
        emit_fts_rev_blocks(&tree.leaves, doc_ids_mmap, &mut def_sink)?;
    }

    // 0x05 FTS doc length blocks (ascending attr_id, then ascending block_id).
    if has_fts {
        for (f_idx, &(attr_id, _)) in fts_attrs.iter().enumerate() {
            let dls = &fts_dls[f_idx];
            for (bid_idx, chunk) in dls.chunks(FTS_BLOCK_SIZE).enumerate() {
                let blk = DocLenBlock::from_dls(chunk);
                if !blk.is_empty() {
                    let key = tagged_key::<7>(
                        FTS_DOC_LEN_TAG,
                        &encode_fts_doc_len_key(attr_id, bid_idx as FtsBlockId),
                    );
                    def_sink.put(&key, &blk.encode())?;
                }
            }
        }
    }

    // 0x06 global attribute counts (streamed in ascending cluster_key order from spill file).
    if has_attr_index {
        let file = File::open(&counts_spill_path)
            .map_err(|e| TreeError::Storage(format!("open counts spill: {e}")))?;
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        while let Some(cluster_key) = read_bytes_lp(&mut reader)
            .map_err(|e| TreeError::Storage(format!("read counts spill: {e}")))?
        {
            let mut cnt_buf = [0u8; 8];
            reader
                .read_exact(&mut cnt_buf)
                .map_err(|e| TreeError::Storage(format!("read count val: {e}")))?;
            let count = u64::from_le_bytes(cnt_buf);
            if count > 0 {
                let key = tagged_vec(ATTR_COUNT_TAG, &cluster_key);
                def_sink.put(&key, &encode_count(count))?;
            }
        }
    }

    cf_files.insert(DEFAULT_CF_NAME.to_owned(), def_sink.finish()?);

    Ok(EmitOutputs {
        cf_files,
        terms_count,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_pass_a_worker(
    w_idx: usize,
    bundle_dir: &Path,
    leaf_opts: &rocksdb::Options,
    target_sst_bytes: u64,
    leaves: &[AssignedLeaf],
    schema: &CollectionSchema,
    quantizer: &RabitqQuantizer,
    dim: usize,
    padded_dim: usize,
    rabitq_words: usize,
    vectors_mmap: &[u8],
    doc_ids_mmap: &[u8],
    docs_idx_mmap: &[u8],
    docs_shards: &[MappedFile],
    has_attr_index: bool,
    fts_attrs: &[(AttrId, &FullTextSearchConfig)],
    mut worker_dls: Option<&mut Vec<Vec<u32>>>,
    mut attr_postings_shard: Option<crate::sorter::SorterShard<AttrPostingItem>>,
    mut attr_clusters_shard: Option<crate::sorter::SorterShard<AttrClusterCountItem>>,
    mut fts_shard: Option<crate::sorter::SorterShard<FtsPostingItem>>,
) -> Result<PassAWorkerOutput> {
    let mut leaf_sink = SstSink::new(
        bundle_dir.to_path_buf(),
        CF_LEAF_ENTRIES,
        leaf_opts,
        target_sst_bytes,
        format!("{CF_LEAF_ENTRIES}/part-{w_idx:02}"),
    );

    let mut scratch = EncodeScratch::default();
    let mut fts_buf = ReusableBuffer::new();
    let fts_analyzers: Vec<FtsAnalyzer> = fts_attrs
        .iter()
        .map(|&(_, cfg)| FtsAnalyzer::new(cfg.clone()))
        .collect();
    let fts_attr_index_map: HashMap<AttrId, usize> = fts_attrs
        .iter()
        .enumerate()
        .map(|(i, &(id, _))| (id, i))
        .collect();
    let mut fts_stats = vec![AttrFtsStats::default(); fts_attrs.len()];
    let mut zone_bounds: HashMap<(AttrId, u8), AttrZoneMap> = HashMap::new();

    let worker_fid_start = leaves.first().map_or(0, |l| l.first_fts_doc_id);
    let mut leaf_doc_ids = Vec::new();
    let mut flat_vecs = Vec::new();
    let mut leaf_attr_slots: BTreeMap<Vec<u8>, Vec<LocalId>> = BTreeMap::new();

    for leaf in leaves {
        leaf_doc_ids.clear();
        flat_vecs.clear();
        leaf_doc_ids.reserve(leaf.rows.len());
        flat_vecs.reserve(leaf.rows.len() * dim);

        for &row in &leaf.rows {
            leaf_doc_ids.push(read_doc_id_at(doc_ids_mmap, row));
            flat_vecs.extend_from_slice(read_vector_at(vectors_mmap, row, dim));
        }

        let packed_leaf = PackedLeafNode::from_entries(
            leaf.parent_id,
            leaf.centroid.clone(),
            &leaf_doc_ids,
            &flat_vecs,
            quantizer,
            dim,
            &mut scratch,
        );
        leaf_sink.put(
            &node_key(leaf.leaf_id),
            &packed_leaf.encode(dim, padded_dim, rabitq_words),
        )?;

        if !has_attr_index && fts_attrs.is_empty() {
            continue;
        }

        leaf_attr_slots.clear();

        for (slot, &row) in leaf.rows.iter().enumerate() {
            let doc_bytes = read_doc_bytes(docs_idx_mmap, docs_shards, row);
            if doc_bytes.is_empty() {
                continue;
            }
            let entries = decode_attribute_entries(doc_bytes)?;

            if has_attr_index {
                for ckey in extract_index_keys_from_entries(schema, &entries)? {
                    leaf_attr_slots.entry(ckey).or_default().push(slot as LocalId);
                }
            }

            if let (Some(w_dls), Some(f_shard)) = (worker_dls.as_deref_mut(), fts_shard.as_mut()) {
                let fid = leaf.first_fts_doc_id + slot as u32;
                let rel_fid = (fid - worker_fid_start) as usize;

                for (attr_id, val) in &entries {
                    let Some(&f_idx) = fts_attr_index_map.get(attr_id) else {
                        continue;
                    };
                    if matches!(val, AttributeValue::Null) {
                        continue;
                    }
                    if let Some(analyzed) =
                        fts_analyzers[f_idx].analyze_attribute_value(val, &mut fts_buf)?
                    {
                        fts_stats[f_idx].doc_count += 1;
                        fts_stats[f_idx].total_tokens += analyzed.doc_len as u64;
                        if analyzed.doc_len > 0 {
                            w_dls[f_idx][rel_fid] = analyzed.doc_len;
                            for (token, term) in analyzed.terms {
                                f_shard.push(FtsPostingItem {
                                    attr_id: *attr_id,
                                    token,
                                    fts_doc_id: fid,
                                    tf: term.tf,
                                    positions: term.positions,
                                })?;
                            }
                        }
                    }
                }
            }
        }

        if has_attr_index {
            let p_shard = attr_postings_shard.as_mut().unwrap();
            let c_shard = attr_clusters_shard.as_mut().unwrap();

            for (ckey, slots) in leaf_attr_slots.iter() {
                let posting_key = make_posting_key(ckey, leaf.leaf_id);
                let posting_val = LeafPosting::encode_sorted_slots(slots)?;
                p_shard.push(AttrPostingItem {
                    posting_key: posting_key.clone(),
                    posting_val,
                })?;
                c_shard.push(AttrClusterCountItem {
                    cluster_key: ckey.clone(),
                    leaf_id: leaf.leaf_id,
                    leaf_doc_count: slots.len() as u32,
                })?;

                let (attr_id, type_tag, _, val_bytes) = split_posting_key(&posting_key)?;
                if type_tag != TAG_PRESENT && type_tag != TAG_BOOL {
                    let zmap = zone_bounds.entry((attr_id, type_tag)).or_default();
                    match zmap.get_mut(&leaf.leaf_id) {
                        Some((min_b, max_b)) => {
                            if val_bytes < min_b.as_slice() {
                                *min_b = val_bytes.to_vec();
                            }
                            if val_bytes > max_b.as_slice() {
                                *max_b = val_bytes.to_vec();
                            }
                        }
                        None => {
                            zmap.insert(leaf.leaf_id, (val_bytes.to_vec(), val_bytes.to_vec()));
                        }
                    }
                }
            }
        }
    }

    if let Some(s) = attr_postings_shard {
        s.finish()?;
    }
    if let Some(s) = attr_clusters_shard {
        s.finish()?;
    }
    if let Some(s) = fts_shard {
        s.finish()?;
    }

    let leaf_ssts = leaf_sink.finish()?;
    Ok((leaf_ssts, zone_bounds, fts_stats))
}

#[allow(clippy::too_many_arguments)]
fn run_pass_b_worker(
    w_idx: usize,
    bundle_dir: &Path,
    docs_opts: &rocksdb::Options,
    vecs_opts: &rocksdb::Options,
    loc_opts: &rocksdb::Options,
    target_sst_bytes: u64,
    dim: usize,
    start: usize,
    end: usize,
    rows_by_doc_id_mmap: &[u8],
    placement_mmap: &[u8],
    doc_ids_mmap: &[u8],
    vectors_mmap: &[u8],
    docs_idx_mmap: &[u8],
    docs_shards: &[MappedFile],
) -> Result<(Vec<SstFileInfo>, Vec<SstFileInfo>, Vec<SstFileInfo>)> {
    let mut docs_sink = SstSink::new(
        bundle_dir.to_path_buf(),
        CF_DOCS,
        docs_opts,
        target_sst_bytes,
        format!("{CF_DOCS}/part-{w_idx:02}"),
    );
    let mut vecs_sink = SstSink::new(
        bundle_dir.to_path_buf(),
        CF_VECTORS,
        vecs_opts,
        target_sst_bytes,
        format!("{CF_VECTORS}/part-{w_idx:02}"),
    );
    let mut loc_sink = SstSink::new(
        bundle_dir.to_path_buf(),
        CF_DOC_LOC,
        loc_opts,
        target_sst_bytes,
        format!("{CF_DOC_LOC}/part-{w_idx:02}"),
    );

    let vec_byte_len = dim * 4;
    for i in start..end {
        let row = u32::from_le_bytes(rows_by_doc_id_mmap[i * 4..(i + 1) * 4].try_into().unwrap());
        let doc_id = read_doc_id_at(doc_ids_mmap, row);
        let key = doc_key(doc_id);

        let doc_bytes = read_doc_bytes(docs_idx_mmap, docs_shards, row);
        docs_sink.put(&key, doc_bytes)?;

        let vec_off = (row as usize) * vec_byte_len;
        vecs_sink.put(&key, &vectors_mmap[vec_off..vec_off + vec_byte_len])?;

        let pl_off = (row as usize) * Placement::SIZE;
        let pl = Placement::decode(&placement_mmap[pl_off..pl_off + Placement::SIZE]);
        let doc_loc = DocLoc {
            leaf_id: pl.leaf_id,
            local_id: pl.local_id,
            fts_doc_id: if pl.fts_doc_id == izzydb::config::FtsDocId::MAX {
                None
            } else {
                Some(pl.fts_doc_id)
            },
        };
        loc_sink.put(&key, &doc_loc.encode())?;
    }

    Ok((docs_sink.finish()?, vecs_sink.finish()?, loc_sink.finish()?))
}

fn emit_fts_rev_blocks(
    leaves: &[AssignedLeaf],
    doc_ids_mmap: &[u8],
    def_sink: &mut SstSink<'_>,
) -> Result<()> {
    let mut cur_bid: Option<FtsBlockId> = None;
    let mut cur_blk = FtsRevBlock::default();

    for leaf in leaves {
        for (local_id, &row) in leaf.rows.iter().enumerate() {
            let fid = leaf.first_fts_doc_id + local_id as u32;
            let bid = fts_block_id(fid);
            let slot = fts_slot(fid);
            if cur_bid != Some(bid) {
                if let Some(prev_bid) = cur_bid {
                    let key = tagged_key::<5>(
                        FTS_REV_BLOCK_TAG,
                        &izzydb::codec::encode_fts_block_key(prev_bid),
                    );
                    def_sink.put(&key, &cur_blk.encode())?;
                    cur_blk = FtsRevBlock::default();
                }
                cur_bid = Some(bid);
            }
            let doc_id = read_doc_id_at(doc_ids_mmap, row);
            cur_blk.set(slot, doc_id, leaf.leaf_id, local_id as LocalId);
        }
    }
    if let Some(prev_bid) = cur_bid
        && !cur_blk.is_empty()
    {
        let key = tagged_key::<5>(
            FTS_REV_BLOCK_TAG,
            &izzydb::codec::encode_fts_block_key(prev_bid),
        );
        def_sink.put(&key, &cur_blk.encode())?;
    }
    Ok(())
}

fn emit_fts_streams(
    iter: &mut crate::sorter::SortedIter<FtsPostingItem>,
    attr_lookup: &HashMap<AttrId, (usize, bool)>,
    fts_dls: &[Vec<u32>],
    terms_sink: &mut SstSink<'_>,
    skips_sink: &mut SstSink<'_>,
    postings_sink: &mut SstSink<'_>,
    positions_sink: &mut SstSink<'_>,
) -> Result<u32> {
    let mut next_term_id: u32 = 0;

    let mut cur_attr: AttrId = 0;
    let mut cur_token: String = String::new();
    let mut has_term = false;
    let mut cur_term_id: u32 = 0;
    let mut cur_f_idx: usize = 0;
    let mut cur_index_pos: bool = false;
    let mut cur_stats = TermStats::default();

    // Current 256-doc block accumulators.
    let mut cur_block_id: Option<FtsBlockId> = None;
    let mut blocks_flushed_for_term: u32 = 0;
    let mut block_tfs: Vec<(u8, u32)> = Vec::new();
    let mut block_pos: Vec<(u8, Vec<u32>)> = Vec::new();

    // Deferred first block for potential inline TermRecord (<= INLINE_TERM_MAX_DF on 1 block).
    let mut deferred_first_block: Option<(FtsBlockId, Vec<u8>, u32, f32)> = None;

    // Current skip chunk accumulator for out-of-line terms.
    let mut cur_skip_chunk_id: Option<u32> = None;
    let mut cur_skip_bounds: Vec<(u8, u32, f32)> = Vec::new();

    let flush_skip_chunk = |term_id: u32,
                            chunk_id_opt: &mut Option<u32>,
                            bounds: &mut Vec<(u8, u32, f32)>,
                            skips_sink: &mut SstSink<'_>|
     -> Result<()> {
        if let Some(chunk_id) = chunk_id_opt.take()
            && !bounds.is_empty()
        {
            let sb = SkipBlock::from_sorted_bounds(bounds);
            let skey = encode_fts_term_id_key(term_id, chunk_id);
            skips_sink.put(&skey, &sb.encode())?;
            bounds.clear();
        }
        Ok(())
    };

    #[allow(clippy::too_many_arguments)]
    let flush_block = |term_id: u32,
                       f_idx: usize,
                       index_pos: bool,
                       block_id_opt: &mut Option<FtsBlockId>,
                       blocks_flushed: &mut u32,
                       block_tfs: &mut Vec<(u8, u32)>,
                       block_pos: &mut Vec<(u8, Vec<u32>)>,
                       deferred: &mut Option<(FtsBlockId, Vec<u8>, u32, f32)>,
                       cur_skip_chunk_id: &mut Option<u32>,
                       cur_skip_bounds: &mut Vec<(u8, u32, f32)>,
                       total_df: u64,
                       is_last_block_of_term: bool,
                       skips_sink: &mut SstSink<'_>,
                       postings_sink: &mut SstSink<'_>,
                       positions_sink: &mut SstSink<'_>|
     -> Result<()> {
        let Some(bid) = block_id_opt.take() else {
            return Ok(());
        };
        let is_first_block = *blocks_flushed == 0;
        *blocks_flushed += 1;

        let dls_slice = &fts_dls[f_idx];
        let base_fid = bid << 8;
        let pblk = PackedPostingBlock::from_sorted_with_dl(block_tfs, |slot| {
            dls_slice[(base_fid + slot as u32) as usize]
        });
        let encoded_p = pblk.encode();
        let hdr = PostingBlockHeader::parse(&encoded_p)?;

        if index_pos {
            let pos_slices: Vec<(u8, &[u32])> = block_pos
                .iter()
                .map(|(s, p)| (*s, p.as_slice()))
                .collect();
            let encoded_pos = PackedPositionBlock::encode_from_sorted_slices(&pos_slices);
            let pos_key = encode_fts_term_id_key(term_id, bid);
            positions_sink.put(&pos_key, &encoded_pos)?;
        }

        block_tfs.clear();
        block_pos.clear();

        if is_first_block && is_last_block_of_term && total_df <= INLINE_TERM_MAX_DF as u64 {
            // Single block and doc_freq <= INLINE_TERM_MAX_DF: keep deferred so finish_term inlines it!
            *deferred = Some((bid, encoded_p, hdr.max_tf, hdr.min_dl_over_tf));
            return Ok(());
        }

        // Otherwise materialize any previously deferred block, then write this block out-of-line.
        if let Some((d_bid, d_bytes, d_max_tf, d_min_ratio)) = deferred.take() {
            let pkey = encode_fts_term_id_key(term_id, d_bid);
            postings_sink.put(&pkey, &d_bytes)?;
            let cid = fts_skip_chunk_id(d_bid);
            *cur_skip_chunk_id = Some(cid);
            cur_skip_bounds.push((fts_skip_slot(d_bid), d_max_tf, d_min_ratio));
        }

        let pkey = encode_fts_term_id_key(term_id, bid);
        postings_sink.put(&pkey, &encoded_p)?;
        let cid = fts_skip_chunk_id(bid);
        if *cur_skip_chunk_id != Some(cid) {
            flush_skip_chunk(term_id, cur_skip_chunk_id, cur_skip_bounds, skips_sink)?;
            *cur_skip_chunk_id = Some(cid);
        }
        cur_skip_bounds.push((fts_skip_slot(bid), hdr.max_tf, hdr.min_dl_over_tf));
        Ok(())
    };

    while let Some(item) = iter.next_item()? {
        if has_term && (item.attr_id != cur_attr || item.token != cur_token) {
            // Flush last block and finish previous term.
            flush_block(
                cur_term_id,
                cur_f_idx,
                cur_index_pos,
                &mut cur_block_id,
                &mut blocks_flushed_for_term,
                &mut block_tfs,
                &mut block_pos,
                &mut deferred_first_block,
                &mut cur_skip_chunk_id,
                &mut cur_skip_bounds,
                cur_stats.doc_freq,
                true,
                skips_sink,
                postings_sink,
                positions_sink,
            )?;
            flush_skip_chunk(
                cur_term_id,
                &mut cur_skip_chunk_id,
                &mut cur_skip_bounds,
                skips_sink,
            )?;
            let rec = match deferred_first_block.take() {
                Some((bid, bytes, _, _)) => {
                    TermRecord::with_inline(cur_term_id, cur_stats, bid, bytes)
                }
                None => TermRecord::out_of_line(cur_term_id, cur_stats),
            };
            terms_sink.put(&encode_fts_term_key(cur_attr, &cur_token), &rec.encode())?;
            has_term = false;
        }

        if !has_term {
            has_term = true;
            cur_attr = item.attr_id;
            cur_token = item.token;
            cur_term_id = next_term_id;
            next_term_id = next_term_id
                .checked_add(1)
                .ok_or_else(|| TreeError::Storage("TermId u32 space exhausted".into()))?;
            let &(f_idx, idx_pos) = attr_lookup.get(&cur_attr).unwrap();
            cur_f_idx = f_idx;
            cur_index_pos = idx_pos;
            cur_stats = TermStats::default();
            cur_block_id = None;
            blocks_flushed_for_term = 0;
            deferred_first_block = None;
            cur_skip_chunk_id = None;
            cur_skip_bounds.clear();
        }

        let bid = fts_block_id(item.fts_doc_id);
        let slot = fts_slot(item.fts_doc_id);
        if cur_block_id != Some(bid) && cur_block_id.is_some() {
            flush_block(
                cur_term_id,
                cur_f_idx,
                cur_index_pos,
                &mut cur_block_id,
                &mut blocks_flushed_for_term,
                &mut block_tfs,
                &mut block_pos,
                &mut deferred_first_block,
                &mut cur_skip_chunk_id,
                &mut cur_skip_bounds,
                cur_stats.doc_freq,
                false,
                skips_sink,
                postings_sink,
                positions_sink,
            )?;
        }
        cur_block_id = Some(bid);

        let dl = fts_dls[cur_f_idx][item.fts_doc_id as usize];
        cur_stats.doc_freq += 1;
        cur_stats.max_tf = cur_stats.max_tf.max(item.tf);
        cur_stats.min_dl_over_tf = cur_stats
            .min_dl_over_tf
            .min(dl_over_tf_lower_bound(dl, item.tf));

        block_tfs.push((slot, item.tf));
        if cur_index_pos {
            block_pos.push((slot, item.positions));
        }
    }

    if has_term {
        flush_block(
            cur_term_id,
            cur_f_idx,
            cur_index_pos,
            &mut cur_block_id,
            &mut blocks_flushed_for_term,
            &mut block_tfs,
            &mut block_pos,
            &mut deferred_first_block,
            &mut cur_skip_chunk_id,
            &mut cur_skip_bounds,
            cur_stats.doc_freq,
            true,
            skips_sink,
            postings_sink,
            positions_sink,
        )?;
        flush_skip_chunk(
            cur_term_id,
            &mut cur_skip_chunk_id,
            &mut cur_skip_bounds,
            skips_sink,
        )?;
        let rec = match deferred_first_block.take() {
            Some((bid, bytes, _, _)) => {
                TermRecord::with_inline(cur_term_id, cur_stats, bid, bytes)
            }
            None => TermRecord::out_of_line(cur_term_id, cur_stats),
        };
        terms_sink.put(&encode_fts_term_key(cur_attr, &cur_token), &rec.encode())?;
    }

    Ok(next_term_id)
}

fn write_count_record(w: &mut impl Write, cluster_key: &[u8], count: u64) -> Result<()> {
    write_bytes_lp(w, cluster_key).map_err(|e| TreeError::Storage(e.to_string()))?;
    w.write_all(&count.to_le_bytes())
        .map_err(|e| TreeError::Storage(e.to_string()))?;
    Ok(())
}
