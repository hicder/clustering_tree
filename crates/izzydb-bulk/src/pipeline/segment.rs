//! Per-segment immutable SST emission.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use rayon::prelude::*;
use roaring::RoaringBitmap;

use izzydb::codec::{
    SEG_FLAG_HAS_COLUMNS, SEG_FLAG_HAS_VECTORS, SegPosting, SegmentMeta, SegmentState, TAG_BOOL,
    TAG_PRESENT, ZoneEntry, column_key, decode_row_entries, detached_count_suffix, detached_key,
    detached_posting_key, doc_id_block_key, encode_doc_id_block, extract_index_keys_for_attr, extract_index_keys_from_entries, make_posting_key, row_key,
    RowView,
};
use izzydb::config::{
    AttrId, ColumnGen, DocId, LocalId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentConfig,
    SegmentId,
};
use izzydb::error::Result;
use izzydb::fts::{
    AttrFtsStats, DocLenBlock, FtsAnalyzer, ReusableBuffer, TermRecord,
};
use izzydb::schema::{
    AttributeType, AttributeValue, CollectionSchema, ColumnStorage, FullTextSearchConfig,
    SchemaEntry,
};
use izzydb::storage::{
    CF_ATTR_POSTINGS, CF_COLUMNS, CF_DETACHED,
    CF_DETACHED_POSTINGS, CF_FTS_POSITIONS, CF_FTS_SKIPS, CF_FTS_TERMS,
    CF_ID_MAP, CF_ROW_DOC_IDS, CF_ROWS, CF_VECTORS, cf_options_for, default_sst_block_cache, doc_key,
};
use izzydb_storage::column::ColumnBlockBuilder;

use crate::common::parquet_reader::{MappedFile, StagedData, decode_staged_doc, read_doc_id_at};
use crate::common::sorter::{
    ExternalSorter, KvItem,
};
use crate::common::sst::{SstFileInfo, SstSink};

use super::*;

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
            let s_u32 = s;
            let e_u32 = (s + segs_per_worker as u32).min(num_segs);
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
                                        .analyze_attribute_value(val, &mut fts_buf)?
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
            let positions_sorter = ExternalSorter::<KvItem>::new(
                "fts_positions_append",
                stage_dir.join("runs_fts_positions"),
                memory_budget_bytes / 4,
                1,
            )?;
            let mut skips_shard = skips_sorter.shard();
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
                    positions_shard.push(KvItem {
                        key: k.to_vec(),
                        val: v.to_vec(),
                    })
                },
            )?;
            skips_shard.finish()?;
            positions_shard.finish()?;

            let skips_files =
                emit_sorted_kv_sst(bundle_dir, CF_FTS_SKIPS, target_sst_bytes, skips_sorter)?;
            let positions_files =
                emit_sorted_kv_sst(bundle_dir, CF_FTS_POSITIONS, target_sst_bytes, positions_sorter)?;
            if !skips_files.is_empty() {
                cf_files.insert(CF_FTS_SKIPS.to_owned(), skips_files);
            }
            if !positions_files.is_empty() {
                cf_files.insert(CF_FTS_POSITIONS.to_owned(), positions_files);
            }
            (new_terms, upd_terms, next_tid)
        } else {
            let terms_opts = cf_options_for(CF_FTS_TERMS, &block_cache)?;
            let skips_opts = cf_options_for(CF_FTS_SKIPS, &block_cache)?;
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
                &mut |k, v| positions_sink.put(k, v),
            )?;

            let tf = terms_sink.finish()?;
            let sf = skips_sink.finish()?;
            let posf = positions_sink.finish()?;
            if !tf.is_empty() {
                cf_files.insert(CF_FTS_TERMS.to_owned(), tf);
            }
            if !sf.is_empty() {
                cf_files.insert(CF_FTS_SKIPS.to_owned(), sf);
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
