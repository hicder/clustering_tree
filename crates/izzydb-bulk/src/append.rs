//! Append-mode bulk import into an existing collection.

use super::*;

/// Bulk-append `parquet_files` onto an existing `target` collection directory crash-safely.
pub fn bulk_append(
    target: impl AsRef<Path>,
    parquet_files: &[PathBuf],
    config: BulkImportConfig,
    progress: Option<&dyn Fn(Progress)>,
) -> Result<BulkAppendReport> {
    bulk_append_checked(target.as_ref(), parquet_files, config, None, progress)
}

#[doc(hidden)]
pub fn bulk_append_with_hook(
    target: impl AsRef<Path>,
    parquet_files: &[PathBuf],
    config: BulkImportConfig,
    mut hook: impl FnMut(BulkAppendHookStage) -> Result<()>,
    progress: Option<&dyn Fn(Progress)>,
) -> Result<BulkAppendReport> {
    bulk_append_checked(
        target.as_ref(),
        parquet_files,
        config,
        Some(&mut hook),
        progress,
    )
}

pub(super) type AppendHook<'a> = &'a mut dyn FnMut(BulkAppendHookStage) -> Result<()>;

pub(super) fn bulk_append_checked(
    target: &Path,
    parquet_files: &[PathBuf],
    config: BulkImportConfig,
    mut hook: Option<AppendHook<'_>>,
    progress: Option<&dyn Fn(Progress)>,
) -> Result<BulkAppendReport> {
    config.validate()?;
    let mut timings = Vec::new();

    report_progress(progress, "prepare", 0, None);
    let t0 = Instant::now();
    let mut col = Collection::open(target, config.collection.clone())?;
    let schema = col.schema().clone();
    let seg_cfg = col.collection_config().segment;
    let seal_rows = seg_cfg.seal_rows as usize;
    let has_vectors = config.collection.vector.is_some();

    let threads = config.threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| TreeError::Storage(format!("build rayon thread pool: {e}")))?;

    let stage_dir = config.work_dir.join("append_stage");
    let bundle_dir = config.work_dir.join("append_ssts");
    for d in [&stage_dir, &bundle_dir] {
        if d.exists() {
            fs::remove_dir_all(d).map_err(|e| TreeError::Storage(format!("clean {}: {e}", d.display())))?;
        }
        fs::create_dir_all(d).map_err(|e| TreeError::Storage(format!("create {}: {e}", d.display())))?;
    }
    timings.push(("prepare", t0.elapsed()));

    // Phase 1: Stage Parquet files.
    report_progress(progress, "scan_and_stage", 0, None);
    let t1 = Instant::now();
    let vec_scan_cfg = config
        .collection
        .vector
        .as_ref()
        .map(|v| (v.dim, v.metric, config.vector_column.as_deref().unwrap()));
    let staged = scan_and_stage(
        parquet_files,
        &schema,
        vec_scan_cfg,
        &config.id_column,
        config.column_map.as_deref(),
        config.on_duplicate_id,
        &stage_dir,
        config.memory_budget_bytes,
        &pool,
    )?;
    timings.push(("scan_and_stage", t1.elapsed()));

    if staged.docs_imported == 0 {
        if !config.keep_work_dir {
            let _ = fs::remove_dir_all(&config.work_dir);
        }
        return Ok(BulkAppendReport {
            rows_read: staged.rows_read,
            docs_appended: 0,
            docs_replaced: 0,
            duplicates_dropped: staged.duplicates_dropped,
            segments_created: 0,
            rebuilt_ann_tree: false,
            sst_files: 0,
            sst_bytes: 0,
            phase_timings: timings,
        });
    }

    let vectors_mmap = MappedFile::open(&staged.vectors_path)?;
    let doc_ids_mmap = MappedFile::open(&staged.doc_ids_by_row_path)?;
    let docs_idx_mmap = MappedFile::open(&staged.docs_idx_path)?;
    let mut docs_shards = Vec::with_capacity(staged.docs_shard_paths.len());
    for p in &staged.docs_shard_paths {
        docs_shards.push(MappedFile::open(p)?);
    }
    let rows_by_doc_id_mmap = MappedFile::open(&staged.rows_by_doc_id_path)?;

    let n_docs = staged.docs_imported as usize;
    let rb_bytes = rows_by_doc_id_mmap.as_bytes();
    let did_bytes = doc_ids_mmap.as_bytes();
    let mut new_doc_mappings: Vec<(DocId, RowId, u32)> = Vec::with_capacity(n_docs);
    let num_new_segs = n_docs.div_ceil(seal_rows) as u32;

    // Reserve start_seg..end_seg as Building before emitting/ingesting SSTs.
    let start_seg = col.storage().get_manifest()?.map_or(0, |m| m.next_segment_id);
    let end_seg = start_seg
        .checked_add(num_new_segs)
        .ok_or_else(|| TreeError::Storage("SegmentId u32 overflow in bulk_append".into()))?;

    let mut all_new_doc_ids = Vec::with_capacity(n_docs);
    for i in 0..n_docs {
        let staged_row = u32::from_le_bytes(rb_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        let doc_id = read_doc_id_at(did_bytes, staged_row);
        let row_id = RowId::new(start_seg + (i / seal_rows) as SegmentId, (i % seal_rows) as LocalId);
        new_doc_mappings.push((doc_id, row_id, staged_row));
        all_new_doc_ids.push(doc_id);
    }

    // Lookup overlapping DocIds to replace.
    let old_row_opts = col.storage().multi_get_id_map(&all_new_doc_ids)?;
    let replaced_old_rows: Vec<RowId> = old_row_opts.iter().flatten().copied().collect();
    let docs_replaced = replaced_old_rows.len() as u64;
    let replaced_set: HashSet<RowId> = replaced_old_rows.iter().copied().collect();

    // Collect existing detached gens & FTS terms / meta.
    let mut detached_gens: HashMap<AttrId, ColumnGen> = HashMap::new();
    for e in schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
    {
        let g = col
            .storage()
            .get_detached_meta(e.id)?
            .map_or(0, |m| m.current_gen);
        detached_gens.insert(e.id, g);
    }

    let existing_fts_meta = if schema.has_fts() {
        match col.storage().get_fts_meta_bytes()? {
            Some(b) => FtsMeta::decode(&b)?,
            None => FtsMeta::default(),
        }
    } else {
        FtsMeta::default()
    };
    let mut existing_fts_terms: HashMap<(AttrId, String), TermRecord> = HashMap::new();
    if schema.has_fts() {
        for (k, v) in col.storage().all_fts_terms()? {
            if k.len() >= 2 {
                let aid = u16::from_be_bytes([k[0], k[1]]);
                if let Ok(tok) = std::str::from_utf8(&k[2..]) {
                    existing_fts_terms.insert((aid, tok.to_owned()), TermRecord::decode(&v)?);
                }
            }
        }
    }

    // Check ANN strategy: incremental insert (<= 30%) vs offline full rebuild (> 30% or empty).
    let existing_live_vecs = col.ann_meta().map_or(0, |m| m.num_vectors);
    let rebuild_ann = has_vectors
        && (existing_live_vecs == 0 || (n_docs as f64) > 0.30 * (existing_live_vecs as f64));
    let mut rebuilt_ann_payload = None;
    if rebuild_ann {
        let vcfg = config.collection.vector.as_ref().unwrap();
        let dim = vcfg.dim;
        let combined_vec_path = stage_dir.join("rebuild_vectors.bin");
        let combined_file = File::create(&combined_vec_path).map_err(map_io)?;
        let mut combined_writer = BufWriter::with_capacity(256 * 1024, combined_file);
        let mut rebuild_entries: Vec<VecEntry> = Vec::new();
        let mut next_vec_idx: u32 = 0;

        // Collect existing surviving live vectors.
        let snap = col.segment_snapshot();
        let mut existing_rows = Vec::new();
        for (&seg, entry) in &snap.segments {
            if !entry.is_visible() {
                continue;
            }
            for local in 0..entry.meta.row_count {
                let r = RowId::new(seg, local);
                let is_del = entry.deleted.as_ref().is_some_and(|bm| bm.contains(local));
                if !is_del && !replaced_set.contains(&r) {
                    existing_rows.push(r);
                }
            }
        }
        for chunk in existing_rows.chunks(4096) {
            let pinned = col.storage().multi_get_vectors_pinned(chunk)?;
            for (&row_id, opt_slice) in chunk.iter().zip(pinned) {
                if let Some(slice) = opt_slice {
                    combined_writer.write_all(&slice).map_err(map_io)?;
                    rebuild_entries.push(VecEntry {
                        row_id,
                        vec_idx: next_vec_idx,
                    });
                    next_vec_idx += 1;
                }
            }
        }

        // Append newly staged vectors.
        let vec_bytes = vectors_mmap.as_bytes();
        let v_byte_len = dim * 4;
        for &(_, row_id, staged_row) in &new_doc_mappings {
            let off = (staged_row as usize) * v_byte_len;
            combined_writer
                .write_all(&vec_bytes[off..off + v_byte_len])
                .map_err(map_io)?;
            rebuild_entries.push(VecEntry {
                row_id,
                vec_idx: next_vec_idx,
            });
            next_vec_idx += 1;
        }
        combined_writer.flush().map_err(map_io)?;
        drop(combined_writer);

        let combined_mmap = MappedFile::open(&combined_vec_path)?;
        let new_seed = rand::random::<u64>();
        let quantizer = RabitqQuantizer::new(dim, new_seed);
        let built = build_tree_and_assign(
            &rebuild_entries,
            combined_mmap.as_bytes(),
            vcfg,
            config.fill_factor,
            config.train_sample,
            config.memory_budget_bytes,
            new_seed,
            &pool,
        )?;
        let cbytes = combined_mmap.as_bytes();
        let packed_leaves: Vec<(NodeId, PackedLeafNode)> = pool.install(|| {
            built
                .leaves
                .par_iter()
                .map(|leaf| {
                    let mut scratch = EncodeScratch::default();
                    let mut rids = Vec::with_capacity(leaf.entries.len());
                    let mut flat = Vec::with_capacity(leaf.entries.len() * dim);
                    for entry in &leaf.entries {
                        rids.push(entry.row_id);
                        flat.extend_from_slice(read_vector_at(cbytes, entry.vec_idx, dim));
                    }
                    let packed = PackedLeafNode::from_entries(
                        leaf.parent_id,
                        leaf.centroid.clone(),
                        &rids,
                        &flat,
                        &quantizer,
                        dim,
                        &mut scratch,
                    );
                    (leaf.leaf_id, packed)
                })
                .collect()
        });
        rebuilt_ann_payload = Some((built.meta, built.internal_nodes, packed_leaves));
    }

    // Step 1: Persist SegmentState::Building + next_segment_id synchronously.
    let created_seq = col.storage().latest_sequence_number();
    col.with_bulk_tx(|tx, _| {
        for rel_seg in 0..num_new_segs {
            let seg = start_seg + rel_seg;
            let ord_start = (rel_seg as usize) * seal_rows;
            let ord_end = ((rel_seg as usize + 1) * seal_rows).min(n_docs);
            let mut bm = SegmentMeta::new_open(created_seq);
            bm.state = SegmentState::Building;
            bm.row_count = (ord_end - ord_start) as u32;
            tx.put_seg_meta(seg, bm);
        }
        tx.manifest_mut().next_segment_id = end_seg;
        Ok(())
    })?;
    if let Some(ref mut h) = hook {
        h(BulkAppendHookStage::AfterBuildingMeta)?;
    }

    // Step 2: Emit & ingest disjoint SSTs for [start_seg..end_seg).
    let t2 = Instant::now();
    drop(rows_by_doc_id_mmap);
    let mut seg_out = emit_segment_range_ssts(
        &bundle_dir,
        &stage_dir,
        &staged,
        &schema,
        &seg_cfg,
        start_seg,
        num_new_segs,
        has_vectors,
        config.collection.vector.as_ref().map_or(0, |v| v.dim),
        &detached_gens,
        &existing_fts_terms,
        existing_fts_meta.next_term_id,
        false,
        true,
        vectors_mmap.as_bytes(),
        doc_ids_mmap.as_bytes(),
        docs_idx_mmap.as_bytes(),
        &docs_shards,
        config.target_sst_bytes,
        config.memory_budget_bytes,
        &pool,
    )?;
    timings.push(("emit_ssts", t2.elapsed()));

    let move_files = same_filesystem(&bundle_dir, target);
    let mut sst_files = 0usize;
    let mut sst_bytes = 0u64;
    for cf in ALL_DATA_CF_NAMES {
        let Some(infos) = seg_out.cf_files.get(cf) else {
            continue;
        };
        if infos.is_empty() {
            continue;
        }
        let paths: Vec<PathBuf> = infos
            .iter()
            .map(|info| {
                sst_files += 1;
                sst_bytes += info.bytes;
                bundle_dir.join(&info.rel_path)
            })
            .collect();
        col.storage().ingest_cf_with_move(cf, &paths, move_files)?;
        if let Some(ref mut h) = hook {
            h(BulkAppendHookStage::AfterIngestCf(cf))?;
        }
    }

    if let Some(ref mut h) = hook {
        h(BulkAppendHookStage::BeforeFinalCommit)?;
    }

    // Step 3: Atomic TxContext commit: tombstone replaced rows, flip Building -> Sealed,
    // update cf_id_map, zones, counts, FTS terms/meta, and ANN tree.
    let t3 = Instant::now();
    {
        let dim = config.collection.vector.as_ref().map_or(0, |v| v.dim);
        let sealed_seq = col.storage().latest_sequence_number();
        let vbytes = vectors_mmap.as_bytes();
        col.with_bulk_tx(|tx, helper| {
            for old_row in replaced_old_rows {
                helper.delete_row_for_replace(tx, old_row)?;
            }

            for (seg, mut meta) in seg_out.seg_metas.drain(..) {
                meta.created_seq = created_seq;
                meta.sealed_seq = sealed_seq;
                tx.put_seg_meta(seg, meta);
            }

            tx.manifest_mut().live_rows += n_docs as u64;
            tx.manifest_mut().total_rows += n_docs as u64;

            for &(doc_id, new_row, _) in &new_doc_mappings {
                tx.put_id_map(doc_id, new_row);
            }

            for (seg, aid, tag, z) in seg_out.zones {
                tx.set_zone_entry(seg, aid, tag, z);
            }
            for (aid, rblk, bytes) in seg_out.doc_len_blocks {
                tx.put_fts_doc_len_bytes(aid, rblk, bytes);
            }
            for (ak, cnt) in seg_out.attr_count_deltas {
                tx.adjust_attr_count(&ak, cnt as i64);
            }
            for (suffix, cnt) in seg_out.detached_count_deltas {
                if suffix.len() >= 7 {
                    let col_gen = u32::from_be_bytes(suffix[2..6].try_into().unwrap());
                    let mut ak = Vec::with_capacity(suffix.len() - 4);
                    ak.extend_from_slice(&suffix[0..2]);
                    ak.extend_from_slice(&suffix[6..]);
                    tx.adjust_detached_count(col_gen, &ak, cnt as i64);
                }
            }

            if schema.has_fts() {
                for (tk, bytes) in seg_out.new_fts_terms {
                    tx.put_fts_term_bytes(tk, bytes);
                }
                for delta in seg_out.updated_fts_terms {
                    if let Some(tbytes) = tx.get_fts_term_bytes(&delta.term_key)? {
                        let mut rec = TermRecord::decode(&tbytes)?;
                        if let Some((old_ib_id, old_ib_bytes)) = rec.inline_block.take() {
                            let hdr = PostingBlockHeader::parse(&old_ib_bytes)?;
                            tx.put_fts_posting_bytes(rec.term_id, old_ib_id, old_ib_bytes);
                            let chunk = old_ib_id.chunk();
                            let mut skip = match tx.get_fts_skip_bytes(rec.term_id, chunk)? {
                                Some(sb) => SkipBlock::decode(&sb)?,
                                None => SkipBlock::default(),
                            };
                            skip.upsert(old_ib_id.rel_block(), hdr.max_tf, hdr.min_dl_over_tf);
                            tx.put_fts_skip_bytes(rec.term_id, chunk, skip.encode());
                        }
                        rec.stats.doc_freq += delta.add_doc_freq;
                        rec.stats.max_tf = rec.stats.max_tf.max(delta.max_tf);
                        rec.stats.min_dl_over_tf =
                            rec.stats.min_dl_over_tf.min(delta.min_dl_over_tf);
                        tx.put_fts_term_bytes(delta.term_key, rec.encode());
                    }
                }
                let mut fts_meta = match tx.get_fts_meta_bytes()? {
                    Some(b) => FtsMeta::decode(&b)?,
                    None => FtsMeta::default(),
                };
                fts_meta.next_term_id = seg_out.next_fts_term_id;
                for (aid, delta) in seg_out.fts_attr_stats {
                    let e = fts_meta.per_attr_stats.entry(aid).or_default();
                    e.doc_count += delta.doc_count;
                    e.total_tokens += delta.total_tokens;
                }
                tx.put_fts_meta_bytes(fts_meta.encode());
            }

            if has_vectors {
                if let Some((new_meta, internals, leaves)) = rebuilt_ann_payload {
                    helper.replace_ann_tree(tx, new_meta, internals, leaves)?;
                } else {
                    for &(_, row_id, staged_row) in &new_doc_mappings {
                        let v = read_vector_at(vbytes, staged_row, dim);
                        helper.insert_vector(tx, row_id, v)?;
                    }
                }
            }
            Ok(())
        })?;
    }
    timings.push(("commit", t3.elapsed()));

    drop(vectors_mmap);
    drop(doc_ids_mmap);
    drop(docs_idx_mmap);
    drop(docs_shards);

    if config.verify_after {
        col.verify_invariants()?;
    }

    if !config.keep_work_dir {
        let _ = fs::remove_dir_all(&config.work_dir);
    }

    Ok(BulkAppendReport {
        rows_read: staged.rows_read,
        docs_appended: staged.docs_imported,
        docs_replaced,
        duplicates_dropped: staged.duplicates_dropped,
        segments_created: num_new_segs,
        rebuilt_ann_tree: rebuild_ann,
        sst_files,
        sst_bytes,
        phase_timings: timings,
    })
}
