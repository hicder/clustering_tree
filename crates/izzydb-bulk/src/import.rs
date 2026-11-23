//! SST build and one-shot bulk import.

use super::*;

/// Bulk-import `parquet_files` into an empty or non-existent `target` collection directory.
pub fn bulk_import(
    target: impl AsRef<Path>,
    parquet_files: &[PathBuf],
    config: BulkImportConfig,
    progress: Option<&dyn Fn(Progress)>,
) -> Result<BulkImportReport> {
    let target = target.as_ref();
    config.validate()?;

    if target.exists() {
        let s = Storage::open_for_bulk_ingest(target, &config.collection)?;
        drop(s);
    }

    let (bundle, mut timings) = build_ssts_timed(parquet_files, &config, progress)?;

    report_progress(progress, "ingest", 0, None);
    let t_ingest = Instant::now();
    ingest_ssts_checked(target, &bundle, Some(&config.collection), None)?;
    timings.push(("ingest", t_ingest.elapsed()));
    report_progress(progress, "ingest", 1, Some(1));

    if config.verify_after {
        report_progress(progress, "verify", 0, None);
        let t_ver = Instant::now();
        let col = Collection::open(target, config.collection.clone())?;
        col.verify_invariants()?;
        timings.push(("verify", t_ver.elapsed()));
        report_progress(progress, "verify", 1, Some(1));
    }

    if !config.keep_work_dir {
        let _ = fs::remove_dir_all(&config.work_dir);
    }

    Ok(BulkImportReport {
        rows_read: bundle.manifest.rows_read,
        docs_imported: bundle.manifest.docs_imported,
        duplicates_dropped: bundle.manifest.duplicates_dropped,
        segments_created: bundle.manifest.segment_info.num_segments,
        leaves: bundle.manifest.leaves,
        internal_nodes: bundle.manifest.internal_nodes,
        tree_height: bundle.manifest.tree_height,
        terms: bundle.manifest.terms,
        sst_files: bundle.manifest.sst_files,
        sst_bytes: bundle.manifest.sst_bytes,
        phase_timings: timings,
    })
}

/// Build an offline [`SstBundle`] under `config.work_dir` without touching any target DB.
pub fn build_ssts(parquet_files: &[PathBuf], config: &BulkImportConfig) -> Result<SstBundle> {
    let (bundle, _) = build_ssts_timed(parquet_files, config, None)?;
    Ok(bundle)
}

pub(super) fn build_ssts_timed(
    parquet_files: &[PathBuf],
    config: &BulkImportConfig,
    progress: Option<&dyn Fn(Progress)>,
) -> Result<(SstBundle, Vec<(&'static str, Duration)>)> {
    config.validate()?;
    let mut timings = Vec::new();

    report_progress(progress, "prepare", 0, None);
    let t0 = Instant::now();
    let schema = CollectionSchema::from_config(&config.collection)?;
    let threads = config.threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| TreeError::Storage(format!("build rayon thread pool: {e}")))?;

    let stage_dir = config.work_dir.join("stage");
    let bundle_dir = config.work_dir.join("ssts");
    for d in [&stage_dir, &bundle_dir] {
        if d.exists() {
            fs::remove_dir_all(d).map_err(|e| TreeError::Storage(format!("clean {}: {e}", d.display())))?;
        }
        fs::create_dir_all(d).map_err(|e| TreeError::Storage(format!("create {}: {e}", d.display())))?;
    }
    timings.push(("prepare", t0.elapsed()));

    // Phase 1: Scan and stage.
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
    report_progress(progress, "scan_and_stage", staged.docs_imported, Some(staged.rows_read));

    let vectors_mmap = MappedFile::open(&staged.vectors_path)?;
    let doc_ids_mmap = MappedFile::open(&staged.doc_ids_by_row_path)?;
    let docs_idx_mmap = MappedFile::open(&staged.docs_idx_path)?;
    let mut docs_shards = Vec::with_capacity(staged.docs_shard_paths.len());
    for p in &staged.docs_shard_paths {
        docs_shards.push(MappedFile::open(p)?);
    }
    let rows_by_doc_id_mmap = MappedFile::open(&staged.rows_by_doc_id_path)?;

    // Phases 2 & 3: Build global ANN tree (if vectors configured).
    let t2 = Instant::now();
    let rabitq_seed = rand::random::<u64>();
    let quantizer = config
        .collection
        .vector
        .as_ref()
        .map(|v| RabitqQuantizer::new(v.dim, rabitq_seed));
    let built_tree = if let Some(ref vcfg) = config.collection.vector {
        report_progress(progress, "build_tree", 0, Some(staged.docs_imported));
        let seal_rows = config.collection.segment.seal_rows as usize;
        let rb_bytes = rows_by_doc_id_mmap.as_bytes();
        let n_docs = staged.docs_imported as usize;
        let mut entries = Vec::with_capacity(n_docs);
        for i in 0..n_docs {
            let staged_row = u32::from_le_bytes(rb_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
            let row_id = RowId::new((i / seal_rows) as SegmentId, (i % seal_rows) as LocalId);
            entries.push(VecEntry {
                row_id,
                vec_idx: staged_row,
            });
        }
        let bt = build_tree_and_assign(
            &entries,
            vectors_mmap.as_bytes(),
            vcfg,
            config.fill_factor,
            config.train_sample,
            config.memory_budget_bytes,
            rabitq_seed,
            &pool,
        )?;
        report_progress(progress, "build_tree", staged.docs_imported, Some(staged.docs_imported));
        Some(bt)
    } else {
        None
    };
    timings.push(("build_tree", t2.elapsed()));

    // Phases 4, 5, 6: Emit SSTs.
    report_progress(progress, "emit_ssts", 0, Some(staged.docs_imported));
    let t3 = Instant::now();
    drop(rows_by_doc_id_mmap);
    let (emitted_files, manifest, ann_meta, num_segments, terms_count) = emit_empty_import_ssts(
        &bundle_dir,
        &stage_dir,
        &staged,
        built_tree.as_ref(),
        config.collection.vector.as_ref(),
        quantizer.as_ref(),
        &schema,
        &config.collection.segment,
        vectors_mmap.as_bytes(),
        doc_ids_mmap.as_bytes(),
        docs_idx_mmap.as_bytes(),
        &docs_shards,
        config.target_sst_bytes,
        config.memory_budget_bytes,
        &pool,
    )?;
    timings.push(("emit_ssts", t3.elapsed()));
    report_progress(progress, "emit_ssts", staged.docs_imported, Some(staged.docs_imported));

    drop(vectors_mmap);
    drop(doc_ids_mmap);
    drop(docs_idx_mmap);
    drop(docs_shards);

    if !config.keep_work_dir {
        let _ = fs::remove_dir_all(&stage_dir);
    }

    let mut cf_file_names: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut sst_files = 0usize;
    let mut sst_bytes = 0u64;
    for (cf, infos) in emitted_files {
        let mut names = Vec::with_capacity(infos.len());
        for SstFileInfo { rel_path, bytes, .. } in infos {
            sst_files += 1;
            sst_bytes += bytes;
            names.push(rel_path);
        }
        if !names.is_empty() {
            cf_file_names.insert(cf, names);
        }
    }

    let sst_manifest = SstManifest {
        format_version: SST_BUNDLE_FORMAT_VERSION,
        config_hash: compute_config_hash(&config.collection)?,
        has_vectors: config.collection.vector.is_some(),
        has_fts: schema.has_fts(),
        schema_bytes: schema.encode(),
        segment_info: ManifestSegmentInfo::from_manifest(&manifest, num_segments),
        ann_meta: ann_meta.as_ref().map(ManifestAnnMeta::from),
        cf_files: cf_file_names,
        rows_read: staged.rows_read,
        docs_imported: staged.docs_imported,
        duplicates_dropped: staged.duplicates_dropped,
        leaves: built_tree.as_ref().map_or(0, |t| t.leaves.len() as u32),
        internal_nodes: built_tree
            .as_ref()
            .map_or(0, |t| t.internal_nodes.len() as u32),
        tree_height: built_tree.as_ref().map_or(0, |t| t.meta.root_level),
        terms: terms_count,
        sst_files,
        sst_bytes,
    };

    let bundle = SstBundle {
        dir: bundle_dir,
        manifest: sst_manifest,
    };
    bundle.save_manifest()?;

    Ok((bundle, timings))
}
