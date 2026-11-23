//! Parallel offline Parquet bulk importer and appender for `izzydb` (v3 segmented storage).

pub mod common;
mod pipeline;
mod tree_builder;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use izzydb::Collection;
use izzydb::codec::{SegmentMeta, SegmentState};
use izzydb::config::{AttrId, ColumnGen, DocId, LocalId, NodeId, RowId, SegmentId};
use izzydb::error::{Result, TreeError};
use izzydb::fts::{FtsMeta, PostingBlockHeader, SkipBlock, TermRecord};
use izzydb::node::PackedLeafNode;
use izzydb::rabitq::{EncodeScratch, RabitqQuantizer};
use izzydb::schema::{CollectionConfig, CollectionSchema, ColumnStorage};
use izzydb::storage::{ALL_DATA_CF_NAMES, DEFAULT_CF_NAME, Storage};

pub use common::parquet_reader::scan_detached_from_parquet;
pub use common::sst::{
    ManifestAnnMeta, ManifestSegmentInfo, SST_BUNDLE_FORMAT_VERSION, SstBundle, SstManifest,
};

use common::parquet_reader::{MappedFile, read_doc_id_at, scan_and_stage};
use common::sst::{SstFileInfo, compute_config_hash};
use pipeline::{emit_empty_import_ssts, emit_segment_range_ssts};
use tree_builder::{VecEntry, build_tree_and_assign, read_vector_at};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DuplicatePolicy {
    #[default]
    Error,
    LastWins,
}

#[derive(Debug, Clone)]
pub struct Progress {
    pub phase: &'static str,
    pub completed: u64,
    pub total: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct BulkImportConfig {
    pub collection: CollectionConfig,
    pub id_column: String,
    pub vector_column: Option<String>,
    /// Parquet column -> schema attribute name; `None` maps every schema attribute by the same name.
    pub column_map: Option<Vec<(String, String)>>,
    pub on_duplicate_id: DuplicatePolicy,
    pub fill_factor: f32,
    pub train_sample: usize,
    pub work_dir: PathBuf,
    pub keep_work_dir: bool,
    pub memory_budget_bytes: usize,
    pub target_sst_bytes: u64,
    pub threads: usize,
    pub verify_after: bool,
}

impl BulkImportConfig {
    pub fn new(
        collection: CollectionConfig,
        id_column: impl Into<String>,
        vector_column: impl Into<String>,
        work_dir: impl Into<PathBuf>,
    ) -> Self {
        let vec_col = if collection.vector.is_some() {
            Some(vector_column.into())
        } else {
            None
        };
        Self {
            collection,
            id_column: id_column.into(),
            vector_column: vec_col,
            column_map: None,
            on_duplicate_id: DuplicatePolicy::Error,
            fill_factor: 0.75,
            train_sample: 1_000_000,
            work_dir: work_dir.into(),
            keep_work_dir: false,
            memory_budget_bytes: 2 * 1024 * 1024 * 1024,
            target_sst_bytes: 256 * 1024 * 1024,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            verify_after: false,
        }
    }

    pub fn new_vectorless(
        collection: CollectionConfig,
        id_column: impl Into<String>,
        work_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            collection,
            id_column: id_column.into(),
            vector_column: None,
            column_map: None,
            on_duplicate_id: DuplicatePolicy::Error,
            fill_factor: 0.75,
            train_sample: 1_000_000,
            work_dir: work_dir.into(),
            keep_work_dir: false,
            memory_budget_bytes: 2 * 1024 * 1024 * 1024,
            target_sst_bytes: 256 * 1024 * 1024,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            verify_after: false,
        }
    }

    pub fn with_vector_column(mut self, col: impl Into<String>) -> Self {
        self.vector_column = Some(col.into());
        self
    }

    pub fn with_column_map(mut self, column_map: Vec<(String, String)>) -> Self {
        self.column_map = Some(column_map);
        self
    }

    pub fn with_on_duplicate_id(mut self, policy: DuplicatePolicy) -> Self {
        self.on_duplicate_id = policy;
        self
    }

    pub fn with_fill_factor(mut self, fill_factor: f32) -> Self {
        self.fill_factor = fill_factor;
        self
    }

    pub fn with_train_sample(mut self, train_sample: usize) -> Self {
        self.train_sample = train_sample;
        self
    }

    pub fn with_keep_work_dir(mut self, keep: bool) -> Self {
        self.keep_work_dir = keep;
        self
    }

    pub fn with_memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.memory_budget_bytes = bytes;
        self
    }

    pub fn with_target_sst_bytes(mut self, bytes: u64) -> Self {
        self.target_sst_bytes = bytes;
        self
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_verify_after(mut self, verify: bool) -> Self {
        self.verify_after = verify;
        self
    }

    pub fn validate(&self) -> Result<()> {
        self.collection.validate()?;
        if self.id_column.is_empty() {
            return Err(TreeError::InvalidConfig("id_column cannot be empty".into()));
        }
        if self.collection.vector.is_some() {
            let Some(ref vc) = self.vector_column else {
                return Err(TreeError::InvalidConfig(
                    "vector_column is required when CollectionConfig::vector is Some".into(),
                ));
            };
            if vc.is_empty() {
                return Err(TreeError::InvalidConfig(
                    "vector_column cannot be empty".into(),
                ));
            }
        }
        if !self.fill_factor.is_finite() || self.fill_factor <= 0.0 || self.fill_factor > 1.0 {
            return Err(TreeError::InvalidConfig(format!(
                "fill_factor must be in (0.0, 1.0], got {}",
                self.fill_factor
            )));
        }
        if self.target_sst_bytes == 0 {
            return Err(TreeError::InvalidConfig(
                "target_sst_bytes must be > 0".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BulkImportReport {
    pub rows_read: u64,
    pub docs_imported: u64,
    pub duplicates_dropped: u64,
    pub segments_created: u32,
    pub leaves: u32,
    pub internal_nodes: u32,
    pub tree_height: u8,
    pub terms: u64,
    pub sst_files: usize,
    pub sst_bytes: u64,
    pub phase_timings: Vec<(&'static str, Duration)>,
}

#[derive(Debug, Clone, Default)]
pub struct BulkAppendReport {
    pub rows_read: u64,
    pub docs_appended: u64,
    pub docs_replaced: u64,
    pub duplicates_dropped: u64,
    pub segments_created: u32,
    pub rebuilt_ann_tree: bool,
    pub sst_files: usize,
    pub sst_bytes: u64,
    pub phase_timings: Vec<(&'static str, Duration)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BulkAppendHookStage {
    AfterBuildingMeta,
    AfterIngestCf(&'static str),
    BeforeFinalCommit,
}

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

fn build_ssts_timed(
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

/// Ingest a pre-built [`SstBundle`] into an empty or non-existent `target` collection directory.
pub fn ingest_ssts(target: impl AsRef<Path>, bundle: &SstBundle) -> Result<()> {
    ingest_ssts_checked(target.as_ref(), bundle, None, None)
}

#[doc(hidden)]
pub fn ingest_ssts_with_hook(
    target: impl AsRef<Path>,
    bundle: &SstBundle,
    expected_config: Option<&CollectionConfig>,
    mut after_cf: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    ingest_ssts_checked(target.as_ref(), bundle, expected_config, Some(&mut after_cf))
}

type AfterCfHook<'a> = &'a mut dyn FnMut(&str) -> Result<()>;

fn ingest_ssts_checked(
    target: &Path,
    bundle: &SstBundle,
    expected_config: Option<&CollectionConfig>,
    mut after_cf: Option<AfterCfHook<'_>>,
) -> Result<()> {
    let col_cfg = match expected_config {
        Some(cfg) => {
            bundle.validate_against(cfg)?;
            cfg.clone()
        }
        None => {
            let schema = CollectionSchema::decode(&bundle.manifest.schema_bytes)?;
            let mut cfg = match &bundle.manifest.ann_meta {
                Some(am) => {
                    let am = am.to_ann_meta()?;
                    CollectionConfig::new(izzydb::config::TreeConfig::new(am.dim as usize, am.metric))
                }
                None => CollectionConfig::without_vectors(),
            };
            cfg.segment.seal_rows = bundle.manifest.segment_info.seal_rows;
            for entry in schema.entries() {
                cfg.attributes.push(izzydb::schema::AttributeConfig {
                    name: entry.name.clone(),
                    attr_type: entry.attr_type,
                    filterable: entry.filterable,
                    full_text_search: entry.full_text_search.clone(),
                    storage: entry.storage,
                    column: entry.column,
                });
            }
            cfg
        }
    };

    let storage = Storage::open_for_bulk_ingest(target, &col_cfg)?;
    let move_files = same_filesystem(&bundle.dir, target);
    let build_id = rand::random::<u64>();
    storage.put_bulk_import_marker(build_id)?;

    for cf in ALL_DATA_CF_NAMES {
        let files = bundle.files(cf);
        if files.is_empty() {
            continue;
        }
        storage.ingest_cf_with_move(cf, &files, move_files)?;
        if let Some(ref mut hook) = after_cf {
            hook(cf)?;
        }
    }

    let default_files = bundle.files(DEFAULT_CF_NAME);
    storage.ingest_cf_with_move(DEFAULT_CF_NAME, &default_files, move_files)?;
    if let Some(ref mut hook) = after_cf {
        hook(DEFAULT_CF_NAME)?;
    }

    storage.clear_bulk_import_marker()?;
    Ok(())
}

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

type AppendHook<'a> = &'a mut dyn FnMut(BulkAppendHookStage) -> Result<()>;

fn bulk_append_checked(
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

fn same_filesystem(a: &Path, b: &Path) -> bool {
    let Ok(ma) = fs::metadata(a) else {
        return false;
    };
    let Ok(mb) = fs::metadata(b) else {
        return false;
    };
    ma.dev() == mb.dev()
}

#[inline]
fn report_progress(
    cb: Option<&dyn Fn(Progress)>,
    phase: &'static str,
    completed: u64,
    total: Option<u64>,
) {
    if let Some(f) = cb {
        f(Progress {
            phase,
            completed,
            total,
        });
    }
}

#[inline]
fn map_io(e: std::io::Error) -> TreeError {
    TreeError::Storage(e.to_string())
}
