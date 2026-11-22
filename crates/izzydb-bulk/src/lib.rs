//! Parallel offline Parquet bulk importer for `izzydb` via external RocksDB SST ingestion.

mod parquet_reader;
mod pipeline;
mod sorter;
mod sst;
mod tree_builder;

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use izzydb::Collection;
use izzydb::error::{Result, TreeError};
use izzydb::rabitq::RabitqQuantizer;
use izzydb::schema::{CollectionConfig, CollectionSchema};
use izzydb::storage::{ALL_DATA_CF_NAMES, DEFAULT_CF_NAME, Storage};

pub use sst::{ManifestTreeMeta, SST_BUNDLE_FORMAT_VERSION, SstBundle, SstManifest};

use parquet_reader::{MappedFile, scan_and_stage};
use pipeline::emit_all_ssts;
use sst::{SstFileInfo, compute_config_hash};
use tree_builder::build_tree_and_assign;

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
    pub vector_column: String,
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
        Self {
            collection,
            id_column: id_column.into(),
            vector_column: vector_column.into(),
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
        if self.vector_column.is_empty() {
            return Err(TreeError::InvalidConfig(
                "vector_column cannot be empty".into(),
            ));
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
    pub leaves: u32,
    pub internal_nodes: u32,
    pub tree_height: u8,
    pub terms: u64,
    pub sst_files: usize,
    pub sst_bytes: u64,
    pub phase_timings: Vec<(&'static str, Duration)>,
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

    // Reject non-empty target before doing any work or creating any files on target.
    if target.exists() {
        let s = Storage::open_for_bulk_ingest(target, &config.collection.tree)?;
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
    let rabitq_seed = rand::random::<u64>();
    let quantizer = RabitqQuantizer::new(config.collection.tree.dim, rabitq_seed);
    let threads = config.threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| TreeError::Storage(format!("build rayon thread pool: {e}")))?;

    let stage_dir = config.work_dir.join("stage");
    let bundle_dir = config.work_dir.join("ssts");
    if stage_dir.exists() {
        fs::remove_dir_all(&stage_dir)
            .map_err(|e| TreeError::Storage(format!("clean stage dir: {e}")))?;
    }
    if bundle_dir.exists() {
        fs::remove_dir_all(&bundle_dir)
            .map_err(|e| TreeError::Storage(format!("clean bundle dir: {e}")))?;
    }
    fs::create_dir_all(&stage_dir)
        .map_err(|e| TreeError::Storage(format!("create stage dir: {e}")))?;
    fs::create_dir_all(&bundle_dir)
        .map_err(|e| TreeError::Storage(format!("create bundle dir: {e}")))?;
    timings.push(("prepare", t0.elapsed()));

    // Phase 1: Scan and stage.
    report_progress(progress, "scan_and_stage", 0, None);
    let t1 = Instant::now();
    let staged = scan_and_stage(
        parquet_files,
        &schema,
        config.collection.tree.dim,
        config.collection.tree.metric,
        &config.id_column,
        &config.vector_column,
        config.column_map.as_deref(),
        config.on_duplicate_id,
        &stage_dir,
        config.memory_budget_bytes,
        &pool,
    )?;
    timings.push(("scan_and_stage", t1.elapsed()));
    report_progress(
        progress,
        "scan_and_stage",
        staged.docs_imported,
        Some(staged.rows_read),
    );

    // Map staged arrays for Phases 2-5.
    let vectors_mmap = MappedFile::open(&staged.vectors_path)?;
    let doc_ids_mmap = MappedFile::open(&staged.doc_ids_by_row_path)?;
    let docs_idx_mmap = MappedFile::open(&staged.docs_idx_path)?;
    let mut docs_shards = Vec::with_capacity(staged.docs_shard_paths.len());
    for p in &staged.docs_shard_paths {
        docs_shards.push(MappedFile::open(p)?);
    }

    // Phases 2 & 3: Train skeleton, route all live vectors, and repair tree.
    report_progress(progress, "build_tree", 0, Some(staged.docs_imported));
    let t2 = Instant::now();
    let built_tree = build_tree_and_assign(
        &staged.live_rows,
        staged.rows_read,
        vectors_mmap.as_bytes(),
        doc_ids_mmap.as_bytes(),
        &config.collection.tree,
        config.fill_factor,
        config.train_sample,
        config.memory_budget_bytes,
        rabitq_seed,
        schema.has_fts(),
        &stage_dir,
        &pool,
    )?;
    timings.push(("build_tree", t2.elapsed()));
    report_progress(
        progress,
        "build_tree",
        staged.docs_imported,
        Some(staged.docs_imported),
    );

    // Phases 4, 5, 6: Emit SSTs.
    report_progress(progress, "emit_ssts", 0, Some(staged.docs_imported));
    let t3 = Instant::now();
    let emitted = emit_all_ssts(
        &bundle_dir,
        &stage_dir,
        &staged,
        &built_tree,
        &schema,
        &quantizer,
        vectors_mmap.as_bytes(),
        doc_ids_mmap.as_bytes(),
        docs_idx_mmap.as_bytes(),
        &docs_shards,
        config.target_sst_bytes,
        config.memory_budget_bytes,
        &pool,
    )?;
    timings.push(("emit_ssts", t3.elapsed()));
    report_progress(
        progress,
        "emit_ssts",
        staged.docs_imported,
        Some(staged.docs_imported),
    );

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
    for (cf, infos) in emitted.cf_files {
        let mut names = Vec::with_capacity(infos.len());
        for SstFileInfo {
            rel_path, bytes, ..
        } in infos
        {
            sst_files += 1;
            sst_bytes += bytes;
            names.push(rel_path);
        }
        if !names.is_empty() {
            cf_file_names.insert(cf, names);
        }
    }

    let manifest = SstManifest {
        format_version: SST_BUNDLE_FORMAT_VERSION,
        config_hash: compute_config_hash(&config.collection)?,
        dim: config.collection.tree.dim,
        metric: config.collection.tree.metric.as_u8(),
        schema_bytes: schema.encode(),
        tree_meta: ManifestTreeMeta::from(&built_tree.meta),
        cf_files: cf_file_names,
        rows_read: staged.rows_read,
        docs_imported: staged.docs_imported,
        duplicates_dropped: staged.duplicates_dropped,
        leaves: built_tree.leaves.len() as u32,
        internal_nodes: built_tree.internal_nodes.len() as u32,
        tree_height: built_tree.meta.root_level,
        terms: emitted.terms_count,
        sst_files,
        sst_bytes,
    };

    let bundle = SstBundle {
        dir: bundle_dir,
        manifest,
    };
    bundle.save_manifest()?;

    Ok((bundle, timings))
}

/// Ingest a pre-built [`SstBundle`] into an empty or non-existent `target` collection directory.
pub fn ingest_ssts(target: impl AsRef<Path>, bundle: &SstBundle) -> Result<()> {
    ingest_ssts_checked(target.as_ref(), bundle, None, None)
}

/// Test/diagnostic variant of [`ingest_ssts`] that validates against `expected_config` and
/// invokes `after_cf(cf_name)` after each CF ingest (allowing crash-injection tests).
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
    if let Some(cfg) = expected_config {
        bundle.validate_against(cfg)?;
    }

    let tree_meta = bundle.manifest.tree_meta.to_tree_meta()?;
    let mut tree_cfg = izzydb::config::TreeConfig::new(tree_meta.dim as usize, tree_meta.metric);
    if let Some(cfg) = expected_config {
        tree_cfg = cfg.tree.clone();
    }

    let storage = Storage::open_for_bulk_ingest(target, &tree_cfg)?;
    let move_files = same_filesystem(&bundle.dir, target);
    if !move_files {
        eprintln!(
            "izzydb-bulk: bundle dir {} and target {} are on different filesystems; copying SSTs instead of hard-linking",
            bundle.dir.display(),
            target.display()
        );
    }

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
