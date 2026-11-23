use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use izzydb::Collection;
use izzydb::config::{DistanceMetric, TreeConfig};
use izzydb::detached::{DetachedDuplicatePolicy, DetachedWriteMode};
use izzydb::schema::CollectionConfig;
use izzydb_bulk::{
    BulkImportConfig, DuplicatePolicy, bulk_append, bulk_import, scan_detached_from_parquet,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMetric {
    L2,
    Cosine,
    Dot,
}

impl From<CliMetric> for DistanceMetric {
    fn from(m: CliMetric) -> Self {
        match m {
            CliMetric::L2 => DistanceMetric::L2Squared,
            CliMetric::Cosine => DistanceMetric::Cosine,
            CliMetric::Dot => DistanceMetric::DotProduct,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliDup {
    Error,
    LastWins,
}

impl From<CliDup> for DuplicatePolicy {
    fn from(d: CliDup) -> Self {
        match d {
            CliDup::Error => DuplicatePolicy::Error,
            CliDup::LastWins => DuplicatePolicy::LastWins,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliDetachedMode {
    Replace,
    Merge,
}

impl From<CliDetachedMode> for DetachedWriteMode {
    fn from(m: CliDetachedMode) -> Self {
        match m {
            CliDetachedMode::Replace => DetachedWriteMode::Replace,
            CliDetachedMode::Merge => DetachedWriteMode::Merge,
        }
    }
}

#[derive(Parser)]
#[command(
    name = "izzydb-import",
    about = "Bulk-import, append, and column-ingest Parquet files into izzydb"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Args)]
struct CommonParquetArgs {
    /// Target izzydb collection directory.
    #[arg(long)]
    target: PathBuf,
    /// Input Parquet files.
    #[arg(long, required = true, num_args = 1..)]
    input: Vec<PathBuf>,
    /// Vector dimensionality (omit for vectorless collections).
    #[arg(long)]
    dim: Option<usize>,
    /// Distance metric.
    #[arg(long, value_enum, default_value_t = CliMetric::Cosine)]
    metric: CliMetric,
    /// ID column name.
    #[arg(long, default_value = "id")]
    id_column: String,
    /// Vector column name.
    #[arg(long, default_value = "vector")]
    vector_column: String,
    /// Duplicate ID policy.
    #[arg(long, value_enum, default_value_t = CliDup::Error)]
    on_duplicate_id: CliDup,
    /// Leaf fill factor in (0.0, 1.0].
    #[arg(long, default_value_t = 0.75)]
    fill_factor: f32,
    /// Training sample size for tree skeleton.
    #[arg(long, default_value_t = 1_000_000)]
    train_sample: usize,
    /// Working directory for staging and SST files.
    #[arg(long)]
    work_dir: Option<PathBuf>,
    /// Memory budget in MiB.
    #[arg(long, default_value_t = 2048)]
    memory_mb: usize,
    /// Target SST file size in MiB.
    #[arg(long, default_value_t = 256)]
    target_sst_mb: u64,
    /// Worker thread count (0 = automatic).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Run invariant verification after operation.
    #[arg(long)]
    verify: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Import Parquet files into an empty or non-existent target collection.
    Import(CommonParquetArgs),
    /// Append Parquet files onto an existing collection (creating new sealed segments).
    Append(CommonParquetArgs),
    /// Whole-column ingest of detached columns from Parquet files into an existing collection.
    Columns {
        /// Target izzydb collection directory.
        #[arg(long)]
        target: PathBuf,
        /// Input Parquet files.
        #[arg(long, required = true, num_args = 1..)]
        input: Vec<PathBuf>,
        /// ID column name.
        #[arg(long, default_value = "id")]
        id_column: String,
        /// Detached schema column names to ingest.
        #[arg(long, required = true, num_args = 1..)]
        columns: Vec<String>,
        /// Detached write mode (`replace` resets unspecified doc IDs to NULL; `merge` preserves prior generation).
        #[arg(long, value_enum, default_value_t = CliDetachedMode::Replace)]
        mode: CliDetachedMode,
        /// Duplicate ID policy.
        #[arg(long, value_enum, default_value_t = CliDup::LastWins)]
        on_duplicate_id: CliDup,
        /// Vector dimensionality (omit for vectorless collections).
        #[arg(long)]
        dim: Option<usize>,
        /// Distance metric.
        #[arg(long, value_enum, default_value_t = CliMetric::Cosine)]
        metric: CliMetric,
    },
}

fn build_import_config(args: &CommonParquetArgs) -> BulkImportConfig {
    let work_dir = args
        .work_dir
        .clone()
        .unwrap_or_else(|| args.target.with_extension("bulk_work"));
    let col_cfg = match args.dim {
        Some(d) => CollectionConfig::new(TreeConfig::new(d, args.metric.into())),
        None => CollectionConfig::without_vectors(),
    };
    let mut cfg = match args.dim {
        Some(_) => BulkImportConfig::new(
            col_cfg,
            args.id_column.clone(),
            args.vector_column.clone(),
            work_dir,
        ),
        None => BulkImportConfig::new_vectorless(col_cfg, args.id_column.clone(), work_dir),
    };
    cfg = cfg
        .with_on_duplicate_id(args.on_duplicate_id.into())
        .with_fill_factor(args.fill_factor)
        .with_train_sample(args.train_sample)
        .with_memory_budget_bytes(args.memory_mb * 1024 * 1024)
        .with_target_sst_bytes(args.target_sst_mb * 1024 * 1024)
        .with_verify_after(args.verify);
    if args.threads > 0 {
        cfg = cfg.with_threads(args.threads);
    }
    cfg
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Import(args) => {
            let cfg = build_import_config(&args);
            match bulk_import(&args.target, &args.input, cfg, None) {
                Ok(report) => {
                    println!(
                        "Imported {} docs ({} rows read, {} duplicates dropped) into {} segments (leaves={}, height={}, internals={}), {} FTS terms, {} SSTs ({:.2} MiB)",
                        report.docs_imported,
                        report.rows_read,
                        report.duplicates_dropped,
                        report.segments_created,
                        report.leaves,
                        report.tree_height,
                        report.internal_nodes,
                        report.terms,
                        report.sst_files,
                        report.sst_bytes as f64 / (1024.0 * 1024.0),
                    );
                }
                Err(e) => {
                    eprintln!("Bulk import failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Command::Append(args) => {
            let cfg = build_import_config(&args);
            match bulk_append(&args.target, &args.input, cfg, None) {
                Ok(report) => {
                    println!(
                        "Appended {} docs ({} rows read, {} replaced, {} duplicates dropped) into {} new segments (rebuilt_ann={}), {} SSTs ({:.2} MiB)",
                        report.docs_appended,
                        report.rows_read,
                        report.docs_replaced,
                        report.duplicates_dropped,
                        report.segments_created,
                        report.rebuilt_ann_tree,
                        report.sst_files,
                        report.sst_bytes as f64 / (1024.0 * 1024.0),
                    );
                }
                Err(e) => {
                    eprintln!("Bulk append failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        Command::Columns {
            target,
            input,
            id_column,
            columns,
            mode,
            on_duplicate_id,
            dim,
            metric,
        } => {
            let base_cfg = match dim {
                Some(d) => CollectionConfig::new(TreeConfig::new(d, metric.into())),
                None => CollectionConfig::without_vectors(),
            };
            let mut col = match Collection::open(&target, base_cfg) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to open target collection: {e}");
                    std::process::exit(1);
                }
            };
            let rows = match scan_detached_from_parquet(
                &input,
                col.schema(),
                &id_column,
                &columns,
                None,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to read Parquet columns: {e}");
                    std::process::exit(1);
                }
            };
            let dup = match on_duplicate_id {
                CliDup::Error => DetachedDuplicatePolicy::Error,
                CliDup::LastWins => DetachedDuplicatePolicy::LastWins,
            };
            let job = match col
                .detached_ingest()
                .columns(columns.clone())
                .mode(mode.into())
                .duplicate_policy(dup)
                .from_iter(rows)
                .build()
            {
                Ok(j) => j,
                Err(e) => {
                    eprintln!("Failed to build detached job: {e}");
                    std::process::exit(1);
                }
            };
            let build_stats = job.stats.clone();
            match col.publish_detached(job) {
                Ok(pub_stats) => {
                    println!(
                        "Published detached columns {:?}: matched={} docs, blocks_written={}, catchup_rows={}",
                        columns,
                        build_stats.matched_rows,
                        build_stats.blocks_written,
                        pub_stats.catchup_rows
                    );
                }
                Err(e) => {
                    eprintln!("Failed to publish detached job: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
