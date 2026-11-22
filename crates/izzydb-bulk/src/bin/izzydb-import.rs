use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use izzydb::config::{DistanceMetric, TreeConfig};
use izzydb::schema::CollectionConfig;
use izzydb_bulk::{BulkImportConfig, DuplicatePolicy, bulk_import};

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

#[derive(Parser)]
#[command(name = "izzydb-import", about = "Bulk-import Parquet files into izzydb")]
struct Cli {
    /// Target izzydb collection directory (must be non-existent or empty).
    #[arg(long)]
    target: PathBuf,
    /// Input Parquet files.
    #[arg(long, required = true, num_args = 1..)]
    input: Vec<PathBuf>,
    /// Vector dimensionality.
    #[arg(long)]
    dim: usize,
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
    /// Run exhaustive invariant verification after import.
    #[arg(long)]
    verify: bool,
}

fn main() {
    let cli = Cli::parse();
    let work_dir = cli
        .work_dir
        .unwrap_or_else(|| cli.target.with_extension("bulk_work"));
    let col_cfg = CollectionConfig::new(TreeConfig::new(cli.dim, cli.metric.into()));
    let mut cfg = BulkImportConfig::new(col_cfg, cli.id_column, cli.vector_column, work_dir)
        .with_on_duplicate_id(cli.on_duplicate_id.into())
        .with_fill_factor(cli.fill_factor)
        .with_train_sample(cli.train_sample)
        .with_memory_budget_bytes(cli.memory_mb * 1024 * 1024)
        .with_target_sst_bytes(cli.target_sst_mb * 1024 * 1024)
        .with_verify_after(cli.verify);
    if cli.threads > 0 {
        cfg = cfg.with_threads(cli.threads);
    }

    match bulk_import(&cli.target, &cli.input, cfg, None) {
        Ok(report) => {
            println!(
                "Imported {} docs ({} rows read, {} duplicates dropped) into {} leaves (height={}, internals={}), {} FTS terms, {} SSTs ({:.2} MiB)",
                report.docs_imported,
                report.rows_read,
                report.duplicates_dropped,
                report.leaves,
                report.tree_height,
                report.internal_nodes,
                report.terms,
                report.sst_files,
                report.sst_bytes as f64 / (1024.0 * 1024.0),
            );
            for (phase, dur) in &report.phase_timings {
                println!("  {phase}: {:.3}s", dur.as_secs_f64());
            }
        }
        Err(e) => {
            eprintln!("Bulk import failed: {e}");
            std::process::exit(1);
        }
    }
}
