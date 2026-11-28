use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use izzydb::fts::SimdTier;
use izzydb::{Collection, RankBy};
use izzydb_bench::{DEFAULT_FTS_PARQUET_PATH, FTS_VOCAB_SIZE, fts_collection_config};
use izzydb_bulk::{BulkImportConfig, bulk_import};
use tempfile::TempDir;

/// Sends a command to `perf record --control fifo:<ctl>,<ack>` when
/// `PERF_CTL_FIFO` is set, so profiles cover only the search loop.
fn perf_ctl(cmd: &str) {
    use std::io::{Read, Write};
    let Ok(ctl) = std::env::var("PERF_CTL_FIFO") else {
        return;
    };
    let mut f = fs::OpenOptions::new().write(true).open(&ctl).unwrap();
    f.write_all(format!("{cmd}\n").as_bytes()).unwrap();
    if let Ok(ack) = std::env::var("PERF_ACK_FIFO") {
        let mut buf = [0u8; 5];
        let _ = fs::File::open(ack).unwrap().read(&mut buf);
    }
}

fn run_search_benchmark(
    col: &Collection,
    num_docs: u64,
    runs: usize,
    only_terms: Option<usize>,
    parallel_segments: Option<usize>,
) {
    let vocab: Vec<String> = (0..FTS_VOCAB_SIZE).map(|i| format!("term{i:04}")).collect();

    println!(
        "\n[n={num_docs}, tier={:?}, runs={runs}, parallel_segments={parallel_segments:?}]",
        SimdTier::detect()
    );

    for (label, num_terms) in [
        ("Short 3-term query", 3usize),
        ("Medium 8-term query", 8),
        ("Long 50-term LLM query", 50),
    ] {
        if only_terms.is_some_and(|n| n != num_terms) {
            continue;
        }
        let q_terms: Vec<String> = (0..num_terms)
            .map(|i| vocab[(i * 19 + 5) % vocab.len()].clone())
            .collect();
        let q_str = q_terms.join(" ");
        let rank_by = RankBy::bm25("body", q_str.as_str());

        // Warmup outside timed loop.
        for _ in 0..5 {
            let _ = std::hint::black_box(col.query_bm25(&rank_by, None, 10).unwrap());
        }

        let mut latencies: Vec<Duration> = Vec::with_capacity(runs);
        let mut hits = 0usize;
        let mut top_id = None;
        perf_ctl("enable");
        let t_total = Instant::now();
        for _ in 0..runs {
            let t0 = Instant::now();
            let res = col.query_bm25(&rank_by, None, 10).unwrap();
            latencies.push(t0.elapsed());
            hits = res.len();
            top_id = res.first().map(|r| (r.id, r.score));
            std::hint::black_box(&res);
        }
        let total_s = t_total.elapsed().as_secs_f64();
        perf_ctl("disable");

        latencies.sort_unstable();
        let sum_us: f64 = latencies.iter().map(|d| d.as_secs_f64() * 1e6).sum();
        let avg_us = sum_us / runs as f64;
        let min_us = latencies[0].as_secs_f64() * 1e6;
        let p50_us = latencies[(runs - 1) * 50 / 100].as_secs_f64() * 1e6;
        let p95_us = latencies[(runs - 1) * 95 / 100].as_secs_f64() * 1e6;
        let qps = runs as f64 / total_s;

        println!(
            "  {label:24} -> avg={avg_us:8.1}us, p50={p50_us:8.1}us, p95={p95_us:8.1}us, min={min_us:8.1}us, qps={qps:7.1}, hits={hits}, top={top_id:?}"
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut parquet_path = PathBuf::from(DEFAULT_FTS_PARQUET_PATH);
    let mut runs = 50usize;
    let mut only_terms = None;
    let mut parallel_segments = None;
    let mut db_dir_arg = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--parquet" => {
                i += 1;
                parquet_path = PathBuf::from(args.get(i).expect("missing value for --parquet"));
            }
            "--runs" => {
                i += 1;
                runs = args
                    .get(i)
                    .expect("missing value for --runs")
                    .parse()
                    .expect("invalid --runs integer");
            }
            "--terms" => {
                i += 1;
                only_terms = Some(
                    args.get(i)
                        .expect("missing value for --terms")
                        .parse()
                        .expect("invalid --terms integer"),
                );
            }
            "--parallel-segments" => {
                i += 1;
                parallel_segments = Some(
                    args.get(i)
                        .expect("missing value for --parallel-segments")
                        .parse()
                        .expect("invalid --parallel-segments integer"),
                );
            }
            "--db-dir" => {
                i += 1;
                db_dir_arg = Some(PathBuf::from(
                    args.get(i).expect("missing value for --db-dir"),
                ));
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }

    if !parquet_path.is_file() {
        eprintln!(
            "Parquet file not found at {}; run `cargo run --release -p izzydb-bench --bin fts_bench_gen` first.",
            parquet_path.display()
        );
        return;
    }

    let mut cfg = fts_collection_config();
    if let Some(n) = parallel_segments {
        cfg = cfg.with_fts_parallel_segments(n);
    }
    // `--db-dir` keeps the imported collection on disk so repeated benchmark passes
    // (and baseline/candidate binaries) can reuse the exact same index data.
    let temp_dir;
    let db_dir = match db_dir_arg {
        Some(path) => {
            fs::create_dir_all(&path).unwrap();
            path
        }
        None => {
            temp_dir = TempDir::new().unwrap();
            temp_dir.path().join("db")
        }
    };
    let work_dir = db_dir.with_extension("import-work");
    if db_dir.join("CURRENT").is_file() {
        println!("Reusing collection at {}", db_dir.display());
    } else {
        let bcfg = BulkImportConfig::new(cfg.clone(), "id", "vector", &work_dir);
        // Prepare the collection outside search benchmark measurement.
        let report = bulk_import(&db_dir, std::slice::from_ref(&parquet_path), bcfg, None).unwrap();
        let _ = fs::remove_dir_all(&work_dir);
        println!(
            "Prepared collection from {} (docs={}, terms={}, segments={})",
            parquet_path.display(),
            report.docs_imported,
            report.terms,
            report.segments_created,
        );
    }

    let col = Collection::open(&db_dir, cfg).unwrap();
    let docs = col.len();

    run_search_benchmark(&col, docs, runs, only_terms, parallel_segments);
}
