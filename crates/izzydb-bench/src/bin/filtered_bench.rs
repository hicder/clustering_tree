use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_array::builder::{
    BooleanBuilder, FixedSizeListBuilder, Float32Builder, Float64Builder, UInt64Builder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, UInt64Type};
use arrow_schema::{DataType, Field, Schema};
use izzydb::metric::{l2_squared, prepare_vector};
use izzydb::{
    AttributeType, Collection, CollectionConfig, DistanceMetric, DocId, Filter, SearchOptions,
    TreeConfig,
};
use izzydb_bulk::{BulkImportConfig, bulk_import};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

const DEFAULT_ROWS: usize = 10_000_000;
const DIM: usize = 128;
const TARGET_CLUSTER_SIZE: usize = 500;
const MIN_CLUSTERS: usize = 150;
const NUM_QUERIES: usize = 100;
const DEFAULT_PARQUET_PATH: &str = "target/bench-data/filtered_10m.parquet";

#[inline]
fn num_clusters_for_rows(n: usize) -> usize {
    (n / TARGET_CLUSTER_SIZE).max(MIN_CLUSTERS)
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[derive(Clone, Copy)]
struct ScoredDoc {
    dist: f32,
    id: DocId,
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.dist.to_bits() == other.dist.to_bits()
    }
}

impl Eq for ScoredDoc {}

impl Ord for ScoredDoc {
    /// Max-heap ordering matching `(dist asc, id asc)` so `peek()` is the worst of top-k.
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn push_top_k(heap: &mut BinaryHeap<ScoredDoc>, k: usize, item: ScoredDoc) {
    if heap.len() < k {
        heap.push(item);
    } else if let Some(&worst) = heap.peek()
        && item < worst
    {
        heap.pop();
        heap.push(item);
    }
}

fn generate_centers(num_clusters: usize) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(42);
    (0..num_clusters)
        .map(|_| (0..DIM).map(|_| rng.random_range(-1.0f32..1.0f32)).collect())
        .collect()
}

fn generate_queries(centers: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(1337);
    let num_clusters = centers.len();
    (0..NUM_QUERIES)
        .map(|i| {
            let c = &centers[(i * 199) % num_clusters];
            c.iter()
                .map(|&x| x + rng.random_range(-0.2f32..0.2f32))
                .collect()
        })
        .collect()
}

fn write_filtered_parquet(path: &Path, n: usize) {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).unwrap();
    }

    let num_clusters = num_clusters_for_rows(n);
    let centers = generate_centers(num_clusters);
    // Advance RNG past centers to keep cluster/data generation deterministic.
    let mut rng = SmallRng::seed_from_u64(4242);

    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                std::sync::Arc::new(Field::new("item", DataType::Float32, false)),
                DIM as i32,
            ),
            false,
        ),
        Field::new("mod100", DataType::UInt64, false),
        Field::new("mod10", DataType::UInt64, false),
        Field::new("even", DataType::Boolean, false),
        Field::new("score", DataType::Float64, false),
    ]));

    let props = WriterProperties::builder()
        .set_max_row_group_size(65_536)
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), schema.clone(), Some(props)).unwrap();

    let batch_size = 16_384usize;
    let mut written = 0usize;
    while written < n {
        let cur = (n - written).min(batch_size);
        let mut id_b = UInt64Builder::with_capacity(cur);
        let mut vec_b = FixedSizeListBuilder::new(Float32Builder::new(), DIM as i32)
            .with_field(Field::new("item", DataType::Float32, false));
        let mut mod100_b = UInt64Builder::with_capacity(cur);
        let mut mod10_b = UInt64Builder::with_capacity(cur);
        let mut even_b = BooleanBuilder::with_capacity(cur);
        let mut score_b = Float64Builder::with_capacity(cur);

        for j in 0..cur {
            let idx = written + j;
            id_b.append_value(idx as u64);
            let c = &centers[idx % num_clusters];
            for &cx in c {
                vec_b
                    .values()
                    .append_value(cx + rng.random_range(-0.2f32..0.2f32));
            }
            vec_b.append(true);
            // Hash row index so scalar attributes are independent of cluster assignment.
            let attr_bucket = splitmix64(idx as u64) % 100;
            mod100_b.append_value(attr_bucket);
            mod10_b.append_value(attr_bucket % 10);
            even_b.append_value(attr_bucket % 2 == 0);
            score_b.append_value(attr_bucket as f64);
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                std::sync::Arc::new(id_b.finish()),
                std::sync::Arc::new(vec_b.finish()),
                std::sync::Arc::new(mod100_b.finish()),
                std::sync::Arc::new(mod10_b.finish()),
                std::sync::Arc::new(even_b.finish()),
                std::sync::Arc::new(score_b.finish()),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        written += cur;
    }
    writer.close().unwrap();
}

struct LoadedDataset {
    ids: Vec<DocId>,
    vecs: Vec<f32>,
    mod100: Vec<u64>,
    mod10: Vec<u64>,
    even: Vec<bool>,
    score: Vec<f64>,
}

fn load_parquet_dataset(path: &Path) -> LoadedDataset {
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let total_rows = builder.metadata().file_metadata().num_rows() as usize;
    let reader = builder.with_batch_size(16_384).build().unwrap();

    let mut ids = Vec::with_capacity(total_rows);
    let mut vecs = Vec::with_capacity(total_rows * DIM);
    let mut mod100 = Vec::with_capacity(total_rows);
    let mut mod10 = Vec::with_capacity(total_rows);
    let mut even = Vec::with_capacity(total_rows);
    let mut score = Vec::with_capacity(total_rows);

    for batch in reader {
        let batch = batch.unwrap();
        let rows = batch.num_rows();
        let id_arr = batch.column(0).as_primitive::<UInt64Type>();
        let vec_arr = batch.column(1).as_fixed_size_list();
        let vec_vals = vec_arr.values().as_primitive::<Float32Type>().values();
        let mod100_arr = batch.column(2).as_primitive::<UInt64Type>();
        let mod10_arr = batch.column(3).as_primitive::<UInt64Type>();
        let even_arr = batch.column(4).as_boolean();
        let score_arr = batch.column(5).as_primitive::<Float64Type>();

        ids.extend_from_slice(id_arr.values());
        vecs.extend_from_slice(&vec_vals[..rows * DIM]);
        mod100.extend_from_slice(mod100_arr.values());
        mod10.extend_from_slice(mod10_arr.values());
        for i in 0..rows {
            even.push(even_arr.value(i));
        }
        score.extend_from_slice(score_arr.values());
    }

    LoadedDataset {
        ids,
        vecs,
        mod100,
        mod10,
        even,
        score,
    }
}

fn compute_ground_truth(
    dataset: &LoadedDataset,
    queries: &[Vec<f32>],
    num_cases: usize,
) -> Vec<Vec<HashSet<DocId>>> {
    let n = dataset.ids.len();
    let mut gt_by_case: Vec<Vec<HashSet<DocId>>> = vec![Vec::with_capacity(queries.len()); num_cases];

    for q in queries {
        let qp = prepare_vector(q, DIM, DistanceMetric::L2Squared).unwrap();
        let mut heaps: Vec<BinaryHeap<ScoredDoc>> =
            (0..num_cases).map(|_| BinaryHeap::with_capacity(11)).collect();

        for i in 0..n {
            let v = &dataset.vecs[i * DIM..(i + 1) * DIM];
            let item = ScoredDoc {
                dist: l2_squared(&qp, v),
                id: dataset.ids[i],
            };
            // Case 0: unfiltered
            push_top_k(&mut heaps[0], 10, item);
            // Case 1: Eq(even=true)
            if dataset.even[i] {
                push_top_k(&mut heaps[1], 10, item);
            }
            // Case 2: Eq(mod10=3)
            if dataset.mod10[i] == 3 {
                push_top_k(&mut heaps[2], 10, item);
            } else {
                // Case 5: Not(Eq(mod10=3))
                push_top_k(&mut heaps[5], 10, item);
            }
            // Case 3: Lt(score<10)
            if dataset.score[i] < 10.0 {
                push_top_k(&mut heaps[3], 10, item);
            }
            // Case 4: Eq(mod100=7)
            if dataset.mod100[i] == 7 {
                push_top_k(&mut heaps[4], 10, item);
            }
        }

        for (case_idx, heap) in heaps.into_iter().enumerate() {
            gt_by_case[case_idx].push(heap.into_iter().map(|d| d.id).collect());
        }
    }

    gt_by_case
}

fn parquet_num_rows(path: &Path) -> usize {
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    builder.metadata().file_metadata().num_rows() as usize
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut parquet_path = PathBuf::from(DEFAULT_PARQUET_PATH);
    let mut db_dir_arg: Option<PathBuf> = None;
    let mut only_case: Option<String> = None;
    let mut duration_secs: Option<u64> = None;
    let mut generate_only = false;
    let mut rows = DEFAULT_ROWS;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--parquet" => {
                i += 1;
                parquet_path = PathBuf::from(&args[i]);
            }
            "--db-dir" => {
                i += 1;
                db_dir_arg = Some(PathBuf::from(&args[i]));
            }
            "--only-case" => {
                i += 1;
                only_case = Some(args[i].clone());
            }
            "--duration-secs" => {
                i += 1;
                duration_secs = Some(args[i].parse().unwrap());
            }
            "--generate-only" => {
                generate_only = true;
            }
            "--rows" => {
                i += 1;
                rows = args[i].parse().unwrap();
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }

    if generate_only || !parquet_path.exists() {
        let t_gen = Instant::now();
        write_filtered_parquet(&parquet_path, rows);
        let size_mib = fs::metadata(&parquet_path).unwrap().len() as f64 / (1024.0 * 1024.0);
        println!(
            "Generated parquet: {} (rows={}, size={size_mib:.2} MiB, elapsed={:.2}s)",
            parquet_path.display(),
            rows,
            t_gen.elapsed().as_secs_f64()
        );
        if generate_only {
            return;
        }
    } else {
        let size_mib = fs::metadata(&parquet_path).unwrap().len() as f64 / (1024.0 * 1024.0);
        println!(
            "Using pregenerated parquet: {} (size={size_mib:.2} MiB)",
            parquet_path.display()
        );
    }

    let mut tree_cfg = TreeConfig::new(DIM, DistanceMetric::L2Squared);
    tree_cfg.enable_lire = true;
    let cfg = CollectionConfig::new(tree_cfg)
        .with_attribute("mod100", AttributeType::Uint, true)
        .with_attribute("mod10", AttributeType::Uint, true)
        .with_attribute("even", AttributeType::Bool, true)
        .with_attribute("score", AttributeType::Float, true);

    // Setup outside benchmark timing: reuse persisted db_dir if present, else bulk-import.
    let _temp_dir: Option<TempDir>;
    let db_dir = if let Some(ref p) = db_dir_arg {
        _temp_dir = None;
        p.clone()
    } else {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("db");
        _temp_dir = Some(dir);
        p
    };

    if db_dir_arg.is_some() && db_dir.exists() {
        let t_open = Instant::now();
        let col = Collection::open(&db_dir, cfg).unwrap();
        println!(
            "Opened existing Collection at {} in {:.2}s",
            db_dir.display(),
            t_open.elapsed().as_secs_f64()
        );
        run_bench(&col, &parquet_path, only_case.as_deref(), duration_secs);
        return;
    }

    let work_dir = db_dir.with_extension("work");
    let bcfg = BulkImportConfig::new(cfg.clone(), "id", "vector", &work_dir);

    let t_bulk = Instant::now();
    let report = bulk_import(&db_dir, std::slice::from_ref(&parquet_path), bcfg, None).unwrap();
    let bulk_s = t_bulk.elapsed().as_secs_f64();
    let _ = fs::remove_dir_all(&work_dir);

    let col = Collection::open(&db_dir, cfg).unwrap();
    println!(
        "Bulk-imported Collection (excluded from bench): docs={}, dim={}, leaves={}, segments={}, sst_files={}, elapsed={bulk_s:.2}s",
        report.docs_imported, DIM, report.leaves, report.segments_created, report.sst_files
    );

    run_bench(&col, &parquet_path, only_case.as_deref(), duration_secs);
}

fn run_bench(
    col: &Collection,
    parquet_path: &Path,
    only_case: Option<&str>,
    duration_secs: Option<u64>,
) {
    let cases: Vec<(&str, Option<Filter>)> = vec![
        ("100% unfiltered", None),
        ("50% Eq(even=true)", Some(Filter::Eq("even".into(), true.into()))),
        ("10% Eq(mod10=3)", Some(Filter::Eq("mod10".into(), 3u64.into()))),
        ("10% Lt(score<10)", Some(Filter::Lt("score".into(), 10.0f64.into()))),
        ("1% Eq(mod100=7)", Some(Filter::Eq("mod100".into(), 7u64.into()))),
        (
            "90% Not(Eq(mod10=3))",
            Some(Filter::Not(Box::new(Filter::Eq("mod10".into(), 3u64.into())))),
        ),
    ];

    let opts = SearchOptions::new(10, 16).with_rerank(200);

    if let Some(secs) = duration_secs {
        let total_rows = parquet_num_rows(parquet_path);
        let centers = generate_centers(num_clusters_for_rows(total_rows));
        let queries = generate_queries(&centers);
        for (label, filter) in &cases {
            if let Some(pat) = only_case
                && !label.contains(pat)
            {
                continue;
            }
            let _ = col.search(&queries[0], opts, filter.as_ref()).unwrap();
            let start = Instant::now();
            let deadline = std::time::Duration::from_secs(secs);
            let mut count = 0usize;
            while start.elapsed() < deadline {
                for q in &queries {
                    let res = col.search(q, opts, filter.as_ref()).unwrap();
                    std::hint::black_box(res);
                    count += 1;
                    if start.elapsed() >= deadline {
                        break;
                    }
                }
            }
            let elapsed = start.elapsed().as_secs_f64();
            let avg_us = elapsed * 1e6 / count as f64;
            let qps = count as f64 / elapsed;
            println!(
                "{label:22} -> queries={count}, elapsed={elapsed:.2}s, avg_latency={avg_us:.1}us, qps={qps:.1}"
            );
        }
        return;
    }

    // Precompute ground truth from the parquet dataset (also excluded from benchmark timing).
    let t_gt = Instant::now();
    let dataset = load_parquet_dataset(parquet_path);
    let centers = generate_centers(num_clusters_for_rows(dataset.ids.len()));
    let queries = generate_queries(&centers);
    let ground_truth = compute_ground_truth(&dataset, &queries, cases.len());
    drop(dataset);
    println!(
        "Prepared {} queries and exact ground truth in {:.2}s",
        queries.len(),
        t_gt.elapsed().as_secs_f64()
    );

    for (case_idx, (label, filter)) in cases.iter().enumerate() {
        if let Some(pat) = only_case
            && !label.contains(pat)
        {
            continue;
        }
        let _ = col.search(&queries[0], opts, filter.as_ref()).unwrap();
        let sq = Instant::now();
        let mut hits = 0usize;
        let mut total = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let res = col.search(q, opts, filter.as_ref()).unwrap();
            hits += res
                .iter()
                .filter(|r| ground_truth[case_idx][qi].contains(&r.id))
                .count();
            total += ground_truth[case_idx][qi].len();
        }
        let avg_us = sq.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        let recall = hits as f64 / total as f64;
        println!("{label:22} -> recall@10={recall:.4}, avg_latency={avg_us:.1}us");
    }
}
