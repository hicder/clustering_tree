use std::collections::HashSet;
use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, Int64Builder, StringBuilder, UInt64Builder};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

use izzydb::metric::{l2_squared, prepare_vector};
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric, DocId,
    FullTextSearchConfig, SearchOptions, TreeConfig,
};
use izzydb_bulk::{BulkImportConfig, bulk_import};

fn dir_size_bytes(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size_bytes(&p);
            } else if let Ok(m) = entry.metadata() {
                total += m.len();
            }
        }
    }
    total
}

fn write_bench_parquet(
    path: &Path,
    n: usize,
    dim: usize,
    with_fts: bool,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let num_clusters = 64.min(n.max(1));
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect())
        .collect();

    let mut fields = vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                dim as i32,
            ),
            false,
        ),
        Field::new("category", DataType::Utf8, false),
        Field::new("rank", DataType::Int64, false),
    ];
    if with_fts {
        fields.push(Field::new("text", DataType::Utf8, false));
    }
    let arrow_schema = Arc::new(Schema::new(fields));

    let props = WriterProperties::builder()
        .set_max_row_group_size(65_536)
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), arrow_schema.clone(), Some(props))
            .unwrap();

    let categories = ["alpha", "beta", "gamma", "delta", "epsilon"];
    let vocab = [
        "vector", "search", "index", "cluster", "rabitq", "parquet", "rocksdb", "bm25", "hybrid",
        "filter", "facet", "quantize", "centroid", "leaf", "posting", "skip",
    ];

    let mut sample_vecs = Vec::new();
    let sample_limit = n.min(20_000);
    let batch_size = 16_384;
    let mut written = 0usize;

    while written < n {
        let cur = (n - written).min(batch_size);
        let mut id_b = UInt64Builder::with_capacity(cur);
        let mut vec_b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32)
            .with_field(Field::new("item", DataType::Float32, false));
        let mut cat_b = StringBuilder::new();
        let mut rank_b = Int64Builder::with_capacity(cur);
        let mut txt_b = StringBuilder::new();

        for j in 0..cur {
            let idx = written + j;
            id_b.append_value(idx as u64);
            let c = &centers[idx % num_clusters];
            let mut v = Vec::with_capacity(if idx < sample_limit { dim } else { 0 });
            for &cx in c {
                let x = cx + rng.random_range(-0.15f32..0.15f32);
                vec_b.values().append_value(x);
                if idx < sample_limit {
                    v.push(x);
                }
            }
            vec_b.append(true);
            if idx < sample_limit {
                sample_vecs.push(v);
            }
            cat_b.append_value(categories[idx % categories.len()]);
            rank_b.append_value((idx % 1000) as i64);
            if with_fts {
                let w0 = vocab[idx % vocab.len()];
                let w1 = vocab[(idx * 7 + 3) % vocab.len()];
                let w2 = vocab[(idx * 13 + 5) % vocab.len()];
                txt_b.append_value(format!("{w0} {w1} {w2} doc{idx}"));
            }
        }

        let mut cols: Vec<arrow_array::ArrayRef> = vec![
            Arc::new(id_b.finish()),
            Arc::new(vec_b.finish()),
            Arc::new(cat_b.finish()),
            Arc::new(rank_b.finish()),
        ];
        if with_fts {
            cols.push(Arc::new(txt_b.finish()));
        }
        let batch = RecordBatch::try_new(arrow_schema.clone(), cols).unwrap();
        writer.write(&batch).unwrap();
        written += cur;
    }
    writer.close().unwrap();

    let queries: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let c = &centers[i % num_clusters];
            c.iter()
                .map(|&x| x + rng.random_range(-0.15f32..0.15f32))
                .collect()
        })
        .collect();
    (sample_vecs, queries)
}

fn run_suite(name: &str, n: usize, dim: usize, with_fts: bool, compare_upsert_n: Option<usize>) {
    let dir = TempDir::new().unwrap();
    let pq_path = dir.path().join("data.parquet");
    let t_gen = Instant::now();
    let (sample_vecs, queries) = write_bench_parquet(&pq_path, n, dim, with_fts, 42);
    let gen_s = t_gen.elapsed().as_secs_f64();

    let mut col_cfg = CollectionConfig::new(TreeConfig::new(dim, DistanceMetric::L2Squared))
        .with_attribute("category", AttributeType::String, true)
        .with_attribute("rank", AttributeType::Int, true);
    if with_fts {
        col_cfg = col_cfg.with_fts_attribute(
            "text",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        );
    }

    let bulk_dir = dir.path().join("bulk_db");
    let work_dir = dir.path().join("work");
    let bcfg = BulkImportConfig::new(col_cfg.clone(), "id", "vector", &work_dir);

    let t0 = Instant::now();
    let report = bulk_import(&bulk_dir, &[pq_path], bcfg, None).unwrap();
    let bulk_s = t0.elapsed().as_secs_f64();
    let bulk_bytes = dir_size_bytes(&bulk_dir);

    println!(
        "[{name}] n={n}, dim={dim}, fts={with_fts} (parquet gen {gen_s:.2}s):"
    );
    println!(
        "  bulk_import: {bulk_s:.2}s ({:.0} docs/s), db={:.2} MiB, leaves={}, height={}, terms={}, sst_files={}",
        n as f64 / bulk_s,
        bulk_bytes as f64 / (1024.0 * 1024.0),
        report.leaves,
        report.tree_height,
        report.terms,
        report.sst_files,
    );
    for (phase, dur) in &report.phase_timings {
        println!("    {phase}: {:.2}s", dur.as_secs_f64());
    }

    let bulk_col = Collection::open(&bulk_dir, col_cfg.clone()).unwrap();
    let mut hits_total = 0usize;
    if n <= sample_vecs.len() {
        for q in &queries {
            let qp = prepare_vector(q, dim, DistanceMetric::L2Squared).unwrap();
            let mut gt: Vec<(DocId, f32)> = sample_vecs
                .iter()
                .enumerate()
                .map(|(i, v)| (i as DocId, l2_squared(&qp, v)))
                .collect();
            gt.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt_set: HashSet<DocId> = gt.iter().take(10).map(|&(id, _)| id).collect();
            for res in bulk_col
                .search(q, SearchOptions::new(10, 8).with_rerank(100), None)
                .unwrap()
            {
                if gt_set.contains(&res.id) {
                    hits_total += 1;
                }
            }
        }
        println!(
            "  bulk recall@10 (beam=8, rerank=100): {:.4}",
            hits_total as f64 / (queries.len() * 10) as f64
        );
    }

    if let Some(up_n) = compare_upsert_n {
        let up_dir = dir.path().join("upsert_db");
        let mut up_col = Collection::open(&up_dir, col_cfg).unwrap();
        let categories = ["alpha", "beta", "gamma", "delta", "epsilon"];
        let t_up = Instant::now();
        for (i, v) in sample_vecs.iter().take(up_n).enumerate() {
            let mut attrs = Attributes::new();
            attrs.insert("category".into(), categories[i % categories.len()].into());
            attrs.insert("rank".into(), ((i % 1000) as i64).into());
            if with_fts {
                attrs.insert("text".into(), format!("vector search index doc{i}").into());
            }
            up_col.upsert(i as DocId, Some(v), attrs).unwrap();
        }
        let up_s = t_up.elapsed().as_secs_f64();
        let up_bytes = dir_size_bytes(&up_dir);
        println!(
            "  upsert baseline (n={up_n}): {up_s:.2}s ({:.0} docs/s), db={:.2} MiB",
            up_n as f64 / up_s,
            up_bytes as f64 / (1024.0 * 1024.0),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let full = args.iter().any(|a| a == "--full");
    if full {
        run_suite("1M x 128d", 1_000_000, 128, false, Some(20_000));
        run_suite("1M x 768d + FTS", 1_000_000, 768, true, Some(5_000));
    } else {
        run_suite("50k x 128d", 50_000, 128, false, Some(10_000));
        run_suite("20k x 768d + FTS", 20_000, 768, true, Some(5_000));
    }
}
