//! Benchmark mixed upsert/patch/delete churn + periodic maintenance (Phase 11).

use std::path::Path;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric, DocId,
    DocumentPatch, MaintenanceOptions, TreeConfig,
};

fn dir_size_bytes(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(path) {
        for entry in rd.flatten() {
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

fn main() {
    let n = 10_000usize;
    let dim = 64usize;
    let mut rng = SmallRng::seed_from_u64(42);
    let mut cfg = CollectionConfig::new(TreeConfig::new(dim, DistanceMetric::Cosine))
        .with_attribute("cat", AttributeType::String, true)
        .with_attribute("score", AttributeType::Float, true);
    cfg.segment.seal_rows = 2_000;

    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    let t_init = Instant::now();
    for id in 0..n as DocId {
        let v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
        let mut attrs = Attributes::new();
        attrs.insert("cat".into(), format!("c{}", id % 20).into());
        attrs.insert("score".into(), ((id % 500) as f64).into());
        col.upsert(id, Some(&v), attrs).unwrap();
    }
    let init_s = t_init.elapsed().as_secs_f64();
    let baseline_bytes = dir_size_bytes(dir.path());

    // Mixed churn: 5,000 attribute patches + 2,500 upserts + 1,000 deletes + maintenance.
    let churn_ops = 8_500usize;
    let t_churn = Instant::now();
    for i in 0..churn_ops {
        if i < 5_000 {
            let id = rng.random_range(0..n as DocId);
            let patch = DocumentPatch::new()
                .with_attribute("score", ((i % 1000) as f64).into());
            let _ = col.patch(id, patch);
        } else if i < 7_500 {
            let id = rng.random_range(0..(n + 2_000) as DocId);
            let v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
            let mut attrs = Attributes::new();
            attrs.insert("cat".into(), format!("c{}", id % 20).into());
            attrs.insert("score".into(), ((i % 1000) as f64).into());
            col.upsert(id, Some(&v), attrs).unwrap();
        } else {
            let id = rng.random_range(0..n as DocId);
            let _ = col.delete(id);
        }
        if (i + 1) % 2_000 == 0 {
            col.maintenance(MaintenanceOptions::default()).unwrap();
        }
    }
    col.maintenance(MaintenanceOptions::default()).unwrap();
    let churn_s = t_churn.elapsed().as_secs_f64();
    let final_bytes = dir_size_bytes(dir.path());

    println!(
        "churn_bench: init n={n} in {init_s:.2}s ({:.0} docs/s), baseline={:.2} MiB | churn {churn_ops} ops + maintenance in {churn_s:.2}s ({:.0} ops/s), final={:.2} MiB ({:.2}x)",
        n as f64 / init_s,
        baseline_bytes as f64 / (1024.0 * 1024.0),
        churn_ops as f64 / churn_s,
        final_bytes as f64 / (1024.0 * 1024.0),
        final_bytes as f64 / baseline_bytes.max(1) as f64,
    );
}
