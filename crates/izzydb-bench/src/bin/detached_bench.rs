//! Benchmark whole-column detached replace ingest & atomic publish (Phase 11).

use std::time::Instant;

use tempfile::TempDir;

use izzydb::{
    AttributeType, AttributeValue, Attributes, Collection, CollectionConfig,
    DetachedDuplicatePolicy, DetachedWriteMode, DocId,
};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);

    let mut cfg = CollectionConfig::without_vectors()
        .with_detached_attribute("popularity", AttributeType::Int, true);
    cfg.segment.seal_rows = 65_536;

    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    // Seed skeleton documents quickly via upsert_batch.
    let batch_docs: Vec<izzydb::Document> = (0..n)
        .map(|i| izzydb::Document {
            id: i as DocId,
            embedding: None,
            attributes: Attributes::new(),
        })
        .collect();
    for chunk in batch_docs.chunks(50_000) {
        col.upsert_batch(chunk).unwrap();
    }
    col.seal().unwrap();

    let rows: Vec<(DocId, Vec<AttributeValue>)> = (0..n)
        .map(|i| (i as DocId, vec![AttributeValue::Int(((i * 37) % 10_000) as i64)]))
        .collect();

    let t_build = Instant::now();
    let job = col
        .detached_ingest()
        .column("popularity")
        .mode(DetachedWriteMode::Replace)
        .duplicate_policy(DetachedDuplicatePolicy::LastWins)
        .from_iter(rows)
        .build()
        .unwrap();
    let build_s = t_build.elapsed().as_secs_f64();

    let t_pub = Instant::now();
    let stats = col.publish_detached(job).unwrap();
    let pub_ms = t_pub.elapsed().as_secs_f64() * 1000.0;

    println!(
        "detached_bench (n={n}): build={build_s:.3}s ({:.2}M rows/s), publish={pub_ms:.2}ms (catchup={})",
        (n as f64 / 1e6) / build_s,
        stats.catchup_rows,
    );
}
