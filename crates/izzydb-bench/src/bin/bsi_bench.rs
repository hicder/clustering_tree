//! Filter-only range benchmark: bit-sliced index vs value-major postings.
//!
//! ```bash
//! cargo run --release -p izzydb-bench --bin bsi_bench -- --rows 1000000 --duration-secs 3
//! ```

use std::time::{Duration, Instant};

use izzydb::bsi_codec::BSI_BASE_PLANE;
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, Document, Filter,
};

fn build_collection(bsi: bool, rows: u64) -> (tempfile::TempDir, Collection) {
    let dir = tempfile::tempdir().unwrap();
    let cfg = CollectionConfig::without_vectors()
        .with_seal_rows(65_536)
        .with_bit_sliced_index(bsi)
        .with_attribute("price", AttributeType::Float, true)
        .with_attribute("uid", AttributeType::Uint, true)
        .with_attribute("tier", AttributeType::Uint, true);
    let mut col = Collection::open(dir.path(), cfg).unwrap();
    let mut batch = Vec::with_capacity(16_384);
    for id in 0..rows {
        let mut attrs = Attributes::new();
        // Deterministic high-cardinality float in [0, 1) plus a low-cardinality tier.
        let price = ((id.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64)
            / ((1u64 << 53) as f64);
        attrs.insert("price".into(), price.into());
        attrs.insert("uid".into(), id.into());
        attrs.insert("tier".into(), (id % 10).into());
        batch.push(Document {
            id,
            embedding: None,
            attributes: attrs,
        });
        if batch.len() == 16_384 {
            col.upsert_batch(&batch).unwrap();
            batch.clear();
        }
    }
    if !batch.is_empty() {
        col.upsert_batch(&batch).unwrap();
    }
    (dir, col)
}

fn measure(col: &Collection, label: &str, queries: &[Filter], duration: Duration) {
    // Warm every variant once.
    for q in queries {
        let _ = col.resolve_filter_doc_ids(q).unwrap();
    }
    let start = Instant::now();
    let mut count = 0usize;
    let mut total_hits = 0usize;
    while start.elapsed() < duration {
        for q in queries {
            total_hits += col.resolve_filter_doc_ids(q).unwrap().len();
            count += 1;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let avg_us = elapsed * 1e6 / count as f64;
    println!(
        "{label:28} queries={count}, avg_latency={avg_us:8.1}us, rows/query={:.0}",
        total_hits as f64 / count as f64
    );
}

fn main() {
    let mut rows = 1_000_000u64;
    let mut duration_secs = 3u64;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--rows" => {
                i += 1;
                rows = args[i].parse().unwrap();
            }
            "--duration-secs" => {
                i += 1;
                duration_secs = args[i].parse().unwrap();
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }
    let duration = Duration::from_secs(duration_secs);
    let filters = [
        (
            "float 10% Lt(price<0.1)",
            Filter::And(vec![
                Filter::Lt("price".into(), 0.1f64.into()),
                Filter::Gte("price".into(), (-0.0f64).into()),
            ]),
        ),
        (
            "float 1% Lt(price<0.01)",
            Filter::And(vec![
                Filter::Lt("price".into(), 0.01f64.into()),
                Filter::Gte("price".into(), (-0.0f64).into()),
            ]),
        ),
        (
            "float window [0.3,0.7)",
            Filter::And(vec![
                Filter::Gte("price".into(), 0.3f64.into()),
                Filter::Lt("price".into(), 0.7f64.into()),
            ]),
        ),
        (
            "uint 10% uid<100k",
            Filter::And(vec![
                Filter::Lt("uid".into(), 100_000u64.into()),
                Filter::Gte("uid".into(), 0u64.into()),
            ]),
        ),
        (
            "uint 1% uid<10k",
            Filter::And(vec![
                Filter::Lt("uid".into(), 10_000u64.into()),
                Filter::Gte("uid".into(), 0u64.into()),
            ]),
        ),
        (
            "low-card tier=3",
            Filter::Eq("tier".into(), 3u64.into()),
        ),
    ];

    for bsi in [false, true] {
        let (_dir, col) = build_collection(bsi, rows);
        let segments = col.segment_snapshot().segments.len();
        let bsi_segments = if bsi {
            (0..segments as u32)
                .filter(|&seg| {
                    col.storage()
                        .get_bsi_plane(seg, 0, 0, BSI_BASE_PLANE)
                        .unwrap()
                        .is_some()
                })
                .count()
        } else {
            0
        };
        println!(
            "\nBSI {} (rows={rows}, segments={segments}, bsi_segments={bsi_segments})",
            if bsi { "enabled" } else { "disabled" }
        );
        for (label, filter) in &filters {
            let queries = vec![filter.clone()];
            measure(&col, label, &queries, duration);
        }
    }
}
