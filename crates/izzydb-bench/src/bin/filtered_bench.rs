use std::collections::HashSet;
use std::time::Instant;

use izzydb::metric::{l2_squared, prepare_vector};
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric, DocId, Filter,
    SearchOptions, TreeConfig,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn main() {
    let n = 20_000usize;
    let dim = 128usize;
    let num_clusters = 150usize;
    let mut rng = SmallRng::seed_from_u64(42);

    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect())
        .collect();
    let data: Vec<(Vec<f32>, Attributes)> = (0..n)
        .map(|i| {
            let c = &centers[i % num_clusters];
            let v: Vec<f32> = c
                .iter()
                .map(|&x| x + rng.random_range(-0.2f32..0.2f32))
                .collect();
            let mut attrs = Attributes::new();
            attrs.insert("mod100".into(), ((i % 100) as u64).into());
            attrs.insert("mod10".into(), ((i % 10) as u64).into());
            attrs.insert("even".into(), (i % 2 == 0).into());
            attrs.insert("score".into(), ((i % 100) as f64).into());
            (v, attrs)
        })
        .collect();

    let queries: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            let c = &centers[i % num_clusters];
            c.iter()
                .map(|&x| x + rng.random_range(-0.2f32..0.2f32))
                .collect()
        })
        .collect();

    let dir = TempDir::new().unwrap();
    let mut tree_cfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    tree_cfg.enable_lire = true;
    let cfg = CollectionConfig::new(tree_cfg)
        .with_attribute("mod100", AttributeType::Uint, true)
        .with_attribute("mod10", AttributeType::Uint, true)
        .with_attribute("even", AttributeType::Bool, true)
        .with_attribute("score", AttributeType::Float, true);

    let mut col = Collection::open(dir.path(), cfg).unwrap();
    let t0 = Instant::now();
    for (id, (vec, attrs)) in data.iter().enumerate() {
        col.upsert(id as DocId, vec, attrs.clone()).unwrap();
    }
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let stats = col.verify_invariants().unwrap();
    println!(
        "Built Collection: n={}, dim={}, leaves={}, clusters={}, postings={}, build={build_ms:.1}ms",
        stats.num_docs,
        dim,
        stats.tree.leaf_nodes,
        stats.num_cluster_bitmaps,
        stats.num_posting_bitmaps
    );

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

    for (label, filter) in cases {
        let mut ground_truth = Vec::with_capacity(queries.len());
        for q in &queries {
            let qp = prepare_vector(q, dim, DistanceMetric::L2Squared).unwrap();
            let mut scored: Vec<(DocId, f32)> = data
                .iter()
                .enumerate()
                .filter(|(_, (_, attrs))| filter.as_ref().is_none_or(|f| f.matches(attrs)))
                .map(|(i, (v, _))| (i as DocId, l2_squared(&qp, v)))
                .collect();
            scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            scored.truncate(10);
            ground_truth.push(scored.into_iter().map(|(id, _)| id).collect::<HashSet<_>>());
        }

        let opts = SearchOptions::new(10, 8)
            .with_rerank(100)
            .with_max_expanded_leaves(256);
        let sq = Instant::now();
        let mut hits = 0usize;
        let mut total = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let res = col.search(q, opts, filter.as_ref()).unwrap();
            hits += res
                .iter()
                .filter(|r| ground_truth[qi].contains(&r.id))
                .count();
            total += ground_truth[qi].len();
        }
        let avg_us = sq.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        let recall = hits as f64 / total as f64;
        println!("{label:22} -> recall@10={recall:.4}, avg_latency={avg_us:.1}us");
    }
}
