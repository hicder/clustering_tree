use std::collections::{HashMap, HashSet};
use std::time::Instant;

use izzydb::metric::{l2_squared, prepare_vector};
use izzydb::{ClusteringTree, DistanceMetric, DocId, TreeConfig};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn gen_clustered_data(
    n: usize,
    dim: usize,
    num_clusters: usize,
    spread: f32,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect())
        .collect();
    let data = (0..n)
        .map(|i| {
            let c = &centers[i % num_clusters];
            c.iter()
                .map(|&x| x + rng.random_range(-spread..spread))
                .collect()
        })
        .collect();
    let queries = (0..100)
        .map(|i| {
            let c = &centers[i % num_clusters];
            c.iter()
                .map(|&x| x + rng.random_range(-spread..spread))
                .collect()
        })
        .collect();
    (data, queries)
}

fn gen_uniform_data(n: usize, dim: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let data = (0..n)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect())
        .collect();
    let queries = (0..100)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect())
        .collect();
    (data, queries)
}

fn run_bench(name: &str, data: &[Vec<f32>], queries: &[Vec<f32>], enable_lire: bool) {
    let dim = data[0].len();
    let dir = TempDir::new().unwrap();
    let mut cfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    cfg.enable_lire = enable_lire;

    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    let t0 = Instant::now();
    for (i, v) in data.iter().enumerate() {
        tree.add_leaf(i as DocId, v).unwrap();
    }
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let stats = tree.verify_invariants().unwrap();

    // Brute force ground truth
    let mut ground_truth = Vec::with_capacity(queries.len());
    for q in queries {
        let qp = prepare_vector(q, dim, DistanceMetric::L2Squared).unwrap();
        let mut scored: Vec<(DocId, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i as DocId, l2_squared(&qp, v)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(10);
        ground_truth.push(scored.into_iter().map(|(id, _)| id).collect::<HashSet<_>>());
    }

    println!(
        "[{name} | LIRE={enable_lire}] n={}, dim={}, height={}, nodes={} (int={}, leaf={}), build={build_ms:.1}ms",
        stats.num_vectors,
        dim,
        stats.root_level,
        stats.total_nodes,
        stats.internal_nodes,
        stats.leaf_nodes
    );

    for beam in [4, 8, 16, 32] {
        let sq = Instant::now();
        let mut q_hits = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let res = tree.search(q, 10, beam).unwrap();
            q_hits += res
                .iter()
                .filter(|&&(id, _)| ground_truth[qi].contains(&id))
                .count();
        }
        let q_us = sq.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        let q_recall = q_hits as f64 / (queries.len() * 10) as f64;

        let sr = Instant::now();
        let mut r_hits = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let res = tree.search_with_rerank(q, 10, beam, 100).unwrap();
            r_hits += res
                .iter()
                .filter(|&&(id, _)| ground_truth[qi].contains(&id))
                .count();
        }
        let r_us = sr.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        let r_recall = r_hits as f64 / (queries.len() * 10) as f64;

        println!(
            "  beam={beam:2} -> rabitq recall@10={q_recall:.4} ({q_us:.1}us) | rerank@100 recall@10={r_recall:.4} ({r_us:.1}us)"
        );
    }
    let _ = HashMap::<DocId, usize>::new();
}

fn main() {
    let (clustered, c_queries) = gen_clustered_data(50_000, 128, 250, 0.20, 1);
    run_bench("Gaussian-50k-128d", &clustered, &c_queries, false);
    run_bench("Gaussian-50k-128d", &clustered, &c_queries, true);

    let (uniform, u_queries) = gen_uniform_data(10_000, 128, 2);
    run_bench("Uniform-10k-128d", &uniform, &u_queries, false);
    run_bench("Uniform-10k-128d", &uniform, &u_queries, true);
}
