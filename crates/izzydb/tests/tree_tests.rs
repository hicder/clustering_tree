use std::collections::{HashMap, HashSet};

use izzydb::{ClusteringTree, DistanceMetric, DocId, TreeConfig, TreeError};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn small_config(dim: usize, metric: DistanceMetric, enable_lire: bool) -> TreeConfig {
    TreeConfig {
        dim,
        metric,
        min_internal_fanout: 2,
        max_internal_fanout: 5,
        min_leaf_size: 2,
        max_leaf_size: 5,
        insert_beam_width: 4,
        lire_neighbor_count: 4,
        enable_lire,
        centroid_drift_eps: 1e-4,
    }
}

fn brute_force_topk(
    docs: &HashMap<DocId, Vec<f32>>,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(DocId, f32)> {
    let q = izzydb::metric::prepare_vector(query, query.len(), metric).unwrap();
    let mut scored: Vec<(DocId, f32)> = docs
        .iter()
        .map(|(&id, v)| {
            let vp = izzydb::metric::prepare_vector(v, v.len(), metric).unwrap();
            (id, izzydb::metric::search_distance(metric, &q, &vp))
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

fn random_vec(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect()
}

#[test]
fn test_config_validation_and_reopen_mismatch() {
    let dir = TempDir::new().unwrap();

    let mut bad = TreeConfig::new(0, DistanceMetric::L2Squared);
    assert!(matches!(
        ClusteringTree::open(dir.path(), bad.clone()),
        Err(TreeError::InvalidConfig(_))
    ));

    bad = TreeConfig::new(8, DistanceMetric::L2Squared);
    bad.min_leaf_size = 5;
    bad.max_leaf_size = 10; // 5 * 2 == 10, must be <
    assert!(matches!(
        ClusteringTree::open(dir.path(), bad),
        Err(TreeError::InvalidConfig(_))
    ));

    let good = TreeConfig::new(8, DistanceMetric::Cosine);
    {
        let _tree = ClusteringTree::open(dir.path(), good).unwrap();
    }

    // Reopen with wrong dimension
    let wrong_dim = TreeConfig::new(16, DistanceMetric::Cosine);
    assert!(matches!(
        ClusteringTree::open(dir.path(), wrong_dim),
        Err(TreeError::ConfigMismatch { field: "dim", .. })
    ));

    // Reopen with wrong metric
    let wrong_metric = TreeConfig::new(8, DistanceMetric::L2Squared);
    assert!(matches!(
        ClusteringTree::open(dir.path(), wrong_metric),
        Err(TreeError::ConfigMismatch {
            field: "metric",
            ..
        })
    ));
}

#[test]
fn test_empty_tree_and_input_validation() {
    let dir = TempDir::new().unwrap();
    let cfg = small_config(4, DistanceMetric::L2Squared, true);
    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();

    assert!(tree.is_empty());
    assert_eq!(tree.len(), 0);
    assert_eq!(tree.search(&[0.0; 4], 5, 4).unwrap(), vec![]);
    assert_eq!(tree.search(&[0.0; 4], 0, 4).unwrap(), vec![]);

    assert!(matches!(
        tree.search(&[0.0; 4], 5, 0),
        Err(TreeError::InvalidConfig(_))
    ));
    assert!(matches!(
        tree.add_leaf(1, &[0.0; 3]),
        Err(TreeError::DimensionMismatch {
            expected: 4,
            actual: 3
        })
    ));
    assert!(matches!(
        tree.add_leaf(1, &[f32::NAN, 0.0, 0.0, 0.0]),
        Err(TreeError::NonFiniteVector)
    ));
    assert!(matches!(
        tree.remove_leaf(1, &[f32::INFINITY, 0.0, 0.0, 0.0]),
        Err(TreeError::NonFiniteVector)
    ));

    // Remove missing id is a clean no-op
    tree.remove_leaf(999, &[0.0; 4]).unwrap();
    tree.verify_invariants().unwrap();
}

#[test]
fn test_identical_vectors_degenerate_splits() {
    let dir = TempDir::new().unwrap();
    let cfg = small_config(8, DistanceMetric::L2Squared, true);
    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();

    let v = vec![1.5f32; 8];
    for id in 0..60 {
        tree.add_leaf(id, &v).unwrap();
    }
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 60);
    assert!(stats.root_level >= 2);

    let res = tree.search(&v, 5, 4).unwrap();
    assert_eq!(res.len(), 5);
    for (_, d) in res {
        assert!(d < 1e-6);
    }

    for id in 0..60 {
        tree.remove_leaf(id, &v).unwrap();
    }
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 0);
    assert_eq!(stats.root_level, 0);
}

#[test]
fn test_small_fanout_multi_level_growth_and_collapse() {
    for enable_lire in [false, true] {
        let dir = TempDir::new().unwrap();
        let cfg = small_config(16, DistanceMetric::L2Squared, enable_lire);
        let mut rng = SmallRng::seed_from_u64(42);
        let mut docs = HashMap::new();

        {
            let mut tree = ClusteringTree::open(dir.path(), cfg.clone()).unwrap();
            for id in 0..400u64 {
                let v = random_vec(&mut rng, 16);
                tree.add_leaf(id, &v).unwrap();
                docs.insert(id, v);
                if id % 40 == 39 {
                    tree.verify_invariants().unwrap();
                }
            }
            let stats = tree.verify_invariants().unwrap();
            assert_eq!(stats.num_vectors, 400);
            assert!(
                stats.root_level >= 3,
                "expected height >= 3 with max=5 and 400 vectors, got {}",
                stats.root_level
            );
        }

        // Reopen from disk and verify invariants + search before collapsing
        {
            let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
            tree.verify_invariants().unwrap();

            let q = random_vec(&mut rng, 16);
            let got_q = tree.search(&q, 10, 16).unwrap();
            let got_reranked = tree.search_with_rerank(&q, 10, 16, 40).unwrap();
            let exact = brute_force_topk(&docs, &q, 10, DistanceMetric::L2Squared);
            let exact_ids: HashSet<DocId> = exact.iter().map(|&(id, _)| id).collect();
            let q_hits = got_q.iter().filter(|&&(id, _)| exact_ids.contains(&id)).count();
            let rerank_hits = got_reranked
                .iter()
                .filter(|&&(id, _)| exact_ids.contains(&id))
                .count();
            assert!(
                q_hits >= 5,
                "expected >= 5/10 pure RaBitQ recall with beam=16, got {q_hits}"
            );
            assert!(
                rerank_hits >= 8,
                "expected >= 8/10 reranked recall with beam=16, got {rerank_hits}"
            );

            // Delete all vectors and watch tree collapse cleanly back to root_level = 0
            for id in 0..400u64 {
                let v = docs.remove(&id).unwrap();
                tree.remove_leaf(id, &v).unwrap();
                if id % 40 == 39 {
                    tree.verify_invariants().unwrap();
                }
            }
            let stats = tree.verify_invariants().unwrap();
            assert_eq!(stats.num_vectors, 0);
            assert_eq!(stats.root_level, 0);
            assert_eq!(stats.total_nodes, 1);
        }
    }
}

#[test]
fn test_all_metrics_search_and_upsert() {
    for metric in [
        DistanceMetric::L2Squared,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ] {
        let dir = TempDir::new().unwrap();
        let cfg = small_config(8, metric, true);
        let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
        let mut rng = SmallRng::seed_from_u64(123 + metric as u64);
        let mut docs = HashMap::new();

        // Include varying norms (especially important for DotProduct)
        for id in 0..120u64 {
            let scale = if id % 5 == 0 { 5.0 } else { 0.5 };
            let v: Vec<f32> = random_vec(&mut rng, 8).into_iter().map(|x| x * scale).collect();
            tree.add_leaf(id, &v).unwrap();
            docs.insert(id, v);
        }

        // Upsert 20 existing IDs with new embeddings
        for id in 0..20u64 {
            let v = random_vec(&mut rng, 8);
            tree.add_leaf(id, &v).unwrap();
            docs.insert(id, v);
        }

        let stats = tree.verify_invariants().unwrap();
        assert_eq!(stats.num_vectors, 120);

        // Verify recall against brute force on 10 queries:
        // 1) Pure RaBitQ quantized posting-list scan
        // 2) RaBitQ + full-vector reranking from cf_vectors
        let mut q_hits = 0usize;
        let mut rerank_hits = 0usize;
        for _ in 0..10 {
            let q = random_vec(&mut rng, 8);
            let got_quantized = tree.search(&q, 5, 12).unwrap();
            let got_reranked = tree.search_with_rerank(&q, 5, 12, 30).unwrap();
            let exact = brute_force_topk(&docs, &q, 5, metric);
            let exact_ids: HashSet<DocId> = exact.iter().map(|&(id, _)| id).collect();
            q_hits += got_quantized
                .iter()
                .filter(|&&(id, _)| exact_ids.contains(&id))
                .count();
            rerank_hits += got_reranked
                .iter()
                .filter(|&&(id, _)| exact_ids.contains(&id))
                .count();
        }
        assert!(
            q_hits >= 25,
            "metric {metric:?} pure RaBitQ expected >= 25/50 hits with beam=12, got {q_hits}"
        );
        assert!(
            rerank_hits >= 40,
            "metric {metric:?} reranked expected >= 40/50 hits with beam=12, got {rerank_hits}"
        );
    }
}

#[test]
fn test_randomized_fuzzer_with_reopen() {
    for enable_lire in [false, true] {
        let dir = TempDir::new().unwrap();
        let cfg = small_config(12, DistanceMetric::Cosine, enable_lire);
        let mut rng = SmallRng::seed_from_u64(9999);
        let mut oracle: HashMap<DocId, Vec<f32>> = HashMap::new();

        let mut tree = ClusteringTree::open(dir.path(), cfg.clone()).unwrap();

        for step in 0..600usize {
            let op = rng.random_range(0..100);
            if op < 55 {
                // Insert new or existing id (from 0..180)
                let id = rng.random_range(0..180u64);
                let v = if rng.random_bool(0.05) {
                    // Occasional zero vector edge case for Cosine
                    vec![0.0; 12]
                } else {
                    random_vec(&mut rng, 12)
                };
                tree.add_leaf(id, &v).unwrap();
                oracle.insert(id, v);
            } else if op < 90 {
                // Delete existing or missing id
                let id = rng.random_range(0..180u64);
                let dummy = vec![0.0; 12];
                tree.remove_leaf(id, &dummy).unwrap();
                oracle.remove(&id);
            } else {
                // Reopen from disk
                drop(tree);
                tree = ClusteringTree::open(dir.path(), cfg.clone()).unwrap();
            }

            if step % 25 == 24 {
                let stats = tree.verify_invariants().unwrap();
                assert_eq!(stats.num_vectors, oracle.len() as u64);
            }
        }

        let stats = tree.verify_invariants().unwrap();
        assert_eq!(stats.num_vectors, oracle.len() as u64);
    }
}

#[test]
fn test_clustered_recall_with_lire_and_default_fanout() {
    // 5,000 32-d vectors across 40 Gaussian clusters; tests ~100x default fanout (32..=160)
    // reaching level >= 2 and verifies high recall@10.
    let dir = TempDir::new().unwrap();
    let dim = 32;
    let mut cfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    cfg.enable_lire = true;

    let mut rng = SmallRng::seed_from_u64(2026);
    let centers: Vec<Vec<f32>> = (0..40).map(|_| random_vec(&mut rng, dim)).collect();
    let mut oracle = HashMap::new();

    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    for id in 0..5_000u64 {
        let c = &centers[(id as usize) % centers.len()];
        let v: Vec<f32> = c
            .iter()
            .map(|&x| x + rng.random_range(-0.15f32..0.15f32))
            .collect();
        tree.add_leaf(id, &v).unwrap();
        oracle.insert(id, v);
    }

    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 5_000);
    assert!(stats.root_level >= 1);

    let mut q_hits = 0usize;
    let mut rerank_hits = 0usize;
    let queries = 25;
    for q_idx in 0..queries {
        let c = &centers[q_idx % centers.len()];
        let q: Vec<f32> = c
            .iter()
            .map(|&x| x + rng.random_range(-0.15f32..0.15f32))
            .collect();
        let got_q = tree.search(&q, 10, 8).unwrap();
        let got_rerank = tree.search_with_rerank(&q, 10, 8, 80).unwrap();
        let exact = brute_force_topk(&oracle, &q, 10, DistanceMetric::L2Squared);
        let exact_ids: HashSet<DocId> = exact.iter().map(|&(id, _)| id).collect();
        q_hits += got_q.iter().filter(|&&(id, _)| exact_ids.contains(&id)).count();
        rerank_hits += got_rerank
            .iter()
            .filter(|&&(id, _)| exact_ids.contains(&id))
            .count();
    }
    let q_recall = q_hits as f64 / (queries * 10) as f64;
    let rerank_recall = rerank_hits as f64 / (queries * 10) as f64;
    assert!(
        q_recall >= 0.55,
        "expected pure RaBitQ recall@10 >= 0.55 on clustered 5k dataset, got {q_recall:.3}"
    );
    assert!(
        rerank_recall >= 0.90,
        "expected reranked recall@10 >= 0.90 on clustered 5k dataset, got {rerank_recall:.3}"
    );
}

#[test]
fn test_posting_list_stores_quantized_code_and_vectors_cf_stores_full_vector() {
    let dir = TempDir::new().unwrap();
    let dim = 128;
    let cfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    let mut tree = ClusteringTree::open(dir.path(), cfg.clone()).unwrap();

    let mut rng = SmallRng::seed_from_u64(777);
    let v0 = random_vec(&mut rng, dim);
    let v1 = random_vec(&mut rng, dim);
    tree.add_leaf(10, &v0).unwrap();
    tree.add_leaf(20, &v1).unwrap();

    // Full vectors are stored in cf_vectors
    assert_eq!(tree.get_vector(10).unwrap(), Some(v0.clone()));
    assert_eq!(tree.get_vector(20).unwrap(), Some(v1.clone()));

    // Upsert updates full vector and remove deletes it
    tree.add_leaf(10, &v1).unwrap();
    assert_eq!(tree.get_vector(10).unwrap(), Some(v1.clone()));
    tree.remove_leaf(10, &v1).unwrap();
    assert_eq!(tree.get_vector(10).unwrap(), None);

    tree.verify_invariants().unwrap();
    drop(tree);

    // Reopen and verify persisted RaBitQ seed + full vectors + quantized posting list
    let tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    assert_eq!(tree.get_vector(20).unwrap(), Some(v1));
    tree.verify_invariants().unwrap();
}

#[test]
fn test_add_leaf_fast_path_and_reencode_paths() {
    let dir = TempDir::new().unwrap();
    let dim = 16;
    let mut cfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    cfg.min_leaf_size = 8;
    cfg.max_leaf_size = 40;
    cfg.centroid_drift_eps = 1e-3;
    cfg.enable_lire = false;

    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    let mut rng = SmallRng::seed_from_u64(5150);
    let mut oracle = HashMap::new();

    // 1..=16: <= 16 and % 16 == 0 re-encode paths
    // 17..=31: tight vectors near [1.0; dim] -> fast path (no drift, not % 16)
    // 32: % 16 == 0 exact mean + re-encode path
    let base = vec![1.0f32; dim];
    for id in 0..33u64 {
        let v: Vec<f32> = if id < 16 {
            base.iter()
                .map(|&x| x + rng.random_range(-0.02f32..0.02f32))
                .collect()
        } else {
            // Very small perturbation keeps drift well below centroid_drift_eps = 1e-3.
            base.iter()
                .map(|&x| x + rng.random_range(-0.001f32..0.001f32))
                .collect()
        };
        tree.add_leaf(id, &v).unwrap();
        oracle.insert(id, v);
    }

    // 34th vector (new_count = 34, not % 16): large offset triggers drifted re-encode path.
    let outlier = vec![10.0f32; dim];
    tree.add_leaf(33, &outlier).unwrap();
    oracle.insert(33, outlier.clone());

    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 34);
    assert_eq!(stats.leaf_nodes, 1);

    for (id, expected) in &oracle {
        assert_eq!(tree.get_vector(*id).unwrap().as_ref(), Some(expected));
    }

    // Both quantized Stage 1 and Stage 2 rerank must locate the outlier immediately.
    let q_res = tree.search(&outlier, 1, 1).unwrap();
    assert_eq!(q_res[0].0, 33);
    let rerank_res = tree.search_with_rerank(&outlier, 3, 1, 10).unwrap();
    assert_eq!(rerank_res[0].0, 33);
    assert!(rerank_res[0].1 < 1e-6);
}

#[test]
fn test_split_and_underflow_merge_and_redistribution() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let mut cfg = small_config(dim, DistanceMetric::L2Squared, false);
    cfg.min_leaf_size = 3;
    cfg.max_leaf_size = 7;
    cfg.min_internal_fanout = 2;
    cfg.max_internal_fanout = 5;

    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    let mut rng = SmallRng::seed_from_u64(31415);
    let mut oracle = HashMap::new();

    // Insert two separated clusters (4 + 4 = 8 > max_leaf_size=7) so split creates 2 leaves of 4 each.
    for id in 0..4u64 {
        let v: Vec<f32> = (0..dim).map(|_| -5.0 + rng.random_range(-0.1..0.1)).collect();
        tree.add_leaf(id, &v).unwrap();
        oracle.insert(id, v);
    }
    for id in 4..8u64 {
        let v: Vec<f32> = (0..dim).map(|_| 5.0 + rng.random_range(-0.1..0.1)).collect();
        tree.add_leaf(id, &v).unwrap();
        oracle.insert(id, v);
    }
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.leaf_nodes, 2);
    assert_eq!(stats.num_vectors, 8);

    // Delete 2 items from the positive cluster (4 -> 3 -> 2 < min_leaf_size=3):
    // combined count is 4 + 2 = 6 <= max_leaf_size=7 -> triggers merge back to 1 root leaf.
    for id in 4..6u64 {
        let v = oracle.remove(&id).unwrap();
        tree.remove_leaf(id, &v).unwrap();
        tree.verify_invariants().unwrap();
    }
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.leaf_nodes, 1);
    assert_eq!(stats.root_level, 0);
    assert_eq!(stats.num_vectors, 6);

    // Grow to 6 negative + 4 positive = 10 vectors across 2 leaves (4 neg + 2 pos + 2 neg + 2 pos):
    // deleting 2 positive vectors leaves 6 + 2 = 8 > max_leaf_size=7 -> triggers redistribution!
    for id in 10..12u64 {
        let v: Vec<f32> = (0..dim).map(|_| -5.0 + rng.random_range(-0.1..0.1)).collect();
        tree.add_leaf(id, &v).unwrap();
        oracle.insert(id, v);
    }
    for id in 20..22u64 {
        let v: Vec<f32> = (0..dim).map(|_| 5.0 + rng.random_range(-0.1..0.1)).collect();
        tree.add_leaf(id, &v).unwrap();
        oracle.insert(id, v);
    }
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.leaf_nodes, 2);
    assert_eq!(stats.num_vectors, 10);

    // Delete positive vectors down to 2 in the positive leaf (total remaining >= 8 > 7).
    let pos_ids: Vec<DocId> = oracle
        .iter()
        .filter(|(_, v)| v[0] > 0.0)
        .map(|(&id, _)| id)
        .collect();
    for &id in pos_ids.iter().take(pos_ids.len() - 2) {
        let v = oracle.remove(&id).unwrap();
        tree.remove_leaf(id, &v).unwrap();
        tree.verify_invariants().unwrap();
    }

    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.leaf_nodes, 2);
    assert_eq!(stats.num_vectors, oracle.len() as u64);
    for (id, expected) in &oracle {
        assert_eq!(tree.get_vector(*id).unwrap().as_ref(), Some(expected));
    }
}

#[test]
fn test_lire_reassignment_preserves_vectors_and_invariants() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let mut cfg = small_config(dim, DistanceMetric::L2Squared, true);
    cfg.min_leaf_size = 3;
    cfg.max_leaf_size = 7;
    cfg.lire_neighbor_count = 4;

    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    let mut rng = SmallRng::seed_from_u64(8080);
    let mut oracle = HashMap::new();

    // Insert 45 vectors across 4 overlapping clusters so splits trigger both outward and inward LIRE moves.
    let centers = [-2.0f32, -0.5, 0.5, 2.0];
    for id in 0..45u64 {
        let c = centers[(id as usize) % centers.len()];
        let v: Vec<f32> = (0..dim)
            .map(|_| c + rng.random_range(-0.8f32..0.8f32))
            .collect();
        tree.add_leaf(id, &v).unwrap();
        oracle.insert(id, v);
    }

    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 45);
    assert!(stats.leaf_nodes >= 7);

    for (id, expected) in &oracle {
        assert_eq!(tree.get_vector(*id).unwrap().as_ref(), Some(expected));
    }

    for c in centers {
        let q = vec![c; dim];
        let got = tree.search_with_rerank(&q, 5, 8, 20).unwrap();
        let exact = brute_force_topk(&oracle, &q, 5, DistanceMetric::L2Squared);
        let exact_ids: HashSet<DocId> = exact.iter().map(|&(id, _)| id).collect();
        let hits = got.iter().filter(|&&(id, _)| exact_ids.contains(&id)).count();
        assert!(hits >= 4, "expected >= 4/5 reranked hits near {c}, got {hits}");
    }
}

#[test]
fn test_remove_by_id_same_leaf_update_and_local_id_compaction() {
    let dir = TempDir::new().unwrap();
    let mut cfg = small_config(4, DistanceMetric::L2Squared, false);
    cfg.min_leaf_size = 2;
    cfg.max_leaf_size = 5;

    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    assert!(!tree.remove(999).unwrap());

    // Insert 4 docs into the root leaf, then churn > 1024 inserts+deletes without splitting.
    for id in 0..4u64 {
        tree.add_leaf(id, &[id as f32, 0.0, 0.0, 0.0]).unwrap();
    }
    // Same-leaf update preserves leaf_id & local_id
    tree.add_leaf(1, &[1.05, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(tree.len(), 4);

    // Churn 1200 single-doc add+remove cycles in the single leaf -> triggers LocalId compaction at 1024.
    for step in 0..1200u64 {
        let temp_id = 10_000 + step;
        tree.add_leaf(temp_id, &[2.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(tree.remove(temp_id).unwrap());
    }
    assert!(!tree.remove(10_000).unwrap());

    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 4);
    assert_eq!(stats.leaf_nodes, 1);

    let hits = tree.search_with_rerank(&[1.05, 0.0, 0.0, 0.0], 1, 2, 4).unwrap();
    assert_eq!(hits[0].0, 1);

    for id in 0..4u64 {
        assert!(tree.remove(id).unwrap());
    }
    assert_eq!(tree.verify_invariants().unwrap().num_vectors, 0);
}
