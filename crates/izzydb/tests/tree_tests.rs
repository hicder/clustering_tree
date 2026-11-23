use std::collections::{HashMap, HashSet};

use izzydb::{
    AttributeType, AttributeValue, Attributes, ClusteringTree, Collection, CollectionConfig,
    DistanceMetric, DocId, Document, DocumentPatch, MaintenanceOptions, RowId, SearchOptions,
    TreeConfig, TreeError, VacuumBudget,
};
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

fn brute_force_topk<K: Ord + Copy>(
    docs: &HashMap<K, Vec<f32>>,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(K, f32)> {
    let q = izzydb::metric::prepare_vector(query, query.len(), metric).unwrap();
    let mut scored: Vec<(K, f32)> = docs
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

    let wrong_dim = TreeConfig::new(16, DistanceMetric::Cosine);
    assert!(matches!(
        ClusteringTree::open(dir.path(), wrong_dim),
        Err(TreeError::ConfigMismatch { field: "dim", .. })
    ));

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
        tree.insert_row(RowId::new(1, 0), &[0.0; 3]),
        Err(TreeError::DimensionMismatch {
            expected: 4,
            actual: 3
        })
    ));
    assert!(matches!(
        tree.insert_row(RowId::new(1, 0), &[f32::NAN, 0.0, 0.0, 0.0]),
        Err(TreeError::NonFiniteVector)
    ));
    assert!(matches!(
        tree.tombstone_row_checked(RowId::new(1, 0), &[f32::INFINITY, 0.0, 0.0, 0.0]),
        Err(TreeError::NonFiniteVector)
    ));

    // Tombstone missing row is a clean no-op.
    assert!(!tree.tombstone_row_checked(RowId::new(1, 999), &[0.0; 4]).unwrap());
    tree.verify_invariants().unwrap();
}

#[test]
fn test_identical_vectors_degenerate_splits() {
    let dir = TempDir::new().unwrap();
    let cfg = small_config(8, DistanceMetric::L2Squared, true);
    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();

    let v = vec![1.5f32; 8];
    for id in 0..60u32 {
        tree.insert_row(RowId::new(1, id), &v).unwrap();
    }
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 60);
    assert!(stats.root_level >= 2);

    let res = tree.search(&v, 5, 4).unwrap();
    assert_eq!(res.len(), 5);
    for (_, d) in res {
        assert!(d < 1e-6);
    }

    for id in 0..60u32 {
        assert!(tree.tombstone_row_checked(RowId::new(1, id), &v).unwrap());
    }
    let mid = tree.verify_invariants().unwrap();
    assert_eq!(mid.num_vectors, 0);
    assert_eq!(mid.stale_rows, 60);

    let vac = tree.vacuum_ann(VacuumBudget::unlimited()).unwrap();
    assert!(vac.completed_pass);
    assert_eq!(vac.stale_slots_dropped, 60);

    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 0);
    assert_eq!(stats.stale_rows, 0);
    assert_eq!(stats.root_level, 0);
}

#[test]
fn test_small_fanout_multi_level_growth_and_collapse() {
    for enable_lire in [false, true] {
        let dir = TempDir::new().unwrap();
        let cfg = small_config(16, DistanceMetric::L2Squared, enable_lire);
        let mut rng = SmallRng::seed_from_u64(42);
        let mut rows = HashMap::new();

        {
            let mut tree = ClusteringTree::open(dir.path(), cfg.clone()).unwrap();
            for id in 0..400u32 {
                let r = RowId::new(1 + id / 100, id % 100);
                let v = random_vec(&mut rng, 16);
                tree.insert_row(r, &v).unwrap();
                rows.insert(r, v);
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

        {
            let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
            tree.verify_invariants().unwrap();

            let q = random_vec(&mut rng, 16);
            let got_q = tree.search(&q, 10, 16).unwrap();
            let got_reranked = tree.search_with_rerank(&q, 10, 16, 40).unwrap();
            let exact = brute_force_topk(&rows, &q, 10, DistanceMetric::L2Squared);
            let exact_ids: HashSet<RowId> = exact.iter().map(|&(id, _)| id).collect();
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

            // Tombstone all rows and vacuum: tree collapses cleanly back to root_level = 0.
            for id in 0..400u32 {
                let r = RowId::new(1 + id / 100, id % 100);
                let v = rows.remove(&r).unwrap();
                assert!(tree.tombstone_row_checked(r, &v).unwrap());
                if id % 100 == 99 {
                    tree.vacuum_ann(VacuumBudget::unlimited()).unwrap();
                    tree.verify_invariants().unwrap();
                }
            }
            let stats = tree.verify_invariants().unwrap();
            assert_eq!(stats.num_vectors, 0);
            assert_eq!(stats.stale_rows, 0);
            assert_eq!(stats.root_level, 0);
            assert_eq!(stats.total_nodes, 1);
        }
    }
}

#[test]
fn test_all_metrics_search_and_cow_update() {
    for metric in [
        DistanceMetric::L2Squared,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ] {
        let dir = TempDir::new().unwrap();
        let cfg = small_config(8, metric, true);
        let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
        let mut rng = SmallRng::seed_from_u64(123 + metric as u64);
        let mut rows = HashMap::new();

        for id in 0..120u32 {
            let scale = if id % 5 == 0 { 5.0 } else { 0.5 };
            let v: Vec<f32> = random_vec(&mut rng, 8).into_iter().map(|x| x * scale).collect();
            let r = RowId::new(1, id);
            tree.insert_row(r, &v).unwrap();
            rows.insert(r, v);
        }

        // CoW update 20 existing rows: tombstone old RowId in seg 1, insert new RowId in seg 2.
        for id in 0..20u32 {
            let old_r = RowId::new(1, id);
            let new_r = RowId::new(2, id);
            let v = random_vec(&mut rng, 8);
            assert!(tree.tombstone_row(old_r).unwrap());
            rows.remove(&old_r);
            tree.insert_row(new_r, &v).unwrap();
            rows.insert(new_r, v);
        }

        let stats = tree.verify_invariants().unwrap();
        assert_eq!(stats.num_vectors, 120);
        assert!(stats.stale_rows <= 20);

        let mut q_hits = 0usize;
        let mut rerank_hits = 0usize;
        for _ in 0..10 {
            let q = random_vec(&mut rng, 8);
            let got_quantized = tree.search(&q, 5, 12).unwrap();
            let got_reranked = tree.search_with_rerank(&q, 5, 12, 30).unwrap();
            for &(r, _) in got_quantized.iter().chain(got_reranked.iter()) {
                assert!(rows.contains_key(&r), "returned tombstoned row {r:?}");
            }
            let exact = brute_force_topk(&rows, &q, 5, metric);
            let exact_ids: HashSet<RowId> = exact.iter().map(|&(id, _)| id).collect();
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
fn test_randomized_fuzzer_with_vacuum_and_reopen() {
    for enable_lire in [false, true] {
        let dir = TempDir::new().unwrap();
        let cfg = small_config(12, DistanceMetric::Cosine, enable_lire);
        let mut rng = SmallRng::seed_from_u64(9999);
        let mut logical_to_row: HashMap<u64, RowId> = HashMap::new();
        let mut next_local = 0u32;

        let mut tree = ClusteringTree::open(dir.path(), cfg.clone()).unwrap();

        for step in 0..500usize {
            let op = rng.random_range(0..100);
            if op < 55 {
                let doc = rng.random_range(0..180u64);
                if let Some(old_r) = logical_to_row.remove(&doc) {
                    assert!(tree.tombstone_row(old_r).unwrap());
                }
                let r = RowId::new(1, next_local);
                next_local += 1;
                let v = if rng.random_bool(0.05) {
                    vec![0.0; 12]
                } else {
                    random_vec(&mut rng, 12)
                };
                tree.insert_row(r, &v).unwrap();
                logical_to_row.insert(doc, r);
            } else if op < 85 {
                let doc = rng.random_range(0..180u64);
                if let Some(old_r) = logical_to_row.remove(&doc) {
                    assert!(tree.tombstone_row(old_r).unwrap());
                }
            } else if op < 93 {
                tree.vacuum_ann(VacuumBudget::leaves(4)).unwrap();
            } else {
                drop(tree);
                tree = ClusteringTree::open(dir.path(), cfg.clone()).unwrap();
            }

            if step % 25 == 24 {
                let stats = tree.verify_invariants().unwrap();
                assert_eq!(stats.num_vectors, logical_to_row.len() as u64);
            }
        }

        tree.vacuum_ann(VacuumBudget::unlimited()).unwrap();
        let stats = tree.verify_invariants().unwrap();
        assert_eq!(stats.num_vectors, logical_to_row.len() as u64);
        assert_eq!(stats.stale_rows, 0);
    }
}

#[test]
fn test_clustered_recall_with_lire_and_default_fanout() {
    let dir = TempDir::new().unwrap();
    let dim = 32;
    let mut cfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    cfg.enable_lire = true;

    let mut rng = SmallRng::seed_from_u64(2026);
    let centers: Vec<Vec<f32>> = (0..40).map(|_| random_vec(&mut rng, dim)).collect();
    let mut oracle = HashMap::new();

    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    for id in 0..5_000u32 {
        let r = RowId::new(1 + id / 1_000, id % 1_000);
        let c = &centers[(id as usize) % centers.len()];
        let v: Vec<f32> = c
            .iter()
            .map(|&x| x + rng.random_range(-0.15f32..0.15f32))
            .collect();
        tree.insert_row(r, &v).unwrap();
        oracle.insert(r, v);
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
        let exact_ids: HashSet<RowId> = exact.iter().map(|&(id, _)| id).collect();
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
    let r10 = RowId::new(1, 10);
    let r20 = RowId::new(1, 20);
    tree.insert_row(r10, &v0).unwrap();
    tree.insert_row(r20, &v1).unwrap();

    assert_eq!(tree.get_vector(r10).unwrap(), Some(v0.clone()));
    assert_eq!(tree.get_vector(r20).unwrap(), Some(v1.clone()));

    // Tombstoning hides get_vector and decrements num_vectors without touching the leaf.
    assert!(tree.tombstone_row(r10).unwrap());
    assert_eq!(tree.get_vector(r10).unwrap(), None);
    let r11 = RowId::new(1, 11);
    tree.insert_row(r11, &v1).unwrap();
    assert_eq!(tree.get_vector(r11).unwrap(), Some(v1.clone()));

    tree.verify_invariants().unwrap();
    drop(tree);

    let tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    assert_eq!(tree.get_vector(r20).unwrap(), Some(v1));
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

    let base = vec![1.0f32; dim];
    for id in 0..33u32 {
        let r = RowId::new(1, id);
        let v: Vec<f32> = if id < 16 {
            base.iter()
                .map(|&x| x + rng.random_range(-0.02f32..0.02f32))
                .collect()
        } else {
            base.iter()
                .map(|&x| x + rng.random_range(-0.001f32..0.001f32))
                .collect()
        };
        tree.insert_row(r, &v).unwrap();
        oracle.insert(r, v);
    }

    let outlier = vec![10.0f32; dim];
    let outlier_row = RowId::new(1, 33);
    tree.insert_row(outlier_row, &outlier).unwrap();
    oracle.insert(outlier_row, outlier.clone());

    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 34);
    assert_eq!(stats.leaf_nodes, 1);

    for (r, expected) in &oracle {
        assert_eq!(tree.get_vector(*r).unwrap().as_ref(), Some(expected));
    }

    let q_res = tree.search(&outlier, 1, 1).unwrap();
    assert_eq!(q_res[0].0, outlier_row);
    let rerank_res = tree.search_with_rerank(&outlier, 3, 1, 10).unwrap();
    assert_eq!(rerank_res[0].0, outlier_row);
    assert!(rerank_res[0].1 < 1e-6);
}

#[test]
fn test_split_and_underflow_merge_and_redistribution_via_vacuum() {
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

    for id in 0..4u32 {
        let r = RowId::new(1, id);
        let v: Vec<f32> = (0..dim).map(|_| -5.0 + rng.random_range(-0.1..0.1)).collect();
        tree.insert_row(r, &v).unwrap();
        oracle.insert(r, v);
    }
    for id in 4..8u32 {
        let r = RowId::new(1, id);
        let v: Vec<f32> = (0..dim).map(|_| 5.0 + rng.random_range(-0.1..0.1)).collect();
        tree.insert_row(r, &v).unwrap();
        oracle.insert(r, v);
    }
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.leaf_nodes, 2);
    assert_eq!(stats.num_vectors, 8);

    // Tombstone 2 items from positive cluster -> vacuum merges back to 1 root leaf.
    for id in 4..6u32 {
        let r = RowId::new(1, id);
        let v = oracle.remove(&r).unwrap();
        assert!(tree.tombstone_row_checked(r, &v).unwrap());
    }
    tree.vacuum_ann(VacuumBudget::unlimited()).unwrap();
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.leaf_nodes, 1);
    assert_eq!(stats.root_level, 0);
    assert_eq!(stats.num_vectors, 6);
    assert_eq!(stats.stale_rows, 0);

    // Grow to 10 live vectors across 2 leaves, then delete positive down to 2 -> redistribution.
    for id in 10..12u32 {
        let r = RowId::new(1, id);
        let v: Vec<f32> = (0..dim).map(|_| -5.0 + rng.random_range(-0.1..0.1)).collect();
        tree.insert_row(r, &v).unwrap();
        oracle.insert(r, v);
    }
    for id in 20..22u32 {
        let r = RowId::new(1, id);
        let v: Vec<f32> = (0..dim).map(|_| 5.0 + rng.random_range(-0.1..0.1)).collect();
        tree.insert_row(r, &v).unwrap();
        oracle.insert(r, v);
    }
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.leaf_nodes, 2);
    assert_eq!(stats.num_vectors, 10);

    let pos_rows: Vec<RowId> = oracle
        .iter()
        .filter(|(_, v)| v[0] > 0.0)
        .map(|(&r, _)| r)
        .collect();
    for &r in pos_rows.iter().take(pos_rows.len() - 2) {
        let v = oracle.remove(&r).unwrap();
        assert!(tree.tombstone_row_checked(r, &v).unwrap());
    }
    tree.vacuum_ann(VacuumBudget::unlimited()).unwrap();

    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.leaf_nodes, 2);
    assert_eq!(stats.num_vectors, oracle.len() as u64);
    assert_eq!(stats.stale_rows, 0);
    for (r, expected) in &oracle {
        assert_eq!(tree.get_vector(*r).unwrap().as_ref(), Some(expected));
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

    let centers = [-2.0f32, -0.5, 0.5, 2.0];
    for id in 0..45u32 {
        let r = RowId::new(1, id);
        let c = centers[(id as usize) % centers.len()];
        let v: Vec<f32> = (0..dim)
            .map(|_| c + rng.random_range(-0.8f32..0.8f32))
            .collect();
        tree.insert_row(r, &v).unwrap();
        oracle.insert(r, v);
    }

    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 45);
    assert!(stats.leaf_nodes >= 7);

    for (r, expected) in &oracle {
        assert_eq!(tree.get_vector(*r).unwrap().as_ref(), Some(expected));
    }

    for c in centers {
        let q = vec![c; dim];
        let got = tree.search_with_rerank(&q, 5, 8, 20).unwrap();
        let exact = brute_force_topk(&oracle, &q, 5, DistanceMetric::L2Squared);
        let exact_ids: HashSet<RowId> = exact.iter().map(|&(id, _)| id).collect();
        let hits = got.iter().filter(|&&(id, _)| exact_ids.contains(&id)).count();
        assert!(hits >= 4, "expected >= 4/5 reranked hits near {c}, got {hits}");
    }
}

#[test]
fn test_vacuum_before_split_and_budgeted_vacuum() {
    let dir = TempDir::new().unwrap();
    let mut cfg = small_config(4, DistanceMetric::L2Squared, false);
    cfg.min_leaf_size = 2;
    cfg.max_leaf_size = 5;

    let mut tree = ClusteringTree::open(dir.path(), cfg).unwrap();
    assert!(!tree.tombstone_row(RowId::new(1, 999)).unwrap());

    // Fill the single root leaf to max_leaf_size = 5, tombstone 3, then insert 3 more.
    // Vacuum-before-split must absorb them without creating a second leaf.
    for id in 0..5u32 {
        tree.insert_row(RowId::new(1, id), &[id as f32, 0.0, 0.0, 0.0])
            .unwrap();
    }
    for id in 0..3u32 {
        assert!(tree.tombstone_row(RowId::new(1, id)).unwrap());
    }
    for id in 5..8u32 {
        tree.insert_row(RowId::new(1, id), &[id as f32, 0.0, 0.0, 0.0])
            .unwrap();
    }
    let stats = tree.verify_invariants().unwrap();
    assert_eq!(stats.num_vectors, 5);
    assert_eq!(stats.stale_rows, 0);
    assert_eq!(stats.leaf_nodes, 1);

    // Grow to several leaves, tombstone half across multiple leaves, then vacuum incrementally.
    for id in 8..30u32 {
        tree.insert_row(RowId::new(1, id), &[id as f32, 0.0, 0.0, 0.0])
            .unwrap();
    }
    for id in (3..30u32).step_by(2) {
        assert!(tree.tombstone_row(RowId::new(1, id)).unwrap());
    }
    let before = tree.verify_invariants().unwrap();
    assert!(before.stale_rows > 0);
    assert!(before.leaf_nodes >= 4);

    let step1 = tree.vacuum_ann(VacuumBudget::leaves(2)).unwrap();
    assert_eq!(step1.leaves_scanned, 2);
    assert!(!step1.completed_pass);

    let mut total_steps = 1usize;
    loop {
        let s = tree.vacuum_ann(VacuumBudget::leaves(2)).unwrap();
        total_steps += 1;
        if s.completed_pass {
            break;
        }
        assert!(total_steps < 20);
    }
    // Finish wrap-around if needed via unlimited pass; stale_rows must be 0 after full pass.
    tree.vacuum_ann(VacuumBudget::unlimited()).unwrap();
    let after = tree.verify_invariants().unwrap();
    assert_eq!(after.stale_rows, 0);
    assert_eq!(after.num_vectors, before.num_vectors);
}

#[test]
fn test_collection_delete_30_percent_and_vacuum_recall_parity() {
    let dim = 16;
    let mut vcfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    vcfg.min_leaf_size = 8;
    vcfg.max_leaf_size = 32;
    vcfg.enable_lire = true;

    let mut rng = SmallRng::seed_from_u64(3030);
    let centers: Vec<Vec<f32>> = (0..12).map(|_| random_vec(&mut rng, dim)).collect();
    let mut all_docs = Vec::with_capacity(1_200);
    for id in 0..1_200u64 {
        let c = &centers[(id as usize) % centers.len()];
        let emb: Vec<f32> = c
            .iter()
            .map(|&x| x + rng.random_range(-0.2f32..0.2f32))
            .collect();
        all_docs.push((id, emb));
    }

    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(vcfg.clone()).with_seal_rows(250);
    let mut col = Collection::open(dir.path(), cfg).unwrap();
    let docs: Vec<Document> = all_docs
        .iter()
        .map(|(id, emb)| Document {
            id: *id,
            embedding: Some(emb.clone()),
            attributes: Attributes::new(),
        })
        .collect();
    col.upsert_batch(&docs).unwrap();

    // Delete 30% of docs (every id % 10 < 3 => 360 docs).
    let to_delete: Vec<DocId> = (0..1_200u64).filter(|id| id % 10 < 3).collect();
    assert_eq!(to_delete.len(), 360);
    let deleted = col.delete_batch(&to_delete).unwrap();
    assert_eq!(deleted, 360);

    let meta = col.ann_meta().unwrap();
    assert_eq!(meta.num_vectors, 840);
    assert_eq!(meta.stale_rows, 360);
    col.verify_ann_invariants().unwrap();

    let deleted_set: HashSet<DocId> = to_delete.iter().copied().collect();
    let surviving_oracle: HashMap<DocId, Vec<f32>> = all_docs
        .iter()
        .filter(|(id, _)| !deleted_set.contains(id))
        .map(|(id, emb)| (*id, emb.clone()))
        .collect();

    // Fresh baseline over surviving docs.
    let fresh_dir = TempDir::new().unwrap();
    let mut fresh = Collection::open(fresh_dir.path(), CollectionConfig::new(vcfg).with_seal_rows(250)).unwrap();
    let fresh_docs: Vec<Document> = surviving_oracle
        .iter()
        .map(|(&id, emb)| Document {
            id,
            embedding: Some(emb.clone()),
            attributes: Attributes::new(),
        })
        .collect();
    fresh.upsert_batch(&fresh_docs).unwrap();

    let queries: Vec<Vec<f32>> = (0..20)
        .map(|i| {
            centers[i % centers.len()]
                .iter()
                .map(|&x| x + rng.random_range(-0.15f32..0.15f32))
                .collect()
        })
        .collect();

    // Pre-vacuum search must never return deleted docs.
    for q in &queries {
        let hits = col
            .search(q, SearchOptions::new(10, 8).with_rerank(40), None)
            .unwrap();
        for h in hits {
            assert!(
                !deleted_set.contains(&h.id),
                "pre-vacuum search returned deleted doc {}",
                h.id
            );
        }
    }

    // maintenance() sees stale_ratio = 360 / 1200 = 0.3 > 0.2 and vacuums to 0 stale rows.
    let mstats = col.maintenance(MaintenanceOptions::default()).unwrap();
    assert!(mstats.vacuum.is_some());
    let stats_after = col.verify_ann_invariants().unwrap().unwrap();
    assert_eq!(stats_after.stale_rows, 0);
    assert_eq!(stats_after.num_vectors, 840);

    let mut vac_hits = 0usize;
    let mut fresh_hits = 0usize;
    for q in &queries {
        let exact = brute_force_topk(&surviving_oracle, q, 10, DistanceMetric::L2Squared);
        let exact_ids: HashSet<DocId> = exact.iter().map(|&(id, _)| id).collect();

        let got = col
            .search(q, SearchOptions::new(10, 8).with_rerank(40), None)
            .unwrap();
        let got_fresh = fresh
            .search(q, SearchOptions::new(10, 8).with_rerank(40), None)
            .unwrap();

        for h in &got {
            assert!(!deleted_set.contains(&h.id));
        }
        vac_hits += got.iter().filter(|h| exact_ids.contains(&h.id)).count();
        fresh_hits += got_fresh
            .iter()
            .filter(|h| exact_ids.contains(&h.id))
            .count();
    }
    let vac_recall = vac_hits as f64 / (queries.len() * 10) as f64;
    let fresh_recall = fresh_hits as f64 / (queries.len() * 10) as f64;
    assert!(vac_recall >= 0.85, "post-vacuum recall too low: {vac_recall:.3}");
    assert!(
        (vac_recall - fresh_recall).abs() <= 0.05,
        "post-vacuum recall {vac_recall:.3} diverged from fresh {fresh_recall:.3}"
    );
}

#[test]
fn test_collection_update_heavy_patch_3x() {
    let dim = 16;
    let mut vcfg = TreeConfig::new(dim, DistanceMetric::Cosine);
    vcfg.min_leaf_size = 6;
    vcfg.max_leaf_size = 24;

    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(vcfg.clone())
        .with_seal_rows(150)
        .with_attribute("ver", AttributeType::Uint, false);
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    let mut rng = SmallRng::seed_from_u64(4242);
    let mut oracle = HashMap::new();
    let mut docs = Vec::with_capacity(400);
    for id in 0..400u64 {
        let emb = random_vec(&mut rng, dim);
        let mut attrs = Attributes::new();
        attrs.insert("ver".into(), AttributeValue::Uint(0));
        docs.push(Document {
            id,
            embedding: Some(emb.clone()),
            attributes: attrs,
        });
        oracle.insert(id, emb);
    }
    col.upsert_batch(&docs).unwrap();

    // Baseline recall before patching.
    let queries: Vec<Vec<f32>> = (0..20).map(|_| random_vec(&mut rng, dim)).collect();
    let baseline_recall: f64 = queries
        .iter()
        .map(|q| {
            let exact: HashSet<DocId> = brute_force_topk(&oracle, q, 10, DistanceMetric::Cosine)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            col.search(q, SearchOptions::new(10, 10).with_rerank(50), None)
                .unwrap()
                .into_iter()
                .filter(|h| exact.contains(&h.id))
                .count()
        })
        .sum::<usize>() as f64
        / (queries.len() * 10) as f64;

    // Patch every doc's attribute 3x (without touching vectors).
    for round in 1..=3u64 {
        let patches: Vec<(DocId, DocumentPatch)> = (0..400u64)
            .map(|id| (id, DocumentPatch::new().with_attribute("ver", AttributeValue::Uint(round))))
            .collect();
        col.patch_batch(&patches).unwrap();
    }

    let stats_pre = col.verify_ann_invariants().unwrap().unwrap();
    assert_eq!(stats_pre.num_vectors, 400);
    assert!(stats_pre.stale_rows <= 1_200);

    // Search even before vacuum returns unique docs and preserves recall.
    let patched_hits: usize = queries
        .iter()
        .map(|q| {
            let exact: HashSet<DocId> = brute_force_topk(&oracle, q, 10, DistanceMetric::Cosine)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let got = col
                .search(q, SearchOptions::new(10, 10).with_rerank(50), None)
                .unwrap();
            let uniq: HashSet<DocId> = got.iter().map(|h| h.id).collect();
            assert_eq!(uniq.len(), got.len(), "duplicate DocId in search results");
            got.into_iter().filter(|h| exact.contains(&h.id)).count()
        })
        .sum();
    let patched_recall = patched_hits as f64 / (queries.len() * 10) as f64;
    assert!(
        patched_recall >= baseline_recall - 0.01,
        "patched recall {patched_recall:.3} vs baseline {baseline_recall:.3}"
    );

    col.vacuum_ann(VacuumBudget::unlimited()).unwrap();
    let stats_post = col.verify_ann_invariants().unwrap().unwrap();
    assert_eq!(stats_post.stale_rows, 0);
    assert_eq!(stats_post.num_vectors, 400);
}

#[test]
fn test_collection_optional_vectors() {
    // 1. Vectorless collection rejects vector writes and vector search.
    let dir1 = TempDir::new().unwrap();
    let cfg1 = CollectionConfig::without_vectors()
        .with_attribute("tag", AttributeType::String, false);
    let mut no_vec = Collection::open(dir1.path(), cfg1).unwrap();
    let mut attrs = Attributes::new();
    attrs.insert("tag".into(), AttributeValue::String("a".into()));
    no_vec.upsert(1, None, attrs.clone()).unwrap();
    assert!(matches!(
        no_vec.upsert(2, Some(&[1.0, 2.0]), attrs.clone()),
        Err(TreeError::InvalidConfig(_))
    ));
    assert!(matches!(
        no_vec.search(&[1.0, 2.0], SearchOptions::new(5, 4), None),
        Err(TreeError::InvalidConfig(_))
    ));

    // 2. Vector collection supports mixed vectorized and vectorless documents.
    let dir2 = TempDir::new().unwrap();
    let cfg2 = CollectionConfig::new(small_config(4, DistanceMetric::L2Squared, false))
        .with_attribute("tag", AttributeType::String, false);
    let mut col = Collection::open(dir2.path(), cfg2).unwrap();
    col.upsert(1, Some(&[1.0, 0.0, 0.0, 0.0]), attrs.clone())
        .unwrap();
    col.upsert(2, None, attrs.clone()).unwrap();
    col.upsert(3, Some(&[0.0, 1.0, 0.0, 0.0]), attrs.clone())
        .unwrap();

    assert_eq!(col.len(), 3);
    assert_eq!(col.ann_meta().unwrap().num_vectors, 2);
    let res = col
        .search(&[1.0, 0.0, 0.0, 0.0], SearchOptions::new(10, 4).with_rerank(10), None)
        .unwrap();
    let ids: Vec<DocId> = res.iter().map(|r| r.id).collect();
    assert_eq!(ids, vec![1, 3]);

    // Transition doc 2 from None -> Some(vec), and doc 3 from Some -> None via upsert.
    col.upsert(2, Some(&[0.9, 0.0, 0.0, 0.0]), attrs.clone())
        .unwrap();
    col.upsert(3, None, attrs).unwrap();
    assert_eq!(col.len(), 3);
    let stats = col.verify_ann_invariants().unwrap().unwrap();
    assert_eq!(stats.num_vectors, 2);
    assert_eq!(stats.stale_rows, 1);

    col.vacuum_ann(VacuumBudget::unlimited()).unwrap();
    let stats = col.verify_ann_invariants().unwrap().unwrap();
    assert_eq!(stats.num_vectors, 2);
    assert_eq!(stats.stale_rows, 0);

    let res = col
        .search(&[1.0, 0.0, 0.0, 0.0], SearchOptions::new(10, 4).with_rerank(10), None)
        .unwrap();
    let ids: Vec<DocId> = res.iter().map(|r| r.id).collect();
    assert_eq!(ids, vec![1, 2]);
}
