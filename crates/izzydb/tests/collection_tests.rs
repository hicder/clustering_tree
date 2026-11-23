#![cfg(any())] // v3: phase 4
use std::collections::{HashMap, HashSet};

use izzydb::{
    AttributeType, AttributeValue, Attributes, Collection, CollectionConfig, DistanceMetric, DocId,
    DocumentPatch, Filter, SearchOptions, TreeConfig, TreeError,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn small_collection_config(
    dim: usize,
    metric: DistanceMetric,
    enable_lire: bool,
) -> CollectionConfig {
    let tree = TreeConfig {
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
    };
    CollectionConfig::new(tree)
        .with_attribute("category", AttributeType::String, true)
        .with_attribute("score", AttributeType::Float, true)
        .with_attribute("rank", AttributeType::Int, true)
        .with_attribute("count", AttributeType::Uint, true)
        .with_attribute("active", AttributeType::Bool, true)
        .with_attribute("tags", AttributeType::StringArray, true)
        .with_attribute("nums", AttributeType::IntArray, true)
        .with_attribute("note", AttributeType::String, false)
}

fn random_vec(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect()
}

fn brute_force_filtered(
    docs: &HashMap<DocId, (Vec<f32>, Attributes)>,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
    filter: Option<&Filter>,
) -> Vec<(DocId, f32)> {
    let q = izzydb::metric::prepare_vector(query, query.len(), metric).unwrap();
    let mut scored: Vec<(DocId, f32)> = docs
        .iter()
        .filter(|(_, (_, attrs))| filter.is_none_or(|f| f.matches(attrs)))
        .map(|(&id, (v, _))| {
            let vp = izzydb::metric::prepare_vector(v, v.len(), metric).unwrap();
            (id, izzydb::metric::search_distance(metric, &q, &vp))
        })
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

#[test]
fn test_schema_validation_and_reopen_reconcile() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(8, DistanceMetric::L2Squared, true);

    {
        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();

        // Unknown attribute rejected
        let mut bad_attrs = Attributes::new();
        bad_attrs.insert("unknown".into(), "x".into());
        assert!(matches!(
            col.upsert(1, &[0.0; 8], bad_attrs),
            Err(TreeError::InvalidAttribute { .. })
        ));

        // Mismatched type rejected
        let mut bad_type = Attributes::new();
        bad_type.insert("rank".into(), "not_an_int".into());
        assert!(matches!(
            col.upsert(1, &[0.0; 8], bad_type),
            Err(TreeError::InvalidAttribute { .. })
        ));

        // Non-filterable attribute rejected in Filter, allowed in upsert/get
        let mut attrs = Attributes::new();
        attrs.insert("note".into(), "secret_payload".into());
        col.upsert(1, &[1.0; 8], attrs).unwrap();
        assert_eq!(
            col.get(1).unwrap().unwrap().attributes.get("note"),
            Some(&AttributeValue::String("secret_payload".into()))
        );
        assert!(matches!(
            col.search(
                &[1.0; 8],
                SearchOptions::new(5, 4),
                Some(&Filter::Eq("note".into(), "secret_payload".into()))
            ),
            Err(TreeError::InvalidFilter(_))
        ));

        col.verify_invariants().unwrap();
    }

    // Reopen with mismatched attribute type fails
    let mut bad_cfg = cfg.clone();
    bad_cfg.attributes[0].attr_type = AttributeType::Int;
    assert!(matches!(
        Collection::open(dir.path(), bad_cfg),
        Err(TreeError::ConfigMismatch {
            field: "attribute_type",
            ..
        })
    ));

    // Reopen with mismatched filterable flag fails
    let mut bad_filt_cfg = cfg.clone();
    bad_filt_cfg.attributes[0].filterable = false;
    assert!(matches!(
        Collection::open(dir.path(), bad_filt_cfg),
        Err(TreeError::ConfigMismatch {
            field: "attribute_filterable",
            ..
        })
    ));

    // Reopen with an additional new attribute succeeds and persists new schema
    let extended_cfg = cfg.with_attribute("extra", AttributeType::Bool, true);
    let col = Collection::open(dir.path(), extended_cfg).unwrap();
    assert!(col.schema().get_by_name("extra").is_some());
    // Old doc has extra=null
    let hits = col
        .search(
            &[1.0; 8],
            SearchOptions::new(5, 4),
            Some(&Filter::IsNull("extra".into())),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 1);
    col.verify_invariants().unwrap();
}

#[test]
fn test_upsert_delete_reopen_and_get() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(8, DistanceMetric::L2Squared, true);

    {
        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();
        let mut attrs1 = Attributes::new();
        attrs1.insert("category".into(), "books".into());
        attrs1.insert("rank".into(), 10i64.into());
        attrs1.insert("note".into(), "v1".into());
        col.upsert(42, &[1.0; 8], attrs1.clone()).unwrap();

        let doc = col.get(42).unwrap().unwrap();
        assert_eq!(doc.id, 42);
        assert_eq!(doc.embedding, vec![1.0; 8]);
        assert_eq!(doc.attributes, attrs1);

        // Upsert same id with updated embedding and attributes
        let mut attrs2 = Attributes::new();
        attrs2.insert("category".into(), "games".into());
        attrs2.insert("rank".into(), 20i64.into());
        col.upsert(42, &[2.0; 8], attrs2.clone()).unwrap();

        let doc2 = col.get(42).unwrap().unwrap();
        assert_eq!(doc2.embedding, vec![2.0; 8]);
        assert_eq!(doc2.attributes, attrs2);

        // Old attribute value must no longer match
        let old_hits = col
            .search(
                &[1.0; 8],
                SearchOptions::new(5, 4),
                Some(&Filter::Eq("category".into(), "books".into())),
            )
            .unwrap();
        assert!(old_hits.is_empty());

        let new_hits = col
            .search(
                &[2.0; 8],
                SearchOptions::new(5, 4),
                Some(&Filter::Eq("category".into(), "games".into())),
            )
            .unwrap();
        assert_eq!(new_hits.len(), 1);
        assert_eq!(new_hits[0].id, 42);
        col.verify_invariants().unwrap();
    }

    // Reopen, delete, verify gone
    {
        let mut col = Collection::open(dir.path(), cfg).unwrap();
        col.verify_invariants().unwrap();
        assert!(col.get(42).unwrap().is_some());
        col.delete(42).unwrap();
        assert!(col.get(42).unwrap().is_none());
        col.delete(999).unwrap(); // non-existent delete is no-op
        assert!(
            col.search(
                &[2.0; 8],
                SearchOptions::new(5, 4),
                Some(&Filter::Eq("category".into(), "games".into())),
            )
            .unwrap()
            .is_empty()
        );
        col.verify_invariants().unwrap();
    }
}

#[test]
fn test_heavy_churn_splits_merges_lire_and_invariants() {
    for enable_lire in [false, true] {
        let dir = TempDir::new().unwrap();
        let cfg = small_collection_config(12, DistanceMetric::Cosine, enable_lire);
        let mut rng = SmallRng::seed_from_u64(4242);
        let mut oracle: HashMap<DocId, (Vec<f32>, Attributes)> = HashMap::new();

        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();
        let categories = ["alpha", "beta", "gamma", "delta"];

        for step in 0..500usize {
            let op = rng.random_range(0..100);
            if op < 60 {
                let id = rng.random_range(0..160u64);
                let vec = random_vec(&mut rng, 12);
                let mut attrs = Attributes::new();
                attrs.insert(
                    "category".into(),
                    categories[rng.random_range(0..categories.len())].into(),
                );
                attrs.insert("rank".into(), rng.random_range(-50i64..50).into());
                attrs.insert("score".into(), rng.random_range(-10.0f64..10.0).into());
                if rng.random_bool(0.7) {
                    attrs.insert("active".into(), rng.random_bool(0.5).into());
                }
                if rng.random_bool(0.5) {
                    attrs.insert(
                        "tags".into(),
                        vec![
                            format!("t{}", rng.random_range(0..5)),
                            format!("t{}", rng.random_range(0..5)),
                        ]
                        .into(),
                    );
                }
                col.upsert(id, &vec, attrs.clone()).unwrap();
                oracle.insert(id, (vec, attrs));
            } else if op < 90 {
                let id = rng.random_range(0..160u64);
                col.delete(id).unwrap();
                oracle.remove(&id);
            } else {
                drop(col);
                col = Collection::open(dir.path(), cfg.clone()).unwrap();
            }

            if step % 25 == 24 {
                let stats = col.verify_invariants().unwrap();
                assert_eq!(stats.tree.num_vectors, oracle.len() as u64);
            }
        }

        let stats = col.verify_invariants().unwrap();
        assert_eq!(stats.tree.num_vectors, oracle.len() as u64);

        // Verify filtered queries match brute force with high recall under churned tree
        let filters = vec![
            Filter::Eq("category".into(), "alpha".into()),
            Filter::And(vec![
                Filter::Eq("active".into(), true.into()),
                Filter::Gt("rank".into(), 0i64.into()),
            ]),
            Filter::Not(Box::new(Filter::Eq("category".into(), "beta".into()))),
            Filter::IsNull("active".into()),
            Filter::Contains("tags".into(), "t1".into()),
        ];

        for filter in &filters {
            let q = random_vec(&mut rng, 12);
            let opts = SearchOptions::new(10, 16)
                .with_rerank(40)
                .with_max_expanded_leaves(128);
            let got = col.search(&q, opts, Some(filter)).unwrap();
            let exact = brute_force_filtered(&oracle, &q, 10, DistanceMetric::Cosine, Some(filter));

            // Every returned result must actually satisfy the filter
            for hit in &got {
                let doc = col.get(hit.id).unwrap().unwrap();
                assert!(
                    filter.matches(&doc.attributes),
                    "returned doc {} failed filter {filter:?}",
                    hit.id
                );
            }

            if !exact.is_empty() {
                let exact_ids: HashSet<DocId> = exact.iter().map(|&(id, _)| id).collect();
                let hits = got.iter().filter(|r| exact_ids.contains(&r.id)).count();
                assert!(
                    hits * 2 >= exact_ids.len(),
                    "low recall for {filter:?}: {hits}/{}",
                    exact_ids.len()
                );
            }
        }
    }
}

#[test]
fn test_false_positive_cluster_pruning_and_dynamic_expansion() {
    // Cluster 1 (near origin): docs 0..4 have x=1, y=0; docs 4..8 have x=0, y=2.
    // Neither doc has BOTH x=1 AND y=2, so cluster-level bitmap intersection includes Cluster 1
    // as a false positive!
    // Cluster 2 (farther away, at [5.0; 8]): docs 100..106 have x=1 AND y=2.
    // Query near origin with `x=1 AND y=2` must reject all docs in Cluster 1 and dynamically
    // expand to Cluster 2 to return docs from Cluster 2.
    let dir = TempDir::new().unwrap();
    let tree = TreeConfig {
        dim: 8,
        metric: DistanceMetric::L2Squared,
        min_internal_fanout: 2,
        max_internal_fanout: 5,
        min_leaf_size: 2,
        max_leaf_size: 8,
        insert_beam_width: 4,
        lire_neighbor_count: 4,
        enable_lire: false,
        centroid_drift_eps: 1e-4,
    };
    let cfg = CollectionConfig::new(tree)
        .with_attribute("x", AttributeType::Int, true)
        .with_attribute("y", AttributeType::Int, true);

    let mut col = Collection::open(dir.path(), cfg).unwrap();
    let mut rng = SmallRng::seed_from_u64(77);

    for id in 0..4u64 {
        let v: Vec<f32> = (0..8).map(|_| rng.random_range(-0.05..0.05)).collect();
        let mut attrs = Attributes::new();
        attrs.insert("x".into(), 1i64.into());
        attrs.insert("y".into(), 0i64.into());
        col.upsert(id, &v, attrs).unwrap();
    }
    for id in 4..8u64 {
        let v: Vec<f32> = (0..8).map(|_| rng.random_range(-0.05..0.05)).collect();
        let mut attrs = Attributes::new();
        attrs.insert("x".into(), 0i64.into());
        attrs.insert("y".into(), 2i64.into());
        col.upsert(id, &v, attrs).unwrap();
    }
    for id in 100..106u64 {
        let v: Vec<f32> = (0..8).map(|_| 5.0 + rng.random_range(-0.05..0.05)).collect();
        let mut attrs = Attributes::new();
        attrs.insert("x".into(), 1i64.into());
        attrs.insert("y".into(), 2i64.into());
        col.upsert(id, &v, attrs).unwrap();
    }

    let stats = col.verify_invariants().unwrap();
    assert!(stats.tree.leaf_nodes >= 2);

    let filter = Filter::And(vec![
        Filter::Eq("x".into(), 1i64.into()),
        Filter::Eq("y".into(), 2i64.into()),
    ]);
    // Disable exhaustive crossover (`exhaustive_cluster_threshold = 0`) and set beam_width = 1
    // to force best-first traversal + dynamic expansion past the false-positive cluster!
    let opts = SearchOptions::new(4, 1)
        .with_rerank(10)
        .with_exhaustive_cluster_threshold(0);
    let hits = col.search(&[0.0; 8], opts, Some(&filter)).unwrap();
    assert_eq!(hits.len(), 4);
    for hit in &hits {
        assert!(
            (100..106).contains(&hit.id),
            "expected hit from cluster 2, got {}",
            hit.id
        );
    }

    // Now test exhaustive crossover too (default threshold = 8):
    let exact_hits = col
        .search(&[0.0; 8], SearchOptions::new(4, 1), Some(&filter))
        .unwrap();
    assert_eq!(exact_hits.len(), 4);
    for hit in &exact_hits {
        assert!((100..106).contains(&hit.id));
    }
}

#[test]
fn test_all_filter_operators_and_selectivities_recall() {
    let dir = TempDir::new().unwrap();
    let dim = 16;
    let mut tree = TreeConfig::new(dim, DistanceMetric::L2Squared);
    tree.min_leaf_size = 8;
    tree.max_leaf_size = 32;
    tree.min_internal_fanout = 4;
    tree.max_internal_fanout = 16;
    tree.enable_lire = true;

    let cfg = CollectionConfig::new(tree)
        .with_attribute("bucket_1000", AttributeType::Uint, true) // 0.1% per value
        .with_attribute("bucket_100", AttributeType::Uint, true) // 1% per value
        .with_attribute("bucket_10", AttributeType::Uint, true) // 10% per value
        .with_attribute("half", AttributeType::Bool, true) // 50%
        .with_attribute("score", AttributeType::Float, true) // continuous [0, 100)
        .with_attribute("label", AttributeType::String, true)
        .with_attribute("tags", AttributeType::StringArray, true)
        .with_attribute("int_tags", AttributeType::IntArray, true)
        .with_attribute("optional_rank", AttributeType::Int, true);

    let mut col = Collection::open(dir.path(), cfg).unwrap();
    let mut rng = SmallRng::seed_from_u64(20261121);
    let n = 2_000usize;
    let centers: Vec<Vec<f32>> = (0..25).map(|_| random_vec(&mut rng, dim)).collect();
    let mut oracle: HashMap<DocId, (Vec<f32>, Attributes)> = HashMap::new();

    for id in 0..n as u64 {
        let c = &centers[(id as usize) % centers.len()];
        let vec: Vec<f32> = c
            .iter()
            .map(|&x| x + rng.random_range(-0.2f32..0.2f32))
            .collect();
        let mut attrs = Attributes::new();
        attrs.insert("bucket_1000".into(), (id % 1000).into());
        attrs.insert("bucket_100".into(), (id % 100).into());
        attrs.insert("bucket_10".into(), (id % 10).into());
        attrs.insert("half".into(), (id % 2 == 0).into());
        attrs.insert("score".into(), ((id % 100) as f64 + 0.5).into());
        attrs.insert("label".into(), format!("lbl_{:02}", id % 20).into());
        attrs.insert(
            "tags".into(),
            vec![format!("g{}", id % 5), format!("h{}", id % 7)].into(),
        );
        attrs.insert("int_tags".into(), vec![(id % 4) as i64, 99].into());
        if id % 10 != 0 {
            attrs.insert("optional_rank".into(), ((id as i64) - 1000).into());
        }
        col.upsert(id, &vec, attrs.clone()).unwrap();
        oracle.insert(id, (vec, attrs));
    }

    let stats = col.verify_invariants().unwrap();
    assert_eq!(stats.tree.num_vectors, n as u64);
    assert!(stats.tree.leaf_nodes > 20);

    let test_cases: Vec<(&str, Filter, usize, f64)> = vec![
        // ~0.1% selectivity (2 docs in 2000 -> exhaustive crossover or tiny subset)
        (
            "0.1% Eq",
            Filter::Eq("bucket_1000".into(), 7u64.into()),
            2,
            1.0,
        ),
        // ~1% selectivity (20 docs)
        (
            "1% Eq",
            Filter::Eq("bucket_100".into(), 13u64.into()),
            10,
            0.90,
        ),
        // ~10% selectivity (200 docs)
        (
            "10% Eq",
            Filter::Eq("bucket_10".into(), 3u64.into()),
            10,
            0.90,
        ),
        // ~50% selectivity (1000 docs)
        ("50% Bool", Filter::Eq("half".into(), true.into()), 10, 0.90),
        // ~90% selectivity (1800 docs)
        (
            "90% NotEq",
            Filter::NotEq("bucket_10".into(), 3u64.into()),
            10,
            0.90,
        ),
        // In / NotIn
        (
            "In 30%",
            Filter::In(
                "bucket_10".into(),
                vec![1u64.into(), 2u64.into(), 3u64.into()],
            ),
            10,
            0.90,
        ),
        (
            "NotIn 80%",
            Filter::NotIn("bucket_10".into(), vec![0u64.into(), 9u64.into()]),
            10,
            0.90,
        ),
        // Range filters: Lt / Lte / Gt / Gte on Float, Int, String
        (
            "Float Lt 15%",
            Filter::Lt("score".into(), 15.0f64.into()),
            10,
            0.90,
        ),
        (
            "Float Gte 85%",
            Filter::Gte("score".into(), 15.5f64.into()),
            10,
            0.90,
        ),
        (
            "Int range And",
            Filter::And(vec![
                Filter::Gt("optional_rank".into(), (-500i64).into()),
                Filter::Lte("optional_rank".into(), 200i64.into()),
            ]),
            10,
            0.90,
        ),
        (
            "String Lt",
            Filter::Lt("label".into(), "lbl_05".into()),
            10,
            0.90,
        ),
        // Array Contains / ContainsAny
        (
            "Contains StringArray",
            Filter::Contains("tags".into(), "g2".into()),
            10,
            0.90,
        ),
        (
            "ContainsAny StringArray",
            Filter::ContainsAny("tags".into(), vec!["g1".into(), "h3".into()]),
            10,
            0.90,
        ),
        (
            "Contains IntArray",
            Filter::Contains("int_tags".into(), 2i64.into()),
            10,
            0.90,
        ),
        // Null / IsNull / IsNotNull
        ("IsNull 10%", Filter::IsNull("optional_rank".into()), 10, 0.90),
        (
            "IsNotNull 90%",
            Filter::IsNotNull("optional_rank".into()),
            10,
            0.90,
        ),
        // Compound Or + Not
        (
            "Or + Not",
            Filter::Or(vec![
                Filter::Eq("bucket_100".into(), 5u64.into()),
                Filter::And(vec![
                    Filter::Eq("half".into(), false.into()),
                    Filter::Not(Box::new(Filter::Eq("bucket_10".into(), 1u64.into()))),
                ]),
            ]),
            10,
            0.90,
        ),
    ];

    let queries: Vec<Vec<f32>> = (0..10)
        .map(|i| {
            let c = &centers[i % centers.len()];
            c.iter()
                .map(|&x| x + rng.random_range(-0.2f32..0.2f32))
                .collect()
        })
        .collect();

    for (name, filter, k, min_recall) in test_cases {
        let mut total_hits = 0usize;
        let mut total_expected = 0usize;
        let opts = SearchOptions::new(k, 8)
            .with_rerank(64)
            .with_max_expanded_leaves(256);

        for q in &queries {
            let got = col.search(q, opts, Some(&filter)).unwrap();
            let exact = brute_force_filtered(&oracle, q, k, DistanceMetric::L2Squared, Some(&filter));
            assert_eq!(
                got.len(),
                exact.len(),
                "{name}: result count mismatch {} vs {}",
                got.len(),
                exact.len()
            );
            for hit in &got {
                let doc = col.get(hit.id).unwrap().unwrap();
                assert!(
                    filter.matches(&doc.attributes),
                    "{name}: returned doc {} did not satisfy filter",
                    hit.id
                );
            }
            let exact_ids: HashSet<DocId> = exact.iter().map(|&(id, _)| id).collect();
            total_hits += got.iter().filter(|r| exact_ids.contains(&r.id)).count();
            total_expected += exact_ids.len();
        }
        let recall = total_hits as f64 / total_expected as f64;
        assert!(
            recall >= min_recall,
            "{name}: recall {recall:.3} < {min_recall:.3} ({total_hits}/{total_expected})"
        );
    }

    // Empty result filter returns empty immediately
    let empty_res = col
        .search(
            &queries[0],
            SearchOptions::new(10, 8),
            Some(&Filter::Eq("bucket_1000".into(), 999_999u64.into())),
        )
        .unwrap();
    assert!(empty_res.is_empty());
}

#[test]
fn test_patch_and_patch_batch_semantics_and_atomicity() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(4, DistanceMetric::L2Squared, true);

    // Seed two well-separated clusters so we can test same-leaf vs cross-leaf vector patches.
    let loc1_before;
    let loc2_before;
    {
        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();
        for id in 1..=4u64 {
            let mut attrs = Attributes::new();
            attrs.insert("category".into(), "near".into());
            attrs.insert("rank".into(), (id as i64).into());
            attrs.insert("tags".into(), vec!["t1".to_string(), "t2".to_string()].into());
            attrs.insert("note".into(), format!("note_{id}").into());
            col.upsert(id, &[0.0 + (id as f32) * 0.01, 0.0, 0.0, 0.0], attrs)
                .unwrap();
        }
        for id in 101..=104u64 {
            let mut attrs = Attributes::new();
            attrs.insert("category".into(), "far".into());
            attrs.insert("rank".into(), (id as i64).into());
            col.upsert(id, &[50.0 + ((id - 100) as f32) * 0.01, 0.0, 0.0, 0.0], attrs)
                .unwrap();
        }
        col.verify_invariants().unwrap();
    }
    {
        let (storage, _) = izzydb::storage::Storage::open(dir.path(), &cfg.tree).unwrap();
        loc1_before = storage.get_doc_loc(1).unwrap().unwrap();
        loc2_before = storage.get_doc_loc(2).unwrap().unwrap();
    }

    {
        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();

        // 1. Attribute-only patch preserves vector & DocLoc, clears fields with Null, keeps omitted fields.
        col.patch(
            1,
            DocumentPatch::new()
                .with_attribute("category", "updated".into())
                .clear_attribute("tags")
                .clear_attribute("note"),
        )
        .unwrap();
        let d1 = col.get(1).unwrap().unwrap();
        assert_eq!(d1.embedding, vec![0.01, 0.0, 0.0, 0.0]);
        assert_eq!(
            d1.attributes.get("category"),
            Some(&AttributeValue::String("updated".into()))
        );
        assert_eq!(d1.attributes.get("rank"), Some(&AttributeValue::Int(1)));
        assert_eq!(d1.attributes.get("tags"), None);
        assert_eq!(d1.attributes.get("note"), None);

        // Old tag/category filters no longer match doc 1; IsNull("tags") now matches doc 1.
        assert!(col.resolve_filter_doc_ids(&Filter::Eq("category".into(), "near".into())).unwrap().len() == 3);
        assert_eq!(
            col.resolve_filter_doc_ids(&Filter::Contains("tags".into(), "t1".into()))
                .unwrap(),
            vec![2, 3, 4]
        );
        assert!(
            col.resolve_filter_doc_ids(&Filter::IsNull("tags".into()))
                .unwrap()
                .contains(&1)
        );

        // 2. Vector-only same-leaf patch preserves DocLoc.
        col.patch(2, DocumentPatch::new().with_embedding(vec![0.025, 0.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(
            col.get(2).unwrap().unwrap().embedding,
            vec![0.025, 0.0, 0.0, 0.0]
        );

        // 3. Vector-only cross-leaf patch moves doc 3 to far cluster and preserves attributes.
        col.patch(3, DocumentPatch::new().with_embedding(vec![50.05, 0.0, 0.0, 0.0]))
            .unwrap();
        let d3 = col.get(3).unwrap().unwrap();
        assert_eq!(d3.embedding, vec![50.05, 0.0, 0.0, 0.0]);
        assert_eq!(
            d3.attributes.get("category"),
            Some(&AttributeValue::String("near".into()))
        );

        // 4. Patching missing DocId returns DocumentNotFound.
        assert!(matches!(
            col.patch(999, DocumentPatch::new().with_attribute("rank", 1i64.into())),
            Err(TreeError::DocumentNotFound(999))
        ));

        // 5. patch_batch atomicity: if one doc is missing or invalid, NO writes occur.
        let err = col
            .patch_batch(&[
                (1, DocumentPatch::new().with_attribute("rank", 777i64.into())),
                (999, DocumentPatch::new().with_attribute("rank", 888i64.into())),
            ])
            .unwrap_err();
        assert!(matches!(err, TreeError::DocumentNotFound(999)));
        assert_eq!(
            col.get(1).unwrap().unwrap().attributes.get("rank"),
            Some(&AttributeValue::Int(1))
        );

        let err_schema = col
            .patch_batch(&[
                (1, DocumentPatch::new().with_attribute("rank", 777i64.into())),
                (2, DocumentPatch::new().with_attribute("rank", "wrong_type".into())),
            ])
            .unwrap_err();
        assert!(matches!(err_schema, TreeError::InvalidAttribute { .. }));
        assert_eq!(
            col.get(1).unwrap().unwrap().attributes.get("rank"),
            Some(&AttributeValue::Int(1))
        );

        // 6. Valid patch_batch commits all atomically.
        col.patch_batch(&[
            (1, DocumentPatch::new().with_attribute("rank", 111i64.into())),
            (2, DocumentPatch::new().with_attribute("rank", 222i64.into())),
        ])
        .unwrap();
        assert_eq!(
            col.get(1).unwrap().unwrap().attributes.get("rank"),
            Some(&AttributeValue::Int(111))
        );
        assert_eq!(
            col.get(2).unwrap().unwrap().attributes.get("rank"),
            Some(&AttributeValue::Int(222))
        );

        col.verify_invariants().unwrap();
    }

    {
        let (storage, _) = izzydb::storage::Storage::open(dir.path(), &cfg.tree).unwrap();
        let loc1_after = storage.get_doc_loc(1).unwrap().unwrap();
        let loc2_after = storage.get_doc_loc(2).unwrap().unwrap();
        let loc3_after = storage.get_doc_loc(3).unwrap().unwrap();
        let loc101 = storage.get_doc_loc(101).unwrap().unwrap();
        assert_eq!(loc1_after, loc1_before);
        assert_eq!(loc2_after, loc2_before);
        assert_eq!(loc3_after.leaf_id, loc101.leaf_id);
    }
}

#[test]
fn test_delete_batch_delete_by_filter_and_update_by_filter() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(4, DistanceMetric::L2Squared, true);
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    // Insert 600 docs (> MUTATION_CHUNK_SIZE=512) to exercise multi-chunk filter ops & multi-level deletes.
    for id in 0..600u64 {
        let mut attrs = Attributes::new();
        attrs.insert(
            "category".into(),
            match id % 3 {
                0 => "a",
                1 => "b",
                _ => "c",
            }
            .into(),
        );
        attrs.insert("rank".into(), (id as i64).into());
        attrs.insert("active".into(), (id % 2 == 0).into());
        if id % 5 == 0 {
            attrs.insert("tags".into(), vec!["special".to_string()].into());
        }
        col.upsert(id, &[((id % 20) as f32) * 0.5, (id as f32) * 0.01, 0.0, 0.0], attrs)
            .unwrap();
    }
    col.verify_invariants().unwrap();

    // delete returns false for missing, true for existing
    assert!(!col.delete(999_999).unwrap());
    assert!(col.delete(599).unwrap());
    assert!(!col.delete(599).unwrap());

    // delete_batch deduplicates and ignores missing IDs
    let deleted = col.delete_batch(&[595, 596, 595, 999_998, 597]).unwrap();
    assert_eq!(deleted, 3);
    col.verify_invariants().unwrap();

    // 0-match filter is a no-op
    let zero_upd = col
        .update_by_filter(
            &Filter::Eq("category".into(), "does_not_exist".into()),
            &DocumentPatch::new().with_attribute("active", false.into()),
        )
        .unwrap();
    assert_eq!(zero_upd, 0);
    let zero_del = col
        .delete_by_filter(&Filter::Eq("category".into(), "does_not_exist".into()))
        .unwrap();
    assert_eq!(zero_del, 0);

    // update_by_filter on range + Not
    let upd_filter = Filter::And(vec![
        Filter::Lt("rank".into(), 100i64.into()),
        Filter::Not(Box::new(Filter::Eq("category".into(), "c".into()))),
    ]);
    let expected_upd = col.resolve_filter_doc_ids(&upd_filter).unwrap();
    let updated = col
        .update_by_filter(
            &upd_filter,
            &DocumentPatch::new().with_attribute("category", "promoted".into()),
        )
        .unwrap();
    assert_eq!(updated as usize, expected_upd.len());
    for id in expected_upd {
        assert_eq!(
            col.get(id).unwrap().unwrap().attributes.get("category"),
            Some(&AttributeValue::String("promoted".into()))
        );
    }
    col.verify_invariants().unwrap();

    // delete_by_filter on IsNull
    let null_tags_filter = Filter::IsNull("tags".into());
    let expected_nulls = col.resolve_filter_doc_ids(&null_tags_filter).unwrap();
    let del_nulls = col.delete_by_filter(&null_tags_filter).unwrap();
    assert_eq!(del_nulls as usize, expected_nulls.len());
    col.verify_invariants().unwrap();

    // Drain everything with empty And(vec![]) and verify clean empty root leaf, then re-insert.
    let remaining = col.len();
    let drained = col.delete_by_filter(&Filter::And(vec![])).unwrap();
    assert_eq!(drained, remaining);
    let stats = col.verify_invariants().unwrap();
    assert_eq!(stats.tree.num_vectors, 0);
    assert_eq!(stats.tree.root_level, 0);
    assert_eq!(stats.tree.total_nodes, 1);

    let mut fresh = Attributes::new();
    fresh.insert("category".into(), "reborn".into());
    col.upsert(1, &[1.0, 2.0, 3.0, 4.0], fresh).unwrap();
    assert_eq!(col.verify_invariants().unwrap().num_docs, 1);
}

#[test]
fn test_long_horizon_mutation_churn_with_patch_and_filter_ops() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(10, DistanceMetric::Cosine, true);
    let mut rng = SmallRng::seed_from_u64(2026_1121);
    let mut oracle: HashMap<DocId, (Vec<f32>, Attributes)> = HashMap::new();
    let categories = ["alpha", "beta", "gamma", "delta"];

    let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();

    for step in 0..400usize {
        let op = rng.random_range(0..100);
        if op < 35 {
            let id = rng.random_range(0..180u64);
            let vec = random_vec(&mut rng, 10);
            let mut attrs = Attributes::new();
            attrs.insert(
                "category".into(),
                categories[rng.random_range(0..categories.len())].into(),
            );
            attrs.insert("rank".into(), rng.random_range(-50i64..50).into());
            attrs.insert("active".into(), rng.random_bool(0.5).into());
            if rng.random_bool(0.5) {
                attrs.insert(
                    "tags".into(),
                    vec![format!("t{}", rng.random_range(0..4))].into(),
                );
            }
            col.upsert(id, &vec, attrs.clone()).unwrap();
            oracle.insert(id, (vec, attrs));
        } else if op < 55 {
            // Single patch (attribute-only, vector-only, or both)
            if let Some(&id) = oracle.keys().nth(rng.random_range(0..oracle.len().max(1))) {
                let mut patch = DocumentPatch::new();
                if rng.random_bool(0.5) {
                    let nv = random_vec(&mut rng, 10);
                    patch.embedding = Some(nv.clone());
                    oracle.get_mut(&id).unwrap().0 = nv;
                }
                if rng.random_bool(0.7) {
                    let cat = categories[rng.random_range(0..categories.len())];
                    patch = patch.with_attribute("category", cat.into());
                    oracle
                        .get_mut(&id)
                        .unwrap()
                        .1
                        .insert("category".into(), cat.into());
                } else {
                    patch = patch.clear_attribute("tags");
                    oracle.get_mut(&id).unwrap().1.remove("tags");
                }
                col.patch(id, patch).unwrap();
            }
        } else if op < 70 && oracle.len() >= 3 {
            // patch_batch
            let ids: Vec<DocId> = oracle.keys().copied().take(3).collect();
            let mut patches = Vec::new();
            for id in ids {
                let new_rank = rng.random_range(-50i64..50);
                oracle
                    .get_mut(&id)
                    .unwrap()
                    .1
                    .insert("rank".into(), new_rank.into());
                patches.push((id, DocumentPatch::new().with_attribute("rank", new_rank.into())));
            }
            col.patch_batch(&patches).unwrap();
        } else if op < 80 {
            // delete_batch
            let ids: Vec<DocId> = (0..4).map(|_| rng.random_range(0..180u64)).collect();
            let expected_del = ids
                .iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .filter(|id| oracle.remove(id).is_some())
                .count() as u64;
            assert_eq!(col.delete_batch(&ids).unwrap(), expected_del);
        } else if op < 87 {
            // update_by_filter
            let f = Filter::Eq("category".into(), "delta".into());
            let mut count = 0u64;
            for (_, attrs) in oracle.values_mut() {
                if f.matches(attrs) {
                    attrs.insert("active".into(), false.into());
                    count += 1;
                }
            }
            let got = col
                .update_by_filter(&f, &DocumentPatch::new().with_attribute("active", false.into()))
                .unwrap();
            assert_eq!(got, count);
        } else if op < 94 {
            // delete_by_filter
            let f = Filter::Lt("rank".into(), (-40i64).into());
            let before = oracle.len();
            oracle.retain(|_, (_, attrs)| !f.matches(attrs));
            let expected = (before - oracle.len()) as u64;
            assert_eq!(col.delete_by_filter(&f).unwrap(), expected);
        } else {
            drop(col);
            col = Collection::open(dir.path(), cfg.clone()).unwrap();
        }

        if step % 40 == 39 {
            let stats = col.verify_invariants().unwrap();
            assert_eq!(stats.num_docs, oracle.len());
        }
    }

    let stats = col.verify_invariants().unwrap();
    assert_eq!(stats.num_docs, oracle.len());

    let q = random_vec(&mut rng, 10);
    let filter = Filter::And(vec![
        Filter::Eq("active".into(), true.into()),
        Filter::Not(Box::new(Filter::Eq("category".into(), "beta".into()))),
    ]);
    let got = col
        .search(
            &q,
            SearchOptions::new(10, 16)
                .with_rerank(40)
                .with_max_expanded_leaves(128),
            Some(&filter),
        )
        .unwrap();
    let exact = brute_force_filtered(&oracle, &q, 10, DistanceMetric::Cosine, Some(&filter));
    if !exact.is_empty() {
        let exact_ids: HashSet<DocId> = exact.iter().map(|&(id, _)| id).collect();
        let hits = got.iter().filter(|r| exact_ids.contains(&r.id)).count();
        assert!(hits * 2 >= exact_ids.len());
    }
}
