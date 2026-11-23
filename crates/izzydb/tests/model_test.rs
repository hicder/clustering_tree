//! Randomized state-machine model test verifying v3 `Collection` against an in-memory oracle.

use std::collections::{BTreeMap, HashMap};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

use izzydb::metric::l2_squared;
use izzydb::{
    AttributeType, AttributeValue, Attributes, Collection, CollectionConfig,
    DetachedDuplicatePolicy, DetachedWriteMode, DistanceMetric, DocId, Document, DocumentPatch,
    FacetSpec, Filter, FullTextSearchConfig, MaintenanceOptions, OrderBySpec, RankBy,
    ScalarValue, SearchOptions, TreeConfig,
};

fn build_model_config(with_vectors: bool, seal_rows: u32) -> CollectionConfig {
    let mut cfg = if with_vectors {
        CollectionConfig::new(TreeConfig {
            dim: 6,
            metric: DistanceMetric::L2Squared,
            min_internal_fanout: 2,
            max_internal_fanout: 5,
            min_leaf_size: 4,
            max_leaf_size: 10,
            insert_beam_width: 4,
            lire_neighbor_count: 4,
            enable_lire: true,
            centroid_drift_eps: 1e-4,
        })
    } else {
        CollectionConfig::without_vectors()
    };
    cfg.segment.seal_rows = seal_rows;
    cfg = cfg
        .with_attribute("cat", AttributeType::String, true)
        .with_attribute("score", AttributeType::Float, true)
        .with_attribute("flag", AttributeType::Bool, true)
        .with_attribute("tags", AttributeType::StringArray, true)
        .with_attribute("note", AttributeType::String, false)
        .with_fts_attribute("body", AttributeType::String, false, FullTextSearchConfig::default())
        .with_detached_attribute("pop", AttributeType::Int, true)
        .with_detached_attribute("badge", AttributeType::String, true);
    cfg
}

fn rand_vec(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.random_range(-2.0f32..2.0f32)).collect()
}

fn rand_attrs(rng: &mut SmallRng, include_detached: bool) -> Attributes {
    let cats = ["alpha", "beta", "gamma", "delta"];
    let vocab = ["rust", "vector", "segment", "index", "search", "column", "compaction"];
    let tags = ["t1", "t2", "t3", "t4"];
    let badges = ["gold", "silver", "bronze"];

    let mut m = Attributes::new();
    if rng.random_bool(0.8) {
        m.insert("cat".into(), cats[rng.random_range(0..cats.len())].into());
    }
    if rng.random_bool(0.8) {
        let s = (rng.random_range(-100..200) as f64) / 10.0;
        m.insert("score".into(), s.into());
    }
    if rng.random_bool(0.75) {
        m.insert("flag".into(), rng.random_bool(0.5).into());
    }
    if rng.random_bool(0.7) {
        let k = rng.random_range(0..=3);
        let ts: Vec<String> = (0..k).map(|_| tags[rng.random_range(0..tags.len())].into()).collect();
        m.insert("tags".into(), ts.into());
    }
    if rng.random_bool(0.6) {
        m.insert("note".into(), format!("n-{}", rng.random_range(0..50)).into());
    }
    if rng.random_bool(0.85) {
        let k = rng.random_range(1..=4);
        let words: Vec<&str> = (0..k).map(|_| vocab[rng.random_range(0..vocab.len())]).collect();
        m.insert("body".into(), words.join(" ").into());
    }
    if include_detached {
        if rng.random_bool(0.5) {
            m.insert("pop".into(), rng.random_range(-20i64..100).into());
        }
        if rng.random_bool(0.5) {
            m.insert("badge".into(), badges[rng.random_range(0..badges.len())].into());
        }
    }
    m
}

fn apply_patch_to_oracle(doc: &mut Document, patch: &DocumentPatch) {
    if let Some(ref v) = patch.embedding {
        doc.embedding = Some(v.clone());
    }
    for (k, v) in &patch.attributes {
        if matches!(v, AttributeValue::Null) {
            doc.attributes.remove(k);
        } else {
            doc.attributes.insert(k.clone(), v.clone());
        }
    }
}

fn eval_filter_oracle(doc: &Document, f: &Filter) -> bool {
    match f {
        Filter::Eq(attr, target) => match (doc.attributes.get(attr), target) {
            (Some(AttributeValue::String(s)), ScalarValue::String(t)) => s == t,
            (Some(AttributeValue::Bool(b)), ScalarValue::Bool(t)) => b == t,
            (Some(AttributeValue::Int(v)), ScalarValue::Int(t)) => v == t,
            _ => false,
        },
        Filter::Gte(attr, target) => match (doc.attributes.get(attr), target) {
            (Some(AttributeValue::Float(v)), ScalarValue::Float(t)) => v >= t,
            (Some(AttributeValue::Int(v)), ScalarValue::Int(t)) => v >= t,
            _ => false,
        },
        Filter::Contains(attr, target) => match (doc.attributes.get(attr), target) {
            (Some(AttributeValue::StringArray(arr)), ScalarValue::String(t)) => {
                arr.iter().any(|x| x == t)
            }
            _ => false,
        },
        Filter::IsNull(attr) => !doc.attributes.contains_key(attr),
        Filter::And(subs) => subs.iter().all(|s| eval_filter_oracle(doc, s)),
        other => panic!("unsupported filter in oracle: {other:?}"),
    }
}

fn check_queries_parity(
    col: &Collection,
    oracle: &BTreeMap<DocId, Document>,
    rng: &mut SmallRng,
    with_vectors: bool,
) {
    // 1. Point gets (sample + missing ID).
    for id in [1u64, 5, 15, 29, 99999] {
        let actual = col.get(id).unwrap();
        let expected = oracle.get(&id).cloned();
        assert_eq!(actual, expected, "get({id}) mismatch");
    }

    // 2. Filter parity (inline + detached + combined).
    let filters = vec![
        Filter::Eq("cat".into(), "alpha".into()),
        Filter::Eq("flag".into(), true.into()),
        Filter::Gte("score".into(), 2.5f64.into()),
        Filter::Contains("tags".into(), "t2".into()),
        Filter::IsNull("cat".into()),
        Filter::Eq("badge".into(), "gold".into()),
        Filter::Gte("pop".into(), 30i64.into()),
        Filter::And(vec![
            Filter::Eq("flag".into(), true.into()),
            Filter::Gte("pop".into(), 10i64.into()),
        ]),
    ];
    for f in &filters {
        let expected_ids: Vec<DocId> = oracle
            .iter()
            .filter(|(_, d)| eval_filter_oracle(d, f))
            .map(|(&id, _)| id)
            .collect();
        let actual_ids = col.resolve_filter_doc_ids(f).unwrap();
        assert_eq!(actual_ids, expected_ids, "filter mismatch for {f:?}");
    }

    // 3. Facets parity (inline & detached terms + stats).
    let specs = vec![
        FacetSpec::terms("cat").with_limit(10),
        FacetSpec::terms("badge").with_limit(10),
        FacetSpec::stats("pop"),
    ];
    let f_actual = col.facet(None, &specs).unwrap();
    let mut cat_counts: HashMap<String, u64> = HashMap::new();
    let mut badge_counts: HashMap<String, u64> = HashMap::new();
    let mut pop_min: Option<i64> = None;
    let mut pop_max: Option<i64> = None;
    let mut pop_sum = 0.0f64;
    let mut pop_cnt = 0u64;
    for d in oracle.values() {
        if let Some(AttributeValue::String(c)) = d.attributes.get("cat") {
            *cat_counts.entry(c.clone()).or_default() += 1;
        }
        if let Some(AttributeValue::String(b)) = d.attributes.get("badge") {
            *badge_counts.entry(b.clone()).or_default() += 1;
        }
        if let Some(AttributeValue::Int(p)) = d.attributes.get("pop") {
            pop_min = Some(pop_min.map_or(*p, |m| m.min(*p)));
            pop_max = Some(pop_max.map_or(*p, |m| m.max(*p)));
            pop_sum += *p as f64;
            pop_cnt += 1;
        }
    }
    let cat_buckets: HashMap<String, u64> = f_actual
        .terms("cat")
        .unwrap()
        .buckets
        .iter()
        .map(|b| match &b.value {
            ScalarValue::String(s) => (s.clone(), b.count),
            other => panic!("expected string bucket, got {other:?}"),
        })
        .collect();
    assert_eq!(cat_buckets, cat_counts);
    let badge_buckets: HashMap<String, u64> = f_actual
        .terms("badge")
        .unwrap()
        .buckets
        .iter()
        .map(|b| match &b.value {
            ScalarValue::String(s) => (s.clone(), b.count),
            other => panic!("expected string bucket, got {other:?}"),
        })
        .collect();
    assert_eq!(badge_buckets, badge_counts);
    let pop_stats = f_actual.stats("pop").unwrap();
    assert_eq!(pop_stats.count, pop_cnt);
    if pop_cnt > 0 {
        assert_eq!(pop_stats.min, pop_min.map(ScalarValue::Int));
        assert_eq!(pop_stats.max, pop_max.map(ScalarValue::Int));
        assert!((pop_stats.sum - pop_sum).abs() < 1e-6);
    }

    // 4. OrderBy parity (inline & detached, NullsOrder::Last default).
    for (attr, is_int) in [("score", false), ("pop", true)] {
        let actual = col
            .order_by(&[OrderBySpec::desc(attr)], None, 10)
            .unwrap();
        let mut expected: Vec<(DocId, Option<f64>)> = oracle
            .iter()
            .map(|(&id, d)| {
                let v = if is_int {
                    match d.attributes.get(attr) {
                        Some(AttributeValue::Int(x)) => Some(*x as f64),
                        _ => None,
                    }
                } else {
                    match d.attributes.get(attr) {
                        Some(AttributeValue::Float(x)) => Some(*x),
                        _ => None,
                    }
                };
                (id, v)
            })
            .collect();
        expected.sort_by(|a, b| match (a.1, b.1) {
            (Some(x), Some(y)) => y.total_cmp(&x).then_with(|| a.0.cmp(&b.0)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.0.cmp(&b.0),
        });
        expected.truncate(10);
        let actual_ids: Vec<DocId> = actual.iter().map(|r| r.id).collect();
        let expected_ids: Vec<DocId> = expected.iter().map(|&(id, _)| id).collect();
        assert_eq!(actual_ids, expected_ids, "order_by({attr}) mismatch");
    }

    // 5. FTS query parity (smoke non-empty & consistency across open/sealed/reopen).
    let bm25_res = col
        .query_bm25(&RankBy::bm25("body", "vector segment"), None, 10)
        .unwrap();
    let expected_matching: usize = oracle
        .values()
        .filter(|d| {
            matches!(
                d.attributes.get("body"),
                Some(AttributeValue::String(s)) if s.contains("vector") || s.contains("segment")
            )
        })
        .count();
    assert_eq!(bm25_res.len(), expected_matching.min(10));

    // 6. Exact-regime filtered vector search parity when configured.
    if with_vectors && !oracle.is_empty() {
        let q = rand_vec(rng, 6);
        let filt = Filter::Eq("flag".into(), true.into());
        let opts = SearchOptions::new(5, 8).with_exact_scan_threshold(oracle.len() + 10);
        let mut trace = izzydb::FilterPlanTrace::default();
        let actual = col.search_with_trace(&q, opts, Some(&filt), &mut trace).unwrap();
        assert_eq!(
            trace.regime,
            Some(izzydb::FilteredSearchRegime::ExactScan)
        );
        let mut expected: Vec<(DocId, f32)> = oracle
            .iter()
            .filter(|(_, d)| eval_filter_oracle(d, &filt))
            .filter_map(|(&id, d)| d.embedding.as_ref().map(|e| (id, l2_squared(&q, e))))
            .collect();
        expected.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        expected.truncate(5);
        assert_eq!(actual.len(), expected.len());
        for (act, (_, exp_dist)) in actual.iter().zip(expected.iter()) {
            let act_dist = l2_squared(&q, oracle[&act.id].embedding.as_ref().unwrap());
            assert!(
                (act_dist - *exp_dist).abs() < 1e-5,
                "exact vector search score mismatch: act_id={} dist={act_dist} vs exp_dist={exp_dist}",
                act.id
            );
        }
    }
}

fn run_model_fuzz(seed: u64, with_vectors: bool, seal_rows: u32, steps: usize) {
    let dir = TempDir::new().unwrap();
    let cfg = build_model_config(with_vectors, seal_rows);
    let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();
    let mut oracle: BTreeMap<DocId, Document> = BTreeMap::new();
    let mut rng = SmallRng::seed_from_u64(seed);
    let max_id = 45u64;

    for step in 0..steps {
        let op = rng.random_range(0..100);
        match op {
            // Upsert (25%)
            0..=24 => {
                let id = rng.random_range(1..=max_id);
                let emb = if with_vectors {
                    Some(rand_vec(&mut rng, 6))
                } else {
                    None
                };
                let mut effective = rand_attrs(&mut rng, true);
                col.upsert(id, emb.as_deref(), effective.clone()).unwrap();
                effective.retain(|_, v| !matches!(v, AttributeValue::Null));
                oracle.insert(
                    id,
                    Document {
                        id,
                        embedding: emb,
                        attributes: effective,
                    },
                );
            }
            // Patch (15%)
            25..=39 => {
                if let Some(&id) = oracle.keys().nth(rng.random_range(0..oracle.len().max(1))) {
                    let mut patch = DocumentPatch::new();
                    if with_vectors && rng.random_bool(0.4) {
                        patch = patch.with_embedding(rand_vec(&mut rng, 6));
                    }
                    if rng.random_bool(0.6) {
                        patch = patch.with_attribute("cat", "alpha".into());
                    } else {
                        patch = patch.clear_attribute("cat");
                    }
                    if rng.random_bool(0.5) {
                        patch = patch.with_attribute("pop", rng.random_range(-5i64..80).into());
                    }
                    col.patch(id, patch.clone()).unwrap();
                    apply_patch_to_oracle(oracle.get_mut(&id).unwrap(), &patch);
                }
            }
            // Delete (10%)
            40..=49 => {
                let id = rng.random_range(1..=max_id);
                let existed = col.delete(id).unwrap();
                assert_eq!(existed, oracle.remove(&id).is_some());
            }
            // Point set_detached (10%)
            50..=59 => {
                if let Some(&id) = oracle.keys().nth(rng.random_range(0..oracle.len().max(1))) {
                    let mut det = Attributes::new();
                    if rng.random_bool(0.7) {
                        det.insert("pop".into(), rng.random_range(0i64..120).into());
                    } else {
                        det.insert("pop".into(), AttributeValue::Null);
                    }
                    if rng.random_bool(0.6) {
                        det.insert("badge".into(), "gold".into());
                    }
                    col.set_detached_batch(&[(id, det.clone())]).unwrap();
                    let doc = oracle.get_mut(&id).unwrap();
                    for (k, v) in det {
                        if matches!(v, AttributeValue::Null) {
                            doc.attributes.remove(&k);
                        } else {
                            doc.attributes.insert(k, v);
                        }
                    }
                }
            }
            // Detached whole-column ingest/publish (8%)
            60..=67 => {
                let mode = if rng.random_bool(0.5) {
                    DetachedWriteMode::Replace
                } else {
                    DetachedWriteMode::Merge
                };
                let mut pairs = Vec::new();
                for id in 1..=max_id {
                    if rng.random_bool(0.45) {
                        let pop_val = AttributeValue::Int(rng.random_range(1..=99));
                        pairs.push((id, vec![pop_val.clone()]));
                    }
                }
                let pairs_map: HashMap<DocId, AttributeValue> = pairs
                    .iter()
                    .map(|(id, vals)| (*id, vals[0].clone()))
                    .collect();
                let job = col
                    .detached_ingest()
                    .column("pop")
                    .mode(mode)
                    .duplicate_policy(DetachedDuplicatePolicy::LastWins)
                    .from_iter(pairs)
                    .build()
                    .unwrap();
                col.publish_detached(job).unwrap();

                for (&id, doc) in &mut oracle {
                    if let Some(val) = pairs_map.get(&id) {
                        doc.attributes.insert("pop".into(), val.clone());
                    } else if mode == DetachedWriteMode::Replace {
                        doc.attributes.remove("pop");
                    }
                }
            }
            // update_by_filter & delete_by_filter (7%)
            68..=74 => {
                if rng.random_bool(0.5) {
                    let f = Filter::Eq("cat".into(), "delta".into());
                    let patch = DocumentPatch::new().with_attribute("flag", false.into());
                    let n = col.update_by_filter(&f, &patch).unwrap();
                    let mut exp = 0u64;
                    for doc in oracle.values_mut() {
                        if eval_filter_oracle(doc, &f) {
                            apply_patch_to_oracle(doc, &patch);
                            exp += 1;
                        }
                    }
                    assert_eq!(n, exp);
                } else {
                    let f = Filter::Eq("badge".into(), "bronze".into());
                    let n = col.delete_by_filter(&f).unwrap();
                    let to_remove: Vec<DocId> = oracle
                        .iter()
                        .filter(|(_, d)| eval_filter_oracle(d, &f))
                        .map(|(&id, _)| id)
                        .collect();
                    assert_eq!(n, to_remove.len() as u64);
                    for id in to_remove {
                        oracle.remove(&id);
                    }
                }
            }
            // Seal / Compact / Maintenance / Vacuum (15%)
            75..=89 => match rng.random_range(0..4) {
                0 => {
                    col.seal().unwrap();
                }
                1 => {
                    col.compact(Default::default()).unwrap();
                }
                2 => {
                    col.maintenance(MaintenanceOptions::default()).unwrap();
                }
                _ => {
                    col.vacuum_ann(Default::default()).unwrap();
                }
            },
            // Reopen (10%)
            _ => {
                drop(col);
                col = Collection::open(dir.path(), cfg.clone()).unwrap();
            }
        }

        col.verify_invariants()
            .unwrap_or_else(|e| panic!("invariant failure at step {step} (op={op}): {e}"));
        assert_eq!(col.len(), oracle.len() as u64, "len mismatch at step {step}");

        if step % 15 == 0 {
            check_queries_parity(&col, &oracle, &mut rng, with_vectors);
        }
    }

    check_queries_parity(&col, &oracle, &mut rng, with_vectors);
}

#[test]
fn randomized_state_machine_model_with_vectors() {
    run_model_fuzz(20_261_123, true, 7, 140);
}

#[test]
fn randomized_state_machine_model_vectorless() {
    run_model_fuzz(9988_7766, false, 5, 100);
}
