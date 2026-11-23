//! Phase 8 tests: detached columns (`cf_detached`, `cf_detached_postings`, `0x07/0x13/0x14`),
//! atomic build/publish ingest (SST + WriteBatch), concurrent catch-up, crash recovery,
//! CoW carry-over, point overwrites with `0x14` correction across filters/order_by/facets/RankBy,
//! and `maintenance()` > 5% overwritten rebuild.

use std::collections::HashMap;

use izzydb::facet::{FacetRange, FacetSpec};
use izzydb::{
    AttributeType, AttributeValue, Attributes, Collection, CollectionConfig, DecayMode,
    DetachedDuplicatePolicy, DetachedWriteMode, DistanceMetric, Document, DocumentPatch, Filter,
    FullTextSearchConfig, MaintenanceOptions, NullsOrder, OrderBySpec, RankBy, ScalarValue,
    SearchOptions, SortDirection, TreeConfig, TreeError,
};
use tempfile::TempDir;

fn test_config(seal_rows: u32) -> CollectionConfig {
    let tree = TreeConfig {
        dim: 4,
        metric: DistanceMetric::L2Squared,
        min_internal_fanout: 2,
        max_internal_fanout: 6,
        min_leaf_size: 2,
        max_leaf_size: 8,
        insert_beam_width: 4,
        lire_neighbor_count: 4,
        enable_lire: true,
        centroid_drift_eps: 1e-4,
    };
    CollectionConfig::new(tree)
        .with_seal_rows(seal_rows)
        .with_attribute("cat", AttributeType::String, true)
        .with_fts_attribute(
            "title",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        )
        .with_detached_attribute("pop", AttributeType::Float, true)
        .with_detached_attribute("rank_i", AttributeType::Int, true)
        .with_detached_attribute("labels", AttributeType::StringArray, true)
        .with_detached_attribute("raw_score", AttributeType::Float, false)
}

fn vec4(x: f32, y: f32) -> Vec<f32> {
    vec![x, y, 0.25, 0.5]
}

fn make_inline_doc(id: u64, cat: &str, title: &str, x: f32) -> Document {
    let mut attrs = Attributes::new();
    attrs.insert("cat".into(), AttributeValue::String(cat.into()));
    attrs.insert("title".into(), AttributeValue::String(title.into()));
    Document {
        id,
        embedding: Some(vec4(x, (id % 7) as f32)),
        attributes: attrs,
    }
}

#[test]
fn detached_ingest_replace_merge_duplicates_and_sst_path() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), test_config(16)).unwrap();

    let docs: Vec<Document> = (1..=60)
        .map(|i| make_inline_doc(i, if i % 2 == 0 { "even" } else { "odd" }, "alpha", i as f32))
        .collect();
    col.upsert_batch(&docs).unwrap();

    // 1. Duplicate policy Error rejects duplicate DocIds.
    let err = col
        .detached_ingest()
        .column("pop")
        .duplicate_policy(DetachedDuplicatePolicy::Error)
        .from_pairs([(1, AttributeValue::Float(1.0)), (1, AttributeValue::Float(2.0))])
        .build()
        .unwrap_err();
    assert!(matches!(err, TreeError::InvalidSchema(_)));

    // 2. Duplicate policy FirstWins & SST ingest path (sst_threshold = 1).
    let job = col
        .detached_ingest()
        .columns(["pop", "labels"])
        .mode(DetachedWriteMode::Replace)
        .duplicate_policy(DetachedDuplicatePolicy::FirstWins)
        .sst_threshold(1)
        .from_iter([
            (
                10,
                vec![
                    AttributeValue::Float(100.0),
                    AttributeValue::StringArray(vec!["hot".into(), "featured".into()]),
                ],
            ),
            (
                10,
                vec![
                    AttributeValue::Float(999.0),
                    AttributeValue::StringArray(vec!["wrong".into()]),
                ],
            ),
            (
                20,
                vec![
                    AttributeValue::Float(200.0),
                    AttributeValue::StringArray(vec!["sale".into()]),
                ],
            ),
            (
                999_999,
                vec![
                    AttributeValue::Float(5.0),
                    AttributeValue::StringArray(vec!["unknown".into()]),
                ],
            ),
        ])
        .build()
        .unwrap();
    assert_eq!(job.stats.columns, 2);
    assert_eq!(job.stats.unknown_ids, 1);
    assert_eq!(job.stats.matched_rows, 2);
    assert!(job.stats.used_sst);
    // Before publish: staged generation must remain invisible!
    assert!(!col.get(10).unwrap().unwrap().attributes.contains_key("pop"));

    let pub_stats = col.publish_detached(job).unwrap();
    assert_eq!(pub_stats.columns, 2);
    assert_eq!(
        col.get(10).unwrap().unwrap().attributes.get("pop"),
        Some(&AttributeValue::Float(100.0))
    );
    assert_eq!(
        col.get(10).unwrap().unwrap().attributes.get("labels"),
        Some(&AttributeValue::StringArray(vec![
            "hot".into(),
            "featured".into()
        ]))
    );
    assert!(!col.get(1).unwrap().unwrap().attributes.contains_key("pop"));

    // 3. Merge overlays without wiping omitted docs; LastWins picks last value.
    let merge_job = col
        .detached_ingest()
        .column("pop")
        .mode(DetachedWriteMode::Merge)
        .duplicate_policy(DetachedDuplicatePolicy::LastWins)
        .from_pairs([
            (20, AttributeValue::Float(250.0)),
            (20, AttributeValue::Float(275.0)),
            (30, AttributeValue::Float(300.0)),
        ])
        .build()
        .unwrap();
    col.publish_detached(merge_job).unwrap();

    assert_eq!(
        col.get(10).unwrap().unwrap().attributes.get("pop"),
        Some(&AttributeValue::Float(100.0)),
        "omitted doc retains prior value under Merge"
    );
    assert_eq!(
        col.get(20).unwrap().unwrap().attributes.get("pop"),
        Some(&AttributeValue::Float(275.0))
    );
    assert_eq!(
        col.get(30).unwrap().unwrap().attributes.get("pop"),
        Some(&AttributeValue::Float(300.0))
    );

    // 4. Replace clears omitted docs (30 -> Null) when replaced cleanly.
    let rep_job = col
        .detached_ingest()
        .column("pop")
        .mode(DetachedWriteMode::Replace)
        .from_pairs([(10, AttributeValue::Float(50.0))])
        .build()
        .unwrap();
    col.publish_detached(rep_job).unwrap();
    assert_eq!(
        col.get(10).unwrap().unwrap().attributes.get("pop"),
        Some(&AttributeValue::Float(50.0))
    );
    assert!(!col.get(20).unwrap().unwrap().attributes.contains_key("pop"));
    assert!(!col.get(30).unwrap().unwrap().attributes.contains_key("pop"));
    assert_eq!(
        col.resolve_filter_doc_ids(&Filter::IsNotNull("pop".into()))
            .unwrap(),
        vec![10]
    );
}

#[test]
fn detached_concurrent_mutations_between_build_and_publish() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), test_config(10)).unwrap();

    // Seed 25 docs across 3 segments.
    let init_docs: Vec<Document> = (1..=25)
        .map(|i| make_inline_doc(i, "seed", "gamma", i as f32))
        .collect();
    col.upsert_batch(&init_docs).unwrap();

    // Build whole-column Replace job updating pop for docs 1..=35 (including not-yet-inserted 26..=32!).
    let job = col
        .detached_ingest()
        .column("pop")
        .mode(DetachedWriteMode::Replace)
        .from_pairs((1..=35).map(|i| (i, AttributeValue::Float(i as f64 * 10.0))))
        .build()
        .unwrap();

    // Mutate concurrently before publish:
    // - Delete docs 2, 3, 15 (< watermark)
    col.delete_batch(&[2, 3, 15]).unwrap();
    // - Inline-update docs 5 and 20 (moves their RowIds >= watermark!)
    col.patch(
        5,
        DocumentPatch::new().with_attribute("cat", AttributeValue::String("moved".into())),
    )
    .unwrap();
    col.upsert(20, Some(&vec4(20.0, 2.0)), {
        let mut a = Attributes::new();
        a.insert("cat".into(), AttributeValue::String("rewritten".into()));
        a
    })
    .unwrap();
    // - Insert new docs 26..=30 (in input) and 99 (not in input)
    let extra: Vec<Document> = (26..=30)
        .chain([99])
        .map(|i| make_inline_doc(i, "new", "gamma", i as f32))
        .collect();
    col.upsert_batch(&extra).unwrap();

    let pub_stats = col.publish_detached(job).unwrap();
    assert_eq!(pub_stats.post_snapshot_deletes, 5); // 3 deletes + 2 old RowIds of CoW docs 5 and 20
    assert!(pub_stats.catchup_rows >= 8);

    // Every live doc in 1..=30 (except deleted 2,3,15) must have pop = id * 10.0; doc 99 must have Null.
    let mut expected_present = Vec::new();
    for id in 1..=30u64 {
        if matches!(id, 2 | 3 | 15) {
            assert!(col.get(id).unwrap().is_none());
        } else {
            expected_present.push(id);
            let doc = col.get(id).unwrap().unwrap();
            assert_eq!(
                doc.attributes.get("pop"),
                Some(&AttributeValue::Float(id as f64 * 10.0)),
                "doc {id} pop mismatch after catch-up publish"
            );
        }
    }
    let doc99 = col.get(99).unwrap().unwrap();
    assert!(!doc99.attributes.contains_key("pop"));

    // Counts and postings must reflect exact live rows (no phantom count from deleted rows!).
    let got_present = col
        .resolve_filter_doc_ids(&Filter::IsNotNull("pop".into()))
        .unwrap();
    assert_eq!(got_present, expected_present);

    // Same test for Merge mode with concurrent inline update carrying prior value when omitted from input.
    let merge_job = col
        .detached_ingest()
        .column("pop")
        .mode(DetachedWriteMode::Merge)
        .from_pairs([(10u64, AttributeValue::Float(999.0))])
        .build()
        .unwrap();
    // Doc 6 is moved to >= watermark before publish while NOT in merge_job's input -> must keep 60.0!
    col.patch(
        6,
        DocumentPatch::new().with_attribute("cat", AttributeValue::String("moved2".into())),
    )
    .unwrap();
    col.publish_detached(merge_job).unwrap();
    assert_eq!(
        col.get(6).unwrap().unwrap().attributes.get("pop"),
        Some(&AttributeValue::Float(60.0))
    );
    assert_eq!(
        col.get(10).unwrap().unwrap().attributes.get("pop"),
        Some(&AttributeValue::Float(999.0))
    );
}

#[test]
fn detached_crash_recovery_discards_pending_generation() {
    let dir = TempDir::new().unwrap();
    let pop_id;
    {
        let mut col = Collection::open(dir.path(), test_config(16)).unwrap();
        pop_id = col.schema().get_by_name("pop").unwrap().id;
        let docs: Vec<Document> = (1..=20)
            .map(|i| make_inline_doc(i, "a", "t", i as f32))
            .collect();
        col.upsert_batch(&docs).unwrap();

        let job1 = col
            .detached_ingest()
            .column("pop")
            .from_pairs((1..=20).map(|i| (i, AttributeValue::Float(1.5))))
            .build()
            .unwrap();
        col.publish_detached(job1).unwrap();

        // Build gen 2 and drop without publishing (simulates crash before publish commit).
        let _uncommitted = col
            .detached_ingest()
            .column("pop")
            .from_pairs((1..=20).map(|i| (i, AttributeValue::Float(888.0))))
            .build()
            .unwrap();
        let meta = col.storage().get_detached_meta(pop_id).unwrap().unwrap();
        assert_eq!(meta.current_gen, 1);
        assert_eq!(meta.pending_gen, 2);
    }

    // Reopen must recover pending_gen = 0, bump retired_below >= 3, and preserve gen 1 reads.
    let mut col = Collection::open(dir.path(), test_config(16)).unwrap();
    let meta = col.storage().get_detached_meta(pop_id).unwrap().unwrap();
    assert_eq!(meta.current_gen, 1);
    assert_eq!(meta.pending_gen, 0);
    assert!(meta.retired_below >= 3);
    assert_eq!(
        col.get(1).unwrap().unwrap().attributes.get("pop"),
        Some(&AttributeValue::Float(1.5))
    );
    assert_eq!(
        col.resolve_filter_doc_ids(&Filter::Eq("pop".into(), ScalarValue::Float(888.0)))
            .unwrap(),
        Vec::<u64>::new()
    );

    // Subsequent build+publish picks gen >= 3 and works cleanly.
    let job3 = col
        .detached_ingest()
        .column("pop")
        .from_pairs((1..=20).map(|i| (i, AttributeValue::Float(3.5))))
        .build()
        .unwrap();
    col.publish_detached(job3).unwrap();
    let meta = col.storage().get_detached_meta(pop_id).unwrap().unwrap();
    assert!(meta.current_gen >= 3);
    assert_eq!(
        col.get(1).unwrap().unwrap().attributes.get("pop"),
        Some(&AttributeValue::Float(3.5))
    );
}

#[test]
fn detached_point_writes_preserve_rowid_and_cow_carryover() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), test_config(8)).unwrap();

    let mut d1 = make_inline_doc(1, "books", "rust", 1.0);
    d1.attributes.insert("pop".into(), AttributeValue::Float(10.0));
    d1.attributes.insert("rank_i".into(), AttributeValue::Int(7));
    d1.attributes
        .insert("raw_score".into(), AttributeValue::Float(0.9));
    col.upsert_batch(&[d1, make_inline_doc(2, "music", "jazz", 2.0)])
        .unwrap();

    let row1_before = col.storage().get_id_map(1).unwrap().unwrap();

    // 1. set_detached & detached-only patch/upsert keep RowId identical.
    col.set_detached(1, "pop", AttributeValue::Float(25.0))
        .unwrap();
    assert_eq!(col.storage().get_id_map(1).unwrap(), Some(row1_before));

    col.patch(
        1,
        DocumentPatch::new().with_attribute("pop", AttributeValue::Float(30.0)),
    )
    .unwrap();
    assert_eq!(col.storage().get_id_map(1).unwrap(), Some(row1_before));

    // Upsert with identical inline attributes & vector preserves RowId and resets omitted detached attrs.
    col.upsert(1, Some(&vec4(1.0, 1.0)), {
        let mut a = Attributes::new();
        a.insert("cat".into(), AttributeValue::String("books".into()));
        a.insert("title".into(), AttributeValue::String("rust".into()));
        a.insert("pop".into(), AttributeValue::Float(40.0));
        a
    })
    .unwrap();
    assert_eq!(col.storage().get_id_map(1).unwrap(), Some(row1_before));
    let doc1 = col.get(1).unwrap().unwrap();
    assert_eq!(doc1.attributes.get("pop"), Some(&AttributeValue::Float(40.0)));
    assert!(
        !doc1.attributes.contains_key("rank_i"),
        "omitted detached attr must be cleared on upsert"
    );

    // Re-set rank_i & raw_score, then perform an inline patch -> RowId moves and detached values carry over!
    col.set_detached_batch(&[(1, {
        let mut a = Attributes::new();
        a.insert("rank_i".into(), AttributeValue::Int(42));
        a.insert("raw_score".into(), AttributeValue::Float(0.75));
        a
    })])
    .unwrap();

    col.patch(
        1,
        DocumentPatch::new()
            .with_attribute("cat", AttributeValue::String("tech".into()))
            .with_attribute("pop", AttributeValue::Float(55.0)),
    )
    .unwrap();

    let row1_after = col.storage().get_id_map(1).unwrap().unwrap();
    assert_ne!(row1_after, row1_before, "inline change must CoW RowId");

    let doc1_after = col.get(1).unwrap().unwrap();
    assert_eq!(
        doc1_after.attributes.get("cat"),
        Some(&AttributeValue::String("tech".into()))
    );
    assert_eq!(
        doc1_after.attributes.get("pop"),
        Some(&AttributeValue::Float(55.0))
    );
    assert_eq!(
        doc1_after.attributes.get("rank_i"),
        Some(&AttributeValue::Int(42)),
        "untouched detached attr must carry over on CoW patch"
    );
    assert_eq!(
        doc1_after.attributes.get("raw_score"),
        Some(&AttributeValue::Float(0.75))
    );
    assert_eq!(
        col.resolve_filter_doc_ids(&Filter::Eq("pop".into(), ScalarValue::Float(40.0)))
            .unwrap(),
        Vec::<u64>::new()
    );
    assert_eq!(
        col.resolve_filter_doc_ids(&Filter::Eq("pop".into(), ScalarValue::Float(55.0)))
            .unwrap(),
        vec![1]
    );
}

#[test]
fn detached_overwritten_queries_orderby_facets_rankby_and_maintenance_rebuild() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), test_config(16)).unwrap();

    // 48 docs across 3 segments (2 sealed, 1 open).
    let docs: Vec<Document> = (1..=48)
        .map(|i| {
            let mut d = make_inline_doc(
                i,
                if i % 3 == 0 { "c" } else if i % 3 == 1 { "a" } else { "b" },
                if i % 2 == 0 { "vector search" } else { "storage engine" },
                i as f32,
            );
            d.attributes.insert(
                "pop".into(),
                AttributeValue::Float((i % 5 + 1) as f64 * 10.0),
            );
            d.attributes
                .insert("rank_i".into(), AttributeValue::Int((i % 4) as i64));
            d.attributes.insert(
                "labels".into(),
                AttributeValue::StringArray(vec![format!("g{}", i % 3)]),
            );
            d
        })
        .collect();
    col.upsert_batch(&docs).unwrap();

    // Point-overwrite 10 docs (> 20% of 48 > 5%) across BOTH sealed and open segments:
    // some get new values, doc 5 becomes Null, doc 48 becomes 999.0.
    let mut truth_pop: HashMap<u64, Option<f64>> = (1..=48)
        .map(|i| (i, Some((i % 5 + 1) as f64 * 10.0)))
        .collect();
    let mut truth_labels: HashMap<u64, Vec<String>> = (1..=48)
        .map(|i| (i, vec![format!("g{}", i % 3)]))
        .collect();

    let mut point_updates = Vec::new();
    for id in [1u64, 5, 10, 16, 17, 25, 32, 33, 40, 48] {
        let mut a = Attributes::new();
        if id == 5 {
            a.insert("pop".into(), AttributeValue::Null);
            a.insert("labels".into(), AttributeValue::StringArray(vec!["cleared".into()]));
            truth_pop.insert(id, None);
            truth_labels.insert(id, vec!["cleared".into()]);
        } else {
            let new_pop = if id == 48 { 999.0 } else { 77.0 + (id % 3) as f64 };
            a.insert("pop".into(), AttributeValue::Float(new_pop));
            a.insert("labels".into(), AttributeValue::StringArray(vec!["updated".into()]));
            truth_pop.insert(id, Some(new_pop));
            truth_labels.insert(id, vec!["updated".into()]);
        }
        point_updates.push((id, a));
    }
    col.set_detached_batch(&point_updates).unwrap();

    let check_all_queries = |c: &Collection| {
        // 1. Old overwritten value filter excludes overwritten docs, includes non-overwritten docs.
        let mut expect_10: Vec<u64> = truth_pop
            .iter()
            .filter_map(|(&id, &v)| (v == Some(10.0)).then_some(id))
            .collect();
        expect_10.sort_unstable();
        assert_eq!(
            c.resolve_filter_doc_ids(&Filter::Eq("pop".into(), ScalarValue::Float(10.0)))
                .unwrap(),
            expect_10
        );

        // 2. Range filter + Null check.
        let mut expect_gt_70: Vec<u64> = truth_pop
            .iter()
            .filter_map(|(&id, &v)| v.is_some_and(|x| x > 70.0).then_some(id))
            .collect();
        expect_gt_70.sort_unstable();
        assert_eq!(
            c.resolve_filter_doc_ids(&Filter::Gt("pop".into(), ScalarValue::Float(70.0)))
                .unwrap(),
            expect_gt_70
        );
        assert_eq!(
            c.resolve_filter_doc_ids(&Filter::IsNull("pop".into()))
                .unwrap(),
            vec![5]
        );

        // 3. ArrayContains on detached StringArray + mixed filter (inline + detached).
        let mut expect_updated: Vec<u64> = truth_labels
            .iter()
            .filter_map(|(&id, ls)| ls.contains(&"updated".to_string()).then_some(id))
            .collect();
        expect_updated.sort_unstable();
        assert_eq!(
            c.resolve_filter_doc_ids(&Filter::Contains(
                "labels".into(),
                ScalarValue::String("updated".into())
            ))
            .unwrap(),
            expect_updated
        );
        let mixed = c
            .resolve_filter_doc_ids(&Filter::And(vec![
                Filter::Eq("cat".into(), ScalarValue::String("a".into())),
                Filter::Gte("pop".into(), ScalarValue::Float(77.0)),
            ]))
            .unwrap();
        let mut expect_mixed: Vec<u64> = truth_pop
            .iter()
            .filter_map(|(&id, &v)| (id % 3 == 1 && v.is_some_and(|x| x >= 77.0)).then_some(id))
            .collect();
        expect_mixed.sort_unstable();
        assert_eq!(mixed, expect_mixed);

        // 4. ORDER BY pop DESC + ORDER BY pop ASC with NullsOrder::First
        let top_desc = c
            .order_by(
                &[OrderBySpec {
                    attr: "pop".into(),
                    direction: SortDirection::Desc,
                    nulls: NullsOrder::Last,
                }],
                None,
                5,
            )
            .unwrap();
        let mut expected_desc: Vec<(u64, f64)> = truth_pop
            .iter()
            .filter_map(|(&id, &v)| v.map(|x| (id, x)))
            .collect();
        expected_desc.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        assert_eq!(
            top_desc.iter().map(|r| r.id).collect::<Vec<_>>(),
            expected_desc.iter().take(5).map(|x| x.0).collect::<Vec<_>>()
        );

        let top_nulls_first = c
            .order_by(
                &[OrderBySpec {
                    attr: "pop".into(),
                    direction: SortDirection::Asc,
                    nulls: NullsOrder::First,
                }],
                None,
                3,
            )
            .unwrap();
        assert_eq!(top_nulls_first[0].id, 5);

        // 5. Facets: unfiltered (global 0x07) & filtered (Strategy 4 postings / Strategy 3).
        let specs = [
            FacetSpec::terms("pop").with_limit(20),
            FacetSpec::stats("pop").with_alias("pop:stats"),
            FacetSpec::ranges(
                "pop",
                vec![
                    FacetRange::below(50.0).with_label("low"),
                    FacetRange::from_value(50.0).with_label("high"),
                ],
            )
            .with_alias("pop:range"),
        ];
        let resp_all = c.facet(None, &specs).unwrap();
        let stats_res = resp_all.stats("pop:stats").unwrap();
        assert_eq!(stats_res.count, 47);
        assert_eq!(stats_res.max, Some(ScalarValue::Float(999.0)));

        let resp_filt = c
            .facet(
                Some(&Filter::Eq("cat".into(), ScalarValue::String("a".into()))),
                &[FacetSpec::terms("pop").with_limit(20)],
            )
            .unwrap();
        let term_sum: u64 = resp_filt
            .terms("pop")
            .unwrap()
            .buckets
            .iter()
            .map(|b| b.count)
            .sum();
        let expect_a_non_null = truth_pop
            .iter()
            .filter(|(id, v)| **id % 3 == 1 && v.is_some())
            .count() as u64;
        assert_eq!(term_sum, expect_a_non_null);

        // 6. RankBy with detached Attribute / Saturate / Decay + BM25 + Vector Search
        let rank_hits = c
            .query(
                &RankBy::Sum(vec![
                    RankBy::Bm25 {
                        attr: "title".into(),
                        query: "vector".into(),
                        params: None,
                    },
                    RankBy::Saturate {
                        attr: "pop".into(),
                        midpoint: 100.0,
                        exponent: 1.0,
                        missing: 0.0,
                    },
                ]),
                Some(&Filter::Eq("cat".into(), ScalarValue::String("c".into()))),
                5,
            )
            .unwrap();
        assert!(!rank_hits.is_empty());
        assert_eq!(rank_hits[0].id, 48);

        let simd_hits = c
            .query(
                &RankBy::Decay {
                    attr: "pop".into(),
                    origin: 999.0,
                    midpoint: 50.0,
                    exponent: 1.0,
                    mode: DecayMode::Symmetric,
                    missing: 0.0,
                },
                None,
                3,
            )
            .unwrap();
        assert_eq!(simd_hits[0].id, 48);

        let vec_hits = c
            .search(
                &vec4(48.0, 6.0),
                SearchOptions::new(3, 16),
                Some(&Filter::Gt("pop".into(), ScalarValue::Float(500.0))),
            )
            .unwrap();
        assert_eq!(vec_hits.len(), 1);
        assert_eq!(vec_hits[0].id, 48);
    };

    check_all_queries(&col);

    // maintenance() must detect >5% overwritten on pop & labels, rebuild them cleanly,
    // clear 0x14 overwritten sets, and preserve all query results identically!
    let mstats = col.maintenance(MaintenanceOptions::default()).unwrap();
    assert_eq!(mstats.detached_rebuilt, 2);

    let pop_id = col.schema().get_by_name("pop").unwrap().id;
    let pop_meta = col.storage().get_detached_meta(pop_id).unwrap().unwrap();
    assert!(
        col.storage()
            .all_detached_overwritten(pop_id, pop_meta.current_gen)
            .unwrap()
            .is_empty()
    );

    check_all_queries(&col);
}
