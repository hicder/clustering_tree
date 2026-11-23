//! Phase 7 tests: sealed-segment `cf_columns`, `ColumnReader` vs `RowStoreReader` parity,
//! `defer_column_build` + `maintenance()`, SIMD batch decay/saturate, and columnar lazy filters.

use std::sync::Mutex;

use izzydb::facet::{FacetRange, FacetSpec};
use izzydb::{
    AttributeConfig, AttributeType, AttributeValue, Attributes, Collection, CollectionConfig,
    DecayMode, DistanceMetric, DocumentPatch, Filter, FullTextSearchConfig, MaintenanceOptions,
    NullsOrder, OrderBySpec, RankBy, ScalarValue, SegmentConfig, TreeConfig, TreeError,
    set_force_column_row_fallback,
};
use izzydb_storage::codec::{SEG_FLAG_HAS_COLUMNS, SegmentState};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

// `set_force_column_row_fallback` is a process-wide toggle, so serialize tests that flip it.
static FALLBACK_LOCK: Mutex<()> = Mutex::new(());

fn base_config(seal_rows: u32, defer_column_build: bool) -> CollectionConfig {
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
        .with_segment_config(SegmentConfig {
            seal_rows,
            defer_column_build,
        })
        .with_attribute("cat", AttributeType::String, true)
        .with_attribute("flag", AttributeType::Bool, true)
        .with_attribute("i", AttributeType::Int, true)
        .with_attribute("u", AttributeType::Uint, true)
        .with_attribute("f", AttributeType::Float, true)
        .with_attribute("nf_num", AttributeType::Float, false)
        .with_attribute_config(
            AttributeConfig::new("note_col", AttributeType::String)
                .filterable(false)
                .with_column(true),
        )
        .with_attribute("tags", AttributeType::StringArray, true)
        .with_attribute("nums", AttributeType::IntArray, true)
        .with_attribute("uints", AttributeType::UintArray, true)
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        )
}

fn populate_diverse(col: &mut Collection, n: u64) {
    let mut rng = SmallRng::seed_from_u64(0xC01_B10C);
    let cats = ["alpha", "beta", "gamma", "delta"];
    for id in 0..n {
        let mut a = Attributes::new();
        if id % 11 != 0 {
            a.insert("cat".into(), cats[(id as usize) % cats.len()].into());
        }
        a.insert("flag".into(), (id % 3 == 0).into());
        if id % 9 != 0 {
            a.insert(
                "i".into(),
                AttributeValue::Int(match id % 7 {
                    0 => -1_000_000_000_000i64 + (id as i64) * 987_654_321,
                    1 => -10 + (id as i64 % 5),
                    _ => rng.random_range(-5000..5000),
                }),
            );
        }
        if id % 10 != 0 {
            a.insert("u".into(), AttributeValue::Uint(100 + (id % 250)));
        }
        if id % 8 != 0 {
            let f = match id % 13 {
                0 => -0.0,
                1 => 0.0,
                _ => (rng.random_range(-100..100) as f64) * 0.25,
            };
            a.insert("f".into(), AttributeValue::Float(f));
        }
        if id % 4 != 0 {
            a.insert("nf_num".into(), AttributeValue::Float((id % 60) as f64 + 1.5));
        }
        if id % 5 != 0 {
            a.insert("note_col".into(), format!("note_{}", id % 17).into());
        }
        if id % 6 == 0 {
            a.insert("tags".into(), Vec::<String>::new().into());
        } else {
            a.insert(
                "tags".into(),
                vec![format!("t{}", id % 4), format!("u{}", id % 3)].into(),
            );
        }
        a.insert("nums".into(), vec![(id as i64 % 5) - 2, 10].into());
        a.insert("uints".into(), vec![id % 7, 100].into());
        a.insert("body".into(), if id % 2 == 0 { "columnar block scan" } else { "row store path" }.into());

        let v = vec![
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        ];
        col.upsert(id, Some(&v), a).unwrap();
    }
}

fn assert_all_queries_match_row_fallback(col: &Collection) {
    let run_suite = || {
        let mut out = Vec::new();
        // Facets (small + large population via filter)
        let f_broad = Filter::Eq("flag".into(), false.into());
        let f_small = Filter::Eq("cat".into(), "alpha".into());
        let specs = [
            FacetSpec::terms("cat").with_missing().with_limit(20),
            FacetSpec::stats("i").with_missing(),
            FacetSpec::histogram("f", 5.0).with_missing(),
            FacetSpec::ranges("nums", vec![FacetRange::between(-5i64, 5i64)]).with_missing(),
            FacetSpec::terms("tags").with_missing().with_limit(20),
        ];
        let fac_small = col.facet(Some(&f_small), &specs).unwrap();
        let fac_broad = col.facet(Some(&f_broad), &specs).unwrap();

        // OrderBy across scalar columns & non-filterable secondary opt-in column + null handling
        let ord1 = col
            .query_ordered(
                &[OrderBySpec::asc("f"), OrderBySpec::desc("i")],
                Some(&f_broad),
                40,
            )
            .unwrap();
        let ord2 = col
            .query_ordered(
                &[OrderBySpec::asc("u"), OrderBySpec::desc("note_col")],
                Some(&f_small),
                25,
            )
            .unwrap();

        // Ranking: Decay, Saturate, Attribute (both SIMD block fast path & multi-attr)
        let r_decay = col
            .query(
                &RankBy::Decay {
                    attr: "nf_num".into(),
                    origin: 20.0,
                    midpoint: 10.0,
                    exponent: 1.0,
                    mode: DecayMode::Symmetric,
                    missing: 0.0,
                },
                Some(&f_broad),
                30,
            )
            .unwrap();
        let r_sat = col
            .query(
                &RankBy::product(
                    2.5,
                    RankBy::Saturate {
                        attr: "u".into(),
                        midpoint: 150.0,
                        exponent: 1.0,
                        missing: 0.0,
                    },
                ),
                None,
                30,
            )
            .unwrap();
        let r_mix = col
            .query(
                &RankBy::Sum(vec![
                    RankBy::bm25("body", "columnar"),
                    RankBy::Decay {
                        attr: "nf_num".into(),
                        origin: 0.0,
                        midpoint: 25.0,
                        exponent: 1.5,
                        mode: DecayMode::Forward,
                        missing: 0.2,
                    },
                ]),
                Some(&Filter::Gt("i".into(), ScalarValue::Int(-100))),
                25,
            )
            .unwrap();

        // Columnar lazy residual filter resolution (`Not` on broad column)
        let lazy_ids = col
            .resolve_filter_doc_ids(&Filter::And(vec![
                Filter::Not(Box::new(Filter::Eq("flag".into(), true.into()))),
                Filter::Gt("f".into(), ScalarValue::Float(0.0)),
            ]))
            .unwrap();

        out.push(format!("{fac_small:?}"));
        out.push(format!("{fac_broad:?}"));
        out.push(format!("{ord1:?}"));
        out.push(format!("{ord2:?}"));
        out.push(format!("{r_decay:?}"));
        out.push(format!("{r_sat:?}"));
        out.push(format!("{r_mix:?}"));
        out.push(format!("{lazy_ids:?}"));
        out
    };

    let _guard = FALLBACK_LOCK.lock().unwrap();
    set_force_column_row_fallback(false);
    let col_res = run_suite();
    set_force_column_row_fallback(true);
    let row_res = run_suite();
    set_force_column_row_fallback(false);

    assert_eq!(col_res, row_res);
}

#[test]
fn test_column_vs_row_parity_sealed_open_and_tombstones() {
    let dir = TempDir::new().unwrap();
    let cfg = base_config(80, false);
    let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();
    // 290 docs -> 3 sealed segments (0, 1, 2) + open segment 3 (50 rows).
    populate_diverse(&mut col, 290);

    // Tombstone & update rows across both sealed and open segments.
    for id in (0..290u64).step_by(5) {
        if id % 10 == 0 {
            col.delete(id).unwrap();
        } else {
            col.patch(
                id,
                DocumentPatch::new()
                    .with_attribute("nf_num", AttributeValue::Float(99.0))
                    .clear_attribute("f"),
            )
            .unwrap();
        }
    }

    let snap = col.segment_snapshot();
    for (&seg, entry) in &snap.segments {
        if entry.meta.state == SegmentState::Sealed {
            assert_ne!(
                entry.meta.flags & SEG_FLAG_HAS_COLUMNS,
                0,
                "sealed segment {seg} must have HAS_COLUMNS"
            );
        } else {
            assert_eq!(
                entry.meta.flags & SEG_FLAG_HAS_COLUMNS,
                0,
                "open segment {seg} must not have HAS_COLUMNS"
            );
        }
    }

    assert_all_queries_match_row_fallback(&col);

    // Reopen and re-verify parity.
    drop(col);
    let col = Collection::open(dir.path(), cfg).unwrap();
    assert_all_queries_match_row_fallback(&col);
}

#[test]
fn test_defer_column_build_maintenance_parity() {
    let dir = TempDir::new().unwrap();
    let cfg = base_config(64, true);
    let mut col = Collection::open(dir.path(), cfg).unwrap();
    populate_diverse(&mut col, 160);

    // Before maintenance: sealed segments exist, but HAS_COLUMNS must NOT be set yet.
    let snap_before = col.segment_snapshot();
    let sealed_before = snap_before
        .segments
        .values()
        .filter(|e| e.meta.state == SegmentState::Sealed)
        .collect::<Vec<_>>();
    assert_eq!(sealed_before.len(), 2);
    for e in &sealed_before {
        assert_eq!(e.meta.flags & SEG_FLAG_HAS_COLUMNS, 0);
    }

    assert_all_queries_match_row_fallback(&col);

    // Run maintenance -> builds deferred columns for the 2 sealed segments.
    let stats = col.maintenance(MaintenanceOptions::default()).unwrap();
    assert_eq!(stats.columns_built, 2);

    let snap_after = col.segment_snapshot();
    for e in snap_after.segments.values() {
        if e.meta.state == SegmentState::Sealed {
            assert_ne!(e.meta.flags & SEG_FLAG_HAS_COLUMNS, 0);
        }
    }

    assert_all_queries_match_row_fallback(&col);

    // Idempotent second maintenance call builds 0 columns.
    let stats2 = col.maintenance(MaintenanceOptions::default()).unwrap();
    assert_eq!(stats2.columns_built, 0);
}

#[test]
fn test_column_schema_evolution_and_reconcile_mismatch() {
    let dir = TempDir::new().unwrap();
    let cfg1 = CollectionConfig::without_vectors()
        .with_seal_rows(32)
        .with_attribute("a", AttributeType::Int, true);

    {
        let mut col = Collection::open(dir.path(), cfg1.clone()).unwrap();
        for id in 0..40u64 {
            let mut attrs = Attributes::new();
            attrs.insert("a".into(), AttributeValue::Int(id as i64));
            col.upsert(id, None, attrs).unwrap();
        }
    }

    // Reopen adding a new columnar attribute `late_str`: sealed segment 0 has no blocks for it
    // (must read as Null), while new docs in segment 1+ get written & sealed into cf_columns.
    let cfg2 = cfg1.with_attribute_config(
        AttributeConfig::new("late_str", AttributeType::String)
            .filterable(false)
            .with_column(true),
    );
    {
        let mut col = Collection::open(dir.path(), cfg2.clone()).unwrap();
        for id in 40..70u64 {
            let mut attrs = Attributes::new();
            attrs.insert("a".into(), AttributeValue::Int(0));
            attrs.insert("late_str".into(), format!("val_{id}").into());
            col.upsert(id, None, attrs).unwrap();
        }
        col.seal().unwrap();

        // Verify segment 0 has no `late_str` column block while segment 1 does.
        let late_id = col.schema().get_by_name("late_str").unwrap().id;
        assert!(col.storage().get_column_block_pinned(0, late_id, 0).unwrap().is_none());
        assert!(col.storage().get_column_block_pinned(1, late_id, 0).unwrap().is_some());

        // Ordering by `(a asc, late_str asc)` reads `late_str` via ColumnReader across both segments.
        let ordered = col
            .query_ordered(
                &[
                    OrderBySpec::asc("a"),
                    OrderBySpec::asc("late_str").with_nulls(NullsOrder::First),
                ],
                None,
                50,
            )
            .unwrap();
        assert_eq!(ordered[0].id, 0);
        assert_eq!(ordered[0].sort_values[1], ScalarValue::Null);
        assert!(ordered.iter().any(|r| r.id >= 40 && r.sort_values[1] == ScalarValue::String("val_40".into())));
    }

    // Changing `column` opt-in on existing attribute must fail with ConfigMismatch.
    let bad_cfg = CollectionConfig::without_vectors()
        .with_seal_rows(32)
        .with_attribute("a", AttributeType::Int, true)
        .with_attribute_config(
            AttributeConfig::new("late_str", AttributeType::String)
                .filterable(false)
                .with_column(false),
        );
    assert!(matches!(
        Collection::open(dir.path(), bad_cfg),
        Err(TreeError::ConfigMismatch {
            field: "attribute_column",
            ..
        })
    ));
}
