//! Phase 5/6 coverage: leaf-colocated scalar columns, node summaries, and bit-sliced indexes.

use std::collections::HashSet;

use izzydb::bsi_codec::{BSI_BASE_PLANE, BSI_PRESENT_PLANE};
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric, Document, Filter,
    Liveness, ScalarValue, SearchOptions, TreeConfig,
};

fn config(seal_rows: u32, bsi: bool) -> CollectionConfig {
    let mut tree = TreeConfig::new(8, DistanceMetric::L2Squared);
    tree.min_leaf_size = 4;
    tree.max_leaf_size = 16;
    CollectionConfig::new(tree)
        .with_seal_rows(seal_rows)
        .with_bit_sliced_index(bsi)
        .with_attribute("score", AttributeType::Float, true)
        .with_attribute("rank", AttributeType::Int, true)
        .with_attribute("qty", AttributeType::Uint, true)
        .with_attribute("active", AttributeType::Bool, true)
        .with_attribute("label", AttributeType::String, true)
}

fn attrs(id: u64) -> Attributes {
    let mut attrs = Attributes::new();
    attrs.insert("label".into(), format!("label_{}", id % 5).into());
    attrs.insert("active".into(), id.is_multiple_of(3).into());
    if !id.is_multiple_of(11) {
        attrs.insert("rank".into(), (id as i64 * 7 - 900).into());
    }
    if !id.is_multiple_of(13) {
        attrs.insert("qty".into(), (id * 17).into());
    }
    if !id.is_multiple_of(17) {
        // Include negative zero and a wide range of magnitudes.
        let score = match id % 9 {
            0 => -0.0,
            1 => 0.0,
            2 => -123.5,
            _ => (id as f64) * 0.25 - 50.0,
        };
        attrs.insert("score".into(), score.into());
    }
    attrs
}

fn build(seal_rows: u32, bsi: bool) -> (tempfile::TempDir, Collection) {
    let dir = tempfile::tempdir().unwrap();
    let mut col = Collection::open(dir.path(), config(seal_rows, bsi)).unwrap();
    let docs: Vec<Document> = (0..600u64)
        .map(|id| Document {
            id,
            embedding: Some(vec![(id % 97) as f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            attributes: attrs(id),
        })
        .collect();
    for chunk in docs.chunks(37) {
        col.upsert_batch(chunk).unwrap();
    }
    (dir, col)
}

#[test]
fn leaf_attrs_survive_churn_reopen_and_search_only_returns_matches() {
    let (dir, mut col) = build(64, true);

    // Patch, delete, and insert after sealing to force LIRE/splits and leaf rewrites.
    for id in (0..600u64).step_by(7) {
        col.patch(
            id,
            izzydb::DocumentPatch {
                embedding: None,
                attributes: Attributes::from([("active".into(), (id % 2 == 0).into())]),
            },
        )
        .unwrap();
    }
    for id in (0..600u64).step_by(19) {
        col.delete(id).unwrap();
    }
    let extra: Vec<Document> = (600..720u64)
        .map(|id| Document {
            id,
            embedding: Some(vec![(id % 97) as f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            attributes: attrs(id),
        })
        .collect();
    col.upsert_batch(&extra).unwrap();

    let filters = [
        Filter::And(vec![
            Filter::Gte("score".into(), ScalarValue::Float(-20.0)),
            Filter::Lt("score".into(), ScalarValue::Float(40.0)),
            Filter::Eq("active".into(), true.into()),
        ]),
        Filter::And(vec![
            Filter::Gt("rank".into(), ScalarValue::Int(-400)),
            Filter::Lte("qty".into(), ScalarValue::Uint(5_000)),
            Filter::Eq("label".into(), "label_3".into()),
        ]),
        Filter::Or(vec![
            Filter::IsNull("score".into()),
            Filter::Eq("rank".into(), 28i64.into()),
        ]),
    ];

    let base_opts = SearchOptions::new(10, 8)
        .with_rerank(64)
        .with_max_expanded_leaves(512);
    let pruned_opts = base_opts.with_summary_pruning(true);

    for filter in &filters {
        let matched: HashSet<u64> = col
            .resolve_filter_doc_ids(filter)
            .unwrap()
            .into_iter()
            .collect();
        assert!(!matched.is_empty());
        let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hits = col.search(&query, base_opts, Some(filter)).unwrap();
        let pruned = col.search(&query, pruned_opts, Some(filter)).unwrap();
        assert_eq!(
            hits.iter().map(|h| h.id).collect::<Vec<_>>(),
            pruned.iter().map(|h| h.id).collect::<Vec<_>>(),
            "summary pruning changed results for {filter:?}"
        );
        assert!(!hits.is_empty());
        for hit in hits {
            assert!(
                matched.contains(&hit.id),
                "search returned {} outside exact filter set",
                hit.id
            );
        }
    }

    drop(col);
    let col = Collection::open(dir.path(), config(64, true)).unwrap();
    col.verify_invariants().unwrap();
    for filter in &filters {
        let matched = col.resolve_filter_doc_ids(filter).unwrap();
        assert!(!matched.is_empty());
    }
}

#[test]
fn bit_sliced_range_parity_and_fallback() {
    for bsi in [true, false] {
        let (dir, col) = build(80, bsi);
        let ranges = [
            Filter::And(vec![
                Filter::Gte("score".into(), (-10.0f64).into()),
                Filter::Lt("score".into(), 30.0f64.into()),
            ]),
            Filter::And(vec![
                Filter::Gte("score".into(), (-0.0f64).into()),
                Filter::Lte("score".into(), 0.0f64.into()),
            ]),
            Filter::Gt("score".into(), (-1000.0f64).into()),
            Filter::Lte("score".into(), 12.5f64.into()),
            Filter::And(vec![
                Filter::Gte("rank".into(), (-250i64).into()),
                Filter::Lte("rank".into(), 250i64.into()),
            ]),
            Filter::And(vec![
                Filter::Gt("qty".into(), 5_000u64.into()),
                Filter::Lte("qty".into(), 20_000u64.into()),
            ]),
        ];
        for filter in ranges {
            let got: HashSet<u64> = col
                .resolve_filter_doc_ids(&filter)
                .unwrap()
                .into_iter()
                .collect();
            let expected: HashSet<u64> =
                (0..600u64).filter(|&id| filter.matches(&attrs(id))).collect();
            assert_eq!(got, expected, "bsi={bsi} filter={filter:?}");
        }

        let bound_filters = [
            Filter::And(vec![
                Filter::Gte("qty".into(), 0u64.into()),
                Filter::Lt("qty".into(), 10_000u64.into()),
            ]),
            Filter::And(vec![
                Filter::Gte("rank".into(), (-1000i64).into()),
                Filter::Lt("rank".into(), 100i64.into()),
            ]),
        ];
        for filter in &bound_filters {
            let matched: HashSet<u64> = col
                .resolve_filter_doc_ids(filter)
                .unwrap()
                .into_iter()
                .collect();
            assert!(!matched.is_empty());
            let expected: HashSet<u64> =
                (0..600u64).filter(|&id| filter.matches(&attrs(id))).collect();
            assert_eq!(matched, expected);
        }

        // Warm path: repeated range queries and a reopen keep results stable.
        let filter = Filter::And(vec![
            Filter::Gte("score".into(), (-0.0f64).into()),
            Filter::Lt("score".into(), 12.5f64.into()),
        ]);
        let first = col.resolve_filter_doc_ids(&filter).unwrap();
        for _ in 0..4 {
            assert_eq!(col.resolve_filter_doc_ids(&filter).unwrap(), first);
        }
        drop(col);
        let col = Collection::open(dir.path(), config(80, bsi)).unwrap();
        assert_eq!(col.resolve_filter_doc_ids(&filter).unwrap(), first);
        col.verify_invariants().unwrap();
    }
}

#[test]
fn bit_sliced_presence_planes_are_written_at_seal() {
    let (_dir, col) = build(40, true);
    let storage = col.storage();
    let snap = col.segment_snapshot();
    let mut found = 0usize;
    for seg in snap.visible_segments() {
        for (attr_id, _) in [(0u16, ()), (1, ()), (2, ())] {
            if storage
                .get_bsi_plane(seg, attr_id, 0, BSI_PRESENT_PLANE)
                .unwrap()
                .is_some()
            {
                found += 1;
                assert!(
                    storage
                        .get_bsi_plane(seg, attr_id, 0, BSI_BASE_PLANE)
                        .unwrap()
                        .is_some()
                );
            }
        }
    }
    assert!(found > 0, "no BSI presence planes were built");
}

#[test]
fn null_and_negative_zero_leaf_filters_match_row_semantics() {
    let (_dir, col) = build(50, true);
    let neg_zero = Filter::Eq("score".into(), (-0.0f64).into());
    let pos_zero = Filter::Eq("score".into(), 0.0f64.into());
    assert_eq!(
        col.resolve_filter_doc_ids(&neg_zero).unwrap(),
        col.resolve_filter_doc_ids(&pos_zero).unwrap()
    );
    let missing = Filter::Eq("score".into(), ScalarValue::Null);
    let present = Filter::NotEq("score".into(), ScalarValue::Null);
    let missing_ids = col.resolve_filter_doc_ids(&missing).unwrap();
    let present_ids = col.resolve_filter_doc_ids(&present).unwrap();
    assert!(!missing_ids.is_empty() && !present_ids.is_empty());
    assert!(missing_ids.iter().all(|id| !present_ids.contains(id)));
}
