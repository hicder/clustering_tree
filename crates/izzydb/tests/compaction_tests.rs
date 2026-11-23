//! Phase 9 tests: segment compaction (`Building -> Sealed -> Forwarding -> Retired`),
//! rank remap, ANN leaf remap during & after forwarding, FTS inline-block consolidation,
//! crash recovery across every step boundary, and RocksDB CompactionFilter orphan cleanup.

use std::collections::BTreeMap;
use std::sync::Arc;

use roaring::RoaringBitmap;

use izzydb::facet::FacetSpec;
use izzydb::{
    AttributeType, AttributeValue, Attributes, Collection, CollectionConfig, CompactionHookStage,
    CompactionOptions, DistanceMetric, Document, DocumentPatch, Filter, FullTextSearchConfig,
    Liveness, MaintenanceOptions, NullsOrder, OrderBySpec, RankBy, RowId, ScalarValue,
    SearchOptions, SortDirection, TokenQuery, TreeConfig, TreeError, VacuumBudget,
};
use izzydb_storage::codec::{
    FTS_DOC_LEN_TAG, SegmentMeta, SegmentState, ZONE_TAG, split_detached_posting_key,
    split_posting_key,
};
use izzydb_storage::{
    CF_ATTR_POSTINGS, CF_COLUMNS, CF_DETACHED, CF_DETACHED_POSTINGS, CF_FTS_POSITIONS,
    CF_FTS_POSTINGS, CF_FTS_SKIPS, CF_ROW_DOC_IDS, CF_ROWS, CF_VECTORS, SegmentEntry,
    SegmentTableSnapshot,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn full_config(seal_rows: u32) -> CollectionConfig {
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
        .with_attribute("score", AttributeType::Int, true)
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default().with_index_positions(true),
        )
        .with_detached_attribute("pop", AttributeType::Float, true)
}

fn make_doc(id: u64, cat: &str, score: i64, body: &str, pop: Option<f64>) -> Document {
    let mut attrs = Attributes::new();
    attrs.insert("cat".into(), AttributeValue::String(cat.into()));
    attrs.insert("score".into(), AttributeValue::Int(score));
    attrs.insert("body".into(), AttributeValue::String(body.into()));
    if let Some(p) = pop {
        attrs.insert("pop".into(), AttributeValue::Float(p));
    }
    Document {
        id,
        embedding: Some(vec![id as f32 * 0.1, (id % 5) as f32, 0.5, 1.0]),
        attributes: attrs,
    }
}

#[test]
fn forwarding_rank_live_matches_naive_table_across_random_bitmaps() {
    let mut rng = SmallRng::seed_from_u64(0xC09A_C710);
    for _ in 0..50 {
        let row_count = rng.random_range(1..300u32);
        let base = rng.random_range(0..500u32);
        let target_seg = 9u32;
        let mut del = RoaringBitmap::new();
        let mut naive = Vec::with_capacity(row_count as usize);
        let mut rank = 0u32;
        for l in 0..row_count {
            if rng.random_bool(0.45) {
                del.insert(l);
                naive.push(None);
            } else {
                naive.push(Some(RowId::new(target_seg, base + rank)));
                rank += 1;
            }
        }
        let mut m0 = SegmentMeta::new_open(0);
        m0.state = SegmentState::Forwarding;
        m0.row_count = row_count;
        m0.deleted_count = del.len() as u32;
        m0.forward_target = target_seg;
        m0.forward_base = base;

        let mut mt = SegmentMeta::new_open(1);
        mt.state = SegmentState::Sealed;
        mt.row_count = base + rank + 5;

        let mut entries = BTreeMap::new();
        entries.insert(
            0,
            SegmentEntry {
                meta: m0,
                deleted: (!del.is_empty()).then(|| Arc::new(del)),
            },
        );
        entries.insert(
            target_seg,
            SegmentEntry {
                meta: mt,
                deleted: None,
            },
        );
        let snap = SegmentTableSnapshot::from_entries(entries);
        for (l, expected) in naive.into_iter().enumerate() {
            assert_eq!(snap.resolve_row(RowId::new(0, l as u32)), expected);
        }
    }
}

#[test]
fn compaction_churn_forwarding_queries_and_physical_orphan_removal() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), full_config(16)).unwrap();

    // Insert 70 docs across 4 sealed segments + 1 small open segment.
    // Give doc 17 a unique rare term "quasar_unique" (inline_block FTS term!) in segment 1.
    let docs: Vec<Document> = (1..=70)
        .map(|i| {
            let body = if i == 17 {
                "quasar_unique vector index"
            } else if i % 2 == 0 {
                "vector storage engine"
            } else {
                "graph search retrieval"
            };
            make_doc(
                i,
                if i % 3 == 0 { "alpha" } else { "beta" },
                (i * 10) as i64,
                body,
                Some((i % 7) as f64 + 1.0),
            )
        })
        .collect();
    col.upsert_batch(&docs).unwrap();
    col.seal().unwrap();

    // Delete ~40% of docs across sealed segments so every sealed segment becomes eligible.
    let to_delete: Vec<u64> = (1..=70).filter(|i| i % 5 == 0 || i % 7 == 0).collect();
    col.delete_batch(&to_delete).unwrap();
    // Patch doc 18 (moves to open seg) and update doc 20->detached-only
    col.patch(
        18,
        DocumentPatch::new().with_attribute("cat", AttributeValue::String("gamma".into())),
    )
    .unwrap();
    col.seal().unwrap();

    let snapshot_queries = |c: &Collection| {
        let all_cats = c
            .resolve_filter_doc_ids(&Filter::Eq("cat".into(), ScalarValue::String("alpha".into())))
            .unwrap();
        let high_pop = c
            .resolve_filter_doc_ids(&Filter::Gte("pop".into(), ScalarValue::Float(5.0)))
            .unwrap();
        let rare_fts = c
            .query(
                &RankBy::Bm25 {
                    attr: "body".into(),
                    query: TokenQuery::Text("quasar_unique".into()),
                    params: None,
                },
                None,
                5,
            )
            .unwrap();
        let common_fts = c
            .query(
                &RankBy::Sum(vec![
                    RankBy::Bm25 {
                        attr: "body".into(),
                        query: TokenQuery::Text("vector engine".into()),
                        params: None,
                    },
                    RankBy::Attribute {
                        attr: "pop".into(),
                        missing: 0.0,
                    },
                ]),
                None,
                10,
            )
            .unwrap();
        let ordered = c
            .order_by(
                &[OrderBySpec {
                    attr: "score".into(),
                    direction: SortDirection::Desc,
                    nulls: NullsOrder::Last,
                }],
                None,
                8,
            )
            .unwrap();
        let facets = c
            .facet(
                None,
                &[
                    FacetSpec::terms("cat"),
                    FacetSpec::stats("pop").with_alias("pop_s"),
                ],
            )
            .unwrap();
        let vec_hits = c
            .search(
                &[1.7, 2.0, 0.5, 1.0],
                SearchOptions::new(6, 16),
                Some(&Filter::IsNotNull("pop".into())),
            )
            .unwrap();
        let doc17 = c.get(17).unwrap();
        (
            all_cats,
            high_pop,
            rare_fts,
            common_fts,
            ordered,
            facets,
            vec_hits,
            doc17,
        )
    };

    let before = snapshot_queries(&col);
    assert_eq!(before.2.iter().map(|r| r.id).collect::<Vec<_>>(), vec![17]);

    // Pause right after Step B (Forwarding, BEFORE Step C ANN remap / Step D Retire)
    let err = col
        .compact_with_hook(CompactionOptions::default(), &mut |stage| {
            if matches!(stage, CompactionHookStage::AfterStepBCommit(_)) {
                return Err(TreeError::InvalidFilter("pause_after_step_b".into()));
            }
            Ok(())
        })
        .unwrap_err();
    assert!(err.to_string().contains("pause_after_step_b"));

    // Forwarding segments must exist now, AND queries/ANN/invariants must still hold!
    assert!(
        col.segment_snapshot()
            .segments
            .values()
            .any(|e| e.meta.state == SegmentState::Forwarding)
    );
    col.verify_ann_invariants().unwrap();
    let during_forwarding = snapshot_queries(&col);
    assert_eq!(during_forwarding, before);

    // Complete Step C & D via vacuum_ann (without calling compact again!), then check retirement.
    let old_forwarding: Vec<u32> = col
        .segment_snapshot()
        .segments
        .iter()
        .filter_map(|(&s, e)| (e.meta.state == SegmentState::Forwarding).then_some(s))
        .collect();
    assert!(!old_forwarding.is_empty());

    col.vacuum_ann(VacuumBudget::unlimited()).unwrap();
    let snap_after = col.segment_snapshot();
    for &fseg in &old_forwarding {
        assert!(
            !snap_after.segments.contains_key(&fseg),
            "forwarded segment {fseg} must be retired after full vacuum pass"
        );
        assert!(col.storage().dropped_segments_snapshot().contains(fseg));
    }

    // No ANN leaf may still reference a dropped or forwarding segment!
    let leaves = col.storage().all_leaf_nodes().unwrap();
    for (_, leaf) in &leaves {
        for slot in 0..leaf.slot_cap {
            if leaf.is_live(slot) {
                let row = leaf.row_id(slot);
                assert!(!col.storage().dropped_segments_snapshot().contains(row.seg()));
                assert!(snap_after.is_visible_segment(row.seg()));
            }
        }
    }

    // Trigger RocksDB LSM compaction -> orphaned keys for retired segments must vanish physically!
    col.compact_fts_stats().unwrap();
    col.storage().compact_range_all().unwrap();

    for &fseg in &old_forwarding {
        assert_eq!(
            col.storage()
                .count_keys_in_cf(Some(CF_ROWS), |k| k.len() >= 4
                    && u32::from_be_bytes(k[0..4].try_into().unwrap()) == fseg)
                .unwrap(),
            0
        );
        assert_eq!(
            col.storage()
                .count_keys_in_cf(Some(CF_ROW_DOC_IDS), |k| k.len() >= 4
                    && u32::from_be_bytes(k[0..4].try_into().unwrap()) == fseg)
                .unwrap(),
            0
        );
        assert_eq!(
            col.storage()
                .count_keys_in_cf(Some(CF_VECTORS), |k| k.len() >= 4
                    && u32::from_be_bytes(k[0..4].try_into().unwrap()) == fseg)
                .unwrap(),
            0
        );
        assert_eq!(
            col.storage()
                .count_keys_in_cf(Some(CF_COLUMNS), |k| k.len() >= 4
                    && u32::from_be_bytes(k[0..4].try_into().unwrap()) == fseg)
                .unwrap(),
            0
        );
        assert_eq!(
            col.storage()
                .count_keys_in_cf(Some(CF_ATTR_POSTINGS), |k| {
                    split_posting_key(k).is_ok_and(|(_, s)| s == fseg)
                })
                .unwrap(),
            0
        );
        assert_eq!(
            col.storage()
                .count_keys_in_cf(Some(CF_DETACHED), |k| k.len() == 14
                    && u32::from_be_bytes(k[6..10].try_into().unwrap()) == fseg)
                .unwrap(),
            0
        );
        assert_eq!(
            col.storage()
                .count_keys_in_cf(Some(CF_DETACHED_POSTINGS), |k| {
                    split_detached_posting_key(k).is_ok_and(|(_, s)| s == fseg)
                })
                .unwrap(),
            0
        );
        for fts_cf in [CF_FTS_SKIPS, CF_FTS_POSTINGS, CF_FTS_POSITIONS] {
            assert_eq!(
                col.storage()
                    .count_keys_in_cf(Some(fts_cf), |k| k.len() >= 8
                        && u32::from_be_bytes(k[4..8].try_into().unwrap()) == fseg)
                    .unwrap(),
                0
            );
        }
        assert_eq!(
            col.storage()
                .count_keys_in_cf(None, |k| match k.first().copied() {
                    Some(FTS_DOC_LEN_TAG) if k.len() >= 7 => {
                        u32::from_be_bytes(k[3..7].try_into().unwrap()) == fseg
                    }
                    Some(ZONE_TAG) if k.len() >= 5 => {
                        u32::from_be_bytes(k[1..5].try_into().unwrap()) == fseg
                    }
                    _ => false,
                })
                .unwrap(),
            0
        );
    }

    // Reopen and verify full query & invariant parity!
    drop(col);
    let col = Collection::open(dir.path(), full_config(16)).unwrap();
    col.verify_ann_invariants().unwrap();
    let after = snapshot_queries(&col);
    assert_eq!(after, before);
}

#[test]
fn compaction_crash_recovery_at_all_step_boundaries_and_pending_detached_guard() {
    let dir = TempDir::new().unwrap();
    {
        let mut col = Collection::open(dir.path(), full_config(16)).unwrap();
        let docs: Vec<Document> = (1..=48)
            .map(|i| make_doc(i, "c", i as i64, "crash test doc", Some(i as f64)))
            .collect();
        col.upsert_batch(&docs).unwrap();
        col.seal().unwrap();
        col.delete_batch(&(1..=24).collect::<Vec<_>>()).unwrap();

        // Guard: pending_gen != 0 must reject compaction.
        let pending_job = col
            .detached_ingest()
            .column("pop")
            .from_pairs([(21, AttributeValue::Float(999.0))])
            .build()
            .unwrap();
        let err = col.compact(CompactionOptions::default()).unwrap_err();
        assert!(matches!(err, TreeError::InvalidConfig(_)));
        col.publish_detached(pending_job).unwrap();

        // Crash 1: abort right after Step A (Building target segment)
        let mut building_seg = None;
        let _ = col.compact_with_hook(CompactionOptions::default().with_max_groups(1), &mut |s| {
            if let CompactionHookStage::AfterStepABuild(t) = s {
                building_seg = Some(t);
                return Err(TreeError::InvalidFilter("crash_step_a".into()));
            }
            Ok(())
        });
        assert!(building_seg.is_some());
    }

    // Reopen after Crash 1: Building segment must be recovered & dropped automatically.
    let retired_after_c2;
    {
        let mut col = Collection::open(dir.path(), full_config(16)).unwrap();
        assert!(
            !col.segment_snapshot()
                .segments
                .values()
                .any(|e| e.meta.state == SegmentState::Building)
        );
        for id in 25..=48u64 {
            assert!(col.get(id).unwrap().is_some());
        }

        // Crash 2: abort right after Step B (Commit Forwarding)
        let _ = col.compact_with_hook(CompactionOptions::default().with_max_groups(1), &mut |s| {
            if matches!(s, CompactionHookStage::AfterStepBCommit(_)) {
                return Err(TreeError::InvalidFilter("crash_step_b".into()));
            }
            Ok(())
        });
        assert!(
            col.segment_snapshot()
                .segments
                .values()
                .any(|e| e.meta.state == SegmentState::Forwarding)
        );
    }

    // Reopen after Crash 2: run maintenance() -> finishes vacuum + compaction + retire!
    {
        let mut col = Collection::open(dir.path(), full_config(16)).unwrap();
        for id in 25..=48u64 {
            assert!(col.get(id).unwrap().is_some());
        }
        let mstats = col.maintenance(MaintenanceOptions::default()).unwrap();
        retired_after_c2 = mstats.compaction.segments_retired;
        assert!(retired_after_c2 > 0 || !col.segment_snapshot().segments.values().any(|e| e.meta.state == SegmentState::Forwarding));
        col.verify_ann_invariants().unwrap();
        assert_eq!(
            col.resolve_filter_doc_ids(&Filter::Eq("cat".into(), ScalarValue::String("c".into())))
                .unwrap()
                .len(),
            24
        );
    }
}
