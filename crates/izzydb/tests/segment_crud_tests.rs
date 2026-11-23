use izzydb::{
    AttributeType, AttributeValue, Attributes, Collection, CollectionConfig, DistanceMetric,
    Document, DocumentPatch, FacetSpec, Filter, FullTextSearchConfig, HybridSearchOptions,
    OrderBySpec, RankBy, SearchOptions, SortDirection, TreeConfig, TreeError, VectorFacetScope,
};
use izzydb_storage::codec::SegmentState;
use tempfile::TempDir;

fn small_collection_config(dim: usize, metric: DistanceMetric, seal_rows: u32) -> CollectionConfig {
    let tree = TreeConfig::new(dim, metric);
    CollectionConfig::new(tree)
        .with_seal_rows(seal_rows)
        .with_attribute("category", AttributeType::String, true)
        .with_attribute("score", AttributeType::Float, true)
        .with_attribute("rank", AttributeType::Int, true)
        .with_attribute("count", AttributeType::Uint, true)
        .with_attribute("active", AttributeType::Bool, true)
        .with_attribute("tags", AttributeType::StringArray, true)
        .with_attribute("note", AttributeType::String, false)
}

#[test]
fn test_schema_validation_and_reopen_reconcile_crud() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(8, DistanceMetric::L2Squared, 100);

    {
        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();

        let mut bad_attrs = Attributes::new();
        bad_attrs.insert("unknown".into(), "x".into());
        assert!(matches!(
            col.upsert(1, Some(&[0.0; 8]), bad_attrs),
            Err(TreeError::InvalidAttribute { .. })
        ));

        let mut bad_type = Attributes::new();
        bad_type.insert("rank".into(), "not_an_int".into());
        assert!(matches!(
            col.upsert(1, Some(&[0.0; 8]), bad_type),
            Err(TreeError::InvalidAttribute { .. })
        ));

        let mut attrs = Attributes::new();
        attrs.insert("note".into(), "secret_payload".into());
        col.upsert(1, Some(&[1.0; 8]), attrs).unwrap();
        assert_eq!(
            col.get(1).unwrap().unwrap().attributes.get("note"),
            Some(&AttributeValue::String("secret_payload".into()))
        );
    }

    let mut bad_cfg = cfg.clone();
    bad_cfg.attributes[0].attr_type = AttributeType::Int;
    assert!(matches!(
        Collection::open(dir.path(), bad_cfg),
        Err(TreeError::ConfigMismatch {
            field: "attribute_type",
            ..
        })
    ));

    let mut bad_filt_cfg = cfg.clone();
    bad_filt_cfg.attributes[0].filterable = false;
    assert!(matches!(
        Collection::open(dir.path(), bad_filt_cfg),
        Err(TreeError::ConfigMismatch {
            field: "attribute_filterable",
            ..
        })
    ));

    let extended_cfg = cfg.with_attribute("extra", AttributeType::Bool, true);
    let col = Collection::open(dir.path(), extended_cfg).unwrap();
    assert!(col.schema().get_by_name("extra").is_some());
    assert_eq!(col.len(), 1);
}

#[test]
fn test_insert_get_overwrite_patch_clear_delete_reinsert() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(4, DistanceMetric::L2Squared, 64);

    {
        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();
        assert!(col.is_empty());

        let mut attrs1 = Attributes::new();
        attrs1.insert("category".into(), "books".into());
        attrs1.insert("rank".into(), 10i64.into());
        attrs1.insert("note".into(), "v1".into());
        col.upsert(42, Some(&[1.0, 2.0, 3.0, 4.0]), attrs1.clone())
            .unwrap();
        assert_eq!(col.len(), 1);

        let doc = col.get(42).unwrap().unwrap();
        assert_eq!(doc.id, 42);
        assert_eq!(doc.embedding, Some(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(doc.attributes, attrs1);

        // Overwrite with updated embedding and attributes
        let mut attrs2 = Attributes::new();
        attrs2.insert("category".into(), "games".into());
        attrs2.insert("rank".into(), 20i64.into());
        col.upsert(42, Some(&[2.0, 3.0, 4.0, 5.0]), attrs2.clone())
            .unwrap();
        assert_eq!(col.len(), 1);
        let doc2 = col.get(42).unwrap().unwrap();
        assert_eq!(doc2.embedding, Some(vec![2.0, 3.0, 4.0, 5.0]));
        assert_eq!(doc2.attributes, attrs2);

        // Patch: update rank, clear category, preserve embedding
        col.patch(
            42,
            DocumentPatch::new()
                .with_attribute("rank", 99i64.into())
                .clear_attribute("category"),
        )
        .unwrap();
        let doc3 = col.get(42).unwrap().unwrap();
        assert_eq!(doc3.embedding, Some(vec![2.0, 3.0, 4.0, 5.0]));
        assert_eq!(doc3.attributes.get("rank"), Some(&AttributeValue::Int(99)));
        assert_eq!(doc3.attributes.get("category"), None);

        // Missing patch returns DocumentNotFound
        assert!(matches!(
            col.patch(999, DocumentPatch::new().with_attribute("rank", 1i64.into())),
            Err(TreeError::DocumentNotFound(999))
        ));

        // patch_batch atomicity
        let err = col
            .patch_batch(&[
                (42, DocumentPatch::new().with_attribute("rank", 777i64.into())),
                (999, DocumentPatch::new().with_attribute("rank", 888i64.into())),
            ])
            .unwrap_err();
        assert!(matches!(err, TreeError::DocumentNotFound(999)));
        assert_eq!(
            col.get(42).unwrap().unwrap().attributes.get("rank"),
            Some(&AttributeValue::Int(99))
        );

        // Delete and re-insert
        assert!(col.delete(42).unwrap());
        assert!(!col.delete(42).unwrap());
        assert!(col.get(42).unwrap().is_none());
        assert_eq!(col.len(), 0);

        let mut attrs4 = Attributes::new();
        attrs4.insert("category".into(), "reborn".into());
        col.upsert(42, None, attrs4.clone()).unwrap();
        let doc4 = col.get(42).unwrap().unwrap();
        assert_eq!(doc4.embedding, None);
        assert_eq!(doc4.attributes, attrs4);
        assert_eq!(col.len(), 1);
    }

    // Reopen persistence
    let col = Collection::open(dir.path(), cfg).unwrap();
    assert_eq!(col.len(), 1);
    let doc = col.get(42).unwrap().unwrap();
    assert_eq!(doc.embedding, None);
    assert_eq!(
        doc.attributes.get("category"),
        Some(&AttributeValue::String("reborn".into()))
    );
}

#[test]
fn test_segment_rollover_updates_and_tombstones() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(4, DistanceMetric::L2Squared, 4);

    {
        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();
        for id in 0..10u64 {
            let mut attrs = Attributes::new();
            attrs.insert("rank".into(), (id as i64).into());
            col.upsert(id, Some(&[id as f32, 0.0, 0.0, 0.0]), attrs)
                .unwrap();
        }
        assert_eq!(col.len(), 10);

        let snap = col.segment_snapshot();
        assert_eq!(snap.segments.len(), 3);
        assert_eq!(snap.get(0).unwrap().meta.state, SegmentState::Sealed);
        assert_eq!(snap.get(0).unwrap().meta.row_count, 4);
        assert_eq!(snap.get(1).unwrap().meta.state, SegmentState::Sealed);
        assert_eq!(snap.get(1).unwrap().meta.row_count, 4);
        assert_eq!(snap.get(2).unwrap().meta.state, SegmentState::Open);
        assert_eq!(snap.get(2).unwrap().meta.row_count, 2);

        // Update doc 1 (in sealed seg 0) -> tombstones (0, 1), appends to open seg 2
        col.patch(1, DocumentPatch::new().with_attribute("rank", 100i64.into()))
            .unwrap();
        // Delete doc 5 (in sealed seg 1)
        assert!(col.delete(5).unwrap());

        let snap2 = col.segment_snapshot();
        assert_eq!(snap2.get(0).unwrap().meta.deleted_count, 1);
        assert!(snap2.get(0).unwrap().deleted.as_ref().unwrap().contains(1));
        assert_eq!(snap2.get(1).unwrap().meta.deleted_count, 1);
        assert!(snap2.get(1).unwrap().deleted.as_ref().unwrap().contains(1)); // doc 5 is local 1 of seg 1
        assert_eq!(snap2.get(2).unwrap().meta.row_count, 3);
        assert_eq!(col.len(), 9);
    }

    // Reopen restores manifest, segment table, and id map accurately
    let col = Collection::open(dir.path(), cfg).unwrap();
    assert_eq!(col.len(), 9);
    let snap = col.segment_snapshot();
    assert_eq!(snap.get(0).unwrap().meta.deleted_count, 1);
    assert_eq!(snap.get(1).unwrap().meta.deleted_count, 1);
    assert_eq!(snap.get(2).unwrap().meta.row_count, 3);
    assert_eq!(
        col.get(1).unwrap().unwrap().attributes.get("rank"),
        Some(&AttributeValue::Int(100))
    );
    assert!(col.get(5).unwrap().is_none());
}

#[test]
fn test_upsert_batch_spanning_seal_boundaries_and_get_batch() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(4, DistanceMetric::Cosine, 3);
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    let docs: Vec<Document> = (0..10u64)
        .map(|i| {
            let mut attributes = Attributes::new();
            attributes.insert("rank".into(), (i as i64).into());
            Document {
                id: i * 10,
                embedding: if i % 2 == 0 {
                    Some(vec![1.0, i as f32, 0.0, 0.0])
                } else {
                    None
                },
                attributes,
            }
        })
        .collect();

    col.upsert_batch(&docs).unwrap();
    assert_eq!(col.len(), 10);

    let snap = col.segment_snapshot();
    // 10 rows with seal_rows=3 => seg 0 (3, sealed), seg 1 (3, sealed), seg 2 (3, sealed), seg 3 (1, open)
    assert_eq!(snap.segments.len(), 4);
    assert_eq!(snap.get(0).unwrap().meta.state, SegmentState::Sealed);
    assert_eq!(snap.get(1).unwrap().meta.state, SegmentState::Sealed);
    assert_eq!(snap.get(2).unwrap().meta.state, SegmentState::Sealed);
    assert_eq!(snap.get(3).unwrap().meta.state, SegmentState::Open);
    assert_eq!(snap.get(3).unwrap().meta.row_count, 1);

    // Force seal open segment
    assert_eq!(col.seal().unwrap(), Some(3));
    assert_eq!(col.seal().unwrap(), None);
    assert_eq!(
        col.segment_snapshot().get(3).unwrap().meta.state,
        SegmentState::Sealed
    );

    let batch = col.get_batch(&[90, 999, 0, 40]).unwrap();
    assert_eq!(batch[0].as_ref().unwrap().id, 90);
    assert!(batch[1].is_none());
    assert_eq!(batch[2].as_ref().unwrap().id, 0);
    assert_eq!(batch[3].as_ref().unwrap().id, 40);
    assert!(batch[0].as_ref().unwrap().embedding.is_none());
    assert!(batch[2].as_ref().unwrap().embedding.is_some());
}

#[test]
fn test_idempotent_upsert_and_patch_do_not_write() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(4, DistanceMetric::L2Squared, 64);
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    let mut attrs = Attributes::new();
    attrs.insert("category".into(), "a".into());
    attrs.insert("rank".into(), 5i64.into());
    col.upsert(1, Some(&[1.0, 2.0, 3.0, 4.0]), attrs.clone())
        .unwrap();

    let seq_before = col.storage().latest_sequence_number();
    col.upsert(1, Some(&[1.0, 2.0, 3.0, 4.0]), attrs).unwrap();
    assert_eq!(col.storage().latest_sequence_number(), seq_before);

    col.patch(
        1,
        DocumentPatch::new()
            .with_embedding(vec![1.0, 2.0, 3.0, 4.0])
            .with_attribute("rank", 5i64.into()),
    )
    .unwrap();
    assert_eq!(col.storage().latest_sequence_number(), seq_before);
    assert_eq!(col.segment_snapshot().get(0).unwrap().meta.row_count, 1);
}

#[test]
fn test_vectorless_collection_and_unsupported_stubs() {
    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::without_vectors()
        .with_seal_rows(8)
        .with_attribute("title", AttributeType::String, true)
        .with_fts_attribute("body", AttributeType::String, false, FullTextSearchConfig::default());
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    let mut attrs = Attributes::new();
    attrs.insert("title".into(), "hello".into());
    attrs.insert("body".into(), "hello vectorless world".into());
    col.upsert(1, None, attrs.clone()).unwrap();
    assert!(matches!(
        col.upsert(2, Some(&[1.0, 2.0]), attrs),
        Err(TreeError::InvalidConfig(_))
    ));
    assert_eq!(col.get(1).unwrap().unwrap().embedding, None);

    // Vector search on a vectorless collection returns InvalidConfig
    assert!(matches!(
        col.search(&[0.0], SearchOptions::new(5, 4), None),
        Err(TreeError::InvalidConfig(_))
    ));
    assert!(matches!(
        col.search(&[0.0], SearchOptions::new(5, 4), Some(&Filter::Eq("title".into(), "hello".into()))),
        Err(TreeError::InvalidConfig(_))
    ));
    // Attribute filters, FTS, order_by, and facets work on vectorless collections
    assert_eq!(
        col.resolve_filter_doc_ids(&Filter::Eq("title".into(), "hello".into()))
            .unwrap(),
        vec![1]
    );
    let bm25_hits = col.query_bm25(&RankBy::bm25("body", "hello"), None, 5).unwrap();
    assert_eq!(bm25_hits.len(), 1);
    assert_eq!(bm25_hits[0].id, 1);
    let ord_hits = col.order_by(&[OrderBySpec::asc("title")], None, 5).unwrap();
    assert_eq!(ord_hits.len(), 1);
    assert_eq!(ord_hits[0].id, 1);
    let facet_res = col.facet(None, &[FacetSpec::terms("title")]).unwrap();
    assert_eq!(facet_res.terms("title").unwrap().buckets.len(), 1);
    assert!(matches!(
        col.hybrid_search(&[0.0], &RankBy::bm25("body", "hello"), HybridSearchOptions::new(5, SearchOptions::new(5, 4)), None),
        Err(TreeError::InvalidConfig(_))
    ));
    assert!(matches!(
        col.search_with_facets(&[0.0], SearchOptions::new(5, 4), None, &[], VectorFacetScope::TopK),
        Err(TreeError::InvalidConfig(_))
    ));
    col.verify_invariants().unwrap();
    let _ = SortDirection::Asc;
}

#[test]
fn test_crash_safety_smoke_drop_between_batches() {
    let dir = TempDir::new().unwrap();
    let cfg = small_collection_config(4, DistanceMetric::L2Squared, 4);

    {
        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();
        let batch1: Vec<Document> = (0..5)
            .map(|i| Document {
                id: i,
                embedding: Some(vec![i as f32; 4]),
                attributes: Attributes::new(),
            })
            .collect();
        col.upsert_batch(&batch1).unwrap();
        // Drop without explicit seal/flush
    }

    let col = Collection::open(dir.path(), cfg).unwrap();
    assert_eq!(col.len(), 5);
    let snap = col.segment_snapshot();
    assert_eq!(snap.get(0).unwrap().meta.state, SegmentState::Sealed);
    assert_eq!(snap.get(1).unwrap().meta.state, SegmentState::Open);
    assert_eq!(snap.get(1).unwrap().meta.row_count, 1);
}
