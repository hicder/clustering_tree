use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{
    BooleanBuilder, FixedSizeListBuilder, Float32Builder, Float64Builder, Int64Builder,
    ListBuilder, StringBuilder, UInt64Builder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

use izzydb::metric::{l2_squared, prepare_vector};
use izzydb::{
    AttributeType, AttributeValue, Attributes, Collection, CollectionConfig,
    DetachedDuplicatePolicy, DetachedWriteMode, DistanceMetric, DocId, DocumentPatch, FacetRange,
    FacetSpec, Filter, FullTextSearchConfig, Language, OrderBySpec, RankBy, SearchOptions,
    TokenQuery, Tokenizer, TreeConfig, TreeError,
};
use izzydb_bulk::{
    BulkAppendHookStage, BulkImportConfig, DuplicatePolicy, SstBundle, build_ssts, bulk_append,
    bulk_append_with_hook, bulk_import, ingest_ssts, ingest_ssts_with_hook,
    scan_detached_from_parquet,
};

fn small_tree_cfg(dim: usize, metric: DistanceMetric) -> TreeConfig {
    TreeConfig {
        dim,
        metric,
        min_internal_fanout: 2,
        max_internal_fanout: 5,
        min_leaf_size: 4,
        max_leaf_size: 12,
        insert_beam_width: 4,
        lire_neighbor_count: 4,
        enable_lire: true,
        centroid_drift_eps: 1e-4,
    }
}

fn full_collection_config(dim: usize, metric: DistanceMetric) -> CollectionConfig {
    let fts_en = FullTextSearchConfig::default()
        .with_language(Language::English)
        .with_stemming(true)
        .with_remove_stopwords(true)
        .with_ascii_folding(true);
    CollectionConfig::new(small_tree_cfg(dim, metric))
        .with_attribute("category", AttributeType::String, true)
        .with_attribute("score", AttributeType::Float, true)
        .with_attribute("rank", AttributeType::Int, true)
        .with_attribute("count", AttributeType::Uint, true)
        .with_attribute("active", AttributeType::Bool, true)
        .with_attribute("tags", AttributeType::StringArray, true)
        .with_attribute("nums", AttributeType::IntArray, true)
        .with_attribute("uids", AttributeType::UintArray, true)
        .with_attribute("note", AttributeType::String, false)
        .with_fts_attribute("title", AttributeType::String, false, fts_en.clone())
        .with_fts_attribute("body", AttributeType::String, false, fts_en)
        .with_fts_attribute(
            "kw",
            AttributeType::StringArray,
            false,
            FullTextSearchConfig::default().with_tokenizer(Tokenizer::PreTokenizedArray),
        )
}

#[derive(Clone)]
struct TestRow {
    id: DocId,
    vec: Vec<f32>,
    attrs: Attributes,
}

fn gen_test_rows(n: usize, dim: usize, seed: u64) -> Vec<TestRow> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let categories = ["books", "electronics", "clothing", "home", "sports"];
    let vocab = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "running", "trail", "waterproof", "lightweight", "shoe", "jacket", "battery",
        "camera", "sensor", "mountain",
    ];
    let tags_pool = ["sale", "new", "featured", "clearance", "eco"];

    (0..n)
        .map(|i| {
            let id = (i as u64 + 1) * 3;
            let vec: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
            let mut attrs = Attributes::new();

            if i % 11 != 0 {
                attrs.insert(
                    "category".into(),
                    categories[rng.random_range(0..categories.len())].into(),
                );
            }
            if i % 9 != 0 {
                attrs.insert(
                    "score".into(),
                    ((rng.random_range(0..1000) as f64) / 10.0).into(),
                );
            }
            if i % 7 != 0 {
                attrs.insert("rank".into(), rng.random_range(-50i64..50).into());
            }
            if i % 8 != 0 {
                attrs.insert("count".into(), rng.random_range(0u64..500).into());
            }
            if i % 6 != 0 {
                attrs.insert("active".into(), rng.random_bool(0.5).into());
            }
            if i % 5 != 0 {
                let k = rng.random_range(0..=3);
                let tags: Vec<String> = (0..k)
                    .map(|_| tags_pool[rng.random_range(0..tags_pool.len())].to_owned())
                    .collect();
                attrs.insert("tags".into(), tags.into());
            }
            if i % 10 != 0 {
                let k = rng.random_range(1..=3);
                let nums: Vec<i64> = (0..k).map(|_| rng.random_range(-10..10)).collect();
                attrs.insert("nums".into(), nums.into());
            }
            if i % 12 != 0 {
                let k = rng.random_range(1..=2);
                let uids: Vec<u64> = (0..k).map(|_| rng.random_range(1..100)).collect();
                attrs.insert("uids".into(), uids.into());
            }
            if i % 4 != 0 {
                attrs.insert("note".into(), format!("doc-{id}-note").into());
            }

            // FTS fields
            if i % 13 != 0 {
                let w1 = vocab[rng.random_range(0..vocab.len())];
                let w2 = vocab[rng.random_range(0..vocab.len())];
                let rare = if i % 97 == 0 { " xylophone" } else { "" };
                attrs.insert("title".into(), format!("{w1} {w2}{rare}").into());
            }
            if i % 15 != 0 {
                let len = rng.random_range(3..=10);
                let words: Vec<&str> = (0..len)
                    .map(|_| vocab[rng.random_range(0..vocab.len())])
                    .collect();
                attrs.insert("body".into(), words.join(" ").into());
            }
            if i % 17 != 0 {
                let k = rng.random_range(1..=3);
                let kws: Vec<String> = (0..k)
                    .map(|_| vocab[rng.random_range(0..vocab.len())].to_owned())
                    .collect();
                attrs.insert("kw".into(), kws.into());
            }

            TestRow { id, vec, attrs }
        })
        .collect()
}

fn write_full_parquet(
    path: &Path,
    rows: &[TestRow],
    dim: usize,
    row_group_size: usize,
    use_f64_vecs: bool,
    col_names: Option<(&str, &str)>,
) {
    let id_name = col_names.map_or("id", |c| c.0);
    let vec_name = col_names.map_or("vector", |c| c.1);

    let vec_field = if use_f64_vecs {
        Field::new(
            vec_name,
            DataType::List(Arc::new(Field::new("item", DataType::Float64, false))),
            false,
        )
    } else {
        Field::new(
            vec_name,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                dim as i32,
            ),
            false,
        )
    };

    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new(id_name, DataType::UInt64, false),
        vec_field,
        Field::new("category", DataType::Utf8, true),
        Field::new("score", DataType::Float64, true),
        Field::new("rank", DataType::Int64, true),
        Field::new("count", DataType::UInt64, true),
        Field::new("active", DataType::Boolean, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
            true,
        ),
        Field::new(
            "nums",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
            true,
        ),
        Field::new(
            "uids",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, false))),
            true,
        ),
        Field::new("note", DataType::Utf8, true),
        Field::new("title", DataType::Utf8, true),
        Field::new("body", DataType::Utf8, true),
        Field::new(
            "kw",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
            true,
        ),
        Field::new("ignored_extra", DataType::Utf8, true),
    ]));

    let props = WriterProperties::builder()
        .set_max_row_group_size(row_group_size.max(1))
        .build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();

    for chunk in rows.chunks(row_group_size.max(1)) {
        let mut id_b = UInt64Builder::new();
        let mut f32_b = FixedSizeListBuilder::new(Float32Builder::new(), dim as i32)
            .with_field(Field::new("item", DataType::Float32, false));
        let mut f64_b = ListBuilder::new(Float64Builder::new())
            .with_field(Field::new("item", DataType::Float64, false));

        let mut cat_b = StringBuilder::new();
        let mut score_b = Float64Builder::new();
        let mut rank_b = Int64Builder::new();
        let mut count_b = UInt64Builder::new();
        let mut active_b = BooleanBuilder::new();
        let mut tags_b = ListBuilder::new(StringBuilder::new())
            .with_field(Field::new("item", DataType::Utf8, false));
        let mut nums_b = ListBuilder::new(Int64Builder::new())
            .with_field(Field::new("item", DataType::Int64, false));
        let mut uids_b = ListBuilder::new(UInt64Builder::new())
            .with_field(Field::new("item", DataType::UInt64, false));
        let mut note_b = StringBuilder::new();
        let mut title_b = StringBuilder::new();
        let mut body_b = StringBuilder::new();
        let mut kw_b = ListBuilder::new(StringBuilder::new())
            .with_field(Field::new("item", DataType::Utf8, false));
        let mut extra_b = StringBuilder::new();

        for r in chunk {
            id_b.append_value(r.id);
            if use_f64_vecs {
                for &v in &r.vec {
                    f64_b.values().append_value(v as f64);
                }
                f64_b.append(true);
            } else {
                for &v in &r.vec {
                    f32_b.values().append_value(v);
                }
                f32_b.append(true);
            }

            match r.attrs.get("category") {
                Some(AttributeValue::String(s)) => cat_b.append_value(s),
                _ => cat_b.append_null(),
            }
            match r.attrs.get("score") {
                Some(AttributeValue::Float(v)) => score_b.append_value(*v),
                _ => score_b.append_null(),
            }
            match r.attrs.get("rank") {
                Some(AttributeValue::Int(v)) => rank_b.append_value(*v),
                _ => rank_b.append_null(),
            }
            match r.attrs.get("count") {
                Some(AttributeValue::Uint(v)) => count_b.append_value(*v),
                _ => count_b.append_null(),
            }
            match r.attrs.get("active") {
                Some(AttributeValue::Bool(v)) => active_b.append_value(*v),
                _ => active_b.append_null(),
            }
            match r.attrs.get("tags") {
                Some(AttributeValue::StringArray(arr)) => {
                    for s in arr {
                        tags_b.values().append_value(s);
                    }
                    tags_b.append(true);
                }
                _ => tags_b.append_null(),
            }
            match r.attrs.get("nums") {
                Some(AttributeValue::IntArray(arr)) => {
                    for &v in arr {
                        nums_b.values().append_value(v);
                    }
                    nums_b.append(true);
                }
                _ => nums_b.append_null(),
            }
            match r.attrs.get("uids") {
                Some(AttributeValue::UintArray(arr)) => {
                    for &v in arr {
                        uids_b.values().append_value(v);
                    }
                    uids_b.append(true);
                }
                _ => uids_b.append_null(),
            }
            match r.attrs.get("note") {
                Some(AttributeValue::String(s)) => note_b.append_value(s),
                _ => note_b.append_null(),
            }
            match r.attrs.get("title") {
                Some(AttributeValue::String(s)) => title_b.append_value(s),
                _ => title_b.append_null(),
            }
            match r.attrs.get("body") {
                Some(AttributeValue::String(s)) => body_b.append_value(s),
                _ => body_b.append_null(),
            }
            match r.attrs.get("kw") {
                Some(AttributeValue::StringArray(arr)) => {
                    for s in arr {
                        kw_b.values().append_value(s);
                    }
                    kw_b.append(true);
                }
                _ => kw_b.append_null(),
            }
            extra_b.append_value("ignored");
        }

        let vec_col: ArrayRef = if use_f64_vecs {
            Arc::new(f64_b.finish())
        } else {
            Arc::new(f32_b.finish())
        };

        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(id_b.finish()),
                vec_col,
                Arc::new(cat_b.finish()),
                Arc::new(score_b.finish()),
                Arc::new(rank_b.finish()),
                Arc::new(count_b.finish()),
                Arc::new(active_b.finish()),
                Arc::new(tags_b.finish()),
                Arc::new(nums_b.finish()),
                Arc::new(uids_b.finish()),
                Arc::new(note_b.finish()),
                Arc::new(title_b.finish()),
                Arc::new(body_b.finish()),
                Arc::new(kw_b.finish()),
                Arc::new(extra_b.finish()),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
}

#[test]
fn test_vectors_only_recall_parity_and_invariants() {
    let dir = TempDir::new().unwrap();
    let dim = 32;
    let n = 1500;
    let num_clusters = 15;
    let mut rng = SmallRng::seed_from_u64(123);
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect())
        .collect();

    let rows: Vec<TestRow> = (0..n)
        .map(|i| {
            let c = &centers[i % num_clusters];
            let v: Vec<f32> = c
                .iter()
                .map(|&x| x + rng.random_range(-0.15f32..0.15f32))
                .collect();
            TestRow {
                id: i as u64,
                vec: v,
                attrs: Attributes::new(),
            }
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let c = &centers[i % num_clusters];
            c.iter()
                .map(|&x| x + rng.random_range(-0.15f32..0.15f32))
                .collect()
        })
        .collect();

    let pq_path = dir.path().join("vecs.parquet");
    write_full_parquet(&pq_path, &rows, dim, 300, false, None);

    let cfg = CollectionConfig::new(TreeConfig {
        dim,
        metric: DistanceMetric::L2Squared,
        min_internal_fanout: 4,
        max_internal_fanout: 16,
        min_leaf_size: 8,
        max_leaf_size: 32,
        insert_beam_width: 4,
        lire_neighbor_count: 8,
        enable_lire: true,
        centroid_drift_eps: 1e-3,
    });

    // Build oracle via upsert.
    let oracle_dir = dir.path().join("oracle");
    let mut oracle = Collection::open(&oracle_dir, cfg.clone()).unwrap();
    for r in &rows {
        oracle.upsert(r.id, Some(&r.vec), Attributes::new()).unwrap();
    }

    // Build bulk-imported collection.
    let bulk_dir = dir.path().join("bulk");
    let work_dir = dir.path().join("work");
    let bcfg = BulkImportConfig::new(cfg.clone(), "id", "vector", &work_dir).with_verify_after(true);
    let report = bulk_import(&bulk_dir, &[pq_path], bcfg, None).unwrap();
    assert_eq!(report.docs_imported, n as u64);
    assert_eq!(report.rows_read, n as u64);
    assert_eq!(report.duplicates_dropped, 0);

    let bulk_col = Collection::open(&bulk_dir, cfg).unwrap();
    bulk_col.verify_invariants().unwrap();

    // Compare recall@10 against ground truth.
    let mut oracle_hits = 0usize;
    let mut bulk_hits = 0usize;
    for q in &queries {
        let qp = prepare_vector(q, dim, DistanceMetric::L2Squared).unwrap();
        let mut gt: Vec<(DocId, f32)> = rows
            .iter()
            .map(|r| (r.id, l2_squared(&qp, &r.vec)))
            .collect();
        gt.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt_set: HashSet<DocId> = gt.iter().take(10).map(|&(id, _)| id).collect();

        let opts = SearchOptions::new(10, 8).with_rerank(100);
        for res in oracle.search(q, opts, None).unwrap() {
            if gt_set.contains(&res.id) {
                oracle_hits += 1;
            }
        }
        for res in bulk_col.search(q, opts, None).unwrap() {
            if gt_set.contains(&res.id) {
                bulk_hits += 1;
            }
        }
    }
    let oracle_recall = oracle_hits as f64 / (queries.len() * 10) as f64;
    let bulk_recall = bulk_hits as f64 / (queries.len() * 10) as f64;
    assert!(
        bulk_recall >= oracle_recall - 0.02 && bulk_recall > 0.90,
        "bulk_recall={bulk_recall:.4} vs oracle_recall={oracle_recall:.4}"
    );
}

#[test]
fn test_full_parity_attributes_facets_fts_and_parquet_edge_cases() {
    let dir = TempDir::new().unwrap();
    let dim = 16;
    let rows = gen_test_rows(450, dim, 777);

    // Split across 3 Parquet files with multiple row groups, Float64 vectors, and Cosine normalization.
    let pq1 = dir.path().join("p1.parquet");
    let pq2 = dir.path().join("p2.parquet");
    let pq3 = dir.path().join("p3.parquet");
    write_full_parquet(&pq1, &rows[..150], dim, 40, true, None);
    write_full_parquet(&pq2, &rows[150..300], dim, 60, true, None);
    write_full_parquet(&pq3, &rows[300..], dim, 50, true, None);

    let cfg = full_collection_config(dim, DistanceMetric::Cosine);

    // Oracle via upsert.
    let oracle_dir = dir.path().join("oracle");
    let mut oracle = Collection::open(&oracle_dir, cfg.clone()).unwrap();
    for r in &rows {
        oracle.upsert(r.id, Some(&r.vec), r.attrs.clone()).unwrap();
    }

    // Bulk import with tiny target_sst_bytes (forcing SST rollover) and small memory budget (forcing spills).
    let bulk_dir = dir.path().join("bulk");
    let work_dir = dir.path().join("work");
    let bcfg = BulkImportConfig::new(cfg.clone(), "id", "vector", &work_dir)
        .with_target_sst_bytes(16 * 1024)
        .with_memory_budget_bytes(1024 * 1024)
        .with_threads(2)
        .with_verify_after(true);

    let phases = std::sync::Mutex::new(Vec::new());
    let progress_cb = |p: izzydb_bulk::Progress| phases.lock().unwrap().push(p.phase);
    let report = bulk_import(
        &bulk_dir,
        &[pq1, pq2, pq3],
        bcfg,
        Some(&progress_cb),
    )
    .unwrap();
    assert_eq!(report.docs_imported, rows.len() as u64);
    assert!(report.sst_files > 10, "expected multi-SST rollover, got {}", report.sst_files);
    assert!(!phases.lock().unwrap().is_empty());

    let bulk_col = Collection::open(&bulk_dir, cfg).unwrap();
    bulk_col.verify_invariants().unwrap();

    // Check every document's stored vector & attributes match oracle.
    for r in &rows {
        let o_doc = oracle.get(r.id).unwrap().unwrap();
        let b_doc = bulk_col.get(r.id).unwrap().unwrap();
        assert_eq!(o_doc.attributes, b_doc.attributes);
        assert_eq!(o_doc.embedding, b_doc.embedding);
    }

    // Filter parity across all operators.
    let filters = vec![
        Filter::Eq("category".into(), "books".into()),
        Filter::NotEq("category".into(), "books".into()),
        Filter::In("category".into(), vec!["books".into(), "sports".into()]),
        Filter::NotIn("category".into(), vec!["books".into(), "sports".into()]),
        Filter::And(vec![
            Filter::Gte("score".into(), 20.0f64.into()),
            Filter::Lt("score".into(), 65.0f64.into()),
            Filter::Eq("active".into(), true.into()),
        ]),
        Filter::Or(vec![
            Filter::Contains("tags".into(), "sale".into()),
            Filter::ContainsAny("nums".into(), vec![0i64.into(), 5i64.into()]),
        ]),
        Filter::Contains("uids".into(), 10u64.into()),
        Filter::IsNull("category".into()),
        Filter::IsNotNull("category".into()),
        Filter::Not(Box::new(Filter::Eq("active".into(), false.into()))),
        Filter::ContainsAnyToken {
            attr: "body".into(),
            query: "running trail".into(),
            last_as_prefix: true,
        },
        Filter::ContainsAllTokens {
            attr: "title".into(),
            query: "alpha".into(),
            last_as_prefix: false,
        },
        Filter::ContainsTokenSequence {
            attr: "body".into(),
            query: "delta echo".into(),
            last_as_prefix: false,
        },
        Filter::ContainsAnyToken {
            attr: "kw".into(),
            query: TokenQuery::PreTokenized(vec!["mountain".into()]),
            last_as_prefix: false,
        },
    ];
    for f in &filters {
        let o_ids = oracle.resolve_filter_doc_ids(f).unwrap();
        let b_ids = bulk_col.resolve_filter_doc_ids(f).unwrap();
        assert_eq!(o_ids, b_ids, "filter mismatch for {f:?}");
    }

    // Facet parity.
    let facet_specs = vec![
        FacetSpec::terms("category").with_limit(10),
        FacetSpec::terms("tags").with_limit(10),
        FacetSpec::stats("score"),
        FacetSpec::ranges(
            "rank",
            vec![
                FacetRange::below(0i64).with_label("neg"),
                FacetRange::from_value(0i64).with_label("pos"),
            ],
        ),
        FacetSpec::histogram("score", 25.0f64).with_alias("score_hist"),
    ];
    let o_facets = oracle.facet(None, &facet_specs).unwrap();
    let b_facets = bulk_col.facet(None, &facet_specs).unwrap();
    assert_eq!(o_facets, b_facets);

    let o_filt_facets = oracle
        .facet(Some(&Filter::Eq("active".into(), true.into())), &facet_specs)
        .unwrap();
    let b_filt_facets = bulk_col
        .facet(Some(&Filter::Eq("active".into(), true.into())), &facet_specs)
        .unwrap();
    assert_eq!(o_filt_facets, b_filt_facets);

    // Exact BM25 top-k equality (IDs and scores!).
    let rank_queries = vec![
        RankBy::bm25("body", "running trail waterproof"),
        RankBy::bm25("title", "xylophone"), // rare inline term
        RankBy::sum(vec![
            RankBy::product(2.0, RankBy::bm25("title", "alpha")),
            RankBy::bm25("body", "mountain sensor"),
        ]),
        RankBy::bm25("kw", TokenQuery::PreTokenized(vec!["camera".into()])),
    ];
    for rq in &rank_queries {
        let o_res = oracle.query_bm25(rq, None, 20).unwrap();
        let b_res = bulk_col.query_bm25(rq, None, 20).unwrap();
        assert_eq!(o_res.len(), b_res.len(), "BM25 len mismatch for {rq:?}");
        for (o, b) in o_res.iter().zip(b_res.iter()) {
            assert_eq!(o.id, b.id, "BM25 id mismatch for {rq:?}");
            assert!(
                (o.score - b.score).abs() < 1e-5,
                "BM25 score mismatch: {} vs {}",
                o.score,
                b.score
            );
        }
    }
}

#[test]
fn test_post_import_mutation_churn_and_invariants() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let rows = gen_test_rows(200, dim, 999);
    let pq_path = dir.path().join("init.parquet");
    write_full_parquet(&pq_path, &rows, dim, 100, false, None);

    let cfg = full_collection_config(dim, DistanceMetric::L2Squared);
    let bulk_dir = dir.path().join("bulk");
    let work_dir = dir.path().join("work");
    let bcfg = BulkImportConfig::new(cfg.clone(), "id", "vector", &work_dir);
    bulk_import(&bulk_dir, &[pq_path], bcfg, None).unwrap();

    let mut col = Collection::open(&bulk_dir, cfg.clone()).unwrap();
    let mut oracle: BTreeMap<DocId, (Vec<f32>, Attributes)> = rows
        .into_iter()
        .map(|r| (r.id, (r.vec, r.attrs)))
        .collect();

    let mut rng = SmallRng::seed_from_u64(4242);
    for step in 0..5 {
        // Upserts (both updates and new docs)
        for j in 0..25 {
            let id = if j % 2 == 0 {
                (rng.random_range(1..=200) * 3) as DocId
            } else {
                (1000 + step * 50 + j) as DocId
            };
            let v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
            let mut attrs = Attributes::new();
            attrs.insert("category".into(), "books".into());
            attrs.insert("rank".into(), (step as i64).into());
            attrs.insert("title".into(), format!("step {step} item {id}").into());
            col.upsert(id, Some(&v), attrs.clone()).unwrap();
            oracle.insert(id, (v, attrs));
        }

        // Patches
        let existing_ids: Vec<DocId> = oracle.keys().copied().take(10).collect();
        for id in existing_ids {
            let patch = DocumentPatch::new()
                .with_attribute("active", AttributeValue::Bool(true))
                .with_attribute("body", format!("patched step {step}").into());
            col.patch(id, patch).unwrap();
            let entry = oracle.get_mut(&id).unwrap();
            entry.1.insert("active".into(), true.into());
            entry
                .1
                .insert("body".into(), format!("patched step {step}").into());
        }

        // Deletes
        let del_ids: Vec<DocId> = oracle.keys().copied().take(8).collect();
        for id in &del_ids {
            assert!(col.delete(*id).unwrap());
            oracle.remove(id);
        }

        assert_eq!(col.len(), oracle.len() as u64);
    }

    drop(col);
    let reopened = Collection::open(&bulk_dir, cfg).unwrap();
    assert_eq!(reopened.len(), oracle.len() as u64);
}

#[test]
fn test_split_build_and_ingest_ssts_and_column_map() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let rows = gen_test_rows(80, dim, 55);
    let pq_path = dir.path().join("mapped.parquet");
    write_full_parquet(&pq_path, &rows, dim, 40, false, Some(("doc_pk", "emb")));

    let cfg = CollectionConfig::new(small_tree_cfg(dim, DistanceMetric::L2Squared))
        .with_attribute("cat", AttributeType::String, true)
        .with_fts_attribute("heading", AttributeType::String, false, FullTextSearchConfig::default());

    let work_dir = dir.path().join("work");
    let bcfg = BulkImportConfig::new(cfg.clone(), "doc_pk", "emb", &work_dir)
        .with_column_map(vec![
            ("category".into(), "cat".into()),
            ("title".into(), "heading".into()),
        ]);

    let bundle = build_ssts(&[pq_path], &bcfg).unwrap();
    let bundle_dir = bundle.dir.clone();
    drop(bundle);

    let loaded = SstBundle::open(&bundle_dir).unwrap();
    // Config mismatch rejected on validate_against
    let bad_cfg = CollectionConfig::new(small_tree_cfg(dim + 1, DistanceMetric::L2Squared));
    assert!(loaded.validate_against(&bad_cfg).is_err());

    let target_dir = dir.path().join("target");
    ingest_ssts(&target_dir, &loaded).unwrap();

    let col = Collection::open(&target_dir, cfg).unwrap();
    assert_eq!(col.len(), 80);
    let doc0 = col.get(rows[0].id).unwrap().unwrap();
    assert_eq!(
        doc0.attributes.get("cat"),
        rows[0].attrs.get("category")
    );
}

#[test]
fn test_failure_modes_crash_injection_non_empty_duplicates_and_edge_sizes() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let cfg = full_collection_config(dim, DistanceMetric::L2Squared);

    // 1. Crash injection after 1st, 5th, and last CF ingest.
    let rows = gen_test_rows(120, dim, 321);
    let pq_path = dir.path().join("crash.parquet");
    write_full_parquet(&pq_path, &rows, dim, 60, false, None);
    let bcfg = BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("work_crash"));
    let bundle = build_ssts(std::slice::from_ref(&pq_path), &bcfg).unwrap();

    for fail_at in [1usize, 5usize, usize::MAX] {
        let target = dir.path().join(format!("crash_target_{fail_at}"));
        let mut count = 0usize;
        let res = ingest_ssts_with_hook(&target, &bundle, Some(&cfg), |cf_name| {
            count += 1;
            let is_last = cf_name == "default";
            if count == fail_at || (fail_at == usize::MAX && is_last) {
                return Err(TreeError::Storage("injected crash".into()));
            }
            Ok(())
        });
        assert!(res.is_err());

        // Re-open MUST fail with incomplete bulk import error.
        let open_res = Collection::open(&target, cfg.clone());
        assert!(
            matches!(open_res, Err(TreeError::CorruptedData(ref m)) if m.contains("incomplete bulk import")),
            "expected incomplete bulk import error, got {:?}",
            open_res.err()
        );

        // Deleting and re-running succeeds.
        std::fs::remove_dir_all(&target).unwrap();
        let b2 = build_ssts(std::slice::from_ref(&pq_path), &bcfg).unwrap();
        ingest_ssts(&target, &b2).unwrap();
        let col = Collection::open(&target, cfg.clone()).unwrap();
        assert_eq!(col.len(), 120);
    }

    // 2. Non-empty target rejected before any write.
    let nonempty_target = dir.path().join("nonempty");
    {
        let mut col = Collection::open(&nonempty_target, cfg.clone()).unwrap();
        col.upsert(1, Some(&vec![1.0; dim]), Attributes::new()).unwrap();
    }
    let res_nonempty = bulk_import(
        &nonempty_target,
        std::slice::from_ref(&pq_path),
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_ne")),
        None,
    );
    assert!(matches!(res_nonempty, Err(TreeError::InvalidConfig(_))));
    // Existing collection still intact.
    let col_ne = Collection::open(&nonempty_target, cfg.clone()).unwrap();
    assert_eq!(col_ne.len(), 1);

    // 3. Importing into an already-initialized empty collection succeeds.
    let empty_init_target = dir.path().join("empty_init");
    {
        let _col = Collection::open(&empty_init_target, cfg.clone()).unwrap();
    }
    bulk_import(
        &empty_init_target,
        std::slice::from_ref(&pq_path),
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_ei")),
        None,
    )
    .unwrap();
    let col_ei = Collection::open(&empty_init_target, cfg.clone()).unwrap();
    assert_eq!(col_ei.len(), 120);

    // 4. Duplicate IDs under Error vs LastWins.
    let dup_rows_a = vec![
        TestRow {
            id: 10,
            vec: vec![1.0; dim],
            attrs: [("category".into(), "first".into())].into_iter().collect(),
        },
        TestRow {
            id: 20,
            vec: vec![2.0; dim],
            attrs: [("category".into(), "only".into())].into_iter().collect(),
        },
    ];
    let dup_rows_b = vec![TestRow {
        id: 10,
        vec: vec![3.0; dim],
        attrs: [("category".into(), "winner".into())].into_iter().collect(),
    }];
    let dup_pq1 = dir.path().join("dup1.parquet");
    let dup_pq2 = dir.path().join("dup2.parquet");
    write_full_parquet(&dup_pq1, &dup_rows_a, dim, 10, false, None);
    write_full_parquet(&dup_pq2, &dup_rows_b, dim, 10, false, None);

    let err_dup = bulk_import(
        dir.path().join("dup_err"),
        &[dup_pq1.clone(), dup_pq2.clone()],
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_dup_err")),
        None,
    );
    assert!(matches!(err_dup, Err(TreeError::InvalidConfig(_))));

    let dup_win_dir = dir.path().join("dup_win");
    let win_rep = bulk_import(
        &dup_win_dir,
        &[dup_pq1, dup_pq2],
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_dup_win"))
            .with_on_duplicate_id(DuplicatePolicy::LastWins),
        None,
    )
    .unwrap();
    assert_eq!(win_rep.rows_read, 3);
    assert_eq!(win_rep.docs_imported, 2);
    assert_eq!(win_rep.duplicates_dropped, 1);
    let dup_col = Collection::open(&dup_win_dir, cfg.clone()).unwrap();
    assert_eq!(
        dup_col.get(10).unwrap().unwrap().attributes.get("category"),
        Some(&AttributeValue::String("winner".into()))
    );
    assert_eq!(dup_col.get(10).unwrap().unwrap().embedding, Some(vec![3.0; dim]));

    // 5. N == 0 (empty collection valid).
    let zero_dir = dir.path().join("zero");
    let zero_rep = bulk_import(
        &zero_dir,
        &[],
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_zero")),
        None,
    )
    .unwrap();
    assert_eq!(zero_rep.docs_imported, 0);
    let zero_col = Collection::open(&zero_dir, cfg.clone()).unwrap();
    assert_eq!(zero_col.len(), 0);

    // 6. N < min_leaf_size (single root leaf).
    let tiny_pq = dir.path().join("tiny.parquet");
    write_full_parquet(&tiny_pq, &rows[..2], dim, 10, false, None);
    let tiny_dir = dir.path().join("tiny");
    let tiny_rep = bulk_import(
        &tiny_dir,
        &[tiny_pq],
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_tiny")),
        None,
    )
    .unwrap();
    assert_eq!(tiny_rep.leaves, 1);
    assert_eq!(tiny_rep.tree_height, 0);

    // 7. Identical vectors (degenerate k-means still gives legal leaf sizes).
    let ident_rows: Vec<TestRow> = (0..100)
        .map(|i| TestRow {
            id: i as u64,
            vec: vec![0.5; dim],
            attrs: Attributes::new(),
        })
        .collect();
    let ident_pq = dir.path().join("ident.parquet");
    write_full_parquet(&ident_pq, &ident_rows, dim, 50, false, None);
    let ident_dir = dir.path().join("ident");
    let ident_rep = bulk_import(
        &ident_dir,
        &[ident_pq],
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_ident")),
        None,
    )
    .unwrap();
    assert_eq!(ident_rep.docs_imported, 100);
    assert!(ident_rep.leaves >= 9);
}

#[test]
fn test_parquet_error_context_and_validation() {
    let dir = TempDir::new().unwrap();
    let dim = 4;
    let cfg = CollectionConfig::new(small_tree_cfg(dim, DistanceMetric::L2Squared))
        .with_attribute("score", AttributeType::Float, true);

    // NaN float attribute rejected with file/rg/row context.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 4),
            false,
        ),
        Field::new("score", DataType::Float64, true),
    ]));
    let mut id_b = UInt64Builder::new();
    id_b.append_value(1);
    let mut v_b = FixedSizeListBuilder::new(Float32Builder::new(), 4)
        .with_field(Field::new("item", DataType::Float32, false));
    for _ in 0..4 {
        v_b.values().append_value(1.0);
    }
    v_b.append(true);
    let mut s_b = Float64Builder::new();
    s_b.append_value(f64::NAN);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(id_b.finish()),
            Arc::new(v_b.finish()),
            Arc::new(s_b.finish()),
        ],
    )
    .unwrap();
    let bad_pq = dir.path().join("bad_nan.parquet");
    let mut writer = ArrowWriter::try_new(File::create(&bad_pq).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let err = bulk_import(
        dir.path().join("out_bad"),
        &[bad_pq],
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_bad")),
        None,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("bad_nan.parquet"), "error missing file: {msg}");
    assert!(msg.contains("rg=0:row=0"), "error missing rg/row: {msg}");
}

#[test]
fn test_bulk_append_equivalence_overlapping_ids_and_ann_strategies() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let mut cfg = full_collection_config(dim, DistanceMetric::L2Squared);
    cfg.segment.seal_rows = 128;

    let rows_a = gen_test_rows(200, dim, 101);
    // Batch B (small <= 30%): overlaps 15 DocIds and adds 30 new DocIds (45 total <= 30% of 200).
    let mut rows_b: Vec<TestRow> = rows_a[..15]
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut attrs = Attributes::new();
            attrs.insert("category".into(), "electronics".into());
            attrs.insert("score".into(), ((i as f64) * 3.5 + 1.0).into());
            attrs.insert("title".into(), format!("updated item {i}").into());
            attrs.insert("body".into(), "mountain waterproof shoe".into());
            TestRow {
                id: r.id,
                vec: vec![0.25 + (i as f32) * 0.01; dim],
                attrs,
            }
        })
        .collect();
    for i in 0..30 {
        let id = 5000 + i as DocId;
        let mut attrs = Attributes::new();
        attrs.insert("category".into(), "books".into());
        attrs.insert("rank".into(), (i as i64 - 10).into());
        attrs.insert("body".into(), "alpha bravo charlie running".into());
        rows_b.push(TestRow {
            id,
            vec: vec![-0.4 + (i as f32) * 0.02; dim],
            attrs,
        });
    }

    // Batch C (large > 30% -> full ANN tree rebuild): 150 rows including 20 overlaps.
    let mut rows_c = gen_test_rows(150, dim, 303);
    for (i, r) in rows_c.iter_mut().enumerate() {
        if i < 20 {
            r.id = rows_a[20 + i].id;
        } else {
            r.id = 10_000 + i as DocId;
        }
    }

    let pq_a = dir.path().join("a.parquet");
    let pq_b = dir.path().join("b.parquet");
    let pq_c = dir.path().join("c.parquet");
    write_full_parquet(&pq_a, &rows_a, dim, 80, false, None);
    write_full_parquet(&pq_b, &rows_b, dim, 20, false, None);
    write_full_parquet(&pq_c, &rows_c, dim, 60, false, None);

    // Oracle built sequentially via upsert.
    let oracle_dir = dir.path().join("oracle");
    let mut oracle = Collection::open(&oracle_dir, cfg.clone()).unwrap();
    for r in rows_a.iter().chain(&rows_b).chain(&rows_c) {
        oracle.upsert(r.id, Some(&r.vec), r.attrs.clone()).unwrap();
    }

    // Bulk import A, then append B (incremental ANN), then append C (> 30% rebuild).
    let bulk_dir = dir.path().join("bulk");
    bulk_import(
        &bulk_dir,
        &[pq_a],
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_a")),
        None,
    )
    .unwrap();

    let rep_b = bulk_append(
        &bulk_dir,
        &[pq_b],
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_b")),
        None,
    )
    .unwrap();
    assert_eq!(rep_b.docs_appended, 45);
    assert_eq!(rep_b.docs_replaced, 15);
    assert!(!rep_b.rebuilt_ann_tree);

    let rep_c = bulk_append(
        &bulk_dir,
        &[pq_c],
        BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join("w_c")),
        None,
    )
    .unwrap();
    assert_eq!(rep_c.docs_appended, 150);
    assert_eq!(rep_c.docs_replaced, 20);
    assert!(rep_c.rebuilt_ann_tree);

    let bulk_col = Collection::open(&bulk_dir, cfg).unwrap();
    bulk_col.verify_invariants().unwrap();
    assert_eq!(bulk_col.len(), oracle.len());

    // Document-for-document equality.
    for r in rows_a.iter().chain(&rows_b).chain(&rows_c) {
        let o_doc = oracle.get(r.id).unwrap().unwrap();
        let b_doc = bulk_col.get(r.id).unwrap().unwrap();
        assert_eq!(o_doc.attributes, b_doc.attributes);
        assert_eq!(o_doc.embedding, b_doc.embedding);
    }

    // Filters, facets, order_by, and BM25 parity.
    for f in [
        Filter::Eq("category".into(), "electronics".into()),
        Filter::And(vec![
            Filter::Gte("rank".into(), (-5i64).into()),
            Filter::Lt("rank".into(), 20i64.into()),
        ]),
        Filter::ContainsAnyToken {
            attr: "body".into(),
            query: "running mountain".into(),
            last_as_prefix: true,
        },
    ] {
        assert_eq!(
            oracle.resolve_filter_doc_ids(&f).unwrap(),
            bulk_col.resolve_filter_doc_ids(&f).unwrap()
        );
    }

    let specs = vec![
        FacetSpec::terms("category").with_limit(10),
        FacetSpec::stats("score"),
    ];
    assert_eq!(
        oracle.facet(None, &specs).unwrap(),
        bulk_col.facet(None, &specs).unwrap()
    );

    let o_ord = oracle
        .order_by(&[OrderBySpec::asc("score")], None, 15)
        .unwrap();
    let b_ord = bulk_col
        .order_by(&[OrderBySpec::asc("score")], None, 15)
        .unwrap();
    assert_eq!(
        o_ord.iter().map(|r| r.id).collect::<Vec<_>>(),
        b_ord.iter().map(|r| r.id).collect::<Vec<_>>()
    );

    let o_bm25 = oracle
        .query_bm25(&RankBy::bm25("body", "mountain running waterproof"), None, 20)
        .unwrap();
    let b_bm25 = bulk_col
        .query_bm25(&RankBy::bm25("body", "mountain running waterproof"), None, 20)
        .unwrap();
    assert_eq!(o_bm25.len(), b_bm25.len());
    for (o, b) in o_bm25.iter().zip(b_bm25.iter()) {
        assert_eq!(o.id, b.id);
        assert!((o.score - b.score).abs() < 1e-5);
    }

    let q = vec![0.25f32; dim];
    let opts = SearchOptions::new(10, 8).with_rerank(50);
    let o_vec = oracle.search(&q, opts, None).unwrap();
    let b_vec = bulk_col.search(&q, opts, None).unwrap();
    assert_eq!(o_vec.len(), b_vec.len());
}

#[test]
fn test_bulk_append_crash_recovery_preserves_original() {
    let dir = TempDir::new().unwrap();
    let dim = 8;
    let mut cfg = full_collection_config(dim, DistanceMetric::L2Squared);
    cfg.segment.seal_rows = 64;

    let init_rows = gen_test_rows(90, dim, 11);
    let append_rows = gen_test_rows(60, dim, 22);
    let pq_init = dir.path().join("init.parquet");
    let pq_app = dir.path().join("app.parquet");
    write_full_parquet(&pq_init, &init_rows, dim, 45, false, None);
    write_full_parquet(&pq_app, &append_rows, dim, 30, false, None);

    for (idx, fail_stage) in [
        BulkAppendHookStage::AfterBuildingMeta,
        BulkAppendHookStage::AfterIngestCf("cf_rows"),
        BulkAppendHookStage::BeforeFinalCommit,
    ]
    .into_iter()
    .enumerate()
    {
        let target = dir.path().join(format!("db_{idx}"));
        bulk_import(
            &target,
            std::slice::from_ref(&pq_init),
            BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join(format!("wi_{idx}"))),
            None,
        )
        .unwrap();

        let err = bulk_append_with_hook(
            &target,
            std::slice::from_ref(&pq_app),
            BulkImportConfig::new(cfg.clone(), "id", "vector", dir.path().join(format!("wa_{idx}"))),
            |stage| {
                if stage == fail_stage {
                    return Err(TreeError::Storage("injected append crash".into()));
                }
                Ok(())
            },
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("injected append crash"));

        let reopened = Collection::open(&target, cfg.clone()).unwrap();
        assert_eq!(reopened.len(), init_rows.len() as u64);
        for r in &init_rows {
            assert_eq!(
                reopened.get(r.id).unwrap().unwrap().attributes,
                r.attrs
            );
        }
        // Appended-only doc IDs from seed 22 (Wait: gen_test_rows uses same IDs (i+1)*3! Verify values stay init_rows[0]!).
        assert_eq!(
            reopened.get(init_rows[0].id).unwrap().unwrap().embedding,
            Some(init_rows[0].vec.clone())
        );
    }
}

#[test]
fn test_vectorless_bulk_import_and_append() {
    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::without_vectors()
        .with_attribute("category", AttributeType::String, true)
        .with_attribute("rank", AttributeType::Int, true)
        .with_fts_attribute("title", AttributeType::String, false, FullTextSearchConfig::default());

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("rank", DataType::Int64, true),
        Field::new("title", DataType::Utf8, true),
    ]));

    let write_vec_less = |path: &Path, ids: &[u64], cats: &[&str], ranks: &[i64], titles: &[&str]| {
        let mut id_b = UInt64Builder::new();
        let mut cat_b = StringBuilder::new();
        let mut rank_b = Int64Builder::new();
        let mut title_b = StringBuilder::new();
        for i in 0..ids.len() {
            id_b.append_value(ids[i]);
            cat_b.append_value(cats[i]);
            rank_b.append_value(ranks[i]);
            title_b.append_value(titles[i]);
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(id_b.finish()),
                Arc::new(cat_b.finish()),
                Arc::new(rank_b.finish()),
                Arc::new(title_b.finish()),
            ],
        )
        .unwrap();
        let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema.clone(), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    };

    let pq1 = dir.path().join("vl1.parquet");
    let pq2 = dir.path().join("vl2.parquet");
    write_vec_less(&pq1, &[1, 2, 3], &["a", "b", "a"], &[10, 5, 30], &["red fox", "blue jay", "green oak"]);
    write_vec_less(&pq2, &[2, 4], &["c", "a"], &[50, 20], &["silver fox", "golden pine"]);

    let target = dir.path().join("vl_db");
    let rep = bulk_import(
        &target,
        &[pq1],
        BulkImportConfig::new_vectorless(cfg.clone(), "id", dir.path().join("w1")),
        None,
    )
    .unwrap();
    assert_eq!(rep.docs_imported, 3);

    let app = bulk_append(
        &target,
        &[pq2],
        BulkImportConfig::new_vectorless(cfg.clone(), "id", dir.path().join("w2")),
        None,
    )
    .unwrap();
    assert_eq!(app.docs_appended, 2);
    assert_eq!(app.docs_replaced, 1);

    let col = Collection::open(&target, cfg).unwrap();
    assert_eq!(col.len(), 4);
    assert!(col.get(2).unwrap().unwrap().embedding.is_none());
    assert_eq!(
        col.resolve_filter_doc_ids(&Filter::Eq("category".into(), "a".into())).unwrap(),
        vec![1, 3, 4]
    );
    let top = col.order_by(&[OrderBySpec::desc("rank")], None, 2).unwrap();
    assert_eq!(top.iter().map(|r| r.id).collect::<Vec<_>>(), vec![2, 3]);
    let bm25 = col.query_bm25(&RankBy::bm25("title", "fox"), None, 5).unwrap();
    assert_eq!(bm25.len(), 2);
}

#[test]
fn test_detached_columns_parquet_import() {
    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::without_vectors()
        .with_attribute("title", AttributeType::String, true)
        .with_detached_attribute("popularity", AttributeType::Int, true)
        .with_detached_attribute("badge", AttributeType::String, true);

    let target = dir.path().join("det_db");
    let mut col = Collection::open(&target, cfg.clone()).unwrap();
    for id in 1..=5u64 {
        let mut attrs = Attributes::new();
        attrs.insert("title".into(), format!("item-{id}").into());
        col.upsert(id, None, attrs).unwrap();
    }

    let pq_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("popularity", DataType::Int64, true),
        Field::new("badge", DataType::Utf8, true),
    ]));
    let mut id_b = UInt64Builder::new();
    let mut pop_b = Int64Builder::new();
    let mut badge_b = StringBuilder::new();
    for (id, pop, badge) in [(1, 100, Some("gold")), (3, 300, Some("silver")), (4, 50, None)] {
        id_b.append_value(id);
        pop_b.append_value(pop);
        match badge {
            Some(s) => badge_b.append_value(s),
            None => badge_b.append_null(),
        }
    }
    let batch = RecordBatch::try_new(
        pq_schema.clone(),
        vec![
            Arc::new(id_b.finish()),
            Arc::new(pop_b.finish()),
            Arc::new(badge_b.finish()),
        ],
    )
    .unwrap();
    let pq = dir.path().join("det_cols.parquet");
    let mut w = ArrowWriter::try_new(File::create(&pq).unwrap(), pq_schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();

    let cols = vec!["popularity".to_owned(), "badge".to_owned()];
    let rows = scan_detached_from_parquet(&[pq], col.schema(), "id", &cols, None).unwrap();
    let job = col
        .detached_ingest()
        .columns(cols)
        .mode(DetachedWriteMode::Replace)
        .duplicate_policy(DetachedDuplicatePolicy::LastWins)
        .from_iter(rows)
        .build()
        .unwrap();
    col.publish_detached(job).unwrap();

    assert_eq!(
        col.resolve_filter_doc_ids(&Filter::Gte("popularity".into(), 100i64.into()))
            .unwrap(),
        vec![1, 3]
    );
    assert_eq!(
        col.get(3).unwrap().unwrap().attributes.get("badge"),
        Some(&AttributeValue::String("silver".into()))
    );
    assert!(!col.get(2).unwrap().unwrap().attributes.contains_key("popularity"));
}
