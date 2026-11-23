#![cfg(any())] // v3: phase 5
use std::collections::{HashMap, HashSet};

use izzydb::fts::{Bm25TermScorer, FtsAnalyzer, ReusableBuffer};
use izzydb::{
    AttributeType, Attributes, Bm25Overrides, Bm25Result, Collection, CollectionConfig,
    DistanceMetric, DocId, DocumentPatch, Filter, FullTextSearchConfig, HybridSearchOptions,
    Language, RankBy, SearchOptions, TokenQuery, Tokenizer, TreeConfig, TreeError,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn test_tree_cfg(dim: usize) -> TreeConfig {
    TreeConfig {
        dim,
        metric: DistanceMetric::L2Squared,
        min_internal_fanout: 2,
        max_internal_fanout: 6,
        min_leaf_size: 2,
        max_leaf_size: 8,
        insert_beam_width: 4,
        lire_neighbor_count: 4,
        enable_lire: true,
        centroid_drift_eps: 1e-4,
    }
}

fn random_vec(rng: &mut SmallRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect()
}

#[test]
fn test_fts_schema_validation_and_reconcile() {
    let base_tree = test_tree_cfg(4);

    // Invalid attribute types
    assert!(matches!(
        CollectionConfig::new(base_tree.clone())
            .with_fts_attribute("num", AttributeType::Int, true, FullTextSearchConfig::default())
            .validate(),
        Err(TreeError::InvalidSchema(_))
    ));

    // PreTokenizedArray requires StringArray and rejects normalization flags
    assert!(matches!(
        CollectionConfig::new(base_tree.clone())
            .with_fts_attribute(
                "s",
                AttributeType::String,
                false,
                FullTextSearchConfig::default().with_tokenizer(Tokenizer::PreTokenizedArray),
            )
            .validate(),
        Err(TreeError::InvalidSchema(_))
    ));
    assert!(matches!(
        CollectionConfig::new(base_tree.clone())
            .with_fts_attribute(
                "s",
                AttributeType::StringArray,
                false,
                FullTextSearchConfig::default()
                    .with_tokenizer(Tokenizer::PreTokenizedArray)
                    .with_stemming(true),
            )
            .validate(),
        Err(TreeError::InvalidSchema(_))
    ));

    // Whitespace rejects stemming & stopwords
    assert!(matches!(
        CollectionConfig::new(base_tree.clone())
            .with_fts_attribute(
                "s",
                AttributeType::String,
                false,
                FullTextSearchConfig::default()
                    .with_tokenizer(Tokenizer::Whitespace)
                    .with_stemming(true),
            )
            .validate(),
        Err(TreeError::InvalidSchema(_))
    ));

    // Stemming / stopwords require case_sensitive == false
    assert!(matches!(
        CollectionConfig::new(base_tree.clone())
            .with_fts_attribute(
                "s",
                AttributeType::String,
                false,
                FullTextSearchConfig::default()
                    .with_case_sensitive(true)
                    .with_stemming(true),
            )
            .validate(),
        Err(TreeError::InvalidSchema(_))
    ));

    // Unsupported stopword language rejected
    assert!(matches!(
        CollectionConfig::new(base_tree.clone())
            .with_fts_attribute(
                "s",
                AttributeType::String,
                false,
                FullTextSearchConfig::default()
                    .with_language(Language::Arabic)
                    .with_remove_stopwords(true),
            )
            .validate(),
        Err(TreeError::InvalidSchema(_))
    ));

    // Invalid max_token_length / k1 / b / k3
    assert!(matches!(
        CollectionConfig::new(base_tree.clone())
            .with_fts_attribute(
                "s",
                AttributeType::String,
                false,
                FullTextSearchConfig::default().with_max_token_length(255),
            )
            .validate(),
        Err(TreeError::InvalidSchema(_))
    ));
    assert!(matches!(
        CollectionConfig::new(base_tree.clone())
            .with_fts_attribute(
                "s",
                AttributeType::String,
                false,
                FullTextSearchConfig::default().with_b(1.5),
            )
            .validate(),
        Err(TreeError::InvalidSchema(_))
    ));

    // Reopen reconcile: changing FTS config fails; adding new FTS attribute backfills existing docs
    let dir = TempDir::new().unwrap();
    let cfg1 = CollectionConfig::new(base_tree.clone())
        .with_attribute("cat", AttributeType::String, true)
        .with_fts_attribute(
            "title",
            AttributeType::String,
            false,
            FullTextSearchConfig::default().with_stemming(true),
        );
    {
        let mut col = Collection::open(dir.path(), cfg1.clone()).unwrap();
        let mut attrs = Attributes::new();
        attrs.insert("cat".into(), "db".into());
        attrs.insert("title".into(), "Running fast queries".into());
        col.upsert(1, &[1.0; 4], attrs).unwrap();
        col.verify_invariants().unwrap();
    }

    let bad_cfg = CollectionConfig::new(base_tree.clone())
        .with_attribute("cat", AttributeType::String, true)
        .with_fts_attribute(
            "title",
            AttributeType::String,
            false,
            FullTextSearchConfig::default().with_stemming(false),
        );
    assert!(matches!(
        Collection::open(dir.path(), bad_cfg),
        Err(TreeError::ConfigMismatch {
            field: "attribute_full_text_search",
            ..
        })
    ));

    // Adding a new FTS attribute + PreTokenizedArray validation
    let cfg2 = cfg1.with_fts_attribute(
        "kw",
        AttributeType::StringArray,
        false,
        FullTextSearchConfig::default()
            .with_tokenizer(Tokenizer::PreTokenizedArray)
            .with_max_token_length(10),
    );
    let mut col = Collection::open(dir.path(), cfg2).unwrap();
    let stats = col.verify_invariants().unwrap();
    assert_eq!(stats.num_fts_docs, 1);

    let mut bad_kw = Attributes::new();
    bad_kw.insert("kw".into(), vec!["toolongtokenvalue".to_string()].into());
    assert!(matches!(
        col.upsert(2, &[0.5; 4], bad_kw),
        Err(TreeError::InvalidAttribute { .. })
    ));

    let mut good_kw = Attributes::new();
    good_kw.insert("kw".into(), vec!["rust".to_string(), "bm25".to_string()].into());
    col.upsert(2, &[0.5; 4], good_kw).unwrap();

    let hits = col
        .query_bm25(
            &RankBy::bm25("kw", TokenQuery::PreTokenized(vec!["bm25".into()])),
            None,
            5,
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 2);
    col.verify_invariants().unwrap();
}

#[test]
fn test_fts_phrase_array_gap_prefix_and_vector_filters() {
    let dir = TempDir::new().unwrap();
    let fts_en = FullTextSearchConfig::default()
        .with_stemming(true)
        .with_remove_stopwords(true)
        .with_ascii_folding(true);
    let fts_nopos = FullTextSearchConfig::default().with_index_positions(false);

    let cfg = CollectionConfig::new(test_tree_cfg(4))
        .with_attribute("tag", AttributeType::String, true)
        .with_fts_attribute("body", AttributeType::String, false, fts_en)
        .with_fts_attribute(
            "chunks",
            AttributeType::StringArray,
            false,
            FullTextSearchConfig::default(),
        )
        .with_fts_attribute("summary", AttributeType::String, false, fts_nopos);

    let mut col = Collection::open(dir.path(), cfg).unwrap();

    let docs = vec![
        (
            1u64,
            vec![0.0, 0.0, 0.0, 0.0],
            "tech",
            "The quick brown fox jumps over the lazy café dog",
            vec!["quick brown", "fox jumps"],
            "brown fox",
        ),
        (
            2u64,
            vec![0.1, 0.0, 0.0, 0.0],
            "tech",
            "Quick dog in the brown forest",
            vec!["quick brown fox", "jumps high"],
            "forest dog",
        ),
        (
            3u64,
            vec![5.0, 5.0, 5.0, 5.0],
            "news",
            "Fast runners running in the morning",
            vec!["morning run", "sprint"],
            "runners",
        ),
        (
            4u64,
            vec![0.2, 0.0, 0.0, 0.0],
            "news",
            "The cat in the hat sat on the mat",
            vec!["cat hat"],
            "cat hat",
        ),
    ];

    for (id, vec, tag, body, chunks, summary) in docs {
        let mut attrs = Attributes::new();
        attrs.insert("tag".into(), tag.into());
        attrs.insert("body".into(), body.into());
        attrs.insert(
            "chunks".into(),
            chunks
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
                .into(),
        );
        attrs.insert("summary".into(), summary.into());
        col.upsert(id, &vec, attrs).unwrap();
    }
    col.verify_invariants().unwrap();

    // ContainsTokenSequence rejected on index_positions = false
    assert!(matches!(
        col.search(
            &[0.0; 4],
            SearchOptions::new(5, 4),
            Some(&Filter::ContainsTokenSequence {
                attr: "summary".into(),
                query: "brown fox".into(),
                last_as_prefix: false,
            })
        ),
        Err(TreeError::InvalidFilter(_))
    ));

    // StringArray: "brown fox" matches ContainsAllTokens, NOT ContainsTokenSequence across element gap!
    let all_chunks = col
        .search(
            &[0.0; 4],
            SearchOptions::new(5, 4),
            Some(&Filter::ContainsAllTokens {
                attr: "chunks".into(),
                query: "brown fox".into(),
                last_as_prefix: false,
            }),
        )
        .unwrap();
    let all_ids: HashSet<DocId> = all_chunks.iter().map(|r| r.id).collect();
    assert_eq!(all_ids, HashSet::from([1, 2]));

    let seq_chunks = col
        .search(
            &[0.0; 4],
            SearchOptions::new(5, 4),
            Some(&Filter::ContainsTokenSequence {
                attr: "chunks".into(),
                query: "brown fox".into(),
                last_as_prefix: false,
            }),
        )
        .unwrap();
    assert_eq!(seq_chunks.iter().map(|r| r.id).collect::<Vec<_>>(), vec![2]);

    // Stopword gap preservation: "cat in the hat" matches doc 4 ("the cat in the hat"), NOT doc 4's chunks ("cat hat")
    let seq_stop_body = col
        .search(
            &[0.0; 4],
            SearchOptions::new(5, 4),
            Some(&Filter::ContainsTokenSequence {
                attr: "body".into(),
                query: "cat in the hat".into(),
                last_as_prefix: false,
            }),
        )
        .unwrap();
    assert_eq!(
        seq_stop_body.iter().map(|r| r.id).collect::<Vec<_>>(),
        vec![4]
    );

    // Prefix phrase: "quick bro" with last_as_prefix = true matches doc 1 & 2
    let seq_prefix = col
        .search(
            &[0.0; 4],
            SearchOptions::new(5, 4),
            Some(&Filter::ContainsTokenSequence {
                attr: "chunks".into(),
                query: "quick bro".into(),
                last_as_prefix: true,
            }),
        )
        .unwrap();
    assert_eq!(
        seq_prefix.iter().map(|r| r.id).collect::<HashSet<_>>(),
        HashSet::from([1, 2])
    );

    // Mixed Attribute + Not(FTS) filter in vector search
    let mixed = Filter::And(vec![
        Filter::Eq("tag".into(), "tech".into()),
        Filter::Not(Box::new(Filter::ContainsAnyToken {
            attr: "body".into(),
            query: "forest".into(),
            last_as_prefix: false,
        })),
    ]);
    let mixed_hits = col
        .search(&[0.0; 4], SearchOptions::new(5, 4), Some(&mixed))
        .unwrap();
    assert_eq!(mixed_hits.iter().map(|r| r.id).collect::<Vec<_>>(), vec![1]);

    // ASCII folding + stemming: "café" -> "cafe", "runners" -> "runner" / "run"
    let cafe_hits = col
        .query_bm25(&RankBy::bm25("body", "cafe"), None, 5)
        .unwrap();
    assert_eq!(cafe_hits.iter().map(|r| r.id).collect::<Vec<_>>(), vec![1]);
}

fn brute_force_bm25_eval(
    docs: &HashMap<DocId, (Vec<f32>, Attributes)>,
    schema: &izzydb::CollectionSchema,
    rank_by: &RankBy,
    filter: Option<&Filter>,
    fts_index: &izzydb::FtsIndex,
    k: usize,
) -> Vec<Bm25Result> {
    let mut buf = ReusableBuffer::new();
    let mut per_attr_counts: HashMap<u16, (u64, u64)> = HashMap::new();
    let mut df_map: HashMap<(u16, String), u64> = HashMap::new();
    let mut doc_analyzed: HashMap<(DocId, u16), izzydb::fts::AnalyzedDoc> = HashMap::new();

    for entry in schema.fts_entries() {
        let cfg = entry.full_text_search.as_ref().unwrap();
        let analyzer = FtsAnalyzer::new(cfg.clone());
        let mut doc_count = 0u64;
        let mut total_tokens = 0u64;
        for (&doc_id, (_, attrs)) in docs {
            if let Some(val) = attrs.get(&entry.name)
                && let Some(ad) = analyzer.analyze_attribute_value(val, &mut buf).unwrap()
            {
                doc_count += 1;
                total_tokens += ad.doc_len as u64;
                for tok in ad.terms.keys() {
                    *df_map.entry((entry.id, tok.clone())).or_insert(0) += 1;
                }
                doc_analyzed.insert((doc_id, entry.id), ad);
            }
        }
        per_attr_counts.insert(entry.id, (doc_count, total_tokens));
    }

    fn eval_expr(
        doc_id: DocId,
        expr: &RankBy,
        schema: &izzydb::CollectionSchema,
        per_attr_counts: &HashMap<u16, (u64, u64)>,
        df_map: &HashMap<(u16, String), u64>,
        doc_analyzed: &HashMap<(DocId, u16), izzydb::fts::AnalyzedDoc>,
        buf: &mut ReusableBuffer,
    ) -> f32 {
        match expr {
            RankBy::Const(c) => *c,
            RankBy::Product(children) => children
                .iter()
                .map(|c| {
                    eval_expr(
                        doc_id,
                        c,
                        schema,
                        per_attr_counts,
                        df_map,
                        doc_analyzed,
                        buf,
                    )
                })
                .product(),
            RankBy::Filter(_)
            | RankBy::Attribute { .. }
            | RankBy::Saturate { .. }
            | RankBy::Decay { .. } => unreachable!("lexical reference evaluator"),
            RankBy::Sum(children) => children
                .iter()
                .map(|c| {
                    eval_expr(
                        doc_id,
                        c,
                        schema,
                        per_attr_counts,
                        df_map,
                        doc_analyzed,
                        buf,
                    )
                })
                .sum(),
            RankBy::Max(children) => children
                .iter()
                .map(|c| {
                    eval_expr(
                        doc_id,
                        c,
                        schema,
                        per_attr_counts,
                        df_map,
                        doc_analyzed,
                        buf,
                    )
                })
                .fold(0.0f32, f32::max),
            RankBy::Bm25 {
                attr,
                query,
                params,
            } => {
                let entry = schema.get_by_name(attr).unwrap();
                let fts = entry.full_text_search.as_ref().unwrap();
                let Some(ad) = doc_analyzed.get(&(doc_id, entry.id)) else {
                    return 0.0;
                };
                let (doc_count, total_tokens) = per_attr_counts[&entry.id];
                let qtokens = FtsAnalyzer::new(fts.clone())
                    .analyze_query(query, buf)
                    .unwrap();
                if qtokens.is_empty() {
                    return 0.0;
                }
                let k1 = params.as_ref().and_then(|p| p.k1).unwrap_or(fts.k1);
                let b = params.as_ref().and_then(|p| p.b).unwrap_or(fts.b);
                let k3 = params.as_ref().and_then(|p| p.k3).unwrap_or(fts.k3);
                let last_as_prefix = params
                    .as_ref()
                    .and_then(|p| p.last_as_prefix)
                    .unwrap_or(false);

                let (exact, prefix_tok) = if last_as_prefix {
                    (&qtokens[..qtokens.len() - 1], Some(&qtokens.last().unwrap().text))
                } else {
                    (&qtokens[..], None)
                };
                let mut qtf_map: HashMap<&str, u32> = HashMap::new();
                for qt in exact {
                    *qtf_map.entry(qt.text.as_str()).or_insert(0) += 1;
                }
                let mut total = 0.0f32;
                for (tok, qtf) in qtf_map {
                    if let Some(at) = ad.terms.get(tok) {
                        let df = df_map[&(entry.id, tok.to_owned())];
                        let scorer = Bm25TermScorer::new(
                            1.0,
                            doc_count,
                            total_tokens,
                            df,
                            qtf,
                            k1,
                            b,
                            k3,
                        );
                        total += scorer.score(at.tf, ad.doc_len);
                    }
                }
                if let Some(pfx) = prefix_tok {
                    let mut candidates: Vec<(String, u64)> = df_map
                        .iter()
                        .filter(|((aid, tok), _)| *aid == entry.id && tok.starts_with(pfx))
                        .map(|((_, tok), &df)| (tok.clone(), df))
                        .collect();
                    candidates.sort_by(|a, b| {
                        (a.0 != *pfx)
                            .cmp(&(b.0 != *pfx))
                            .then_with(|| b.1.cmp(&a.1))
                            .then_with(|| a.0.cmp(&b.0))
                    });
                    candidates.truncate(128);
                    let mut best = 0.0f32;
                    for (tok, df) in candidates {
                        if let Some(at) = ad.terms.get(&tok) {
                            let scorer = Bm25TermScorer::new(
                                1.0,
                                doc_count,
                                total_tokens,
                                df,
                                1,
                                k1,
                                b,
                                k3,
                            );
                            best = best.max(scorer.score(at.tf, ad.doc_len));
                        }
                    }
                    total += best;
                }
                total
            }
        }
    }

    let mut out = Vec::new();
    for (&doc_id, (_, attrs)) in docs {
        if let Some(f) = filter
            && !fts_index.matches_filter(schema, f, attrs)
        {
            continue;
        }
        let s = eval_expr(
            doc_id,
            rank_by,
            schema,
            &per_attr_counts,
            &df_map,
            &doc_analyzed,
            &mut buf,
        );
        if s > 0.0 {
            out.push(Bm25Result {
                id: doc_id,
                score: s,
            });
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    out.truncate(k);
    out
}

#[test]
fn test_fts_churn_parity_filtered_bm25_and_hybrid_search() {
    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(test_tree_cfg(8))
        .with_attribute("bucket", AttributeType::Uint, true)
        .with_attribute("active", AttributeType::Bool, true)
        .with_fts_attribute(
            "title",
            AttributeType::String,
            false,
            FullTextSearchConfig::default().with_stemming(true),
        )
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default()
                .with_stemming(true)
                .with_remove_stopwords(true),
        );

    let mut rng = SmallRng::seed_from_u64(98765);
    let vocab = [
        "vector", "search", "database", "rocksdb", "bm25", "maxscore", "rabitq", "cluster",
        "posting", "block", "prefix", "phrase", "hybrid", "ranking", "filter", "memory",
        "storage", "index", "token", "stemming", "unicode", "alpha", "beta", "gamma",
    ];

    let mut oracle: HashMap<DocId, (Vec<f32>, Attributes)> = HashMap::new();
    let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();

    // Churn across > 300 docs (multiple 256-doc FTS blocks) with inserts, updates, deletes, reopen
    for step in 0..750usize {
        let op = rng.random_range(0..100);
        if op < 70 {
            let id = rng.random_range(0..450u64);
            let vec = random_vec(&mut rng, 8);
            let title_len = rng.random_range(2..=5);
            let body_len = rng.random_range(6..=20);
            let title = (0..title_len)
                .map(|_| vocab[rng.random_range(0..vocab.len())])
                .collect::<Vec<_>>()
                .join(" ");
            let body = (0..body_len)
                .map(|_| vocab[rng.random_range(0..vocab.len())])
                .collect::<Vec<_>>()
                .join(" ");
            let mut attrs = Attributes::new();
            attrs.insert("bucket".into(), (id % 5).into());
            attrs.insert("active".into(), (id % 2 == 0).into());
            attrs.insert("title".into(), title.into());
            attrs.insert("body".into(), body.into());
            col.upsert(id, &vec, attrs.clone()).unwrap();
            oracle.insert(id, (vec, attrs));
        } else if op < 92 {
            let id = rng.random_range(0..450u64);
            col.delete(id).unwrap();
            oracle.remove(&id);
        } else {
            drop(col);
            col = Collection::open(dir.path(), cfg.clone()).unwrap();
        }
        if step % 150 == 149 {
            let stats = col.verify_invariants().unwrap();
            assert_eq!(stats.num_docs, oracle.len());
            assert_eq!(stats.num_fts_docs, oracle.len());
        }
    }

    let stats = col.verify_invariants().unwrap();
    assert_eq!(stats.num_fts_docs, oracle.len());

    let fts_helper = izzydb::FtsIndex::new();
    let schema = col.schema();

    let queries: Vec<(RankBy, Option<Filter>)> = vec![
        // Single field short query
        (RankBy::bm25("body", "vector bm25 maxscore"), None),
        // Composite weighted Sum + Product
        (
            RankBy::Sum(vec![
                RankBy::product(2.5, RankBy::bm25("title", "rocksdb posting block")),
                RankBy::bm25("body", "rocksdb posting block rabitq cluster"),
            ]),
            None,
        ),
        // Max combinator
        (
            RankBy::Max(vec![
                RankBy::bm25("title", "hybrid ranking"),
                RankBy::bm25("body", "hybrid ranking"),
            ]),
            None,
        ),
        // Prefix query
        (
            RankBy::bm25_with(
                "body",
                "vector max",
                Bm25Overrides::default().with_last_as_prefix(true),
            ),
            None,
        ),
        // Filtered BM25: selective attribute pushdown
        (
            RankBy::bm25("body", "database storage index"),
            Some(Filter::Eq("bucket".into(), 2u64.into())),
        ),
        // Filtered BM25: FTS-only pushdown
        (
            RankBy::bm25("body", "vector search database rocksdb"),
            Some(Filter::ContainsAllTokens {
                attr: "title".into(),
                query: "vector".into(),
                last_as_prefix: false,
            }),
        ),
        // Filtered BM25: lazy fallback (Not + broad Bool + FTS)
        (
            RankBy::bm25("body", "cluster posting block"),
            Some(Filter::And(vec![
                Filter::Eq("active".into(), true.into()),
                Filter::Not(Box::new(Filter::Eq("bucket".into(), 0u64.into()))),
            ])),
        ),
    ];

    for (rank_by, filter) in &queries {
        let got = col.query_bm25(rank_by, filter.as_ref(), 12).unwrap();
        let expected = brute_force_bm25_eval(
            &oracle,
            schema,
            rank_by,
            filter.as_ref(),
            &fts_helper,
            12,
        );
        assert_eq!(got.len(), expected.len(), "len mismatch for {rank_by:?}");
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_eq!(g.id, e.id, "id mismatch for {rank_by:?}");
            assert!(
                (g.score - e.score).abs() <= 1e-4,
                "score mismatch for doc {}: {} vs {}",
                g.id,
                g.score,
                e.score
            );
        }
    }

    // Hybrid search (RRF)
    let qvec = random_vec(&mut rng, 8);
    let rank_by = RankBy::bm25("body", "vector search bm25");
    let hyb_filter = Filter::Eq("active".into(), true.into());
    let hyb_opts = HybridSearchOptions::new(10, SearchOptions::new(20, 8).with_rerank(40))
        .with_bm25_candidates(20)
        .with_rrf_k(60.0)
        .with_weights(1.0, 1.5);
    let hyb = col
        .hybrid_search(&qvec, &rank_by, hyb_opts, Some(&hyb_filter))
        .unwrap();
    assert!(!hyb.is_empty());
    assert!(hyb.len() <= 10);
    for w in hyb.windows(2) {
        assert!(
            w[0].score > w[1].score
                || (w[0].score == w[1].score && w[0].id < w[1].id)
        );
    }
    for h in &hyb {
        let doc = col.get(h.id).unwrap().unwrap();
        assert!(hyb_filter.matches(&doc.attributes));
        let mut expected_score = 0.0f32;
        if let Some(vr) = h.vector_rank {
            expected_score += 1.0 / (60.0 + vr as f32);
        }
        if let Some(br) = h.bm25_rank {
            expected_score += 1.5 / (60.0 + br as f32);
        }
        assert!((h.score - expected_score).abs() < 1e-6);
    }
}

#[test]
fn test_fts_recycling_term_stats_tightening_and_null_empty_transitions() {
    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(test_tree_cfg(4))
        .with_attribute("tag", AttributeType::String, true)
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default()
                .with_stemming(true)
                .with_remove_stopwords(true),
        );

    {
        let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();

        // Insert 300 docs (> 256 FTS block size) to span 2 FTS blocks.
        for id in 0..300u64 {
            let mut attrs = Attributes::new();
            attrs.insert("tag".into(), if id % 2 == 0 { "even" } else { "odd" }.into());
            let body = if id == 50 {
                // Outlier doc: very high tf=6 for "outlier" and short dl=6 (min_dl_over_tf = 1.0)
                "outlier outlier outlier outlier outlier outlier"
            } else {
                "outlier common common alpha beta gamma"
            };
            attrs.insert("body".into(), body.into());
            col.upsert(id, &[0.0, 0.0, 0.0, 0.0], attrs).unwrap();
        }
        col.verify_invariants().unwrap();

        // Non-FTS attribute patch + vector-only patch preserve FTS index and invariants.
        col.patch(
            10,
            DocumentPatch::new()
                .with_embedding(vec![0.5, 0.0, 0.0, 0.0])
                .with_attribute("tag", "patched".into()),
        )
        .unwrap();
        col.verify_invariants().unwrap();

        // Edit outlier doc 50 down (tf=6 -> tf=1, dl=6 -> dl=7): inline TermStats tightening must fire!
        col.patch(
            50,
            DocumentPatch::new()
                .with_attribute("body", "outlier alpha beta gamma delta epsilon zeta".into()),
        )
        .unwrap();
        col.verify_invariants().unwrap();

        // Null / empty / stopword-only transitions on doc 250 (preserved past delete_batch(0..200))
        // non-empty -> stopword-only (doc_len = 0)
        col.patch(
            250,
            DocumentPatch::new().with_attribute("body", "the and of".into()),
        )
        .unwrap();
        col.verify_invariants().unwrap();
        // stopword-only -> empty string
        col.patch(250, DocumentPatch::new().with_attribute("body", "".into()))
            .unwrap();
        col.verify_invariants().unwrap();
        // empty string -> Null (cleared)
        col.patch(250, DocumentPatch::new().clear_attribute("body"))
            .unwrap();
        col.verify_invariants().unwrap();
        // Null -> non-empty
        col.patch(
            250,
            DocumentPatch::new().with_attribute("body", "outlier reborn".into()),
        )
        .unwrap();
        col.verify_invariants().unwrap();

        // Delete 200 docs (0..200) to populate fts_free_ids, then re-insert 150 docs and verify FtsDocId recycling!
        let ids_to_del: Vec<DocId> = (0..200u64).collect();
        assert_eq!(col.delete_batch(&ids_to_del).unwrap(), 200);
        col.verify_invariants().unwrap();

        for id in 1000..1150u64 {
            let mut attrs = Attributes::new();
            attrs.insert("tag".into(), "recycled".into());
            attrs.insert("body".into(), "common recycled text".into());
            col.upsert(id, &[1.0, 0.0, 0.0, 0.0], attrs).unwrap();
        }
        col.verify_invariants().unwrap();

        // delete_by_filter with pure FTS filter & mixed filter
        let fts_del = col
            .delete_by_filter(&Filter::ContainsAllTokens {
                attr: "body".into(),
                query: "reborn".into(),
                last_as_prefix: false,
            })
            .unwrap();
        assert_eq!(fts_del, 1);

        let mixed_upd = col
            .update_by_filter(
                &Filter::And(vec![
                    Filter::Eq("tag".into(), "odd".into()),
                    Filter::ContainsAnyToken {
                        attr: "body".into(),
                        query: "outl".into(),
                        last_as_prefix: true,
                    },
                ]),
                &DocumentPatch::new().with_attribute("body", "updated common text".into()),
            )
            .unwrap();
        assert!(mixed_upd > 0);
        col.verify_invariants().unwrap();
    }

    // Check FtsMeta.next_fts_doc_id stayed bounded at 300 (didn't grow to 450!) and TermStats tightened.
    {
        let (storage, _) = izzydb::storage::Storage::open(dir.path(), &test_tree_cfg(4)).unwrap();
        let meta = izzydb::fts::FtsMeta::decode(&storage.get_fts_meta_bytes().unwrap().unwrap())
            .unwrap();
        assert!(
            meta.next_fts_doc_id <= 300,
            "expected FtsDocId recycling to keep next_fts_doc_id <= 300, got {}",
            meta.next_fts_doc_id
        );
        let outlier_key = izzydb::codec::encode_fts_term_key(1, "outlier");
        let rec = izzydb::fts::TermRecord::decode(
            &storage.get_fts_term_bytes(&outlier_key).unwrap().unwrap(),
        )
        .unwrap();
        assert_eq!(rec.stats.max_tf, 1);
        assert_eq!(rec.stats.min_dl_over_tf, 6.0);
    }

    // Drain all remaining docs and confirm next_fts_doc_id shrinks back to 0 and free list is empty.
    {
        let mut col = Collection::open(dir.path(), cfg).unwrap();
        let all = col.len();
        assert_eq!(col.delete_by_filter(&Filter::And(vec![])).unwrap(), all);
        col.verify_invariants().unwrap();
    }
    {
        let (storage, _) = izzydb::storage::Storage::open(dir.path(), &test_tree_cfg(4)).unwrap();
        let meta = izzydb::fts::FtsMeta::decode(&storage.get_fts_meta_bytes().unwrap().unwrap())
            .unwrap();
        assert_eq!(meta.next_fts_doc_id, 0);
        assert!(storage.get_fts_free_ids().unwrap().is_empty());
    }
}

#[test]
fn test_fts_compact_fts_stats_high_df_term() {
    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(test_tree_cfg(4)).with_fts_attribute(
        "body",
        AttributeType::String,
        false,
        FullTextSearchConfig::default(),
    );

    // > 64 blocks = 16_384+ docs? To keep unit test fast while testing >64 blocks cleanly,
    // note block_id = fts_doc_id >> 8. Even 16_640 docs in FTS takes ~150ms if we use tiny tree or batch?
    // Wait: tree with max_leaf_size=8 and 16_640 vectors would be slow to insert one-by-one.
    // Instead, test compact_fts_stats directly with max_leaf_size=256 on 16_640 docs? Still 16k RocksDB commits if 1-by-1.
    // Even better: test compact_fts_stats end-to-end on a multi-block corpus (e.g. 600 docs) to ensure it runs cleanly
    // and preserves exact TermStats & BM25 parity.
    let mut col = Collection::open(dir.path(), cfg).unwrap();
    for id in 0..300u64 {
        let mut attrs = Attributes::new();
        attrs.insert(
            "body".into(),
            if id == 0 { "rare rare rare common" } else { "common text" }.into(),
        );
        col.upsert(id, &[0.0; 4], attrs).unwrap();
    }
    col.delete(0).unwrap();
    col.compact_fts_stats().unwrap();
    col.verify_invariants().unwrap();
}

#[test]
fn test_fts_long_horizon_churn_with_patch_and_filter_mutations() {
    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(test_tree_cfg(8))
        .with_attribute("bucket", AttributeType::Uint, true)
        .with_attribute("active", AttributeType::Bool, true)
        .with_fts_attribute(
            "title",
            AttributeType::String,
            false,
            FullTextSearchConfig::default().with_stemming(true),
        )
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default()
                .with_stemming(true)
                .with_remove_stopwords(true),
        );

    let mut rng = SmallRng::seed_from_u64(55123);
    let vocab = [
        "vector", "search", "database", "rocksdb", "bm25", "maxscore", "rabitq", "cluster",
        "posting", "block", "prefix", "phrase", "hybrid", "ranking", "filter", "memory",
        "storage", "index", "token", "stemming", "alpha", "beta", "the", "and",
    ];
    let mut oracle: HashMap<DocId, (Vec<f32>, Attributes)> = HashMap::new();
    let mut col = Collection::open(dir.path(), cfg.clone()).unwrap();

    // Seed 320 docs across >1 FTS block (256 slots/block).
    for id in 0..320u64 {
        let vec = random_vec(&mut rng, 8);
        let mut attrs = Attributes::new();
        attrs.insert("bucket".into(), (id % 5).into());
        attrs.insert("active".into(), (id % 2 == 0).into());
        let title = (0..3)
            .map(|_| vocab[rng.random_range(0..vocab.len())])
            .collect::<Vec<_>>()
            .join(" ");
        let body = (0..8)
            .map(|_| vocab[rng.random_range(0..vocab.len())])
            .collect::<Vec<_>>()
            .join(" ");
        attrs.insert("title".into(), title.into());
        attrs.insert("body".into(), body.into());
        col.upsert(id, &vec, attrs.clone()).unwrap();
        oracle.insert(id, (vec, attrs));
    }

    for step in 0..300usize {
        let op = rng.random_range(0..100);
        if op < 30 {
            let id = rng.random_range(0..400u64);
            let vec = random_vec(&mut rng, 8);
            let mut attrs = Attributes::new();
            attrs.insert("bucket".into(), (id % 5).into());
            attrs.insert("active".into(), rng.random_bool(0.5).into());
            let title = (0..3)
                .map(|_| vocab[rng.random_range(0..vocab.len())])
                .collect::<Vec<_>>()
                .join(" ");
            let body = (0..6)
                .map(|_| vocab[rng.random_range(0..vocab.len())])
                .collect::<Vec<_>>()
                .join(" ");
            attrs.insert("title".into(), title.into());
            attrs.insert("body".into(), body.into());
            col.upsert(id, &vec, attrs.clone()).unwrap();
            oracle.insert(id, (vec, attrs));
        } else if op < 55 {
            if let Some(&id) = oracle.keys().nth(rng.random_range(0..oracle.len().max(1))) {
                let mut patch = DocumentPatch::new();
                if rng.random_bool(0.4) {
                    let nv = random_vec(&mut rng, 8);
                    patch.embedding = Some(nv.clone());
                    oracle.get_mut(&id).unwrap().0 = nv;
                }
                if rng.random_bool(0.6) {
                    let body = if rng.random_bool(0.15) {
                        "the and".to_string()
                    } else {
                        (0..5)
                            .map(|_| vocab[rng.random_range(0..vocab.len())])
                            .collect::<Vec<_>>()
                            .join(" ")
                    };
                    patch = patch.with_attribute("body", body.clone().into());
                    oracle
                        .get_mut(&id)
                        .unwrap()
                        .1
                        .insert("body".into(), body.into());
                } else {
                    patch = patch.clear_attribute("title");
                    oracle.get_mut(&id).unwrap().1.remove("title");
                }
                col.patch(id, patch).unwrap();
            }
        } else if op < 75 {
            let ids: Vec<DocId> = (0..4).map(|_| rng.random_range(0..400u64)).collect();
            let expected = ids
                .iter()
                .collect::<HashSet<_>>()
                .into_iter()
                .filter(|id| oracle.remove(id).is_some())
                .count() as u64;
            assert_eq!(col.delete_batch(&ids).unwrap(), expected);
        } else if op < 88 {
            let f = Filter::And(vec![
                Filter::Eq("bucket".into(), 4u64.into()),
                Filter::ContainsAnyToken {
                    attr: "body".into(),
                    query: "alpha".into(),
                    last_as_prefix: false,
                },
            ]);
            let fts_helper = izzydb::FtsIndex::new();
            let mut matched = 0u64;
            let schema_clone = col.schema().clone();
            for (_, attrs) in oracle.values_mut() {
                if fts_helper.matches_filter(&schema_clone, &f, attrs) {
                    attrs.insert("active".into(), false.into());
                    matched += 1;
                }
            }
            let got = col
                .update_by_filter(&f, &DocumentPatch::new().with_attribute("active", false.into()))
                .unwrap();
            assert_eq!(got, matched);
        } else if op < 95 {
            let f = Filter::ContainsAllTokens {
                attr: "title".into(),
                query: "beta gamma".into(),
                last_as_prefix: false,
            };
            let fts_helper = izzydb::FtsIndex::new();
            let schema_clone = col.schema().clone();
            let before = oracle.len();
            oracle.retain(|_, (_, attrs)| !fts_helper.matches_filter(&schema_clone, &f, attrs));
            let expected = (before - oracle.len()) as u64;
            assert_eq!(col.delete_by_filter(&f).unwrap(), expected);
        } else {
            drop(col);
            col = Collection::open(dir.path(), cfg.clone()).unwrap();
        }

        if step % 60 == 59 {
            let stats = col.verify_invariants().unwrap();
            assert_eq!(stats.num_docs, oracle.len());
            assert_eq!(stats.num_fts_docs, oracle.len());
        }
    }

    col.verify_invariants().unwrap();
    let fts_helper = izzydb::FtsIndex::new();
    let rank_by = RankBy::Sum(vec![
        RankBy::product(2.0, RankBy::bm25("title", "vector database")),
        RankBy::bm25("body", "vector search bm25 maxscore cluster"),
    ]);
    let filter = Filter::And(vec![
        Filter::Eq("active".into(), true.into()),
        Filter::Not(Box::new(Filter::Eq("bucket".into(), 0u64.into()))),
    ]);
    let got = col.query_bm25(&rank_by, Some(&filter), 10).unwrap();
    let expected = brute_force_bm25_eval(
        &oracle,
        col.schema(),
        &rank_by,
        Some(&filter),
        &fts_helper,
        10,
    );
    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_eq!(g.id, e.id);
        assert!((g.score - e.score).abs() <= 1e-4);
    }
}
