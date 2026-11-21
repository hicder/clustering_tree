use std::collections::HashSet;
use std::time::Instant;

use izzydb::metric::{l2_squared, prepare_vector};
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric, DocId, Filter,
    FullTextSearchConfig, HybridSearchOptions, RankBy, SearchOptions, TreeConfig,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn bench_high_dim_ann() {
    let n = 10_000usize;
    let dim = 768usize;
    let num_clusters = 100usize;
    let mut rng = SmallRng::seed_from_u64(101);

    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect())
        .collect();
    let data: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let c = &centers[i % num_clusters];
            c.iter()
                .map(|&x| x + rng.random_range(-0.15f32..0.15f32))
                .collect()
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            let c = &centers[i % num_clusters];
            c.iter()
                .map(|&x| x + rng.random_range(-0.15f32..0.15f32))
                .collect()
        })
        .collect();

    let dir = TempDir::new().unwrap();
    let mut tree_cfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    tree_cfg.enable_lire = false;
    let mut tree = izzydb::ClusteringTree::open(dir.path(), tree_cfg).unwrap();
    let t0 = Instant::now();
    for (i, v) in data.iter().enumerate() {
        tree.add_leaf(i as DocId, v).unwrap();
    }
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut ground_truth = Vec::with_capacity(queries.len());
    for q in &queries {
        let qp = prepare_vector(q, dim, DistanceMetric::L2Squared).unwrap();
        let mut scored: Vec<(DocId, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i as DocId, l2_squared(&qp, v)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(10);
        ground_truth.push(scored.into_iter().map(|(id, _)| id).collect::<HashSet<_>>());
    }

    println!("\n=== 1. High-Dim ANN (n={n}, dim={dim}, build={build_ms:.1}ms) ===");
    for beam in [4, 16, 32] {
        // Warmup
        let _ = tree.search(&queries[0], 10, beam).unwrap();
        let sq = Instant::now();
        let mut q_hits = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let res = tree.search(q, 10, beam).unwrap();
            q_hits += res
                .iter()
                .filter(|&&(id, _)| ground_truth[qi].contains(&id))
                .count();
        }
        let q_us = sq.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        let q_recall = q_hits as f64 / (queries.len() * 10) as f64;

        let sr = Instant::now();
        let mut r_hits = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let res = tree.search_with_rerank(q, 10, beam, 100).unwrap();
            r_hits += res
                .iter()
                .filter(|&&(id, _)| ground_truth[qi].contains(&id))
                .count();
        }
        let r_us = sr.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        let r_recall = r_hits as f64 / (queries.len() * 10) as f64;
        println!(
            "  beam={beam:2} -> rabitq={q_us:6.1}us (r@10={q_recall:.4}) | rerank@100={r_us:6.1}us (r@10={r_recall:.4})"
        );
    }
}

fn bench_filtered_and_range() {
    let n = 20_000usize;
    let dim = 128usize;
    let num_clusters = 150usize;
    let mut rng = SmallRng::seed_from_u64(42);

    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect())
        .collect();

    let dir = TempDir::new().unwrap();
    let mut tree_cfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    tree_cfg.enable_lire = true;
    let cfg = CollectionConfig::new(tree_cfg)
        .with_attribute("mod10", AttributeType::Uint, true)
        .with_attribute("even", AttributeType::Bool, true)
        .with_attribute("opt_flag", AttributeType::Uint, true)
        .with_attribute("cont_score", AttributeType::Float, true);

    let mut col = Collection::open(dir.path(), cfg).unwrap();
    let t0 = Instant::now();
    for i in 0..n {
        let c = &centers[i % num_clusters];
        let v: Vec<f32> = c
            .iter()
            .map(|&x| x + rng.random_range(-0.2f32..0.2f32))
            .collect();
        let mut attrs = Attributes::new();
        attrs.insert("mod10".into(), ((i % 10) as u64).into());
        attrs.insert("even".into(), (i % 2 == 0).into());
        if i % 5 != 0 {
            attrs.insert("opt_flag".into(), ((i % 4) as u64).into());
        }
        // High-cardinality continuous float: 20,000 distinct values in [0, 1000)
        let cont = rng.random_range(0.0f64..1000.0);
        attrs.insert("cont_score".into(), cont.into());
        col.upsert(i as DocId, &v, attrs).unwrap();
    }
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let queries: Vec<Vec<f32>> = (0..50)
        .map(|i| {
            let c = &centers[i % num_clusters];
            c.iter()
                .map(|&x| x + rng.random_range(-0.2f32..0.2f32))
                .collect()
        })
        .collect();

    let cases: Vec<(&str, Filter)> = vec![
        ("50% Eq(even=true)", Filter::Eq("even".into(), true.into())),
        ("10% Eq(mod10=3)", Filter::Eq("mod10".into(), 3u64.into())),
        (
            "90% Not(Eq(mod10=3))",
            Filter::Not(Box::new(Filter::Eq("mod10".into(), 3u64.into()))),
        ),
        ("20% IsNull(opt_flag)", Filter::IsNull("opt_flag".into())),
        (
            "10% HiCard Lt(cont_score<100)",
            Filter::Lt("cont_score".into(), 100.0f64.into()),
        ),
        (
            "20% HiCard Range[200,400]",
            Filter::And(vec![
                Filter::Gte("cont_score".into(), 200.0f64.into()),
                Filter::Lte("cont_score".into(), 400.0f64.into()),
            ]),
        ),
    ];

    println!("\n=== 2. Filtered & High-Cardinality Range ANN (n={n}, dim={dim}, build={build_ms:.1}ms) ===");
    let opts = SearchOptions::new(10, 8)
        .with_rerank(100)
        .with_max_expanded_leaves(256);
    for (label, filter) in &cases {
        let _ = col.search(&queries[0], opts, Some(filter)).unwrap();
        let sq = Instant::now();
        let mut count_sum = 0usize;
        for q in &queries {
            let res = col.search(q, opts, Some(filter)).unwrap();
            count_sum += res.len();
        }
        let avg_us = sq.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        println!("  {label:30} -> avg={avg_us:7.1}us (avg_hits={:.1})", count_sum as f64 / queries.len() as f64);
    }
}

fn bench_fts_and_hybrid() {
    let n = 40_000usize;
    let dim = 64usize;
    let vocab_size = 2_000usize;
    let mut rng = SmallRng::seed_from_u64(777);
    let vocab: Vec<String> = (0..vocab_size).map(|i| format!("word{i:04}")).collect();
    let rare_terms: Vec<String> = (0..50).map(|i| format!("rare{i:03}")).collect();

    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(TreeConfig::new(dim, DistanceMetric::L2Squared))
        .with_attribute("bucket", AttributeType::Uint, true)
        .with_attribute("tag", AttributeType::String, true)
        .with_fts_attribute(
            "title",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        )
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        );
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    let t0 = Instant::now();
    for id in 0..n as DocId {
        let vec: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
        let title = format!(
            "{} {} {}",
            vocab[rng.random_range(0..100)],
            vocab[rng.random_range(0..vocab.len())],
            vocab[rng.random_range(0..vocab.len())]
        );
        let body_len = rng.random_range(40..=80);
        let mut words: Vec<&str> = (0..body_len)
            .map(|_| {
                let idx = if rng.random_bool(0.45) {
                    rng.random_range(0..60)
                } else {
                    rng.random_range(0..vocab.len())
                };
                vocab[idx].as_str()
            })
            .collect();
        if id % 100 == 0 {
            // Embed an exact 2-word phrase and rare terms periodically
            words[5] = "word0001";
            words[6] = "word0002";
        }
        let mut body = words.join(" ");
        if id < 5 {
            body.push(' ');
            body.push_str(&rare_terms[0]);
            body.push(' ');
            body.push_str(&rare_terms[1]);
        }

        let mut attrs = Attributes::new();
        attrs.insert("bucket".into(), (id % 10).into());
        attrs.insert("tag".into(), format!("tag_{}", id % 20).into());
        attrs.insert("title".into(), title.into());
        attrs.insert("body".into(), body.into());
        col.upsert(id, &vec, attrs).unwrap();
    }
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\n=== 3. FTS, Lazy-Filter BM25, Phrase & Hybrid (n={n}, vocab={vocab_size}, build={build_ms:.1}ms) ===");

    let runs = 25usize;
    let q3 = format!("{} {} {}", vocab[3], vocab[17], vocab[42]);
    let q8 = (0..8)
        .map(|i| vocab[(i * 19 + 5) % vocab.len()].as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let q30 = (0..30)
        .map(|i| vocab[(i * 29 + 7) % vocab.len()].as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let q_rare = format!("{} {}", rare_terms[0], rare_terms[1]);

    let bm25_cases: Vec<(&str, RankBy, Option<Filter>)> = vec![
        ("BM25 3-term", RankBy::bm25("body", q3.as_str()), None),
        ("BM25 8-term", RankBy::bm25("body", q8.as_str()), None),
        ("BM25 30-term", RankBy::bm25("body", q30.as_str()), None),
        ("BM25 rare (df<=5)", RankBy::bm25("body", q_rare.as_str()), None),
        (
            "BM25 lazy-filter Not(bucket=0)",
            RankBy::bm25("body", q8.as_str()),
            Some(Filter::Not(Box::new(Filter::Eq("bucket".into(), 0u64.into())))),
        ),
    ];

    for (label, rank_by, filter) in &bm25_cases {
        let _ = col.query_bm25(rank_by, filter.as_ref(), 10).unwrap();
        let t = Instant::now();
        let mut hits = 0usize;
        for _ in 0..runs {
            hits = col.query_bm25(rank_by, filter.as_ref(), 10).unwrap().len();
        }
        let avg_us = t.elapsed().as_secs_f64() * 1e6 / runs as f64;
        println!("  {label:32} -> avg={avg_us:7.1}us (hits={hits})");
    }

    // FTS-filtered vector search & Phrase search & Hybrid
    let qvec: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
    let v_opts = SearchOptions::new(10, 8).with_rerank(100);
    let fts_vec_cases: Vec<(&str, Filter)> = vec![
        (
            "Vec + ContainsAllTokens(2)",
            Filter::ContainsAllTokens {
                attr: "body".into(),
                query: "word0001 word0002".into(),
                last_as_prefix: false,
            },
        ),
        (
            "Vec + ContainsTokenSequence(2)",
            Filter::ContainsTokenSequence {
                attr: "body".into(),
                query: "word0001 word0002".into(),
                last_as_prefix: false,
            },
        ),
    ];
    for (label, filter) in &fts_vec_cases {
        let _ = col.search(&qvec, v_opts, Some(filter)).unwrap();
        let t = Instant::now();
        let mut hits = 0usize;
        for _ in 0..runs {
            hits = col.search(&qvec, v_opts, Some(filter)).unwrap().len();
        }
        let avg_us = t.elapsed().as_secs_f64() * 1e6 / runs as f64;
        println!("  {label:32} -> avg={avg_us:7.1}us (hits={hits})");
    }

    let hyb_opts = HybridSearchOptions::new(10, SearchOptions::new(20, 8).with_rerank(50));
    let hyb_rank = RankBy::bm25("body", q8.as_str());
    let hyb_filter = Filter::Not(Box::new(Filter::Eq("bucket".into(), 0u64.into())));
    let _ = col
        .hybrid_search(&qvec, &hyb_rank, hyb_opts, Some(&hyb_filter))
        .unwrap();
    let t_hyb = Instant::now();
    let mut hyb_hits = 0usize;
    for _ in 0..runs {
        hyb_hits = col
            .hybrid_search(&qvec, &hyb_rank, hyb_opts, Some(&hyb_filter))
            .unwrap()
            .len();
    }
    let hyb_us = t_hyb.elapsed().as_secs_f64() * 1e6 / runs as f64;
    println!("  {:32} -> avg={hyb_us:7.1}us (hits={hyb_hits})", "Hybrid (vec+BM25+Not)");
}

fn main() {
    bench_high_dim_ann();
    bench_filtered_and_range();
    bench_fts_and_hybrid();
}
