//! Faceting latency benchmark: unfiltered (`default[0x06]`), selective (`DocView` pass), broad
//! (inverted leaf masks), complement, and disjunctive BM25 + vector faceted queries.
//!
//! Run with `cargo run --release --example facet_bench`.

use std::time::Instant;

use izzydb::facet::{
    FacetRange, FacetSpec, HybridFacetScope, VectorFacetScope, take_strategy_trace,
};
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric, DocId, Filter,
    FullTextSearchConfig, HybridSearchOptions, RankBy, ScalarValue, SearchOptions, TreeConfig,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

const WORDS: [&str; 12] = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu",
];

fn time<T>(label: &str, iters: usize, mut f: impl FnMut() -> T) -> T {
    let mut out = f(); // warm-up
    let t0 = Instant::now();
    for _ in 0..iters {
        out = f();
    }
    let per = t0.elapsed() / iters as u32;
    println!("{label:<52} {per:>12.2?}");
    out
}

fn main() {
    let n = 20_000usize;
    let dim = 64usize;
    let mut rng = SmallRng::seed_from_u64(7);

    let mut tree_cfg = TreeConfig::new(dim, DistanceMetric::L2Squared);
    tree_cfg.enable_lire = true;
    let cfg = CollectionConfig::new(tree_cfg)
        .with_attribute("brand", AttributeType::String, true)
        .with_attribute("tags", AttributeType::StringArray, true)
        .with_attribute("price", AttributeType::Float, true)
        .with_attribute("rating", AttributeType::Int, true)
        .with_attribute("in_stock", AttributeType::Bool, true)
        .with_attribute("status", AttributeType::String, true)
        .with_fts_attribute("title", AttributeType::String, false, FullTextSearchConfig::default());

    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    let t0 = Instant::now();
    for i in 0..n {
        let v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
        let mut a = Attributes::new();
        a.insert("brand".into(), format!("brand{}", i % 50).into());
        a.insert(
            "tags".into(),
            (0..3)
                .map(|_| format!("tag{}", rng.random_range(0..200)))
                .collect::<Vec<_>>()
                .into(),
        );
        a.insert("price".into(), rng.random_range(1.0f64..500.0).into());
        a.insert("rating".into(), (rng.random_range(1i64..=5)).into());
        a.insert("in_stock".into(), rng.random_bool(0.8).into());
        // `deleted` is rare: complement facets over `status != "deleted"` hit global counts.
        a.insert(
            "status".into(),
            if i % 2000 == 0 { "deleted" } else { "live" }.into(),
        );
        let title: Vec<&str> = (0..6).map(|_| WORDS[rng.random_range(0..WORDS.len())]).collect();
        a.insert("title".into(), title.join(" ").into());
        col.upsert(i as DocId, &v, a).unwrap();
    }
    println!("indexed {n} docs in {:?}\n", t0.elapsed());

    let q: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
    let facets = vec![
        FacetSpec::terms("brand").with_limit(10).exclude_self(),
        FacetSpec::terms("tags").with_limit(10),
        FacetSpec::ranges(
            "price",
            vec![
                FacetRange::below(50.0f64),
                FacetRange::between(50.0f64, 200.0f64),
                FacetRange::from_value(200.0f64),
            ],
        ),
        FacetSpec::stats("rating"),
        FacetSpec::histogram("price", 50.0).with_alias("price_hist"),
    ];

    // ~10 documents (`i % 2000 == 0`) spread over <= 16 leaves -> forward `DocView` pass.
    let selective = Filter::And(vec![
        Filter::Eq("status".into(), "deleted".into()),
        Filter::Gte("rating".into(), ScalarValue::Int(1)),
    ]);
    let broad = Filter::Eq("in_stock".into(), true.into());
    let complement = Filter::NotEq("status".into(), "deleted".into());
    let facets_no_self: Vec<FacetSpec> = facets
        .iter()
        .filter(|s| s.attr != "brand")
        .cloned()
        .collect();

    println!("{:<52} {:>12}", "scenario", "latency");
    println!("{}", "-".repeat(66));
    for (label, filter, specs) in [
        ("unfiltered terms+stats (default[0x06])", None, &facets_no_self),
        ("complement status != deleted (global - inner)", Some(&complement), &facets_no_self),
        ("selective filter (forward DocView pass)", Some(&selective), &facets_no_self),
        ("broad filter in_stock (inverted leaf masks)", Some(&broad), &facets_no_self),
        ("broad filter + disjunctive ExcludeSelf(brand)", Some(&broad), &facets),
    ] {
        take_strategy_trace();
        let res = time(label, 20, || col.facet(filter, specs).unwrap());
        let strategies = take_strategy_trace();
        let mut uniq: Vec<_> = strategies.clone();
        uniq.sort_unstable();
        uniq.dedup();
        println!("{:<52} {:?}  ({} facets)", "", uniq, res.facets.len());
    }

    println!();
    let opts = SearchOptions::new(10, 8).with_rerank(50);
    time("vector search only (filter in_stock)", 20, || {
        col.search(&q, opts, Some(&broad)).unwrap()
    });
    time("vector + facets FilterMatches (in_stock)", 20, || {
        col.search_with_facets(&q, opts, Some(&broad), &facets, VectorFacetScope::FilterMatches)
            .unwrap()
    });
    time("vector + facets TopK", 20, || {
        col.search_with_facets(&q, opts, Some(&broad), &facets, VectorFacetScope::TopK)
            .unwrap()
    });

    let rank_by = RankBy::bm25("title", "alpha gamma");
    time("bm25 only (filter in_stock)", 20, || {
        col.query_bm25(&rank_by, Some(&broad), 10).unwrap()
    });
    time("bm25 + disjunctive facets (in_stock)", 20, || {
        col.query_bm25_with_facets(&rank_by, Some(&broad), 10, &facets).unwrap()
    });

    let hopts = HybridSearchOptions::new(10, opts);
    time("hybrid + facets Bm25AndFilterMatches", 20, || {
        col.hybrid_search_with_facets(
            &q,
            &rank_by,
            hopts,
            Some(&broad),
            &facets,
            HybridFacetScope::Bm25AndFilterMatches,
        )
        .unwrap()
    });
}
