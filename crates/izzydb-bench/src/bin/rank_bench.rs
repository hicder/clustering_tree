//! Latency of ranking/boosting features relative to plain BM25 (`cargo run --release --bin rank_bench`).
//!
//! Verifies plain BM25 is not regressed by the generalized executor and reports the cost of
//! filter boosts, multiplicative / additive attribute scoring, and index-order `OrderBy`.

use std::time::Instant;

use izzydb::{
    AttributeType, AttributeValue, Attributes, Collection, CollectionConfig, DecayMode,
    DistanceMetric, DocId, Filter, FullTextSearchConfig, OrderBySpec, RankBy, ScalarValue,
    TokenQuery, TreeConfig,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn time_us<T>(runs: usize, mut f: impl FnMut() -> T) -> f64 {
    let _ = f(); // warmup
    let t = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(f());
    }
    t.elapsed().as_secs_f64() * 1e6 / runs as f64
}

fn main() {
    let n = 20_000usize;
    let dim = 16usize;
    let mut rng = SmallRng::seed_from_u64(1);
    let vocab: Vec<String> = (0..1000).map(|i| format!("term{i:04}")).collect();

    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(TreeConfig::new(dim, DistanceMetric::L2Squared))
        .with_attribute("created", AttributeType::Int, true)
        .with_attribute("popularity", AttributeType::Uint, true)
        .with_attribute("cat", AttributeType::String, true)
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        );
    let mut col = Collection::open(dir.path(), cfg).unwrap();
    let t0 = Instant::now();
    for id in 0..n as DocId {
        let v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        let len = rng.random_range(15..40);
        let text: Vec<&str> = (0..len)
            .map(|_| {
                let i = if rng.random_bool(0.5) {
                    rng.random_range(0..50)
                } else {
                    rng.random_range(0..vocab.len())
                };
                vocab[i].as_str()
            })
            .collect();
        let mut a = Attributes::new();
        a.insert("body".into(), text.join(" ").into());
        a.insert("created".into(), AttributeValue::Int(rng.random_range(0..1_000_000)));
        a.insert("popularity".into(), AttributeValue::Uint(rng.random_range(0..100_000)));
        a.insert("cat".into(), format!("c{}", rng.random_range(0..20)).into());
        col.upsert(id, &v, a).unwrap();
    }
    println!("indexed {n} docs in {:.0}ms", t0.elapsed().as_secs_f64() * 1e3);

    let q = format!("{} {} {}", vocab[5], vocab[19], vocab[300]);
    let plain = RankBy::bm25("body", q.as_str());
    let phrase = Filter::ContainsTokenSequence {
        attr: "body".into(),
        query: TokenQuery::Text(format!("{} {}", vocab[5], vocab[19])),
        last_as_prefix: false,
    };
    let runs = 30;
    let k = 10;

    let cases: Vec<(&str, RankBy)> = vec![
        ("BM25 (baseline)", plain.clone()),
        ("2 x BM25 (constant product)", RankBy::product(2.0, plain.clone())),
        (
            "BM25 + 3 x Filter(phrase)",
            RankBy::Sum(vec![
                plain.clone(),
                RankBy::product(3.0, RankBy::filter(phrase.clone())),
            ]),
        ),
        (
            "BM25 x (1 + 2 x Decay(created))",
            RankBy::boost_multiplicative(
                plain.clone(),
                2.0,
                RankBy::decay_from("created", 1_000_000.0, 100_000.0)
                    .with_decay_mode(DecayMode::Backward),
            ),
        ),
        (
            "BM25 x Saturate(popularity)",
            RankBy::Product(vec![plain.clone(), RankBy::saturate("popularity", 10_000.0)]),
        ),
        (
            "BM25 + 0.05 x Decay(created) (backfill)",
            RankBy::Sum(vec![
                plain.clone(),
                RankBy::product(
                    0.05,
                    RankBy::decay_from("created", 1_000_000.0, 100_000.0)
                        .with_decay_mode(DecayMode::Backward),
                ),
            ]),
        ),
        (
            "BM25 x BM25 (conjunctive)",
            RankBy::Product(vec![plain.clone(), RankBy::bm25("body", vocab[19].as_str())]),
        ),
        ("Saturate(popularity) (monotone index scan)", RankBy::saturate("popularity", 10_000.0)),
        ("Sum(Saturate, Filter) (candidate scan)", RankBy::Sum(vec![
            RankBy::saturate("popularity", 10_000.0),
            RankBy::filter(Filter::Eq("cat".into(), ScalarValue::String("c3".into()))),
        ])),
    ];
    println!("\ntop-{k} latency over {n} docs (avg of {runs} runs):");
    let mut baseline = 0.0;
    for (i, (label, rb)) in cases.iter().enumerate() {
        let runs = if label.contains("candidate scan") || label.contains("conjunctive") { 5 } else { runs };
        let us = time_us(runs, || col.query(rb, None, k).unwrap());
        if i == 0 {
            baseline = us;
        }
        println!("  {label:44} {us:10.1} us  ({:5.1}x baseline)", us / baseline);
    }

    println!("\nOrderBy top-{k}:");
    let cat_filter = Filter::Eq("cat".into(), ScalarValue::String("c3".into()));
    let ob_cases: Vec<(&str, Vec<OrderBySpec>, Option<&Filter>)> = vec![
        ("created DESC (unfiltered)", vec![OrderBySpec::desc("created")], None),
        ("created ASC (unfiltered)", vec![OrderBySpec::asc("created")], None),
        (
            "popularity DESC, created ASC",
            vec![OrderBySpec::desc("popularity"), OrderBySpec::asc("created")],
            None,
        ),
        ("created DESC, cat = c3", vec![OrderBySpec::desc("created")], Some(&cat_filter)),
    ];
    for (label, specs, f) in &ob_cases {
        let us = time_us(runs, || col.query_ordered(specs, *f, k).unwrap());
        println!("  {label:44} {us:10.1} us");
    }
}
