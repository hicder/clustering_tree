//! Exactness tests for filter boosts, attribute scoring, and general `Product` ranking against a
//! brute-force reference evaluator.

use std::collections::{HashMap, HashSet};

use izzydb::rank::{eval_decay, eval_saturate};
use izzydb::{
    AttributeType, AttributeValue, Attributes, Bm25Result, Collection, CollectionConfig,
    DecayMode, DistanceMetric, DocId, DocumentPatch, FacetSpec, Filter,
    FullTextSearchConfig, HybridFacetScope, HybridSearchOptions, RankBy, ScalarValue,
    SearchOptions, TokenQuery, TreeConfig, TreeError,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

const N: u64 = 150;

fn tree_cfg() -> TreeConfig {
    TreeConfig {
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
    }
}

fn config() -> CollectionConfig {
    CollectionConfig::new(tree_cfg())
        .with_attribute("cat", AttributeType::String, true)
        .with_attribute("price", AttributeType::Float, true)
        .with_attribute("views", AttributeType::Uint, true)
        .with_attribute("ts", AttributeType::Int, true)
        .with_attribute("plain", AttributeType::Float, false)
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        )
        .with_fts_attribute(
            "title",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        )
}

const WORDS: [&str; 8] = [
    "alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa",
];

fn build(dir: &TempDir) -> (Collection, HashMap<DocId, Attributes>) {
    let mut col = Collection::open(dir.path(), config()).unwrap();
    let mut rng = SmallRng::seed_from_u64(7);
    let mut docs = HashMap::new();
    for id in 0..N {
        let mut attrs = Attributes::new();
        attrs.insert("cat".into(), ["a", "b", "c"][(id % 3) as usize].into());
        if id % 11 != 0 {
            attrs.insert("price".into(), (rng.random_range(0.0f64..100.0)).into());
        }
        if id % 7 != 0 {
            attrs.insert("views".into(), AttributeValue::Uint(rng.random_range(0..5000)));
        }
        attrs.insert("ts".into(), AttributeValue::Int(rng.random_range(-500..2000)));
        if id % 5 != 0 {
            attrs.insert("plain".into(), (rng.random_range(0.0f64..50.0)).into());
        }
        let nb = rng.random_range(2..9);
        let body: Vec<&str> = (0..nb).map(|_| WORDS[rng.random_range(0..8)]).collect();
        attrs.insert("body".into(), body.join(" ").into());
        if id % 4 != 0 {
            let nt = rng.random_range(1..4);
            let title: Vec<&str> = (0..nt).map(|_| WORDS[rng.random_range(0..8)]).collect();
            attrs.insert("title".into(), title.join(" ").into());
        }
        col.upsert(id, &[0.0, 0.0, 0.0, 0.0], attrs.clone()).unwrap();
        docs.insert(id, attrs);
    }
    (col, docs)
}

// ---------------------------------------------------------------------------------------
// Brute-force reference
// ---------------------------------------------------------------------------------------

struct Reference<'a> {
    col: &'a Collection,
    docs: &'a HashMap<DocId, Attributes>,
    lex: HashMap<String, HashMap<DocId, f32>>,
    filters: HashMap<String, HashSet<DocId>>,
}

impl<'a> Reference<'a> {
    fn new(col: &'a Collection, docs: &'a HashMap<DocId, Attributes>) -> Self {
        Self {
            col,
            docs,
            lex: HashMap::new(),
            filters: HashMap::new(),
        }
    }

    fn numeric(&self, id: DocId, attr: &str) -> Option<f64> {
        match self.docs[&id].get(attr)? {
            AttributeValue::Int(v) => Some(*v as f64),
            AttributeValue::Uint(v) => Some(*v as f64),
            AttributeValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    fn eval(&mut self, e: &RankBy, id: DocId) -> f32 {
        if e.is_pure_lexical() {
            let key = format!("{e:?}");
            if !self.lex.contains_key(&key) {
                let hits = self.col.query_bm25(e, None, N as usize + 10).unwrap();
                self.lex
                    .insert(key.clone(), hits.into_iter().map(|h| (h.id, h.score)).collect());
            }
            return self.lex[&key].get(&id).copied().unwrap_or(0.0);
        }
        match e {
            RankBy::Const(c) => *c,
            RankBy::Bm25 { .. } => unreachable!(),
            RankBy::Filter(f) => {
                let key = format!("{f:?}");
                if !self.filters.contains_key(&key) {
                    let ids: HashSet<DocId> =
                        self.col.resolve_filter_doc_ids(f).unwrap().into_iter().collect();
                    self.filters.insert(key.clone(), ids);
                }
                if self.filters[&key].contains(&id) { 1.0 } else { 0.0 }
            }
            RankBy::Attribute { attr, missing } => match self.numeric(id, attr) {
                Some(v) => v.max(0.0) as f32,
                None => *missing,
            },
            RankBy::Saturate {
                attr,
                midpoint,
                exponent,
                missing,
            } => match self.numeric(id, attr) {
                Some(v) => eval_saturate(v, *midpoint, *exponent),
                None => *missing,
            },
            RankBy::Decay {
                attr,
                origin,
                midpoint,
                exponent,
                mode,
                missing,
            } => match self.numeric(id, attr) {
                Some(v) => eval_decay(v, *origin, *midpoint, *exponent, *mode),
                None => *missing,
            },
            RankBy::Sum(c) => c.iter().map(|x| self.eval(x, id)).sum(),
            RankBy::Max(c) => c.iter().map(|x| self.eval(x, id)).fold(0.0, f32::max),
            RankBy::Product(c) => c.iter().map(|x| self.eval(x, id)).product(),
        }
    }

    /// Full reference scores (score > 0) for docs matching `filter`.
    fn scores(&mut self, e: &RankBy, filter: Option<&Filter>) -> HashMap<DocId, f32> {
        let allowed: Option<HashSet<DocId>> = filter
            .map(|f| self.col.resolve_filter_doc_ids(f).unwrap().into_iter().collect());
        let mut out = HashMap::new();
        for id in 0..N {
            if allowed.as_ref().is_some_and(|a| !a.contains(&id)) {
                continue;
            }
            let s = self.eval(e, id);
            if s > 0.0 {
                out.insert(id, s);
            }
        }
        out
    }
}

fn tol(s: f32) -> f32 {
    1e-4 * s.abs().max(1.0)
}

fn assert_matches_reference(
    label: &str,
    actual: &[Bm25Result],
    reference: &HashMap<DocId, f32>,
    k: usize,
    strict: bool,
) {
    let mut expected: Vec<(DocId, f32)> = reference.iter().map(|(&i, &s)| (i, s)).collect();
    expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    expected.truncate(k);
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: result count (k={k})\nactual={actual:?}\nexpected={expected:?}"
    );
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a.score - e.1).abs() <= tol(e.1),
            "{label}: score at rank {i}: actual {a:?} expected {e:?}"
        );
        if strict {
            // Non-lexical scores are computed with identical f32 operations: exact order.
            assert_eq!(a.id, e.0, "{label}: id at rank {i} (strict)\nactual={actual:?}\nexpected={expected:?}");
            assert_eq!(a.score, e.1, "{label}: score at rank {i} (strict)");
        } else if a.id != e.0 {
            // Allowed only for float near-ties.
            let ref_score = reference.get(&a.id).copied().unwrap_or(-1.0);
            assert!(
                (ref_score - e.1).abs() <= tol(e.1),
                "{label}: id mismatch at rank {i}: actual {a:?} expected {e:?} (ref {ref_score})"
            );
        }
    }
    for w in actual.windows(2) {
        assert!(w[0].score >= w[1].score - tol(w[0].score), "{label}: not sorted");
    }
}

fn check(col: &Collection, docs: &HashMap<DocId, Attributes>, label: &str, e: &RankBy) {
    let filters: Vec<Option<Filter>> = vec![
        None,
        Some(Filter::Eq("cat".into(), ScalarValue::String("b".into()))),
        Some(Filter::Gt("price".into(), ScalarValue::Float(40.0))),
        Some(Filter::ContainsAnyToken {
            attr: "body".into(),
            query: TokenQuery::Text("omega theta".into()),
            last_as_prefix: false,
        }),
    ];
    let mut reference = Reference::new(col, docs);
    for filter in &filters {
        let scores = reference.scores(e, filter.as_ref());
        for k in [1usize, 3, 10, 400] {
            let actual = col.query(e, filter.as_ref(), k).unwrap();
            assert_matches_reference(
                &format!("{label} filter={filter:?}"),
                &actual,
                &scores,
                k,
                !e.has_bm25(),
            );
        }
    }
}

fn bm(attr: &str, q: &str) -> RankBy {
    RankBy::bm25(attr, q)
}

fn phrase(attr: &str, q: &str) -> Filter {
    Filter::ContainsTokenSequence {
        attr: attr.into(),
        query: TokenQuery::Text(q.into()),
        last_as_prefix: false,
    }
}

// ---------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------

#[test]
fn filter_as_boost() {
    let dir = TempDir::new().unwrap();
    let (col, docs) = build(&dir);

    // Attribute predicate boost (additive, includes non-lexical docs via backfill).
    check(
        &col,
        &docs,
        "additive attr filter boost",
        &RankBy::Sum(vec![
            bm("body", "alpha"),
            RankBy::product(
                3.0,
                RankBy::filter(Filter::Eq("cat".into(), ScalarValue::String("a".into()))),
            ),
        ]),
    );
    // Phrase boost over a different FTS attribute.
    check(
        &col,
        &docs,
        "phrase boost",
        &RankBy::Sum(vec![
            bm("body", "alpha beta"),
            RankBy::product(2.0, RankBy::filter(phrase("title", "alpha beta"))),
        ]),
    );
    // Boolean tree filter boost, tiny weight (should leave most lexical order intact).
    check(
        &col,
        &docs,
        "boolean tree boost",
        &RankBy::Sum(vec![
            bm("body", "gamma delta"),
            RankBy::product(
                0.05,
                RankBy::filter(Filter::Or(vec![
                    Filter::Lt("price".into(), ScalarValue::Float(20.0)),
                    Filter::Not(Box::new(Filter::Eq(
                        "cat".into(),
                        ScalarValue::String("c".into()),
                    ))),
                ])),
            ),
        ]),
    );
    // Standalone filter score (non-lexical): matching docs score 1.0, ties by DocId.
    check(
        &col,
        &docs,
        "filter only",
        &RankBy::filter(Filter::Eq("cat".into(), ScalarValue::String("c".into()))),
    );
    // Multiplicative filter gate.
    check(
        &col,
        &docs,
        "bm25 x filter",
        &RankBy::Product(vec![
            bm("body", "omega"),
            RankBy::filter(Filter::Gte("views".into(), ScalarValue::Uint(1000))),
        ]),
    );
}

#[test]
fn attribute_scoring() {
    let dir = TempDir::new().unwrap();
    let (col, docs) = build(&dir);

    check(&col, &docs, "attribute", &RankBy::attribute("price"));
    check(&col, &docs, "attribute missing", &RankBy::attribute_with_missing("views", 25.0));
    check(&col, &docs, "attribute negatives clamp", &RankBy::attribute("ts"));
    check(&col, &docs, "saturate", &RankBy::saturate("views", 800.0));
    check(
        &col,
        &docs,
        "saturate exp",
        &RankBy::saturate("views", 800.0).with_exponent(3.0).with_missing(0.2),
    );
    check(&col, &docs, "decay forward", &RankBy::decay("price", 20.0));
    check(
        &col,
        &docs,
        "decay forward origin",
        &RankBy::decay_from("price", 30.0, 10.0).with_decay_mode(DecayMode::Forward),
    );
    check(
        &col,
        &docs,
        "decay backward",
        &RankBy::decay_from("ts", 1500.0, 200.0).with_decay_mode(DecayMode::Backward),
    );
    check(
        &col,
        &docs,
        "decay symmetric",
        &RankBy::decay_from("price", 50.0, 10.0).with_exponent(2.0),
    );
    check(
        &col,
        &docs,
        "decay missing",
        &RankBy::decay("price", 20.0).with_missing(0.4),
    );
    check(
        &col,
        &docs,
        "weighted saturate",
        &RankBy::product(2.5, RankBy::saturate("price", 30.0)),
    );
    // Non-filterable attribute: must work through the exact scan path.
    check(&col, &docs, "non-filterable attribute", &RankBy::attribute("plain"));
    check(&col, &docs, "non-filterable decay", &RankBy::decay("plain", 10.0));
    check(
        &col,
        &docs,
        "sum of signals",
        &RankBy::Sum(vec![
            RankBy::saturate("views", 500.0),
            RankBy::product(
                0.5,
                RankBy::filter(Filter::Eq("cat".into(), ScalarValue::String("a".into()))),
            ),
            RankBy::decay("price", 30.0),
        ]),
    );
    check(
        &col,
        &docs,
        "max of signals",
        &RankBy::Max(vec![
            RankBy::saturate("views", 500.0),
            RankBy::decay("price", 30.0),
        ]),
    );
    check(
        &col,
        &docs,
        "product of signals",
        &RankBy::Product(vec![
            RankBy::saturate("views", 500.0),
            RankBy::decay_from("ts", 1000.0, 400.0),
        ]),
    );
}

#[test]
fn additive_bm25_plus_attribute() {
    let dir = TempDir::new().unwrap();
    let (col, docs) = build(&dir);

    // Decay is > 0 for every document: non-lexical docs must backfill the top-k.
    check(
        &col,
        &docs,
        "bm25 + decay",
        &RankBy::Sum(vec![
            bm("body", "alpha"),
            RankBy::product(2.0, RankBy::decay("price", 25.0)),
        ]),
    );
    // Small decay weight: lexical docs dominate, backfill is provably unnecessary.
    check(
        &col,
        &docs,
        "bm25 + small decay",
        &RankBy::Sum(vec![
            bm("body", "alpha beta gamma"),
            RankBy::product(0.001, RankBy::decay("price", 25.0)),
        ]),
    );
    check(
        &col,
        &docs,
        "bm25 + saturate + const",
        &RankBy::Sum(vec![
            bm("title", "omega"),
            bm("body", "sigma"),
            RankBy::saturate("views", 1000.0),
            RankBy::Const(0.25),
        ]),
    );
    // Rare lexical match (fewer than k hits).
    check(
        &col,
        &docs,
        "rare bm25 + attribute",
        &RankBy::Sum(vec![bm("title", "kappa theta omega"), RankBy::attribute("price")]),
    );
}

#[test]
fn multiplicative_boosting() {
    let dir = TempDir::new().unwrap();
    let (col, docs) = build(&dir);

    check(
        &col,
        &docs,
        "bm25 x (1 + w decay)",
        &RankBy::boost_multiplicative(
            bm("body", "alpha beta"),
            2.0,
            RankBy::decay_from("ts", 2000.0, 300.0).with_decay_mode(DecayMode::Backward),
        ),
    );
    check(
        &col,
        &docs,
        "bm25 x saturate",
        &RankBy::Product(vec![bm("body", "gamma"), RankBy::saturate("views", 700.0)]),
    );
    check(
        &col,
        &docs,
        "bm25 x attribute (filterable)",
        &RankBy::Product(vec![bm("body", "delta omega"), RankBy::attribute("price")]),
    );
    // Non-filterable attribute has no index-derived bound: exact fallback.
    check(
        &col,
        &docs,
        "bm25 x attribute (non-filterable)",
        &RankBy::Product(vec![bm("body", "delta omega"), RankBy::attribute("plain")]),
    );
    check(
        &col,
        &docs,
        "bm25 x phrase filter",
        &RankBy::Product(vec![
            bm("body", "alpha"),
            RankBy::Sum(vec![
                RankBy::Const(1.0),
                RankBy::product(4.0, RankBy::filter(phrase("body", "alpha beta"))),
            ]),
        ]),
    );
    check(
        &col,
        &docs,
        "weighted multi-field x saturate",
        &RankBy::Product(vec![
            RankBy::Sum(vec![
                RankBy::product(2.0, bm("title", "alpha")),
                bm("body", "alpha"),
            ]),
            RankBy::saturate("views", 600.0),
        ]),
    );
    check(
        &col,
        &docs,
        "max x decay",
        &RankBy::Product(vec![
            RankBy::Max(vec![bm("title", "omega"), bm("body", "omega")]),
            RankBy::decay("price", 50.0),
        ]),
    );
    // Conjunctive lexical product.
    check(
        &col,
        &docs,
        "bm25 x bm25",
        &RankBy::Product(vec![bm("body", "alpha"), bm("title", "beta")]),
    );
    check(
        &col,
        &docs,
        "(bm25 + bm25) x bm25 x saturate",
        &RankBy::Product(vec![
            RankBy::Sum(vec![bm("body", "alpha"), bm("body", "gamma")]),
            bm("title", "omega sigma"),
            RankBy::saturate("views", 1000.0),
        ]),
    );
    // Nested products / zero factors.
    check(
        &col,
        &docs,
        "zero factor",
        &RankBy::Product(vec![bm("body", "alpha"), RankBy::Const(0.0)]),
    );
    check(
        &col,
        &docs,
        "product of one",
        &RankBy::Product(vec![bm("body", "alpha")]),
    );
}

#[test]
fn plain_bm25_unchanged_by_constant_products() {
    let dir = TempDir::new().unwrap();
    let (col, _docs) = build(&dir);
    let base = col.query(&bm("body", "alpha beta"), None, 20).unwrap();
    let scaled = col
        .query(&RankBy::product(2.0, bm("body", "alpha beta")), None, 20)
        .unwrap();
    let nested = col
        .query(
            &RankBy::Product(vec![
                RankBy::Const(2.0),
                RankBy::Product(vec![RankBy::Const(1.0), bm("body", "alpha beta")]),
            ]),
            None,
            20,
        )
        .unwrap();
    assert_eq!(base.len(), scaled.len());
    for ((b, s), n) in base.iter().zip(&scaled).zip(&nested) {
        assert_eq!(b.id, s.id);
        assert_eq!(b.id, n.id);
        assert!((s.score - 2.0 * b.score).abs() < 1e-5);
        assert!((n.score - s.score).abs() < 1e-6);
    }
}

#[test]
fn updates_and_deletes_are_reflected() {
    let dir = TempDir::new().unwrap();
    let (mut col, mut docs) = build(&dir);
    let e = RankBy::Product(vec![bm("body", "alpha"), RankBy::saturate("views", 700.0)]);

    for id in (0..N).step_by(9) {
        col.delete(id).unwrap();
        docs.remove(&id);
    }
    for id in (1..N).step_by(13) {
        if docs.contains_key(&id) {
            col.patch(
                id,
                DocumentPatch::new()
                    .with_attribute("views", AttributeValue::Uint(4999))
                    .with_attribute("body", "alpha alpha alpha".into()),
            )
            .unwrap();
            let d = docs.get_mut(&id).unwrap();
            d.insert("views".into(), AttributeValue::Uint(4999));
            d.insert("body".into(), "alpha alpha alpha".into());
        }
    }
    let mut reference = Reference::new(&col, &docs);
    let scores: HashMap<DocId, f32> = (0..N)
        .filter(|id| docs.contains_key(id))
        .filter_map(|id| {
            let s = reference.eval(&e, id);
            (s > 0.0).then_some((id, s))
        })
        .collect();
    let actual = col.query(&e, None, 25).unwrap();
    assert_matches_reference("after mutation", &actual, &scores, 25, false);

    let sat = RankBy::saturate("views", 700.0);
    let scores: HashMap<DocId, f32> = (0..N)
        .filter(|id| docs.contains_key(id))
        .filter_map(|id| {
            let s = reference.eval(&sat, id);
            (s > 0.0).then_some((id, s))
        })
        .collect();
    let actual = col.query(&sat, None, 15).unwrap();
    assert_matches_reference("monotone after mutation", &actual, &scores, 15, true);
}

#[test]
fn validation_errors() {
    let dir = TempDir::new().unwrap();
    let (col, _docs) = build(&dir);
    let bad: Vec<RankBy> = vec![
        RankBy::Const(-1.0),
        RankBy::Const(f32::NAN),
        RankBy::attribute("cat"),                   // string attribute
        RankBy::attribute("nope"),                  // unknown
        RankBy::attribute_with_missing("price", -1.0),
        RankBy::saturate("price", 0.0),
        RankBy::saturate("price", 5.0).with_exponent(0.0),
        RankBy::decay("price", -3.0),
        RankBy::decay_from("price", f64::NAN, 3.0),
        RankBy::decay("price", 3.0).with_exponent(f64::INFINITY),
        RankBy::Product(vec![]),
        RankBy::filter(Filter::Eq("plain".into(), ScalarValue::Float(1.0))), // not filterable
    ];
    for e in bad {
        assert!(
            matches!(col.query(&e, None, 5), Err(TreeError::InvalidQuery(_))),
            "{e:?} should be rejected"
        );
    }
}

#[test]
fn hybrid_and_facets_over_boosted_queries() {
    let dir = TempDir::new().unwrap();
    let (col, docs) = build(&dir);

    let e = RankBy::Sum(vec![
        bm("body", "alpha"),
        RankBy::product(
            2.0,
            RankBy::filter(Filter::Eq("cat".into(), ScalarValue::String("a".into()))),
        ),
    ]);
    let filter = Filter::Gt("price".into(), ScalarValue::Float(10.0));
    let facets = [FacetSpec::terms("cat")];

    let mut reference = Reference::new(&col, &docs);
    let positives = reference.scores(&e, Some(&filter));
    let resp = col
        .query_bm25_with_facets(&e, Some(&filter), 5, &facets)
        .unwrap();
    assert_eq!(resp.total_hits as usize, positives.len());
    assert_matches_reference("faceted hits", &resp.hits, &positives, 5, false);
    let mut expected_counts: HashMap<String, u64> = HashMap::new();
    for id in positives.keys() {
        if let Some(AttributeValue::String(c)) = docs[id].get("cat") {
            *expected_counts.entry(c.clone()).or_default() += 1;
        }
    }
    let terms = resp.facets.terms("cat").unwrap();
    assert_eq!(terms.buckets.len(), expected_counts.len());
    for b in &terms.buckets {
        let ScalarValue::String(v) = &b.value else { panic!() };
        assert_eq!(b.count, expected_counts[v], "facet count for {v}");
    }

    // Hybrid RRF inherits the boosted lexical leg.
    let opts = HybridSearchOptions::new(10, SearchOptions::new(20, 8).with_rerank(40));
    let hits = col
        .hybrid_search(&[0.0; 4], &e, opts, Some(&filter))
        .unwrap();
    assert!(!hits.is_empty());
    let lexical = col.query(&e, Some(&filter), 10).unwrap();
    let lexical_ids: HashSet<DocId> = lexical.iter().map(|h| h.id).collect();
    for h in &hits {
        if let Some(rank) = h.bm25_rank {
            assert!(rank >= 1);
        }
    }
    assert!(hits.iter().any(|h| h
        .bm25_rank
        .is_some_and(|_| lexical_ids.contains(&h.id))));

    let resp = col
        .hybrid_search_with_facets(
            &[0.0; 4],
            &e,
            HybridSearchOptions::new(10, SearchOptions::new(20, 8).with_rerank(40)),
            Some(&filter),
            &facets,
            HybridFacetScope::Bm25AndFilterMatches,
        )
        .unwrap();
    assert_eq!(resp.total_hits as usize, positives.len());
}
