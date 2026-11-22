//! Exact-parity tests for faceting against a brute-force oracle over the in-memory document model.

use std::collections::{BTreeMap, HashMap, HashSet};

use izzydb::facet::{
    FacetFilterMode, FacetKind, FacetRange, FacetResult, FacetResults, FacetSort, FacetSpec,
    HybridFacetScope, VectorFacetScope, cmp_scalar_value, take_strategy_trace,
};
use izzydb::{
    AttributeType, AttributeValue, Attributes, Bm25Overrides, Collection, CollectionConfig,
    DistanceMetric, DocId, DocumentPatch, Filter, FullTextSearchConfig, HybridSearchOptions,
    RankBy, ScalarValue, SearchOptions, TreeConfig, TreeError,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

const DIM: usize = 4;
const WORDS: [&str; 8] = [
    "alpha", "alto", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
];

type Model = HashMap<DocId, (Vec<f32>, Attributes)>;

fn tree_cfg(max_leaf: usize, min_leaf: usize) -> TreeConfig {
    TreeConfig {
        dim: DIM,
        metric: DistanceMetric::L2Squared,
        min_internal_fanout: 2,
        max_internal_fanout: 6,
        min_leaf_size: min_leaf,
        max_leaf_size: max_leaf,
        insert_beam_width: 4,
        lire_neighbor_count: 4,
        enable_lire: true,
        centroid_drift_eps: 1e-4,
    }
}

fn config(tree: TreeConfig) -> CollectionConfig {
    CollectionConfig::new(tree)
        .with_attribute("category", AttributeType::String, true)
        .with_attribute("score", AttributeType::Float, true)
        .with_attribute("rank", AttributeType::Int, true)
        .with_attribute("big", AttributeType::Int, true)
        .with_attribute("count", AttributeType::Uint, true)
        .with_attribute("active", AttributeType::Bool, true)
        .with_attribute("tags", AttributeType::StringArray, true)
        .with_attribute("nums", AttributeType::IntArray, true)
        .with_attribute("uints", AttributeType::UintArray, true)
        .with_attribute("note", AttributeType::String, false)
        .with_fts_attribute("body", AttributeType::String, false, FullTextSearchConfig::default())
}

fn small_config() -> CollectionConfig {
    config(tree_cfg(8, 2))
}

fn random_vec(rng: &mut SmallRng) -> Vec<f32> {
    (0..DIM).map(|_| rng.random_range(-1.0f32..1.0f32)).collect()
}

fn gen_attrs(rng: &mut SmallRng, i: u64) -> Attributes {
    let mut a = Attributes::new();
    let cats = ["red", "green", "blue", "a\0b", "héllo", "a\u{1F600}"];
    if rng.random_bool(0.85) {
        let c = if i % 97 == 0 {
            "rare".to_string()
        } else {
            cats[rng.random_range(0..cats.len())].to_string()
        };
        a.insert("category".into(), c.into());
    }
    if rng.random_bool(0.8) {
        let floats = [-0.0f64, 0.0, 1.5, 2.5, -3.25, 100.0, 7.75];
        a.insert("score".into(), floats[rng.random_range(0..floats.len())].into());
    }
    if rng.random_bool(0.8) {
        a.insert("rank".into(), AttributeValue::Int(rng.random_range(-12i64..25)));
    }
    if rng.random_bool(0.3) {
        let big = [i64::MIN, i64::MAX, 0, -1];
        a.insert("big".into(), AttributeValue::Int(big[rng.random_range(0..big.len())]));
    }
    if rng.random_bool(0.8) {
        let v = if rng.random_bool(0.05) {
            u64::MAX
        } else {
            rng.random_range(0u64..30)
        };
        a.insert("count".into(), AttributeValue::Uint(v));
    }
    if rng.random_bool(0.7) {
        a.insert("active".into(), rng.random_bool(0.5).into());
    }
    if rng.random_bool(0.8) {
        let pool = ["x", "y", "z", "w", "xy"];
        let n = rng.random_range(0..5);
        let tags: Vec<String> = (0..n)
            .map(|_| pool[rng.random_range(0..pool.len())].to_string())
            .collect();
        a.insert("tags".into(), tags.into());
    }
    if rng.random_bool(0.8) {
        let n = rng.random_range(0..5);
        let nums: Vec<i64> = (0..n).map(|_| rng.random_range(-5i64..15)).collect();
        a.insert("nums".into(), nums.into());
    }
    if rng.random_bool(0.6) {
        let n = rng.random_range(0..4);
        let u: Vec<u64> = (0..n).map(|_| rng.random_range(0u64..10)).collect();
        a.insert("uints".into(), u.into());
    }
    if rng.random_bool(0.5) {
        a.insert("note".into(), "n".into());
    }
    let n = rng.random_range(2..8);
    let body: Vec<&str> = (0..n).map(|_| WORDS[rng.random_range(0..WORDS.len())]).collect();
    a.insert("body".into(), body.join(" ").into());
    a
}

fn populate(col: &mut Collection, n: u64, seed: u64) -> Model {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut model = Model::new();
    for id in 0..n {
        let v = random_vec(&mut rng);
        let a = gen_attrs(&mut rng, id);
        col.upsert(id, &v, a.clone()).unwrap();
        model.insert(id, (v, a));
    }
    model
}

// ------------------------------------------------------------------------------------------
// Oracle
// ------------------------------------------------------------------------------------------

fn norm(v: ScalarValue) -> ScalarValue {
    match v {
        ScalarValue::Float(f) if f == 0.0 => ScalarValue::Float(0.0),
        o => o,
    }
}

/// Elements of a value (`None` = null/missing; `Some(vec![])` = empty array).
fn elems(v: Option<&AttributeValue>) -> Option<Vec<ScalarValue>> {
    Some(match v? {
        AttributeValue::Null => return None,
        AttributeValue::Bool(b) => vec![ScalarValue::Bool(*b)],
        AttributeValue::Int(x) => vec![ScalarValue::Int(*x)],
        AttributeValue::Uint(x) => vec![ScalarValue::Uint(*x)],
        AttributeValue::Float(x) => vec![norm(ScalarValue::Float(*x))],
        AttributeValue::String(s) => vec![ScalarValue::String(s.clone())],
        AttributeValue::StringArray(a) => a.iter().map(|s| ScalarValue::String(s.clone())).collect(),
        AttributeValue::IntArray(a) => a.iter().map(|x| ScalarValue::Int(*x)).collect(),
        AttributeValue::UintArray(a) => a.iter().map(|x| ScalarValue::Uint(*x)).collect(),
    })
}

fn as_f64(v: &ScalarValue) -> f64 {
    match v {
        ScalarValue::Int(x) => *x as f64,
        ScalarValue::Uint(x) => *x as f64,
        ScalarValue::Float(x) => *x,
        _ => panic!("non numeric"),
    }
}

fn in_range(v: &ScalarValue, r: &FacetRange) -> bool {
    use std::cmp::Ordering::*;
    if let Some(from) = &r.from {
        match cmp_scalar_value(v, from) {
            Less => return false,
            Equal if !r.from_inclusive => return false,
            _ => {}
        }
    }
    if let Some(to) = &r.to {
        match cmp_scalar_value(v, to) {
            Greater => return false,
            Equal if !r.to_inclusive => return false,
            _ => {}
        }
    }
    true
}

fn oracle_one(spec: &FacetSpec, pop: &[&Attributes]) -> FacetResult {
    let total = pop.len() as u64;
    let per_doc: Vec<Option<Vec<ScalarValue>>> =
        pop.iter().map(|a| elems(a.get(&spec.attr))).collect();
    let present = per_doc.iter().filter(|e| e.is_some()).count() as u64;
    let missing = if spec.include_missing { total - present } else { 0 };
    match &spec.kind {
        FacetKind::Terms {
            limit,
            sort,
            min_count,
            prefix,
        } => {
            let mut counts: Vec<(ScalarValue, u64)> = Vec::new();
            for e in per_doc.iter().flatten() {
                let mut seen: Vec<&ScalarValue> = Vec::new();
                for v in e {
                    if seen.iter().any(|s| cmp_scalar_value(s, v).is_eq()) {
                        continue;
                    }
                    seen.push(v);
                    match counts.iter_mut().find(|(k, _)| cmp_scalar_value(k, v).is_eq()) {
                        Some((_, c)) => *c += 1,
                        None => counts.push((v.clone(), 1)),
                    }
                }
            }
            if let Some(p) = prefix {
                counts.retain(|(k, _)| matches!(k, ScalarValue::String(s) if s.starts_with(p.as_str())));
            }
            counts.retain(|(_, c)| *c >= (*min_count).max(1));
            let total_values = counts.len() as u64;
            counts.sort_by(|a, b| cmp_scalar_value(&a.0, &b.0));
            match sort {
                FacetSort::ValueAsc => {}
                FacetSort::ValueDesc => counts.reverse(),
                FacetSort::CountDesc => counts.sort_by(|a, b| {
                    b.1.cmp(&a.1).then_with(|| cmp_scalar_value(&a.0, &b.0))
                }),
                FacetSort::CountAsc => counts.sort_by(|a, b| {
                    a.1.cmp(&b.1).then_with(|| cmp_scalar_value(&a.0, &b.0))
                }),
            }
            counts.truncate(*limit);
            FacetResult::Terms(izzydb::facet::TermFacetResult {
                buckets: counts
                    .into_iter()
                    .map(|(value, count)| izzydb::facet::FacetBucket { value, count })
                    .collect(),
                total_values,
                missing_count: missing,
            })
        }
        FacetKind::Ranges { ranges } => FacetResult::Ranges(izzydb::facet::RangeFacetResult {
            buckets: ranges
                .iter()
                .map(|r| izzydb::facet::RangeBucket {
                    label: r.label.clone(),
                    from: r.from.clone(),
                    to: r.to.clone(),
                    count: per_doc
                        .iter()
                        .flatten()
                        .filter(|e| e.iter().any(|v| in_range(v, r)))
                        .count() as u64,
                })
                .collect(),
            missing_count: missing,
        }),
        FacetKind::Stats => {
            let vals: Vec<&ScalarValue> = per_doc.iter().flatten().flatten().collect();
            let sum: f64 = vals.iter().map(|v| as_f64(v)).sum();
            let min = vals.iter().copied().min_by(|a, b| cmp_scalar_value(a, b)).cloned();
            let max = vals.iter().copied().max_by(|a, b| cmp_scalar_value(a, b)).cloned();
            FacetResult::Stats(izzydb::facet::StatsFacetResult {
                count: vals.len() as u64,
                missing_count: missing,
                min,
                max,
                sum,
                avg: (!vals.is_empty()).then(|| sum / vals.len() as f64),
            })
        }
        FacetKind::Histogram {
            interval,
            offset,
            min_count,
        } => {
            let mut b: BTreeMap<i64, u64> = BTreeMap::new();
            for v in per_doc.iter().flatten().flatten() {
                *b.entry(((as_f64(v) - offset) / interval).floor() as i64).or_default() += 1;
            }
            FacetResult::Histogram(izzydb::facet::HistogramFacetResult {
                buckets: b
                    .into_iter()
                    .filter(|&(_, c)| c >= (*min_count).max(1))
                    .map(|(i, count)| {
                        let start = offset + i as f64 * interval;
                        izzydb::facet::HistogramBucket {
                            start,
                            end: start + interval,
                            count,
                        }
                    })
                    .collect(),
                missing_count: missing,
            })
        }
    }
}

fn pop_by_filter<'a>(model: &'a Model, f: Option<&Filter>) -> Vec<&'a Attributes> {
    model
        .values()
        .map(|(_, a)| a)
        .filter(|a| f.is_none_or(|f| f.matches(a)))
        .collect()
}

fn pop_by_ids<'a>(model: &'a Model, ids: &[DocId]) -> Vec<&'a Attributes> {
    ids.iter().map(|id| &model[id].1).collect()
}

/// Oracle for a full request: each spec's population is `pop_for_spec(spec)`.
fn oracle<'a>(
    specs: &[FacetSpec],
    pop_for_spec: impl Fn(&FacetSpec) -> Vec<&'a Attributes>,
) -> FacetResults {
    let mut r = FacetResults::default();
    for s in specs {
        r.facets
            .insert(s.output_name().to_owned(), oracle_one(s, &pop_for_spec(s)));
    }
    r
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn assert_facets_eq(got: &FacetResults, want: &FacetResults, ctx: &str) {
    assert_eq!(
        got.facets.keys().collect::<Vec<_>>(),
        want.facets.keys().collect::<Vec<_>>(),
        "{ctx}: facet names"
    );
    for (name, g) in &got.facets {
        let w = &want.facets[name];
        match (g, w) {
            (FacetResult::Stats(g), FacetResult::Stats(w)) => {
                assert_eq!(g.count, w.count, "{ctx}: {name} count");
                assert_eq!(g.missing_count, w.missing_count, "{ctx}: {name} missing");
                assert_eq!(g.min, w.min, "{ctx}: {name} min");
                assert_eq!(g.max, w.max, "{ctx}: {name} max");
                assert!(approx(g.sum, w.sum), "{ctx}: {name} sum {} vs {}", g.sum, w.sum);
                assert_eq!(g.avg.is_some(), w.avg.is_some(), "{ctx}: {name} avg presence");
                if let (Some(a), Some(b)) = (g.avg, w.avg) {
                    assert!(approx(a, b), "{ctx}: {name} avg");
                }
            }
            (FacetResult::Histogram(g), FacetResult::Histogram(w)) => {
                assert_eq!(g.missing_count, w.missing_count, "{ctx}: {name} missing");
                assert_eq!(g.buckets.len(), w.buckets.len(), "{ctx}: {name} bucket count");
                for (gb, wb) in g.buckets.iter().zip(&w.buckets) {
                    assert!(approx(gb.start, wb.start) && approx(gb.end, wb.end), "{ctx}: {name}");
                    assert_eq!(gb.count, wb.count, "{ctx}: {name} bucket count");
                }
            }
            (g, w) => assert_eq!(g, w, "{ctx}: facet '{name}'"),
        }
    }
}

fn all_specs() -> Vec<FacetSpec> {
    dedup_aliases(vec![
        FacetSpec::terms("category").with_limit(100).with_missing(),
        FacetSpec::terms("category")
            .with_alias("cat_cnt_asc")
            .with_limit(3)
            .with_sort(FacetSort::CountAsc),
        FacetSpec::terms("category")
            .with_alias("cat_val_desc")
            .with_limit(100)
            .with_sort(FacetSort::ValueDesc),
        FacetSpec::terms("category")
            .with_alias("cat_prefix_nul")
            .with_prefix("a\0")
            .with_limit(100),
        FacetSpec::terms("category")
            .with_alias("cat_prefix_r")
            .with_prefix("r")
            .with_limit(100)
            .with_sort(FacetSort::ValueAsc),
        FacetSpec::terms("tags").with_limit(100).with_missing(),
        FacetSpec::terms("tags")
            .with_alias("tags_x")
            .with_prefix("x")
            .with_limit(100),
        FacetSpec::terms("nums").with_limit(100).with_missing(),
        FacetSpec::terms("uints").with_limit(100),
        FacetSpec::terms("active").with_missing(),
        FacetSpec::terms("rank")
            .with_limit(5)
            .with_sort(FacetSort::ValueDesc),
        FacetSpec::terms("count")
            .with_limit(7)
            .with_sort(FacetSort::CountAsc)
            .with_min_count(2),
        FacetSpec::terms("score").with_limit(100),
        FacetSpec::terms("big").with_limit(100).with_missing(),
        FacetSpec::ranges(
            "rank",
            vec![
                FacetRange::below(0i64).with_label("neg"),
                FacetRange::between(0i64, 10i64),
                FacetRange::from_value(10i64),
                FacetRange::between(-5i64, 5i64).with_inclusive(false, true),
            ],
        )
        .with_missing(),
        FacetSpec::ranges(
            "score",
            vec![
                FacetRange::between(-1.0f64, 2.0f64),
                FacetRange::from_value(2.5f64).with_inclusive(false, false),
            ],
        ),
        FacetSpec::ranges(
            "nums",
            vec![
                FacetRange::between(0i64, 5i64),
                FacetRange::between(3i64, 10i64),
                FacetRange::from_value(8i64),
                FacetRange::below(100i64),
            ],
        )
        .with_missing(),
        FacetSpec::ranges("uints", vec![FacetRange::between(2u64, 6u64)]),
        FacetSpec::ranges("count", vec![FacetRange::from_value(u64::MAX)]),
        FacetSpec::ranges(
            "category",
            vec![FacetRange::between("a", "h"), FacetRange::from_value("h")],
        ),
        FacetSpec::ranges(
            "tags",
            vec![FacetRange::between("w", "y"), FacetRange::from_value("x")],
        ),
        FacetSpec::ranges("big", vec![FacetRange::below(0i64), FacetRange::from_value(0i64)]),
        FacetSpec::stats("rank").with_missing(),
        FacetSpec::stats("score"),
        FacetSpec::stats("count"),
        FacetSpec::histogram("rank", 4.0).with_offset(1.0).with_missing(),
        FacetSpec::histogram("score", 2.5),
        FacetSpec::histogram("count", 8.0).with_min_count(2),
    ])
}

/// Give every un-aliased spec whose output name was already taken a unique alias.
fn dedup_aliases(specs: Vec<FacetSpec>) -> Vec<FacetSpec> {
    let mut seen: HashSet<String> = HashSet::new();
    specs
        .into_iter()
        .map(|mut s| {
            let mut n = 0;
            while !seen.insert(s.output_name().to_owned()) {
                n += 1;
                s.alias = Some(format!("{}_{n}", s.attr));
            }
            s
        })
        .collect()
}

/// Drop `Ranges` facets over array attributes (they never use global counts / complement).
fn without_array_ranges(specs: Vec<FacetSpec>) -> Vec<FacetSpec> {
    specs
        .into_iter()
        .filter(|s| {
            !(matches!(s.kind, FacetKind::Ranges { .. })
                && ["nums", "uints", "tags"].contains(&s.attr.as_str()))
        })
        .collect()
}

fn check_filter(col: &Collection, model: &Model, filter: Option<&Filter>, ctx: &str) {
    let specs = all_specs();
    let got = col.facet(filter, &specs).unwrap();
    let pop = pop_by_filter(model, filter);
    let want = oracle(&specs, |_| pop.clone());
    assert_facets_eq(&got, &want, ctx);
}

fn eq(a: &str, v: impl Into<ScalarValue>) -> Filter {
    Filter::Eq(a.into(), v.into())
}

// ------------------------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------------------------

#[test]
fn test_facet_unfiltered_all_kinds_parity() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), small_config()).unwrap();
    let model = populate(&mut col, 400, 1);
    col.verify_invariants().unwrap();

    take_strategy_trace();
    let specs = without_array_ranges(all_specs());
    let got = col.facet(None, &specs).unwrap();
    assert_eq!(take_strategy_trace(), vec!["global"], "unfiltered must use global counts");
    let pop = pop_by_filter(&model, None);
    assert_facets_eq(&got, &oracle(&specs, |_| pop.clone()), "unfiltered/global");

    // Array ranges can't use global counts (documents must count once per bucket).
    check_filter(&col, &model, None, "unfiltered/all kinds incl. array ranges");
}

#[test]
fn test_facet_filtered_strategies_parity_and_crossover() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), small_config()).unwrap();
    let model = populate(&mut col, 500, 2);

    let cases: Vec<(&str, Filter, &str)> = vec![
        ("selective eq", eq("category", "rare"), "docs"),
        ("broad eq", eq("category", "red"), "leaf_masks_no_views"),
        (
            "broad range",
            Filter::Gte("rank".into(), ScalarValue::Int(0)),
            "leaf_masks_no_views",
        ),
        (
            "contains",
            Filter::Contains("tags".into(), "x".into()),
            "leaf_masks_no_views",
        ),
        (
            "and of broad",
            Filter::And(vec![eq("category", "red"), Filter::Gt("rank".into(), ScalarValue::Int(-5))]),
            "leaf_masks_no_views",
        ),
        (
            "or",
            Filter::Or(vec![eq("category", "rare"), eq("category", "blue")]),
            "leaf_masks_no_views",
        ),
        (
            "complement of selective",
            Filter::Not(Box::new(eq("category", "rare"))),
            "complement",
        ),
        (
            "noteq selective",
            Filter::NotEq("category".into(), "rare".into()),
            "complement",
        ),
        (
            "notin selective",
            Filter::NotIn("category".into(), vec!["rare".into()]),
            "complement",
        ),
        (
            "complement of broad",
            Filter::Not(Box::new(eq("category", "red"))),
            "leaf_masks_views",
        ),
        ("is_null", Filter::IsNull("rank".into()), "leaf_masks_views"),
        ("is_not_null", Filter::IsNotNull("rank".into()), "leaf_masks_no_views"),
        (
            "mixed not/or",
            Filter::And(vec![
                Filter::Or(vec![eq("category", "green"), Filter::Not(Box::new(eq("active", true)))]),
                Filter::Lt("score".into(), ScalarValue::Float(50.0)),
            ]),
            "leaf_masks_views",
        ),
        (
            "empty result",
            Filter::And(vec![eq("category", "red"), eq("category", "green")]),
            "",
        ),
    ];
    for (name, f, expected_strategy) in cases {
        take_strategy_trace();
        let specs = all_specs();
        // Array-range facets force a leaf/doc pass for complement filters; check the strategy on
        // a request without them, then full parity with them.
        let simple = without_array_ranges(specs);
        let got = col.facet(Some(&f), &simple).unwrap();
        let trace = take_strategy_trace();
        if !expected_strategy.is_empty() {
            assert!(
                trace.contains(&expected_strategy),
                "{name}: expected strategy {expected_strategy}, got {trace:?}"
            );
        }
        let pop = pop_by_filter(&model, Some(&f));
        assert_facets_eq(&got, &oracle(&simple, |_| pop.clone()), name);
        check_filter(&col, &model, Some(&f), &format!("{name} (all kinds)"));
    }
    col.verify_invariants().unwrap();
}

#[test]
fn test_facet_crossover_beyond_forward_pass_limit() {
    // Few, large leaves: the filter resolves to <= 16 candidate leaves but > 2048 documents, so
    // the forward `DocView` pass must be skipped in favor of inverted leaf masks.
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), config(tree_cfg(500, 100))).unwrap();
    let mut rng = SmallRng::seed_from_u64(9);
    let mut model = Model::new();
    for id in 0..2300u64 {
        let v = random_vec(&mut rng);
        let mut a = gen_attrs(&mut rng, id);
        a.insert("category".into(), if id % 400 == 0 { "few" } else { "many" }.into());
        col.upsert(id, &v, a.clone()).unwrap();
        model.insert(id, (v, a));
    }
    let specs = vec![
        FacetSpec::terms("tags").with_limit(100).with_missing(),
        FacetSpec::stats("rank"),
        FacetSpec::ranges("nums", vec![FacetRange::between(0i64, 5i64), FacetRange::from_value(3i64)]),
    ];
    // > 2048 matching docs.
    take_strategy_trace();
    let f = eq("category", "many");
    let got = col.facet(Some(&f), &specs).unwrap();
    assert_eq!(take_strategy_trace(), vec!["leaf_masks_no_views"]);
    let pop = pop_by_filter(&model, Some(&f));
    assert!(pop.len() > 2048);
    assert_facets_eq(&got, &oracle(&specs, |_| pop.clone()), "over 2048");

    // <= 2048 matching docs in few leaves -> forward pass.
    take_strategy_trace();
    let f2 = eq("category", "few");
    let got2 = col.facet(Some(&f2), &specs).unwrap();
    assert_eq!(take_strategy_trace(), vec!["docs"]);
    let pop2 = pop_by_filter(&model, Some(&f2));
    assert_facets_eq(&got2, &oracle(&specs, |_| pop2.clone()), "under 2048");
    col.verify_invariants().unwrap();
}

#[test]
fn test_facet_disjunctive_modes() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), small_config()).unwrap();
    let model = populate(&mut col, 400, 3);

    let f_cat = eq("category", "red");
    let f_rank = Filter::Gte("rank".into(), ScalarValue::Int(0));
    let f_tag = Filter::Contains("tags".into(), "x".into());
    let query = Filter::And(vec![f_cat.clone(), Filter::And(vec![f_rank.clone(), f_tag.clone()])]);
    let custom = eq("active", true);

    let specs = vec![
        FacetSpec::terms("category").with_limit(50).with_missing(),
        FacetSpec::terms("category").with_alias("cat_self").with_limit(50).exclude_self(),
        FacetSpec::terms("rank").with_limit(50).exclude_self(),
        FacetSpec::stats("rank").with_alias("rank_stats_self").exclude_self(),
        FacetSpec::terms("tags")
            .with_limit(50)
            .with_filter_mode(FacetFilterMode::ExcludeAttributes(vec!["category".into(), "rank".into()])),
        FacetSpec::terms("active")
            .with_missing()
            .with_filter_mode(FacetFilterMode::ExcludeAttributes(vec![
                "category".into(),
                "rank".into(),
                "tags".into(),
            ])),
        FacetSpec::terms("category")
            .with_alias("cat_global")
            .with_limit(50)
            .with_filter_mode(FacetFilterMode::Custom(None)),
        FacetSpec::terms("category")
            .with_alias("cat_custom")
            .with_limit(50)
            .with_filter_mode(FacetFilterMode::Custom(Some(custom.clone()))),
        // Non-self attribute with ExcludeSelf drops nothing referencing other attributes.
        FacetSpec::terms("active").with_alias("active_self").exclude_self(),
    ];
    take_strategy_trace();
    let got = col.facet(Some(&query), &specs).unwrap();
    let trace = take_strategy_trace();
    assert!(trace.contains(&"global"), "Custom(None) facet should use global counts: {trace:?}");

    let both = Filter::And(vec![f_rank.clone(), f_tag.clone()]);
    let rank_tag_cat_excl = Filter::And(vec![f_cat.clone(), f_tag.clone()]);
    let want = oracle(&specs, |s| match s.output_name() {
        "category" | "tags" => pop_by_filter(&model, Some(&query)),
        "cat_self" => pop_by_filter(&model, Some(&both)),
        "rank" | "rank_stats_self" => pop_by_filter(&model, Some(&rank_tag_cat_excl)),
        // tags: exclude category & rank -> only the tags conjunct remains
        "active" => pop_by_filter(&model, Some(&f_tag)).into_iter().collect(),
        "cat_global" => pop_by_filter(&model, None),
        "cat_custom" => pop_by_filter(&model, Some(&custom)),
        "active_self" => pop_by_filter(&model, Some(&query)),
        other => panic!("unexpected {other}"),
    });
    // `tags` uses ExcludeAttributes(category, rank): effective filter == f_tag, not the query.
    let mut want = want;
    want.facets.insert(
        "tags".into(),
        oracle_one(&specs[4], &pop_by_filter(&model, Some(&f_tag))),
    );
    // `active` excludes everything -> global population.
    want.facets.insert(
        "active".into(),
        oracle_one(&specs[5], &pop_by_filter(&model, None)),
    );
    assert_facets_eq(&got, &want, "disjunctive");

    // Mixed Or/Not conjuncts referencing the excluded attribute and others are preserved.
    let mixed = Filter::Or(vec![eq("category", "green"), Filter::Gt("rank".into(), ScalarValue::Int(5))]);
    let q2 = Filter::And(vec![mixed.clone(), f_tag.clone()]);
    let specs2 = vec![FacetSpec::terms("category").with_limit(50).exclude_self()];
    let got2 = col.facet(Some(&q2), &specs2).unwrap();
    let want2 = oracle(&specs2, |_| pop_by_filter(&model, Some(&q2)));
    assert_facets_eq(&got2, &want2, "mixed or preserved");
}

#[test]
fn test_facet_validation_errors() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), small_config()).unwrap();
    populate(&mut col, 20, 4);
    let bad = |specs: Vec<FacetSpec>| {
        matches!(col.facet(None, &specs), Err(TreeError::InvalidQuery(_)))
    };
    assert!(bad(vec![FacetSpec::terms("note")]), "non-filterable");
    assert!(bad(vec![FacetSpec::terms("body")]), "non-filterable fts");
    assert!(bad(vec![FacetSpec::terms("nope")]), "unknown");
    assert!(bad(vec![FacetSpec::terms("rank"), FacetSpec::stats("count").with_alias("rank")]), "dup");
    assert!(bad(vec![FacetSpec::terms("rank").with_prefix("a")]), "prefix on int");
    assert!(bad(vec![FacetSpec::stats("category")]), "stats on string");
    assert!(bad(vec![FacetSpec::histogram("rank", -1.0)]), "bad interval");
    assert!(bad(vec![FacetSpec::ranges("active", vec![])]), "ranges on bool");
    assert!(bad(vec![FacetSpec::ranges("rank", vec![FacetRange::between(9i64, 1i64)])]));
    assert!(matches!(
        col.facet(
            Some(&Filter::Eq("nope".into(), 1i64.into())),
            &[FacetSpec::terms("rank")]
        ),
        Err(TreeError::InvalidFilter(_))
    ));
    assert!(col.facet(None, &[]).unwrap().facets.is_empty());
}

#[test]
fn test_facet_edge_values_exact() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), small_config()).unwrap();
    let v = [0.0f32; DIM];
    let mut put = |id: DocId, attrs: Vec<(&str, AttributeValue)>| {
        let a: Attributes = attrs.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        col.upsert(id, &v, a).unwrap();
    };
    put(1, vec![("score", (-0.0).into()), ("big", AttributeValue::Int(i64::MIN)), ("category", "a\0b".into()),
        ("tags", vec!["t".to_string(), "t".to_string(), "u".to_string()].into()), ("nums", vec![10i64, 20].into())]);
    put(2, vec![("score", 0.0.into()), ("big", AttributeValue::Int(i64::MAX)), ("category", "héllo".into()),
        ("tags", Vec::<String>::new().into()), ("nums", Vec::<i64>::new().into())]);
    put(3, vec![("score", AttributeValue::Null), ("category", "a\0b".into()), ("nums", vec![10i64, 10].into())]);
    put(4, vec![("big", AttributeValue::Int(i64::MAX))]);

    let specs = [
        FacetSpec::terms("score").with_missing(),
        FacetSpec::terms("big").with_missing().with_sort(FacetSort::ValueAsc),
        FacetSpec::terms("category").with_prefix("a\0").with_missing(),
        FacetSpec::terms("tags").with_missing().with_sort(FacetSort::ValueAsc),
        FacetSpec::ranges("nums", vec![FacetRange::between(0i64, 50i64)])
            .with_alias("nums_r")
            .with_missing(),
        FacetSpec::terms("nums").with_missing().with_sort(FacetSort::ValueAsc),
    ];
    for filter in [None, Some(Filter::IsNotNull("category".into()).clone()), Some(Filter::Not(Box::new(eq("category", "zzz"))))] {
        let got = col.facet(filter.as_ref(), &specs).unwrap();
        if filter.is_none() {
            let score = got.terms("score").unwrap();
            // -0.0 and +0.0 merge into one bucket; the Null doc is missing.
            assert_eq!(score.buckets.len(), 1);
            assert_eq!(score.buckets[0].count, 2);
            assert_eq!(score.missing_count, 2);
            let big = got.terms("big").unwrap();
            assert_eq!(big.buckets[0].value, ScalarValue::Int(i64::MIN));
            assert_eq!(big.buckets[1].value, ScalarValue::Int(i64::MAX));
            assert_eq!(big.buckets[1].count, 2);
            let cat = got.terms("category").unwrap();
            assert_eq!(cat.buckets.len(), 1);
            assert_eq!(cat.buckets[0].value, ScalarValue::String("a\0b".into()));
            assert_eq!(cat.buckets[0].count, 2);
            assert_eq!(cat.missing_count, 1); // doc 4 has no category (doc 2's héllo is present)
            let tags = got.terms("tags").unwrap();
            // doc 1's duplicate "t" counts once; doc 2's empty array is present, not missing.
            assert_eq!(
                tags.buckets.iter().map(|b| (b.value.clone(), b.count)).collect::<Vec<_>>(),
                vec![(ScalarValue::String("t".into()), 1), (ScalarValue::String("u".into()), 1)]
            );
            assert_eq!(tags.missing_count, 2); // docs 3 and 4 lack tags
            let r = got.ranges("nums_r").unwrap();
            assert_eq!(r.buckets[0].count, 2); // docs 1 and 3; the empty array matches nothing
            assert_eq!(r.missing_count, 1); // doc 4
            let nums = got.terms("nums").unwrap();
            assert_eq!(nums.buckets.len(), 2);
        }
        let mut model = Model::new();
        for id in 1..=4u64 {
            let d = col.get(id).unwrap().unwrap();
            model.insert(id, (d.embedding, d.attributes));
        }
        let pop = pop_by_filter(&model, filter.as_ref());
        assert_facets_eq(&got, &oracle(&specs, |_| pop.clone()), &format!("edge {filter:?}"));
    }
    col.verify_invariants().unwrap();
}

fn bm25_cases() -> Vec<(&'static str, RankBy, Box<dyn Fn(&str) -> bool>)> {
    let has = |w: &'static str| move |b: &str| b.split_whitespace().any(|t| t == w);
    let has_prefix = |p: &'static str| move |b: &str| b.split_whitespace().any(|t| t.starts_with(p));
    vec![
        ("single", RankBy::bm25("body", "alpha beta"), Box::new(move |b: &str| has("alpha")(b) || has("beta")(b))),
        (
            "sum",
            RankBy::sum(vec![RankBy::bm25("body", "gamma"), RankBy::bm25("body", "zeta")]),
            Box::new(move |b: &str| has("gamma")(b) || has("zeta")(b)),
        ),
        (
            "max",
            RankBy::max(vec![RankBy::bm25("body", "delta"), RankBy::bm25("body", "eta")]),
            Box::new(move |b: &str| has("delta")(b) || has("eta")(b)),
        ),
        ("product", RankBy::product(2.5, RankBy::bm25("body", "epsilon")), Box::new(move |b: &str| has("epsilon")(b))),
        (
            "product_zero",
            RankBy::sum(vec![RankBy::product(0.0, RankBy::bm25("body", "alpha")), RankBy::bm25("body", "beta")]),
            Box::new(move |b: &str| has("beta")(b)),
        ),
        (
            "last_as_prefix",
            RankBy::bm25_with("body", "beta al", Bm25Overrides::default().with_last_as_prefix(true)),
            Box::new(move |b: &str| has("beta")(b) || has_prefix("al")(b)),
        ),
        ("no_match", RankBy::bm25("body", "nonexistentword"), Box::new(|_: &str| false)),
    ]
}

fn body_of(model: &Model, id: DocId) -> String {
    match model[&id].1.get("body") {
        Some(AttributeValue::String(s)) => s.clone(),
        _ => String::new(),
    }
}

#[test]
fn test_facet_bm25_scopes_parity() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), small_config()).unwrap();
    let model = populate(&mut col, 400, 5);
    let specs = vec![
        FacetSpec::terms("category").with_limit(50).with_missing(),
        FacetSpec::terms("category").with_alias("cat_self").with_limit(50).exclude_self(),
        FacetSpec::stats("rank"),
        FacetSpec::ranges("nums", vec![FacetRange::between(0i64, 6i64)]),
    ];
    let filters: Vec<Option<Filter>> = vec![
        None,
        Some(eq("category", "red")),
        Some(Filter::Gte("rank".into(), ScalarValue::Int(3))),
        Some(Filter::And(vec![eq("category", "blue"), Filter::Gte("rank".into(), ScalarValue::Int(-2))])),
    ];
    for (name, rank_by, matcher) in bm25_cases() {
        for filter in &filters {
            let ctx = format!("bm25 {name} filter={filter:?}");
            let resp = col.query_bm25_with_facets(&rank_by, filter.as_ref(), 10, &specs).unwrap();
            let plain = col.query_bm25(&rank_by, filter.as_ref(), 10).unwrap();
            assert_eq!(resp.hits, plain, "{ctx}: hits unchanged");

            let matching = |extra: Option<&Filter>| -> Vec<&Attributes> {
                model
                    .iter()
                    .filter(|(id, (_, a))| {
                        matcher(&body_of(&model, **id)) && extra.is_none_or(|f| f.matches(a))
                    })
                    .map(|(_, (_, a))| a)
                    .collect()
            };
            let pop = matching(filter.as_ref());
            assert_eq!(resp.total_hits, pop.len() as u64, "{ctx}: total_hits");
            let want = oracle(&specs, |s| {
                if s.output_name() == "cat_self" {
                    let eff = s.effective_filter(filter.as_ref());
                    matching(eff.as_ref())
                } else {
                    pop.clone()
                }
            });
            assert_facets_eq(&resp.facets, &want, &ctx);
            for h in &resp.hits {
                assert!(matcher(&body_of(&model, h.id)), "{ctx}: hit must be in match set");
            }
        }
    }
}

#[test]
fn test_facet_vector_scopes_parity() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), small_config()).unwrap();
    let model = populate(&mut col, 400, 6);
    let mut rng = SmallRng::seed_from_u64(77);
    let specs = dedup_aliases(vec![
        FacetSpec::terms("category").with_limit(50).with_missing(),
        FacetSpec::terms("tags").with_limit(50),
        FacetSpec::stats("rank"),
        FacetSpec::histogram("rank", 5.0),
        FacetSpec::ranges("nums", vec![FacetRange::between(0i64, 6i64)]),
    ]);
    let filters: Vec<Option<Filter>> = vec![
        None,
        Some(eq("category", "red")),
        Some(Filter::Not(Box::new(eq("category", "green")))),
        Some(eq("category", "rare")),
    ];
    let opts = SearchOptions::new(10, 8).with_rerank(40);
    for filter in &filters {
        let q = random_vec(&mut rng);
        let ctx = format!("vector filter={filter:?}");

        let resp = col
            .search_with_facets(&q, opts, filter.as_ref(), &specs, VectorFacetScope::FilterMatches)
            .unwrap();
        assert_eq!(resp.hits, col.search(&q, opts, filter.as_ref()).unwrap(), "{ctx}: hits");
        let pop = pop_by_filter(&model, filter.as_ref());
        assert_eq!(resp.total_hits, pop.len() as u64);
        assert_facets_eq(&resp.facets, &oracle(&specs, |_| pop.clone()), &format!("{ctx} FilterMatches"));

        let top = col
            .search_with_facets(&q, opts, filter.as_ref(), &specs, VectorFacetScope::TopK)
            .unwrap();
        let ids: Vec<DocId> = top.hits.iter().map(|h| h.id).collect();
        assert_eq!(top.total_hits, ids.len() as u64);
        assert_facets_eq(&top.facets, &oracle(&specs, |_| pop_by_ids(&model, &ids)), &format!("{ctx} TopK"));

        let cand = col
            .search_with_facets(&q, opts, filter.as_ref(), &specs, VectorFacetScope::Candidates(60))
            .unwrap();
        let wide = SearchOptions { k: 60, ..opts };
        let cand_ids: Vec<DocId> = col.search(&q, wide, filter.as_ref()).unwrap().iter().map(|h| h.id).collect();
        assert_eq!(cand.total_hits, cand_ids.len() as u64);
        assert_eq!(cand.hits.len(), cand_ids.len().min(10));
        assert_facets_eq(&cand.facets, &oracle(&specs, |_| pop_by_ids(&model, &cand_ids)), &format!("{ctx} Candidates"));
    }
}

#[test]
fn test_facet_hybrid_scopes_parity() {
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), small_config()).unwrap();
    let model = populate(&mut col, 400, 8);
    let mut rng = SmallRng::seed_from_u64(78);
    let specs = vec![
        FacetSpec::terms("category").with_limit(50).with_missing(),
        FacetSpec::stats("rank"),
        FacetSpec::ranges("nums", vec![FacetRange::between(0i64, 6i64)]),
    ];
    let rank_by = RankBy::bm25("body", "alpha gamma");
    let matcher = |id: DocId| {
        let b = body_of(&model, id);
        b.split_whitespace().any(|t| t == "alpha" || t == "gamma")
    };
    let hopts = HybridSearchOptions::new(6, SearchOptions::new(8, 8).with_rerank(30)).with_bm25_candidates(12);
    for filter in [None, Some(eq("category", "red")), Some(Filter::Gte("rank".into(), ScalarValue::Int(0)))] {
        let q = random_vec(&mut rng);
        let ctx = format!("hybrid filter={filter:?}");
        let plain = col.hybrid_search(&q, &rank_by, hopts, filter.as_ref()).unwrap();

        let r = col
            .hybrid_search_with_facets(&q, &rank_by, hopts, filter.as_ref(), &specs, HybridFacetScope::FilterMatches)
            .unwrap();
        assert_eq!(r.hits, plain);
        let pop = pop_by_filter(&model, filter.as_ref());
        assert_eq!(r.total_hits, pop.len() as u64);
        assert_facets_eq(&r.facets, &oracle(&specs, |_| pop.clone()), &format!("{ctx} FilterMatches"));

        let fused = col
            .hybrid_search_with_facets(&q, &rank_by, hopts, filter.as_ref(), &specs, HybridFacetScope::FusedCandidates)
            .unwrap();
        let vec_ids: Vec<DocId> = col.search(&q, hopts.vector_search, filter.as_ref()).unwrap().iter().map(|h| h.id).collect();
        let bm_ids: Vec<DocId> = col.query_bm25(&rank_by, filter.as_ref(), 12).unwrap().iter().map(|h| h.id).collect();
        let union: HashSet<DocId> = vec_ids.into_iter().chain(bm_ids).collect();
        let mut union: Vec<DocId> = union.into_iter().collect();
        union.sort_unstable();
        assert_eq!(fused.total_hits, union.len() as u64);
        assert_facets_eq(&fused.facets, &oracle(&specs, |_| pop_by_ids(&model, &union)), &format!("{ctx} Fused"));

        let b = col
            .hybrid_search_with_facets(&q, &rank_by, hopts, filter.as_ref(), &specs, HybridFacetScope::Bm25AndFilterMatches)
            .unwrap();
        let bpop: Vec<&Attributes> = model
            .iter()
            .filter(|(id, (_, a))| matcher(**id) && filter.as_ref().is_none_or(|f| f.matches(a)))
            .map(|(_, (_, a))| a)
            .collect();
        assert_eq!(b.total_hits, bpop.len() as u64);
        assert_facets_eq(&b.facets, &oracle(&specs, |_| bpop.clone()), &format!("{ctx} Bm25AndFilter"));
    }
}

#[test]
fn test_facet_empty_collection() {
    let dir = TempDir::new().unwrap();
    let col = Collection::open(dir.path(), small_config()).unwrap();
    let specs = all_specs();
    let model = Model::new();
    for f in [None, Some(eq("category", "red"))] {
        let got = col.facet(f.as_ref(), &specs).unwrap();
        assert_facets_eq(&got, &oracle(&specs, |_| pop_by_filter(&model, None)), "empty");
    }
    let resp = col
        .query_bm25_with_facets(&RankBy::bm25("body", "alpha"), None, 5, &specs)
        .unwrap();
    assert_eq!(resp.total_hits, 0);
}

#[test]
fn test_facet_churn_parity_and_reopen() {
    let dir = TempDir::new().unwrap();
    let mut rng = SmallRng::seed_from_u64(11);
    let mut model: Model;
    {
        let mut col = Collection::open(dir.path(), small_config()).unwrap();
        model = populate(&mut col, 250, 10);
        let mut next_id = 1000u64;
        for step in 0..160 {
            match rng.random_range(0..8) {
                0 | 1 => {
                    // upsert new or replace
                    let id = if rng.random_bool(0.5) { next_id } else { rng.random_range(0..250) };
                    next_id += 1;
                    let v = random_vec(&mut rng);
                    let a = gen_attrs(&mut rng, id);
                    col.upsert(id, &v, a.clone()).unwrap();
                    model.insert(id, (v, a));
                }
                2 => {
                    // patch attrs (+ sometimes embedding)
                    let ids: Vec<DocId> = model.keys().copied().collect();
                    let id = ids[rng.random_range(0..ids.len())];
                    let mut patch = DocumentPatch::new();
                    let extra = gen_attrs(&mut rng, id);
                    for key in ["category", "rank", "tags", "nums", "score"] {
                        if rng.random_bool(0.5) {
                            match extra.get(key) {
                                Some(v) => patch = patch.with_attribute(key, v.clone()),
                                None => patch = patch.clear_attribute(key),
                            }
                        }
                    }
                    if rng.random_bool(0.3) {
                        patch = patch.with_embedding(random_vec(&mut rng));
                    }
                    col.patch(id, patch.clone()).unwrap();
                    let entry = model.get_mut(&id).unwrap();
                    if let Some(e) = patch.embedding {
                        entry.0 = e;
                    }
                    for (k, v) in patch.attributes {
                        if matches!(v, AttributeValue::Null) {
                            entry.1.remove(&k);
                        } else {
                            entry.1.insert(k, v);
                        }
                    }
                }
                3 => {
                    let ids: Vec<DocId> = model.keys().copied().collect();
                    let batch: Vec<(DocId, DocumentPatch)> = (0..4)
                        .map(|_| {
                            let id = ids[rng.random_range(0..ids.len())];
                            (id, DocumentPatch::new().with_attribute("rank", AttributeValue::Int(rng.random_range(-12..25))))
                        })
                        .collect();
                    // Last write wins for duplicate ids in a batch.
                    col.patch_batch(&batch).unwrap();
                    for (id, p) in &batch {
                        for (k, v) in &p.attributes {
                            model.get_mut(id).unwrap().1.insert(k.clone(), v.clone());
                        }
                    }
                }
                4 => {
                    let ids: Vec<DocId> = model.keys().copied().collect();
                    let id = ids[rng.random_range(0..ids.len())];
                    assert!(col.delete(id).unwrap());
                    model.remove(&id);
                }
                5 => {
                    let ids: Vec<DocId> = model.keys().copied().collect();
                    let batch: Vec<DocId> = (0..5).map(|_| ids[rng.random_range(0..ids.len())]).collect();
                    col.delete_batch(&batch).unwrap();
                    for id in batch {
                        model.remove(&id);
                    }
                }
                6 => {
                    let f = Filter::And(vec![eq("category", "rare"), Filter::Gt("rank".into(), ScalarValue::Int(8))]);
                    col.delete_by_filter(&f).unwrap();
                    model.retain(|_, (_, a)| !f.matches(a));
                }
                _ => {
                    let f = eq("category", "red");
                    let patch = DocumentPatch::new().with_attribute("active", true.into());
                    col.update_by_filter(&f, &patch).unwrap();
                    for (_, a) in model.values_mut() {
                        if f.matches(a) {
                            a.insert("active".into(), true.into());
                        }
                    }
                }
            }
            if step % 40 == 39 {
                col.verify_invariants().unwrap();
                check_filter(&col, &model, None, &format!("churn step {step} global"));
                check_filter(&col, &model, Some(&eq("category", "blue")), &format!("churn step {step} filtered"));
            }
        }
        col.verify_invariants().unwrap();
        check_filter(&col, &model, Some(&Filter::Not(Box::new(eq("category", "red")))), "churn complement");
    }
    // Reopen: persisted global counts must still be exact (and the marker must suppress backfill).
    let col = Collection::open(dir.path(), small_config()).unwrap();
    col.verify_invariants().unwrap();
    check_filter(&col, &model, None, "reopen global");
    check_filter(&col, &model, Some(&Filter::Gte("rank".into(), ScalarValue::Int(0))), "reopen filtered");
}

#[test]
fn test_attr_counts_backfill_on_legacy_database() {
    let dir = TempDir::new().unwrap();
    let model;
    {
        let mut col = Collection::open(dir.path(), small_config()).unwrap();
        model = populate(&mut col, 150, 12);
        col.verify_invariants().unwrap();
    }
    // Reproduce a database created before global counts existed.
    {
        let (storage, _) = izzydb::storage::Storage::open(dir.path(), &small_config().tree).unwrap();
        storage.clear_attr_counts().unwrap();
        assert!(!storage.has_attr_counts_marker().unwrap());
        assert!(storage.all_attr_counts().unwrap().is_empty());
    }
    let mut col = Collection::open(dir.path(), small_config()).unwrap();
    col.verify_invariants().unwrap(); // asserts exact counts + marker
    check_filter(&col, &model, None, "backfilled global");

    // Subsequent mutations keep counts exact.
    col.delete(3).unwrap();
    col.upsert(9999, &[0.0; DIM], Attributes::from([("category".to_string(), "red".into())])).unwrap();
    col.verify_invariants().unwrap();
}
