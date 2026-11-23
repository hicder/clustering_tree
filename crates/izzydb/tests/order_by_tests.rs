//! Attribute-ordered top-k (`query_ordered`) against a brute-force in-memory sort.

#![cfg(any())] // v3: phase 6
use std::cmp::Ordering;
use std::collections::HashMap;

use izzydb::{
    AttributeType, AttributeValue, Attributes, Collection, CollectionConfig, DistanceMetric,
    DocId, DocumentPatch, FacetSpec, Filter, NullsOrder, OrderBySpec, OrderedResult, ScalarValue,
    SortDirection, TreeConfig, TreeError,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

const N: u64 = 220;

fn config() -> CollectionConfig {
    CollectionConfig::new(TreeConfig {
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
    })
    .with_attribute("i", AttributeType::Int, true)
    .with_attribute("u", AttributeType::Uint, true)
    .with_attribute("f", AttributeType::Float, true)
    .with_attribute("s", AttributeType::String, true)
    .with_attribute("g", AttributeType::Int, true) // low-cardinality group key
    .with_attribute("cat", AttributeType::String, true)
    .with_attribute("plain", AttributeType::Int, false)
    .with_attribute("flag", AttributeType::Bool, true)
}

fn build(dir: &TempDir) -> (Collection, HashMap<DocId, Attributes>) {
    let mut col = Collection::open(dir.path(), config()).unwrap();
    let mut rng = SmallRng::seed_from_u64(99);
    let mut docs = HashMap::new();
    for id in 0..N {
        let mut a = Attributes::new();
        if id % 6 != 0 {
            a.insert("i".into(), AttributeValue::Int(rng.random_range(-40..40)));
        }
        if id % 8 != 1 {
            a.insert("u".into(), AttributeValue::Uint(rng.random_range(0..30)));
        }
        if id % 9 != 2 {
            let f = match id % 13 {
                0 => -0.0,
                1 => 0.0,
                _ => rng.random_range(-10.0f64..10.0).round() / 2.0,
            };
            a.insert("f".into(), AttributeValue::Float(f));
        }
        if id % 7 != 3 {
            // Includes embedded NUL and unicode to exercise the escaped string encoding.
            let pool = ["apple", "apple pie", "banana", "ba\0na", "cherry", "éclair", "", "zebra"];
            a.insert("s".into(), pool[rng.random_range(0..pool.len())].into());
        }
        a.insert("g".into(), AttributeValue::Int(rng.random_range(0..4)));
        a.insert("cat".into(), ["x", "y", "z"][(id % 3) as usize].into());
        a.insert("plain".into(), AttributeValue::Int(id as i64));
        col.upsert(id, &[0.0; 4], a.clone()).unwrap();
        docs.insert(id, a);
    }
    (col, docs)
}

fn scalar(v: Option<&AttributeValue>) -> ScalarValue {
    match v {
        Some(AttributeValue::Int(x)) => ScalarValue::Int(*x),
        Some(AttributeValue::Uint(x)) => ScalarValue::Uint(*x),
        Some(AttributeValue::Float(x)) => ScalarValue::Float(*x),
        Some(AttributeValue::String(x)) => ScalarValue::String(x.clone()),
        Some(AttributeValue::Bool(x)) => ScalarValue::Bool(*x),
        _ => ScalarValue::Null,
    }
}

fn cmp_scalar(a: &ScalarValue, b: &ScalarValue) -> Ordering {
    match (a, b) {
        (ScalarValue::Int(x), ScalarValue::Int(y)) => x.cmp(y),
        (ScalarValue::Uint(x), ScalarValue::Uint(y)) => x.cmp(y),
        (ScalarValue::Float(x), ScalarValue::Float(y)) => x.partial_cmp(y).unwrap(),
        (ScalarValue::String(x), ScalarValue::String(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

fn oracle(
    col: &Collection,
    docs: &HashMap<DocId, Attributes>,
    specs: &[OrderBySpec],
    filter: Option<&Filter>,
    k: usize,
) -> Vec<OrderedResult> {
    let allowed: Option<std::collections::HashSet<DocId>> =
        filter.map(|f| col.resolve_filter_doc_ids(f).unwrap().into_iter().collect());
    let mut rows: Vec<(DocId, Vec<ScalarValue>)> = docs
        .iter()
        .filter(|(id, _)| allowed.as_ref().is_none_or(|a| a.contains(id)))
        .map(|(&id, a)| (id, specs.iter().map(|s| scalar(a.get(&s.attr))).collect()))
        .collect();
    rows.sort_by(|a, b| {
        for (spec, (x, y)) in specs.iter().zip(a.1.iter().zip(&b.1)) {
            let (xn, yn) = (matches!(x, ScalarValue::Null), matches!(y, ScalarValue::Null));
            let ord = match (xn, yn) {
                (true, true) => Ordering::Equal,
                (true, false) => {
                    if spec.nulls == NullsOrder::First { Ordering::Less } else { Ordering::Greater }
                }
                (false, true) => {
                    if spec.nulls == NullsOrder::First { Ordering::Greater } else { Ordering::Less }
                }
                (false, false) => {
                    let o = cmp_scalar(x, y);
                    if spec.direction == SortDirection::Desc { o.reverse() } else { o }
                }
            };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        a.0.cmp(&b.0)
    });
    rows.truncate(k);
    rows.into_iter()
        .map(|(id, sort_values)| OrderedResult { id, sort_values })
        .collect()
}

fn filters() -> Vec<Option<Filter>> {
    vec![
        None,
        Some(Filter::Eq("cat".into(), ScalarValue::String("y".into()))), // broad
        Some(Filter::Gt("i".into(), ScalarValue::Int(10))),
        Some(Filter::And(vec![
            Filter::Eq("g".into(), ScalarValue::Int(2)),
            Filter::Not(Box::new(Filter::Eq("cat".into(), ScalarValue::String("x".into())))),
        ])),
        Some(Filter::In("i".into(), vec![ScalarValue::Int(7), ScalarValue::Int(-3)])), // selective
        Some(Filter::Eq("cat".into(), ScalarValue::String("nope".into()))), // empty
    ]
}

fn run(col: &Collection, docs: &HashMap<DocId, Attributes>, specs: &[OrderBySpec]) {
    for filter in filters().into_iter().flatten() {
        for k in [1usize, 2, 7, 25, 1000] {
            let got = col.query_ordered(specs, Some(&filter), k).unwrap();
            let want = oracle(col, docs, specs, Some(&filter), k);
            assert_eq!(got, want, "specs={specs:?} filter={filter:?} k={k}");
        }
    }
    for k in [1usize, 2, 7, 25, 1000] {
        let got = col.query_ordered(specs, None, k).unwrap();
        let want = oracle(col, docs, specs, None, k);
        assert_eq!(got, want, "specs={specs:?} unfiltered k={k}");
    }
}

#[test]
fn single_key_all_types_directions_and_nulls() {
    let dir = TempDir::new().unwrap();
    let (col, docs) = build(&dir);
    for attr in ["i", "u", "f", "s"] {
        for dir_ in [SortDirection::Asc, SortDirection::Desc] {
            for nulls in [NullsOrder::Last, NullsOrder::First] {
                let spec = OrderBySpec {
                    attr: attr.into(),
                    direction: dir_,
                    nulls,
                };
                run(&col, &docs, &[spec]);
            }
        }
    }
}

#[test]
fn multi_key_ties() {
    let dir = TempDir::new().unwrap();
    let (col, docs) = build(&dir);
    run(
        &col,
        &docs,
        &[OrderBySpec::asc("g"), OrderBySpec::desc("u"), OrderBySpec::asc("s")],
    );
    run(
        &col,
        &docs,
        &[
            OrderBySpec::desc("g").with_nulls(NullsOrder::First),
            OrderBySpec::asc("i").with_nulls(NullsOrder::First),
        ],
    );
    // Secondary key on a non-filterable attribute is allowed.
    run(&col, &docs, &[OrderBySpec::desc("u"), OrderBySpec::desc("plain")]);
    // Primary with nulls first, secondary breaks null ties.
    run(
        &col,
        &docs,
        &[OrderBySpec::asc("i").with_nulls(NullsOrder::First), OrderBySpec::desc("f")],
    );
}

#[test]
fn float_negative_zero_and_negative_ints() {
    let dir = TempDir::new().unwrap();
    let (col, _docs) = build(&dir);
    let asc = col.query_ordered(&[OrderBySpec::asc("f")], None, N as usize).unwrap();
    let vals: Vec<f64> = asc
        .iter()
        .filter_map(|r| match r.sort_values[0] {
            ScalarValue::Float(v) => Some(v),
            _ => None,
        })
        .collect();
    assert!(vals.windows(2).all(|w| w[0] <= w[1]));
    assert!(vals.first().unwrap() < &0.0);
    // -0.0 and 0.0 are the same key: ordered purely by DocId inside the zero group.
    let zeros: Vec<DocId> = asc
        .iter()
        .filter(|r| matches!(r.sort_values[0], ScalarValue::Float(v) if v == 0.0))
        .map(|r| r.id)
        .collect();
    assert!(zeros.windows(2).all(|w| w[0] < w[1]));
    let ints = col.query_ordered(&[OrderBySpec::asc("i")], None, 5).unwrap();
    assert!(matches!(ints[0].sort_values[0], ScalarValue::Int(v) if v < 0));
}

#[test]
fn survives_updates_deletes_and_splits() {
    let dir = TempDir::new().unwrap();
    let (mut col, mut docs) = build(&dir);
    for id in (0..N).step_by(5) {
        col.delete(id).unwrap();
        docs.remove(&id);
    }
    for id in (1..N).step_by(7) {
        if docs.contains_key(&id) {
            col.patch(
                id,
                DocumentPatch::new()
                    .with_attribute("i", AttributeValue::Int(1000 + id as i64))
                    .with_attribute("u", AttributeValue::Null),
            )
            .unwrap();
            let d = docs.get_mut(&id).unwrap();
            d.insert("i".into(), AttributeValue::Int(1000 + id as i64));
            d.remove("u");
        }
    }
    // Grow the collection after deletes to force further splits.
    for id in N..N + 60 {
        let mut a = Attributes::new();
        a.insert("i".into(), AttributeValue::Int(-(id as i64)));
        a.insert("g".into(), AttributeValue::Int(1));
        a.insert("cat".into(), "y".into());
        a.insert("plain".into(), AttributeValue::Int(id as i64));
        col.upsert(id, &[0.0; 4], a.clone()).unwrap();
        docs.insert(id, a);
    }
    col.verify_invariants().unwrap();
    run(&col, &docs, &[OrderBySpec::desc("i")]);
    run(&col, &docs, &[OrderBySpec::asc("u").with_nulls(NullsOrder::First), OrderBySpec::asc("i")]);
}

#[test]
fn validation_and_edge_cases() {
    let dir = TempDir::new().unwrap();
    let (col, _docs) = build(&dir);
    let invalid = |specs: &[OrderBySpec]| {
        matches!(col.query_ordered(specs, None, 5), Err(TreeError::InvalidQuery(_)))
    };
    assert!(invalid(&[]));
    assert!(invalid(&[OrderBySpec::asc("nope")]));
    assert!(invalid(&[OrderBySpec::asc("plain")])); // primary must be filterable
    assert!(invalid(&[OrderBySpec::asc("flag")])); // bool is not ordered
    assert!(invalid(&[OrderBySpec::asc("i"), OrderBySpec::asc("nope")]));
    assert!(col.query_ordered(&[OrderBySpec::asc("i")], None, 0).unwrap().is_empty());
    assert!(
        col.query_ordered(
            &[OrderBySpec::asc("i")],
            Some(&Filter::Eq("plain".into(), ScalarValue::Int(1))),
            3
        )
        .is_err()
    );
}

#[test]
fn ordered_with_facets_counts_all_filter_matches() {
    let dir = TempDir::new().unwrap();
    let (col, docs) = build(&dir);
    let filter = Filter::Gt("i".into(), ScalarValue::Int(0));
    let resp = col
        .query_ordered_with_facets(
            &[OrderBySpec::desc("i")],
            Some(&filter),
            5,
            &[FacetSpec::terms("cat")],
        )
        .unwrap();
    let expected = oracle(&col, &docs, &[OrderBySpec::desc("i")], Some(&filter), 5);
    assert_eq!(resp.hits, expected);
    let matching = col.resolve_filter_doc_ids(&filter).unwrap().len() as u64;
    assert_eq!(resp.total_hits, matching);
    let sum: u64 = resp.facets.terms("cat").unwrap().buckets.iter().map(|b| b.count).sum();
    assert_eq!(sum, matching);
}

#[test]
fn top_group_fills_result_with_docid_tiebreak() {
    // The first (highest) value group already fills the result; ties inside it are by DocId.
    let dir = TempDir::new().unwrap();
    let mut col = Collection::open(dir.path(), config()).unwrap();
    for id in 0..200u64 {
        let mut a = Attributes::new();
        a.insert("u".into(), AttributeValue::Uint(id / 100));
        col.upsert(id, &[0.0; 4], a).unwrap();
    }
    let got = col.query_ordered(&[OrderBySpec::desc("u")], None, 3).unwrap();
    let ids: Vec<DocId> = got.iter().map(|r| r.id).collect();
    assert_eq!(ids, vec![100, 101, 102]);
}
