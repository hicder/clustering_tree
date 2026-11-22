# Faceting

Facets summarize a result set: "12 shoes, 7 hats", "price histogram", "average rating". IzzyDB facets are **exact** (not sampled or approximate) and run over the same filters used for search.

Every attribute declared `filterable = true` can be faceted.

```rust
use izzydb::{FacetRange, FacetSort, FacetSpec, Filter};

let facets = vec![
    FacetSpec::terms("category").with_limit(5),
    FacetSpec::ranges("price", vec![
        FacetRange::below(25.0_f64).with_label("budget"),
        FacetRange::between(25.0_f64, 100.0_f64),
        FacetRange::from_value(100.0_f64),
    ]),
    FacetSpec::stats("price").with_alias("price_stats"),
    FacetSpec::histogram("price", 20.0).with_alias("price_hist"),
];

let filter = Filter::Eq("in_stock".into(), true.into());
let results = col.facet(Some(&filter), &facets)?;

if let Some(cats) = results.terms("category") {
    for b in &cats.buckets {
        println!("{:?}: {}", b.value, b.count);
    }
}
```

`Collection::facet(filter, specs)` counts over **all documents matching `filter`** (or the whole collection when `None`). It does no ranking and needs no query vector.

## Facet kinds

### `Terms` — counts per distinct value

`FacetSpec::terms(attr)`; available on `String`, `Int`, `Uint`, `Float`, `Bool` and array attributes (arrays count each element).

| Builder | Default | Effect |
|---|---|---|
| `.with_limit(n)` | 10 | Max buckets returned. |
| `.with_sort(s)` | `CountDesc` | `CountDesc`, `CountAsc`, `ValueAsc`, `ValueDesc`. Ties break by ascending value. |
| `.with_min_count(n)` | 1 | Drop buckets with fewer documents. |
| `.with_prefix(p)` | none | Only values starting with `p`: **typeahead over facet values**. `String`/`StringArray` only. |

Result `TermFacetResult { buckets: Vec<FacetBucket{value, count}>, total_values, missing_count }`. `total_values` is the number of distinct values that passed `prefix`/`min_count` *before* `limit`, so you can show "+37 more".

### `Ranges` — counts per custom bucket

`FacetSpec::ranges(attr, vec![FacetRange…])`; not available on `Bool`.

| Constructor | Interval |
|---|---|
| `FacetRange::between(a, b)` | `[a, b)` |
| `FacetRange::below(b)` | `(-∞, b)` |
| `FacetRange::from_value(a)` | `[a, +∞)` |
| `.with_label("budget")` | attach a display label |
| `.with_inclusive(from_incl, to_incl)` | change bound inclusivity |

Bounds must have the attribute's type and satisfy `from < to`. Result `RangeFacetResult { buckets: Vec<RangeBucket{label, from, to, count}>, missing_count }`. Ranges may overlap; a document is counted in every range it falls into.

### `Stats` — numeric summary

`FacetSpec::stats(attr)` on `Int`/`Uint`/`Float`. Result `StatsFacetResult { count, missing_count, min, max, sum, avg }` (`min`/`max`/`avg` are `None` when no document has a value).

### `Histogram` — fixed-width buckets

`FacetSpec::histogram(attr, interval)` on numeric attributes. `interval` must be `> 0`. Optional `.with_offset(x)` shifts bucket boundaries; `.with_min_count(n)` hides sparse buckets (default 1, so empty buckets are omitted). Result `HistogramFacetResult { buckets: Vec<HistogramBucket{start, end, count}>, missing_count }`.

## Naming and several facets on one attribute

Results are keyed by the facet's **output name**: the `alias` if set, else the attribute name. Output names must be unique per call, so use `.with_alias(...)` when you want, e.g., both stats and a histogram of `price`:

```rust
FacetSpec::stats("price").with_alias("price_stats")
FacetSpec::histogram("price", 20.0).with_alias("price_hist")
```

Look results up with `results.terms("name")`, `.ranges(..)`, `.stats(..)`, `.histogram(..)`, or `.get(name)` for the enum.

## Counting missing values

By default `missing_count` is reported as `0` (not computed). Call `.with_missing()` to count documents lacking the attribute:

```rust
FacetSpec::terms("category").with_missing()
```

## Multi-select (disjunctive) facets

In a faceted UI the user ticks "shoes"; the category facet should still show "hats (7)" so they can add it. If the facet obeyed the active `category = shoes` filter, every other category would show zero.

`exclude_self()` solves this: it drops top-level `And` clauses that reference **only this facet's attribute** when computing *this* facet, while other facets and the hits still honor the full filter.

```rust
let filter = Filter::And(vec![
    Filter::Eq("category".into(), "shoes".into()),
    Filter::Eq("in_stock".into(), true.into()),
]);

let facets = vec![
    FacetSpec::terms("category").exclude_self(),   // sees in_stock=true, ignores category=shoes
    FacetSpec::terms("tags"),                      // sees the full filter
];
```

More control via `with_filter_mode`:

| `FacetFilterMode` | Meaning |
|---|---|
| `QueryFilter` (default) | the full query filter |
| `ExcludeSelf` | drop top-level `And` clauses touching only this attribute |
| `ExcludeAttributes(vec!["a","b"])` | drop top-level `And` clauses touching only attributes in the list |
| `Custom(Some(f))` / `Custom(None)` | replace the filter entirely (`None` = global counts over all documents) |

Only **top-level** `And` clauses are removed. A clause buried inside an `Or` or `Not` is left in place.

## Combining facets with search

Each search mode has a `*_with_facets` variant returning `FacetedSearchResponse { hits, facets, total_hits }`:

```rust
use izzydb::VectorFacetScope;

let resp = col.search_with_facets(
    &query,
    SearchOptions::new(10, 8),
    Some(&filter),
    &facets,
    VectorFacetScope::FilterMatches,
)?;
println!("{} matching docs, showing {}", resp.total_hits, resp.hits.len());
```

`total_hits` is the size of the facet population (what the facets were computed over).

### Vector search scope (`VectorFacetScope`)

| Variant | Facets computed over |
|---|---|
| `FilterMatches` (default) | all documents matching `filter` (the usual facet-sidebar behavior) |
| `TopK` | only the returned `k` hits |
| `Candidates(n)` | the top-`n` ANN candidates |

For `TopK` and `Candidates`, the population is a fixed set, so per-facet `FacetFilterMode`s are ignored.

### BM25

```rust
let resp = col.query_bm25_with_facets(&rank_by, Some(&filter), 10, &facets)?;
```

The population is every document with BM25 score `> 0` that also passes `filter`. This is exact, not limited to the top `top_k`.

### Hybrid

<a id="hybrid"></a>

```rust
use izzydb::HybridFacetScope;

let resp = col.hybrid_search_with_facets(
    &query, &rank_by, hybrid_opts, Some(&filter), &facets,
    HybridFacetScope::Bm25AndFilterMatches,
)?;
```

| `HybridFacetScope` | Facets computed over |
|---|---|
| `FilterMatches` (default) | all documents matching `filter` |
| `FusedCandidates` | union of the vector and BM25 candidates that entered fusion |
| `Bm25AndFilterMatches` | all documents with BM25 score `> 0` that match `filter` |

## Errors

`InvalidQuery` for: unknown or non-`filterable` attribute, duplicate output names, `prefix` on a non-string attribute, `Stats`/`Histogram` on a non-numeric attribute, `Ranges` on `Bool`, a bad range, or a non-positive histogram interval.
