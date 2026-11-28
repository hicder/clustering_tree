# Ranking, Boosting & Sorting

Beyond plain BM25, a `RankBy` can blend text relevance with filters and numeric attributes, and you can order results by attributes alone. All of it runs through the same `col.query(&rank_by, filter, top_k)`.

Every expression produces a score `>= 0`; documents scoring `0` are not returned; ties break by ascending `DocId`.

## Building blocks

| Expression | Score |
|---|---|
| `RankBy::bm25(attr, "text")` | BM25 relevance |
| `RankBy::constant(c)` | the constant `c` (`>= 0`) |
| `RankBy::filter(filter)` | `1.0` if the document matches `filter`, else `0.0` |
| `RankBy::attribute("n")` | the numeric value (`Int` / `Uint` / `Float`), clamped to `>= 0` |
| `RankBy::saturate("n", midpoint)` | `x / (x + midpoint)`: rises from 0 toward 1, `0.5` at `midpoint` |
| `RankBy::decay("n", midpoint)` | `midpoint / (x + midpoint)`: `1` at `0`, `0.5` at `midpoint`, falling toward 0 |
| `RankBy::decay_from("n", origin, midpoint)` | same, measured as `\|x - origin\|` |
| `sum` / `max` / `product` / `product_all` | combinators |

Tweaks: `.with_exponent(e)` (steeper or flatter curve), `.with_missing(s)` (score for documents without the attribute, default `0`), `.with_decay_mode(DecayMode::{Symmetric, Backward, Forward})` (`Backward` penalizes values below the origin, e.g. old timestamps; `Forward` penalizes values above it).

Attributes used in `attribute` / `saturate` / `decay` must be numeric scalars. They do **not** need to be `filterable`, but a `filterable` attribute lets IzzyDB prune and use index order.

## Boost with a filter

```rust
// Phrase matches in the title count 3 extra points; in-stock items get a small nudge.
let rank_by = RankBy::sum(vec![
    RankBy::bm25("description", "trail shoe"),
    RankBy::product(3.0, RankBy::filter(Filter::ContainsTokenSequence {
        attr: "title".into(),
        query: "trail shoe".into(),
        last_as_prefix: false,
    })),
    RankBy::product(0.5, RankBy::filter(Filter::Eq("in_stock".into(), true.into()))),
]);
```

Documents that match a boost but no BM25 term can still be returned (their score is just the boost). Use a multiplicative form (below) if boosts should only reorder text matches.

## Multiply by a signal

```rust
// Recency: documents newer than "now" keep full score, older ones decay (half score at 30 days old).
let recency = RankBy::decay_from("created_at", now, 30.0 * DAY).with_decay_mode(DecayMode::Backward);

// BM25 * (1 + 2 * recency)
let rank_by = RankBy::boost_multiplicative(RankBy::bm25("description", "trail shoe"), 2.0, recency);

// BM25 * popularity, where popularity is squashed into 0..1
let rank_by = RankBy::product_all(vec![
    RankBy::bm25("description", "trail shoe"),
    RankBy::saturate("views", 10_000.0),
]);

// Conjunctive: both fields must match
let rank_by = RankBy::product_all(vec![
    RankBy::bm25("title", "trail"),
    RankBy::bm25("description", "waterproof"),
]);
```

A product is `0` (excluded) whenever any factor is `0`.

## Ranking by attributes only

An expression without BM25 is scored from attributes:

```rust
let hot = col.query(&RankBy::saturate("views", 10_000.0), Some(&filter), 20)?;
```

A single monotone score over a `filterable` attribute (`attribute`, `saturate`, `decay` in `Forward` / `Backward` mode, optionally scaled by a constant) reads the index in order and stops after about `top_k` entries. Other attribute-only expressions evaluate every document that passes the filter, so give them a selective `filter` on big collections.

## Sorting by attributes (`query_ordered` / `order_by`)

```rust
use izzydb::{NullsOrder, OrderBySpec};

let hits = col.query_ordered(
    &[
        OrderBySpec::desc("price").with_nulls(NullsOrder::Last),
        OrderBySpec::asc("name"),
    ],
    Some(&Filter::Eq("category".into(), "shoes".into())),
    20,
)?;
// `col.order_by(...)` is an alias for `col.query_ordered(...)`.
// Vec<OrderedResult { id, sort_values }>
```

- The primary key must be `filterable` and an ordered scalar (`Int`, `Uint`, `Float`, `String`); further keys only need to be ordered scalars.
- Equal keys tie-break by the next key, then ascending `DocId`.
- `NullsOrder::Last` (default) / `First` places documents missing the attribute; it is independent of `Asc` / `Desc`.
- `query_ordered_with_facets(order_by, filter, top_k, facets)` also returns facets over all documents matching `filter`.
- On large collections the first `top_k` rows of an unfiltered or broadly filtered query are read straight from the attribute index.

## Facets and hybrid search

`query_bm25_with_facets` and `hybrid_search_with_facets` accept boosted expressions. The facet population is every document with a score `> 0` that matches `filter`. For expressions beyond plain BM25, per-facet filter modes (`ExcludeSelf`, ...) are ignored, as with other fixed document sets. Hybrid search uses the boosted ranking as its lexical leg.

## Errors

`InvalidQuery`: unknown or non-numeric attribute, negative or non-finite weights / `missing`, non-positive `midpoint` / `exponent`, an empty `product_all`, a filter that is invalid, an empty or invalid `order_by`, or a non-`filterable` primary sort attribute.

For the execution strategy and performance, see [`docs/architecture/ranking-and-boosting.md`](../architecture/ranking-and-boosting.md).
