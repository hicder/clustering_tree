# Ranking, Boosting, and Attribute Sorting

This document describes how `izzydb` executes the generalized `RankBy` AST (BM25, filter boosts, attribute scores, multiplicative products) and attribute-ordered queries (`OrderBy`). Implementation: [`rank.rs`](../../crates/izzydb-core/src/rank.rs) (AST, math, normalization), [`maxscore.rs`](../../crates/izzydb-fts/src/maxscore/mod.rs) (pruned lexical execution), [`rank_eval.rs`](../../crates/izzydb/src/rank_eval/mod.rs) (planner and exact executors), [`order_by.rs`](../../crates/izzydb/src/order_by.rs) (index-order scan).

Attribute scores read through `AttrReader` (`cf_columns` on sealed segments, `cf_detached` on detached columns, and zero-copy `RowView` in `cf_rows` on open segments); filter boosts reuse `FilterPlan`; `OrderBy` streams value-major `cf_attr_postings` / `cf_detached_postings` keys.

## 1. The AST and its invariants

```text
RankBy = Const(f32) | Bm25{..} | Filter(Filter)
       | Attribute{attr, missing} | Saturate{attr, midpoint, exponent, missing}
       | Decay{attr, origin, midpoint, exponent, mode, missing}
       | Sum(Vec) | Max(Vec) | Product(Vec)
```

- **Scores are finite and `>= 0`.** Negative numeric values clamp to `0` in `Attribute` / `Saturate`; every node therefore yields a non-negative number, which is what makes the product bounds below sound.
- **Zero scores are excluded**; ties break by ascending `DocId`.
- **Curves** (overflow-safe, computed in `f64`):
  - `Saturate = 1 / (1 + (x/midpoint)^-e)` = `x^e / (x^e + midpoint^e)`, `0` for `x <= 0`.
  - `Decay = 1 / (1 + (d/midpoint)^e)` where `d` is the distance from `origin` by `DecayMode` (`Symmetric` `|x-o|`, `Backward` `max(o-x,0)`, `Forward` `max(x-o,0)`); `1.0` when `d <= 0`.
  - `missing` is the score for documents lacking the attribute.
- `Filter(f)` is `1.0` when the document matches `f`, else `0.0`; `f` may contain attribute, FTS token/phrase, and boolean predicates.

`RankBy::normalize()` canonicalizes an expression: it flattens nested `Sum` / `Max` / `Product`, folds constants, drops zero terms, and rewrites `w * Sum(xs)` to `Sum(w * x)`. `Product([Const(w), e])` is exactly the old weighted product, so existing queries compile to the same MAXSCORE clauses.

## 2. Execution regimes

`Collection::query` normalizes the expression and picks a plan (`rank_eval::classify`):

| Normalized shape | Plan | Cost model |
|---|---|---|
| **Pure lexical**: BM25 leaves under `Sum` / `Max` / constant-weight `Product` | Block-Max MAXSCORE, unchanged | identical to plain BM25 |
| **`L * m(doc)`**: one pure-lexical factor `L`, other factors have no BM25 (`Saturate`, `Decay`, `Filter`, `1 + w*Decay(..)`, ...) | MAXSCORE on `L * UB(m)`; survivors rescored exactly | forward attribute read only for candidates with `L * UB(m) > theta` |
| **`L + a(doc)`**: lexical terms plus non-lexical additive terms | MAXSCORE with threshold `theta - UB(a)`; exact backfill of non-lexical docs when needed | backfill skipped when the lexical top-k is full and its weakest score `> UB(a)` |
| **No BM25** | single monotone `filterable` attribute: index-order scan; otherwise candidate scan | `O(k)` index entries (monotone) / `O(candidates)` |
| **Anything else** (e.g. `BM25 x BM25`) | exact evaluation over the positive candidate universe | unpruned over candidates |

### 2.1 MAXSCORE with a modifier

`izzydb-fts::execute_rank_by_with` accepts a `RankModifier { mult_ub, add_ub, rescore }`. The true score is

```text
score(doc) = L(doc) * m(doc) + a(doc),   0 <= m <= mult_ub,   0 <= a <= add_ub
```

- The lexical clauses are compiled with weight `mult_ub`, so the SIMD score buffers hold `L * mult_ub >= L * m`, an upper bound of the lexical part.
- Every pruning decision (block skipping, essential/non-essential partitioning, the candidate threshold) uses `prune_theta = max(theta - add_ub, 0)`. A document can only enter the top-k if `L * mult_ub + add_ub > theta`, so no qualifying document is pruned.
- Candidates that survive reach `rescore(doc_id, L * mult_ub)`, which reads the document once, evaluates `m` and `a`, and returns the exact final score. The heap and `theta` hold exact scores.

Upper bounds: `Const(c) = c`, `Filter = 1`, `Saturate`/`Decay = max(1, missing)`, `Sum = sum`, `Max = max`, `Product = product`. `Attribute` is unbounded by definition; its bound is read from per-segment `0x12` zone maps (or largest key in `cf_attr_postings` / `cf_detached_postings`) for a `filterable` attribute. A modifier with no finite bound (e.g. `Attribute` on a non-filterable attribute) falls back to the exact plan.

### 2.2 Additive backfill

`Decay` and `Saturate` can be `> 0` for documents that match no BM25 term, so `BM25 + w*Decay(..)` may rank non-lexical documents. After MAXSCORE returns the exact top-k *over lexical documents*, the planner scans non-lexical documents (those outside the lexical match bitmap) by `a(doc)` alone and merges the two sets. This is exact because a non-lexical document scores exactly `a(doc)`. The scan is skipped when the lexical result is full and its weakest score strictly exceeds `add_ub`. When the additive term is `w * Filter(..)`, the scan universe is just that filter's matches.

### 2.3 Filter leaves

`RankBy::Filter` leaves whose predicate is exactly resolvable from the indexes (FTS token/phrase predicates, selective filters) are resolved once per query into a `RowSet`, making per-document scoring a bitmap/set lookup instead of re-analyzing text. Other filters are evaluated per row via `RowView::matches_filter` (plus `cf_detached` / attribute decoding when detached or FTS predicates are present).

### 2.4 Exact fallback

For shapes MAXSCORE cannot prune (conjunctive `BM25 x BM25`, nested sums inside products, ...), `general_eval`:

1. derives a candidate superset from the expression (`Product` = intersection of children's positive sets, `Sum` / `Max` = union; non-lexical leaves widen to "all documents");
2. applies the query filter;
3. fetches BM25 scores of each maximal pure-lexical subtree for those candidates via MAXSCORE restricted to the candidate bitmap;
4. evaluates the full expression per candidate and keeps the top-k.

This is exact but not pruned. A pruned conjunctive clause (block-level `max(current_block)` skipping across factors) is a possible future optimization; it does not change results.

### 2.5 Monotone attribute fast path

`Attribute`, `Saturate`, `Decay(Forward | Backward)` over a `filterable` attribute with `missing = 0`, optionally scaled by a constant, are monotone in the stored value. The planner streams the attribute's value groups in score order (via the `OrderBy` scanner), stops after the group in which `k` results were reached **and** every following group with an equal score (equal-score groups must be merged by `DocId`), then sorts by `score DESC, DocId ASC`.

## 3. Facets over boosted queries

Pure lexical expressions keep the posting-union bitmap population, so per-facet filter modes (`ExcludeSelf`, ...) work as before. Expressions with boosts / attribute scores are evaluated exactly into a fixed document set of all documents with `score > 0` that match the filter; as for `TopK` / `Candidates` populations, per-facet filter modes are ignored for those. Hybrid search inherits boosted expressions on its lexical leg unchanged.

## 4. Attribute sorting (`OrderBy`)

`Collection::order_by(&[OrderBySpec], filter, top_k)` returns the top-k documents by one or more ordered scalar attributes (`Int`, `Uint`, `Float`, `String`), `Asc` / `Desc`, `NullsFirst` / `NullsLast` per key. The primary key must be `filterable` (inline or detached); secondary keys are read via `AttrReader`.

- **Streaming path (default).** Walks value-major `cf_attr_postings` (or `cf_detached_postings`, merging with `0x14` overwritten sets) for the primary `(attr_id, type_tag)` prefix, forward for `Asc` and reverse for `Desc`; keys are grouped by `(encoded_val, seg)` in sorted value order. For each value group, segment `SegPosting`s are merged into a `RowSet`, ANDed with `SegmentTableSnapshot` liveness and the filter, resolved to `DocId`s via `RowResolver`, and ordered by any secondary keys (`DocId` tie-breaker). The scan **stops as soon as `top_k` rows exist** at a value boundary.
- **Selective-filter crossover.** When the filter materializes to at most `8,192` rows, matches are fetched via `AttrReader` and sorted in memory.
- **Nulls.** Documents with an absent/`Null` primary key (`IsNull`) are emitted before the scan for `NullsFirst`, or appended after it (only if rows remain) for `NullsLast`.
- **Float ordering.** `-0.0` and `0.0` share one key (the index normalizes them), so they tie and break by `DocId`; NaN is rejected at write time.

## 5. Performance

`cargo run --release -p izzydb-bench --bin rank_bench` (20k docs, top-10, one core, release build):

| Query | Latency |
|---|---|
| BM25 (baseline) | ~97 us |
| `2 x BM25` (constant product) | ~95 us (no regression: same clauses) |
| `BM25 + 3 x Filter(phrase)` | ~1.3 ms |
| `BM25 x (1 + 2 x Decay(created))` | ~0.8 ms |
| `BM25 x Saturate(popularity)` | ~0.35 ms |
| `BM25 + 0.05 x Decay(created)` | ~0.26 ms |
| `Saturate(popularity)` (monotone index scan) | ~20 us |
| `OrderBy created DESC` (unfiltered) | ~23 us |
| `OrderBy popularity DESC, created ASC` | ~30 us |
| `BM25 x BM25` (exact fallback) | ~6 ms |
| `Sum(Saturate, Filter)` (candidate scan, no BM25) | ~12 ms |

Modifier queries cost more than plain BM25 because surviving candidates read attributes via `AttrReader` / `cf_rows`; a strong modifier bound (e.g. a large additive weight) weakens pruning because `theta - add_ub` stays small.

## 6. Testing

`crates/izzydb/tests/rank_boost_tests.rs` compares every plan against a brute-force reference evaluator (full-collection scores, with and without filters, `k` in `{1, 3, 10, all}`, after updates and deletes); non-BM25 expressions are compared with exact `(score, DocId)` equality. `crates/izzydb/tests/order_by_tests.rs` compares `query_ordered` against an in-memory sort across all ordered types, directions, null placements, multi-key ties, selective and broad filters, and mutated / split trees.
