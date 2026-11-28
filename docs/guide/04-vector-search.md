# Vector Search

```rust
use izzydb::SearchOptions;

let hits = col.search(&query, SearchOptions::new(10, 8), None)?;
// Vec<SearchResult { id: u64, distance: f32 }>, nearest first
```

`Collection::search(query, options, filter)` returns up to `k` approximate nearest neighbors, ordered by ascending `distance`. The third argument is an optional [`Filter`](05-filtering.md).

Errors: `DimensionMismatch` and `NonFiniteVector` for a bad query; `InvalidFilter` for a bad filter. If `k == 0` or the collection is empty, you get an empty `Vec`.

## How the search works (just enough to tune it)

Vectors live in a tree of clusters with ~100 children per node. A query:

1. **Descends** the tree with a *beam*: at each level it keeps the `beam_width` closest clusters.
2. **Scans** the vectors in the surviving leaf clusters using 1-bit quantized codes (RaBitQ), which is very fast but approximate.
3. Optionally **reranks** the best candidates using their exact full-precision vectors.

Wider beams look at more clusters (higher recall, more latency). Reranking fixes the ordering error that 1-bit codes introduce.

## `SearchOptions`

| Builder | Field | Default | Effect |
|---|---|---|---|
| `SearchOptions::new(k, beam_width)` | `k`, `beam_width` | — | Number of results; clusters kept per tree level. |
| `.with_rerank(n)` | `rerank_candidates` | `None` (no rerank) | Take the best `n` candidates by quantized score and re-score them exactly. Reported distances are then computed from the full-precision vectors. |
| `.with_exact_scan_threshold(t)` | `exact_scan_threshold` | 4,096 | If an exact materialized filter produces ≤ `t` matching vector rows, skip ANN traversal and compute exact distances directly. |
| `.with_max_expanded_leaves(m)` | `max_expanded_leaves` | `beam_width * 16` | Upper bound on leaves probed when a **filter** forces the search to keep expanding to find enough matches. |

### Choosing values

- **Start with** `SearchOptions::new(k, 8).with_rerank(10 * k)`. This is a good recall/latency balance for most embeddings.
- **Recall too low?** Raise `rerank_candidates` first (cheap), then `beam_width`.
- **Latency too high?** Lower `beam_width` (try 4), and lower `rerank_candidates`.
- `rerank_candidates` should be ≥ `k`. Without reranking, returned `distance` values are *quantized estimates*, so don't compare them to exact distances you compute yourself.
- Measure recall on your own data. The repo's `recall_bench` shows how: `cargo run --release -p izzydb-bench --bin recall_bench`.

## Filtered vector search

```rust
let filter = Filter::And(vec![
    Filter::Eq("category".into(), "shoes".into()),
    Filter::Lte("price".into(), 100.0_f64.into()),
]);
let opts = SearchOptions::new(10, 8).with_rerank(100).with_max_expanded_leaves(256);
let hits = col.search(&query, opts, Some(&filter))?;
```

The planner chooses the cheapest strategy automatically:
- **Selective filters** ($\le$ `exact_scan_threshold`, default `4,096` vector rows) switch to exact distance scoring (`ExactScan`) for 100% exact recall.
- **Medium-selectivity filters** prune ANN leaves directly against the matching `RowId` set (`MaskedTraversal`), skipping clusters with no matching documents.
- **Broad or unselective filters** (`PostFilter`) avoid upfront materialization, checking broad inline equality predicates via cached per-segment posting bitmaps and verifying general scalar filters in distance-ordered chunks from sealed `cf_columns` blocks or row storage, stopping once enough candidates pass inside the top-`k`/rerank distance cutoff.

When a filter is very selective above `exact_scan_threshold` and matches are scattered across many clusters, the search keeps expanding to further leaves until it finds `k` matches or reaches `max_expanded_leaves`; raise that cap if highly selective filters return fewer than `k` hits that you know exist.

See [Filtering](05-filtering.md) for the full predicate language.

## Choosing a metric

Pick the metric your embedding model was trained for (most text embedders: `Cosine`). The metric is fixed at collection creation. See [Collections & Schema](02-collections-and-schema.md#distance-metrics).

## Search + facets

To get facet counts alongside hits in one call use `search_with_facets` — see [Faceting](09-faceting.md#combining-facets-with-search).
