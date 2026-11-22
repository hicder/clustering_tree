# Hybrid Search (Vector + BM25)

Vector search finds *similar meaning*; BM25 finds *matching words*. Neither is best alone: embeddings blur exact names, codes, and rare terms, while keyword search misses paraphrases. `hybrid_search` runs both and merges the two ranked lists with **Reciprocal Rank Fusion (RRF)**.

```rust
use izzydb::{HybridSearchOptions, RankBy, SearchOptions};

let rank_by = RankBy::sum(vec![
    RankBy::product(2.0, RankBy::bm25("title", "trail runner")),
    RankBy::bm25("description", "waterproof trail shoe"),
]);

let opts = HybridSearchOptions::new(10, SearchOptions::new(20, 8).with_rerank(100));

let hits = col.hybrid_search(&query_embedding, &rank_by, opts, Some(&filter))?;
for h in hits {
    println!(
        "doc {} fused={:.4} vector_rank={:?} bm25_rank={:?}",
        h.id, h.score, h.vector_rank, h.bm25_rank
    );
}
```

`hybrid_search(query_vec, rank_by, options, filter)` returns `Vec<HybridSearchResult { id, score, vector_rank, bm25_rank }>` sorted by descending fused `score`. `vector_rank` / `bm25_rank` are 1-based positions in each list, or `None` if the document appeared in only one.

The `filter` is applied to **both** retrievals.

## How fusion works

For each document, its fused score is the sum over the lists it appears in:

```
score(d) = vector_weight / (rrf_k + vector_rank(d))
         + bm25_weight   / (rrf_k + bm25_rank(d))
```

Because RRF uses **ranks**, not raw scores, you never have to reconcile cosine distances with BM25 scores, which live on unrelated scales. A document that ranks well in both lists beats one that is first in only one.

Ties are broken by ascending `DocId`.

## `HybridSearchOptions`

| Builder | Default | Effect |
|---|---|---|
| `HybridSearchOptions::new(k, vector_search)` | — | `k` = final results returned; `vector_search` = the `SearchOptions` for the vector leg. Its own `k` is how many vector candidates enter fusion. |
| `.with_bm25_candidates(n)` | `max(vector_search.k, 2 * k)` | How many BM25 hits enter fusion. |
| `.with_rrf_k(x)` | 60.0 | Damping constant (≥ 0). Larger values flatten the difference between ranks; smaller values reward top ranks more. 60 is the conventional choice. |
| `.with_weights(v, b)` | 1.0, 1.0 | Weight of the vector and BM25 lists (each ≥ 0). Set `b > v` to favor keyword matches. |

Invalid values return `TreeError::InvalidQuery`.

### Tuning

- **Candidate depth:** Fusion only sees what each leg returns. Make the vector `k` and `bm25_candidates` comfortably larger than the final `k` (2–5×).
- **Weights:** Start at 1:1. If users type exact product names or codes, raise the BM25 weight; if queries are natural-language questions, raise the vector weight. Evaluate on real queries.
- **Vector-only or keyword-only?** Set the other weight to `0.0`, or call `search` / `query_bm25` directly.

## What if a leg returns nothing?

Fusion still works: documents from the other list are returned (with `None` rank for the missing leg).

## Hybrid with facets

`hybrid_search_with_facets` returns hits plus aggregations over a chosen population — see [Faceting](08-faceting.md#hybrid).
