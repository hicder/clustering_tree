# Search Path

This document describes both the unfiltered and attribute-filtered ANN search paths in `Collection` ([`crates/izzydb/src/collection.rs`](../../crates/izzydb/src/collection.rs)) and [`crates/izzydb/src/search.rs`](../../crates/izzydb/src/search.rs).

## Entry Points

- `Collection::search(query, opts, None)` / `ClusteringTree::search*`: unfiltered beam traversal fast path with zero filtering overhead.
- `Collection::search(query, opts, Some(filter))`: cluster-aware filtered vector search (`search::search_filtered`, a thin wrapper over `search::search_filtered_prepared`).
- `Collection::{search_with_facets, query_bm25_with_facets, hybrid_search_with_facets}`: the same ranking paths plus exact facets (see below).

---

## Unfiltered Path (`search::search`)

1. **Query preparation:** `prepare_vector_cow(query, config.dim, config.metric)` validates length/finiteness and L2-normalizes for `Cosine`.
2. **Top-down beam traversal:** Walks `internal_cache` from `meta.root_id` to `level == 0`, keeping the top `opts.beam_width` nodes per level by `search_distance`.
3. **Stage 1 — Batched RaBitQ leaf scoring:** Sorts leaf IDs, calls `storage.multi_get_leaf_nodes_pinned`, rotates the query once (`QueryScratch`), and runs `score_leaf_rabitq` (`score_leaf_rabitq_masked` with `live_mask` when a leaf has tombstones) over `LeafNodeView`s using pre-stored `code_popcounts` and 8-wide SIMD-tiled codes (AVX-512 VPOPCNTDQ / NEON / scalar).
4. **Stage 2 — Optional full-vector reranking:** When `opts.rerank_candidates` is `Some(r)`, shortlists `r.max(k)` candidates, batch-fetches full vectors from `cf_vectors`, computes exact `search_distance_bytes`, and returns top `k`.

---

## Filtered Path (`search::search_filtered`)

### Step 1: Cluster-Level Pruning (`AttributeIndex::prepare_filter_with_fts`)

Evaluates the validated `Filter` AST against `cf_attr_clusters`, cached attribute zones in `default` (`AttrZoneMap`), and `FtsIndex` to produce `ClusterCandidates`:

- `Eq(attr, val)` / `Contains(attr, val)` / `IsNotNull(attr)`: point lookup in `cf_attr_clusters` $\to$ `Subset(bm)`.
- `In(attr, vals)` / `ContainsAny(attr, vals)`: batched `multi_get_attr_cluster_bitmaps` + bitwise `|` $\to$ `Subset(union)`.
- `Lt` / `Lte` / `Gt` / `Gte`: pruned in RAM via cached attribute zones in `default` (`AttrZoneMap`) against per-leaf `[min_b, max_b]` bounds (0 cluster RocksDB lookups), producing a `PreparedNode::Range` evaluated later via prefix-bounded cluster-first range scans on `cf_attr_postings`. Adjacent/sibling `Range` nodes on the same attribute inside `And` are coalesced into one `[lo, hi]` scan and ordered after selective equality/FTS predicates.
- `ContainsAllTokens` / `ContainsAnyToken` / `ContainsTokenSequence`: evaluates matching `FtsDocId`s in `FtsIndex` and resolves `(leaf_id, slot_mask)` directly in RAM via cached FTS reverse-map blocks in `default` (`FtsRevBlock`) into `PreparedNode::LeafSlotMasks`.
- `And`: intersects `Subset` bitmaps (`&=`), short-circuiting immediately to empty if any child is empty.
- `Or`: returns `All` if any child is `All`, otherwise unions `Subset` bitmaps (`|=`).
- `Not` / `NotEq` / `NotIn` / `IsNull`: returns `All` (except `NotEq(attr, Null)`, which prunes like `IsNotNull`), deferring exact negation to per-leaf evaluation.

If `ClusterCandidates::Subset(bm)` is empty, search returns `Ok(Vec::new())` immediately with zero leaf or vector reads.

### Step 2: Exhaustive-Scan Crossover (Highly Selective Filters)

When `ClusterCandidates::Subset(bm)` satisfies `bm.len() <= opts.exhaustive_cluster_threshold` (default `8`):

1. Batch-fetches `PackedLeafNode` blobs for all `bm` leaves.
2. Evaluates exact per-leaf slot bitmasks (`LeafMask`) via `AttributeIndex::eval_leaves_masks` (one sorted `batched_multi_get_cf` on `cf_attr_postings` via `multi_get_attr_posting_slices` + any `Range` prefix scans).
3. Iterates set bits via `.trailing_zeros()` (`slot == local_id`), collects live `view.doc_id(slot)`s, batch-fetches full vectors from `cf_vectors`, scores exact `search_distance_bytes`, and returns top `k` (100% exact recall, bypassing tree traversal and RaBitQ).

### Step 3: Pruned Best-First Traversal + Dynamic Cluster Expansion

For broader filters (`All` or `bm.len() > exhaustive_cluster_threshold`):

1. **Ancestor pruning:** When `Subset(bm)` covers `< 50%` of leaves, walks `leaf_parent` and `internal_cache` upward from each qualifying leaf to build `allowed_internal: RoaringBitmap`. Internal subtrees with zero qualifying leaves are skipped.
2. **Best-first priority queue (`BinaryHeap` min-heap):** Pops closest nodes from `(meta.root_id, 0.0, meta.root_level)`, pushing only allowed internal and leaf children.
3. **Batched leaf evaluation & masked RaBitQ scoring:**
   - Pulls batches of closest qualifying leaves (starting at `opts.beam_width` and doubling on subsequent rounds).
   - Batch-fetches `PackedLeafNode` blobs and evaluates exact slot-indexed `LeafMask`s directly via `AttributeIndex::eval_leaves_masks` (including exact complement `live_mask & !sub` for `Not` / `NotEq` / `NotIn` / `IsNull`, zero-copy `LeafPostingView` decoding, and zero-alloc cluster-first `scan_attr_posting_range_into_words`).
   - Intersects with `view.live_mask_bytes` when `view.has_tombstones()` and skips leaves whose resulting mask is empty (e.g. false-positive clusters under `And`).
   - Passes `mask.as_slice()` directly to `score_leaf_rabitq_masked` (`0..view.slot_cap`) with no `local_id -> slot` translation loop, skipping 8-entry SIMD tiles whose mask byte is `0`.
4. **Dynamic expansion:** Continues pulling leaf batches until `candidates.len() >= target && probed_leaves >= opts.beam_width` (where `target = rerank_candidates.unwrap_or(k).max(k)`), the heap is empty, or `probed_leaves >= opts.max_expanded_leaves.unwrap_or(beam_width * 16)`.
5. **Stage 2 exact reranking:** Identical to unfiltered search (`cf_vectors` batched multi-get + exact `search_distance_bytes`).

---

## Faceted Search (`Collection::*_with_facets`)

Ranking is executed exactly as above; faceting is a second, independent pass over the *facet population* (details in [filtering-and-inverted-index.md](filtering-and-inverted-index.md#7-faceting)).

| Entry point | Facet population |
|---|---|
| `facet(filter, facets)` | all live docs matching `filter` (whole collection if `None`) |
| `search_with_facets(.., VectorFacetScope::FilterMatches)` | all docs matching `filter`; the `PreparedFilter` built for ANN search is reused via `search_filtered_prepared` |
| `search_with_facets(.., TopK)` | the returned top-`k` hits |
| `search_with_facets(.., Candidates(n))` | the top-`max(n, k)` scored ANN candidates (hits are the first `k`) |
| `query_bm25_with_facets` | docs with BM25 score `> 0` for `rank_by` **and** matching `filter` (`FtsIndex::eval_rank_by_match_bitmap`) |
| `hybrid_search_with_facets(.., FilterMatches)` | all docs matching `filter` |
| `hybrid_search_with_facets(.., FusedCandidates)` | union of vector and BM25 candidates entering RRF |
| `hybrid_search_with_facets(.., Bm25AndFilterMatches)` | docs matching `rank_by` and `filter` |

`FacetedSearchResponse::total_hits` is the size of the primary population (the effective filter equal to the query filter).
