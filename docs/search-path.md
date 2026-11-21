# Search Path

This document describes both the unfiltered and attribute-filtered ANN search paths in `Collection` ([`crates/izzydb/src/collection.rs`](../crates/izzydb/src/collection.rs)) and [`crates/izzydb/src/search.rs`](../crates/izzydb/src/search.rs).

## Entry Points

- `Collection::search(query, opts, None)` / `ClusteringTree::search*`: unfiltered beam traversal fast path with zero filtering overhead.
- `Collection::search(query, opts, Some(filter))`: cluster-aware filtered vector search (`search::search_filtered`).

---

## Unfiltered Path (`search::search`)

1. **Query preparation:** `prepare_vector_cow(query, config.dim, config.metric)` validates length/finiteness and L2-normalizes for `Cosine`.
2. **Top-down beam traversal:** Walks `internal_cache` from `meta.root_id` to `level == 0`, keeping the top `opts.beam_width` nodes per level by `search_distance`.
3. **Stage 1 — Batched RaBitQ leaf scoring:** Sorts leaf IDs, calls `storage.multi_get_leaf_nodes_pinned`, rotates the query once (`QueryScratch`), and runs `score_leaf_rabitq` (AVX-512 VPOPCNTDQ / POPCNT / scalar) over `LeafNodeView`s.
4. **Stage 2 — Optional full-vector reranking:** When `opts.rerank_candidates` is `Some(r)`, shortlists `r.max(k)` candidates, batch-fetches full vectors from `cf_vectors`, computes exact `search_distance_bytes`, and returns top `k`.

---

## Filtered Path (`search::search_filtered`)

### Step 1: Cluster-Level Pruning (`AttributeIndex::prepare_filter`)

Evaluates the validated `Filter` AST against `cf_attr_clusters` (`RoaringBitmap<NodeId>`) to produce `ClusterCandidates`:

- `Eq(attr, val)` / `Contains(attr, val)` / `IsNotNull(attr)`: point lookup in `cf_attr_clusters` $\to$ `Subset(bm)`.
- `In(attr, vals)` / `ContainsAny(attr, vals)`: batched `multi_get_attr_cluster_bitmaps` + bitwise `|` $\to$ `Subset(union)`.
- `Lt` / `Lte` / `Gt` / `Gte`: prefix-scoped RocksDB range scan (`scan_attr_cluster_range`) over `(attr_id, type_tag, [lo..hi])` + bitwise `|` $\to$ `Subset(union)`, caching matching keys in `PreparedFilter`.
- `And`: intersects `Subset` bitmaps (`&=`), short-circuiting immediately to empty if any child is empty.
- `Or`: returns `All` if any child is `All`, otherwise unions `Subset` bitmaps (`|=`).
- `Not` / `NotEq` / `NotIn` / `IsNull`: returns `All` (except `NotEq(attr, Null)`, which prunes like `IsNotNull`), deferring exact negation to per-leaf evaluation.

If `ClusterCandidates::Subset(bm)` is empty, search returns `Ok(Vec::new())` immediately with zero leaf or vector reads.

### Step 2: Exhaustive-Scan Crossover (Highly Selective Filters)

When `ClusterCandidates::Subset(bm)` satisfies `bm.len() <= opts.exhaustive_cluster_threshold` (default `8`):

1. Batch-fetches `PackedLeafNode` blobs for all `bm` leaves.
2. Evaluates exact per-leaf `local_id` masks via `AttributeIndex::eval_leaves_masks` (one `batched_multi_get_cf` on `cf_attr_postings`).
3. Maps surviving `(leaf_id, local_id)` pairs to `DocId`s via `LeafNodeView`, batch-fetches full vectors from `cf_vectors`, scores exact `search_distance_bytes`, and returns top `k` (100% exact recall, bypassing tree traversal and RaBitQ).

### Step 3: Pruned Best-First Traversal + Dynamic Cluster Expansion

For broader filters (`All` or `bm.len() > exhaustive_cluster_threshold`):

1. **Ancestor pruning:** When `Subset(bm)` covers `< 50%` of leaves, walks `leaf_parent` and `internal_cache` upward from each qualifying leaf to build `allowed_internal: RoaringBitmap`. Internal subtrees with zero qualifying leaves are skipped.
2. **Best-first priority queue (`BinaryHeap` min-heap):** Pops closest nodes from `(meta.root_id, 0.0, meta.root_level)`, pushing only allowed internal and leaf children.
3. **Batched leaf evaluation & masked RaBitQ scoring:**
   - Pulls batches of closest qualifying leaves (starting at `opts.beam_width` and doubling on subsequent rounds).
   - Batch-fetches `PackedLeafNode` blobs and evaluates exact `local_id` bitmaps via `AttributeIndex::eval_leaves_masks` (including exact complement `all_local_ids - sub` for `Not` / `NotEq` / `NotIn` / `IsNull`).
   - Skips leaves whose resulting `local_mask` is empty (e.g., false-positive clusters where `And` terms matched different documents).
   - Builds an `allowed_slots: Vec<u64>` bitmask over `0..view.entry_count` and runs `score_leaf_rabitq_masked`, which skips 8-entry SIMD chunks whose mask byte is `0`.
4. **Dynamic expansion:** Continues pulling leaf batches until `candidates.len() >= target && probed_leaves >= opts.beam_width` (where `target = rerank_candidates.unwrap_or(k).max(k)`), the heap is empty, or `probed_leaves >= opts.max_expanded_leaves.unwrap_or(beam_width * 16)`.
5. **Stage 2 exact reranking:** Identical to unfiltered search (`cf_vectors` batched multi-get + exact `search_distance_bytes`).
