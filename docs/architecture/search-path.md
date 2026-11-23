# Search Path

This document specifies how IzzyDB v3 executes vector search (`Collection::search`), filtered vector search (five-regime planner), BM25/boosted ranking (`query`, `query_bm25`, `order_by`), hybrid RRF search, and faceted search over segmented storage.

Implementation sources:
- Entry points: [`crates/izzydb/src/collection.rs`](../../crates/izzydb/src/collection.rs)
- Filter planner (`FilterPlan`, cardinality estimation, zone pruning): [`crates/izzydb/src/filter_plan.rs`](../../crates/izzydb/src/filter_plan.rs)
- ANN search, exact scan, masked traversal & post-filter: [`crates/izzydb/src/search.rs`](../../crates/izzydb/src/search.rs)
- Segment/block row resolution (`RowId -> DocId`): [`crates/izzydb/src/segments.rs`](../../crates/izzydb/src/segments.rs)

---

## 1. Filter Planner (`filter_plan::plan_filter_with_trace`)

Unlike v2's per-leaf cluster bitmaps, v3 plans predicates over **`RowSet`** (`BTreeMap<SegmentId, SegmentRows>` — `All` or `RoaringBitmap<LocalId>`):

1. **Zero-I/O cardinality estimation:** Consults `default[0x06]` (inline) and `default[0x07]` (detached) live-doc counts, FTS `TermRecord.doc_freq`, and cached `0x12` per-segment zone maps to estimate selectivity without scanning postings.
2. **Materialization budget (`materialize_limit = max(65_536, live_rows / 8)`):**
   - Selective subtrees whose estimated match count fits under `materialize_limit` materialize an exact live `RowSet` (using `cf_attr_postings` / `cf_detached_postings` value-major seeks, `0x14` overwritten-set correction for detached columns, and `FtsIndex` block evaluation, ANDed with `SegmentTableSnapshot` live bitmaps).
   - Broad predicates (or large negations / unselective ranges exceeding `materialize_limit`) produce `FilterPlan::Lazy { residual, seed }`, deferring exact evaluation to zero-copy `RowView::matches_filter` (plus `cf_detached` fallback when detached attributes appear) on surviving ANN candidates.
   - `Empty` short-circuits immediately with zero ANN or vector reads.

---

## 2. Five Filtered Vector Search Regimes (`FilteredSearchRegime`)

`search::search_filtered_with_trace` selects one of five regimes based on the vector index (`has_vectors`), `FilterPlan`, and `SearchOptions`:

| Regime | Trigger | Execution |
|---|---|---|
| 1. `Unfiltered` | `filter = None` or `FilterPlan::All` | Top-down beam traversal over `internal_cache` (`opts.beam_width`), batched `cf_leaf_entries` RaBitQ scoring + `Liveness` mask, optional exact rerank from `cf_vectors`, `RowResolver::resolve_all` (`cf_row_doc_ids`) |
| 2. `Empty` | `FilterPlan::Empty` or 0 matching rows | Immediate empty `Vec::new()` (0 I/O) |
| 3. `ExactScan` | `FilterPlan::Materialized(rs)` with $\le$ `opts.exact_scan_threshold` (default `4,096`) vector rows | Skips the ANN tree completely: batch-fetches exact `[f32; dim]` vectors from `cf_vectors` in `RowId` order, computes exact `search_distance_bytes`, guarantees **100% exact recall** |
| 4. `MaskedTraversal` | `FilterPlan::Materialized(rs)` with $> \text{exact\_scan\_threshold}$ rows | Best-first beam traversal (`BinaryHeap`), testing each leaf's physical `row_ids` directly against `rs` (through `SegmentTableSnapshot::resolve_forwarding`), scoring matching live slots via `score_leaf_rabitq_masked`, dynamically expanding up to `max_expanded_leaves.unwrap_or(beam_width * 16)`, then exact rerank |
| 5. `PostFilter` | `FilterPlan::Lazy { residual, seed }` | Best-first leaf expansion fetching candidate batches, checking optional `seed` `RowSet`, lazily evaluating `residual` against `cf_rows` (`RowView`) / `cf_detached`, then scoring surviving full/quantized candidates up to `k` / expansion limit |

*(Collections created with `CollectionConfig::without_vectors()` reject `Collection::search` with `TreeError::InvalidConfig` and serve FTS, `filter`, `order_by`, and `facet` queries directly over `SegmentStore`.)*

---

## 3. Liveness, Forwarding Resolution & `DocId` Output

Because ANN leaves (`PackedLeafNode` v3) may briefly retain tombstoned or compaction-forwarded `RowId`s until `vacuum_ann` scrubs them:
- During both unfiltered and filtered leaf scoring, every leaf slot's `row_id` is checked against `SegmentTableSnapshot`: `resolve_forwarding` maps any `Forwarding` segment's `LocalId` via rank in its deletion bitmap, and dead/retired slots are masked out before scoring or reranking.
- Top-$k$ `RowId` winners are translated back to user `DocId`s once at the end via sorted batched lookups into `cf_row_doc_ids` (`RowResolver::resolve_all`, using the bounded `doc_id_block_cache`).

---

## 4. Faceted Search (`Collection::*_with_facets`)

Ranking executes identically to the paths above; faceting evaluates the requested `FacetSpec`s in a second pass over the selected scope (`FilterMatches`, `TopK`, `Candidates(n)`, `FusedCandidates`, `Bm25AndFilterMatches`), using `default[0x06]`/`default[0x07]` global counts (`"global"` / `"complement"`), `AttrReader` forward block passes (`"docs"`), or per-segment posting intersection (`"seg_masks"`).
