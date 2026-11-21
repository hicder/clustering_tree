# Full-Text Search & BM25 Architecture (`FtsIndex`)

`izzydb` implements native BM25 full-text search, token/phrase filtering, multi-field `RankBy` combinators, and Reciprocal Rank Fusion (RRF) hybrid search decoupled from vector clustering.

## 1. Crate Structure

- **`izzydb-core`:** `FullTextSearchConfig`, `Tokenizer`, `Language`, `TokenQuery`, `Filter::Contains*`, `RankBy`, `Bm25Overrides`, `TermId`, and `FtsDocId` / `FtsBlockId` / `FtsSkipChunkId` helpers.
- **`izzydb-fts`:**
  - `analyzer.rs`: `FtsAnalyzer` backed by `alyze` (`WordV4` UAX #29 DFA, `Whitespace`, `PreTokenizedArray`, optional `fts-cjk` feature for `SmartCn`/`Kuromoji`/`Nori`), Snowball stemming (18 languages), stopword removal (13 languages), ASCII folding, and `+2` position gaps between `StringArray` elements.
  - `block.rs`: `FtsMeta` (`next_fts_doc_id`, `next_term_id`, per-attr corpus stats), `TermRecord` (`term_id`, `TermStats`, inline posting block when single-block and `doc_freq <= 8`), 256-slot `DocLenBlock` / `DocLenBlockView` (`u8`/`u16`/`u32`), 256-block `SkipBlock` / `SkipBlockView` / `SkipBlockLayout`, v3 TF-only Sparse/Dense `PackedPostingBlock` / `PostingBlockView` (`u8`/`u16`/`u32`, `max_tf`, `min_dl_over_tf` bounds), and delta-varint `PackedPositionBlock` / `PositionBlockView` (`u16` or `u32` offsets, `slot_contains_pos`).
  - `simd.rs`: Runtime-dispatched AVX-512F, AVX2+FMA, aarch64 NEON, and scalar kernels for `dl_norm` preparation, dense/sparse BM25 block accumulation, masked max-reduction, threshold bitmask filtering (`filter_ge_256`), and `PrefixMax`/`MaxBranches` vector merges.
  - `bm25.rs`: BM25 IDF/TF/QTF scoring, `min_dl_over_tf`-based upper bounds, and SIMD block scatter/dense accumulators.
  - `maxscore.rs`: SIMD Vectorized Block-Max MAXSCORE top-$k$ execution supporting `Sum`, `Max`, `Product`, `last_as_prefix` disjunctions, batched skip cursor opening, progressive non-essential slot pruning, pushdown `RoaringBitmap<FtsDocId>` masks, and lazy filter predicates.
- **`izzydb-storage`:** 5 dedicated FTS RocksDB column families (`cf_fts_rev_doc_map`, `cf_fts_terms`, `cf_fts_skips`, `cf_fts_postings`, `cf_fts_positions`, `cf_fts_doc_lens` — 6 including `cf_fts_doc_lens`; forward `fts_doc_id` lives inside `cf_doc_loc`), `RocksSkipCursor` (batched chunk-0 pinned skip fast path + lazy sequential `cf_fts_postings` iterator), cached `FtsRevBlock` table, cached `RocksDocLenCursor`, and atomic `TxContext` overlays.
- **`izzydb`:** `FtsIndex`, `Collection::query_bm25` / `Collection::query`, `Collection::hybrid_search`, `Collection::compact_fts_stats`, FTS filter integration in `Collection::search` and `Collection::resolve_filter_doc_ids`, and exhaustive `verify_collection_invariants` parity checks.

## 2. Cluster-Decoupled Posting Blocks, `TermId` Dictionary & `FtsDocId` Recycling

Rather than keying FTS postings by vector `(cluster_id, local_id)` (which causes heavy write amplification during vector tree splits, merges, and LIRE boundary reassignments), `FtsIndex` assigns each FTS document a dense `FtsDocId: u32` (reserving `FtsDocId::MAX` as the sentinel for "no FTS ID" in `cf_doc_loc`):

- `block_id = fts_doc_id >> 8`
- `slot = (fts_doc_id & 0xFF) as u8`

Reverse mapping (`FtsDocId -> (DocId, leaf_id, local_id)`) is stored in 256-slot `FtsRevBlock` blobs in `cf_fts_rev_doc_map` and cached in `Storage::fts_rev_map_cache`, enabling RAM-speed translation from `RoaringBitmap<FtsDocId>` hits to `(leaf_id, slot_mask)` without RocksDB lookups during hybrid/filtered vector search.

To avoid block fragmentation under churn, deleted `FtsDocId`s are persisted in `cf_meta["fts_free_ids"]` (`RoaringBitmap`, managed transactionally via `TxContext`):
- **Deallocation (`remove_doc`):** inserts `fts_doc_id` into `fts_free_ids` and shrinks `meta.next_fts_doc_id` while `next_fts_doc_id - 1` is present in `fts_free_ids`. When tail blocks empty out, `TxContext::commit` trims trailing `None` entries from `Storage::fts_doc_lens_cache` and `Storage::fts_rev_map_cache`.
- **Allocation (`upsert_doc`):** pops `fts_free_ids.min()` first to refill earlier 256-slot blocks densely before advancing `meta.next_fts_doc_id`.

Each `(attr_id, token)` in `cf_fts_terms` stores a `TermRecord` containing a unique monotonic `term_id: TermId (u32)`, `TermStats`, and—for single-block terms with `doc_freq <= INLINE_TERM_MAX_DF` (`8`)—the encoded v3 posting block inline. Out-of-line `cf_fts_skips`, `cf_fts_postings`, and `cf_fts_positions` rows use compact 8-byte keys `term_id (u32 BE) || block_or_chunk_id (u32 BE)` with 4-byte prefix Bloom filters (`cf_fts_postings` and `cf_fts_skips` uncompressed).

Document lengths are stored once per `(attr_id, block_id)` in `cf_fts_doc_lens` (`DocLenBlock`), while each posting block holds a v3 `PackedPostingBlock` storing only `tf` (switching automatically between sparse `(tfs, slots)` and dense `(bitmap, [tf; 256])` at `POSTING_DENSE_THRESHOLD = 96`) plus exact `(max_tf, min_dl_over_tf)` bounds mirrored into 256-block `SkipBlock` chunks in `cf_fts_skips`.

When a delete or update lowers a posting's `(tf, dl/tf)` contribution at a term's current `TermStats` extreme (`max_tf` or `min_dl_over_tf`) and the term spans `<= 64` posting blocks (`INLINE_TERM_STATS_RECOMPUTE_MAX_BLOCKS`), `FtsIndex` recomputes exact `TermStats` inline from `cf_fts_skips` chunk summaries (or inline `TermRecord` headers); `Collection::compact_fts_stats` recomputes exact bounds across all terms on demand. Updates whose FTS attributes did not change skip `FtsIndex` analysis and I/O altogether.

## 3. Query Execution & Hybrid Search

- **Vectorized Block-Max MAXSCORE:** Terms are sorted by ascending global upper bound and dynamically partitioned into non-essential and essential sets as the top-$k$ threshold $\theta$ rises. Block pruning consults only inline `TermRecord` headers or `cf_fts_skips` chunks (`RocksSkipCursor`); `cf_fts_postings` is only read via a lazy sequential iterator on blocks that survive pruning, accumulating into a stack-allocated `[f32; 256]` buffer. See [SIMD Block-Max MAXSCORE & Data Layout](blockmax-simd.md) for the exact binary formats and SIMD kernel pipeline.
- **Filters:**
  - `ContainsAnyToken`, `ContainsAllTokens`, and `ContainsTokenSequence` gallop over `RocksSkipCursor` / inline `TermRecord`s before reading surviving posting blocks, then batch-fetch `cf_fts_positions` via `multi_get_fts_positions_pinned` and scan `PositionBlockView` (`slot_contains_pos`) without allocating per-slot vectors for follow-up tokens.
  - Vector `Collection::search` resolves FTS subtrees via `Storage::resolve_fts_bitmap_to_leaf_masks` into `(leaf_id, LeafMask)` directly in RAM inside `PreparedFilter`, supporting arbitrary `And`/`Or`/`Not` combinations with attribute filters.
  - Filtered BM25 pushes down pure-FTS and selective attribute filters into `RoaringBitmap<FtsDocId>` block masks and falls back to zero-copy `DocView::matches_filter` lazy evaluation over `cf_docs` (`DBPinnableSlice`) for broad/negated non-FTS filters.
- **Hybrid Search:** `Collection::hybrid_search` executes vector search and `query_bm25` under the same `Option<&Filter>` and fuses rankings via Reciprocal Rank Fusion (RRF).
