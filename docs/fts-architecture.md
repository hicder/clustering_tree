# Full-Text Search & BM25 Architecture (`FtsIndex`)

`izzydb` implements native BM25 full-text search, token/phrase filtering, multi-field `RankBy` combinators, and Reciprocal Rank Fusion (RRF) hybrid search decoupled from vector clustering.

## 1. Crate Structure

- **`izzydb-core`:** `FullTextSearchConfig`, `Tokenizer`, `Language`, `TokenQuery`, `Filter::Contains*`, `RankBy`, `Bm25Overrides`, and `FtsDocId`/`FtsBlockId` helpers.
- **`izzydb-fts`:**
  - `analyzer.rs`: `FtsAnalyzer` backed by `alyze` (`WordV4` UAX #29 DFA, `Whitespace`, `PreTokenizedArray`, optional `fts-cjk` feature for `SmartCn`/`Kuromoji`/`Nori`), Snowball stemming (18 languages), stopword removal (13 languages), ASCII folding, and `+2` position gaps between `StringArray` elements.
  - `block.rs`: `FtsMeta`, `TermStats` (`doc_freq`, `max_tf`, `min_dl_over_tf`), 256-slot `DocLenBlock` / `DocLenBlockView` (`u8`/`u16`/`u32`), v2 TF-only Sparse/Dense `PackedPostingBlock` / `PostingBlockView` (`u8`/`u16`/`u32`, `max_tf`, `min_dl_over_tf` bounds), and delta-varint `PackedPositionBlock` / `PositionBlockView`.
  - `simd.rs`: Runtime-dispatched AVX-512F, AVX2+FMA, aarch64 NEON, and scalar kernels for `dl_norm` preparation, dense/sparse BM25 block accumulation, masked max-reduction, threshold bitmask filtering (`filter_ge_256`), and `PrefixMax`/`MaxBranches` vector merges.
  - `bm25.rs`: BM25 IDF/TF/QTF scoring, `min_dl_over_tf`-based upper bounds, and SIMD block scatter/dense accumulators.
  - `maxscore.rs`: SIMD Vectorized Block-Max MAXSCORE top-$k$ execution supporting `Sum`, `Max`, `Product`, `last_as_prefix` disjunctions, progressive non-essential slot pruning, pushdown `RoaringBitmap<FtsDocId>` masks, and lazy filter predicates.
- **`izzydb-storage`:** 6 FTS RocksDB column families (`cf_fts_doc_map`, `cf_fts_rev_doc_map`, `cf_fts_terms`, `cf_fts_postings`, `cf_fts_positions`, `cf_fts_doc_lens`), prefix-bounded `RocksPostingCursor` with reusable seek buffer, and atomic `TxContext` overlays.
- **`izzydb`:** `FtsIndex`, `Collection::query_bm25` / `Collection::query`, `Collection::hybrid_search`, FTS filter integration in `Collection::search`, and exhaustive `verify_collection_invariants` parity checks.

## 2. Cluster-Decoupled Posting Blocks

Rather than keying FTS postings by vector `(cluster_id, local_id)` (which causes heavy write amplification during vector tree splits, merges, and LIRE boundary reassignments), `FtsIndex` assigns each document a dense monotonic `FtsDocId: u32`:

- `block_id = fts_doc_id >> 8`
- `slot = (fts_doc_id & 0xFF) as u8`

Document lengths are stored once per `(attr_id, block_id)` in `cf_fts_doc_lens` (`DocLenBlock`), while each `(attr_id, token, block_id)` maps to a v2 `PackedPostingBlock` storing only `tf` (switching automatically between sparse `(slots, tfs)` and dense `(bitmap, [tf; 256])` at `POSTING_DENSE_THRESHOLD = 96`) plus exact `(max_tf, min_dl_over_tf)` block-max bounds. Because `avgdl` is only supplied at query time, block upper bounds remain mathematically sound across arbitrary inserts and deletes without background re-indexing.

## 3. Query Execution & Hybrid Search

- **Vectorized Block-Max MAXSCORE:** Terms are sorted by ascending global upper bound and dynamically partitioned into non-essential and essential sets as the top-$k$ threshold $\theta$ rises. Blocks whose header upper bound cannot beat $\theta$ are skipped without decoding postings; surviving blocks accumulate into a stack-allocated `[f32; 256]` buffer.
- **Filters:**
  - `ContainsAnyToken`, `ContainsAllTokens`, and `ContainsTokenSequence` execute directly on `cf_fts_postings` (fetching `cf_fts_positions` only for surviving candidate blocks in multi-term phrase queries).
  - Vector `Collection::search` resolves FTS subtrees to `(leaf_id, local_id)` masks inside `PreparedFilter`, supporting arbitrary `And`/`Or`/`Not` combinations with attribute filters.
  - Filtered BM25 pushes down pure-FTS and selective attribute filters into `RoaringBitmap<FtsDocId>` block masks and falls back to sound lazy evaluation for broad or negated filters.
- **Hybrid Search:** `Collection::hybrid_search` executes vector search and `query_bm25` under the same `Option<&Filter>` and fuses rankings via Reciprocal Rank Fusion (RRF).
