# Full-Text Search & BM25 Architecture (`FtsIndex`)

`izzydb` implements native BM25 full-text search, token/phrase filtering, generalized `RankBy` combinators, and Reciprocal Rank Fusion (RRF) hybrid search keyed natively on **segment coordinates (`RowId` / `RowBlock`)**.

## 1. Crate Structure

- **`izzydb-core`:** `FullTextSearchConfig`, `Tokenizer`, `Language`, `TokenQuery`, `Filter::Contains*`, `RankBy`, `Bm25Overrides`, `CollectionConfig::fts_parallel_segments`, `TermId`, and `RowId` / `RowBlock` / `SkipChunk` (`chunk = block >> 8`).
- **`izzydb-fts`:**
  - `analyzer.rs`: `FtsAnalyzer` backed by `alyze` (`WordV4`, `Whitespace`, `PreTokenizedArray`, optional `fts-cjk` for `SmartCn`/`Kuromoji`/`Nori`), Snowball stemming (18 languages), stopwords (13 languages), ASCII folding, and `+2` position gaps between `StringArray` elements.
  - `block/`: `FtsMeta` v4 (`next_term_id`, per-attr live `(doc_count, total_tokens)`), `TermRecord` v4 (`term_id`, `TermStats`, inline standalone v3 posting block when single-`(seg, block)` and `doc_freq <= 8`), 256-slot `DocLenBlock` (`u8`/`u16`/`u32`), unified v4 `FtsChunk` per `(term_id, seg, chunk)` (combining skip bounds with compact `TfEncoding` posting payloads), and delta-varint `PackedPositionBlock`.
  - `simd/`, `bm25.rs`, `maxscore/`: Runtime-dispatched AVX-512F / AVX2+FMA / NEON / scalar Block-Max MAXSCORE top-$k$ execution supporting `Sum`, `Max`, constant `Product`, per-document `RankModifier` (see [ranking-and-boosting.md](ranking-and-boosting.md)), `last_as_prefix`, `BlockMaskProvider` segment liveness & filter pushdown, lazy filter predicates, single-clause chunk skipping, and segment-sharded Rayon parallelism (`execute_rank_by_with_shards`).
- **`izzydb-storage`:** FTS column families (`cf_fts_terms`, `cf_fts_skips`, `cf_fts_positions`, plus deprecated empty `cf_fts_postings`), `default[0x05]` doc lengths, `RocksSkipCursor`, and cached `RocksDocLenCursor` — all installed with the `dropped_segments` compaction filter.
- **`izzydb`:** `FtsIndex` (`RowIndexHook`, `max_parallel_segments`), `RowResolver` (`cf_row_doc_ids`), `LivenessMaskProvider`, `Collection::query_bm25` / `query` / `hybrid_search` / `compact_fts_stats`.

---

## 2. Segment-Keyed Posting Chunks (`(seg, chunk)` & `(seg, block, slot)`)

Instead of maintaining a separate dense `FtsDocId` namespace (`cf_doc_loc`, `FtsRevBlock`, `fts_free_ids` removed in v3), FTS keys directly off **`RowId`**:

- `seg: SegmentId = row.seg()`
- `block: u32 = row.local() >> 8` (`RowBlock = (seg, block)`)
- `chunk: u32 = block >> 8` (`SkipChunk = (seg, chunk)`)
- `slot: u8 = (row.local() & 0xFF) as u8`

Keys in RocksDB:
- **`cf_fts_terms`:** `attr_id (u16 BE) || token_utf8` $\to$ `TermRecord` v4. Rare single-block terms (`doc_freq <= 8`) store `seg: u32 LE || block: u32 LE || v3 PackedPostingBlock` inline inside `TermRecord`. A second distinct `(seg, block)` promotes the term out-of-line into `cf_fts_skips`.
- **`cf_fts_skips`:** `term_id (u32 BE) || seg (u32 BE) || chunk (u32 BE)` (12 B) $\to$ unified v4 **`FtsChunk`** (up to 256 blocks' `(max_tf, min_dl_over_tf)` skip bounds + compact `TfEncoding` posting payloads; 4-byte prefix extractor on `term_id`).
- **`cf_fts_postings`:** Deprecated and kept empty in v4 (`verify_invariants` asserts zero visible keys; retained as an empty CF handle with the `dropped_segments` compaction filter).
- **`cf_fts_positions`:** `term_id (u32 BE) || seg (u32 BE) || block (u32 BE)` (12 B) $\to$ `PackedPositionBlock`.
- **`default[0x05]`:** `0x05 || attr_id (u16 BE) || seg (u32 BE) || block (u32 BE)` (11 B) $\to$ `DocLenBlock` (lazily cached per `(attr_id, seg, block)` in `Storage::fts_doc_lens_cache`).

### Deletes, Updates, and Compaction

- **Live mutations:** Because `RowId`s are append-only in `Open` segments and immutable once sealed, sealed FTS chunks/position blocks never undergo random deletes. Tombstoning a row decrements `TermRecord.doc_freq` and `FtsMeta` live `(doc_count, total_tokens)`; `LivenessMaskProvider` masks out deleted slots in RAM during MAXSCORE and filter evaluation at `[u64; 4]` bitmask speed.
- **Compaction (`Collection::compact`):** Copies live FTS terms/chunks/positions/doc-lens densely into new target segments and re-computes tight `TermStats` (`max_tf`, `min_dl_over_tf`); old segments' `cf_fts_skips` / `cf_fts_positions` / `default[0x05]` (and any legacy `cf_fts_postings`) keys are dropped automatically by the `dropped_segments` compaction filter.
- **`Collection::compact_fts_stats`:** Recomputes exact `TermStats` across all terms on demand.

---

## 3. Query Execution & Hybrid Search

- **Vectorized Block-Max MAXSCORE:** Compiles the expression once (`CompiledPlan`), sorts clauses by upper bound, and skips chunks/blocks using `FtsChunk` bounds / inline `TermRecord` headers (including whole-chunk skips when a single essential clause remains). Surviving blocks borrow compact posting payloads directly from the active chunk slice (`FtsSkipCursor::current_posting_encoded`) and the BM25 scorer decodes them at SIMD-kernel dispatch, avoiding a wide `PostingBlockView` copy on the hot path. Each 256-slot block is masked with `LivenessMaskProvider` (`snap.block_live_mask & filter_rowset_mask`) and accumulated into L1-resident `[f32; 256]` stack buffers. All-single-clause plans use the specialized `run_shard_single` loop with shard-local `(block, bound)` arrays. Top-$k$ `RowId` winners are resolved to `DocId`s on tie eviction (`ensure_theta_tie_doc_ids`) and on final output via `RowResolver::resolve_all` (`cf_row_doc_ids`). See [SIMD Block-Max MAXSCORE & Data Layout](blockmax-simd.md) for block formats and SIMD kernels.
- **Segment-Parallel MAXSCORE:** When `CollectionConfig::fts_parallel_segments` is unset (`None`, using Rayon thread count) or `> 1`, and there is no `lazy_filter` or `RankModifier` across $\ge 4$ visible segments (`StorageFtsSource::segment_ranges`), MAXSCORE pre-warms `DocLenBlock` caches and runs disjoint `(seg_start, seg_end)` shards in parallel via Rayon with a shared `AtomicU32` `shared_theta`, merging shard heaps with `DocId`-resolved tie eviction (`merge_shard_heaps`). Set `with_fts_parallel_segments(1)` to force single-threaded execution.
- **Token Filters:** `ContainsAllTokens`, `ContainsAnyToken`, and `ContainsTokenSequence` produce exact live `RowSet`s directly usable by the v3 filter planner (`FilterPlan`), vector search, `OrderBy`, and faceting.
- **Hybrid Search (`Collection::hybrid_search`):** Executes vector search and `query` / `query_bm25` under the same `Option<&Filter>` and fuses rankings with Reciprocal Rank Fusion (RRF).
