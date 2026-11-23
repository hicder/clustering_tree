# Full-Text Search & BM25 Architecture (`FtsIndex`)

`izzydb` implements native BM25 full-text search, token/phrase filtering, generalized `RankBy` combinators, and Reciprocal Rank Fusion (RRF) hybrid search keyed natively on **segment coordinates (`RowId` / `RowBlock`)**.

## 1. Crate Structure

- **`izzydb-core`:** `FullTextSearchConfig`, `Tokenizer`, `Language`, `TokenQuery`, `Filter::Contains*`, `RankBy`, `Bm25Overrides`, `TermId`, and `RowId` / `RowBlock` / `SkipChunk` (`chunk = block >> 8`).
- **`izzydb-fts`:**
  - `analyzer.rs`: `FtsAnalyzer` backed by `alyze` (`WordV4`, `Whitespace`, `PreTokenizedArray`, optional `fts-cjk` for `SmartCn`/`Kuromoji`/`Nori`), Snowball stemming (18 languages), stopwords (13 languages), ASCII folding, and `+2` position gaps between `StringArray` elements.
  - `block/`: `FtsMeta` v4 (`next_term_id`, per-attr live `(doc_count, total_tokens)`), `TermRecord` v4 (`term_id`, `TermStats`, inline posting block when single-`(seg, block)` and `doc_freq <= 8`), 256-slot `DocLenBlock` (`u8`/`u16`/`u32`), 256-block `SkipBlock` chunk per `(term_id, seg, chunk)`, v3 TF-only Sparse/Dense `PackedPostingBlock`, and delta-varint `PackedPositionBlock`.
  - `simd.rs`, `bm25.rs`, `maxscore/`: Runtime-dispatched AVX-512F / AVX2+FMA / NEON / scalar Block-Max MAXSCORE top-$k$ execution supporting `Sum`, `Max`, constant `Product`, per-document `RankModifier` (see [ranking-and-boosting.md](ranking-and-boosting.md)), `last_as_prefix`, `BlockMaskProvider` segment liveness & filter pushdown, and lazy filter predicates.
- **`izzydb-storage`:** Dedicated FTS column families (`cf_fts_terms`, `cf_fts_skips`, `cf_fts_postings`, `cf_fts_positions`), `default[0x05]` doc lengths, `RocksSkipCursor`, and cached `RocksDocLenCursor` — all installed with the `dropped_segments` compaction filter.
- **`izzydb`:** `FtsIndex` (`RowIndexHook`), `RowResolver` (`cf_row_doc_ids`), `LivenessMaskProvider`, `Collection::query_bm25` / `query` / `hybrid_search` / `compact_fts_stats`.

---

## 2. Segment-Keyed Posting Blocks (`(seg, block, slot)`)

Instead of maintaining a separate dense `FtsDocId` namespace (`cf_doc_loc`, `FtsRevBlock`, `fts_free_ids` removed in v3), FTS keys directly off **`RowId`**:

- `seg: SegmentId = row.seg()`
- `block: u32 = row.local() >> 8` (`RowBlock = (seg, block)`)
- `chunk: u32 = block >> 8` (`SkipChunk = (seg, chunk)`)
- `slot: u8 = (row.local() & 0xFF) as u8`

Keys in RocksDB:
- **`cf_fts_terms`:** `attr_id (u16 BE) || token_utf8` $\to$ `TermRecord` v4. Rare single-block terms (`doc_freq <= 8`) store `seg: u32 LE || block: u32 LE || v3 PackedPostingBlock` inline inside `TermRecord`. A second distinct `(seg, block)` promotes the term out-of-line.
- **`cf_fts_skips`:** `term_id (u32 BE) || seg (u32 BE) || chunk (u32 BE)` (12 B) $\to$ `SkipBlock` (4-byte prefix Bloom on `term_id`).
- **`cf_fts_postings` & `cf_fts_positions`:** `term_id (u32 BE) || seg (u32 BE) || block (u32 BE)` (12 B) $\to$ `PackedPostingBlock` / `PackedPositionBlock`.
- **`default[0x05]`:** `0x05 || attr_id (u16 BE) || seg (u32 BE) || block (u32 BE)` (11 B) $\to$ `DocLenBlock` (lazily cached per `(attr_id, seg, block)` in `Storage::fts_doc_lens_cache`).

### Deletes, Updates, and Compaction

- **Live mutations:** Because `RowId`s are append-only in `Open` segments and immutable once sealed, sealed FTS posting/position blocks never undergo random deletes. Tombstoning a row decrements `TermRecord.doc_freq` and `FtsMeta` live `(doc_count, total_tokens)`; `LivenessMaskProvider` masks out deleted slots in RAM during MAXSCORE and filter evaluation at `[u64; 4]` bitmask speed.
- **Compaction (`Collection::compact`):** Copies live FTS terms/postings/positions/doc-lens densely into new target segments and re-computes tight `TermStats` (`max_tf`, `min_dl_over_tf`); old segments' `cf_fts_skips` / `cf_fts_postings` / `cf_fts_positions` / `default[0x05]` keys are dropped automatically by the `dropped_segments` compaction filter.
- **`Collection::compact_fts_stats`:** Recomputes exact `TermStats` across all terms on demand.

---

## 3. Query Execution & Hybrid Search

- **Vectorized Block-Max MAXSCORE:** Sorts terms by upper bound and skips chunks/blocks using `SkipBlock` / inline `TermRecord` headers, masking each 256-slot block with `LivenessMaskProvider` (`snap.block_live_mask & filter_rowset_mask`) before accumulating BM25 scores into L1-resident `[f32; 256]` stack buffers. Top-$k$ `RowId` winners are resolved to `DocId`s once via `RowResolver::resolve_all` (`cf_row_doc_ids`). See [SIMD Block-Max MAXSCORE & Data Layout](blockmax-simd.md) for block formats and SIMD kernels.
- **Token Filters:** `ContainsAllTokens`, `ContainsAnyToken`, and `ContainsTokenSequence` produce exact live `RowSet`s directly usable by the v3 filter planner (`FilterPlan`), vector search, `OrderBy`, and faceting.
- **Hybrid Search (`Collection::hybrid_search`):** Executes vector search and `query` / `query_bm25` under the same `Option<&Filter>` and fuses rankings with Reciprocal Rank Fusion (RRF).
