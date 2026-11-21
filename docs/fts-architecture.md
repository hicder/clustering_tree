# Full-Text Search & BM25 Architecture (`FtsIndex`)

`izzydb` implements native BM25 full-text search, token/phrase filtering, multi-field `RankBy` combinators, and Reciprocal Rank Fusion (RRF) hybrid search decoupled from vector clustering.

## 1. Crate Structure

- **`izzydb-core`:** `FullTextSearchConfig`, `Tokenizer`, `Language`, `TokenQuery`, `Filter::Contains*`, `RankBy`, `Bm25Overrides`, and `FtsDocId`/`FtsBlockId` helpers.
- **`izzydb-fts`:**
  - `analyzer.rs`: `FtsAnalyzer` backed by `alyze` (`WordV4` UAX #29 DFA, `Whitespace`, `PreTokenizedArray`, optional `fts-cjk` feature for `SmartCn`/`Kuromoji`/`Nori`), Snowball stemming (18 languages), stopword removal (13 languages), ASCII folding, and `+2` position gaps between `StringArray` elements.
  - `block.rs`: `FtsMeta`, `TermStats`, 256-document SoA `PackedPostingBlock` / `PostingBlockView` (with inline `tf`, `dl`, and exact `(max_tf, min_dl)` block-max headers), and delta-varint `PackedPositionBlock` / `PositionBlockView`.
  - `bm25.rs`: BM25 IDF/TF/QTF scoring and L1-resident `[f32; 256]` block scatter-accumulation.
  - `maxscore.rs`: Vectorized Block-Max MAXSCORE top-$k$ execution supporting `Sum`, `Max`, `Product`, `last_as_prefix` disjunctions, pushdown `RoaringBitmap<FtsDocId>` masks, and lazy filter predicates.
- **`izzydb-storage`:** 5 FTS RocksDB column families (`cf_fts_doc_map`, `cf_fts_rev_doc_map`, `cf_fts_terms`, `cf_fts_postings`, `cf_fts_positions`), prefix-bounded `RocksPostingCursor`, and atomic `TxContext` overlays.
- **`izzydb`:** `FtsIndex`, `Collection::query_bm25` / `Collection::query`, `Collection::hybrid_search`, FTS filter integration in `Collection::search`, and exhaustive `verify_collection_invariants` parity checks.

## 2. Cluster-Decoupled Posting Blocks

Rather than keying FTS postings by vector `(cluster_id, local_id)` (which causes heavy write amplification during vector tree splits, merges, and LIRE boundary reassignments), `FtsIndex` assigns each document a dense monotonic `FtsDocId: u32`:

- `block_id = fts_doc_id >> 8`
- `slot = (fts_doc_id & 0xFF) as u8`

Each `(attr_id, token, block_id)` maps to a `PackedPostingBlock` (<= 256 postings, typically <= 1.3 KB) storing inline `tf`, `dl`, and exact `(max_tf, min_dl)` bounds. Because `avgdl` is only supplied at query time, block upper bounds remain mathematically sound across arbitrary inserts and deletes without background re-indexing.

## 3. Query Execution & Hybrid Search

- **Vectorized Block-Max MAXSCORE:** Terms are sorted by ascending global upper bound and dynamically partitioned into non-essential and essential sets as the top-$k$ threshold $\theta$ rises. Blocks whose header upper bound cannot beat $\theta$ are skipped without decoding postings; surviving blocks accumulate into a stack-allocated `[f32; 256]` buffer.
- **Filters:**
  - `ContainsAnyToken`, `ContainsAllTokens`, and `ContainsTokenSequence` execute directly on `cf_fts_postings` (fetching `cf_fts_positions` only for surviving candidate blocks in multi-term phrase queries).
  - Vector `Collection::search` resolves FTS subtrees to `(leaf_id, local_id)` masks inside `PreparedFilter`, supporting arbitrary `And`/`Or`/`Not` combinations with attribute filters.
  - Filtered BM25 pushes down pure-FTS and selective attribute filters into `RoaringBitmap<FtsDocId>` block masks and falls back to sound lazy evaluation for broad or negated filters.
- **Hybrid Search:** `Collection::hybrid_search` executes vector search and `query_bm25` under the same `Option<&Filter>` and fuses rankings via Reciprocal Rank Fusion (RRF).
