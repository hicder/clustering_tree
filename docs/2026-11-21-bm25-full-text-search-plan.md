# Plan: BM25 Full-Text Search (`FtsIndex` + `alyze` + Block-Max MAXSCORE)

## Goal

Add native **BM25 Full-Text Search (FTS)**, **token/phrase filtering**, **multi-field ranking combinators**, and **hybrid vector + BM25 search** to `izzydb`, modeled after **turbopuffer FTS v2**:

1. **Per-attribute FTS schema configuration** on `AttributeType::String` and `AttributeType::StringArray` (`FullTextSearchConfig`), supporting configurable tokenizers (`word_v4`, `whitespace`, `pre_tokenized_array`, and optional CJK), language-aware stemming & stopword removal, case sensitivity, ASCII folding, `max_token_length`, and BM25 tuning (`k1`, `b`, `k3`).
2. **Production-grade natural text analysis pipeline** powered by turbopuffer's open-source **`alyze`** crate (plus documented modular crate alternatives for stemming, stopwords, Unicode segmentation, ASCII folding, and CJK).
3. **KV-store-native, cluster-decoupled inverted index (`FtsIndex`)** stored in dedicated RocksDB column families using **fixed 256-document posting blocks** with inline term frequencies, document lengths, and exact block-max upper-bound metadata `(max_tf, min_dl)`—eliminating write amplification on vector tree splits/merges/LIRE while keeping single-doc upserts/deletes O(1) per unique token.
4. **Vectorized Block-Max MAXSCORE** top-$k$ query execution (excelling on both short human queries and long LLM-generated queries), supporting `last_as_prefix` prefix matching, `ContainsAllTokens` / `ContainsAnyToken` / `ContainsTokenSequence` filters, composite `RankBy` expressions (`Sum`, `Max`, `Product`), filtered BM25, and Reciprocal Rank Fusion (RRF) hybrid search.

### Compatibility
- Backward-incompatible on-disk schema bump (`CollectionSchema` version `1 -> 2` and new RocksDB column families). Existing databases are not migrated.
- Existing `Collection::upsert`, `Collection::delete`, `Collection::get`, and `Collection::search` APIs continue to work seamlessly; attributes without `full_text_search` enabled incur zero FTS storage or indexing overhead.

---

## 1. Natural Language Processing (NLP) Crates

### 1.1 Recommended Primary Crate: `alyze = "0.2.0"`

Turbopuffer open-sourced its own Rust text analysis crate, **`alyze`** (`github.com/turbopuffer/alyze`, dual-licensed `MIT AND Apache-2.0`, Rust Edition 2024, MSRV 1.88 — matching `izzydb`'s Edition 2024 workspace). It directly powers turbopuffer's `word_v4` pipeline and bundles all core NLP primitives in one fast, low-allocation crate:

| Capability | Provided by `alyze = "0.2.0"` | Details |
|---|---|---|
| **Word segmentation (`word_v4`)** | `alyze::uax29::word` | Hand-rolled DFA implementing Unicode 17.0 UAX #29 word boundaries (~300–500 MiB/s) |
| **Whitespace tokenizer** | `alyze::analyze::TokenizerOptions::Whitespace` | Splits strictly on Unicode `White_Space` (via `tpuf_icu_properties_211`) |
| **Case folding / lowercasing** | `AnalysisOptions::case_sensitive` | Unicode 17.0 lowercasing (`u17_to_lower`) |
| **ASCII folding** | `AnalysisOptions::ascii_folding` | Lucene-compatible diacritic/accent folding (`café` → `cafe`, `naïve` → `naive`) |
| **Stemming (18 languages)** | `AnalysisOptions::stemming` | Wraps `rust-stemmers = "1.2.0"` (Snowball) with an internal `StemmingCache` (32k entries) |
| **Stopword removal (13 languages)** | `AnalysisOptions::stopword_removal` | Compile-time `phf` perfect-hash stopword tables |
| **Multi-string (`[]string`) analysis** | `Analyzer::analyze_inputs` | Streams tokens with positions across string slices; reuses `ReusableBuffer` |
| **CJK tokenizers (optional)** | Features `smartcn`, `kuromoji`, `nori` (`cjk`) | Lucene-derived Chinese/Japanese/Korean morphological tokenizers (keep behind an opt-in `fts-cjk` feature because `alyze`'s `build.rs` fetches dictionaries over HTTP when enabled) |

**Supported languages in `alyze` (matching turbopuffer):**
- **Stemming (18):** Arabic, Danish, Dutch, English *(default)*, Finnish, French, German, Greek, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Tamil, Turkish.
- **Stopwords (13):** Danish, Dutch, English, Finnish, French, German, Hungarian, Italian, Norwegian, Portuguese, Russian, Spanish, Swedish. *(Not supported for Arabic, Greek, Romanian, Tamil, Turkish—mirroring turbopuffer.)*

### 1.2 Modular Alternative / Supplemental Crates

If `izzydb` later needs custom analysis stages outside `alyze` (e.g., extra stopword languages, NFKC normalization, transliteration, or offline-embedded CJK dictionaries without build-time downloads), the Rust NLP ecosystem provides clean drop-in crates:

| Stage | Crate | Latest Version | Notes |
|---|---|---|---|
| **UAX #29 segmentation** | `unicode-segmentation` | `1.13.3` | Standard `unicode_words()` iterator (slower than `alyze`'s DFA, zero-config) |
| **Stemming (Snowball)** | `rust-stemmers` | `1.2.0` | Same 18 Snowball stemmers used inside `alyze` |
| **Stopwords (50+ languages)** | `stop-words` | `0.10.1` | ISO & NLTK stopword lists across 50+ languages (covers Arabic/Greek/Romanian/Turkish if needed) |
| **Unicode normalization** | `unicode-normalization` | `0.1.26` | NFC / NFD / NFKC / NFKD decomposition & recomposition |
| **ASCII transliteration** | `deunicode` | `1.6.2` | Transliterates arbitrary Unicode (including Cyrillic/Greek/CJK/emoji) to ASCII |
| **Offline CJK (embedded dicts)** | `lindera` | `6.2.0` | Japanese (`ipadic`/`unidic`), Korean (`ko-dic`), Chinese (`jieba`/`cc-cedict`) with `embed-*` features |
| **All-in-one multilingual** | `charabia` | `0.9.x` | Meilisearch's auto-detecting multilingual tokenizer (heavier dependency tree) |

**Recommendation:** Use **`alyze = "0.2.0"`** as the default analysis engine in a new **`izzydb-fts`** crate (or module inside `izzydb-core` / `izzydb`), with an optional Cargo feature `fts-cjk = ["alyze/cjk"]` so default builds remain 100% offline and fast to compile.

---

## 2. Crate & Module Layout

Add a dedicated crate `crates/izzydb-fts` (or keep schema/AST types in `izzydb-core` and analysis/scoring in `izzydb-fts`) to keep compile times modular:

- **Workspace `Cargo.toml`:**
  - Add `alyze = "0.2"` to `[workspace.dependencies]`.
  - Add `izzydb-fts = { path = "crates/izzydb-fts" }` to workspace members & dependencies.
- **`crates/izzydb-core/src/`:**
  - `config.rs`: add `FtsDocId = u32`, `FtsBlockId = u32` (`block_id = fts_doc_id >> 8`, `slot = (fts_doc_id & 0xFF) as u8`).
  - `schema.rs`: extend `AttributeConfig` with `pub full_text_search: Option<FullTextSearchConfig>`, add `FullTextSearchConfig`, `Tokenizer`, `Language`, and schema version `2` serialization/reconciliation.
  - `filter.rs`: add FTS filter predicates to `Filter`:
    - `ContainsAllTokens { attr: String, query: TokenQuery, last_as_prefix: bool }`
    - `ContainsAnyToken { attr: String, query: TokenQuery, last_as_prefix: bool }`
    - `ContainsTokenSequence { attr: String, query: TokenQuery, last_as_prefix: bool }`
    *(where `TokenQuery` is `Text(String)` or `PreTokenized(Vec<String>)`).*
  - `rank.rs` *(new)*: `RankBy` expression AST (`Bm25`, `Sum`, `Max`, `Product`) and `Bm25Params` (`k1`, `b`, `k3`, `last_as_prefix`).
- **`crates/izzydb-fts/src/`** *(new crate)*:
  - `analyzer.rs`: wraps `alyze::analyze::Analyzer` + `ReusableBuffer`, implements `PreTokenizedArray`, emits `(token: String, tf: u32, positions: Vec<u32>)` and `doc_len: u32` with array-element position gaps.
  - `block.rs`: `PackedPostingBlock` & `PostingBlockView` (256-slot SoA binary format, header with `(count, max_tf, min_dl)`), plus `PackedPositionBlock` (delta-varint positions for phrase matching).
  - `bm25.rs`: BM25 IDF/TF scoring math, exact block-max upper bound calculation, and vectorized 256-slot block accumulator.
  - `maxscore.rs`: Block-Max MAXSCORE multi-term top-$k$ execution engine (supporting single-field BM25, `Sum`/`Max`/`Product` combinators, `last_as_prefix` union terms, and optional `FtsDocId` filter bitmask).
- **`crates/izzydb-storage/src/`:**
  - `storage.rs`: add FTS column families (`cf_fts_terms`, `cf_fts_postings`, `cf_fts_positions`, `cf_fts_doc_map`, `cf_fts_rev_doc_map`), `FtsMeta` in `cf_meta`, and `TxContext` dirty overlays for atomic FTS mutations alongside `ClusteringTree` and `AttributeIndex`.
- **`crates/izzydb/src/`:**
  - `fts_index.rs` *(new)*: `FtsIndex` coordinating writes (`upsert_doc`, `remove_doc`), prefix term expansion, `Contains*` filter evaluation, and `RankBy` BM25 search.
  - `collection.rs`: wire `FtsIndex` into `Collection::open`, `upsert`, `delete`, add `query_bm25` / `query` (`RankBy` + `Option<&Filter>` + `top_k`), and `hybrid_search` (vector + BM25 via RRF).
  - `verify.rs`: extend `verify_collection_invariants` to exhaustively verify `FtsIndex` parity against `cf_docs`.

---

## 3. Schema & Tokenizer Configuration

Modeled directly on turbopuffer's `full_text_search` schema object:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct FullTextSearchConfig {
    pub tokenizer: Tokenizer,        // default: Tokenizer::WordV4
    pub language: Language,          // default: Language::English
    pub case_sensitive: bool,        // default: false
    pub stemming: bool,              // default: false
    pub remove_stopwords: bool,      // default: false
    pub ascii_folding: bool,         // default: false
    pub max_token_length: usize,     // default: 39 (valid: 1..=254)
    pub k1: f32,                     // default: 1.2 (valid: > 0.0)
    pub b: f32,                      // default: 0.75 (valid: 0.0..=1.0)
    pub k3: f32,                     // default: 8.0 (valid: >= 0.0)
    pub index_positions: bool,       // default: true (required for ContainsTokenSequence)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tokenizer {
    WordV4,             // alyze UAX #29 DFA
    Whitespace,         // alyze Unicode whitespace split
    PreTokenizedArray,  // []string elements used verbatim as tokens
    #[cfg(feature = "fts-cjk")]
    SmartCn,            // Chinese
    #[cfg(feature = "fts-cjk")]
    Kuromoji,           // Japanese
    #[cfg(feature = "fts-cjk")]
    Nori,               // Korean
}
```

### Schema Validation Rules (matching `alyze` & turbopuffer)
1. `full_text_search` is only valid on `AttributeType::String` and `AttributeType::StringArray`.
2. `Tokenizer::PreTokenizedArray` is only valid on `AttributeType::StringArray`, and rejects `stemming`, `remove_stopwords`, `case_sensitive: true`, `ascii_folding`.
3. `Tokenizer::Whitespace` rejects `stemming: true` and `remove_stopwords: true`.
4. `stemming: true` or `remove_stopwords: true` requires `case_sensitive == false`.
5. `remove_stopwords: true` rejects languages without stopword lists (`Arabic`, `Greek`, `Romanian`, `Tamil`, `Turkish`) with `TreeError::InvalidSchema`.
6. CJK tokenizers (`SmartCn`, `Kuromoji`, `Nori`) apply their own language-specific normalization and reject custom `stemming`/`remove_stopwords`/`case_sensitive`/`ascii_folding` overrides.
7. `max_token_length` must be in `1..=254`, `k1 > 0.0 && k1.is_finite()`, `0.0 <= b <= 1.0`, `k3 >= 0.0 && k3.is_finite()`.

### `StringArray` Position Semantics
For `AttributeType::StringArray` (e.g., `["quick brown", "fox jumps"]`), `Analyzer::analyze_inputs` indexes all elements into the same field's token stream (contributing to one combined `doc_len`), but inserts a **position gap of `+2`** between consecutive array elements (e.g., `"quick"@0, "brown"@1, "fox"@3, "jumps"@4`). That way:
- `ContainsAllTokens("brown fox")` matches across elements.
- `ContainsTokenSequence("brown fox")` does **not** falsely match across element boundaries (matching turbopuffer's `[]string` phrase semantics).

---

## 4. Storage Layout & On-Disk Binary Format

### 4.1 Why Decouple `FtsIndex` from `ClusteringTree`?

`AttributeIndex` couples postings to vector clusters `(cluster_id, local_id)` because filtered vector search probes clusters one by one. In contrast, **BM25 search does not traverse vector clusters**—it traverses posting lists per query term. Turbopuffer FTS v2 specifically decoupled its inverted index from vector clusters (achieving a ~10x size reduction and up to 20x speedup) because:
- Vector splits, merges, and LIRE reassignments move documents across clusters constantly; decoupling avoids rewriting FTS postings during vector maintenance.
- Fixed **256-document posting blocks** are uniform, contiguous, and SIMD-friendly, enabling Block-Max MAXSCORE to skip entire 256-doc blocks at once.

### 4.2 Dense Monotonic `FtsDocId: u32`

User-supplied `DocId` is `u64` (potentially sparse/UUID-derived). On a document's first insert into a `Collection` (when any FTS attribute exists in the schema), `Collection` allocates a dense monotonic `FtsDocId: u32` (stored in `cf_meta["fts_meta"]`):
- `block_id: u32 = fts_doc_id >> 8` (256 docs per block)
- `slot: u8 = (fts_doc_id & 0xFF) as u8` (1-byte intra-block doc ID)

On `upsert` of an existing `DocId`, its existing `FtsDocId` is reused (updating only the diff between old and new tokens in that document's single `block_id`). On `delete`, its `FtsDocId` is removed from its blocks and mappings.

### 4.3 RocksDB Column Families

| Column Family | Constant | Key Format | Value Format | Tuning |
|---|---|---|---|---|
| `cf_meta` | `CF_META` | `b"fts_meta"` | `FtsMeta { next_fts_doc_id: u32, per_attr_stats: Vec<(AttrId, doc_count: u64, total_tokens: u64)> }` | Point `get_cf` |
| `cf_fts_doc_map` | `CF_FTS_DOC_MAP` | `doc_id` (`u64` BE, 8 B) | `fts_doc_id` (`u32` BE, 4 B) | Point-lookup CF (10 bpk Bloom) |
| `cf_fts_rev_doc_map` | `CF_FTS_REV_DOC_MAP` | `fts_doc_id` (`u32` BE, 4 B) | `doc_id` (`u64` BE, 8 B) | Point-lookup CF (10 bpk Bloom) |
| `cf_fts_terms` | `CF_FTS_TERMS` | `attr_id` (`u16` BE) \|\| `token_utf8` | `TermStats` (16 B: `doc_freq: u64 LE`, `max_tf: u32 LE`, `min_dl: u32 LE`) | 2-byte prefix extractor (`attr_id`), full-key Bloom + prefix scan for `last_as_prefix` |
| `cf_fts_postings` | `CF_FTS_POSTINGS` | `attr_id` (`u16` BE) \|\| `escaped_token` \|\| `block_id` (`u32` BE) | `PackedPostingBlock` | Point-lookup + prefix scan on `(attr_id \|\| escaped_token)` |
| `cf_fts_positions` | `CF_FTS_POSITIONS` | `attr_id` (`u16` BE) \|\| `escaped_token` \|\| `block_id` (`u32` BE) | `PackedPositionBlock` | Point-lookup CF (read **only** for `ContainsTokenSequence`) |

*Key notes:*
- `escaped_token` uses the existing `encode_string_ordered` escaping (`0x00 -> 0x00 0xFF`, terminated with `0x00 0x00`) from `izzydb-storage/src/codec.rs`, guaranteeing unambiguous separation from the trailing `block_id: u32 BE` and exact lexicographical ordering by `(attr_id, token, block_id)`.
- Because `cf_fts_terms` keys are `attr_id (2 B) || raw token_utf8` (no trailing suffix), a prefix query `"wal"` is simply a RocksDB iterator seek to `attr_id || b"wal"` reading until `!key.starts_with(prefix)`.
- Keeping `cf_fts_positions` in its own CF ensures normal BM25 queries never pull position bytes into the RocksDB block cache.

### 4.4 `PackedPostingBlock` Binary Format (256-doc bucket)

Fixed 12-byte header + compact Structure-of-Arrays (SoA) body:

```text
Header (12 bytes):
  [count: u16 LE]        // 1..=256 (empty blocks are deleted from RocksDB)
  [flags: u16 LE]        // reserved (e.g., 0 = u16 tf/dl, 1 = u32 overflow)
  [max_tf: u32 LE]       // exact max(tf) among all entries in this block
  [min_dl: u32 LE]       // exact min(dl) among all entries in this block

Body (sorted strictly ascending by slot: u8):
  [slots: [u8; count]]          // 1 byte per posting
  [tfs:   [u16; count] LE]      // saturated at u16::MAX (or u32 if flags == 1)
  [dls:   [u16; count] LE]      // doc length in tokens (or u32 if flags == 1)
```

**Why this layout works so well:**
1. **Tiny footprint:** ~5 bytes per posting (≤ 1.3 KB per full 256-doc block).
2. **Zero random doc-length reads:** `dl` is stored inline with each posting, eliminating per-document lookups during BM25 scoring.
3. **Exact `avgdl`-independent block-max bounds:** Storing `(max_tf, min_dl)` instead of a pre-baked float score means the block's upper bound $\text{BM25}(\text{idf}, \text{max\_tf}, \text{min\_dl}, \text{avgdl})$ is **always mathematically valid** even as `avgdl` evolves with inserts and deletes—no background re-indexing needed!
4. **O(1) mutable updates:** Upserting or deleting a document modifies only one small `<= 1.3 KB` block per unique token in that document.

### 4.5 `PackedPositionBlock` Binary Format (`cf_fts_positions`)

Stored only when `index_positions: true` and read only for `ContainsTokenSequence`:
```text
[count: u16 LE]
[slots: [u8; count]]                    // matches PackedPostingBlock.slots
[pos_offsets: [u32; count + 1] LE]      // byte slice offsets into delta_varints
[delta_varints: [u8; ...]]              // LEB128 varint-encoded position deltas per slot
```

---

## 5. Write Path & Atomic Transactions

All FTS updates execute inside the **same `TxContext` / `rocksdb::WriteBatch`** as `cf_docs`, `ClusteringTree`, and `AttributeIndex`:

### 5.1 `Collection::upsert(id, embedding, new_attrs)`
1. Validate `new_attrs` against `schema` (including `PreTokenizedArray` checks).
2. Look up `fts_doc_id = tx.get_fts_doc_id(id)?`:
   - If existing: read `old_doc_bytes` from `cf_docs`, re-analyze old FTS-enabled attributes (using a reusable `alyze::analyze::ReusableBuffer` stored on `FtsIndex`), and compute `(old_tokens, old_dl)` per FTS attribute.
   - If new: allocate `fts_doc_id = tx.alloc_fts_doc_id()`, stage `cf_fts_doc_map[id] = fts_doc_id` and `cf_fts_rev_doc_map[fts_doc_id] = id`.
3. Analyze `new_attrs` for each FTS-enabled attribute to get `(new_tokens, new_dl)` (map of `token -> (tf, positions)`).
4. Compute per-attribute diff between old and new tokens at `block_id = fts_doc_id >> 8`, `slot = (fts_doc_id & 0xFF) as u8`:
   - **Removed tokens:** remove `slot` from `PackedPostingBlock` (and `PackedPositionBlock`), decrement `TermStats.doc_freq` (deleting key if `0`).
   - **Added tokens:** insert `(slot, tf, new_dl)` into `PackedPostingBlock` (and positions), increment `TermStats.doc_freq`, update `TermStats.max_tf = max(max_tf, tf)` and `TermStats.min_dl = min(min_dl, new_dl)`.
   - **Changed tokens** (same token, different `tf`, `dl`, or `positions`): update slot in place; update `max_tf` / `min_dl`.
   - **Update `AttrFtsStats`:** adjust `doc_count` (when transitioning null ↔ non-null) and `total_tokens += new_dl - old_dl`.
5. Commit `tx` atomically.

*Note on `TermStats.max_tf` / `min_dl` tightening on delete:* Deleting the single document that held a term's global `max_tf` or `min_dl` leaves `TermStats` as a **conservative upper bound** (still 100% sound for MAXSCORE, never causing false pruning!). For `PackedPostingBlock`, we recompute exact `(block.max_tf, block.min_dl)` on every block mutation since scanning ≤ 256 `u16`s takes < 50 ns.

### 5.2 `Collection::delete(id)`
1. Look up `fts_doc_id` and `old_doc_bytes` inside `tx`.
2. Re-analyze old FTS attributes and remove `slot` from all affected posting/position blocks, decrement `TermStats.doc_freq`, decrement `AttrFtsStats`, and delete `cf_fts_doc_map[id]` and `cf_fts_rev_doc_map[fts_doc_id]`.
3. Commit `tx` atomically alongside `ClusteringTree` and `AttributeIndex` removal.

---

## 6. Query Execution: BM25, Vectorized Block-Max MAXSCORE, Filters & Hybrid Search

### 6.1 BM25 Scoring Formula

For corpus size $N = \text{attr\_stats.doc\_count}$, average document length $\text{avgdl} = \text{total\_tokens} / N$, term document frequency $df_t$, document term frequency $tf_{t,d}$, document length $|d|$, and query term frequency $qtf_t$:

$$\text{IDF}(t) = \ln\!\left(1 + \frac{N - df_t + 0.5}{df_t + 0.5}\right)$$

$$\text{score}(d, t) = \text{IDF}(t) \cdot \frac{tf_{t,d} \cdot (k_1 + 1)}{tf_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)} \cdot \frac{qtf_t \cdot (k_3 + 1)}{qtf_t + k_3}$$

- Documents with score `<= 0.0` are excluded (matching turbopuffer).
- For `last_as_prefix: true`, the final analyzed token $p$ is expanded via a RocksDB prefix scan on `cf_fts_terms` into matching terms $\{t_1, \dots, t_m\}$ (capped at a configurable `max_prefix_expansions`, e.g., 128, prioritized by `doc_freq` or lexicographical order). Those expanded terms form a **disjunctive sub-clause** whose combined contribution for document $d$ takes the sum (or max) of matching prefix terms, with a unified block-max upper bound.

### 6.2 Vectorized Block-Max MAXSCORE Execution

Following turbopuffer FTS v2's choice of **Vectorized MAXSCORE over WAND** (especially strong for long 10–100+ term LLM-generated queries):

1. **Term preparation & upper bounds:**
   - Analyze query string using the attribute's `FullTextSearchConfig` (producing deduplicated `(token, qtf)` pairs; handle `last_as_prefix` on the final token).
   - Batch-fetch `TermStats` from `cf_fts_terms` (`multi_get_cf`). Drop terms with `doc_freq == 0`.
   - Compute each term's global upper bound:
     $$\text{term\_max}_i = \text{BM25}(\text{IDF}_i, \text{term.max\_tf}, \text{term.min\_dl}, \text{avgdl}, qtf_i)$$
   - Sort terms by **ascending** $\text{term\_max}_i$ and compute prefix sums `prefix_max_sum[i] = sum(term_max[0..=i])`.
2. **Essential vs. non-essential partitioning:**
   - Maintain a min-heap of size $k$ with current threshold $\theta$ (`0.0` until $k$ docs are collected).
   - Find the largest index `non_essential_end` such that `prefix_max_sum[non_essential_end] <= theta`.
   - Terms `0..=non_essential_end` are **non-essential** (their combined scores alone can never beat $\theta$); remaining terms are **essential**.
3. **Block-at-a-time cursor advancement (256-doc blocks):**
   - Each term holds a RocksDB prefix iterator over `cf_fts_postings` for `(attr_id, escaped_token)`, yielding `(block_id, PackedPostingBlock)`.
   - Advance to the smallest `block_id` among all **essential** terms.
   - **Block-Max pruning:** Compute the sum of exact block upper bounds for all essential terms present at `block_id` plus `prefix_max_sum[non_essential_end]` (tightened by seeking non-essential terms if needed). If `block_upper_bound <= theta`, skip `block_id` completely!
   - **Vectorized L1-cache scoring (`[f32; 256]`):**
     - Zero a stack-allocated `scores: [f32; 256]` (1 KB) and `touched: [u64; 4]` bitset (32 B).
     - For each essential term present at `block_id`, scatter-add its BM25 contribution into `scores[slot]` and set `touched` bit.
     - If any non-essential terms exist and `max(scores) + prefix_max_sum[non_essential_end] > theta`, seek non-essential iterators to `block_id` and accumulate only for `touched` slots (or scatter-add into `scores`).
     - Scan `touched` bits (using `trailing_zeros()`), apply any optional `FtsDocId` filter mask, and push `(score, fts_doc_id)` into the top-$k$ min-heap whenever `score > theta`.
     - Whenever $\theta$ increases, recompute `non_essential_end`—shrinking the essential set dynamically!
4. **Final ID resolution:**
   - Batch-fetch `DocId`s for the final top-$k$ `FtsDocId`s from `cf_fts_rev_doc_map` and return `Vec<Bm25Result { id: DocId, score: f32 }>` sorted by descending score (tie-broken by ascending `DocId`).

### 6.3 Composite `RankBy` Expressions (`Sum`, `Max`, `Product`)

```rust
pub enum RankBy {
    Bm25 {
        attr: String,
        query: TokenQuery,
        params: Option<Bm25Overrides>, // optional query-time last_as_prefix, k1, b, k3
    },
    Sum(Vec<RankBy>),
    Max(Vec<RankBy>),
    Product(f32, Box<RankBy>),
}
```

- **`Sum` & `Product` (e.g., weighted multi-field BM25 like `2.0 * title + 1.0 * body`):** Flatten linearly into a single weighted term list `(weight * term_score)` and execute in **one unified MAXSCORE pass**!
- **`Max`:** Evaluate child clauses (or maintain per-child `[f32; 256]` accumulators per block) with `block_max = max(child_block_maxes)` for sound block skipping.

### 6.4 FTS Filters (`ContainsAllTokens`, `ContainsAnyToken`, `ContainsTokenSequence`)

Implemented in `FtsIndex` and integrated into `Filter`:
- **`ContainsAnyToken`:** Union of `FtsDocId`s across all query tokens (and prefix expansions if `last_as_prefix`).
- **`ContainsAllTokens`:** Intersect posting blocks sorted by ascending `doc_freq` (skipping non-common `block_id`s via RocksDB `seek`), intersecting `slot` lists per block.
- **`ContainsTokenSequence`:** Run `ContainsAllTokens` first to find candidate `(block_id, slot)` pairs, then fetch `cf_fts_positions` for those blocks only and verify contiguous positions $p_0, p_0 + 1, \dots, p_0 + (m - 1)$ (using merge-join over sorted position lists).

### 6.5 Bridging `Filter` with BM25 and Vector Search

1. **Filtered BM25 (`Collection::query_bm25(rank_by, filter, top_k)`):**
   - **FTS-only filters (`Contains*`):** Evaluate directly in `FtsDocId` space into a `RoaringBitmap<FtsDocId>` and pass directly into MAXSCORE as a block/slot mask (skipping any `block_id` with zero set bits!).
   - **Selective attribute filters (`ClusterCandidates::Subset(bm)` with small cluster count):** Resolve matching `(cluster_id, local_id)` → `DocId` → `FtsDocId` into a `RoaringBitmap<FtsDocId>` and push down into MAXSCORE.
   - **Broad attribute filters (`All` or large `Subset`):** Evaluate lazily on candidate `DocId`s that beat $\theta$ (fetching `cf_docs` and running `Filter::matches`), only updating $\theta$ when a document passes the filter.
2. **Vector search with FTS filters (`Collection::search(query, opts, Some(filter))`):**
   - Evaluate FTS filter predicates via `FtsIndex` → `RoaringBitmap<FtsDocId>` → resolve `DocId`s → `cf_doc_loc` `(leaf_id, local_id)` to build `ClusterCandidates::Subset` and per-leaf `local_id` masks, plugging seamlessly into `search_filtered`'s existing cluster pruning, exhaustive crossover, and masked RaBitQ scoring!
3. **Hybrid Search (`Collection::hybrid_search`):**
   - Run `Collection::search` (vector top-$k_v$) and `Collection::query_bm25` (BM25 top-$k_b$) under the same `Option<&Filter>`.
   - Fuse results via **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF}(d) = \frac{w_{\text{vec}}}{k_{\text{rrf}} + \text{rank}_{\text{vec}}(d)} + \frac{w_{\text{bm25}}}{k_{\text{rrf}} + \text{rank}_{\text{bm25}}(d)}$$
     (default $k_{\text{rrf}} = 60$, configurable weights $w_{\text{vec}}, w_{\text{bm25}}$).

---

## 7. Invariant Verification (`Collection::verify_invariants`)

Extend `verify_collection_invariants` ([`crates/izzydb/src/verify.rs`](file:///home/hieu/code/izzydb/crates/izzydb/src/verify.rs)) to verify:
1. **`FtsDocId` bijection:** `cf_fts_doc_map` and `cf_fts_rev_doc_map` form a 1-to-1 bijection with `cf_docs` (when the schema has ≥ 1 FTS attribute), with every `fts_doc_id < fts_meta.next_fts_doc_id`.
2. **Full FTS reconstruction check:** Re-analyze every document's FTS-enabled attributes from `cf_docs`, rebuild expected `AttrFtsStats`, `cf_fts_terms` (`doc_freq` exact; `max_tf` ≥ true max, `min_dl` ≤ true min), `cf_fts_postings` (exact slots, `tf`, `dl`, `block.max_tf`, `block.min_dl`, no empty blocks), and `cf_fts_positions`, and assert exact parity with disk.

---

## 8. Implementation Phases

1. **Phase 1 — Schema, `izzydb-fts` Analysis Pipeline & Unit Tests:**
   - Add `alyze = "0.2"` to workspace `Cargo.toml`; create `crates/izzydb-fts`.
   - Implement `FullTextSearchConfig`, `Tokenizer`, `Language`, `RankBy`, and `Filter::Contains*Tokens` in `izzydb-core` (bump schema version to `2`).
   - Implement `analyzer.rs` wrapping `alyze::analyze::Analyzer` + `PreTokenizedArray` + `StringArray` position gaps.
   - Unit tests for UAX #29 tokenization, lowercasing, ASCII folding, stopwords, Snowball stemming across multiple languages, `max_token_length`, and array boundary gaps.
2. **Phase 2 — Block Codecs, RocksDB CFs & Atomic Write Path (`izzydb-storage`, `izzydb`):**
   - Implement `PackedPostingBlock`, `PostingBlockView`, and `PackedPositionBlock` in `izzydb-fts/src/block.rs`.
   - Add FTS column families (`cf_fts_terms`, `cf_fts_postings`, `cf_fts_positions`, `cf_fts_doc_map`, `cf_fts_rev_doc_map`), `FtsMeta`, and `TxContext` overlays in `izzydb-storage/src/storage.rs`.
   - Implement `FtsIndex` write methods (`upsert_doc`, `remove_doc`) and wire atomically into `Collection::open`, `upsert`, and `delete`.
   - Extend `verify_collection_invariants` with full FTS parity checks.
3. **Phase 3 — BM25 Scoring, Vectorized Block-Max MAXSCORE & `RankBy`:**
   - Implement `bm25.rs` and `maxscore.rs` (essential/non-essential partitioning, block-max skipping, `[f32; 256]` L1 accumulator, `last_as_prefix` expansion, `Sum`/`Max`/`Product` combinators).
   - Expose `Collection::query_bm25` and verify exact score/ranking parity against a brute-force reference BM25 implementation across random corpora, upserts, deletes, and long (50-term) queries.
4. **Phase 4 — Text Filters, Filtered BM25 & Hybrid Search:**
   - Implement `ContainsAllTokens`, `ContainsAnyToken`, and `ContainsTokenSequence` (both standalone in `Filter` for vector search and combined with BM25).
   - Implement filtered BM25 pushdown + lazy fallback and `Collection::hybrid_search` (RRF).
   - Add end-to-end integration tests in `crates/izzydb/tests/fts_tests.rs` and a benchmark example comparing MAXSCORE vs. exhaustive BM25 on short and long LLM-style queries.
   - Document the new CFs and formats in `docs/key-encoding.md` and `docs/fts-architecture.md`.
