# SIMD Block-Max MAXSCORE & v2 Posting Layout (2026-11-21)

This note documents the storage layout, upper-bound tightening, runtime-dispatched SIMD kernels, and query-execution improvements introduced in commit [`66d8ec00`](../../crates/izzydb-fts/src/simd.rs) (`perf(fts)!: implement SIMD Block-Max MAXSCORE with v2 posting encoding`).

## Baseline Bottlenecks (v1 Posting Layout)

Prior to this change, `FtsIndex` already executed Block-Max MAXSCORE over 256-document blocks, but four bottlenecks limited throughput and bound tightness:

1. **Duplicated document lengths in every posting block:** v1 `PackedPostingBlock` stored `(slots, tfs, dls)` inline per `(attr_id, token, block_id)`. Every term's posting block repeated `dl` values (`u16` or `u32`), bloating posting blobs and forcing `k1 * (1 - b) + (k1 * b / avgdl) * dl` to be recomputed per posting in every term's inner loop.
2. **Loose `(max_tf, min_dl)` upper bounds:** `TermStats` and `PostingBlockHeader` tracked `max_tf` and `min_dl` independently. Pairing one document's `max_tf` with a different document's `min_dl` produced unnecessarily loose block and term upper bounds, reducing block-skipping effectiveness.
3. **Scalar block decoding & accumulation:** `Bm25TermScorer::accumulate_block` and `accumulate_block_masked` decoded `u16`/`u32` values element-by-element in scalar loops, and `execute_rank_by` scanned `[f32; 256]` score buffers and evaluated non-essential clauses without progressive slot pruning.
4. **Per-seek key allocations:** `RocksPostingCursor::seek_block` allocated/copied seek keys rather than reusing a scratch buffer.

---

## Changes Implemented

### 1. Out-of-Line `DocLenBlock` (`cf_fts_doc_lens`) & Shared `dl_norm` Cache
([`crates/izzydb-fts/src/block.rs`](../../crates/izzydb-fts/src/block.rs), [`crates/izzydb-fts/src/maxscore.rs`](../../crates/izzydb-fts/src/maxscore.rs), [`crates/izzydb-storage/src/storage.rs`](../../crates/izzydb-storage/src/storage.rs), [`crates/izzydb-storage/src/codec.rs`](../../crates/izzydb-storage/src/codec.rs))

- **Dedicated column family (`CF_FTS_DOC_LENS` / `cf_fts_doc_lens`):** Document lengths are now stored once per `(attr_id, block_id)` using a 6-byte big-endian key (`attr_id: u16 BE || block_id: u32 BE`), configured with `point_lookup_cf_opts(&block_cache, true)` (uncompressed, Bloom-filtered, pinned L0 index/filter blocks).
- **`DocLenBlock` & `DocLenBlockView`:** Each block holds 256 `dl` slots with a 4-byte header (`width: u8`, `reserved: u8 = 0`, `count: u16 LE`) followed by 256 packed integers at `IntWidth` (`U8`, `U16`, or `U32` selected via `IntWidth::min_for_max(max_dl)`).
- **Shared `DlNormCache`:** `execute_rank_by` interns `(attr_id, c1, c2)` groups (where `c1 = k1 * (1 - b)` and `c2 = k1 * b / avgdl`). For each visited `block_id`, `prepare_dl_norm_256` computes `dl_norm[s] = max(c1 + c2 * dl[s], f32::MIN_POSITIVE)` once via SIMD and shares the `[f32; 256]` slice across all query terms in that group.

### 2. v2 TF-Only Sparse/Dense `PackedPostingBlock`
([`crates/izzydb-fts/src/block.rs`](../../crates/izzydb-fts/src/block.rs), [`crates/izzydb-core/src/schema.rs`](../../crates/izzydb-core/src/schema.rs))

- Removed `dls` from `PackedPostingBlock` so posting blocks store only term frequencies (`tf`) using `IntWidth` (`U8` / `U16` / `U32`), and switched automatically between two layouts at `POSTING_DENSE_THRESHOLD = 96`:
  - **Sparse (`count < 96`):** 12-byte header + `slots: [u8; count]` + packed `tfs: [u8|u16|u32; count]` LE.
  - **Dense (`count >= 96`):** 12-byte header + 256-bit presence `bitmap: [u64; 4]` (32 bytes) + 256-slot contiguous `tfs: [u8|u16|u32; 256]` LE, enabling linear SIMD vector loads across 256 slots without indirect `tf` gathers.
- **Strict v2 header validation:** `PostingBlockHeader` (12 bytes: `count: u16`, `flags: u16`, `max_tf: u32`, `min_dl_over_tf: f32`) encodes `POSTING_VERSION_V2 = 0x0200`, `FLAG_DENSE = 0x0001`, and `WIDTH_MASK = 0x0006`. `PostingBlockHeader::parse` and `PostingBlockView::parse` validate version bits, unknown flag bits, width tag, `1 <= count <= 256`, non-negative non-NaN `min_dl_over_tf`, exact payload length, and dense bitmap popcount (`pop == count`).
- Bumped `CollectionSchema` persistence format from version `2` to version `3`.

### 3. Tighter `min_dl_over_tf` Upper Bounds
([`crates/izzydb-fts/src/block.rs`](../../crates/izzydb-fts/src/block.rs), [`crates/izzydb-fts/src/bm25.rs`](../../crates/izzydb-fts/src/bm25.rs))

- Replaced `min_dl: u32` with `min_dl_over_tf: f32` in both `TermStats` (`cf_fts_terms`) and `PostingBlockHeader`.
- `dl_over_tf_lower_bound(dl, tf)` computes `(dl as f64) / (tf as f64)` and applies `f32::next_down()` whenever `(r as f64) > exact`, ensuring `min_dl_over_tf <= dl / tf` for every posting in the term or block.
- `Bm25TermScorer::upper_bound(max_tf, min_dl_over_tf)` evaluates:
  $$\text{ub} = \frac{\text{num\_scale}}{1 + \frac{k_1(1 - b)}{\text{max\_tf}} + \frac{k_1 b}{\text{avgdl}} \cdot \text{min\_dl\_over\_tf}}$$
  in `f64` before casting to `f32`, giving a mathematically sound upper bound that is significantly tighter when high-`tf` documents are longer than the block's shortest document.

### 4. Runtime-Dispatched SIMD Kernels
([`crates/izzydb-fts/src/simd.rs`](../../crates/izzydb-fts/src/simd.rs), [`crates/izzydb-fts/src/bm25.rs`](../../crates/izzydb-fts/src/bm25.rs))

`SimdTier::detect()` selects `Avx512` (`avx512f`), `Avx2` (`avx2` + `fma`), `Neon` (`aarch64` `neon`), or `Scalar` at runtime for all 256-slot block operations:

| Kernel | Purpose | Vectorization Details |
|---|---|---|
| `prepare_dl_norm_256` | Precompute `max(c1 + c2 * dl, MIN_POSITIVE)` | Widens `U8`/`U16`/`U32` to `f32` and runs FMA + max across 16 / 8 / 4 lanes |
| `accumulate_dense_256` | Dense block BM25 accumulation | Loads contiguous 16 / 8 / 4 `tf` values, computes `(num_scale * tf) / (tf + dl_norm)`, skips zero 64-doc bitmap words |
| `accumulate_dense_masked_256` | Masked dense BM25 accumulation | Intersects `bitmap & mask`; uses scalar bit-iteration for `pop <= 24` and masked SIMD updates (`_mm512_mask_add_ps`, AVX2 blend, NEON `vbslq_f32`) otherwise |
| `accumulate_sparse` / `accumulate_sparse_masked` | Sparse block BM25 accumulation | Vectorizes BM25 arithmetic in 16 / 8 / 4-lane chunks (`slots.len() >= 16 / 8 / 4`) before scattering into `scores[slot]`; masked variant pre-filters active slots into `accumulate_u32_pairs` |
| `max_score_256` | Masked block max-reduction | Computes max over active slots (`_mm512_reduce_max_ps` on AVX-512) |
| `filter_ge_256` | Candidate threshold mask | Produces `[u64; 4]` bitmask of active slots with `scores[slot] >= threshold` |
| `merge_max_256` / `merge_add_256` | Multi-clause score merging | Vectorized elementwise `max` and `add` across 256 floats for `PrefixMax` (`last_as_prefix`) and `MaxBranches` (`RankBy::Max`) |

### 5. Progressive Non-Essential Pruning, Zero-Alloc Cursor Seeks & Verification
([`crates/izzydb-fts/src/maxscore.rs`](../../crates/izzydb-fts/src/maxscore.rs), [`crates/izzydb-storage/src/storage.rs`](../../crates/izzydb-storage/src/storage.rs), [`crates/izzydb/src/fts_index.rs`](../../crates/izzydb/src/fts_index.rs), [`crates/izzydb/src/verify.rs`](../../crates/izzydb/src/verify.rs))

- **Progressive `active_mask` pruning:** After essential clauses execute on `block_id`, `execute_rank_by` computes `max_ess = max_score_256(...)`. If `max_ess + non_ess_ub >= theta`, it seeks non-essential clauses to `block_id` to form a tighter per-block bound `tight_ub = max_ess + non_ess_block_ub`. Surviving blocks iterate non-essential clauses in **descending** global `term_max` order, narrowing `active_mask` before each clause via `filter_ge_256(tier, &scores, &active_mask, (theta - rem_ub) - tight_eps)` and breaking early once `active_mask == [0; 4]`.
- **Zero-allocation cursor seek:** `RocksPostingCursor::seek_block` reuses a pre-reserved `seek_buf` (`truncate(prefix_len)` + `extend_from_slice(&target.to_be_bytes())`).
- **Invariant verification:** `verify_collection_invariants` checks `cf_fts_doc_lens` parity and `min_dl_over_tf` bounds against `cf_docs`.

---

## Benchmark & Speedup

Measured via `cargo run --release --example fts_bench` (`n = 10,000` docs, `vocab = 1,000` with Zipf-like skew producing dense and sparse blocks, `top_k = 10` with exact top-10 ID agreement verified against exhaustive BM25 scoring) on an **AMD Ryzen 9 7950X (`SimdTier::Avx512`)**:

### Live Benchmark Run (`n = 10,000`, `tier = Avx512`)

| Query | `SIMD-MAXSCORE` | Exhaustive Baseline | Speedup vs. Exhaustive |
|---|---:|---:|---:|
| Short 3-term query | **90.6 µs** | 1,636.5 µs | **18.1×** |
| Medium 8-term query | **114.8 µs** | 3,586.4 µs | **31.2×** |
| Long 50-term LLM query | **516.3 µs** | 13,431.4 µs | **26.0×** |

### Commit `66d8ec00` Recorded Benchmark (`n = 10,000`, `AVX-512F`)

Commit [`66d8ec00`](../../crates/izzydb/examples/fts_bench.rs) also recorded a direct 3-term comparison against the pre-SIMD Block-Max MAXSCORE baseline:

| Query | Baseline MAXSCORE | `SIMD-MAXSCORE` | Exhaustive | Speedup vs. Baseline MAXSCORE | Speedup vs. Exhaustive |
|---|---:|---:|---:|---:|---:|
| Short 3-term | 93.4 µs | **86.6 µs** | 2,168.3 µs | **1.08×** (-7.3%) | **25.1×** |
| Medium 8-term | — | **109.3 µs** | 3,586.8 µs | — | **32.8×** |
| Long 50-term LLM | — | **555.1 µs** | 13,364.8 µs | — | **24.1×** |

*(Note: `Avx2` and `aarch64` `Neon` kernels are compiled and unit-tested against `Scalar` parity in [`crates/izzydb-fts/src/simd.rs`](../../crates/izzydb-fts/src/simd.rs), though hardware benchmarks above reflect `Avx512` on AMD Ryzen 9 7950X.)*
