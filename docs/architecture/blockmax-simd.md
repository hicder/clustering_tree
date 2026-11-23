# SIMD Block-Max MAXSCORE & Data Layout

This document describes the on-disk/in-memory data layout for full-text search (FTS) blocks and how the Block-Max MAXSCORE query engine executes vectorized BM25 scoring over them using runtime-dispatched SIMD kernels.

Implementation sources:
- `RowId` / `RowBlock` / `SkipChunk` bit-splitting (`ROW_BLOCK_SIZE = 256`, `FTS_SKIP_CHUNK_SIZE = 256`): [`crates/izzydb-core/src/config.rs`](../../crates/izzydb-core/src/config.rs)
- `FtsMeta` v4, `TermRecord` v4, `SkipBlock`, `DocLenBlock`, v3 `PackedPostingBlock`, `IntWidth`, and `dl_over_tf_lower_bound`: [`crates/izzydb-fts/src/block/mod.rs`](../../crates/izzydb-fts/src/block/mod.rs)
- BM25 scoring parameters, upper-bound formula, and SIMD dispatch wrappers: [`crates/izzydb-fts/src/bm25.rs`](../../crates/izzydb-fts/src/bm25.rs)
- Runtime-dispatched SIMD kernels (`Avx512`, `Avx2`, `Neon`, `Scalar`): [`crates/izzydb-fts/src/simd.rs`](../../crates/izzydb-fts/src/simd.rs)
- MAXSCORE execution loop, `DlNormCache`, `BlockMaskProvider`, `FtsSkipCursor`, and `Clause` evaluation: [`crates/izzydb-fts/src/maxscore/mod.rs`](../../crates/izzydb-fts/src/maxscore/mod.rs)
- RocksDB column families, `cf_row_doc_ids`, `RocksSkipCursor`, and cached `RocksDocLenCursor`: [`crates/izzydb-storage/src/storage/`](../../crates/izzydb-storage/src/storage/), [`crates/izzydb-codec/src/lib.rs`](../../crates/izzydb-codec/src/lib.rs)

---

## 1. Design Overview: Why 256-Document Blocks?

Rather than storing variable-length posting lists keyed by arbitrary document IDs, `FtsIndex` keys postings directly on v3 **`RowId = (SegmentId, LocalId)`** coordinates and partitions each segment into fixed-size blocks of **`ROW_BLOCK_SIZE = 256`** rows:

```text
RowId (u64):     [ seg: 32 bits | local: 32 bits ]
RowBlock:        (seg, block = local >> 8)   slot = (local & 0xFF) as u8
SkipChunk:       (seg, chunk = block >> 8)   rel  = (block & 0xFF) as u8
```

A fixed 256-slot block geometry aligns storage, caching, and SIMD execution around four hardware-friendly primitives:

1. **Compact 256-bit masks (`[u64; 4]`, 32 bytes):** Presence bitmaps, filter masks, touched-slot sets, and candidate hit masks all share the same four `u64` words. Bitwise set operations (`&`, `|`), popcount checks, and SIMD predicate comparisons map 1-to-1 onto 256 slots.
2. **L1-resident working buffers (`[f32; 256]`, 1 KiB each):** Both the per-block score accumulator (`scores`) and precomputed document-length normalization array (`dl_norm`) fit in 1 KiB on the stack—16 AVX-512 vectors, 32 AVX2 vectors, or 64 NEON vectors.
3. **Out-of-line document lengths in `default`:** Document lengths (`dl`) depend only on `(attr_id, block_id)`, not on individual query terms. Storing them once per block eliminates `dl` duplication from every term's posting block and lets all query terms on the same attribute share one SIMD-computed `dl_norm` array.
4. **Out-of-line skip summaries (`cf_fts_skips`) + inline rare terms:** Block-Max upper bounds `(max_tf, min_dl_over_tf)` for 256 posting blocks are packed into a single `SkipBlock` KV so MAXSCORE pruning reads compact skip chunks (`RocksSkipCursor`) and touches `cf_fts_postings` only on blocks that survive pruning. Single-block rare terms (`doc_freq <= 8`) inline their v3 posting block directly inside `cf_fts_terms` (`TermRecord`).

---

## 2. Data Layout

### 2.1 RocksDB Column Families & Keys

Block-Max MAXSCORE reads from `cf_fts_terms`, `cf_fts_skips`, FTS document lengths in `default`, and lazily from `cf_fts_postings` (plus `default[b"\x01fts_meta"]` for `doc_count`/`total_tokens`, `SegmentTableSnapshot` for liveness masking, and `cf_row_doc_ids` to resolve top-$k$ `RowId -> DocId`):

| Column Family | Key Format | Value Format | Access & Caching Pattern |
|---|---|---|---|
| `cf_fts_terms` | `attr_id` (`u16` BE) \|\| `token_utf8` | `TermRecord` v4 (`term_id`, `TermStats`, optional inline v3 block with `(seg, block)`) | Sorted `batched_multi_get_cf` (or 2-byte prefix scan for `last_as_prefix`) |
| `cf_fts_skips` | `term_id` BE \|\| `seg` BE \|\| `chunk` BE (12 B) | `SkipBlock` (up to 256 blocks' bounds) | Uncompressed CF, 4 B prefix Bloom; `RocksSkipCursor` |
| FTS document lengths in `default` | `0x05` \|\| `attr_id` BE \|\| `seg` BE \|\| `block` BE (11 B) | `DocLenBlock` (260 / 516 / 1,028 B) | Lazily loaded into `Storage::fts_doc_lens_cache` per `(attr_id, seg, block)` |
| `cf_fts_postings` | `term_id` BE \|\| `seg` BE \|\| `block` BE (12 B) | v3 `PackedPostingBlock` (TF-only) | Uncompressed CF, 4 B prefix Bloom; lazy sequential `posting_iter` inside `RocksSkipCursor` |
| `cf_row_doc_ids` | `seg` BE \|\| `block` BE (8 B) | `[u64; 256]` LE `DocId`s (2,048 B) | LRU cached in `Storage::doc_id_block_cache` (32 MiB) |

### 2.2 Adaptive Integer Width (`IntWidth`)

`DocLenBlock`, `SkipBlock` (`max_tf`), and `PackedPostingBlock` (`tf`) pack integer arrays using the smallest width (`IntWidth::min_for_max`) that fits the block's maximum value:

| `IntWidth` Tag | Rust Variant | Element Size | Max Value |
|---:|---|---:|---:|
| `0` | `IntWidth::U8` | 1 byte | `255` |
| `1` | `IntWidth::U16` | 2 bytes (LE) | `65,535` |
| `2` | `IntWidth::U32` | 4 bytes (LE) | `4,294,967,295` |

All multi-byte integers inside packed blocks are **little-endian**, matching `x86_64` and `aarch64` host endianness so SIMD kernels can load raw byte slices directly into vector registers without byte-swapping.

### 2.3 `TermRecord` (`cf_fts_terms`)

21-byte fixed header (`TERM_RECORD_HEADER_LEN = 21`) with optional inline v3 posting block for single-`(seg, block)` rare terms (`doc_freq <= INLINE_TERM_MAX_DF = 8`):

| Offset | Size | Field | Type | Meaning |
|---:|---:|---|---|---|
| `0` | 4 | `term_id` | `u32` LE | Monotonic numeric `TermId` indexing `cf_fts_skips`/`cf_fts_postings`/`cf_fts_positions` |
| `4` | 8 | `stats.doc_freq` | `u64` LE | Live documents containing this term |
| `12` | 4 | `stats.max_tf` | `u32` LE | Maximum term frequency across all postings of this term |
| `16` | 4 | `stats.min_dl_over_tf` | `f32` LE | Sound lower bound on `dl / tf` across all postings (`>= 0.0`, non-NaN; `f32::INFINITY` when empty) |
| `20` | 1 | `flags` | `u8` | `0x00`: out-of-line in `cf_fts_postings` + `cf_fts_skips`; `0x01`: inline `seg: u32 LE \|\| block: u32 LE \|\| v3 PackedPostingBlock` follows byte `20` |

### 2.4 `SkipBlock` (`cf_fts_skips`)

Chunked Block-Max skip summary covering up to `FTS_SKIP_CHUNK_SIZE = 256` posting blocks per KV (`chunk_id = block_id >> 8`):

```text
Offset                 Size                  Field
0                      1 B                   width: u8 (IntWidth tag for max_tfs)
1                      1 B                   reserved: u8 (must be 0)
2                      2 B                   count: u16 LE (1..=256)
4                      count * 4 B           min_dl_over_tfs: [f32; count] LE
4 + count * 4          count * width B       max_tfs: [u8 | u16 | u32; count] LE
4 + count*(4+width)    count B               rel_block_ids: [u8; count] (sorted ascending block_id & 0xFF)
```

`SkipBlockLayout::parse` validates a chunk once on load and slices views without re-parsing the header across `next_block()` / `seek_block()`.

### 2.5 `DocLenBlock` (FTS document lengths in `default`)

Fixed-slot 256-entry block stored once per `(attr_id, block_id)` whenever at least one document in the block has a non-zero token count (`count: 1..=256`). Deleted or absent slots store `dl = 0`.

```text
Offset   Size              Field
0        1 B               width: u8 (IntWidth tag: 0=U8, 1=U16, 2=U32)
1        1 B               reserved: u8 (must be 0)
2        2 B               count: u16 LE (number of non-zero slots, 1..=256)
4        256 * width B     dls: [u8 | u16 | u32; 256] LE (slot-indexed 0..255)
```

Total size is **260 bytes** (`U8`), **516 bytes** (`U16`), or **1,028 bytes** (`U32`). Because all 256 slots are always present in contiguous slot order, `prepare_dl_norm_256` streams through `dls` linearly with full-width SIMD loads.

### 2.6 v3 `PackedPostingBlock` (`cf_fts_postings` & Inline `TermRecord`)

Posting blocks store **only term frequencies (`tf`)**—never `dl`—and automatically choose between **Sparse** and **Dense** payloads at **`POSTING_DENSE_THRESHOLD = 96`** (`count >= 96` uses Dense; `count < 96` uses Sparse).

#### Common 12-Byte `PostingBlockHeader`

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 2 | `count` | `u16` LE | Number of postings in this block (`1..=256`) |
| `2` | 2 | `flags` | `u16` LE | Bit layout below (`POSTING_VERSION_V3 = 0x0300`) |
| `4` | 4 | `max_tf` | `u32` LE | Maximum `tf` in this block |
| `8` | 4 | `min_dl_over_tf` | `f32` LE | Sound block-level lower bound on `dl / tf` |

`flags` bit layout:
- Bits `8..15` (`0xFF00`, `POSTING_VERSION_MASK`): format version, must equal `0x0300` (`POSTING_VERSION_V3`).
- Bits `1..2` (`0x0006`, `WIDTH_MASK`, `WIDTH_SHIFT = 1`): `(flags & 0x0006) >> 1` encodes `IntWidth` (`0 = U8`, `1 = U16`, `2 = U32`).
- Bit `0` (`0x0001`, `FLAG_DENSE`): `0` = Sparse payload, `1` = Dense payload.
- All other bits must be `0`.

#### v3 Sparse Payload (`FLAG_DENSE == 0`, `count < 96`)

```text
Offset                  Size                 Field
0                       12 B                 PostingBlockHeader
12                      count * width B      tfs: [u8 | u16 | u32; count] LE (4-byte aligned after header)
12 + count * width      count B              slots: [u8; count] (strictly sorted ascending, parallel to tfs)
```

Placing `tfs` directly after the 12-byte header keeps multi-byte `tf` arrays 4-byte aligned for contiguous SIMD loads before scattering into `scores[slot]`.

#### Dense Payload (`FLAG_DENSE == 1`, `count >= 96`)

```text
Offset              Size                 Field
0                   12 B                 PostingBlockHeader
12                  32 B                 bitmap: [u64; 4] LE (256-bit presence mask; popcount == count)
44                  256 * width B        tfs: [u8 | u16 | u32; 256] LE (slot-indexed 0..255; 0 if absent)
```

At $\ge 96/256$ ($\ge 37.5$%) occupancy—especially with `U8` or `U16` `tf`s—a 256-slot array plus a 32-byte bitmap eliminates `slots` indirection: every slot `s` maps directly to `tfs[s]`, `dl_norm[s]`, and `scores[s]` at the same vector lane index.

---

## 3. Sound `min_dl_over_tf` Upper Bounds

For a posting `(tf, dl)`, the BM25 term contribution is:

$$\text{score}(tf, dl) = \frac{\text{num\textunderscore{}scale} \cdot tf}{tf + k_1(1 - b) + \frac{k_1 b}{\text{avgdl}} \cdot dl} = \frac{\text{num\textunderscore{}scale}}{1 + \frac{k_1(1 - b)}{tf} + \frac{k_1 b}{\text{avgdl}} \cdot \frac{dl}{tf}}$$

where $\text{num\textunderscore{}scale} = \text{weight} \cdot \text{idf} \cdot \text{qtf\textunderscore{}norm} \cdot (k_1 + 1)$.

- **Write-time bound tracking (`dl_over_tf_lower_bound`):** For every posting `(tf, dl)`, `PackedPostingBlock::recompute_bounds` and `TermStats::observe` compute `exact = (dl as f64) / (tf as f64)`, cast `r = exact as f32`, and apply `r.next_down()` if `(r as f64) > exact`. This guarantees $\text{min\textunderscore{}dl\textunderscore{}over\textunderscore{}tf} \le dl / tf$ in exact arithmetic.
- **Query-time upper bound (`Bm25TermScorer::upper_bound`):** Evaluates the denominator in `f64` using the term or block `max_tf` and `min_dl_over_tf`:

  $$\text{ub} = \frac{\text{num\textunderscore{}scale}}{1 + \frac{k_1(1 - b)}{\text{max\textunderscore{}tf}} + \frac{k_1 b}{\text{avgdl}} \cdot \text{min\textunderscore{}dl\textunderscore{}over\textunderscore{}tf}}$$

  Because `avgdl` is only applied at query time, skip bounds and `TermStats` never go stale across inserts or deletes.

---

## 4. How Data Is Used with SIMD

### 4.1 Runtime Tier Selection (`SimdTier::detect`)

At query startup (`execute_rank_by`), `SimdTier::detect()` selects the widest supported little-endian SIMD backend once per query:

| `SimdTier` | Target & ISA Guard | Vector Width (`f32` lanes) | Notes |
|---|---|---:|---|
| `Avx512` | `x86_64`, `avx512f` | 16 lanes (512-bit) | Native unsigned `u32 -> f32` (`_mm512_cvtepu32_ps`) and hardware 16-bit opmask arithmetic (`_mm512_mask_add_ps`, `_mm512_cmp_ps_mask`, `_mm512_reduce_max_ps`) |
| `Avx2` | `x86_64`, `avx2` + `fma` | 8 lanes (256-bit) | Uses `_mm256_cvtepi32_ps` for `U32`; falls back to scalar if `slice_u32_fits_i32` is false (`>= 2^31`) |
| `Neon` | `aarch64`, `neon` | 4 lanes (128-bit) | Native `vcvtq_f32_u32`, `vfmaq_f32`, `vbslq_f32`, `vmaxvq_f32` |
| `Scalar` | Portable fallback | 1 lane | Bit-twiddling (`trailing_zeros`) loops |

*(Every SIMD kernel in `crates/izzydb-fts/src/simd.rs` is cross-checked in unit tests against `SimdTier::Scalar` across `U8`/`U16`/`U32` widths, sparse/dense layouts, and arbitrary masks.)*

### 4.2 Shared `DlNormCache` & `prepare_dl_norm_256`

Notice that in the BM25 formula, the document-length term:

$$\text{dl\textunderscore{}norm}[s] = \max\left(c_1 + c_2 \cdot dl[s],\ \text{f32::MIN\textunderscore{}POSITIVE}\right), \qquad c_1 = k_1(1 - b),\quad c_2 = \frac{k_1 b}{\text{avgdl}}$$

is identical for every query term targeting the same `(attr_id, c1, c2)`.

1. During query compilation (`compile_clauses`), `DlNormCache::intern(attr_id, &scorer)` deduplicates `(attr_id, c1.to_bits(), c2.to_bits())` into a `dl_group` ID.
2. When a term actually decodes a surviving block (`DlNormCache::get_or_load`), `RocksDocLenCursor::seek_block_bytes(block_id)` returns the cached `DocLenBlock` slice (zero RocksDB lookups on warm queries).
3. `prepare_dl_norm_256` loads packed integers from `dls_bytes` (`U8`/`U16`/`U32`), widens them to `f32` in registers, and computes `max(fma(c2, dl, c1), MIN_POSITIVE)` across all 256 slots into `group.dl_norm: [f32; 256]`.

### 4.3 SIMD Kernels Reference

All 256-slot kernels live in [`crates/izzydb-fts/src/simd.rs`](../../crates/izzydb-fts/src/simd.rs):

| Kernel | Input Layout | Operation & Vectorization Strategy |
|---|---|---|
| `prepare_dl_norm_256` | `DocLenBlockView::dls_bytes` | Widens `U8`/`U16`/`U32` $\to$ `f32` and computes `max(c2 * dl + c1, MIN_POSITIVE)` linearly across 256 slots (16 / 8 / 4 lanes). |
| `accumulate_dense_256` | Dense `bitmap: [u64; 4]` + `tfs_bytes` | ORs `bitmap` into `touched`. Loops over the four 64-doc words, **skipping any word where `bw == 0`**. For non-zero words, loads contiguous 16 / 8 / 4 `tf`s, `dl_norm`, and `scores`, adding `(num_scale * tf) / (tf + dl_norm)`. |
| `accumulate_dense_masked_256` | Dense `bitmap`, `mask: [u64; 4]`, `tfs_bytes` | Computes `active = bitmap & mask`. If `popcount == 0`, returns; if **`popcount <= 24`**, runs scalar `trailing_zeros` bit-iteration to touch only active slots. For `popcount > 24`, skips zero sub-chunks (`16`/`8`/`4`-bit slices) and uses masked SIMD adds. |
| `accumulate_sparse` | Sparse `slots: &[u8]` + aligned `tfs_bytes` | When `slots.len() >= lane_width`, gathers `dl_norm[slots[i + lane]]` into a stack buffer, loads contiguous `tfs` via SIMD, computes `(num_scale * tf) / (tf + dl_norm)`, and scatters sums to `scores[slot]`. Remainder uses `IntWidth`-specialized scalar loops. |
| `accumulate_sparse_masked` | Sparse `slots`, `tfs_bytes`, `mask` | If `slots.len() < 8` or `mask_pop_le_8(mask)`, runs a scalar bit-test loop over `slots`. Otherwise compacts matching `(slot, tf: u32)` pairs into stack `[u8; 96]` / `[u32; 96]` buffers and runs `accumulate_u32_pairs`. |
| `max_score_256` | `scores: &[f32; 256]`, `mask: &[u64; 4]` | Returns max `scores[slot]` over set bits of `mask`. Uses scalar bit-iteration for `pop <= 8`; otherwise SIMD reduction. |
| `filter_ge_256` | `scores: &[f32; 256]`, `mask`, `threshold` | Returns `[u64; 4]` bitmask of slots in `mask` with `scores[slot] >= threshold`. |
| `merge_max_256` / `merge_add_256` | Two `[f32; 256]` buffers | Linear 16 / 8 / 4-lane elementwise `max` and `add` for `PrefixMax` (`last_as_prefix`) and `MaxBranches` (`RankBy::Max`). |

### 4.4 Block-Max MAXSCORE Execution Flow (`execute_rank_by`)

```text
  Compile RankBy -> Clauses sorted ascending by global term_max
    • multi_get_term_records (cf_fts_terms) + batched open_skip_cursors (cf_fts_skips chunk 0 / inline)
        │
        ▼
  Next RowBlock = (seg, block) = min(essential cursors [essential_start..])
        │
        ├──[BlockMaskProvider misses (seg, block)]────────────────────► advance essential, skip
        │
        ├──[theta > 0 && ess_block_ub + non_ess_ub + eps < theta]─────► advance essential, skip
        │   (reads only in-memory SkipBlock bounds / inline header — 0 cf_fts_postings reads)
        ▼
  Essential SIMD Accumulation (scores[256] = 0, touched = 0)
    • For each essential clause at (seg, block):
      lazy current_posting_bytes() (cf_fts_postings) ─► accumulate_block_simd + advance_past
    • active_mask = touched & allowed_mask (liveness & pushed-down filter RowSet)
    • max_ess = max_score_256(scores, active_mask)
        │
        ├──[max_ess + non_ess_ub + eps < theta]───────────────────────► prune all non-essential
        ▼
  Descending Non-Essential Seek & Progressive Slot Pruning
    • Seek j = (essential_start - 1) down to 0 via skip cursor bounds only:
      break early if (max_ess + non_ess_block_ub + rem_global + eps < theta)
    • Loop surviving non-essential clauses (descending term_max):
      - active_mask = filter_ge_256(scores, active_mask, theta - rem_ub - eps)
      - if active_mask == [0; 4]: break
      - lazy current_posting_bytes() ─► accumulate_block_masked_simd
        │
        ▼
  Top-k Candidate Extraction & Dynamic Partitioning
    • hit_mask = filter_ge_256(scores, active_mask, cand_thresh)
    • Extract (score, RowId::new(seg, (block << 8) | slot)) via trailing_zeros
    • Update min-heap & theta (resolving DocIds via cf_row_doc_ids on ties/lazy filter)
    • Advance essential_start while prefix_max_sum[essential_start] + eps < theta
```

Key execution details:

1. **Out-of-line skip pruning:** `TermStream::block_upper_bound` reads `(max_tf, min_dl_over_tf)` directly from the `SkipBlockView` pinned in `RocksSkipCursor` (or an inline `TermRecord`). A block must survive pruning before the cursor reads postings or document lengths for it.
2. **Batched `chunk 0` & lazy sequential `cf_fts_postings` cursors:** `open_fts_skip_cursors` batch-fetches `chunk 0` of `cf_fts_skips` (covering docs `0..65,535`) via a single sorted `batched_multi_get_cf`, avoiding per-term skip iterators; each `RocksSkipCursor` opens a sequential `DBRawIterator` on `cf_fts_postings` lazily only on its first surviving block.
3. **Descending non-essential seek & early exit:** Non-essential clauses are sought in **descending** global `term_max` order using skip bounds only, stopping as soon as `(max_ess + non_ess_block_ub + rem_global) < theta`.
4. **Progressive `active_mask` narrowing:** Before each surviving non-essential clause, `filter_ge_256` clears bits in `active_mask` that can no longer reach `theta`, driving masked accumulation into scalar fast paths and terminating early when `active_mask == [0; 4]`.
