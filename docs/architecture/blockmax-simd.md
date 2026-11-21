# SIMD Block-Max MAXSCORE & Data Layout

This document describes the on-disk/in-memory data layout for full-text search (FTS) blocks and how the Block-Max MAXSCORE query engine executes vectorized BM25 scoring over them using runtime-dispatched SIMD kernels.

Implementation sources:
- Block ID / slot bit-splitting (`FTS_BLOCK_SIZE = 256`): [`crates/izzydb-core/src/config.rs`](../../crates/izzydb-core/src/config.rs)
- Term stats, `DocLenBlock`, v2 `PackedPostingBlock`, `IntWidth`, and `dl_over_tf_lower_bound`: [`crates/izzydb-fts/src/block.rs`](../../crates/izzydb-fts/src/block.rs)
- BM25 scoring parameters, upper-bound formula, and SIMD dispatch wrappers: [`crates/izzydb-fts/src/bm25.rs`](../../crates/izzydb-fts/src/bm25.rs)
- Runtime-dispatched SIMD kernels (`Avx512`, `Avx2`, `Neon`, `Scalar`): [`crates/izzydb-fts/src/simd.rs`](../../crates/izzydb-fts/src/simd.rs)
- MAXSCORE execution loop, `DlNormCache`, `BlockFilterMask`, and `Clause` evaluation: [`crates/izzydb-fts/src/maxscore.rs`](../../crates/izzydb-fts/src/maxscore.rs)
- RocksDB column families, `RocksPostingCursor`, and cached `RocksDocLenCursor`: [`crates/izzydb-storage/src/storage.rs`](../../crates/izzydb-storage/src/storage.rs), [`crates/izzydb-storage/src/codec.rs`](../../crates/izzydb-storage/src/codec.rs)

---

## 1. Design Overview: Why 256-Document Blocks?

Rather than storing variable-length posting lists keyed by arbitrary document IDs, `FtsIndex` assigns each document a dense monotonic `FtsDocId: u32` and partitions the `u32` ID space into fixed-size blocks of **`FTS_BLOCK_SIZE = 256`** documents:

```text
FtsDocId (u32):  [ block_id: 24 bits (fts_doc_id >> 8) | slot: 8 bits (fts_doc_id & 0xFF) ]
```

A fixed 256-slot block geometry aligns storage, caching, and SIMD execution around three hardware-friendly primitives:

1. **Compact 256-bit masks (`[u64; 4]`, 32 bytes):** Presence bitmaps, filter masks, touched-slot sets, and candidate hit masks all share the same four `u64` words. Bitwise set operations (`&`, `|`), popcount checks, and SIMD predicate comparisons map 1-to-1 onto 256 slots.
2. **L1-resident working buffers (`[f32; 256]`, 1 KiB each):** Both the per-block score accumulator (`scores`) and precomputed document-length normalization array (`dl_norm`) fit in 1 KiB on the stack—16 AVX-512 vectors, 32 AVX2 vectors, or 64 NEON vectors.
3. **Out-of-line document lengths (`cf_fts_doc_lens`):** Document lengths (`dl`) depend only on `(attr_id, block_id)`, not on individual query terms. Storing them once per block eliminates `dl` duplication from every term's posting block and lets all query terms on the same attribute share one SIMD-computed `dl_norm` array.

---

## 2. Data Layout

### 2.1 RocksDB Column Families & Keys

Block-Max MAXSCORE reads from three FTS column families (plus `cf_meta["fts_meta"]` for per-attribute `doc_count` / `total_tokens` and `cf_fts_rev_doc_map` to resolve top-$k$ `FtsDocId -> DocId`):

| Column Family | Key Format | Value Format | Access & Caching Pattern |
|---|---|---|---|
| `cf_fts_terms` | `attr_id` (`u16` BE) \|\| `token_utf8` | `TermStats` (16 B LE) | Sorted `batched_multi_get_cf` (or 2-byte prefix scan for `last_as_prefix`) |
| `cf_fts_doc_lens` | `attr_id` (`u16` BE) \|\| `block_id` (`u32` BE) | `DocLenBlock` (260 / 516 / 1028 B) | Uncompressed CF; lazily loaded into `Storage::fts_doc_lens_cache` (`Arc<Vec<Option<Box<[u8]>>>>`) per `attr_id` and updated write-through on commit |
| `cf_fts_postings` | `attr_id` (`u16` BE) \|\| `escaped_token` \|\| `block_id` (`u32` BE) | v2 `PackedPostingBlock` | `RocksPostingCursor` with cached `current_block`, reusable `seek_buf`, and single-step `iter.next()` when `cur + 1 == target` |

*(In `cf_fts_postings`, `escaped_token` escapes `0x00 -> 0x00 0xFF` and terminates with `0x00 0x00`, keeping `(attr_id, token)` keys prefix-free before the big-endian `block_id`.)*

### 2.2 Adaptive Integer Width (`IntWidth`)

Both `DocLenBlock` and `PackedPostingBlock` pack integer arrays using the smallest width (`IntWidth::min_for_max`) that fits the block's maximum value:

| `IntWidth` Tag | Rust Variant | Element Size | Max Value |
|---:|---|---:|---:|
| `0` | `IntWidth::U8` | 1 byte | `255` |
| `1` | `IntWidth::U16` | 2 bytes (LE) | `65,535` |
| `2` | `IntWidth::U32` | 4 bytes (LE) | `4,294,967,295` |

All multi-byte integers inside packed blocks are **little-endian**, matching `x86_64` and `aarch64` host endianness so SIMD kernels can load raw byte slices directly into vector registers without byte-swapping.

### 2.3 `TermStats` (`cf_fts_terms`)

Fixed **16-byte** little-endian record:

| Offset | Size | Field | Type | Meaning |
|---:|---:|---|---|---|
| `0` | 8 | `doc_freq` | `u64` LE | Number of documents containing this term |
| `8` | 4 | `max_tf` | `u32` LE | Maximum term frequency across all postings of this term |
| `12` | 4 | `min_dl_over_tf` | `f32` LE | Sound lower bound on `dl / tf` across all postings (`>= 0.0`, non-NaN; `f32::INFINITY` when empty) |

### 2.4 `DocLenBlock` (`cf_fts_doc_lens`)

Fixed-slot 256-entry block stored once per `(attr_id, block_id)` whenever at least one document in the block has a non-zero token count (`count: 1..=256`). Deleted or absent slots store `dl = 0`.

```text
Offset   Size              Field
0        1 B               width: u8 (IntWidth tag: 0=U8, 1=U16, 2=U32)
1        1 B               reserved: u8 (must be 0)
2        2 B               count: u16 LE (number of non-zero slots, 1..=256)
4        256 * width B     dls: [u8 | u16 | u32; 256] LE (slot-indexed 0..255)
```

Total size is **260 bytes** (`U8`), **516 bytes** (`U16`), or **1,028 bytes** (`U32`). Because all 256 slots are always present in contiguous slot order, `prepare_dl_norm_256` streams through `dls` linearly with full-width SIMD loads.

### 2.5 v2 `PackedPostingBlock` (`cf_fts_postings`)

Posting blocks store **only term frequencies (`tf`)**—never `dl`—and automatically choose between **Sparse** and **Dense** payloads at **`POSTING_DENSE_THRESHOLD = 96`** (`count >= 96` uses Dense; `count < 96` uses Sparse).

#### Common 12-Byte `PostingBlockHeader`

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 2 | `count` | `u16` LE | Number of postings in this block (`1..=256`) |
| `2` | 2 | `flags` | `u16` LE | Bit layout below (`POSTING_VERSION_V2 = 0x0200`) |
| `4` | 4 | `max_tf` | `u32` LE | Maximum `tf` in this block |
| `8` | 4 | `min_dl_over_tf` | `f32` LE | Sound block-level lower bound on `dl / tf` |

`flags` bit layout:
- Bits `8..15` (`0xFF00`, `POSTING_VERSION_MASK`): format version, must equal `0x0200` (`POSTING_VERSION_V2`).
- Bits `1..2` (`0x0006`, `WIDTH_MASK`, `WIDTH_SHIFT = 1`): `(flags & 0x0006) >> 1` encodes `IntWidth` (`0 = U8`, `1 = U16`, `2 = U32`).
- Bit `0` (`0x0001`, `FLAG_DENSE`): `0` = Sparse payload, `1` = Dense payload.
- All other bits must be `0`.

#### Sparse Payload (`FLAG_DENSE == 0`, `count < 96`)

```text
Offset              Size                 Field
0                   12 B                 PostingBlockHeader
12                  count B              slots: [u8; count] (strictly sorted ascending)
12 + count          count * width B      tfs: [u8 | u16 | u32; count] LE (parallel to slots)
```

Why Sparse? When fewer than 96 of the 256 slots match, packing `(slots, tfs)` in parallel Structure-of-Arrays (SoA) avoids padding 256 `tf` slots and keeps `tfs` contiguous for SIMD arithmetic before scattering into `scores[slot]`.

#### Dense Payload (`FLAG_DENSE == 1`, `count >= 96`)

```text
Offset              Size                 Field
0                   12 B                 PostingBlockHeader
12                  32 B                 bitmap: [u64; 4] LE (256-bit presence mask; popcount == count)
44                  256 * width B        tfs: [u8 | u16 | u32; 256] LE (slot-indexed 0..255; 0 if absent)
```

Why Dense? At $\ge 96/256$ ($\ge 37.5\%$) occupancy—especially with `U8` or `U16` `tf`s—a 256-slot array plus a 32-byte bitmap is compact while eliminating `slots` indirection. Every slot `s` maps directly to `tfs[s]`, `dl_norm[s]`, and `scores[s]` at the same vector lane index, enabling aligned linear SIMD loads/stores without gathers or scatters.

---

## 3. Sound `min_dl_over_tf` Upper Bounds

For a posting `(tf, dl)`, the BM25 term contribution is:

$$\text{score}(tf, dl) = \frac{\text{num\_scale} \cdot tf}{tf + k_1(1 - b) + \frac{k_1 b}{\text{avgdl}} \cdot dl} = \frac{\text{num\_scale}}{1 + \frac{k_1(1 - b)}{tf} + \frac{k_1 b}{\text{avgdl}} \cdot \frac{dl}{tf}}$$

where $\text{num\_scale} = \text{weight} \cdot \text{idf} \cdot \text{qtf\_norm} \cdot (k_1 + 1)$.

- **Write-time bound tracking (`dl_over_tf_lower_bound`):** For every posting `(tf, dl)`, `PackedPostingBlock::recompute_bounds` and `TermStats::observe` compute `exact = (dl as f64) / (tf as f64)`, cast `r = exact as f32`, and apply `r.next_down()` if `(r as f64) > exact`. This guarantees $\text{min\_dl\_over\_tf} \le dl / tf$ in exact arithmetic.
- **Query-time upper bound (`Bm25TermScorer::upper_bound`):** Evaluates the denominator in `f64` using the term or block `max_tf` and `min_dl_over_tf`:

  $$\text{ub} = \frac{\text{num\_scale}}{1 + \frac{k_1(1 - b)}{\max\_tf} + \frac{k_1 b}{\text{avgdl}} \cdot \text{min\_dl\_over\_tf}}$$

  Because `avgdl` is only applied at query time, block headers and `TermStats` never go stale across inserts or deletes. Pairing `max_tf` with `min_dl_over_tf` instead of an independent `min_dl` yields much tighter block bounds when high-`tf` documents are longer than the shortest document in the block.

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

$$\text{dl\_norm}[s] = \max\left(c_1 + c_2 \cdot dl[s],\ \text{f32::MIN\_POSITIVE}\right), \qquad c_1 = k_1(1 - b),\quad c_2 = \frac{k_1 b}{\text{avgdl}}$$

is identical for every query term targeting the same `(attr_id, c1, c2)`.

1. During query compilation (`compile_clauses`), `DlNormCache::intern(attr_id, &scorer)` deduplicates `(attr_id, c1.to_bits(), c2.to_bits())` into a `dl_group` ID.
2. When a term actually decodes a block (`DlNormCache::get_or_load`), if `group.cached_block_id != Some(block_id)`, `RocksDocLenCursor::seek_block_bytes(block_id)` returns the cached `DocLenBlock` slice (zero RocksDB lookups on warm queries).
3. `prepare_dl_norm_256` loads 16 / 8 / 4 packed integers from `dls_bytes` (`U8`/`U16`/`U32`), widens them to `f32` in registers, and computes `max(fma(c2, dl, c1), MIN_POSITIVE)` across all 256 slots into `group.dl_norm: [f32; 256]`. Subsequent terms in the same block reuse `group.dl_norm` directly from L1 cache.

### 4.3 SIMD Kernels Reference

All 256-slot kernels live in [`crates/izzydb-fts/src/simd.rs`](../../crates/izzydb-fts/src/simd.rs):

| Kernel | Input Layout | Operation & Vectorization Strategy |
|---|---|---|
| `prepare_dl_norm_256` | `DocLenBlockView::dls_bytes` | Widens `U8`/`U16`/`U32` $\to$ `f32` and computes `max(c2 * dl + c1, MIN_POSITIVE)` linearly across 256 slots (16 / 8 / 4 lanes). |
| `accumulate_dense_256` | Dense `bitmap: [u64; 4]` + `tfs_bytes` | ORs `bitmap` into `touched`. Loops over the four 64-doc words, **skipping any word where `bw == 0`**. For non-zero words, loads contiguous 16 / 8 / 4 `tf`s, `dl_norm`, and `scores`, adding `(num_scale * tf) / (tf + dl_norm)`. |
| `accumulate_dense_masked_256` | Dense `bitmap`, `mask: [u64; 4]`, `tfs_bytes` | Computes `active = bitmap & mask`. If `popcount == 0`, returns; if **`popcount <= 24`**, runs scalar `trailing_zeros` bit-iteration (`accumulate_active_bits_scalar`) to touch only active slots. For `popcount > 24`, skips zero sub-chunks (`16`/`8`/`4`-bit slices of `active[w]`) and uses masked SIMD adds (`_mm512_mask_add_ps`, AVX2 bit-mask blend, NEON `vbslq_f32`). |
| `accumulate_sparse` | Sparse `slots: &[u8]` + `tfs_bytes` | When `slots.len() >= lane_width` (`16` / `8` / `4`), gathers `dl_norm[slots[i + lane]]` into a stack buffer, loads contiguous `tfs` via SIMD, computes `(num_scale * tf) / (tf + dl_norm)` in vector lanes, and scatters sums to `scores[slot]` while setting `touched` bits. Remainder `< lane_width` uses `IntWidth`-specialized scalar loops. |
| `accumulate_sparse_masked` | Sparse `slots`, `tfs_bytes`, `mask` | If `slots.len() < 8` or `mask_pop_le_8(mask)` (`popcount <= 8`), runs a scalar bit-test loop directly over `slots`. Otherwise compacts matching `(slot, tf: u32)` pairs into stack `[u8; 96]` / `[u32; 96]` buffers and runs `accumulate_u32_pairs` (16 / 8 / 4-wide SIMD). |
| `max_score_256` | `scores: &[f32; 256]`, `mask: &[u64; 4]` | Returns max `scores[slot]` over set bits of `mask`. Uses scalar bit-iteration when `mask_pop_le_8(mask)`; otherwise skips zero sub-chunks and computes SIMD max (`_mm512_maskz_mov_ps` + `_mm512_reduce_max_ps` on AVX-512; lane masking + horizontal shuffle max on AVX2; `vtstq_u32`/`vbslq_f32` + `vmaxvq_f32` on NEON). |
| `filter_ge_256` | `scores: &[f32; 256]`, `mask`, `threshold` | Returns `[u64; 4]` bitmask of slots in `mask` with `scores[slot] >= threshold`. Uses scalar bit-iteration for `mask_pop_le_8(mask)`; otherwise skips zero sub-chunks and extracts comparison bitmasks directly (`_mm512_cmp_ps_mask::<_CMP_GE_OQ>`, `_mm256_cmp_ps` + `_mm256_movemask_ps`, NEON `vcgeq_f32` + weighted `vaddvq_u32`). |
| `merge_max_256` / `merge_add_256` | Two `[f32; 256]` buffers | Linear 16 / 8 / 4-lane elementwise `max` and `add` used when combining `PrefixMax` (`last_as_prefix`) and `MaxBranches` (`RankBy::Max`) clauses. |

### 4.4 Block-Max MAXSCORE Execution Flow (`execute_rank_by`)

```text
  Compile RankBy -> Clauses sorted ascending by global term_max
        │
        ▼
  Next block_id = min(essential cursors [essential_start..])
        │
        ├──[BlockFilterMask misses block_id]──────────────────────────► advance essential, skip
        │
        ├──[theta > 0 && ess_block_ub + non_ess_ub + eps < theta]─────► advance essential, skip
        │   (reads only 12-byte PostingBlockHeaders)
        ▼
  Essential SIMD Accumulation (scores[256] = 0, touched = 0)
    • For each essential clause at block_id:
      accumulate_block_simd (Dense or Sparse) + advance_past(block_id)
    • active_mask = touched & allowed_mask
    • max_ess = max_score_256(scores, active_mask)
        │
        ├──[max_ess + non_ess_ub + eps < theta]───────────────────────► prune all non-essential
        ▼
  Descending Non-Essential Seek & Progressive Slot Pruning
    • Seek j = (essential_start - 1) down to 0:
      break early if (max_ess + non_ess_block_ub + rem_global + eps < theta)
    • Loop surviving non-essential clauses (descending term_max):
      - active_mask = filter_ge_256(scores, active_mask, theta - rem_ub - eps)
      - if active_mask == [0; 4]: break
      - accumulate_block_masked_simd (Dense masked or Sparse masked)
        │
        ▼
  Top-k Candidate Extraction & Dynamic Partitioning
    • hit_mask = filter_ge_256(scores, active_mask, cand_thresh)
    • Extract (score, make_fts_doc_id(block_id, slot)) via trailing_zeros
    • Update min-heap & theta (resolving DocIds lazily for filters/ties)
    • Advance essential_start while prefix_max_sum[essential_start] + eps < theta
```

Key execution details that maximize SIMD and pruning synergy:

1. **Header-only skipping before decoding:** `TermStream::block_upper_bound` parses only the 12-byte `PostingBlockHeader` (`max_tf`, `min_dl_over_tf`). If the block cannot beat `theta`, neither `PostingBlockView` nor `DocLenBlockView` (`prepare_dl_norm_256`) is touched.
2. **Fused cursor advance:** Essential clauses call `advance_past(block_id)` right after accumulating `block_id`, keeping cursors aligned even if `essential_start` grows at the end of the iteration.
3. **Descending non-essential seek & early exit:** Non-essential clauses are sought in **descending** global `term_max` order. Absent terms (`block_upper_bound == 0`) or low-bound blocks often drop `max_ess + non_ess_block_ub + rem_global` below `theta` immediately, avoiding seeks on the remaining long tail of low-IDF terms.
4. **Progressive `active_mask` narrowing:** Before each surviving non-essential clause, `filter_ge_256` clears bits in `active_mask` for slots whose accumulated score plus the remaining non-essential block upper bounds (`rem_ub`) can no longer reach `theta`. Because `active_mask` shrinks monotonically from clause to clause, masked accumulation rapidly crosses into the scalar fast paths (`pop <= 24` for Dense, `pop <= 8` for Sparse) and terminates the loop as soon as `active_mask == [0; 4]`.
5. **Composite `RankBy` handling:** `RankBy::Sum` flattens into additive clauses and `RankBy::Product` distributes scalar weights into `Bm25TermScorer::num_scale`. Multi-term `last_as_prefix` (`PrefixMax`) and `RankBy::Max` (`MaxBranches`) evaluate their branches into temporary `[f32; 256]` buffers (fast-pathing directly into `scores` when only one branch matches `block_id`) and combine them with `merge_max_256` and `merge_add_256`.
6. **Filter pushdown via `BlockFilterMask`:** When a `Filter` produces a `RoaringBitmap<FtsDocId>`, `BlockFilterMask::from_bitmap` pre-slices it into per-block `[u64; 4]` masks. Entire blocks missing from the map are skipped before header checks, and matching blocks intersect `touched &= mask` right after essential accumulation so `max_score_256` and `filter_ge_256` ignore filtered-out slots.
