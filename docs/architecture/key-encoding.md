# Key & Binary Encoding Specification

This document specifies the persistent RocksDB key-value encodings, packed node/vector/row/columnar blob formats, order-preserving attribute encodings, FTS block formats, and RaBitQ bit-packing layouts in IzzyDB v3.

Implementation sources:
- Column families, RocksDB keys, compaction filters, and CF options: [`crates/izzydb-storage/src/storage/`](../../crates/izzydb-storage/src/storage/)
- Packed ANN node/metadata serialization and zero-copy views: [`crates/izzydb-storage/src/node/mod.rs`](../../crates/izzydb-storage/src/node/mod.rs)
- `RowView` v3, `SegPosting`, `Manifest`, `SegmentMeta`, `DetachedMeta`, zone maps: [`crates/izzydb-codec/src/lib.rs`](../../crates/izzydb-codec/src/lib.rs)
- `ColumnBlock` & `ColumnBlockBuilder` (`cf_columns`, `cf_detached`): [`crates/izzydb-storage/src/column/mod.rs`](../../crates/izzydb-storage/src/column/mod.rs)
- FTS metadata, `TermRecord` v4, `FtsChunk` v4, `DocLenBlock`, compact `TfEncoding` / v3 `PackedPostingBlock`, and `PackedPositionBlock`: [`crates/izzydb-fts/src/block/mod.rs`](../../crates/izzydb-fts/src/block/mod.rs)
- ID aliases, `RowId`/`RowBlock`/`SkipChunk`, and schema types: [`crates/izzydb-core/src/config.rs`](../../crates/izzydb-core/src/config.rs), [`crates/izzydb-core/src/schema/mod.rs`](../../crates/izzydb-core/src/schema/mod.rs)
- Structural and index invariants: [`crates/izzydb/src/verify/mod.rs`](../../crates/izzydb/src/verify/mod.rs)

*(For how the FTS SIMD Block-Max MAXSCORE engine consumes FTS blocks, see [`blockmax-simd.md`](blockmax-simd.md). For segment compaction and detached columns, see [`segments-and-compaction.md`](segments-and-compaction.md) and [`detached-columns.md`](detached-columns.md).)*

---

## 1. Identifiers & Endianness Conventions

| Type | Rust Definition | Width | Endianness in RocksDB Keys | Endianness Inside Packed Blobs (`Manifest`, `Packed*`, FTS/Column Blocks) |
|---|---|---:|---|---|
| `DocId` | `u64` | 8 B | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `SegmentId` | `u32` | 4 B | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) — monotonic from `0`, never reused |
| `LocalId` | `u32` | 4 B | **Big-endian** (`to_be_bytes`) | **Little-endian** — dense append-only within segment (`< MAX_SEGMENT_ROWS = 1 << 20`) |
| `RowId` | `u64` | 8 B | **Big-endian** (`to_be_bytes`) | **Little-endian** (`(seg as u64) << 32 \| local`) |
| `RowBlock` | `(SegmentId, u32)` | 8 B | `seg BE \|\| block BE` | `block = local >> 8`, `slot = (local & 0xFF) as u8` ($\le 4,096$ blocks/seg) |
| `SkipChunk` | `(SegmentId, u32)` | 8 B | `seg BE \|\| chunk BE` | `chunk = block >> 8` (256 blocks = 65,536 rows) |
| `NodeId` | `u32` | 4 B | **Big-endian** (`to_be_bytes`) | **Little-endian** (ANN tree node) |
| `AttrId` | `u16` | 2 B | **Big-endian** (`to_be_bytes`) | **Little-endian** (shared id space for `Inline` & `Detached`) |
| `TermId` | `u32` | 4 B | **Big-endian** (`to_be_bytes`) | **Little-endian** |
| `ColumnGen` | `u32` | 4 B | **Big-endian** (`to_be_bytes`) | **Little-endian** (monotonic generation per detached column) |

- **Big-endian in RocksDB keys** guarantees lexicographical byte order equals numeric/value order (`cf_id_map` sorted `batched_multi_get_cf`, value-major `cf_attr_postings` range scans, and FTS `(seg, block)` ordering).
- **Little-endian inside packed blobs** enables zero-copy borrowing of aligned slices (`row_ids`, `codes`, `code_popcounts`, `doc_ids`, `dls`, `tfs`, columnar buffers) directly from `DBPinnableSlice`.

---

## 2. RocksDB Column Families (16 CFs)

| Column Family | Key Format | Value Format | Compression | Notes |
|---|---|---|---|---|
| `default` | tagged, see §3 | metadata / bitmaps / counts | Default (Snappy) | point lookups + small scans; Bloom 10 bpk |
| `cf_id_map` | `doc_id` (`u64` BE, 8 B) | `row_id` (`u64` BE, 8 B) | Default (Snappy) | Bloom 10 bpk; two-level partitioned index/filter; live docs only (bijection to live `cf_rows`) |
| `cf_rows` | `row_id` (`u64` BE, 8 B) | `RowView` v3 (inline attrs, §4.1) | Default (Snappy) | No Bloom (keys known to exist); two-level index; dead rows remain until compaction |
| `cf_row_doc_ids` | `seg` BE \|\| `block` BE (8 B) | `[u64; 256]` LE `DocId`s (2,048 B) | `None` | No Bloom; two-level index; reverse map; liveness decides whether slot holds live `DocId` |
| `cf_vectors` | `row_id` (`u64` BE, 8 B) | `[f32; dim]` LE | `None` | No Bloom; two-level index; 16 KiB data blocks + intra-block hash; absent when `has_vectors == false` or row has no vector |
| `cf_leaf_entries` | `leaf_id` (`u32` BE, 4 B) | `PackedLeafNode` v2 (§4.3) | `None` | No Bloom; two-level index; 16 KiB data blocks; stores `row_ids` + `stale_hint` |
| `cf_node_summaries` | `node_id` (`u32` BE, 4 B) | `NodeSummary` v1 (§4.9) | `None` | Leaf and internal min/max/presence/equality-bitset rollups; loaded into RAM at open |
| `cf_bsi` | `seg` BE \|\| `attr_id` BE \|\| `chunk` BE \|\| `plane` (11 B) | `[u64; 1024]` LE (8,192 B) or base record | `None` | Per-segment bit-sliced ordered numeric ranges; plane `0xFE` = base+width, `0xFF` = presence |
| `cf_attr_postings` | `attr_id` BE \|\| `tag` \|\| `encoded_val` \|\| `seg` BE | `SegPosting` (§4.2) | `None` | **Value-major**, Bloom 10 bpk, prefix extractor 3 B (`attr_id \|\| tag`) |
| `cf_columns` | `seg` BE \|\| `attr_id` BE \|\| `block` BE (10 B) | `ColumnBlock` (§4.4) | `None` | Bloom 10 bpk; two-level partitioned index/filter; 16 KiB blocks + intra-block hash; sealed segments only (inline attrs) |
| `cf_detached` | `attr_id` BE \|\| `gen` BE \|\| `seg` BE \|\| `block` BE (14 B) | `ColumnBlock` (§4.4) | `None` | Bloom 10 bpk; two-level partitioned index/filter; active + pending generations |
| `cf_detached_postings` | `attr_id` BE \|\| `gen` BE \|\| `tag` \|\| `encoded_val` \|\| `seg` BE | `SegPosting` (§4.2) | `None` | Filterable detached columns, Bloom 10 bpk, prefix 7 B (`attr \|\| gen \|\| tag`) |
| `cf_fts_terms` | `attr_id` BE \|\| `token_utf8` | `TermRecord` v4 (§4.5) | Default (Snappy) | Prefix 2 B (`attr_id`), Bloom 10 bpk |
| `cf_fts_skips` | `term_id` BE \|\| `seg` BE \|\| `chunk` BE (12 B) | `FtsChunk` v4 (§4.5) | `None` | Prefix 4 B (`term_id`); unified skip bounds + compact postings; no Bloom filter |
| `cf_fts_postings` | `term_id` BE \|\| `seg` BE \|\| `chunk` BE (12 B) | *(deprecated — empty)* | `None` | Retained empty in v4; postings live inside `FtsChunk` (`cf_fts_skips`) |
| `cf_fts_positions` | `term_id` BE \|\| `seg` BE \|\| `block` BE (12 B) | `PackedPositionBlock` | Default (Snappy) | Prefix 4 B (`term_id`), Bloom 10 bpk |

### RocksDB Table & Block Cache Tuning

Per-CF options live in [`crates/izzydb-storage/src/storage/cf_options.rs`](../../crates/izzydb-storage/src/storage/cf_options.rs). Search hot paths (`cf_vectors`, `cf_leaf_entries`, `cf_node_summaries`, `cf_bsi`, `cf_rows`, `cf_row_doc_ids`) read keys already known to exist, so they omit Bloom filters and use two-level partitioned indexes that keep metadata reloads small. `cf_bsi` is uncompressed with 16 KiB blocks + intra-block hash because every plane value is an 8 KiB bitset. Large point-lookup CFs with retained Bloom filters (`cf_id_map`, `cf_columns`, `cf_detached`) also partition filters into 4 KiB blocks. `cf_columns` and `cf_detached` store uncompressed blocks because their ~2 KiB fixed-width `ColumnBlock`s inline strings/arrays and Snappy decompression dominated filtered scans. The shared `LRUCache` defaults to 128 MiB and is configurable per collection with `CollectionConfig::with_block_cache_bytes(...)`.

### Compaction Filters

`cf_rows`, `cf_row_doc_ids`, `cf_vectors`, `cf_columns`, `cf_bsi`, `cf_detached`, `cf_attr_postings`, `cf_detached_postings`, `cf_fts_skips`, `cf_fts_positions`, `default[0x05]` (and deprecated `cf_fts_postings`) install a RocksDB `CompactionFilter` that drops any key whose embedded `SegmentId` belongs to `Storage::dropped_segments` (`Arc<ArcSwap<RoaringBitmap>>`). For `cf_detached`, `cf_detached_postings`, `default[0x07]`, and `default[0x14]`, the filter additionally drops keys whose `ColumnGen < retired_below[attr_id]`. `cf_node_summaries` is node-keyed and is deleted explicitly when a leaf or internal node is removed. Because segment IDs and generation IDs are never reused, filtering is lock-free and race-free.

---

## 3. `default` CF Keyspaces

| Key | Value | Notes |
|---|---|---|
| `b"\x01schema"` | `CollectionSchema` v4 | Adds `ColumnStorage::{Inline, Detached}` per attr and `has_vectors: bool` |
| `b"\x01manifest"` | `Manifest` (§4.6, 32 B) | `next_segment_id`, `open_segment`, `seal_rows`, `live_rows`, `total_rows` |
| `b"\x01ann_meta"` | `AnnMeta` | `TreeMeta` fields + `stale_rows`, `vacuum_epoch`, `vacuum_cursor` (absent when vectorless) |
| `b"\x01fts_meta"` | `FtsMeta` v4 | `next_term_id` + per-attr `(doc_count, total_tokens)` over live rows |
| `b"\x01attr_counts_v1"` | empty marker | Marks inline `0x06` live-doc counts initialized |
| `b"\x01dropped_segments"` | `RoaringBitmap<SegmentId>` | Monotonic bitmap of retired segments feeding compaction filters |
| `b"\x01bulk_import_v3"` | build id (`u64` LE) | Crash-detection marker during SST ingestion (`bulk_import` / `bulk_append`) |
| `0x02 \|\| node_id` BE (5 B) | `PackedInternalNode` | ANN tree internal node |
| `0x05 \|\| attr_id` BE \|\| `seg` BE \|\| `block` BE (11 B) | `DocLenBlock` | FTS document lengths per `(attr_id, seg, block)` |
| `0x06 \|\| attr_id` BE \|\| `tag` \|\| `encoded_val` | `doc_count: u64` LE (8 B) | Exact **live-doc** count per inline value (including `TAG_PRESENT = 0x00`; deleted at `0`) |
| `0x07 \|\| attr_id` BE \|\| `gen` BE \|\| `tag` \|\| `encoded_val` | `doc_count: u64` LE (8 B) | Exact **live-doc** count per detached value in `gen` |
| `0x10 \|\| seg` BE (5 B) | `RoaringBitmap<LocalId>` | Deleted `LocalId` tombstones in segment `seg` (absent = no deletions) |
| `0x11 \|\| seg` BE (5 B) | `SegmentMeta` (§4.7) | One per non-dropped segment (`Open`, `Sealed`, `Building`, `Forwarding`, `Retired`) |
| `0x12 \|\| seg` BE \|\| `attr_id` BE \|\| `tag` (8 B) | `ZoneEntry { min_len, max_len, min, max }` | Per-segment min/max of ordered scalars (widen-only on insert) |
| `0x13 \|\| attr_id` BE (3 B) | `DetachedMeta` (§4.8) | Metadata + generation pointers for one detached column |
| `0x14 \|\| attr_id` BE \|\| `gen` BE \|\| `seg` BE (11 B) | `RoaringBitmap<LocalId>` | Detached point-write *overwritten set* (`LocalId`s modified after bulk gen build) |

---

## 4. Value & Metadata Formats

### 4.1 `RowView` v3 (`cf_rows`)

Header (4 B): `[version: u8 = 0x03][flags: u8][count: u16 LE]`, followed by `count * 8` B directory sorted ascending by `attr_id`: `[attr_id: u16 LE][val_kind: u8][reserved: u8 = 0][payload_offset: u32 LE]`, then packed payloads (`val_kind` tags `0..=8`: `Null`, `Bool`, `Int`, `Uint`, `Float`, `String`, `StringArray`, `IntArray`, `UintArray`).

Differs from v2 `DocView`:
- `flags & 0x01` = `has_vector` (lets `Collection::get` skip `cf_vectors` reads when absent).
- Stores only `ColumnStorage::Inline` attributes (`Detached` attributes live exclusively in `cf_detached`).
- Owning `DocId` is not stored inline (resolved via `cf_row_doc_ids`).

### 4.2 `SegPosting` (`cf_attr_postings`, `cf_detached_postings`)

Compact set of `LocalId`s within one segment:

| Tag | Variant | Threshold | Payload |
|---:|---|---|---|
| `0x00` | `Sparse` | `len <= 32` | `[u32; len]` LE strictly ascending |
| `0x01` | `Roaring` | `len > 32` | `RoaringBitmap::serialize_into` |

Postings in sealed segments may include dead `LocalId`s from later tombstones; readers intersect with segment liveness (`SegmentTableSnapshot`), and `default[0x14]` overwritten sets for detached columns.

Order-preserving `type_tag` / `encoded_val` rules (`0x00 TAG_PRESENT`, `0x01 Bool`, `0x02 Int` sign-flipped BE, `0x03 Uint` BE, `0x04 Float` IEEE-ordered BE with `-0.0 -> +0.0`, `0x05 String` `0x00`-escaped UTF-8 terminated by `0x00 0x00`) are documented in [filtering-and-inverted-index.md](filtering-and-inverted-index.md#42-order-preserving-inverted-key-encoding).

### 4.3 `PackedLeafNode` v2 (`cf_leaf_entries`)

Slot-indexed RaBitQ leaf layout (`LEAF_HEADER_LEN = 24` B):
- Replaces v2 `doc_ids: [u64; padded_slots]` with `row_ids: [u64; padded_slots]` LE holding `RowId`s.
- Header bytes `16..20` store `stale_hint: u32 LE`; byte `20` stores `LEAF_FORMAT_VERSION = 3` and bytes `21..24` store the trailing `LeafAttrBlock` length (`u24 LE`).
- A slot is **effective** iff `live_mask[slot]` is set and `liveness.is_live(resolve(row_ids[slot]))` after resolving forwarding segments.
- `PackedInternalNode` headers carry `INTERNAL_FORMAT_VERSION = 2` at byte `12`; older collections fail clearly on open instead of being misread.

### 4.4 `ColumnBlock` (`cf_columns`, `cf_detached`)

One attribute for one 256-slot `RowBlock` (`block = local >> 8`):

- **Header (8 B):** `[version: u8 = 1][kind: u8][width: u8][flags: u8][present_count: u16 LE][reserved: u16]`.
  - `flags & 0x01`: all 256 slots present (`present` bitmap omitted).
  - `flags & 0x06`: integer frame-of-reference narrowing (`width` = `1`/`2`/`4`/`8` bytes + `base: i64` LE for narrowed `Int`/`Uint`).
- **`present`:** `[u64; 4]` LE (32 B, unless `all_present`).
- **Payload:** Fixed-width `[T; 256]` slot-indexed (scalars) or `offsets: [u32; 257]` LE + contiguous payloads (strings / arrays).

### 4.5 `TermRecord` v4 (`cf_fts_terms`) & `FtsChunk` v4 (`cf_fts_skips`)

- **`TermRecord` v4:** 21-byte header (`term_id: u32 LE`, `doc_freq: u64 LE` over **live** rows, `max_tf: u32 LE`, `min_dl_over_tf: f32 LE`, `flags: u8`). When `flags == 0x01` (single-block term with `doc_freq <= 8`), the inline payload appends `seg: u32 LE || block: u32 LE || v3 PackedPostingBlock` (256-slot TF-only posting block). A second distinct `(seg, block)` promotes the term to `cf_fts_skips`.
- **`FtsChunk` v4 (`cf_fts_skips`):** 12-byte header (`version_flags: u8 = 0x40 | optional 0x01 u16_offsets`, `skip_tf_width: u8`, `count_minus_1: u8`, `reserved: u8 = 0`, `chunk_max_tf: u32 LE`, `chunk_min_dl_over_tf: f32 LE`) followed by per-block `min_dl_over_tfs`, cumulative payload `block_ends` (`u16` or `u32`), `max_tfs`, `rel_block_ids`, and concatenated compact `TfEncoding` posting payloads (see [`blockmax-simd.md`](blockmax-simd.md)).

### 4.6 `Manifest` (`default[b"\x01manifest"]`, 32 B)

`[version: u8 = 1][has_open: u8][reserved: u16][next_segment_id: u32 LE][open_segment: u32 LE][seal_rows: u32 LE][reserved: u32][live_rows: u64 LE][total_rows: u64 LE]`.

### 4.7 `SegmentMeta` (`default[0x11 || seg BE]`)

`state: u8` (`0 Open`, `1 Sealed`, `2 Building`, `3 Forwarding`, `4 Retired`), `flags: u8` (`0x01` has `cf_columns`, `0x02` has vectors), `row_count: u32`, `deleted_count: u32`, `forward_target: u32`, `forward_base: u32`, `forwarded_at_epoch: u64`, `created_seq: u64`, `sealed_seq: u64`.

### 4.8 `DetachedMeta` (`default[0x13 || attr_id BE]`)

`[version: u8][state: u8][reserved: u16][current_gen: u32 LE][pending_gen: u32 LE][retired_below: u32 LE][built_against_watermark: u64 LE (RowId)]`.

### 4.9 Trailing `LeafAttrBlock` v2 (`cf_leaf_entries`) & `NodeSummary` v1 (`cf_node_summaries`)

Each `PackedLeafNode` appends a slot-aligned scalar block for every eligible inline filterable attribute (`Bool`, `Int`, `Uint`, `Float`; strings, arrays, and detached columns stay in their existing stores). Fetching the leaf for RaBitQ also fetches all filter values, so filtered traversal needs no second column-family read:

- **Header (7 B):** `[version: u8 = 2][slot_cap: u32 LE][attr_count: u16 LE]`.
- **Per column (8 B header):** `[attr_id: u16 LE][attr_type: u8][encoding: u8][payload_len: u32 LE]`, then `present: ceil(slot_cap / 64) * 8 B` followed by:
  - `Bool`: `ceil(slot_cap / 64) * 8 B` value mask.
  - `Int` / `Uint`: frame-of-reference `[base: u64 LE][width: u8][slot_cap * width B]` deltas (`width` in `0/1/2/4/8`).
  - `Float`: `[f64; slot_cap]` or `[f32; slot_cap]` when every present value round-trips losslessly (`encoding = 1`).

`cf_node_summaries[node_id]` stores a conservative rollup for every leaf and internal node:

- **Header (8 B):** `[version: u8 = 1][complete: u8][row_count: u32 LE][attr_count: u16 LE]`.
- **Per attribute (32 B):** `[attr_id: u16 LE][attr_type: u8][present_count: u32 LE][has_values: u8][min: u64 LE][max: u64 LE][bits: u64 LE]`.
- `min`/`max`/`bits` use the unsigned order-preserving encoding (Int sign-flip, Uint verbatim, Float IEEE total order with `-0.0 -> +0.0`). `bits` is a 64-bit hash bloom over values and may only produce false positives. Internal summaries are the union of child summaries; `complete = false` disables pruning for that subtree.

### 4.10 Bit-Sliced Index (`cf_bsi`)

Key: `seg (u32 BE) || attr_id (u16 BE) || chunk (u32 BE) || plane (u8)`, where `chunk = local / 65,536` and values are `[u64; 1024]` LE (8,192 B):

- `plane 0xFE`: frame-of-reference record `[base: u64 LE][width: u8]`.
- `plane 0xFF`: presence bitmap.
- `plane 0..width-1`: bit-slices of `value - base` using the same unsigned order-preserving encoding.

Range evaluation starts from the presence bitmap and applies wordwise `&`/`|`/`!` over only the stored planes, then intersects with segment liveness (`RowSet::into_live`). Segments without a complete BSI fall back to value-major postings.

---

## 5. Consistency Invariants (`Collection::verify_invariants`)

Verified exhaustively by `crates/izzydb/src/verify/mod.rs`:
1. **Schema & Manifest / Segment Table:** `Manifest.next_segment_id > max(seg)` across all `0x11` keys; at most one `Open` segment equal to `Manifest.open_segment`; zero dropped segment leaks into `0x11` or `0x10`; `0x10` deletion bitmaps bounded by `< row_count` with `|bitmap| == deleted_count`.
2. **Row / DocId Bijection:** For every non-dropped segment, `cf_rows` keys equal `0..row_count`; `cf_id_map` is an exact bijection onto live `(RowId, DocId)` entries mirrored in `cf_row_doc_ids`.
3. **ANN Tree (when `has_vectors`):** Every live row with `has_vector` has **exactly one** effective leaf entry across `cf_leaf_entries` (after resolving `Forwarding` segments); every non-effective entry points to a tombstoned/forwarded-dead/dropped row; `AnnMeta.num_vectors == live_vector_rows` and `AnnMeta.stale_rows == physical_entries - effective_entries`. (Vectorless collections have zero ANN keys.)
4. **Inline Attribute Indexes, Counts & Zones:** `cf_attr_postings` $\cap$ live equals exact keys re-extracted from live `cf_rows`; `0x06` counts equal live-doc counts; `0x12` zones enclose all live values; sealed `cf_columns` match live rows.
5. **Leaf-Colocated Scalars & Summaries (when `has_vectors`):** every live leaf slot's trailing `LeafAttrBlock` value equals the resolved row's `cf_rows` value for each eligible attribute; each leaf summary equals its `LeafAttrView` rollup; every internal summary equals the merge of its children (or is marked incomplete).
6. **Detached Columns:** `cf_detached`, `cf_detached_postings`, `0x07`, `0x14`, and `0x12` zones at `current_gen` match point-corrected live values; no retired generations leak outside compaction-filter bounds.
7. **FTS Parity:** `cf_fts_terms`, `cf_fts_skips` (`FtsChunk` v4) $\cap$ live, `cf_fts_positions`, `default[0x05]` doc lengths, and `FtsMeta` stats match a full re-analysis of live rows, with sound upper bounds; deprecated `cf_fts_postings` has zero visible keys.
