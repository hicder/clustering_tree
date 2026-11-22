# Key & Binary Encoding Specification

This document specifies the persistent RocksDB key-value encodings, packed node/vector/document blob formats, order-preserving attribute encodings, FTS block formats, and RaBitQ bit-packing layouts used by `Collection`, `ClusteringTree`, `AttributeIndex`, and `FtsIndex`.

Implementation sources:
- Column families, RocksDB keys, `DocLoc`, `FtsRevBlock`, and CF options: [`crates/izzydb-storage/src/storage.rs`](../../crates/izzydb-storage/src/storage.rs)
- Packed node/metadata serialization and zero-copy views: [`crates/izzydb-storage/src/node.rs`](../../crates/izzydb-storage/src/node.rs)
- Attribute key/value (`LeafPosting`), zone map, and v2 `cf_docs` (`DocView`) codecs: [`crates/izzydb-storage/src/codec.rs`](../../crates/izzydb-storage/src/codec.rs)
- FTS metadata, `TermRecord`, `SkipBlock`, `DocLenBlock`, v3 `PackedPostingBlock`, and `PackedPositionBlock`: [`crates/izzydb-fts/src/block.rs`](../../crates/izzydb-fts/src/block.rs)
- Metric tags, ID aliases, and schema/filter types: [`crates/izzydb-core/src/config.rs`](../../crates/izzydb-core/src/config.rs), [`crates/izzydb-core/src/schema.rs`](../../crates/izzydb-core/src/schema.rs)
- RaBitQ code and query bit-plane packing: [`crates/izzydb-rabitq/src/quantizer.rs`](../../crates/izzydb-rabitq/src/quantizer.rs)
- Structural and index invariants: [`crates/izzydb/src/verify.rs`](../../crates/izzydb/src/verify.rs)

*(For how the FTS SIMD Block-Max MAXSCORE engine consumes `TermRecord`, `SkipBlock`, `DocLenBlock`, and v3 `PackedPostingBlock`, see [`blockmax-simd.md`](blockmax-simd.md).)*

---

## 1. Identifiers and Endianness Conventions

| Type | Rust Definition | Width | Endianness in RocksDB Keys & `DocLoc` Values | Endianness Inside Packed Blobs (`TreeMeta`, `Packed*`, FTS Blocks) |
|---|---|---:|---|---|
| `DocId` | `u64` | 8 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `NodeId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `LocalId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | — (`local_id == slot` in `PackedLeafNode`) |
| `AttrId` | `u16` | 2 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `FtsDocId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`next_fts_doc_id` in `FtsMeta`) |
| `FtsBlockId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | — (derived as `fts_doc_id >> 8`) |
| `TermId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes` in `TermRecord` / `FtsMeta`) |
| `FtsSkipChunkId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | — (derived as `block_id >> 8`, 256 blocks per chunk) |
| `f32` scalars & vectors | `f32` | 4 bytes | — | **Little-endian** (`to_le_bytes`) |
| `u16`/`u32`/`u64` counts & scalars | `u16`/`u32`/`u64` | 2/4/8 bytes | — | **Little-endian** (`to_le_bytes`) |

> **Why big-endian for RocksDB keys?**
> `doc_key(id: u64)` (`id.to_be_bytes()`), `node_key(id: u32)`, `term_id`/`block_id`/`chunk_id` suffixes, and attribute index keys encode IDs and ordered scalars in big-endian so lexicographical byte order matches numeric/logical order for `batched_multi_get_cf(..., sorted_input = true)`, FTS skip/posting seeks, and cluster-first range scans.
>
> **Why little-endian inside packed blobs?**
> On little-endian hosts (`#[cfg(target_endian = "little")]`), aligned `doc_ids`, `codes`, `code_popcounts`, `centroid`, `rotated_centroid`, `res_norm_sqs`, and `scales` slices can be borrowed directly from RocksDB `DBPinnableSlice` buffers, and FTS SIMD kernels can load `DocLenBlock` / v3 `PackedPostingBlock` arrays directly into vector registers.

---

## 2. RocksDB Column Families & Key-Value Map

`Storage::open` opens eleven column families. The `default` family groups metadata and cached indexes by one-byte tags. Scans are bounded to a tag or, for document lengths, a tag and attribute ID. Databases created with the previous layout require a fresh database path.

| Column Family | Constant | Key Format | Value Format | Compression | Read Pattern |
|---|---|---|---|---|---|
| `default` (metadata) | `DEFAULT_COLUMN_FAMILY_NAME` | `b"\x01tree_meta"` / `b"\x01schema"` / `b"\x01fts_meta"` / `b"\x01fts_free_ids"` | `TreeMeta` (31 B) / `CollectionSchema` (v3) / `FtsMeta` / `RoaringBitmap` | Default | Point `get` |
| `default` (internal nodes) | `DEFAULT_COLUMN_FAMILY_NAME` | `0x02` \|\| `node_id` (`u32` BE, 5 B total) | `PackedInternalNode` | Default | Namespace load on `open()` |
| `cf_leaf_entries` | `CF_LEAF_ENTRIES` | `leaf_id` (`u32` BE, 4 B) | Slot-indexed `PackedLeafNode` (`live_mask`, `code_popcounts`, 8-wide tiled `codes`) | `None` | `batched_multi_get_cf` (pinned) |
| `cf_vectors` | `CF_VECTORS` | `doc_id` (`u64` BE, 8 B) | `[f32; dim]` LE (`dim * 4` B) | `None` | `get_pinned_cf` / `batched_multi_get_cf` |
| `cf_doc_loc` | `CF_DOC_LOC` | `doc_id` (`u64` BE, 8 B) | `leaf_id` (`u32` BE) \|\| `local_id` (`u32` BE) \|\| `fts_doc_id` (`u32` BE, `u32::MAX` = None) (12 B) | Default | Point `get_pinned_cf` / `batched_multi_get_cf` |
| `cf_docs` | `CF_DOCS` | `doc_id` (`u64` BE, 8 B) | v2 directory-indexed forward `Attributes` (`DocView`) | Default | Point pinned / `batched_multi_get_cf` |
| `cf_attr_clusters` | `CF_ATTR_CLUSTERS` | `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) \|\| `encoded_val` | `RoaringBitmap` of `cluster_id: u32` | Default | Point `multi_get_cf` / `AttrZoneMap` for ranges |
| `cf_attr_postings` | `CF_ATTR_POSTINGS` | `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) \|\| `cluster_id` (`u32` BE) \|\| `encoded_val` | `LeafPosting` (sparse `u16` / dense bitset / Roaring) | `None` | `batched_multi_get_cf` / 7-byte prefix range scan |
| `default` (attribute zones) | `DEFAULT_COLUMN_FAMILY_NAME` | `0x03` \|\| `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) (4 B total) | `AttrZoneMap` (`leaf_id -> (min_b, max_b)`) | Default | Loaded into `Storage::attr_zones_cache` on demand |
| `default` (FTS reverse-map blocks) | `DEFAULT_COLUMN_FAMILY_NAME` | `0x04` \|\| `block_id` (`u32` BE, 5 B total) | `FtsRevBlock` (`present`, `doc_ids[256]`, `leaf_ids[256]`, `local_ids[256]`, 4,128 B) | Default | Cached in `Storage::fts_rev_map_cache` |
| `cf_fts_terms` | `CF_FTS_TERMS` | `attr_id` (`u16` BE) \|\| `token_utf8` | `TermRecord` (`term_id` + `TermStats` + optional inline v3 posting block) | Default | Sorted `batched_multi_get_cf` / 2-byte prefix scan |
| `cf_fts_skips` | `CF_FTS_SKIPS` | `term_id` (`u32` BE) \|\| `chunk_id` (`u32` BE) (8 B) | `SkipBlock` (up to 256 blocks' bounds) | `None` | Batched `chunk 0` pin / 4-byte prefix `RocksSkipCursor` |
| `cf_fts_postings` | `CF_FTS_POSTINGS` | `term_id` (`u32` BE) \|\| `block_id` (`u32` BE) (8 B) | v3 `PackedPostingBlock` (TF-only, aligned `tfs`) | `None` | Lazy sequential 4-byte prefix iterator / prefix visit |
| `cf_fts_positions` | `CF_FTS_POSITIONS` | `term_id` (`u32` BE) \|\| `block_id` (`u32` BE) (8 B) | `PackedPositionBlock` (`u16` or `u32` offsets + LEB128 deltas) | Default | Batched `multi_get_fts_positions_pinned` (`ContainsTokenSequence` only) |
| `default` (FTS document lengths) | `DEFAULT_COLUMN_FAMILY_NAME` | `0x05` \|\| `attr_id` (`u16` BE) \|\| `block_id` (`u32` BE) (7 B total) | `DocLenBlock` (260 / 516 / 1,028 B) | Default | Lazy per-`attr_id` load into `fts_doc_lens_cache` |

### RocksDB Tuning

- Shared 128 MiB LRU block cache (`BLOCK_CACHE_BYTES = 128 * 1024 * 1024`) across all column families.
- Shared 128 MiB write-buffer threshold across column families.
- `default`, `cf_leaf_entries`, `cf_vectors`, `cf_doc_loc`, and `cf_docs` use `point_lookup_cf_opts` (10.0 bits/key full-key Bloom filter, `set_memtable_whole_key_filtering(true)`, cached & pinned L0 filter/index blocks).
- `cf_attr_clusters` uses a 3-byte fixed prefix extractor (`attr_id (2 B) || type_tag (1 B)`), 10.0 bits/key Bloom filter with `set_whole_key_filtering(true)`.
- `cf_attr_postings` uses a **7-byte** fixed prefix extractor (`attr_id (2 B) || type_tag (1 B) || cluster_id (4 B)`), `set_whole_key_filtering(true)`, and `set_prefix_same_as_start(true)` on per-leaf range scans.
- `cf_fts_terms` uses a 2-byte fixed prefix extractor (`attr_id (2 B)`), `set_whole_key_filtering(true)`, and `set_prefix_same_as_start(true)` on `last_as_prefix` term scans.
- `cf_fts_skips`, `cf_fts_postings`, and `cf_fts_positions` use a **4-byte** fixed prefix extractor (`term_id (4 B)`), 10.0 bits/key Bloom filter with `set_whole_key_filtering(true)`.
- `DBCompressionType::None` on `cf_leaf_entries`, `cf_vectors`, `cf_attr_postings`, `cf_fts_skips`, and `cf_fts_postings` for zero-copy `DBPinnableSlice` / raw iterator access on hot query paths.

---

## 3. Binary Value Specifications

### 3.1 `TreeMeta` (`default`, key `b"\x01tree_meta"`)

Fixed size: **31 bytes** (`META_LEN = 31`). Any other byte length fails decoding with `TreeError::CorruptedData`.

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 4 | `root_id` | `u32` LE | Must satisfy `root_id < next_node_id` |
| `4` | 1 | `root_level` | `u8` | `0` when root is a leaf; `> 0` when root is internal |
| `5` | 4 | `next_node_id` | `u32` LE | Next unallocated `NodeId` |
| `9` | 4 | `dim` | `u32` LE | Vector dimensionality (`> 0`) |
| `13` | 1 | `metric` | `u8` | `0` = `L2Squared`, `1` = `Cosine`, `2` = `DotProduct` |
| `14` | 8 | `num_vectors` | `u64` LE | Total indexed documents across all leaves |
| `22` | 8 | `rabitq_seed` | `u64` LE | Seed for deterministic FWHT sign-flip rotation $P^{-1}$ |
| `30` | 1 | `rabitq_bits` | `u8` | Must be `1` (1-bit RaBitQ); other values rejected on `open()` |

### 3.2 `CollectionSchema` (`default`, key `b"\x01schema"`)

Header: `version: u8` (`3`), `next_attr_id: u16` LE, `count: u32` LE (7 bytes), followed by `count` entries:

- `id: u16` LE
- `attr_type: u8` (`1=String`, `2=Int`, `3=Uint`, `4=Float`, `5=Bool`, `6=StringArray`, `7=IntArray`, `8=UintArray`)
- `filterable: u8` (`0`/`1`)
- `has_fts: u8` (`0`/`1`)
- If `has_fts == 1`, `FullTextSearchConfig` (21 B):
  `tokenizer: u8` (`1=WordV4`, `2=Whitespace`, `3=PreTokenizedArray`, `4=SmartCn`, `5=Kuromoji`, `6=Nori`), `language: u8` (`1..=18`), `case_sensitive: u8`, `stemming: u8`, `remove_stopwords: u8`, `ascii_folding: u8`, `max_token_length: u16` LE, `k1: f32` LE, `b: f32` LE, `k3: f32` LE, `index_positions: u8`
- `name_len: u32` LE \|\| `name_utf8`

### 3.2b `FtsMeta` & `fts_free_ids` (`default`, keys `b"\x01fts_meta"`, `b"\x01fts_free_ids"`)

- **`b"\x01fts_meta"`:** `next_fts_doc_id: u32` LE, `next_term_id: u32` LE, `count: u32` LE (12-byte header), followed by `count` entries of `(attr_id: u16 LE, doc_count: u64 LE, total_tokens: u64 LE)` (18 B each). `FtsDocId::MAX` (`u32::MAX`) is reserved as the absent sentinel in `cf_doc_loc`.
- **`b"\x01fts_free_ids"`:** Optional serialized `RoaringBitmap` of recycled `FtsDocId: u32` values strictly `< next_fts_doc_id` (key deleted when empty).

### 3.3 `cf_doc_loc` Value (`DocLoc`)

Fixed size: **12 bytes**: `leaf_id` (`u32` BE, `0..4`) \|\| `local_id` (`u32` BE, `4..8`) \|\| `fts_doc_id` (`u32` BE, `8..12`; `u32::MAX` when absent).

### 3.4 `PackedInternalNode` (internal nodes in `default`)

Variable size: **`24 + 12 * n + (1 + n) * dim * 4` bytes**, where `n` is `entry_count`.

#### Header (`INTERNAL_HEADER_LEN = 24` bytes)

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 1 | `level` | `u8` | Tree level (`>= 1`; leaves are level `0`) |
| `1` | 1 | `has_parent` | `u8` | `0` = `None` (root), `1` = `Some(parent_id)` |
| `2` | 2 | reserved | `[u8; 2]` | Written as `0` |
| `4` | 4 | `n` | `u32` LE | Number of child entries |
| `8` | 4 | `parent_id` | `u32` LE | Parent `NodeId` when `has_parent == 1`; `0` otherwise |
| `12` | 4 | reserved | `[u8; 4]` | Keeps 8-byte alignment for `subtree_count` and body |
| `16` | 8 | `subtree_count` | `u64` LE | Sum of `child_subtree_counts` |

#### Body (starting at byte `24`)

| Offset | Size | Field | Type | Ordering / Layout |
|---|---|---|---|---|
| `24` | `n * 8` | `child_subtree_counts` | `[u64; n]` LE | Parallel to `child_ids` |
| `24 + 8n` | `n * 4` | `child_ids` | `[u32; n]` LE | Strictly sorted ascending by `NodeId` |
| `24 + 12n` | `dim * 4` | `centroid` | `[f32; dim]` LE | Weighted mean of child centroids |
| `24 + 12n + 4*dim` | `n * dim * 4` | `child_centroids` | `[f32; n * dim]` LE | Row-major; child `i` occupies `[i * dim .. (i + 1) * dim]` |

### 3.5 Slot-Indexed `PackedLeafNode` (`cf_leaf_entries`)

Derived parameters:
- `padded_dim = dim.next_power_of_two().max(64)`
- `rabitq_words = padded_dim / 64`
- `mask_words = slot_cap.div_ceil(64)`
- `padded_slots = slot_cap.next_multiple_of(8)` (allows unmasked 8-wide SIMD tail loops without out-of-bounds loads)
- `pop_width = if padded_dim > u16::MAX as usize { 4 } else { 2 }` (`popcount_u32` header flag)

Every entry's stable `LocalId` equals its slot index (`local_id == slot`), eliminating any separate `local_ids` array.

#### Header (`LEAF_HEADER_LEN = 24` bytes)

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 1 | `has_parent` | `u8` | `0` = `None` (root leaf), `1` = `Some(parent_id)` |
| `1` | 1 | `popcount_u32` | `u8` | `0` = `u16` per slot, `1` = `u32` per slot |
| `2` | 2 | `mask_words` | `u16` LE | Number of `u64` words in `live_mask` (`slot_cap.div_ceil(64)`) |
| `4` | 4 | `slot_cap` | `u32` LE | High-water slot count (trailing dead slots trimmed on delete) |
| `8` | 4 | `parent_id` | `u32` LE | Parent `NodeId` when `has_parent == 1`; `0` otherwise |
| `12` | 4 | `live_count` | `u32` LE | Number of set bits in `live_mask` (`<= slot_cap`) |
| `16` | 8 | reserved | `u64` | Written as `0` |

#### Body (starting at byte `24`)

| Section | Size (bytes) | Type | Layout / Semantics |
|---|---:|---|---|
| `live_mask` | `mask_words * 8` | `[u64; mask_words]` LE | Bit `slot` is `1` iff slot is occupied |
| `doc_ids` | `padded_slots * 8` | `[u64; padded_slots]` LE | Indexed by `slot`; `0` for dead/padding slots |
| `codes` | `padded_slots * rabitq_words * 8` | `[u64; padded_slots * rabitq_words]` LE | **8-entry SIMD-tiled** layout: `((slot / 8) * rabitq_words + w) * 8 + (slot % 8)` |
| `res_norm_sqs` | `padded_slots * 4` | `[f32; padded_slots]` LE | Squared residual norm $\|x_i - c\|_2^2$ |
| `scales` | `padded_slots * 4` | `[f32; padded_slots]` LE | $\|x_i - c\|_2 / \langle \bar{o}_i, \tilde{o}_i \rangle$, or `0.0` when `res_norm_sq == 0.0` |
| `centroid` | `dim * 4` | `[f32; dim]` LE | Leaf centroid $c$ |
| `rotated_centroid` | `padded_dim * 4` | `[f32; padded_dim]` LE | Precomputed $P^{-1} c$ |
| `code_popcounts` | `padded_slots * pop_width` | `[u16 | u32; padded_slots]` LE | Precomputed $\sum_w \text{popcnt}(\text{code}[w])$ per slot |

### 3.6 Global Vectors (`cf_vectors`)

- **Key:** `doc_key(doc_id)` (`doc_id.to_be_bytes()`, 8 bytes).
- **Value:** raw little-endian `[f32; dim]` (`dim * 4` bytes, no header). L2-normalized at insert when `metric == Cosine`.

### 3.7 v2 Forward Document Attributes (`cf_docs`, `DocView`)

- **Key:** `doc_key(doc_id)` (`doc_id.to_be_bytes()`, 8 bytes).
- **Value Layout:**
  - Header (4 B): `[version: u8 = 0x02][reserved: u8 = 0][count: u16 LE]`
  - Directory (`count * 8` B, sorted ascending by `attr_id`): `[attr_id: u16 LE][val_kind: u8][reserved: u8 = 0][payload_offset: u32 LE]`
  - Payloads (packed starting at `4 + count * 8`):

| `val_kind` | Rust Variant | Payload Encoding |
|---:|---|---|
| `0` | `Null` | Empty (0 B) |
| `1` | `Bool` | `0x00` (`false`) / `0x01` (`true`) (1 B) |
| `2` | `Int` | `i64` LE (8 B) |
| `3` | `Uint` | `u64` LE (8 B) |
| `4` | `Float` | `f64` LE (8 B; `NaN` rejected) |
| `5` | `String` | Raw `utf8_bytes` (byte length derived from next directory `payload_offset` or end-of-buffer) |
| `6` | `StringArray` | `arr_len: u32` LE \|\| `arr_len` × (`len: u32` LE \|\| `utf8_bytes`) |
| `7` | `IntArray` | `arr_len: u32` LE \|\| `[i64; arr_len]` LE |
| `8` | `UintArray` | `arr_len: u32` LE \|\| `[u64; arr_len]` LE |

`DocView` binary-searches the 8-byte directory on `attr_id` and evaluates non-FTS filter predicates directly on borrowed slices without allocating an `Attributes` `BTreeMap`.

### 3.8 Order-Preserving Attribute Index Keys, `LeafPosting`, and attribute zones in `default`

- **`cf_attr_clusters` key:** `attr_id (u16 BE) || type_tag (u8) || encoded_val` (value: `RoaringBitmap<NodeId>`)
- **`cf_attr_postings` key (cluster-first):** `attr_id (u16 BE) || type_tag (u8) || cluster_id (u32 BE) || encoded_val`
- **`default` attribute-zone key:** `0x03 || attr_id (u16 BE) || type_tag (u8)` (4 B)

| `type_tag` | Meaning | `encoded_val` |
|---:|---|---|
| `0x00` | `TAG_PRESENT` | Empty (0 B) — marks non-null presence for `IsNull` / `IsNotNull` |
| `0x01` | `Bool` | 1 byte (`0x00` = `false`, `0x01` = `true`) |
| `0x02` | `Int(i64)` | 8 bytes BE with sign bit flipped: `(v as u64) ^ (1u64 << 63)` |
| `0x03` | `Uint(u64)` | 8 bytes BE: `v.to_be_bytes()` |
| `0x04` | `Float(f64)` | 8 bytes BE IEEE-754 order-preserving (`-0.0 -> +0.0`, `NaN` rejected; negative bits inverted, non-negative sign bit flipped) |
| `0x05` | `String` | Escaped-null-terminated UTF-8 (`0x00 -> 0x00 0xFF`, terminated with `0x00 0x00`) |

Array attributes (`StringArray`, `IntArray`, `UintArray`) index each distinct element under their element's scalar `type_tag`.

#### `LeafPosting` (`cf_attr_postings` value)

| Tag | Variant | Threshold | Payload |
|---:|---|---|---|
| `0x00` | `Sparse` | `len <= 16` (`max_id <= u16::MAX`) | `[slot_0: u16 LE, ..., slot_{k-1}: u16 LE]` |
| `0x01` | `Dense` | `len > 16` and `max_id <= 4095` | `num_words: u16 LE || [u64; num_words] LE` |
| `0x02` | `Roaring` | Fallback | `RoaringBitmap::serialize_into` bytes |

#### `AttrZoneMap` (attribute zones in `default` value, scalar ordered types only)

- `leaf_count: u32 LE` followed by `leaf_count` entries of `(leaf_id: u32 LE, min_len: u16 LE, max_len: u16 LE, min_bytes, max_bytes)` sorted by `leaf_id` (storing each leaf's minimum and maximum order-preserving `encoded_val`).

### 3.9 FTS Keys, `TermRecord`, `SkipBlock`, `FtsRevBlock`, `DocLenBlock`, Posting & Position Blocks

Each document in an FTS-enabled collection is assigned a dense `FtsDocId: u32` (`< FtsDocId::MAX`), partitioned into 256-document blocks (`FTS_BLOCK_SIZE = 256`):

- `block_id: FtsBlockId = fts_doc_id >> 8` (`u32`)
- `slot: u8 = (fts_doc_id & 0xFF) as u8` (`0..=255`)

#### FTS Key Formats

- **`default` FTS reverse-map key:** `0x04 || block_id (u32 BE)` (5 B) $\to$ fixed **4,128-byte** `FtsRevBlock`: `present: [u64; 4]` LE (32 B) \|\| `doc_ids: [u64; 256]` LE \|\| `leaf_ids: [u32; 256]` LE \|\| `local_ids: [u32; 256]` LE.
- **`cf_fts_terms`:** `attr_id` (`u16` BE, 2 B) \|\| `token_utf8`.
- **`default` FTS document-length key:** `0x05 || attr_id (u16 BE) || block_id (u32 BE)` (7 B).
- **`cf_fts_skips`:** `term_id` (`u32` BE, 4 B) \|\| `chunk_id` (`u32` BE, 4 B) (`chunk_id = block_id >> 8`, 8 B).
- **`cf_fts_postings` & `cf_fts_positions`:** `term_id` (`u32` BE, 4 B) \|\| `block_id` (`u32` BE, 4 B) (8 B).

#### Shared `IntWidth` Tag (`DocLenBlock`, `SkipBlock` `max_tf`, & v3 `PackedPostingBlock`)

| `IntWidth` Tag | Variant | Element Width | Max Value |
|---:|---|---:|---:|
| `0` | `IntWidth::U8` | 1 byte | `255` |
| `1` | `IntWidth::U16` | 2 bytes LE | `65,535` |
| `2` | `IntWidth::U32` | 4 bytes LE | `4,294,967,295` |

#### `TermRecord` (`cf_fts_terms`)

Header: `TERM_RECORD_HEADER_LEN = 21` bytes:

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 4 | `term_id` | `u32` LE | Unique monotonic `TermId` (`< FtsMeta.next_term_id`) |
| `4` | 8 | `stats.doc_freq` | `u64` LE | Number of documents containing this term |
| `12` | 4 | `stats.max_tf` | `u32` LE | Maximum `tf` across all postings of this term |
| `16` | 4 | `stats.min_dl_over_tf` | `f32` LE | Sound lower bound on `dl / tf` (`>= 0.0`, non-NaN; `f32::INFINITY` when empty) |
| `20` | 1 | `flags` | `u8` | `0x00` = out-of-line in `cf_fts_postings`/`cf_fts_skips`; `0x01` = single-block (`doc_freq <= 8`) posting block appended inline (`block_id: u32 LE \|\| v3 PackedPostingBlock`) |

#### `SkipBlock` (`cf_fts_skips`, up to 256 posting blocks per chunk)

Stored when a term has `flags == 0x00` (`out-of-line`):

- **Header (4 B):** `width: u8` (`IntWidth` for `max_tf`), `reserved: u8` (`0`), `count: u16` LE (`1..=256`).
- **Body (`count * (4 + width + 1)` B):** `min_dl_over_tfs: [f32; count]` LE \|\| `max_tfs: [u8 | u16 | u32; count]` LE \|\| `rel_block_ids: [u8; count]` (strictly sorted ascending `block_id & 0xFF`).

#### `DocLenBlock` (FTS document lengths in `default`, 260 / 516 / 1,028 bytes)

Stored once per `(attr_id, block_id)` whenever at least one slot in the block has `dl > 0`:

- **Header (4 B):** `width: u8` (`IntWidth`), `reserved: u8` (`0`), `count: u16` LE (`1..=256`, non-zero slot count).
- **Body (`256 * width` B):** `dls: [u8 | u16 | u32; 256]` LE, slot-indexed `0..=255` (`0` for absent/deleted slots).

#### v3 `PackedPostingBlock` (TF-only, up to 256 docs per block)

Stored inline in `TermRecord` when single-block and `doc_freq <= 8`, or out-of-line in `cf_fts_postings`:

- **Header (12 B):** `count: u16` LE (`1..=256`), `flags: u16` LE, `max_tf: u32` LE, `min_dl_over_tf: f32` LE.
  - `flags` bits `8..15` (`0xFF00`): format version, must equal `0x0300` (`POSTING_VERSION_V3`).
  - `flags` bits `1..2` (`0x0006`): `(flags & 0x0006) >> 1` encodes `IntWidth` (`0 = U8`, `1 = U16`, `2 = U32`).
  - `flags` bit `0` (`0x0001`, `FLAG_DENSE`): `0` = Sparse (`count < 96`), `1` = Dense (`count >= 96`, `POSTING_DENSE_THRESHOLD = 96`).
  - All other bits in `flags` must be `0`.
- **Sparse Payload (`FLAG_DENSE == 0`, `12 + count * (width + 1)` B):**
  - `tfs: [u8 | u16 | u32; count]` LE (stored immediately after the 12-byte header so multi-byte `tf`s stay 4-byte aligned) \|\| `slots: [u8; count]` (strictly sorted ascending, parallel to `tfs`).
- **Dense Payload (`FLAG_DENSE == 1`, `44 + 256 * width` B):**
  - `bitmap: [u64; 4]` LE (32 B, 256-bit presence mask with popcount == `count`) \|\| `tfs: [u8 | u16 | u32; 256]` LE (slot-indexed `0..=255`; `0` if absent).

#### `PackedPositionBlock` (`cf_fts_positions`, read only for `ContainsTokenSequence`)

- `[flags: u8][count_minus_1: u8][slots: [u8; count]][pos_offsets: [u16 | u32; count + 1] LE][delta_varints: [u8; ...]]`
- `flags & 0x01 == 0` uses 2-byte `u16` offsets whenever `delta_varints.len() <= u16::MAX` (65,535 B), and `0x01` selects `u32` offsets otherwise.
- `slots` are strictly sorted ascending (`count = count_minus_1 + 1`); `pos_offsets[i]..pos_offsets[i + 1]` slices the unsigned 7-bit LEB128 varint-encoded position deltas for `slots[i]` (`last_off == delta_varints.len()`).

---

## 4. Consistency Invariants

Verified exhaustively by `verify_invariants` and `verify_collection_invariants` ([`crates/izzydb/src/verify.rs`](../../crates/izzydb/src/verify.rs)):

1. **Node keyspaces & cache parity:** Reachable internal nodes match `internal_cache` and the internal-node records in `default`; reachable leaves match `leaf_parent` and `cf_leaf_entries` with no orphans.
2. **Slot-indexed leaf invariants:** Every leaf satisfies `slot_cap <= config.max_leaf_size`, `live_mask` popcount equals `live_count`, and every live slot satisfies vector validity (`cf_vectors`) and non-negative finite `res_norm_sqs` / `scales`.
3. **Vector, doc, and `DocLoc` completeness:** `cf_vectors`, `cf_docs`, and `cf_doc_loc` are exact 1-to-1 bijections with reachable leaf `(doc_id, leaf_id, local_id == slot)` entries and `TreeMeta.num_vectors`.
4. **Attribute postings, zone maps & cluster summary exactness:** Every key in `cf_attr_postings` (`LeafPosting`) is non-empty and matches the exact set of slots reconstructed from `cf_docs` + `cf_doc_loc`; `cf_attr_clusters[k]` equals the exact `RoaringBitmap` of non-empty leaves; attribute zones in `default` match the exact `(min_val, max_val)` per leaf for ordered scalar types.
5. **FTS index parity:** `cf_doc_loc` `fts_doc_id`s and FTS reverse-map blocks in `default` (`FtsRevBlock`) form a 1-to-1 bijection with `cf_docs` and `(doc_id, leaf_id, local_id)` when FTS attributes exist; `default[b"\x01fts_free_ids"]` is strictly disjoint from live `FtsDocId`s, bounded below `next_fts_doc_id`, and trimmed at the tail; `default[b"\x01fts_meta"]` per-attribute `(doc_count, total_tokens)`, `cf_fts_terms` (`TermRecord` with unique `term_id < next_term_id`, valid inline terms `df <= 8` on 1 block, `TermStats` sound globally and exact for `<= 64`-block terms), `cf_fts_skips`, FTS document lengths in `default`, `cf_fts_postings`, and `cf_fts_positions` match a full re-analysis of `cf_docs`.
6. **Atomic updates:** All mutations across the column families commit in a single `rocksdb::WriteBatch` before updating in-memory caches (`internal_cache`, `leaf_parent`, `attr_zones_cache`, `fts_rev_map_cache`, and `fts_doc_lens_cache`).
