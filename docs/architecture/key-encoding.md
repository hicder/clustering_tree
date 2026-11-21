# Key & Binary Encoding Specification

This document specifies the persistent RocksDB key-value encodings, packed node/vector/document blob formats, order-preserving attribute encodings, FTS block formats, and RaBitQ bit-packing layouts used by `Collection`, `ClusteringTree`, `AttributeIndex`, and `FtsIndex`.

Implementation sources:
- Column families, RocksDB keys, and CF options: [`crates/izzydb-storage/src/storage.rs`](../../crates/izzydb-storage/src/storage.rs)
- Packed node/metadata serialization and zero-copy views: [`crates/izzydb-storage/src/node.rs`](../../crates/izzydb-storage/src/node.rs)
- Attribute key/forward codecs and `RoaringBitmap` serialization: [`crates/izzydb-storage/src/codec.rs`](../../crates/izzydb-storage/src/codec.rs)
- FTS metadata, `TermStats`, `DocLenBlock`, v2 `PackedPostingBlock`, and `PackedPositionBlock`: [`crates/izzydb-fts/src/block.rs`](../../crates/izzydb-fts/src/block.rs)
- Metric tags, ID aliases, and schema/filter types: [`crates/izzydb-core/src/config.rs`](../../crates/izzydb-core/src/config.rs), [`crates/izzydb-core/src/schema.rs`](../../crates/izzydb-core/src/schema.rs)
- RaBitQ code and query bit-plane packing: [`crates/izzydb-rabitq/src/quantizer.rs`](../../crates/izzydb-rabitq/src/quantizer.rs)
- Structural and index invariants: [`crates/izzydb/src/verify.rs`](../../crates/izzydb/src/verify.rs)

*(For how the FTS SIMD Block-Max MAXSCORE engine consumes `DocLenBlock` and v2 `PackedPostingBlock`, see [`blockmax-simd.md`](blockmax-simd.md).)*

---

## 1. Identifiers and Endianness Conventions

| Type | Rust Definition | Width | Endianness in RocksDB Keys & Map/Loc Values | Endianness Inside Packed Blobs (`TreeMeta`, `Packed*`, FTS Blocks) |
|---|---|---:|---|---|
| `DocId` | `u64` | 8 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `NodeId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `LocalId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `AttrId` | `u16` | 2 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `FtsDocId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`next_fts_doc_id` in `FtsMeta`) |
| `FtsBlockId` | `u32` | 4 bytes | **Big-endian** (`to_be_bytes`) | — (derived as `fts_doc_id >> 8`) |
| `f32` scalars & vectors | `f32` | 4 bytes | — | **Little-endian** (`to_le_bytes`) |
| `u16`/`u32`/`u64` counts & scalars | `u16`/`u32`/`u64` | 2/4/8 bytes | — | **Little-endian** (`to_le_bytes`) |

> **Why big-endian for RocksDB keys?**
> `doc_key(id: u64)` (`id.to_be_bytes()`), `node_key(id: u32)` (`id.to_be_bytes()`), `fts_doc_key(id: u32)`, `block_id` suffixes, and attribute index keys encode IDs and ordered scalars in big-endian so lexicographical byte order matches numeric/logical order for `batched_multi_get_cf(..., sorted_input = true)`, FTS block cursor seeks, and range scans.
>
> **Why little-endian inside packed blobs?**
> On little-endian hosts (`#[cfg(target_endian = "little")]`), aligned `doc_ids`, `codes`, `local_ids`, `centroid`, `rotated_centroid`, `res_norm_sqs`, and `scales` slices can be borrowed directly from RocksDB `DBPinnableSlice` buffers via `bytemuck::try_cast_slice` without copying or byte-swapping, and FTS SIMD kernels can load `DocLenBlock` / `PackedPostingBlock` arrays directly into vector registers.

---

## 2. RocksDB Column Families & Key-Value Map

`Storage::open` opens fifteen column families (fourteen custom CFs plus RocksDB's required `default` CF, which is unused):

| Column Family | Constant | Key Format | Value Format | Compression | Read Pattern |
|---|---|---|---|---|---|
| `default` | `DEFAULT_COLUMN_FAMILY_NAME` | — | — | Default | Unused |
| `cf_meta` | `CF_META` | `b"tree_meta"` / `b"schema"` / `b"fts_meta"` / `b"fts_free_ids"` | `TreeMeta` (31 B) / `CollectionSchema` (v3) / `FtsMeta` / `RoaringBitmap` | Default | Point `get_cf` |
| `cf_internal_nodes` | `CF_INTERNAL_NODES` | `node_id` (`u32` BE, 4 B) | `PackedInternalNode` | Default | Full CF load on `open()` |
| `cf_leaf_entries` | `CF_LEAF_ENTRIES` | `leaf_id` (`u32` BE, 4 B) | `PackedLeafNode` (with `local_ids`) | `None` | `batched_multi_get_cf` (pinned) |
| `cf_vectors` | `CF_VECTORS` | `doc_id` (`u64` BE, 8 B) | `[f32; dim]` LE (`dim * 4` B) | `None` | `get_pinned_cf` / `batched_multi_get_cf` |
| `cf_doc_loc` | `CF_DOC_LOC` | `doc_id` (`u64` BE, 8 B) | `leaf_id` (`u32` BE) \|\| `local_id` (`u32` BE) (8 B) | Default | Point `get_pinned_cf` / `batched_multi_get_cf` |
| `cf_docs` | `CF_DOCS` | `doc_id` (`u64` BE, 8 B) | Forward binary-encoded `Attributes` | Default | Point / `batched_multi_get_cf` |
| `cf_attr_clusters` | `CF_ATTR_CLUSTERS` | `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) \|\| `encoded_val` | `RoaringBitmap` of `cluster_id: u32` | Default | Point `multi_get_cf` / 3-byte prefix range scan |
| `cf_attr_postings` | `CF_ATTR_POSTINGS` | `cluster_key` \|\| `cluster_id` (`u32` BE, 4 B) | `RoaringBitmap` of `local_id: u32` | Default | `batched_multi_get_cf` |
| `cf_fts_doc_map` | `CF_FTS_DOC_MAP` | `doc_id` (`u64` BE, 8 B) | `fts_doc_id` (`u32` BE, 4 B) | Default | Point / `batched_multi_get_cf` |
| `cf_fts_rev_doc_map` | `CF_FTS_REV_DOC_MAP` | `fts_doc_id` (`u32` BE, 4 B) | `doc_id` (`u64` BE, 8 B) | Default | Point / `batched_multi_get_cf` |
| `cf_fts_terms` | `CF_FTS_TERMS` | `attr_id` (`u16` BE) \|\| `token_utf8` | `TermStats` (16 B LE) | Default | `batched_multi_get_cf` / 2-byte prefix scan |
| `cf_fts_postings` | `CF_FTS_POSTINGS` | `attr_id` (`u16` BE) \|\| `escaped_token` \|\| `block_id` (`u32` BE) | v2 `PackedPostingBlock` (TF-only) | Default | `RocksPostingCursor` prefix seek / point `get_cf` |
| `cf_fts_positions` | `CF_FTS_POSITIONS` | `attr_id` (`u16` BE) \|\| `escaped_token` \|\| `block_id` (`u32` BE) | `PackedPositionBlock` | Default | Point `get_cf` (`ContainsTokenSequence` only) |
| `cf_fts_doc_lens` | `CF_FTS_DOC_LENS` | `attr_id` (`u16` BE) \|\| `block_id` (`u32` BE) (6 B) | `DocLenBlock` (260 / 516 / 1,028 B) | `None` | Lazy per-`attr_id` load into `fts_doc_lens_cache` / point `get_cf` |

### RocksDB Tuning

- Shared 128 MiB LRU block cache (`BLOCK_CACHE_BYTES = 128 * 1024 * 1024`) across all point-lookup, attribute, and FTS CFs.
- `cf_internal_nodes`, `cf_leaf_entries`, `cf_vectors`, `cf_doc_loc`, `cf_docs`, `cf_attr_postings`, `cf_fts_doc_map`, `cf_fts_rev_doc_map`, `cf_fts_postings`, `cf_fts_positions`, and `cf_fts_doc_lens` use `point_lookup_cf_opts` (10.0 bits/key full-key Bloom filter, `set_memtable_whole_key_filtering(true)`, cached & pinned L0 filter/index blocks).
- `cf_attr_clusters` uses `attr_cluster_cf_opts` with a 3-byte fixed prefix extractor (`attr_id (2 B) || type_tag (1 B)`), 10.0 bits/key Bloom filter with `set_whole_key_filtering(true)`, and `set_prefix_same_as_start(true)` on range scans.
- `cf_fts_terms` uses `fts_terms_cf_opts` with a 2-byte fixed prefix extractor (`attr_id (2 B)`), 10.0 bits/key Bloom filter with `set_whole_key_filtering(true)`, and `set_prefix_same_as_start(true)` on `last_as_prefix` term scans.
- `DBCompressionType::None` on `cf_leaf_entries`, `cf_vectors`, and `cf_fts_doc_lens` to keep pinned `DBPinnableSlice` reads zero-copy on hot query paths.

---

## 3. Binary Value Specifications

### 3.1 `TreeMeta` (`cf_meta`, key `b"tree_meta"`)

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

### 3.2 `CollectionSchema` (`cf_meta`, key `b"schema"`)

Header: `version: u8` (`3`), `next_attr_id: u16` LE, `count: u32` LE (7 bytes), followed by `count` entries:

- `id: u16` LE
- `attr_type: u8` (`1=String`, `2=Int`, `3=Uint`, `4=Float`, `5=Bool`, `6=StringArray`, `7=IntArray`, `8=UintArray`)
- `filterable: u8` (`0`/`1`)
- `has_fts: u8` (`0`/`1`)
- If `has_fts == 1`, `FullTextSearchConfig` (21 B):
  `tokenizer: u8` (`1=WordV4`, `2=Whitespace`, `3=PreTokenizedArray`, `4=SmartCn`, `5=Kuromoji`, `6=Nori`), `language: u8` (`1..=18`), `case_sensitive: u8`, `stemming: u8`, `remove_stopwords: u8`, `ascii_folding: u8`, `max_token_length: u16` LE, `k1: f32` LE, `b: f32` LE, `k3: f32` LE, `index_positions: u8`
- `name_len: u32` LE || `name_utf8`

### 3.2b `FtsMeta` & `fts_free_ids` (`cf_meta`, keys `b"fts_meta"`, `b"fts_free_ids"`)

- **`b"fts_meta"`:** `next_fts_doc_id: u32` LE, `count: u32` LE, followed by `count` entries of `(attr_id: u16 LE, doc_count: u64 LE, total_tokens: u64 LE)` (18 B each).
- **`b"fts_free_ids"`:** Optional serialized `RoaringBitmap` of recycled `FtsDocId: u32` values strictly `< next_fts_doc_id` (key deleted when empty).

### 3.3 `cf_doc_loc` Value (`DocLoc`)

Fixed size: **8 bytes**: `leaf_id` (`u32` BE, bytes `0..4`) || `local_id` (`u32` BE, bytes `4..8`).

### 3.4 `PackedInternalNode` (`cf_internal_nodes`)

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

### 3.5 `PackedLeafNode` (`cf_leaf_entries`)

Derived parameters:
- `padded_dim = dim.next_power_of_two().max(64)`
- `rabitq_words = padded_dim / 64`

Total size: **`24 + (dim + padded_dim) * 4 + n * (20 + rabitq_words * 8)` bytes**, where `n` is the number of vectors in the leaf.

#### Header (`LEAF_HEADER_LEN = 24` bytes)

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 1 | `has_parent` | `u8` | `0` = `None` (root leaf), `1` = `Some(parent_id)` |
| `1` | 3 | reserved | `[u8; 3]` | Written as `[0, 0, 0]` |
| `4` | 4 | `n` | `u32` LE | Number of entries in this leaf |
| `8` | 4 | `parent_id` | `u32` LE | Parent `NodeId` when `has_parent == 1`; `0` otherwise |
| `12` | 4 | `next_local_id` | `u32` LE | Monotonic per-leaf `LocalId` allocator |
| `16` | 8 | reserved | `u64` | Written as `0` |

#### Body (starting at byte `24`)

| Section | Size (bytes) | Type | Layout / Semantics |
|---|---:|---|---|
| `doc_ids` | `n * 8` | `[u64; n]` LE | Strictly sorted ascending by `DocId` |
| `codes` | `n * rabitq_words * 8` | `[u64; n * rabitq_words]` LE | **Word-major** SoA: `codes[w * n + i]` |
| `local_ids` | `n * 4` | `[u32; n]` LE | Parallel to `doc_ids`; stable across normal leaf inserts/deletes |
| `centroid` | `dim * 4` | `[f32; dim]` LE | Leaf centroid $c$ |
| `rotated_centroid` | `padded_dim * 4` | `[f32; padded_dim]` LE | Precomputed $P^{-1} c$ |
| `res_norm_sqs` | `n * 4` | `[f32; n]` LE | Squared residual norm $\|x_i - c\|_2^2$ |
| `scales` | `n * 4` | `[f32; n]` LE | $\|x_i - c\|_2 / \langle \bar{o}_i, \tilde{o}_i \rangle$, or `0.0` when `res_norm_sq == 0.0` |

### 3.6 Global Vectors (`cf_vectors`)

- **Key:** `doc_key(doc_id)` (`doc_id.to_be_bytes()`, 8 bytes).
- **Value:** raw little-endian `[f32; dim]` (`dim * 4` bytes, no header). L2-normalized at insert when `metric == Cosine`.

### 3.7 Forward Document Attributes (`cf_docs`)

- **Key:** `doc_key(doc_id)` (`doc_id.to_be_bytes()`, 8 bytes).
- **Value:** `count: u16` LE followed by `count` entries sorted ascending by `attr_id: u16` LE, each storing `(attr_id: u16 LE, val_kind: u8, payload)`:

| `val_kind` | Rust Variant | Payload Encoding |
|---:|---|---|
| `0` | `Null` | Empty (0 B) |
| `1` | `Bool` | `0x00` (`false`) / `0x01` (`true`) (1 B) |
| `2` | `Int` | `i64` LE (8 B) |
| `3` | `Uint` | `u64` LE (8 B) |
| `4` | `Float` | `f64` LE (8 B; `NaN` rejected) |
| `5` | `String` | `len: u32` LE \|\| `utf8_bytes` |
| `6` | `StringArray` | `arr_len: u32` LE \|\| `arr_len` × (`len: u32` LE \|\| `utf8_bytes`) |
| `7` | `IntArray` | `arr_len: u32` LE \|\| `[i64; arr_len]` LE |
| `8` | `UintArray` | `arr_len: u32` LE \|\| `[u64; arr_len]` LE |

### 3.8 Order-Preserving Attribute Index Keys (`cf_attr_clusters` & `cf_attr_postings`)

- **`cf_attr_clusters` key:** `attr_id (u16 BE) || type_tag (u8) || encoded_val`
- **`cf_attr_postings` key:** `cf_attr_clusters_key || cluster_id (u32 BE)`

| `type_tag` | Meaning | `encoded_val` |
|---:|---|---|
| `0x00` | `TAG_PRESENT` | Empty (0 B) — marks non-null presence for `IsNull` / `IsNotNull` |
| `0x01` | `Bool` | 1 byte (`0x00` = `false`, `0x01` = `true`) |
| `0x02` | `Int(i64)` | 8 bytes BE with sign bit flipped: `(v as u64) ^ (1u64 << 63)` |
| `0x03` | `Uint(u64)` | 8 bytes BE: `v.to_be_bytes()` |
| `0x04` | `Float(f64)` | 8 bytes BE IEEE-754 order-preserving (`-0.0 -> +0.0`, `NaN` rejected; negative bits inverted, non-negative sign bit flipped) |
| `0x05` | `String` | Escaped-null-terminated UTF-8 (`0x00 -> 0x00 0xFF`, terminated with `0x00 0x00`) |

Array attributes (`StringArray`, `IntArray`, `UintArray`) index each distinct element under their element's scalar `type_tag`. Values in both CFs are `RoaringBitmap` payloads serialized with `RoaringBitmap::serialize_into`.

### 3.9 FTS Keys, Term Stats, Document Lengths, Posting & Position Blocks

Each document in an FTS-enabled collection is assigned a dense `FtsDocId: u32`, partitioned into 256-document blocks (`FTS_BLOCK_SIZE = 256`):

- `block_id: FtsBlockId = fts_doc_id >> 8` (`u32`)
- `slot: u8 = (fts_doc_id & 0xFF) as u8` (`0..=255`)

#### FTS Key Formats

- **`cf_fts_doc_map`:** `doc_id` (`u64` BE, 8 B) $\to$ `fts_doc_id` (`u32` BE, 4 B).
- **`cf_fts_rev_doc_map`:** `fts_doc_id` (`u32` BE, 4 B) $\to$ `doc_id` (`u64` BE, 8 B).
- **`cf_fts_terms`:** `attr_id` (`u16` BE, 2 B) || `token_utf8`.
- **`cf_fts_doc_lens`:** `attr_id` (`u16` BE, 2 B) || `block_id` (`u32` BE, 4 B) (6 B).
- **`cf_fts_postings` & `cf_fts_positions`:** `attr_id` (`u16` BE, 2 B) || `escaped_token` || `block_id` (`u32` BE, 4 B), where `escaped_token` uses `encode_str_ordered` (`0x00 -> 0x00 0xFF`, terminated with `0x00 0x00`) so `(attr_id, token)` prefixes remain prefix-free before `block_id`.

#### Shared `IntWidth` Tag (`DocLenBlock` & v2 `PackedPostingBlock`)

| `IntWidth` Tag | Variant | Element Width | Max Value |
|---:|---|---:|---:|
| `0` | `IntWidth::U8` | 1 byte | `255` |
| `1` | `IntWidth::U16` | 2 bytes LE | `65,535` |
| `2` | `IntWidth::U32` | 4 bytes LE | `4,294,967,295` |

#### `TermStats` (`cf_fts_terms`, fixed 16 bytes LE)

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 8 | `doc_freq` | `u64` LE | Number of documents containing this term |
| `8` | 4 | `max_tf` | `u32` LE | Maximum `tf` across all postings of this term |
| `12` | 4 | `min_dl_over_tf` | `f32` LE | Sound lower bound on `dl / tf` (`>= 0.0`, non-NaN; `f32::INFINITY` when empty) |

#### `DocLenBlock` (`cf_fts_doc_lens`, 260 / 516 / 1,028 bytes)

Stored once per `(attr_id, block_id)` whenever at least one slot in the block has `dl > 0`:

- **Header (4 B):** `width: u8` (`IntWidth`), `reserved: u8` (`0`), `count: u16` LE (`1..=256`, non-zero slot count).
- **Body (`256 * width` B):** `dls: [u8 | u16 | u32; 256]` LE, slot-indexed `0..=255` (`0` for absent/deleted slots).

#### v2 `PackedPostingBlock` (`cf_fts_postings`, TF-only, up to 256 docs per block)

- **Header (12 B):** `count: u16` LE (`1..=256`), `flags: u16` LE, `max_tf: u32` LE, `min_dl_over_tf: f32` LE.
  - `flags` bits `8..15` (`0xFF00`): format version, must equal `0x0200` (`POSTING_VERSION_V2`).
  - `flags` bits `1..2` (`0x0006`): `(flags & 0x0006) >> 1` encodes `IntWidth` (`0 = U8`, `1 = U16`, `2 = U32`).
  - `flags` bit `0` (`0x0001`, `FLAG_DENSE`): `0` = Sparse (`count < 96`), `1` = Dense (`count >= 96`, `POSTING_DENSE_THRESHOLD = 96`).
  - All other bits in `flags` must be `0`.
- **Sparse Payload (`FLAG_DENSE == 0`, `12 + count * (1 + width)` B):**
  - `slots: [u8; count]` (strictly sorted ascending) || `tfs: [u8 | u16 | u32; count]` LE (parallel to `slots`).
- **Dense Payload (`FLAG_DENSE == 1`, `44 + 256 * width` B):**
  - `bitmap: [u64; 4]` LE (32 B, 256-bit presence mask with popcount == `count`) || `tfs: [u8 | u16 | u32; 256]` LE (slot-indexed `0..=255`; `0` if absent).

#### `PackedPositionBlock` (`cf_fts_positions`, read only for `ContainsTokenSequence`)

- `[count: u16 LE (1..=256)][slots: [u8; count]][pos_offsets: [u32; count + 1] LE][delta_varints: [u8; ...]]`
- `slots` are strictly sorted ascending; `pos_offsets[i]..pos_offsets[i + 1]` slices the unsigned 7-bit LEB128 varint-encoded position deltas for `slots[i]` (`last_off == delta_varints.len()`).

---

## 4. Consistency Invariants

Verified exhaustively by `verify_invariants` and `verify_collection_invariants` ([`crates/izzydb/src/verify.rs`](../../crates/izzydb/src/verify.rs)):

1. **Disjoint node CFs & cache parity:** Reachable internal nodes match `internal_cache` and `cf_internal_nodes`; reachable leaves match `leaf_parent` and `cf_leaf_entries` with no orphans.
2. **Per-leaf `LocalId` uniqueness:** Every `local_id < leaf.next_local_id` and all `local_ids` within a leaf are unique.
3. **Vector & doc completeness:** `cf_vectors`, `cf_docs`, and `cf_doc_loc` are exact 1-to-1 bijections with reachable leaf `(doc_id, leaf_id, local_id)` entries and `TreeMeta.num_vectors`.
4. **Attribute postings & cluster summary exactness:** Every key in `cf_attr_postings` is non-empty and matches the exact set of `local_id`s reconstructed from `cf_docs` + `cf_doc_loc`; bit `cluster_id` is set in `cf_attr_clusters[k]` iff `cf_attr_postings[k || cluster_id]` exists and is non-empty.
5. **FTS index parity:** `cf_fts_doc_map` and `cf_fts_rev_doc_map` form a 1-to-1 bijection with `cf_docs` when FTS attributes exist; `cf_meta["fts_free_ids"]` is strictly disjoint from live `FtsDocId`s, bounded below `next_fts_doc_id`, and trimmed at the tail; `cf_meta["fts_meta"]` per-attribute `(doc_count, total_tokens)`, `cf_fts_terms` (`TermStats` sound globally and exact for `<= 64`-block terms), `cf_fts_doc_lens`, `cf_fts_postings`, and `cf_fts_positions` match a full re-analysis of `cf_docs`.
6. **Per-leaf `LocalId` bound:** Every leaf satisfies `next_local_id < (max_leaf_size * 4).max(1024).min(LocalId::MAX)`.
7. **Atomic updates:** All mutations across `cf_meta`, `cf_internal_nodes`, `cf_leaf_entries`, `cf_vectors`, `cf_doc_loc`, `cf_docs`, `cf_attr_clusters`, `cf_attr_postings`, and all FTS column families commit in a single `rocksdb::WriteBatch` before updating in-memory caches (`internal_cache`, `leaf_parent`, and write-through `fts_doc_lens_cache`).
