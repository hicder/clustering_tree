# Filtering & Inverted Index Architecture

This document provides a detailed specification of `izzydb`'s typed attribute schema, filter AST, order-preserving binary encodings, cluster-aware inverted index (`AttributeIndex`), atomic mutation lifecycle, and filtered ANN query execution.

Primary implementation sources:

- Schema & attribute types: [`crates/izzydb-core/src/schema.rs`](../../crates/izzydb-core/src/schema.rs)
- Filter AST, validation, and in-memory evaluator: [`crates/izzydb-core/src/filter.rs`](../../crates/izzydb-core/src/filter.rs)
- Attribute key/value (`LeafPosting`), zone map, & forward `DocView` codecs: [`crates/izzydb-storage/src/codec.rs`](../../crates/izzydb-storage/src/codec.rs)
- RocksDB storage, prefix extractors, and `TxContext`: [`crates/izzydb-storage/src/storage.rs`](../../crates/izzydb-storage/src/storage.rs)
- Slot-indexed `PackedLeafNode` (`local_id == slot`, `live_mask`): [`crates/izzydb-storage/src/node.rs`](../../crates/izzydb-storage/src/node.rs)
- Top-level `Collection` coordinator: [`crates/izzydb/src/collection.rs`](../../crates/izzydb/src/collection.rs)
- Two-level `AttributeIndex` & `LeafMask`: [`crates/izzydb/src/attr_index.rs`](../../crates/izzydb/src/attr_index.rs)
- Filtered search & dynamic expansion: [`crates/izzydb/src/search.rs`](../../crates/izzydb/src/search.rs)
- Masked RaBitQ leaf scoring: [`crates/izzydb-rabitq/src/simd.rs`](../../crates/izzydb-rabitq/src/simd.rs)
- Invariant verification: [`crates/izzydb/src/verify.rs`](../../crates/izzydb/src/verify.rs)

---

## 1. High-Level Architecture

`Collection` owns a single shared RocksDB instance (`Arc<Storage>`) and coordinates three storage subsystems inside one atomic `rocksdb::WriteBatch` per operation:

1. **Forward Document Store (`cf_docs` + `cf_vectors`):**
   Stores full-precision vectors (`cf_vectors`) and v2 offset-directory document attributes (`cf_docs`, zero-copy `DocView`) keyed by `DocId` (`u64` BE). Storing forward attributes allows `Collection::get` to reconstruct documents, lazy BM25 filters (`DocView::matches_filter`) to skip full deserialization, and `Collection::upsert`, `Collection::delete`, and tree reorganizations (`ClusterDelta`) to look up which index keys a document owns without scanning the inverted index.
2. **Vector Index (`ClusteringTree`):**
   Owns internal nodes in `default`, `cf_leaf_entries`, and `cf_doc_loc` (`doc_id -> (leaf_id, local_id, fts_doc_id)`). Each leaf cluster has a `NodeId` (`u32`), and each resident vector occupies a stable slot `slot` where **`local_id == slot`** tracked via `live_mask`.
3. **Cluster-Aware Inverted Index (`AttributeIndex`):**
   Maintains cluster summaries, compact per-leaf postings, and per-leaf zone maps:
   - **Cluster summary index (`cf_attr_clusters`):** maps `(attr_id, type_tag, encoded_val)` $\to$ `RoaringBitmap` of leaf `NodeId`s (`cluster_id: u32`) containing at least one document with that attribute value.
   - **Per-leaf zone maps in `default`:** maps `(attr_id, type_tag)` $\to$ `BTreeMap<NodeId, (min_val, max_val)>`, cached in RAM (`Storage::attr_zones_cache`) for zero-I/O cluster pruning on range queries (`Lt`/`Lte`/`Gt`/`Gte`).
   - **Cluster-first per-leaf posting index (`cf_attr_postings`):** maps `(attr_id, type_tag, cluster_id, encoded_val)` $\to$ compact `LeafPosting` (sparse sorted `u16` slots for `<= 16`, inline `[u64; words]` bitset for denser leaves, `RoaringBitmap` fallback). Supports both single-point `batched_multi_get_cf` lookups and prefix-bounded intra-leaf range scans (`7`-byte prefix `(attr_id || type_tag || cluster_id)`).

```text
Filter AST
   │
   ▼
[1] cf_attr_clusters + attr_zones_cache + fts_rev_map_cache
   │  └─► ClusterCandidates::Subset(leaf NodeId bitmap) | All
   │
   ├─► Subset.len() <= exhaustive_cluster_threshold (default 8)
   │      └─► Exhaustive crossover: fetch leaves ─► eval_leaves_masks (LeafMask) ─► exact cf_vectors top-k
   │
   └─► Pruned best-first traversal + dynamic expansion
          ├─► Ancestor pruning (if Subset < 50% of leaves)
          ├─► Min-heap pops batches of closest allowed leaves (beam_width, 2x, 4x, ...)
          ├─► eval_leaves_masks: batched cf_attr_postings + Range prefix scans ─► stack-inline LeafMask
          ├─► Mask already in slot space (local_id == slot) ─► AND live_mask ─► score_leaf_rabitq_masked
          └─► Optional Stage 2 exact rerank via cf_vectors
```

---

## 2. Schema & Attribute Type System

Schemas are configured via `CollectionConfig` and persisted as `CollectionSchema` in `default` under the key `b"\x01schema"`.

### 2.1 Supported `AttributeType`s

| `AttributeType` | Discriminant (`u8`) | Rust Payload (`AttributeValue`) | `is_array()` | `is_ordered()` | `scalar_type()` |
|---|---:|---|---|---|---:|
| `String` | `1` | `AttributeValue::String(String)` | No | Yes | `String` |
| `Int` | `2` | `AttributeValue::Int(i64)` | No | Yes | `Int` |
| `Uint` | `3` | `AttributeValue::Uint(u64)` | No | Yes | `Uint` |
| `Float` | `4` | `AttributeValue::Float(f64)` | No | Yes | `Float` |
| `Bool` | `5` | `AttributeValue::Bool(bool)` | No | No | `Bool` |
| `StringArray` | `6` | `AttributeValue::StringArray(Vec<String>)` | Yes | No | `String` |
| `IntArray` | `7` | `AttributeValue::IntArray(Vec<i64>)` | Yes | No | `Int` |
| `UintArray` | `8` | `AttributeValue::UintArray(Vec<u64>)` | Yes | No | `Uint` |

### 2.2 Schema Registration & `filterable` Flag

Each `AttributeConfig` declares:

- `name: String` (must be non-empty and unique across the collection).
- `attr_type: AttributeType`.
- `filterable: bool`:
  - **`true`:** The attribute is stored in `cf_docs` **and** indexed in `cf_attr_clusters`, `cf_attr_postings`, and the attribute zones in `default`. Queries may reference it in `Filter` predicates.
  - **`false`:** Payload-only attribute. Stored in `cf_docs` and returned by `Collection::get`, but skipped during inverted index extraction (`extract_index_keys`) and rejected by `Filter::validate` with `TreeError::InvalidFilter`.

`CollectionSchema` assigns each attribute a monotonic `AttrId = u16` starting at `0` (up to `AttrId::MAX` attributes). Using `AttrId` instead of attribute name strings keeps `cf_docs` payloads and inverted-index RocksDB keys compact.

### 2.3 Validation, `Null`, and `NaN` Rules

- **Undeclared attributes:** `upsert` rejects any attribute name not present in `CollectionSchema` (`TreeError::InvalidAttribute`).
- **Missing vs `Null`:** Attributes omitted from a document's `Attributes` map (`BTreeMap<String, AttributeValue>`) and attributes explicitly set to `AttributeValue::Null` are both accepted for any declared type (including array types during `validate_attributes`) and produce no inverted-index keys.
- **`NaN` floats:** `f64::NAN` is rejected everywhere—in `CollectionSchema::validate_attributes`, `encode_attributes`, `ScalarValue::matches_scalar_type`, and `encode_f64_ordered`.
- **Negative zero:** `-0.0_f64` is normalized to `+0.0_f64` in both `Filter::matches`, `DocView::matches_filter`, and `encode_f64_ordered` so equality and range behavior are identical between in-memory evaluation and index scans.

### 2.4 Schema Persistence & Evolution (`reconcile`)

On `Collection::open`:

1. If `default[b"\x01schema"]` is absent and `meta.num_vectors == 0`, `CollectionSchema::from_config` creates and writes the initial schema (`version = 3`).
2. If `default[b"\x01schema"]` is absent and `meta.num_vectors > 0`, opening fails with `TreeError::InvalidSchema`.
3. If `default[b"\x01schema"]` exists, `CollectionSchema::reconcile` verifies every declared attribute in `CollectionConfig`:
   - Existing attributes must match both `attr_type` (`field: "attribute_type"`) and `filterable` (`field: "attribute_filterable"`), or `open` fails with `TreeError::ConfigMismatch`.
   - New attributes not yet in the persisted schema are appended with new monotonic `AttrId`s, and the updated schema is persisted back to `default[b"\x01schema"]`. Pre-existing documents do not need rewriting; they implicitly evaluate as missing/`Null` for newly added attributes.

---

## 3. Filter AST & Semantics

`Filter` ([`crates/izzydb-core/src/filter.rs`](../../crates/izzydb-core/src/filter.rs)) defines the predicate AST evaluated by `Filter::matches(&Attributes)`, `DocView::matches_filter`, and `AttributeIndex` against RocksDB.

### 3.1 Variants & Validation Rules (`Filter::validate`)

| Variant | Allowed Attribute Types | `ScalarValue::Null` Allowed? | Notes |
|---|---|---|---|
| `And(Vec<Filter>)` | Recursive | — | Empty `And(vec![])` matches all documents (`True`) |
| `Or(Vec<Filter>)` | Recursive | — | Empty `Or(vec![])` matches no documents (`False`) |
| `Not(Box<Filter>)` | Recursive | — | Logical complement over documents in the collection |
| `Eq(attr, val)` | Scalar (`String`, `Int`, `Uint`, `Float`, `Bool`) | Yes | Rejected on array attributes (`use Contains`) |
| `NotEq(attr, val)` | Scalar | Yes | Equivalent to `Not(Eq(attr, val))` |
| `In(attr, vals)` | Scalar | Yes | Empty `vals` matches no documents; rejected on array attributes (`use ContainsAny`) |
| `NotIn(attr, vals)` | Scalar | Yes | Equivalent to `Not(In(attr, vals))` |
| `Lt` / `Lte` / `Gt` / `Gte` | Ordered scalars (`String`, `Int`, `Uint`, `Float`) | **No** | Rejected on `Bool` and array types |
| `Contains(attr, val)` | Arrays (`StringArray`, `IntArray`, `UintArray`) | **No** | `val` must match `attr_type.scalar_type()` |
| `ContainsAny(attr, vals)` | Arrays | **No** | Each element in `vals` must match `attr_type.scalar_type()` |
| `IsNull(attr)` | Any filterable attribute | — | Matches missing attributes and explicit `AttributeValue::Null` |
| `IsNotNull(attr)` | Any filterable attribute | — | Matches any document with a non-null value (including empty arrays!) |
| `ContainsAllTokens` / `ContainsAnyToken` / `ContainsTokenSequence` | FTS-enabled attributes | — | Evaluated against `FtsIndex` and resolved to `LeafSlotMasks` via `FtsRevBlock` |

### 3.2 Subtle Semantic Details

- **Empty arrays and `IsNotNull`:** An empty array (e.g. `AttributeValue::StringArray(vec![])`) is non-null, so `append_index_keys_for_value` still emits `TAG_PRESENT` (`0x00`) for it even though it emits zero element keys. Consequently, `IsNotNull` matches empty arrays and `IsNull` does not, matching `Filter::matches`.
- **Negation includes `Null`/missing:** `NotEq("category", "books")` is defined as `!Eq("category", "books")`. Documents where `"category"` is omitted or `AttributeValue::Null` satisfy `NotEq("category", "books")`, `NotIn("category", [...])`, and `Not(Eq(...))`.
- **Range predicates exclude `Null`/missing:** `Lt`, `Lte`, `Gt`, and `Gte` only match documents that have a non-null value of that attribute satisfying the comparison.

---

## 4. Storage Layout & Order-Preserving Encodings

### 4.1 Column Families

| Column Family | Key Format | Value Format | RocksDB Options |
|---|---|---|---|
| `cf_docs` | `doc_id` (`u64` BE, 8 B) | v2 offset-directory `Attributes` (`DocView`) | Point-lookup CF (10 bpk full-key Bloom, whole-key memtable filtering, shared 128 MiB LRU block cache) |
| `cf_doc_loc` | `doc_id` (`u64` BE, 8 B) | `leaf_id` (`u32` BE) \|\| `local_id` (`u32` BE) \|\| `fts_doc_id` (`u32` BE, `u32::MAX` = None) | Point-lookup CF |
| `cf_attr_clusters` | `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) \|\| `encoded_val` | `RoaringBitmap` of `cluster_id: u32` | 3-byte fixed prefix extractor (`attr_id \|\| type_tag`), 10 bpk Bloom (`whole_key_filtering = true`) |
| `cf_attr_postings` | `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) \|\| `cluster_id` (`u32` BE) \|\| `encoded_val` | `LeafPosting` (`Sparse` / `Dense` / `Roaring`) | 7-byte fixed prefix extractor (`attr_id \|\| type_tag \|\| cluster_id`), 10 bpk Bloom, uncompressed |
| attribute zones in `default` | `0x03` \|\| `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) (4 B) | `AttrZoneMap` (`leaf_id -> (min_b, max_b)`) | Namespace scan on first load into `Storage::attr_zones_cache` |

Only non-empty `RoaringBitmap`s, `LeafPosting`s, and `AttrZoneMap`s are persisted. Whenever a mutation empties an entry, `TxContext::commit` deletes that key from RocksDB.

### 4.2 v2 Forward Document Format (`cf_docs`, `DocView`)

`encode_attributes` sorts entries by `AttrId` ascending and encodes a binary offset directory for $O(\log F)$ single-attribute lookup and zero-allocation filter evaluation:

```text
[version: u8 = 0x02][reserved: u8 = 0][count: u16 LE]
  directory (count * 8 B, sorted by attr_id):
    [attr_id: u16 LE][val_kind: u8][reserved: u8 = 0][payload_offset: u32 LE]
  payloads:
    packed bytes for entry 0, entry 1, ..., entry (count - 1)
```

`val_kind` tags (`0..=8`) and payloads:

- `0` (`Null`): 0 bytes
- `1` (`Bool`): 1 byte (`0` or `1`)
- `2` (`Int`): `i64` LE (8 bytes)
- `3` (`Uint`): `u64` LE (8 bytes)
- `4` (`Float`): `f64` LE (8 bytes; `NaN` rejected)
- `5` (`String`): raw UTF-8 bytes (length derived from consecutive `payload_offset`s)
- `6` (`StringArray`): `count: u32 LE` + `count` length-prefixed UTF-8 strings (`len: u32 LE` + bytes)
- `7` (`IntArray`): `count: u32 LE` + `count` `i64` LE values
- `8` (`UintArray`): `count: u32 LE` + `count` `u64` LE values

`decode_attribute_entries` reads `(AttrId, AttributeValue)` pairs directly without string allocations, while `DocView::matches_filter` binary-searches the directory for only the attributes referenced by a filter.

### 4.3 Order-Preserving Inverted Key Encoding (`cf_attr_clusters` & `cf_attr_postings`)

Every key in `cf_attr_clusters` has the form:

```text
[attr_id: u16 BE (2 B)] [type_tag: u8 (1 B)] [encoded_val: variable]
```

And every key in cluster-first `cf_attr_postings` places `cluster_id` before `encoded_val`:

```text
[attr_id: u16 BE (2 B)] [type_tag: u8 (1 B)] [cluster_id: u32 BE (4 B)] [encoded_val: variable]
```

Moving `cluster_id` before `encoded_val` groups every `(attr_id, type_tag, encoded_val)` posting owned by the same leaf contiguously in RocksDB under a single 7-byte prefix, turning arbitrary intra-leaf range queries (`[lo, hi]`) into one contiguous prefix-Bloom-accelerated scan.

The `type_tag` and `encoded_val` encodings preserve **RocksDB's bytewise lexicographical order = logical value order**:

| `type_tag` | Constant | Meaning | `encoded_val` Encoding |
|---:|---|---|---|
| `0x00` | `TAG_PRESENT` | Non-null presence marker | Empty (`0` bytes). Key is 3 bytes in `cf_attr_clusters`, 7 bytes in `cf_attr_postings`. |
| `0x01` | `TAG_BOOL` | `Bool(bool)` | `1` byte: `0x00` (`false`) or `0x01` (`true`). |
| `0x02` | `TAG_INT` | `Int(i64)` / `IntArray` elem | `8` bytes BE with sign bit flipped: `((v as u64) ^ (1u64 << 63)).to_be_bytes()`. Maps `i64::MIN` $\to$ `0x0000...` and `i64::MAX` $\to$ `0xFFFF...`. |
| `0x03` | `TAG_UINT` | `Uint(u64)` / `UintArray` elem | `8` bytes BE: `v.to_be_bytes()`. |
| `0x04` | `TAG_FLOAT` | `Float(f64)` | `8` bytes BE IEEE-754 order-preserving: normalize `-0.0` to `+0.0`; if sign bit is `1` (negative), invert all 64 bits (`!bits`); otherwise flip the sign bit (`bits ^ (1u64 << 63)`). Preserves order from `NEG_INFINITY` through `INFINITY`. |
| `0x05` | `TAG_STRING` | `String` / `StringArray` elem | Escaped-null-terminated UTF-8: each `0x00` byte becomes `[0x00, 0xFF]`, followed by terminator `[0x00, 0x00]`. |

### 4.4 `LeafPosting` Compact Per-Leaf Bitsets (`cf_attr_postings`)

Instead of serializing a full `RoaringBitmap` header for small leaf slot sets (`0 <= slot < slot_cap`), `LeafPosting` selects the smallest self-describing representation:

- **`0x00` (`Sparse`, `len <= 16` and `max_id <= u16::MAX`):** `1 + 2 * len` bytes storing sorted `u16` LE slots (`3..33` bytes total).
- **`0x01` (`Dense`, `max_id <= 4095`):** `3 + num_words * 8` bytes storing `num_words: u16 LE` followed by raw `[u64; num_words]` LE bitset words (no heap allocation on `or_into_words`).
- **`0x02` (`Roaring`):** `RoaringBitmap` fallback for larger IDs.

Both `Sparse` and `Dense` decode via zero-copy `LeafPostingView::or_into_words` directly into stack-inline `LeafMask` buffers (`inline: [u64; 4]` covering up to 256 slots).

---

## 5. Slot-Indexed `local_id == slot` & Atomic Mutation Lifecycle

`PackedLeafNode` ([`crates/izzydb-storage/src/node.rs`](../../crates/izzydb-storage/src/node.rs)) stores entries **slot-indexed (`local_id == slot`)** with `live_mask: Vec<u64>`, eliminating the per-leaf `local_ids` array:

1. **Normal leaf insert / delete:**
   - **Insert (`alloc_slot`):** reuses the lowest cleared bit in `live_mask` when `live_count < slot_cap`, or appends at `slot_cap`. All existing entries keep their slot index unchanged.
   - **Delete (`remove_entry`):** clears bit `slot` in `live_mask` and shrinks `slot_cap` if trailing slots are empty.
2. **Compaction (`needs_leaf_compaction`):**
   - When `slot_cap > max_leaf_size` or `slot_cap > 32 && live_count * 2 <= slot_cap`, surviving entries are compactly repacked to `0..n` and `ClusterDelta` updates `AttributeIndex` + `FtsRevBlock`.
3. **Structural mutations (leaf split, underflow merge/redistribution, or SPFresh LIRE reassignment):**
   - Surviving entries in affected leaves are compactly packed to `0..n` via `PackedLeafNode::reset_and_encode` (updating `cf_doc_loc` via `tx.put_doc_pos` while preserving `fts_doc_id`), and `tree.rs` emits `ClusterDelta { removed, added }` processed by `AttributeIndex::apply_cluster_delta` (removals before additions) inside `TxContext`.

---

## 6. Filtered Search Execution

`Collection::search(query, opts, Some(filter))` dispatches to `search::search_filtered` ([`crates/izzydb/src/search.rs`](../../crates/izzydb/src/search.rs)). Unfiltered queries (`filter = None`) take `search::search` with zero filtering overhead.

### 6.1 Step 1: Cluster-Level Pruning (`AttributeIndex::prepare_filter_with_fts`)

After `filter.validate(schema)` and `prepare_vector_cow`, `AttributeIndex::prepare_filter_with_fts` evaluates the `Filter` AST:

| `Filter` Node | Cluster Pruning Source | `ClusterCandidates` | `PreparedNode` |
|---|---|---|---|
| `Eq(attr, Null)` | `cf_attr_clusters` (`TAG_PRESENT`) | `All` | `Not(Keys([(present_key, bm)]))` (or `True`) |
| `Eq(attr, val)` / `Contains(attr, val)` | Point lookup `cf_attr_clusters` | `Subset(bm)` (empty $\to$ `False`) | `Keys([(key, bm)])` |
| `NotEq(attr, Null)` | `cf_attr_clusters` (`TAG_PRESENT`) | `Subset(bm)` | `Keys([(present_key, bm)])` |
| `NotEq(attr, val)` (non-null) | Evaluates inner `Eq` | `All` | `Not(inner)` |
| `In(attr, vals)` / `ContainsAny(attr, vals)` | Deduplicated `multi_get_attr_cluster_bitmaps` + bitwise `\|` | `Subset(union)` (or `All` if `Null`) | `Keys(items)` (or `Or([Keys, is_null])`) |
| `NotIn(attr, vals)` | Evaluates inner `In` | `All` | `Not(inner)` |
| `Lt` / `Lte` / `Gt` / `Gte` | In-RAM `AttrZoneMap` (`attr_zones_cache`) interval check | `Subset(bm)` | `Range { attr_id, type_tag, lo, hi, cands }` |
| `ContainsAllTokens` / `ContainsAnyToken` / `ContainsTokenSequence` | `FtsIndex` + in-RAM `fts_rev_map_cache` | `Subset(leaf_bm)` | `LeafSlotMasks(per_leaf)` |
| `And(children)` | Intersect `Subset`s (`&=`), coalesce same-attr `Range`s, order non-`Range` before `Range` | `Subset` if any child is `Subset`; else `All` | `And(children)` (or `False` on contradiction/empty) |
| `Or(children)` | Union `Subset` bitmaps (`\|=`) | `All` if any child is `All`; else `Subset(union)` | `Or(children)` |
| `Not(child)` | Evaluates `child` | `All` | `Not(child)` |

### 6.2 Step 2 & Step 3: Crossover and Dynamic Expansion

- **Exhaustive-scan crossover (`bm.len() <= exhaustive_cluster_threshold`, default `8`):** fetches qualifying leaves, evaluates `LeafMask`s via `AttributeIndex::eval_leaves_masks` (batched point gets on `cf_attr_postings` + zero-alloc `scan_attr_posting_range_into_words` on `Range` nodes), iterates set bits via `.trailing_zeros()` (`slot == local_id`), fetches exact `cf_vectors`, and returns top `k`.
- **Pruned best-first traversal + dynamic expansion:** applies ancestor pruning when `Subset` covers `< 50%` of leaves, expands closest qualifying leaves in batches (`beam_width`, doubling), evaluates `LeafMask`s (with short-circuiting so selective equality/FTS predicates skip `Range` scans), intersects with `view.live_mask_bytes` on tombstoned leaves, passes `mask.as_slice()` directly to `score_leaf_rabitq_masked`, and reranks candidates if `opts.rerank_candidates` is set.

---

## 7. Consistency Invariants (`Collection::verify_invariants`)

`verify_collection_invariants` ([`crates/izzydb/src/verify.rs`](../../crates/izzydb/src/verify.rs)) exhaustively verifies:

1. **Schema parity:** `default[b"\x01schema"]` matches the in-memory `CollectionSchema`.
2. **Document bijection:** `cf_docs` forms a 1-to-1 bijection with `cf_doc_loc`, `cf_vectors`, and reachable slot-indexed leaf entries (`local_id == slot`, `slot_cap <= max_leaf_size`).
3. **Full inverted-index & zone-map reconstruction check:** re-decodes every `cf_docs` entry, pairs it with `cf_doc_loc`, and asserts byte/set equality for `cf_attr_postings` (`LeafPosting`), `cf_attr_clusters` (`RoaringBitmap`), and attribute zones in `default` (`AttrZoneMap`).
