# Filtering & Inverted Index Architecture

This document provides a detailed specification of `izzydb`'s typed attribute schema, filter AST, order-preserving binary encodings, two-level cluster-aware inverted index (`AttributeIndex`), atomic mutation lifecycle, and filtered ANN query execution.

Primary implementation sources:

- Schema & attribute types: [`crates/izzydb-core/src/schema.rs`](../../crates/izzydb-core/src/schema.rs)
- Filter AST, validation, and in-memory evaluator: [`crates/izzydb-core/src/filter.rs`](../../crates/izzydb-core/src/filter.rs)
- Attribute key & forward document codecs: [`crates/izzydb-storage/src/codec.rs`](../../crates/izzydb-storage/src/codec.rs)
- RocksDB storage, prefix extractors, and `TxContext`: [`crates/izzydb-storage/src/storage.rs`](../../crates/izzydb-storage/src/storage.rs)
- Packed leaf `local_ids` & `next_local_id`: [`crates/izzydb-storage/src/node.rs`](../../crates/izzydb-storage/src/node.rs)
- Top-level `Collection` coordinator: [`crates/izzydb/src/collection.rs`](../../crates/izzydb/src/collection.rs)
- Two-level `AttributeIndex`: [`crates/izzydb/src/attr_index.rs`](../../crates/izzydb/src/attr_index.rs)
- Filtered search & dynamic expansion: [`crates/izzydb/src/search.rs`](../../crates/izzydb/src/search.rs)
- Masked RaBitQ leaf scoring: [`crates/izzydb-rabitq/src/simd.rs`](../../crates/izzydb-rabitq/src/simd.rs)
- Invariant verification: [`crates/izzydb/src/verify.rs`](../../crates/izzydb/src/verify.rs)

---

## 1. High-Level Architecture

`Collection` owns a single shared RocksDB instance (`Arc<Storage>`) and coordinates three storage subsystems inside one atomic `rocksdb::WriteBatch` per operation:

1. **Forward Document Store (`cf_docs` + `cf_vectors`):**
   Stores full-precision vectors (`cf_vectors`) and binary-encoded document attributes (`cf_docs`) keyed by `DocId` (`u64` BE). Storing forward attributes allows `Collection::get` to reconstruct documents and enables `Collection::upsert`, `Collection::delete`, and tree reorganizations (`ClusterDelta`) to look up which index keys a document owns without scanning the inverted index.
2. **Vector Index (`ClusteringTree`):**
   Owns `cf_internal_nodes`, `cf_leaf_entries`, and `cf_doc_loc` (`doc_id -> (leaf_id, local_id)`). Each leaf cluster has a `NodeId` (`u32`) and assigns each resident vector a compact per-leaf `LocalId` (`u32`).
3. **Cluster-Aware Inverted Index (`AttributeIndex`):**
   Maintains a two-level bitmap index using `roaring::RoaringBitmap`:
   - **Cluster summary index (`cf_attr_clusters`):** maps `(attr_id, type_tag, encoded_val)` $\to$ `RoaringBitmap` of leaf `NodeId`s (`cluster_id: u32`) containing at least one document with that attribute value. Used for zero-leaf-I/O cluster pruning and tree ancestor pruning.
   - **Per-cluster posting index (`cf_attr_postings`):** maps `(attr_id, type_tag, encoded_val, cluster_id)` $\to$ `RoaringBitmap` of per-leaf `LocalId`s (`local_id: u32`). Used for exact intra-leaf document filtering via a single batched `multi_get_cf` per probed leaf batch.

```text
Filter AST
   │
   ▼
[1] cf_attr_clusters ──► ClusterCandidates::Subset(leaf NodeId bitmap) | All
   │
   ├─► Subset.len() <= exhaustive_cluster_threshold (default 8)
   │      └─► Exhaustive crossover: fetch leaves ─► eval_leaves_masks ─► exact cf_vectors top-k
   │
   └─► Pruned best-first traversal + dynamic expansion
          ├─► Ancestor pruning (if Subset < 50% of leaves)
          ├─► Min-heap pops batches of closest allowed leaves (beam_width, 2x, 4x, ...)
          ├─► cf_attr_postings (1 batched_multi_get_cf per batch) ─► local_id RoaringBitmap
          ├─► Map local_id mask ─► slot_mask_words ─► score_leaf_rabitq_masked
          └─► Optional Stage 2 exact rerank via cf_vectors
```

---

## 2. Schema & Attribute Type System

Schemas are configured via `CollectionConfig` and persisted as `CollectionSchema` in `cf_meta` under the key `b"schema"`.

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
  - **`true`:** The attribute is stored in `cf_docs` **and** indexed in `cf_attr_clusters` / `cf_attr_postings`. Queries may reference it in `Filter` predicates.
  - **`false`:** Payload-only attribute. Stored in `cf_docs` and returned by `Collection::get`, but skipped during inverted index extraction (`extract_index_keys`) and rejected by `Filter::validate` with `TreeError::InvalidFilter`.

`CollectionSchema` assigns each attribute a monotonic `AttrId = u16` starting at `0` (up to `AttrId::MAX` attributes). Using `AttrId` instead of attribute name strings keeps `cf_docs` payloads and inverted-index RocksDB keys compact.

### 2.3 Validation, `Null`, and `NaN` Rules

- **Undeclared attributes:** `upsert` rejects any attribute name not present in `CollectionSchema` (`TreeError::InvalidAttribute`).
- **Missing vs `Null`:** Attributes omitted from a document's `Attributes` map (`BTreeMap<String, AttributeValue>`) and attributes explicitly set to `AttributeValue::Null` are both accepted for any declared type (including array types during `validate_attributes`) and produce no inverted-index keys.
- **`NaN` floats:** `f64::NAN` is rejected everywhere—in `CollectionSchema::validate_attributes`, `encode_attributes`, `ScalarValue::matches_scalar_type`, and `encode_f64_ordered`.
- **Negative zero:** `-0.0_f64` is normalized to `+0.0_f64` in both `Filter::matches` and `encode_f64_ordered` so equality and range behavior are identical between in-memory evaluation and index scans.

### 2.4 Schema Persistence & Evolution (`reconcile`)

On `Collection::open`:

1. If `cf_meta["schema"]` is absent and `meta.num_vectors == 0`, `CollectionSchema::from_config` creates and writes the initial schema (`version = 1`).
2. If `cf_meta["schema"]` is absent and `meta.num_vectors > 0`, opening fails with `TreeError::InvalidSchema`.
3. If `cf_meta["schema"]` exists, `CollectionSchema::reconcile` verifies every declared attribute in `CollectionConfig`:
   - Existing attributes must match both `attr_type` (`field: "attribute_type"`) and `filterable` (`field: "attribute_filterable"`), or `open` fails with `TreeError::ConfigMismatch`.
   - New attributes not yet in the persisted schema are appended with new monotonic `AttrId`s, and the updated schema is persisted back to `cf_meta["schema"]`. Pre-existing documents do not need rewriting; they implicitly evaluate as missing/`Null` for newly added attributes.

---

## 3. Filter AST & Semantics

`Filter` ([`crates/izzydb-core/src/filter.rs`](../../crates/izzydb-core/src/filter.rs)) defines the predicate AST evaluated both by `Filter::matches(&Attributes)` in memory and by `AttributeIndex` against RocksDB.

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

### 3.2 Subtle Semantic Details

- **Empty arrays and `IsNotNull`:** An empty array (e.g. `AttributeValue::StringArray(vec![])`) is non-null, so `append_index_keys_for_value` still emits `TAG_PRESENT` (`0x00`) for it even though it emits zero element keys. Consequently, `IsNotNull` matches empty arrays and `IsNull` does not, matching `Filter::matches`.
- **Negation includes `Null`/missing:** `NotEq("category", "books")` is defined as `!Eq("category", "books")`. Documents where `"category"` is omitted or `AttributeValue::Null` satisfy `NotEq("category", "books")`, `NotIn("category", [...])`, and `Not(Eq(...))`.
- **Range predicates exclude `Null`/missing:** `Lt`, `Lte`, `Gt`, and `Gte` only match documents that have a non-null value of that attribute satisfying the comparison.

---

## 4. Storage Layout & Order-Preserving Encodings

### 4.1 Column Families

| Column Family | Key Format | Value Format | RocksDB Options |
|---|---|---|---|
| `cf_docs` | `doc_id` (`u64` BE, 8 B) | Forward binary `Attributes` | Point-lookup CF (10 bpk full-key Bloom, whole-key memtable filtering, shared 128 MiB LRU block cache) |
| `cf_doc_loc` | `doc_id` (`u64` BE, 8 B) | `leaf_id` (`u32` BE) \|\| `local_id` (`u32` BE) | Point-lookup CF |
| `cf_attr_clusters` | `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) \|\| `encoded_val` | `RoaringBitmap` of `cluster_id: u32` | 3-byte fixed prefix extractor (`attr_id \|\| type_tag`), 10 bpk Bloom (`whole_key_filtering = true`), `prefix_same_as_start = true` on range scans |
| `cf_attr_postings` | `cluster_key` \|\| `cluster_id` (`u32` BE, 4 B) | `RoaringBitmap` of `local_id: u32` | Point-lookup CF (10 bpk full-key Bloom, whole-key memtable filtering) |

Only non-empty `RoaringBitmap`s are persisted. Whenever a mutation empties a bitmap in either `cf_attr_postings` or `cf_attr_clusters`, `TxContext::commit` deletes that key from RocksDB.

### 4.2 Forward Document Format (`cf_docs`)

`encode_attributes` sorts entries by `AttrId` ascending and encodes:

```text
[count: u16 LE]
  repeat count times:
    [attr_id: u16 LE] [val_kind: u8] [payload...]
```

`val_kind` tags (`0..=8`) and payloads:

- `0` (`Null`): 0 bytes
- `1` (`Bool`): 1 byte (`0` or `1`)
- `2` (`Int`): `i64` LE (8 bytes)
- `3` (`Uint`): `u64` LE (8 bytes)
- `4` (`Float`): `f64` LE (8 bytes; `NaN` rejected)
- `5` (`String`): `len: u32 LE` + UTF-8 bytes
- `6` (`StringArray`): `count: u32 LE` + `count` length-prefixed UTF-8 strings (`len: u32 LE` + bytes)
- `7` (`IntArray`): `count: u32 LE` + `count` `i64` LE values
- `8` (`UintArray`): `count: u32 LE` + `count` `u64` LE values

`decode_attribute_entries` reads `(AttrId, AttributeValue)` pairs directly without string allocations or hash lookups, which keeps `Collection::upsert`, `Collection::delete`, and `AttributeIndex::apply_cluster_delta` fast.

### 4.3 Order-Preserving Inverted Key Encoding (`cf_attr_clusters` & `cf_attr_postings`)

Every key in `cf_attr_clusters` has the form:

```text
[attr_id: u16 BE (2 B)] [type_tag: u8 (1 B)] [encoded_val: variable]
```

And every key in `cf_attr_postings` appends the 4-byte cluster ID:

```text
[attr_id: u16 BE (2 B)] [type_tag: u8 (1 B)] [encoded_val: variable] [cluster_id: u32 BE (4 B)]
```

The `type_tag` and `encoded_val` encodings are chosen so that **RocksDB's bytewise lexicographical order matches logical value order**:

| `type_tag` | Constant | Meaning | `encoded_val` Encoding |
|---:|---|---|---|
| `0x00` | `TAG_PRESENT` | Non-null presence marker | Empty (`0` bytes). Key is 3 bytes in `cf_attr_clusters`, 7 bytes in `cf_attr_postings`. |
| `0x01` | `TAG_BOOL` | `Bool(bool)` | `1` byte: `0x00` (`false`) or `0x01` (`true`). |
| `0x02` | `TAG_INT` | `Int(i64)` / `IntArray` elem | `8` bytes BE with sign bit flipped: `((v as u64) ^ (1u64 << 63)).to_be_bytes()`. Maps `i64::MIN` $\to$ `0x0000...` and `i64::MAX` $\to$ `0xFFFF...`. |
| `0x03` | `TAG_UINT` | `Uint(u64)` / `UintArray` elem | `8` bytes BE: `v.to_be_bytes()`. |
| `0x04` | `TAG_FLOAT` | `Float(f64)` | `8` bytes BE IEEE-754 order-preserving: normalize `-0.0` to `+0.0`; if sign bit is `1` (negative), invert all 64 bits (`!bits`); otherwise flip the sign bit (`bits ^ (1u64 << 63)`). Preserves order from `NEG_INFINITY` through `INFINITY`. |
| `0x05` | `TAG_STRING` | `String` / `StringArray` elem | Escaped-null-terminated UTF-8: each `0x00` byte becomes `[0x00, 0xFF]`, followed by terminator `[0x00, 0x00]`. |

#### Why Escaped-Null-Termination (`0x00 -> 0x00 0xFF`, terminator `0x00 0x00`)?

1. **Order preservation for prefixes and embedded NULs:** Because the terminator `0x00 0x00` is strictly smaller than any escaped NUL `0x00 0xFF` and any non-zero UTF-8 byte `>= 0x01`, `"a"` (`[0x61, 0x00, 0x00]`) sorts before `"ab"` (`[0x61, 0x62, 0x00, 0x00]`) and `"\0"` (`[0x00, 0xFF, 0x00, 0x00]`) sorts before `"\0a"` (`[0x00, 0xFF, 0x61, 0x00, 0x00]`), while `""` (`[0x00, 0x00]`) sorts first.
2. **Unambiguous `cf_attr_postings` suffixing:** In `cf_attr_postings`, `cluster_id` (`u32` BE) is appended immediately after `encoded_val`. A fixed terminator ensures `"a" || cluster_id` can never collide with `"a<bytes>" || other_cluster_id`.

#### Key Extraction for a Document (`extract_index_keys`)

For each attribute in a document:

1. If `!schema_entry.filterable` or `val == AttributeValue::Null`: emit nothing.
2. Otherwise, emit `encode_present_cluster_key(attr_id)` (`TAG_PRESENT`).
3. Emit the scalar key, or for array attributes (`StringArray`, `IntArray`, `UintArray`), emit one key per element using the element's scalar `type_tag` (`TAG_STRING`, `TAG_INT`, `TAG_UINT`).
4. Sort and deduplicate the resulting key list (`keys.sort_unstable(); keys.dedup()`), so duplicate values within an array (e.g. `["a", "b", "a"]`) produce one key.

### 4.4 RocksDB Prefix Extraction & Range Scans

`cf_attr_clusters` configures `SliceTransform::create_fixed_prefix(3)` over `(attr_id (2 B) || type_tag (1 B))` alongside `set_whole_key_filtering(true)`.

- Point lookups (`Eq`, `In`, `Contains`, `IsNotNull`) use full-key Bloom filters (`get_pinned_cf` / `batched_multi_get_cf`).
- Range predicates (`Lt`, `Lte`, `Gt`, `Gte`) call `Storage::scan_attr_cluster_range` with `ReadOptions::set_prefix_same_as_start(true)`:
  - `Lt` / `Lte`: seek to `prefix = [attr_id BE, type_tag]` (which is strictly `<` any key with a value payload) and iterate forward until `!k.starts_with(&prefix)` or `k >= bound` (`>` for `Lte`).
  - `Gt` / `Gte`: seek to `bound` (`encode_scalar_cluster_key(attr_id, val)`), skipping `k == bound` for `Gt`, and iterate forward until `!k.starts_with(&prefix)`.

### 4.5 Why `RoaringBitmap` Fits Naturally

- `NodeId` was narrowed to `u32` and `LocalId` is `u32`, matching `roaring::RoaringBitmap`'s native 32-bit element domain.
- In `cf_attr_postings`, every `local_id` within a leaf satisfies `0 <= local_id < next_local_id` (typically `<= max_leaf_size`, default `160`), so every per-cluster posting bitmap fits inside a single 16-bit Roaring container (`high16 == 0`). Bitwise intersections, unions, and complements execute in-memory without decompressing into `Vec<u32>`.

---

## 5. Stable Per-Leaf `LocalId`s & Atomic Mutation Lifecycle

### 5.1 Why `LocalId` Exists

`PackedLeafNode` stores `doc_ids: [u64; n]` in strictly ascending `DocId` order so point updates/removals can locate entries via binary search. If `AttributeIndex` stored raw array slot indices (`0..n`), inserting a document whose `DocId` lands in the middle of a leaf would shift all higher slot indices and force rewriting every attribute posting bitmap in that leaf.

Instead, `PackedLeafNode` ([`crates/izzydb-storage/src/node.rs`](../../crates/izzydb-storage/src/node.rs)) stores:

- `next_local_id: u32` in the 24-byte leaf header (bytes `12..16`).
- `local_ids: [u32; n]` parallel to `doc_ids` in the leaf body.

Their lifecycle follows two regimes:

1. **Normal leaf insert / delete (no split, merge, or redistribution):**
   - **Insert:** Allocates `local_id = leaf.next_local_id; leaf.next_local_id += 1;`. All existing entries keep their `local_id`s unchanged, so `AttributeIndex` only inserts bits for the newly added document.
   - **Delete:** Removes the entry and keeps surviving entries' `local_id`s unchanged (resetting `next_local_id = 0` only when the leaf becomes empty). `AttributeIndex` only removes bits for the deleted document.
2. **Per-leaf `LocalId` compaction:**
   - When `leaf.next_local_id >= (config.max_leaf_size * 4).max(1024)`, surviving entries are compactly renumbered to `0..n` (`next_local_id = n`) and `ClusterDelta` updates `AttributeIndex` atomically.
3. **Structural mutations (leaf split, underflow merge/redistribution, or SPFresh LIRE reassignment):**
   - Surviving entries in affected leaves are compactly renumbered to `0..n` (`next_local_id = n`) via `PackedLeafNode::reset_and_encode` (updating `cf_doc_loc` as well), and `tree.rs` records a `ClusterDelta { removed, added }`:
     - `removed`: `(doc_id, old_leaf_id, old_local_id)` for every document deleted, moved out, or renumbered.
     - `added`: `(doc_id, new_leaf_id, new_local_id)` for every document inserted, moved in, or renumbered.

### 5.2 Applying `ClusterDelta` & Automatic Summary Downsampling

`AttributeIndex::apply_cluster_delta` processes `ClusterDelta` inside `TxContext`:

1. Deduplicates all affected `DocId`s from `delta.removed` and `delta.added`, batch-fetches their forward payloads via `tx.multi_get_doc_bytes` (honoring uncommitted `dirty_docs` in `TxContext`), and extracts their cluster keys once per `DocId`.
2. **Applies all `removed` entries before all `added` entries.** Ordering removals before additions is essential when a leaf is renumbered in place: if document A moves from `local_id = 5` to `local_id = 2` while document B moves from `local_id = 2` to `local_id = 0`, removing old assignments first ensures B's removal of `local_id = 2` does not erase A's newly assigned `local_id = 2`.
3. Every bit update goes through `TxContext::mutate_attr_posting(cluster_key, cluster_id, mutate)`:
   - Loads `cf_attr_postings[cluster_key || cluster_id]` into `dirty_attr_postings` if not already cached in the transaction.
   - Observes `was_empty` before `mutate(bm)` and `is_empty` after `mutate(bm)`:
     - **`was_empty && !is_empty`:** inserts `cluster_id` into `dirty_attr_clusters[cluster_key]`.
     - **`!was_empty && is_empty`:** removes `cluster_id` from `dirty_attr_clusters[cluster_key]`.
   - On `TxContext::commit()`, all dirty tree nodes, `cf_vectors`, `cf_doc_loc`, `cf_docs`, `cf_attr_clusters`, and `cf_attr_postings` (deleting keys whose bitmaps became empty) are written in a single atomic `rocksdb::WriteBatch`, and in-memory `internal_cache` / `leaf_parent` are updated only after `write_batch` succeeds.

### 5.3 End-to-End Mutation Flows (`upsert`, `patch`, `patch_batch`, `update_by_filter`, `delete`, `delete_batch`, `delete_by_filter`)

- **`Collection::upsert` / `patch` / `patch_batch`:** when an existing document's vector is unchanged (or `patch.embedding` is `None`), `ClusteringTree` is untouched and `AttributeIndex::diff_doc_in_place` / `diff_patched_in_place` updates only changed keys via a sorted two-pointer diff. When a vector changes and routes to the same leaf (`update_vector_in_tx` Case A), `local_id` is preserved and `AttributeIndex` diffs in place. Cross-leaf vector moves remove old keys and apply `rem_delta` before updating `cf_docs`, applying `ins_delta`, and inserting new keys at `new_loc`, followed by LIRE if splits occurred.
- **`Collection::delete` / `delete_batch`:** `delete` returns `Result<bool>` and `delete_batch` deduplicates IDs, runs `remove_many_in_tx` (updating each affected leaf once before `handle_underflow`), batch-fetches `cf_docs` via `multi_get_doc_bytes`, and removes deleted entries from `AttributeIndex` and `FtsIndex` before applying `ClusterDelta`.
- **`Collection::delete_by_filter` / `update_by_filter`:** resolve matching `DocId`s via `Collection::resolve_filter_doc_ids` (using a direct `FtsIndex` fast path for positive FTS-only filters or chunked leaf evaluation with `eval_leaves_masks` for general filters) and execute mutations in bounded atomic chunks (`512` docs per `TxContext` commit).

---

## 6. Filtered Search Execution

`Collection::search(query, opts, Some(filter))` dispatches to `search::search_filtered` ([`crates/izzydb/src/search.rs`](../../crates/izzydb/src/search.rs)). Unfiltered queries (`filter = None`) take `search::search` with zero filtering overhead.

### 6.1 Step 1: Cluster-Level Pruning (`AttributeIndex::prepare_filter`)

After `filter.validate(schema)` and `prepare_vector_cow`, `AttributeIndex::prepare_filter` evaluates the `Filter` AST against `cf_attr_clusters`, returning `PreparedFilter { cluster_candidates, root }`.

`ClusterCandidates` is either:
- `Subset(RoaringBitmap)` — a guaranteed **superset** of leaf `NodeId`s that could contain matching documents.
- `All` — the filter cannot prune clusters at the summary level (e.g. top-level `Not`, `IsNull`, or `NotEq` against a non-null scalar).

| `Filter` Node | `cf_attr_clusters` Operation | `ClusterCandidates` | `PreparedNode` |
|---|---|---|---|
| `Eq(attr, Null)` | Lookup `TAG_PRESENT` | `All` | `Not(Keys([(present_key, bm)]))` (or `True` if `TAG_PRESENT` absent) |
| `Eq(attr, val)` / `Contains(attr, val)` | Point lookup `get_attr_cluster_bitmap` | `Subset(bm)` (empty $\to$ `False`) | `Keys([(key, bm)])` |
| `NotEq(attr, Null)` | Lookup `TAG_PRESENT` | `Subset(bm)` | `Keys([(present_key, bm)])` |
| `NotEq(attr, val)` (non-null) | Evaluates inner `Eq` | `All` | `Not(inner)` |
| `In(attr, vals)` / `ContainsAny(attr, vals)` | Deduplicated `multi_get_attr_cluster_bitmaps` + bitwise `\|` | `Subset(union)` (or `All` if `vals` includes `Null`) | `Keys(non_empty_items)` (or `Or([Keys, is_null])`) |
| `NotIn(attr, vals)` | Evaluates inner `In` | `All` | `Not(inner)` |
| `Lt` / `Lte` / `Gt` / `Gte` | `scan_attr_cluster_range` within 3-byte prefix + bitwise `\|` | `Subset(union)` | `Keys(entries)` (caches keys + cluster bitmaps!) |
| `IsNull(attr)` | Lookup `TAG_PRESENT` | `All` | `Not(Keys([(present_key, bm)]))` (or `True`) |
| `IsNotNull(attr)` | Lookup `TAG_PRESENT` | `Subset(bm)` | `Keys([(present_key, bm)])` |
| `And(children)` | Intersect `Subset` bitmaps (`&=`), short-circuit on empty | `Subset` if any child is `Subset`; else `All` | `And(children)` (or `False` on short-circuit; `True` if empty) |
| `Or(children)` | Union `Subset` bitmaps (`\|=`) | `All` if any child is `All`; else `Subset(union)` | `Or(children)` (`False` if empty) |
| `Not(child)` | Evaluates `child` to build `PreparedNode` | `All` | `Not(child)` |

> **Important optimization for range filters:** `eval_cluster_node` caches every matching `(cluster_key, cluster_bitmap)` pair discovered during `scan_attr_cluster_range` inside `PreparedNode::Keys`. Later, `eval_leaves_masks` only requests `cf_attr_postings` for `(cluster_key, leaf_id)` pairs where `cluster_bm.contains(leaf_id)`, avoiding redundant RocksDB range scans on every leaf batch.

If `prepared.cluster_candidates.is_empty()`, `search_filtered` returns `Ok(Vec::new())` immediately without touching `cf_leaf_entries` or `cf_vectors`.

### 6.2 Step 2: Exhaustive-Scan Crossover

When `ClusterCandidates::Subset(bm)` satisfies:

```rust
bm.len() as usize <= opts.exhaustive_cluster_threshold // default: 8
```

the filter is selective enough that scanning all qualifying leaves directly is faster and yields **100% exact recall**:

1. Batch-fetches `PackedLeafNode` blobs for all `bm` leaves (`multi_get_leaf_nodes_pinned`).
2. Evaluates exact per-leaf `local_id` masks via `AttributeIndex::eval_leaves_masks`.
3. Iterates `0..view.entry_count`, collecting `view.doc_id(slot)` where `mask.contains(view.local_id(slot))`.
4. Sorts and deduplicates matching `DocId`s, batch-fetches full vectors from `cf_vectors`, computes exact `search_distance_bytes`, and returns the top `k`.

### 6.3 Step 3: Pruned Best-First Traversal & Dynamic Expansion

For broader filters (`All` or `bm.len() > exhaustive_cluster_threshold`):

1. **Ancestor pruning:** When `Subset(bm)` covers strictly less than 50% of leaves (`(bm.len() as usize) * 2 < total_leaves && meta.root_level > 0`), `search_filtered` walks upward from each leaf in `bm` using `leaf_parent` and `internal_cache` to construct `allowed_internal: RoaringBitmap`. During traversal, internal nodes not in `allowed_internal` and leaves not in `allowed_leaves` are never pushed onto the search heap.
2. **Best-first priority queue:** A `BinaryHeap<HeapEntry>` min-heap ordered by `(dist, node_id)` expands internal nodes lazily by exact `search_distance(metric, &q, child_centroid)`, yielding qualifying leaves in strict best-first order of centroid distance.
3. **Batched dynamic expansion:** Each round pops up to `batch_target` leaves (starting at `opts.beam_width`, then doubling: `beam_width`, `2 * beam_width`, `4 * beam_width`, ...), up to `max_expanded = opts.max_expanded_leaves.unwrap_or(opts.beam_width * 16).max(opts.beam_width)`.
4. **Batched intra-leaf mask evaluation (`AttributeIndex::eval_leaves_masks`):**
   - `collect_posting_keys` traverses `PreparedNode` for the batch's `leaf_ids`, emitting `make_posting_key(cluster_key, leaf_id)` **only** where `cluster_bm.contains(leaf_id)`.
   - Sorts/dedups keys and executes **one** `storage.multi_get_attr_posting_bitmaps(&needed_keys)`.
   - If `node_needs_complement` is true (any `PreparedNode::True` or `PreparedNode::Not` in the AST), builds `all_local` from `view.local_id(0..view.entry_count)` for each non-empty leaf so `Not(sub)` computes exact `all_local - sub`.
   - Evaluates `eval_prepared_leaf` (with short-circuiting on empty `And` intersections).
   - Leaves with an empty resulting `local_mask`—such as **false-positive clusters** under `And(Eq(a, 1), Eq(b, 2))` where one document in the leaf has `a = 1` and a different document has `b = 2`—are skipped immediately without RaBitQ scoring.
5. **Masked SIMD RaBitQ scoring (`score_leaf_rabitq_masked`):**
   - Maps the matching `local_id` `RoaringBitmap` into a slot-indexed bitset `slot_mask_words: Vec<u64>` (`entry_count.div_ceil(64)` words) by testing `local_mask.contains(view.local_id(slot))` for `slot in 0..view.entry_count`.
   - Calls `score_leaf_rabitq_masked` (`MASKED = true`):
     - In AVX-512 (`score_leaf_avx512`), 8-entry blocks whose 8-bit slot mask byte `m == 0` are skipped entirely (`if MASKED && m == 0 { continue; }`), and non-zero lanes only append candidates whose mask bit is set.
     - In POPCNT and scalar paths, entries with unset mask bits are skipped without computing popcounts or pushing candidates.
6. **Stopping condition & Stage 2 rerank:**
   - Expansion stops once `(candidates.len() >= target && probed_leaves >= opts.beam_width)` where `target = opts.rerank_candidates.unwrap_or(opts.k).max(opts.k)`, or when `heap.is_empty()`, or when `probed_leaves >= max_expanded`.
   - `finalize_candidates` shortlists top `rerank_candidates` (if set), batch-fetches exact vectors from `cf_vectors`, computes exact distances, and returns top `k`.

---

## 7. Consistency Invariants (`Collection::verify_invariants`)

`verify_collection_invariants` ([`crates/izzydb/src/verify.rs`](../../crates/izzydb/src/verify.rs)) exhaustively verifies both tree invariants and inverted-index parity:

1. **Schema parity:** `cf_meta["schema"]` exists and equals the in-memory `CollectionSchema`.
2. **Document bijection:** `cf_docs` has exactly `meta.num_vectors` entries, and its keys form a 1-to-1 bijection with `cf_doc_loc`, `cf_vectors`, and reachable leaf entries.
3. **Per-leaf `local_id` uniqueness:** Within every reachable leaf, all `local_id`s are unique and strictly `< leaf.next_local_id`.
4. **Full inverted-index reconstruction check:** Re-decodes every `cf_docs` entry, pairs it with its `cf_doc_loc`, rebuilds `expected_postings` and `expected_clusters` from scratch, and asserts:
   - Neither `cf_attr_postings` nor `cf_attr_clusters` contains any persisted empty `RoaringBitmap`.
   - `cf_attr_postings` on disk equals `expected_postings` byte-for-byte / set-for-set.
   - `cf_attr_clusters` on disk equals `expected_clusters` set-for-set.

---

## 8. Configuration & Performance Notes

- **Selectivity adaptation via `SearchOptions`:**
  - `exhaustive_cluster_threshold` (default `8`): raise to favor exact exhaustive evaluation over quantized traversal when filters match a small number of clusters; set to `0` to disable exhaustive crossover.
  - `max_expanded_leaves` (default `beam_width * 16`): controls the upper bound of leaves probed when searching for `target` matching candidates under sparse or unpruned filters (e.g., top-level `Not` / `IsNull`).
  - `rerank_candidates`: controls Stage 2 exact reranking over quantized candidates produced by masked RaBitQ scanning.
- **`Collection` vs `ClusteringTree`:** Standalone `ClusteringTree` only manages vectors (`cf_internal_nodes`, `cf_leaf_entries`, `cf_vectors`, `cf_doc_loc`) and does not maintain `cf_docs` or `AttributeIndex`. Use `Collection` whenever document attributes or filtering are needed.
- **Range-query cardinality:** `Lt`/`Lte`/`Gt`/`Gte` scan every distinct indexed value in `(attr_id, type_tag)` within the requested range in `cf_attr_clusters` and fetch matching postings per leaf batch. Discrete or moderately bucketed ordered attributes scan very few keys; extremely high-cardinality continuous floats over wide ranges will touch more keys during `prepare_filter` and `eval_leaves_masks`.
