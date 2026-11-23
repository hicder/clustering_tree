# Plan: Global Per-Doc Vector Storage (`cf_vectors`)

## Goal

Replace the per-leaf full-precision vector blob in `cf_leaf_vectors` (`leaf_id -> PackedLeafVectors`) with a global per-document entry in `cf_vectors` (`doc_id -> [f32; dim]`).

### Scope
- **Changed:** `cf_leaf_vectors` $\to$ `cf_vectors` (`doc_id` 8-byte BE $\to$ raw little-endian `[f32; dim]`).
- **Unchanged:** `cf_leaf_entries` (`PackedLeafNode` word-major SoA, 1 blob per leaf for Stage 1 SIMD scanning), `cf_internal_nodes`, `cf_doc_loc`, and `cf_meta`.
- **Compatibility:** Backward-incompatible on-disk schema change (existing databases with `cf_leaf_vectors` are not migrated).

### Motivation
- Eliminates O(leaf_size * dim) RocksDB write amplification on `add_leaf` and `remove_leaf` (e.g., at `max_leaf_size = 160` and `dim = 1536`, per-insert vector writes drop from ~984 KB to 6 KB).
- Avoids reading any full-precision vectors on the common `add_leaf` fast path (non-reencode, non-split).
- Eliminates vector key moves on leaf splits, merges/redistributions, and SPFresh LIRE reassignments.
- Reduces Stage 2 `search_with_rerank` read amplification and block cache pressure by fetching only the `rerank_candidates` vectors instead of full leaf blobs across the beam.

---

## 1. Storage Schema & Key Layout (`crates/izzydb-storage/src/storage.rs`)

- **Column family rename:**
  - Replace `CF_LEAF_VECTORS = "cf_leaf_vectors"` with `CF_VECTORS = "cf_vectors"`.
  - Keep `point_lookup_cf_opts(&block_cache, true)` (bloom filter + `DBCompressionType::None` for zero-copy `DBPinnableSlice` reads).
- **Key / value format:**
  - **Key:** `node_key(doc_id)` (`doc_id.to_be_bytes()`, 8 bytes).
  - **Value:** flat little-endian `[f32; dim]` (`dim * 4` bytes, no header).
- **Initialization (`Storage::open`):**
  - Remove the initial empty `PackedLeafVectors` write for root leaf `1`.
- **`Storage` read methods:**
  - Simplify `Storage::get_vector(&self, doc_id: DocId) -> Result<Option<Vec<f32>>>` to a single `get_pinned_cf(CF_VECTORS, node_key(doc_id))` + `decode_f32_slice` (no `cf_doc_loc` lookup needed).
  - Remove `get_leaf_vectors`, `multi_get_leaf_vectors_pinned`, and `all_leaf_vectors`.
  - Add:
    - `multi_get_vectors_pinned(&self, sorted_doc_ids: &[DocId]) -> Result<Vec<Option<DBPinnableSlice<'_>>>>` backed by `batched_multi_get_pinned(CF_VECTORS, sorted_doc_ids, true)`.
    - `all_vectors(&self) -> Result<Vec<(DocId, Vec<f32>)>>` for invariant verification.
- **`StorageBatch` write methods:**
  - Replace `put_leaf_vectors` / `delete_leaf_vectors` with:
    - `put_vector(&mut self, doc_id: DocId, vec: &[f32])`
    - `delete_vector(&mut self, doc_id: DocId)`
- **`TxContext` read-your-own-writes overlay:**
  - Replace `leaf_vectors: HashMap<NodeId, Option<PackedLeafVectors>>` and `dirty_leaf_vecs: HashSet<NodeId>` with:
    - `dirty_vectors: HashMap<DocId, Option<Vec<f32>>>`
  - Add `put_vector(&mut self, doc_id: DocId, vec: Vec<f32>)` and `delete_vector(&mut self, doc_id: DocId)`.
  - Add `get_vectors_flat(&self, doc_ids: &[DocId]) -> Result<Vec<f32>>`:
    - Allocates a flat `doc_ids.len() * dim` output buffer aligned with `doc_ids`.
    - Fills any entries present in `dirty_vectors` directly (returning `CorruptedData` if marked deleted `None`).
    - Sorts the remaining `doc_ids` ascending, fetches them in a single `storage.multi_get_vectors_pinned(...)` call, validates `slice.len() == dim * 4`, and decodes into the corresponding slots.
  - Flush `dirty_vectors` in `TxContext::commit` via `batch.put_vector` / `batch.delete_vector`.

---

## 2. Node Encoding & Re-quantization (`crates/izzydb-storage/src/node/mod.rs`)

- Remove `PackedLeafVectors`, `LeafVectorsView`, and `LEAF_VECS_HEADER_LEN`.
- Add `encode_f32_slice(vals: &[f32]) -> Vec<u8>` (reusing `write_f32_le`).
- Update `PackedLeafNode::re_encode_all` signature to accept `(doc_ids: &[DocId], flat_vectors: &[f32], quantizer: &RabitqQuantizer, dim: usize, scratch: &mut EncodeScratch)`:
  - Validates `flat_vectors.len() == doc_ids.len() * dim`.
  - Updates `self.doc_ids`, `self.res_norm_sqs`, `self.scales`, and word-major `self.codes` relative to `self.centroid`.

---

## 3. Write, Split, Merge & LIRE Paths (`crates/izzydb/src/tree/mod.rs`, `crates/izzydb/src/lire.rs`)

### `ClusteringTree::add_leaf` (`crates/izzydb/src/tree/mod.rs`)
- Stage `tx.put_vector(id, vec.clone())` and `tx.put_doc_loc(id, target_leaf)`.
- Do **not** read full vectors up front; derive `new_count = leaf.entry_count() + 1` (existing `id` is already removed before routing if upserting).
- **Non-overflow (`new_count <= max_leaf_size`):**
  - `new_count == 1`: set centroid to `vec`, rotate, encode single entry, `leaf.upsert_entry(...)` (0 vector reads).
  - Fast path (`new_count > 16`, `new_count % 16 != 0`, and `!drifted`): update centroid with `incremental_mean`, rotate, encode single entry, `leaf.upsert_entry(...)` (**0 vector reads from `cf_vectors`**).
  - Re-encode path (`new_count <= 16`, `new_count % 16 == 0`, or `drifted`): build sorted `doc_ids` including `id`, fetch `flat_vecs = tx.get_vectors_flat(&doc_ids)?`, recompute exact mean centroid if `new_count % 16 == 0`, and call `leaf.re_encode_all(&doc_ids, &flat_vecs, ...)`.
- **Overflow (`new_count > max_leaf_size`):**
  - Insert `id` into `leaf` (or pass `(id, &vec)` alongside `leaf`), stage `tx.put_leaf_node(target_leaf, leaf)`, and call `split_node`.

### `ClusteringTree::remove_doc_internal` (`crates/izzydb/src/tree/mod.rs`)
- Stage `tx.delete_vector(id)` and `tx.delete_doc_loc(id)`; remove `id` from `leaf` via `leaf.remove_entry(id, words)`.
- Compute `new_count = leaf.entry_count()`.
- If `new_count == 0`, zero out `leaf.centroid` and `leaf.rotated_centroid` (0 vector reads).
- If `will_underflow`, persist `leaf` to `tx` and delegate directly to `handle_underflow` (avoids double-fetching vectors).
- Otherwise, fetch remaining vectors via `tx.get_vectors_flat(&leaf.doc_ids)?`, recompute exact mean centroid, and re-encode if `new_count <= 16` or drifted.

### `split_node` (`crates/izzydb/src/tree/mod.rs`)
- Fetch `flat_vecs = tx.get_vectors_flat(&leaf.doc_ids)?`.
- Run `balanced_two_means`, partition into `(left_doc_ids, left_vecs)` and `(right_doc_ids, right_vecs)`.
- Re-encode `left_leaf` and `right_leaf`, update `cf_leaf_entries` and `cf_doc_loc` for `right_doc_ids`.
- **Zero writes to `cf_vectors`** (vectors stay at their `doc_id` keys).

### `handle_underflow` (`crates/izzydb/src/tree/mod.rs`)
- Remove all `tx.delete_leaf_vectors` and `tx.put_leaf_vectors` calls.
- Fetch `left_vecs = tx.get_vectors_flat(&left_leaf.doc_ids)?` and `sib_vecs = tx.get_vectors_flat(&sib_leaf.doc_ids)?`.
- On merge or redistribution, update `PackedLeafNode`s in `cf_leaf_entries` and moved entries in `cf_doc_loc`.
- **Zero writes to `cf_vectors`**.

### SPFresh LIRE (`crates/izzydb/src/lire.rs`)
- Read `doc_ids` from `u_leaf`, `v_leaf`, and `npa_neighbors` (`cf_leaf_entries`).
- Batch-fetch all required full vectors once via `storage.multi_get_vectors_pinned(&sorted_unique_doc_ids)` into an in-memory map `HashMap<DocId, Vec<f32>>` (or flat buffer).
- Apply `moves` by updating `from_leaf.doc_ids` / `to_leaf.doc_ids` (in `cf_leaf_entries`) and `tx.put_doc_loc(doc_id, to_leaf)`.
- For each `touched_leaf`, compute the new centroid and call `leaf.re_encode_all` using the fetched vectors (falling back to `tx.get_vectors_flat` for any untouched docs in a touched neighbor leaf).
- **Zero writes to `cf_vectors`**.

---

## 4. Search Path & Reranking (`crates/izzydb/src/search/mod.rs`, `crates/izzydb-core/src/simd.rs`, `crates/izzydb-rabitq/src/simd.rs`)

- In `search` (`crates/izzydb/src/search/mod.rs`):
  - After selecting the top `shortlist_len` candidates from Stage 1, sort `candidates` by `doc_id` ascending.
  - Fetch full vectors with one `storage.multi_get_vectors_pinned(&sorted_doc_ids)?` call.
  - Validate each slice has length `dim * 4` and score directly via `search_distance_bytes(config.metric, &q, slice)` (zero-copy over `DBPinnableSlice`).
- In `crates/izzydb-rabitq/src/simd.rs`:
  - Remove now-unused `leaf_id` and `slot` fields from `LeafCandidate` (simplifying SIMD candidate emission).

---

## 5. Invariant Verification & Documentation (`crates/izzydb/src/verify/mod.rs`, `docs/`)

- **`verify_invariants` (`crates/izzydb/src/verify/mod.rs`):**
  - Replace `storage.all_leaf_vectors()` with `storage.all_vectors()`.
  - Check `disk_vectors.len() == disk_meta.num_vectors as usize`.
  - For every reachable leaf, verify every `doc_id` in `leaf.doc_ids` exists in `disk_vectors`, has `len == config.dim`, and has all finite components.
  - Verify `disk_vectors.len() == seen_docs.len()` (no orphaned keys in `cf_vectors`).
- **Docs (`docs/key-encoding.md`):**
  - Update the column family table and invariants to document `cf_vectors` (`doc_id` 8-byte BE $\to$ raw `[f32; dim]` LE) and remove the `PackedLeafVectors` layout section.

---

## 6. Testing & Verification

1. Run `cargo test` to verify all unit and integration tests (`crates/izzydb/tests/tree_tests.rs`, `crates/izzydb-storage/src/node/mod.rs`, `crates/izzydb-rabitq/src/simd.rs`), including `verify_invariants` and `get_vector` upsert/delete/reopen checks.
2. Add targeted tests covering `add_leaf` fast-path vs. re-encode, splits, underflow merges/redistributions, and LIRE reassignments under `cf_vectors`.
3. Run `cargo run --release -p izzydb-bench --bin recall_bench` to verify recall parity and measure insert + `search_with_rerank` latency.
