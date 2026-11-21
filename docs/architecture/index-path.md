# Index Path

This document describes how `Collection` coordinates `ClusteringTree`, the forward document store (`cf_docs` + `cf_vectors`), and `AttributeIndex` (`cf_attr_clusters` + `cf_attr_postings`) on upserts, deletes, splits, merges, and SPFresh LIRE rounds.

## 1. Architecture & Shared RocksDB Layout

`Collection` ([`crates/izzydb/src/collection.rs`](../../crates/izzydb/src/collection.rs)) owns a shared `Arc<Storage>` ([`crates/izzydb-storage/src/storage.rs`](../../crates/izzydb-storage/src/storage.rs)) and coordinates:

1. **Document Store:** `cf_docs` (forward binary-encoded `Attributes`) and `cf_vectors` (`[f32; dim]` LE).
2. **`ClusteringTree`** ([`crates/izzydb/src/tree.rs`](../../crates/izzydb/src/tree.rs)): manages `cf_internal_nodes`, `cf_leaf_entries` (`PackedLeafNode` with RaBitQ codes and stable per-leaf `local_id`s), `cf_doc_loc` (`leaf_id || local_id`), and in-memory `internal_cache` + `leaf_parent` maps.
3. **`AttributeIndex`** ([`crates/izzydb/src/attr_index.rs`](../../crates/izzydb/src/attr_index.rs)): maintains downsampled cluster-level `RoaringBitmap<NodeId>`s in `cf_attr_clusters` and per-cluster document-level `RoaringBitmap<LocalId>`s in `cf_attr_postings`.

All mutations in an operation execute inside a single `TxContext` overlay (`dirty_internal`, `dirty_leaf`, `dirty_vectors`, `dirty_doc_loc`, `dirty_docs`, `dirty_attr_clusters`, `dirty_attr_postings`) and commit atomically in one `rocksdb::WriteBatch`.

## 2. Stable Per-Leaf `LocalId` Lifecycle

To keep attribute bitmaps independent of sorted `doc_ids` slot shifts on normal inserts/deletes:

- **Normal leaf insert:** allocates `local_id = leaf.next_local_id; leaf.next_local_id += 1;`. All existing entries in the leaf keep their `local_id` untouched, so only the newly inserted document updates `AttributeIndex`.
- **Normal leaf delete:** removes the entry and leaves surviving entries' `local_id`s untouched (resetting `next_local_id = 0` when the leaf becomes empty).
- **Per-leaf `LocalId` compaction:** when `leaf.next_local_id >= (config.max_leaf_size * 4).max(1024)`, surviving entries in the leaf are compactly renumbered to `0..n` (`next_local_id = n`) and `ClusterDelta` updates `AttributeIndex` in the same `TxContext`, keeping `RoaringBitmap`s within a single 16-bit container and preventing `u32` exhaustion under heavy churn.
- **Split, merge, redistribution, and LIRE:** renumber surviving entries compactly to `0..n`, reset `next_local_id = n`, update `cf_doc_loc`, and emit a `ClusterDelta { removed, added }` so `AttributeIndex::apply_cluster_delta` updates all affected `(cluster_key, cluster_id)` bitmaps in the same `TxContext`.

## 3. Upsert & Patch Flows (`Collection::upsert`, `patch`, `patch_batch`, `update_by_filter`)

1. **Validation & encoding:**
   - Validates `attributes` (or `DocumentPatch::attributes`) against `CollectionSchema` (`self.schema.validate_attributes`).
   - Validates and prepares `embedding` (when provided) via `prepare_vector` (L2-normalizing when `metric == Cosine`).
   - Merges patched attributes (`AttributeValue::Null` clears an attribute; omitted attributes are preserved) and encodes forward attributes into `cf_docs` (`encode_attributes`).
2. **Existing document fast paths (when `id` already exists):**
   - **Zero-touch vector fast path (`patch.embedding` is `None`, or `upsert` vector is bit-identical to `cf_vectors`):** skips loading or writing `PackedLeafNode`, centroids, RaBitQ codes, and `cf_vectors`, and diffs `AttributeIndex` in place at `(old_loc.leaf_id, old_loc.local_id)` via `diff_doc_in_place` / `diff_patched_in_place`.
   - **In-place same-leaf vector update (`update_vector_in_tx` Case A):** when the new vector routes back to `old_loc.leaf_id`, updates `cf_vectors`, recomputes centroid / RaBitQ codes while preserving `local_ids`, propagates centroid upward if drifted, and diffs `AttributeIndex` in place—avoiding spurious `handle_underflow` merges.
   - **Cross-leaf vector move (`update_vector_in_tx` Case B):** runs `remove_existing_in_tx` then `insert_in_tx` in the same `TxContext`, removing old `AttributeIndex` keys and applying `rem_delta` before updating `cf_docs`, applying `ins_delta`, and inserting new keys at `new_loc`.
3. **New document insert & route (`insert_in_tx`):**
   - Routes via `search::route_to_leaf_in_tx` (reading through `tx` overlay).
   - **Non-overflow (`new_count <= max_leaf_size`):** updates centroid (incremental mean or exact mean on `% 16 == 0`), re-encodes existing entries without changing `local_id`s when `new_count <= 16 || new_count % 16 == 0 || drifted`, propagates centroid upward when drift exceeds `centroid_drift_eps`, and compacts `LocalId`s if `next_local_id` reaches the compaction threshold.
   - **Overflow (`new_count > max_leaf_size`):** calls `split_node` (partitioning via `balanced_two_means`, cascading internal splits upward as needed), returning `(new_loc, ins_delta, Some(split_pair))`.
4. **Attribute & FTS index update & atomic commit:**
   - Skips `FtsIndex` completely when no FTS-enabled attribute changed on an existing document.
   - Applies coalesced `ClusterDelta` (`net_changes`, removals before additions) to `AttributeIndex` via `tx.mutate_attr_posting`, which automatically maintains `cf_attr_clusters` on `empty <-> non-empty` transitions.
   - Commits `tx` atomically and updates `internal_cache`, `leaf_parent`, and `fts_doc_lens_cache`.
   - `update_by_filter` resolves matching `DocId`s via `Collection::resolve_filter_doc_ids` and applies patches in bounded chunks (`512` docs per `TxContext` commit).
5. **SPFresh LIRE (`lire::run_lire_with_hook`):**
   - When split(s) occurred and `enable_lire && root_level > 0`, runs bounded LIRE (`MAX_LIRE_ROUNDS = 4`) and any maintenance splits/underflows. Each step runs inside a `TxContext` with an `on_delta` hook calling `AttributeIndex::apply_cluster_delta` before `tx.commit()`.

## 4. Delete Flows (`Collection::delete`, `delete_batch`, `delete_by_filter`)

1. **`Collection::delete(id) -> Result<bool>`:** calls `tree::remove_existing_in_tx`, removes `id` from `AttributeIndex` and `FtsIndex`, stages `tx.delete_doc(id)`, applies any underflow/compaction `ClusterDelta`, and commits `tx` atomically. Returns `false` if `id` was absent. (Standalone `ClusteringTree::remove(id) -> Result<bool>` provides vector-only deletion by `DocId`.)
2. **`Collection::delete_batch(ids) -> Result<u64>`:** deduplicates `ids`, calls `tree::remove_many_in_tx` (updating each affected leaf once before running `handle_underflow` and `LocalId` compaction), batch-fetches `cf_docs` via `multi_get_doc_bytes`, purges `AttributeIndex` and `FtsIndex` before applying `ClusterDelta`, and commits atomically.
3. **`Collection::delete_by_filter(filter) -> Result<u64>`:** resolves matching `DocId`s via `resolve_filter_doc_ids` and deletes in bounded chunks (`512` docs per `TxContext` commit).
