# Index Path

This document describes how `Collection` coordinates `ClusteringTree`, the forward document store (`cf_docs` + `cf_vectors`), and `AttributeIndex` (`cf_attr_clusters` + `cf_attr_postings` + attribute zones in `default`) on upserts, deletes, splits, merges, and SPFresh LIRE rounds.

## 1. Architecture & Shared RocksDB Layout

`Collection` ([`crates/izzydb/src/collection.rs`](../../crates/izzydb/src/collection.rs)) owns a shared `Arc<Storage>` ([`crates/izzydb-storage/src/storage.rs`](../../crates/izzydb-storage/src/storage.rs)) and coordinates:

1. **Document Store:** `cf_docs` (v2 offset-indexed forward `Attributes`) and `cf_vectors` (`[f32; dim]` LE).
2. **`ClusteringTree`** ([`crates/izzydb/src/tree.rs`](../../crates/izzydb/src/tree.rs)): manages internal nodes in `default`, `cf_leaf_entries` (slot-indexed `PackedLeafNode` with `live_mask`, precomputed `code_popcounts`, and 8-wide SIMD-tiled RaBitQ codes), `cf_doc_loc` (`leaf_id || local_id || fts_doc_id` — 12 B), and in-memory `internal_cache` + `leaf_parent` maps.
3. **`AttributeIndex`** ([`crates/izzydb/src/attr_index.rs`](../../crates/izzydb/src/attr_index.rs)): maintains cluster-level `RoaringBitmap<NodeId>`s in `cf_attr_clusters`, cluster-first `LeafPosting` slot sets in `cf_attr_postings`, and cached per-leaf `(min, max)` bounds (`AttrZoneMap`) in `default`.

All mutations in an operation execute inside a single `TxContext` overlay (`dirty_internal`, `dirty_leaf`, `dirty_vectors`, `dirty_doc_pos`, `dirty_fts_doc_map`, `dirty_docs`, `dirty_attr_clusters`, `dirty_attr_postings`, and dirty FTS maps) and commit atomically in one `rocksdb::WriteBatch`.

## 2. Slot-Indexed Per-Leaf `LocalId` Lifecycle

Because `PackedLeafNode` is **slot-indexed (`local_id == slot`)** and tracks occupancy via `live_mask: Vec<u64>`, entries keep their slot across normal inserts and deletes without a separate `local_ids` array:

- **Normal leaf insert:** `alloc_slot` reuses the lowest dead `0` bit in `live_mask` when `live_count < slot_cap`, or appends at `slot_cap` (`slot_cap += 1`). Existing live slots do not move, so only the newly inserted document updates `AttributeIndex` and `FtsRevBlock`.
- **Normal leaf delete:** clears the bit in `live_mask` and trims any trailing dead slots (`slot_cap`), leaving all lower surviving slots unchanged.
- **Leaf compaction (`needs_leaf_compaction`):** triggers when `slot_cap > max_leaf_size` or sparse deletes leave `slot_cap > 32 && live_count * 2 <= slot_cap`, re-packing surviving slots to `0..n` and updating `AttributeIndex` + `FtsRevBlock` via `ClusterDelta`.
- **Split, merge, redistribution, and LIRE:** re-pack surviving entries compactly to `0..n` via `PackedLeafNode::reset_and_encode` (preserving `fts_doc_id` in `cf_doc_loc` via `tx.put_doc_pos`) and emit `ClusterDelta { removed, added }` so `AttributeIndex::apply_cluster_delta` atomically updates affected postings in `cf_attr_postings`, cluster summaries in `cf_attr_clusters`, and attribute zones in `default`.

## 3. Upsert & Patch Flows (`Collection::upsert`, `patch`, `patch_batch`, `update_by_filter`)

1. **Validation & encoding:**
   - Validates `attributes` (or `DocumentPatch::attributes`) against `CollectionSchema` (`self.schema.validate_attributes`).
   - Validates and prepares `embedding` (when provided) via `prepare_vector` (L2-normalizing when `metric == Cosine`).
   - Merges patched attributes (`AttributeValue::Null` clears an attribute; omitted attributes are preserved) and encodes v2 forward attributes with an `AttrId`-sorted offset directory into `cf_docs` (`encode_attributes`).
2. **Existing document fast paths (when `id` already exists):**
   - **Zero-touch vector fast path (`patch.embedding` is `None`, or `upsert` vector is bit-identical to `cf_vectors`):** skips loading or writing `PackedLeafNode`, centroids, RaBitQ codes, and `cf_vectors`, and diffs `AttributeIndex` in place at `(old_loc.leaf_id, old_loc.local_id)` via `diff_doc_in_place` / `diff_patched_in_place`.
   - **In-place same-leaf vector update (`update_vector_in_tx` Case A):** when the new vector routes back to `old_loc.leaf_id`, updates `cf_vectors`, recomputes centroid / RaBitQ codes (`encode_into_with_popcnt` on the fast path) while preserving `old_loc.local_id`, propagates centroid upward if drifted, and diffs `AttributeIndex` in place—avoiding spurious `handle_underflow` merges.
   - **Cross-leaf vector move (`update_vector_in_tx` Case B):** runs `remove_existing_in_tx` then `insert_in_tx` in the same `TxContext` (preserving any existing `fts_doc_id` across `cf_doc_loc` reconciliation), removing old `AttributeIndex` keys and applying `rem_delta` before updating `cf_docs`, applying `ins_delta`, and inserting new keys at `new_loc`, followed by LIRE if splits occurred.
3. **New document insert & route (`insert_in_tx`):**
   - Routes via `search::route_to_leaf_in_tx` (reading through `tx` overlay).
   - **Non-overflow (`new_count <= max_leaf_size`):** updates centroid (incremental mean or exact mean on `% 16 == 0`), re-encodes live slots in place when `new_count <= 16 || new_count % 16 == 0 || drifted` (or encodes just the new slot via `encode_into_with_popcnt`), and propagates centroid upward when drift exceeds `centroid_drift_eps`.
   - **Overflow (`new_count > max_leaf_size`):** calls `split_node` (partitioning via `balanced_two_means`, cascading internal splits upward as needed), returning `(new_loc, ins_delta, Some(split_pair))`.
4. **Attribute, Zone Map & FTS index update & atomic commit:**
   - Skips `FtsIndex` completely when no FTS-enabled attribute changed on an existing document.
   - Applies coalesced `ClusterDelta` (`net_changes`, removals before additions) to `AttributeIndex` via `tx.mutate_attr_posting`, which maintains `cf_attr_clusters` on `empty <-> non-empty` transitions.
   - `TxContext::commit` atomically updates attribute zones in `default` (recomputing only when a leaf's min/max posting key is deleted; skipping re-writes when bounds did not change) and reconciles `cf_doc_loc` + `FtsRevBlock` in `default`.
   - Commits `tx` in one `WriteBatch`, then writes through to in-memory `internal_cache`, `leaf_parent`, `attr_zones_cache`, `fts_rev_map_cache`, and `fts_doc_lens_cache`.
   - `update_by_filter` resolves matching `DocId`s via `Collection::resolve_filter_doc_ids` and applies patches in bounded chunks (`512` docs per `TxContext` commit).
5. **SPFresh LIRE (`lire::run_lire_with_hook`):**
   - When split(s) occurred and `enable_lire && root_level > 0`, runs bounded LIRE (`MAX_LIRE_ROUNDS = 4`) and any maintenance splits/underflows. Each step runs inside a `TxContext` with an `on_delta` hook calling `AttributeIndex::apply_cluster_delta` before `tx.commit()`.

## 4. Delete Flows (`Collection::delete`, `delete_batch`, `delete_by_filter`)

1. **`Collection::delete(id) -> Result<bool>`:** calls `tree::remove_existing_in_tx`, removes `id` from `AttributeIndex` and `FtsIndex`, stages `tx.delete_doc(id)`, applies any underflow/compaction `ClusterDelta`, and commits `tx` atomically. Returns `false` if `id` was absent. (Standalone `ClusteringTree::remove(id) -> Result<bool>` provides vector-only deletion by `DocId`.)
2. **`Collection::delete_batch(ids) -> Result<u64>`:** deduplicates `ids`, calls `tree::remove_many_in_tx` (updating each affected leaf once before running `handle_underflow` and `needs_leaf_compaction`), batch-fetches `cf_docs` via `multi_get_doc_bytes`, purges `AttributeIndex` and `FtsIndex` before applying `ClusterDelta`, and commits atomically.
3. **`Collection::delete_by_filter(filter) -> Result<u64>`:** resolves matching `DocId`s via `resolve_filter_doc_ids` and deletes in bounded chunks (`512` docs per `TxContext` commit).
