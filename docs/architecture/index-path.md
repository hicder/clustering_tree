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
- **Split, merge, redistribution, and LIRE:** renumber surviving entries compactly to `0..n`, reset `next_local_id = n`, update `cf_doc_loc`, and emit a `ClusterDelta { removed, added }` so `AttributeIndex::apply_cluster_delta` updates all affected `(cluster_key, cluster_id)` bitmaps in the same `TxContext`.

## 3. Upsert Flow (`Collection::upsert`)

1. **Validation & encoding:**
   - Validates `attributes` against `CollectionSchema` (`self.schema.validate_attributes`).
   - Validates and prepares `embedding` via `prepare_vector` (L2-normalizing when `metric == Cosine`).
   - Encodes forward attributes into `cf_docs` payload (`encode_attributes`).
2. **Existing document removal (if `id` already exists):**
   - Calls `tree::remove_existing_in_tx(&mut tx, id, ...)`, which removes `id` from its old leaf (triggering `handle_underflow` if needed) and returns `(old_loc, rem_delta)`.
   - Removes `id`'s old attribute keys at `(old_loc.leaf_id, old_loc.local_id)` and applies `rem_delta` (any other docs moved/renumbered by underflow) to `AttributeIndex`.
3. **Insert & route:**
   - Stages `tx.put_doc_bytes(id, new_doc_bytes)`.
   - Calls `tree::insert_in_tx(&mut tx, id, &vec, ...)`:
     - Routes via `search::route_to_leaf_in_tx` (reading through `tx` overlay).
     - **Non-overflow (`new_count <= max_leaf_size`):** updates centroid (incremental mean or exact mean on `% 16 == 0`), re-encodes existing entries without changing `local_id`s when `new_count <= 16 || new_count % 16 == 0 || drifted`, propagates centroid upward when drift exceeds `centroid_drift_eps`, and returns `(new_loc, empty_delta, None)`.
     - **Overflow (`new_count > max_leaf_size`):** calls `split_node` (partitioning via `balanced_two_means`, cascading internal splits upward as needed), returning `(new_loc, ins_delta, Some(split_pair))`.
4. **Attribute index update & atomic commit:**
   - Applies `ins_delta` (removals before additions) to `AttributeIndex` via `tx.mutate_attr_posting`, which automatically maintains the downsampled `cf_attr_clusters` bit on `empty <-> non-empty` transitions.
   - Inserts `id`'s new attributes at `(new_loc.leaf_id, new_loc.local_id)`.
   - Commits `tx` atomically and updates `internal_cache` and `leaf_parent`.
5. **SPFresh LIRE (`lire::run_lire_with_hook`):**
   - When a split occurred and `enable_lire && root_level > 0`, runs bounded LIRE (`MAX_LIRE_ROUNDS = 4`) and any maintenance splits/underflows. Each step runs inside a `TxContext` with an `on_delta` hook calling `AttributeIndex::apply_cluster_delta` before `tx.commit()`.

## 4. Delete Flow (`Collection::delete`)

1. Calls `tree::remove_existing_in_tx(&mut tx, id, ...)`; returns `Ok(())` if `id` is absent.
2. Decodes `id`'s old `cf_docs` attributes and removes them from `AttributeIndex` at `(old_loc.leaf_id, old_loc.local_id)`.
3. Stages `tx.delete_doc(id)`, applies any underflow `ClusterDelta` to `AttributeIndex`, and commits `tx` atomically.
