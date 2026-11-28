# Index Path

This document describes how `Collection` coordinates `SegmentStore`, `AnnIndex`, `AttributeIndex`, `FtsIndex`, and detached columns during live mutations (`upsert`, `patch`, `delete`) and background maintenance (`seal`, `compact`, `vacuum_ann`).

Implementation sources:
- Top-level write coordinator: [`crates/izzydb/src/collection/mod.rs`](../../crates/izzydb/src/collection/mod.rs)
- Append-only segment allocator & `RowIndexHook`: [`crates/izzydb/src/segments.rs`](../../crates/izzydb/src/segments.rs)
- Global ANN tree over `RowId`s: [`crates/izzydb/src/tree/mod.rs`](../../crates/izzydb/src/tree/mod.rs)
- Value-major segment attribute index: [`crates/izzydb/src/attr_index.rs`](../../crates/izzydb/src/attr_index.rs)
- Segment-scoped FTS index: [`crates/izzydb/src/fts_index/mod.rs`](../../crates/izzydb/src/fts_index/mod.rs)

*(For offline compaction and forwarding details, see [`segments-and-compaction.md`](segments-and-compaction.md); for detached columns, see [`detached-columns.md`](detached-columns.md).)*

---

## 1. Architecture & Write Hooks

`Collection` owns `Arc<Storage>` and executes every mutation inside a single `TxContext` committed as one atomic `rocksdb::WriteBatch`:

1. **`SegmentStore`:** Appends new rows to `Manifest.open_segment` (`RowId = (seg, local)`), writes `cf_rows` (`RowView` v3), `cf_id_map[doc_id] = row_id`, `cf_row_doc_ids`, and optional `cf_vectors[row_id]`. Seals segments at `SegmentConfig.seal_rows` (building `cf_columns` unless `defer_column_build = true`).
2. **`AttributeIndex`** (`RowIndexHook`): Inserts segment-scoped `SegPosting` entries (`cf_attr_postings`), updates live `default[0x06]` counts, and widens `default[0x12]` per-segment zone maps. On delete/replace, decrements `0x06` only (`cf_attr_postings` and zones are never shrunk in place).
3. **`FtsIndex`** (`RowIndexHook`): Analyzes FTS-enabled attributes directly into `(seg, chunk)`-keyed `cf_fts_skips` (`FtsChunk` v4), `cf_fts_terms` (inline rare terms), `cf_fts_positions`, and `default[0x05]` doc lens. On delete/replace, decrements `TermRecord.doc_freq` and `FtsMeta` stats without touching sealed chunks.
4. **`AnnIndex` (`ClusteringTree`, when `has_vectors`):** Inserts new `(RowId, quantized_code)` entries into `cf_leaf_entries` and updates internal nodes (`default[0x02]`). Stale entries left behind by deleted/replaced rows increment `AnnMeta.stale_rows` and are cleaned up opportunistically on leaf split/re-encode or incrementally via `vacuum_ann`.

---

## 2. Copy-on-Write (`upsert`, `patch`, `delete`)

### `upsert(id, embedding, attributes)`
1. Validates vector (`Some` required when `has_vectors`, rejected or ignored cleanly when vectorless) and `Inline` vs `Detached` attributes against `CollectionSchema`.
2. Allocates a fresh `new_row` via `SegmentStore::alloc_row`.
3. If `id` already exists at `old_row`:
   - Reads `old_row` from `cf_rows` (`RowView`) and runs delete hooks (tombstoning `old_row` in `default[0x10 || old_seg]`, decrementing `0x06` and FTS `doc_freq`, decrementing `0x07` detached counts, incrementing `AnnMeta.stale_rows` if `old_row` had a vector).
   - `upsert` replaces the whole document (`carry_over = false`): any unmentioned detached column resets to `Null` at `new_row` unless provided in `attributes`.
4. Writes `cf_rows[new_row]` (`Inline` attrs), `cf_id_map[id] = new_row`, `cf_row_doc_ids`, optional `cf_vectors`, runs `AttributeIndex::on_insert` + `FtsIndex::on_insert`, writes any detached values in `attributes` to `cf_detached` / `cf_detached_postings` / `0x07`, and inserts `new_row` into `AnnIndex` (running SPFresh LIRE rounds if splits occur).

### `patch(id, patch)` / `patch_batch`
1. Looks up existing `old_row` via `cf_id_map` (`DocumentNotFound` if missing).
2. Reads `old_row`'s `RowView` (and old vector from `cf_vectors` when `patch.embedding` is `None` and `has_vectors`).
3. Merges `patch.attributes` onto old `Inline` attributes, allocates `new_row`, and tombstones `old_row` as above (`carry_over = true`, preserving unmodified detached columns on `new_row`).
4. **FTS diffing:** if FTS attributes are unchanged, raw FTS token analysis can still follow standard segment row hooks cleanly while inline/vector state moves atomically in `TxContext`.
5. `patch_batch` validates and applies all patches inside one `TxContext`; `update_by_filter` resolves matching `DocId`s first and applies patches in bounded atomic chunks (`512` docs/commit).

### `delete(id)` / `delete_batch` / `delete_by_filter`
- Removes `id` from `cf_id_map`, adds `old_row.local()` to `default[0x10 || old_seg]`, decrements `0x06`, `0x07`, and FTS `doc_freq` / `FtsMeta`, increments `AnnMeta.stale_rows` (when `old_had_vector`), and updates `Manifest.live_rows`.
- Returns `false` if absent (`delete`), total deleted count (`delete_batch` / `delete_by_filter`).

---

## 3. ANN Tree Maintenance & Forwarding Resolution

- **Slot lifecycle in `PackedLeafNode` v3:** Leaf slots store `row_ids: [RowId; padded_slots]`. On `insert_vector`, any slot whose `RowId` is no longer live (after resolving `Forwarding` segments via `SegmentTableSnapshot::resolve_forwarding`) is immediately reusable before allocating a new slot.
- **Splits & SPFresh LIRE (`lire::run_lire_with_hook`):** Re-pack surviving slots to `0..n` without touching `AttributeIndex` or `FtsIndex` (because `RowId`s live in segment space, NOT leaf space—eliminating v2's cross-leaf `ClusterDelta` cascade!).
- **`vacuum_ann(VacuumBudget)`:** Incrementally scans `cf_leaf_entries` from `AnnMeta.vacuum_cursor`, drops stale slots, rewrites `Forwarding` `RowId`s to their compaction target `(T, new_local)`, merges underflowing leaves, and advances `AnnMeta.vacuum_epoch` upon full-pass completion so `Forwarding` segments can be retired.
