# Detached Columns & Whole-Column Ingest

Detached columns store attributes **columnarly outside `cf_rows`**, allowing whole-column replacement (`M rows/s` throughput, `< 100 ms` atomic publish) and point updates without triggering segment copy-on-write (CoW) or re-indexing vectors/FTS.

Implementation sources:
- Schema `ColumnStorage::Detached`: [`crates/izzydb-core/src/schema/mod.rs`](../../crates/izzydb-core/src/schema/mod.rs)
- Detached build, concurrent catch-up, atomic publish & point writes: [`crates/izzydb/src/detached/mod.rs`](../../crates/izzydb/src/detached/mod.rs)
- Filter / facet / `OrderBy` / `get` readers (`AttrReader`): [`crates/izzydb/src/attr_reader.rs`](../../crates/izzydb/src/attr_reader.rs)
- Compaction-time detached preservation & GC: [`crates/izzydb/src/compaction.rs`](../../crates/izzydb/src/compaction.rs) & [`crates/izzydb-storage/src/storage/mod.rs`](../../crates/izzydb-storage/src/storage/mod.rs)

---

## 1. Inline vs. Detached Attributes

Every schema attribute declares `ColumnStorage`:

| Mode | Stored In | Columnar Reads (`cf_columns`)? | Whole-Column Replace? | Allowed Features |
|---|---|---|---|---|
| `Inline` (default) | `cf_rows` (`RowView`) | Sealed segments only (per-segment blocks) | No (row CoW) | Vector payload, FTS, `filterable`, `facet`, `order_by` |
| `Detached` | `cf_detached` (always columnar, all segments) | **Always** (`ColumnBlock` keyed by `attr_id \|\| gen \|\| seg \|\| block`) | **Yes** (`detached_ingest` + `publish_detached`) | `filterable`, `facet`, `order_by`, `RankBy` (`fts` disallowed) |

Detached columns are ideal for signals updated orthogonally to documents—for example: offline popularity scores, PageRank/CTR weights, moderation flags, price tiers, or ML feature refreshes over millions of rows.

---

## 2. Persistence Layout & Generations

Every detached attribute has a persistent metadata record `DetachedMeta` under `default[0x13 || attr_id BE]`:

```text
current_gen: u32          // active readable generation (initially 0)
pending_gen: u32          // staged generation during in-flight build (0 = idle)
retired_below: u32        // generations < retired_below are eligible for RocksDB compaction-filter GC
built_against_watermark: RowId
```

All detached data is versioned by `(attr_id, gen)`:

| Keyspace | Key | Value | Purpose |
|---|---|---|---|
| `cf_detached` | `attr_id BE \|\| gen BE \|\| seg BE \|\| block BE` (14 B) | `ColumnBlock` (256-slot columnar block) | Forward column values for generation `gen` |
| `cf_detached_postings` | `attr_id BE \|\| gen BE \|\| tag \|\| encoded_val \|\| seg BE` | `SegPosting` (`Sparse` / `Roaring`) | Inverted index when `filterable = true` |
| `default[0x07]` | `0x07 \|\| attr_id BE \|\| gen BE \|\| tag \|\| encoded_val` | `doc_count: u64` LE | Live-doc counts for `gen` (`0x06` analogue) |
| `default[0x14]` | `0x14 \|\| attr_id BE \|\| gen BE \|\| seg BE` | `RoaringBitmap<LocalId>` | Overwritten-slot correction set (see §4) |

---

## 3. Lock-Free Whole-Column Build & Atomic Publish (`detached_ingest`)

Whole-column replacement works in two phases so heavy Parquet/iterator scanning does **not** hold a write lock (`&mut Collection`) across millions of rows:

1. **Stage & Build (`collection.detached_ingest()...build()` — takes `&self`):**
   - Captures a snapshot of active segments (`snap_open_end: Option<(seg, row_count)>`, `snap_next_segment_id`, and snapshot deletion bitmaps) plus the highest snapshot `watermark: RowId`.
   - Allocates `next_gen = current_gen + 1`, sets `pending_gen = next_gen` (preventing concurrent compaction/publish races).
   - Resolves input `(DocId, Vec<AttributeValue>)` pairs against `cf_id_map`:
     - In `DetachedWriteMode::Replace`: live snapshot docs omitted from input become `Null`.
     - In `DetachedWriteMode::Merge`: live snapshot docs omitted from input inherit their `current_gen` value from `cf_detached`.
   - Builds sorted `ColumnBlock`s (`cf_detached`), `SegPosting`s (`cf_detached_postings`), live counts (`0x07`), and zone entries for `next_gen`, writing directly via external SST ingestion when block count meets `sst_threshold` (default `64`) or `rocksdb::WriteBatch` for small jobs. Returns a `DetachedJob`.

2. **Atomic Publish (`collection.publish_detached(&mut self, job)` — `< 20 ms`):**
   - **Catch-up delta:** Any rows allocated *after* the build snapshot (`row.seg() >= snap_next_segment_id` or `row` in `snap_open_seg` with `local >= snap_open_end`) are matched against `job.sorted_input` (`DetachedWriteMode::Replace` / `Merge`) and written to `next_gen`.
   - **Post-snapshot deletes:** Any row `< snapshot` deleted between `build()` and `publish_detached()` decrements `next_gen`'s `0x07` counts.
   - **Atomic pointer swap:** In one `WriteBatch`, sets `current_gen = next_gen`, `pending_gen = 0`, updates `retired_below = old_gen`, updates `Storage::detached_retired_below`, and swaps `zone_cache` for `(attr_id, ordered_tag)`—instantaneously activating the new generation across all readers.
   - RocksDB's compaction filter subsequently drops all `cf_detached`, `cf_detached_postings`, `default[0x07]`, and `default[0x14]` keys with `gen < retired_below`.

---

## 4. Point Writes, CoW Carry-Over & Overwritten Sets (`0x14`)

Normal online operations continue to work seamlessly on detached columns:

- **`set_detached(id, attrs)` / `set_detached_batch`:** Updates a detached column for an existing document **in place without allocating a new `RowId` or copying `cf_rows` / `cf_vectors` / FTS**.
  - Patches `cf_detached[attr_id || current_gen || seg || block]` (single 256-slot block RMW), appends the new value to `cf_detached_postings` (postings are append-only!), updates `default[0x07]` counts (`-1` old, `+1` new), and records `local` in `default[0x14 || attr_id || current_gen || seg]` (`overwritten` bitmap) if its prior posting value changed.
  - Because `cf_detached_postings` only grows via appends for `current_gen`, a filter/facet/order-by scan over `cf_detached_postings` evaluates:
    $$\text{matches} = (\text{posting}(k) \setminus \text{overwritten}[g, s]) \cup \text{forward\_scan}(\text{overwritten}[g, s] \text{ matching } k)$$
    yielding **100% exact results** even after mixed point updates while keeping bulk postings immutable!
- **`patch(id, patch)`:** Reads existing detached values at `old_row` and carries them over to `new_row` (`carry_over = true`) unless overwritten by `patch`.
- **`upsert(id, vec, attrs)`:** Replaces the entire document (`carry_over = false`); omitted detached columns reset to `Null` at `new_row`!
- **Deletion (`delete`):** Tombstoning `old_row` decrements `0x07` live counts for `current_gen` from `cf_detached`.
- **Periodic rebuild:** `maintenance()` checks `0x14` overwritten bitmaps and re-packs fragmented detached columns to a clean generation when needed.
