# Segments, CoW Updates, Compaction & ANN Vacuum

This document specifies IzzyDB v3's append-only segment store, copy-on-write (CoW) document lifecycle, background segment compaction, ANN vacuuming, and RocksDB compaction filter reclamation.

Implementation sources:
- Write-path segment store & hooks: [`crates/izzydb/src/segments.rs`](../../crates/izzydb/src/segments.rs)
- Segment table & compaction filter: [`crates/izzydb-storage/src/storage/`](../../crates/izzydb-storage/src/storage/)
- Segment compaction & RowId forwarding: [`crates/izzydb/src/compaction.rs`](../../crates/izzydb/src/compaction.rs)
- ANN vacuum & forwarding retire coordination: [`crates/izzydb/src/tree/mod.rs`](../../crates/izzydb/src/tree/mod.rs)
- Sealed-segment columnar builder: [`crates/izzydb/src/columns.rs`](../../crates/izzydb/src/columns.rs)

---

## 1. Coordinate System (`RowId = (SegmentId, LocalId)`)

In v3, user primary keys (`DocId: u64`) are decoupled from physical storage locations (`RowId: u64`):

```text
RowId (u64):  [ SegmentId: u32 (seg) | LocalId: u32 (local) ]
RowBlock:     (SegmentId, block = local >> 8)   // 256 rows/block
SkipChunk:    (SegmentId, chunk = block >> 8)   // 256 blocks = 65,536 rows
```

- **`cf_id_map`:** maps live `DocId (u64 BE) -> RowId (u64 BE)`.
- **`cf_row_doc_ids`:** stores 256-slot `[DocId; 256]` LE blocks keyed by `seg BE || block BE` (`doc_id_block_cache` LRU in RAM).
- **`SegmentId`s are strictly monotonic** (`Manifest.next_segment_id`) and **never reused**.
- **`LocalId`s are dense and append-only** inside a segment (`0..row_count`, max `MAX_SEGMENT_ROWS = 1 << 20`).

Because every `RowId` is immutable until compaction retires its segment, sealed-segment inverted postings (`cf_attr_postings`), FTS postings (`cf_fts_postings`), columnar blocks (`cf_columns`), and detached blocks (`cf_detached`) are **write-once** for sealed rows.

---

## 2. Segment State Machine & Lifecycle

Every non-dropped segment has a `SegmentMeta` record under `default[0x11 || seg BE]`:

```text
          alloc first row
[None] ───────────────────────► Open (state = 0)
                                  │
       row_count >= seal_rows     │  (or explicit col.seal())
                                  ▼
                                Sealed (state = 1)
                                  │
       compaction Step A/B        │  live rows copied to new Building -> Sealed target segment
                                  ▼
                              Forwarding (state = 3)
                                  │
       vacuum_epoch advances      │  (all ANN leaves scrubbed beyond forwarded_at_epoch)
                                  ▼
                               Retired (state = 4) ──► dropped_segments bitmap + delete_range
```

| State | Code | Visible to Queries? | Mutable? | Purpose |
|---|---:|---|---|---|
| `Open` | `0` | Yes | Append + tombstone | Active write tail (`<= SegmentConfig.seal_rows`, default `65,536`) |
| `Sealed` | `1` | Yes | Tombstone only (`0x10`) | Immutable rows + per-segment `cf_columns` |
| `Building` | `2` | **No** | Compaction output only | Crash-safe compaction target prior to Step B commit |
| `Forwarding` | `3` | **No** | Tombstone redirects | Holds rank-based `LocalId -> target LocalId` mapping for stale ANN leaf entries |
| `Retired` | `4` | **No** | Tombstone removed | Recorded in `dropped_segments`; keys reclaimed asynchronously by compaction filter |

### Liveness & Tombstones

A row `(seg, local)` is live iff:
1. `SegmentMeta` exists with `state` in `{Open, Sealed}` and `local < seg_meta.row_count`, and
2. Bit `local` is **not** set in `default[0x10 || seg BE]` (deleted `LocalId`s `RoaringBitmap`).

`Storage` keeps an immutable `Arc<SegmentTableSnapshot>` updated via `ArcSwap` on every transaction commit, giving queries lock-free RAM access to segment visibility and per-block `[u64; 4]` live masks (`block_live_mask(seg, block)`).

---

## 3. Copy-on-Write (`upsert`, `patch`, `delete`)

Every mutation (`upsert`, `patch`, `delete`, and `*_batch`) executes inside one `TxContext` atomic `WriteBatch`:

- **Tombstone old row (on `delete` or existing `upsert`/`patch`):**
  1. Adds `old_row.local()` to `default[0x10 || old_seg]` (incrementing `SegmentMeta.deleted_count` and decrementing `Manifest.live_rows`).
  2. Reads `old_row` from `cf_rows` (`RowView`) to decrement `default[0x06]` live attribute counts and FTS `TermRecord.doc_freq` + `FtsMeta` stats; never modifies sealed `cf_attr_postings`, `0x12` zone maps, or `cf_fts_postings` blocks.
  3. If the collection has vectors and the old row had one, marks it stale by incrementing `AnnMeta.stale_rows` (deferred leaf vacuum; the vector is only deleted from the ANN tree eagerly if the old leaf happens to split/repack or when `vacuum_ann` scans it).
- **Append new row (on `upsert` or `patch`):**
  1. Allocates a fresh `new_row = (open_seg, local)` via `SegmentStore::alloc_row`, seals `open_seg` if `row_count >= seal_rows` (building `cf_columns` unless `defer_column_build` is true).
  2. Writes `cf_rows[new_row]`, `cf_id_map[doc_id] = new_row`, `cf_row_doc_ids`, optional `cf_vectors[new_row]`, inline `cf_attr_postings` + `0x06` counts + `0x12` zone widening, FTS blocks, and (on `patch`) carries over per-row detached attribute values from `old_row` to `new_row` (`upsert` does **not** carry over unmentioned detached columns).
  3. Routes into the global ANN tree (`ClusteringTree`), inserting `(new_row, quant_code)` (and reusing/evicting any dead slot in the target leaf).

**Cost implication:** Updating a single attribute via `patch` rewrites the row's inline attributes (CoW) and leaves sealed segments untouched except for tombstones. Frequently mutated attributes that shouldn't force CoW row rewrites can be marked `ColumnStorage::Detached` (see [detached-columns.md](detached-columns.md)).

---

## 4. Two-Phase Segment Compaction & RowId Forwarding

Over time, tombstones accumulate in sealed segments and sparse deletions leave undersized segments. `Collection::compact(CompactionOptions)` merges eligible segments (`deleted_count / row_count >= 0.30` or `live < seal_rows * 0.25`) safely with full crash recovery:

1. **Pre-flight & Recovery:** Any leftover `Building` segment from a crashed prior compaction run is deleted via `drop_segments` before starting.
2. **Step A (Build target segment — invisible):**
   - Allocates one or more target segments (`SegmentState::Building`), copying all live rows (`cf_rows`, `cf_vectors`, `cf_columns`), live inline `cf_attr_postings` / `0x12` zones, FTS blocks, and live `cf_detached` / `cf_detached_postings` / `0x14` slices across all active detached generations into dense `0..M` locals.
   - Commits Step A so `Building` rows exist durably while source segments are still live.
3. **Step B (Atomic cutover):**
   - Atomically switches `target` from `Building -> Sealed`, updates `cf_id_map` for every moved `DocId`, marks source segments as `SegmentState::Forwarding { forward_target, forward_base, forwarded_at_epoch }` (or deletes vectorless segments immediately), and deletes `0x10` tombstones / `0x12` zones on forwarded sources.
   - Note: `cf_attr_postings` (`0x12` excluded), FTS postings, and row bytes for `Forwarding` sources are dropped once retired.
4. **Rank-Based Forwarding Without Per-Row Map:**
   - Because live rows in source segment $S$ were copied in dense ascending `LocalId` order into $[B, B + L)$ inside target $T$, any old `RowId(S, l)` maps in $O(\log |\text{deleted}_S|)$ via `deleted_bitmap_S.rank(l)`:
     $$\text{new\_local} = \text{forward\_base} + (l - \text{deleted}_S.\text{rank}(l))$$
   - Dead `LocalId`s (`deleted_S.contains(l)`) map to tombstoned (`None`). ANN leaf scans and `vacuum_ann` resolve `RowResolver` / `SegmentTableSnapshot::resolve_forwarding` transparently without rewriting the entire ANN tree synchronously during compaction!

---

## 5. ANN Vacuum (`vacuum_ann`) & Compaction Filter Reclamation

- **`Collection::vacuum_ann(VacuumBudget)`** resumes from `AnnMeta.vacuum_cursor`, scanning up to `max_leaves` / `max_bytes` of `cf_leaf_entries`:
  - Resolves every `RowId` in each leaf through forwarding chains: drops dead/retired entries, rewrites forwarded `RowId`s to their current target `(T, new_local)`, re-packs `PackedLeafNode`, merges underflowing leaves, and decrements `AnnMeta.stale_rows`.
  - When `vacuum_cursor` wraps around `next_node_id`, it increments `AnnMeta.vacuum_epoch += 1` (`completed_pass = true`).
- **Retiring `Forwarding` segments:** Once `AnnMeta.vacuum_epoch > seg_meta.forwarded_at_epoch`, **no** ANN leaf anywhere in the tree can still hold a reference to that `Forwarding` segment. Compaction/vacuum immediately transitions it to `dropped_segments` (`\x01dropped_segments`), issues bounded `delete_range`s, and lets the RocksDB compaction filter finish physical key reclamation.
- **RocksDB Compaction Filter:** Installed on `cf_rows`, `cf_row_doc_ids`, `cf_vectors`, `cf_columns`, `cf_detached`, `cf_attr_postings`, `cf_detached_postings`, `cf_fts_skips`, `cf_fts_postings`, `cf_fts_positions`, and `default[0x05]`. Each filter shares an `Arc<ArcSwap<RoaringBitmap>>` of `dropped_segments` (plus retired detached generations) and drops keys belonging to dead segments/generations during normal RocksDB SST compaction—race-free because segment IDs and generation IDs are never reused.

## 6. Combined Maintenance (`Collection::maintenance`)

`Collection::maintenance(MaintenanceOptions)` runs one bounded housekeeping cycle:
1. Materializes `cf_columns` for any `Sealed` segment missing `SEG_FLAG_HAS_COLUMNS` (when `defer_column_build` was set).
2. Runs `vacuum_ann` when `stale_rows / (num_vectors + stale_rows) > vacuum_stale_ratio` (default `0.2`).
3. Runs `compact` (unless a detached column build has `pending_gen != 0`, preventing races) and tightens FTS `TermRecord` upper bounds if segments were forwarded or retired.
4. Rebuilds detached columns whose `0x14` overwritten sets accumulated significant churn.
