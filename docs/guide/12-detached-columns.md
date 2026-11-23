# Detached Columns

**Detached columns** are attributes stored outside each document's forward row (`cf_rows`), versioned in columnar blocks (`cf_detached`) by monotonic generation.

Use detached columns for fields that are **computed or refreshed independently from documents**—such as offline PageRank / popularity scores, ML ranking signals, moderation states, inventory flags, or daily batch updates over millions of rows:

- **Whole-column replacement (`> 1M rows/s` build, `< 20 ms` atomic publish):** Replace or merge an entire column across all live rows without copying `cf_rows`, `cf_vectors`, or re-indexing vectors/FTS.
- **No row CoW on point writes:** `set_detached` / `set_detached_batch` update detached column values in place without allocating new `RowId`s or tombstoning rows.
- **Full query integration:** Filterable detached columns support `Filter` predicates, `OrderBy`, `RankBy` (`Attribute`/`Saturate`/`Decay`), and `Facet`s seamlessly alongside inline attributes.

*(Constraints: detached columns cannot use `FullTextSearchConfig`, and changing an attribute between `Inline` and `Detached` on an existing collection is rejected with `TreeError::ConfigMismatch`.)*

---

## Declaring detached attributes

```rust
use izzydb::{AttributeType, Collection, CollectionConfig, DistanceMetric, TreeConfig};

let config = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine))
    // Standard inline attribute
    .with_attribute("category", AttributeType::String, true)
    // Detached attributes: (name, type, filterable)
    .with_detached_attribute("popularity", AttributeType::Int, true)
    .with_detached_attribute("score_tier", AttributeType::String, true);

// Or for document/BM25/filter/column workloads without vectors:
let vectorless = CollectionConfig::without_vectors()
    .with_detached_attribute("popularity", AttributeType::Int, true);

let mut col = Collection::open("/tmp/col", config)?;
```

You can also add a **new** detached attribute to an existing collection simply by adding `.with_detached_attribute(...)` to `CollectionConfig` and reopening (`CollectionSchema::reconcile` starts it at generation `0` with `Null` values for existing rows).

---

## Whole-column bulk ingest (`detached_ingest` + `publish_detached`)

Detached ingest splits heavy scanning/building (`&self`, read-only!) from the fast pointer swap (`&mut self`):

```rust
use izzydb::{AttributeValue, DetachedDuplicatePolicy, DetachedWriteMode, DocId};

let updates: Vec<(DocId, Vec<AttributeValue>)> = vec![
    (1, vec![950_i64.into(), "gold".into()]),
    (2, vec![120_i64.into(), "silver".into()]),
];

// Phase 1 (&self): build next-generation columnar SST blocks without holding write lock
let job = col
    .detached_ingest()
    .columns(["popularity", "score_tier"])
    .mode(DetachedWriteMode::Replace)                // or DetachedWriteMode::Merge
    .duplicate_policy(DetachedDuplicatePolicy::LastWins)
    .from_iter(updates)
    .build()?;

// Phase 2 (&mut self): reconcile concurrent writes since snapshot & swap generation (< 20 ms)
let pub_stats = col.publish_detached(job)?;
println!("Published {} columns (catchup_rows={})", pub_stats.columns, pub_stats.catchup_rows);
```

### `DetachedWriteMode`

| Mode | Unmentioned live `DocId`s | Unknown `DocId`s |
|---|---|---|
| `DetachedWriteMode::Replace` (default) | Reset to `Null` | Ignored (`stats.unknown_ids`) |
| `DetachedWriteMode::Merge` | Preserve prior generation's value | Ignored |

For a single column, you can also use `.column("popularity")` with `.from_pairs(iter_of_doc_id_and_value)` instead of `.columns([...]).from_iter(...)`.

### Concurrent writes during `build()`

Writes (`upsert`, `patch`, `delete`) remain valid while a `DetachedJob` is staged (only another concurrent `detached_ingest` or `compact` waits while `pending_gen != 0`). When `publish_detached(job)` commits, it automatically catches up any rows appended or deleted between `build()` and publish.

### From the CLI (`izzydb-import columns`)

```bash
cargo run --release -p izzydb-bulk --bin izzydb-import -- columns \
  --target /data/my_collection \
  --input metrics.parquet \
  --id-column id \
  --columns popularity score_tier \
  --mode replace \
  --dim 128 --metric cosine
```

---

## Point updates with `set_detached`

```rust
use izzydb::{AttributeValue, Attributes};

let mut patch = Attributes::new();
patch.insert("popularity".into(), 999_i64.into());
col.set_detached(1, patch)?;

// Clear a detached value to Null:
let mut clear = Attributes::new();
clear.insert("score_tier".into(), AttributeValue::Null);
col.set_detached(2, clear)?;

// Batch multiple documents atomically:
col.set_detached_batch(&[(1, attrs_1), (2, attrs_2)])?;
```

`set_detached` updates `cf_detached` in place without CoW row duplication or ANN/FTS writes.

---

## Interaction with `upsert` and `patch`

- **`patch(id, DocumentPatch)`:** Preserves existing detached attribute values across the CoW move unless explicitly overwritten or cleared in `patch.attributes`.
- **`upsert(id, embedding, attrs)`:** **Full document replacement.** Unmentioned detached attributes are **not** carried over—they become `Null` unless included in `attrs`.

---

## Housekeeping

After many `set_detached` updates on a `filterable` detached column, point-write *overwritten sets* (`default[0x14]`) accumulate small per-segment bitmaps. Calling [`Collection::maintenance`](11-operations.md#segment-lifecycle--maintenance) automatically rebuilds fragmented detached columns cleanly to a fresh generation and lets RocksDB compaction filters reclaim retired generations.
