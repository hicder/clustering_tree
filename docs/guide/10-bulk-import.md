# Bulk Import, Append & Column Ingest from Parquet

Calling `upsert` once per document is right for live traffic, but slow for loading millions of rows: each call is a transaction that appends rows, encodes vectors, and updates secondary indexes. The `izzydb-bulk` crate builds sealed segments **offline** from Parquet files, writes sorted RocksDB SST files, and ingests them straight into RocksDB. See [Bulk Import Architecture](../architecture/bulk-import.md) for internals.

Three modes are supported:
- **`bulk_import` / `izzydb-import import`**: Populate a **new or empty** collection (vectored or vectorless).
- **`bulk_append` / `izzydb-import append`**: Append new Parquet files as new sealed segments into an **existing populated** collection (replacing any colliding `DocId`s).
- **`izzydb-import columns`**: Bulk-update [Detached Columns](12-detached-columns.md) from Parquet on an existing collection.

---

## Input format

One or more Parquet files with:

- an **id column** — integer (`Int32/64`, `UInt32/64`), one per document (nulls and negative values are rejected; duplicates follow `on_duplicate_id`);
- an optional **vector column** — `FixedSizeList<Float32|Float64>` or `List<Float32|Float64>` with `dim` elements per row (required unless the collection uses `CollectionConfig::without_vectors()`);
- one column per schema attribute (inline and/or detached), with a compatible Arrow type:

| `AttributeType` | Accepted Parquet / Arrow types |
|---|---|
| `String` | `Utf8`, `LargeUtf8`, `Utf8View` |
| `Int` | `Int8/16/32/64` |
| `Uint` | `UInt8/16/32/64` |
| `Float` | `Float32`, `Float64` |
| `Bool` | `Boolean` |
| `StringArray` | list of the string types |
| `IntArray` | list of `Int8/16/32/64` |
| `UintArray` | list of `UInt8/16/32/64` |

Null attribute values become missing attributes (`Null`).

---

## Initial Import (`bulk_import`)

```rust
use std::path::PathBuf;
use izzydb::{AttributeType, CollectionConfig, DistanceMetric, FullTextSearchConfig, TreeConfig};
use izzydb_bulk::{BulkImportConfig, DuplicatePolicy, bulk_import};

let col_cfg = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine))
    .with_attribute("category", AttributeType::String, true)
    .with_fts_attribute("title", AttributeType::String, false, FullTextSearchConfig::default());

let cfg = BulkImportConfig::new(col_cfg.clone(), "id", "vector", "/tmp/bulk_work")
    .with_on_duplicate_id(DuplicatePolicy::LastWins)
    .with_fill_factor(0.75)
    .with_verify_after(true);

let files = vec![PathBuf::from("part-0.parquet"), PathBuf::from("part-1.parquet")];
let report = bulk_import("/data/my_collection", &files, cfg, None)?;
println!(
    "{} docs in {} segments ({} leaves, {} FTS terms)",
    report.docs_imported, report.segments_created, report.leaves, report.terms
);
```

For vectorless collections, use `BulkImportConfig::new_vectorless(col_cfg, "id", "/tmp/bulk_work")`.

### `BulkImportConfig` options

| Builder | Default | Meaning |
|---|---|---|
| `.with_column_map(vec![("parquet_col", "attr_name")])` | map by attribute name | Map differently named Parquet columns to schema attributes. |
| `.with_on_duplicate_id(p)` | `Error` | `Error` rejects duplicate IDs; `LastWins` keeps the last occurrence. |
| `.with_fill_factor(f)` | `0.75` | Target ANN leaf fullness in `(0, 1]`. |
| `.with_train_sample(n)` | `1,000,000` | Vectors sampled to train the ANN clustering tree. |
| `.with_memory_budget_bytes(n)` | `2 GiB` | Memory budget for sorting/staging. |
| `.with_target_sst_bytes(n)` | `256 MiB` | Target SST file size. |
| `.with_threads(n)` | all cores | Worker threads. |
| `.with_keep_work_dir(bool)` | `false` | Keep intermediate files after completion. |
| `.with_verify_after(bool)` | `false` | Run `verify_invariants()` after import/append. |

**Work directory:** place `work_dir` on the **same filesystem** as the target database so SST files are hard-linked (`move_files = true`) instead of copied.

---

## Appending to a Populated Collection (`bulk_append`)

To ingest new Parquet batches into an existing collection without rebuilding from scratch:

```rust
use izzydb_bulk::bulk_append;

let report = bulk_append(
    "/data/my_collection",
    &[PathBuf::from("batch-2.parquet")],
    cfg,
    None,
)?;
println!(
    "Appended {} docs (replaced {}) across {} new segments (rebuilt_ann={})",
    report.docs_appended, report.docs_replaced, report.segments_created, report.rebuilt_ann_tree
);
```

`bulk_append` creates fresh sealed segments starting at `Manifest.next_segment_id`, tombstones replaced `DocId`s atomically, and either inserts new vectors incrementally into the existing ANN tree or rebuilds the global ANN tree when large batches shift the distribution.

---

## Command Line (`izzydb-import`)

The CLI uses subcommands `import`, `append`, and `columns`:

```bash
# 1. Create / import into an empty collection (omit --dim for vectorless)
cargo run --release -p izzydb-bulk --bin izzydb-import -- import \
  --target /data/my_collection \
  --input part-0.parquet part-1.parquet \
  --dim 128 --metric cosine \
  --id-column id --vector-column vector \
  --on-duplicate-id last-wins \
  --verify

# 2. Append another Parquet batch to an existing collection
cargo run --release -p izzydb-bulk --bin izzydb-import -- append \
  --target /data/my_collection \
  --input part-2.parquet \
  --dim 128 --metric cosine \
  --on-duplicate-id last-wins

# 3. Bulk-ingest detached columns from Parquet into an existing collection
cargo run --release -p izzydb-bulk --bin izzydb-import -- columns \
  --target /data/my_collection \
  --input scores.parquet \
  --id-column id \
  --columns popularity rank_tier \
  --mode replace
```

Run `izzydb-import --help` or `izzydb-import <subcommand> --help` for all flags.

---

## Two-Stage Import (`build_ssts` / `ingest_ssts`)

```rust
use izzydb_bulk::{build_ssts, ingest_ssts, SstBundle};

let bundle = build_ssts(&files, &cfg)?;
let bundle = SstBundle::open("/path/to/bundle")?;
ingest_ssts("/data/my_collection", &bundle)?;
```

## Failure and Recovery

Ingestion is bracketed by `default[b"\x01bulk_import_v3"]` with `sync = true`. If a crash occurs during SST ingestion, opening the collection fails cleanly with `CorruptedData("incomplete bulk import...")`. For `bulk_import`, delete the target directory and re-run; for `build_ssts`, intermediate target touches never occur until `ingest_ssts`.
