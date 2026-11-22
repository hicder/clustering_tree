# Bulk Import from Parquet

Calling `upsert` once per document is right for live traffic, but slow for loading millions of rows: each call is a transaction that updates the tree, quantized codes, attribute indexes, and BM25 postings. The `izzydb-bulk` crate builds the whole collection **offline** from Parquet files, writes sorted RocksDB SST files, and links them into the database in one step. See the [Bulk Import Architecture](../architecture/bulk-import.md) and [benchmark](../perf/2026-11-22-bulk-import.md) for internals and numbers.

> **The target must be empty.** The importer creates a *new* collection (or fills an existing, empty one). It does not append to a populated collection. After import, use `upsert` / `patch` / `delete` normally.

## Input format

One or more Parquet files with:

- an **id column** — integer (`Int32/64`, `UInt32/64`), one per document (nulls and negative values are rejected; duplicates follow `on_duplicate_id`);
- a **vector column** — `FixedSizeList<Float32|Float64>` or `List<Float32|Float64>` with exactly `dim` elements per row (null vectors are rejected);
- one column per schema attribute, with a compatible Arrow type:

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

Null attribute values become missing attributes.

## From Rust

```rust
use std::path::PathBuf;
use izzydb::{AttributeType, CollectionConfig, DistanceMetric, FullTextSearchConfig, TreeConfig};
use izzydb_bulk::{BulkImportConfig, DuplicatePolicy, bulk_import};

let col_cfg = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine))
    .with_attribute("category", AttributeType::String, true)
    .with_fts_attribute("title", AttributeType::String, false, FullTextSearchConfig::default());

let cfg = BulkImportConfig::new(col_cfg, "id", "vector", "/tmp/bulk_work")
    .with_on_duplicate_id(DuplicatePolicy::LastWins)
    .with_fill_factor(0.75)
    .with_verify_after(true);

let files = vec![PathBuf::from("part-0.parquet"), PathBuf::from("part-1.parquet")];
let report = bulk_import("/data/my_collection", &files, cfg, None)?;
println!("{} docs, {} leaves, {} FTS terms", report.docs_imported, report.leaves, report.terms);
for (phase, dur) in &report.phase_timings {
    println!("  {phase}: {:.2}s", dur.as_secs_f64());
}
```

Add `izzydb-bulk` to your `Cargo.toml` alongside `izzydb` (path/git dependency, same as `izzydb`).

Afterwards open it normally with the **same** `CollectionConfig`:

```rust
let col = izzydb::Collection::open("/data/my_collection", col_cfg)?;
```

### `BulkImportConfig` options

`BulkImportConfig::new(collection, id_column, vector_column, work_dir)` plus:

| Builder | Default | Meaning |
|---|---|---|
| `.with_column_map(vec![("parquet_col", "attr_name")])` | all schema attributes map to same-named columns | Map differently named Parquet columns to attributes. |
| `.with_on_duplicate_id(p)` | `Error` | `Error` aborts on a repeated id; `LastWins` keeps the last occurrence (by file order, then row). |
| `.with_fill_factor(f)` | 0.75 | Target leaf fullness, in `(0, 1]`. Lower leaves headroom for future inserts without splits; higher packs tighter. |
| `.with_train_sample(n)` | 1,000,000 | Vectors sampled to train the cluster tree skeleton. |
| `.with_memory_budget_bytes(n)` | 2 GiB | Memory budget for sorting/staging. |
| `.with_target_sst_bytes(n)` | 256 MiB | Target SST file size. |
| `.with_threads(n)` | all cores | Worker threads. |
| `.with_keep_work_dir(bool)` | `false` | Keep intermediate files (otherwise deleted on success). |
| `.with_verify_after(bool)` | `false` | Run full invariant verification after import (slow on big data; great for first runs). |

**Work directory:** put it on the **same filesystem** as the target so SST files are hard-linked into place instead of copied. It needs room for the staged vectors/documents and the generated SST files, so check free space before large imports.

The last argument of `bulk_import` is an optional progress callback `Option<&dyn Fn(Progress)>`; `Progress { phase, completed, total }`.

## From the command line

```bash
cargo run --release -p izzydb-bulk --bin izzydb-import -- \
  --target /data/my_collection \
  --input part-0.parquet part-1.parquet \
  --dim 128 --metric cosine \
  --id-column id --vector-column vector \
  --on-duplicate-id last-wins \
  --verify
```

Run with `--help` for everything (`--fill-factor`, `--train-sample`, `--work-dir`, `--memory-mb`, `--target-sst-mb`, `--threads`). The CLI builds a collection with **vector only** (no attributes); for attributes or full-text, use the Rust API so you can pass a full schema.

## Two-stage import (build on one machine, ingest on another)

```rust
use izzydb_bulk::{build_ssts, ingest_ssts, SstBundle};

// Machine A: heavy CPU/IO, touches no database
let bundle = build_ssts(&files, &cfg)?;     // writes SSTs + manifest.json under cfg.work_dir

// Copy the bundle directory to machine B, then:
let bundle = SstBundle::open("/path/to/bundle")?;
ingest_ssts("/data/my_collection", &bundle)?;
```

`ingest_ssts` checks the bundle's manifest (format version, config hash, dimension, metric) against the target before ingesting.

## Failure and recovery

Ingestion is bracketed by a durable marker. If the process dies mid-ingest, opening the database fails with:

```
corrupted storage data: incomplete bulk import; delete the database directory and re-run
```

Do exactly that: delete the target directory and run the import again. Building SSTs never touches the target, so a failure *during build* leaves nothing to clean up except the work directory.

## Tips

- Run with `--verify` / `.with_verify_after(true)` the first time you load a new dataset shape.
- `memory_budget_bytes` bounds how much the importer keeps in RAM while staging and sorting; raise it on machines with memory to spare, lower it on small ones.
- `fill_factor` controls leaf headroom: a lower value leaves room for later `upsert`s before leaves split, a higher value packs leaves more tightly. The default (0.75) is a reasonable middle.
