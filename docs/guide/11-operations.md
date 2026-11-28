# Operations, Maintenance & Troubleshooting

## On-disk layout

A collection is a RocksDB directory (`CURRENT`, `MANIFEST`, `.sst` files, `.log` WALs). Treat it as opaque: back it up while closed or via RocksDB checkpoints.

## Opening, closing, and concurrency

- `Collection::open(path, config)` loads ANN internal nodes (`default[0x02]`), the `SegmentTable` snapshot, `zone_cache`, and `dropped_segments`.
- RocksDB holds an exclusive `LOCK` per process. Dropping `Collection` closes it.
- Reads (`search`, `query`, `query_bm25`, `hybrid_search`, `query_ordered`, `order_by`, `facet`, `get`, `resolve_filter_doc_ids`, `detached_ingest().build()`) take `&self` and run concurrently.
- Writes (`upsert*`, `patch*`, `delete*`, `set_detached*`, `publish_detached`, `seal`, `compact`, `vacuum_ann`, `maintenance`, `compact_fts_stats`) take `&mut self`.

---

## Segment lifecycle & maintenance

IzzyDB v3 writes documents into an active append-only **segment** (`SegmentConfig::seal_rows`, default `65,536` rows per segment):

- **Sealing (`col.seal()`):** Seals the current open segment early, freezing its postings and building per-segment columnar blocks (`cf_columns`) for inline attributes (unless `segment.defer_column_build = true`).
- **ANN vacuum (`col.vacuum_ann(VacuumBudget)`):** Incrementally sweeps `cf_leaf_entries` from `ann_meta.vacuum_cursor`, dropping tombstoned entries, rewriting compaction-forwarded `RowId`s, and advancing `vacuum_epoch` once a full pass finishes so `Forwarding` segments can retire cleanly.
- **Compaction (`col.compact(CompactionOptions)`):** Merges sealed segments whose tombstone ratio exceeds `0.30` or live rows fall below `25%` of `seal_rows`, forwarding surviving `RowId`s in dense rank order and retiring obsolete segments once `vacuum_epoch` advances. RocksDB compaction filters then reclaim all dead segment/generation keys asynchronously during SST compaction.
- **Combined housekeeping (`col.maintenance(MaintenanceOptions)`):** Runs deferred `cf_columns` builds, triggers `vacuum_ann` when `stale_rows / (num_vectors + stale_rows) > vacuum_stale_ratio` (default `0.2`), compacts eligible segments, tightens FTS bounds (`compact_fts_stats`), and rebuilds fragmented detached columns.

```rust
use izzydb::{MaintenanceOptions, VacuumBudget};

// Periodic maintenance (e.g., from a background worker taking write lock briefly)
let stats = col.maintenance(MaintenanceOptions::default())?;
println!(
    "vacuum={:?}, compacted_segments={}, rebuilt_detached={}",
    stats.vacuum, stats.compaction.segments_retired, stats.detached_rebuilt
);

// Or run steps explicitly:
col.seal()?;
col.vacuum_ann(VacuumBudget::leaves(128))?;
```

### Tuning `SegmentConfig`

- **`seal_rows` (default `65,536`, range `1..=1_048_576`):**
  - Smaller values (`4,096–16,384`) reduce open-segment scan work, tighten `0x12` zone-map bounds, and accelerate sealed `cf_columns` builds/compactions at high write churn.
  - Larger values (`65,536–262,144`) reduce total segment count, minimize per-segment overhead on `SegPosting`s, and match bulk Parquet batch sizes.
- **`defer_column_build` (default `false`):** Set `true` on latency-sensitive write paths to skip synchronous `cf_columns` generation at seal time and defer it to `maintenance()`.

### I/O & Concurrency Tuning (`CollectionConfig`)

- **`.with_direct_io(true)` (default `false`):** Enables RocksDB `use_direct_reads` and `use_direct_io_for_flush_and_compaction` (`O_DIRECT`) to bypass the OS page cache and rely on RocksDB's block cache instead.
- **`.with_fts_parallel_segments(n)` (default `None`):** Caps the number of segment shards executed in parallel during FTS Block-Max MAXSCORE queries (`None` uses Rayon's thread count; `1` runs single-threaded; `0` is rejected with `InvalidConfig`).

---

## Durability

Single-call mutations (`upsert`, `patch`, `patch_batch`, `delete`, `delete_batch`, `set_detached*`, `publish_detached`, `seal`) commit via one atomic RocksDB `WriteBatch` through RocksDB's WAL (no per-write `fsync` by default). `update_by_filter` and `delete_by_filter` apply changes in bounded `512`-doc atomic chunks. `compact` uses two-phase crash-safe `Building -> Sealed -> Forwarding -> Retired` state transitions so interrupted compactions recover cleanly on re-open.

---

## Verifying invariants

```rust
let stats = col.verify_invariants()?;
println!("{} docs across {} segments", stats.num_docs, stats.num_segments);
```

`verify_invariants` exhaustively checks manifest counters, `SegmentMeta` / tombstones / dropped segments, `cf_id_map <-> cf_rows <-> cf_row_doc_ids` bijection, ANN leaf parity (or emptiness for vectorless collections), inline/detached inverted postings & counts & zone maps, `cf_columns` / `cf_detached`, and FTS parity. Intended for tests, post-import validation, or offline audits.

---

## Benchmarks

```bash
cargo test --workspace
cargo run --release -p izzydb-bench --bin recall_bench
cargo run --release -p izzydb-bench --bin filtered_bench
cargo run --release -p izzydb-bench --bin fts_bench
cargo run --release -p izzydb-bench --bin rank_bench
cargo run --release -p izzydb-bench --bin facet_bench
cargo run --release -p izzydb-bench --bin bulk_import_bench
cargo run --release -p izzydb-bench --bin churn_bench
cargo run --release -p izzydb-bench --bin detached_bench
```

See [Segmented Storage v3 Performance Report](../perf/2026-11-23-segmented-storage-v3.md) for v3 baseline measurements.

---

## Errors

All fallible calls return `izzydb::Result<T>` (`Result<T, TreeError>`):

| Variant | Typical cause |
|---|---|
| `InvalidConfig` / `InvalidSchema` | Out-of-range `TreeConfig` / `SegmentConfig` / `fts_parallel_segments = 0`, vector search on a vectorless collection, or invalid attribute definition |
| `ConfigMismatch` | Re-opening a collection with a different `dim`, `metric`, `has_vectors`, or existing attribute type/flags |
| `DimensionMismatch` / `NonFiniteVector` | Embedding or query length mismatch, or `NaN` / `Inf` values |
| `InvalidAttribute` | Undeclared attribute name, wrong `AttributeValue` variant, or `NaN` float |
| `InvalidFilter` / `InvalidQuery` | Invalid `Filter`, `RankBy`, `OrderBySpec`, or `FacetSpec` |
| `DocumentNotFound` | `patch` / `patch_batch` / `set_detached` on a missing `DocId` |
| `Storage` / `CorruptedData` / `InvariantViolation` | RocksDB I/O failure, incomplete bulk import marker, or storage corruption |

---

## Cargo features & FAQ

| Feature | Crates | Effect |
|---|---|---|
| `fts-cjk` | `izzydb`, `izzydb-core`, `izzydb-fts` | Enables `SmartCn`, `Kuromoji`, and `Nori` tokenizers for Chinese, Japanese, and Korean. |
| `io-uring` | `izzydb`, `izzydb-storage`, `izzydb-bulk`, `izzydb-bench` | Enables `rocksdb/io-uring` for async Linux I/O (opt-in; requires `liburing` development headers on Linux). |

- **Can I create a collection without vector embeddings?** Yes: use `CollectionConfig::without_vectors()` and pass `None` as the embedding to `upsert`.
- **How do I update an attribute across millions of docs without rewriting documents or vectors?** Declare it as a [Detached Column](12-detached-columns.md) (`.with_detached_attribute(...)`) and load updates with `detached_ingest().build()` + `publish_detached()`.
- **Do I need to flush after writing?** No; committed `WriteBatch`es are visible to the next read immediately.
