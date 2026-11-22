# Bulk Import Benchmark (`izzydb-bulk`, 2026-11-22)

Measured in `--release` via:

```bash
cargo run --release --offline -p izzydb-bench --bin bulk_import_bench
cargo run --release --offline -p izzydb-bench --bin bulk_import_bench -- --full
```

Source: [`crates/izzydb-bench/src/bin/bulk_import_bench.rs`](../../crates/izzydb-bench/src/bin/bulk_import_bench.rs), [`crates/izzydb-bulk/`](../../crates/izzydb-bulk/).

## 1. 1M-Scale Runs (`--full`)

Datasets: 64-cluster Gaussian-like vectors (`spread = 0.15`), `L2Squared`, `category: String (filterable)`, `rank: Int (filterable)`, default `TreeConfig` (`min_leaf_size = 32`, `max_leaf_size = 160`, `fill_factor = 0.75`). In the FTS run, `text: String` has 4 tokens per document including a unique per-document token (`1,000,016` unique terms total).

| Dataset | `bulk_import` Wall Time | `bulk_import` Throughput | Final DB Size | Leaves / Height | FTS Terms | SST Files | `upsert` Baseline (small subset) | Speedup vs. Subset `upsert` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **1M × 128-d** | **14.17 s** | **70,588 docs/s** | **606.42 MiB** | 12,559 / 2 | 0 | 131 | 13,363 docs/s (`n = 20,000`, 172.07 MiB DB) | **5.3×** |
| **1M × 768-d + FTS** | **22.11 s** | **45,237 docs/s** | **3,337.12 MiB** | 12,287 / 2 | 1,000,016 | 135 | 1,338 docs/s (`n = 5,000`, 285.99 MiB DB) | **33.8×** |

### Phase Breakdown (`--full`)

| Phase | 1M × 128-d | 1M × 768-d + FTS |
|---|---:|---:|
| `prepare` | 0.00 s | 0.00 s |
| `scan_and_stage` | 0.23 s | 1.68 s |
| `build_tree` | 12.77 s | 17.02 s |
| `emit_ssts` | 0.62 s | 2.81 s |
| `ingest` | 0.54 s | 0.57 s |

## 2. Recall & Medium-Scale Runs (Default Mode)

| Dataset | `bulk_import` Wall Time | `bulk_import` Throughput | Final DB Size | Recall@10 (`beam=8`, `rerank=100`) | `upsert` Baseline |
|---|---:|---:|---:|---:|---:|
| **50k × 128-d** | 0.87 s | 57,714 docs/s | 30.76 MiB | — | 11,276 docs/s (`n = 10,000`, 103.52 MiB) |
| **20k × 768-d + FTS** | 1.02 s | 19,527 docs/s | 67.30 MiB | **0.9920** | 1,302 docs/s (`n = 5,000`, 285.99 MiB) |

## Observations

- **Zero WAL / compaction amplification:** At `1M × 128-d`, the entire collection is `606.42 MiB` (`~636 B/doc` including raw 512 B vectors, forward docs, RaBitQ leaf blobs, and attribute indexes), compared to `172.07 MiB` for just `20,000` incremental `upsert`s (`~9,021 B/doc` pre-compaction).
- **Sub-second SST emission & ingestion:** At `1M × 128-d`, `emit_ssts` (`0.62 s`) and `ingest` (`0.54 s`) together take `1.16 s`; the dominant cost is `build_tree` (`12.77 s`, dominated by training the skeleton on `1,000,000` sample vectors and routing all `1M` vectors through the 2-level hierarchy).
