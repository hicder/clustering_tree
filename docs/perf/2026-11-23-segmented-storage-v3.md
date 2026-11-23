# Segmented Storage v3 Performance Report (2026-11-23)

Comparison of `segments-v3` against the pre-overhaul `master` baseline (`1dbdb8d`), measured in `--release` on the same machine across the `izzydb-bench` suite:

```bash
cargo build --release -p izzydb-bench
for b in recall_bench filtered_bench fts_bench rank_bench facet_bench bulk_import_bench churn_bench detached_bench; do
  ./target/release/$b
done
./target/release/detached_bench 2000000
```

Source: [`crates/izzydb-bench/src/bin/`](../../crates/izzydb-bench/src/bin/).

---

## 1. Summary

- **Bulk import throughput:** **1.39×–1.55× faster** (`86,107 docs/s` vs `55,519 docs/s` on 50k×128d; `26,734 docs/s` vs `19,165 docs/s` on 20k×768d+FTS) with **3.3× fewer SST files** (`39–40` vs `131–132`) and higher recall (`0.9940` vs `0.9840`).
- **Faceted queries (broad / disjunctive / hybrid):** **3.3×–4.1× faster** (`3.92–4.48 ms` vs `14.35–16.39 ms`), because per-segment inverted postings (`1` segment vs `150` leaves) eliminate cross-leaf bitmap merging.
- **Ranking, boosted BM25, and `OrderBy`:** **All 13 `rank_bench` queries improved** (`0.28×–0.97×` latency, indexing `25%` faster), with BM25 baseline dropping from `113.8 µs` to `63.7 µs` (`0.56×`) and `OrderBy created ASC` from `26.5 µs` to `7.4 µs` (`0.28×`).
- **Unfiltered ANN recall & latency:** Virtually identical recall and latency on Gaussian-50k (`0.9970` @ `84.7 µs` vs `0.9950` @ `86.3 µs` for `beam=8, LIRE=true`).
- **Detached whole-column ingest:** **`1.40M rows/s`** build at 2M rows with **`18.75 ms` atomic publish** (`4.68 ms` at 500k rows), well below the `< 100 ms` publish gate.
- **Mixed churn + maintenance:** **`15,171 ops/s`** across 8,500 mixed patches/upserts/deletes plus periodic compaction and ANN vacuum (`final / baseline` on-disk directory ratio `1.93×` prior to RocksDB manual SST compaction, driven by RocksDB WAL + L0 SST retention).

---

## 2. Unfiltered ANN (`recall_bench`)

| Dataset | Mode | Metric (`beam=8`, `rerank=100`) | `master` | `segments-v3` | Ratio (`v3 / master`) |
|---|---|---|---:|---:|---:|
| Gaussian 50k × 128d | `LIRE=false` | Build time | `3800.8 ms` | `3735.9 ms` | `0.98×` |
| Gaussian 50k × 128d | `LIRE=false` | RaBitQ `recall@10` / latency | `0.4800` (`30.1 µs`) | `0.4810` (`29.3 µs`) | `0.97×` |
| Gaussian 50k × 128d | `LIRE=false` | Reranked `recall@10` / latency | `0.9940` (`90.0 µs`) | `0.9920` (`88.8 µs`) | `0.99×` |
| Gaussian 50k × 128d | `LIRE=true` | Build time | `4045.5 ms` | `3831.1 ms` | `0.95×` |
| Gaussian 50k × 128d | `LIRE=true` | RaBitQ `recall@10` / latency | `0.4890` (`28.5 µs`) | `0.5140` (`29.3 µs`) | `1.03×` |
| Gaussian 50k × 128d | `LIRE=true` | Reranked `recall@10` / latency | `0.9950` (`86.3 µs`) | `0.9970` (`84.7 µs`) | `0.98×` |
| Uniform 10k × 128d | `LIRE=false` | Reranked `recall@10` / latency | `0.2700` (`59.9 µs`) | `0.2710` (`74.8 µs`) | `1.25×` |
| Uniform 10k × 128d | `LIRE=true` | Reranked `recall@10` / latency | `0.2770` (`62.1 µs`) | `0.2820` (`74.3 µs`) | `1.20×` |

On realistic clustered embeddings (`Gaussian-50k`), v3 matches `master` recall and latency (`0.98×–0.99×`). On uniform vectors where `cf_vectors` point reads and `cf_row_doc_ids` block resolution touch scattered keys, resolving `RowId -> DocId` via `cf_row_doc_ids` adds ~`12–15 µs` on cold blocks.

---

## 3. Filtered Vector Search (`filtered_bench`, 20k × 128d)

| Selectivity / Filter | `master` recall@10 (latency) | `segments-v3` recall@10 (latency) | Ratio | Regime / Notes |
|---|---:|---:|---:|---|
| 100% unfiltered | `1.0000` (`89.9 µs`) | `0.9990` (`86.0 µs`) | `0.96×` | `Unfiltered` |
| 50% `Eq(even=true)` | `0.9930` (`108.2 µs`) | `0.9770` (`101.2 µs`) | `0.94×` | `MaskedTraversal` |
| 10% `Eq(mod10=3)` | `0.9900` (`105.9 µs`) | **`1.0000`** (`837.2 µs`) | `7.91×` | 2,000 rows $\le$ `exact_scan_threshold (4,096)` $\to$ `ExactScan` |
| 10% `Lt(score<10)` | `0.9960` (`108.8 µs`) | **`1.0000`** (`791.2 µs`) | `7.27×` | 2,000 rows $\le$ `exact_scan_threshold (4,096)` $\to$ `ExactScan` |
| 1% `Eq(mod100=7)` | `1.0000` (`88.9 µs`) | `1.0000` (`92.2 µs`) | `1.04×` | `ExactScan` (200 rows) |
| 90% `Not(Eq(mod10=3))` | `0.9980` (`96.8 µs`) | `0.9970` (`163.0 µs`) | `1.68×` | Complement evaluation against segment postings |

**Crossover note:** Because v3 replaces per-leaf cluster bitmaps (`cf_attr_clusters`) with global per-segment postings (`cf_attr_postings`) and exact `RowSet` planning, any materialized filter with $\le$ `exact_scan_threshold` (default `4,096`) matching vector rows takes `ExactScan` to guarantee **`1.0000` exact recall** by scoring full vectors directly in `cf_vectors`. At `10%` of `20k` docs (`2,000 <= 4,096`), `filtered_bench` hits `ExactScan` (trading `837 µs` exact evaluation for `1.0000` recall); callers prioritizing latency over exact recall on 1k–4k filtered sets can lower `.with_exact_scan_threshold(...)` (e.g. `512`) so mid-selectivity queries stay on `MaskedTraversal`.

---

## 4. Full-Text Search (`fts_bench`, 10k docs, AVX-512)

| Metric / Query | `master` | `segments-v3` | Ratio |
|---|---:|---:|---:|
| Index build | `2336.5 ms` | **`2152.6 ms`** | **`0.92×`** |
| Short 3-term query (MAXSCORE) | `43.5 µs` | `89.5 µs` | `2.06×` |
| Medium 8-term query (MAXSCORE) | `78.4 µs` | `135.9 µs` | `1.73×` |
| Long 50-term LLM query (MAXSCORE) | `502.2 µs` | `665.7 µs` | `1.33×` |
| Exhaustive baseline (3 / 8 / 50 terms) | `1861 / 3557 / 14017 µs` | `1839 / 3398 / 13236 µs` | `0.94×–0.99×` |

In v3, FTS postings key directly off `(seg, block)` coordinates rather than maintaining a separate global `FtsDocId` allocation map (`FtsRevBlock` / `cf_doc_loc` / `fts_free_ids`). On a single-segment 10k collection, resolving `RowId -> DocId` via `cf_row_doc_ids` and consulting segment liveness masks adds ~`45–60 µs` on synthetic `fts_bench`, while larger multi-term / `rank_bench` BM25 queries (`20k` docs below) run **faster** end-to-end (`63.7 µs` vs `113.8 µs`).

---

## 5. Generalized `RankBy` & `OrderBy` (`rank_bench`, 20k docs)

| Operation | `master` | `segments-v3` | Ratio (`v3 / master`) |
|---|---:|---:|---:|
| Index build (20k docs) | `6512 ms` | **`4882 ms`** | **`0.75×`** |
| `BM25` (baseline) | `113.8 µs` | **`63.7 µs`** | **`0.56×`** |
| `2 × BM25` | `103.0 µs` | **`58.4 µs`** | **`0.57×`** |
| `BM25 + 3 × Filter(phrase)` | `1356.7 µs` | **`1136.1 µs`** | **`0.84×`** |
| `BM25 × (1 + 2 × Decay(created))` | `817.5 µs` | **`678.5 µs`** | **`0.83×`** |
| `BM25 × Saturate(popularity)` | `352.0 µs` | **`285.6 µs`** | **`0.81×`** |
| `BM25 + 0.05 × Decay(created)` (backfill) | `261.4 µs` | **`209.2 µs`** | **`0.80×`** |
| `BM25 × BM25` (conjunctive) | `8599.9 µs` | **`3710.2 µs`** | **`0.43×`** |
| `Saturate(popularity)` (monotone scan) | `22.9 µs` | **`8.3 µs`** | **`0.36×`** |
| `Sum(Saturate, Filter)` (candidate scan) | `13859.6 µs` | **`8285.2 µs`** | **`0.60×`** |
| `OrderBy created DESC` (unfiltered) | `22.7 µs` | **`8.9 µs`** | **`0.39×`** |
| `OrderBy created ASC` (unfiltered) | `26.5 µs` | **`7.4 µs`** | **`0.28×`** |
| `OrderBy popularity DESC, created ASC` | `30.0 µs` | **`29.2 µs`** | **`0.97×`** |
| `OrderBy created DESC, cat = c3` | `604.2 µs` | **`564.5 µs`** | **`0.93×`** |

Gate (`<= 1.0×`): **all 13 queries pass**. `cf_columns` / zero-copy segment reads and value-major `cf_attr_postings` eliminate per-leaf posting scatter across 150 leaves during `OrderBy` and modifier rescoring.

---

## 6. Faceting (`facet_bench`, 20k docs)

| Scenario | `master` | `segments-v3` | Ratio |
|---|---:|---:|---:|
| Index build (20k docs) | `5.88 s` | **`5.29 s`** | **`0.90×`** |
| Unfiltered terms + stats (`global`) | `4.67 ms` | **`4.28 ms`** | **`0.92×`** |
| Complement `status != deleted` (`complement`) | `4.73 ms` | **`4.00 ms`** | **`0.85×`** |
| Selective filter (`docs`) | `63.79 µs` | `92.97 µs` | `1.46×` |
| Broad filter `in_stock` | `14.35 ms` | **`4.33 ms`** | **`0.30×`** |
| Broad filter + disjunctive `ExcludeSelf(brand)` | `16.39 ms` | **`4.41 ms`** | **`0.27×`** |
| Vector search only (`in_stock`) | `37.55 µs` | `48.81 µs` | `1.30×` |
| Vector + facets `FilterMatches` (`in_stock`) | `16.11 ms` | **`4.48 ms`** | **`0.28×`** |
| Vector + facets `TopK` | `50.19 µs` | `70.99 µs` | `1.41×` |
| BM25 only (`in_stock`) | `276.44 µs` | **`133.35 µs`** | **`0.48×`** |
| BM25 + disjunctive facets (`in_stock`) | `15.95 ms` | **`3.92 ms`** | **`0.25×`** |
| Hybrid + facets `Bm25AndFilterMatches` | `16.10 ms` | **`3.98 ms`** | **`0.25×`** |

Broad and disjunctive faceted queries accelerate by **3.3×–4.1×** because v3 evaluates postings at segment granularity (`SegPosting` prefix scans) instead of sweeping 150 ANN leaves.

---

## 7. Bulk Import (`bulk_import_bench`)

| Dataset | Metric | `master` | `segments-v3` | Ratio / Delta |
|---|---|---:|---:|---:|
| `50k × 128d` | Wall time / Throughput | `0.90 s` (`55,519 docs/s`) | **`0.58 s` (`86,107 docs/s`)** | **`1.55×` faster** |
| `50k × 128d` | `emit_ssts` + `ingest` | `0.10 s + 0.56 s` | **`0.09 s + 0.25 s`** | **`1.94×` faster** |
| `50k × 128d` | DB Size / SST files | `30.76 MiB` / `131` SSTs | **`30.45 MiB` / `39` SSTs** | smaller, `3.4×` fewer SSTs |
| `20k × 768d + FTS` | Wall time / Throughput | `1.04 s` (`19,165 docs/s`) | **`0.75 s` (`26,734 docs/s`)** | **`1.39×` faster** |
| `20k × 768d + FTS` | `emit_ssts` + `ingest` | `0.14 s + 0.57 s` | **`0.13 s + 0.29 s`** | **`1.69×` faster** |
| `20k × 768d + FTS` | `recall@10` (`beam=8, rerank=100`) | `0.9840` | **`0.9940`** | `+0.0100` |
| `20k × 768d + FTS` | DB Size / SST files | `67.30 MiB` / `132` SSTs | **`66.92 MiB` / `40` SSTs** | smaller, `3.3×` fewer SSTs |

Gate (`throughput >= 1.0×`, `DB size <=`): **passed across every configuration**.

---

## 8. Mixed Churn & Detached Whole-Column Benchmarks (New in v3)

| Benchmark | Workload | Result |
|---|---|---|
| `churn_bench` | `init n=10,000` (64-d vectors, `seal_rows=2,000`, 2 indexed attrs) + `8,500` mixed ops (`5,000` patches + `2,500` upserts + `1,000` deletes) + `5` `maintenance()` runs | Init: **`18,717 docs/s`** (`110.46 MiB`) \| Churn + maintenance: **`15,171 ops/s`** (`0.56 s` total), final directory **`213.28 MiB` (`1.93×`)** |
| `detached_bench` (`n=500,000`) | Empty vectorless schema + 1 filterable `Int` detached column (`Replace` mode) | Build: `0.442 s` (**`1.13M rows/s`**) \| Atomic publish: **`4.68 ms`** (`catchup=0`) |
| `detached_bench` (`n=2,000,000`) | Empty vectorless schema + 1 filterable `Int` detached column (`Replace` mode) | Build: `1.430 s` (**`1.40M rows/s`**) \| Atomic publish: **`18.75 ms`** (`catchup=0`) |

**Notes:**
- `detached_bench` easily clears the `< 100 ms` publish gate (`4.68 ms` at 500k rows, `18.75 ms` at 2M rows) because `publish_detached` atomically swaps `DetachedMeta.current_gen` and live counts (`default[0x07]`) in one `WriteBatch` while bulk data (`cf_detached` / `cf_detached_postings`) is already ingested via SSTs during `build()`.
- `churn_bench` measures raw directory byte size immediately after 8,500 random writes (`1.93×` baseline including RocksDB WAL logs and uncompacted L0 SSTs created by the compaction filter + CoW row appends).
