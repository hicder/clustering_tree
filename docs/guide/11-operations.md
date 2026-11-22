# Operations & Troubleshooting

## On-disk layout

A collection is one directory managed by RocksDB (many `.sst` files, a `MANIFEST`, `LOG`, write-ahead logs). Treat it as opaque: don't edit files inside it. Back it up with the process closed (copy the whole directory), or use RocksDB-level tooling.

## Opening and closing

- `Collection::open(path, config)` creates or opens. Opening loads the tree's internal nodes into memory, so large collections take longer to open.
- A RocksDB database can be opened by **one process at a time**; a second `open` of the same path fails (RocksDB's `LOCK` file).
- There is no explicit `close`: dropping the `Collection` closes it.

## Concurrency model

- Reads (`search`, `query_bm25`, `hybrid_search`, `facet`, `get`, `resolve_filter_doc_ids`) take `&self`.
- Writes (`upsert`, `patch*`, `delete*`, `update_by_filter`, `compact_fts_stats`) take `&mut self`.

Rust's borrow rules therefore give you a single writer *or* many readers within a single `&mut` / `&` borrow. To share a collection across threads, choose a wrapper that fits your app, e.g. `Arc<RwLock<Collection>>` (readers in parallel, writers exclusive). Keep write critical sections short, as a writer blocks readers.

## Durability

Each write is applied as one atomic RocksDB `WriteBatch`: after an error or a crash you will see the whole write or none of it, never a half-updated index. Writes go through RocksDB's write-ahead log with its default (non-fsync-per-write) settings, so they survive a process crash, but the most recent writes could be lost if the operating system or machine itself crashes before the OS flushes the log. If you need to be certain the latest data is on disk before reporting success, plan for that at the application level (for instance, re-derivable sources of truth, or bulk reloads).

Two operations are *not* single-transaction, because they touch an unbounded number of documents: `update_by_filter` and `delete_by_filter` apply changes in bounded atomic chunks. A crash in the middle leaves some chunks applied and some not; re-running the same call is safe and finishes the job.

## Errors

All APIs return `izzydb::Result<T>`; the error type is `TreeError` (implements `std::error::Error`).

| Variant | Typical cause | What to do |
|---|---|---|
| `InvalidConfig(msg)` | Bad `TreeConfig` / bulk config (e.g. `min_leaf_size * 2 >= max_leaf_size`, `dim = 0`). | Fix the config. |
| `InvalidSchema(msg)` | Invalid attribute or full-text declaration (duplicate names, `stemming` + `case_sensitive`, FTS on a non-string attribute). | Fix the schema. |
| `ConfigMismatch { field, expected, actual }` | Re-opening with a different `dim`, `metric`, attribute type, `filterable`, or full-text config. | Pass the original config; to change it, re-create and re-ingest. |
| `DimensionMismatch { expected, actual }` | Vector length ≠ `dim`. | Check your embedding model. |
| `NonFiniteVector` | `NaN` or `±Inf` in a vector. | Sanitize inputs. |
| `InvalidAttribute { name, reason }` | Undeclared attribute, wrong value type, `NaN` float. | Match the schema. |
| `InvalidFilter(msg)` | Unknown/non-filterable attribute, wrong value type, `Eq` on an array, range on `Bool`, … | See [Filtering](05-filtering.md). |
| `InvalidQuery(msg)` | Bad `RankBy`, hybrid options, or facet spec. | See [Full-Text](06-full-text-search.md), [Hybrid](07-hybrid-search.md), [Ranking & Boosting](08-ranking-and-boosting.md), [Faceting](09-faceting.md). |
| `DocumentNotFound(id)` | `patch` / `patch_batch` on a missing id. | `upsert` instead, or check first with `get`. |
| `Storage(msg)` | RocksDB reported an I/O or lock error (disk full, permissions, DB already open elsewhere). | Check the message and the environment. |
| `CorruptedData(msg)` | Unreadable stored data, or an interrupted bulk import. | For an interrupted import, delete the directory and re-run the import. Otherwise restore from backup. |
| `InvariantViolation(msg)` | An internal consistency check failed. | Run `verify_invariants` and report a bug with the output. |

## Verifying a collection

```rust
let stats = col.verify_invariants()?;
println!("{} docs in {} leaves", stats.num_docs, stats.tree.leaf_nodes);
```

`verify_invariants` exhaustively checks the tree structure, document locations, attribute indexes, and the full-text index, and returns `CollectionStats` (document, bitmap, term, posting counts, and tree shape). It reads the **entire** collection, so use it in tests, after bulk imports, or in an offline maintenance job, not on a hot path.

## Performance tips

**Writes**

- Use [bulk import](10-bulk-import.md) for the initial load.
- Index only what you query: each `filterable` and full-text attribute adds work to every write ([Collections & Schema](02-collections-and-schema.md#declaring-attributes)).
- `patch` a few attributes instead of `upsert`ing a whole document.
- Prefer `patch_batch` / `delete_batch` to loops of single calls.

**Vector search**

- Tune `beam_width` and `rerank_candidates` against measured recall ([Vector Search](04-vector-search.md#choosing-values)).
- Use `Cosine` with normalized-friendly embeddings; IzzyDB normalizes for you.

**Full-text**

- Turn off `index_positions` on attributes where you never run phrase queries.
- After heavy churn, `compact_fts_stats` can restore tight pruning bounds.

**Build**

- Always benchmark and deploy with `--release`; debug builds are very slow for SIMD-heavy code.
- Distance and BM25 kernels pick the best SIMD instruction set at runtime (AVX2/AVX-512 on x86_64, NEON on aarch64), so no special flags are needed.

## Benchmarks you can run

```bash
cargo run --release -p izzydb-bench --bin recall_bench      # ANN recall / latency
cargo run --release -p izzydb-bench --bin filtered_bench    # filtered vector search
cargo run --release -p izzydb-bench --bin fts_bench         # BM25
cargo run --release -p izzydb-bench --bin facet_bench       # faceting
cargo run --release -p izzydb-bench --bin bulk_import_bench # bulk import
```

## Cargo features

| Feature (crate `izzydb`) | Effect |
|---|---|
| `fts-cjk` | Adds the `SmartCn`, `Kuromoji`, and `Nori` tokenizers for Chinese, Japanese, and Korean. |

## Frequently asked questions

**Do I have to run an indexing/refresh step after writing?** No. Writes update all indexes atomically, and the next read sees them.

**Can a document have no attributes?** Yes. Pass an empty `Attributes::new()`. Every document needs a vector.

**Can I store multiple vectors per document?** A document has exactly one embedding. To model several (title and body embeddings, chunks of a long document), create one document per vector, give them distinct ids, and put the parent id in an attribute such as `parent_id` (`Uint`, filterable) so you can filter and group.

**What are the id limits?** `DocId` is any `u64`.

**Why do I get fewer than `k` results?** The collection may hold fewer than `k` matching documents; a very selective filter may need a larger `max_expanded_leaves` ([Vector Search](04-vector-search.md#filtered-vector-search)); or `k` of a BM25 query is limited to documents matching at least one query term.

**How do I change a schema attribute's type?** You can't in place; see [Collections & Schema](02-collections-and-schema.md#opening-re-opening-and-schema-evolution).

**Is it a server?** No, it is a library; wrap it in your own service if you need network access.
