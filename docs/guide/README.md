# IzzyDB User Guide

IzzyDB is an **embedded** hybrid search engine for Rust. It runs inside your process (no server), stores everything in a [RocksDB](https://rocksdb.org) directory, and combines three kinds of retrieval over a single set of documents:

| Capability | Question it answers | Guide |
|---|---|---|
| Vector search | "Which documents are *semantically closest* to this embedding?" | [Vector Search](04-vector-search.md) |
| Attribute filtering | "…but only `category = shoes AND price < 100`." | [Filtering](05-filtering.md) |
| Full-text (BM25) | "Which documents best match these *words*?" | [Full-Text Search](06-full-text-search.md) |
| Hybrid (RRF) | "Combine vector and keyword rankings." | [Hybrid Search](07-hybrid-search.md) |
| Facets | "How many results per category / price range?" | [Faceting](08-faceting.md) |

## Reading order

New to IzzyDB? Read 1 → 3, then jump to whichever search mode you need.

1. [Getting Started](01-getting-started.md) — add the dependency, create a collection, insert, query.
2. [Collections & Schema](02-collections-and-schema.md) — `CollectionConfig`, `TreeConfig`, attribute types, schema evolution.
3. [Documents](03-documents.md) — `upsert`, `get`, `patch`, `delete`, and the `*_by_filter` bulk mutations.
4. [Vector Search](04-vector-search.md) — distance metrics, `SearchOptions`, recall/latency tuning.
5. [Filtering](05-filtering.md) — the `Filter` expression tree and its exact semantics (nulls, negation, ranges).
6. [Full-Text Search](06-full-text-search.md) — analyzers, BM25, `RankBy` composition, prefix search.
7. [Hybrid Search](07-hybrid-search.md) — Reciprocal Rank Fusion of vector + BM25.
8. [Faceting](08-faceting.md) — `Terms`, `Ranges`, `Stats`, `Histogram`, multi-select facets.
9. [Bulk Import](09-bulk-import.md) — loading millions of rows from Parquet with `izzydb-bulk`.
10. [Operations & Troubleshooting](10-operations.md) — durability, reopening, concurrency, verification, errors, performance tips.

## Mental model in 60 seconds

- A **`Collection`** is a directory on disk. You open it with a `CollectionConfig` that declares the vector dimension, the distance metric, and a typed **schema** of attributes.
- A **document** is `(DocId: u64, embedding: [f32; dim], attributes)`. Every document must have a vector.
- Attributes can be **filterable** (indexed for `Filter` and facets) and/or **full-text** (indexed for BM25 and token predicates).
- Every write (`upsert`, `patch`, `delete`, …) updates the vector index, attribute indexes, and full-text index in **one atomic RocksDB write batch**. There is no "refresh" step: after a write returns, the next query sees it.
- Writes take `&mut Collection`; searches take `&self`.

## Where to go for internals

This guide is about *using* IzzyDB. For how it works inside (key encodings, clustering tree, RaBitQ, Block-Max MAXSCORE, …) see [`docs/architecture`](../architecture/), and for measurements see [`docs/perf`](../perf/).
