# IzzyDB

A mutable, RocksDB-backed hierarchical clustering tree (~100x fanout) for approximate nearest neighbor search, inspired by turbopuffer ANN v3 and SPFresh (LIRE).

- **Quantized posting lists:** Leaf posting lists (`cf_leaf_entries`) store 1-bit RaBitQ codes relative to each leaf centroid (~21x smaller than `f32` at 128d).
- **Full-vector store:** Full-precision vectors are stored separately in `cf_vectors` for exact tree splits, LIRE reassignment, and optional Stage 2 reranking.
- **Mutable & persistent:** Supports inserts, upserts, and deletes with balanced 2-means splits, underflow merges/redistributions, SPFresh LIRE boundary reassignment, and atomic `WriteBatch` commits.
- **Cluster-aware attribute filtering:** `Collection` pairs `ClusteringTree` with a typed attribute schema, forward document storage, and cluster/leaf bitmap indexes for filtered ANN search with dynamic expansion and exhaustive-scan crossover.

## Workspace Crates

- [`izzydb-core`](crates/izzydb-core) — configuration, error types, schema & filter AST, distance metrics, SIMD vector kernels, and balanced 2-means clustering.
- [`izzydb-rabitq`](crates/izzydb-rabitq) — 1-bit RaBitQ quantization, randomized FWHT rotation, and SIMD popcount leaf scoring.
- [`izzydb-storage`](crates/izzydb-storage) — packed node/metadata codecs and RocksDB storage/transaction layer.
- [`izzydb`](crates/izzydb) — `Collection` and `ClusteringTree` indexes, filtered & unfiltered search paths, SPFresh LIRE reassignment, and structural invariant verification.

## Quick Start

Define a schema on `CollectionConfig` (setting `filterable = true` for attributes you want indexed for filtering, or `false` for payload-only stored attributes), upsert vectors with `Attributes`, and pass a `Filter` to `Collection::search`:

```rust
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric, Filter,
    SearchOptions, TreeConfig,
};

let config = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine))
    .with_attribute("category", AttributeType::String, true)
    .with_attribute("price", AttributeType::Float, true)
    .with_attribute("in_stock", AttributeType::Bool, true)
    .with_attribute("tags", AttributeType::StringArray, true)
    .with_attribute("title", AttributeType::String, false); // stored, not indexed

let mut col = Collection::open("/tmp/my_collection", config)?;

let mut attrs = Attributes::new();
attrs.insert("category".into(), "shoes".into());
attrs.insert("price".into(), 79.99_f64.into());
attrs.insert("in_stock".into(), true.into());
attrs.insert("tags".into(), vec!["running".to_string(), "sale".to_string()].into());
attrs.insert("title".into(), "Trail Runner".into());

col.upsert(1, &vec![0.1; 128], attrs)?;

// Unfiltered search
let opts = SearchOptions::new(10, 8).with_rerank(100);
let all_hits = col.search(&query, opts, None)?;

// Filtered search with boolean combinators, ranges, and array predicates
let filter = Filter::And(vec![
    Filter::Eq("category".into(), "shoes".into()),
    Filter::Eq("in_stock".into(), true.into()),
    Filter::Lte("price".into(), 100.0_f64.into()),
    Filter::ContainsAny("tags".into(), vec!["running".into(), "trail".into()]),
]);

let filtered_opts = SearchOptions::new(10, 8)
    .with_rerank(100)
    .with_max_expanded_leaves(256);
let hits = col.search(&query, filtered_opts, Some(&filter))?;
```

### Filter Examples

Supported attribute types are `String`, `Int` (`i64`), `Uint` (`u64`), `Float` (`f64`), `Bool`, `StringArray`, `IntArray`, and `UintArray`. All attributes referenced in a `Filter` must be registered with `filterable = true`.

```rust
use izzydb::Filter;

// Equality and set membership (scalar attributes)
let eq = Filter::Eq("category".into(), "books".into());
let neq = Filter::NotEq("category".into(), "archive".into());
let in_set = Filter::In("category".into(), vec!["books".into(), "games".into()]);
let not_in = Filter::NotIn("rank".into(), vec![0_i64.into(), 1_i64.into()]);

// Range comparisons (String, Int, Uint, Float)
let range = Filter::And(vec![
    Filter::Gte("price".into(), 10.0_f64.into()),
    Filter::Lt("price".into(), 50.0_f64.into()),
]);

// Array membership (StringArray, IntArray, UintArray)
let has_tag = Filter::Contains("tags".into(), "sale".into());
let has_any = Filter::ContainsAny("tags".into(), vec!["sale".into(), "new".into()]);

// Null / missing attribute checks
let missing = Filter::IsNull("in_stock".into());
let present = Filter::IsNotNull("in_stock".into());

// Boolean logic: And, Or, Not
let complex = Filter::Or(vec![
    Filter::And(vec![
        Filter::Eq("category".into(), "shoes".into()),
        Filter::Not(Box::new(Filter::Eq("in_stock".into(), false.into()))),
    ]),
    Filter::Contains("tags".into(), "featured".into()),
]);
```

## Documentation

- [Key & Binary Encoding](docs/key-encoding.md) — RocksDB column families, big-endian key formats, packed SoA node/vector layouts, and RaBitQ bit-packing.
- [Index Path](docs/index-path.md) — write path, RocksDB layout, RaBitQ quantization, splits, merges, and LIRE.
- [Search Path](docs/search-path.md) — beam traversal, asymmetric 4-bit RaBitQ `popcount` scanning, cluster-aware filtering, and full-vector reranking.
- [Filtering & Inverted Index](docs/filtering-and-inverted-index.md) — schema & filter semantics, order-preserving encodings, two-level `RoaringBitmap` index, `LocalId` lifecycle, and filtered execution.

## Tests & Benchmarks

```bash
cargo test
cargo run --release -p izzydb --example recall_bench
cargo run --release -p izzydb --example filtered_bench
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
