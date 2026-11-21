# IzzyDB

A mutable, RocksDB-backed hierarchical clustering tree (~100x fanout) for approximate nearest neighbor search, inspired by turbopuffer ANN v3 and SPFresh (LIRE).

- **Quantized posting lists:** Leaf posting lists (`cf_leaf_entries`) store 1-bit RaBitQ codes relative to each leaf centroid (~21x smaller than `f32` at 128d).
- **Full-vector store:** Full-precision vectors are stored separately in `cf_vectors` for exact tree splits, LIRE reassignment, and optional Stage 2 reranking.
- **Mutable & persistent:** Supports inserts, upserts, and deletes with balanced 2-means splits, underflow merges/redistributions, SPFresh LIRE boundary reassignment, and atomic `WriteBatch` commits.

## Workspace Crates

- [`izzydb-core`](crates/izzydb-core) — configuration, error types, distance metrics, SIMD vector kernels, and balanced 2-means clustering.
- [`izzydb-rabitq`](crates/izzydb-rabitq) — 1-bit RaBitQ quantization, randomized FWHT rotation, and SIMD popcount leaf scoring.
- [`izzydb-storage`](crates/izzydb-storage) — packed node/metadata codecs and RocksDB storage/transaction layer.
- [`izzydb`](crates/izzydb) — `ClusteringTree` index, search path, SPFresh LIRE reassignment, and structural invariant verification.

## Quick Start

```rust
use izzydb::{ClusteringTree, DistanceMetric, TreeConfig};

let config = TreeConfig::new(128, DistanceMetric::L2Squared);
let mut tree = ClusteringTree::open("/tmp/my_tree", config)?;

tree.add_leaf(1, &vec![0.1; 128])?;
let approx = tree.search(&query, 10, 8)?;
let reranked = tree.search_with_rerank(&query, 10, 8, 100)?;
```

## Documentation

- [Key & Binary Encoding](docs/key-encoding.md) — RocksDB column families, big-endian key formats, packed SoA node/vector layouts, and RaBitQ bit-packing.
- [Index Path](docs/index-path.md) — write path, RocksDB layout, RaBitQ quantization, splits, merges, and LIRE.
- [Search Path](docs/search-path.md) — beam traversal, asymmetric 4-bit RaBitQ `popcount` scanning, and full-vector reranking.

## Tests & Benchmarks

```bash
cargo test
cargo run --release -p izzydb --example recall_bench
```
