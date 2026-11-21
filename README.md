# IzzyDB

A mutable, RocksDB-backed hierarchical clustering tree (~100x fanout) for approximate nearest neighbor search, inspired by turbopuffer ANN v3 and SPFresh (LIRE).

- **Quantized posting lists:** Leaf posting lists (`cf_leaf_entries`) store 1-bit RaBitQ codes relative to each leaf centroid (~21x smaller than `f32` at 128d).
- **Full-vector store:** Full-precision vectors are stored separately in `cf_vectors` for exact tree splits, LIRE reassignment, and optional Stage 2 reranking.
- **Mutable & persistent:** Supports inserts, upserts, and deletes with balanced 2-means splits, underflow merges/redistributions, SPFresh LIRE boundary reassignment, and atomic `WriteBatch` commits.
- **Cluster-aware attribute filtering:** `Collection` pairs `ClusteringTree` with a typed attribute schema, forward document storage, and cluster/leaf bitmap indexes for filtered ANN search with dynamic expansion and exhaustive-scan crossover.
- **Full-text search & hybrid RRF:** Cluster-decoupled `FtsIndex` with `alyze` tokenization, Snowball stemming, stopword removal, ASCII folding, phrase position indexing, vectorized Block-Max MAXSCORE BM25, and Reciprocal Rank Fusion (`hybrid_search`).

## Workspace Crates

- [`izzydb-core`](crates/izzydb-core) — configuration, error types, schema & filter AST, `RankBy` AST, distance metrics, SIMD vector kernels, and balanced 2-means clustering.
- [`izzydb-fts`](crates/izzydb-fts) — text analyzer, 256-doc SoA posting/position blocks, and Block-Max MAXSCORE BM25 execution.
- [`izzydb-rabitq`](crates/izzydb-rabitq) — 1-bit RaBitQ quantization, randomized FWHT rotation, and SIMD popcount leaf scoring.
- [`izzydb-storage`](crates/izzydb-storage) — packed node/metadata codecs and RocksDB storage/transaction layer.
- [`izzydb`](crates/izzydb) — `Collection` and `ClusteringTree` indexes, filtered & unfiltered vector search, BM25 & hybrid search, SPFresh LIRE reassignment, and structural invariant verification.

## Quick Start

Define a schema on `CollectionConfig` (setting `filterable = true` for attributes you want indexed for filtering, or `.with_fts_attribute(...)` to enable full-text indexing), upsert vectors with `Attributes`, and query via `Collection::search`, `Collection::query_bm25`, or `Collection::hybrid_search`:

```rust
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric, Filter,
    FullTextSearchConfig, HybridSearchOptions, Language, RankBy, SearchOptions, TreeConfig,
};

let fts_en = FullTextSearchConfig::default()
    .with_language(Language::English)
    .with_stemming(true)
    .with_remove_stopwords(true)
    .with_ascii_folding(true);

let config = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine))
    .with_attribute("category", AttributeType::String, true)
    .with_attribute("price", AttributeType::Float, true)
    .with_attribute("in_stock", AttributeType::Bool, true)
    .with_attribute("tags", AttributeType::StringArray, true)
    .with_fts_attribute("title", AttributeType::String, false, fts_en.clone())
    .with_fts_attribute("description", AttributeType::String, false, fts_en);

let mut col = Collection::open("/tmp/my_collection", config)?;

let mut attrs = Attributes::new();
attrs.insert("category".into(), "shoes".into());
attrs.insert("price".into(), 79.99_f64.into());
attrs.insert("in_stock".into(), true.into());
attrs.insert("tags".into(), vec!["running".to_string(), "sale".to_string()].into());
attrs.insert("title".into(), "Trail Runner Pro".into());
attrs.insert(
    "description".into(),
    "Lightweight waterproof trail running shoe with high-traction sole.".into(),
);

col.upsert(1, &vec![0.1; 128], attrs)?;

// Unfiltered vector search
let opts = SearchOptions::new(10, 8).with_rerank(100);
let all_hits = col.search(&query, opts, None)?;

// Filtered vector search combining attribute predicates and full-text phrase matching
let filter = Filter::And(vec![
    Filter::Eq("category".into(), "shoes".into()),
    Filter::Eq("in_stock".into(), true.into()),
    Filter::Lte("price".into(), 100.0_f64.into()),
    Filter::ContainsAny("tags".into(), vec!["running".into(), "trail".into()]),
    Filter::ContainsTokenSequence {
        attr: "description".into(),
        query: "trail running".into(),
        last_as_prefix: false,
    },
]);

let filtered_opts = SearchOptions::new(10, 8)
    .with_rerank(100)
    .with_max_expanded_leaves(256);
let hits = col.search(&query, filtered_opts, Some(&filter))?;

// BM25 full-text search (single field or weighted multi-field)
let rank_by = RankBy::sum(vec![
    RankBy::product(2.0, RankBy::bm25("title", "trail runner")),
    RankBy::bm25("description", "waterproof trail shoe"),
]);
let bm25_hits = col.query_bm25(&rank_by, Some(&filter), 10)?;

// Hybrid vector + BM25 search fused with Reciprocal Rank Fusion (RRF)
let hybrid_opts = HybridSearchOptions::new(10, SearchOptions::new(20, 8).with_rerank(100))
    .with_bm25_candidates(20)
    .with_rrf_k(60.0)
    .with_weights(1.0, 1.5);
let hybrid_hits = col.hybrid_search(&query, &rank_by, hybrid_opts, Some(&filter))?;
```

### Full-Text Search Examples

`FullTextSearchConfig` supports `Tokenizer::WordV4` (default UAX #29), `Tokenizer::Whitespace`, and `Tokenizer::PreTokenizedArray` (for `StringArray` attributes), along with Snowball stemming, stopword removal, ASCII folding, BM25 parameter tuning (`k1`, `b`, `k3`), and optional position indexing (`index_positions`).

```rust
use izzydb::{
    AttributeType, Bm25Overrides, CollectionConfig, DistanceMetric, Filter, FullTextSearchConfig,
    RankBy, TokenQuery, Tokenizer, TreeConfig,
};

// Pre-tokenized string array attribute (tokens stored verbatim)
let config = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine))
    .with_fts_attribute("title", AttributeType::String, false, FullTextSearchConfig::default())
    .with_fts_attribute("body", AttributeType::String, false, FullTextSearchConfig::default())
    .with_fts_attribute(
        "keywords",
        AttributeType::StringArray,
        false,
        FullTextSearchConfig::default().with_tokenizer(Tokenizer::PreTokenizedArray),
    );

// Single-field BM25 query (col.query is an alias for col.query_bm25)
let simple = RankBy::bm25("body", "waterproof trail");
let hits = col.query_bm25(&simple, None, 10)?;

// Prefix / autocomplete matching on the last token and query-time BM25 overrides
let prefix_rank = RankBy::bm25_with(
    "title",
    "trail run",
    Bm25Overrides::default()
        .with_last_as_prefix(true)
        .with_k1(1.5)
        .with_b(0.5),
);

// Composite multi-field ranking with Sum, Max, and Product weights
let multi_field = RankBy::max(vec![
    RankBy::product(2.5, RankBy::bm25("title", "trail shoe")),
    RankBy::bm25("body", "trail shoe"),
]);

// Querying a PreTokenizedArray attribute requires TokenQuery::PreTokenized
let pre_tokenized = RankBy::bm25(
    "keywords",
    TokenQuery::PreTokenized(vec!["gore-tex".into(), "vibram".into()]),
);

// Filtered BM25 using any Filter (attribute, FTS, or combined)
let only_in_stock = Filter::Eq("in_stock".into(), true.into());
let filtered_bm25 = col.query_bm25(&multi_field, Some(&only_in_stock), 10)?;
```

### Filter Examples

Supported attribute types are `String`, `Int` (`i64`), `Uint` (`u64`), `Float` (`f64`), `Bool`, `StringArray`, `IntArray`, and `UintArray`. Attributes used in scalar/array predicates must be registered with `filterable = true`; attributes used in token predicates must have `FullTextSearchConfig` enabled (`ContainsTokenSequence` also requires `index_positions = true`, the default).

```rust
use izzydb::{Filter, TokenQuery};

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

// Full-text token & phrase matching
let all_tokens = Filter::ContainsAllTokens {
    attr: "description".into(),
    query: "waterproof trail".into(),
    last_as_prefix: false,
};
let any_token_prefix = Filter::ContainsAnyToken {
    attr: "title".into(),
    query: "trail run".into(),
    last_as_prefix: true,
};
let phrase = Filter::ContainsTokenSequence {
    attr: "description".into(),
    query: "trail running shoe".into(),
    last_as_prefix: false,
};
let pre_tok_filter = Filter::ContainsAnyToken {
    attr: "keywords".into(),
    query: TokenQuery::PreTokenized(vec!["vibram".into()]),
    last_as_prefix: false,
};

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
- [Full-Text Search Architecture](docs/fts-architecture.md) — `FtsIndex`, cluster-decoupled 256-doc posting blocks, Block-Max MAXSCORE BM25, FTS filters, and hybrid RRF.

## Tests & Benchmarks

```bash
cargo test
cargo run --release -p izzydb --example recall_bench
cargo run --release -p izzydb --example filtered_bench
cargo run --release -p izzydb --example fts_bench
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
