# IzzyDB

An embedded, RocksDB-backed hybrid search engine in Rust for vector, full-text, and filtered queries.

- **Segmented storage:** Append-only segments (`RowId = (SegmentId, LocalId)`) with copy-on-write updates, background compaction + rank-based forwarding, lazy ANN vacuuming, and RocksDB compaction filters.
- **Vector search (optional):** Mutable global clustering tree over `RowId`s with RaBitQ quantization and optional exact reranking — or create a vectorless collection with `CollectionConfig::without_vectors()`.
- **Attribute filtering & detached columns:** Typed `Inline` and `Detached` attribute schemas with value-major per-segment inverted indexes, exact live counts, per-segment zone maps, zero-copy lazy row evaluation, and lock-free whole-column detached replacement (`> 1M rows/s` build, `< 20 ms` publish).
- **Full-text & hybrid:** Segment-keyed Block-Max MAXSCORE BM25 with unified v4 `FtsChunk` skip+posting chunks (`TfEncoding` compression, inline rare-term postings, segment-parallel MAXSCORE via `fts_parallel_segments`), stemming, phrase matching, and Reciprocal Rank Fusion hybrid search.
- **Ranking, boosting & sorting:** Composable scoring expressions (filter boosts, saturation and decay attribute signals, additive and multiplicative combinators) and multi-key attribute sorting with index-order top-k early termination.
- **Faceting:** Exact terms, ranges, stats, and histogram facets over filter, BM25, vector, and hybrid queries, with multi-select (disjunctive) faceting and O(values) unfiltered counts from maintained per-value counters.
- **Persistent & accelerated:** Atomic RocksDB batch commits, per-CF table tuning (partitioned indexes, no Bloom on guaranteed-hit CFs, uncompressed `cf_columns`/`cf_detached`), a configurable shared block cache (`CollectionConfig::with_block_cache_bytes`), optional `direct_io` (`O_DIRECT`) and opt-in `io-uring` Cargo feature, and runtime-dispatched SIMD (`x86_64` AVX2/AVX-512, `aarch64` NEON).

## Workspace Crates

- [`izzydb-core`](crates/izzydb-core) — configuration, error types, schema & filter AST, `RankBy` AST, distance metrics, SIMD vector kernels, and balanced 2-means clustering.
- [`izzydb-fts`](crates/izzydb-fts) — text analyzer, 256-doc SoA posting/position blocks, and Block-Max MAXSCORE BM25 execution.
- [`izzydb-rabitq`](crates/izzydb-rabitq) — 1-bit RaBitQ quantization, randomized FWHT rotation, and SIMD popcount leaf scoring.
- [`izzydb-codec`](crates/izzydb-codec) — keyspace layout and binary codecs: key builders, order-preserving scalar encodings, posting lists, segment metadata, and the packed row format.
- [`izzydb-storage`](crates/izzydb-storage) — packed ANN node/column block formats and the RocksDB storage/transaction layer (re-exports `izzydb-codec` as `codec`).
- [`izzydb`](crates/izzydb) — `Collection` (segments, compaction, detached columns, filtered & unfiltered vector search, BM25 & hybrid search, SPFresh LIRE reassignment, and structural invariant verification).
- [`izzydb-bulk`](crates/izzydb-bulk) — parallel offline Parquet bulk importer and appender (`bulk_import`, `bulk_append`, `build_ssts`, `ingest_ssts`, and the `izzydb-import` CLI with `import`, `append`, and `columns`) building sorted external SSTs.

## Quick Start

Define a schema on `CollectionConfig` (setting `filterable = true` for attributes indexed for filtering and faceting, or `.with_fts_attribute(...)` to enable full-text indexing), upsert vectors with `Attributes`, and query via `Collection::search`, `Collection::query_bm25`, `Collection::hybrid_search`, or `Collection::facet`:

- **Full-text search:** `FullTextSearchConfig` supports `Tokenizer::WordV4` (default UAX #29), `Tokenizer::Whitespace`, and `Tokenizer::PreTokenizedArray` (for `StringArray` attributes), along with Snowball stemming, stopword removal, ASCII folding, BM25 parameter tuning (`k1`, `b`, `k3`), and optional position indexing (`index_positions`).
- **Filters:** Supported attribute types are `String`, `Int` (`i64`), `Uint` (`u64`), `Float` (`f64`), `Bool`, `StringArray`, `IntArray`, and `UintArray`. Attributes used in scalar/array predicates must be registered with `filterable = true`; attributes used in token predicates must have `FullTextSearchConfig` enabled (`ContainsTokenSequence` also requires `index_positions = true`, the default).
- **Facets:** Every `filterable` attribute is facetable. Facets are exact; each `FacetSpec` needs a unique output name (use `with_alias` for several facets on one attribute).

```rust
use izzydb::{
    AttributeType, Attributes, Bm25Overrides, Collection, CollectionConfig, DecayMode,
    DistanceMetric, FacetRange, FacetSort, FacetSpec, Filter, FullTextSearchConfig,
    HybridFacetScope, HybridSearchOptions, Language, OrderBySpec, RankBy, SearchOptions,
    TokenQuery, Tokenizer, TreeConfig, VectorFacetScope,
};

let fts_en = FullTextSearchConfig::default()
    .with_language(Language::English)
    .with_stemming(true)
    .with_remove_stopwords(true)
    .with_ascii_folding(true);

let config = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine))
    .with_attribute("category", AttributeType::String, true)
    .with_attribute("rank", AttributeType::Int, true)
    .with_attribute("price", AttributeType::Float, true)
    .with_attribute("in_stock", AttributeType::Bool, true)
    .with_attribute("tags", AttributeType::StringArray, true)
    .with_fts_attribute("title", AttributeType::String, false, fts_en.clone())
    .with_fts_attribute("description", AttributeType::String, false, fts_en)
    .with_fts_attribute(
        "keywords",
        AttributeType::StringArray,
        false,
        FullTextSearchConfig::default().with_tokenizer(Tokenizer::PreTokenizedArray),
    );

let mut col = Collection::open("/tmp/my_collection", config)?;

let mut attrs = Attributes::new();
attrs.insert("category".into(), "shoes".into());
attrs.insert("rank".into(), 5_i64.into());
attrs.insert("price".into(), 79.99_f64.into());
attrs.insert("in_stock".into(), true.into());
attrs.insert("tags".into(), vec!["running".to_string(), "sale".to_string()].into());
attrs.insert("title".into(), "Trail Runner Pro".into());
attrs.insert(
    "description".into(),
    "Lightweight waterproof trail running shoe with high-traction sole.".into(),
);
attrs.insert("keywords".into(), vec!["gore-tex".to_string(), "vibram".to_string()].into());
col.upsert(1, Some(&vec![0.1; 128]), attrs)?;

let query = vec![0.1_f32; 128];

// --- Vector Search ---

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

// --- Full-Text Search (BM25) ---

// Single-field BM25 query (col.query is an alias for col.query_bm25)
let simple = RankBy::bm25("description", "waterproof trail");
let simple_hits = col.query_bm25(&simple, None, 10)?;

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
let rank_by = RankBy::sum(vec![
    RankBy::product(2.0, RankBy::bm25("title", "trail runner")),
    RankBy::bm25("description", "waterproof trail shoe"),
]);
let multi_field = RankBy::max(vec![
    RankBy::product(2.5, RankBy::bm25("title", "trail shoe")),
    RankBy::bm25("description", "trail shoe"),
]);
let bm25_hits = col.query_bm25(&rank_by, Some(&filter), 10)?;

// Boosting: filter boosts, recency / popularity signals, multiplicative products
let boosted = RankBy::boost_multiplicative(
    RankBy::bm25("description", "waterproof trail shoe"),
    2.0,
    RankBy::decay_from("created_at", now, 30.0 * DAY).with_decay_mode(DecayMode::Backward),
);
let boosted_hits = col.query(&boosted, Some(&filter), 10)?;

// Sort by attributes (index-order scan, stops after top_k)
let cheapest = col.query_ordered(&[OrderBySpec::asc("price")], Some(&filter), 10)?;

// Querying a PreTokenizedArray attribute requires TokenQuery::PreTokenized
let pre_tokenized = RankBy::bm25(
    "keywords",
    TokenQuery::PreTokenized(vec!["gore-tex".into(), "vibram".into()]),
);

// --- Hybrid Vector + BM25 Search (RRF) ---

let hybrid_opts = HybridSearchOptions::new(10, SearchOptions::new(20, 8).with_rerank(100))
    .with_bm25_candidates(20)
    .with_rrf_k(60.0)
    .with_weights(1.0, 1.5);
let hybrid_hits = col.hybrid_search(&query, &rank_by, hybrid_opts.clone(), Some(&filter))?;

// --- Filters ---

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

// --- Facets ---

let facets = vec![
    // Top categories; `exclude_self` drops the category clause from this facet's filter so the
    // other categories stay visible (multi-select / disjunctive faceting).
    FacetSpec::terms("category").with_limit(5).with_missing().exclude_self(),
    // Typeahead over array values, ordered by value.
    FacetSpec::terms("tags").with_alias("tag_search").with_prefix("sa").with_sort(FacetSort::ValueAsc),
    FacetSpec::ranges("price", vec![
        FacetRange::below(25.0_f64).with_label("budget"),
        FacetRange::between(25.0_f64, 100.0_f64),
        FacetRange::from_value(100.0_f64),
    ]),
    FacetSpec::stats("price").with_alias("price_stats"),
    FacetSpec::histogram("price", 20.0).with_alias("price_hist"),
];

let facet_filter = Filter::And(vec![
    Filter::Eq("category".into(), "shoes".into()),
    Filter::Eq("in_stock".into(), true.into()),
]);

// Filter-only: facets over every document matching `facet_filter`.
let results = col.facet(Some(&facet_filter), &facets)?;
if let Some(cats) = results.terms("category") {
    for b in &cats.buckets {
        println!("{:?}: {}", b.value, b.count);
    }
}
let price = results.stats("price_stats").unwrap();
println!("avg price {:?}, missing {}", price.avg, price.missing_count);

// Vector search + facets (default scope: all docs matching `facet_filter`).
let resp = col.search_with_facets(
    &query, SearchOptions::new(10, 8), Some(&facet_filter), &facets, VectorFacetScope::FilterMatches,
)?;
println!("{} matches, {} hits", resp.total_hits, resp.hits.len());

// BM25 + facets: population is every document with BM25 score > 0 that matches `facet_filter`.
let resp = col.query_bm25_with_facets(&prefix_rank, Some(&facet_filter), 10, &facets)?;

// Hybrid RRF + facets over the BM25 matches that also pass `facet_filter`.
let resp = col.hybrid_search_with_facets(
    &query,
    &pre_tokenized,
    hybrid_opts,
    Some(&facet_filter),
    &facets,
    HybridFacetScope::Bm25AndFilterMatches,
)?;
```

## Bulk Import & Append (`izzydb-bulk`)

For loading Parquet datasets without per-row WAL/memtable/compaction overhead, `izzydb-bulk` builds sealed segments, columnar blocks, attribute/FTS indexes, and the optional global ANN tree offline as sorted external RocksDB SST files (`SstFileWriter`) and ingests them via `ingest_external_file_cf_opts`. It supports initial imports (`bulk_import`), populating new segments into existing collections (`bulk_append`), and whole-column detached Parquet refreshes (`izzydb-import columns`):

```rust
use std::path::PathBuf;
use izzydb::{CollectionConfig, DistanceMetric, TreeConfig};
use izzydb_bulk::{BulkImportConfig, DuplicatePolicy, bulk_import};

let col_cfg = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine));
let cfg = BulkImportConfig::new(col_cfg, "id", "vector", "/tmp/bulk_work")
    .with_on_duplicate_id(DuplicatePolicy::LastWins)
    .with_fill_factor(0.75);

let report = bulk_import("/tmp/my_collection", &[PathBuf::from("data.parquet")], cfg, None)?;
println!("Imported {} docs across {} segments ({} leaves)", report.docs_imported, report.segments_created, report.leaves);
```

You can also split SST generation and ingestion across machines with `build_ssts` and `ingest_ssts`, or use the `izzydb-import` CLI (`cargo run --release -p izzydb-bulk --bin izzydb-import -- import|append|columns --help`). See [Bulk Import Architecture](docs/architecture/bulk-import.md) for details on the pipeline and crash-safe marker protocol.

## Documentation

- **[User Guide](docs/guide/README.md)** — task-oriented walkthrough: getting started, schema, documents, vector/filtered/full-text/hybrid search, ranking & boosting, faceting, bulk import/append, operations & maintenance, and [Detached Columns](docs/guide/12-detached-columns.md).

### Architecture & Performance

- [Key & Binary Encoding](docs/architecture/key-encoding.md) — v3 RocksDB column families, `RowId` / segment keys, `RowView` v3, `ColumnBlock`, `SegPosting`, `FtsChunk` v4, and RaBitQ leaf layouts.
- [Segments, Compaction & Vacuum](docs/architecture/segments-and-compaction.md) — append-only segment states, CoW updates, rank-based `RowId` forwarding, `vacuum_ann`, and compaction filters.
- [Detached Columns](docs/architecture/detached-columns.md) — out-of-row generational columnar storage, lock-free `detached_ingest`, atomic `< 20 ms` publish, and `0x14` overwritten sets.
- [Index Path](docs/architecture/index-path.md) — `SegmentStore`, `AnnIndex`, `AttributeIndex`, and `FtsIndex` atomic write coordination.
- [Search Path](docs/architecture/search-path.md) — five-regime `FilterPlan` planner (`ExactScan`, `MaskedTraversal`, `PostFilter` with `PostingMembership` & `ColumnFilter`), beam traversal, and `RowResolver` output.
- [Filtering & Inverted Index](docs/architecture/filtering-and-inverted-index.md) — `Inline`/`Detached` schemas, value-major per-segment postings, `0x06`/`0x07` live counts, `0x12` zone maps, and faceting.
- [Full-Text Search Architecture](docs/architecture/fts-architecture.md) — unified `FtsChunk` v4 `(seg, chunk)` blocks, `TermRecord` v4, segment-parallel Block-Max MAXSCORE BM25, and hybrid RRF.
- [SIMD Block-Max MAXSCORE & Data Layout](docs/architecture/blockmax-simd.md) — 256-slot block geometry, `DocLenBlock` & `FtsChunk` v4, compact `TfEncoding` / v3 posting layouts, and runtime-dispatched SIMD kernels.
- [Bulk Import & Append Architecture](docs/architecture/bulk-import.md) — offline Parquet scan, skeleton training, `bulk_import`, `bulk_append`, and `izzydb-import` subcommands.
- [Segmented Storage v3 Performance Report](docs/perf/2026-11-23-segmented-storage-v3.md) — `master` vs. `segments-v3` release comparison across ANN, filtering, BM25, faceting, bulk import, churn, and detached ingest.
- [ANN Latency Optimization](docs/perf/2026-11-20-latency-optimization.md) — packed node blobs, internal node cache, single-FWHT query path, and SIMD RaBitQ scoring.
- [SIMD Block-Max MAXSCORE](docs/perf/2026-11-21-simd-maxscore.md) — out-of-line `DocLenBlock`, v2 TF-only Sparse/Dense posting blocks, `min_dl_over_tf` bounds, and runtime-dispatched FTS SIMD kernels.

## Breaking Changes (v3 Segmented Storage)

- **On-disk format incompatible with v2:** databases must be recreated or re-imported.
- **Optional embeddings & vectorless collections:** `Collection::upsert` and `Document.embedding` take/return `Option` embeddings (`Some(&vec)` / `Option<Vec<f32>>`); use `CollectionConfig::without_vectors()` and `BulkImportConfig::new_vectorless` when vectors are omitted.
- **Search tuning knob:** `SearchOptions::with_exhaustive_cluster_threshold` is replaced by `SearchOptions::with_exact_scan_threshold` (default `4_096` vector rows).
- **Bulk CLI subcommands:** `izzydb-import` now takes a subcommand: `izzydb-import import`, `izzydb-import append`, or `izzydb-import columns`.
- **`ClusteringTree` standalone removal:** collection APIs live on `Collection`; ANN indexing is coordinated internally as a secondary index over `RowId`s.

## Tests & Benchmarks

```bash
cargo test --workspace
cargo run --release -p izzydb-bench --bin recall_bench
cargo run --release -p izzydb-bench --bin filtered_bench
cargo run --release -p izzydb-bench --bin fts_bench
cargo run --release -p izzydb-bench --bin rank_bench
cargo run --release -p izzydb-bench --bin facet_bench
cargo run --release -p izzydb-bench --bin encoding_bench
cargo run --release -p izzydb-bench --bin bulk_import_bench
cargo run --release -p izzydb-bench --bin churn_bench
cargo run --release -p izzydb-bench --bin detached_bench
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
