# Getting Started

This page takes you from an empty `Cargo.toml` to a running filtered vector search plus keyword search.

## 1. Add the dependency

IzzyDB is a Cargo workspace; the crate you use as an application developer is [`izzydb`](../../crates/izzydb). Depend on it by path (or by git URL of wherever you host the repository):

```toml
[dependencies]
izzydb = { path = "../izzydb/crates/izzydb" }
```

Requirements:

- Rust **2024 edition** (a recent stable toolchain).
- A C++ toolchain and `clang`/`libclang`, because the `rocksdb` crate compiles RocksDB from source on the first build (expect a few minutes).
- Optional feature `fts-cjk` enables Chinese/Japanese/Korean tokenizers (see [Full-Text Search](06-full-text-search.md#tokenizers)).

## 2. Create a collection

```rust
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric,
    FullTextSearchConfig, Language, TreeConfig,
};

const DIM: usize = 4; // use your embedding model's dimension, e.g. 384, 768, 1536

let config = CollectionConfig::new(TreeConfig::new(DIM, DistanceMetric::Cosine))
    // Indexed for filters and facets.
    .with_attribute("category", AttributeType::String, true)
    .with_attribute("price", AttributeType::Float, true)
    // Indexed for BM25 keyword search (not used in equality filters, hence `false`).
    .with_fts_attribute(
        "title",
        AttributeType::String,
        false,
        FullTextSearchConfig::default()
            .with_language(Language::English)
            .with_stemming(true),
    );

let mut col = Collection::open("./my_collection", config)?;
```

`Collection::open` **creates** the directory if it does not exist and **re-opens** it (validating the config against what is stored) if it does. See [Collections & Schema](02-collections-and-schema.md).

## 3. Insert documents

```rust
let mut attrs = Attributes::new();
attrs.insert("category".into(), "shoes".into());
attrs.insert("price".into(), 79.99_f64.into());
attrs.insert("title".into(), "Trail Runner Pro".into());

col.upsert(1, Some(&[0.1, 0.2, 0.3, 0.4]), attrs)?;
```

`upsert(id, embedding, attributes)` inserts a new document or replaces the document with the same `id` (`Some(&vec)` for vectored collections, or `None` for vectorless collections). Attribute values convert with `.into()` from `&str`, `String`, `i32`/`i64`, `u32`/`u64`, `f64`, `bool`, `Vec<String>`, `Vec<i64>`, and `Vec<u64>`.

## 4. Query

### Vector search

```rust
use izzydb::SearchOptions;

let query = [0.1_f32, 0.2, 0.3, 0.4];
let hits = col.search(&query, SearchOptions::new(10, 8), None)?;
for h in hits {
    println!("doc {} distance {:.4}", h.id, h.distance); // smaller = closer
}
```

### Filtered vector search

```rust
use izzydb::Filter;

let filter = Filter::And(vec![
    Filter::Eq("category".into(), "shoes".into()),
    Filter::Lt("price".into(), 100.0_f64.into()),
]);
let hits = col.search(&query, SearchOptions::new(10, 8), Some(&filter))?;
```

### Keyword search (BM25)

```rust
use izzydb::RankBy;

let hits = col.query_bm25(&RankBy::bm25("title", "trail runner"), None, 10)?;
for h in hits {
    println!("doc {} score {:.3}", h.id, h.score); // larger = better
}
```

### Read a document back

```rust
if let Some(doc) = col.get(1)? {
    println!("{:?} {:?}", doc.embedding, doc.attributes);
}
```

## 5. Everything returns `Result`

All fallible calls return `izzydb::Result<T>` with a [`TreeError`](11-operations.md#errors). The `?` operator in the examples above assumes your function returns `Result<_, izzydb::TreeError>` (or something `TreeError` converts into, e.g. `Box<dyn std::error::Error>`, since `TreeError` implements `std::error::Error`).

A complete minimal program:

```rust
use izzydb::{
    AttributeType, Attributes, Collection, CollectionConfig, DistanceMetric, Filter, RankBy,
    SearchOptions, TreeConfig, FullTextSearchConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::Cosine))
        .with_attribute("category", AttributeType::String, true)
        .with_fts_attribute("title", AttributeType::String, false, FullTextSearchConfig::default());
    let mut col = Collection::open("./demo_collection", config)?;

    let docs = [
        (1, [1.0, 0.0, 0.0, 0.0], "shoes", "trail running shoe"),
        (2, [0.9, 0.1, 0.0, 0.0], "shoes", "road running shoe"),
        (3, [0.0, 1.0, 0.0, 0.0], "hats",  "wool winter hat"),
    ];
    for (id, vec, cat, title) in docs {
        let mut a = Attributes::new();
        a.insert("category".into(), cat.into());
        a.insert("title".into(), title.into());
        col.upsert(id, Some(&vec), a)?;
    }

    let only_shoes = Filter::Eq("category".into(), "shoes".into());
    let near = col.search(&[1.0, 0.0, 0.0, 0.0], SearchOptions::new(2, 4), Some(&only_shoes))?;
    println!("{near:?}");

    let kw = col.query_bm25(&RankBy::bm25("title", "winter hat"), None, 5)?;
    println!("{kw:?}");
    Ok(())
}
```

## Next steps

- Model your data: [Collections & Schema](02-collections-and-schema.md)
- Keep it fresh: [Documents](03-documents.md)
- Loading a big dataset? Skip ahead to [Bulk Import](10-bulk-import.md).
