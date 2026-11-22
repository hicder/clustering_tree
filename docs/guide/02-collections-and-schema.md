# Collections & Schema

A `Collection` is configured by a [`CollectionConfig`], which has two parts:

- a [`TreeConfig`] describing the **vector index** (dimension, metric, node sizes), and
- a list of **attributes** (the typed schema) describing the non-vector fields of every document.

## `TreeConfig`

```rust
use izzydb::{DistanceMetric, TreeConfig};

let tree = TreeConfig::new(768, DistanceMetric::Cosine);
```

`TreeConfig::new(dim, metric)` fills in sensible defaults. `dim` and `metric` are **fixed for the life of the collection**: re-opening with a different value fails with `TreeError::ConfigMismatch`.

### Distance metrics

| `DistanceMetric` | Distance (smaller = closer) | Use when |
|---|---|---|
| `Cosine` | `1 - dot(a, b)` on L2-normalized vectors | Text/image embeddings where only direction matters. IzzyDB normalizes vectors for you on write and on query. |
| `L2Squared` | `Σ (aᵢ - bᵢ)²` | Embeddings where magnitude matters (e.g. raw feature vectors). |
| `DotProduct` | `-dot(a, b)` | Maximum-inner-product search (e.g. recommender embeddings). |

Search results always report a `distance` where **smaller is better**, whatever the metric.

### Tuning fields (rarely needed)

| Field | Default | Meaning |
|---|---|---|
| `min_leaf_size` / `max_leaf_size` | 32 / 160 | Vectors per leaf cluster. A leaf splits above `max` and merges below `min`. |
| `min_internal_fanout` / `max_internal_fanout` | 32 / 160 | Children per internal node (~100× fanout per level). |
| `insert_beam_width` | 4 | Beam width used to route a new vector to its leaf. Wider = better placement, slower writes. |
| `enable_lire` | `true` | Reassign boundary vectors to better neighbor clusters after a split (SPFresh LIRE). Improves recall under churn. |
| `lire_neighbor_count` | 8 | Neighbor leaves examined by LIRE. |
| `centroid_drift_eps` | `1e-3` | Skip propagating centroid updates upward when drift is below this. |

These are public fields; set them directly:

```rust
let mut tree = TreeConfig::new(768, DistanceMetric::Cosine);
tree.insert_beam_width = 8;
```

`Collection::open` validates them. Constraints: `dim > 0`, `min_leaf_size * 2 < max_leaf_size`, `min_internal_fanout * 2 < max_internal_fanout` (this prevents split/merge oscillation), and `insert_beam_width >= 1`. Violations return `TreeError::InvalidConfig`.

> The defaults are tuned for general use. Change leaf/fanout sizes only if you have measured a reason to.

## Attribute types

| `AttributeType` | Rust value (via `.into()`) | Filter ops | Range ops (`Lt`…`Gte`) | Facets |
|---|---|---|---|---|
| `String` | `&str`, `String` | ✔ | ✔ (bytewise order) | Terms, Ranges |
| `Int` | `i64`, `i32` | ✔ | ✔ | Terms, Ranges, Stats, Histogram |
| `Uint` | `u64`, `u32` | ✔ | ✔ | Terms, Ranges, Stats, Histogram |
| `Float` | `f64` | ✔ | ✔ | Terms, Ranges, Stats, Histogram |
| `Bool` | `bool` | ✔ | ✘ | Terms |
| `StringArray` | `Vec<String>` | `Contains*` | ✘ | Terms, Ranges |
| `IntArray` | `Vec<i64>` | `Contains*` | ✘ | Terms, Ranges |
| `UintArray` | `Vec<u64>` | `Contains*` | ✘ | Terms, Ranges |

Notes:

- `Float` rejects `NaN`; `-0.0` and `+0.0` are treated as equal.
- Array attributes use `Contains` / `ContainsAny` instead of `Eq` / `In`.
- Any attribute may be omitted from a document or set to `AttributeValue::Null`; both mean "missing" (see [Filtering](05-filtering.md#nulls-and-negation)).

## Declaring attributes

Three builder methods on `CollectionConfig`:

```rust
use izzydb::{AttributeConfig, AttributeType, CollectionConfig, FullTextSearchConfig, TreeConfig, DistanceMetric};

let config = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine))
    // (name, type, filterable)
    .with_attribute("category", AttributeType::String, true)
    // (name, type, filterable, full-text config)
    .with_fts_attribute("title", AttributeType::String, false, FullTextSearchConfig::default())
    // Equivalent explicit form
    .with_attribute_config(
        AttributeConfig::new("body", AttributeType::String)
            .filterable(false)
            .full_text_search(FullTextSearchConfig::default()),
    );
```

Each attribute has two independent switches:

| Switch | Enables | Cost |
|---|---|---|
| `filterable = true` | `Filter::Eq/In/Lt/…/Contains/IsNull` predicates **and** facets on this attribute. | An inverted index entry per value plus per-value counters. |
| full-text config (`FullTextSearchConfig`) | `RankBy::bm25`, and `ContainsAllTokens` / `ContainsAnyToken` / `ContainsTokenSequence` filters. Only valid on `String` / `StringArray`. | A BM25 inverted index (postings, optional positions). |

An attribute with **neither** switch is just stored: you get it back from `Collection::get`, but you cannot filter, facet, or rank by it. That is fine for payload such as a URL or a JSON blob.

> **Index only what you query.** Every filterable/full-text attribute adds write cost and disk space. A `String` description that you only ever BM25-search should be `filterable = false`.

Constraints checked at open: attribute names must be non-empty and unique.

## Opening, re-opening, and schema evolution

```rust
let mut col = Collection::open("./my_collection", config)?;
```

- **Path does not exist:** it is created and the schema is stored in the database.
- **Path exists:** the stored schema is compared with your `CollectionConfig`:
  - every attribute you declare that already exists must have the **same type, same `filterable` flag, and same full-text config**, otherwise `Collection::open` fails with `TreeError::ConfigMismatch { field, expected, actual }`;
  - **new attributes are added.** Old documents simply read as missing for them. If a new attribute is full-text indexed, existing documents are back-filled into the FTS index during `open`.
- Attributes that are in the database but **not** in your config are kept as-is. Still, pass your complete schema every time so the config in your code stays the source of truth.
- Vector `dim` and `metric` must match the stored values.

Things you **cannot** do in place: change an attribute's type, toggle `filterable`, change its full-text settings, or change `dim`/`metric`. To make such a change, create a new collection and re-ingest (see [Bulk Import](09-bulk-import.md)).

## Inspecting a collection

```rust
col.len();            // number of documents (u64)
col.is_empty();
col.config();         // &TreeConfig
col.schema();         // &CollectionSchema: iterate col.schema().entries()
```

[`CollectionConfig`]: ../../crates/izzydb-core/src/schema.rs
[`TreeConfig`]: ../../crates/izzydb-core/src/config.rs
