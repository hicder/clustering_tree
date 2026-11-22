# Full-Text Search (BM25)

IzzyDB includes a native BM25 inverted index. Enable it per attribute with a `FullTextSearchConfig`, then query with `RankBy` expressions.

```rust
use izzydb::{AttributeType, CollectionConfig, DistanceMetric, FullTextSearchConfig, Language, RankBy, TreeConfig};

let fts = FullTextSearchConfig::default()
    .with_language(Language::English)
    .with_stemming(true)
    .with_remove_stopwords(true)
    .with_ascii_folding(true);

let config = CollectionConfig::new(TreeConfig::new(128, DistanceMetric::Cosine))
    .with_fts_attribute("title", AttributeType::String, false, fts.clone())
    .with_fts_attribute("description", AttributeType::String, false, fts);

// ... open, upsert ...
let hits = col.query_bm25(&RankBy::bm25("description", "waterproof trail shoe"), None, 10)?;
// Vec<Bm25Result { id: u64, score: f32 }>, highest score first
```

`query_bm25(rank_by, filter, top_k)` (alias: `query`) returns the top `top_k` documents by score. Only documents that match at least one query term are returned. The optional `filter` restricts the candidate set ([Filtering](05-filtering.md)).

Which attributes can be full-text? `String` and `StringArray` only. For a `StringArray`, all elements are indexed as one field, with a position gap between elements so phrases never span two elements.

## Analyzer configuration

The analyzer runs identically at index time and query time, so a query like `"Running Shoes"` matches a document containing `run shoe` when stemming is on.

| Builder | Default | Meaning |
|---|---|---|
| `.with_tokenizer(t)` | `Tokenizer::WordV4` | How text is split into tokens (below). |
| `.with_language(l)` | `English` | Language for stemming and stopwords. 18 languages are supported. |
| `.with_case_sensitive(b)` | `false` | Keep case. Incompatible with stemming/stopwords. |
| `.with_stemming(b)` | `false` | Snowball stemming (`running → run`). |
| `.with_remove_stopwords(b)` | `false` | Drop common words (`the`, `and`, …). Not available for Arabic, Greek, Romanian, Tamil, Turkish. |
| `.with_ascii_folding(b)` | `false` | Fold accents (`café → cafe`). |
| `.with_max_token_length(n)` | 39 | Longer tokens are ignored. Range `1..=254`. |
| `.with_index_positions(b)` | `true` | Store token positions, required for phrase matching (`ContainsTokenSequence`). Turning it off saves space. |
| `.with_k1(x)` / `.with_b(x)` / `.with_k3(x)` | 1.2 / 0.75 / 8.0 | BM25 parameters (below). |

### Tokenizers

| `Tokenizer` | Behavior | Notes |
|---|---|---|
| `WordV4` (default) | Unicode word segmentation (UAX #29) | Good general-purpose choice. |
| `Whitespace` | Split on Unicode whitespace only | No stemming/stopwords allowed. Useful for IDs, SKUs, and codes. |
| `PreTokenizedArray` | Each element of a `StringArray` **is** a token, verbatim | Only on `StringArray`; no stemming, stopwords, folding, or case-sensitivity flags. Queries must use `TokenQuery::PreTokenized`. |
| `SmartCn`, `Kuromoji`, `Nori` | Chinese, Japanese, Korean | Requires the `fts-cjk` Cargo feature; the normalization flags above are not allowed. |

Validation rules are enforced when the collection opens. For example `stemming` with `case_sensitive`, or `PreTokenizedArray` on a `String` attribute, return `TreeError::InvalidSchema`.

### BM25 parameters

BM25 scores a document higher when it contains rare query terms many times, relative to document length.

- **`k1`** (> 0, default 1.2): term-frequency saturation. Higher = repeated terms keep adding score for longer.
- **`b`** (0..=1, default 0.75): length normalization. `1` fully penalizes long documents, `0` ignores length.
- **`k3`** (≥ 0, default 8.0): how much repeated terms *in the query* count.

Defaults are a sound starting point. They are stored in the schema (and therefore fixed), but you can override them per query ([below](#per-query-overrides)).

## Queries with `RankBy`

### Single field

```rust
RankBy::bm25("title", "trail runner")
```

Multiple query terms are combined as a **disjunction**: documents matching any term score, and documents matching more (and rarer) terms rank higher. If you need *all* terms present, add a filter:

```rust
let must_have_all = Filter::ContainsAllTokens {
    attr: "title".into(), query: "trail runner".into(), last_as_prefix: false,
};
col.query_bm25(&RankBy::bm25("title", "trail runner"), Some(&must_have_all), 10)?;
```

### Multiple fields

Combine per-field scores with `sum`, `max`, and weighted `product`:

```rust
// Title matches count double; scores add up across fields.
let rank_by = RankBy::sum(vec![
    RankBy::product(2.0, RankBy::bm25("title", "trail runner")),
    RankBy::bm25("description", "waterproof trail shoe"),
]);

// Best single field wins (no accumulation).
let best_field = RankBy::max(vec![
    RankBy::product(2.5, RankBy::bm25("title", "trail shoe")),
    RankBy::bm25("description", "trail shoe"),
]);
```

| Combinator | Score |
|---|---|
| `RankBy::sum(vec)` | sum of children |
| `RankBy::max(vec)` | maximum of children |
| `RankBy::product(w, expr)` | `w × score(expr)`, with `w ≥ 0` and finite |
| `RankBy::product_all(vec)` | product of all children (see [Ranking & Boosting](08-ranking-and-boosting.md)) |

Expressions nest. Each leaf may use a different attribute and a different query string.

### Per-query overrides

```rust
use izzydb::Bm25Overrides;

let rank = RankBy::bm25_with(
    "title",
    "trail run",
    Bm25Overrides::default()
        .with_last_as_prefix(true)   // "run" also matches run*, runner, running…
        .with_k1(1.5)
        .with_b(0.5)
        .with_max_prefix_expansions(64),
);
```

### Prefix / autocomplete search

`with_last_as_prefix(true)` treats the **last** query token as a prefix. The prefix expands to the indexed terms that start with it, up to `max_prefix_expansions` (default 128). This is the mechanism for type-ahead: query with whatever the user has typed so far.

The prefix token goes through the attribute's analyzer (lowercasing, folding, and stemming if enabled) before being matched against indexed terms. With stemming on, a half-typed word can therefore be stemmed into something that is not a prefix of the full word's indexed form. For type-ahead fields, consider a dedicated attribute with stemming disabled, and test with real partial input.

### Pre-tokenized attributes

```rust
use izzydb::TokenQuery;

RankBy::bm25("keywords", TokenQuery::PreTokenized(vec!["gore-tex".into(), "vibram".into()]))
```

## Combining with filters

```rust
let filter = Filter::And(vec![
    Filter::Eq("in_stock".into(), true.into()),
    Filter::Lte("price".into(), 100.0_f64.into()),
]);
let hits = col.query_bm25(&rank_by, Some(&filter), 10)?;
```

The filter can include token predicates over other full-text attributes too.

## Index maintenance

- Inserting, updating, and deleting documents keep the index current automatically. An update that doesn't change any full-text attribute skips re-indexing.
- BM25 corpus statistics (document counts and average length) are maintained incrementally.
- After a lot of deletes/updates, run `col.compact_fts_stats()` to tighten internal score bounds (see [Documents](03-documents.md#compact_fts_stats)). It affects speed only, never results.

## Errors

| Error | Cause |
|---|---|
| `InvalidQuery` | `RankBy` references an attribute without full-text enabled, a `PreTokenizedArray` attribute is queried with text, or an override/weight is out of range. |
| `InvalidFilter` | Same problems in a token *filter*, or `ContainsTokenSequence` on an attribute with `index_positions = false`. |

For fusing keyword and vector rankings, continue to [Hybrid Search](07-hybrid-search.md).
