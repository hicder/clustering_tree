# Filtering

A [`Filter`] is a boolean expression tree over attributes. Pass it to `search`, `query_bm25`, `hybrid_search`, `facet`, and the `*_by_filter` mutations.

```rust
use izzydb::Filter;

let f = Filter::And(vec![
    Filter::Eq("category".into(), "shoes".into()),
    Filter::Gte("price".into(), 10.0_f64.into()),
    Filter::Lt("price".into(), 50.0_f64.into()),
]);
```

Values inside predicates are `ScalarValue`s; use `.into()` from `&str`, `String`, `i32`/`i64`, `u32`/`u64`, `f64`, `bool`. **The value's type must match the attribute's type**: comparing an `Int` attribute to `3.0_f64` is an `InvalidFilter` error, not an implicit conversion.

## Requirements

- Attributes used in scalar/array predicates must be declared `filterable = true`.
- Attributes used in token predicates (`ContainsAllTokens`, …) must have a full-text config.
- Filters are validated against the schema before they run; unknown attributes, type mismatches, and unsupported combinations return `TreeError::InvalidFilter`.

## Predicate reference

### Scalar attributes (`String`, `Int`, `Uint`, `Float`, `Bool`)

| Predicate | Meaning |
|---|---|
| `Eq(attr, v)` / `NotEq(attr, v)` | equal / not equal |
| `In(attr, [v…])` / `NotIn(attr, [v…])` | member / not a member of the set |
| `Lt` / `Lte` / `Gt` / `Gte` `(attr, v)` | ordered comparison — **not** available on `Bool` |
| `IsNull(attr)` / `IsNotNull(attr)` | attribute missing / present |

`Eq`, `NotEq`, `In`, `NotIn` on an **array** attribute are rejected — use the array predicates below.

String comparison is bytewise (UTF-8), so `"Z" < "a"`.

### Array attributes (`StringArray`, `IntArray`, `UintArray`)

| Predicate | Meaning |
|---|---|
| `Contains(attr, v)` | array contains `v` |
| `ContainsAny(attr, [v…])` | array contains at least one of the values |
| `IsNull` / `IsNotNull` | as above |

To require *all* of several values, combine with `And`:

```rust
Filter::And(vec![
    Filter::Contains("tags".into(), "sale".into()),
    Filter::Contains("tags".into(), "new".into()),
])
```

### Full-text token predicates

These analyze the query text with the attribute's configured analyzer (same tokenizer, stemming, stopwords, folding as at index time), then test the document:

| Predicate | Matches documents where the attribute… |
|---|---|
| `ContainsAllTokens { attr, query, last_as_prefix }` | contains **every** query token (any order) |
| `ContainsAnyToken { attr, query, last_as_prefix }` | contains **at least one** query token |
| `ContainsTokenSequence { attr, query, last_as_prefix }` | contains the tokens **consecutively as a phrase** (requires `index_positions = true`, the default) |

```rust
use izzydb::{Filter, TokenQuery};

let phrase = Filter::ContainsTokenSequence {
    attr: "description".into(),
    query: "trail running".into(),   // &str / String -> TokenQuery::Text
    last_as_prefix: false,
};

// Autocomplete-style: last token is treated as a prefix ("run" matches "running", "runner", …)
let typeahead = Filter::ContainsAllTokens {
    attr: "title".into(),
    query: "trail run".into(),
    last_as_prefix: true,
};

// PreTokenizedArray attributes take pre-split tokens verbatim:
let kw = Filter::ContainsAnyToken {
    attr: "keywords".into(),
    query: TokenQuery::PreTokenized(vec!["gore-tex".into(), "vibram".into()]),
    last_as_prefix: false,
};
```

An attribute using `Tokenizer::PreTokenizedArray` *requires* `TokenQuery::PreTokenized`; passing text is an error.

### Boolean logic

`And(vec![…])`, `Or(vec![…])`, `Not(Box::new(f))`. They nest arbitrarily and may freely mix scalar, array, and token predicates.

```rust
let f = Filter::Or(vec![
    Filter::And(vec![
        Filter::Eq("category".into(), "shoes".into()),
        Filter::Not(Box::new(Filter::Eq("in_stock".into(), false.into()))),
    ]),
    Filter::Contains("tags".into(), "featured".into()),
]);
```

## Nulls and negation

These rules matter; they are where filters most often surprise people.

- A **missing** attribute and an attribute explicitly set to `AttributeValue::Null` are the same thing.
- `IsNull` matches missing; `IsNotNull` matches present — including an **empty array**, which is "present".
- **Negations include missing.** `NotEq("category", "books")` means *"not (category == books)"*, so documents with no `category` **match**. The same holds for `NotIn` and `Not(Eq(…))`. If you want "has a category and it isn't books", write:

  ```rust
  Filter::And(vec![
      Filter::IsNotNull("category".into()),
      Filter::NotEq("category".into(), "books".into()),
  ])
  ```
- **Range predicates exclude missing.** `Lt/Lte/Gt/Gte` only match documents that have a value.
- `Float` values: `NaN` is not allowed anywhere; `-0.0 == 0.0`.
- `Eq(attr, ScalarValue::Null)` is allowed and is equivalent to `IsNull(attr)`; `Null` is not allowed in range predicates, `Contains`, or `ContainsAny`.

## Checking a filter without running a search

```rust
let ids = col.resolve_filter_doc_ids(&f)?;   // also validates the filter
```

## Performance tips

- Filters on `filterable` attributes are answered from inverted indexes organized **by cluster**, so vector search can skip clusters that contain no matches, and range predicates are pruned using per-cluster min/max values.
- Mixed filters are fine: `And`/`Or`/`Not` may combine attribute and token predicates freely.
- If you filter on the same text pattern constantly, consider a derived `filterable` attribute (a `category` or tag array) instead of a token predicate.

[`Filter`]: ../../crates/izzydb-core/src/filter.rs
