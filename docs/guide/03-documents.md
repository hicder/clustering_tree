# Documents: Writing and Reading

A document is `(DocId, embedding, attributes)`:

- `DocId` is a `u64` that **you** choose. It is the primary key.
- `embedding` is `&[f32]` of exactly `dim` finite values (no `NaN`/`Inf`).
- `attributes` is `Attributes = BTreeMap<String, AttributeValue>`. Every key must be declared in the schema.

All write methods are on `Collection` and take `&mut self`. Each call is **atomic** across the vector index, attribute indexes, and full-text index; if it returns `Err`, nothing was written.

## Building attributes

```rust
use izzydb::{AttributeValue, Attributes};

let mut attrs = Attributes::new();
attrs.insert("category".into(), "shoes".into());              // String
attrs.insert("rank".into(), 5_i64.into());                    // Int
attrs.insert("views".into(), 12_u64.into());                  // Uint
attrs.insert("price".into(), 79.99_f64.into());               // Float
attrs.insert("in_stock".into(), true.into());                 // Bool
attrs.insert("tags".into(), vec!["a".to_string(), "b".to_string()].into()); // StringArray
attrs.insert("discount".into(), AttributeValue::Null);        // explicitly missing
```

The value's variant must match the declared type (`Int` attribute ← `i64`, not `u64` or `f64`). A mismatch returns `TreeError::InvalidAttribute { name, reason }`. Undeclared names are rejected the same way.

## `upsert` — insert or replace

```rust
col.upsert(42, &embedding, attrs)?;
```

If `42` exists, the document is **fully replaced**: any attribute you leave out of `attrs` becomes missing. Re-upserting an identical vector is cheap (IzzyDB detects that the vector is unchanged and skips re-indexing it).

Errors: `DimensionMismatch`, `NonFiniteVector`, `InvalidAttribute`.

## `get`

```rust
match col.get(42)? {
    Some(doc) => println!("{} {:?}", doc.embedding.len(), doc.attributes),
    None => println!("not found"),
}
```

`Document { id, embedding, attributes }`. For `Cosine` collections the returned embedding is the L2-normalized form IzzyDB indexes, not necessarily the exact floats you supplied.

## `patch` — partial update

Use `DocumentPatch` when you only want to change some fields. Omitted attributes are preserved.

```rust
use izzydb::DocumentPatch;

col.patch(
    42,
    DocumentPatch::new()
        .with_attribute("price", 59.99_f64.into())   // set
        .clear_attribute("discount"),                 // set to Null
)?;

// Replace the vector only:
col.patch(42, DocumentPatch::new().with_embedding(new_embedding))?;
```

- Returns `TreeError::DocumentNotFound(id)` if the document does not exist (unlike `upsert`, `patch` never creates).
- Only the indexes touched by the change are updated, so patching one attribute is cheaper than a full `upsert`.

### `patch_batch`

```rust
col.patch_batch(&[
    (1, DocumentPatch::new().with_attribute("in_stock", false.into())),
    (2, DocumentPatch::new().with_attribute("in_stock", false.into())),
])?;
```

All patches are validated first and applied in **one** transaction. If any id is missing, `DocumentNotFound` is returned and **nothing** is written.

### `update_by_filter`

Apply the same patch to every document matching a [`Filter`](05-filtering.md):

```rust
use izzydb::Filter;

let n = col.update_by_filter(
    &Filter::Eq("category".into(), "clearance".into()),
    &DocumentPatch::new().with_attribute("in_stock", false.into()),
)?;
println!("updated {n} documents");
```

The matching ids are resolved first, then updated in bounded atomic chunks. The whole operation is therefore **not** one transaction: a crash midway leaves some chunks applied. Each chunk is individually atomic.

## Deleting

```rust
let existed: bool = col.delete(42)?;                 // false if it was absent
let n: u64 = col.delete_batch(&[1, 2, 3])?;          // dedups; ignores missing ids
let n: u64 = col.delete_by_filter(&Filter::Lt("price".into(), 1.0_f64.into()))?;
```

`delete_batch` is atomic. `delete_by_filter` is chunked like `update_by_filter`.

Deletion fully removes the document from the vector, attribute, and full-text indexes. The only residue is described next.

### `compact_fts_stats`

BM25 uses per-term upper bounds to skip work. After many deletes/updates those bounds can become looser than necessary (still correct, just less selective). If you have a full-text collection with heavy churn, occasionally run:

```rust
col.compact_fts_stats()?;
```

It recomputes exact bounds. It is a no-op without full-text attributes and is never needed for correctness.

## Finding ids without searching

```rust
let ids: Vec<u64> = col.resolve_filter_doc_ids(&filter)?;
```

Returns every `DocId` matching a filter (attribute and/or full-text predicates), with no vector involved.

## Batching advice

- Each `upsert` is its own transaction. For **initial** loads of large datasets use [Bulk Import](09-bulk-import.md), which is dramatically faster.
- For ongoing changes of many documents, prefer `patch_batch` / `delete_batch` over loops of single calls.
- There is no explicit `flush`/`commit`: a returned write has been applied to RocksDB and its write-ahead log (see [Durability](10-operations.md#durability) for what that does and does not guarantee).
