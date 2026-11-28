# Filtering & Inverted Index Architecture

This document specifies IzzyDB v3's attribute schema (`Inline` and `Detached`), filter AST, order-preserving encodings, **value-major per-segment** inverted index (`AttributeIndex`), exact live counts (`0x06`/`0x07`), per-segment zone maps (`0x12`), and faceting engine.

Implementation sources:
- Schema & `ColumnStorage::{Inline, Detached}`: [`crates/izzydb-core/src/schema/mod.rs`](../../crates/izzydb-core/src/schema/mod.rs)
- Filter AST & validation: [`crates/izzydb-core/src/filter.rs`](../../crates/izzydb-core/src/filter.rs)
- `RowView` v3, `SegPosting`, `ZoneEntry`, and scalar order-preserving codecs: [`crates/izzydb-codec/src/lib.rs`](../../crates/izzydb-codec/src/lib.rs)
- `ColumnBlock` readers (`AttrReader`, `ColumnReader`, `RowStoreReader`): [`crates/izzydb/src/attr_reader.rs`](../../crates/izzydb/src/attr_reader.rs)
- `AttributeIndex` write hook: [`crates/izzydb/src/attr_index.rs`](../../crates/izzydb/src/attr_index.rs)
- Filter planner (`RowSet`): [`crates/izzydb/src/filter_plan/mod.rs`](../../crates/izzydb/src/filter_plan/mod.rs)
- Faceting engine: [`crates/izzydb/src/facet/mod.rs`](../../crates/izzydb/src/facet/mod.rs)

---

## 1. Overview: Segment-Scoped & Value-Major (v3)

In v2, `AttributeIndex` indexed documents by ANN `(cluster_id, local_id)`, causing index churn whenever vectors split, merged, or moved under LIRE. In v3, `AttributeIndex` is decoupled from the ANN tree and keyed by **`RowId = (SegmentId, LocalId)`**:

- **No cross-index churn on ANN splits/LIRE:** Moving a vector between ANN leaves changes zero attribute index keys.
- **Value-major postings (`cf_attr_postings` & `cf_detached_postings`):** Storing `encoded_val` *before* `seg` groups all segments holding a given `(attr_id, type_tag, encoded_val)` together so `OrderBy`, `Terms` facets, and range scans stream values in sorted order without scattering across hundreds of ANN leaves.
- **Append-only sealed postings + RAM liveness:** Once a segment seals (`Open -> Sealed`), `cf_attr_postings` and `0x12` zone entries are immutable; deletions decrement `default[0x06]` live counts and mark `0x10` tombstones. Compaction reclaims dead postings cleanly via `dropped_segments`.

---

## 2. Schema, Storage Modes & Validation

Each `AttributeConfig` declares:
- `name: String` (unique, non-empty)
- `attr_type: AttributeType` (`String`, `Int`, `Uint`, `Float`, `Bool`, `StringArray`, `IntArray`, `UintArray`)
- `filterable: bool` (enables inverted index keys in `cf_attr_postings`/`cf_detached_postings`, live counts `0x06`/`0x07`, zone maps `0x12`, `Filter` predicates, and facets)
- `storage: ColumnStorage` (`Inline` stored in `cf_rows` + sealed `cf_columns`; `Detached` stored exclusively in `cf_detached` + `cf_detached_postings`, see [detached-columns.md](detached-columns.md))
- `column: bool` (optional `.with_column(true)` / `.with_column_attribute(...)` to opt non-filterable `Inline` string/array attributes into sealed `cf_columns`; `filterable` attributes and scalar `Bool`/`Int`/`Uint`/`Float` are automatically columnar on seal via `SchemaEntry::is_columnar()`)
- Optional `FullTextSearchConfig` (`Inline` string/string-array attributes only)

Validation rules (`Filter::validate` & `CollectionSchema`):
- `NaN` floats are rejected everywhere; `-0.0` normalizes to `+0.0`.
- Missing attributes and explicit `AttributeValue::Null` are equivalent.
- Negations (`NotEq`, `NotIn`, `Not`) include missing/`Null` documents (`NotEq("cat", "books")` matches docs with no `cat`). Ranges (`Lt`/`Lte`/`Gt`/`Gte`) exclude `Null`.
- `Contains`/`ContainsAny` apply to array attributes (`StringArray`, `IntArray`, `UintArray`); empty arrays still emit `TAG_PRESENT` (`0x00`) so `IsNotNull` matches them.

---

## 3. Columnar Forward Blocks (`cf_columns` & `cf_detached`)

While `cf_rows` stores per-row `RowView` v3 directories (fast for multi-attribute point `get` and `RowView::matches_filter`), reading one attribute across many rows uses 256-slot columnar `ColumnBlock`s:
- **`cf_columns` (`seg BE || attr_id BE || block BE`, 10 B):** built for `Inline` attributes when a segment is sealed (or deferred to `maintenance()`).
- **`cf_detached` (`attr_id BE || gen BE || seg BE || block BE`, 14 B):** always columnar across all segments.
- `AttrReader` abstracts over `cf_columns` (sealed inline), `cf_detached` (detached), and `cf_rows` (open-segment inline or deferred-column segments), accelerating `OrderBy`, `RankBy` modifier rescoring, `"docs"` facet passes, and `ColumnFilter` residual verification during ANN `PostFilter` search.

---

## 4. Inverted Keyspaces & Binary Encodings

### 4.1 Column Families & Keys

- **`cf_attr_postings`:** `attr_id (u16 BE) || type_tag (u8) || encoded_val || seg (u32 BE)` $\to$ `SegPosting` (`0x00` Sparse `[u32; len]` LE for `<= 32` locals; `0x01` `RoaringBitmap`). Prefix extractor: 3 B (`attr_id || type_tag`). Uncompressed (`None`). `SegPostingView` adds zero-allocation `contains(local)` (binary search over sparse/array values, direct container bit tests) and ORs raw serialized containers into a flat `SegBitsetAccum` for `In` / range / `Or` unions.
- **`cf_detached_postings`:** `attr_id (u16 BE) || gen (u32 BE) || type_tag (u8) || encoded_val || seg (u32 BE)` $\to$ `SegPosting`. Prefix extractor: 7 B (`attr_id || gen || type_tag`). Uncompressed.
- **Sealed posting cache:** `Storage::attr_posting_cache` is a bounded 64 MiB FIFO of decoded sealed-segment bitmaps keyed by the full `(attr_key, seg)` posting key. Entries are immutable and dropped only when the segment retires; open-segment postings bypass the cache.
- **`default[0x06]` / `default[0x07]`:** Global live-doc counts (`u64` LE; deleted at `0`) keyed identically to `(attr_id, [gen], type_tag, encoded_val)`. Gives $O(1)$ cardinality estimation for `FilterPlan`, `"global"`/`"complement"` facets, and exact `TotalHits` without scanning postings.
- **`default[0x12]` (`ZoneEntry`):** `0x12 || seg (u32 BE) || attr_id (u16 BE) || type_tag (u8)` $\to$ `(min_bytes, max_bytes)` for ordered scalar attributes. Cached in `Storage::zone_cache` to prune segments with zero posting I/O on range predicates.
- **`default[0x14]` (detached overwritten set):** `0x14 || attr_id BE || gen BE || seg BE` $\to$ `RoaringBitmap<LocalId>` marking rows mutated via `set_detached` after bulk generation build so scans over immutable `cf_detached_postings` stay 100% exact.

### 4.2 Order-Preserving Inverted Key Encoding

| `type_tag` | Constant | `encoded_val` |
|---:|---|---|
| `0x00` | `TAG_PRESENT` | Empty (0 B) — marks non-null presence |
| `0x01` | `TAG_BOOL` | `0x00` (`false`) / `0x01` (`true`) |
| `0x02` | `TAG_INT` | `((v as u64) ^ (1u64 << 63)).to_be_bytes()` (`i64::MIN -> 0x0000...`, `MAX -> 0xFFFF...`) |
| `0x03` | `TAG_UINT` | `v.to_be_bytes()` |
| `0x04` | `TAG_FLOAT` | `-0.0 -> +0.0`; negative bits inverted (`!bits`), non-negative sign bit flipped |
| `0x05` | `TAG_STRING` | Escaped-null UTF-8 (`0x00 -> 0x00 0xFF`), terminated with `[0x00, 0x00]` |

Every `encoded_val` is **self-delimiting**, so parsers can unambiguously split the trailing 4-byte `seg: u32 BE` suffix both in range scans and inside the RocksDB compaction filter.

---

## 5. Faceting Engine (`crates/izzydb/src/facet/mod.rs`)

Every `filterable = true` attribute (both `Inline` and `Detached`) supports exact `Terms`, `Ranges`, `Stats`, and `Histogram` facets plus disjunctive `FacetFilterMode` (`ExcludeSelf`, etc.). Strategy dispatch per `FacetGroup`:

| Strategy | Trigger | Mechanism |
|---|---|---|
| `global` | Unfiltered, non-array-range group | Ordered scan over `default[0x06]` (inline) or `default[0x07]` (detached) with prefix/range bounds; 0 segment/row reads |
| `complement` | Top-level `Not` / `NotEq` / `NotIn` whose positive inner filter resolves to `<= 2,048` live rows | Subtracts forward `AttrReader` pass counts from `global` |
| `docs` | Explicit doc sets (`TopK`, `Candidates`, `FusedCandidates`) or materialized `<= 2,048` rows | Columnar/row forward pass via `AttrReader` in 512-row batches |
| `seg_masks` | Broad filters (`> 2,048` rows) | Evaluates segment-level `SegPosting` bitmaps once against `SegmentTableSnapshot` liveness (plus `0x14` correction for detached attrs) and counts via bitwise intersection |
