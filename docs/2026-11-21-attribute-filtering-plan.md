# Plan: Attribute Filtering with Vector Search (`Collection` + `AttributeIndex`)

## Goal

Add **native attribute filtering** for vector search in `izzydb`, modeled after turbopuffer's cluster-aware inverted index and dynamic expansion:

1. **Promote `Collection` to the top-level interface**, owning a shared RocksDB instance (`Storage`) and coordinating:
   - A **Document Store** (`cf_docs` + `cf_vectors`) for forward attributes and full-precision vectors.
   - **`ClusteringTree`** as one index inside `Collection` (owning centroids, `PackedLeafNode` RaBitQ codes, splits/merges, and SPFresh LIRE).
   - **`AttributeIndex`** as a second index inside `Collection` (sharing the same RocksDB and atomic `WriteBatch`), referencing `cluster_id` (`NodeId`) and `local_id` from `ClusteringTree`.
2. **Use `roaring::RoaringBitmap`** for compact, SIMD-friendly set representation at both:
   - **Cluster granularity** (downsampled summary bitmap per `(attr_id, value)` over `cluster_id: u32`) for zero-I/O cluster pruning.
   - **Document granularity** (per-cluster posting bitmap per `(attr_id, value, cluster_id)` over `local_id: u32`) for exact intra-leaf filtering.
3. **Support filtered vector search with cluster pruning, dynamic expansion, and exhaustive-scan crossover**, maintaining high recall across any filter selectivity (0.01% to 100%).

### Compatibility
- Backward-incompatible on-disk schema change (existing databases are not migrated).
- `ClusteringTree` remains usable internally (and optionally as a standalone vector-only index if desired), while `Collection` becomes the primary document + vector + filter API.

---

## 1. Crate & Dependency Changes

- **Workspace `Cargo.toml`:**
  - Add `roaring = "0.10"` (or latest `0.11`) to `[workspace.dependencies]` for `RoaringBitmap`.
- **Module layout:**
  - `crates/izzydb-core/src/`:
    - `config.rs`: narrow `NodeId` from `u64` to `u32` (supports up to $4.29 \times 10^9$ tree nodes / hundreds of billions of vectors, halves node-ID key/payload size, and enables native 32-bit `RoaringBitmap` for cluster IDs). Add `LocalId = u32`, `AttrId = u16`.
    - `schema.rs` *(new)*: `CollectionConfig`, `CollectionSchema`, `AttributeType`, `AttributeConfig` (`filterable: bool`), `AttributeValue`, `ScalarValue`, `Attributes` (`BTreeMap<String, AttributeValue>`).
    - `filter.rs` *(new)*: `Filter` AST (`And`, `Or`, `Not`, `Eq`, `NotEq`, `In`, `NotIn`, `Lt`, `Lte`, `Gt`, `Gte`, `Contains`, `ContainsAny`, `IsNull`, `IsNotNull`) and validation against `CollectionSchema`.
  - `crates/izzydb-storage/src/`:
    - `codec.rs` *(new)*: order-preserving key encoding for scalar attribute values (`i64`, `u64`, `f64`, `bool`, escaped-null-terminated `String`), forward `Attributes` binary codec, and `RoaringBitmap` serialize/deserialize helpers.
    - `node.rs`: update `PackedLeafNode` / `LeafNodeView` to include `next_local_id: u32` and `local_ids: [u32; n]`, and update `NodeId` fields to `u32`.
    - `storage.rs`: add `cf_docs`, `cf_attr_clusters`, `cf_attr_postings`; update `cf_doc_loc` value to `(leaf_id: u32 BE, local_id: u32 BE)`; extend `StorageBatch` and `TxContext` so `ClusteringTree` and `AttributeIndex` commit in one atomic `WriteBatch`.
  - `crates/izzydb/src/`:
    - `collection.rs` *(new)*: top-level `Collection` struct (`open`, `upsert`, `delete`, `get`, `search`, `verify_invariants`).
    - `attr_index.rs` *(new)*: `AttributeIndex` write/maintenance helpers (`apply_doc_changes`, `apply_cluster_moves`) and query-time filter evaluation (`eval_cluster_candidates`, `eval_leaf_mask`).
    - `tree.rs` & `lire.rs`: refactor `ClusteringTree` to borrow/share `Storage`, maintain stable `local_id`s in leaves, and emit `ClusterMutation` events (`Added`, `Removed`, `Moved`, `LeafRenumbered`) on every insert, delete, split, merge, redistribution, and LIRE round.
    - `search.rs`: add filtered best-first search with cluster pruning, dynamic expansion, and exhaustive-scan crossover; add masked RaBitQ scoring.

---

## 2. Data Format & RocksDB Schema

All components share a single RocksDB instance. Column families:

| Column Family | Constant | Owner | Key Format | Value Format | Compression |
|---|---|---|---|---|---|
| `cf_meta` | `CF_META` | Shared | `b"tree_meta"`, `b"schema"` | `TreeMeta`, encoded `CollectionSchema` | Default |
| `cf_docs` | `CF_DOCS` | Collection | `doc_id` (`u64` BE, 8 B) | Forward binary-encoded `Attributes` | Default |
| `cf_vectors` | `CF_VECTORS` | Collection | `doc_id` (`u64` BE, 8 B) | `[f32; dim]` LE (`dim * 4` B) | `None` |
| `cf_doc_loc` | `CF_DOC_LOC` | `ClusteringTree` | `doc_id` (`u64` BE, 8 B) | `leaf_id` (`u32` BE) \|\| `local_id` (`u32` BE) (8 B) | Default |
| `cf_internal_nodes` | `CF_INTERNAL_NODES` | `ClusteringTree` | `node_id` (`u32` BE, 4 B) | `PackedInternalNode` | Default |
| `cf_leaf_entries` | `CF_LEAF_ENTRIES` | `ClusteringTree` | `leaf_id` (`u32` BE, 4 B) | `PackedLeafNode` (with `local_ids`) | `None` |
| `cf_attr_clusters` | `CF_ATTR_CLUSTERS` | `AttributeIndex` | `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) \|\| `encoded_val` | `RoaringBitmap` of `cluster_id: u32` | Default |
| `cf_attr_postings` | `CF_ATTR_POSTINGS` | `AttributeIndex` | `attr_id` (`u16` BE) \|\| `type_tag` (`u8`) \|\| `encoded_val` \|\| `cluster_id` (`u32` BE) | `RoaringBitmap` of `local_id: u32` | Default |

### 2.1 Stable Per-Leaf `local_id` in `PackedLeafNode`

Because `PackedLeafNode.doc_ids` is kept sorted ascending by `DocId`, inserting a new document into the middle of a leaf shifts the array slot indices of subsequent documents. To avoid invalidating all attribute bitmaps in a leaf on every insert:

- **Header (`LEAF_HEADER_LEN = 24` bytes, unchanged size):**
  - Replace 4 of the existing reserved bytes with `next_local_id: u32` LE (and `parent_id` becomes `u32` LE).
- **Body:**
  - Add `local_ids: [u32; n]` LE immediately parallel to `doc_ids: [u64; n]`.
- **Lifecycle:**
  - **Normal insert into leaf:** allocate `local_id = leaf.next_local_id; leaf.next_local_id += 1;`. Existing entries keep their `local_id`s untouched $\implies$ only the newly inserted document's attribute bitmaps are updated.
  - **Normal delete from leaf:** remove the entry; remaining entries keep their `local_id`s untouched.
  - **Leaf split / merge / redistribution / LIRE:** renumber surviving entries compactly to `0..n` and reset `next_local_id = n`, emitting `LeafRenumbered` / `Moved` events so `AttributeIndex` rewrites only the affected clusters' bitmaps. Because `n <= max_leaf_size` (~160), every per-cluster `RoaringBitmap` stays inside a single 16-bit container (`high16 == 0`).

### 2.2 Order-Preserving Attribute Value Encoding (`codec.rs`)

To support both point lookups (`Eq`, `In`, `Contains`) and RocksDB range scans (`Lt`, `Lte`, `Gt`, `Gte`) over `cf_attr_clusters` and `cf_attr_postings`:

| `type_tag` | Type | Key Encoding (`encoded_val`) |
|---|---|---|
| `0x00` | `Null` / Missing | Empty (`0` bytes) — optional presence/null tracking |
| `0x01` | `Bool` | `1` byte (`0x00` = `false`, `0x01` = `true`) |
| `0x02` | `Int(i64)` | `8` bytes BE with sign bit flipped: `(v as u64) ^ (1u64 << 63)` |
| `0x03` | `Uint(u64)` | `8` bytes BE: `v.to_be_bytes()` |
| `0x04` | `Float(f64)` | `8` bytes BE IEEE-754 order-preserving (`-0.0 -> +0.0`, reject `NaN`; flip sign bit if positive, invert all bits if negative) |
| `0x05` | `String` | Escaped-null-terminated UTF-8 (`0x00` $\to$ `0x00 0xFF`, terminated with `0x00 0x00`) so `encoded_val \|\| cluster_id` is unambiguous and preserves exact lexicographical order |

- **Array attributes (`StringArray`, `IntArray`, `UintArray`):** Each distinct element is indexed under the scalar `type_tag` of its element type.
- **Schema registry:** `CollectionSchema` assigns each attribute name a monotonic `AttrId: u16` persisted in `cf_meta["schema"]`, avoiding repeated string field names in RocksDB keys.
- **Presence / Null checks (`IsNull`, `IsNotNull`, `Eq(..., Null)`):** Either maintain a reserved `type_tag = 0x00` bitmap per attribute for present docs (or null docs) so `IsNull` / `IsNotNull` prune clusters cleanly without scanning all values.

### 2.3 RocksDB Prefix / Point-Lookup Tuning

- `cf_attr_clusters`: bloom filter + prefix extractor on `attr_id (2B) + type_tag (1B)` for fast range scans (`Lt`/`Gt`/`Lte`/`Gte`) and point `multi_get_cf` (`Eq`/`In`).
- `cf_attr_postings`: point-lookup bloom filter (shared block cache) for fast `batched_multi_get_cf` when fetching `(attr_id, val, cluster_id)` bitmaps for probed leaves.

---

## 3. Public Interface (`Collection` vs `ClusteringTree`)

### 3.1 `Collection` (`crates/izzydb/src/collection.rs`)

```rust
pub struct Collection {
    storage: Arc<Storage>,
    schema: CollectionSchema,
    vector_index: ClusteringTree,
    attr_index: AttributeIndex,
}

impl Collection {
    pub fn open(path: impl AsRef<Path>, config: CollectionConfig) -> Result<Self>;

    pub fn upsert(
        &mut self,
        id: DocId,
        embedding: &[f32],
        attributes: Attributes,
    ) -> Result<()>;

    pub fn delete(&mut self, id: DocId) -> Result<()>;

    pub fn get(&self, id: DocId) -> Result<Option<Document>>;

    pub fn search(
        &self,
        query: &[f32],
        opts: SearchOptions,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>>;

    pub fn verify_invariants(&self) -> Result<CollectionStats>;
}
```

### 3.2 `ClusteringTree` as an Index Component

- `ClusteringTree` no longer owns the RocksDB `Storage` exclusively; it holds `Arc<Storage>` (or borrows `&Storage` + `&mut TxContext` during mutations).
- Maintain an in-memory `leaf_parent: HashMap<NodeId, NodeId>` alongside `internal_cache` (updated on open and on commit) so filtered search can map any qualifying leaf `NodeId` to its ancestor internal nodes in $O(\text{tree height})$ (~2–3 hops) without touching disk.
- Expose mutation methods that participate in a caller-owned `TxContext` and return a `ClusterDelta`:
  ```rust
  pub struct ClusterDelta {
      /// Docs whose (leaf_id, local_id) was removed (deleted or moved out).
      pub removed: Vec<(DocId, NodeId, LocalId)>,
      /// Docs whose (leaf_id, local_id) was added (inserted or moved in).
      pub added: Vec<(DocId, NodeId, LocalId)>,
  }
  ```
  *(On a split/merge/LIRE where a leaf's `local_id`s are renumbered, emitting the old `(doc_id, old_leaf, old_local)` in `removed` and new `(doc_id, new_leaf, new_local)` in `added`—or rebuilding the affected leaf's attribute postings directly from `cf_docs`—keeps the sync logic completely uniform and bug-free.)*

---

## 4. Write Path & Cluster Split / Merge / LIRE Coordination

All index updates for a single operation (or each LIRE maintenance step) execute inside **one `TxContext` / `rocksdb::WriteBatch`**:

### 4.1 `Collection::upsert(id, embedding, new_attrs)`
1. Validate `new_attrs` against `self.schema` and prepare `embedding`.
2. Read `old_attrs = tx.get_doc(id)?` and `old_loc = tx.get_doc_loc(id)?`.
3. Stage `tx.put_doc(id, &new_attrs)`.
4. Run `vector_index.upsert_in_tx(&mut tx, id, &vec)?`, returning `(ClusterDelta, Option<SplitPair>)`.
5. Update `AttributeIndex` in the same `tx`:
   - For `id` itself: remove `old_attrs` at `old_loc` (if any); insert `new_attrs` at `id`'s new `(leaf_id, local_id)`.
   - For any **other** documents touched by a leaf split (in `ClusterDelta`): batch-fetch their attributes from `cf_docs`, clear the old cluster's affected `(attr_id, val, old_cluster_id)` postings (or remove moved `local_id`s), and insert into the new `(attr_id, val, new_cluster_id)` postings.
   - Maintain **downsampled `cf_attr_clusters` invariant**:
     - When `cf_attr_postings[(attr_id, val, cluster_id)]` transitions `empty -> non-empty`: set `cluster_id` in `cf_attr_clusters[(attr_id, val)]`.
     - When `cf_attr_postings[(attr_id, val, cluster_id)]` transitions `non-empty -> empty`: delete the key from `cf_attr_postings` and clear `cluster_id` in `cf_attr_clusters[(attr_id, val)]` (deleting the cluster bitmap key if it becomes empty).
6. Commit `tx` atomically (`rocksdb::WriteBatch`) and update `internal_cache` + `leaf_parent`.
7. If `SplitPair` was produced and `enable_lire` is enabled: run `lire::run_lire` with a callback/hook so **each LIRE reassignment, split, or underflow merge/redistribution** applies its `ClusterDelta` to `AttributeIndex` inside the same `TxContext` before committing.

### 4.2 `Collection::delete(id)`
1. Look up `old_loc = tx.get_doc_loc(id)?`; if `None`, return `Ok(())`.
2. Read `old_attrs = tx.get_doc(id)?` and stage `tx.delete_doc(id)`.
3. Run `vector_index.remove_in_tx(&mut tx, id)?`, returning `ClusterDelta` (which includes any underflow merge/redistribution moves).
4. Apply `old_attrs` removal for `id` plus any merge/redistribution moves to `AttributeIndex` inside `tx`.
5. Commit `tx` atomically.

### 4.3 `TxContext` Overlay for `AttributeIndex`
- Cache dirty `RoaringBitmap`s in `TxContext`:
  - `dirty_attr_clusters: HashMap<Vec<u8>, RoaringBitmap>`
  - `dirty_attr_postings: HashMap<Vec<u8>, RoaringBitmap>`
- Multiple doc moves within the same split/merge/LIRE round mutate the in-memory `RoaringBitmap` in `TxContext` and serialize it **once** on `tx.commit()`, avoiding repeated decode/encode churn.

---

## 5. Filtered Vector Search Execution Path (`search.rs` + `attr_index.rs`)

When `filter` is `None`, `Collection::search` delegates directly to the existing unfiltered `ClusteringTree::search` fast path (zero overhead).

When `filter` is `Some(filter)`:

### Step 1: Cluster-Level Pruning (`eval_cluster_candidates`)
Evaluate the `Filter` AST against `cf_attr_clusters` (downsampled `RoaringBitmap`s of `cluster_id: u32`) using conservative upper-bound semantics:

```rust
pub enum ClusterCandidates {
    /// Filter cannot safely prune clusters (e.g., top-level `Not(...)`).
    All,
    /// Guaranteed superset of leaf `NodeId`s that may contain matching docs.
    Subset(RoaringBitmap),
}
```

- `Eq(attr, val)` / `Contains(attr, val)`: point lookup in `cf_attr_clusters` $\to$ `Subset(bm)`.
- `In(attr, vals)` / `ContainsAny(attr, vals)`: batched `multi_get_cf` + bitwise `|` $\to$ `Subset(union)`.
- `Lt` / `Lte` / `Gt` / `Gte`: RocksDB range scan over `cf_attr_clusters` for `(attr_id, type_tag, [lo..hi])` + bitwise `|` $\to$ `Subset(union)`.
- `And(children)`: intersect all `Subset`s (`&`); if any child is empty, short-circuit to `Subset(empty)`.
- `Or(children)`: if any child is `All` $\to$ `All`; otherwise union (`|`).
- `Not(_)` / `NotEq` / `NotIn`: returns `All` (since a `1` bit in a downsampled bitmap means "at least 1 doc has this value", not "all docs have this value"). Exact negation happens at the `local_id` level in Step 3.

If `ClusterCandidates::Subset(bm)` and `bm.is_empty()`: **return `Ok(Vec::new())` immediately** (0 leaf reads, 0 vector reads).

### Step 2: Exhaustive-Scan Crossover (Highly Selective Filters)
If `ClusterCandidates::Subset(bm)` has `bm.len() <= opts.exhaustive_cluster_threshold` (e.g., default `<= 8` clusters, or configurable):
1. Batch-fetch `PackedLeafNode` blobs for all `cluster_id`s in `bm` and evaluate exact `local_id` bitmaps in `cf_attr_postings`.
2. Map surviving `(cluster_id, local_id)` $\to$ `DocId` via `LeafNodeView.local_ids` and `LeafNodeView.doc_ids`.
3. Batch-fetch full vectors for those `DocId`s from `cf_vectors`, compute exact `search_distance_bytes`, select top `k`, and return (**100% recall**, skips tree traversal and RaBitQ quantization).

### Step 3: Pruned Best-First Traversal + Dynamic Cluster Expansion
For broader filters (`All` or `bm.len() > exhaustive_cluster_threshold`):

1. **Ancestor pruning (when `Subset(bm)` is sparse, e.g. `bm.len() < total_leaves / 4`):**
   - Walk `leaf_parent` upward from each `leaf_id` in `bm` (1–3 hops in a 100x-fanout tree) to build `allowed_internal: RoaringBitmap`.
   - Any internal node not in `allowed_internal` has **zero** qualifying leaves in its entire subtree and is skipped during traversal.
2. **Best-first priority queue (`BinaryHeap` min-heap by `(distance, node_id)`):**
   - Start with `(meta.root_id, 0.0, meta.root_level)`.
   - Pop closest internal nodes, pushing only allowed children (internal nodes in `allowed_internal`, leaf nodes in `bm`).
   - Collect the next batch of closest qualifying leaves (batch size = `opts.beam_width`, growing on subsequent expansion rounds).
3. **Exact per-leaf filtering + masked RaBitQ scoring:**
   - For the batch of qualifying leaves:
     - Batch-fetch `PackedLeafNode` blobs from `cf_leaf_entries`.
     - Evaluate the exact `Filter` AST per leaf against `cf_attr_postings` (using `batched_multi_get_cf` across all `(attr_id, val, leaf_id)` keys needed for the batch):
       - `Not(sub)` inside a leaf computes `leaf_all_local_ids - eval_leaf(sub)` (where `leaf_all_local_ids` is built from `view.local_ids`), making negation 100% exact.
     - If a leaf's resulting `matching_local_ids: RoaringBitmap` is empty (e.g., `And` of two attributes that existed in the same cluster on *different* docs), skip scoring that leaf.
     - Otherwise, build a compact slot mask over `0..view.entry_count` (e.g. checking `matching_local_ids.contains(local_id)` per slot) and run **`score_leaf_rabitq_masked`** so only matching documents are pushed to `candidates`.
4. **Dynamic expansion condition:**
   - Target Stage-1 candidate count: `target = opts.rerank_candidates.unwrap_or(opts.k).max(opts.k)`.
   - Keep pulling the next batch of closest qualifying leaves from the best-first heap until:
     - `candidates.len() >= target` **and** `probed_leaves >= opts.beam_width`, **OR**
     - the priority queue is empty, **OR**
     - `probed_leaves >= opts.max_expanded_leaves` (safety cap, e.g., `beam_width * 16` or configurable).
5. **Stage 2 exact reranking:**
   - Unchanged: take top `rerank_candidates.max(k)` from `candidates`, sort by `doc_id`, batch-fetch from `cf_vectors`, score with `search_distance_bytes`, and return top `k`.

---

## 6. Masked RaBitQ Scoring (`crates/izzydb-rabitq/src/simd.rs`)

- Add `score_leaf_rabitq_masked` (or an optional `allowed_slots: Option<&[u64]>` bitset in `RabitqLeafScoreParams`, 1 bit per slot `0..entry_count`, e.g., `[u64; 3]` for `max_leaf_size <= 192`):
  - The existing SIMD 8-wide loop computes distances in tight word-major batches; emitting into `candidates` simply checks `(allowed_slots[slot / 64] >> (slot % 64)) & 1 != 0` (or skips an entire 8-slot chunk when its 8 mask bits are all `0`!).
  - Keeps the hot unfiltered path unchanged and gives near-zero overhead when filtering dense clusters while skipping empty 8-entry SIMD lanes on sparse clusters.

---

## 7. Invariant Verification (`crates/izzydb/src/verify.rs`)

Extend `verify_invariants` to `Collection::verify_invariants`:
1. Existing `ClusteringTree` invariants (disjoint internal/leaf CFs, no orphans, `cf_vectors` completeness, `cf_doc_loc` bijection including `local_id` uniqueness and `< leaf.next_local_id`).
2. `cf_docs` completeness: exact 1-to-1 match with `cf_vectors` and `TreeMeta.num_vectors`.
3. `cf_attr_postings` exactness: rebuild expected `(attr_id, type_tag, encoded_val, cluster_id) -> RoaringBitmap<local_id>` from `cf_docs` + `cf_doc_loc` and assert exact equality with disk (no missing entries, no stale `local_id`s, no empty bitmaps).
4. `cf_attr_clusters` exactness: assert bit `cluster_id` is set in `cf_attr_clusters[(attr_id, type_tag, encoded_val)]` **iff** `cf_attr_postings[(attr_id, type_tag, encoded_val, cluster_id)]` exists and is non-empty.

---

## 8. Implementation Phases

1. **Phase 1 — Core Types & Codecs (`izzydb-core`, `izzydb-storage`):**
   - Narrow `NodeId` to `u32`, add `LocalId = u32`, `AttrId = u16`, add `roaring` dependency.
   - Implement `schema.rs`, `filter.rs`, `codec.rs` (order-preserving scalar encodings + `Attributes` binary codec + `RoaringBitmap` codec).
   - Update `PackedLeafNode` / `LeafNodeView` with `next_local_id` and `local_ids`.
   - Add `cf_docs`, `cf_attr_clusters`, `cf_attr_postings` and `TxContext` overlays in `storage.rs`.
2. **Phase 2 — `ClusteringTree` Delta Emission & `AttributeIndex`:**
   - Refactor `ClusteringTree` (`tree.rs`, `lire.rs`) to accept shared `Arc<Storage>` / `&mut TxContext`, maintain `leaf_parent`, and emit `ClusterDelta` on insert, delete, split, merge, redistribution, and LIRE.
   - Implement `attr_index.rs` (`apply_doc_changes`, `apply_cluster_moves`, `eval_cluster_candidates`, `eval_leaf_mask`).
   - Implement `Collection` (`collection.rs`) wiring `ClusteringTree` + `AttributeIndex` + `cf_docs` atomically.
3. **Phase 3 — Filtered Search & Masked RaBitQ:**
   - Add `score_leaf_rabitq_masked` in `izzydb-rabitq/src/simd.rs`.
   - Implement pruned best-first traversal, dynamic expansion, and exhaustive-scan crossover in `search.rs`.
4. **Phase 4 — Verification, Tests & Benchmarks:**
   - Extend `verify_invariants` and add comprehensive tests in `crates/izzydb/tests/collection_tests.rs`:
     - Upsert (attribute change + vector change), delete, reopen persistence.
     - Heavy churn triggering leaf splits, underflow merges, redistributions, and multi-round LIRE with `verify_invariants` after every stage.
     - Filtered search correctness & recall against brute-force ground truth across 0.1%, 1%, 10%, 50%, and 90% selectivities for `Eq`, `NotEq`, `In`, `NotIn`, `Lt`/`Lte`/`Gt`/`Gte`, `Contains`, `ContainsAny`, `And`, `Or`, `Not`.
     - False-positive cluster pruning test (cluster has doc A with `x=1` and doc B with `y=2`, query `x=1 AND y=2` must skip or return 0 from that cluster and dynamically expand to other clusters).
   - Update `docs/key-encoding.md`, `docs/index-path.md`, `docs/search-path.md` and add a filtered recall/latency benchmark example.
