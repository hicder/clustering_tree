# RocksDB-Backed Hierarchical Clustering Tree — Design & Implementation Plan

A mutable, multi-dimensional B-tree of vector centroids (~100× fanout) backed by **RocksDB**, inspired by **turbopuffer ANN v3** and **SPFresh (LIRE)**.

---

## 1. Architecture Overview

Instead of a flat centroid list with a graph index on top (SPTAG), the tree recursively clusters vectors into leaves and clusters centroids into internal nodes up to a single root:

```text
Level 2 (Root):      [ Node 7 ]  (~100 child centroids)
                     /    |    \
Level 1 (Internal): [N4] [N5] [N6]  (~100 child centroids each)
                    / | \
Level 0 (Leaves): [L1][L2][L3] ...  (~100 vectors each)
```

### Core Properties
- **Target fanout (~100×):** Each internal node targets ~100 children; each leaf targets ~100 vectors (independently configurable). At 100× fanout, 100M vectors fit in just **4 levels** (`100^4 = 100M`).
- **Dynamic height:** Starts as a single empty leaf (`root_level = 0`). Grows upward when the root splits; collapses downward when the root is reduced to a single child.
- **Incremental rebalancing (LIRE):** Local splits, merges/redistributions, and bounded boundary reassignments keep clusters balanced and recall high without full rebuilds.
- **Crash-safe persistence:** All structural mutations (splits, merges, root growth/collapse, doc location updates) commit atomically via a single RocksDB `WriteBatch`.

---

## 2. Public Rust API

```rust
pub type DocId = u64;
pub type NodeId = u64;

pub enum DistanceMetric {
    L2Squared,
    Cosine,     // normalize vectors on ingest, use inner product / L2
    DotProduct,
}

pub struct TreeConfig {
    pub dim: usize,
    pub metric: DistanceMetric,

    // Internal node fanout bounds (~100x target, hysteresis avoids thrashing)
    pub min_internal_fanout: usize,    // default: 32
    pub max_internal_fanout: usize,    // default: 160

    // Leaf cluster size bounds
    pub min_leaf_size: usize,          // default: 32
    pub max_leaf_size: usize,          // default: 160

    // Search / routing parameters
    pub insert_beam_width: usize,      // default: 4  (probes per level during routing)
    pub lire_neighbor_count: usize,    // default: 8  (neighbor leaves checked on split)
    pub enable_lire: bool,             // default: true
    pub centroid_drift_eps: f32,       // default: 1e-3 (skip parent rewrite if drift < eps)
}

pub struct ClusteringTree { /* RocksDB handle, config, in-memory meta/cache */ }

impl ClusteringTree {
    pub fn open(path: impl AsRef<Path>, config: TreeConfig) -> Result<Self>;

    /// Insert (or update) a vector into the tree, triggering splits / root growth as needed.
    pub fn add_leaf(&mut self, id: DocId, embedding: &[f32]) -> Result<()>;

    /// Remove a vector from the tree, triggering merges / root collapse as needed.
    pub fn remove_leaf(&mut self, id: DocId, embedding: &[f32]) -> Result<()>;

    /// Approximate nearest neighbor search (beam search from root to leaves).
    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Result<Vec<(DocId, f32)>>;

    /// Verify all structural invariants (for testing & fuzzing).
    pub fn verify_invariants(&self) -> Result<TreeStats>;
}
```

> **Note on `min` / `max` hysteresis:**
> Always enforce `min_size * 2 < max_size` (e.g., `32 * 2 = 64 < 160`).
> - Splitting a node of size `161` produces two nodes of ~`80` each — well above `32` and below `160`.
> - Merging two nodes of size `< 32` produces `< 64` — well below `160`.
> This prevents split/merge oscillation on alternating inserts and deletes.

---

## 3. RocksDB Storage Schema

### Why Prefix-Keyed Entries Instead of Whole-Node Blobs?
At 768–1536 dimensions (`f32`), a single node with 100 entries is **300–600 KB**. Serializing an entire node as one RocksDB value would rewrite 300–600 KB on every single `add_leaf` / `remove_leaf`.

Instead, we store each entry under a **fixed 8-byte `NodeId` big-endian prefix** and enable a RocksDB `SliceTransform::create_fixed_prefix(8)` prefix bloom filter.
- **Point insert/delete:** Writes/deletes only **1 entry** (~3–6 KB) + a tiny header update.
- **Node scan (search / split / merge):** A single sequential prefix seek reads all ~100 contiguous entries from RocksDB's block cache / SSTs.

### Column Families

| Column Family | Key Format (Big-Endian) | Value Format | Purpose |
| :--- | :--- | :--- | :--- |
| `cf_meta` | `b"tree_meta"` | `TreeMeta { root_id, root_level, next_node_id, dim, metric, num_vectors }` | Global tree state |
| `cf_node_header` | `node_id: [u8; 8]` | `NodeHeader { level: u8, parent_id: Option<u64>, entry_count: u32, subtree_count: u64, centroid: Vec<f32> }` | Node metadata, parent pointer, total leaf count, cached centroid |
| `cf_internal_entries` | `node_id: [u8; 8] \|\| child_id: [u8; 8]` | `InternalEntry { centroid: Vec<f32>, subtree_count: u64 }` | Child centroids & weights of an internal node (`level >= 1`) |
| `cf_leaf_entries` | `leaf_id: [u8; 8] \|\| doc_id: [u8; 8]` | Raw `f32` LE bytes (`dim * 4` bytes) | Vectors stored in a leaf (`level == 0`) |
| `cf_doc_loc` | `doc_id: [u8; 8]` | `leaf_id: [u8; 8]` | Exact `DocId -> LeafId` reverse lookup |

### Why `cf_doc_loc` Is Essential
Because routing is approximate and centroids shift over time (splits, merges, LIRE reassignments), routing `embedding` from the root during `remove_leaf(id, embedding)` is **not guaranteed** to land on the exact leaf currently holding `id`.
- `cf_doc_loc` (16 bytes per vector) gives an **$O(1)$ exact lookup** of `leaf_id` for `remove_leaf` and for handling duplicate/updated `DocId`s in `add_leaf`.
- Combined with `parent_id` in `cf_node_header`, any mutation (including LIRE neighbor updates) can walk upward to the root in $O(\text{height})$ lookups.

### RocksDB Tuning
- `cf_internal_entries` & `cf_leaf_entries`: set `prefix_extractor = SliceTransform::create_fixed_prefix(8)` and enable `memtable_prefix_bloom_ratio`.
- `cf_doc_loc` & `cf_node_header`: enable full-key bloom filters (10 bits/key) for fast point lookups.
- **In-memory upper-level cache (Phase 2):** Pin or LRU-cache internal nodes (`level >= 1`) in memory since internal levels are ~1% the size of leaves.

---

## 4. Core Algorithms

### 4.1 Routing / Search (`route_to_leaf` & `search`)
Given a query vector `q` and `beam_width` `B`:
1. Start with `frontier = [(root_id, 0.0)]` at `level = root_level`.
2. While `level > 0`:
   - For each `node_id` in `frontier`, scan `cf_internal_entries` with prefix `node_id` and compute `dist(q, child.centroid)` for every child.
   - Keep the top `B` closest children across all expanded nodes as the next `frontier`.
   - `level -= 1`.
3. At `level == 0`:
   - For `route_to_leaf`: return the single closest leaf in `frontier` (using `B = insert_beam_width`, e.g., 4, avoids local minima during insert).
   - For `search(q, k, B)`: scan `cf_leaf_entries` for all `B` leaves in `frontier`, compute exact distances, and return the top `k` `(DocId, dist)`.

---

### 4.2 `add_leaf(id, embedding)`
1. **Validate & normalize:** Check `embedding.len() == dim`. If `Cosine`, L2-normalize.
2. **Handle existing `DocId` (idempotency / upsert):**
   - Look up `id` in `cf_doc_loc`. If present, call `remove_leaf` (or remove old entry first) so a `DocId` never exists in two leaves.
3. **Route to target leaf:**
   - If the tree is empty (`root_level == 0` and `root.entry_count == 0`), target is `root_id`.
   - Otherwise, `leaf_id = route_to_leaf(embedding, config.insert_beam_width)`.
4. **Insert into leaf (in a `WriteBatch`):**
   - Put `cf_leaf_entries[leaf_id || id] = embedding`.
   - Put `cf_doc_loc[id] = leaf_id`.
   - Increment `leaf.entry_count` and update `leaf.centroid` (incremental mean: `c' = c + (x - c) / n`, or recompute on split).
   - Increment `meta.num_vectors`.
5. **Check leaf overflow:**
   - If `leaf.entry_count > config.max_leaf_size`:
     - Run `split_leaf(leaf_id)` (Section 4.4), which may propagate splits upward to internal nodes and grow a new root level.
   - Else if `dist(old_centroid, new_centroid) > config.centroid_drift_eps`:
     - Update `(leaf_id, new_centroid)` in `leaf.parent_id` (if non-root).

---

### 4.3 `remove_leaf(id, embedding)`
1. **Locate leaf:**
   - Look up `leaf_id = cf_doc_loc.get(id)`. If `None`, return `Ok(())` (or `Err(NotFound)` — decide based on desired semantics; returning `bool` or `Ok(())` is clean).
2. **Delete from leaf (in a `WriteBatch`):**
   - Delete `cf_leaf_entries[leaf_id || id]`.
   - Delete `cf_doc_loc[id]`.
   - Decrement `leaf.entry_count` and `meta.num_vectors`.
3. **Handle leaf state:**
   - **If `leaf_id == root_id`** (meaning `root_level == 0`):
     - Recompute `leaf.centroid` (or zero if empty) and commit. Done.
   - **If `leaf.entry_count == 0`** (empty non-root leaf):
     - Delete `leaf_id` from `cf_node_header`.
     - Remove `leaf_id` from `parent_id` via `remove_child_from_parent(parent_id, leaf_id)`.
   - **If `leaf.entry_count < config.min_leaf_size`**:
     - Run `handle_underflow(leaf_id, level = 0)` (Section 4.6).
   - **Else**:
     - Update `leaf.centroid` (incremental downdate or recompute); if drift `> centroid_drift_eps`, update parent's `cf_internal_entries[parent_id || leaf_id]`.

---

### 4.4 Splitting a Node & Maintaining ~100× Fanout (`split_node`)

Splitting works identically for **leaves (`level == 0`)** and **internal nodes (`level >= 1`)**, differing only in whether the items being clustered are `(DocId, Vec<f32>)` or `(ChildNodeId, Centroid)`.

#### Balanced 2-Means Clustering
Given `N > max_size` items `[(item_id, vec)]`:
1. **Initialization (k-means++ for `k = 2`):**
   - Pick `c1` randomly (or farthest from the node's current centroid).
   - Pick `c2` with probability proportional to `dist(x, c1)^2` (or simply farthest point from `c1` for determinism).
2. **Lloyd's iterations** (e.g., 10–15 iterations or until convergence):
   - Assign each item to `argmin(dist(x, c1), dist(x, c2))`.
   - Recompute `c1, c2` as the means of their partitions (normalized if `Cosine`).
3. **Balance enforcement (critical guard against degenerate/outlier splits):**
   - Require both partitions to have at least `min_split = min_size` items (e.g., `>= 32`).
   - If `|P1| < min_split` (or `|P2| < min_split`):
     - Score items in the larger partition by `margin(x) = dist(x, c_small) - dist(x, c_large)` (smallest margin = closest to boundary).
     - Move the lowest-margin items to the smaller partition until `|P_small| == min_split`.
     - Recompute `c1, c2`.
   - If all vectors are identical (`c1 == c2`), split 50/50 by index to guarantee termination.

#### Applying the Split (Atomically in `WriteBatch`)
Suppose node `U` at `level = L` splits into `P1` (stays in `U`, new centroid `c1`) and `P2` (moves to new node `V = alloc_node_id()`, new centroid `c2`):
1. **Write `U` and `V`:**
   - For every item moved to `P2`:
     - Delete from `U`'s entry CF (`cf_leaf_entries` or `cf_internal_entries`), insert into `V`'s entry CF.
     - If `L == 0` (leaf): update `cf_doc_loc[doc_id] = V`.
     - If `L >= 1` (internal): update `cf_node_header[child_id].parent_id = Some(V)`.
   - Write updated `NodeHeader` for `U` (`entry_count = |P1|`, `centroid = c1`) and new `NodeHeader` for `V` (`level = L`, `parent_id = U.parent_id`, `entry_count = |P2|`, `centroid = c2`).
2. **Attach `V` to parent (or grow a new root level!):**
   - **Case A: `U` is the current root (`U.parent_id == None`):**
     - **Create a new level!** (See Section 4.5).
   - **Case B: `U` has a parent `P` (`level = L + 1`):**
     - Update `cf_internal_entries[P || U] = c1`.
     - Insert `cf_internal_entries[P || V] = c2`.
     - Increment `P.entry_count` by 1 and update `P.centroid`.
     - **Recursive cascade:** If `P.entry_count > config.max_internal_fanout`, call `split_node(P)`!
3. **LIRE boundary reassignment (for `L == 0`, when `enable_lire` is true):**
   - Run after the split (see Section 4.7).

---

### 4.5 Growing & Shrinking Tree Levels (Root Management)

#### Creating a New Level (Root Split)
Triggered when `U == meta.root_id` overflows and splits into `(U, c1)` and `(V, c2)` at `level = L`:
1. Allocate `new_root = alloc_node_id()`.
2. Create `cf_node_header[new_root] = NodeHeader { level: L + 1, parent_id: None, entry_count: 2, centroid: mean(c1, c2) }`.
3. Insert `cf_internal_entries[new_root || U] = c1` and `cf_internal_entries[new_root || V] = c2`.
4. Update `U.parent_id = Some(new_root)` and `V.parent_id = Some(new_root)`.
5. Update `meta.root_id = new_root`, `meta.root_level = L + 1`.

**Tree growth progression example (`max_leaf = 160`, `max_internal = 160`):**
- `0 ..= 160` vectors → `root_level = 0` (1 leaf).
- `161` vectors → root leaf splits → `root_level = 1` (1 internal root → 2 leaves).
- `~16,000` vectors (~161 leaves) → level-1 root splits → `root_level = 2` (1 root → 2 level-1 nodes → ~161 leaves).
- `~1.6M` vectors → `root_level = 3`.
- `~160M` vectors → `root_level = 4`.

#### Collapsing a Level (Root Shrink)
Triggered after a merge or empty-node deletion removes a child from `meta.root_id`:
1. While `meta.root_level > 0` and `root.entry_count == 1`:
   - Read the single remaining `only_child_id` from `cf_internal_entries[root_id || *]`.
   - Set `cf_node_header[only_child_id].parent_id = None`.
   - Delete `cf_node_header[root_id]` and `cf_internal_entries[root_id || only_child_id]`.
   - Update `meta.root_id = only_child_id`, `meta.root_level -= 1`.

---

### 4.6 Underflow: Merge or Redistribute (`handle_underflow`)

When a non-root node `U` at `level = L` drops below `min_size` (`min_leaf_size` or `min_internal_fanout`):
1. **Find nearest sibling `S`:**
   - Scan `U`'s parent `P` (`cf_internal_entries[P || *]`) for the sibling `S != U` with minimum `dist(U.centroid, S.centroid)`.
   - *(Edge case: if `P` has only 1 child, `P` itself must be the root and will collapse; if `P` is somehow non-root with 1 child, merge `P` first.)*
2. **Branch on combined size `total = U.entry_count + S.entry_count`:**
   - **Case 1 — Merge (`total <= max_size`):**
     - Move all entries from `U` into `S` (updating `cf_doc_loc` if `L == 0`, or child `parent_id = S` if `L >= 1`).
     - Recompute `S.centroid` and `S.entry_count = total`.
     - Delete `U` (`cf_node_header[U]`, `cf_internal_entries[P || U]`).
     - Update `cf_internal_entries[P || S] = S.centroid`.
     - Decrement `P.entry_count`.
     - **Propagate upward:**
       - If `P == root_id` and `P.entry_count == 1` → **collapse root** (Section 4.5).
       - Else if `P != root_id` and `P.entry_count < config.min_internal_fanout` → recursively call `handle_underflow(P)`.
   - **Case 2 — Redistribute (`total > max_size`):**
     - Load all entries of `U` and `S` together and run **Balanced 2-Means** to partition them into two balanced sets `P1, P2` (both `>= min_size` and `<= max_size`).
     - Rewrite `U` with `P1` and `S` with `P2` (updating `cf_doc_loc` or child `parent_id` for moved items).
     - Update both centroids `c1, c2` in `P`.
     - No node is deleted, so underflow stops here.

---

### 4.7 SPFresh LIRE Boundary Reassignment (Leaves)

After leaf `U` (old centroid `c_old`) splits into `U` (`c1`) and `V` (`c2`):
1. Find `M = config.lire_neighbor_count` nearest neighbor leaves to `c_old` via `search_leaves(c_old, M + 2)` (excluding `U` and `V`).
2. **Outward check (from `U` and `V` to neighbors):**
   - For each vector `(doc_id, x)` in `U` (with centroid `c1`): if some neighbor `N` has `dist(x, c_N) < dist(x, c1)`, move `(doc_id, x)` to `N`. Same for `V`.
3. **Inward check (from neighbors to `U` or `V`):**
   - For each neighbor `N` (with centroid `c_N`):
     - **Fast NPA filter:** Only scan `N` if `min(dist(c_N, c1), dist(c_N, c2)) < dist(c_N, c_old)` (the split moved a centroid closer to `N`).
     - For each `(doc_id, x)` in `N`: let `best = argmin_{c1, c2} dist(x, c)`. If `dist(x, best) < dist(x, c_N)`, move `(doc_id, x)` from `N` to `U` or `V`.
4. **Preventing cascade storms:**
   - Reassignment moves are collected and applied in a `WriteBatch`.
   - Maintain a **maintenance queue** `VecDeque<NodeId>` of any neighbor leaf that now overflows (`> max_leaf_size`) or underflows (`< min_leaf_size`).
   - Process the queue with a **depth/iteration cap** (e.g., max 4 LIRE rounds per `add_leaf`) so a single insert has bounded latency; any remaining splits can still execute without triggering further LIRE.

---

### 4.8 Computing & Propagating Internal Node Centroids

Yes — every internal node's embedding is the **centroid of its children**, specifically the **subtree-weighted mean** (or optionally unweighted mean) of its child centroids:

#### Formula (Exact Subtree Mean)
For an internal node `P` with children `[(c_1, w_1), ..., (c_m, w_m)]` where `c_i` is child `i`'s centroid and `w_i = subtree_count` is the number of leaf vectors in child `i`'s subtree:

```text
w_total = sum(w_i)
centroid(P) = (1 / w_total) * sum(w_i * c_i)
```

- **Why weighted?** Because each child `i` has `sum_{v in subtree(i)} v = w_i * c_i`, weighting by `subtree_count` (8 bytes per child) gives the **exact mean of all leaf vectors** in `P`'s subtree without ever reading leaves, even when sibling subtrees have uneven sizes (e.g., 35 vs. 150).
- **Unweighted alternative:** `(1 / m) * sum(c_i)` is simpler and works well when fanout is tightly balanced, though slightly skewed when sibling sizes diverge.
- **Metric-specific normalization:**
  - `L2Squared` / `DotProduct`: use the weighted arithmetic mean as-is (do **not** normalize for `DotProduct` since norms matter).
  - `Cosine`: L2-normalize the resulting vector (`centroid / ||centroid||_2`), which is the standard **spherical k-means** centroid.

#### Recompute vs. Incremental Drift
- **Avoid unbounded incremental downdates on deletes:** Subtracting vectors incrementally (`(n * c - x) / (n - 1)`) accumulates floating-point drift over many deletes. Because each node has only ~100 entries, **recomputing the centroid directly from the node's entries** takes only a few microseconds and stays drift-free.
- **When to recompute and propagate upward (avoiding a hot root):**
  1. **Structural events (always):** Recompute `P.centroid` whenever `P` splits, merges, redistributes, or adds/removes a child.
  2. **Leaf inserts/deletes (drift-thresholded):** Recompute the leaf's centroid (or update incrementally and recompute every `K` mutations, e.g., `K = 16`). Only update `cf_internal_entries[parent || leaf]` if `dist(old_centroid, new_centroid) > config.centroid_drift_eps`.
  3. **Upward cascade stops naturally:** When a parent `P` updates a child's centroid, recompute `P.centroid` from its ~100 children; only propagate to `P.parent_id` if `dist(old_P, new_P) > centroid_drift_eps`. Since level-2+ nodes average over 10,000+ vectors, normal inserts almost never rewrite the root.

---

## 5. Proposed Rust Crate Structure

```text
clustering_tree/
├── Cargo.toml
├── PLAN.md
└── src/
    ├── lib.rs          # Public API & re-exports
    ├── error.rs        # TreeError (thiserror)
    ├── config.rs       # TreeConfig & DistanceMetric
    ├── metric.rs       # L2, Cosine, DotProduct + vector normalization (SIMD-friendly loops)
    ├── storage.rs      # RocksDB open/CF setup, key codecs, WriteBatch builder, prefix scans
    ├── node.rs         # TreeMeta, NodeHeader, LeafEntry, InternalEntry serialization
    ├── kmeans.rs       # k-means++ init, Lloyd's 2-means, balanced split enforcement
    ├── tree.rs         # ClusteringTree: add_leaf, remove_leaf, split_node, handle_underflow, root grow/collapse
    ├── lire.rs         # SPFresh NPA boundary reassignment & bounded maintenance queue
    ├── search.rs       # Beam search & route_to_leaf
    └── verify.rs       # Full tree invariant checker & stats collector
```

### Suggested `Cargo.toml` Dependencies
- `rocksdb = "0.23"` (or latest 0.22/0.23)
- `thiserror = "2"`
- `rand = "0.9"` (or `0.8`, with `SmallRng` seeded deterministically for reproducible tests)
- `bytemuck = { version = "1", features = ["extern_crate_alloc"] }` (zero-copy `&[f32] <-> &[u8]` casting)

---

## 6. Phased Implementation Roadmap

### Phase 1: Storage & Primitives
- [ ] Implement `config.rs`, `error.rs`, and `metric.rs` (L2², Cosine, Dot, mean/centroid helpers).
- [ ] Implement `storage.rs` and `node.rs`:
  - Open RocksDB with the 5 column families + 8-byte prefix extractor on `cf_internal_entries` and `cf_leaf_entries`.
  - Clean `StorageBatch` wrapper around `rocksdb::WriteBatch` so tree logic never deals with raw byte keys.
  - Initialize empty tree (`root_id = 1`, `root_level = 0`) on fresh DB; reload `TreeMeta` on reopen.
- [ ] Implement `kmeans.rs`: deterministic balanced 2-means with degenerate/identical-vector handling.

### Phase 2: Core Mutable Tree (`add_leaf`, `remove_leaf`, Splits, Merges, Levels)
- [ ] Implement `search.rs`: `route_to_leaf` and `search(query, k, beam_width)`.
- [ ] Implement `add_leaf` + `split_node` + **root level growth** (recursively splitting internal nodes when fanout `> max_internal_fanout`).
- [ ] Implement `remove_leaf` + `handle_underflow` (merge & redistribute) + **root level collapse**.
- [ ] Implement `verify.rs`: exhaustive `verify_invariants()` checking:
  1. Every node reachable from `root_id` has the exact expected `level` (`parent.level - 1`).
  2. Every child's `parent_id` matches its parent, and `root.parent_id == None`.
  3. Every `NodeHeader.entry_count` equals the actual count of entries in its CF.
  4. `cf_doc_loc` is a 1-to-1 bijection with all entries in `cf_leaf_entries`.
  5. Every non-root node satisfies `min_size <= entry_count <= max_size` (and root has `>= 2` children if `root_level > 0`).
  6. No orphaned nodes exist in `cf_node_header`.

### Phase 3: SPFresh LIRE & Recall Quality
- [ ] Implement `lire.rs`: NPA boundary reassignment on leaf splits with bounded maintenance queue.
- [ ] Benchmark recall@10 vs brute force on synthetic Gaussian clusters + random uniform data (e.g., 10k → 100k vectors, 128-d) with and without LIRE.

### Phase 4: Testing, Fuzzing & Performance Polish
- [ ] **Small-fanout stress tests:** Run with `min = 2, max = 5` to force rapid multi-level growth (4–5 levels) and collapse back down to 0 levels in milliseconds.
- [ ] **Randomized operation fuzzer:** Random sequences of `add_leaf` (including duplicate IDs), `remove_leaf` (including missing IDs), and reopen-from-disk, running `verify_invariants()` after every batch.
- [ ] **Scale test:** Test with default ~100× fanout (`min = 32, max = 160`) up to 50k–200k vectors to verify level transitions (`0 → 1 → 2 → 3`) and recall@10.
- [ ] *(Optional Phase 5)*: Add in-memory cache for internal nodes (`level >= 1`), batch insert API (`add_leaves`), and RaBitQ / SQ8 quantization for leaf scanning.
