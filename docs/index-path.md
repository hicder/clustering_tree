# Index Path

This document describes how `ClusteringTree` persists state in RocksDB and maintains tree structure, centroids, and RaBitQ posting-list codes on inserts, upserts, and deletes.

## 1. RocksDB Storage Layout

All state is persisted across six column families in [`src/storage.rs`](../src/storage.rs):

| Column Family | Key | Value | Access Pattern |
|---|---|---|---|
| `cf_meta` | `"tree_meta"` | `TreeMeta` (39B: `root_id`, `root_level`, `next_node_id`, `dim`, `metric`, `num_vectors`, `rabitq_seed`, `rabitq_bits`) | Point lookup |
| `cf_node_header` | `node_id` (8B BE) | `NodeHeader` (`level`, `parent_id`, `entry_count`, `subtree_count`, `centroid`) | Point lookup (bloom filter) |
| `cf_internal_entries` | `node_id` (8B BE) \|\| `child_id` (8B BE) | `InternalEntry` (`subtree_count`, `centroid`) | 8-byte prefix scan (prefix bloom) |
| `cf_leaf_entries` | `leaf_id` (8B BE) \|\| `doc_id` (8B BE) | `QuantizedEntry` (`residual_norm: f32`, `o_bar_dot_o: f32`, `code: [u64; padded_dim / 64]`) | 8-byte prefix scan (prefix bloom) |
| `cf_doc_loc` | `doc_id` (8B BE) | `leaf_id` (8B BE) | Point lookup (bloom filter) |
| `cf_vectors` | `doc_id` (8B BE) | Full `f32` vector (`dim * 4` bytes, L2-normalized for `Cosine`) | Point / batched `multi_get_cf` |

All mutations in an operation go through `TxContext`, a read-your-own-writes overlay that commits changes to `cf_meta`, `cf_node_header`, `cf_internal_entries`, `cf_leaf_entries`, `cf_doc_loc`, and `cf_vectors` in a single atomic `rocksdb::WriteBatch`.

## 2. Insert & Upsert Flow (`add_leaf`)

Implemented in [`ClusteringTree::add_leaf`](../src/tree.rs):

1. **Vector preparation (`prepare_vector`)**:
   - Validates `embedding.len() == config.dim` and all components are finite.
   - If `DistanceMetric::Cosine`, L2-normalizes the vector in place.
2. **Idempotent upsert**:
   - Checks `cf_doc_loc` for `id`. If present, removes the old entry via `remove_doc_internal(id)` first.
3. **Leaf routing**:
   - If `root_level == 0`, inserts directly into `root_id`.
   - Otherwise, `search::route_to_leaf` runs top-down beam search (`beam_width = config.insert_beam_width`) using `cluster_distance` (`L2Squared` for `L2Squared`/`DotProduct`, `1 - dot` for `Cosine`) to select the closest leaf.
4. **Full vector + doc location persistence**:
   - Writes the prepared full vector to `cf_vectors`, maps `id -> target_leaf` in `cf_doc_loc`, and increments `TreeMeta.num_vectors`.
5. **Leaf update vs. split**:
   - **Non-overflow (`new_count <= config.max_leaf_size`):**
     - `new_count == 1`: sets `header.centroid = vec` and writes the quantized entry (`residual_norm = 0.0`).
     - `new_count % 16 == 0`: batch-fetches all full vectors for the leaf from `cf_vectors`, recomputes the exact mean centroid (`compute_mean_centroid`), and re-quantizes all entries in the leaf against the updated centroid.
     - Otherwise: updates `header.centroid` via `incremental_mean`. If `new_count <= 16` or centroid drift (`cluster_distance(old, new)`) exceeds `config.centroid_drift_eps`, re-quantizes all entries in the leaf; otherwise quantizes only the newly inserted vector.
     - Calls `propagate_centroid_upward` to update ancestor `InternalEntry` and `NodeHeader` centroids/counts while drift exceeds `centroid_drift_eps`.
     - Commits `TxContext`.
   - **Overflow (`new_count > config.max_leaf_size`):**
     - Inserts the new quantized entry, runs `split_node`, commits `TxContext`, and (if `config.enable_lire && root_level > 0`) runs SPFresh LIRE.

## 3. RaBitQ Quantization (`RabitqQuantizer::encode`)

Implemented in [`src/rabitq.rs`](../src/rabitq.rs):

- **Dimensions & rotation:** Pads `dim` to `padded_dim = dim.next_power_of_two().max(64)`. Applies a deterministic orthogonal transform $P^{-1}$ built from 3 rounds of random $\pm 1$ sign flips (seeded by `TreeMeta.rabitq_seed`) followed by a normalized Fast Walsh-Hadamard Transform ($O(D \log D)$).
- **Encoding relative to leaf centroid $c$:**
  1. Computes residual $r = x - c$ and Euclidean norm $\|r\|_2$. If $\|r\|_2 = 0$, emits `residual_norm = 0.0`, `o_bar_dot_o = 1.0`, and zeroed code words.
  2. Normalizes $o = r / \|r\|_2$ and rotates $\tilde{o} = P^{-1} o$.
  3. Packs sign bits ($\tilde{o}_i \ge 0 \implies 1$, else $0$) into `padded_dim / 64` little-endian `u64` words.
  4. Computes the alignment scalar $\langle \bar{o}, \tilde{o} \rangle = \frac{1}{\sqrt{D_{\text{pad}}}} \sum_i |\tilde{o}_i|$ (`o_bar_dot_o`) for unbiased distance estimation.

## 4. Splits, Root Growth, and SPFresh LIRE

- **Leaf & internal splits (`split_node` in [`src/tree.rs`](../src/tree.rs)):**
  - Leaf splits fetch full vectors for all leaf docs from `cf_vectors` and partition them via `balanced_two_means` ([`src/kmeans.rs`](../src/kmeans.rs)), which seeds two centroids deterministically (k-means++ style), runs up to 15 iterations of 2-means, and rebalances lowest-margin items if either partition falls below `min_leaf_size` (or `max_leaf_size` overflow).
  - Both halves are quantized against their new centroids (`centroid1`, `centroid2`) and written to `cf_leaf_entries` (`cf_vectors` is not rewritten).
  - Internal splits partition child `InternalEntry` centroids weighted by `subtree_count` with bounds `[min_internal_fanout, max_internal_fanout]`.
  - Splitting the root allocates a new root node at `level + 1`; splitting a non-root node cascades upward if the parent exceeds `max_internal_fanout`.
- **SPFresh LIRE (`lire::run_lire` in [`src/lire.rs`](../src/lire.rs)):**
  - Bounded to `MAX_LIRE_ROUNDS = 4`.
  - Searches `lire_neighbor_count` nearest leaves to `pair.old_centroid`.
  - **Outward check:** Scans full vectors (from `cf_vectors`) in the split pair `(left_id, right_id)` and moves any vector closer to a neighbor centroid.
  - **Inward check:** Filters neighbor leaves with the NPA condition ($\min(d(c_n, c_u), d(c_n, c_v)) < d(c_n, c_{\text{old}})$) and moves vectors closer to `left_id` or `right_id`.
  - Updates `cf_doc_loc`, recomputes touched leaf centroids, re-quantizes touched leaves, propagates centroid updates upward, and queues any newly overflowing or underflowing leaves for maintenance.

## 5. Deletion & Underflow (`remove_leaf`)

1. Deletes from `cf_leaf_entries`, `cf_doc_loc`, and `cf_vectors`, and decrements `num_vectors`.
2. Recomputes the leaf centroid from remaining full vectors. If the leaf does not underflow (or is the root), re-quantizes remaining entries when `new_count <= 16` or centroid drift exceeds `centroid_drift_eps`, then propagates upward.
3. If a non-root node falls below `min_leaf_size` / `min_internal_fanout`, `handle_underflow` picks the closest sibling:
   - **Merge** if combined entries $\le \text{max\_size}$ (recomputing centroid and re-quantizing merged leaf entries).
   - **Redistribute** via `balanced_two_means` otherwise.
   - Collapses internal roots down via `collapse_root` if a root ends up with a single child.
