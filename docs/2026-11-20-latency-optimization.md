# Latency Optimization Note (2026-11-20)

This note documents the storage layout, caching, algorithmic, and SIMD changes made to reduce ANN query latency (targeting `beam = 4`) and index build time in `clustering_tree`.

## Bottlenecks in the Baseline

Before these changes, running `cargo run --release --example recall_bench` on an AMD Ryzen 9 7950X showed ~556–668 µs per query at `beam = 4` (~70–135 µs per leaf scanned). Over 95% of query time was spent outside distance math:

1. **Per-entry RocksDB keys & prefix iterators:** `cf_internal_entries` and `cf_leaf_entries` stored one KV pair per child/vector, plus separate `cf_node_header` lookups per leaf. Every leaf or internal scan opened a `DBIterator`, allocated per entry, and walked over overwritten versions and tombstones from splits, drift re-quantization, and LIRE.
2. **Repeated RocksDB tree traversal:** Top-down beam search scanned `cf_internal_entries` from RocksDB on every query and every `add_leaf` insert.
3. **Scattered full-vector rerank lookups:** `search_with_rerank` issued `rerank_candidates` (100) separate point lookups against `cf_vectors` (`doc_id -> vec`).
4. **Per-leaf FWHT & allocations:** `prepare_leaf_query` recomputed the 3-round Walsh-Hadamard rotation `P^{-1}(q - c)` for every leaf in the beam, allocated bit-plane `Vec<u64>`s per leaf, copied non-`Cosine` queries in `prepare_vector`, and ran full $O(N \log N)$ sorts for small top-`k` / beam selection.
5. **No runtime SIMD dispatch:** Default release builds targeted baseline `x86-64` (SSE2), missing hardware `POPCNT`, AVX2+FMA, and AVX-512 `VPOPCNTDQ`.

---

## Changes Implemented

### 1. Packed Node Blobs & Leaf-Grouped Full Vectors ([`src/node.rs`](../src/node.rs), [`src/storage.rs`](../src/storage.rs))

Replaced per-entry column families (`cf_node_header`, `cf_internal_entries`, `cf_leaf_entries`, `cf_vectors`) with one packed blob per node:

- **`cf_internal_nodes` (`node_id -> PackedInternalNode`):** Packs node metadata (`level`, `parent_id`, `subtree_count`, `centroid`) together with sorted `child_ids`, `child_subtree_counts`, and contiguous `child_centroids`.
- **`cf_leaf_entries` (`leaf_id -> PackedLeafNode`):** Stores a Struct-of-Arrays (SoA) leaf blob containing `parent_id`, `centroid`, pre-rotated `rotated_centroid` ($P^{-1}c$), sorted `doc_ids`, precomputed `res_norm_sqs` ($\|x - c\|_2^2$) and `scales` ($\|x - c\|_2 / \langle \bar{o}, \tilde{o} \rangle$), and word-major 1-bit RaBitQ `codes`. Zero-copy [`LeafNodeView`](../src/node.rs) parses directly over `DBPinnableSlice`.
- **`cf_leaf_vectors` (`leaf_id -> PackedLeafVectors`):** Co-locates full-precision `f32` vectors by `leaf_id` in the exact same sorted `doc_ids` slot order as `cf_leaf_entries`. `search_with_rerank` now fetches at most `beam_width` (4) blobs instead of 100 individual `doc_id` lookups, and `get_vector(doc_id)` resolves via `cf_doc_loc` + binary search inside [`LeafVectorsView`](../src/node.rs).
- **RocksDB tuning & zero-copy reads:** Switched batched leaf and vector reads to `batched_multi_get_cf` (`sorted_input = true`) and `get_pinned_cf`, added a shared 128 MiB LRU block cache, pinned L0 index/filter blocks in cache, enabled memtable whole-key filtering, and disabled compression on node/vector CFs.

### 2. In-Memory Internal Node Cache ([`src/tree.rs`](../src/tree.rs), [`src/storage.rs`](../src/storage.rs), [`src/lire.rs`](../src/lire.rs), [`src/verify.rs`](../src/verify.rs))

- `ClusteringTree` loads all `cf_internal_nodes` entries into `internal_cache: HashMap<NodeId, PackedInternalNode>` on `open()`.
- `TxContext::commit` writes the RocksDB `WriteBatch` first and updates `internal_cache` only after persistence succeeds.
- Top-down beam traversal in `search`, `search_with_rerank`, `route_to_leaf`, and `search_leaves` executes entirely in memory with zero RocksDB calls.

### 3. Pre-Rotated Centroids & Zero-Allocation Query Path ([`src/rabitq.rs`](../src/rabitq.rs), [`src/search.rs`](../src/search.rs), [`src/metric.rs`](../src/metric.rs))

- **Single FWHT per query:** Using linearity ($P^{-1}(q - c) = P^{-1}q - P^{-1}c$), `search` runs `rotate_into` once on `q` and subtracts each leaf's precomputed `rotated_centroid` into reusable `QueryScratch` buffers.
- **Precomputed RaBitQ scalars:** `res_norm_sq` and `scale = norm / o_bar_dot_o` are computed at encode time, removing a per-candidate float division and square in the hot scan loop.
- **Reduced allocations & partial sorting:** `prepare_vector_cow` borrows non-`Cosine` queries without copying, and `select_top_k` uses `select_nth_unstable_by` + `sort_unstable_by` on only the truncated top elements.

### 4. Runtime-Dispatched SIMD Kernels ([`src/simd.rs`](../src/simd.rs))

Added runtime feature detection (`is_x86_feature_detected!`) with scalar fallbacks so standard `cargo build --release` binaries use native vector instructions automatically:

- **`score_leaf_rabitq`:** Processes 8 entries in parallel over word-major RaBitQ codes using AVX-512 `vpopcntq` (`_mm512_popcnt_epi64`), AVX2+FMA distance evaluation, and branchless zero-residual handling (`_mm256_cmp_ps` + `_mm256_blendv_ps`), with `popcnt` and scalar tail fallbacks.
- **`l2_squared` / `dot_product` & `*_bytes` variants:** AVX-512F and AVX2+FMA kernels, including unaligned byte-slice loads for scoring directly out of RocksDB `DBPinnableSlice` buffers during Stage 2 reranking.

---

## Benchmark Comparison

Measured with `cargo run --release --example recall_bench` on AMD Ryzen 9 7950X:

### `beam = 4` Summary

| Dataset | Stage | Before | After | Speedup | Recall@10 (Before → After) |
|---|---|---:|---:|---:|---|
| Gaussian-50k-128d (`LIRE=false`) | `search` (RaBitQ) | 601.1 µs | **22.0 µs** | **27.3×** | 0.5260 → 0.5010 |
| Gaussian-50k-128d (`LIRE=false`) | `rerank@100` | 604.5 µs | **37.6 µs** | **16.1×** | 0.9920 → 0.9930 |
| Gaussian-50k-128d (`LIRE=true`) | `search` (RaBitQ) | 559.1 µs | **22.6 µs** | **24.7×** | 0.5030 → 0.5320 |
| Gaussian-50k-128d (`LIRE=true`) | `rerank@100` | 555.7 µs | **40.0 µs** | **13.9×** | 0.9990 → 0.9950 |
| Uniform-10k-128d (`LIRE=false`) | `search` (RaBitQ) | 608.6 µs | **15.1 µs** | **40.3×** | 0.1280 → 0.1270 |
| Uniform-10k-128d (`LIRE=false`) | `rerank@100` | 632.1 µs | **26.8 µs** | **23.6×** | 0.1670 → 0.1680 |
| Uniform-10k-128d (`LIRE=true`) | `search` (RaBitQ) | 650.9 µs | **14.1 µs** | **46.2×** | 0.1300 → 0.1360 |
| Uniform-10k-128d (`LIRE=true`) | `rerank@100` | 668.3 µs | **24.4 µs** | **27.4×** | 0.1880 → 0.1890 |

*(Note: slight run-to-run variation in Stage 1 RaBitQ recall is due to random `rabitq_seed` generation per fresh database in `Storage::open`.)*

### Full Beam Sweep & Build Times

| Dataset | Build (Before → After) | `beam=4` | `beam=8` | `beam=16` | `beam=32` |
|---|---|---:|---:|---:|---:|
| Gaussian-50k-128d (`LIRE=false`) | 13,109.4 ms → **6,089.4 ms** | 22.0 / 37.6 µs | 31.1 / 41.7 µs | 51.1 / 59.5 µs | 93.4 / 101.4 µs |
| Gaussian-50k-128d (`LIRE=true`) | 13,078.5 ms → **6,177.1 ms** | 22.6 / 40.0 µs | 30.5 / 41.2 µs | 50.3 / 58.9 µs | 93.4 / 102.1 µs |
| Uniform-10k-128d (`LIRE=false`) | 3,971.6 ms → **1,608.5 ms** | 15.1 / 26.8 µs | 24.3 / 41.2 µs | 45.9 / 68.2 µs | 88.9 / 120.6 µs |
| Uniform-10k-128d (`LIRE=true`) | 4,168.7 ms → **1,611.4 ms** | 14.1 / 24.4 µs | 24.2 / 40.7 µs | 45.4 / 74.3 µs | 88.6 / 138.3 µs |
