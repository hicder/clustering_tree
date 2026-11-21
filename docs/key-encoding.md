# Key & Binary Encoding Specification

This document specifies the persistent RocksDB key-value encodings, packed node/vector blob formats, and RaBitQ bit-packing layouts used by `ClusteringTree`.

Implementation sources:
- Column families, RocksDB keys, and CF options: [`src/storage.rs`](../src/storage.rs)
- Packed node/metadata serialization and zero-copy views: [`src/node.rs`](../src/node.rs)
- Metric tags and ID aliases: [`src/config.rs`](../src/config.rs)
- RaBitQ code and query bit-plane packing: [`src/rabitq.rs`](../src/rabitq.rs)
- Structural invariants: [`src/verify.rs`](../src/verify.rs)

---

## 1. Identifiers and Endianness Conventions

| Type | Rust Definition | Width | Endianness in RocksDB Keys & `cf_doc_loc` Values | Endianness Inside Packed Blobs (`TreeMeta`, `Packed*`) |
|---|---|---|---|---|
| `DocId` | `u64` | 8 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `NodeId` | `u64` | 8 bytes | **Big-endian** (`to_be_bytes`) | **Little-endian** (`to_le_bytes`) |
| `f32` scalars & vectors | `f32` | 4 bytes | — | **Little-endian** (`to_le_bytes`) |
| `u32` counts / `dim` | `u32` | 4 bytes | — | **Little-endian** (`to_le_bytes`) |

> **Why big-endian for RocksDB keys?**
> `node_key(id: u64) -> [u8; 8]` encodes IDs via `id.to_be_bytes()`. Lexicographical byte ordering in RocksDB therefore matches ascending numeric `u64` ordering, allowing `batched_multi_get_cf(..., sorted_input = true)` on ascending `sorted_leaf_ids` and ordered full-CF scans.
>
> **Why little-endian inside packed blobs?**
> On little-endian hosts (`#[cfg(target_endian = "little")]`), aligned `doc_ids`, `codes`, `centroid`, `rotated_centroid`, `res_norm_sqs`, and `scales` slices can be borrowed directly from RocksDB `DBPinnableSlice` buffers via `bytemuck::try_cast_slice` without copying or byte-swapping.

---

## 2. RocksDB Column Families & Key-Value Map

`Storage::open` opens six column families (plus RocksDB's required `default` CF, which is unused):

| Column Family | Constant | Key Format | Value Format | Compression | Read Pattern |
|---|---|---|---|---|---|
| `default` | `DEFAULT_COLUMN_FAMILY_NAME` | — | — | Default | Unused |
| `cf_meta` | `CF_META` | ASCII literal `b"tree_meta"` (9 B) | `TreeMeta` (39 B) | Default | Point `get_cf` |
| `cf_internal_nodes` | `CF_INTERNAL_NODES` | `node_id` (`u64` BE, 8 B) | `PackedInternalNode` | Default | Full CF load on `open()` |
| `cf_leaf_entries` | `CF_LEAF_ENTRIES` | `leaf_id` (`u64` BE, 8 B) | `PackedLeafNode` | `None` | `batched_multi_get_cf` (pinned) |
| `cf_leaf_vectors` | `CF_LEAF_VECTORS` | `leaf_id` (`u64` BE, 8 B) | `PackedLeafVectors` | `None` | `get_pinned_cf` / `batched_multi_get_cf` |
| `cf_doc_loc` | `CF_DOC_LOC` | `doc_id` (`u64` BE, 8 B) | `leaf_id` (`u64` BE, 8 B) | Default | Point `get_pinned_cf` |

### RocksDB Tuning for Point-Lookup CFs

`cf_internal_nodes`, `cf_leaf_entries`, `cf_leaf_vectors`, and `cf_doc_loc` share `point_lookup_cf_opts`:
- Shared 128 MiB LRU block cache (`BLOCK_CACHE_BYTES = 128 * 1024 * 1024`)
- 10.0 bits/key full-key Bloom filter (`set_bloom_filter(10.0, false)`)
- `set_memtable_whole_key_filtering(true)`
- `set_cache_index_and_filter_blocks(true)` and `set_pin_l0_filter_and_index_blocks_in_cache(true)`
- `DBCompressionType::None` on `cf_leaf_entries` and `cf_leaf_vectors` to keep pinned `DBPinnableSlice` reads zero-copy on hot query paths

### ID Allocation & Initial Database State

- `NodeId` and `DocId` are independent `u64` spaces.
- On fresh database creation (`cf_meta` missing `"tree_meta"`), `Storage::open` atomically writes:
  - `cf_meta["tree_meta"]`: `root_id = 1`, `root_level = 0`, `next_node_id = 2`, `dim = config.dim as u32`, `metric = config.metric`, `num_vectors = 0`, `rabitq_seed = rand::random::<u64>()`, `rabitq_bits = 1`.
  - `cf_leaf_entries[1u64.to_be_bytes()]`: empty `PackedLeafNode` (`parent_id = None`, `n = 0`).
  - `cf_leaf_vectors[1u64.to_be_bytes()]`: empty `PackedLeafVectors` (`n = 0`).
- Every subsequent node allocation increments `TreeMeta.next_node_id` (`TxContext::alloc_node_id`).

---

## 3. Binary Value Specifications

### 3.1 `TreeMeta` (`cf_meta`, key `b"tree_meta"`)

Fixed size: **39 bytes** (`META_LEN = 39`). Any other byte length fails decoding with `TreeError::CorruptedData`.

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 8 | `root_id` | `u64` LE | Must satisfy `root_id < next_node_id` |
| `8` | 1 | `root_level` | `u8` | `0` when root is a leaf; `> 0` when root is internal |
| `9` | 8 | `next_node_id` | `u64` LE | Next unallocated `NodeId` |
| `17` | 4 | `dim` | `u32` LE | Vector dimensionality (`> 0`) |
| `21` | 1 | `metric` | `u8` | `0` = `L2Squared`, `1` = `Cosine`, `2` = `DotProduct` |
| `22` | 8 | `num_vectors` | `u64` LE | Total indexed documents across all leaves |
| `30` | 8 | `rabitq_seed` | `u64` LE | Seed for deterministic FWHT sign-flip rotation $P^{-1}$ |
| `38` | 1 | `rabitq_bits` | `u8` | Must be `1` (1-bit RaBitQ); other values rejected on `open()` |

### 3.2 `cf_doc_loc` Value

Fixed size: **8 bytes**. Encoded using `node_key(leaf_id)` (`leaf_id.to_be_bytes()`). Notice this is **big-endian**, matching key encoding rather than the little-endian blob format used in `Packed*` structs.

### 3.3 `PackedInternalNode` (`cf_internal_nodes`)

Variable size: **`24 + 16 * n + (1 + n) * dim * 4` bytes**, where `n` is the number of children (`entry_count`).

#### Header (`INTERNAL_HEADER_LEN = 24` bytes)

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 1 | `level` | `u8` | Tree level (`>= 1`; leaves are level `0`) |
| `1` | 1 | `has_parent` | `u8` | `0` = `None` (root), `1` = `Some(parent_id)`; any other value fails decode |
| `2` | 2 | reserved | `[u8; 2]` | Written as `0` |
| `4` | 4 | `n` | `u32` LE | Number of child entries |
| `8` | 8 | `parent_id` | `u64` LE | Parent `NodeId` when `has_parent == 1`; `0` when `has_parent == 0` |
| `16` | 8 | `subtree_count` | `u64` LE | Sum of `child_subtree_counts` |

#### Body (starting at byte `24`)

| Offset | Size | Field | Type | Ordering / Layout |
|---|---|---|---|-|
| `24` | `n * 8` | `child_ids` | `[u64; n]` LE | Strictly sorted ascending by `NodeId` |
| `24 + 8n` | `n * 8` | `child_subtree_counts` | `[u64; n]` LE | Parallel to `child_ids` |
| `24 + 16n` | `dim * 4` | `centroid` | `[f32; dim]` LE | Weighted mean of child centroids |
| `24 + 16n + 4*dim` | `n * dim * 4` | `child_centroids` | `[f32; n * dim]` LE | Row-major; child `i` occupies `[i * dim .. (i + 1) * dim]` |

### 3.4 `PackedLeafNode` (`cf_leaf_entries`)

Struct-of-Arrays (SoA) blob storing the leaf centroid, pre-rotated centroid, sorted document IDs, precomputed RaBitQ scalars, and word-major 1-bit codes.

Derived parameters:
- `padded_dim = dim.next_power_of_two().max(64)`
- `rabitq_words = padded_dim / 64`

Total size: **`24 + (dim + padded_dim) * 4 + n * (16 + rabitq_words * 8)` bytes**, where `n` is the number of vectors in the leaf.

#### Header (`LEAF_HEADER_LEN = 24` bytes)

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 1 | `has_parent` | `u8` | `0` = `None` (root leaf), `1` = `Some(parent_id)` |
| `1` | 3 | reserved | `[u8; 3]` | Written as `[0, 0, 0]` |
| `4` | 4 | `n` | `u32` LE | Number of entries in this leaf |
| `8` | 8 | `parent_id` | `u64` LE | Parent `NodeId` when `has_parent == 1`; `0` when `has_parent == 0` |
| `16` | 8 | reserved | `u64` | Written as `0` |

#### Body (starting at byte `24`)

Note that `doc_ids` and `codes` are placed immediately after the 24-byte header (at 8-byte aligned offsets from the blob start) for fast SIMD and `u64` access in `LeafNodeView`:

| Section | Size (bytes) | Type | Layout / Semantics |
|---|---:|---|---|
| `doc_ids` | `n * 8` | `[u64; n]` LE | Strictly sorted ascending by `DocId` |
| `codes` | `n * rabitq_words * 8` | `[u64; n * rabitq_words]` LE | **Word-major** SoA: `codes[w * n + i]` is word `w` (`0..rabitq_words`) for entry slot `i` (`0..n`) |
| `centroid` | `dim * 4` | `[f32; dim]` LE | Leaf centroid $c$ |
| `rotated_centroid` | `padded_dim * 4` | `[f32; padded_dim]` LE | Precomputed $P^{-1} c$ |
| `res_norm_sqs` | `n * 4` | `[f32; n]` LE | Squared residual norm $\|x_i - c\|_2^2$ |
| `scales` | `n * 4` | `[f32; n]` LE | $\|x_i - c\|_2 / \langle \bar{o}_i, \tilde{o}_i \rangle$, or `0.0` when `res_norm_sq == 0.0` |

### 3.5 `PackedLeafVectors` (`cf_leaf_vectors`)

Co-locates all full-precision vectors belonging to a leaf in the exact same sorted `doc_ids` slot order as `PackedLeafNode`.

Total size: **`8 + n * 8 + n * dim * 4` bytes**, where `n` is the number of vectors in the leaf.

#### Header (`LEAF_VECS_HEADER_LEN = 8` bytes)

| Offset | Size | Field | Type | Notes |
|---:|---:|---|---|---|
| `0` | 4 | `n` | `u32` LE | Number of vectors in this leaf |
| `4` | 4 | reserved | `u32` | Written as `0` |

#### Body (starting at byte `8`)

| Section | Size (bytes) | Type | Layout / Semantics |
|---|---:|---|---|
| `doc_ids` | `n * 8` | `[u64; n]` LE | Strictly sorted ascending; must equal `PackedLeafNode.doc_ids` |
| `vectors` | `n * dim * 4` | `[f32; n * dim]` LE | Slot `i` occupies `[i * dim .. (i + 1) * dim]`; L2-normalized at insert when `metric == Cosine` |

`LeafVectorsView::find_vec_bytes` binary-searches `doc_ids` in-place on the pinned RocksDB slice and returns the `dim * 4`-byte vector slice for slot `i`.

---

## 4. RaBitQ Code & Query Bit-Plane Encoding

### 4.1 Orthogonal Rotation $P^{-1}$

Defined in `RabitqQuantizer::new(dim, seed)` and `RabitqQuantizer::rotate_into`:
1. Zero-pad the `dim`-dimensional input vector to `padded_dim = dim.next_power_of_two().max(64)`.
2. Seed `rand::rngs::SmallRng::seed_from_u64(seed)` (using `TreeMeta.rabitq_seed`).
3. Generate `ROTATION_ROUNDS = 3` sign vectors $s^{(r)} \in \{-1.0, +1.0\}^{D_{\text{pad}}}$ (`rng.random_bool(0.5)` $\implies +1.0$, else $-1.0$).
4. For each round $r \in \{0, 1, 2\}$, multiply element-wise by $s^{(r)}$ and apply an in-place Fast Walsh-Hadamard Transform scaled by `inv_sqrt_padded_dim = 1.0 / (padded_dim as f32).sqrt()`.

### 4.2 1-Bit Data Vector Encoding (`encode_into`)

Given full vector $x \in \mathbb{R}^D$ and leaf centroid $c \in \mathbb{R}^D$:
1. Compute residual $r = x - c$ and `norm_sq` $= \sum_{j=0}^{D-1} r_j^2$ in `f64`.
2. **Zero-residual special case (`norm_sq <= 0.0`):** emit `res_norm_sq = 0.0`, `scale = 0.0`, and all-zero `code` words (`[0u64; rabitq_words]`).
3. Otherwise normalize $o = r / \sqrt{\text{norm\_sq}}$ and rotate $\tilde{o} = P^{-1} o \in \mathbb{R}^{D_{\text{pad}}}$.
4. **Bit packing:** for coordinate $j \in \{0, \dots, D_{\text{pad}} - 1\}$, word index is `w = j / 64` and bit position is `b = j % 64`:
   $$\text{code}[w] \mathrel{|}= \begin{cases} 1 \ll b & \text{if } \tilde{o}_j \ge 0.0 \\ 0 & \text{otherwise} \end{cases}$$
5. Compute alignment scalar:
   $$\langle \bar{o}, \tilde{o} \rangle = \max\!\left(10^{-12},\; \frac{1}{\sqrt{D_{\text{pad}}}} \sum_{j=0}^{D_{\text{pad}}-1} |\tilde{o}_j|\right)$$
   and store `res_norm_sq = norm_sq as f32` and `scale = (norm / o_bar_dot_o) as f32`.
6. Persist in `PackedLeafNode` in **word-major** order: for slot `i` in a leaf of `n` entries, word `w` is stored at `codes[w * n + i]`. This allows SIMD kernels (`score_leaf_rabitq`) to load 8 consecutive entries' word `w` in a single contiguous 64-byte (`__m512i`) load.

### 4.3 Transient 4-Bit Query Bit-Plane Encoding (`pack_query_bits_into`)

Query encodings are ephemeral (kept in `QueryScratch`, never persisted):
- Using linearity ($P^{-1}(q - c) = P^{-1}q - P^{-1}c$ for `L2Squared`/`Cosine`, or $P^{-1}q$ for `DotProduct`), the rotated query vector $\tilde{q} \in \mathbb{R}^{D_{\text{pad}}}$ is uniformly quantized over $[v_{\min}, v_{\max}]$ with step $\Delta = (v_{\max} - v_{\min}) / 15$ into 4-bit integers $q_{\text{int}, j} = \text{clamp}(\text{round}((\tilde{q}_j - v_{\min}) / \Delta), 0, 15) \in \{0, \dots, 15\}$ (or all-zero planes with $\Delta = 0.0$ when $v_{\max} - v_{\min} \le 10^{-12}$).
- The 4 bits of $q_{\text{int}, j}$ are transposed into 4 bit-planes `planes[0..4]`, each of length `rabitq_words`, where bit `b = j % 64` of `planes[k][j / 64]` stores bit `k` of $q_{\text{int}, j}$ (`(q_int >> k) & 1`).

---

## 5. Key Consistency Invariants

Verified exhaustively by `verify_invariants` ([`src/verify.rs`](../src/verify.rs)):

1. **Disjoint node CFs:** Every `NodeId` reachable at `level > 0` exists only in `cf_internal_nodes`; every reachable `NodeId` at `level == 0` exists in both `cf_leaf_entries` and `cf_leaf_vectors`, and never in `cf_internal_nodes`.
2. **No orphaned blobs:** Every key present in `cf_internal_nodes`, `cf_leaf_entries`, and `cf_leaf_vectors` is reachable from `TreeMeta.root_id`.
3. **Leaf slot alignment:** For every leaf `leaf_id`, `PackedLeafNode.doc_ids == PackedLeafVectors.doc_ids` (identical sorted order and length).
4. **Bijection in `cf_doc_loc`:** Every `(doc_id, leaf_id)` in `cf_doc_loc` matches the exact leaf containing `doc_id`, `cf_doc_loc` contains no orphans, and `cf_doc_loc.len() == TreeMeta.num_vectors`.
5. **Atomic updates:** All key/value mutations across `cf_meta`, `cf_internal_nodes`, `cf_leaf_entries`, `cf_leaf_vectors`, and `cf_doc_loc` are staged in `TxContext` and committed in a single `rocksdb::WriteBatch` before updating the in-memory `internal_cache`.
