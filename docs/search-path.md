# Search Path

This document describes how `ClusteringTree` executes approximate nearest neighbor queries using beam traversal over internal nodes, RaBitQ bitwise scanning over leaf posting lists, and optional full-vector reranking.

## Entry Points

Defined on `ClusteringTree` in [`src/tree.rs`](../src/tree.rs) and implemented in [`src/search.rs`](../src/search.rs):

- `search(query, k, beam_width)` — Stage 1 only. Scores candidates exclusively from quantized codes in `cf_leaf_entries` without reading `cf_vectors`.
- `search_with_rerank(query, k, beam_width, rerank_candidates)` — Stage 1 RaBitQ shortlist + Stage 2 exact reranking against full vectors in `cf_vectors`.

## Step 1: Query Preparation

`prepare_vector(query, config.dim, config.metric)` validates length and finiteness (`NaN`/`Inf` rejected). For `DistanceMetric::Cosine`, the query vector is L2-normalized. If `k == 0` or `num_vectors == 0`, search returns immediately.

## Step 2: Top-Down Beam Traversal

Starting at `frontier = [(meta.root_id, 0.0)]` and `level = meta.root_level`:

1. While `level > 0`, scans `cf_internal_entries` for every node in `frontier` (`Storage::scan_internal_with`).
2. Computes `search_distance(metric, &q, centroid)` for each child entry:
   - `L2Squared`: $\sum_i (q_i - c_i)^2$
   - `Cosine`: $\max(0, 1 - \langle q, c \rangle)$
   - `DotProduct`: $-\langle q, c \rangle$
3. Sorts candidates by `(distance, node_id)` and truncates `frontier` to `beam_width`.
4. Repeats until `level == 0`, leaving `frontier` with up to `beam_width` leaf `(leaf_id, distance)` pairs.

## Step 3: Stage 1 — RaBitQ Posting-List Scan

For each selected leaf in `frontier`:

1. **Fetch leaf centroid:** Reads `NodeHeader` from `cf_node_header` so distances are estimated against the exact centroid used when quantizing that leaf's entries.
2. **Prepare asymmetric 4-bit query state (`RabitqQuantizer::prepare_leaf_query` in [`src/rabitq.rs`](../src/rabitq.rs)):**
   - **`L2Squared` & `Cosine`:** Centers query residual $q_r = q - c$, computes $\|q_r\|_2^2$, and rotates $q_r$ via the 3-round sign-flip + normalized FWHT ($\tilde{q} = P^{-1} q_r$).
   - **`DotProduct`:** Reuses a single rotated query $\tilde{q} = P^{-1} q$ computed once per search (`prepare_dot_query`), and records $-\langle c, q \rangle$ for the leaf.
   - **4-bit scalar quantization:** Quantizes $\tilde{q}$ uniformly over $[\min \tilde{q}, \max \tilde{q}]$ into 4-bit integers $q_{u,i} \in \{0, \dots, 15\}$ split across 4 bit-planes (`planes[0..4]`) of `padded_dim / 64` `u64` words, along with `vl`, `delta`, and `sum_q`.
3. **Bitwise leaf scan (`Storage::scan_leaf_codes_with`):**
   - Streams `cf_leaf_entries` for `leaf_id` using a prefix-bloom-enabled iterator, casting aligned code bytes directly to `&[u64]` with zero copying on little-endian hosts.
   - For each `(doc_id, residual_norm, o_bar_dot_o, code_words)`:
     - If `residual_norm == 0.0`, the vector equals the leaf centroid and `estimate_distance` returns the precomputed exact `zero_res_dist`.
     - Otherwise, computes 5 popcounts per `u64` word:
       - `p_x = popcount(x)`
       - `p_b = popcount(x & planes[b])` for $b \in \{0, 1, 2, 3\}$
     - Reconstructs the unbiased inner-product estimate:
       $$\widehat{\langle r, q_{\text{target}} \rangle} = \|r\|_2 \cdot \frac{(2 (\text{vl} \cdot p_x + \Delta \sum_{b=0}^{3} 2^b p_b) - S_q) / \sqrt{D_{\text{pad}}}}{\langle \bar{o}, \tilde{o} \rangle}$$
     - Evaluates metric distance:
       - `L2Squared`: $\max(0, \|r\|_2^2 + \|q_r\|_2^2 - 2\widehat{\langle r, q_r \rangle})$
       - `Cosine`: $\max(0, \frac{1}{2}(\|r\|_2^2 + \|q_r\|_2^2 - 2\widehat{\langle r, q_r \rangle}))$
       - `DotProduct`: $-\langle c, q \rangle - \widehat{\langle r, q \rangle}$

If reranking is disabled (`search`), candidates are sorted by `(estimated_distance, doc_id)` and truncated to `k`.

## Step 4: Stage 2 — Optional Full-Vector Reranking

When called via `search_with_rerank`:

1. Sorts Stage 1 candidates and truncates to `shortlist_len = rerank_candidates.max(k)`.
2. Batch-fetches full-precision vectors for all shortlisted `DocId`s in a single `Storage::multi_get_vectors` (`rocksdb::DB::multi_get_cf`) lookup against `cf_vectors`.
3. Computes exact `search_distance(config.metric, &q, &vec)` for each shortlisted candidate, sorts by `(exact_distance, doc_id)`, and returns the top `k`.
