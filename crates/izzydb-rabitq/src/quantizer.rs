use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use izzydb_core::config::DistanceMetric;
use izzydb_core::metric::{dot_product, l2_squared, search_distance};

/// Number of randomized sign-flip + Walsh-Hadamard rounds used to approximate
/// a Haar-uniform orthogonal rotation in O(D log D) time.
const ROTATION_ROUNDS: usize = 3;

/// Quantized RaBitQ posting-list entry.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedEntry {
    /// Squared Euclidean norm of the residual `||x - c||_2^2`.
    pub res_norm_sq: f32,
    /// Precomputed scale `||x - c||_2 / <o_bar, o_rot>` (or `0.0` when residual norm is `0.0`).
    pub scale: f32,
    /// Packed 1-bit RaBitQ code (`padded_dim / 64` `u64` words).
    pub code: Vec<u64>,
}

/// Reusable scratch buffers for zero-allocation RaBitQ encoding.
#[derive(Debug, Clone, Default)]
pub struct EncodeScratch {
    residual: Vec<f32>,
    rotated: Vec<f32>,
}

/// Metadata returned when quantizing a rotated query vector into 4 bit-planes.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedQueryMeta {
    pub vl: f32,
    pub delta: f32,
    pub sum_q: f32,
}

/// Reusable scratch buffers for zero-allocation query preparation across leaves in a beam.
#[derive(Debug, Clone)]
pub struct QueryScratch {
    pub rot_q: Vec<f32>,
    pub rot_diff: Vec<f32>,
    pub planes: [Vec<u64>; 4],
}

impl QueryScratch {
    pub fn new(padded_dim: usize, num_words: usize) -> Self {
        Self {
            rot_q: Vec::with_capacity(padded_dim),
            rot_diff: vec![0.0f32; padded_dim],
            planes: std::array::from_fn(|_| vec![0u64; num_words]),
        }
    }
}

/// 1-bit RaBitQ quantizer using a deterministic randomized Fast Walsh-Hadamard Transform.
#[derive(Debug, Clone)]
pub struct RabitqQuantizer {
    dim: usize,
    padded_dim: usize,
    num_words: usize,
    inv_sqrt_padded_dim: f32,
    signs: [Vec<f32>; ROTATION_ROUNDS],
}

impl RabitqQuantizer {
    /// Construct a quantizer for vectors of dimension `dim` using `seed` to derive
    /// the deterministic random orthogonal transform `P`.
    pub fn new(dim: usize, seed: u64) -> Self {
        let padded_dim = dim.next_power_of_two().max(64);
        let num_words = padded_dim / 64;
        let inv_sqrt_padded_dim = 1.0 / (padded_dim as f32).sqrt();

        let mut rng = SmallRng::seed_from_u64(seed);
        let signs = std::array::from_fn(|_| {
            (0..padded_dim)
                .map(|_| if rng.random_bool(0.5) { 1.0f32 } else { -1.0f32 })
                .collect()
        });

        Self {
            dim,
            padded_dim,
            num_words,
            inv_sqrt_padded_dim,
            signs,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    pub fn num_words(&self) -> usize {
        self.num_words
    }

    pub fn inv_sqrt_padded_dim(&self) -> f32 {
        self.inv_sqrt_padded_dim
    }

    /// Apply `P^{-1}` (3 rounds of random sign-flip + normalized FWHT) to `input`,
    /// zero-padding from `dim` to `padded_dim`.
    pub fn rotate_into(&self, input: &[f32], out: &mut Vec<f32>) {
        debug_assert_eq!(input.len(), self.dim);
        out.clear();
        out.extend_from_slice(input);
        out.resize(self.padded_dim, 0.0);

        for sign_vec in &self.signs {
            for (x, &s) in out.iter_mut().zip(sign_vec.iter()) {
                *x *= s;
            }
            fwht_in_place(out, self.inv_sqrt_padded_dim);
        }
    }

    /// Convenience helper returning `P^{-1}(input)` as a new `Vec<f32>`.
    pub fn rotate_new(&self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.padded_dim);
        self.rotate_into(input, &mut out);
        out
    }

    /// Quantize `vec` relative to its leaf `centroid`, writing the packed words into `code_out`
    /// and returning `(res_norm_sq, scale)` without heap allocations.
    pub fn encode_into(
        &self,
        vec: &[f32],
        centroid: &[f32],
        scratch: &mut EncodeScratch,
        code_out: &mut [u64],
    ) -> (f32, f32) {
        let (res_norm_sq, scale, _) = self.encode_into_with_popcnt(vec, centroid, scratch, code_out);
        (res_norm_sq, scale)
    }

    /// Quantize `vec` relative to its leaf `centroid`, writing packed words into `code_out`
    /// and returning `(res_norm_sq, scale, code_popcount)` without heap allocations.
    pub fn encode_into_with_popcnt(
        &self,
        vec: &[f32],
        centroid: &[f32],
        scratch: &mut EncodeScratch,
        code_out: &mut [u64],
    ) -> (f32, f32, u32) {
        debug_assert_eq!(vec.len(), self.dim);
        debug_assert_eq!(centroid.len(), self.dim);
        debug_assert_eq!(code_out.len(), self.num_words);

        scratch.residual.clear();
        let mut norm_sq = 0.0f64;
        for (&x, &c) in vec.iter().zip(centroid.iter()) {
            let d = x - c;
            scratch.residual.push(d);
            norm_sq += (d as f64) * (d as f64);
        }

        if norm_sq <= 0.0 {
            code_out.fill(0);
            return (0.0, 0.0, 0);
        }

        let norm = norm_sq.sqrt();
        let inv_norm = (1.0 / norm) as f32;
        for r in &mut scratch.residual {
            *r *= inv_norm;
        }

        self.rotate_into(&scratch.residual, &mut scratch.rotated);

        let mut abs_sum = 0.0f64;
        let mut popcnt = 0u32;
        for (word_idx, chunk) in scratch.rotated.as_chunks::<64>().0.iter().enumerate() {
            let mut word = 0u64;
            for (bit, &val) in chunk.iter().enumerate() {
                abs_sum += val.abs() as f64;
                if val >= 0.0 {
                    word |= 1u64 << bit;
                }
            }
            popcnt += word.count_ones();
            code_out[word_idx] = word;
        }

        let o_bar_dot_o = (abs_sum * (self.inv_sqrt_padded_dim as f64)).max(1e-12);
        let scale = (norm / o_bar_dot_o) as f32;
        (norm_sq as f32, scale, popcnt)
    }

    /// Quantize `vec` relative to its leaf `centroid`.
    pub fn encode(&self, vec: &[f32], centroid: &[f32]) -> QuantizedEntry {
        let mut scratch = EncodeScratch::default();
        let mut code = vec![0u64; self.num_words];
        let (res_norm_sq, scale) = self.encode_into(vec, centroid, &mut scratch, &mut code);
        QuantizedEntry {
            res_norm_sq,
            scale,
            code,
        }
    }

    /// Pre-rotate a query once for `DotProduct` searches (where query rotation is leaf-independent).
    pub fn prepare_dot_query(&self, query: &[f32]) -> QuantizedQueryBits {
        let mut rot_q = Vec::with_capacity(self.padded_dim);
        self.rotate_into(query, &mut rot_q);
        QuantizedQueryBits::from_rotated(&rot_q, self.num_words)
    }

    /// Prepare a leaf-specific 4-bit quantized query state for asymmetric RaBitQ scanning.
    pub fn prepare_leaf_query(
        &self,
        query: &[f32],
        centroid: &[f32],
        metric: DistanceMetric,
        dot_query_cache: Option<&QuantizedQueryBits>,
    ) -> LeafQueryState {
        match metric {
            DistanceMetric::L2Squared | DistanceMetric::Cosine => {
                let mut q_res = Vec::with_capacity(self.dim);
                for (&q, &c) in query.iter().zip(centroid.iter()) {
                    q_res.push(q - c);
                }
                let q_res_norm_sq = l2_squared(query, centroid);
                let zero_res_dist = search_distance(metric, query, centroid);

                let mut rot_q = Vec::with_capacity(self.padded_dim);
                self.rotate_into(&q_res, &mut rot_q);
                let q_bits = QuantizedQueryBits::from_rotated(&rot_q, self.num_words);

                LeafQueryState {
                    metric,
                    q_res_norm_sq,
                    zero_res_dist,
                    inv_sqrt_padded_dim: self.inv_sqrt_padded_dim,
                    q_bits,
                }
            }
            DistanceMetric::DotProduct => {
                let zero_res_dist = -dot_product(centroid, query);
                let q_bits = match dot_query_cache {
                    Some(cached) => cached.clone(),
                    None => self.prepare_dot_query(query),
                };
                LeafQueryState {
                    metric,
                    q_res_norm_sq: 0.0,
                    zero_res_dist,
                    inv_sqrt_padded_dim: self.inv_sqrt_padded_dim,
                    q_bits,
                }
            }
        }
    }
}

/// Quantize `rot_q` into 4 bit-planes in-place inside `planes`.
pub fn pack_query_bits_into(rot_q: &[f32], planes: &mut [Vec<u64>; 4]) -> QuantizedQueryMeta {
    let num_words = rot_q.len() / 64;
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::NEG_INFINITY;
    let mut sum_q = 0.0f64;

    for &val in rot_q {
        if val < v_min {
            v_min = val;
        }
        if val > v_max {
            v_max = val;
        }
        sum_q += val as f64;
    }

    let span = v_max - v_min;
    if span <= 1e-12 {
        for plane in planes.iter_mut() {
            plane.resize(num_words, 0);
            plane.fill(0);
        }
        return QuantizedQueryMeta {
            vl: v_min,
            delta: 0.0,
            sum_q: sum_q as f32,
        };
    }

    let delta = span / 15.0;
    let inv_delta = 15.0 / span;
    for plane in planes.iter_mut() {
        plane.resize(num_words, 0);
    }

    for (word_idx, chunk) in rot_q.as_chunks::<64>().0.iter().enumerate() {
        let mut w0 = 0u64;
        let mut w1 = 0u64;
        let mut w2 = 0u64;
        let mut w3 = 0u64;
        for (bit, &val) in chunk.iter().enumerate() {
            let q_int = (((val - v_min) * inv_delta).round() as i32).clamp(0, 15) as u8;
            let mask = 1u64 << bit;
            if (q_int & 1) != 0 {
                w0 |= mask;
            }
            if (q_int & 2) != 0 {
                w1 |= mask;
            }
            if (q_int & 4) != 0 {
                w2 |= mask;
            }
            if (q_int & 8) != 0 {
                w3 |= mask;
            }
        }
        planes[0][word_idx] = w0;
        planes[1][word_idx] = w1;
        planes[2][word_idx] = w2;
        planes[3][word_idx] = w3;
    }

    QuantizedQueryMeta {
        vl: v_min,
        delta,
        sum_q: sum_q as f32,
    }
}

/// Compute `rot_q - rot_c` (using pre-rotated centroid bytes) and pack 4 bit-planes into `scratch`.
pub fn pack_query_bits_from_diff_bytes_into(
    rot_q: &[f32],
    rot_c_bytes: &[u8],
    rot_diff: &mut Vec<f32>,
    planes: &mut [Vec<u64>; 4],
) -> QuantizedQueryMeta {
    debug_assert_eq!(rot_q.len() * 4, rot_c_bytes.len());
    rot_diff.resize(rot_q.len(), 0.0);

    #[cfg(target_endian = "little")]
    if let Ok(rot_c) = bytemuck::try_cast_slice::<u8, f32>(rot_c_bytes) {
        for ((d, &q), &c) in rot_diff.iter_mut().zip(rot_q.iter()).zip(rot_c.iter()) {
            *d = q - c;
        }
        return pack_query_bits_into(rot_diff, planes);
    }

    for ((d, &q), &chunk) in rot_diff
        .iter_mut()
        .zip(rot_q.iter())
        .zip(rot_c_bytes.as_chunks::<4>().0.iter())
    {
        let c = f32::from_le_bytes(chunk);
        *d = q - c;
    }
    pack_query_bits_into(rot_diff, planes)
}

/// 4-bit scalar-quantized rotated query split into 4 bit-planes for `AND + popcount` scans.
#[derive(Debug, Clone)]
pub struct QuantizedQueryBits {
    pub vl: f32,
    pub delta: f32,
    pub sum_q: f32,
    pub planes: [Vec<u64>; 4],
}

impl QuantizedQueryBits {
    fn from_rotated(rot_q: &[f32], num_words: usize) -> Self {
        let mut planes = std::array::from_fn(|_| vec![0u64; num_words]);
        let meta = pack_query_bits_into(rot_q, &mut planes);
        Self {
            vl: meta.vl,
            delta: meta.delta,
            sum_q: meta.sum_q,
            planes,
        }
    }
}

/// Precomputed query state for scanning quantized entries in a single leaf.
#[derive(Debug, Clone)]
pub struct LeafQueryState {
    metric: DistanceMetric,
    q_res_norm_sq: f32,
    zero_res_dist: f32,
    inv_sqrt_padded_dim: f32,
    q_bits: QuantizedQueryBits,
}

impl LeafQueryState {
    /// Estimate the search distance to a quantized entry using bitwise `AND + popcount`.
    #[inline]
    pub fn estimate_distance(
        &self,
        res_norm_sq: f32,
        scale: f32,
        code_words: &[u64],
    ) -> f32 {
        if res_norm_sq == 0.0 {
            return self.zero_res_dist;
        }

        let q0 = &self.q_bits.planes[0];
        let q1 = &self.q_bits.planes[1];
        let q2 = &self.q_bits.planes[2];
        let q3 = &self.q_bits.planes[3];

        debug_assert_eq!(code_words.len(), q0.len());

        let mut p_x = 0u32;
        let mut p0 = 0u32;
        let mut p1 = 0u32;
        let mut p2 = 0u32;
        let mut p3 = 0u32;

        for (i, &x) in code_words.iter().enumerate() {
            p_x += x.count_ones();
            p0 += (x & q0[i]).count_ones();
            p1 += (x & q1[i]).count_ones();
            p2 += (x & q2[i]).count_ones();
            p3 += (x & q3[i]).count_ones();
        }

        let weighted_pop = (p0 + (p1 << 1) + (p2 << 2) + (p3 << 3)) as f32;
        let sum_x_q = self.q_bits.vl * (p_x as f32) + self.q_bits.delta * weighted_pop;
        let o_bar_dot_q = (2.0 * sum_x_q - self.q_bits.sum_q) * self.inv_sqrt_padded_dim;
        let est_r_dot_q = scale * o_bar_dot_q;

        match self.metric {
            DistanceMetric::L2Squared => {
                (res_norm_sq + self.q_res_norm_sq - 2.0 * est_r_dot_q).max(0.0)
            }
            DistanceMetric::Cosine => {
                (0.5 * (res_norm_sq + self.q_res_norm_sq - 2.0 * est_r_dot_q)).max(0.0)
            }
            DistanceMetric::DotProduct => self.zero_res_dist - est_r_dot_q,
        }
    }
}

#[inline]
fn fwht_in_place(buf: &mut [f32], inv_sqrt_n: f32) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for chunk in buf.chunks_exact_mut(h * 2) {
            let (left, right) = chunk.split_at_mut(h);
            for (a, b) in left.iter_mut().zip(right.iter_mut()) {
                let x = *a;
                let y = *b;
                *a = x + y;
                *b = x - y;
            }
        }
        h *= 2;
    }
    for x in buf.iter_mut() {
        *x *= inv_sqrt_n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izzydb_core::metric::prepare_vector;

    #[test]
    fn test_fwht_preserves_norm_and_dot_product() {
        let mut rng = SmallRng::seed_from_u64(7);
        for dim in [8, 12, 96, 128, 384] {
            let qtz = RabitqQuantizer::new(dim, 42);
            let a: Vec<f32> = (0..dim).map(|_| rng.random_range(-2.0..2.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.random_range(-2.0..2.0)).collect();

            let mut ra = Vec::new();
            let mut rb = Vec::new();
            qtz.rotate_into(&a, &mut ra);
            qtz.rotate_into(&b, &mut rb);

            let orig_dot = dot_product(&a, &b);
            let rot_dot = dot_product(&ra, &rb);
            assert!(
                (orig_dot - rot_dot).abs() < 1e-3,
                "dim={dim}: orig_dot={orig_dot}, rot_dot={rot_dot}"
            );

            let orig_norm = dot_product(&a, &a);
            let rot_norm = dot_product(&ra, &ra);
            assert!(
                (orig_norm - rot_norm).abs() < 1e-3,
                "dim={dim}: orig_norm={orig_norm}, rot_norm={rot_norm}"
            );
        }
    }

    #[test]
    fn test_rabitq_estimator_unbiased_across_metrics() {
        let mut rng = SmallRng::seed_from_u64(123);
        let dim = 256;
        let qtz = RabitqQuantizer::new(dim, 99);

        for metric in [
            DistanceMetric::L2Squared,
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
        ] {
            let raw_c: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let c = prepare_vector(&raw_c, dim, metric).unwrap();
            let raw_q: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let q = prepare_vector(&raw_q, dim, metric).unwrap();

            let dot_cache = if metric == DistanceMetric::DotProduct {
                Some(qtz.prepare_dot_query(&q))
            } else {
                None
            };
            let state = qtz.prepare_leaf_query(&q, &c, metric, dot_cache.as_ref());

            let mut sum_err = 0.0f64;
            let trials = 200;
            for _ in 0..trials {
                let raw_x: Vec<f32> = c
                    .iter()
                    .map(|&ci| ci + rng.random_range(-0.3..0.3))
                    .collect();
                let x = prepare_vector(&raw_x, dim, metric).unwrap();
                let entry = qtz.encode(&x, &c);
                let exact = search_distance(metric, &q, &x);
                let est = state.estimate_distance(entry.res_norm_sq, entry.scale, &entry.code);
                sum_err += (est - exact) as f64;
            }
            let mean_bias = (sum_err / trials as f64).abs();
            assert!(
                mean_bias < 0.25,
                "metric={metric:?}: mean_bias={mean_bias} unexpectedly high"
            );
        }
    }

    #[test]
    fn test_zero_residual_exact_distance() {
        let qtz = RabitqQuantizer::new(16, 1);
        let c = vec![0.25f32; 16];
        let q = vec![-0.5f32; 16];
        let entry = qtz.encode(&c, &c);
        assert_eq!(entry.res_norm_sq, 0.0);
        assert_eq!(entry.scale, 0.0);

        for metric in [DistanceMetric::L2Squared, DistanceMetric::DotProduct] {
            let state = qtz.prepare_leaf_query(&q, &c, metric, None);
            let est = state.estimate_distance(entry.res_norm_sq, entry.scale, &entry.code);
            let exact = search_distance(metric, &q, &c);
            assert!((est - exact).abs() < 1e-6);
        }
    }
}
