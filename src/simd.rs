use crate::config::{DistanceMetric, DocId};

/// Candidate scored during Stage 1 RaBitQ leaf scanning.
#[derive(Debug, Clone, Copy)]
pub struct LeafCandidate {
    pub doc_id: DocId,
    pub dist: f32,
}

/// Squared Euclidean distance `sum((a[i] - b[i])^2)` using runtime-dispatched SIMD.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // Safety: checked `avx512f` at runtime; slices have equal length.
            return unsafe { l2_squared_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: checked `avx2` + `fma` at runtime; slices have equal length.
            return unsafe { l2_squared_avx2(a, b) };
        }
    }
    l2_squared_scalar(a, b)
}

/// Inner product `sum(a[i] * b[i])` using runtime-dispatched SIMD.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // Safety: checked `avx512f` at runtime; slices have equal length.
            return unsafe { dot_product_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: checked `avx2` + `fma` at runtime; slices have equal length.
            return unsafe { dot_product_avx2(a, b) };
        }
    }
    dot_product_scalar(a, b)
}

/// Squared Euclidean distance between `a` and a little-endian `f32` byte slice `b_bytes`
/// without requiring 4-byte alignment on `b_bytes`.
#[inline]
pub fn l2_squared_bytes(a: &[f32], b_bytes: &[u8]) -> f32 {
    debug_assert_eq!(a.len() * 4, b_bytes.len());
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if is_x86_feature_detected!("avx512f") {
            // Safety: checked `avx512f` at runtime; `b_bytes.len() == a.len() * 4` and unaligned loads are used.
            return unsafe { l2_squared_bytes_avx512(a, b_bytes) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: checked `avx2` + `fma` at runtime; `b_bytes.len() == a.len() * 4` and unaligned loads are used.
            return unsafe { l2_squared_bytes_avx2(a, b_bytes) };
        }
    }
    #[cfg(target_endian = "little")]
    if let Ok(b) = bytemuck::try_cast_slice::<u8, f32>(b_bytes) {
        return l2_squared_scalar(a, b);
    }
    let mut sum = 0.0f32;
    for (&x, &chunk) in a.iter().zip(b_bytes.as_chunks::<4>().0.iter()) {
        let y = f32::from_le_bytes(chunk);
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Inner product between `a` and a little-endian `f32` byte slice `b_bytes`
/// without requiring 4-byte alignment on `b_bytes`.
#[inline]
pub fn dot_product_bytes(a: &[f32], b_bytes: &[u8]) -> f32 {
    debug_assert_eq!(a.len() * 4, b_bytes.len());
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if is_x86_feature_detected!("avx512f") {
            // Safety: checked `avx512f` at runtime; `b_bytes.len() == a.len() * 4` and unaligned loads are used.
            return unsafe { dot_product_bytes_avx512(a, b_bytes) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safety: checked `avx2` + `fma` at runtime; `b_bytes.len() == a.len() * 4` and unaligned loads are used.
            return unsafe { dot_product_bytes_avx2(a, b_bytes) };
        }
    }
    #[cfg(target_endian = "little")]
    if let Ok(b) = bytemuck::try_cast_slice::<u8, f32>(b_bytes) {
        return dot_product_scalar(a, b);
    }
    let mut sum = 0.0f32;
    for (&x, &chunk) in a.iter().zip(b_bytes.as_chunks::<4>().0.iter()) {
        let y = f32::from_le_bytes(chunk);
        sum += x * y;
    }
    sum
}

/// Parameters needed to batch-score a packed leaf node using RaBitQ bitwise popcount.
pub struct RabitqLeafScoreParams<'a> {
    pub entry_count: usize,
    pub rabitq_words: usize,
    pub doc_ids_bytes: &'a [u8],
    pub codes_bytes: &'a [u8],
    pub res_norm_sqs_bytes: &'a [u8],
    pub scales_bytes: &'a [u8],
    pub q0: &'a [u64],
    pub q1: &'a [u64],
    pub q2: &'a [u64],
    pub q3: &'a [u64],
    pub vl: f32,
    pub delta: f32,
    pub sum_q: f32,
    pub inv_sqrt_padded_dim: f32,
    pub q_res_norm_sq: f32,
    pub zero_res_dist: f32,
    pub metric: DistanceMetric,
}

/// Score all `entry_count` vectors in a packed leaf and append [`LeafCandidate`]s to `out`.
pub fn score_leaf_rabitq(params: &RabitqLeafScoreParams<'_>, out: &mut Vec<LeafCandidate>) {
    if params.entry_count == 0 {
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vpopcntdq")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            // Safety: runtime-checked AVX-512F + VPOPCNTDQ + AVX2 + FMA; slices validated by LeafNodeView.
            unsafe {
                match params.metric {
                    DistanceMetric::L2Squared => score_leaf_avx512::<0>(params, out),
                    DistanceMetric::Cosine => score_leaf_avx512::<1>(params, out),
                    DistanceMetric::DotProduct => score_leaf_avx512::<2>(params, out),
                }
            }
            return;
        }
        if is_x86_feature_detected!("popcnt") {
            // Safety: runtime-checked POPCNT.
            unsafe {
                match params.metric {
                    DistanceMetric::L2Squared => score_leaf_popcnt::<0>(params, out),
                    DistanceMetric::Cosine => score_leaf_popcnt::<1>(params, out),
                    DistanceMetric::DotProduct => score_leaf_popcnt::<2>(params, out),
                }
            }
            return;
        }
    }
    match params.metric {
        DistanceMetric::L2Squared => score_leaf_scalar::<0>(params, out),
        DistanceMetric::Cosine => score_leaf_scalar::<1>(params, out),
        DistanceMetric::DotProduct => score_leaf_scalar::<2>(params, out),
    }
}

#[inline]
fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    let (chunks_a, rem_a) = a.as_chunks::<4>();
    let (chunks_b, rem_b) = b.as_chunks::<4>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        let d0 = ca[0] - cb[0];
        let d1 = ca[1] - cb[1];
        let d2 = ca[2] - cb[2];
        let d3 = ca[3] - cb[3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    let mut sum = (s0 + s1) + (s2 + s3);
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    let (chunks_a, rem_a) = a.as_chunks::<4>();
    let (chunks_b, rem_b) = b.as_chunks::<4>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b.iter()) {
        s0 += ca[0] * cb[0];
        s1 += ca[1] * cb[1];
        s2 += ca[2] * cb[2];
        s3 += ca[3] * cb[3];
    }

    let mut sum = (s0 + s1) + (s2 + s3);
    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        sum += x * y;
    }
    sum
}

#[inline(always)]
fn score_single_entry_scalar<const METRIC: u8>(
    params: &RabitqLeafScoreParams<'_>,
    i: usize,
) -> LeafCandidate {
    let n = params.entry_count;
    let words = params.rabitq_words;
    let doc_id =
        u64::from_le_bytes(params.doc_ids_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    let res_norm_sq = f32::from_le_bytes(
        params.res_norm_sqs_bytes[i * 4..(i + 1) * 4]
            .try_into()
            .unwrap(),
    );
    let dist = if res_norm_sq == 0.0 {
        params.zero_res_dist
    } else {
        let scale =
            f32::from_le_bytes(params.scales_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        let mut p_x = 0u32;
        let mut p0 = 0u32;
        let mut p1 = 0u32;
        let mut p2 = 0u32;
        let mut p3 = 0u32;
        for w in 0..words {
            let off = (w * n + i) * 8;
            let x = u64::from_le_bytes(params.codes_bytes[off..off + 8].try_into().unwrap());
            p_x += x.count_ones();
            p0 += (x & params.q0[w]).count_ones();
            p1 += (x & params.q1[w]).count_ones();
            p2 += (x & params.q2[w]).count_ones();
            p3 += (x & params.q3[w]).count_ones();
        }
        let weighted_pop = (p0 + (p1 << 1) + (p2 << 2) + (p3 << 3)) as f32;
        let sum_x_q = params.vl * (p_x as f32) + params.delta * weighted_pop;
        let o_bar_dot_q = (2.0 * sum_x_q - params.sum_q) * params.inv_sqrt_padded_dim;
        let est_r_dot_q = scale * o_bar_dot_q;
        match METRIC {
            0 => (res_norm_sq + params.q_res_norm_sq - 2.0 * est_r_dot_q).max(0.0),
            1 => (0.5 * (res_norm_sq + params.q_res_norm_sq - 2.0 * est_r_dot_q)).max(0.0),
            _ => params.zero_res_dist - est_r_dot_q,
        }
    };
    LeafCandidate { doc_id, dist }
}

#[inline]
fn score_leaf_scalar<const METRIC: u8>(
    params: &RabitqLeafScoreParams<'_>,
    out: &mut Vec<LeafCandidate>,
) {
    let n = params.entry_count;
    out.reserve(n);
    for i in 0..n {
        out.push(score_single_entry_scalar::<METRIC>(params, i));
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "popcnt")]
unsafe fn score_leaf_popcnt<const METRIC: u8>(
    params: &RabitqLeafScoreParams<'_>,
    out: &mut Vec<LeafCandidate>,
) {
    score_leaf_scalar::<METRIC>(params, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut i = 0usize;
    let mut acc = _mm512_setzero_ps();
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    while i + 16 <= len {
        unsafe {
            let va = _mm512_loadu_ps(ap.add(i));
            let vb = _mm512_loadu_ps(bp.add(i));
            let d = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(d, d, acc);
        }
        i += 16;
    }

    let mut sum = _mm512_reduce_add_ps(acc);
    while i < len {
        unsafe {
            let d = *ap.add(i) - *bp.add(i);
            sum += d * d;
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut i = 0usize;
    let mut acc = _mm512_setzero_ps();
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    while i + 16 <= len {
        unsafe {
            let va = _mm512_loadu_ps(ap.add(i));
            let vb = _mm512_loadu_ps(bp.add(i));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        i += 16;
    }

    let mut sum = _mm512_reduce_add_ps(acc);
    while i < len {
        unsafe {
            sum += *ap.add(i) * *bp.add(i);
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    while i + 8 <= len {
        unsafe {
            let va = _mm256_loadu_ps(ap.add(i));
            let vb = _mm256_loadu_ps(bp.add(i));
            let d = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        i += 8;
    }

    let mut sum = unsafe { hsum_avx2(acc) };
    while i < len {
        unsafe {
            let d = *ap.add(i) - *bp.add(i);
            sum += d * d;
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    while i + 8 <= len {
        unsafe {
            let va = _mm256_loadu_ps(ap.add(i));
            let vb = _mm256_loadu_ps(bp.add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        i += 8;
    }

    let mut sum = unsafe { hsum_avx2(acc) };
    while i < len {
        unsafe {
            sum += *ap.add(i) * *bp.add(i);
        }
        i += 1;
    }
    sum
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn l2_squared_bytes_avx512(a: &[f32], b_bytes: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut i = 0usize;
    let mut acc = _mm512_setzero_ps();
    let ap = a.as_ptr();
    let bp = b_bytes.as_ptr() as *const f32;

    while i + 16 <= len {
        unsafe {
            let va = _mm512_loadu_ps(ap.add(i));
            let vb = _mm512_loadu_ps(bp.add(i));
            let d = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(d, d, acc);
        }
        i += 16;
    }

    let mut sum = _mm512_reduce_add_ps(acc);
    while i < len {
        let y = f32::from_le_bytes(b_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        let d = unsafe { *ap.add(i) } - y;
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_bytes_avx512(a: &[f32], b_bytes: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut i = 0usize;
    let mut acc = _mm512_setzero_ps();
    let ap = a.as_ptr();
    let bp = b_bytes.as_ptr() as *const f32;

    while i + 16 <= len {
        unsafe {
            let va = _mm512_loadu_ps(ap.add(i));
            let vb = _mm512_loadu_ps(bp.add(i));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        i += 16;
    }

    let mut sum = _mm512_reduce_add_ps(acc);
    while i < len {
        let y = f32::from_le_bytes(b_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        sum += unsafe { *ap.add(i) } * y;
        i += 1;
    }
    sum
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_bytes_avx2(a: &[f32], b_bytes: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    let ap = a.as_ptr();
    let bp = b_bytes.as_ptr() as *const f32;

    while i + 8 <= len {
        unsafe {
            let va = _mm256_loadu_ps(ap.add(i));
            let vb = _mm256_loadu_ps(bp.add(i));
            let d = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        i += 8;
    }

    let mut sum = unsafe { hsum_avx2(acc) };
    while i < len {
        let y = f32::from_le_bytes(b_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        let d = unsafe { *ap.add(i) } - y;
        sum += d * d;
        i += 1;
    }
    sum
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_bytes_avx2(a: &[f32], b_bytes: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    let ap = a.as_ptr();
    let bp = b_bytes.as_ptr() as *const f32;

    while i + 8 <= len {
        unsafe {
            let va = _mm256_loadu_ps(ap.add(i));
            let vb = _mm256_loadu_ps(bp.add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        i += 8;
    }

    let mut sum = unsafe { hsum_avx2(acc) };
    while i < len {
        let y = f32::from_le_bytes(b_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        sum += unsafe { *ap.add(i) } * y;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f,avx512vpopcntdq,avx2,fma")]
unsafe fn score_leaf_avx512<const METRIC: u8>(
    params: &RabitqLeafScoreParams<'_>,
    out: &mut Vec<LeafCandidate>,
) {
    use std::arch::x86_64::*;

    let n = params.entry_count;
    let words = params.rabitq_words;
    out.reserve(n);

    let v_vl = _mm256_set1_ps(params.vl);
    let v_delta = _mm256_set1_ps(params.delta);
    let v_two = _mm256_set1_ps(2.0);
    let v_sum_q = _mm256_set1_ps(params.sum_q);
    let v_inv_sqrt = _mm256_set1_ps(params.inv_sqrt_padded_dim);
    let v_q_res_sq = _mm256_set1_ps(params.q_res_norm_sq);
    let v_half = _mm256_set1_ps(0.5);
    let v_zero = _mm256_setzero_ps();
    let v_zero_res_dist = _mm256_set1_ps(params.zero_res_dist);

    let codes_ptr = params.codes_bytes.as_ptr();
    let res_sq_ptr = params.res_norm_sqs_bytes.as_ptr() as *const f32;
    let scales_ptr = params.scales_bytes.as_ptr() as *const f32;
    let doc_ids_ptr = params.doc_ids_bytes.as_ptr();

    let mut i = 0usize;
    while i + 8 <= n {
        let mut acc_x = _mm512_setzero_si512();
        let mut acc_0 = _mm512_setzero_si512();
        let mut acc_1 = _mm512_setzero_si512();
        let mut acc_2 = _mm512_setzero_si512();
        let mut acc_3 = _mm512_setzero_si512();

        for w in 0..words {
            unsafe {
                let x = _mm512_loadu_si512(codes_ptr.add((w * n + i) * 8) as *const __m512i);
                let q0w = _mm512_set1_epi64(*params.q0.get_unchecked(w) as i64);
                let q1w = _mm512_set1_epi64(*params.q1.get_unchecked(w) as i64);
                let q2w = _mm512_set1_epi64(*params.q2.get_unchecked(w) as i64);
                let q3w = _mm512_set1_epi64(*params.q3.get_unchecked(w) as i64);

                acc_x = _mm512_add_epi64(acc_x, _mm512_popcnt_epi64(x));
                acc_0 = _mm512_add_epi64(acc_0, _mm512_popcnt_epi64(_mm512_and_si512(x, q0w)));
                acc_1 = _mm512_add_epi64(acc_1, _mm512_popcnt_epi64(_mm512_and_si512(x, q1w)));
                acc_2 = _mm512_add_epi64(acc_2, _mm512_popcnt_epi64(_mm512_and_si512(x, q2w)));
                acc_3 = _mm512_add_epi64(acc_3, _mm512_popcnt_epi64(_mm512_and_si512(x, q3w)));
            }
        }

        let w1 = _mm512_slli_epi64(acc_1, 1);
        let w2 = _mm512_slli_epi64(acc_2, 2);
        let w3 = _mm512_slli_epi64(acc_3, 3);
        let weighted_i64 = _mm512_add_epi64(_mm512_add_epi64(acc_0, w1), _mm512_add_epi64(w2, w3));

        let px_f = _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(acc_x));
        let wp_f = _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(weighted_i64));

        let sum_x_q = _mm256_fmadd_ps(v_delta, wp_f, _mm256_mul_ps(v_vl, px_f));
        let o_bar_dot_q = _mm256_mul_ps(_mm256_fmsub_ps(v_two, sum_x_q, v_sum_q), v_inv_sqrt);

        let v_scale = unsafe { _mm256_loadu_ps(scales_ptr.add(i)) };
        let v_res_sq = unsafe { _mm256_loadu_ps(res_sq_ptr.add(i)) };
        let est_r_dot_q = _mm256_mul_ps(v_scale, o_bar_dot_q);

        let raw_dist = match METRIC {
            0 => {
                // max(0, res_sq + q_res_sq - 2 * est_r_dot_q)
                let d = _mm256_fnmadd_ps(v_two, est_r_dot_q, _mm256_add_ps(v_res_sq, v_q_res_sq));
                _mm256_max_ps(v_zero, d)
            }
            1 => {
                let d = _mm256_fnmadd_ps(v_two, est_r_dot_q, _mm256_add_ps(v_res_sq, v_q_res_sq));
                _mm256_max_ps(v_zero, _mm256_mul_ps(v_half, d))
            }
            _ => _mm256_sub_ps(v_zero_res_dist, est_r_dot_q),
        };

        let zero_mask = _mm256_cmp_ps(v_res_sq, v_zero, _CMP_EQ_OQ);
        let final_dist = _mm256_blendv_ps(raw_dist, v_zero_res_dist, zero_mask);

        let mut dists = [0.0f32; 8];
        let mut doc_ids = [0u64; 8];
        unsafe {
            _mm256_storeu_ps(dists.as_mut_ptr(), final_dist);
            std::ptr::copy_nonoverlapping(doc_ids_ptr.add(i * 8), doc_ids.as_mut_ptr() as *mut u8, 64);
        }

        for lane in 0..8 {
            out.push(LeafCandidate {
                doc_id: doc_ids[lane],
                dist: dists[lane],
            });
        }

        i += 8;
    }

    // Scalar tail for remaining `< 8` entries
    while i < n {
        out.push(score_single_entry_scalar::<METRIC>(params, i));
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::prepare_vector;
    use crate::node::PackedLeafNode;
    use crate::rabitq::{
        EncodeScratch, QueryScratch, RabitqQuantizer, pack_query_bits_from_diff_bytes_into,
        pack_query_bits_into,
    };
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_simd_leaf_scoring_matches_scalar_reference() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dim = 128;
        let qtz = RabitqQuantizer::new(dim, 777);

        for metric in [
            DistanceMetric::L2Squared,
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
        ] {
            let raw_c: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let c = prepare_vector(&raw_c, dim, metric).unwrap();
            let raw_q: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let q = prepare_vector(&raw_q, dim, metric).unwrap();

            let mut leaf = PackedLeafNode::empty(None, dim, qtz.padded_dim());
            leaf.centroid = c.clone();
            qtz.rotate_into(&c, &mut leaf.rotated_centroid);

            let mut scratch_enc = EncodeScratch::default();
            let mut entries = Vec::new();
            // 19 entries: 2 full 8-wide AVX-512 blocks + 3 scalar tail entries, including a zero-residual entry.
            for idx in 0..19u64 {
                let x = if idx == 5 {
                    c.clone()
                } else {
                    let raw_x: Vec<f32> =
                        c.iter().map(|&v| v + rng.random_range(-0.3..0.3)).collect();
                    prepare_vector(&raw_x, dim, metric).unwrap()
                };
                let mut code = vec![0u64; qtz.num_words()];
                let (res_norm_sq, scale) = qtz.encode_into(&x, &c, &mut scratch_enc, &mut code);
                leaf.upsert_entry(idx * 10 + 1, res_norm_sq, scale, &code, qtz.num_words());
                entries.push((idx * 10 + 1, res_norm_sq, scale, code));
            }

            let blob = leaf.encode(dim, qtz.padded_dim(), qtz.num_words());
            let view =
                crate::node::LeafNodeView::parse(&blob, dim, qtz.padded_dim(), qtz.num_words())
                    .unwrap();

            let mut q_scratch = QueryScratch::new(qtz.padded_dim(), qtz.num_words());
            qtz.rotate_into(&q, &mut q_scratch.rot_q);

            let (q_res_norm_sq, zero_res_dist, q_meta) = match metric {
                DistanceMetric::L2Squared => {
                    let d2 = l2_squared(&q, &c);
                    let meta = pack_query_bits_from_diff_bytes_into(
                        &q_scratch.rot_q,
                        view.rotated_centroid_bytes,
                        &mut q_scratch.rot_diff,
                        &mut q_scratch.planes,
                    );
                    (d2, d2, meta)
                }
                DistanceMetric::Cosine => {
                    let d2 = l2_squared(&q, &c);
                    let zero = (1.0 - dot_product(&q, &c)).max(0.0);
                    let meta = pack_query_bits_from_diff_bytes_into(
                        &q_scratch.rot_q,
                        view.rotated_centroid_bytes,
                        &mut q_scratch.rot_diff,
                        &mut q_scratch.planes,
                    );
                    (d2, zero, meta)
                }
                DistanceMetric::DotProduct => {
                    let zero = -dot_product(&q, &c);
                    let meta = pack_query_bits_into(&q_scratch.rot_q, &mut q_scratch.planes);
                    (0.0, zero, meta)
                }
            };

            let mut simd_out = Vec::new();
            score_leaf_rabitq(
                &RabitqLeafScoreParams {
                    entry_count: view.entry_count,
                    rabitq_words: qtz.num_words(),
                    doc_ids_bytes: view.doc_ids_bytes,
                    codes_bytes: view.codes_bytes,
                    res_norm_sqs_bytes: view.res_norm_sqs_bytes,
                    scales_bytes: view.scales_bytes,
                    q0: &q_scratch.planes[0],
                    q1: &q_scratch.planes[1],
                    q2: &q_scratch.planes[2],
                    q3: &q_scratch.planes[3],
                    vl: q_meta.vl,
                    delta: q_meta.delta,
                    sum_q: q_meta.sum_q,
                    inv_sqrt_padded_dim: qtz.inv_sqrt_padded_dim(),
                    q_res_norm_sq,
                    zero_res_dist,
                    metric,
                },
                &mut simd_out,
            );

            let dot_cache = if metric == DistanceMetric::DotProduct {
                Some(qtz.prepare_dot_query(&q))
            } else {
                None
            };
            let ref_state = qtz.prepare_leaf_query(&q, &c, metric, dot_cache.as_ref());

            assert_eq!(simd_out.len(), entries.len());
            for (cand, &(doc_id, res_norm_sq, scale, ref code)) in
                simd_out.iter().zip(entries.iter())
            {
                assert_eq!(cand.doc_id, doc_id);
                let ref_dist = ref_state.estimate_distance(res_norm_sq, scale, code);
                assert!(
                    (cand.dist - ref_dist).abs() < 1e-3,
                    "metric={metric:?}, doc={doc_id}: simd={} vs ref={ref_dist}",
                    cand.dist
                );
            }
        }
    }
}
