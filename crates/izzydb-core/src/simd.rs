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
    #[cfg(target_arch = "aarch64")]
    {
        if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
            // Safety: checked `neon` at compile/runtime; slices have equal length.
            return unsafe { l2_squared_neon(a, b) };
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
    #[cfg(target_arch = "aarch64")]
    {
        if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
            // Safety: checked `neon` at compile/runtime; slices have equal length.
            return unsafe { dot_product_neon(a, b) };
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
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
            // Safety: checked `neon` at compile/runtime; `b_bytes.len() == a.len() * 4` and unaligned `vld1q_u8` loads are used.
            return unsafe { l2_squared_bytes_neon(a, b_bytes) };
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
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
            // Safety: checked `neon` at compile/runtime; `b_bytes.len() == a.len() * 4` and unaligned `vld1q_u8` loads are used.
            return unsafe { dot_product_bytes_neon(a, b_bytes) };
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe { neon_reduce_f32::<true, false>(a, b.as_ptr() as *const u8) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    unsafe { neon_reduce_f32::<false, false>(a, b.as_ptr() as *const u8) }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn l2_squared_bytes_neon(a: &[f32], b_bytes: &[u8]) -> f32 {
    unsafe { neon_reduce_f32::<true, true>(a, b_bytes.as_ptr()) }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_product_bytes_neon(a: &[f32], b_bytes: &[u8]) -> f32 {
    unsafe { neon_reduce_f32::<false, true>(a, b_bytes.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_reduce_f32<const IS_L2: bool, const UNALIGNED_BYTES: bool>(
    a: &[f32],
    b_ptr: *const u8,
) -> f32 {
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn load_b<const UNALIGNED: bool>(b_ptr: *const u8, idx: usize) -> float32x4_t {
        unsafe {
            if UNALIGNED {
                vreinterpretq_f32_u8(vld1q_u8(b_ptr.add(idx * 4)))
            } else {
                vld1q_f32((b_ptr as *const f32).add(idx))
            }
        }
    }

    #[inline(always)]
    unsafe fn step<const L2: bool>(
        acc: float32x4_t,
        va: float32x4_t,
        vb: float32x4_t,
    ) -> float32x4_t {
        unsafe {
            if L2 {
                let d = vsubq_f32(va, vb);
                vfmaq_f32(acc, d, d)
            } else {
                vfmaq_f32(acc, va, vb)
            }
        }
    }

    let len = a.len();
    let ap = a.as_ptr();
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    while i + 16 <= len {
        unsafe {
            acc0 = step::<IS_L2>(acc0, vld1q_f32(ap.add(i)), load_b::<UNALIGNED_BYTES>(b_ptr, i));
            acc1 = step::<IS_L2>(
                acc1,
                vld1q_f32(ap.add(i + 4)),
                load_b::<UNALIGNED_BYTES>(b_ptr, i + 4),
            );
            acc2 = step::<IS_L2>(
                acc2,
                vld1q_f32(ap.add(i + 8)),
                load_b::<UNALIGNED_BYTES>(b_ptr, i + 8),
            );
            acc3 = step::<IS_L2>(
                acc3,
                vld1q_f32(ap.add(i + 12)),
                load_b::<UNALIGNED_BYTES>(b_ptr, i + 12),
            );
        }
        i += 16;
    }

    let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    while i + 4 <= len {
        unsafe {
            acc = step::<IS_L2>(acc, vld1q_f32(ap.add(i)), load_b::<UNALIGNED_BYTES>(b_ptr, i));
        }
        i += 4;
    }

    let mut sum = vaddvq_f32(acc);
    while i < len {
        let x = unsafe { *ap.add(i) };
        let y = if UNALIGNED_BYTES {
            let bytes = unsafe { *(b_ptr.add(i * 4) as *const [u8; 4]) };
            f32::from_le_bytes(bytes)
        } else {
            unsafe { *(b_ptr as *const f32).add(i) }
        };
        if IS_L2 {
            let d = x - y;
            sum += d * d;
        } else {
            sum += x * y;
        }
        i += 1;
    }
    sum
}

/// Evaluate `RankBy::Saturate` across up to 256 contiguous values (`present_mask` bit indicates
/// whether each slot holds a finite numeric value; absent slots write `missing`).
#[inline]
pub fn batch_eval_saturate(
    vals: &[f64],
    present_mask: &[u64; 4],
    midpoint: f64,
    exponent: f64,
    missing: f32,
    out: &mut [f32],
) {
    let n = vals.len().min(out.len()).min(256);
    if exponent == 1.0 {
        for (i, (&x, dst)) in vals[..n].iter().zip(&mut out[..n]).enumerate() {
            let present = ((present_mask[i >> 6] >> (i & 63)) & 1) != 0;
            *dst = if !present {
                missing
            } else {
                crate::rank::eval_saturate(x, midpoint, 1.0)
            };
        }
    } else {
        for (i, (&x, dst)) in vals[..n].iter().zip(&mut out[..n]).enumerate() {
            let present = ((present_mask[i >> 6] >> (i & 63)) & 1) != 0;
            *dst = if !present {
                missing
            } else {
                crate::rank::eval_saturate(x, midpoint, exponent)
            };
        }
    }
}

/// Evaluate `RankBy::Decay` across up to 256 contiguous values (`present_mask` bit indicates
/// whether each slot holds a finite numeric value; absent slots write `missing`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn batch_eval_decay(
    vals: &[f64],
    present_mask: &[u64; 4],
    origin: f64,
    midpoint: f64,
    exponent: f64,
    mode: crate::rank::DecayMode,
    missing: f32,
    out: &mut [f32],
) {
    let n = vals.len().min(out.len()).min(256);
    for (i, (&x, dst)) in vals[..n].iter().zip(&mut out[..n]).enumerate() {
        let present = ((present_mask[i >> 6] >> (i & 63)) & 1) != 0;
        *dst = if !present {
            missing
        } else {
            crate::rank::eval_decay(x, origin, midpoint, exponent, mode)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_f32(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((*state >> 32) as u32) as f32 / (u32::MAX as f32);
        bits * 2.0 - 1.0
    }

    #[test]
    fn test_simd_distances_match_scalar_across_lengths_and_unaligned_offsets() {
        let mut rng = 0x1234_5678_9abc_def0u64;
        for len in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 64, 128, 384, 1536] {
            let a: Vec<f32> = (0..len).map(|_| next_f32(&mut rng)).collect();
            let b: Vec<f32> = (0..len).map(|_| next_f32(&mut rng)).collect();

            let ref_l2 = l2_squared_scalar(&a, &b);
            let ref_dot = dot_product_scalar(&a, &b);

            let got_l2 = l2_squared(&a, &b);
            let got_dot = dot_product(&a, &b);

            let tol = 1e-4f32.max(ref_l2.abs() * 1e-5);
            assert!(
                (got_l2 - ref_l2).abs() <= tol,
                "l2_squared len={len}: got={got_l2} vs ref={ref_l2}"
            );
            let dot_tol = 1e-4f32.max(ref_dot.abs() * 1e-5);
            assert!(
                (got_dot - ref_dot).abs() <= dot_tol,
                "dot_product len={len}: got={got_dot} vs ref={ref_dot}"
            );

            // Test unaligned byte offsets (0..=3 byte shift)
            let b_raw: Vec<u8> = b.iter().flat_map(|x| x.to_le_bytes()).collect();
            for offset in 0..4 {
                let mut padded = vec![0xA5u8; offset + b_raw.len() + 4];
                padded[offset..offset + b_raw.len()].copy_from_slice(&b_raw);
                let b_bytes = &padded[offset..offset + b_raw.len()];

                let got_l2_b = l2_squared_bytes(&a, b_bytes);
                let got_dot_b = dot_product_bytes(&a, b_bytes);
                assert!(
                    (got_l2_b - ref_l2).abs() <= tol,
                    "l2_squared_bytes len={len} offset={offset}: got={got_l2_b} vs ref={ref_l2}"
                );
                assert!(
                    (got_dot_b - ref_dot).abs() <= dot_tol,
                    "dot_product_bytes len={len} offset={offset}: got={got_dot_b} vs ref={ref_dot}"
                );
            }
        }
    }
}
