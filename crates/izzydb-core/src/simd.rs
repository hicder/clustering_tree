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
