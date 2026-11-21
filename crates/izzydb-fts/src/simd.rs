use izzydb_core::config::FTS_BLOCK_SIZE;

use crate::block::{IntWidth, POSTING_DENSE_THRESHOLD, read_tf_at};

/// Runtime-detected SIMD execution tier for 256-doc FTS block kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdTier {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Scalar,
}

impl SimdTier {
    #[inline]
    pub fn detect() -> Self {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Self::Avx2;
            }
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        {
            if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
                return Self::Neon;
            }
        }
        Self::Scalar
    }
}

/// Compute `dl_norm[s] = max(c1 + c2 * dl[s], f32::MIN_POSITIVE)` for all 256 slots.
#[inline]
pub fn prepare_dl_norm_256(
    tier: SimdTier,
    width: IntWidth,
    dls_bytes: &[u8],
    c1: f32,
    c2: f32,
    out: &mut [f32; FTS_BLOCK_SIZE],
) {
    debug_assert_eq!(dls_bytes.len(), FTS_BLOCK_SIZE * width.byte_size());
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe { prepare_dl_norm_256_avx512(width, dls_bytes, c1, c2, out) },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { prepare_dl_norm_256_avx2(width, dls_bytes, c1, c2, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe { prepare_dl_norm_256_neon(width, dls_bytes, c1, c2, out) },
        _ => prepare_dl_norm_256_scalar(width, dls_bytes, c1, c2, out),
    }
}

/// Accumulate BM25 scores for a dense 256-slot TF block into `scores` and OR `bitmap` into `touched`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn accumulate_dense_256(
    tier: SimdTier,
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    bitmap: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    touched: &mut [u64; 4],
) {
    debug_assert_eq!(tfs_bytes.len(), FTS_BLOCK_SIZE * width.byte_size());
    if num_scale <= 0.0 {
        return;
    }
    for w in 0..4 {
        touched[w] |= bitmap[w];
    }
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe {
            accumulate_dense_256_avx512(width, tfs_bytes, num_scale, dl_norm, bitmap, scores)
        },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe {
            accumulate_dense_256_avx2(width, tfs_bytes, num_scale, dl_norm, bitmap, scores)
        },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe {
            accumulate_dense_256_neon(width, tfs_bytes, num_scale, dl_norm, bitmap, scores)
        },
        _ => accumulate_dense_256_scalar(width, tfs_bytes, num_scale, dl_norm, bitmap, scores),
    }
}

/// Accumulate BM25 scores for a dense 256-slot TF block only on slots where `mask` is set.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn accumulate_dense_masked_256(
    tier: SimdTier,
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    bitmap: &[u64; 4],
    mask: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    debug_assert_eq!(tfs_bytes.len(), FTS_BLOCK_SIZE * width.byte_size());
    if num_scale <= 0.0 {
        return;
    }
    let active = [
        bitmap[0] & mask[0],
        bitmap[1] & mask[1],
        bitmap[2] & mask[2],
        bitmap[3] & mask[3],
    ];
    let pop: u32 = active.iter().map(|w| w.count_ones()).sum();
    if pop == 0 {
        return;
    }
    if pop <= 24 {
        accumulate_active_bits_scalar(width, tfs_bytes, num_scale, dl_norm, &active, scores);
        return;
    }
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe {
            accumulate_dense_masked_avx512(width, tfs_bytes, num_scale, dl_norm, &active, scores)
        },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe {
            accumulate_dense_masked_avx2(width, tfs_bytes, num_scale, dl_norm, &active, scores)
        },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe {
            accumulate_dense_masked_neon(width, tfs_bytes, num_scale, dl_norm, &active, scores)
        },
        _ => accumulate_active_bits_scalar(width, tfs_bytes, num_scale, dl_norm, &active, scores),
    }
}

/// Accumulate BM25 scores for a sparse posting block (`slots.len() < POSTING_DENSE_THRESHOLD`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn accumulate_sparse(
    tier: SimdTier,
    width: IntWidth,
    slots: &[u8],
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    touched: &mut [u64; 4],
) {
    debug_assert_eq!(tfs_bytes.len(), slots.len() * width.byte_size());
    if num_scale <= 0.0 || slots.is_empty() {
        return;
    }
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 if slots.len() >= 16 => unsafe {
            accumulate_sparse_avx512(width, slots, tfs_bytes, num_scale, dl_norm, scores, touched)
        },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 | SimdTier::Avx512 if slots.len() >= 8 => unsafe {
            accumulate_sparse_avx2(width, slots, tfs_bytes, num_scale, dl_norm, scores, touched)
        },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon if slots.len() >= 4 => unsafe {
            accumulate_sparse_neon(width, slots, tfs_bytes, num_scale, dl_norm, scores, touched)
        },
        _ => accumulate_sparse_scalar(width, slots, tfs_bytes, num_scale, dl_norm, scores, touched),
    }
}

#[inline]
fn mask_pop_le_8(mask: &[u64; 4]) -> bool {
    let mut sum = mask[0].count_ones() + mask[1].count_ones();
    if sum > 8 {
        return false;
    }
    sum += mask[2].count_ones() + mask[3].count_ones();
    sum <= 8
}

/// Accumulate BM25 scores for a sparse posting block only on slots where `mask` is set.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn accumulate_sparse_masked(
    tier: SimdTier,
    width: IntWidth,
    slots: &[u8],
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    mask: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    debug_assert_eq!(tfs_bytes.len(), slots.len() * width.byte_size());
    if num_scale <= 0.0 || slots.is_empty() {
        return;
    }
    if slots.len() < 8 || slots.len() > POSTING_DENSE_THRESHOLD || mask_pop_le_8(mask) {
        match width {
            IntWidth::U8 => {
                for (i, &slot) in slots.iter().enumerate() {
                    if ((mask[(slot >> 6) as usize] >> (slot & 63)) & 1) != 0 {
                        let tf_f = tfs_bytes[i] as f32;
                        let idx = slot as usize;
                        scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                    }
                }
            }
            IntWidth::U16 => {
                for (i, &slot) in slots.iter().enumerate() {
                    if ((mask[(slot >> 6) as usize] >> (slot & 63)) & 1) != 0 {
                        let off = i * 2;
                        let tf_f =
                            u16::from_le_bytes([tfs_bytes[off], tfs_bytes[off + 1]]) as f32;
                        let idx = slot as usize;
                        scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                    }
                }
            }
            IntWidth::U32 => {
                for (i, &slot) in slots.iter().enumerate() {
                    if ((mask[(slot >> 6) as usize] >> (slot & 63)) & 1) != 0 {
                        let off = i * 4;
                        let tf_f = u32::from_le_bytes(
                            tfs_bytes[off..off + 4].try_into().unwrap(),
                        ) as f32;
                        let idx = slot as usize;
                        scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                    }
                }
            }
        }
        return;
    }
    let mut matched_slots = [0u8; POSTING_DENSE_THRESHOLD];
    let mut matched_tfs = [0u32; POSTING_DENSE_THRESHOLD];
    let mut m = 0usize;
    for (i, &slot) in slots.iter().enumerate() {
        if ((mask[(slot >> 6) as usize] >> (slot & 63)) & 1) != 0 {
            matched_slots[m] = slot;
            matched_tfs[m] = read_tf_at(width, tfs_bytes, i);
            m += 1;
        }
    }
    if m == 0 {
        return;
    }
    accumulate_u32_pairs(
        tier,
        &matched_slots[..m],
        &matched_tfs[..m],
        num_scale,
        dl_norm,
        scores,
    );
}

/// Return the maximum `scores[slot]` among slots where `mask` bit is 1 (0.0 if empty).
#[inline]
pub fn max_score_256(tier: SimdTier, scores: &[f32; FTS_BLOCK_SIZE], mask: &[u64; 4]) -> f32 {
    if *mask == [0u64; 4] {
        return 0.0;
    }
    if mask_pop_le_8(mask) {
        return max_score_256_scalar(scores, mask);
    }
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe { max_score_256_avx512(scores, mask) },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { max_score_256_avx2(scores, mask) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe { max_score_256_neon(scores, mask) },
        _ => max_score_256_scalar(scores, mask),
    }
}

/// Return a 256-bit mask of all slots in `mask` satisfying `scores[slot] >= threshold`.
#[inline]
pub fn filter_ge_256(
    tier: SimdTier,
    scores: &[f32; FTS_BLOCK_SIZE],
    mask: &[u64; 4],
    threshold: f32,
) -> [u64; 4] {
    if *mask == [0u64; 4] {
        return [0u64; 4];
    }
    if mask_pop_le_8(mask) {
        return filter_ge_256_scalar(scores, mask, threshold);
    }
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe { filter_ge_256_avx512(scores, mask, threshold) },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { filter_ge_256_avx2(scores, mask, threshold) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe { filter_ge_256_neon(scores, mask, threshold) },
        _ => filter_ge_256_scalar(scores, mask, threshold),
    }
}

/// Elementwise `dst[i] = dst[i].max(src[i])` across 256 floats.
#[inline]
pub fn merge_max_256(
    tier: SimdTier,
    dst: &mut [f32; FTS_BLOCK_SIZE],
    src: &[f32; FTS_BLOCK_SIZE],
) {
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe { merge_max_256_avx512(dst, src) },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { merge_max_256_avx2(dst, src) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe { merge_max_256_neon(dst, src) },
        _ => merge_max_256_scalar(dst, src),
    }
}

/// Elementwise `dst[i] += src[i]` across 256 floats.
#[inline]
pub fn merge_add_256(
    tier: SimdTier,
    dst: &mut [f32; FTS_BLOCK_SIZE],
    src: &[f32; FTS_BLOCK_SIZE],
) {
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe { merge_add_256_avx512(dst, src) },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { merge_add_256_avx2(dst, src) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe { merge_add_256_neon(dst, src) },
        _ => merge_add_256_scalar(dst, src),
    }
}

// ============================================================================
// Scalar implementations
// ============================================================================

#[inline]
fn read_dl_at(width: IntWidth, dls_bytes: &[u8], idx: usize) -> u32 {
    match width {
        IntWidth::U8 => dls_bytes[idx] as u32,
        IntWidth::U16 => {
            let off = idx * 2;
            u16::from_le_bytes([dls_bytes[off], dls_bytes[off + 1]]) as u32
        }
        IntWidth::U32 => {
            let off = idx * 4;
            u32::from_le_bytes(dls_bytes[off..off + 4].try_into().unwrap())
        }
    }
}

#[inline]
fn prepare_dl_norm_256_scalar(
    width: IntWidth,
    dls_bytes: &[u8],
    c1: f32,
    c2: f32,
    out: &mut [f32; FTS_BLOCK_SIZE],
) {
    for (i, dst) in out.iter_mut().enumerate() {
        let dl_f = read_dl_at(width, dls_bytes, i) as f32;
        *dst = c2.mul_add(dl_f, c1).max(f32::MIN_POSITIVE);
    }
}

#[inline]
fn accumulate_dense_256_scalar(
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    bitmap: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    accumulate_active_bits_scalar(width, tfs_bytes, num_scale, dl_norm, bitmap, scores);
}

#[inline]
fn accumulate_active_bits_scalar(
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    active: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    for (w, &bits) in active.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let bit = b.trailing_zeros() as usize;
            b &= b - 1;
            let slot = (w << 6) | bit;
            let tf_f = read_tf_at(width, tfs_bytes, slot) as f32;
            scores[slot] += (num_scale * tf_f) / (tf_f + dl_norm[slot]);
        }
    }
}

#[inline]
fn accumulate_sparse_scalar(
    width: IntWidth,
    slots: &[u8],
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    touched: &mut [u64; 4],
) {
    match width {
        IntWidth::U8 => {
            for (i, &slot) in slots.iter().enumerate() {
                let tf_f = tfs_bytes[i] as f32;
                let idx = slot as usize;
                scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        }
        IntWidth::U16 => {
            for (i, &slot) in slots.iter().enumerate() {
                let off = i * 2;
                let tf_f = u16::from_le_bytes([tfs_bytes[off], tfs_bytes[off + 1]]) as f32;
                let idx = slot as usize;
                scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        }
        IntWidth::U32 => {
            for (i, &slot) in slots.iter().enumerate() {
                let off = i * 4;
                let tf_f = u32::from_le_bytes(tfs_bytes[off..off + 4].try_into().unwrap()) as f32;
                let idx = slot as usize;
                scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        }
    }
}

#[inline]
fn accumulate_u32_pairs(
    tier: SimdTier,
    slots: &[u8],
    tfs: &[u32],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    let n = slots.len();
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if tier == SimdTier::Avx512 && n >= 16 {
        unsafe {
            use std::arch::x86_64::*;
            let v_scale = _mm512_set1_ps(num_scale);
            while i + 16 <= n {
                let mut dln_buf = [0.0f32; 16];
                for lane in 0..16 {
                    dln_buf[lane] = dl_norm[slots[i + lane] as usize];
                }
                let v_tf = _mm512_cvtepu32_ps(_mm512_loadu_si512(
                    tfs.as_ptr().add(i) as *const __m512i
                ));
                let v_dln = _mm512_loadu_ps(dln_buf.as_ptr());
                let v_res = _mm512_div_ps(_mm512_mul_ps(v_scale, v_tf), _mm512_add_ps(v_tf, v_dln));
                let mut res = [0.0f32; 16];
                _mm512_storeu_ps(res.as_mut_ptr(), v_res);
                for lane in 0..16 {
                    scores[slots[i + lane] as usize] += res[lane];
                }
                i += 16;
            }
        }
    }
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if matches!(tier, SimdTier::Avx2 | SimdTier::Avx512) && i + 8 <= n {
        let all_i32 = tfs[i..].iter().all(|&v| v <= i32::MAX as u32);
        if all_i32 {
            unsafe {
                use std::arch::x86_64::*;
                let v_scale = _mm256_set1_ps(num_scale);
                while i + 8 <= n {
                    let mut dln_buf = [0.0f32; 8];
                    for lane in 0..8 {
                        dln_buf[lane] = dl_norm[slots[i + lane] as usize];
                    }
                    let v_tf = _mm256_cvtepi32_ps(_mm256_loadu_si256(
                        tfs.as_ptr().add(i) as *const __m256i
                    ));
                    let v_dln = _mm256_loadu_ps(dln_buf.as_ptr());
                    let v_res =
                        _mm256_div_ps(_mm256_mul_ps(v_scale, v_tf), _mm256_add_ps(v_tf, v_dln));
                    let mut res = [0.0f32; 8];
                    _mm256_storeu_ps(res.as_mut_ptr(), v_res);
                    for lane in 0..8 {
                        scores[slots[i + lane] as usize] += res[lane];
                    }
                    i += 8;
                }
            }
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if tier == SimdTier::Neon && i + 4 <= n {
        unsafe {
            use std::arch::aarch64::*;
            let v_scale = vdupq_n_f32(num_scale);
            while i + 4 <= n {
                let mut dln_buf = [0.0f32; 4];
                for lane in 0..4 {
                    dln_buf[lane] = dl_norm[slots[i + lane] as usize];
                }
                let v_tf = vcvtq_f32_u32(vld1q_u32(tfs.as_ptr().add(i)));
                let v_dln = vld1q_f32(dln_buf.as_ptr());
                let v_res = vdivq_f32(vmulq_f32(v_scale, v_tf), vaddq_f32(v_tf, v_dln));
                let mut res = [0.0f32; 4];
                vst1q_f32(res.as_mut_ptr(), v_res);
                for lane in 0..4 {
                    scores[slots[i + lane] as usize] += res[lane];
                }
                i += 4;
            }
        }
    }
    let _ = tier;
    while i < n {
        let tf_f = tfs[i] as f32;
        let idx = slots[i] as usize;
        scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
        i += 1;
    }
}

#[inline]
fn max_score_256_scalar(scores: &[f32; FTS_BLOCK_SIZE], mask: &[u64; 4]) -> f32 {
    let mut best = 0.0f32;
    for (w, &bits) in mask.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let bit = b.trailing_zeros() as usize;
            b &= b - 1;
            let s = scores[(w << 6) | bit];
            if s > best {
                best = s;
            }
        }
    }
    best
}

#[inline]
fn filter_ge_256_scalar(
    scores: &[f32; FTS_BLOCK_SIZE],
    mask: &[u64; 4],
    threshold: f32,
) -> [u64; 4] {
    let mut out = [0u64; 4];
    for (w, &bits) in mask.iter().enumerate() {
        let mut b = bits;
        let mut keep = 0u64;
        while b != 0 {
            let bit = b.trailing_zeros() as usize;
            b &= b - 1;
            if scores[(w << 6) | bit] >= threshold {
                keep |= 1u64 << bit;
            }
        }
        out[w] = keep;
    }
    out
}

#[inline]
fn merge_max_256_scalar(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        if s > *d {
            *d = s;
        }
    }
}

#[inline]
fn merge_add_256_scalar(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

// ============================================================================
// x86_64 AVX-512F implementations
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn load_16_int_as_f32_avx512(
    width: IntWidth,
    ptr: *const u8,
    idx: usize,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    unsafe {
        match width {
            IntWidth::U8 => {
                let v = _mm_loadu_si128(ptr.add(idx) as *const __m128i);
                _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v))
            }
            IntWidth::U16 => {
                let v = _mm256_loadu_si256(ptr.add(idx * 2) as *const __m256i);
                _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(v))
            }
            IntWidth::U32 => {
                let v = _mm512_loadu_si512(ptr.add(idx * 4) as *const __m512i);
                _mm512_cvtepu32_ps(v)
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn prepare_dl_norm_256_avx512(
    width: IntWidth,
    dls_bytes: &[u8],
    c1: f32,
    c2: f32,
    out: &mut [f32; FTS_BLOCK_SIZE],
) {
    use std::arch::x86_64::*;
    let v_c1 = _mm512_set1_ps(c1);
    let v_c2 = _mm512_set1_ps(c2);
    let v_min = _mm512_set1_ps(f32::MIN_POSITIVE);
    let ptr = dls_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0usize;
    while i < FTS_BLOCK_SIZE {
        unsafe {
            let v_dl = load_16_int_as_f32_avx512(width, ptr, i);
            let v_norm = _mm512_max_ps(_mm512_fmadd_ps(v_c2, v_dl, v_c1), v_min);
            _mm512_storeu_ps(out_ptr.add(i), v_norm);
        }
        i += 16;
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_dense_256_avx512(
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    bitmap: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    use std::arch::x86_64::*;
    let v_scale = _mm512_set1_ps(num_scale);
    let tf_ptr = tfs_bytes.as_ptr();
    let dln_ptr = dl_norm.as_ptr();
    let sc_ptr = scores.as_mut_ptr();
    for (w, &bw) in bitmap.iter().enumerate() {
        if bw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..4 {
            let i = base + (chunk << 4);
            unsafe {
                let v_tf = load_16_int_as_f32_avx512(width, tf_ptr, i);
                let v_dln = _mm512_loadu_ps(dln_ptr.add(i));
                let v_cur = _mm512_loadu_ps(sc_ptr.add(i));
                let v_num = _mm512_mul_ps(v_scale, v_tf);
                let v_den = _mm512_add_ps(v_tf, v_dln);
                let v_new = _mm512_add_ps(v_cur, _mm512_div_ps(v_num, v_den));
                _mm512_storeu_ps(sc_ptr.add(i), v_new);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_dense_masked_avx512(
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    active: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    use std::arch::x86_64::*;
    let v_scale = _mm512_set1_ps(num_scale);
    let tf_ptr = tfs_bytes.as_ptr();
    let dln_ptr = dl_norm.as_ptr();
    let sc_ptr = scores.as_mut_ptr();
    for (w, &aw) in active.iter().enumerate() {
        if aw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..4 {
            let k = ((aw >> (chunk << 4)) & 0xFFFF) as u16;
            if k == 0 {
                continue;
            }
            let i = base + (chunk << 4);
            unsafe {
                let v_tf = load_16_int_as_f32_avx512(width, tf_ptr, i);
                let v_dln = _mm512_loadu_ps(dln_ptr.add(i));
                let v_cur = _mm512_loadu_ps(sc_ptr.add(i));
                let v_delta = _mm512_div_ps(_mm512_mul_ps(v_scale, v_tf), _mm512_add_ps(v_tf, v_dln));
                let v_new = _mm512_mask_add_ps(v_cur, k, v_cur, v_delta);
                _mm512_storeu_ps(sc_ptr.add(i), v_new);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_sparse_avx512(
    width: IntWidth,
    slots: &[u8],
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    touched: &mut [u64; 4],
) {
    use std::arch::x86_64::*;
    let n = slots.len();
    let v_scale = _mm512_set1_ps(num_scale);
    let tf_ptr = tfs_bytes.as_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let mut dln_buf = [0.0f32; 16];
        for lane in 0..16 {
            dln_buf[lane] = dl_norm[slots[i + lane] as usize];
        }
        let mut res = [0.0f32; 16];
        unsafe {
            let v_tf = load_16_int_as_f32_avx512(width, tf_ptr, i);
            let v_dln = _mm512_loadu_ps(dln_buf.as_ptr());
            let v_res = _mm512_div_ps(_mm512_mul_ps(v_scale, v_tf), _mm512_add_ps(v_tf, v_dln));
            _mm512_storeu_ps(res.as_mut_ptr(), v_res);
        }
        for lane in 0..16 {
            let s = slots[i + lane];
            scores[s as usize] += res[lane];
            touched[(s >> 6) as usize] |= 1u64 << (s & 63);
        }
        i += 16;
    }
    if i < n {
        accumulate_sparse_scalar(
            width,
            &slots[i..],
            &tfs_bytes[i * width.byte_size()..],
            num_scale,
            dl_norm,
            scores,
            touched,
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn max_score_256_avx512(scores: &[f32; FTS_BLOCK_SIZE], mask: &[u64; 4]) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_ps();
    let ptr = scores.as_ptr();
    for (w, &mw) in mask.iter().enumerate() {
        if mw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..4 {
            let k = ((mw >> (chunk << 4)) & 0xFFFF) as u16;
            if k == 0 {
                continue;
            }
            let v = unsafe { _mm512_maskz_mov_ps(k, _mm512_loadu_ps(ptr.add(base + (chunk << 4)))) };
            acc = _mm512_max_ps(acc, v);
        }
    }
    _mm512_reduce_max_ps(acc)
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn filter_ge_256_avx512(
    scores: &[f32; FTS_BLOCK_SIZE],
    mask: &[u64; 4],
    threshold: f32,
) -> [u64; 4] {
    use std::arch::x86_64::*;
    let vt = _mm512_set1_ps(threshold);
    let ptr = scores.as_ptr();
    let mut out = [0u64; 4];
    for (w, &mw) in mask.iter().enumerate() {
        if mw == 0 {
            continue;
        }
        let base = w << 6;
        let mut word = 0u64;
        for chunk in 0..4 {
            if ((mw >> (chunk << 4)) & 0xFFFF) == 0 {
                continue;
            }
            let vs = unsafe { _mm512_loadu_ps(ptr.add(base + (chunk << 4))) };
            let k = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(vs, vt) as u64;
            word |= k << (chunk << 4);
        }
        out[w] = word & mw;
    }
    out
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn merge_max_256_avx512(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    use std::arch::x86_64::*;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i < FTS_BLOCK_SIZE {
        unsafe {
            let vd = _mm512_loadu_ps(dp.add(i));
            let vs = _mm512_loadu_ps(sp.add(i));
            _mm512_storeu_ps(dp.add(i), _mm512_max_ps(vd, vs));
        }
        i += 16;
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f")]
unsafe fn merge_add_256_avx512(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    use std::arch::x86_64::*;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i < FTS_BLOCK_SIZE {
        unsafe {
            let vd = _mm512_loadu_ps(dp.add(i));
            let vs = _mm512_loadu_ps(sp.add(i));
            _mm512_storeu_ps(dp.add(i), _mm512_add_ps(vd, vs));
        }
        i += 16;
    }
}

// ============================================================================
// x86_64 AVX2 + FMA implementations
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_8_int_as_f32_avx2(
    width: IntWidth,
    ptr: *const u8,
    idx: usize,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    unsafe {
        match width {
            IntWidth::U8 => {
                let v = _mm_loadl_epi64(ptr.add(idx) as *const __m128i);
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v))
            }
            IntWidth::U16 => {
                let v = _mm_loadu_si128(ptr.add(idx * 2) as *const __m128i);
                _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v))
            }
            IntWidth::U32 => {
                // Only called when values fit in i32 (<= i32::MAX); u32 >= 2^31 falls back to scalar.
                let v = _mm256_loadu_si256(ptr.add(idx * 4) as *const __m256i);
                _mm256_cvtepi32_ps(v)
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[inline]
fn slice_u32_fits_i32(bytes: &[u8]) -> bool {
    bytes
        .chunks_exact(4)
        .all(|c| (c[3] & 0x80) == 0)
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn prepare_dl_norm_256_avx2(
    width: IntWidth,
    dls_bytes: &[u8],
    c1: f32,
    c2: f32,
    out: &mut [f32; FTS_BLOCK_SIZE],
) {
    if width == IntWidth::U32 && !slice_u32_fits_i32(dls_bytes) {
        prepare_dl_norm_256_scalar(width, dls_bytes, c1, c2, out);
        return;
    }
    use std::arch::x86_64::*;
    let v_c1 = _mm256_set1_ps(c1);
    let v_c2 = _mm256_set1_ps(c2);
    let v_min = _mm256_set1_ps(f32::MIN_POSITIVE);
    let ptr = dls_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0usize;
    while i < FTS_BLOCK_SIZE {
        unsafe {
            let v_dl = load_8_int_as_f32_avx2(width, ptr, i);
            let v_norm = _mm256_max_ps(_mm256_fmadd_ps(v_c2, v_dl, v_c1), v_min);
            _mm256_storeu_ps(out_ptr.add(i), v_norm);
        }
        i += 8;
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_dense_256_avx2(
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    bitmap: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    if width == IntWidth::U32 && !slice_u32_fits_i32(tfs_bytes) {
        accumulate_dense_256_scalar(width, tfs_bytes, num_scale, dl_norm, bitmap, scores);
        return;
    }
    use std::arch::x86_64::*;
    let v_scale = _mm256_set1_ps(num_scale);
    let tf_ptr = tfs_bytes.as_ptr();
    let dln_ptr = dl_norm.as_ptr();
    let sc_ptr = scores.as_mut_ptr();
    for (w, &bw) in bitmap.iter().enumerate() {
        if bw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..8 {
            let i = base + (chunk << 3);
            unsafe {
                let v_tf = load_8_int_as_f32_avx2(width, tf_ptr, i);
                let v_dln = _mm256_loadu_ps(dln_ptr.add(i));
                let v_cur = _mm256_loadu_ps(sc_ptr.add(i));
                let v_new = _mm256_add_ps(
                    v_cur,
                    _mm256_div_ps(_mm256_mul_ps(v_scale, v_tf), _mm256_add_ps(v_tf, v_dln)),
                );
                _mm256_storeu_ps(sc_ptr.add(i), v_new);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_dense_masked_avx2(
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    active: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    if width == IntWidth::U32 && !slice_u32_fits_i32(tfs_bytes) {
        accumulate_active_bits_scalar(width, tfs_bytes, num_scale, dl_norm, active, scores);
        return;
    }
    use std::arch::x86_64::*;
    let v_scale = _mm256_set1_ps(num_scale);
    let bit_weights = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    let tf_ptr = tfs_bytes.as_ptr();
    let dln_ptr = dl_norm.as_ptr();
    let sc_ptr = scores.as_mut_ptr();
    for (w, &aw) in active.iter().enumerate() {
        if aw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..8 {
            let bm = ((aw >> (chunk << 3)) & 0xFF) as i32;
            if bm == 0 {
                continue;
            }
            let i = base + (chunk << 3);
            unsafe {
                let v_tf = load_8_int_as_f32_avx2(width, tf_ptr, i);
                let v_dln = _mm256_loadu_ps(dln_ptr.add(i));
                let v_cur = _mm256_loadu_ps(sc_ptr.add(i));
                let mut v_delta =
                    _mm256_div_ps(_mm256_mul_ps(v_scale, v_tf), _mm256_add_ps(v_tf, v_dln));
                if bm != 0xFF {
                    let vm = _mm256_cmpeq_epi32(
                        _mm256_and_si256(_mm256_set1_epi32(bm), bit_weights),
                        bit_weights,
                    );
                    v_delta = _mm256_and_ps(v_delta, _mm256_castsi256_ps(vm));
                }
                _mm256_storeu_ps(sc_ptr.add(i), _mm256_add_ps(v_cur, v_delta));
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_sparse_avx2(
    width: IntWidth,
    slots: &[u8],
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    touched: &mut [u64; 4],
) {
    if width == IntWidth::U32 && !slice_u32_fits_i32(tfs_bytes) {
        accumulate_sparse_scalar(width, slots, tfs_bytes, num_scale, dl_norm, scores, touched);
        return;
    }
    use std::arch::x86_64::*;
    let n = slots.len();
    let v_scale = _mm256_set1_ps(num_scale);
    let tf_ptr = tfs_bytes.as_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let mut dln_buf = [0.0f32; 8];
        for lane in 0..8 {
            dln_buf[lane] = dl_norm[slots[i + lane] as usize];
        }
        let mut res = [0.0f32; 8];
        unsafe {
            let v_tf = load_8_int_as_f32_avx2(width, tf_ptr, i);
            let v_dln = _mm256_loadu_ps(dln_buf.as_ptr());
            let v_res = _mm256_div_ps(_mm256_mul_ps(v_scale, v_tf), _mm256_add_ps(v_tf, v_dln));
            _mm256_storeu_ps(res.as_mut_ptr(), v_res);
        }
        for lane in 0..8 {
            let s = slots[i + lane];
            scores[s as usize] += res[lane];
            touched[(s >> 6) as usize] |= 1u64 << (s & 63);
        }
        i += 8;
    }
    if i < n {
        accumulate_sparse_scalar(
            width,
            &slots[i..],
            &tfs_bytes[i * width.byte_size()..],
            num_scale,
            dl_norm,
            scores,
            touched,
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2")]
unsafe fn max_score_256_avx2(scores: &[f32; FTS_BLOCK_SIZE], mask: &[u64; 4]) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let bit_weights = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    let ptr = scores.as_ptr();
    for (w, &mw) in mask.iter().enumerate() {
        if mw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..8 {
            let bm = ((mw >> (chunk << 3)) & 0xFF) as i32;
            if bm == 0 {
                continue;
            }
            let vs = unsafe { _mm256_loadu_ps(ptr.add(base + (chunk << 3))) };
            let masked = if bm == 0xFF {
                vs
            } else {
                let vm = _mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_set1_epi32(bm), bit_weights),
                    bit_weights,
                );
                _mm256_and_ps(vs, _mm256_castsi256_ps(vm))
            };
            acc = _mm256_max_ps(acc, masked);
        }
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let m128 = _mm_max_ps(lo, hi);
    let shuf = _mm_movehdup_ps(m128);
    let m64 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_movehl_ps(m64, m64);
    _mm_cvtss_f32(_mm_max_ss(m64, shuf2))
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2")]
unsafe fn filter_ge_256_avx2(
    scores: &[f32; FTS_BLOCK_SIZE],
    mask: &[u64; 4],
    threshold: f32,
) -> [u64; 4] {
    use std::arch::x86_64::*;
    let vt = _mm256_set1_ps(threshold);
    let ptr = scores.as_ptr();
    let mut out = [0u64; 4];
    for (w, &mw) in mask.iter().enumerate() {
        if mw == 0 {
            continue;
        }
        let base = w << 6;
        let mut word = 0u64;
        for chunk in 0..8 {
            if ((mw >> (chunk << 3)) & 0xFF) == 0 {
                continue;
            }
            let vs = unsafe { _mm256_loadu_ps(ptr.add(base + (chunk << 3))) };
            let cmp = _mm256_cmp_ps::<_CMP_GE_OQ>(vs, vt);
            let bits = (_mm256_movemask_ps(cmp) as u32) as u64;
            word |= bits << (chunk << 3);
        }
        out[w] = word & mw;
    }
    out
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2")]
unsafe fn merge_max_256_avx2(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    use std::arch::x86_64::*;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i < FTS_BLOCK_SIZE {
        unsafe {
            let vd = _mm256_loadu_ps(dp.add(i));
            let vs = _mm256_loadu_ps(sp.add(i));
            _mm256_storeu_ps(dp.add(i), _mm256_max_ps(vd, vs));
        }
        i += 8;
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx2")]
unsafe fn merge_add_256_avx2(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    use std::arch::x86_64::*;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i < FTS_BLOCK_SIZE {
        unsafe {
            let vd = _mm256_loadu_ps(dp.add(i));
            let vs = _mm256_loadu_ps(sp.add(i));
            _mm256_storeu_ps(dp.add(i), _mm256_add_ps(vd, vs));
        }
        i += 8;
    }
}

// ============================================================================
// aarch64 NEON implementations
// ============================================================================

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_4_int_as_f32_neon(
    width: IntWidth,
    ptr: *const u8,
    idx: usize,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        match width {
            IntWidth::U8 => {
                let mut tmp = [0u8; 8];
                std::ptr::copy_nonoverlapping(ptr.add(idx), tmp.as_mut_ptr(), 4);
                let v8 = vld1_u8(tmp.as_ptr());
                let v16 = vget_low_u16(vmovl_u8(v8));
                vcvtq_f32_u32(vmovl_u16(v16))
            }
            IntWidth::U16 => {
                let v16 = vld1_u16(ptr.add(idx * 2) as *const u16);
                vcvtq_f32_u32(vmovl_u16(v16))
            }
            IntWidth::U32 => {
                let v32 = vld1q_u32(ptr.add(idx * 4) as *const u32);
                vcvtq_f32_u32(v32)
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn prepare_dl_norm_256_neon(
    width: IntWidth,
    dls_bytes: &[u8],
    c1: f32,
    c2: f32,
    out: &mut [f32; FTS_BLOCK_SIZE],
) {
    use std::arch::aarch64::*;
    let v_c1 = vdupq_n_f32(c1);
    let v_c2 = vdupq_n_f32(c2);
    let v_min = vdupq_n_f32(f32::MIN_POSITIVE);
    let ptr = dls_bytes.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0usize;
    while i < FTS_BLOCK_SIZE {
        unsafe {
            let v_dl = load_4_int_as_f32_neon(width, ptr, i);
            let v_norm = vmaxq_f32(vfmaq_f32(v_c1, v_c2, v_dl), v_min);
            vst1q_f32(out_ptr.add(i), v_norm);
        }
        i += 4;
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn accumulate_dense_256_neon(
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    bitmap: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    use std::arch::aarch64::*;
    let v_scale = vdupq_n_f32(num_scale);
    let tf_ptr = tfs_bytes.as_ptr();
    let dln_ptr = dl_norm.as_ptr();
    let sc_ptr = scores.as_mut_ptr();
    for (w, &bw) in bitmap.iter().enumerate() {
        if bw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..16 {
            let i = base + (chunk << 2);
            unsafe {
                let v_tf = load_4_int_as_f32_neon(width, tf_ptr, i);
                let v_dln = vld1q_f32(dln_ptr.add(i));
                let v_cur = vld1q_f32(sc_ptr.add(i));
                let v_new = vaddq_f32(v_cur, vdivq_f32(vmulq_f32(v_scale, v_tf), vaddq_f32(v_tf, v_dln)));
                vst1q_f32(sc_ptr.add(i), v_new);
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn accumulate_dense_masked_neon(
    width: IntWidth,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    active: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    use std::arch::aarch64::*;
    let v_scale = vdupq_n_f32(num_scale);
    let v_zero = vdupq_n_f32(0.0);
    let bit_weights: [u32; 4] = [1, 2, 4, 8];
    let v_bw = unsafe { vld1q_u32(bit_weights.as_ptr()) };
    let tf_ptr = tfs_bytes.as_ptr();
    let dln_ptr = dl_norm.as_ptr();
    let sc_ptr = scores.as_mut_ptr();
    for (w, &aw) in active.iter().enumerate() {
        if aw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..16 {
            let nm = ((aw >> (chunk << 2)) & 0xF) as u32;
            if nm == 0 {
                continue;
            }
            let i = base + (chunk << 2);
            unsafe {
                let v_tf = load_4_int_as_f32_neon(width, tf_ptr, i);
                let v_dln = vld1q_f32(dln_ptr.add(i));
                let v_cur = vld1q_f32(sc_ptr.add(i));
                let mut v_delta = vdivq_f32(vmulq_f32(v_scale, v_tf), vaddq_f32(v_tf, v_dln));
                if nm != 0xF {
                    let vm = vtstq_u32(vdupq_n_u32(nm), v_bw);
                    v_delta = vbslq_f32(vm, v_delta, v_zero);
                }
                vst1q_f32(sc_ptr.add(i), vaddq_f32(v_cur, v_delta));
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn accumulate_sparse_neon(
    width: IntWidth,
    slots: &[u8],
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    touched: &mut [u64; 4],
) {
    use std::arch::aarch64::*;
    let n = slots.len();
    let v_scale = vdupq_n_f32(num_scale);
    let tf_ptr = tfs_bytes.as_ptr();
    let mut i = 0usize;
    while i + 4 <= n {
        let mut dln_buf = [0.0f32; 4];
        for lane in 0..4 {
            dln_buf[lane] = dl_norm[slots[i + lane] as usize];
        }
        let mut res = [0.0f32; 4];
        unsafe {
            let v_tf = load_4_int_as_f32_neon(width, tf_ptr, i);
            let v_dln = vld1q_f32(dln_buf.as_ptr());
            let v_res = vdivq_f32(vmulq_f32(v_scale, v_tf), vaddq_f32(v_tf, v_dln));
            vst1q_f32(res.as_mut_ptr(), v_res);
        }
        for lane in 0..4 {
            let s = slots[i + lane];
            scores[s as usize] += res[lane];
            touched[(s >> 6) as usize] |= 1u64 << (s & 63);
        }
        i += 4;
    }
    if i < n {
        accumulate_sparse_scalar(
            width,
            &slots[i..],
            &tfs_bytes[i * width.byte_size()..],
            num_scale,
            dl_norm,
            scores,
            touched,
        );
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn max_score_256_neon(scores: &[f32; FTS_BLOCK_SIZE], mask: &[u64; 4]) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_f32(0.0);
    let v_zero = vdupq_n_f32(0.0);
    let bit_weights: [u32; 4] = [1, 2, 4, 8];
    let v_bw = unsafe { vld1q_u32(bit_weights.as_ptr()) };
    let ptr = scores.as_ptr();
    for (w, &mw) in mask.iter().enumerate() {
        if mw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..16 {
            let nm = ((mw >> (chunk << 2)) & 0xF) as u32;
            if nm == 0 {
                continue;
            }
            let vs = unsafe { vld1q_f32(ptr.add(base + (chunk << 2))) };
            let masked = if nm == 0xF {
                vs
            } else {
                let vm = vtstq_u32(vdupq_n_u32(nm), v_bw);
                vbslq_f32(vm, vs, v_zero)
            };
            acc = vmaxq_f32(acc, masked);
        }
    }
    vmaxvq_f32(acc)
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn filter_ge_256_neon(
    scores: &[f32; FTS_BLOCK_SIZE],
    mask: &[u64; 4],
    threshold: f32,
) -> [u64; 4] {
    use std::arch::aarch64::*;
    let vt = vdupq_n_f32(threshold);
    let bit_weights: [u32; 4] = [1, 2, 4, 8];
    let v_bw = unsafe { vld1q_u32(bit_weights.as_ptr()) };
    let ptr = scores.as_ptr();
    let mut out = [0u64; 4];
    for (w, &mw) in mask.iter().enumerate() {
        if mw == 0 {
            continue;
        }
        let base = w << 6;
        let mut word = 0u64;
        for chunk in 0..16 {
            if ((mw >> (chunk << 2)) & 0xF) == 0 {
                continue;
            }
            let vs = unsafe { vld1q_f32(ptr.add(base + (chunk << 2))) };
            let cmp = vcgeq_f32(vs, vt);
            let bits = vaddvq_u32(vandq_u32(cmp, v_bw)) as u64;
            word |= bits << (chunk << 2);
        }
        out[w] = word & mw;
    }
    out
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn merge_max_256_neon(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    use std::arch::aarch64::*;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i < FTS_BLOCK_SIZE {
        unsafe {
            let vd = vld1q_f32(dp.add(i));
            let vs = vld1q_f32(sp.add(i));
            vst1q_f32(dp.add(i), vmaxq_f32(vd, vs));
        }
        i += 4;
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn merge_add_256_neon(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    use std::arch::aarch64::*;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i < FTS_BLOCK_SIZE {
        unsafe {
            let vd = vld1q_f32(dp.add(i));
            let vs = vld1q_f32(sp.add(i));
            vst1q_f32(dp.add(i), vaddq_f32(vd, vs));
        }
        i += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{DocLenBlock, PackedPostingBlock, PostingBlockPayload, PostingBlockView};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn available_tiers() -> Vec<SimdTier> {
        let mut tiers = vec![SimdTier::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                tiers.push(SimdTier::Avx2);
            }
            if is_x86_feature_detected!("avx512f") {
                tiers.push(SimdTier::Avx512);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
                tiers.push(SimdTier::Neon);
            }
        }
        tiers
    }

    #[test]
    fn test_simd_kernels_match_scalar_across_widths_layouts_and_masks() {
        let mut rng = SmallRng::seed_from_u64(2026);
        let tiers = available_tiers();

        for (max_dl, max_tf) in [(100u32, 20u32), (10_000, 1_000), (100_000, 100_000)] {
            let mut dls = DocLenBlock::default();
            for s in 0..=255u8 {
                dls.set(s, rng.random_range(1..=max_dl));
            }
            let dl_bytes = dls.encode();
            let dl_view = crate::block::DocLenBlockView::parse(&dl_bytes).unwrap();

            for (c1, c2) in [(0.3f32, 0.045f32), (0.0, 0.0), (0.0, 0.1)] {
                let mut ref_dln = [0.0f32; 256];
                prepare_dl_norm_256(SimdTier::Scalar, dl_view.width, dl_view.dls_bytes, c1, c2, &mut ref_dln);

                for &tier in &tiers {
                    let mut got_dln = [0.0f32; 256];
                    prepare_dl_norm_256(tier, dl_view.width, dl_view.dls_bytes, c1, c2, &mut got_dln);
                    for i in 0..256 {
                        let tol = 1e-6f32.max(ref_dln[i].abs() * 1e-6);
                        assert!(
                            (got_dln[i] - ref_dln[i]).abs() <= tol,
                            "tier={tier:?} i={i}: got={} vs ref={}",
                            got_dln[i],
                            ref_dln[i]
                        );
                    }
                }

                // Test sparse counts (0, 1, 7, 8, 15, 16, 17, 63, 95) and dense (150, 256)
                for count in [1usize, 7, 8, 15, 16, 17, 63, 95, 150, 256] {
                    let mut slots: Vec<u8> = (0..=255u8).collect();
                    // Deterministic partial shuffle
                    for i in 0..count {
                        let j = rng.random_range(i..256);
                        slots.swap(i, j);
                    }
                    slots.truncate(count);
                    slots.sort_unstable();

                    let mut blk = PackedPostingBlock::default();
                    for &s in &slots {
                        blk.upsert(s, rng.random_range(1..=max_tf), &dls);
                    }
                    let enc = blk.encode();
                    let view = PostingBlockView::parse(&enc).unwrap();
                    let num_scale = 3.75f32;

                    let mask = [
                        rng.random::<u64>(),
                        rng.random::<u64>(),
                        rng.random::<u64>(),
                        rng.random::<u64>(),
                    ];

                    let mut ref_scores = [0.0f32; 256];
                    let mut ref_touched = [0u64; 4];
                    let mut ref_masked = [0.0f32; 256];
                    match &view.payload {
                        PostingBlockPayload::Sparse { slots, tfs_bytes } => {
                            accumulate_sparse(
                                SimdTier::Scalar,
                                view.tf_width,
                                slots,
                                tfs_bytes,
                                num_scale,
                                &ref_dln,
                                &mut ref_scores,
                                &mut ref_touched,
                            );
                            accumulate_sparse_masked(
                                SimdTier::Scalar,
                                view.tf_width,
                                slots,
                                tfs_bytes,
                                num_scale,
                                &ref_dln,
                                &mask,
                                &mut ref_masked,
                            );
                        }
                        PostingBlockPayload::Dense { bitmap, tfs_bytes } => {
                            accumulate_dense_256(
                                SimdTier::Scalar,
                                view.tf_width,
                                tfs_bytes,
                                num_scale,
                                &ref_dln,
                                bitmap,
                                &mut ref_scores,
                                &mut ref_touched,
                            );
                            accumulate_dense_masked_256(
                                SimdTier::Scalar,
                                view.tf_width,
                                tfs_bytes,
                                num_scale,
                                &ref_dln,
                                bitmap,
                                &mask,
                                &mut ref_masked,
                            );
                        }
                    }

                    let ref_max = max_score_256(SimdTier::Scalar, &ref_scores, &mask);
                    let ref_ge = filter_ge_256(SimdTier::Scalar, &ref_scores, &mask, ref_max * 0.5);

                    for &tier in &tiers {
                        let mut got_scores = [0.0f32; 256];
                        let mut got_touched = [0u64; 4];
                        let mut got_masked = [0.0f32; 256];
                        match &view.payload {
                            PostingBlockPayload::Sparse { slots, tfs_bytes } => {
                                accumulate_sparse(
                                    tier,
                                    view.tf_width,
                                    slots,
                                    tfs_bytes,
                                    num_scale,
                                    &ref_dln,
                                    &mut got_scores,
                                    &mut got_touched,
                                );
                                accumulate_sparse_masked(
                                    tier,
                                    view.tf_width,
                                    slots,
                                    tfs_bytes,
                                    num_scale,
                                    &ref_dln,
                                    &mask,
                                    &mut got_masked,
                                );
                            }
                            PostingBlockPayload::Dense { bitmap, tfs_bytes } => {
                                accumulate_dense_256(
                                    tier,
                                    view.tf_width,
                                    tfs_bytes,
                                    num_scale,
                                    &ref_dln,
                                    bitmap,
                                    &mut got_scores,
                                    &mut got_touched,
                                );
                                accumulate_dense_masked_256(
                                    tier,
                                    view.tf_width,
                                    tfs_bytes,
                                    num_scale,
                                    &ref_dln,
                                    bitmap,
                                    &mask,
                                    &mut got_masked,
                                );
                            }
                        }

                        assert_eq!(got_touched, ref_touched);
                        for i in 0..256 {
                            assert!(
                                (got_scores[i] - ref_scores[i]).abs() <= 1e-6,
                                "tier={tier:?} count={count} slot={i}: {} vs {}",
                                got_scores[i],
                                ref_scores[i]
                            );
                            assert!(
                                (got_masked[i] - ref_masked[i]).abs() <= 1e-6,
                                "masked tier={tier:?} count={count} slot={i}: {} vs {}",
                                got_masked[i],
                                ref_masked[i]
                            );
                        }

                        let got_max = max_score_256(tier, &got_scores, &mask);
                        assert!((got_max - ref_max).abs() <= 1e-6);
                        let got_ge = filter_ge_256(tier, &got_scores, &mask, ref_max * 0.5);
                        assert_eq!(got_ge, ref_ge);

                        let mut dst_max = ref_masked;
                        let mut ref_dst_max = ref_masked;
                        merge_max_256(tier, &mut dst_max, &ref_scores);
                        merge_max_256(SimdTier::Scalar, &mut ref_dst_max, &ref_scores);
                        assert_eq!(dst_max, ref_dst_max);

                        let mut dst_add = ref_masked;
                        let mut ref_dst_add = ref_masked;
                        merge_add_256(tier, &mut dst_add, &ref_scores);
                        merge_add_256(SimdTier::Scalar, &mut ref_dst_add, &ref_scores);
                        for i in 0..256 {
                            assert!((dst_add[i] - ref_dst_add[i]).abs() <= 1e-6);
                        }
                    }
                }
            }
        }
    }
}
