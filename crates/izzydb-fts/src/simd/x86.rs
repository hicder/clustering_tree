use izzydb_core::config::FTS_BLOCK_SIZE;

use crate::block::{IntWidth, TfEncoding};

use super::scalar::{
    accumulate_active_bits_scalar, accumulate_dense_256_scalar, accumulate_sparse_scalar,
    prepare_dl_norm_256_scalar,
};

#[inline]
fn tf_tail_slice(tf_enc: TfEncoding, tfs_bytes: &[u8], i: usize) -> &[u8] {
    match tf_enc {
        TfEncoding::TfOne => &[],
        TfEncoding::U2 => &tfs_bytes[i >> 2..],
        TfEncoding::U4 => &tfs_bytes[i >> 1..],
        TfEncoding::U8 => &tfs_bytes[i..],
        TfEncoding::U16 => &tfs_bytes[i * 2..],
        TfEncoding::U32 => &tfs_bytes[i * 4..],
    }
}

// ============================================================================
// x86_64 AVX-512F implementations
// ============================================================================

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn load_16_tf_as_f32_avx512(
    tf_enc: TfEncoding,
    ptr: *const u8,
    idx: usize,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    unsafe {
        match tf_enc {
            TfEncoding::TfOne => _mm512_set1_ps(1.0),
            TfEncoding::U2 => {
                let packed = std::ptr::read_unaligned(ptr.add(idx >> 2) as *const u32);
                let v_raw = _mm512_set1_epi32(packed as i32);
                let shifts = _mm512_setr_epi32(
                    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                );
                let v_u2 = _mm512_and_epi32(_mm512_srlv_epi32(v_raw, shifts), _mm512_set1_epi32(3));
                _mm512_cvtepi32_ps(_mm512_add_epi32(v_u2, _mm512_set1_epi32(1)))
            }
            TfEncoding::U4 => {
                let lo = std::ptr::read_unaligned(ptr.add(idx >> 1) as *const u32) as i32;
                let hi = std::ptr::read_unaligned(ptr.add((idx >> 1) + 4) as *const u32) as i32;
                let v_raw = _mm512_set_epi32(
                    hi, hi, hi, hi, hi, hi, hi, hi, lo, lo, lo, lo, lo, lo, lo, lo,
                );
                let shifts = _mm512_setr_epi32(
                    0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28,
                );
                let v_u4 = _mm512_and_epi32(_mm512_srlv_epi32(v_raw, shifts), _mm512_set1_epi32(15));
                _mm512_cvtepi32_ps(_mm512_add_epi32(v_u4, _mm512_set1_epi32(1)))
            }
            TfEncoding::U8 => {
                let v = _mm_loadu_si128(ptr.add(idx) as *const __m128i);
                _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v))
            }
            TfEncoding::U16 => {
                let v = _mm256_loadu_si256(ptr.add(idx * 2) as *const __m256i);
                _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(v))
            }
            TfEncoding::U32 => {
                let v = _mm512_loadu_si512(ptr.add(idx * 4) as *const __m512i);
                _mm512_cvtepu32_ps(v)
            }
        }
    }
}

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

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn prepare_dl_norm_256_avx512(
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

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn accumulate_dense_256_avx512(
    tf_enc: TfEncoding,
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
    if matches!(tf_enc, TfEncoding::TfOne | TfEncoding::U2 | TfEncoding::U4) {
        unsafe {
            accumulate_dense_masked_avx512(tf_enc, tfs_bytes, num_scale, dl_norm, bitmap, scores);
        }
        return;
    }
    for (w, &bw) in bitmap.iter().enumerate() {
        if bw == 0 {
            continue;
        }
        let base = w << 6;
        for chunk in 0..4 {
            let i = base + (chunk << 4);
            unsafe {
                let v_tf = load_16_tf_as_f32_avx512(tf_enc, tf_ptr, i);
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

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn accumulate_dense_masked_avx512(
    tf_enc: TfEncoding,
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
                let v_tf = load_16_tf_as_f32_avx512(tf_enc, tf_ptr, i);
                let v_dln = _mm512_loadu_ps(dln_ptr.add(i));
                let v_cur = _mm512_loadu_ps(sc_ptr.add(i));
                let v_delta = _mm512_div_ps(_mm512_mul_ps(v_scale, v_tf), _mm512_add_ps(v_tf, v_dln));
                let v_new = _mm512_mask_add_ps(v_cur, k, v_cur, v_delta);
                _mm512_storeu_ps(sc_ptr.add(i), v_new);
            }
        }
    }
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn accumulate_sparse_avx512(
    tf_enc: TfEncoding,
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
            let v_tf = load_16_tf_as_f32_avx512(tf_enc, tf_ptr, i);
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
            tf_enc,
            &slots[i..],
            tf_tail_slice(tf_enc, tfs_bytes, i),
            num_scale,
            dl_norm,
            scores,
            touched,
        );
    }
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn accumulate_u32_pairs_avx512(
    slots: &[u8],
    tfs: &[u32],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    mut i: usize,
) -> usize {
    use std::arch::x86_64::*;
    let n = slots.len();
    let v_scale = _mm512_set1_ps(num_scale);
    while i + 16 <= n {
        let mut dln_buf = [0.0f32; 16];
        for lane in 0..16 {
            dln_buf[lane] = dl_norm[slots[i + lane] as usize];
        }
        unsafe {
            let v_tf =
                _mm512_cvtepu32_ps(_mm512_loadu_si512(tfs.as_ptr().add(i) as *const __m512i));
            let v_dln = _mm512_loadu_ps(dln_buf.as_ptr());
            let v_res = _mm512_div_ps(_mm512_mul_ps(v_scale, v_tf), _mm512_add_ps(v_tf, v_dln));
            let mut res = [0.0f32; 16];
            _mm512_storeu_ps(res.as_mut_ptr(), v_res);
            for lane in 0..16 {
                scores[slots[i + lane] as usize] += res[lane];
            }
        }
        i += 16;
    }
    i
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn max_score_256_avx512(
    scores: &[f32; FTS_BLOCK_SIZE],
    mask: &[u64; 4],
) -> f32 {
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
            let v =
                unsafe { _mm512_maskz_mov_ps(k, _mm512_loadu_ps(ptr.add(base + (chunk << 4)))) };
            acc = _mm512_max_ps(acc, v);
        }
    }
    _mm512_reduce_max_ps(acc)
}

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn filter_ge_256_avx512(
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

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn merge_max_256_avx512(
    dst: &mut [f32; FTS_BLOCK_SIZE],
    src: &[f32; FTS_BLOCK_SIZE],
) {
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

#[target_feature(enable = "avx512f")]
pub(super) unsafe fn merge_add_256_avx512(
    dst: &mut [f32; FTS_BLOCK_SIZE],
    src: &[f32; FTS_BLOCK_SIZE],
) {
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

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_8_tf_as_f32_avx2(
    tf_enc: TfEncoding,
    ptr: *const u8,
    idx: usize,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    unsafe {
        match tf_enc {
            TfEncoding::TfOne => _mm256_set1_ps(1.0),
            TfEncoding::U2 => {
                let packed = std::ptr::read_unaligned(ptr.add(idx >> 2) as *const u16) as i32;
                let v_raw = _mm256_set1_epi32(packed);
                let shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
                let v_u2 = _mm256_and_si256(
                    _mm256_srlv_epi32(v_raw, shifts),
                    _mm256_set1_epi32(3),
                );
                _mm256_cvtepi32_ps(_mm256_add_epi32(v_u2, _mm256_set1_epi32(1)))
            }
            TfEncoding::U4 => {
                let packed = std::ptr::read_unaligned(ptr.add(idx >> 1) as *const u32) as i32;
                let v_raw = _mm256_set1_epi32(packed);
                let shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
                let v_u4 = _mm256_and_si256(
                    _mm256_srlv_epi32(v_raw, shifts),
                    _mm256_set1_epi32(15),
                );
                _mm256_cvtepi32_ps(_mm256_add_epi32(v_u4, _mm256_set1_epi32(1)))
            }
            TfEncoding::U8 => {
                let v = _mm_loadl_epi64(ptr.add(idx) as *const __m128i);
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v))
            }
            TfEncoding::U16 => {
                let v = _mm_loadu_si128(ptr.add(idx * 2) as *const __m128i);
                _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v))
            }
            TfEncoding::U32 => {
                let v = _mm256_loadu_si256(ptr.add(idx * 4) as *const __m256i);
                _mm256_cvtepi32_ps(v)
            }
        }
    }
}

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

#[inline]
fn slice_u32_fits_i32(bytes: &[u8]) -> bool {
    bytes.as_chunks::<4>().0.iter().all(|c| (c[3] & 0x80) == 0)
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn prepare_dl_norm_256_avx2(
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

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn accumulate_dense_256_avx2(
    tf_enc: TfEncoding,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    bitmap: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    if matches!(tf_enc, TfEncoding::TfOne | TfEncoding::U2 | TfEncoding::U4) {
        unsafe {
            accumulate_dense_masked_avx2(tf_enc, tfs_bytes, num_scale, dl_norm, bitmap, scores);
        }
        return;
    }
    if tf_enc == TfEncoding::U32 && !slice_u32_fits_i32(tfs_bytes) {
        accumulate_dense_256_scalar(tf_enc, tfs_bytes, num_scale, dl_norm, bitmap, scores);
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
                let v_tf = load_8_tf_as_f32_avx2(tf_enc, tf_ptr, i);
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

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn accumulate_dense_masked_avx2(
    tf_enc: TfEncoding,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    active: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    if tf_enc == TfEncoding::U32 && !slice_u32_fits_i32(tfs_bytes) {
        accumulate_active_bits_scalar(tf_enc, tfs_bytes, num_scale, dl_norm, active, scores);
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
                let v_tf = load_8_tf_as_f32_avx2(tf_enc, tf_ptr, i);
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

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn accumulate_sparse_avx2(
    tf_enc: TfEncoding,
    slots: &[u8],
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    touched: &mut [u64; 4],
) {
    if tf_enc == TfEncoding::U32 && !slice_u32_fits_i32(tfs_bytes) {
        accumulate_sparse_scalar(tf_enc, slots, tfs_bytes, num_scale, dl_norm, scores, touched);
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
            let v_tf = load_8_tf_as_f32_avx2(tf_enc, tf_ptr, i);
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
            tf_enc,
            &slots[i..],
            tf_tail_slice(tf_enc, tfs_bytes, i),
            num_scale,
            dl_norm,
            scores,
            touched,
        );
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn accumulate_u32_pairs_avx2(
    slots: &[u8],
    tfs: &[u32],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    mut i: usize,
) -> usize {
    use std::arch::x86_64::*;
    let n = slots.len();
    let v_scale = _mm256_set1_ps(num_scale);
    while i + 8 <= n {
        let mut dln_buf = [0.0f32; 8];
        for lane in 0..8 {
            dln_buf[lane] = dl_norm[slots[i + lane] as usize];
        }
        unsafe {
            let v_tf =
                _mm256_cvtepi32_ps(_mm256_loadu_si256(tfs.as_ptr().add(i) as *const __m256i));
            let v_dln = _mm256_loadu_ps(dln_buf.as_ptr());
            let v_res = _mm256_div_ps(_mm256_mul_ps(v_scale, v_tf), _mm256_add_ps(v_tf, v_dln));
            let mut res = [0.0f32; 8];
            _mm256_storeu_ps(res.as_mut_ptr(), v_res);
            for lane in 0..8 {
                scores[slots[i + lane] as usize] += res[lane];
            }
        }
        i += 8;
    }
    i
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn max_score_256_avx2(scores: &[f32; FTS_BLOCK_SIZE], mask: &[u64; 4]) -> f32 {
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

#[target_feature(enable = "avx2")]
pub(super) unsafe fn filter_ge_256_avx2(
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

#[target_feature(enable = "avx2")]
pub(super) unsafe fn merge_max_256_avx2(
    dst: &mut [f32; FTS_BLOCK_SIZE],
    src: &[f32; FTS_BLOCK_SIZE],
) {
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

#[target_feature(enable = "avx2")]
pub(super) unsafe fn merge_add_256_avx2(
    dst: &mut [f32; FTS_BLOCK_SIZE],
    src: &[f32; FTS_BLOCK_SIZE],
) {
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
