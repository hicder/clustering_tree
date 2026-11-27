use izzydb_core::config::FTS_BLOCK_SIZE;

use crate::block::{IntWidth, TfEncoding, read_tf_enc_at};

use super::scalar::accumulate_sparse_scalar;

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

#[target_feature(enable = "neon")]
pub(super) unsafe fn prepare_dl_norm_256_neon(
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

#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_4_tf_as_f32_neon(
    tf_enc: TfEncoding,
    tfs_bytes: &[u8],
    idx: usize,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        match tf_enc {
            TfEncoding::TfOne => vdupq_n_f32(1.0),
            TfEncoding::U2 | TfEncoding::U4 => {
                let vals = [
                    read_tf_enc_at(tf_enc, tfs_bytes, idx),
                    read_tf_enc_at(tf_enc, tfs_bytes, idx + 1),
                    read_tf_enc_at(tf_enc, tfs_bytes, idx + 2),
                    read_tf_enc_at(tf_enc, tfs_bytes, idx + 3),
                ];
                vcvtq_f32_u32(vld1q_u32(vals.as_ptr()))
            }
            TfEncoding::U8 => load_4_int_as_f32_neon(IntWidth::U8, tfs_bytes.as_ptr(), idx),
            TfEncoding::U16 => load_4_int_as_f32_neon(IntWidth::U16, tfs_bytes.as_ptr(), idx),
            TfEncoding::U32 => load_4_int_as_f32_neon(IntWidth::U32, tfs_bytes.as_ptr(), idx),
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn accumulate_dense_256_neon(
    tf_enc: TfEncoding,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    bitmap: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    if matches!(tf_enc, TfEncoding::TfOne | TfEncoding::U2 | TfEncoding::U4) {
        unsafe {
            accumulate_dense_masked_neon(tf_enc, tfs_bytes, num_scale, dl_norm, bitmap, scores);
        }
        return;
    }
    use std::arch::aarch64::*;
    let v_scale = vdupq_n_f32(num_scale);
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
                let v_tf = load_4_tf_as_f32_neon(tf_enc, tfs_bytes, i);
                let v_dln = vld1q_f32(dln_ptr.add(i));
                let v_cur = vld1q_f32(sc_ptr.add(i));
                let v_new =
                    vaddq_f32(v_cur, vdivq_f32(vmulq_f32(v_scale, v_tf), vaddq_f32(v_tf, v_dln)));
                vst1q_f32(sc_ptr.add(i), v_new);
            }
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn accumulate_dense_masked_neon(
    tf_enc: TfEncoding,
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
                let v_tf = load_4_tf_as_f32_neon(tf_enc, tfs_bytes, i);
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

#[target_feature(enable = "neon")]
pub(super) unsafe fn accumulate_sparse_neon(
    tf_enc: TfEncoding,
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
    let mut i = 0usize;
    while i + 4 <= n {
        let mut dln_buf = [0.0f32; 4];
        for lane in 0..4 {
            dln_buf[lane] = dl_norm[slots[i + lane] as usize];
        }
        let mut res = [0.0f32; 4];
        unsafe {
            let v_tf = load_4_tf_as_f32_neon(tf_enc, tfs_bytes, i);
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
        let tail = match tf_enc {
            TfEncoding::TfOne => &[],
            TfEncoding::U2 => &tfs_bytes[i >> 2..],
            TfEncoding::U4 => &tfs_bytes[i >> 1..],
            TfEncoding::U8 => &tfs_bytes[i..],
            TfEncoding::U16 => &tfs_bytes[i * 2..],
            TfEncoding::U32 => &tfs_bytes[i * 4..],
        };
        accumulate_sparse_scalar(
            tf_enc,
            &slots[i..],
            tail,
            num_scale,
            dl_norm,
            scores,
            touched,
        );
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn accumulate_u32_pairs_neon(
    slots: &[u8],
    tfs: &[u32],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    mut i: usize,
) -> usize {
    use std::arch::aarch64::*;
    let n = slots.len();
    let v_scale = vdupq_n_f32(num_scale);
    while i + 4 <= n {
        let mut dln_buf = [0.0f32; 4];
        for lane in 0..4 {
            dln_buf[lane] = dl_norm[slots[i + lane] as usize];
        }
        unsafe {
            let v_tf = vcvtq_f32_u32(vld1q_u32(tfs.as_ptr().add(i)));
            let v_dln = vld1q_f32(dln_buf.as_ptr());
            let v_res = vdivq_f32(vmulq_f32(v_scale, v_tf), vaddq_f32(v_tf, v_dln));
            let mut res = [0.0f32; 4];
            vst1q_f32(res.as_mut_ptr(), v_res);
            for lane in 0..4 {
                scores[slots[i + lane] as usize] += res[lane];
            }
        }
        i += 4;
    }
    i
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn max_score_256_neon(scores: &[f32; FTS_BLOCK_SIZE], mask: &[u64; 4]) -> f32 {
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

#[target_feature(enable = "neon")]
pub(super) unsafe fn filter_ge_256_neon(
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

#[target_feature(enable = "neon")]
pub(super) unsafe fn merge_max_256_neon(
    dst: &mut [f32; FTS_BLOCK_SIZE],
    src: &[f32; FTS_BLOCK_SIZE],
) {
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

#[target_feature(enable = "neon")]
pub(super) unsafe fn merge_add_256_neon(
    dst: &mut [f32; FTS_BLOCK_SIZE],
    src: &[f32; FTS_BLOCK_SIZE],
) {
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
