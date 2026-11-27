use izzydb_core::config::{FTS_BLOCK_SIZE, for_each_set_bit_256};

use crate::block::{IntWidth, TfEncoding, read_tf_enc_at};

use super::SimdTier;

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
pub(super) fn prepare_dl_norm_256_scalar(
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
pub(super) fn accumulate_dense_256_scalar(
    tf_enc: TfEncoding,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    bitmap: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    accumulate_active_bits_scalar(tf_enc, tfs_bytes, num_scale, dl_norm, bitmap, scores);
}

#[inline]
pub(super) fn accumulate_active_bits_scalar(
    tf_enc: TfEncoding,
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    active: &[u64; 4],
    scores: &mut [f32; FTS_BLOCK_SIZE],
) {
    if tf_enc == TfEncoding::TfOne {
        for_each_set_bit_256(active, |slot| {
            let idx = slot as usize;
            scores[idx] += num_scale / (1.0 + dl_norm[idx]);
        });
        return;
    }
    for_each_set_bit_256(active, |slot| {
        let idx = slot as usize;
        let tf_f = read_tf_enc_at(tf_enc, tfs_bytes, idx) as f32;
        scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
    });
}

#[inline]
pub(super) fn accumulate_sparse_scalar(
    tf_enc: TfEncoding,
    slots: &[u8],
    tfs_bytes: &[u8],
    num_scale: f32,
    dl_norm: &[f32; FTS_BLOCK_SIZE],
    scores: &mut [f32; FTS_BLOCK_SIZE],
    touched: &mut [u64; 4],
) {
    match tf_enc {
        TfEncoding::TfOne => {
            for &slot in slots {
                let idx = slot as usize;
                scores[idx] += num_scale / (1.0 + dl_norm[idx]);
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        }
        TfEncoding::U2 => {
            for (i, &slot) in slots.iter().enumerate() {
                let tf_f =
                    ((((tfs_bytes[i >> 2] >> ((i & 3) * 2)) & 0x03) + 1) as u32) as f32;
                let idx = slot as usize;
                scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        }
        TfEncoding::U4 => {
            for (i, &slot) in slots.iter().enumerate() {
                let tf_f =
                    ((((tfs_bytes[i >> 1] >> ((i & 1) * 4)) & 0x0F) + 1) as u32) as f32;
                let idx = slot as usize;
                scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        }
        TfEncoding::U8 => {
            for (i, &slot) in slots.iter().enumerate() {
                let tf_f = tfs_bytes[i] as f32;
                let idx = slot as usize;
                scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        }
        TfEncoding::U16 => {
            for (i, &slot) in slots.iter().enumerate() {
                let off = i * 2;
                let tf_f = u16::from_le_bytes([tfs_bytes[off], tfs_bytes[off + 1]]) as f32;
                let idx = slot as usize;
                scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        }
        TfEncoding::U32 => {
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
pub(super) fn accumulate_u32_pairs(
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
        i = unsafe { super::x86::accumulate_u32_pairs_avx512(slots, tfs, num_scale, dl_norm, scores, i) };
    }
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if matches!(tier, SimdTier::Avx2 | SimdTier::Avx512)
        && i + 8 <= n
        && tfs[i..].iter().all(|&v| v <= i32::MAX as u32)
    {
        i = unsafe { super::x86::accumulate_u32_pairs_avx2(slots, tfs, num_scale, dl_norm, scores, i) };
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if tier == SimdTier::Neon && i + 4 <= n {
        i = unsafe { super::aarch64::accumulate_u32_pairs_neon(slots, tfs, num_scale, dl_norm, scores, i) };
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
pub(super) fn max_score_256_scalar(scores: &[f32; FTS_BLOCK_SIZE], mask: &[u64; 4]) -> f32 {
    let mut best = 0.0f32;
    for_each_set_bit_256(mask, |slot| {
        let s = scores[slot as usize];
        if s > best {
            best = s;
        }
    });
    best
}

#[inline]
pub(super) fn filter_ge_256_scalar(
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
pub(super) fn merge_max_256_scalar(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        if s > *d {
            *d = s;
        }
    }
}

#[inline]
pub(super) fn merge_add_256_scalar(dst: &mut [f32; FTS_BLOCK_SIZE], src: &[f32; FTS_BLOCK_SIZE]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}
