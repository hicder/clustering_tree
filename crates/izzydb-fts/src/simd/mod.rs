//! Runtime-dispatched SIMD kernels for 256-slot FTS block scoring, filtering, and reduction.

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod aarch64;
mod scalar;
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
mod x86;

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
        SimdTier::Avx512 => unsafe {
            x86::prepare_dl_norm_256_avx512(width, dls_bytes, c1, c2, out)
        },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { x86::prepare_dl_norm_256_avx2(width, dls_bytes, c1, c2, out) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe {
            aarch64::prepare_dl_norm_256_neon(width, dls_bytes, c1, c2, out)
        },
        _ => scalar::prepare_dl_norm_256_scalar(width, dls_bytes, c1, c2, out),
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
            x86::accumulate_dense_256_avx512(width, tfs_bytes, num_scale, dl_norm, bitmap, scores)
        },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe {
            x86::accumulate_dense_256_avx2(width, tfs_bytes, num_scale, dl_norm, bitmap, scores)
        },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe {
            aarch64::accumulate_dense_256_neon(width, tfs_bytes, num_scale, dl_norm, bitmap, scores)
        },
        _ => scalar::accumulate_dense_256_scalar(width, tfs_bytes, num_scale, dl_norm, bitmap, scores),
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
        scalar::accumulate_active_bits_scalar(width, tfs_bytes, num_scale, dl_norm, &active, scores);
        return;
    }
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe {
            x86::accumulate_dense_masked_avx512(
                width, tfs_bytes, num_scale, dl_norm, &active, scores,
            )
        },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe {
            x86::accumulate_dense_masked_avx2(width, tfs_bytes, num_scale, dl_norm, &active, scores)
        },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe {
            aarch64::accumulate_dense_masked_neon(
                width, tfs_bytes, num_scale, dl_norm, &active, scores,
            )
        },
        _ => scalar::accumulate_active_bits_scalar(
            width, tfs_bytes, num_scale, dl_norm, &active, scores,
        ),
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
            x86::accumulate_sparse_avx512(width, slots, tfs_bytes, num_scale, dl_norm, scores, touched)
        },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 | SimdTier::Avx512 if slots.len() >= 8 => unsafe {
            x86::accumulate_sparse_avx2(width, slots, tfs_bytes, num_scale, dl_norm, scores, touched)
        },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon if slots.len() >= 4 => unsafe {
            aarch64::accumulate_sparse_neon(
                width, slots, tfs_bytes, num_scale, dl_norm, scores, touched,
            )
        },
        _ => scalar::accumulate_sparse_scalar(
            width, slots, tfs_bytes, num_scale, dl_norm, scores, touched,
        ),
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
        for (i, &slot) in slots.iter().enumerate() {
            if ((mask[(slot >> 6) as usize] >> (slot & 63)) & 1) != 0 {
                let tf_f = read_tf_at(width, tfs_bytes, i) as f32;
                let idx = slot as usize;
                scores[idx] += (num_scale * tf_f) / (tf_f + dl_norm[idx]);
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
    scalar::accumulate_u32_pairs(
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
        return scalar::max_score_256_scalar(scores, mask);
    }
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe { x86::max_score_256_avx512(scores, mask) },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { x86::max_score_256_avx2(scores, mask) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe { aarch64::max_score_256_neon(scores, mask) },
        _ => scalar::max_score_256_scalar(scores, mask),
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
        return scalar::filter_ge_256_scalar(scores, mask, threshold);
    }
    match tier {
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx512 => unsafe { x86::filter_ge_256_avx512(scores, mask, threshold) },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { x86::filter_ge_256_avx2(scores, mask, threshold) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe { aarch64::filter_ge_256_neon(scores, mask, threshold) },
        _ => scalar::filter_ge_256_scalar(scores, mask, threshold),
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
        SimdTier::Avx512 => unsafe { x86::merge_max_256_avx512(dst, src) },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { x86::merge_max_256_avx2(dst, src) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe { aarch64::merge_max_256_neon(dst, src) },
        _ => scalar::merge_max_256_scalar(dst, src),
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
        SimdTier::Avx512 => unsafe { x86::merge_add_256_avx512(dst, src) },
        #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
        SimdTier::Avx2 => unsafe { x86::merge_add_256_avx2(dst, src) },
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        SimdTier::Neon => unsafe { aarch64::merge_add_256_neon(dst, src) },
        _ => scalar::merge_add_256_scalar(dst, src),
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
                prepare_dl_norm_256(
                    SimdTier::Scalar,
                    dl_view.width,
                    dl_view.dls_bytes,
                    c1,
                    c2,
                    &mut ref_dln,
                );

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

                for count in [1usize, 7, 8, 15, 16, 17, 63, 95, 150, 256] {
                    let mut slots: Vec<u8> = (0..=255u8).collect();
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
                    let ref_ge =
                        filter_ge_256(SimdTier::Scalar, &ref_scores, &mask, ref_max * 0.5);

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
