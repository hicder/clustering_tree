//! SIMD-accelerated RaBitQ leaf scoring across AVX-512, x86 POPCNT, NEON, and scalar backends.

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod aarch64;
mod scalar;
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
mod x86;

use izzydb_core::config::{DistanceMetric, DocId};

pub(super) const METRIC_L2: u8 = DistanceMetric::L2Squared as u8;
pub(super) const METRIC_COSINE: u8 = DistanceMetric::Cosine as u8;
pub(super) const METRIC_DOT: u8 = DistanceMetric::DotProduct as u8;

/// Candidate scored during Stage 1 RaBitQ leaf scanning.
#[derive(Debug, Clone, Copy)]
pub struct LeafCandidate {
    pub doc_id: DocId,
    pub dist: f32,
}

/// Parameters needed to batch-score a packed leaf node using RaBitQ bitwise popcount.
///
/// `codes_bytes` is laid out in 8-entry SIMD tiles:
/// `tile = slot / 8`, `lane = slot % 8`, word `w` of `slot` is at
/// `((tile * rabitq_words + w) * 8 + lane) * 8`.
/// `doc_ids_bytes`, `code_popcounts_bytes`, `res_norm_sqs_bytes`, and `scales_bytes`
/// are padded to `slot_cap.next_multiple_of(8)` entries so SIMD loops never read out of bounds.
pub struct RabitqLeafScoreParams<'a> {
    pub slot_cap: usize,
    pub rabitq_words: usize,
    pub popcount_u32: bool,
    pub doc_ids_bytes: &'a [u8],
    pub codes_bytes: &'a [u8],
    pub code_popcounts_bytes: &'a [u8],
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

/// Score all `slot_cap` vectors in a packed leaf and append [`LeafCandidate`]s to `out`.
pub fn score_leaf_rabitq(params: &RabitqLeafScoreParams<'_>, out: &mut Vec<LeafCandidate>) {
    score_leaf_rabitq_masked(params, None, out);
}

/// Score matching entries in a packed leaf (where bit `slot` in `allowed_slots` is set, or all entries
/// in `0..slot_cap` when `allowed_slots` is `None`) and append [`LeafCandidate`]s to `out`.
pub fn score_leaf_rabitq_masked(
    params: &RabitqLeafScoreParams<'_>,
    allowed_slots: Option<&[u64]>,
    out: &mut Vec<LeafCandidate>,
) {
    if params.slot_cap == 0 {
        return;
    }
    if let Some(mask) = allowed_slots {
        debug_assert!(mask.len() * 64 >= params.slot_cap);
        if mask.iter().all(|&w| w == 0) {
            return;
        }
        dispatch_score_leaf::<true>(params, mask, out);
    } else {
        dispatch_score_leaf::<false>(params, &[], out);
    }
}

#[inline]
fn dispatch_score_leaf<const MASKED: bool>(
    params: &RabitqLeafScoreParams<'_>,
    allowed_slots: &[u64],
    out: &mut Vec<LeafCandidate>,
) {
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vpopcntdq")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            // Safety: runtime-checked AVX-512F + VPOPCNTDQ + AVX2 + FMA; slices validated by caller.
            unsafe {
                match params.metric {
                    DistanceMetric::L2Squared => {
                        x86::score_leaf_avx512::<METRIC_L2, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::Cosine => {
                        x86::score_leaf_avx512::<METRIC_COSINE, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::DotProduct => {
                        x86::score_leaf_avx512::<METRIC_DOT, MASKED>(params, allowed_slots, out)
                    }
                }
            }
            return;
        }
        if is_x86_feature_detected!("popcnt") {
            // Safety: runtime-checked POPCNT.
            unsafe {
                match params.metric {
                    DistanceMetric::L2Squared => {
                        x86::score_leaf_popcnt::<METRIC_L2, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::Cosine => {
                        x86::score_leaf_popcnt::<METRIC_COSINE, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::DotProduct => {
                        x86::score_leaf_popcnt::<METRIC_DOT, MASKED>(params, allowed_slots, out)
                    }
                }
            }
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        if cfg!(target_feature = "neon") || std::arch::is_aarch64_feature_detected!("neon") {
            // Safety: checked `neon` at compile/runtime; slices validated by caller and unaligned byte loads are used.
            unsafe {
                match params.metric {
                    DistanceMetric::L2Squared => {
                        aarch64::score_leaf_neon::<METRIC_L2, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::Cosine => {
                        aarch64::score_leaf_neon::<METRIC_COSINE, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::DotProduct => {
                        aarch64::score_leaf_neon::<METRIC_DOT, MASKED>(params, allowed_slots, out)
                    }
                }
            }
            return;
        }
    }
    match params.metric {
        DistanceMetric::L2Squared => {
            scalar::score_leaf_scalar::<METRIC_L2, MASKED>(params, allowed_slots, out)
        }
        DistanceMetric::Cosine => {
            scalar::score_leaf_scalar::<METRIC_COSINE, MASKED>(params, allowed_slots, out)
        }
        DistanceMetric::DotProduct => {
            scalar::score_leaf_scalar::<METRIC_DOT, MASKED>(params, allowed_slots, out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantizer::{
        EncodeScratch, QueryScratch, RabitqQuantizer, pack_query_bits_from_diff_bytes_into,
        pack_query_bits_into,
    };
    use izzydb_core::metric::{dot_product, l2_squared, prepare_vector};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_simd_leaf_scoring_matches_scalar_reference() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dim = 128;
        let qtz = RabitqQuantizer::new(dim, 777);
        let words = qtz.num_words();

        for metric in [
            DistanceMetric::L2Squared,
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
        ] {
            let raw_c: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let c = prepare_vector(&raw_c, dim, metric).unwrap();
            let raw_q: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let q = prepare_vector(&raw_q, dim, metric).unwrap();

            let rotated_c = qtz.rotate_new(&c);
            let rotated_c_bytes: Vec<u8> = rotated_c.iter().flat_map(|x| x.to_le_bytes()).collect();

            let n = 19usize;
            let padded_n = n.next_multiple_of(8);
            let mut scratch_enc = EncodeScratch::default();
            let mut entries = Vec::with_capacity(n);
            let mut doc_ids = vec![0u64; padded_n];
            let mut res_norm_sqs = vec![0.0f32; padded_n];
            let mut scales = vec![0.0f32; padded_n];
            let mut popcounts = vec![0u16; padded_n];
            let mut tiled_codes = vec![0u64; padded_n * words];

            for idx in 0..n {
                let doc_id = (idx as u64) * 10 + 1;
                let x = if idx == 5 {
                    c.clone()
                } else {
                    let raw_x: Vec<f32> =
                        c.iter().map(|&v| v + rng.random_range(-0.3..0.3)).collect();
                    prepare_vector(&raw_x, dim, metric).unwrap()
                };
                let mut code = vec![0u64; words];
                let (res_norm_sq, scale, popcnt) =
                    qtz.encode_into_with_popcnt(&x, &c, &mut scratch_enc, &mut code);
                doc_ids[idx] = doc_id;
                res_norm_sqs[idx] = res_norm_sq;
                scales[idx] = scale;
                popcounts[idx] = popcnt as u16;
                let tile = idx / 8;
                let lane = idx % 8;
                for (w, &word) in code.iter().enumerate() {
                    tiled_codes[(tile * words + w) * 8 + lane] = word;
                }
                entries.push((doc_id, res_norm_sq, scale, code));
            }

            let doc_ids_bytes: Vec<u8> = doc_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
            let codes_bytes: Vec<u8> = tiled_codes.iter().flat_map(|v| v.to_le_bytes()).collect();
            let pop_bytes: Vec<u8> = popcounts.iter().flat_map(|v| v.to_le_bytes()).collect();
            let res_norm_sqs_bytes: Vec<u8> =
                res_norm_sqs.iter().flat_map(|v| v.to_le_bytes()).collect();
            let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();

            let mut q_scratch = QueryScratch::new(qtz.padded_dim(), words);
            qtz.rotate_into(&q, &mut q_scratch.rot_q);

            let (q_res_norm_sq, zero_res_dist, q_meta) = match metric {
                DistanceMetric::L2Squared => {
                    let d2 = l2_squared(&q, &c);
                    let meta = pack_query_bits_from_diff_bytes_into(
                        &q_scratch.rot_q,
                        &rotated_c_bytes,
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
                        &rotated_c_bytes,
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
                    slot_cap: n,
                    rabitq_words: words,
                    popcount_u32: false,
                    doc_ids_bytes: &doc_ids_bytes,
                    codes_bytes: &codes_bytes,
                    code_popcounts_bytes: &pop_bytes,
                    res_norm_sqs_bytes: &res_norm_sqs_bytes,
                    scales_bytes: &scales_bytes,
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
            let mut masked_out = Vec::new();
            let mask: [u64; 1] = [(1u64 << 9) | (1u64 << 13) | (1u64 << 16) | (1u64 << 18)];
            score_leaf_rabitq_masked(
                &RabitqLeafScoreParams {
                    slot_cap: n,
                    rabitq_words: words,
                    popcount_u32: false,
                    doc_ids_bytes: &doc_ids_bytes,
                    codes_bytes: &codes_bytes,
                    code_popcounts_bytes: &pop_bytes,
                    res_norm_sqs_bytes: &res_norm_sqs_bytes,
                    scales_bytes: &scales_bytes,
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
                Some(&mask),
                &mut masked_out,
            );
            assert_eq!(masked_out.len(), 4);
            for (cand, slot) in masked_out.iter().zip([9usize, 13, 16, 18]) {
                assert_eq!(cand.doc_id, simd_out[slot].doc_id);
                assert!(
                    (cand.dist - simd_out[slot].dist).abs() < 1e-4,
                    "slot={slot}: masked={} vs unmasked={}",
                    cand.dist,
                    simd_out[slot].dist
                );
            }
        }
    }

    #[test]
    fn test_simd_leaf_scoring_multiword_mask_and_unaligned_slices() {
        let mut rng = SmallRng::seed_from_u64(99);
        let dim = 384;
        let qtz = RabitqQuantizer::new(dim, 1234);
        let words = qtz.num_words();

        let raw_c: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let c = prepare_vector(&raw_c, dim, DistanceMetric::Cosine).unwrap();
        let raw_q: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let q = prepare_vector(&raw_q, dim, DistanceMetric::Cosine).unwrap();
        let rotated_c = qtz.rotate_new(&c);
        let rotated_c_bytes: Vec<u8> = rotated_c.iter().flat_map(|x| x.to_le_bytes()).collect();

        let n = 70usize;
        let padded_n = n.next_multiple_of(8);
        let mut scratch_enc = EncodeScratch::default();
        let mut entries = Vec::with_capacity(n);
        let mut doc_ids = vec![0u64; padded_n];
        let mut res_norm_sqs = vec![0.0f32; padded_n];
        let mut scales = vec![0.0f32; padded_n];
        let mut popcounts = vec![0u16; padded_n];
        let mut tiled_codes = vec![0u64; padded_n * words];

        for idx in 0..n {
            let doc_id = (idx as u64) * 3 + 7;
            let raw_x: Vec<f32> = c.iter().map(|&v| v + rng.random_range(-0.3..0.3)).collect();
            let x = prepare_vector(&raw_x, dim, DistanceMetric::Cosine).unwrap();
            let mut code = vec![0u64; words];
            let (res_norm_sq, scale, popcnt) =
                qtz.encode_into_with_popcnt(&x, &c, &mut scratch_enc, &mut code);
            doc_ids[idx] = doc_id;
            res_norm_sqs[idx] = res_norm_sq;
            scales[idx] = scale;
            popcounts[idx] = popcnt as u16;
            let tile = idx / 8;
            let lane = idx % 8;
            for (w, &word) in code.iter().enumerate() {
                tiled_codes[(tile * words + w) * 8 + lane] = word;
            }
            entries.push((doc_id, res_norm_sq, scale, code));
        }

        let shift_bytes = |raw: Vec<u8>, shift: usize| -> Vec<u8> {
            let mut out = vec![0xA5u8; shift + raw.len() + 3];
            out[shift..shift + raw.len()].copy_from_slice(&raw);
            out
        };
        let doc_ids_buf = shift_bytes(doc_ids.iter().flat_map(|v| v.to_le_bytes()).collect(), 1);
        let codes_buf = shift_bytes(tiled_codes.iter().flat_map(|v| v.to_le_bytes()).collect(), 3);
        let pop_buf = shift_bytes(popcounts.iter().flat_map(|v| v.to_le_bytes()).collect(), 1);
        let res_buf = shift_bytes(
            res_norm_sqs.iter().flat_map(|v| v.to_le_bytes()).collect(),
            1,
        );
        let scales_buf = shift_bytes(scales.iter().flat_map(|v| v.to_le_bytes()).collect(), 2);

        let mut q_scratch = QueryScratch::new(qtz.padded_dim(), words);
        qtz.rotate_into(&q, &mut q_scratch.rot_q);
        let q_res_norm_sq = l2_squared(&q, &c);
        let zero_res_dist = (1.0 - dot_product(&q, &c)).max(0.0);
        let q_meta = pack_query_bits_from_diff_bytes_into(
            &q_scratch.rot_q,
            &rotated_c_bytes,
            &mut q_scratch.rot_diff,
            &mut q_scratch.planes,
        );

        let params = RabitqLeafScoreParams {
            slot_cap: n,
            rabitq_words: words,
            popcount_u32: false,
            doc_ids_bytes: &doc_ids_buf[1..1 + padded_n * 8],
            codes_bytes: &codes_buf[3..3 + padded_n * words * 8],
            code_popcounts_bytes: &pop_buf[1..1 + padded_n * 2],
            res_norm_sqs_bytes: &res_buf[1..1 + padded_n * 4],
            scales_bytes: &scales_buf[2..2 + padded_n * 4],
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
            metric: DistanceMetric::Cosine,
        };

        let mut out = Vec::new();
        score_leaf_rabitq(&params, &mut out);
        assert_eq!(out.len(), n);

        let ref_state = qtz.prepare_leaf_query(&q, &c, DistanceMetric::Cosine, None);
        for (cand, &(doc_id, res_norm_sq, scale, ref code)) in out.iter().zip(entries.iter()) {
            assert_eq!(cand.doc_id, doc_id);
            let ref_dist = ref_state.estimate_distance(res_norm_sq, scale, code);
            assert!(
                (cand.dist - ref_dist).abs() < 1e-3,
                "doc={doc_id}: simd={} vs ref={ref_dist}",
                cand.dist
            );
        }

        let kept_slots = [4usize, 7, 63, 64, 67, 69];
        let mut mask = [0u64; 2];
        for &slot in &kept_slots {
            mask[slot / 64] |= 1u64 << (slot % 64);
        }
        let mut masked_out = Vec::new();
        score_leaf_rabitq_masked(&params, Some(&mask), &mut masked_out);
        assert_eq!(masked_out.len(), kept_slots.len());
        for (cand, &slot) in masked_out.iter().zip(kept_slots.iter()) {
            assert_eq!(cand.doc_id, out[slot].doc_id);
            assert!((cand.dist - out[slot].dist).abs() < 1e-6);
        }
    }
}
