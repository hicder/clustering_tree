use izzydb_core::config::{DistanceMetric, DocId};

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
                        score_leaf_avx512::<0, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::Cosine => {
                        score_leaf_avx512::<1, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::DotProduct => {
                        score_leaf_avx512::<2, MASKED>(params, allowed_slots, out)
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
                        score_leaf_popcnt::<0, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::Cosine => {
                        score_leaf_popcnt::<1, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::DotProduct => {
                        score_leaf_popcnt::<2, MASKED>(params, allowed_slots, out)
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
                        score_leaf_neon::<0, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::Cosine => {
                        score_leaf_neon::<1, MASKED>(params, allowed_slots, out)
                    }
                    DistanceMetric::DotProduct => {
                        score_leaf_neon::<2, MASKED>(params, allowed_slots, out)
                    }
                }
            }
            return;
        }
    }
    match params.metric {
        DistanceMetric::L2Squared => score_leaf_scalar::<0, MASKED>(params, allowed_slots, out),
        DistanceMetric::Cosine => score_leaf_scalar::<1, MASKED>(params, allowed_slots, out),
        DistanceMetric::DotProduct => score_leaf_scalar::<2, MASKED>(params, allowed_slots, out),
    }
}

#[inline(always)]
fn read_popcount_scalar(params: &RabitqLeafScoreParams<'_>, slot: usize) -> u32 {
    if params.popcount_u32 {
        let off = slot * 4;
        u32::from_le_bytes(
            params.code_popcounts_bytes[off..off + 4]
                .try_into()
                .unwrap(),
        )
    } else {
        let off = slot * 2;
        u16::from_le_bytes([
            params.code_popcounts_bytes[off],
            params.code_popcounts_bytes[off + 1],
        ]) as u32
    }
}

#[inline(always)]
fn score_single_entry_scalar<const METRIC: u8>(
    params: &RabitqLeafScoreParams<'_>,
    i: usize,
) -> LeafCandidate {
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
        let p_x = read_popcount_scalar(params, i);
        let tile = i >> 3;
        let lane = i & 7;
        let tile_base = tile * words * 8 + lane;
        let mut p0 = 0u32;
        let mut p1 = 0u32;
        let mut p2 = 0u32;
        let mut p3 = 0u32;
        for w in 0..words {
            let off = (tile_base + w * 8) * 8;
            let x = u64::from_le_bytes(params.codes_bytes[off..off + 8].try_into().unwrap());
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
fn score_leaf_scalar<const METRIC: u8, const MASKED: bool>(
    params: &RabitqLeafScoreParams<'_>,
    allowed_slots: &[u64],
    out: &mut Vec<LeafCandidate>,
) {
    let n = params.slot_cap;
    if !MASKED {
        out.reserve(n);
    }
    for i in 0..n {
        if !MASKED || ((allowed_slots[i / 64] >> (i % 64)) & 1 != 0) {
            out.push(score_single_entry_scalar::<METRIC>(params, i));
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "popcnt")]
unsafe fn score_leaf_popcnt<const METRIC: u8, const MASKED: bool>(
    params: &RabitqLeafScoreParams<'_>,
    allowed_slots: &[u64],
    out: &mut Vec<LeafCandidate>,
) {
    score_leaf_scalar::<METRIC, MASKED>(params, allowed_slots, out);
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
#[target_feature(enable = "avx512f,avx512vpopcntdq,avx2,fma")]
unsafe fn score_leaf_avx512<const METRIC: u8, const MASKED: bool>(
    params: &RabitqLeafScoreParams<'_>,
    allowed_slots: &[u64],
    out: &mut Vec<LeafCandidate>,
) {
    use std::arch::x86_64::*;

    let n = params.slot_cap;
    let padded_n = n.next_multiple_of(8);
    let words = params.rabitq_words;
    if !MASKED {
        out.reserve(n);
    }

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
    let pop_ptr = params.code_popcounts_bytes.as_ptr();
    let res_sq_ptr = params.res_norm_sqs_bytes.as_ptr() as *const f32;
    let scales_ptr = params.scales_bytes.as_ptr() as *const f32;
    let doc_ids_ptr = params.doc_ids_bytes.as_ptr();
    let pop_u32 = params.popcount_u32;

    let loop_end = if MASKED { padded_n } else { n & !7 };
    let mut i = 0usize;
    let mut tile = 0usize;
    while i < loop_end {
        let byte_mask = if MASKED {
            let m = ((allowed_slots[i / 64] >> (i % 64)) & 0xFF) as u8;
            if m == 0 {
                i += 8;
                tile += 1;
                continue;
            }
            m
        } else {
            0xFF
        };

        let mut acc_0 = _mm512_setzero_si512();
        let mut acc_1 = _mm512_setzero_si512();
        let mut acc_2 = _mm512_setzero_si512();
        let mut acc_3 = _mm512_setzero_si512();

        let tile_byte_base = tile * words * 64;
        for w in 0..words {
            unsafe {
                let x = _mm512_loadu_si512(codes_ptr.add(tile_byte_base + w * 64) as *const __m512i);
                let q0w = _mm512_set1_epi64(*params.q0.get_unchecked(w) as i64);
                let q1w = _mm512_set1_epi64(*params.q1.get_unchecked(w) as i64);
                let q2w = _mm512_set1_epi64(*params.q2.get_unchecked(w) as i64);
                let q3w = _mm512_set1_epi64(*params.q3.get_unchecked(w) as i64);

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

        let px_i32 = unsafe {
            if pop_u32 {
                _mm256_loadu_si256(pop_ptr.add(i * 4) as *const __m256i)
            } else {
                _mm256_cvtepu16_epi32(_mm_loadu_si128(pop_ptr.add(i * 2) as *const __m128i))
            }
        };
        let px_f = _mm256_cvtepi32_ps(px_i32);
        let wp_f = _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(weighted_i64));

        let sum_x_q = _mm256_fmadd_ps(v_delta, wp_f, _mm256_mul_ps(v_vl, px_f));
        let o_bar_dot_q = _mm256_mul_ps(_mm256_fmsub_ps(v_two, sum_x_q, v_sum_q), v_inv_sqrt);

        let v_scale = unsafe { _mm256_loadu_ps(scales_ptr.add(i)) };
        let v_res_sq = unsafe { _mm256_loadu_ps(res_sq_ptr.add(i)) };
        let est_r_dot_q = _mm256_mul_ps(v_scale, o_bar_dot_q);

        let raw_dist = match METRIC {
            0 => {
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

        if !MASKED || byte_mask == 0xFF {
            for lane in 0..8 {
                out.push(LeafCandidate {
                    doc_id: doc_ids[lane],
                    dist: dists[lane],
                });
            }
        } else {
            let mut m = byte_mask;
            while m != 0 {
                let lane = m.trailing_zeros() as usize;
                m &= m - 1;
                if i + lane < n {
                    out.push(LeafCandidate {
                        doc_id: doc_ids[lane],
                        dist: dists[lane],
                    });
                }
            }
        }

        i += 8;
        tile += 1;
    }

    if !MASKED {
        while i < n {
            out.push(score_single_entry_scalar::<METRIC>(params, i));
            i += 1;
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn score_leaf_neon<const METRIC: u8, const MASKED: bool>(
    params: &RabitqLeafScoreParams<'_>,
    allowed_slots: &[u64],
    out: &mut Vec<LeafCandidate>,
) {
    use std::arch::aarch64::*;

    let n = params.slot_cap;
    let padded_n = n.next_multiple_of(8);
    let words = params.rabitq_words;
    if !MASKED {
        out.reserve(n);
    }

    let v_vl = vdupq_n_f32(params.vl);
    let v_delta = vdupq_n_f32(params.delta);
    let v_two = vdupq_n_f32(2.0);
    let v_neg_sum_q = vdupq_n_f32(-params.sum_q);
    let v_inv_sqrt = vdupq_n_f32(params.inv_sqrt_padded_dim);
    let v_q_res_sq = vdupq_n_f32(params.q_res_norm_sq);
    let v_half = vdupq_n_f32(0.5);
    let v_zero = vdupq_n_f32(0.0);
    let v_zero_res_dist = vdupq_n_f32(params.zero_res_dist);

    let codes_ptr = params.codes_bytes.as_ptr();
    let pop_ptr = params.code_popcounts_bytes.as_ptr();
    let res_sq_ptr = params.res_norm_sqs_bytes.as_ptr();
    let scales_ptr = params.scales_bytes.as_ptr();
    let doc_ids_ptr = params.doc_ids_bytes.as_ptr();
    let pop_u32 = params.popcount_u32;

    let loop_end = if MASKED { padded_n } else { n & !3 };
    let mut i = 0usize;
    while i < loop_end {
        let nibble_mask = if MASKED {
            let m = ((allowed_slots[i / 64] >> (i % 64)) & 0xF) as u8;
            if m == 0 {
                i += 4;
                continue;
            }
            m
        } else {
            0xF
        };

        let mut total_0 = vdupq_n_u32(0);
        let mut total_1 = vdupq_n_u32(0);
        let mut total_2 = vdupq_n_u32(0);
        let mut total_3 = vdupq_n_u32(0);

        let tile = i >> 3;
        let half_off = (i & 4) * 8; // 0 for lanes 0..4, 32 for lanes 4..8
        let tile_byte_base = tile * words * 64 + half_off;

        let mut w_start = 0usize;
        while w_start < words {
            let w_end = (w_start + 1024).min(words);
            let mut acc_0_lo = vdupq_n_u16(0);
            let mut acc_1_lo = vdupq_n_u16(0);
            let mut acc_2_lo = vdupq_n_u16(0);
            let mut acc_3_lo = vdupq_n_u16(0);

            let mut acc_0_hi = vdupq_n_u16(0);
            let mut acc_1_hi = vdupq_n_u16(0);
            let mut acc_2_hi = vdupq_n_u16(0);
            let mut acc_3_hi = vdupq_n_u16(0);

            for w in w_start..w_end {
                unsafe {
                    let base = tile_byte_base + w * 64;
                    let x_lo = vld1q_u8(codes_ptr.add(base));
                    let x_hi = vld1q_u8(codes_ptr.add(base + 16));

                    let q0w = vreinterpretq_u8_u64(vdupq_n_u64(*params.q0.get_unchecked(w)));
                    let q1w = vreinterpretq_u8_u64(vdupq_n_u64(*params.q1.get_unchecked(w)));
                    let q2w = vreinterpretq_u8_u64(vdupq_n_u64(*params.q2.get_unchecked(w)));
                    let q3w = vreinterpretq_u8_u64(vdupq_n_u64(*params.q3.get_unchecked(w)));

                    acc_0_lo = vpadalq_u8(acc_0_lo, vcntq_u8(vandq_u8(x_lo, q0w)));
                    acc_1_lo = vpadalq_u8(acc_1_lo, vcntq_u8(vandq_u8(x_lo, q1w)));
                    acc_2_lo = vpadalq_u8(acc_2_lo, vcntq_u8(vandq_u8(x_lo, q2w)));
                    acc_3_lo = vpadalq_u8(acc_3_lo, vcntq_u8(vandq_u8(x_lo, q3w)));

                    acc_0_hi = vpadalq_u8(acc_0_hi, vcntq_u8(vandq_u8(x_hi, q0w)));
                    acc_1_hi = vpadalq_u8(acc_1_hi, vcntq_u8(vandq_u8(x_hi, q1w)));
                    acc_2_hi = vpadalq_u8(acc_2_hi, vcntq_u8(vandq_u8(x_hi, q2w)));
                    acc_3_hi = vpadalq_u8(acc_3_hi, vcntq_u8(vandq_u8(x_hi, q3w)));
                }
            }

            total_0 = vaddq_u32(total_0, vpaddlq_u16(vpaddq_u16(acc_0_lo, acc_0_hi)));
            total_1 = vaddq_u32(total_1, vpaddlq_u16(vpaddq_u16(acc_1_lo, acc_1_hi)));
            total_2 = vaddq_u32(total_2, vpaddlq_u16(vpaddq_u16(acc_2_lo, acc_2_hi)));
            total_3 = vaddq_u32(total_3, vpaddlq_u16(vpaddq_u16(acc_3_lo, acc_3_hi)));
            w_start = w_end;
        }

        let w1 = vshlq_n_u32::<1>(total_1);
        let w2 = vshlq_n_u32::<2>(total_2);
        let w3 = vshlq_n_u32::<3>(total_3);
        let weighted_u32 = vaddq_u32(vaddq_u32(total_0, w1), vaddq_u32(w2, w3));

        let total_x = unsafe {
            if pop_u32 {
                vreinterpretq_u32_u8(vld1q_u8(pop_ptr.add(i * 4)))
            } else {
                vmovl_u16(vreinterpret_u16_u8(vld1_u8(pop_ptr.add(i * 2))))
            }
        };
        let px_f = vcvtq_f32_u32(total_x);
        let wp_f = vcvtq_f32_u32(weighted_u32);

        let sum_x_q = vfmaq_f32(vmulq_f32(v_vl, px_f), v_delta, wp_f);
        let o_bar_dot_q = vmulq_f32(vfmaq_f32(v_neg_sum_q, v_two, sum_x_q), v_inv_sqrt);

        let v_scale = unsafe { vreinterpretq_f32_u8(vld1q_u8(scales_ptr.add(i * 4))) };
        let v_res_sq = unsafe { vreinterpretq_f32_u8(vld1q_u8(res_sq_ptr.add(i * 4))) };
        let est_r_dot_q = vmulq_f32(v_scale, o_bar_dot_q);

        let raw_dist = match METRIC {
            0 => {
                let d = vfmsq_f32(vaddq_f32(v_res_sq, v_q_res_sq), v_two, est_r_dot_q);
                vmaxq_f32(v_zero, d)
            }
            1 => {
                let d = vfmsq_f32(vaddq_f32(v_res_sq, v_q_res_sq), v_two, est_r_dot_q);
                vmaxq_f32(v_zero, vmulq_f32(v_half, d))
            }
            _ => vsubq_f32(v_zero_res_dist, est_r_dot_q),
        };

        let zero_mask = vceqq_f32(v_res_sq, v_zero);
        let final_dist = vbslq_f32(zero_mask, v_zero_res_dist, raw_dist);

        let mut dists = [0.0f32; 4];
        let mut doc_ids = [0u64; 4];
        unsafe {
            vst1q_f32(dists.as_mut_ptr(), final_dist);
            std::ptr::copy_nonoverlapping(doc_ids_ptr.add(i * 8), doc_ids.as_mut_ptr() as *mut u8, 32);
        }

        if !MASKED || nibble_mask == 0xF {
            for lane in 0..4 {
                out.push(LeafCandidate {
                    doc_id: doc_ids[lane],
                    dist: dists[lane],
                });
            }
        } else {
            for lane in 0..4 {
                if (nibble_mask >> lane) & 1 != 0 && i + lane < n {
                    out.push(LeafCandidate {
                        doc_id: doc_ids[lane],
                        dist: dists[lane],
                    });
                }
            }
        }

        i += 4;
    }

    if !MASKED {
        while i < n {
            out.push(score_single_entry_scalar::<METRIC>(params, i));
            i += 1;
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
