use super::scalar::score_single_entry_scalar;
use super::{LeafCandidate, METRIC_COSINE, METRIC_L2, RabitqLeafScoreParams};

#[target_feature(enable = "neon")]
pub(super) unsafe fn score_leaf_neon<const METRIC: u8, const MASKED: bool>(
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
            METRIC_L2 => {
                let d = vfmsq_f32(vaddq_f32(v_res_sq, v_q_res_sq), v_two, est_r_dot_q);
                vmaxq_f32(v_zero, d)
            }
            METRIC_COSINE => {
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
