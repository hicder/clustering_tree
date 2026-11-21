use super::scalar::{score_leaf_scalar, score_single_entry_scalar};
use super::{LeafCandidate, METRIC_COSINE, METRIC_L2, RabitqLeafScoreParams};

#[target_feature(enable = "popcnt")]
pub(super) unsafe fn score_leaf_popcnt<const METRIC: u8, const MASKED: bool>(
    params: &RabitqLeafScoreParams<'_>,
    allowed_slots: &[u64],
    out: &mut Vec<LeafCandidate>,
) {
    score_leaf_scalar::<METRIC, MASKED>(params, allowed_slots, out);
}

#[target_feature(enable = "avx512f,avx512vpopcntdq,avx2,fma")]
pub(super) unsafe fn score_leaf_avx512<const METRIC: u8, const MASKED: bool>(
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
            METRIC_L2 => {
                let d = _mm256_fnmadd_ps(v_two, est_r_dot_q, _mm256_add_ps(v_res_sq, v_q_res_sq));
                _mm256_max_ps(v_zero, d)
            }
            METRIC_COSINE => {
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
