#[inline(always)]
fn read_popcount_scalar(params: &super::RabitqLeafScoreParams<'_>, slot: usize) -> u32 {
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
pub(super) fn score_single_entry_scalar<const METRIC: u8>(
    params: &super::RabitqLeafScoreParams<'_>,
    i: usize,
) -> super::LeafCandidate {
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
            super::METRIC_L2 => (res_norm_sq + params.q_res_norm_sq - 2.0 * est_r_dot_q).max(0.0),
            super::METRIC_COSINE => {
                (0.5 * (res_norm_sq + params.q_res_norm_sq - 2.0 * est_r_dot_q)).max(0.0)
            }
            _ => params.zero_res_dist - est_r_dot_q,
        }
    };
    super::LeafCandidate { doc_id, dist }
}

#[inline]
pub(super) fn score_leaf_scalar<const METRIC: u8, const MASKED: bool>(
    params: &super::RabitqLeafScoreParams<'_>,
    allowed_slots: &[u64],
    out: &mut Vec<super::LeafCandidate>,
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
