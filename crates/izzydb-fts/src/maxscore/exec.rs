//! Block-Max MAXSCORE execution (single-threaded and segment-parallel).

use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU32, Ordering};

use alyze::analyze::ReusableBuffer;
use rayon::prelude::*;

use izzydb_core::config::{
    DocId, FTS_BLOCK_SIZE, RowBlock, RowId, for_each_set_bit_256,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rank::RankBy;
use izzydb_core::schema::CollectionSchema;

use crate::simd::{
    SimdTier, filter_ge_256, max_score_256,
};

use super::*;

/// Execute a validated pure-lexical [`RankBy`] query using SIMD Vectorized Block-Max MAXSCORE.
pub fn execute_rank_by<S: FtsPostingSource>(
    source: &S,
    schema: &CollectionSchema,
    rank_by: &RankBy,
    top_k: usize,
    block_mask: Option<&dyn BlockMaskProvider>,
    lazy_filter: Option<&mut dyn FnMut(RowId) -> Result<bool>>,
    buffer: &mut ReusableBuffer,
) -> Result<Vec<Bm25Result>> {
    execute_rank_by_with(
        source,
        schema,
        rank_by,
        top_k,
        block_mask,
        lazy_filter,
        None,
        buffer,
    )
}

/// [`execute_rank_by`] with an optional [`RankModifier`] (multiplicative and/or additive
/// per-row factors evaluated lazily on surviving candidates).
#[allow(clippy::too_many_arguments)]
pub fn execute_rank_by_with<S: FtsPostingSource>(
    source: &S,
    schema: &CollectionSchema,
    rank_by: &RankBy,
    top_k: usize,
    block_mask: Option<&dyn BlockMaskProvider>,
    lazy_filter: Option<&mut dyn FnMut(RowId) -> Result<bool>>,
    modifier: Option<RankModifier<'_>>,
    buffer: &mut ReusableBuffer,
) -> Result<Vec<Bm25Result>> {
    execute_rank_by_with_shards(
        source,
        schema,
        rank_by,
        top_k,
        block_mask,
        lazy_filter,
        modifier,
        None,
        buffer,
    )
}

/// [`execute_rank_by_with`] with an explicit cap on parallel segment shards (`None` uses all
/// Rayon worker threads; `Some(1)` forces single-threaded execution).
#[allow(clippy::too_many_arguments)]
pub fn execute_rank_by_with_shards<S: FtsPostingSource>(
    source: &S,
    schema: &CollectionSchema,
    rank_by: &RankBy,
    top_k: usize,
    block_mask: Option<&dyn BlockMaskProvider>,
    mut lazy_filter: Option<&mut dyn FnMut(RowId) -> Result<bool>>,
    modifier: Option<RankModifier<'_>>,
    max_shards: Option<usize>,
    buffer: &mut ReusableBuffer,
) -> Result<Vec<Bm25Result>> {
    rank_by.validate(schema)?;
    let rank_by = rank_by.normalize();
    if !rank_by.is_pure_lexical() {
        return Err(not_lexical_err());
    }
    let (mult_ub, add_ub) = match &modifier {
        Some(m) => {
            if !(m.mult_ub.is_finite() && m.mult_ub > 0.0 && m.add_ub.is_finite() && m.add_ub >= 0.0)
            {
                return Err(TreeError::InvalidQuery(
                    "RankModifier bounds must be finite (mult_ub > 0, add_ub >= 0)".into(),
                ));
            }
            (m.mult_ub, m.add_ub)
        }
        None => (1.0, 0.0),
    };
    let mut rescore_fn = modifier.map(|m| m.rescore);
    if top_k == 0 {
        return Ok(Vec::new());
    }
    let Some(meta) = source.get_fts_meta()? else {
        return Ok(Vec::new());
    };

    let tier = SimdTier::detect();
    let plan = compile_plan(source, schema, &meta, &rank_by, mult_ub, buffer)?;
    if plan.clauses.is_empty() {
        return Ok(Vec::new());
    }

    let shard_budget = max_shards.unwrap_or_else(rayon::current_num_threads);
    if lazy_filter.is_none()
        && rescore_fn.is_none()
        && shard_budget > 1
        && let Some(ranges) = source.segment_ranges(shard_budget)
        && ranges.len() >= 2
    {
        plan.prewarm_doc_lens(source)?;
        let shared_theta = AtomicU32::new(0);
        let shard_heaps: Vec<BinaryHeap<TopKEntry>> = ranges
            .par_iter()
            .map(|&(seg_start, seg_end)| -> Result<BinaryHeap<TopKEntry>> {
                let (mut clauses, prefix_max_sum) =
                    plan.instantiate(source, Some((seg_start, seg_end)))?;
                if clauses.is_empty() {
                    return Ok(BinaryHeap::new());
                }
                let mut dl_cache = DlNormCache::from_specs(&plan.dl_specs);
                if clauses
                    .iter()
                    .all(|c| matches!(c.kind, ClauseKind::Single(_)))
                {
                    let mut streams = into_single_streams(clauses)
                        .expect("all-single plan must convert to term streams");
                    run_shard_single(
                        source,
                        &mut streams,
                        &prefix_max_sum,
                        &mut dl_cache,
                        top_k,
                        add_ub,
                        block_mask,
                        &mut None,
                        None,
                        Some(&shared_theta),
                        tier,
                    )
                } else {
                    run_shard(
                        source,
                        &mut clauses,
                        &prefix_max_sum,
                        &mut dl_cache,
                        top_k,
                        add_ub,
                        block_mask,
                        &mut None,
                        None,
                        Some(&shared_theta),
                        tier,
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let merged = merge_shard_heaps(source, shard_heaps, top_k)?;
        return resolve_and_sort_results(source, merged);
    }

    let (mut clauses, prefix_max_sum) = plan.instantiate(source, None)?;
    if clauses.is_empty() {
        return Ok(Vec::new());
    }
    let mut dl_cache = DlNormCache::from_specs(&plan.dl_specs);
    let heap = if clauses
        .iter()
        .all(|c| matches!(c.kind, ClauseKind::Single(_)))
    {
        let mut streams =
            into_single_streams(clauses).expect("all-single plan must convert to term streams");
        run_shard_single(
            source,
            &mut streams,
            &prefix_max_sum,
            &mut dl_cache,
            top_k,
            add_ub,
            block_mask,
            &mut lazy_filter,
            rescore_fn.as_deref_mut(),
            None,
            tier,
        )?
    } else {
        run_shard(
            source,
            &mut clauses,
            &prefix_max_sum,
            &mut dl_cache,
            top_k,
            add_ub,
            block_mask,
            &mut lazy_filter,
            rescore_fn,
            None,
            tier,
        )?
    };

    resolve_and_sort_results(source, heap)
}

#[allow(clippy::too_many_arguments)]
fn run_shard<'a, 'r, S: FtsPostingSource>(
    source: &'a S,
    clauses: &mut [Clause<'a, S>],
    prefix_max_sum: &[f32],
    dl_cache: &mut DlNormCache<'a, S>,
    top_k: usize,
    add_ub: f32,
    block_mask: Option<&dyn BlockMaskProvider>,
    lazy_filter: &mut Option<&mut dyn FnMut(RowId) -> Result<bool>>,
    mut rescore: Option<&mut (dyn FnMut(RowId, f32) -> Result<f32> + 'r)>,
    shared_theta: Option<&AtomicU32>,
    tier: SimdTier,
) -> Result<BinaryHeap<TopKEntry>> {
    let n_clauses = clauses.len();
    let mut essential_start: usize = 0;
    let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(top_k + 1);
    let mut theta: f32 = 0.0;

    let mut scores = [0.0f32; FTS_BLOCK_SIZE];
    let mut touched: [u64; 4];
    let mut candidates: Vec<(f32, RowId)> = Vec::with_capacity(32);
    let mut non_ess_ubs: Vec<f32> = Vec::with_capacity(n_clauses);

    while let Some(block_id) = next_essential_block(&clauses[essential_start..]) {
        let non_ess_ub = if essential_start > 0 {
            prefix_max_sum[essential_start - 1]
        } else {
            0.0
        };
        let global_theta = shared_theta.map_or(0.0, |st| f32::from_bits(st.load(Ordering::Relaxed)));
        let eff_theta = theta.max(global_theta);
        let prune_theta = if eff_theta > add_ub {
            eff_theta - add_ub
        } else {
            0.0
        };

        // Chunk-level skip when only one essential clause remains.
        if essential_start + 1 == n_clauses
            && prune_theta > 0.0
            && let Some((chunk, chunk_ub)) = clauses[essential_start].single_chunk_upper_bound()
            && ub_below_threshold(chunk_ub + non_ess_ub, prune_theta)
        {
            clauses[essential_start].seek_past_single_chunk(chunk)?;
            continue;
        }

        let allowed_mask = match block_mask {
            Some(mp) => match mp.mask(block_id) {
                Some([0, 0, 0, 0]) => {
                    advance_clauses_past(&mut clauses[essential_start..], block_id)?;
                    continue;
                }
                other => other,
            },
            None => None,
        };

        // Block-Max skip pruning BEFORE reading any posting blocks or doc_len blocks.
        // Lexical-scale pruning threshold: a doc needs `L * mult_ub + add_ub > theta`.
        if prune_theta > 0.0
            && essential_block_pruned(&clauses[essential_start..], block_id, non_ess_ub, prune_theta)
        {
            advance_clauses_past(&mut clauses[essential_start..], block_id)?;
            continue;
        }

        // Accumulate essential clauses into L1-resident [f32; 256] buffer via SIMD,
        // and advance each processed cursor past `block_id` before shrinking essential_start.
        scores.fill(0.0);
        touched = [0u64; 4];
        for c in &mut clauses[essential_start..] {
            if c.next_block_id() == Some(block_id) {
                c.accumulate_block(block_id, &mut scores, &mut touched, dl_cache, source, tier)?;
                c.advance_past(block_id)?;
            }
        }

        if let Some(mask) = allowed_mask {
            for w in 0..4 {
                touched[w] &= mask[w];
            }
        }

        let active_mask = if touched == [0u64; 4] || essential_start == 0 {
            touched
        } else {
            eval_non_essential_clauses(
                &mut clauses[..essential_start],
                &prefix_max_sum[..essential_start],
                block_id,
                touched,
                non_ess_ub,
                prune_theta,
                &mut scores,
                &mut non_ess_ubs,
                dl_cache,
                source,
                tier,
            )?
        };

        // SIMD threshold filtering for top-k candidates.
        if active_mask != [0u64; 4] {
            let cand_thresh = if heap.len() < top_k && global_theta == 0.0 {
                f32::MIN_POSITIVE
            } else {
                prune_theta.max(f32::MIN_POSITIVE)
            };
            let hit_mask = filter_ge_256(tier, &scores, &active_mask, cand_thresh);
            if hit_mask != [0u64; 4] {
                collect_block_candidates(block_id, &hit_mask, &scores, &mut candidates);
                if !candidates.is_empty() {
                    theta = insert_candidates_into_heap(
                        source,
                        &candidates,
                        top_k,
                        theta,
                        global_theta,
                        &mut heap,
                        lazy_filter,
                        rescore.as_deref_mut(),
                    )?;
                    if let Some(st) = shared_theta
                        && heap.len() == top_k
                        && theta > global_theta
                    {
                        st.fetch_max(theta.to_bits(), Ordering::Relaxed);
                    }
                }
            }
        }

        // Dynamically shrink essential set as theta rises.
        let global_theta = shared_theta.map_or(0.0, |st| f32::from_bits(st.load(Ordering::Relaxed)));
        let eff_theta = theta.max(global_theta);
        let prune_theta = if eff_theta > add_ub {
            eff_theta - add_ub
        } else {
            0.0
        };
        if (heap.len() == top_k || global_theta > 0.0) && prune_theta > 0.0 {
            while essential_start < n_clauses
                && ub_below_threshold(prefix_max_sum[essential_start], prune_theta)
            {
                essential_start += 1;
            }
            if essential_start >= n_clauses {
                break;
            }
        }
    }

    Ok(heap)
}

/// Move compiled clauses into plain term streams when every live clause is a single term.
fn into_single_streams<'a, S: FtsPostingSource + 'a>(
    clauses: Vec<Clause<'a, S>>,
) -> Option<Vec<TermStream<'a, S>>> {
    clauses
        .into_iter()
        .map(|c| match c.kind {
            ClauseKind::Single(t) => Some(t),
            _ => None,
        })
        .collect()
}

#[inline]
fn refresh_stream_state<'a, S: FtsPostingSource>(
    stream: &TermStream<'a, S>,
    block: &mut Option<RowBlock>,
    bound: &mut f32,
) {
    let cur = stream.current_block_id();
    *block = cur;
    *bound = if cur.is_some() {
        stream.block_upper_bound()
    } else {
        0.0
    };
}

#[inline]
fn min_cached_block(blocks: &[Option<RowBlock>]) -> Option<RowBlock> {
    blocks.iter().filter_map(|b| *b).min()
}

/// Specialized MAXSCORE loop for plans made up only of single-term clauses.
///
/// Caches each stream's current block and Block-Max bound in shard-local arrays so the
/// next-block scan, bound sum, skip advance, and score pass each touch the essential
/// clauses once per block instead of re-querying every cursor several times.
#[allow(clippy::too_many_arguments)]
fn run_shard_single<'a, 'r, S: FtsPostingSource>(
    source: &'a S,
    streams: &mut [TermStream<'a, S>],
    prefix_max_sum: &[f32],
    dl_cache: &mut DlNormCache<'a, S>,
    top_k: usize,
    add_ub: f32,
    block_mask: Option<&dyn BlockMaskProvider>,
    lazy_filter: &mut Option<&mut dyn FnMut(RowId) -> Result<bool>>,
    mut rescore: Option<&mut (dyn FnMut(RowId, f32) -> Result<f32> + 'r)>,
    shared_theta: Option<&AtomicU32>,
    tier: SimdTier,
) -> Result<BinaryHeap<TopKEntry>> {
    let n_clauses = streams.len();
    let mut essential_start: usize = 0;
    let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(top_k + 1);
    let mut theta: f32 = 0.0;

    let mut scores = [0.0f32; FTS_BLOCK_SIZE];
    let mut touched: [u64; 4];
    let mut candidates: Vec<(f32, RowId)> = Vec::with_capacity(32);
    let mut non_ess_ubs: Vec<f32> = Vec::with_capacity(n_clauses);

    let mut blocks: Vec<Option<RowBlock>> = Vec::with_capacity(n_clauses);
    let mut bounds: Vec<f32> = Vec::with_capacity(n_clauses);
    for stream in streams.iter() {
        let mut block = None;
        let mut bound = 0.0;
        refresh_stream_state(stream, &mut block, &mut bound);
        blocks.push(block);
        bounds.push(bound);
    }
    let mut matching: Vec<usize> = Vec::with_capacity(n_clauses);

    while let Some(block_id) = min_cached_block(&blocks[essential_start..]) {
        let non_ess_ub = if essential_start > 0 {
            prefix_max_sum[essential_start - 1]
        } else {
            0.0
        };
        let global_theta = shared_theta.map_or(0.0, |st| f32::from_bits(st.load(Ordering::Relaxed)));
        let eff_theta = theta.max(global_theta);
        let prune_theta = if eff_theta > add_ub {
            eff_theta - add_ub
        } else {
            0.0
        };

        // Chunk-level skip when only one essential clause remains.
        if essential_start + 1 == n_clauses
            && prune_theta > 0.0
            && let Some((chunk, chunk_ub)) = streams[essential_start].chunk_upper_bound()
            && ub_below_threshold(chunk_ub + non_ess_ub, prune_theta)
        {
            streams[essential_start].seek_past_chunk(chunk)?;
            refresh_stream_state(
                &streams[essential_start],
                &mut blocks[essential_start],
                &mut bounds[essential_start],
            );
            continue;
        }

        let allowed_mask = match block_mask {
            Some(mp) => match mp.mask(block_id) {
                Some([0, 0, 0, 0]) => {
                    for i in essential_start..n_clauses {
                        if blocks[i] == Some(block_id) {
                            streams[i].advance_past(block_id)?;
                            refresh_stream_state(&streams[i], &mut blocks[i], &mut bounds[i]);
                        }
                    }
                    continue;
                }
                other => other,
            },
            None => None,
        };

        // One pass over the essential clauses finds the matching block and sums its bounds.
        matching.clear();
        let mut ess_block_ub = 0.0f64;
        for i in essential_start..n_clauses {
            if blocks[i] == Some(block_id) {
                matching.push(i);
                ess_block_ub += bounds[i] as f64;
            }
        }

        if prune_theta > 0.0
            && ub_below_threshold((ess_block_ub + non_ess_ub as f64) as f32, prune_theta)
        {
            for &i in &matching {
                streams[i].advance_past(block_id)?;
                refresh_stream_state(&streams[i], &mut blocks[i], &mut bounds[i]);
            }
            continue;
        }

        scores.fill(0.0);
        touched = [0u64; 4];
        for &i in &matching {
            let stream = &mut streams[i];
            if !stream.scorer.is_zero() {
                stream.accumulate_into(
                    block_id,
                    AccumMode::Unmasked {
                        touched: &mut touched,
                    },
                    &mut scores,
                    dl_cache,
                    source,
                    tier,
                )?;
            }
            stream.advance_past(block_id)?;
            refresh_stream_state(&streams[i], &mut blocks[i], &mut bounds[i]);
        }

        if let Some(mask) = allowed_mask {
            for w in 0..4 {
                touched[w] &= mask[w];
            }
        }

        let active_mask = if touched == [0u64; 4] || essential_start == 0 {
            touched
        } else {
            eval_non_essential_streams(
                &mut streams[..essential_start],
                &mut blocks[..essential_start],
                &mut bounds[..essential_start],
                &prefix_max_sum[..essential_start],
                block_id,
                touched,
                non_ess_ub,
                prune_theta,
                &mut scores,
                &mut non_ess_ubs,
                dl_cache,
                source,
                tier,
            )?
        };

        if active_mask != [0u64; 4] {
            let cand_thresh = if heap.len() < top_k && global_theta == 0.0 {
                f32::MIN_POSITIVE
            } else {
                prune_theta.max(f32::MIN_POSITIVE)
            };
            let hit_mask = filter_ge_256(tier, &scores, &active_mask, cand_thresh);
            if hit_mask != [0u64; 4] {
                collect_block_candidates(block_id, &hit_mask, &scores, &mut candidates);
                if !candidates.is_empty() {
                    theta = insert_candidates_into_heap(
                        source,
                        &candidates,
                        top_k,
                        theta,
                        global_theta,
                        &mut heap,
                        lazy_filter,
                        rescore.as_deref_mut(),
                    )?;
                    if let Some(st) = shared_theta
                        && heap.len() == top_k
                        && theta > global_theta
                    {
                        st.fetch_max(theta.to_bits(), Ordering::Relaxed);
                    }
                }
            }
        }

        let global_theta = shared_theta.map_or(0.0, |st| f32::from_bits(st.load(Ordering::Relaxed)));
        let eff_theta = theta.max(global_theta);
        let prune_theta = if eff_theta > add_ub {
            eff_theta - add_ub
        } else {
            0.0
        };
        if (heap.len() == top_k || global_theta > 0.0) && prune_theta > 0.0 {
            while essential_start < n_clauses
                && ub_below_threshold(prefix_max_sum[essential_start], prune_theta)
            {
                essential_start += 1;
            }
            if essential_start >= n_clauses {
                break;
            }
        }
    }

    Ok(heap)
}

/// Single-term counterpart of [`eval_non_essential_clauses`] using cached block/bound arrays.
#[allow(clippy::too_many_arguments)]
fn eval_non_essential_streams<'a, S: FtsPostingSource + 'a>(
    non_essential: &mut [TermStream<'a, S>],
    blocks: &mut [Option<RowBlock>],
    bounds: &mut [f32],
    prefix_max_sum: &[f32],
    block_id: RowBlock,
    mut active_mask: [u64; 4],
    non_ess_ub: f32,
    theta: f32,
    scores: &mut [f32; FTS_BLOCK_SIZE],
    non_ess_ubs: &mut Vec<f32>,
    dl_cache: &mut DlNormCache<'a, S>,
    source: &'a S,
    tier: SimdTier,
) -> Result<[u64; 4]> {
    let max_ess = max_score_256(tier, scores, &active_mask);
    if ub_below_threshold(max_ess + non_ess_ub, theta) {
        return Ok([0u64; 4]);
    }

    non_ess_ubs.clear();
    let mut non_ess_block_ub = 0.0f64;
    for j in (0..non_essential.len()).rev() {
        non_essential[j].seek(block_id)?;
        let cur = non_essential[j].current_block_id();
        blocks[j] = cur;
        let ub = if cur == Some(block_id) {
            non_essential[j].block_upper_bound()
        } else {
            0.0
        };
        bounds[j] = ub;
        non_ess_ubs.push(ub);
        non_ess_block_ub += ub as f64;
        if theta > 0.0 {
            let rem_global = if j > 0 { prefix_max_sum[j - 1] } else { 0.0 };
            let upper = (max_ess as f64 + non_ess_block_ub + rem_global as f64) as f32;
            if ub_below_threshold(upper, theta) {
                return Ok([0u64; 4]);
            }
        }
    }

    let tight_ub = (max_ess as f64 + non_ess_block_ub) as f32;
    let tight_eps = ub_slack(tight_ub);
    let mut rem_ub = non_ess_block_ub;
    for (idx, j) in (0..non_essential.len()).rev().enumerate() {
        let ub_j = non_ess_ubs[idx];
        if ub_j <= 0.0 {
            continue;
        }
        if theta > 0.0 {
            let min_needed = (theta - rem_ub as f32) - tight_eps;
            if min_needed > 0.0 {
                active_mask = filter_ge_256(tier, scores, &active_mask, min_needed);
                if active_mask == [0u64; 4] {
                    break;
                }
            }
        }
        if blocks[j] == Some(block_id) && !non_essential[j].scorer.is_zero() {
            non_essential[j].accumulate_into(
                block_id,
                AccumMode::Masked { mask: &active_mask },
                scores,
                dl_cache,
                source,
                tier,
            )?;
        }
        rem_ub = (rem_ub - ub_j as f64).max(0.0);
    }
    Ok(active_mask)
}

fn merge_shard_heaps<S: FtsPostingSource>(
    source: &S,
    shard_heaps: Vec<BinaryHeap<TopKEntry>>,
    top_k: usize,
) -> Result<BinaryHeap<TopKEntry>> {
    let mut all_entries: Vec<TopKEntry> = Vec::new();
    for h in shard_heaps {
        all_entries.extend(h);
    }
    if all_entries.len() <= top_k {
        return Ok(all_entries.into_iter().collect());
    }
    all_entries.sort_by(|a, b| b.score.total_cmp(&a.score));
    let cutoff = all_entries[top_k - 1].score;
    let first_tie = all_entries[..top_k]
        .iter()
        .position(|e| e.score == cutoff)
        .unwrap_or(top_k - 1);
    let last_tie = all_entries[top_k..]
        .iter()
        .rposition(|e| e.score == cutoff)
        .map_or(top_k - 1, |pos| top_k + pos);

    let mut out: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(top_k);
    for e in all_entries.drain(..first_tie) {
        out.push(e);
    }
    let remaining_slots = top_k - out.len();
    let tie_count = last_tie - first_tie + 1;
    let mut ties: Vec<TopKEntry> = all_entries.drain(..tie_count).collect();
    let missing_rows: Vec<RowId> = ties
        .iter()
        .filter(|e| e.doc_id.is_none())
        .map(|e| e.row_id)
        .collect();
    if !missing_rows.is_empty() {
        let fetched = source.multi_get_doc_ids(&missing_rows)?;
        let map: HashMap<RowId, DocId> = missing_rows
            .into_iter()
            .zip(fetched)
            .filter_map(|(rid, did)| did.map(|d| (rid, d)))
            .collect();
        for e in &mut ties {
            if e.doc_id.is_none() {
                e.doc_id = map.get(&e.row_id).copied();
            }
        }
    }
    ties.retain(|e| e.doc_id.is_some());
    ties.sort_by_key(|e| e.doc_id.unwrap());
    for e in ties.into_iter().take(remaining_slots) {
        out.push(e);
    }
    Ok(out)
}

#[inline]
pub(super) fn next_essential_block<'a, S: FtsPostingSource + 'a>(
    essential: &[Clause<'a, S>],
) -> Option<RowBlock> {
    essential
        .iter()
        .filter_map(Clause::next_block_id)
        .min()
}

#[inline]
pub(super) fn advance_clauses_past<'a, S: FtsPostingSource + 'a>(
    clauses: &mut [Clause<'a, S>],
    block_id: RowBlock,
) -> Result<()> {
    for c in clauses {
        c.advance_past(block_id)?;
    }
    Ok(())
}

#[inline]
pub(super) fn essential_block_pruned<'a, S: FtsPostingSource + 'a>(
    essential: &[Clause<'a, S>],
    block_id: RowBlock,
    non_ess_ub: f32,
    theta: f32,
) -> bool {
    let ess_block_ub: f64 = essential
        .iter()
        .map(|c| c.block_upper_bound(block_id) as f64)
        .sum();
    let block_ub = (ess_block_ub + non_ess_ub as f64) as f32;
    ub_below_threshold(block_ub, theta)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn eval_non_essential_clauses<'a, S: FtsPostingSource + 'a>(
    non_essential: &mut [Clause<'a, S>],
    prefix_max_sum: &[f32],
    block_id: RowBlock,
    mut active_mask: [u64; 4],
    non_ess_ub: f32,
    theta: f32,
    scores: &mut [f32; FTS_BLOCK_SIZE],
    non_ess_ubs: &mut Vec<f32>,
    dl_cache: &mut DlNormCache<'a, S>,
    source: &'a S,
    tier: SimdTier,
) -> Result<[u64; 4]> {
    let max_ess = max_score_256(tier, scores, &active_mask);
    if ub_below_threshold(max_ess + non_ess_ub, theta) {
        return Ok([0u64; 4]);
    }

    non_ess_ubs.clear();
    let mut non_ess_block_ub = 0.0f64;
    // Seek descending by global term_max using out-of-line skip cursors only.
    for j in (0..non_essential.len()).rev() {
        non_essential[j].seek_to(block_id)?;
        let ub = non_essential[j].block_upper_bound(block_id);
        non_ess_ubs.push(ub);
        non_ess_block_ub += ub as f64;
        if theta > 0.0 {
            let rem_global = if j > 0 { prefix_max_sum[j - 1] } else { 0.0 };
            let upper = (max_ess as f64 + non_ess_block_ub + rem_global as f64) as f32;
            if ub_below_threshold(upper, theta) {
                return Ok([0u64; 4]);
            }
        }
    }

    let tight_ub = (max_ess as f64 + non_ess_block_ub) as f32;
    let tight_eps = ub_slack(tight_ub);
    let mut rem_ub = non_ess_block_ub;
    for (idx, j) in (0..non_essential.len()).rev().enumerate() {
        let ub_j = non_ess_ubs[idx];
        if ub_j <= 0.0 {
            continue;
        }
        if theta > 0.0 {
            let min_needed = (theta - rem_ub as f32) - tight_eps;
            if min_needed > 0.0 {
                active_mask = filter_ge_256(tier, scores, &active_mask, min_needed);
                if active_mask == [0u64; 4] {
                    break;
                }
            }
        }
        non_essential[j].accumulate_block_masked(
            block_id,
            &active_mask,
            scores,
            dl_cache,
            source,
            tier,
        )?;
        rem_ub = (rem_ub - ub_j as f64).max(0.0);
    }
    Ok(active_mask)
}

#[inline]
pub(super) fn collect_block_candidates(
    block_id: RowBlock,
    hit_mask: &[u64; 4],
    scores: &[f32; FTS_BLOCK_SIZE],
    candidates: &mut Vec<(f32, RowId)>,
) {
    candidates.clear();
    for_each_set_bit_256(hit_mask, |slot| {
        candidates.push((scores[slot as usize], block_id.row(slot)));
    });
}

/// Ensure every entry in `heap` tied at `theta` has a resolved `doc_id` so `TopKEntry::cmp`
/// evicts by largest `DocId` rather than `RowId`. Returns `true` if `heap` remains at `top_k`.
fn ensure_theta_tie_doc_ids<S: FtsPostingSource>(
    source: &S,
    heap: &mut BinaryHeap<TopKEntry>,
    theta: f32,
    top_k: usize,
) -> Result<bool> {
    let mut missing_needed = false;
    let mut tie_count = 0usize;
    for e in heap.iter() {
        if e.score == theta {
            tie_count += 1;
            if e.doc_id.is_none() {
                missing_needed = true;
            }
        }
    }
    if !missing_needed || tie_count == 0 {
        return Ok(heap.len() == top_k);
    }
    let mut entries: Vec<TopKEntry> = heap.drain().collect();
    let missing: Vec<RowId> = entries
        .iter()
        .filter(|e| e.score == theta && e.doc_id.is_none())
        .map(|e| e.row_id)
        .collect();
    let fetched = source.multi_get_doc_ids(&missing)?;
    let mut idx_map = HashMap::with_capacity(missing.len());
    for (rid, did) in missing.into_iter().zip(fetched) {
        if let Some(d) = did {
            idx_map.insert(rid, d);
        }
    }
    entries.retain_mut(|e| {
        if e.score == theta && e.doc_id.is_none() {
            if let Some(&d) = idx_map.get(&e.row_id) {
                e.doc_id = Some(d);
                true
            } else {
                false
            }
        } else {
            true
        }
    });
    *heap = BinaryHeap::from(entries);
    Ok(heap.len() == top_k)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn insert_candidates_into_heap<'r, S: FtsPostingSource>(
    source: &S,
    candidates: &[(f32, RowId)],
    top_k: usize,
    mut theta: f32,
    floor_theta: f32,
    heap: &mut BinaryHeap<TopKEntry>,
    lazy_filter: &mut Option<&mut dyn FnMut(RowId) -> Result<bool>>,
    mut rescore: Option<&mut (dyn FnMut(RowId, f32) -> Result<f32> + 'r)>,
) -> Result<f32> {
    let needs_tie_doc_ids =
        heap.len() == top_k && candidates.iter().any(|&(s, _)| s == theta);

    let mut tie_doc_ids: Option<Vec<Option<DocId>>> = None;

    for (idx, &(s, row_id)) in candidates.iter().enumerate() {
        if let Some(pred) = lazy_filter
            && !pred(row_id)?
        {
            continue;
        }
        let s = match rescore.as_mut() {
            Some(f) => {
                let exact = f(row_id, s)?;
                if exact.is_nan()
                    || exact <= 0.0
                    || !exact.is_finite()
                    || (heap.len() == top_k && exact < theta)
                {
                    continue;
                }
                exact
            }
            None => s,
        };

        if floor_theta > 0.0 && s < floor_theta {
            continue;
        }

        if heap.len() < top_k {
            heap.push(TopKEntry {
                score: s,
                row_id,
                doc_id: None,
            });
            if heap.len() == top_k {
                theta = heap.peek().unwrap().score;
            }
        } else if s > theta {
            let has_multi_ties = heap.iter().filter(|e| e.score == theta).take(2).count() > 1;
            if has_multi_ties && !ensure_theta_tie_doc_ids(source, heap, theta, top_k)? {
                heap.push(TopKEntry {
                    score: s,
                    row_id,
                    doc_id: None,
                });
                if heap.len() == top_k {
                    theta = heap.peek().unwrap().score;
                }
                continue;
            }
            heap.pop();
            heap.push(TopKEntry {
                score: s,
                row_id,
                doc_id: None,
            });
            theta = heap.peek().unwrap().score;
        } else if s == theta {
            let new_doc_id = if needs_tie_doc_ids {
                let resolved = tie_doc_ids.get_or_insert_with(|| {
                    let ids: Vec<RowId> = candidates.iter().map(|&(_, r)| r).collect();
                    source.multi_get_doc_ids(&ids).unwrap_or_default()
                });
                match resolved.get(idx).copied().flatten() {
                    Some(id) => id,
                    None => continue,
                }
            } else {
                match source.multi_get_doc_ids(&[row_id])?[0] {
                    Some(id) => id,
                    None => continue,
                }
            };
            if !ensure_theta_tie_doc_ids(source, heap, theta, top_k)? {
                heap.push(TopKEntry {
                    score: s,
                    row_id,
                    doc_id: Some(new_doc_id),
                });
                if heap.len() == top_k {
                    theta = heap.peek().unwrap().score;
                }
                continue;
            }
            let root = heap.pop().unwrap();
            let root_doc_id = root.doc_id.unwrap_or(DocId::MAX);
            if new_doc_id < root_doc_id {
                heap.push(TopKEntry {
                    score: s,
                    row_id,
                    doc_id: Some(new_doc_id),
                });
                theta = heap.peek().unwrap().score;
            } else {
                heap.push(root);
            }
        }
    }
    Ok(theta)
}

pub(super) fn resolve_and_sort_results<S: FtsPostingSource>(
    source: &S,
    heap: BinaryHeap<TopKEntry>,
) -> Result<Vec<Bm25Result>> {
    if heap.is_empty() {
        return Ok(Vec::new());
    }

    let mut entries: Vec<TopKEntry> = heap.into_vec();
    let missing_rows: Vec<RowId> = entries
        .iter()
        .filter(|e| e.doc_id.is_none())
        .map(|e| e.row_id)
        .collect();
    if !missing_rows.is_empty() {
        let fetched = source.multi_get_doc_ids(&missing_rows)?;
        let map: HashMap<RowId, DocId> = missing_rows
            .into_iter()
            .zip(fetched)
            .filter_map(|(rid, did)| did.map(|d| (rid, d)))
            .collect();
        for e in &mut entries {
            if e.doc_id.is_none() {
                e.doc_id = map.get(&e.row_id).copied();
            }
        }
    }
    entries.retain(|e| e.doc_id.is_some());
    entries.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.unwrap().cmp(&b.doc_id.unwrap()))
    });

    Ok(entries
        .into_iter()
        .map(|e| Bm25Result {
            id: e.doc_id.unwrap(),
            score: e.score,
        })
        .collect())
}
