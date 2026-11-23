//! Block-Max MAXSCORE execution.

use std::collections::{BinaryHeap, HashMap};

use alyze::analyze::ReusableBuffer;

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
    mut lazy_filter: Option<&mut dyn FnMut(RowId) -> Result<bool>>,
    modifier: Option<RankModifier<'_>>,
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
    let mut dl_cache = DlNormCache::new();
    let mut clauses = Vec::new();
    {
        let mut ctx = CompileContext {
            source,
            schema,
            meta: &meta,
            buffer,
            dl_cache: &mut dl_cache,
        };
        compile_clauses(&mut ctx, &rank_by, mult_ub, &mut clauses)?;
    }
    if clauses.is_empty() {
        return Ok(Vec::new());
    }

    // Sort clauses ascending by global upper bound `term_max` for MAXSCORE partitioning.
    clauses.sort_by(|a, b| a.term_max.total_cmp(&b.term_max));
    let n_clauses = clauses.len();
    let mut prefix_max_sum = Vec::with_capacity(n_clauses);
    let mut acc = 0.0f64;
    for c in &clauses {
        acc += c.term_max as f64;
        prefix_max_sum.push(acc as f32);
    }

    let mut essential_start: usize = 0;
    let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(top_k + 1);
    let mut theta: f32 = 0.0;

    let mut scores = [0.0f32; FTS_BLOCK_SIZE];
    let mut touched: [u64; 4];
    let mut candidates: Vec<(f32, RowId)> = Vec::with_capacity(32);
    let mut non_ess_ubs: Vec<f32> = Vec::with_capacity(n_clauses);

    while let Some(block_id) = next_essential_block(&clauses[essential_start..]) {
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

        let non_ess_ub = if essential_start > 0 {
            prefix_max_sum[essential_start - 1]
        } else {
            0.0
        };

        // Block-Max skip pruning BEFORE reading any posting blocks or doc_len blocks.
        // Lexical-scale pruning threshold: a doc needs `L * mult_ub + add_ub > theta`.
        let prune_theta = if theta > add_ub { theta - add_ub } else { 0.0 };
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
                c.accumulate_block(block_id, &mut scores, &mut touched, &mut dl_cache, source, tier)?;
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
                &mut dl_cache,
                source,
                tier,
            )?
        };

        // SIMD threshold filtering for top-k candidates.
        if active_mask != [0u64; 4] {
            let cand_thresh = if heap.len() < top_k {
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
                        &mut heap,
                        &mut lazy_filter,
                        rescore_fn.as_deref_mut(),
                    )?;
                }
            }
        }

        // Dynamically shrink essential set as theta rises.
        let prune_theta = if theta > add_ub { theta - add_ub } else { 0.0 };
        if heap.len() == top_k && prune_theta > 0.0 {
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

    resolve_and_sort_results(source, heap)
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

pub(super) fn insert_candidates_into_heap<'r, S: FtsPostingSource>(
    source: &S,
    candidates: &[(f32, RowId)],
    top_k: usize,
    mut theta: f32,
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
            let mut root = heap.pop().unwrap();
            let root_doc_id = match root.doc_id {
                Some(id) => id,
                None => {
                    let fetched = source.multi_get_doc_ids(&[root.row_id])?;
                    let id = fetched[0].unwrap_or(DocId::MAX);
                    root.doc_id = Some(id);
                    id
                }
            };
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

    let mut results: Vec<Bm25Result> = entries
        .into_iter()
        .filter_map(|e| {
            e.doc_id.map(|id| Bm25Result {
                id,
                score: e.score,
            })
        })
        .collect();
    results.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    Ok(results)
}
