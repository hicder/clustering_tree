//! Query clauses and score accumulation.



use izzydb_core::config::{
    FTS_BLOCK_SIZE, RowBlock, SkipChunk,
};
use izzydb_core::error::Result;

use crate::block::PostingBlockView;
use crate::bm25::Bm25TermScorer;
use crate::simd::{
    SimdTier, merge_add_256, merge_max_256,
};

use super::*;

pub(super) enum AccumMode<'m, 't> {
    Unmasked { touched: &'t mut [u64; 4] },
    Masked { mask: &'m [u64; 4] },
}

impl AccumMode<'_, '_> {
    #[inline]
    pub(super) fn apply(
        self,
        scorer: &Bm25TermScorer,
        tier: SimdTier,
        view: &PostingBlockView<'_>,
        dl_norm: &[f32; FTS_BLOCK_SIZE],
        scores: &mut [f32; FTS_BLOCK_SIZE],
    ) {
        match self {
            Self::Unmasked { touched } => {
                scorer.accumulate_block_simd(tier, view, dl_norm, scores, touched);
            }
            Self::Masked { mask } => {
                scorer.accumulate_block_masked_simd(tier, view, dl_norm, mask, scores);
            }
        }
    }
}

pub(super) enum ClauseKind<'a, S: FtsPostingSource + 'a> {
    Single(TermStream<'a, S>),
    PrefixMax(Vec<TermStream<'a, S>>),
    MaxBranches(Vec<Vec<Clause<'a, S>>>),
}

pub(super) struct Clause<'a, S: FtsPostingSource + 'a> {
    pub(super) term_max: f32,
    pub(super) kind: ClauseKind<'a, S>,
}

impl<'a, S: FtsPostingSource + 'a> Clause<'a, S> {
    pub(super) fn next_block_id(&self) -> Option<RowBlock> {
        match &self.kind {
            ClauseKind::Single(t) => t.current_block_id(),
            ClauseKind::PrefixMax(ts) => ts.iter().filter_map(|t| t.current_block_id()).min(),
            ClauseKind::MaxBranches(branches) => branches
                .iter()
                .flat_map(|b| b.iter())
                .filter_map(|c| c.next_block_id())
                .min(),
        }
    }

    pub(super) fn block_upper_bound(&self, target: RowBlock) -> f32 {
        match &self.kind {
            ClauseKind::Single(t) => {
                if t.current_block_id() == Some(target) {
                    t.block_upper_bound()
                } else {
                    0.0
                }
            }
            ClauseKind::PrefixMax(ts) => ts
                .iter()
                .filter(|t| t.current_block_id() == Some(target))
                .map(TermStream::block_upper_bound)
                .fold(0.0f32, f32::max),
            ClauseKind::MaxBranches(branches) => branches
                .iter()
                .map(|branch| {
                    branch
                        .iter()
                        .map(|c| c.block_upper_bound(target))
                        .sum::<f32>()
                })
                .fold(0.0f32, f32::max),
        }
    }

    #[inline]
    pub(super) fn single_chunk_upper_bound(&self) -> Option<(SkipChunk, f32)> {
        match &self.kind {
            ClauseKind::Single(t) => t.chunk_upper_bound(),
            _ => None,
        }
    }

    #[inline]
    pub(super) fn seek_past_single_chunk(&mut self, chunk: SkipChunk) -> Result<()> {
        if let ClauseKind::Single(t) = &mut self.kind {
            t.seek_past_chunk(chunk)?;
        }
        Ok(())
    }

    pub(super) fn seek_to(&mut self, target: RowBlock) -> Result<()> {
        match &mut self.kind {
            ClauseKind::Single(t) => t.seek(target),
            ClauseKind::PrefixMax(ts) => {
                for t in ts {
                    t.seek(target)?;
                }
                Ok(())
            }
            ClauseKind::MaxBranches(branches) => {
                for branch in branches {
                    for c in branch {
                        c.seek_to(target)?;
                    }
                }
                Ok(())
            }
        }
    }

    pub(super) fn advance_past(&mut self, target: RowBlock) -> Result<()> {
        match &mut self.kind {
            ClauseKind::Single(t) => t.advance_past(target),
            ClauseKind::PrefixMax(ts) => {
                for t in ts {
                    t.advance_past(target)?;
                }
                Ok(())
            }
            ClauseKind::MaxBranches(branches) => {
                for branch in branches {
                    for c in branch {
                        c.advance_past(target)?;
                    }
                }
                Ok(())
            }
        }
    }

    pub(super) fn accumulate_block(
        &mut self,
        target: RowBlock,
        scores: &mut [f32; FTS_BLOCK_SIZE],
        touched: &mut [u64; 4],
        dl_cache: &mut DlNormCache<'a, S>,
        source: &'a S,
        tier: SimdTier,
    ) -> Result<()> {
        self.accumulate_impl(target, None, scores, Some(touched), dl_cache, source, tier)
    }

    pub(super) fn accumulate_block_masked(
        &mut self,
        target: RowBlock,
        mask: &[u64; 4],
        scores: &mut [f32; FTS_BLOCK_SIZE],
        dl_cache: &mut DlNormCache<'a, S>,
        source: &'a S,
        tier: SimdTier,
    ) -> Result<()> {
        self.accumulate_impl(target, Some(mask), scores, None, dl_cache, source, tier)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn accumulate_impl(
        &mut self,
        target: RowBlock,
        mask: Option<&[u64; 4]>,
        scores: &mut [f32; FTS_BLOCK_SIZE],
        touched: Option<&mut [u64; 4]>,
        dl_cache: &mut DlNormCache<'a, S>,
        source: &'a S,
        tier: SimdTier,
    ) -> Result<()> {
        let mut mode_for = |t: &mut TermStream<'a, S>,
                            local_touched: Option<&mut [u64; 4]>,
                            dst_scores: &mut [f32; FTS_BLOCK_SIZE]|
         -> Result<()> {
            let mode = match (mask, local_touched) {
                (Some(m), _) => AccumMode::Masked { mask: m },
                (None, Some(tr)) => AccumMode::Unmasked { touched: tr },
                (None, None) => unreachable!("unmasked mode requires touched mask"),
            };
            t.accumulate_into(target, mode, dst_scores, dl_cache, source, tier)
        };

        match &mut self.kind {
            ClauseKind::Single(t) => {
                if t.current_block_id() == Some(target) && !t.scorer.is_zero() {
                    mode_for(t, touched, scores)?;
                }
                Ok(())
            }
            ClauseKind::PrefixMax(ts) => {
                let matching = ts
                    .iter()
                    .filter(|t| t.current_block_id() == Some(target) && !t.scorer.is_zero())
                    .count();
                if matching == 0 {
                    return Ok(());
                }
                if matching == 1 {
                    let t = ts
                        .iter_mut()
                        .find(|t| t.current_block_id() == Some(target) && !t.scorer.is_zero())
                        .unwrap();
                    return mode_for(t, touched, scores);
                }

                let mut local = [0.0f32; FTS_BLOCK_SIZE];
                let mut tmp = [0.0f32; FTS_BLOCK_SIZE];
                let mut local_touched = [0u64; 4];
                let mut first = true;
                for t in ts.iter_mut() {
                    if t.current_block_id() != Some(target) || t.scorer.is_zero() {
                        continue;
                    }
                    let dst = if first {
                        first = false;
                        &mut local
                    } else {
                        tmp.fill(0.0);
                        &mut tmp
                    };
                    mode_for(t, touched.as_ref().map(|_| &mut local_touched), dst)?;
                    if std::ptr::eq(dst, &tmp) {
                        merge_max_256(tier, &mut local, &tmp);
                    }
                }
                merge_add_256(tier, scores, &local);
                if let Some(tr) = touched {
                    for w in 0..4 {
                        tr[w] |= local_touched[w];
                    }
                }
                Ok(())
            }
            ClauseKind::MaxBranches(branches) => {
                let matching_branches = branches
                    .iter()
                    .filter(|b| b.iter().any(|c| c.next_block_id() == Some(target)))
                    .count();
                if matching_branches == 0 {
                    return Ok(());
                }
                if matching_branches == 1 {
                    let branch = branches
                        .iter_mut()
                        .find(|b| b.iter().any(|c| c.next_block_id() == Some(target)))
                        .unwrap();
                    let mut tr = touched;
                    for c in branch.iter_mut() {
                        if c.next_block_id() == Some(target) {
                            c.accumulate_impl(
                                target,
                                mask,
                                scores,
                                tr.as_deref_mut(),
                                dl_cache,
                                source,
                                tier,
                            )?;
                        }
                    }
                    return Ok(());
                }

                let mut best = [0.0f32; FTS_BLOCK_SIZE];
                let mut branch_scores = [0.0f32; FTS_BLOCK_SIZE];
                let mut any_touched = [0u64; 4];
                let mut first = true;
                for branch in branches.iter_mut() {
                    if !branch.iter().any(|c| c.next_block_id() == Some(target)) {
                        continue;
                    }
                    let is_first = first;
                    first = false;
                    let dst = if is_first {
                        &mut best
                    } else {
                        branch_scores.fill(0.0);
                        &mut branch_scores
                    };
                    for c in branch.iter_mut() {
                        if c.next_block_id() == Some(target) {
                            c.accumulate_impl(
                                target,
                                mask,
                                dst,
                                touched.as_ref().map(|_| &mut any_touched),
                                dl_cache,
                                source,
                                tier,
                            )?;
                        }
                    }
                    if !is_first {
                        merge_max_256(tier, &mut best, &branch_scores);
                    }
                }
                merge_add_256(tier, scores, &best);
                if let Some(tr) = touched {
                    for w in 0..4 {
                        tr[w] |= any_touched[w];
                    }
                }
                Ok(())
            }
        }
    }
}
