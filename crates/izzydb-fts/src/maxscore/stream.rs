//! Per-term posting streams and doc-length normalization caches.



use izzydb_core::config::{
    AttrId, FTS_BLOCK_SIZE, RowBlock, SkipChunk, TermId,
};
use izzydb_core::error::{Result, TreeError};

use crate::block::{DocLenBlockView, EncodedPosting};
use crate::bm25::Bm25TermScorer;
use crate::simd::{
    SimdTier, prepare_dl_norm_256,
};

use super::*;

/// Specification of a `(attr_id, c1, c2)` doc-length normalization group shared across clauses.
///
/// Built once during query compilation so worker shards can construct thread-local [`DlNormCache`]s
/// without re-compiling query terms.
///
/// ```text
/// DlGroupSpec { attr_id: 1, c1: 0.3, c2: 0.045 }
/// ```
#[derive(Debug, Clone, Copy)]
pub(super) struct DlGroupSpec {
    /// Indexed text attribute identifier.
    pub(super) attr_id: AttrId,
    /// BM25 `k1 * (1.0 - b)`.
    pub(super) c1: f32,
    /// BM25 `k1 * b / avgdl`.
    pub(super) c2: f32,
}

pub(super) struct DlNormGroup<'a, S: FtsPostingSource + 'a> {
    pub(super) attr_id: AttrId,
    pub(super) c1: f32,
    pub(super) c2: f32,
    pub(super) cached_block_id: Option<RowBlock>,
    pub(super) cursor: Option<S::DocLenCursor<'a>>,
    pub(super) dl_norm: [f32; FTS_BLOCK_SIZE],
}

pub(super) struct DlNormCache<'a, S: FtsPostingSource + 'a> {
    pub(super) groups: Vec<DlNormGroup<'a, S>>,
}

impl<'a, S: FtsPostingSource + 'a> DlNormCache<'a, S> {
    pub(super) fn from_specs(specs: &[DlGroupSpec]) -> Self {
        Self {
            groups: specs
                .iter()
                .map(|s| DlNormGroup {
                    attr_id: s.attr_id,
                    c1: s.c1,
                    c2: s.c2,
                    cached_block_id: None,
                    cursor: None,
                    dl_norm: [f32::MIN_POSITIVE; FTS_BLOCK_SIZE],
                })
                .collect(),
        }
    }

    #[inline]
    pub(super) fn get_or_load(
        &mut self,
        source: &'a S,
        tier: SimdTier,
        group_id: usize,
        block_id: RowBlock,
    ) -> Result<&[f32; FTS_BLOCK_SIZE]> {
        let group = &mut self.groups[group_id];
        if group.cached_block_id != Some(block_id) {
            let attr_id = group.attr_id;
            if group.cursor.is_none() {
                group.cursor = Some(source.open_doc_len_cursor(attr_id)?);
            }
            let cursor = group.cursor.as_mut().unwrap();
            let Some(bytes) = cursor.seek_block_bytes(block_id)? else {
                return Err(TreeError::CorruptedData(format!(
                    "missing cf_fts_doc_lens block for attr {attr_id} block {block_id:?}"
                )));
            };
            let view = DocLenBlockView::parse(bytes)?;
            prepare_dl_norm_256(
                tier,
                view.width,
                view.dls_bytes,
                group.c1,
                group.c2,
                &mut group.dl_norm,
            );
            group.cached_block_id = Some(block_id);
        }
        Ok(&self.groups[group_id].dl_norm)
    }
}

pub(super) fn intern_dl_group(
    specs: &mut Vec<DlGroupSpec>,
    attr_id: AttrId,
    scorer: &Bm25TermScorer,
) -> usize {
    let c1 = scorer.k1_one_minus_b;
    let c2 = scorer.k1_b_over_avgdl;
    if let Some(pos) = specs.iter().position(|g| {
        g.attr_id == attr_id && g.c1.to_bits() == c1.to_bits() && g.c2.to_bits() == c2.to_bits()
    }) {
        return pos;
    }
    let id = specs.len();
    specs.push(DlGroupSpec { attr_id, c1, c2 });
    id
}

pub(super) struct TermStream<'a, S: FtsPostingSource + 'a> {
    pub(super) term_id: TermId,
    pub(super) scorer: Bm25TermScorer,
    pub(super) dl_group: usize,
    pub(super) cursor: S::SkipCursor<'a>,
}

impl<'a, S: FtsPostingSource + 'a> TermStream<'a, S> {
    #[inline]
    pub(super) fn current_block_id(&self) -> Option<RowBlock> {
        self.cursor.current_block_id()
    }

    #[inline]
    pub(super) fn block_upper_bound(&self) -> f32 {
        let (max_tf, min_dl_over_tf) = self.cursor.current_block_bounds();
        self.scorer.upper_bound(max_tf, min_dl_over_tf)
    }

    #[inline]
    pub(super) fn chunk_upper_bound(&self) -> Option<(SkipChunk, f32)> {
        let (chunk, max_tf, min_dl_over_tf) = self.cursor.current_chunk_bounds()?;
        Some((chunk, self.scorer.upper_bound(max_tf, min_dl_over_tf)))
    }

    #[inline]
    pub(super) fn seek(&mut self, target: RowBlock) -> Result<()> {
        if let Some(cur) = self.cursor.current_block_id()
            && cur < target
        {
            self.cursor.seek_block(target)?;
        }
        Ok(())
    }

    #[inline]
    pub(super) fn seek_past_chunk(&mut self, chunk: SkipChunk) -> Result<()> {
        self.cursor.seek_past_chunk(chunk)
    }

    #[inline]
    pub(super) fn advance_past(&mut self, block_id: RowBlock) -> Result<()> {
        if self.cursor.current_block_id() == Some(block_id) {
            self.cursor.next_block()?;
        }
        Ok(())
    }

    #[inline]
    pub(super) fn accumulate_into(
        &mut self,
        target: RowBlock,
        mode: AccumMode<'_, '_>,
        scores: &mut [f32; FTS_BLOCK_SIZE],
        dl_cache: &mut DlNormCache<'a, S>,
        source: &'a S,
        tier: SimdTier,
    ) -> Result<()> {
        let dl_norm = dl_cache.get_or_load(source, tier, self.dl_group, target)?;
        if let Some(encoded) = self.cursor.current_posting_encoded()? {
            return mode.apply_encoded(&self.scorer, tier, encoded, dl_norm, scores);
        }
        let Some(guard) = source.get_posting_block(self.term_id, target)? else {
            return Err(TreeError::CorruptedData(format!(
                "missing posting block for term_id {} block {target:?}",
                self.term_id
            )));
        };
        mode.apply_encoded(
            &self.scorer,
            tier,
            EncodedPosting::Standalone(guard.as_ref()),
            dl_norm,
            scores,
        )
    }
}
