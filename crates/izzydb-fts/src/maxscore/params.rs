//! Resolved BM25 parameters, top-k heap entries, and rank modifiers.

use std::cmp::Ordering;


use izzydb_core::config::{
    DocId, RowBlock, RowId,
};
use izzydb_core::error::Result;
use izzydb_core::rank::Bm25Overrides;
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::FullTextSearchConfig;


use super::*;

pub(super) struct ResolvedBm25Params {
    pub(super) k1: f32,
    pub(super) b: f32,
    pub(super) k3: f32,
    pub(super) last_as_prefix: bool,
    pub(super) max_prefix_expansions: usize,
}

pub(super) fn resolve_params(
    fts: &FullTextSearchConfig,
    overrides: Option<&Bm25Overrides>,
) -> ResolvedBm25Params {
    match overrides {
        Some(o) => ResolvedBm25Params {
            k1: o.k1.unwrap_or(fts.k1),
            b: o.b.unwrap_or(fts.b),
            k3: o.k3.unwrap_or(fts.k3),
            last_as_prefix: o.last_as_prefix.unwrap_or(false),
            max_prefix_expansions: o
                .max_prefix_expansions
                .unwrap_or(DEFAULT_MAX_PREFIX_EXPANSIONS),
        },
        None => ResolvedBm25Params {
            k1: fts.k1,
            b: fts.b,
            k3: fts.k3,
            last_as_prefix: false,
            max_prefix_expansions: DEFAULT_MAX_PREFIX_EXPANSIONS,
        },
    }
}

#[inline]
pub(super) fn ub_slack(ub: f32) -> f32 {
    ub * 1e-6 + 1e-7
}

#[inline]
pub(super) fn ub_below_threshold(ub: f32, theta: f32) -> bool {
    ub + ub_slack(ub) < theta
}

/// Min-heap entry for top-$k$ BM25 (smallest score / largest DocId at the top so it is evicted first).
#[derive(Debug, Clone, Copy)]
pub(super) struct TopKEntry {
    pub(super) score: f32,
    pub(super) row_id: RowId,
    pub(super) doc_id: Option<DocId>,
}

impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.row_id == other.row_id
    }
}

impl Eq for TopKEntry {}

impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| match (self.doc_id, other.doc_id) {
                (Some(a), Some(b)) => a.cmp(&b),
                _ => self.row_id.cmp(&other.row_id),
            })
    }
}

impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Optional per-block slot mask combining segment liveness and pushed-down filter [`RowSet`]s.
///
/// Returning `None` means all 256 slots in the block are allowed (fast path for segments
/// without tombstones when no filter is active). Returning `Some([0; 4])` skips the block.
pub trait BlockMaskProvider {
    fn mask(&self, block: RowBlock) -> Option<[u64; 4]>;
}

impl BlockMaskProvider for RowSet {
    #[inline]
    fn mask(&self, block: RowBlock) -> Option<[u64; 4]> {
        Some(self.block_mask(block.seg(), block.block()))
    }
}

/// Per-row score modifier applied on top of a pure lexical expression `L(row)`:
/// `score(row) = L(row) * m(row) + a(row)` with `0 <= m <= mult_ub` and `0 <= a <= add_ub`.
///
/// MAXSCORE runs on `L * mult_ub` (an upper bound of the lexical part of the true score) and
/// prunes against `theta - add_ub`; only candidates that survive pruning reach `rescore`.
pub struct RankModifier<'f> {
    /// Finite upper bound `> 0` of the multiplicative factor `m(row)`.
    pub mult_ub: f32,
    /// Finite upper bound `>= 0` of the additive term `a(row)`.
    pub add_ub: f32,
    /// Maps `(row_id, L(row) * mult_ub)` to the exact final score (`<= 0` drops the document).
    pub rescore: &'f mut dyn FnMut(RowId, f32) -> Result<f32>,
}
