use izzydb_core::config::FTS_BLOCK_SIZE;

use crate::block::{PostingBlockPayload, PostingBlockView};
use crate::simd::{
    SimdTier, accumulate_dense_256, accumulate_dense_masked_256, accumulate_sparse,
    accumulate_sparse_masked,
};

/// Robertson-Walker / Lucene BM25+ style IDF: `ln(1 + (N - df + 0.5) / (df + 0.5))`.
#[inline]
pub fn bm25_idf(doc_count: u64, doc_freq: u64) -> f64 {
    if doc_count == 0 || doc_freq == 0 {
        return 0.0;
    }
    let n = doc_count as f64;
    let df = doc_freq as f64;
    let num = (n - df + 0.5).max(0.0);
    (1.0 + num / (df + 0.5)).ln()
}

/// Query term frequency saturation factor: `qtf * (k3 + 1) / (qtf + k3)`.
#[inline]
pub fn bm25_qtf_norm(qtf: u32, k3: f32) -> f64 {
    if qtf == 0 {
        return 0.0;
    }
    let q = qtf as f64;
    let k3 = k3 as f64;
    if k3 == 0.0 {
        return 1.0;
    }
    (q * (k3 + 1.0)) / (q + k3)
}

/// Precomputed per-term scoring parameters for fast inner-loop BM25 accumulation.
#[derive(Debug, Clone, Copy)]
pub struct Bm25TermScorer {
    /// Combined scalar multiplier: `weight * idf * qtf_norm * (k1 + 1.0)`.
    pub num_scale: f32,
    /// `k1 * (1.0 - b)`.
    pub k1_one_minus_b: f32,
    /// `k1 * b / avgdl`.
    pub k1_b_over_avgdl: f32,
}

impl Bm25TermScorer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weight: f32,
        doc_count: u64,
        total_tokens: u64,
        doc_freq: u64,
        qtf: u32,
        k1: f32,
        b: f32,
        k3: f32,
    ) -> Self {
        if doc_count == 0 || total_tokens == 0 || doc_freq == 0 || qtf == 0 || weight <= 0.0 {
            return Self {
                num_scale: 0.0,
                k1_one_minus_b: 0.0,
                k1_b_over_avgdl: 0.0,
            };
        }
        let avgdl = (total_tokens as f64) / (doc_count as f64);
        let idf = bm25_idf(doc_count, doc_freq);
        let qtf_norm = bm25_qtf_norm(qtf, k3);
        let k1_f = k1 as f64;
        let b_f = b as f64;
        let num_scale = (weight as f64 * idf * qtf_norm * (k1_f + 1.0)) as f32;
        let k1_one_minus_b = (k1_f * (1.0 - b_f)) as f32;
        let k1_b_over_avgdl = ((k1_f * b_f) / avgdl) as f32;
        Self {
            num_scale,
            k1_one_minus_b,
            k1_b_over_avgdl,
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.num_scale <= 0.0
    }

    /// Exact BM25 contribution for a single posting `(tf, dl)`.
    #[inline]
    pub fn score(&self, tf: u32, dl: u32) -> f32 {
        if self.num_scale <= 0.0 || tf == 0 || dl == 0 {
            return 0.0;
        }
        let tf_f = tf as f32;
        let dl_f = dl as f32;
        (self.num_scale * tf_f) / (tf_f + self.k1_one_minus_b + self.k1_b_over_avgdl * dl_f)
    }

    /// Sound upper bound on score for any posting with `tf <= max_tf` and `dl / tf >= min_dl_over_tf`.
    #[inline]
    pub fn upper_bound(&self, max_tf: u32, min_dl_over_tf: f32) -> f32 {
        if self.num_scale <= 0.0 || max_tf == 0 || !min_dl_over_tf.is_finite() {
            return 0.0;
        }
        let c1 = self.k1_one_minus_b as f64;
        let c2 = self.k1_b_over_avgdl as f64;
        let inv_tf = if c1 > 0.0 { c1 / (max_tf as f64) } else { 0.0 };
        let denom = 1.0 + inv_tf + c2 * (min_dl_over_tf as f64);
        ((self.num_scale as f64) / denom) as f32
    }

    /// Accumulate BM25 scores for all postings in `view` into `scores` using SIMD kernels.
    #[inline]
    pub fn accumulate_block_simd(
        &self,
        tier: SimdTier,
        view: &PostingBlockView<'_>,
        dl_norm: &[f32; FTS_BLOCK_SIZE],
        scores: &mut [f32; FTS_BLOCK_SIZE],
        touched: &mut [u64; 4],
    ) {
        if self.is_zero() {
            return;
        }
        match &view.payload {
            PostingBlockPayload::Dense { bitmap, tfs_bytes } => {
                accumulate_dense_256(
                    tier,
                    view.tf_enc,
                    tfs_bytes,
                    self.num_scale,
                    dl_norm,
                    bitmap,
                    scores,
                    touched,
                );
            }
            PostingBlockPayload::Sparse { slots, tfs_bytes } => {
                accumulate_sparse(
                    tier,
                    view.tf_enc,
                    slots,
                    tfs_bytes,
                    self.num_scale,
                    dl_norm,
                    scores,
                    touched,
                );
            }
        }
    }

    /// Accumulate BM25 scores only for slots already set in `mask` using SIMD kernels.
    #[inline]
    pub fn accumulate_block_masked_simd(
        &self,
        tier: SimdTier,
        view: &PostingBlockView<'_>,
        dl_norm: &[f32; FTS_BLOCK_SIZE],
        mask: &[u64; 4],
        scores: &mut [f32; FTS_BLOCK_SIZE],
    ) {
        if self.is_zero() {
            return;
        }
        match &view.payload {
            PostingBlockPayload::Dense { bitmap, tfs_bytes } => {
                accumulate_dense_masked_256(
                    tier,
                    view.tf_enc,
                    tfs_bytes,
                    self.num_scale,
                    dl_norm,
                    bitmap,
                    mask,
                    scores,
                );
            }
            PostingBlockPayload::Sparse { slots, tfs_bytes } => {
                accumulate_sparse_masked(
                    tier,
                    view.tf_enc,
                    slots,
                    tfs_bytes,
                    self.num_scale,
                    dl_norm,
                    mask,
                    scores,
                );
            }
        }
    }
}
