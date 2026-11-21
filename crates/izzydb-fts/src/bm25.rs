use crate::block::PostingBlockView;

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

    /// Sound upper bound on score for any posting with `tf <= max_tf` and `dl >= min_dl`.
    #[inline]
    pub fn upper_bound(&self, max_tf: u32, min_dl: u32) -> f32 {
        self.score(max_tf, min_dl.max(1))
    }

    /// Scatter-add BM25 scores for all postings in `view` into `scores` and mark `touched`.
    #[inline]
    pub fn accumulate_block(
        &self,
        view: &PostingBlockView<'_>,
        scores: &mut [f32; 256],
        touched: &mut [u64; 4],
    ) {
        if self.is_zero() {
            return;
        }
        let n = view.header.count as usize;
        if view.is_u16() {
            let tfs = view.tfs_bytes;
            let dls = view.dls_bytes;
            for i in 0..n {
                let slot = view.slots[i];
                let off = i * 2;
                let tf = u16::from_le_bytes([tfs[off], tfs[off + 1]]) as u32;
                let dl = u16::from_le_bytes([dls[off], dls[off + 1]]) as u32;
                let s = self.score(tf, dl);
                scores[slot as usize] += s;
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        } else {
            for i in 0..n {
                let slot = view.slots[i];
                let s = self.score(view.tf(i), view.dl(i));
                scores[slot as usize] += s;
                touched[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            }
        }
    }

    /// Scatter-add BM25 scores only for slots already set in `mask`.
    #[inline]
    pub fn accumulate_block_masked(
        &self,
        view: &PostingBlockView<'_>,
        mask: &[u64; 4],
        scores: &mut [f32; 256],
    ) {
        if self.is_zero() {
            return;
        }
        let n = view.header.count as usize;
        if view.is_u16() {
            let tfs = view.tfs_bytes;
            let dls = view.dls_bytes;
            for i in 0..n {
                let slot = view.slots[i];
                if ((mask[(slot >> 6) as usize] >> (slot & 63)) & 1) == 0 {
                    continue;
                }
                let off = i * 2;
                let tf = u16::from_le_bytes([tfs[off], tfs[off + 1]]) as u32;
                let dl = u16::from_le_bytes([dls[off], dls[off + 1]]) as u32;
                scores[slot as usize] += self.score(tf, dl);
            }
        } else {
            for i in 0..n {
                let slot = view.slots[i];
                if ((mask[(slot >> 6) as usize] >> (slot & 63)) & 1) == 0 {
                    continue;
                }
                scores[slot as usize] += self.score(view.tf(i), view.dl(i));
            }
        }
    }
}
