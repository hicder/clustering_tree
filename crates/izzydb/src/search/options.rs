//! Search options.



/// Parameters controlling quantized vector search, optional full-vector reranking, and filtered expansion.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    pub k: usize,
    pub beam_width: usize,
    pub rerank_candidates: Option<usize>,
    /// Materialized filtered sets with `<= exact_scan_threshold` vector rows use exact scan.
    pub exact_scan_threshold: usize,
    /// Maximum number of leaves probed during filtered dynamic expansion (defaults to `beam_width * 16`).
    pub max_expanded_leaves: Option<usize>,
    /// Use in-memory leaf/internal value summaries to prune subtrees before I/O.
    ///
    /// Off by default: it helps correlated attributes but costs more than it saves on
    /// uncorrelated ones. Enable for tenant/category/timestamp-style clustered schemas.
    pub summary_pruning: bool,
}

impl SearchOptions {
    pub fn new(k: usize, beam_width: usize) -> Self {
        Self {
            k,
            beam_width,
            rerank_candidates: None,
            exact_scan_threshold: 4_096,
            max_expanded_leaves: None,
            summary_pruning: false,
        }
    }

    pub fn with_rerank(mut self, rerank_candidates: usize) -> Self {
        self.rerank_candidates = Some(rerank_candidates);
        self
    }

    pub fn with_exact_scan_threshold(mut self, threshold: usize) -> Self {
        self.exact_scan_threshold = threshold;
        self
    }

    pub fn with_max_expanded_leaves(mut self, max_leaves: usize) -> Self {
        self.max_expanded_leaves = Some(max_leaves);
        self
    }

    /// Enable summary-based subtree pruning for correlated attributes.
    pub fn with_summary_pruning(mut self, enabled: bool) -> Self {
        self.summary_pruning = enabled;
        self
    }
}
