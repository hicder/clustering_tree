//! Segmented storage engine with global ANN, inverted attribute filters, and Block-Max MAXSCORE FTS.

pub use izzydb_core::{config, error, facet as facet_types, filter, kmeans, metric, rank, rowset, schema};
pub use izzydb_core::facet;
pub use izzydb_fts as fts;
pub use izzydb_rabitq as rabitq;
pub use izzydb_storage::{codec, node, storage};

/// High-level `Collection` coordinating segments and secondary indexes.
pub mod collection;
/// Segment store, row allocation, sealing, and `RowId -> DocId` resolution.
pub mod segments;

// Legacy/unported v2 modules compiled out progressively as Phases 3-11 land.
#[cfg(any())]
pub mod attr_index;
#[cfg(any())]
pub mod fts_index;
#[cfg(any())]
pub mod lire;
#[cfg(any())]
pub mod order_by;
#[cfg(any())]
pub mod rank_eval;
#[cfg(any())]
pub mod search;
#[cfg(any())]
pub mod simd;
#[cfg(any())]
pub mod tree;
#[cfg(any())]
pub mod verify;

pub use collection::{
    Collection, Document, DocumentPatch, HybridSearchOptions, HybridSearchResult, SearchResult,
};
pub use config::{
    AttrId, ColumnGen, DEFAULT_SEAL_ROWS, DistanceMetric, DocId, LocalId, MAX_SEGMENT_ROWS,
    NodeId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentConfig, SegmentId, SkipChunk, TermId,
    TreeConfig,
};
pub use error::{Result, TreeError};
pub use facet::{
    FacetBucket, FacetFilterMode, FacetKind, FacetRange, FacetResult, FacetResults, FacetSort,
    FacetSpec, FacetedSearchResponse, HistogramBucket, HistogramFacetResult, HybridFacetScope,
    RangeBucket, RangeFacetResult, StatsFacetResult, TermFacetResult, VectorFacetScope,
};
pub use filter::{Filter, FtsPredicateKind, TokenQuery};
pub use izzydb_fts::Bm25Result;
pub use rank::{
    Bm25Overrides, DecayMode, NullsOrder, OrderBySpec, OrderedResult, RankBy, SortDirection,
};
pub use rowset::{Liveness, RowSet};
pub use schema::{
    AttributeConfig, AttributeType, AttributeValue, Attributes, CollectionConfig, CollectionSchema,
    ColumnStorage, FullTextSearchConfig, Language, ScalarValue, Tokenizer,
};

/// Parameters controlling quantized vector search and exact reranking.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    pub k: usize,
    pub beam_width: usize,
    pub rerank_candidates: Option<usize>,
    /// Materialized filtered sets with `<= exact_scan_threshold` vector rows use exact scan.
    pub exact_scan_threshold: usize,
    /// Maximum number of leaves probed during filtered dynamic expansion.
    pub max_expanded_leaves: Option<usize>,
}

impl SearchOptions {
    pub fn new(k: usize, beam_width: usize) -> Self {
        Self {
            k,
            beam_width,
            rerank_candidates: None,
            exact_scan_threshold: 4_096,
            max_expanded_leaves: None,
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
}

/// Summary statistics for the ANN clustering tree (Phase 3 / 11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeStats {
    pub num_vectors: u64,
    pub stale_rows: u64,
    pub root_id: NodeId,
    pub root_level: u8,
    pub total_nodes: usize,
    pub internal_nodes: usize,
    pub leaf_nodes: usize,
    pub min_leaf_entries: usize,
    pub max_leaf_entries: usize,
    pub min_internal_fanout: usize,
    pub max_internal_fanout: usize,
}

/// Summary statistics collected by [`Collection::verify_invariants`] (Phase 11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionStats {
    pub tree: Option<TreeStats>,
    pub num_docs: usize,
    pub num_segments: usize,
    pub num_posting_bitmaps: usize,
    pub num_fts_docs: usize,
    pub num_fts_terms: usize,
    pub num_fts_postings: usize,
    pub num_fts_positions: usize,
}
