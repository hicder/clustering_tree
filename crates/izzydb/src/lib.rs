//! Segmented storage engine with global ANN, inverted attribute filters, and Block-Max MAXSCORE FTS.

pub use izzydb_core::{config, error, facet as facet_types, filter, kmeans, metric, rank, rowset, schema};
pub use izzydb_fts as fts;
pub use izzydb_rabitq as rabitq;
pub use izzydb_storage::{codec, node, storage};

/// High-level `Collection` coordinating segments and secondary indexes.
pub mod collection;
/// Segment store, row allocation, sealing, and `RowId -> DocId` resolution.
pub mod segments;

/// Batched `RowId`-sorted attribute readers (row store, columns, detached columns).
pub(crate) mod attr_reader;
/// Per-segment inverted attribute index.
pub mod attr_index;
/// Seal-time and maintenance transposition into `cf_columns`.
pub(crate) mod columns;
/// Segment compaction, forwarding remap, and orphan retirement.
pub mod compaction;
/// Detached columns (`cf_detached`, whole-column ingest/publish, point overwrites).
pub mod detached;
/// Exact faceting engine over `RowSet` populations and global counts.
pub mod facet;
/// `RowSet` filter planner, cardinality estimation, and zone pruning.
pub mod filter_plan;
/// Full-text inverted index and Block-Max MAXSCORE engine.
pub mod fts_index;
/// SPFresh LIRE boundary-vector reassignment.
pub mod lire;
/// Attribute-ordered top-k (`ORDER BY`) query execution.
pub mod order_by;
/// Generalized `RankBy` scoring and boost/decay evaluation.
pub mod rank_eval;
/// Beam-search traversal, RaBitQ scanning, and exact reranking.
pub mod search;
/// SIMD distance and RaBitQ leaf-scoring re-exports.
pub mod simd;
/// Global ANN clustering tree over `RowId`s.
pub mod tree;
/// Exhaustive storage-format invariant verifier.
pub mod verify;

pub use attr_index::AttributeIndex;
pub use collection::{
    Collection, Document, DocumentPatch, HybridSearchOptions, HybridSearchResult, SearchResult,
};
pub use compaction::{CompactionHookStage, CompactionOptions, CompactionPolicy, CompactionStats};
pub use detached::{
    DetachedBuildStats, DetachedDuplicatePolicy, DetachedIngestBuilder, DetachedJob,
    DetachedPublishStats, DetachedWriteMode,
};
#[doc(hidden)]
pub use columns::set_force_column_row_fallback;
pub use filter_plan::{FilterPlanTrace, FilteredSearchRegime};
pub use fts_index::{FtsIndex, LivenessMaskProvider};
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
pub use search::SearchOptions;
pub use tree::{
    AnnIndex, ClusteringTree, MaintenanceOptions, MaintenanceStats, VacuumBudget, VacuumStats,
};

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
