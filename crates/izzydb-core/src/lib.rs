//! Core configuration, error types, schema, filters, distance metrics, SIMD kernels, and 2-means clustering for `izzydb`.

pub mod config;
pub mod error;
pub mod facet;
pub mod filter;
pub mod kmeans;
pub mod metric;
pub mod rank;
pub mod rowset;
pub mod schema;
pub mod simd;

pub use config::{
    AttrId, ColumnGen, DEFAULT_SEAL_ROWS, DistanceMetric, DocId, FTS_BLOCK_SIZE, LocalId,
    MAX_SEGMENT_ROWS, NodeId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentConfig, SegmentId,
    SkipChunk, TermId, TreeConfig,
};
pub use error::{Result, TreeError};
pub use facet::{
    FacetBucket, FacetFilterMode, FacetKind, FacetRange, FacetResult, FacetResults, FacetSort,
    FacetSpec, FacetedSearchResponse, HistogramBucket, HistogramFacetResult, HybridFacetScope,
    RangeBucket, RangeFacetResult, StatsFacetResult, TermFacetResult, VectorFacetScope,
};
pub use filter::{Filter, FtsPredicateKind, TokenQuery};
pub use rank::{
    Bm25Overrides, DecayMode, NullsOrder, OrderBySpec, OrderedResult, RankBy, SortDirection,
    eval_decay, eval_saturate, validate_order_by,
};
pub use rowset::{Liveness, RowSet, RowSetCursor};
pub use schema::{
    AttributeConfig, AttributeType, AttributeValue, Attributes, CollectionConfig, CollectionSchema,
    ColumnStorage, FullTextSearchConfig, Language, ScalarValue, Tokenizer,
};
