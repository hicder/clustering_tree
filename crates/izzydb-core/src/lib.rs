//! Core configuration, error types, schema, filters, distance metrics, SIMD kernels, and 2-means clustering for `izzydb`.

pub mod config;
pub mod error;
pub mod facet;
pub mod filter;
pub mod kmeans;
pub mod metric;
pub mod rank;
pub mod schema;
pub mod simd;

pub use config::{
    AttrId, DistanceMetric, DocId, FTS_BLOCK_SIZE, FtsBlockId, FtsDocId, LocalId, NodeId,
    TreeConfig, fts_block_id, fts_slot, make_fts_doc_id,
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
pub use schema::{
    AttributeConfig, AttributeType, AttributeValue, Attributes, CollectionConfig, CollectionSchema,
    FullTextSearchConfig, Language, ScalarValue, Tokenizer,
};
