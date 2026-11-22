//! A mutable, RocksDB-backed hierarchical clustering tree (~100x fanout)
//! with native cluster-aware attribute filtering and Block-Max MAXSCORE full-text search.

pub use izzydb_core::{config, error, filter, kmeans, metric, rank, schema};
pub use izzydb_fts as fts;
pub use izzydb_rabitq as rabitq;
pub use izzydb_storage::{codec, node, storage};

/// Cluster-aware inverted attribute index and two-phase filter evaluator.
pub mod attr_index;
/// High-level `Collection` coordinating vector, attribute, and full-text indexes.
pub mod collection;
/// Exact faceting engine (global counts, forward doc passes, inverted leaf-mask intersection).
pub mod facet;
/// Inverted full-text index manager and BM25 query execution.
pub mod fts_index;
/// SPFresh LIRE boundary reassignment after leaf splits.
pub mod lire;
/// Beam-search tree traversal, RaBitQ leaf scanning, and exact reranking.
pub mod search;
/// Re-exported RaBitQ SIMD leaf-scoring kernels.
pub mod simd;
/// Mutable hierarchical clustering tree (`ClusteringTree`) and split/merge maintenance.
pub mod tree;
/// Exhaustive structural, attribute-index, and FTS invariant verification.
pub mod verify;

pub use collection::{
    Collection, Document, DocumentPatch, HybridSearchOptions, HybridSearchResult, SearchResult,
};
pub use facet::{
    FacetBucket, FacetFilterMode, FacetKind, FacetRange, FacetResult, FacetResults, FacetSort,
    FacetSpec, FacetedSearchResponse, HistogramBucket, HistogramFacetResult, HybridFacetScope,
    RangeBucket, RangeFacetResult, StatsFacetResult, TermFacetResult, VectorFacetScope,
};
pub use config::{AttrId, DistanceMetric, DocId, FtsBlockId, FtsDocId, LocalId, NodeId, TreeConfig};
pub use error::{Result, TreeError};
pub use filter::{Filter, FtsPredicateKind, TokenQuery};
pub use fts_index::FtsIndex;
pub use izzydb_fts::Bm25Result;
pub use rank::{Bm25Overrides, RankBy};
pub use schema::{
    AttributeConfig, AttributeType, AttributeValue, Attributes, CollectionConfig, CollectionSchema,
    FullTextSearchConfig, Language, ScalarValue, Tokenizer,
};
pub use search::SearchOptions;
pub use tree::ClusteringTree;
pub use verify::{CollectionStats, TreeStats};
