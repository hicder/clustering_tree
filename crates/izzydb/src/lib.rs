//! A mutable, RocksDB-backed hierarchical clustering tree (~100x fanout)
//! with native cluster-aware attribute filtering.

pub use izzydb_core::{config, error, filter, kmeans, metric, rank, schema};
pub use izzydb_fts as fts;
pub use izzydb_rabitq as rabitq;
pub use izzydb_storage::{codec, node, storage};

pub mod attr_index;
pub mod collection;
pub mod fts_index;
pub mod lire;
pub mod search;
pub mod simd;
pub mod tree;
pub mod verify;

pub use collection::{
    Collection, Document, HybridSearchOptions, HybridSearchResult, SearchResult,
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
