//! A mutable, RocksDB-backed hierarchical clustering tree (~100x fanout)
//! with native cluster-aware attribute filtering.

pub use izzydb_core::{config, error, filter, kmeans, metric, schema};
pub use izzydb_rabitq as rabitq;
pub use izzydb_storage::{codec, node, storage};

pub mod attr_index;
pub mod collection;
pub mod lire;
pub mod search;
pub mod simd;
pub mod tree;
pub mod verify;

pub use collection::{Collection, Document, SearchResult};
pub use config::{AttrId, DistanceMetric, DocId, LocalId, NodeId, TreeConfig};
pub use error::{Result, TreeError};
pub use filter::Filter;
pub use schema::{
    AttributeConfig, AttributeType, AttributeValue, Attributes, CollectionConfig, CollectionSchema,
    ScalarValue,
};
pub use search::SearchOptions;
pub use tree::ClusteringTree;
pub use verify::{CollectionStats, TreeStats};
