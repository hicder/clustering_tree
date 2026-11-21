//! A mutable, RocksDB-backed hierarchical clustering tree (~100x fanout)
//! inspired by turbopuffer ANN v3 and SPFresh (LIRE).

pub mod config;
pub mod error;
pub mod kmeans;
pub mod lire;
pub mod metric;
pub mod node;
pub mod search;
pub mod storage;
pub mod tree;
pub mod verify;

pub use config::{DistanceMetric, DocId, NodeId, TreeConfig};
pub use error::{Result, TreeError};
pub use tree::ClusteringTree;
pub use verify::TreeStats;
