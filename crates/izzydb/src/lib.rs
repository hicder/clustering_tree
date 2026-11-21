//! A mutable, RocksDB-backed hierarchical clustering tree (~100x fanout)
//! inspired by turbopuffer ANN v3 and SPFresh (LIRE).

pub use izzydb_core::{config, error, kmeans, metric};
pub use izzydb_rabitq as rabitq;
pub use izzydb_storage::{node, storage};

pub mod lire;
pub mod search;
pub mod simd;
pub mod tree;
pub mod verify;

pub use config::{DistanceMetric, DocId, NodeId, TreeConfig};
pub use error::{Result, TreeError};
pub use tree::ClusteringTree;
pub use verify::TreeStats;
