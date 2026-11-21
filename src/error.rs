use thiserror::Error;

/// Result type alias for clustering tree operations.
pub type Result<T> = std::result::Result<T, TreeError>;

/// Errors returned by [`crate::ClusteringTree`] operations.
#[derive(Debug, Error)]
pub enum TreeError {
    /// Underlying RocksDB error.
    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),

    /// Invalid configuration parameters.
    #[error("invalid tree config: {0}")]
    InvalidConfig(String),

    /// Input embedding or query has a different dimension than configured.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Input embedding or query contains NaN or infinite values.
    #[error("vector contains non-finite (NaN or Inf) values")]
    NonFiniteVector,

    /// Reopened database metadata disagrees with provided [`crate::TreeConfig`].
    #[error("config mismatch for {field}: database has {expected}, config has {actual}")]
    ConfigMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },

    /// Corrupted or unexpected serialized data in storage.
    #[error("corrupted storage data: {0}")]
    CorruptedData(String),

    /// Structural invariant violation detected during verification or mutation.
    #[error("tree invariant violation: {0}")]
    InvariantViolation(String),
}
