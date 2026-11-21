use thiserror::Error;

/// Result type alias for clustering tree operations.
pub type Result<T> = std::result::Result<T, TreeError>;

/// Errors returned by clustering tree operations.
#[derive(Debug, Error)]
pub enum TreeError {
    /// Underlying storage backend error.
    #[error("storage error: {0}")]
    Storage(String),

    /// Invalid configuration parameters.
    #[error("invalid tree config: {0}")]
    InvalidConfig(String),

    /// Input embedding or query has a different dimension than configured.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Input embedding or query contains NaN or infinite values.
    #[error("vector contains non-finite (NaN or Inf) values")]
    NonFiniteVector,

    /// Reopened database metadata disagrees with provided [`TreeConfig`](crate::TreeConfig).
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

    /// Invalid collection schema definition or schema mismatch.
    #[error("invalid schema: {0}")]
    InvalidSchema(String),

    /// Document attribute failed schema validation.
    #[error("invalid attribute '{name}': {reason}")]
    InvalidAttribute { name: String, reason: String },

    /// Search filter failed validation against collection schema.
    #[error("invalid filter: {0}")]
    InvalidFilter(String),

    /// Search or ranking query failed validation.
    #[error("invalid query: {0}")]
    InvalidQuery(String),
}
