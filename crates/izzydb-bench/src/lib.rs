use izzydb::{
    AttributeType, CollectionConfig, DistanceMetric, FullTextSearchConfig, TreeConfig,
};

/// Default path for the generated 10M-row FTS benchmark Parquet file.
pub const DEFAULT_FTS_PARQUET_PATH: &str = "target/bench-data/fts_10m.parquet";

/// Default row count for the generated FTS benchmark dataset.
pub const DEFAULT_FTS_ROWS: usize = 10_000_000;

/// Vector dimension matching the original `fts_bench` schema.
pub const FTS_VECTOR_DIM: usize = 32;

/// Vocabulary size for synthetic FTS documents.
pub const FTS_VOCAB_SIZE: usize = 1_000;

/// Minimum token count per generated document.
pub const FTS_MIN_DOC_LEN: usize = 15;

/// Maximum token count per generated document.
pub const FTS_MAX_DOC_LEN: usize = 40;

/// Build the `CollectionConfig` shared by Parquet generation and `fts_bench`.
pub fn fts_collection_config() -> CollectionConfig {
    CollectionConfig::new(TreeConfig::new(FTS_VECTOR_DIM, DistanceMetric::L2Squared))
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        )
}
