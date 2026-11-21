//! Core configuration, error types, schema, filters, distance metrics, SIMD kernels, and 2-means clustering for `izzydb`.

pub mod config;
pub mod error;
pub mod filter;
pub mod kmeans;
pub mod metric;
pub mod schema;
pub mod simd;

pub use config::{AttrId, DistanceMetric, DocId, LocalId, NodeId, TreeConfig};
pub use error::{Result, TreeError};
pub use filter::Filter;
pub use schema::{
    AttributeConfig, AttributeType, AttributeValue, Attributes, CollectionConfig, CollectionSchema,
    ScalarValue,
};
