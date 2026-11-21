//! Core configuration, error types, distance metrics, SIMD kernels, and 2-means clustering for `izzydb`.

pub mod config;
pub mod error;
pub mod kmeans;
pub mod metric;
pub mod simd;

pub use config::{DistanceMetric, DocId, NodeId, TreeConfig};
pub use error::{Result, TreeError};
