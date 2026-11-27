//! Column storage mode and attribute/collection configuration.

use std::collections::HashSet;
use crate::config::{AttrId, SegmentConfig, TreeConfig};
use crate::error::{Result, TreeError};

use super::*;

/// Where an attribute's values are physically stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ColumnStorage {
    /// Stored inline in the segment row store (`cf_rows`) and transposed into `cf_columns` on seal.
    #[default]
    Inline = 0,
    /// Stored outside the row store in versioned `cf_detached` column blocks.
    Detached = 1,
}

impl ColumnStorage {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::Inline),
            1 => Ok(Self::Detached),
            other => Err(TreeError::CorruptedData(format!(
                "unknown ColumnStorage byte: {other}"
            ))),
        }
    }
}

/// Configuration for a single attribute column in a collection.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeConfig {
    pub name: String,
    pub attr_type: AttributeType,
    pub filterable: bool,
    pub full_text_search: Option<FullTextSearchConfig>,
    pub storage: ColumnStorage,
    /// Opt non-filterable inline strings / string arrays into sealed-segment `cf_columns`.
    pub column: bool,
}

impl AttributeConfig {
    /// Create a filterable inline attribute definition.
    pub fn new(name: impl Into<String>, attr_type: AttributeType) -> Self {
        Self {
            name: name.into(),
            attr_type,
            filterable: true,
            full_text_search: None,
            storage: ColumnStorage::Inline,
            column: false,
        }
    }

    /// Set whether this attribute is indexed for filtering.
    pub fn filterable(mut self, filterable: bool) -> Self {
        self.filterable = filterable;
        self
    }

    /// Enable full-text search indexing and BM25 scoring on this attribute.
    pub fn full_text_search(mut self, config: FullTextSearchConfig) -> Self {
        self.full_text_search = Some(config);
        self
    }

    /// Set whether this attribute is stored inline or as a detached column.
    pub fn storage(mut self, storage: ColumnStorage) -> Self {
        self.storage = storage;
        self
    }

    /// Opt an inline attribute into sealed-segment columnar storage (`cf_columns`).
    pub fn with_column(mut self, column: bool) -> Self {
        self.column = column;
        self
    }
}

/// Top-level configuration for opening or creating a [`Collection`](crate::Collection) (in `izzydb`).
///
/// Controls vector tree geometry, segment sealing thresholds, attribute/FTS schema, RocksDB I/O
/// mode, and FTS query concurrency.
///
/// ```text
/// let cfg = CollectionConfig::without_vectors()
///     .with_fts_attribute("body", AttributeType::String, false, FullTextSearchConfig::default())
///     .with_fts_parallel_segments(4);
/// ```
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    /// Optional RaBitQ vector clustering tree configuration (`None` for vectorless collections).
    pub vector: Option<TreeConfig>,
    /// Segment sealing and column transposition lifecycle settings.
    pub segment: SegmentConfig,
    /// Schema definitions for indexed, filterable, column, and full-text attributes.
    pub attributes: Vec<AttributeConfig>,
    /// Bypass the OS page cache using `O_DIRECT` for RocksDB SST reads and flush/compaction I/O.
    pub direct_io: bool,
    /// Maximum number of segment shards executed in parallel during FTS MAXSCORE queries
    /// (`None` uses all Rayon threads; `Some(1)` runs single-threaded across all segments).
    pub fts_parallel_segments: Option<usize>,
}

impl CollectionConfig {
    pub fn new(tree: TreeConfig) -> Self {
        Self {
            vector: Some(tree),
            segment: SegmentConfig::default(),
            attributes: Vec::new(),
            direct_io: false,
            fts_parallel_segments: None,
        }
    }

    pub fn without_vectors() -> Self {
        Self {
            vector: None,
            segment: SegmentConfig::default(),
            attributes: Vec::new(),
            direct_io: false,
            fts_parallel_segments: None,
        }
    }

    pub fn with_direct_io(mut self, direct_io: bool) -> Self {
        self.direct_io = direct_io;
        self
    }

    /// Cap the number of segment shards executed in parallel for FTS MAXSCORE queries.
    ///
    /// Pass `1` to force single-threaded MAXSCORE across all segments.
    pub fn with_fts_parallel_segments(mut self, parallel_segments: usize) -> Self {
        self.fts_parallel_segments = Some(parallel_segments);
        self
    }

    pub fn with_seal_rows(mut self, seal_rows: u32) -> Self {
        self.segment.seal_rows = seal_rows;
        self
    }

    pub fn with_segment_config(mut self, segment: SegmentConfig) -> Self {
        self.segment = segment;
        self
    }

    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        attr_type: AttributeType,
        filterable: bool,
    ) -> Self {
        self.attributes.push(AttributeConfig {
            name: name.into(),
            attr_type,
            filterable,
            full_text_search: None,
            storage: ColumnStorage::Inline,
            column: false,
        });
        self
    }

    pub fn with_column_attribute(
        mut self,
        name: impl Into<String>,
        attr_type: AttributeType,
        filterable: bool,
        column: bool,
    ) -> Self {
        self.attributes.push(AttributeConfig {
            name: name.into(),
            attr_type,
            filterable,
            full_text_search: None,
            storage: ColumnStorage::Inline,
            column,
        });
        self
    }

    pub fn with_detached_attribute(
        mut self,
        name: impl Into<String>,
        attr_type: AttributeType,
        filterable: bool,
    ) -> Self {
        self.attributes.push(AttributeConfig {
            name: name.into(),
            attr_type,
            filterable,
            full_text_search: None,
            storage: ColumnStorage::Detached,
            column: false,
        });
        self
    }

    pub fn with_attribute_config(mut self, attr: AttributeConfig) -> Self {
        self.attributes.push(attr);
        self
    }

    pub fn with_fts_attribute(
        mut self,
        name: impl Into<String>,
        attr_type: AttributeType,
        filterable: bool,
        fts: FullTextSearchConfig,
    ) -> Self {
        self.attributes.push(AttributeConfig {
            name: name.into(),
            attr_type,
            filterable,
            full_text_search: Some(fts),
            storage: ColumnStorage::Inline,
            column: false,
        });
        self
    }

    pub fn validate(&self) -> Result<()> {
        if let Some(ref tree) = self.vector {
            tree.validate()?;
        }
        self.segment.validate()?;
        if self.fts_parallel_segments == Some(0) {
            return Err(TreeError::InvalidConfig(
                "fts_parallel_segments must be >= 1".into(),
            ));
        }
        let mut seen = HashSet::with_capacity(self.attributes.len());
        for attr in &self.attributes {
            if attr.name.is_empty() {
                return Err(TreeError::InvalidSchema(
                    "attribute name cannot be empty".into(),
                ));
            }
            if !seen.insert(attr.name.as_str()) {
                return Err(TreeError::InvalidSchema(format!(
                    "duplicate attribute name '{}'",
                    attr.name
                )));
            }
            if attr.storage == ColumnStorage::Detached && attr.full_text_search.is_some() {
                return Err(TreeError::InvalidSchema(format!(
                    "detached attribute '{}' cannot have full_text_search",
                    attr.name
                )));
            }
            if let Some(fts) = &attr.full_text_search {
                fts.validate(&attr.name, attr.attr_type)?;
            }
        }
        if self.attributes.len() > AttrId::MAX as usize {
            return Err(TreeError::InvalidSchema(
                "too many attributes for u16 AttrId".into(),
            ));
        }
        Ok(())
    }
}
