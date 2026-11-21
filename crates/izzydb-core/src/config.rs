use crate::error::{Result, TreeError};

/// External document identifier.
pub type DocId = u64;

/// Internal tree node identifier.
pub type NodeId = u32;

/// Dense per-leaf local identifier assigned to documents in a leaf.
pub type LocalId = u32;

/// Compact numeric identifier assigned to each attribute in a [`CollectionSchema`](crate::schema::CollectionSchema).
pub type AttrId = u16;

/// Dense monotonic document identifier used by the FTS inverted index (`block_id = fts_doc_id >> 8`).
pub type FtsDocId = u32;

/// 256-document posting block identifier (`fts_doc_id >> 8`).
pub type FtsBlockId = u32;

/// Number of documents per fixed FTS posting block.
pub const FTS_BLOCK_SIZE: usize = 256;

/// Extract the posting block ID for an [`FtsDocId`].
#[inline]
pub const fn fts_block_id(doc_id: FtsDocId) -> FtsBlockId {
    doc_id >> 8
}

/// Extract the intra-block slot (`0..=255`) for an [`FtsDocId`].
#[inline]
pub const fn fts_slot(doc_id: FtsDocId) -> u8 {
    (doc_id & 0xFF) as u8
}

/// Combine a block ID and intra-block slot into an [`FtsDocId`].
#[inline]
pub const fn make_fts_doc_id(block_id: FtsBlockId, slot: u8) -> FtsDocId {
    (block_id << 8) | (slot as u32)
}

/// Distance metric used for clustering and nearest neighbor search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DistanceMetric {
    /// Squared Euclidean distance: `sum((a[i] - b[i])^2)`.
    L2Squared = 0,
    /// Cosine distance: `1.0 - dot(a, b)` on L2-normalized vectors.
    Cosine = 1,
    /// Maximum inner product distance: `-dot(a, b)` (smaller is closer).
    DotProduct = 2,
}

impl DistanceMetric {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::L2Squared),
            1 => Ok(Self::Cosine),
            2 => Ok(Self::DotProduct),
            other => Err(TreeError::CorruptedData(format!(
                "unknown DistanceMetric byte: {other}"
            ))),
        }
    }
}

/// Configuration for a clustering tree database.
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Similarity/distance metric.
    pub metric: DistanceMetric,

    /// Minimum number of children in a non-root internal node.
    pub min_internal_fanout: usize,
    /// Maximum number of children in an internal node before splitting.
    pub max_internal_fanout: usize,

    /// Minimum number of vectors in a non-root leaf node.
    pub min_leaf_size: usize,
    /// Maximum number of vectors in a leaf node before splitting.
    pub max_leaf_size: usize,

    /// Beam width used per level when routing vectors during `add_leaf`.
    pub insert_beam_width: usize,
    /// Number of neighbor leaves checked during SPFresh LIRE boundary reassignment.
    pub lire_neighbor_count: usize,
    /// Whether to run SPFresh LIRE boundary reassignment after leaf splits.
    pub enable_lire: bool,
    /// Skip propagating centroid updates upward when drift <= this threshold.
    pub centroid_drift_eps: f32,
}

impl TreeConfig {
    /// Create a configuration with default ~100x fanout bounds for the given dimension and metric.
    pub fn new(dim: usize, metric: DistanceMetric) -> Self {
        Self {
            dim,
            metric,
            min_internal_fanout: 32,
            max_internal_fanout: 160,
            min_leaf_size: 32,
            max_leaf_size: 160,
            insert_beam_width: 4,
            lire_neighbor_count: 8,
            enable_lire: true,
            centroid_drift_eps: 1e-3,
        }
    }

    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(TreeError::InvalidConfig("dim must be > 0".into()));
        }
        if self.min_leaf_size < 1 {
            return Err(TreeError::InvalidConfig(
                "min_leaf_size must be >= 1".into(),
            ));
        }
        if self.min_internal_fanout < 2 {
            return Err(TreeError::InvalidConfig(
                "min_internal_fanout must be >= 2".into(),
            ));
        }
        if self.min_leaf_size * 2 >= self.max_leaf_size {
            return Err(TreeError::InvalidConfig(format!(
                "min_leaf_size * 2 ({}) must be < max_leaf_size ({}) to avoid split/merge oscillation",
                self.min_leaf_size * 2,
                self.max_leaf_size
            )));
        }
        if self.min_internal_fanout * 2 >= self.max_internal_fanout {
            return Err(TreeError::InvalidConfig(format!(
                "min_internal_fanout * 2 ({}) must be < max_internal_fanout ({}) to avoid split/merge oscillation",
                self.min_internal_fanout * 2,
                self.max_internal_fanout
            )));
        }
        if self.insert_beam_width == 0 {
            return Err(TreeError::InvalidConfig(
                "insert_beam_width must be >= 1".into(),
            ));
        }
        if !self.centroid_drift_eps.is_finite() || self.centroid_drift_eps < 0.0 {
            return Err(TreeError::InvalidConfig(
                "centroid_drift_eps must be finite and >= 0.0".into(),
            ));
        }
        Ok(())
    }
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self::new(128, DistanceMetric::L2Squared)
    }
}
