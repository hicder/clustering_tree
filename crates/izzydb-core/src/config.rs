use std::fmt;

use crate::error::{Result, TreeError};

/// External document identifier (user primary key).
pub type DocId = u64;

/// Monotonic segment identifier (never reused).
pub type SegmentId = u32;

/// Dense, append-only local row identifier within a segment (`< MAX_SEGMENT_ROWS`).
pub type LocalId = u32;

/// Internal tree node identifier.
pub type NodeId = u32;

/// Compact numeric identifier assigned to each attribute in a [`CollectionSchema`](crate::schema::CollectionSchema).
pub type AttrId = u16;

/// Unique numeric identifier assigned to each `(AttrId, token)` term in the FTS inverted index.
pub type TermId = u32;

/// Monotonic generation identifier for detached columns.
pub type ColumnGen = u32;

/// Universal 256-row block granularity (FTS posting blocks, doc lengths, column blocks, `RowId -> DocId` blocks).
pub const ROW_BLOCK_SIZE: usize = 256;

/// Alias for [`ROW_BLOCK_SIZE`] used by FTS block structures.
pub const FTS_BLOCK_SIZE: usize = ROW_BLOCK_SIZE;

/// Number of 256-row blocks per FTS skip summary chunk.
pub const FTS_SKIP_CHUNK_SIZE: usize = 256;

/// Hard cap on rows per segment (`2^20 = 1,048,576`, at most 4,096 blocks).
pub const MAX_SEGMENT_ROWS: u32 = 1 << 20;

/// Default row count threshold at which an open segment is sealed (`256 * 256 = 65,536`).
pub const DEFAULT_SEAL_ROWS: u32 = 65_536;

/// Internal segment-addressed row coordinate: `(segment_id << 32) | local_id`.
///
/// Numeric ordering is segment-major (`(segment_id, local_id)` ascending).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct RowId(pub u64);

impl RowId {
    #[inline]
    pub const fn new(seg: SegmentId, local: LocalId) -> Self {
        Self(((seg as u64) << 32) | (local as u64))
    }

    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn seg(self) -> SegmentId {
        (self.0 >> 32) as SegmentId
    }

    #[inline]
    pub const fn local(self) -> LocalId {
        self.0 as LocalId
    }

    #[inline]
    pub const fn block(self) -> RowBlock {
        RowBlock::new(self.seg(), self.local() >> 8)
    }

    #[inline]
    pub const fn block_idx(self) -> u32 {
        self.local() >> 8
    }

    #[inline]
    pub const fn slot(self) -> u8 {
        (self.local() & 0xFF) as u8
    }

    #[inline]
    pub const fn to_be_bytes(self) -> [u8; 8] {
        self.0.to_be_bytes()
    }

    #[inline]
    pub const fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_be_bytes(bytes))
    }

    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_le_bytes(bytes))
    }
}

impl fmt::Debug for RowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RowId({}, {})", self.seg(), self.local())
    }
}

impl fmt::Display for RowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.seg(), self.local())
    }
}

/// 256-row block coordinate within a segment: `(segment_id << 32) | block_idx`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct RowBlock(pub u64);

impl RowBlock {
    #[inline]
    pub const fn new(seg: SegmentId, block: u32) -> Self {
        Self(((seg as u64) << 32) | (block as u64))
    }

    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn seg(self) -> SegmentId {
        (self.0 >> 32) as SegmentId
    }

    #[inline]
    pub const fn block(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    pub const fn chunk(self) -> SkipChunk {
        SkipChunk::new(self.seg(), self.block() >> 8)
    }

    #[inline]
    pub const fn chunk_idx(self) -> u32 {
        self.block() >> 8
    }

    #[inline]
    pub const fn rel_block(self) -> u8 {
        (self.block() & 0xFF) as u8
    }

    #[inline]
    pub const fn row(self, slot: u8) -> RowId {
        RowId::new(self.seg(), (self.block() << 8) | (slot as u32))
    }

    #[inline]
    pub const fn to_be_bytes(self) -> [u8; 8] {
        self.0.to_be_bytes()
    }

    #[inline]
    pub const fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_be_bytes(bytes))
    }
}

impl fmt::Debug for RowBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RowBlock({}, {})", self.seg(), self.block())
    }
}

/// 256-block FTS skip chunk coordinate within a segment: `(segment_id << 32) | chunk_idx`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct SkipChunk(pub u64);

impl SkipChunk {
    #[inline]
    pub const fn new(seg: SegmentId, chunk: u32) -> Self {
        Self(((seg as u64) << 32) | (chunk as u64))
    }

    #[inline]
    pub const fn seg(self) -> SegmentId {
        (self.0 >> 32) as SegmentId
    }

    #[inline]
    pub const fn chunk(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    pub const fn block(self, rel_block: u8) -> RowBlock {
        RowBlock::new(self.seg(), (self.chunk() << 8) | (rel_block as u32))
    }

    #[inline]
    pub const fn to_be_bytes(self) -> [u8; 8] {
        self.0.to_be_bytes()
    }

    #[inline]
    pub const fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_be_bytes(bytes))
    }
}

impl fmt::Debug for SkipChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SkipChunk({}, {})", self.seg(), self.chunk())
    }
}

/// Segment lifecycle configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentConfig {
    /// Maximum rows appended to an open segment before it is sealed.
    pub seal_rows: u32,
    /// If `true`, sealing skips building `cf_columns` inline and defers it to `maintenance()`.
    pub defer_column_build: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            seal_rows: DEFAULT_SEAL_ROWS,
            defer_column_build: false,
        }
    }
}

impl SegmentConfig {
    pub fn new(seal_rows: u32) -> Self {
        Self {
            seal_rows,
            defer_column_build: false,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.seal_rows == 0 || self.seal_rows > MAX_SEGMENT_ROWS {
            return Err(TreeError::InvalidConfig(format!(
                "seal_rows must be in 1..={MAX_SEGMENT_ROWS}, got {}",
                self.seal_rows
            )));
        }
        Ok(())
    }
}

/// Iterate over each set bit index (`0..=255`) in a 256-bit `[u64; 4]` block mask in ascending order.
#[inline]
pub fn for_each_set_bit_256(mask: &[u64; 4], mut f: impl FnMut(u8)) {
    for (w, &bits) in mask.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let bit = b.trailing_zeros() as usize;
            b &= b - 1;
            f(((w << 6) | bit) as u8);
        }
    }
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
