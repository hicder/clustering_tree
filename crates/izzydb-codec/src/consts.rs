use izzydb_core::config::ROW_BLOCK_SIZE;

/// Reserved `type_tag` tracking non-null attribute presence for `IsNull` / `IsNotNull`.
pub const TAG_PRESENT: u8 = 0x00;
pub const TAG_BOOL: u8 = 0x01;
pub const TAG_INT: u8 = 0x02;
pub const TAG_UINT: u8 = 0x03;
pub const TAG_FLOAT: u8 = 0x04;
pub const TAG_STRING: u8 = 0x05;

// `default` CF keyspace tags.
pub const INTERNAL_NODE_TAG: u8 = 0x02;
pub const FTS_DOC_LEN_TAG: u8 = 0x05;
pub const ATTR_COUNT_TAG: u8 = 0x06;
pub const DETACHED_COUNT_TAG: u8 = 0x07;
pub const SEG_DELETES_TAG: u8 = 0x10;
pub const SEG_META_TAG: u8 = 0x11;
pub const ZONE_TAG: u8 = 0x12;
pub const DETACHED_META_TAG: u8 = 0x13;
pub const DETACHED_OVERWRITTEN_TAG: u8 = 0x14;

pub(super) const ROW_FORMAT_V3: u8 = 0x03;
pub(super) const ROW_FLAG_HAS_VECTOR: u8 = 0x01;
pub(super) const ROW_HEADER_LEN: usize = 4;
pub(super) const ROW_DIR_ENTRY_LEN: usize = 8;

pub(super) const ROW_VAL_NULL: u8 = 0;
pub(super) const ROW_VAL_BOOL: u8 = 1;
pub(super) const ROW_VAL_INT: u8 = 2;
pub(super) const ROW_VAL_UINT: u8 = 3;
pub(super) const ROW_VAL_FLOAT: u8 = 4;
pub(super) const ROW_VAL_STRING: u8 = 5;
pub(super) const ROW_VAL_STRING_ARRAY: u8 = 6;
pub(super) const ROW_VAL_INT_ARRAY: u8 = 7;
pub(super) const ROW_VAL_UINT_ARRAY: u8 = 8;

pub(super) const SEG_POSTING_SPARSE: u8 = 0x00;
pub(super) const SEG_POSTING_ROARING: u8 = 0x01;
pub const SEG_POSTING_SPARSE_MAX: usize = 32;

pub const DOC_ID_BLOCK_BYTES: usize = ROW_BLOCK_SIZE * 8;
pub const MANIFEST_BYTES: usize = 32;
pub const SEGMENT_META_BYTES: usize = 44;
pub const DETACHED_META_BYTES: usize = 24;

pub const SEG_FLAG_HAS_COLUMNS: u8 = 0x01;
pub const SEG_FLAG_HAS_VECTORS: u8 = 0x02;

// ---------------------------------------------------------------------------
