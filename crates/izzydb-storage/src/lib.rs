//! RocksDB-backed packed node storage, binary codecs, and transaction overlay for `izzydb`.

pub use izzydb_codec as codec;
pub mod bsi;
pub mod column;
pub mod leaf_attrs;
pub mod node;
pub mod storage;

pub use column::{ColumnBlockBuilder, ColumnBlockView};

pub use leaf_attrs::{
    AttrSummary, LeafAttrBlock, LeafAttrBlockBuilder, LeafAttrColumn, LeafAttrSpec, LeafAttrValue,
    LeafAttrView, NodeSummary,
};
pub use node::{
    AnnMeta, LeafNodeView, PackedInternalNode, PackedLeafNode, TreeMeta, borrow_f32_slice,
    decode_f32_into, decode_f32_slice, encode_f32_slice,
};
pub use rocksdb::DBPinnableSlice;
pub use storage::{
    ALL_DATA_CF_NAMES, ANN_META_KEY, ATTR_COUNTS_VERSION_KEY, BULK_IMPORT_MARKER_KEY, ByteRange,
    CF_ATTR_POSTINGS, CF_BSI, CF_COLUMNS, CF_DETACHED, CF_DETACHED_POSTINGS, CF_FTS_POSITIONS,
    CF_FTS_POSTINGS, CF_FTS_SKIPS, CF_FTS_TERMS, CF_ID_MAP, CF_LEAF_ENTRIES, CF_NODE_SUMMARIES,
    CF_ROW_DOC_IDS, CF_ROWS, CF_VECTORS, DEFAULT_CF_NAME, DROPPED_SEGMENTS_KEY,
    DocLenBlockTable, FTS_META_KEY, MANIFEST_KEY, LeafVisitAction, RocksDocLenCursor,
    RocksSkipCursor, SCHEMA_KEY, SegmentEntry, SegmentTable, SegmentTableSnapshot, Storage,
    StorageBatch, TxContext, ZoneTable, cf_options_for, default_sst_block_cache, doc_key,
    encode_count, node_key, tagged_key, tagged_vec,
};
