//! RocksDB-backed packed node storage, binary codecs, and transaction overlay for `izzydb`.

pub mod codec;
pub mod node;
pub mod storage;

pub use node::{
    LeafNodeView, PackedInternalNode, PackedLeafNode, TreeMeta, borrow_f32_slice,
    decode_f32_into, decode_f32_slice, encode_f32_slice,
};
pub use rocksdb::DBPinnableSlice;
pub use storage::{
    ALL_DATA_CF_NAMES, ATTR_COUNT_TAG, ATTR_COUNTS_VERSION_KEY, ATTR_ZONE_TAG,
    BULK_IMPORT_MARKER_KEY, AttrZoneMap, ByteRange, CF_ATTR_CLUSTERS, CF_ATTR_POSTINGS, CF_DOC_LOC,
    CF_DOCS, CF_FTS_POSITIONS, CF_FTS_POSTINGS, CF_FTS_SKIPS, CF_FTS_TERMS, CF_LEAF_ENTRIES,
    CF_VECTORS, DEFAULT_CF_NAME, DocLoc, FTS_DOC_LEN_TAG, FTS_FREE_IDS_KEY, FTS_META_KEY,
    FTS_REV_BLOCK_TAG, FtsRevBlock, INTERNAL_NODE_TAG, META_KEY, RocksDocLenCursor,
    RocksSkipCursor, SCHEMA_KEY, Storage, StorageBatch, TxContext, cf_options_for,
    default_sst_block_cache, doc_key, encode_count, node_key, tagged_key, tagged_vec,
};
