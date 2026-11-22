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
    AttrZoneMap, ByteRange, CF_ATTR_CLUSTERS, CF_ATTR_POSTINGS, CF_DOC_LOC, CF_DOCS, CF_FTS_POSITIONS,
    CF_FTS_POSTINGS, CF_FTS_SKIPS, CF_FTS_TERMS, CF_LEAF_ENTRIES, CF_VECTORS, DocLoc, FtsRevBlock,
    RocksDocLenCursor, RocksSkipCursor, Storage, StorageBatch, TxContext,
};
