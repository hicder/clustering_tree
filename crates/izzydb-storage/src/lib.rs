//! RocksDB-backed packed node storage, binary codecs, and transaction overlay for `izzydb`.

pub mod codec;
pub mod node;
pub mod storage;

pub use node::{
    LeafNodeView, PackedInternalNode, PackedLeafNode, TreeMeta, borrow_f32_slice,
    decode_f32_into, decode_f32_slice, encode_f32_slice,
};
pub use storage::{
    CF_ATTR_CLUSTERS, CF_ATTR_POSTINGS, CF_DOC_LOC, CF_DOCS, CF_FTS_DOC_LENS, CF_FTS_DOC_MAP,
    CF_FTS_POSITIONS, CF_FTS_POSTINGS, CF_FTS_REV_DOC_MAP, CF_FTS_TERMS, CF_INTERNAL_NODES,
    CF_LEAF_ENTRIES, CF_META, CF_VECTORS, DocLoc, RocksDocLenCursor, RocksPostingCursor,
    Storage, StorageBatch, TxContext,
};
