use rocksdb::WriteBatch;

use izzydb_core::config::{DocId, FtsBlockId, NodeId};
use izzydb_core::schema::CollectionSchema;

use crate::codec::encode_fts_block_key;
use crate::node::{PackedInternalNode, PackedLeafNode, TreeMeta, encode_f32_slice};

use super::types::{DocLoc, doc_key, node_key};
use super::{
    CF_ATTR_CLUSTERS, CF_ATTR_POSTINGS, CF_ATTR_ZONES, CF_DOC_LOC, CF_DOCS, CF_FTS_DOC_LENS,
    CF_FTS_POSITIONS, CF_FTS_POSTINGS, CF_FTS_REV_DOC_MAP, CF_FTS_SKIPS, CF_FTS_TERMS,
    CF_INTERNAL_NODES, CF_LEAF_ENTRIES, CF_META, CF_VECTORS, FTS_FREE_IDS_KEY, FTS_META_KEY,
    META_KEY, SCHEMA_KEY, Storage,
};

/// Typed wrapper around `rocksdb::WriteBatch` for atomic tree and attribute index mutations.
pub struct StorageBatch<'a> {
    pub(super) storage: &'a Storage,
    pub(super) batch: WriteBatch,
}

impl<'a> StorageBatch<'a> {
    fn put_cf(&mut self, cf_name: &str, key: &[u8], val: &[u8]) {
        let cf = self.storage.db.cf_handle(cf_name).unwrap();
        self.batch.put_cf(cf, key, val);
    }

    fn delete_cf(&mut self, cf_name: &str, key: &[u8]) {
        let cf = self.storage.db.cf_handle(cf_name).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_meta(&mut self, meta: &TreeMeta) {
        self.put_cf(CF_META, META_KEY, &meta.encode());
    }

    pub fn put_schema(&mut self, schema: &CollectionSchema) {
        self.put_cf(CF_META, SCHEMA_KEY, &schema.encode());
    }

    pub fn put_internal_node(&mut self, node_id: NodeId, node: &PackedInternalNode) {
        self.put_cf(CF_INTERNAL_NODES, &node_key(node_id), &node.encode(self.storage.dim));
    }

    pub fn delete_internal_node(&mut self, node_id: NodeId) {
        self.delete_cf(CF_INTERNAL_NODES, &node_key(node_id));
    }

    pub fn put_leaf_node(&mut self, leaf_id: NodeId, leaf: &PackedLeafNode) {
        let bytes = leaf.encode(
            self.storage.dim,
            self.storage.padded_dim,
            self.storage.rabitq_words,
        );
        self.put_cf(CF_LEAF_ENTRIES, &node_key(leaf_id), &bytes);
    }

    pub fn delete_leaf_node(&mut self, leaf_id: NodeId) {
        self.delete_cf(CF_LEAF_ENTRIES, &node_key(leaf_id));
    }

    pub fn put_vector(&mut self, doc_id: DocId, vec: &[f32]) {
        self.put_cf(CF_VECTORS, &doc_key(doc_id), &encode_f32_slice(vec));
    }

    pub fn delete_vector(&mut self, doc_id: DocId) {
        self.delete_cf(CF_VECTORS, &doc_key(doc_id));
    }

    pub fn put_doc_loc(&mut self, doc_id: DocId, loc: DocLoc) {
        self.put_cf(CF_DOC_LOC, &doc_key(doc_id), &loc.encode());
    }

    pub fn delete_doc_loc(&mut self, doc_id: DocId) {
        self.delete_cf(CF_DOC_LOC, &doc_key(doc_id));
    }

    pub fn put_doc_bytes(&mut self, doc_id: DocId, bytes: &[u8]) {
        self.put_cf(CF_DOCS, &doc_key(doc_id), bytes);
    }

    pub fn delete_doc(&mut self, doc_id: DocId) {
        self.delete_cf(CF_DOCS, &doc_key(doc_id));
    }

    pub fn put_attr_cluster(&mut self, key: &[u8], bytes: &[u8]) {
        self.put_cf(CF_ATTR_CLUSTERS, key, bytes);
    }

    pub fn delete_attr_cluster(&mut self, key: &[u8]) {
        self.delete_cf(CF_ATTR_CLUSTERS, key);
    }

    pub fn put_attr_posting(&mut self, key: &[u8], bytes: &[u8]) {
        self.put_cf(CF_ATTR_POSTINGS, key, bytes);
    }

    pub fn delete_attr_posting(&mut self, key: &[u8]) {
        self.delete_cf(CF_ATTR_POSTINGS, key);
    }

    pub fn put_attr_zone(&mut self, key: &[u8; 3], bytes: &[u8]) {
        self.put_cf(CF_ATTR_ZONES, key, bytes);
    }

    pub fn delete_attr_zone(&mut self, key: &[u8; 3]) {
        self.delete_cf(CF_ATTR_ZONES, key);
    }

    pub fn put_fts_meta_bytes(&mut self, bytes: &[u8]) {
        self.put_cf(CF_META, FTS_META_KEY, bytes);
    }

    pub fn put_fts_free_ids(&mut self, bytes: &[u8]) {
        self.put_cf(CF_META, FTS_FREE_IDS_KEY, bytes);
    }

    pub fn delete_fts_free_ids(&mut self) {
        self.delete_cf(CF_META, FTS_FREE_IDS_KEY);
    }

    pub fn put_fts_rev_block(&mut self, block_id: FtsBlockId, bytes: &[u8]) {
        self.put_cf(CF_FTS_REV_DOC_MAP, &encode_fts_block_key(block_id), bytes);
    }

    pub fn delete_fts_rev_block(&mut self, block_id: FtsBlockId) {
        self.delete_cf(CF_FTS_REV_DOC_MAP, &encode_fts_block_key(block_id));
    }

    pub fn put_fts_term(&mut self, key: &[u8], bytes: &[u8]) {
        self.put_cf(CF_FTS_TERMS, key, bytes);
    }

    pub fn delete_fts_term(&mut self, key: &[u8]) {
        self.delete_cf(CF_FTS_TERMS, key);
    }

    pub fn put_fts_skip(&mut self, key: &[u8; 8], bytes: &[u8]) {
        self.put_cf(CF_FTS_SKIPS, key, bytes);
    }

    pub fn delete_fts_skip(&mut self, key: &[u8; 8]) {
        self.delete_cf(CF_FTS_SKIPS, key);
    }

    pub fn put_fts_posting(&mut self, key: &[u8; 8], bytes: &[u8]) {
        self.put_cf(CF_FTS_POSTINGS, key, bytes);
    }

    pub fn delete_fts_posting(&mut self, key: &[u8; 8]) {
        self.delete_cf(CF_FTS_POSTINGS, key);
    }

    pub fn put_fts_position(&mut self, key: &[u8; 8], bytes: &[u8]) {
        self.put_cf(CF_FTS_POSITIONS, key, bytes);
    }

    pub fn delete_fts_position(&mut self, key: &[u8; 8]) {
        self.delete_cf(CF_FTS_POSITIONS, key);
    }

    pub fn put_fts_doc_len(&mut self, key: &[u8; 6], bytes: &[u8]) {
        self.put_cf(CF_FTS_DOC_LENS, key, bytes);
    }

    pub fn delete_fts_doc_len(&mut self, key: &[u8; 6]) {
        self.delete_cf(CF_FTS_DOC_LENS, key);
    }
}
