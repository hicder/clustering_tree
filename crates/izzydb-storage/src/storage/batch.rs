use rocksdb::WriteBatch;

use izzydb_core::config::{AttrId, ColumnGen, DocId, NodeId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentId, SkipChunk, TermId};
use izzydb_core::schema::CollectionSchema;

use crate::codec::{
    ATTR_COUNT_TAG, DETACHED_COUNT_TAG, DETACHED_META_TAG, DETACHED_OVERWRITTEN_TAG, DetachedMeta,
    FTS_DOC_LEN_TAG, INTERNAL_NODE_TAG, Manifest, SEG_DELETES_TAG, SEG_META_TAG, SegmentMeta,
    ZONE_TAG, ZoneEntry, detached_meta_key, detached_overwritten_key, doc_id_block_key,
    doc_len_key_coord, encode_doc_id_block, fts_block_key_coord, fts_chunk_key_coord, row_key,
    seg_deletes_key, seg_meta_key, zone_key,
};
use crate::node::{AnnMeta, PackedInternalNode, PackedLeafNode, encode_f32_slice};

use super::{
    ANN_META_KEY, CF_ATTR_POSTINGS, CF_COLUMNS, CF_DETACHED, CF_DETACHED_POSTINGS,
    CF_FTS_POSITIONS, CF_FTS_POSTINGS, CF_FTS_SKIPS, CF_FTS_TERMS, CF_ID_MAP, CF_LEAF_ENTRIES,
    CF_ROW_DOC_IDS, CF_ROWS, CF_VECTORS, DROPPED_SEGMENTS_KEY, FTS_META_KEY, MANIFEST_KEY,
    SCHEMA_KEY, Storage, doc_key, node_key, tagged_key,
};

/// Typed wrapper around `rocksdb::WriteBatch` for atomic v3 storage mutations.
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

    pub fn delete_range_cf(&mut self, cf_name: &str, from: &[u8], to: &[u8]) {
        let cf = self.storage.db.cf_handle(cf_name).unwrap();
        self.batch.delete_range_cf(cf, from, to);
    }

    pub fn delete_range_default(&mut self, from: &[u8], to: &[u8]) {
        self.batch.delete_range(from, to);
    }

    pub fn put_manifest(&mut self, manifest: &Manifest) {
        self.batch.put(MANIFEST_KEY, manifest.encode());
    }

    pub fn put_ann_meta(&mut self, meta: &AnnMeta) {
        self.batch.put(ANN_META_KEY, meta.encode());
    }

    pub fn put_schema(&mut self, schema: &CollectionSchema) {
        self.batch.put(SCHEMA_KEY, schema.encode());
    }

    pub fn put_dropped_segments(&mut self, bytes: &[u8]) {
        self.batch.put(DROPPED_SEGMENTS_KEY, bytes);
    }

    pub fn put_seg_meta(&mut self, seg: SegmentId, meta: &SegmentMeta) {
        self.batch.put(seg_meta_key(seg), meta.encode());
    }

    pub fn delete_seg_meta(&mut self, seg: SegmentId) {
        let _ = SEG_META_TAG;
        self.batch.delete(seg_meta_key(seg));
    }

    pub fn put_seg_deletes(&mut self, seg: SegmentId, bytes: &[u8]) {
        self.batch.put(seg_deletes_key(seg), bytes);
    }

    pub fn delete_seg_deletes(&mut self, seg: SegmentId) {
        let _ = SEG_DELETES_TAG;
        self.batch.delete(seg_deletes_key(seg));
    }

    pub fn put_zone_entry(&mut self, seg: SegmentId, attr_id: AttrId, tag: u8, entry: &ZoneEntry) {
        let _ = ZONE_TAG;
        self.batch.put(zone_key(seg, attr_id, tag), entry.encode());
    }

    pub fn delete_zone_entry(&mut self, seg: SegmentId, attr_id: AttrId, tag: u8) {
        self.batch.delete(zone_key(seg, attr_id, tag));
    }

    pub fn put_detached_meta(&mut self, attr_id: AttrId, meta: &DetachedMeta) {
        let _ = DETACHED_META_TAG;
        self.batch.put(detached_meta_key(attr_id), meta.encode());
    }

    pub fn put_detached_overwritten(
        &mut self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        seg: SegmentId,
        bytes: &[u8],
    ) {
        let _ = DETACHED_OVERWRITTEN_TAG;
        self.batch
            .put(detached_overwritten_key(attr_id, col_gen, seg), bytes);
    }

    pub fn delete_detached_overwritten(
        &mut self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        seg: SegmentId,
    ) {
        self.batch
            .delete(detached_overwritten_key(attr_id, col_gen, seg));
    }

    pub fn put_internal_node(&mut self, node_id: NodeId, node: &PackedInternalNode) {
        let key = tagged_key::<5>(INTERNAL_NODE_TAG, &node_key(node_id));
        self.batch.put(key, node.encode(self.storage.dim));
    }

    pub fn delete_internal_node(&mut self, node_id: NodeId) {
        self.batch
            .delete(tagged_key::<5>(INTERNAL_NODE_TAG, &node_key(node_id)));
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

    pub fn put_id_map(&mut self, doc_id: DocId, row_id: RowId) {
        self.put_cf(CF_ID_MAP, &doc_key(doc_id), &row_key(row_id));
    }

    pub fn delete_id_map(&mut self, doc_id: DocId) {
        self.delete_cf(CF_ID_MAP, &doc_key(doc_id));
    }

    pub fn put_row_bytes(&mut self, row: RowId, bytes: &[u8]) {
        self.put_cf(CF_ROWS, &row_key(row), bytes);
    }

    pub fn delete_row(&mut self, row: RowId) {
        self.delete_cf(CF_ROWS, &row_key(row));
    }

    pub fn put_doc_id_block(&mut self, seg: SegmentId, block: u32, ids: &[DocId; ROW_BLOCK_SIZE]) {
        self.put_cf(
            CF_ROW_DOC_IDS,
            &doc_id_block_key(seg, block),
            &encode_doc_id_block(ids),
        );
    }

    pub fn put_vector(&mut self, row: RowId, vec: &[f32]) {
        self.put_cf(CF_VECTORS, &row_key(row), &encode_f32_slice(vec));
    }

    pub fn put_vector_bytes(&mut self, row: RowId, bytes: &[u8]) {
        self.put_cf(CF_VECTORS, &row_key(row), bytes);
    }

    pub fn delete_vector(&mut self, row: RowId) {
        self.delete_cf(CF_VECTORS, &row_key(row));
    }

    pub fn put_attr_posting(&mut self, key: &[u8], bytes: &[u8]) {
        self.put_cf(CF_ATTR_POSTINGS, key, bytes);
    }

    pub fn delete_attr_posting(&mut self, key: &[u8]) {
        self.delete_cf(CF_ATTR_POSTINGS, key);
    }

    pub fn put_column_block(&mut self, key: &[u8; 10], bytes: &[u8]) {
        self.put_cf(CF_COLUMNS, key, bytes);
    }

    pub fn put_detached_block(&mut self, key: &[u8; 14], bytes: &[u8]) {
        self.put_cf(CF_DETACHED, key, bytes);
    }

    pub fn delete_detached_block(&mut self, key: &[u8; 14]) {
        self.delete_cf(CF_DETACHED, key);
    }

    pub fn put_detached_posting(&mut self, key: &[u8], bytes: &[u8]) {
        self.put_cf(CF_DETACHED_POSTINGS, key, bytes);
    }

    pub fn delete_detached_posting(&mut self, key: &[u8]) {
        self.delete_cf(CF_DETACHED_POSTINGS, key);
    }

    pub fn put_attr_count(&mut self, attr_key: &[u8], count: u64) {
        let mut key = Vec::with_capacity(1 + attr_key.len());
        key.push(ATTR_COUNT_TAG);
        key.extend_from_slice(attr_key);
        self.batch.put(key, count.to_le_bytes());
    }

    pub fn delete_attr_count(&mut self, attr_key: &[u8]) {
        let mut key = Vec::with_capacity(1 + attr_key.len());
        key.push(ATTR_COUNT_TAG);
        key.extend_from_slice(attr_key);
        self.batch.delete(key);
    }

    pub fn put_detached_count(&mut self, suffix: &[u8], count: u64) {
        let mut key = Vec::with_capacity(1 + suffix.len());
        key.push(DETACHED_COUNT_TAG);
        key.extend_from_slice(suffix);
        self.batch.put(key, count.to_le_bytes());
    }

    pub fn delete_detached_count(&mut self, suffix: &[u8]) {
        let mut key = Vec::with_capacity(1 + suffix.len());
        key.push(DETACHED_COUNT_TAG);
        key.extend_from_slice(suffix);
        self.batch.delete(key);
    }

    pub fn put_fts_meta_bytes(&mut self, bytes: &[u8]) {
        self.batch.put(FTS_META_KEY, bytes);
    }

    pub fn put_fts_term(&mut self, key: &[u8], bytes: &[u8]) {
        self.put_cf(CF_FTS_TERMS, key, bytes);
    }

    pub fn delete_fts_term(&mut self, key: &[u8]) {
        self.delete_cf(CF_FTS_TERMS, key);
    }

    pub fn put_fts_skip(&mut self, term_id: TermId, chunk: SkipChunk, bytes: &[u8]) {
        self.put_cf(CF_FTS_SKIPS, &fts_chunk_key_coord(term_id, chunk), bytes);
    }

    pub fn delete_fts_skip(&mut self, term_id: TermId, chunk: SkipChunk) {
        self.delete_cf(CF_FTS_SKIPS, &fts_chunk_key_coord(term_id, chunk));
    }

    pub fn put_fts_posting(&mut self, term_id: TermId, block: RowBlock, bytes: &[u8]) {
        self.put_cf(CF_FTS_POSTINGS, &fts_block_key_coord(term_id, block), bytes);
    }

    pub fn delete_fts_posting(&mut self, term_id: TermId, block: RowBlock) {
        self.delete_cf(CF_FTS_POSTINGS, &fts_block_key_coord(term_id, block));
    }

    pub fn put_fts_position(&mut self, term_id: TermId, block: RowBlock, bytes: &[u8]) {
        self.put_cf(
            CF_FTS_POSITIONS,
            &fts_block_key_coord(term_id, block),
            bytes,
        );
    }

    pub fn delete_fts_position(&mut self, term_id: TermId, block: RowBlock) {
        self.delete_cf(CF_FTS_POSITIONS, &fts_block_key_coord(term_id, block));
    }

    pub fn put_fts_doc_len(&mut self, attr_id: AttrId, block: RowBlock, bytes: &[u8]) {
        let k = doc_len_key_coord(attr_id, block);
        self.batch.put(tagged_key::<11>(FTS_DOC_LEN_TAG, &k), bytes);
    }

    pub fn delete_fts_doc_len(&mut self, attr_id: AttrId, block: RowBlock) {
        let k = doc_len_key_coord(attr_id, block);
        self.batch.delete(tagged_key::<11>(FTS_DOC_LEN_TAG, &k));
    }
}
