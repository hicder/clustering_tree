use std::collections::{HashMap, HashSet};
use std::path::Path;

use roaring::RoaringBitmap;
use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamily, ColumnFamilyDescriptor, DBCompressionType,
    DBPinnableSlice, DBRawIteratorWithThreadMode, Direction, IteratorMode, Options, ReadOptions,
    SliceTransform, WriteBatch, DB, DEFAULT_COLUMN_FAMILY_NAME,
};

use izzydb_core::config::{DocId, FtsBlockId, FtsDocId, LocalId, NodeId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::CollectionSchema;

use crate::codec::{decode_roaring_bitmap, encode_roaring_bitmap, make_posting_key};
use crate::node::{
    PackedInternalNode, PackedLeafNode, TreeMeta, decode_f32_into, decode_f32_slice,
    encode_f32_slice,
};

#[inline]
fn map_db_err(err: rocksdb::Error) -> TreeError {
    TreeError::Storage(err.to_string())
}

pub const CF_META: &str = "cf_meta";
pub const CF_INTERNAL_NODES: &str = "cf_internal_nodes";
pub const CF_LEAF_ENTRIES: &str = "cf_leaf_entries";
pub const CF_VECTORS: &str = "cf_vectors";
pub const CF_DOC_LOC: &str = "cf_doc_loc";
pub const CF_DOCS: &str = "cf_docs";
pub const CF_ATTR_CLUSTERS: &str = "cf_attr_clusters";
pub const CF_ATTR_POSTINGS: &str = "cf_attr_postings";
pub const CF_FTS_DOC_MAP: &str = "cf_fts_doc_map";
pub const CF_FTS_REV_DOC_MAP: &str = "cf_fts_rev_doc_map";
pub const CF_FTS_TERMS: &str = "cf_fts_terms";
pub const CF_FTS_POSTINGS: &str = "cf_fts_postings";
pub const CF_FTS_POSITIONS: &str = "cf_fts_positions";

const META_KEY: &[u8] = b"tree_meta";
const SCHEMA_KEY: &[u8] = b"schema";
const FTS_META_KEY: &[u8] = b"fts_meta";
const BLOCK_CACHE_BYTES: usize = 128 * 1024 * 1024;

/// Physical location of a document in the clustering tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocLoc {
    pub leaf_id: NodeId,
    pub local_id: LocalId,
}

impl DocLoc {
    #[inline]
    pub fn encode(self) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[0..4].copy_from_slice(&self.leaf_id.to_be_bytes());
        out[4..8].copy_from_slice(&self.local_id.to_be_bytes());
        out
    }

    #[inline]
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 8 {
            return Err(TreeError::CorruptedData(format!(
                "DocLoc expected 8 bytes, got {}",
                bytes.len()
            )));
        }
        Ok(Self {
            leaf_id: u32::from_le_bytes_or_be(bytes[0..4].try_into().unwrap()),
            local_id: u32::from_be_bytes(bytes[4..8].try_into().unwrap()),
        })
    }
}

trait U32FromBe {
    fn from_le_bytes_or_be(bytes: [u8; 4]) -> u32;
}
impl U32FromBe for u32 {
    #[inline]
    fn from_le_bytes_or_be(bytes: [u8; 4]) -> u32 {
        u32::from_be_bytes(bytes)
    }
}

/// RocksDB storage wrapper managing packed node, document, and attribute index column families.
pub struct Storage {
    db: DB,
    _block_cache: Cache,
    dim: usize,
    padded_dim: usize,
    rabitq_words: usize,
}

impl Storage {
    /// Open or initialize the RocksDB instance and validate/create `TreeMeta`.
    pub fn open(path: impl AsRef<Path>, config: &TreeConfig) -> Result<(Self, TreeMeta)> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        let block_cache = Cache::new_lru_cache(BLOCK_CACHE_BYTES);

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(DEFAULT_COLUMN_FAMILY_NAME, Options::default()),
            ColumnFamilyDescriptor::new(CF_META, Options::default()),
            ColumnFamilyDescriptor::new(CF_INTERNAL_NODES, point_lookup_cf_opts(&block_cache, false)),
            ColumnFamilyDescriptor::new(CF_LEAF_ENTRIES, point_lookup_cf_opts(&block_cache, true)),
            ColumnFamilyDescriptor::new(CF_VECTORS, point_lookup_cf_opts(&block_cache, true)),
            ColumnFamilyDescriptor::new(CF_DOC_LOC, point_lookup_cf_opts(&block_cache, false)),
            ColumnFamilyDescriptor::new(CF_DOCS, point_lookup_cf_opts(&block_cache, false)),
            ColumnFamilyDescriptor::new(CF_ATTR_CLUSTERS, attr_cluster_cf_opts(&block_cache)),
            ColumnFamilyDescriptor::new(CF_ATTR_POSTINGS, point_lookup_cf_opts(&block_cache, false)),
            ColumnFamilyDescriptor::new(CF_FTS_DOC_MAP, point_lookup_cf_opts(&block_cache, false)),
            ColumnFamilyDescriptor::new(
                CF_FTS_REV_DOC_MAP,
                point_lookup_cf_opts(&block_cache, false),
            ),
            ColumnFamilyDescriptor::new(CF_FTS_TERMS, fts_terms_cf_opts(&block_cache)),
            ColumnFamilyDescriptor::new(CF_FTS_POSTINGS, point_lookup_cf_opts(&block_cache, false)),
            ColumnFamilyDescriptor::new(
                CF_FTS_POSITIONS,
                point_lookup_cf_opts(&block_cache, false),
            ),
        ];

        let db = DB::open_cf_descriptors(&db_opts, path, cf_descriptors).map_err(map_db_err)?;
        let padded_dim = config.dim.next_power_of_two().max(64);
        let rabitq_words = padded_dim / 64;
        let storage = Self {
            db,
            _block_cache: block_cache,
            dim: config.dim,
            padded_dim,
            rabitq_words,
        };

        let meta = match storage.get_meta()? {
            Some(existing) => {
                if existing.dim as usize != config.dim {
                    return Err(TreeError::ConfigMismatch {
                        field: "dim",
                        expected: existing.dim.to_string(),
                        actual: config.dim.to_string(),
                    });
                }
                if existing.metric != config.metric {
                    return Err(TreeError::ConfigMismatch {
                        field: "metric",
                        expected: format!("{:?}", existing.metric),
                        actual: format!("{:?}", config.metric),
                    });
                }
                if existing.rabitq_bits != 1 {
                    return Err(TreeError::CorruptedData(format!(
                        "unsupported rabitq_bits: {}",
                        existing.rabitq_bits
                    )));
                }
                existing
            }
            None => {
                let init_meta = TreeMeta {
                    root_id: 1,
                    root_level: 0,
                    next_node_id: 2,
                    dim: config.dim as u32,
                    metric: config.metric,
                    num_vectors: 0,
                    rabitq_seed: rand::random::<u64>(),
                    rabitq_bits: 1,
                };
                let init_leaf = PackedLeafNode::empty(None, config.dim, padded_dim);
                let mut batch = storage.new_batch();
                batch.put_meta(&init_meta);
                batch.put_leaf_node(1, &init_leaf);
                storage.write_batch(batch)?;
                init_meta
            }
        };

        Ok((storage, meta))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    pub fn rabitq_words(&self) -> usize {
        self.rabitq_words
    }

    pub fn new_batch(&self) -> StorageBatch<'_> {
        StorageBatch {
            storage: self,
            batch: WriteBatch::default(),
        }
    }

    pub fn write_batch(&self, batch: StorageBatch<'_>) -> Result<()> {
        self.db.write(batch.batch).map_err(map_db_err)?;
        Ok(())
    }

    pub fn get_meta(&self) -> Result<Option<TreeMeta>> {
        let cf = self.cf(CF_META)?;
        match self.db.get_cf(cf, META_KEY).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(TreeMeta::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn get_schema(&self) -> Result<Option<CollectionSchema>> {
        let cf = self.cf(CF_META)?;
        match self.db.get_cf(cf, SCHEMA_KEY).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(CollectionSchema::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn put_schema(&self, schema: &CollectionSchema) -> Result<()> {
        let mut batch = self.new_batch();
        batch.put_schema(schema);
        self.write_batch(batch)
    }

    pub fn get_doc_loc(&self, doc_id: DocId) -> Result<Option<DocLoc>> {
        let cf = self.cf(CF_DOC_LOC)?;
        match self.db.get_pinned_cf(cf, doc_key(doc_id)).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(DocLoc::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn multi_get_doc_locs(&self, doc_ids: &[DocId]) -> Result<Vec<Option<DocLoc>>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_DOC_LOC)?;
        let keys: Vec<[u8; 8]> = doc_ids.iter().map(|&id| doc_key(id)).collect();
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), false);
        let mut out = Vec::with_capacity(doc_ids.len());
        for res in results {
            match res.map_err(map_db_err)? {
                Some(slice) => out.push(Some(DocLoc::decode(&slice)?)),
                None => out.push(None),
            }
        }
        Ok(out)
    }

    pub fn get_doc_bytes(&self, doc_id: DocId) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_DOCS)?;
        self.db.get_cf(cf, doc_key(doc_id)).map_err(map_db_err)
    }

    pub fn multi_get_doc_bytes_pinned(
        &self,
        sorted_doc_ids: &[DocId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        self.batched_multi_get_doc_pinned(CF_DOCS, sorted_doc_ids, true)
    }

    pub fn get_vector(&self, doc_id: DocId) -> Result<Option<Vec<f32>>> {
        let cf = self.cf(CF_VECTORS)?;
        match self.db.get_pinned_cf(cf, doc_key(doc_id)).map_err(map_db_err)? {
            Some(slice) => Ok(Some(decode_f32_slice(&slice, self.dim)?)),
            None => Ok(None),
        }
    }

    pub fn get_internal_node(&self, node_id: NodeId) -> Result<Option<PackedInternalNode>> {
        let cf = self.cf(CF_INTERNAL_NODES)?;
        match self.db.get_pinned_cf(cf, node_key(node_id)).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(PackedInternalNode::decode(&bytes, self.dim)?)),
            None => Ok(None),
        }
    }

    pub fn get_leaf_node(&self, leaf_id: NodeId) -> Result<Option<PackedLeafNode>> {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        match self.db.get_pinned_cf(cf, node_key(leaf_id)).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(PackedLeafNode::decode(
                &bytes,
                self.dim,
                self.padded_dim,
                self.rabitq_words,
            )?)),
            None => Ok(None),
        }
    }

    /// Zero-copy batched lookup of quantized leaf blobs, assuming `sorted_leaf_ids` is sorted ascending.
    pub fn multi_get_leaf_nodes_pinned(
        &self,
        sorted_leaf_ids: &[NodeId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if sorted_leaf_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        let keys: Vec<[u8; 4]> = sorted_leaf_ids.iter().map(|&id| node_key(id)).collect();
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), true);
        let mut out = Vec::with_capacity(sorted_leaf_ids.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    /// Zero-copy batched lookup of full-precision vectors, assuming `sorted_doc_ids` is sorted ascending.
    pub fn multi_get_vectors_pinned(
        &self,
        sorted_doc_ids: &[DocId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        self.batched_multi_get_doc_pinned(CF_VECTORS, sorted_doc_ids, true)
    }

    fn batched_multi_get_doc_pinned(
        &self,
        cf_name: &str,
        ids: &[DocId],
        sorted_input: bool,
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(cf_name)?;
        let keys: Vec<[u8; 8]> = ids.iter().map(|&id| doc_key(id)).collect();
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted_input);
        let mut out = Vec::with_capacity(ids.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn get_attr_cluster_bitmap(&self, key: &[u8]) -> Result<Option<RoaringBitmap>> {
        let cf = self.cf(CF_ATTR_CLUSTERS)?;
        match self.db.get_pinned_cf(cf, key).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(decode_roaring_bitmap(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn multi_get_attr_cluster_bitmaps(
        &self,
        sorted_keys: &[Vec<u8>],
    ) -> Result<Vec<Option<RoaringBitmap>>> {
        if sorted_keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_ATTR_CLUSTERS)?;
        let results = self.db.batched_multi_get_cf(cf, sorted_keys.iter(), true);
        let mut out = Vec::with_capacity(sorted_keys.len());
        for res in results {
            match res.map_err(map_db_err)? {
                Some(slice) => out.push(Some(decode_roaring_bitmap(&slice)?)),
                None => out.push(None),
            }
        }
        Ok(out)
    }

    /// Scan `cf_attr_clusters` within a single `(attr_id, type_tag)` 3-byte prefix.
    ///
    /// Returns `(cluster_key, RoaringBitmap)` pairs for keys in `[start_key, end_key)` (or `<= end_key` if `end_inclusive`).
    pub fn scan_attr_cluster_range(
        &self,
        prefix: [u8; 3],
        start_key: &[u8],
        start_inclusive: bool,
        end_key: Option<&[u8]>,
        end_inclusive: bool,
    ) -> Result<Vec<(Vec<u8>, RoaringBitmap)>> {
        let cf = self.cf(CF_ATTR_CLUSTERS)?;
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self.db.iterator_cf_opt(
            cf,
            ro,
            IteratorMode::From(start_key, Direction::Forward),
        );
        let mut out = Vec::new();
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            if !start_inclusive && k.as_ref() == start_key {
                continue;
            }
            if let Some(end) = end_key {
                let cmp = k.as_ref().cmp(end);
                if if end_inclusive {
                    cmp == std::cmp::Ordering::Greater
                } else {
                    cmp != std::cmp::Ordering::Less
                } {
                    break;
                }
            }
            out.push((k.to_vec(), decode_roaring_bitmap(&v)?));
        }
        Ok(out)
    }

    pub fn get_attr_posting_bitmap(&self, key: &[u8]) -> Result<Option<RoaringBitmap>> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        match self.db.get_pinned_cf(cf, key).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(decode_roaring_bitmap(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn multi_get_attr_posting_bitmaps(
        &self,
        sorted_keys: &[Vec<u8>],
    ) -> Result<Vec<Option<RoaringBitmap>>> {
        if sorted_keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let results = self.db.batched_multi_get_cf(cf, sorted_keys.iter(), true);
        let mut out = Vec::with_capacity(sorted_keys.len());
        for res in results {
            match res.map_err(map_db_err)? {
                Some(slice) => out.push(Some(decode_roaring_bitmap(&slice)?)),
                None => out.push(None),
            }
        }
        Ok(out)
    }

    pub fn all_internal_nodes(&self) -> Result<Vec<(NodeId, PackedInternalNode)>> {
        let cf = self.cf(CF_INTERNAL_NODES)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((
                decode_u32_key(&k)?,
                PackedInternalNode::decode(&v, self.dim)?,
            ));
        }
        Ok(out)
    }

    pub fn all_leaf_nodes(&self) -> Result<Vec<(NodeId, PackedLeafNode)>> {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((
                decode_u32_key(&k)?,
                PackedLeafNode::decode(&v, self.dim, self.padded_dim, self.rabitq_words)?,
            ));
        }
        Ok(out)
    }

    pub fn all_vectors(&self) -> Result<Vec<(DocId, Vec<f32>)>> {
        let cf = self.cf(CF_VECTORS)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((decode_u64_key(&k)?, decode_f32_slice(&v, self.dim)?));
        }
        Ok(out)
    }

    pub fn all_doc_locs(&self) -> Result<Vec<(DocId, DocLoc)>> {
        let cf = self.cf(CF_DOC_LOC)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((decode_u64_key(&k)?, DocLoc::decode(&v)?));
        }
        Ok(out)
    }

    pub fn all_docs_bytes(&self) -> Result<Vec<(DocId, Vec<u8>)>> {
        let cf = self.cf(CF_DOCS)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((decode_u64_key(&k)?, v.to_vec()));
        }
        Ok(out)
    }

    pub fn all_attr_clusters(&self) -> Result<Vec<(Vec<u8>, RoaringBitmap)>> {
        let cf = self.cf(CF_ATTR_CLUSTERS)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((k.to_vec(), decode_roaring_bitmap(&v)?));
        }
        Ok(out)
    }

    pub fn all_attr_postings(&self) -> Result<Vec<(Vec<u8>, RoaringBitmap)>> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((k.to_vec(), decode_roaring_bitmap(&v)?));
        }
        Ok(out)
    }

    pub fn get_fts_meta_bytes(&self) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_META)?;
        self.db.get_cf(cf, FTS_META_KEY).map_err(map_db_err)
    }

    pub fn get_fts_doc_id(&self, doc_id: DocId) -> Result<Option<FtsDocId>> {
        let cf = self.cf(CF_FTS_DOC_MAP)?;
        match self.db.get_pinned_cf(cf, doc_key(doc_id)).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(decode_u32_key(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn multi_get_fts_doc_ids(&self, doc_ids: &[DocId]) -> Result<Vec<Option<FtsDocId>>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_FTS_DOC_MAP)?;
        let keys: Vec<[u8; 8]> = doc_ids.iter().map(|&id| doc_key(id)).collect();
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), false);
        let mut out = Vec::with_capacity(doc_ids.len());
        for res in results {
            match res.map_err(map_db_err)? {
                Some(slice) => out.push(Some(decode_u32_key(&slice)?)),
                None => out.push(None),
            }
        }
        Ok(out)
    }

    pub fn get_doc_id_from_fts(&self, fts_doc_id: FtsDocId) -> Result<Option<DocId>> {
        let cf = self.cf(CF_FTS_REV_DOC_MAP)?;
        match self
            .db
            .get_pinned_cf(cf, fts_doc_key(fts_doc_id))
            .map_err(map_db_err)?
        {
            Some(bytes) => Ok(Some(decode_u64_key(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn multi_get_doc_ids_from_fts(&self, fts_doc_ids: &[FtsDocId]) -> Result<Vec<Option<DocId>>> {
        if fts_doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_FTS_REV_DOC_MAP)?;
        let keys: Vec<[u8; 4]> = fts_doc_ids.iter().map(|&id| fts_doc_key(id)).collect();
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), false);
        let mut out = Vec::with_capacity(fts_doc_ids.len());
        for res in results {
            match res.map_err(map_db_err)? {
                Some(slice) => out.push(Some(decode_u64_key(&slice)?)),
                None => out.push(None),
            }
        }
        Ok(out)
    }

    pub fn get_fts_term_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_TERMS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn multi_get_fts_term_bytes(&self, keys: &[Vec<u8>]) -> Result<Vec<Option<Vec<u8>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_FTS_TERMS)?;
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), false);
        let mut out = Vec::with_capacity(keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?.map(|s| s.to_vec()));
        }
        Ok(out)
    }

    pub fn scan_fts_term_prefix_bytes(
        &self,
        prefix: &[u8],
        scan_limit: usize,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf = self.cf(CF_FTS_TERMS)?;
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(prefix, Direction::Forward));
        let mut out = Vec::new();
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if !k.starts_with(prefix) {
                break;
            }
            out.push((k.to_vec(), v.to_vec()));
            if out.len() >= scan_limit {
                break;
            }
        }
        Ok(out)
    }

    pub fn get_fts_posting_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn get_fts_position_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_POSITIONS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn open_fts_posting_cursor(&self, prefix: Vec<u8>) -> Result<RocksPostingCursor<'_>> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        let mut iter = self.db.raw_iterator_cf(cf);
        iter.seek(&prefix);
        let valid = check_posting_iter(&iter, &prefix)?;
        Ok(RocksPostingCursor {
            iter,
            prefix,
            valid,
        })
    }

    pub fn all_fts_doc_map(&self) -> Result<Vec<(DocId, FtsDocId)>> {
        let cf = self.cf(CF_FTS_DOC_MAP)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((decode_u64_key(&k)?, decode_u32_key(&v)?));
        }
        Ok(out)
    }

    pub fn all_fts_rev_doc_map(&self) -> Result<Vec<(FtsDocId, DocId)>> {
        let cf = self.cf(CF_FTS_REV_DOC_MAP)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((decode_u32_key(&k)?, decode_u64_key(&v)?));
        }
        Ok(out)
    }

    pub fn all_fts_terms(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf = self.cf(CF_FTS_TERMS)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((k.to_vec(), v.to_vec()));
        }
        Ok(out)
    }

    pub fn all_fts_postings(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((k.to_vec(), v.to_vec()));
        }
        Ok(out)
    }

    pub fn all_fts_positions(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf = self.cf(CF_FTS_POSITIONS)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((k.to_vec(), v.to_vec()));
        }
        Ok(out)
    }

    fn cf(&self, name: &str) -> Result<&ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| TreeError::CorruptedData(format!("missing column family {name}")))
    }
}

/// Typed wrapper around `rocksdb::WriteBatch` for atomic tree and attribute index mutations.
pub struct StorageBatch<'a> {
    storage: &'a Storage,
    batch: WriteBatch,
}

impl<'a> StorageBatch<'a> {
    pub fn put_meta(&mut self, meta: &TreeMeta) {
        let cf = self.storage.db.cf_handle(CF_META).unwrap();
        self.batch.put_cf(cf, META_KEY, meta.encode());
    }

    pub fn put_schema(&mut self, schema: &CollectionSchema) {
        let cf = self.storage.db.cf_handle(CF_META).unwrap();
        self.batch.put_cf(cf, SCHEMA_KEY, schema.encode());
    }

    pub fn put_internal_node(&mut self, node_id: NodeId, node: &PackedInternalNode) {
        let cf = self.storage.db.cf_handle(CF_INTERNAL_NODES).unwrap();
        self.batch
            .put_cf(cf, node_key(node_id), node.encode(self.storage.dim));
    }

    pub fn delete_internal_node(&mut self, node_id: NodeId) {
        let cf = self.storage.db.cf_handle(CF_INTERNAL_NODES).unwrap();
        self.batch.delete_cf(cf, node_key(node_id));
    }

    pub fn put_leaf_node(&mut self, leaf_id: NodeId, leaf: &PackedLeafNode) {
        let cf = self.storage.db.cf_handle(CF_LEAF_ENTRIES).unwrap();
        self.batch.put_cf(
            cf,
            node_key(leaf_id),
            leaf.encode(
                self.storage.dim,
                self.storage.padded_dim,
                self.storage.rabitq_words,
            ),
        );
    }

    pub fn delete_leaf_node(&mut self, leaf_id: NodeId) {
        let cf = self.storage.db.cf_handle(CF_LEAF_ENTRIES).unwrap();
        self.batch.delete_cf(cf, node_key(leaf_id));
    }

    pub fn put_vector(&mut self, doc_id: DocId, vec: &[f32]) {
        debug_assert_eq!(vec.len(), self.storage.dim);
        let cf = self.storage.db.cf_handle(CF_VECTORS).unwrap();
        self.batch
            .put_cf(cf, doc_key(doc_id), encode_f32_slice(vec));
    }

    pub fn delete_vector(&mut self, doc_id: DocId) {
        let cf = self.storage.db.cf_handle(CF_VECTORS).unwrap();
        self.batch.delete_cf(cf, doc_key(doc_id));
    }

    pub fn put_doc_loc(&mut self, doc_id: DocId, loc: DocLoc) {
        let cf = self.storage.db.cf_handle(CF_DOC_LOC).unwrap();
        self.batch.put_cf(cf, doc_key(doc_id), loc.encode());
    }

    pub fn delete_doc_loc(&mut self, doc_id: DocId) {
        let cf = self.storage.db.cf_handle(CF_DOC_LOC).unwrap();
        self.batch.delete_cf(cf, doc_key(doc_id));
    }

    pub fn put_doc_bytes(&mut self, doc_id: DocId, bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_DOCS).unwrap();
        self.batch.put_cf(cf, doc_key(doc_id), bytes);
    }

    pub fn delete_doc(&mut self, doc_id: DocId) {
        let cf = self.storage.db.cf_handle(CF_DOCS).unwrap();
        self.batch.delete_cf(cf, doc_key(doc_id));
    }

    pub fn put_attr_cluster(&mut self, key: &[u8], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_ATTR_CLUSTERS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_attr_cluster(&mut self, key: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_ATTR_CLUSTERS).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_attr_posting(&mut self, key: &[u8], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_ATTR_POSTINGS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_attr_posting(&mut self, key: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_ATTR_POSTINGS).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_fts_meta_bytes(&mut self, bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_META).unwrap();
        self.batch.put_cf(cf, FTS_META_KEY, bytes);
    }

    pub fn put_fts_doc_map(&mut self, doc_id: DocId, fts_doc_id: FtsDocId) {
        let cf = self.storage.db.cf_handle(CF_FTS_DOC_MAP).unwrap();
        self.batch
            .put_cf(cf, doc_key(doc_id), fts_doc_key(fts_doc_id));
    }

    pub fn delete_fts_doc_map(&mut self, doc_id: DocId) {
        let cf = self.storage.db.cf_handle(CF_FTS_DOC_MAP).unwrap();
        self.batch.delete_cf(cf, doc_key(doc_id));
    }

    pub fn put_fts_rev_doc_map(&mut self, fts_doc_id: FtsDocId, doc_id: DocId) {
        let cf = self.storage.db.cf_handle(CF_FTS_REV_DOC_MAP).unwrap();
        self.batch
            .put_cf(cf, fts_doc_key(fts_doc_id), doc_key(doc_id));
    }

    pub fn delete_fts_rev_doc_map(&mut self, fts_doc_id: FtsDocId) {
        let cf = self.storage.db.cf_handle(CF_FTS_REV_DOC_MAP).unwrap();
        self.batch.delete_cf(cf, fts_doc_key(fts_doc_id));
    }

    pub fn put_fts_term(&mut self, key: &[u8], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_TERMS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_fts_term(&mut self, key: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_TERMS).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_fts_posting(&mut self, key: &[u8], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_POSTINGS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_fts_posting(&mut self, key: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_POSTINGS).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_fts_position(&mut self, key: &[u8], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_POSITIONS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_fts_position(&mut self, key: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_POSITIONS).unwrap();
        self.batch.delete_cf(cf, key);
    }
}

/// Prefix-bounded RocksDB cursor over `cf_fts_postings` for a single `(attr_id, escaped_token)`.
pub struct RocksPostingCursor<'a> {
    iter: DBRawIteratorWithThreadMode<'a, DB>,
    prefix: Vec<u8>,
    valid: bool,
}

impl<'a> RocksPostingCursor<'a> {
    #[inline]
    pub fn current_block_id(&self) -> Option<FtsBlockId> {
        if !self.valid {
            return None;
        }
        let k = self.iter.key()?;
        let off = self.prefix.len();
        if k.len() != off + 4 {
            return None;
        }
        Some(u32::from_be_bytes(k[off..off + 4].try_into().unwrap()))
    }

    #[inline]
    pub fn current_block_bytes(&self) -> &[u8] {
        if !self.valid {
            return &[];
        }
        self.iter.value().unwrap_or(&[])
    }

    #[inline]
    pub fn next_block(&mut self) -> Result<()> {
        if !self.valid {
            return Ok(());
        }
        self.iter.next();
        self.valid = check_posting_iter(&self.iter, &self.prefix)?;
        Ok(())
    }

    #[inline]
    pub fn seek_block(&mut self, target: FtsBlockId) -> Result<()> {
        if !self.valid {
            return Ok(());
        }
        let mut seek_key = Vec::with_capacity(self.prefix.len() + 4);
        seek_key.extend_from_slice(&self.prefix);
        seek_key.extend_from_slice(&target.to_be_bytes());
        self.iter.seek(&seek_key);
        self.valid = check_posting_iter(&self.iter, &self.prefix)?;
        Ok(())
    }
}

#[inline]
fn check_posting_iter(iter: &DBRawIteratorWithThreadMode<'_, DB>, prefix: &[u8]) -> Result<bool> {
    if !iter.valid() {
        iter.status().map_err(map_db_err)?;
        return Ok(false);
    }
    let Some(k) = iter.key() else {
        return Ok(false);
    };
    Ok(k.starts_with(prefix) && k.len() == prefix.len() + 4)
}

#[inline]
pub(crate) fn fts_doc_key(id: FtsDocId) -> [u8; 4] {
    id.to_be_bytes()
}

#[inline]
pub(crate) fn node_key(id: NodeId) -> [u8; 4] {
    id.to_be_bytes()
}

#[inline]
pub(crate) fn doc_key(id: DocId) -> [u8; 8] {
    id.to_be_bytes()
}

#[inline]
fn decode_u32_key(bytes: &[u8]) -> Result<u32> {
    if bytes.len() != 4 {
        return Err(TreeError::CorruptedData(format!(
            "expected 4-byte key, got {}",
            bytes.len()
        )));
    }
    Ok(u32::from_be_bytes(bytes.try_into().unwrap()))
}

#[inline]
fn decode_u64_key(bytes: &[u8]) -> Result<u64> {
    if bytes.len() != 8 {
        return Err(TreeError::CorruptedData(format!(
            "expected 8-byte key, got {}",
            bytes.len()
        )));
    }
    Ok(u64::from_be_bytes(bytes.try_into().unwrap()))
}

fn point_lookup_cf_opts(block_cache: &Cache, disable_compression: bool) -> Options {
    let mut opts = Options::default();
    opts.set_memtable_whole_key_filtering(true);
    if disable_compression {
        opts.set_compression_type(DBCompressionType::None);
    }

    let mut bbo = BlockBasedOptions::default();
    bbo.set_block_cache(block_cache);
    bbo.set_bloom_filter(10.0, false);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_pin_l0_filter_and_index_blocks_in_cache(true);
    opts.set_block_based_table_factory(&bbo);
    opts
}

fn attr_cluster_cf_opts(block_cache: &Cache) -> Options {
    let mut opts = Options::default();
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(3));
    opts.set_memtable_whole_key_filtering(true);

    let mut bbo = BlockBasedOptions::default();
    bbo.set_block_cache(block_cache);
    bbo.set_bloom_filter(10.0, false);
    bbo.set_whole_key_filtering(true);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_pin_l0_filter_and_index_blocks_in_cache(true);
    opts.set_block_based_table_factory(&bbo);
    opts
}

fn fts_terms_cf_opts(block_cache: &Cache) -> Options {
    let mut opts = Options::default();
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(2));
    opts.set_memtable_whole_key_filtering(true);

    let mut bbo = BlockBasedOptions::default();
    bbo.set_block_cache(block_cache);
    bbo.set_bloom_filter(10.0, false);
    bbo.set_whole_key_filtering(true);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_pin_l0_filter_and_index_blocks_in_cache(true);
    opts.set_block_based_table_factory(&bbo);
    opts
}

/// Read-your-own-writes transaction overlay that commits all mutations across tree, document store,
/// attribute index, and FTS index in one atomic `WriteBatch`, updating in-memory caches only after persistence succeeds.
pub struct TxContext<'a> {
    storage: &'a Storage,
    internal_cache: &'a mut HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &'a mut HashMap<NodeId, NodeId>,
    meta: TreeMeta,
    meta_dirty: bool,
    internal_nodes: HashMap<NodeId, Option<PackedInternalNode>>,
    dirty_internal: HashSet<NodeId>,
    leaf_nodes: HashMap<NodeId, Option<PackedLeafNode>>,
    dirty_leaf: HashSet<NodeId>,
    dirty_vectors: HashMap<DocId, Option<Vec<f32>>>,
    dirty_doc_loc: HashMap<DocId, Option<DocLoc>>,
    dirty_docs: HashMap<DocId, Option<Vec<u8>>>,
    dirty_attr_clusters: HashMap<Vec<u8>, RoaringBitmap>,
    dirty_attr_postings: HashMap<Vec<u8>, RoaringBitmap>,
    fts_meta_bytes: Option<Vec<u8>>,
    fts_meta_dirty: bool,
    dirty_fts_doc_map: HashMap<DocId, Option<FtsDocId>>,
    dirty_fts_rev_doc_map: HashMap<FtsDocId, Option<DocId>>,
    dirty_fts_terms: HashMap<Vec<u8>, Option<Vec<u8>>>,
    dirty_fts_postings: HashMap<Vec<u8>, Option<Vec<u8>>>,
    dirty_fts_positions: HashMap<Vec<u8>, Option<Vec<u8>>>,
}

impl<'a> TxContext<'a> {
    pub fn new(
        storage: &'a Storage,
        internal_cache: &'a mut HashMap<NodeId, PackedInternalNode>,
        leaf_parent: &'a mut HashMap<NodeId, NodeId>,
        meta: TreeMeta,
    ) -> Self {
        Self {
            storage,
            internal_cache,
            leaf_parent,
            meta,
            meta_dirty: false,
            internal_nodes: HashMap::new(),
            dirty_internal: HashSet::new(),
            leaf_nodes: HashMap::new(),
            dirty_leaf: HashSet::new(),
            dirty_vectors: HashMap::new(),
            dirty_doc_loc: HashMap::new(),
            dirty_docs: HashMap::new(),
            dirty_attr_clusters: HashMap::new(),
            dirty_attr_postings: HashMap::new(),
            fts_meta_bytes: None,
            fts_meta_dirty: false,
            dirty_fts_doc_map: HashMap::new(),
            dirty_fts_rev_doc_map: HashMap::new(),
            dirty_fts_terms: HashMap::new(),
            dirty_fts_postings: HashMap::new(),
            dirty_fts_positions: HashMap::new(),
        }
    }

    pub fn meta(&self) -> &TreeMeta {
        &self.meta
    }

    pub fn meta_mut(&mut self) -> &mut TreeMeta {
        self.meta_dirty = true;
        &mut self.meta
    }

    pub fn alloc_node_id(&mut self) -> NodeId {
        let id = self.meta.next_node_id;
        self.meta.next_node_id = id.checked_add(1).expect("NodeId u32 overflow");
        self.meta_dirty = true;
        id
    }

    pub fn is_internal_node(&self, node_id: NodeId) -> bool {
        if let Some(opt) = self.internal_nodes.get(&node_id) {
            return opt.is_some();
        }
        self.internal_cache.contains_key(&node_id)
    }

    pub fn get_internal_node_ref(&self, node_id: NodeId) -> Result<&PackedInternalNode> {
        if let Some(opt) = self.internal_nodes.get(&node_id) {
            return opt.as_ref().ok_or_else(|| {
                TreeError::CorruptedData(format!("internal node {node_id} was deleted in tx"))
            });
        }
        self.internal_cache
            .get(&node_id)
            .ok_or_else(|| TreeError::CorruptedData(format!("missing internal node {node_id}")))
    }

    pub fn get_internal_node(&mut self, node_id: NodeId) -> Result<PackedInternalNode> {
        if let Some(opt) = self.internal_nodes.get(&node_id) {
            return opt.clone().ok_or_else(|| {
                TreeError::CorruptedData(format!("internal node {node_id} was deleted in tx"))
            });
        }
        let node = self
            .internal_cache
            .get(&node_id)
            .cloned()
            .ok_or_else(|| {
                TreeError::CorruptedData(format!("missing internal node {node_id}"))
            })?;
        self.internal_nodes.insert(node_id, Some(node.clone()));
        Ok(node)
    }

    pub fn put_internal_node(&mut self, node_id: NodeId, node: PackedInternalNode) {
        self.internal_nodes.insert(node_id, Some(node));
        self.dirty_internal.insert(node_id);
    }

    pub fn delete_internal_node(&mut self, node_id: NodeId) {
        self.internal_nodes.insert(node_id, None);
        self.dirty_internal.insert(node_id);
    }

    pub fn get_leaf_node(&mut self, leaf_id: NodeId) -> Result<PackedLeafNode> {
        if let Some(opt) = self.leaf_nodes.get(&leaf_id) {
            return opt.clone().ok_or_else(|| {
                TreeError::CorruptedData(format!("leaf {leaf_id} was deleted in tx"))
            });
        }
        let leaf = self
            .storage
            .get_leaf_node(leaf_id)?
            .ok_or_else(|| TreeError::CorruptedData(format!("missing leaf node {leaf_id}")))?;
        self.leaf_nodes.insert(leaf_id, Some(leaf.clone()));
        Ok(leaf)
    }

    pub fn put_leaf_node(&mut self, leaf_id: NodeId, leaf: PackedLeafNode) {
        self.leaf_nodes.insert(leaf_id, Some(leaf));
        self.dirty_leaf.insert(leaf_id);
    }

    pub fn delete_leaf_node(&mut self, leaf_id: NodeId) {
        self.leaf_nodes.insert(leaf_id, None);
        self.dirty_leaf.insert(leaf_id);
    }

    pub fn put_vector(&mut self, doc_id: DocId, vec: Vec<f32>) {
        debug_assert_eq!(vec.len(), self.storage.dim);
        self.dirty_vectors.insert(doc_id, Some(vec));
    }

    pub fn delete_vector(&mut self, doc_id: DocId) {
        self.dirty_vectors.insert(doc_id, None);
    }

    /// Fetch full-precision vectors for `doc_ids` into a flat `doc_ids.len() * dim` buffer
    /// aligned with `doc_ids`, honoring read-your-own-writes mutations in `dirty_vectors`.
    pub fn get_vectors_flat(&self, doc_ids: &[DocId]) -> Result<Vec<f32>> {
        let dim = self.storage.dim;
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![0.0f32; doc_ids.len() * dim];
        let mut missing: Vec<(DocId, usize)> = Vec::new();
        for (slot, &doc_id) in doc_ids.iter().enumerate() {
            match self.dirty_vectors.get(&doc_id) {
                Some(Some(vec)) => {
                    out[slot * dim..(slot + 1) * dim].copy_from_slice(vec);
                }
                Some(None) => {
                    return Err(TreeError::CorruptedData(format!(
                        "vector {doc_id} was deleted in tx"
                    )));
                }
                None => missing.push((doc_id, slot)),
            }
        }
        if !missing.is_empty() {
            missing.sort_unstable_by_key(|&(doc_id, _)| doc_id);
            let sorted_ids: Vec<DocId> = missing.iter().map(|&(doc_id, _)| doc_id).collect();
            let slices = self.storage.multi_get_vectors_pinned(&sorted_ids)?;
            let expected_bytes = dim * 4;
            for (&(doc_id, slot), opt_slice) in missing.iter().zip(slices.iter()) {
                let slice = opt_slice.as_ref().ok_or_else(|| {
                    TreeError::CorruptedData(format!("missing vector {doc_id} in cf_vectors"))
                })?;
                if slice.len() != expected_bytes {
                    return Err(TreeError::CorruptedData(format!(
                        "vector {doc_id} expected {expected_bytes} bytes, got {}",
                        slice.len()
                    )));
                }
                decode_f32_into(slice, &mut out[slot * dim..(slot + 1) * dim])?;
            }
        }
        Ok(out)
    }

    pub fn get_doc_loc(&self, doc_id: DocId) -> Result<Option<DocLoc>> {
        if let Some(opt) = self.dirty_doc_loc.get(&doc_id) {
            return Ok(*opt);
        }
        self.storage.get_doc_loc(doc_id)
    }

    pub fn put_doc_loc(&mut self, doc_id: DocId, loc: DocLoc) {
        self.dirty_doc_loc.insert(doc_id, Some(loc));
    }

    pub fn delete_doc_loc(&mut self, doc_id: DocId) {
        self.dirty_doc_loc.insert(doc_id, None);
    }

    pub fn get_doc_bytes(&self, doc_id: DocId) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_docs.get(&doc_id) {
            return Ok(opt.clone());
        }
        self.storage.get_doc_bytes(doc_id)
    }

    pub fn multi_get_doc_bytes(&self, doc_ids: &[DocId]) -> Result<Vec<Option<Vec<u8>>>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![None; doc_ids.len()];
        let mut missing: Vec<(DocId, usize)> = Vec::new();
        for (slot, &doc_id) in doc_ids.iter().enumerate() {
            if let Some(opt) = self.dirty_docs.get(&doc_id) {
                out[slot] = opt.clone();
            } else {
                missing.push((doc_id, slot));
            }
        }
        if !missing.is_empty() {
            missing.sort_unstable_by_key(|&(doc_id, _)| doc_id);
            let sorted_ids: Vec<DocId> = missing.iter().map(|&(doc_id, _)| doc_id).collect();
            let slices = self.storage.multi_get_doc_bytes_pinned(&sorted_ids)?;
            for (&(_, slot), opt_slice) in missing.iter().zip(slices.iter()) {
                out[slot] = opt_slice.as_ref().map(|s| s.to_vec());
            }
        }
        Ok(out)
    }

    pub fn put_doc_bytes(&mut self, doc_id: DocId, bytes: Vec<u8>) {
        self.dirty_docs.insert(doc_id, Some(bytes));
    }

    pub fn delete_doc(&mut self, doc_id: DocId) {
        self.dirty_docs.insert(doc_id, None);
    }

    /// Mutate the posting bitmap at `(cluster_key, cluster_id)` and automatically maintain the
    /// downsampled `cf_attr_clusters[cluster_key]` bit for `cluster_id` on `empty <-> non-empty` transitions.
    pub fn mutate_attr_posting(
        &mut self,
        cluster_key: &[u8],
        cluster_id: NodeId,
        mutate: impl FnOnce(&mut RoaringBitmap),
    ) -> Result<()> {
        let posting_key = make_posting_key(cluster_key, cluster_id);
        if !self.dirty_attr_postings.contains_key(&posting_key) {
            let existing = self
                .storage
                .get_attr_posting_bitmap(&posting_key)?
                .unwrap_or_default();
            self.dirty_attr_postings.insert(posting_key.clone(), existing);
        }
        let bm = self.dirty_attr_postings.get_mut(&posting_key).unwrap();
        let was_empty = bm.is_empty();
        mutate(bm);
        let is_empty = bm.is_empty();

        if was_empty && !is_empty {
            self.mutate_attr_cluster(cluster_key, |cbm| {
                cbm.insert(cluster_id);
            })?;
        } else if !was_empty && is_empty {
            self.mutate_attr_cluster(cluster_key, |cbm| {
                cbm.remove(cluster_id);
            })?;
        }
        Ok(())
    }

    fn mutate_attr_cluster(
        &mut self,
        cluster_key: &[u8],
        mutate: impl FnOnce(&mut RoaringBitmap),
    ) -> Result<()> {
        if !self.dirty_attr_clusters.contains_key(cluster_key) {
            let existing = self
                .storage
                .get_attr_cluster_bitmap(cluster_key)?
                .unwrap_or_default();
            self.dirty_attr_clusters
                .insert(cluster_key.to_vec(), existing);
        }
        let bm = self.dirty_attr_clusters.get_mut(cluster_key).unwrap();
        mutate(bm);
        Ok(())
    }

    pub fn get_fts_meta_bytes(&self) -> Result<Option<Vec<u8>>> {
        if let Some(ref bytes) = self.fts_meta_bytes {
            return Ok(Some(bytes.clone()));
        }
        self.storage.get_fts_meta_bytes()
    }

    pub fn put_fts_meta_bytes(&mut self, bytes: Vec<u8>) {
        self.fts_meta_bytes = Some(bytes);
        self.fts_meta_dirty = true;
    }

    pub fn get_fts_doc_id(&self, doc_id: DocId) -> Result<Option<FtsDocId>> {
        if let Some(opt) = self.dirty_fts_doc_map.get(&doc_id) {
            return Ok(*opt);
        }
        self.storage.get_fts_doc_id(doc_id)
    }

    pub fn put_fts_doc_mapping(&mut self, doc_id: DocId, fts_doc_id: FtsDocId) {
        self.dirty_fts_doc_map.insert(doc_id, Some(fts_doc_id));
        self.dirty_fts_rev_doc_map.insert(fts_doc_id, Some(doc_id));
    }

    pub fn delete_fts_doc_mapping(&mut self, doc_id: DocId, fts_doc_id: FtsDocId) {
        self.dirty_fts_doc_map.insert(doc_id, None);
        self.dirty_fts_rev_doc_map.insert(fts_doc_id, None);
    }

    pub fn get_fts_term_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_terms.get(key) {
            return Ok(opt.clone());
        }
        self.storage.get_fts_term_bytes(key)
    }

    pub fn put_fts_term_bytes(&mut self, key: Vec<u8>, val: Vec<u8>) {
        self.dirty_fts_terms.insert(key, Some(val));
    }

    pub fn delete_fts_term(&mut self, key: Vec<u8>) {
        self.dirty_fts_terms.insert(key, None);
    }

    pub fn get_fts_posting_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_postings.get(key) {
            return Ok(opt.clone());
        }
        self.storage.get_fts_posting_bytes(key)
    }

    pub fn put_fts_posting_bytes(&mut self, key: Vec<u8>, val: Vec<u8>) {
        self.dirty_fts_postings.insert(key, Some(val));
    }

    pub fn delete_fts_posting(&mut self, key: Vec<u8>) {
        self.dirty_fts_postings.insert(key, None);
    }

    pub fn get_fts_position_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_positions.get(key) {
            return Ok(opt.clone());
        }
        self.storage.get_fts_position_bytes(key)
    }

    pub fn put_fts_position_bytes(&mut self, key: Vec<u8>, val: Vec<u8>) {
        self.dirty_fts_positions.insert(key, Some(val));
    }

    pub fn delete_fts_position(&mut self, key: Vec<u8>) {
        self.dirty_fts_positions.insert(key, None);
    }

    pub fn commit(mut self) -> Result<TreeMeta> {
        let mut batch = self.storage.new_batch();
        if self.meta_dirty {
            batch.put_meta(&self.meta);
        }
        for &node_id in &self.dirty_internal {
            match self.internal_nodes.get(&node_id).and_then(|o| o.as_ref()) {
                Some(node) => batch.put_internal_node(node_id, node),
                None => batch.delete_internal_node(node_id),
            }
        }
        for &leaf_id in &self.dirty_leaf {
            match self.leaf_nodes.get(&leaf_id).and_then(|o| o.as_ref()) {
                Some(leaf) => batch.put_leaf_node(leaf_id, leaf),
                None => batch.delete_leaf_node(leaf_id),
            }
        }
        for (doc_id, opt) in self.dirty_vectors {
            match opt {
                Some(vec) => batch.put_vector(doc_id, &vec),
                None => batch.delete_vector(doc_id),
            }
        }
        for (doc_id, opt) in self.dirty_doc_loc {
            match opt {
                Some(loc) => batch.put_doc_loc(doc_id, loc),
                None => batch.delete_doc_loc(doc_id),
            }
        }
        for (doc_id, opt) in self.dirty_docs {
            match opt {
                Some(bytes) => batch.put_doc_bytes(doc_id, &bytes),
                None => batch.delete_doc(doc_id),
            }
        }
        for (key, bm) in self.dirty_attr_clusters {
            if bm.is_empty() {
                batch.delete_attr_cluster(&key);
            } else {
                let bytes = encode_roaring_bitmap(&bm)?;
                batch.put_attr_cluster(&key, &bytes);
            }
        }
        for (key, bm) in self.dirty_attr_postings {
            if bm.is_empty() {
                batch.delete_attr_posting(&key);
            } else {
                let bytes = encode_roaring_bitmap(&bm)?;
                batch.put_attr_posting(&key, &bytes);
            }
        }
        if self.fts_meta_dirty
            && let Some(ref bytes) = self.fts_meta_bytes
        {
            batch.put_fts_meta_bytes(bytes);
        }
        for (doc_id, opt) in self.dirty_fts_doc_map {
            match opt {
                Some(fid) => batch.put_fts_doc_map(doc_id, fid),
                None => batch.delete_fts_doc_map(doc_id),
            }
        }
        for (fts_doc_id, opt) in self.dirty_fts_rev_doc_map {
            match opt {
                Some(did) => batch.put_fts_rev_doc_map(fts_doc_id, did),
                None => batch.delete_fts_rev_doc_map(fts_doc_id),
            }
        }
        for (key, opt) in self.dirty_fts_terms {
            match opt {
                Some(bytes) => batch.put_fts_term(&key, &bytes),
                None => batch.delete_fts_term(&key),
            }
        }
        for (key, opt) in self.dirty_fts_postings {
            match opt {
                Some(bytes) => batch.put_fts_posting(&key, &bytes),
                None => batch.delete_fts_posting(&key),
            }
        }
        for (key, opt) in self.dirty_fts_positions {
            match opt {
                Some(bytes) => batch.put_fts_position(&key, &bytes),
                None => batch.delete_fts_position(&key),
            }
        }

        self.storage.write_batch(batch)?;

        // Update in-memory caches only after RocksDB commit succeeds.
        for node_id in self.dirty_internal {
            match self.internal_nodes.remove(&node_id).flatten() {
                Some(node) => {
                    self.internal_cache.insert(node_id, node);
                }
                None => {
                    self.internal_cache.remove(&node_id);
                }
            }
        }
        for leaf_id in self.dirty_leaf {
            match self.leaf_nodes.remove(&leaf_id).flatten() {
                Some(leaf) => {
                    if let Some(pid) = leaf.parent_id {
                        self.leaf_parent.insert(leaf_id, pid);
                    } else {
                        self.leaf_parent.remove(&leaf_id);
                    }
                }
                None => {
                    self.leaf_parent.remove(&leaf_id);
                }
            }
        }
        Ok(self.meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izzydb_core::config::DistanceMetric;
    use tempfile::TempDir;

    #[test]
    fn test_tx_get_vectors_flat_overlay_and_sorted_multiget() {
        let dir = TempDir::new().unwrap();
        let cfg = TreeConfig::new(4, DistanceMetric::L2Squared);
        let (storage, meta) = Storage::open(dir.path(), &cfg).unwrap();
        let mut cache = HashMap::new();
        let mut leaf_parent = HashMap::new();

        {
            let mut tx = TxContext::new(&storage, &mut cache, &mut leaf_parent, meta.clone());
            tx.put_vector(30, vec![3.0, 3.1, 3.2, 3.3]);
            tx.put_vector(10, vec![1.0, 1.1, 1.2, 1.3]);
            tx.commit().unwrap();
        }

        {
            let mut tx = TxContext::new(&storage, &mut cache, &mut leaf_parent, meta.clone());
            tx.put_vector(40, vec![4.0, 4.1, 4.2, 4.3]);
            tx.commit().unwrap();
        }

        let mut tx = TxContext::new(&storage, &mut cache, &mut leaf_parent, meta);
        tx.put_vector(10, vec![9.0, 9.1, 9.2, 9.3]);
        tx.put_vector(20, vec![2.0, 2.1, 2.2, 2.3]);
        tx.delete_vector(30);

        let flat = tx.get_vectors_flat(&[40, 10, 20]).unwrap();
        assert_eq!(
            flat,
            vec![
                4.0, 4.1, 4.2, 4.3, // doc 40 (from RocksDB)
                9.0, 9.1, 9.2, 9.3, // doc 10 (overwritten in tx)
                2.0, 2.1, 2.2, 2.3, // doc 20 (new in tx)
            ]
        );

        assert!(tx.get_vectors_flat(&[10, 30]).is_err());

        tx.commit().unwrap();
        assert_eq!(storage.get_vector(10).unwrap(), Some(vec![9.0, 9.1, 9.2, 9.3]));
        assert_eq!(storage.get_vector(20).unwrap(), Some(vec![2.0, 2.1, 2.2, 2.3]));
        assert_eq!(storage.get_vector(30).unwrap(), None);
        assert_eq!(storage.get_vector(40).unwrap(), Some(vec![4.0, 4.1, 4.2, 4.3]));
    }
}
