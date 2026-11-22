//! RocksDB storage engine, column family descriptors, read/write caches,
//! FTS skip/doc-len cursors, and atomic [`TxContext`] overlay.

mod batch;
mod fts_cursors;
mod tx;
mod types;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use roaring::RoaringBitmap;
use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamily, ColumnFamilyDescriptor, DB, DBCompressionType,
    DBIteratorWithThreadMode, DBPinnableSlice, DEFAULT_COLUMN_FAMILY_NAME, Direction, IngestExternalFileOptions,
    IteratorMode, Options, PrefixRange, ReadOptions, SliceTransform, WriteBatch, WriteOptions,
};

use izzydb_core::config::{
    AttrId, DocId, FtsBlockId, FtsDocId, NodeId, TermId, TreeConfig, fts_block_id, fts_slot,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::CollectionSchema;
use izzydb_fts::TermRecord;

use crate::codec::{
    LeafPosting, LeafPostingView, decode_roaring_bitmap, decode_zone_map, encode_fts_block_key,
    encode_fts_term_id_key, encode_fts_term_id_prefix, encode_posting_prefix,
};
use crate::node::{PackedInternalNode, PackedLeafNode, TreeMeta, decode_f32_slice};

pub use batch::StorageBatch;
pub use fts_cursors::{RocksDocLenCursor, RocksSkipCursor};
pub use tx::TxContext;
pub use types::{AttrZoneMap, DocLoc, FtsRevBlock, doc_key, node_key};

use types::{AttrZoneTable, DocLenBlockTable, FtsRevBlockTable, decode_u32_key, decode_u64_key, map_db_err};

pub const DEFAULT_CF_NAME: &str = DEFAULT_COLUMN_FAMILY_NAME;
pub const CF_LEAF_ENTRIES: &str = "cf_leaf_entries";
pub const CF_VECTORS: &str = "cf_vectors";
pub const CF_DOC_LOC: &str = "cf_doc_loc";
pub const CF_DOCS: &str = "cf_docs";
pub const CF_ATTR_CLUSTERS: &str = "cf_attr_clusters";
pub const CF_ATTR_POSTINGS: &str = "cf_attr_postings";
pub const CF_FTS_TERMS: &str = "cf_fts_terms";
pub const CF_FTS_SKIPS: &str = "cf_fts_skips";
pub const CF_FTS_POSTINGS: &str = "cf_fts_postings";
pub const CF_FTS_POSITIONS: &str = "cf_fts_positions";

pub const ALL_DATA_CF_NAMES: [&str; 10] = [
    CF_VECTORS,
    CF_DOCS,
    CF_DOC_LOC,
    CF_LEAF_ENTRIES,
    CF_ATTR_CLUSTERS,
    CF_ATTR_POSTINGS,
    CF_FTS_POSITIONS,
    CF_FTS_POSTINGS,
    CF_FTS_SKIPS,
    CF_FTS_TERMS,
];

// Each keyspace in the default column family starts with a distinct byte.
pub const INTERNAL_NODE_TAG: u8 = 2;
pub const ATTR_ZONE_TAG: u8 = 3;
pub const FTS_REV_BLOCK_TAG: u8 = 4;
pub const FTS_DOC_LEN_TAG: u8 = 5;
/// `default` CF keyspace tag for global per-value document counts:
/// `0x06 || cluster_key (attr_id BE || type_tag || encoded_val) -> doc_count: u64 LE`.
pub const ATTR_COUNT_TAG: u8 = 6;

pub const META_KEY: &[u8] = b"\x01tree_meta";
pub const SCHEMA_KEY: &[u8] = b"\x01schema";
pub const FTS_META_KEY: &[u8] = b"\x01fts_meta";
pub const FTS_FREE_IDS_KEY: &[u8] = b"\x01fts_free_ids";
/// Marker recording that `default[0x06]` global attribute counts are present and maintained.
pub const ATTR_COUNTS_VERSION_KEY: &[u8] = b"\x01attr_counts_v1";
/// Marker persisted during external SST bulk ingestion to detect incomplete multi-CF ingests.
pub const BULK_IMPORT_MARKER_KEY: &[u8] = b"\x01bulk_import_v1";
const BLOCK_CACHE_BYTES: usize = 128 * 1024 * 1024;
const DB_WRITE_BUFFER_BYTES: usize = 128 * 1024 * 1024;
const ATTR_POSTING_PREFIX_LEN: usize = 7;

/// Ordered byte range bounds (`(value_bytes, inclusive)`) for prefix-bounded RocksDB scans.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteRange<'a> {
    pub lo: Option<(&'a [u8], bool)>,
    pub hi: Option<(&'a [u8], bool)>,
}

/// RocksDB storage wrapper managing packed node, document, and attribute index column families.
/// Lazy iterator over `cf_attr_clusters` entries; see [`Storage::iter_attr_clusters_prefix`].
pub struct AttrClusterIter<'a> {
    inner: DBIteratorWithThreadMode<'a, DB>,
}

impl Iterator for AttrClusterIter<'_> {
    type Item = Result<(Vec<u8>, RoaringBitmap)>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        Some((|| {
            let (k, v) = item.map_err(map_db_err)?;
            if k.len() < 3 {
                return Err(TreeError::CorruptedData("cf_attr_clusters key too short".into()));
            }
            Ok((k[3..].to_vec(), decode_roaring_bitmap(&v)?))
        })())
    }
}

pub struct Storage {
    pub(super) db: DB,
    _block_cache: Cache,
    pub(super) dim: usize,
    pub(super) padded_dim: usize,
    pub(super) rabitq_words: usize,
    pub(super) fts_doc_lens_cache: RwLock<HashMap<u16, DocLenBlockTable>>,
    pub(super) fts_rev_map_cache: RwLock<Option<FtsRevBlockTable>>,
    pub(super) attr_zones_cache: RwLock<Option<AttrZoneTable>>,
}

impl Storage {
    /// Open or initialize the RocksDB instance and validate/create `TreeMeta`.
    pub fn open(path: impl AsRef<Path>, config: &TreeConfig) -> Result<(Self, TreeMeta)> {
        let block_cache = default_sst_block_cache();
        let storage = Self::open_raw(path, config, &block_cache)?;

        if storage.has_bulk_import_marker()? {
            return Err(TreeError::CorruptedData(
                "incomplete bulk import; delete the database directory and re-run".into(),
            ));
        }

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
                let init_leaf = PackedLeafNode::empty(None, config.dim, storage.padded_dim);
                let mut batch = storage.new_batch();
                batch.put_meta(&init_meta);
                batch.put_leaf_node(1, &init_leaf);
                storage.write_batch(batch)?;
                init_meta
            }
        };

        Ok((storage, meta))
    }

    fn open_raw(
        path: impl AsRef<Path>,
        config: &TreeConfig,
        block_cache: &Cache,
    ) -> Result<Self> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.set_db_write_buffer_size(DB_WRITE_BUFFER_BYTES);

        let cf_names = [
            DEFAULT_COLUMN_FAMILY_NAME,
            CF_LEAF_ENTRIES,
            CF_VECTORS,
            CF_DOC_LOC,
            CF_DOCS,
            CF_ATTR_CLUSTERS,
            CF_ATTR_POSTINGS,
            CF_FTS_TERMS,
            CF_FTS_SKIPS,
            CF_FTS_POSTINGS,
            CF_FTS_POSITIONS,
        ];
        let mut cf_descriptors = Vec::with_capacity(cf_names.len());
        for name in cf_names {
            cf_descriptors.push(ColumnFamilyDescriptor::new(
                name,
                cf_options_for(name, block_cache)?,
            ));
        }

        let db = DB::open_cf_descriptors(&db_opts, path, cf_descriptors).map_err(map_db_err)?;
        let padded_dim = config.dim.next_power_of_two().max(64);
        let rabitq_words = padded_dim / 64;
        Ok(Self {
            db,
            _block_cache: block_cache.clone(),
            dim: config.dim,
            padded_dim,
            rabitq_words,
            fts_doc_lens_cache: RwLock::new(HashMap::new()),
            fts_rev_map_cache: RwLock::new(None),
            attr_zones_cache: RwLock::new(None),
        })
    }

    /// Open a fresh or empty database path for external SST bulk ingestion.
    ///
    /// Does not write initial `TreeMeta` or root leaf; refuses if a bulk-import marker is
    /// already present or if the database contains any vectors, FTS state, or attribute state.
    pub fn open_for_bulk_ingest(path: impl AsRef<Path>, config: &TreeConfig) -> Result<Self> {
        config.validate()?;
        let block_cache = default_sst_block_cache();
        let storage = Self::open_raw(path, config, &block_cache)?;

        if storage.has_bulk_import_marker()? {
            return Err(TreeError::CorruptedData(
                "incomplete bulk import; delete the database directory and re-run".into(),
            ));
        }

        if let Some(meta) = storage.get_meta()?
            && meta.num_vectors > 0
        {
            return Err(TreeError::InvalidConfig(
                "bulk import requires an empty collection".into(),
            ));
        }

        for cf_name in [
            CF_VECTORS,
            CF_DOCS,
            CF_DOC_LOC,
            CF_ATTR_CLUSTERS,
            CF_ATTR_POSTINGS,
            CF_FTS_TERMS,
            CF_FTS_SKIPS,
            CF_FTS_POSTINGS,
            CF_FTS_POSITIONS,
        ] {
            let cf = storage.cf(cf_name)?;
            if storage.db.iterator_cf(cf, IteratorMode::Start).next().is_some() {
                return Err(TreeError::InvalidConfig(
                    "bulk import requires an empty collection".into(),
                ));
            }
        }

        for item in storage.db.iterator(IteratorMode::Start) {
            let (k, _) = item.map_err(map_db_err)?;
            let allowed = k.as_ref() == META_KEY
                || k.as_ref() == SCHEMA_KEY
                || k.as_ref() == ATTR_COUNTS_VERSION_KEY;
            if !allowed {
                return Err(TreeError::InvalidConfig(
                    "bulk import requires an empty collection".into(),
                ));
            }
        }

        let leaf_cf = storage.cf(CF_LEAF_ENTRIES)?;
        for (idx, item) in storage.db.iterator_cf(leaf_cf, IteratorMode::Start).enumerate() {
            let (k, v) = item.map_err(map_db_err)?;
            if idx > 0 || k.as_ref() != node_key(1) {
                return Err(TreeError::InvalidConfig(
                    "bulk import requires an empty collection".into(),
                ));
            }
            let leaf = PackedLeafNode::decode(
                &v,
                storage.dim,
                storage.padded_dim,
                storage.rabitq_words,
            )?;
            if leaf.live_count > 0 {
                return Err(TreeError::InvalidConfig(
                    "bulk import requires an empty collection".into(),
                ));
            }
        }

        Ok(storage)
    }

    /// Build RocksDB [`Options`] matching this storage instance's live CF configuration.
    pub fn cf_options(&self, cf_name: &str) -> Result<Options> {
        cf_options_for(cf_name, &self._block_cache)
    }

    /// Whether the incomplete-bulk-import marker is currently set in `default`.
    pub fn has_bulk_import_marker(&self) -> Result<bool> {
        Ok(self
            .db
            .get_pinned(BULK_IMPORT_MARKER_KEY)
            .map_err(map_db_err)?
            .is_some())
    }

    /// Persist the pending bulk-import marker with WAL sync, clearing any empty initial root leaf.
    pub fn put_bulk_import_marker(&self, build_id: u64) -> Result<()> {
        let mut val = [0u8; 9];
        val[0] = 1;
        val[1..9].copy_from_slice(&build_id.to_le_bytes());

        let mut batch = WriteBatch::default();
        batch.put(BULK_IMPORT_MARKER_KEY, val);
        let leaf_cf = self.cf(CF_LEAF_ENTRIES)?;
        if self
            .db
            .get_pinned_cf(leaf_cf, node_key(1))
            .map_err(map_db_err)?
            .is_some()
        {
            batch.delete_cf(leaf_cf, node_key(1));
        }
        let mut wo = WriteOptions::default();
        wo.set_sync(true);
        self.db.write_opt(batch, &wo).map_err(map_db_err)?;
        Ok(())
    }

    /// Clear the bulk-import marker with WAL sync after all column families have been ingested.
    pub fn clear_bulk_import_marker(&self) -> Result<()> {
        let mut wo = WriteOptions::default();
        wo.set_sync(true);
        self.db
            .delete_opt(BULK_IMPORT_MARKER_KEY, &wo)
            .map_err(map_db_err)?;
        Ok(())
    }

    /// Ingest external SST files into `cf_name` by moving (hard-linking) them.
    pub fn ingest_cf(&self, cf_name: &str, files: &[PathBuf]) -> Result<()> {
        self.ingest_cf_with_move(cf_name, files, true)
    }

    /// Ingest external SST files into `cf_name`, optionally moving (`true`) or copying (`false`).
    pub fn ingest_cf_with_move(
        &self,
        cf_name: &str,
        files: &[PathBuf],
        move_files: bool,
    ) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }
        let cf = self.cf(cf_name)?;
        let mut opts = IngestExternalFileOptions::default();
        opts.set_move_files(move_files);
        opts.set_allow_global_seqno(true);
        opts.set_allow_blocking_flush(true);
        opts.set_snapshot_consistency(false);
        self.db
            .ingest_external_file_cf_opts(cf, &opts, files.to_vec())
            .map_err(map_db_err)?;
        Ok(())
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
        match self.db.get(META_KEY).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(TreeMeta::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn get_schema(&self) -> Result<Option<CollectionSchema>> {
        match self.db.get(SCHEMA_KEY).map_err(map_db_err)? {
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
        let sorted = doc_ids.windows(2).all(|w| w[0] <= w[1]);
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted);
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

    pub fn get_doc_bytes_pinned(&self, doc_id: DocId) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_DOCS)?;
        self.db.get_pinned_cf(cf, doc_key(doc_id)).map_err(map_db_err)
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
        let key = tagged_key::<5>(INTERNAL_NODE_TAG, &node_key(node_id));
        match self.db.get_pinned(key).map_err(map_db_err)? {
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
                let past_end = if end_inclusive {
                    cmp == std::cmp::Ordering::Greater
                } else {
                    cmp != std::cmp::Ordering::Less
                };
                if past_end {
                    break;
                }
            }
            out.push((k.to_vec(), decode_roaring_bitmap(&v)?));
        }
        Ok(out)
    }

    /// Lazily stream `cf_attr_clusters` entries under one `(attr_id, type_tag)` 3-byte `prefix`
    /// in ascending (`reverse == false`) or descending (`reverse == true`) value order, without
    /// materializing the attribute's whole value domain.
    ///
    /// Yields `(encoded_val_bytes, leaf_bitmap)` where `encoded_val_bytes` excludes the 3-byte prefix.
    pub fn iter_attr_clusters_prefix(
        &self,
        prefix: [u8; 3],
        reverse: bool,
    ) -> Result<AttrClusterIter<'_>> {
        let cf = self.cf(CF_ATTR_CLUSTERS)?;
        let mut ro = ReadOptions::default();
        ro.set_iterate_range(PrefixRange(prefix.to_vec()));
        let mode = if reverse {
            IteratorMode::End
        } else {
            IteratorMode::Start
        };
        Ok(AttrClusterIter {
            inner: self.db.iterator_cf_opt(cf, ro, mode),
        })
    }

    /// Load or return cached per-attribute zone maps (`(attr_id, type_tag) -> BTreeMap<leaf_id, (min, max)>`).
    pub fn get_attr_zones_cached(&self) -> Result<AttrZoneTable> {
        {
            let guard = self
                .attr_zones_cache
                .read()
                .expect("attr_zones_cache poisoned");
            if let Some(ref table) = *guard {
                return Ok(Arc::clone(table));
            }
        }
        let mut write_guard = self
            .attr_zones_cache
            .write()
            .expect("attr_zones_cache poisoned");
        if let Some(ref table) = *write_guard {
            return Ok(Arc::clone(table));
        }

        let mut map: HashMap<(AttrId, u8), AttrZoneMap> = HashMap::new();
        self.visit_default(&[ATTR_ZONE_TAG], |k, v| {
            if k.len() != 3 {
                return Err(TreeError::CorruptedData(format!(
                    "attribute zone key expected 3 bytes after its tag, got {}",
                    k.len()
                )));
            }
            let attr_id = u16::from_be_bytes([k[0], k[1]]);
            let type_tag = k[2];
            map.insert((attr_id, type_tag), decode_zone_map(v)?);
            Ok(())
        })?;
        let table = Arc::new(map);
        *write_guard = Some(Arc::clone(&table));
        Ok(table)
    }

    pub fn get_attr_posting(&self, key: &[u8]) -> Result<Option<LeafPosting>> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        match self.db.get_pinned_cf(cf, key).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(LeafPosting::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Zero-copy batched lookup of raw `LeafPosting` slices from `cf_attr_postings`.
    pub fn multi_get_attr_posting_slices(
        &self,
        sorted_keys: &[Vec<u8>],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if sorted_keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let results = self.db.batched_multi_get_cf(cf, sorted_keys.iter(), true);
        let mut out = Vec::with_capacity(sorted_keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    /// Prefix-bounded range scan over cluster-first `cf_attr_postings` for `(attr_id, type_tag, leaf_id)`,
    /// ORing matching `LeafPostingView` entries directly into `dst_words` with zero heap allocations.
    pub fn scan_attr_posting_range_into_words(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        leaf_id: NodeId,
        range: ByteRange<'_>,
        dst_words: &mut [u64],
    ) -> Result<()> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let prefix = encode_posting_prefix(attr_id, type_tag, leaf_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);

        let start_key = match range.lo {
            Some((v, _)) => {
                let mut k = Vec::with_capacity(ATTR_POSTING_PREFIX_LEN + v.len());
                k.extend_from_slice(&prefix);
                k.extend_from_slice(v);
                k
            }
            None => prefix.to_vec(),
        };

        let mut iter = self.db.raw_iterator_cf_opt(cf, ro);
        iter.seek(&start_key);
        while iter.valid() {
            let Some(k) = iter.key() else { break };
            if !k.starts_with(&prefix) {
                break;
            }
            let val_bytes = &k[ATTR_POSTING_PREFIX_LEN..];
            if let Some((lo_b, lo_inc)) = range.lo
                && !lo_inc
                && val_bytes == lo_b
            {
                iter.next();
                continue;
            }
            if let Some((hi_b, hi_inc)) = range.hi {
                let cmp = val_bytes.cmp(hi_b);
                let past_hi = if hi_inc {
                    cmp == std::cmp::Ordering::Greater
                } else {
                    cmp != std::cmp::Ordering::Less
                };
                if past_hi {
                    break;
                }
            }
            if let Some(v) = iter.value() {
                LeafPostingView::parse(v)?.or_into_words(dst_words)?;
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)?;
        Ok(())
    }

    fn scan_cf<K, V>(
        &self,
        cf_name: &str,
        mut decode_key: impl FnMut(&[u8]) -> Result<K>,
        mut decode_val: impl FnMut(&[u8]) -> Result<V>,
    ) -> Result<Vec<(K, V)>> {
        let cf = self.cf(cf_name)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((decode_key(&k)?, decode_val(&v)?));
        }
        Ok(out)
    }

    fn visit_default(
        &self,
        prefix: &[u8],
        mut visit: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(prefix));
        let mut iter = self.db.raw_iterator_opt(options);
        iter.seek(prefix);
        while let Some((key, value)) = iter.item() {
            visit(&key[prefix.len()..], value)?;
            iter.next();
        }
        iter.status().map_err(map_db_err)
    }

    fn scan_default<K, V>(
        &self,
        tag: u8,
        mut decode_key: impl FnMut(&[u8]) -> Result<K>,
        mut decode_val: impl FnMut(&[u8]) -> Result<V>,
    ) -> Result<Vec<(K, V)>> {
        let mut out = Vec::new();
        self.visit_default(&[tag], |key, value| {
            out.push((decode_key(key)?, decode_val(value)?));
            Ok(())
        })?;
        Ok(out)
    }

    pub fn all_internal_nodes(&self) -> Result<Vec<(NodeId, PackedInternalNode)>> {
        self.scan_default(INTERNAL_NODE_TAG, decode_u32_key, |v| {
            PackedInternalNode::decode(v, self.dim)
        })
    }

    pub fn all_leaf_nodes(&self) -> Result<Vec<(NodeId, PackedLeafNode)>> {
        self.scan_cf(CF_LEAF_ENTRIES, decode_u32_key, |v| {
            PackedLeafNode::decode(v, self.dim, self.padded_dim, self.rabitq_words)
        })
    }

    pub fn all_vectors(&self) -> Result<Vec<(DocId, Vec<f32>)>> {
        self.scan_cf(CF_VECTORS, decode_u64_key, |v| decode_f32_slice(v, self.dim))
    }

    pub fn all_doc_locs(&self) -> Result<Vec<(DocId, DocLoc)>> {
        self.scan_cf(CF_DOC_LOC, decode_u64_key, DocLoc::decode)
    }

    /// Ids of every stored document, ascending.
    pub fn all_doc_ids(&self) -> Result<Vec<DocId>> {
        Ok(self
            .scan_cf(CF_DOC_LOC, decode_u64_key, |_| Ok(()))?
            .into_iter()
            .map(|(id, _)| id)
            .collect())
    }

    pub fn all_docs_bytes(&self) -> Result<Vec<(DocId, Vec<u8>)>> {
        self.scan_cf(CF_DOCS, decode_u64_key, |v| Ok(v.to_vec()))
    }

    pub fn all_attr_clusters(&self) -> Result<Vec<(Vec<u8>, RoaringBitmap)>> {
        self.scan_cf(CF_ATTR_CLUSTERS, |k| Ok(k.to_vec()), decode_roaring_bitmap)
    }

    pub fn all_attr_postings(&self) -> Result<Vec<(Vec<u8>, LeafPosting)>> {
        self.scan_cf(CF_ATTR_POSTINGS, |k| Ok(k.to_vec()), LeafPosting::decode)
    }

    /// Global number of documents holding the index key `cluster_key`
    /// (`attr_id BE || type_tag || encoded_val`), from `default[0x06]`. `0` if absent.
    pub fn get_attr_count(&self, cluster_key: &[u8]) -> Result<u64> {
        let mut key = Vec::with_capacity(1 + cluster_key.len());
        key.push(ATTR_COUNT_TAG);
        key.extend_from_slice(cluster_key);
        match self.db.get_pinned(key).map_err(map_db_err)? {
            Some(bytes) => decode_count(&bytes),
            None => Ok(0),
        }
    }

    /// Visit `default[0x06]` counts for `(attr_id, type_tag)` in ascending `encoded_val` order,
    /// seeking to `start_val`. `visit(encoded_val, count)` returns `false` to stop early.
    pub fn visit_attr_counts(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        start_val: &[u8],
        mut visit: impl FnMut(&[u8], u64) -> Result<bool>,
    ) -> Result<()> {
        let a = attr_id.to_be_bytes();
        let prefix = [ATTR_COUNT_TAG, a[0], a[1], type_tag];
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        let mut iter = self.db.raw_iterator_opt(options);
        let mut seek_key = Vec::with_capacity(prefix.len() + start_val.len());
        seek_key.extend_from_slice(&prefix);
        seek_key.extend_from_slice(start_val);
        iter.seek(&seek_key);
        while let Some((key, value)) = iter.item() {
            if !visit(&key[prefix.len()..], decode_count(value)?)? {
                break;
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)
    }

    /// Prefix-scan `cf_attr_postings` for `(attr_id, type_tag, leaf_id)` in ascending `encoded_val`
    /// order, seeking to `start_val`. `visit(encoded_val, raw_posting)` returns `false` to stop.
    pub fn visit_leaf_attr_postings(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        leaf_id: NodeId,
        start_val: &[u8],
        mut visit: impl FnMut(&[u8], &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let prefix = encode_posting_prefix(attr_id, type_tag, leaf_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let mut iter = self.db.raw_iterator_cf_opt(cf, ro);
        let mut seek_key = Vec::with_capacity(prefix.len() + start_val.len());
        seek_key.extend_from_slice(&prefix);
        seek_key.extend_from_slice(start_val);
        iter.seek(&seek_key);
        while iter.valid() {
            let Some(k) = iter.key() else { break };
            if !k.starts_with(&prefix) {
                break;
            }
            if let Some(v) = iter.value()
                && !visit(&k[ATTR_POSTING_PREFIX_LEN..], v)?
            {
                break;
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)
    }

    /// All `default[0x06]` counts as `(cluster_key, count)` pairs.
    pub fn all_attr_counts(&self) -> Result<Vec<(Vec<u8>, u64)>> {
        self.scan_default(ATTR_COUNT_TAG, |k| Ok(k.to_vec()), decode_count)
    }

    /// Delete every `default[0x06]` count and the global-counts marker, reproducing the on-disk
    /// state of a database created before global counts existed (the next `Collection::open`
    /// backfills them). Intended for migration tests.
    #[doc(hidden)]
    pub fn clear_attr_counts(&self) -> Result<()> {
        let mut batch = self.new_batch();
        for (key, _) in self.all_attr_counts()? {
            batch.delete_attr_count(&key);
        }
        batch.delete_attr_counts_marker();
        self.write_batch(batch)
    }

    /// Whether the `default[0x06]` global-counts marker is persisted.
    pub fn has_attr_counts_marker(&self) -> Result<bool> {
        Ok(self
            .db
            .get_pinned(ATTR_COUNTS_VERSION_KEY)
            .map_err(map_db_err)?
            .is_some())
    }

    pub fn all_attr_zones(&self) -> Result<Vec<(Vec<u8>, AttrZoneMap)>> {
        self.scan_default(ATTR_ZONE_TAG, |k| Ok(k.to_vec()), decode_zone_map)
    }

    pub fn get_fts_meta_bytes(&self) -> Result<Option<Vec<u8>>> {
        self.db.get(FTS_META_KEY).map_err(map_db_err)
    }

    pub fn get_fts_free_ids(&self) -> Result<RoaringBitmap> {
        match self.db.get_pinned(FTS_FREE_IDS_KEY).map_err(map_db_err)? {
            Some(bytes) => decode_roaring_bitmap(&bytes),
            None => Ok(RoaringBitmap::new()),
        }
    }

    pub fn get_fts_doc_id(&self, doc_id: DocId) -> Result<Option<FtsDocId>> {
        Ok(self.get_doc_loc(doc_id)?.and_then(|l| l.fts_doc_id))
    }

    pub fn multi_get_fts_doc_ids(&self, doc_ids: &[DocId]) -> Result<Vec<Option<FtsDocId>>> {
        Ok(self
            .multi_get_doc_locs(doc_ids)?
            .into_iter()
            .map(|opt| opt.and_then(|l| l.fts_doc_id))
            .collect())
    }

    fn get_fts_rev_table_cached(&self) -> Result<FtsRevBlockTable> {
        {
            let guard = self
                .fts_rev_map_cache
                .read()
                .expect("fts_rev_map_cache poisoned");
            if let Some(ref table) = *guard {
                return Ok(Arc::clone(table));
            }
        }
        let mut write_guard = self
            .fts_rev_map_cache
            .write()
            .expect("fts_rev_map_cache poisoned");
        if let Some(ref table) = *write_guard {
            return Ok(Arc::clone(table));
        }

        let mut blocks: Vec<Option<Arc<FtsRevBlock>>> = Vec::new();
        self.visit_default(&[FTS_REV_BLOCK_TAG], |k, v| {
            let block_id = decode_u32_key(k)? as usize;
            let blk = Arc::new(FtsRevBlock::decode(v)?);
            if blocks.len() <= block_id {
                blocks.resize_with(block_id + 1, || None);
            }
            blocks[block_id] = Some(blk);
            Ok(())
        })?;
        let table = Arc::new(blocks);
        *write_guard = Some(Arc::clone(&table));
        Ok(table)
    }

    pub fn get_fts_rev_block(&self, block_id: FtsBlockId) -> Result<Option<FtsRevBlock>> {
        let key = tagged_key::<5>(FTS_REV_BLOCK_TAG, &encode_fts_block_key(block_id));
        match self.db.get_pinned(key).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(FtsRevBlock::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn get_doc_id_from_fts(&self, fts_doc_id: FtsDocId) -> Result<Option<DocId>> {
        let table = self.get_fts_rev_table_cached()?;
        let bid = fts_block_id(fts_doc_id) as usize;
        let slot = fts_slot(fts_doc_id);
        Ok(table
            .get(bid)
            .and_then(|o| o.as_ref())
            .and_then(|b| b.get_doc_id(slot)))
    }

    pub fn multi_get_doc_ids_from_fts(&self, fts_doc_ids: &[FtsDocId]) -> Result<Vec<Option<DocId>>> {
        if fts_doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let table = self.get_fts_rev_table_cached()?;
        let mut out = Vec::with_capacity(fts_doc_ids.len());
        for &fid in fts_doc_ids {
            let bid = fts_block_id(fid) as usize;
            let slot = fts_slot(fid);
            out.push(
                table
                    .get(bid)
                    .and_then(|o| o.as_ref())
                    .and_then(|b| b.get_doc_id(slot)),
            );
        }
        Ok(out)
    }

    /// Resolve a `RoaringBitmap<FtsDocId>` directly into `(leaf_bm, per_leaf_local_masks)` in RAM
    /// using the cached block-packed `cf_fts_rev_doc_map`.
    pub fn resolve_fts_bitmap_to_leaf_masks(
        &self,
        fts_bm: &RoaringBitmap,
    ) -> Result<(RoaringBitmap, HashMap<NodeId, Vec<u64>>)> {
        let table = self.get_fts_rev_table_cached()?;
        let mut leaf_bm = RoaringBitmap::new();
        let mut per_leaf: HashMap<NodeId, Vec<u64>> = HashMap::new();
        let mut last_bid: Option<usize> = None;
        let mut last_blk: Option<&FtsRevBlock> = None;

        for fid in fts_bm.iter() {
            let bid = fts_block_id(fid) as usize;
            if last_bid != Some(bid) {
                last_bid = Some(bid);
                last_blk = table.get(bid).and_then(|o| o.as_deref());
            }
            if let Some(blk) = last_blk
                && let Some((_, leaf_id, local_id)) = blk.get(fts_slot(fid))
            {
                leaf_bm.insert(leaf_id);
                let words = per_leaf.entry(leaf_id).or_default();
                let w = (local_id as usize) >> 6;
                if words.len() <= w {
                    words.resize(w + 1, 0);
                }
                words[w] |= 1u64 << (local_id & 63);
            }
        }
        Ok((leaf_bm, per_leaf))
    }

    pub fn get_fts_term_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_TERMS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn multi_get_fts_term_bytes_sorted(
        &self,
        keys: &[Vec<u8>],
        mut on_item: impl FnMut(Option<&[u8]>) -> Result<()>,
    ) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let cf = self.cf(CF_FTS_TERMS)?;
        let sorted = keys.windows(2).all(|w| w[0] <= w[1]);
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted);
        for res in results {
            match res.map_err(map_db_err)? {
                Some(slice) => on_item(Some(&slice))?,
                None => on_item(None)?,
            }
        }
        Ok(())
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

    pub fn get_fts_skip_bytes(&self, key: &[u8; 8]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_SKIPS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn get_fts_posting_bytes(&self, key: &[u8; 8]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn get_fts_posting_pinned(
        &self,
        term_id: TermId,
        block_id: FtsBlockId,
    ) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        let key = encode_fts_term_id_key(term_id, block_id);
        self.db.get_pinned_cf(cf, key).map_err(map_db_err)
    }

    pub fn visit_fts_postings_for_term(
        &self,
        term_id: TermId,
        mut on_block: impl FnMut(FtsBlockId, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        let prefix = encode_fts_term_id_prefix(term_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let mut iter = self.db.raw_iterator_cf_opt(cf, ro);
        iter.seek(prefix);
        while iter.valid() {
            let Some(k) = iter.key() else { break };
            if k.len() != 8 || k[0..4] != prefix {
                break;
            }
            let bid = u32::from_be_bytes(k[4..8].try_into().unwrap());
            if let Some(v) = iter.value() {
                on_block(bid, v)?;
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)?;
        Ok(())
    }

    pub fn get_fts_position_bytes(&self, key: &[u8; 8]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_POSITIONS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn multi_get_fts_positions_pinned(
        &self,
        keys: &[[u8; 8]],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_FTS_POSITIONS)?;
        let sorted = keys.windows(2).all(|w| w[0] <= w[1]);
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted);
        let mut out = Vec::with_capacity(keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn get_fts_doc_len_bytes(&self, key: &[u8; 6]) -> Result<Option<Vec<u8>>> {
        let key = tagged_key::<7>(FTS_DOC_LEN_TAG, key);
        self.db.get(key).map_err(map_db_err)
    }

    fn build_skip_cursor<'a>(
        &'a self,
        record: TermRecord,
        chunk0_pin: Option<DBPinnableSlice<'a>>,
    ) -> Result<RocksSkipCursor<'a>> {
        if let Some((bid, bytes)) = record.inline_block {
            return RocksSkipCursor::from_inline(&self.db, record.term_id, bid, bytes);
        }
        if let Some(pin) = chunk0_pin {
            return RocksSkipCursor::from_chunk0_pin(&self.db, record.term_id, pin);
        }
        RocksSkipCursor::open_out_of_line(&self.db, record.term_id)
    }

    pub fn open_fts_skip_cursor(&self, record: TermRecord) -> Result<RocksSkipCursor<'_>> {
        if record.inline_block.is_some() {
            return self.build_skip_cursor(record, None);
        }
        let cf = self.cf(CF_FTS_SKIPS)?;
        let pin = self
            .db
            .get_pinned_cf(cf, encode_fts_term_id_key(record.term_id, 0))
            .map_err(map_db_err)?;
        self.build_skip_cursor(record, pin)
    }

    pub fn open_fts_skip_cursors(&self, records: &[TermRecord]) -> Result<Vec<RocksSkipCursor<'_>>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        if records.len() == 1 {
            return Ok(vec![self.open_fts_skip_cursor(records[0].clone())?]);
        }

        let mut chunk0_jobs: Vec<(u32, usize, [u8; 8])> = Vec::with_capacity(records.len());
        for (idx, rec) in records.iter().enumerate() {
            if rec.inline_block.is_none() {
                chunk0_jobs.push((rec.term_id, idx, encode_fts_term_id_key(rec.term_id, 0)));
            }
        }
        chunk0_jobs.sort_unstable_by_key(|&(tid, _, _)| tid);

        let mut chunk0_pins: Vec<Option<DBPinnableSlice<'_>>> = Vec::new();
        chunk0_pins.resize_with(records.len(), || None);
        if !chunk0_jobs.is_empty() {
            let cf = self.cf(CF_FTS_SKIPS)?;
            let results = self
                .db
                .batched_multi_get_cf(cf, chunk0_jobs.iter().map(|(_, _, k)| k), true);
            for ((_, orig_idx, _), res) in chunk0_jobs.into_iter().zip(results) {
                chunk0_pins[orig_idx] = res.map_err(map_db_err)?;
            }
        }

        let mut out = Vec::with_capacity(records.len());
        for (rec, pin_opt) in records.iter().zip(chunk0_pins) {
            out.push(self.build_skip_cursor(rec.clone(), pin_opt)?);
        }
        Ok(out)
    }

    pub fn open_fts_doc_len_cursor(&self, attr_id: u16) -> Result<RocksDocLenCursor> {
        {
            let guard = self
                .fts_doc_lens_cache
                .read()
                .expect("fts_doc_lens_cache poisoned");
            if let Some(blocks) = guard.get(&attr_id) {
                return Ok(RocksDocLenCursor {
                    blocks: Arc::clone(blocks),
                });
            }
        }

        let mut write_guard = self
            .fts_doc_lens_cache
            .write()
            .expect("fts_doc_lens_cache poisoned");
        if let Some(blocks) = write_guard.get(&attr_id) {
            return Ok(RocksDocLenCursor {
                blocks: Arc::clone(blocks),
            });
        }

        let prefix = tagged_key::<3>(FTS_DOC_LEN_TAG, &attr_id.to_be_bytes());
        let mut blocks: Vec<Option<Box<[u8]>>> = Vec::new();
        self.visit_default(&prefix, |key, value| {
            let block_id = decode_u32_key(key)? as usize;
            if blocks.len() <= block_id {
                blocks.resize_with(block_id + 1, || None);
            }
            blocks[block_id] = Some(value.into());
            Ok(())
        })?;
        let table = Arc::new(blocks);
        write_guard.insert(attr_id, Arc::clone(&table));
        Ok(RocksDocLenCursor { blocks: table })
    }

    pub fn all_fts_rev_blocks(&self) -> Result<Vec<(FtsBlockId, FtsRevBlock)>> {
        self.scan_default(FTS_REV_BLOCK_TAG, decode_u32_key, FtsRevBlock::decode)
    }

    pub fn all_fts_terms(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.scan_cf(CF_FTS_TERMS, |k| Ok(k.to_vec()), |v| Ok(v.to_vec()))
    }

    pub fn all_fts_skips(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.scan_cf(CF_FTS_SKIPS, |k| Ok(k.to_vec()), |v| Ok(v.to_vec()))
    }

    pub fn all_fts_postings(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.scan_cf(CF_FTS_POSTINGS, |k| Ok(k.to_vec()), |v| Ok(v.to_vec()))
    }

    pub fn all_fts_positions(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.scan_cf(CF_FTS_POSITIONS, |k| Ok(k.to_vec()), |v| Ok(v.to_vec()))
    }

    pub fn all_fts_doc_lens(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.scan_default(FTS_DOC_LEN_TAG, |k| Ok(k.to_vec()), |v| Ok(v.to_vec()))
    }

    pub(super) fn cf(&self, name: &str) -> Result<&ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| TreeError::CorruptedData(format!("missing column family {name}")))
    }
}

#[inline]
pub fn encode_count(count: u64) -> [u8; 8] {
    count.to_le_bytes()
}

fn decode_count(bytes: &[u8]) -> Result<u64> {
    let arr: [u8; 8] = bytes.try_into().map_err(|_| {
        TreeError::CorruptedData(format!("attr count expected 8 bytes, got {}", bytes.len()))
    })?;
    Ok(u64::from_le_bytes(arr))
}

#[inline]
pub fn tagged_key<const N: usize>(tag: u8, key: &[u8]) -> [u8; N] {
    let mut tagged = [tag; N];
    tagged[1..].copy_from_slice(key);
    tagged
}

#[inline]
pub fn tagged_vec(tag: u8, key: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + key.len());
    out.push(tag);
    out.extend_from_slice(key);
    out
}

/// Create a shared LRU block cache configured identically to [`Storage::open`] for offline SST builders.
pub fn default_sst_block_cache() -> Cache {
    Cache::new_lru_cache(BLOCK_CACHE_BYTES)
}

/// Build the exact RocksDB [`Options`] used by [`Storage::open`] for `cf_name`.
pub fn cf_options_for(cf_name: &str, block_cache: &Cache) -> Result<Options> {
    match cf_name {
        DEFAULT_COLUMN_FAMILY_NAME | CF_DOC_LOC | CF_DOCS => {
            Ok(point_lookup_cf_opts(block_cache, false))
        }
        CF_LEAF_ENTRIES | CF_VECTORS => Ok(point_lookup_cf_opts(block_cache, true)),
        CF_ATTR_CLUSTERS => Ok(prefix_cf_opts(block_cache, 3, false)),
        CF_ATTR_POSTINGS => Ok(prefix_cf_opts(block_cache, ATTR_POSTING_PREFIX_LEN, true)),
        CF_FTS_TERMS => Ok(prefix_cf_opts(block_cache, 2, false)),
        CF_FTS_SKIPS | CF_FTS_POSTINGS => Ok(prefix_cf_opts(block_cache, 4, true)),
        CF_FTS_POSITIONS => Ok(prefix_cf_opts(block_cache, 4, false)),
        other => Err(TreeError::InvalidConfig(format!(
            "unknown column family '{other}'"
        ))),
    }
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

fn prefix_cf_opts(block_cache: &Cache, prefix_len: usize, disable_compression: bool) -> Options {
    let mut opts = Options::default();
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(prefix_len));
    opts.set_memtable_whole_key_filtering(true);
    if disable_compression {
        opts.set_compression_type(DBCompressionType::None);
    }

    let mut bbo = BlockBasedOptions::default();
    bbo.set_block_cache(block_cache);
    bbo.set_bloom_filter(10.0, false);
    bbo.set_whole_key_filtering(true);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_pin_l0_filter_and_index_blocks_in_cache(true);
    opts.set_block_based_table_factory(&bbo);
    opts
}

#[cfg(test)]
mod tests {
    use super::*;
    use izzydb_core::config::DistanceMetric;
    use rocksdb::SstFileWriter;
    use tempfile::TempDir;

    #[test]
    fn test_sst_ingest_and_bulk_import_marker() {
        let dir = TempDir::new().unwrap();
        let sst_dir = TempDir::new().unwrap();
        let cfg = TreeConfig::new(4, DistanceMetric::L2Squared);

        let storage = Storage::open_for_bulk_ingest(dir.path(), &cfg).unwrap();
        storage.put_bulk_import_marker(42).unwrap();
        assert!(storage.has_bulk_import_marker().unwrap());

        let sst_path = sst_dir.path().join("docs.sst");
        {
            let opts = storage.cf_options(CF_DOCS).unwrap();
            let mut writer = SstFileWriter::create(&opts);
            writer.open(&sst_path).unwrap();
            writer.put(doc_key(10), b"doc-10").unwrap();
            writer.put(doc_key(20), b"doc-20").unwrap();
            writer.finish().unwrap();
        }
        storage.ingest_cf(CF_DOCS, std::slice::from_ref(&sst_path)).unwrap();
        drop(storage);

        // Leftover marker must make Storage::open fail.
        assert!(matches!(
            Storage::open(dir.path(), &cfg),
            Err(TreeError::CorruptedData(msg)) if msg.contains("incomplete bulk import")
        ));

        // Re-open raw / clear marker, then normal open succeeds and reads ingested docs.
        {
            let block_cache = default_sst_block_cache();
            let raw = Storage::open_raw(dir.path(), &cfg, &block_cache).unwrap();
            raw.clear_bulk_import_marker().unwrap();
        }
        let (reopened, _meta) = Storage::open(dir.path(), &cfg).unwrap();
        assert_eq!(reopened.get_doc_bytes(10).unwrap().as_deref(), Some(b"doc-10".as_slice()));
        assert_eq!(reopened.get_doc_bytes(20).unwrap().as_deref(), Some(b"doc-20".as_slice()));
        drop(reopened);

        // Non-empty target must be rejected by open_for_bulk_ingest.
        assert!(matches!(
            Storage::open_for_bulk_ingest(dir.path(), &cfg),
            Err(TreeError::InvalidConfig(_))
        ));
    }
}
