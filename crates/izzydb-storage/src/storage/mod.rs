pub mod batch;
pub mod fts_cursors;
pub mod segments;
pub mod tx;
pub mod types;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use roaring::RoaringBitmap;
use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamily, ColumnFamilyDescriptor, DB, DBCompressionType,
    DBPinnableSlice, DEFAULT_COLUMN_FAMILY_NAME, Direction, IngestExternalFileOptions,
    IteratorMode, Options, PrefixRange, ReadOptions, SliceTransform, SstFileWriter, WriteBatch,
    WriteOptions,
};

use izzydb_core::config::{
    AttrId, ColumnGen, DocId, NodeId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentId, TermId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{CollectionConfig, CollectionSchema};
use izzydb_fts::TermRecord;

use crate::codec::{
    ATTR_COUNT_TAG, DETACHED_COUNT_TAG, DETACHED_META_TAG, DETACHED_OVERWRITTEN_TAG, DetachedMeta,
    FTS_DOC_LEN_TAG, INTERNAL_NODE_TAG, Manifest, SEG_DELETES_TAG, SEG_META_TAG, SegPosting,
    SegmentMeta, ZONE_TAG, ZoneEntry, column_key, decode_doc_id_block, decode_fts_block_key,
    decode_roaring_bitmap, decode_row_key, detached_count_suffix, detached_key,
    detached_meta_key, detached_overwritten_key, detached_posting_prefix, doc_id_block_key,
    encode_attr_prefix, encode_fts_term_id_prefix, fts_block_key_coord, row_key,
    split_detached_posting_key, split_posting_key,
};
use crate::node::{AnnMeta, PackedInternalNode, PackedLeafNode, decode_f32_slice};

pub use batch::StorageBatch;
pub use fts_cursors::{RocksDocLenCursor, RocksSkipCursor};
pub use segments::{SegmentEntry, SegmentTable, SegmentTableSnapshot};
pub use tx::TxContext;
pub use types::{DocLenBlockTable, ZoneTable};

/// Control flow returned by [`Storage::visit_leaf_nodes_from`] callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafVisitAction {
    Continue,
    StopBefore,
    StopAfter,
}

use types::{decode_u32_key, decode_u64_key, map_db_err};

pub const DEFAULT_CF_NAME: &str = DEFAULT_COLUMN_FAMILY_NAME;
pub const CF_ID_MAP: &str = "cf_id_map";
pub const CF_ROWS: &str = "cf_rows";
pub const CF_ROW_DOC_IDS: &str = "cf_row_doc_ids";
pub const CF_VECTORS: &str = "cf_vectors";
pub const CF_LEAF_ENTRIES: &str = "cf_leaf_entries";
pub const CF_ATTR_POSTINGS: &str = "cf_attr_postings";
pub const CF_COLUMNS: &str = "cf_columns";
pub const CF_DETACHED: &str = "cf_detached";
pub const CF_DETACHED_POSTINGS: &str = "cf_detached_postings";
pub const CF_FTS_TERMS: &str = "cf_fts_terms";
pub const CF_FTS_SKIPS: &str = "cf_fts_skips";
pub const CF_FTS_POSTINGS: &str = "cf_fts_postings";
pub const CF_FTS_POSITIONS: &str = "cf_fts_positions";

pub const ALL_DATA_CF_NAMES: [&str; 13] = [
    CF_ID_MAP,
    CF_ROWS,
    CF_ROW_DOC_IDS,
    CF_VECTORS,
    CF_LEAF_ENTRIES,
    CF_ATTR_POSTINGS,
    CF_COLUMNS,
    CF_DETACHED,
    CF_DETACHED_POSTINGS,
    CF_FTS_TERMS,
    CF_FTS_SKIPS,
    CF_FTS_POSTINGS,
    CF_FTS_POSITIONS,
];

pub const SCHEMA_KEY: &[u8] = b"\x01schema";
pub const MANIFEST_KEY: &[u8] = b"\x01manifest";
pub const ANN_META_KEY: &[u8] = b"\x01ann_meta";
pub const FTS_META_KEY: &[u8] = b"\x01fts_meta";
pub const ATTR_COUNTS_VERSION_KEY: &[u8] = b"\x01attr_counts_v1";
pub const DROPPED_SEGMENTS_KEY: &[u8] = b"\x01dropped_segments";
pub const BULK_IMPORT_MARKER_KEY: &[u8] = b"\x01bulk_import_v3";

const BLOCK_CACHE_BYTES: usize = 128 * 1024 * 1024;
const DB_WRITE_BUFFER_BYTES: usize = 128 * 1024 * 1024;
const ATTR_POSTING_PREFIX_LEN: usize = 3;
const DETACHED_POSTING_PREFIX_LEN: usize = 7;
/// Default bounded `cf_row_doc_ids` cache: 16,384 blocks × 2,048 B = 32 MiB.
const DOC_ID_BLOCK_CACHE_CAP: usize = 16_384;

/// Ordered byte range bounds (`(value_bytes, inclusive)`) for prefix-bounded RocksDB scans.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteRange<'a> {
    pub lo: Option<(&'a [u8], bool)>,
    pub hi: Option<(&'a [u8], bool)>,
}

pub(super) struct DocIdBlockCache {
    cap: usize,
    map: HashMap<(SegmentId, u32), Arc<[DocId; ROW_BLOCK_SIZE]>>,
    order: VecDeque<(SegmentId, u32)>,
}

impl DocIdBlockCache {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&self, key: &(SegmentId, u32)) -> Option<Arc<[DocId; ROW_BLOCK_SIZE]>> {
        self.map.get(key).map(Arc::clone)
    }

    fn insert(&mut self, key: (SegmentId, u32), val: Arc<[DocId; ROW_BLOCK_SIZE]>) {
        if self.map.contains_key(&key) {
            self.map.insert(key, val);
            return;
        }
        if self.map.len() >= self.cap
            && let Some(old) = self.order.pop_front()
        {
            self.map.remove(&old);
        }
        self.order.push_back(key);
        self.map.insert(key, val);
    }

    fn remove_segment(&mut self, seg: SegmentId) {
        self.map.retain(|&(s, _), _| s != seg);
        self.order.retain(|&(s, _)| s != seg);
    }
}

pub struct Storage {
    pub(super) db: DB,
    _block_cache: Cache,
    pub(super) dim: usize,
    pub(super) padded_dim: usize,
    pub(super) rabitq_words: usize,
    pub(super) segments: SegmentTable,
    pub(super) dropped_segments: Arc<RwLock<Arc<RoaringBitmap>>>,
    pub(super) doc_id_block_cache: RwLock<DocIdBlockCache>,
    pub(super) zone_cache: RwLock<Option<ZoneTable>>,
    pub(super) fts_doc_lens_cache: RwLock<HashMap<AttrId, DocLenBlockTable>>,
}

impl Storage {
    /// Open or initialize the v3 RocksDB instance, loading `Manifest`, optional `AnnMeta`,
    /// `SegmentTable`, and `dropped_segments`.
    pub fn open(
        path: impl AsRef<Path>,
        config: &CollectionConfig,
    ) -> Result<(Self, Manifest, Option<AnnMeta>)> {
        config.validate()?;
        let block_cache = default_sst_block_cache();
        let dim = config.vector.as_ref().map_or(0, |v| v.dim);
        let storage = Self::open_raw(path, dim, &block_cache)?;

        if storage.has_bulk_import_marker()? {
            return Err(TreeError::CorruptedData(
                "incomplete bulk import; delete the database directory and re-run".into(),
            ));
        }

        let manifest = match storage.get_manifest()? {
            Some(existing) => existing,
            None => {
                let init = Manifest::new(config.segment.seal_rows);
                let mut batch = storage.new_batch();
                batch.put_manifest(&init);
                storage.write_batch(batch)?;
                init
            }
        };

        let ann_meta = match (&config.vector, storage.get_ann_meta()?) {
            (Some(vcfg), Some(existing)) => {
                if existing.dim as usize != vcfg.dim {
                    return Err(TreeError::ConfigMismatch {
                        field: "dim",
                        expected: existing.dim.to_string(),
                        actual: vcfg.dim.to_string(),
                    });
                }
                if existing.metric != vcfg.metric {
                    return Err(TreeError::ConfigMismatch {
                        field: "metric",
                        expected: format!("{:?}", existing.metric),
                        actual: format!("{:?}", vcfg.metric),
                    });
                }
                if existing.rabitq_bits != 1 {
                    return Err(TreeError::CorruptedData(format!(
                        "unsupported rabitq_bits: {}",
                        existing.rabitq_bits
                    )));
                }
                Some(existing)
            }
            (Some(vcfg), None) => {
                if manifest.total_rows > 0 || storage.get_schema()?.is_some_and(|s| !s.has_vectors()) {
                    return Err(TreeError::ConfigMismatch {
                        field: "has_vectors",
                        expected: "false".into(),
                        actual: "true".into(),
                    });
                }
                let init_meta = AnnMeta {
                    root_id: 1,
                    root_level: 0,
                    next_node_id: 2,
                    dim: vcfg.dim as u32,
                    metric: vcfg.metric,
                    num_vectors: 0,
                    rabitq_seed: rand::random::<u64>(),
                    rabitq_bits: 1,
                    stale_rows: 0,
                    vacuum_epoch: 0,
                    vacuum_cursor: 0,
                };
                let init_leaf = PackedLeafNode::empty(None, vcfg.dim, storage.padded_dim);
                let mut batch = storage.new_batch();
                batch.put_ann_meta(&init_meta);
                batch.put_leaf_node(1, &init_leaf);
                storage.write_batch(batch)?;
                Some(init_meta)
            }
            (None, Some(_)) => {
                return Err(TreeError::ConfigMismatch {
                    field: "has_vectors",
                    expected: "true".into(),
                    actual: "false".into(),
                });
            }
            (None, None) => None,
        };

        storage.load_segment_state()?;
        storage.recover_pending_detached()?;
        Ok((storage, manifest, ann_meta))
    }

    fn open_raw(path: impl AsRef<Path>, dim: usize, block_cache: &Cache) -> Result<Self> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.set_db_write_buffer_size(DB_WRITE_BUFFER_BYTES);

        let mut cf_names = Vec::with_capacity(1 + ALL_DATA_CF_NAMES.len());
        cf_names.push(DEFAULT_COLUMN_FAMILY_NAME);
        cf_names.extend_from_slice(&ALL_DATA_CF_NAMES);

        let mut cf_descriptors = Vec::with_capacity(cf_names.len());
        for name in cf_names {
            cf_descriptors.push(ColumnFamilyDescriptor::new(
                name,
                cf_options_for(name, block_cache)?,
            ));
        }

        let db = DB::open_cf_descriptors(&db_opts, path, cf_descriptors).map_err(map_db_err)?;
        let padded_dim = if dim == 0 {
            0
        } else {
            dim.next_power_of_two().max(64)
        };
        let rabitq_words = padded_dim / 64;
        Ok(Self {
            db,
            _block_cache: block_cache.clone(),
            dim,
            padded_dim,
            rabitq_words,
            segments: SegmentTable::new(SegmentTableSnapshot::default()),
            dropped_segments: Arc::new(RwLock::new(Arc::new(RoaringBitmap::new()))),
            doc_id_block_cache: RwLock::new(DocIdBlockCache::new(DOC_ID_BLOCK_CACHE_CAP)),
            zone_cache: RwLock::new(None),
            fts_doc_lens_cache: RwLock::new(HashMap::new()),
        })
    }

    fn load_segment_state(&self) -> Result<()> {
        let mut entries: BTreeMap<SegmentId, SegmentEntry> = BTreeMap::new();
        self.visit_default(&[SEG_META_TAG], |k, v| {
            let seg = decode_u32_key(k)?;
            let meta = SegmentMeta::decode(v)?;
            entries.insert(seg, SegmentEntry { meta, deleted: None });
            Ok(())
        })?;
        self.visit_default(&[SEG_DELETES_TAG], |k, v| {
            let seg = decode_u32_key(k)?;
            let bm = decode_roaring_bitmap(v)?;
            if let Some(entry) = entries.get_mut(&seg) {
                entry.deleted = (!bm.is_empty()).then(|| Arc::new(bm));
            }
            Ok(())
        })?;
        self.segments
            .publish(SegmentTableSnapshot::from_entries(entries));

        let dropped = match self.db.get_pinned(DROPPED_SEGMENTS_KEY).map_err(map_db_err)? {
            Some(bytes) => decode_roaring_bitmap(&bytes)?,
            None => RoaringBitmap::new(),
        };
        *self
            .dropped_segments
            .write()
            .expect("dropped_segments poisoned") = Arc::new(dropped);
        Ok(())
    }

    /// Open a fresh or empty database path for external SST bulk ingestion.
    pub fn open_for_bulk_ingest(path: impl AsRef<Path>, config: &CollectionConfig) -> Result<Self> {
        config.validate()?;
        let block_cache = default_sst_block_cache();
        let dim = config.vector.as_ref().map_or(0, |v| v.dim);
        let storage = Self::open_raw(path, dim, &block_cache)?;

        if storage.has_bulk_import_marker()? {
            return Err(TreeError::CorruptedData(
                "incomplete bulk import; delete the database directory and re-run".into(),
            ));
        }

        if let Some(m) = storage.get_manifest()?
            && m.total_rows > 0
        {
            return Err(TreeError::InvalidConfig(
                "bulk import requires an empty collection".into(),
            ));
        }

        for cf_name in ALL_DATA_CF_NAMES {
            if cf_name == CF_LEAF_ENTRIES {
                continue;
            }
            let cf = storage.cf(cf_name)?;
            if storage.db.iterator_cf(cf, IteratorMode::Start).next().is_some() {
                return Err(TreeError::InvalidConfig(
                    "bulk import requires an empty collection".into(),
                ));
            }
        }

        for item in storage.db.iterator(IteratorMode::Start) {
            let (k, _) = item.map_err(map_db_err)?;
            let allowed = k.as_ref() == MANIFEST_KEY
                || k.as_ref() == ANN_META_KEY
                || k.as_ref() == SCHEMA_KEY
                || k.as_ref() == ATTR_COUNTS_VERSION_KEY;
            if !allowed {
                return Err(TreeError::InvalidConfig(
                    "bulk import requires an empty collection".into(),
                ));
            }
        }

        Ok(storage)
    }

    pub fn cf_options(&self, cf_name: &str) -> Result<Options> {
        cf_options_for(cf_name, &self._block_cache)
    }

    pub fn has_bulk_import_marker(&self) -> Result<bool> {
        Ok(self
            .db
            .get_pinned(BULK_IMPORT_MARKER_KEY)
            .map_err(map_db_err)?
            .is_some())
    }

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

    pub fn clear_bulk_import_marker(&self) -> Result<()> {
        let mut wo = WriteOptions::default();
        wo.set_sync(true);
        self.db
            .delete_opt(BULK_IMPORT_MARKER_KEY, &wo)
            .map_err(map_db_err)?;
        Ok(())
    }

    pub fn ingest_cf(&self, cf_name: &str, files: &[PathBuf]) -> Result<()> {
        self.ingest_cf_with_move(cf_name, files, true)
    }

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

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    #[inline]
    pub fn rabitq_words(&self) -> usize {
        self.rabitq_words
    }

    #[inline]
    pub fn segments(&self) -> &SegmentTable {
        &self.segments
    }

    #[inline]
    pub fn segment_snapshot(&self) -> Arc<SegmentTableSnapshot> {
        self.segments.snapshot()
    }

    #[inline]
    pub fn dropped_segments_snapshot(&self) -> Arc<RoaringBitmap> {
        Arc::clone(
            &self
                .dropped_segments
                .read()
                .expect("dropped_segments poisoned"),
        )
    }

    #[inline]
    pub fn latest_sequence_number(&self) -> u64 {
        self.db.latest_sequence_number()
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

    pub fn write_batch_sync(&self, batch: StorageBatch<'_>) -> Result<()> {
        let mut opts = WriteOptions::default();
        opts.set_sync(true);
        self.db.write_opt(batch.batch, &opts).map_err(map_db_err)?;
        Ok(())
    }

    /// Discard any uncommitted `pending_gen != 0` detached builds left behind by a crash before publish.
    fn recover_pending_detached(&self) -> Result<()> {
        let metas = self.all_detached_metas()?;
        let mut to_fix = Vec::new();
        for (attr_id, mut meta) in metas {
            if meta.pending_gen != 0 {
                meta.retired_below = meta.retired_below.max(meta.pending_gen.saturating_add(1));
                meta.pending_gen = 0;
                meta.built_against_watermark = RowId(0);
                to_fix.push((attr_id, meta));
            }
        }
        if !to_fix.is_empty() {
            let mut batch = self.new_batch();
            for (attr_id, meta) in to_fix {
                batch.put_detached_meta(attr_id, &meta);
            }
            self.write_batch_sync(batch)?;
        }
        Ok(())
    }

    pub fn get_detached_meta(&self, attr_id: AttrId) -> Result<Option<DetachedMeta>> {
        match self.db.get_pinned(detached_meta_key(attr_id)).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(DetachedMeta::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn all_detached_metas(&self) -> Result<Vec<(AttrId, DetachedMeta)>> {
        self.scan_default(
            DETACHED_META_TAG,
            |k| {
                let arr: [u8; 2] = k.try_into().map_err(|_| {
                    TreeError::CorruptedData("DetachedMeta key expected 2 bytes".into())
                })?;
                Ok(AttrId::from_be_bytes(arr))
            },
            DetachedMeta::decode,
        )
    }

    pub fn put_detached_meta_sync(&self, attr_id: AttrId, meta: &DetachedMeta) -> Result<()> {
        let mut batch = self.new_batch();
        batch.put_detached_meta(attr_id, meta);
        self.write_batch_sync(batch)
    }

    pub fn get_detached_block_pinned(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        seg: SegmentId,
        block: u32,
    ) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_DETACHED)?;
        self.db
            .get_pinned_cf(cf, detached_key(attr_id, col_gen, seg, block))
            .map_err(map_db_err)
    }

    pub fn multi_get_detached_blocks_pinned(
        &self,
        keys: &[(AttrId, ColumnGen, SegmentId, u32)],
        sorted_input: bool,
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_DETACHED)?;
        let raw_keys: Vec<[u8; 14]> = keys
            .iter()
            .map(|&(attr, g, seg, block)| detached_key(attr, g, seg, block))
            .collect();
        let results = self.db.batched_multi_get_cf(cf, raw_keys.iter(), sorted_input);
        let mut out = Vec::with_capacity(keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn get_detached_overwritten(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        seg: SegmentId,
    ) -> Result<Option<RoaringBitmap>> {
        match self
            .db
            .get_pinned(detached_overwritten_key(attr_id, col_gen, seg))
            .map_err(map_db_err)?
        {
            Some(bytes) => Ok(Some(decode_roaring_bitmap(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn all_detached_overwritten(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
    ) -> Result<Vec<(SegmentId, RoaringBitmap)>> {
        let a = attr_id.to_be_bytes();
        let g = col_gen.to_be_bytes();
        let prefix = [
            DETACHED_OVERWRITTEN_TAG,
            a[0],
            a[1],
            g[0],
            g[1],
            g[2],
            g[3],
        ];
        let mut out = Vec::new();
        self.visit_default(&prefix, |k, v| {
            let arr: [u8; 4] = k.try_into().map_err(|_| {
                TreeError::CorruptedData("detached overwritten seg suffix expected 4 bytes".into())
            })?;
            out.push((SegmentId::from_be_bytes(arr), decode_roaring_bitmap(v)?));
            Ok(())
        })?;
        Ok(out)
    }

    pub fn get_detached_count(&self, col_gen: ColumnGen, attr_key: &[u8]) -> Result<u64> {
        let suffix = detached_count_suffix(col_gen, attr_key);
        let mut key = Vec::with_capacity(1 + suffix.len());
        key.push(DETACHED_COUNT_TAG);
        key.extend_from_slice(&suffix);
        match self.db.get_pinned(key).map_err(map_db_err)? {
            Some(bytes) => decode_count(&bytes),
            None => Ok(0),
        }
    }

    pub fn visit_detached_counts(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
        start_val: &[u8],
        mut visit: impl FnMut(&[u8], u64) -> Result<bool>,
    ) -> Result<()> {
        let p = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut prefix = [0u8; 8];
        prefix[0] = DETACHED_COUNT_TAG;
        prefix[1..8].copy_from_slice(&p);
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        options.set_total_order_seek(true);
        let mut iter = self.db.raw_iterator_opt(options);
        let mut seek_key = Vec::with_capacity(prefix.len() + start_val.len());
        seek_key.extend_from_slice(&prefix);
        seek_key.extend_from_slice(start_val);
        iter.seek(&seek_key);
        while let Some((key, value)) = iter.item() {
            if !key.starts_with(&prefix) {
                break;
            }
            if !visit(&key[prefix.len()..], decode_count(value)?)? {
                break;
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)
    }

    pub fn max_detached_count_key(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
    ) -> Result<Option<Vec<u8>>> {
        let p = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut prefix = [0u8; 8];
        prefix[0] = DETACHED_COUNT_TAG;
        prefix[1..8].copy_from_slice(&p);
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        options.set_total_order_seek(true);
        let mut iter = self.db.raw_iterator_opt(options);
        iter.seek_to_last();
        while let Some((key, value)) = iter.item() {
            if !key.starts_with(&prefix) {
                break;
            }
            if decode_count(value)? > 0 {
                return Ok(Some(key[prefix.len()..].to_vec()));
            }
            iter.prev();
        }
        iter.status().map_err(map_db_err)?;
        Ok(None)
    }

    pub fn multi_get_detached_posting_slices(
        &self,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_DETACHED_POSTINGS)?;
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), true);
        let mut out = Vec::with_capacity(keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn visit_detached_postings_for_value(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
        encoded_val: &[u8],
        mut visit: impl FnMut(SegmentId, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_DETACHED_POSTINGS)?;
        let p = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut key_prefix = Vec::with_capacity(7 + encoded_val.len());
        key_prefix.extend_from_slice(&p);
        key_prefix.extend_from_slice(encoded_val);
        let mut ro = ReadOptions::default();
        ro.set_iterate_range(PrefixRange(&key_prefix[..]));
        ro.set_total_order_seek(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&key_prefix, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            let (prefix, seg) = split_detached_posting_key(&k)?;
            if prefix != key_prefix.as_slice() {
                break;
            }
            visit(seg, &v)?;
        }
        Ok(())
    }

    pub fn visit_detached_postings_range(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
        start_val: &[u8],
        mut visit: impl FnMut(&[u8], SegmentId, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.cf(CF_DETACHED_POSTINGS)?;
        let prefix = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut seek_key = Vec::with_capacity(7 + start_val.len());
        seek_key.extend_from_slice(&prefix);
        seek_key.extend_from_slice(start_val);
        let mut ro = ReadOptions::default();
        ro.set_iterate_range(PrefixRange(&prefix[..]));
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&seek_key, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            let (val_prefix, seg) = split_detached_posting_key(&k)?;
            if !visit(&val_prefix[7..], seg, &v)? {
                break;
            }
        }
        Ok(())
    }

    pub fn visit_detached_postings_prefix(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
        reverse: bool,
        mut visit: impl FnMut(&[u8], SegmentId, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.cf(CF_DETACHED_POSTINGS)?;
        let prefix = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        options.set_total_order_seek(true);
        let mut iter = self.db.raw_iterator_cf_opt(cf, options);
        if reverse {
            iter.seek_to_last();
        } else {
            iter.seek(&prefix);
        }
        while let Some((k, v)) = iter.item() {
            if !k.starts_with(&prefix) {
                break;
            }
            let (val_prefix, seg) = split_detached_posting_key(k)?;
            if !visit(&val_prefix[7..], seg, v)? {
                break;
            }
            if reverse {
                iter.prev();
            } else {
                iter.next();
            }
        }
        iter.status().map_err(map_db_err)
    }

    pub fn visit_detached_blocks_for_gen(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        mut visit: impl FnMut(SegmentId, u32, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_DETACHED)?;
        let a = attr_id.to_be_bytes();
        let g = col_gen.to_be_bytes();
        let prefix = [a[0], a[1], g[0], g[1], g[2], g[3]];
        let mut ro = ReadOptions::default();
        ro.set_iterate_range(PrefixRange(&prefix[..]));
        ro.set_total_order_seek(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&prefix, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if !k.starts_with(&prefix) || k.len() != 14 {
                continue;
            }
            let seg = u32::from_be_bytes(k[6..10].try_into().unwrap());
            let block = u32::from_be_bytes(k[10..14].try_into().unwrap());
            visit(seg, block, &v)?;
        }
        Ok(())
    }

    pub fn ingest_detached_sst(&self, sorted_entries: &[([u8; 14], Vec<u8>)]) -> Result<()> {
        if sorted_entries.is_empty() {
            return Ok(());
        }
        let path = self.db.path().join(format!(
            ".detached_stage_{}_{}.sst",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let opts = self.cf_options(CF_DETACHED)?;
        let mut writer = SstFileWriter::create(&opts);
        let res = (|| -> Result<()> {
            writer.open(&path).map_err(map_db_err)?;
            for (k, v) in sorted_entries {
                writer.put(k, v).map_err(map_db_err)?;
            }
            writer.finish().map_err(map_db_err)?;
            self.ingest_cf(CF_DETACHED, std::slice::from_ref(&path))?;
            Ok(())
        })();
        let _ = std::fs::remove_file(&path);
        res
    }

    pub fn write_detached_stage(
        &self,
        counts: &[(Vec<u8>, u64)],
        postings: &[(Vec<u8>, Vec<u8>)],
        blocks: Option<&[([u8; 14], Vec<u8>)]>,
    ) -> Result<()> {
        let mut batch = self.new_batch();
        if let Some(blks) = blocks {
            for (k, v) in blks {
                batch.put_detached_block(k, v);
            }
        }
        for (suffix, cnt) in counts {
            if *cnt > 0 {
                batch.put_detached_count(suffix, *cnt);
            }
        }
        for (k, v) in postings {
            batch.put_detached_posting(k, v);
        }
        self.write_batch(batch)
    }

    /// Take a point-in-time RocksDB snapshot and read `(Manifest, SegmentTableSnapshot, cf_id_map)`
    /// at that exact snapshot (used by `DetachedIngest::build`).
    pub fn snapshot_id_map_and_state(
        &self,
    ) -> Result<(Manifest, SegmentTableSnapshot, Vec<(DocId, RowId)>)> {
        let snap = self.db.snapshot();
        let mut ro_def = ReadOptions::default();
        ro_def.set_snapshot(&snap);

        let manifest_bytes = self
            .db
            .get_opt(MANIFEST_KEY, &ro_def)
            .map_err(map_db_err)?
            .ok_or_else(|| TreeError::CorruptedData("missing manifest in snapshot".into()))?;
        let manifest = Manifest::decode(&manifest_bytes)?;

        let mut entries: BTreeMap<SegmentId, SegmentEntry> = BTreeMap::new();
        {
            let mut ro = ReadOptions::default();
            ro.set_snapshot(&snap);
            ro.set_iterate_range(PrefixRange(&[SEG_META_TAG][..]));
            let mut it = self.db.raw_iterator_opt(ro);
            it.seek(&[SEG_META_TAG]);
            while let Some((k, v)) = it.item() {
                let seg = decode_u32_key(&k[1..])?;
                entries.insert(
                    seg,
                    SegmentEntry {
                        meta: SegmentMeta::decode(v)?,
                        deleted: None,
                    },
                );
                it.next();
            }
            it.status().map_err(map_db_err)?;
        }
        {
            let mut ro = ReadOptions::default();
            ro.set_snapshot(&snap);
            ro.set_iterate_range(PrefixRange(&[SEG_DELETES_TAG][..]));
            let mut it = self.db.raw_iterator_opt(ro);
            it.seek(&[SEG_DELETES_TAG]);
            while let Some((k, v)) = it.item() {
                let seg = decode_u32_key(&k[1..])?;
                let bm = decode_roaring_bitmap(v)?;
                if let Some(entry) = entries.get_mut(&seg) {
                    entry.deleted = (!bm.is_empty()).then(|| Arc::new(bm));
                }
                it.next();
            }
            it.status().map_err(map_db_err)?;
        }
        let seg_snap = SegmentTableSnapshot::from_entries(entries);

        let cf_id_map = self.cf(CF_ID_MAP)?;
        let mut ro_id = ReadOptions::default();
        ro_id.set_snapshot(&snap);
        let mut id_map = Vec::with_capacity(manifest.live_rows as usize);
        for item in self.db.iterator_cf_opt(cf_id_map, ro_id, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            let doc_id = decode_u64_key(&k)?;
            let r_arr: [u8; 8] = v.as_ref().try_into().map_err(|_| {
                TreeError::CorruptedData("cf_id_map value expected 8 bytes".into())
            })?;
            id_map.push((doc_id, RowId::from_be_bytes(r_arr)));
        }
        Ok((manifest, seg_snap, id_map))
    }

    pub fn get_manifest(&self) -> Result<Option<Manifest>> {
        match self.db.get_pinned(MANIFEST_KEY).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(Manifest::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn get_ann_meta(&self) -> Result<Option<AnnMeta>> {
        match self.db.get_pinned(ANN_META_KEY).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(AnnMeta::decode(&bytes)?)),
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

    pub fn get_id_map(&self, doc_id: DocId) -> Result<Option<RowId>> {
        let cf = self.cf(CF_ID_MAP)?;
        match self
            .db
            .get_pinned_cf(cf, doc_key(doc_id))
            .map_err(map_db_err)?
        {
            Some(bytes) => {
                let arr: [u8; 8] = bytes.as_ref().try_into().map_err(|_| {
                    TreeError::CorruptedData(format!(
                        "cf_id_map expected 8 bytes, got {}",
                        bytes.len()
                    ))
                })?;
                Ok(Some(RowId::from_be_bytes(arr)))
            }
            None => Ok(None),
        }
    }

    pub fn multi_get_id_map(&self, doc_ids: &[DocId]) -> Result<Vec<Option<RowId>>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_ID_MAP)?;
        let keys: Vec<[u8; 8]> = doc_ids.iter().map(|&id| doc_key(id)).collect();
        let sorted = doc_ids.windows(2).all(|w| w[0] <= w[1]);
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted);
        let mut out = Vec::with_capacity(doc_ids.len());
        for res in results {
            match res.map_err(map_db_err)? {
                Some(slice) => {
                    let arr: [u8; 8] = slice.as_ref().try_into().map_err(|_| {
                        TreeError::CorruptedData(format!(
                            "cf_id_map expected 8 bytes, got {}",
                            slice.len()
                        ))
                    })?;
                    out.push(Some(RowId::from_be_bytes(arr)));
                }
                None => out.push(None),
            }
        }
        Ok(out)
    }

    pub fn get_row_bytes(&self, row: RowId) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_ROWS)?;
        self.db.get_cf(cf, row_key(row)).map_err(map_db_err)
    }

    pub fn get_row_bytes_pinned(&self, row: RowId) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_ROWS)?;
        self.db.get_pinned_cf(cf, row_key(row)).map_err(map_db_err)
    }

    pub fn multi_get_row_bytes_pinned(
        &self,
        sorted_rows: &[RowId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        self.batched_multi_get_row_pinned(CF_ROWS, sorted_rows, true)
    }

    pub fn get_column_block_pinned(
        &self,
        seg: SegmentId,
        attr_id: AttrId,
        block: u32,
    ) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_COLUMNS)?;
        self.db
            .get_pinned_cf(cf, column_key(seg, attr_id, block))
            .map_err(map_db_err)
    }

    /// Batch pinned `multi_get` against `cf_columns` for `(seg, attr_id, block)` keys.
    /// When `sorted_input` is true, `keys` must be in ascending `(seg, attr_id, block)` order.
    pub fn multi_get_column_blocks_pinned(
        &self,
        keys: &[(SegmentId, AttrId, u32)],
        sorted_input: bool,
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_COLUMNS)?;
        let raw_keys: Vec<[u8; 10]> = keys
            .iter()
            .map(|&(seg, attr_id, block)| column_key(seg, attr_id, block))
            .collect();
        let results = self.db.batched_multi_get_cf(cf, raw_keys.iter(), sorted_input);
        let mut out = Vec::with_capacity(keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn get_vector(&self, row: RowId) -> Result<Option<Vec<f32>>> {
        let cf = self.cf(CF_VECTORS)?;
        match self
            .db
            .get_pinned_cf(cf, row_key(row))
            .map_err(map_db_err)?
        {
            Some(slice) => Ok(Some(decode_f32_slice(&slice, self.dim)?)),
            None => Ok(None),
        }
    }

    pub fn multi_get_vectors_pinned(
        &self,
        sorted_rows: &[RowId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        self.batched_multi_get_row_pinned(CF_VECTORS, sorted_rows, true)
    }

    fn batched_multi_get_row_pinned(
        &self,
        cf_name: &str,
        rows: &[RowId],
        sorted_input: bool,
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(cf_name)?;
        let keys: Vec<[u8; 8]> = rows.iter().map(|&r| row_key(r)).collect();
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted_input);
        let mut out = Vec::with_capacity(rows.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn get_doc_id_block(
        &self,
        seg: SegmentId,
        block: u32,
    ) -> Result<Option<Arc<[DocId; ROW_BLOCK_SIZE]>>> {
        if let Some(cached) = self
            .doc_id_block_cache
            .read()
            .expect("doc_id_block_cache poisoned")
            .get(&(seg, block))
        {
            return Ok(Some(cached));
        }
        let cf = self.cf(CF_ROW_DOC_IDS)?;
        let Some(bytes) = self
            .db
            .get_pinned_cf(cf, doc_id_block_key(seg, block))
            .map_err(map_db_err)?
        else {
            return Ok(None);
        };
        let decoded = decode_doc_id_block(&bytes)?;
        let arc: Arc<[DocId; ROW_BLOCK_SIZE]> = Arc::from(decoded);
        self.doc_id_block_cache
            .write()
            .expect("doc_id_block_cache poisoned")
            .insert((seg, block), Arc::clone(&arc));
        Ok(Some(arc))
    }

    /// Resolve a slice of [`RowId`]s to [`DocId`]s through the bounded `doc_id_block_cache`.
    pub fn resolve_doc_ids(&self, rows: &[RowId]) -> Result<Vec<Option<DocId>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(rows.len());
        let mut last_key: Option<(SegmentId, u32)> = None;
        let mut last_blk: Option<Arc<[DocId; ROW_BLOCK_SIZE]>> = None;
        for &row in rows {
            let key = (row.seg(), row.block_idx());
            if last_key != Some(key) {
                last_key = Some(key);
                last_blk = self.get_doc_id_block(key.0, key.1)?;
            }
            out.push(last_blk.as_ref().map(|blk| blk[row.slot() as usize]));
        }
        Ok(out)
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
        match self
            .db
            .get_pinned_cf(cf, node_key(leaf_id))
            .map_err(map_db_err)?
        {
            Some(bytes) => Ok(Some(PackedLeafNode::decode(
                &bytes,
                self.dim,
                self.padded_dim,
                self.rabitq_words,
            )?)),
            None => Ok(None),
        }
    }

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

    pub fn get_attr_posting(&self, key: &[u8]) -> Result<Option<SegPosting>> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        match self.db.get_pinned_cf(cf, key).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(SegPosting::decode(&bytes)?)),
            None => Ok(None),
        }
    }

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

    pub fn visit_attr_postings_for_value(
        &self,
        attr_key: &[u8],
        mut visit: impl FnMut(SegmentId, &[u8]) -> Result<()>,
    ) -> Result<()> {
        debug_assert!(attr_key.len() >= 3);
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(attr_key, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            let (prefix, seg) = split_posting_key(&k)?;
            if prefix != attr_key {
                break;
            }
            visit(seg, &v)?;
        }
        Ok(())
    }

    pub fn visit_attr_postings_range(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        start_val: &[u8],
        mut visit: impl FnMut(&[u8], SegmentId, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let prefix = encode_attr_prefix(attr_id, type_tag);
        let mut seek_key = Vec::with_capacity(prefix.len() + start_val.len());
        seek_key.extend_from_slice(&prefix);
        seek_key.extend_from_slice(start_val);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&seek_key, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            let (val_prefix, seg) = split_posting_key(&k)?;
            if !visit(&val_prefix[3..], seg, &v)? {
                break;
            }
        }
        Ok(())
    }

    pub fn visit_segment_rows(
        &self,
        seg: SegmentId,
        mut visit: impl FnMut(RowId, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_ROWS)?;
        let start = row_key(RowId::new(seg, 0));
        let iter = self
            .db
            .iterator_cf(cf, IteratorMode::From(&start, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            let row = decode_row_key(&k)?;
            if row.seg() != seg {
                break;
            }
            visit(row, &v)?;
        }
        Ok(())
    }

    /// Visit all `(val_bytes, seg, posting_bytes)` for `(attr_id, type_tag)` in `cf_attr_postings`,
    /// ordered forward (`reverse == false`) or reverse (`reverse == true`) by `(val_bytes, seg)`.
    pub fn visit_attr_postings_prefix(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        reverse: bool,
        mut visit: impl FnMut(&[u8], SegmentId, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let prefix = encode_attr_prefix(attr_id, type_tag);
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        options.set_total_order_seek(true);
        let mut iter = self.db.raw_iterator_cf_opt(cf, options);
        if reverse {
            iter.seek_to_last();
        } else {
            iter.seek(&prefix);
        }
        while let Some((k, v)) = iter.item() {
            if !k.starts_with(&prefix) {
                break;
            }
            let (val_prefix, seg) = split_posting_key(k)?;
            if !visit(&val_prefix[3..], seg, v)? {
                break;
            }
            if reverse {
                iter.prev();
            } else {
                iter.next();
            }
        }
        iter.status().map_err(map_db_err)
    }

    /// Largest encoded value in `default[0x06]` with live count `> 0` for `(attr_id, type_tag)`.
    pub fn max_attr_count_key(&self, attr_id: AttrId, type_tag: u8) -> Result<Option<Vec<u8>>> {
        let a = attr_id.to_be_bytes();
        let prefix = [ATTR_COUNT_TAG, a[0], a[1], type_tag];
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        options.set_total_order_seek(true);
        let mut iter = self.db.raw_iterator_opt(options);
        iter.seek_to_last();
        while let Some((key, value)) = iter.item() {
            if !key.starts_with(&prefix) {
                break;
            }
            if decode_count(value)? > 0 {
                return Ok(Some(key[prefix.len()..].to_vec()));
            }
            iter.prev();
        }
        iter.status().map_err(map_db_err)?;
        Ok(None)
    }

    pub fn get_zones_cached(&self) -> Result<ZoneTable> {
        {
            let guard = self.zone_cache.read().expect("zone_cache poisoned");
            if let Some(ref table) = *guard {
                return Ok(Arc::clone(table));
            }
        }
        let mut write_guard = self.zone_cache.write().expect("zone_cache poisoned");
        if let Some(ref table) = *write_guard {
            return Ok(Arc::clone(table));
        }

        let mut map: HashMap<(AttrId, u8), HashMap<SegmentId, ZoneEntry>> = HashMap::new();
        self.visit_default(&[ZONE_TAG], |k, v| {
            if k.len() != 7 {
                return Err(TreeError::CorruptedData(format!(
                    "zone key expected 7 bytes after tag, got {}",
                    k.len()
                )));
            }
            let seg = u32::from_be_bytes(k[0..4].try_into().unwrap());
            let attr_id = u16::from_be_bytes([k[4], k[5]]);
            let tag = k[6];
            map.entry((attr_id, tag))
                .or_default()
                .insert(seg, ZoneEntry::decode(v)?);
            Ok(())
        })?;
        let table = Arc::new(map);
        *write_guard = Some(Arc::clone(&table));
        Ok(table)
    }

    pub fn get_attr_count(&self, attr_key: &[u8]) -> Result<u64> {
        let mut key = Vec::with_capacity(1 + attr_key.len());
        key.push(ATTR_COUNT_TAG);
        key.extend_from_slice(attr_key);
        match self.db.get_pinned(key).map_err(map_db_err)? {
            Some(bytes) => decode_count(&bytes),
            None => Ok(0),
        }
    }

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

    pub fn has_attr_counts_marker(&self) -> Result<bool> {
        Ok(self
            .db
            .get_pinned(ATTR_COUNTS_VERSION_KEY)
            .map_err(map_db_err)?
            .is_some())
    }

    pub fn get_fts_meta_bytes(&self) -> Result<Option<Vec<u8>>> {
        self.db.get(FTS_META_KEY).map_err(map_db_err)
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
        let iter =
            self.db
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

    pub fn get_fts_skip_bytes(&self, key: &[u8; 12]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_SKIPS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn get_fts_posting_bytes(&self, key: &[u8; 12]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn get_fts_posting_pinned(
        &self,
        term_id: TermId,
        block_id: RowBlock,
    ) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        let key = fts_block_key_coord(term_id, block_id);
        self.db.get_pinned_cf(cf, key).map_err(map_db_err)
    }

    pub fn all_fts_terms(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.scan_cf(CF_FTS_TERMS, |k| Ok(k.to_vec()), |v| Ok(v.to_vec()))
    }

    pub fn visit_fts_postings_for_term(
        &self,
        term_id: TermId,
        mut visit: impl FnMut(RowBlock, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        let prefix = encode_fts_term_id_prefix(term_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&prefix, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if k.len() != 12 || k[0..4] != prefix {
                break;
            }
            let (tid, block) = decode_fts_block_key(&k)?;
            if tid != term_id {
                break;
            }
            visit(block, &v)?;
        }
        Ok(())
    }

    pub fn get_fts_position_bytes(&self, key: &[u8; 12]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_POSITIONS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn multi_get_fts_positions_pinned(
        &self,
        keys: &[[u8; 12]],
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

    pub fn get_fts_doc_len_bytes(&self, key: &[u8; 10]) -> Result<Option<Vec<u8>>> {
        let key = tagged_key::<11>(FTS_DOC_LEN_TAG, key);
        self.db.get(key).map_err(map_db_err)
    }

    pub fn open_fts_skip_cursor(&self, record: TermRecord) -> Result<RocksSkipCursor<'_>> {
        if let Some((bid, bytes)) = record.inline_block {
            return RocksSkipCursor::from_inline(&self.db, record.term_id, bid, bytes);
        }
        RocksSkipCursor::open_out_of_line(&self.db, record.term_id)
    }

    pub fn open_fts_skip_cursors(&self, records: &[TermRecord]) -> Result<Vec<RocksSkipCursor<'_>>> {
        records
            .iter()
            .map(|r| self.open_fts_skip_cursor(r.clone()))
            .collect()
    }

    pub fn open_fts_doc_len_cursor(&self, attr_id: AttrId) -> Result<RocksDocLenCursor> {
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
        let mut blocks: HashMap<RowBlock, Box<[u8]>> = HashMap::new();
        self.visit_default(&prefix, |key, value| {
            if key.len() != 8 {
                return Err(TreeError::CorruptedData(format!(
                    "doc_len suffix expected 8 bytes, got {}",
                    key.len()
                )));
            }
            let seg = u32::from_be_bytes(key[0..4].try_into().unwrap());
            let blk = u32::from_be_bytes(key[4..8].try_into().unwrap());
            blocks.insert(RowBlock::new(seg, blk), value.into());
            Ok(())
        })?;
        let table = Arc::new(blocks);
        write_guard.insert(attr_id, Arc::clone(&table));
        Ok(RocksDocLenCursor { blocks: table })
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

    /// Iterate `cf_leaf_entries` in ascending `NodeId` order starting at `start_leaf_id`.
    /// When `visit` returns `Ok(false)`, stops immediately at `leaf_id` (if `stop_after` was not
    /// requested) or after `leaf_id`: passing `(bool_should_continue, bool_advance_past_current)`.
    pub fn visit_leaf_nodes_from(
        &self,
        start_leaf_id: NodeId,
        mut visit: impl FnMut(NodeId, &[u8]) -> Result<LeafVisitAction>,
    ) -> Result<Option<NodeId>> {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        let start_key = node_key(start_leaf_id);
        let mut iter = self
            .db
            .iterator_cf(cf, IteratorMode::From(&start_key, rocksdb::Direction::Forward));
        while let Some(item) = iter.next() {
            let (k, v) = item.map_err(map_db_err)?;
            let leaf_id = decode_u32_key(&k)?;
            match visit(leaf_id, &v)? {
                LeafVisitAction::Continue => {}
                LeafVisitAction::StopBefore => return Ok(Some(leaf_id)),
                LeafVisitAction::StopAfter => {
                    return match iter.next() {
                        Some(next_item) => {
                            let (nk, _) = next_item.map_err(map_db_err)?;
                            Ok(Some(decode_u32_key(&nk)?))
                        }
                        None => Ok(None),
                    };
                }
            }
        }
        Ok(None)
    }

    pub fn all_vectors(&self) -> Result<Vec<(RowId, Vec<f32>)>> {
        self.scan_cf(
            CF_VECTORS,
            |k| {
                let arr: [u8; 8] = k.try_into().map_err(|_| {
                    TreeError::CorruptedData(format!(
                        "cf_vectors key expected 8 bytes, got {}",
                        k.len()
                    ))
                })?;
                Ok(RowId::from_be_bytes(arr))
            },
            |v| decode_f32_slice(v, self.dim),
        )
    }

    pub fn all_id_map(&self) -> Result<Vec<(DocId, RowId)>> {
        self.scan_cf(CF_ID_MAP, decode_u64_key, |v| {
            let arr: [u8; 8] = v.try_into().map_err(|_| {
                TreeError::CorruptedData("cf_id_map value expected 8 bytes".into())
            })?;
            Ok(RowId::from_be_bytes(arr))
        })
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

    pub fn visit_default(
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

    pub(super) fn cf(&self, name: &str) -> Result<&ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| TreeError::CorruptedData(format!("missing column family {name}")))
    }
}

#[inline]
pub fn node_key(id: NodeId) -> [u8; 4] {
    id.to_be_bytes()
}

#[inline]
pub fn doc_key(id: DocId) -> [u8; 8] {
    id.to_be_bytes()
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

pub fn default_sst_block_cache() -> Cache {
    Cache::new_lru_cache(BLOCK_CACHE_BYTES)
}

pub fn cf_options_for(cf_name: &str, block_cache: &Cache) -> Result<Options> {
    match cf_name {
        DEFAULT_COLUMN_FAMILY_NAME | CF_ID_MAP | CF_ROWS | CF_COLUMNS | CF_DETACHED => {
            Ok(point_lookup_cf_opts(block_cache, false))
        }
        CF_ROW_DOC_IDS | CF_VECTORS | CF_LEAF_ENTRIES => {
            Ok(point_lookup_cf_opts(block_cache, true))
        }
        CF_ATTR_POSTINGS => Ok(prefix_cf_opts(block_cache, ATTR_POSTING_PREFIX_LEN, true)),
        CF_DETACHED_POSTINGS => Ok(prefix_cf_opts(
            block_cache,
            DETACHED_POSTING_PREFIX_LEN,
            true,
        )),
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
    use izzydb_core::config::{DistanceMetric, TreeConfig};
    use rocksdb::SstFileWriter;
    use tempfile::TempDir;

    #[test]
    fn test_sst_ingest_and_bulk_import_marker() {
        let dir = TempDir::new().unwrap();
        let sst_dir = TempDir::new().unwrap();
        let cfg = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared));

        let storage = Storage::open_for_bulk_ingest(dir.path(), &cfg).unwrap();
        storage.put_bulk_import_marker(42).unwrap();
        assert!(storage.has_bulk_import_marker().unwrap());

        let sst_path = sst_dir.path().join("rows.sst");
        {
            let opts = storage.cf_options(CF_ROWS).unwrap();
            let mut writer = SstFileWriter::create(&opts);
            writer.open(&sst_path).unwrap();
            writer.put(row_key(RowId::new(0, 10)), b"row-10").unwrap();
            writer.put(row_key(RowId::new(0, 20)), b"row-20").unwrap();
            writer.finish().unwrap();
        }
        storage
            .ingest_cf(CF_ROWS, std::slice::from_ref(&sst_path))
            .unwrap();
        drop(storage);

        assert!(matches!(
            Storage::open(dir.path(), &cfg),
            Err(TreeError::CorruptedData(msg)) if msg.contains("incomplete bulk import")
        ));

        {
            let block_cache = default_sst_block_cache();
            let raw = Storage::open_raw(dir.path(), 4, &block_cache).unwrap();
            raw.clear_bulk_import_marker().unwrap();
        }
        let (reopened, _manifest, _ann) = Storage::open(dir.path(), &cfg).unwrap();
        assert_eq!(
            reopened.get_row_bytes(RowId::new(0, 10)).unwrap().as_deref(),
            Some(b"row-10".as_slice())
        );
        assert_eq!(
            reopened.get_row_bytes(RowId::new(0, 20)).unwrap().as_deref(),
            Some(b"row-20".as_slice())
        );
    }

    #[test]
    fn test_vectorless_collection_open_and_mismatch() {
        let dir = TempDir::new().unwrap();
        let cfg_no_vec = CollectionConfig::without_vectors().with_seal_rows(128);
        let (storage, manifest, ann) = Storage::open(dir.path(), &cfg_no_vec).unwrap();
        assert!(ann.is_none());
        assert_eq!(manifest.seal_rows, 128);
        storage
            .put_schema(&CollectionSchema::from_config(&cfg_no_vec).unwrap())
            .unwrap();
        drop(storage);

        // Reopening vectorless DB with vectors must fail with ConfigMismatch
        let cfg_vec = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared));
        assert!(matches!(
            Storage::open(dir.path(), &cfg_vec),
            Err(TreeError::ConfigMismatch { .. })
        ));
    }
}
