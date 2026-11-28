mod attr_postings;
pub mod batch;
mod cf_options;
mod compaction_filter;
mod detached;
mod fts;
pub mod fts_cursors;
mod reads;
mod scans;
pub mod segments;
#[cfg(test)]
mod tests;
pub mod tx;
pub mod types;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use roaring::RoaringBitmap;
use rocksdb::{
    Cache, ColumnFamily, ColumnFamilyDescriptor, DB, DEFAULT_COLUMN_FAMILY_NAME,
    IngestExternalFileOptions, IteratorMode, Options, WriteBatch, WriteOptions,
};

use izzydb_core::config::{AttrId, ColumnGen, DocId, NodeId, ROW_BLOCK_SIZE, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::CollectionConfig;

use crate::codec::{
    Manifest, SEG_DELETES_TAG, SEG_META_TAG, SegmentMeta, SegmentState, ZONE_TAG,
    decode_roaring_bitmap,
};
use crate::node::{AnnMeta, PackedLeafNode};

pub use batch::StorageBatch;
use cf_options::cf_options_with_filters;
pub use cf_options::{
    DEFAULT_BLOCK_CACHE_BYTES, cf_options_for, default_sst_block_cache, sst_block_cache,
};
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
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.map.entry(key) {
            e.insert(val);
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
    pub(super) retired_detached: Arc<RwLock<Arc<HashMap<AttrId, ColumnGen>>>>,
    pub(super) doc_id_block_cache: RwLock<DocIdBlockCache>,
    /// Bounded cache of decoded sealed-segment attribute postings.
    pub(super) attr_posting_cache: RwLock<attr_postings::AttrPostingCache>,
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
        let block_cache = sst_block_cache(
            config
                .block_cache_bytes
                .unwrap_or(DEFAULT_BLOCK_CACHE_BYTES),
        );
        let dim = config.vector.as_ref().map_or(0, |v| v.dim);
        let storage = Self::open_raw(path, dim, &block_cache, config.direct_io)?;

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
                if manifest.total_rows > 0
                    || storage.get_schema()?.is_some_and(|s| !s.has_vectors())
                {
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
        storage.recover_building_segments()?;
        Ok((storage, manifest, ann_meta))
    }

    fn open_raw(
        path: impl AsRef<Path>,
        dim: usize,
        block_cache: &Cache,
        direct_io: bool,
    ) -> Result<Self> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.set_db_write_buffer_size(DB_WRITE_BUFFER_BYTES);
        if direct_io {
            db_opts.set_use_direct_reads(true);
            db_opts.set_use_direct_io_for_flush_and_compaction(true);
        }

        let dropped_segments = Arc::new(RwLock::new(Arc::new(RoaringBitmap::new())));
        let retired_detached = Arc::new(RwLock::new(Arc::new(HashMap::new())));

        let mut cf_names = Vec::with_capacity(1 + ALL_DATA_CF_NAMES.len());
        cf_names.push(DEFAULT_COLUMN_FAMILY_NAME);
        cf_names.extend_from_slice(&ALL_DATA_CF_NAMES);

        let mut cf_descriptors = Vec::with_capacity(cf_names.len());
        for name in cf_names {
            cf_descriptors.push(ColumnFamilyDescriptor::new(
                name,
                cf_options_with_filters(
                    name,
                    block_cache,
                    Some((&dropped_segments, &retired_detached)),
                )?,
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
            dropped_segments,
            retired_detached,
            doc_id_block_cache: RwLock::new(DocIdBlockCache::new(DOC_ID_BLOCK_CACHE_CAP)),
            attr_posting_cache: RwLock::new(attr_postings::AttrPostingCache::new(
                attr_postings::ATTR_POSTING_CACHE_BYTES,
            )),
            zone_cache: RwLock::new(None),
            fts_doc_lens_cache: RwLock::new(HashMap::new()),
        })
    }

    fn load_segment_state(&self) -> Result<()> {
        let mut entries: BTreeMap<SegmentId, SegmentEntry> = BTreeMap::new();
        self.visit_default(&[SEG_META_TAG], |k, v| {
            let seg = decode_u32_key(k)?;
            let meta = SegmentMeta::decode(v)?;
            entries.insert(
                seg,
                SegmentEntry {
                    meta,
                    deleted: None,
                },
            );
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

        let dropped = match self
            .db
            .get_pinned(DROPPED_SEGMENTS_KEY)
            .map_err(map_db_err)?
        {
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
        let block_cache = sst_block_cache(
            config
                .block_cache_bytes
                .unwrap_or(DEFAULT_BLOCK_CACHE_BYTES),
        );
        let dim = config.vector.as_ref().map_or(0, |v| v.dim);
        let storage = Self::open_raw(path, dim, &block_cache, config.direct_io)?;

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
            if storage
                .db
                .iterator_cf(cf, IteratorMode::Start)
                .next()
                .is_some()
            {
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
        let mut retired_map = HashMap::with_capacity(metas.len());
        for (attr_id, mut meta) in metas {
            if meta.pending_gen != 0 {
                meta.retired_below = meta.retired_below.max(meta.pending_gen.saturating_add(1));
                meta.pending_gen = 0;
                meta.built_against_watermark = RowId(0);
                to_fix.push((attr_id, meta.clone()));
            }
            if meta.retired_below > 0 {
                retired_map.insert(attr_id, meta.retired_below);
            }
        }
        if !to_fix.is_empty() {
            let mut batch = self.new_batch();
            for (attr_id, meta) in to_fix {
                batch.put_detached_meta(attr_id, &meta);
            }
            self.write_batch_sync(batch)?;
        }
        *self
            .retired_detached
            .write()
            .expect("retired_detached poisoned") = Arc::new(retired_map);
        Ok(())
    }

    /// Discard any `SegmentState::Building` segments left by a crash mid-compaction.
    fn recover_building_segments(&self) -> Result<()> {
        let snap = self.segments.snapshot();
        let building: Vec<SegmentId> = snap
            .segments
            .iter()
            .filter_map(|(&seg, e)| (e.meta.state == SegmentState::Building).then_some(seg))
            .collect();
        if building.is_empty() {
            return Ok(());
        }
        let mut dropped = (*self.dropped_segments_snapshot()).clone();
        let mut batch = self.new_batch();
        for &seg in &building {
            dropped.insert(seg);
            batch.delete_seg_meta(seg);
            batch.delete_seg_deletes(seg);
            if let Some(next_seg) = seg.checked_add(1) {
                let lo = seg.to_be_bytes();
                let hi = next_seg.to_be_bytes();
                for cf_name in [CF_ROWS, CF_ROW_DOC_IDS, CF_VECTORS, CF_COLUMNS] {
                    batch.delete_range_cf(cf_name, &lo, &hi);
                }
                let mut zlo = [0u8; 5];
                zlo[0] = ZONE_TAG;
                zlo[1..5].copy_from_slice(&lo);
                let mut zhi = [0u8; 5];
                zhi[0] = ZONE_TAG;
                zhi[1..5].copy_from_slice(&hi);
                batch.delete_range_default(&zlo, &zhi);
            }
        }
        batch.put_dropped_segments(&crate::codec::encode_roaring_bitmap(&dropped)?);
        self.write_batch_sync(batch)?;

        self.segments.mutate_and_publish(|s| {
            for seg in &building {
                s.segments.remove(seg);
            }
        });
        *self
            .dropped_segments
            .write()
            .expect("dropped_segments poisoned") = Arc::new(dropped);
        Ok(())
    }

    /// Trigger RocksDB manual compaction across `default` and all data CFs (reclaiming
    /// tombstones, range deletes, and compaction-filtered orphaned segment/generation keys).
    pub fn compact_range_all(&self) -> Result<()> {
        self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
        for cf_name in ALL_DATA_CF_NAMES {
            let cf = self.cf(cf_name)?;
            self.db.compact_range_cf(cf, None::<&[u8]>, None::<&[u8]>);
        }
        Ok(())
    }

    /// Count physical keys currently stored in `cf_name` matching `pred` (test / diagnostics).
    pub fn count_keys_in_cf(
        &self,
        cf_name: Option<&str>,
        mut pred: impl FnMut(&[u8]) -> bool,
    ) -> Result<usize> {
        let mut count = 0usize;
        match cf_name {
            None => {
                for item in self.db.iterator(IteratorMode::Start) {
                    let (k, _) = item.map_err(map_db_err)?;
                    if pred(&k) {
                        count += 1;
                    }
                }
            }
            Some(name) => {
                let cf = self.cf(name)?;
                for item in self.db.iterator_cf(cf, IteratorMode::Start) {
                    let (k, _) = item.map_err(map_db_err)?;
                    if pred(&k) {
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
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
