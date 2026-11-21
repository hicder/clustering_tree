use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

use roaring::RoaringBitmap;
use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamily, ColumnFamilyDescriptor, DBCompressionType,
    DBPinnableSlice, DBRawIteratorWithThreadMode, Direction, IteratorMode, Options, ReadOptions,
    SliceTransform, WriteBatch, DB, DEFAULT_COLUMN_FAMILY_NAME,
};

use izzydb_core::config::{
    AttrId, DocId, FTS_BLOCK_SIZE, FtsBlockId, FtsDocId, LocalId, NodeId, TermId, TreeConfig,
    fts_block_id, fts_slot, make_fts_block_id,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::CollectionSchema;
use izzydb_fts::{PostingBlockHeader, SkipBlockLayout, SkipBlockView, TermRecord};

use crate::codec::{
    LeafPosting, LeafPostingView, decode_roaring_bitmap, decode_zone_map, encode_attr_prefix,
    encode_fts_block_key, encode_fts_term_id_key, encode_fts_term_id_prefix,
    encode_posting_prefix, encode_roaring_bitmap, encode_zone_map, make_posting_key,
    split_posting_key,
};
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
pub const CF_ATTR_ZONES: &str = "cf_attr_zones";
pub const CF_FTS_REV_DOC_MAP: &str = "cf_fts_rev_doc_map";
pub const CF_FTS_TERMS: &str = "cf_fts_terms";
pub const CF_FTS_SKIPS: &str = "cf_fts_skips";
pub const CF_FTS_POSTINGS: &str = "cf_fts_postings";
pub const CF_FTS_POSITIONS: &str = "cf_fts_positions";
pub const CF_FTS_DOC_LENS: &str = "cf_fts_doc_lens";

const META_KEY: &[u8] = b"tree_meta";
const SCHEMA_KEY: &[u8] = b"schema";
const FTS_META_KEY: &[u8] = b"fts_meta";
const FTS_FREE_IDS_KEY: &[u8] = b"fts_free_ids";
const BLOCK_CACHE_BYTES: usize = 128 * 1024 * 1024;
const FTS_REV_BLOCK_HEADER_LEN: usize = 32;
const FTS_REV_BLOCK_LEN: usize =
    FTS_REV_BLOCK_HEADER_LEN + FTS_BLOCK_SIZE * (8 + 4 + 4);

/// Physical location of a document in the clustering tree (plus optional `FtsDocId`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocLoc {
    pub leaf_id: NodeId,
    pub local_id: LocalId,
    pub fts_doc_id: Option<FtsDocId>,
}

impl DocLoc {
    #[inline]
    pub fn encode(self) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&self.leaf_id.to_be_bytes());
        out[4..8].copy_from_slice(&self.local_id.to_be_bytes());
        out[8..12].copy_from_slice(&self.fts_doc_id.unwrap_or(FtsDocId::MAX).to_be_bytes());
        out
    }

    #[inline]
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 12 {
            return Err(TreeError::CorruptedData(format!(
                "DocLoc expected 12 bytes, got {}",
                bytes.len()
            )));
        }
        let raw_fts = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
        Ok(Self {
            leaf_id: u32::from_be_bytes(bytes[0..4].try_into().unwrap()),
            local_id: u32::from_be_bytes(bytes[4..8].try_into().unwrap()),
            fts_doc_id: if raw_fts == FtsDocId::MAX {
                None
            } else {
                Some(raw_fts)
            },
        })
    }
}

/// 256-document block-packed reverse map in `cf_fts_rev_doc_map` mapping `slot -> (DocId, leaf_id, local_id)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtsRevBlock {
    pub present: [u64; 4],
    pub doc_ids: [DocId; FTS_BLOCK_SIZE],
    pub leaf_ids: [NodeId; FTS_BLOCK_SIZE],
    pub local_ids: [LocalId; FTS_BLOCK_SIZE],
}

impl Default for FtsRevBlock {
    fn default() -> Self {
        Self {
            present: [0u64; 4],
            doc_ids: [0u64; FTS_BLOCK_SIZE],
            leaf_ids: [0u32; FTS_BLOCK_SIZE],
            local_ids: [0u32; FTS_BLOCK_SIZE],
        }
    }
}

impl FtsRevBlock {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.present == [0u64; 4]
    }

    #[inline]
    pub fn get(&self, slot: u8) -> Option<(DocId, NodeId, LocalId)> {
        let s = slot as usize;
        if ((self.present[s >> 6] >> (s & 63)) & 1) == 0 {
            return None;
        }
        Some((self.doc_ids[s], self.leaf_ids[s], self.local_ids[s]))
    }

    #[inline]
    pub fn get_doc_id(&self, slot: u8) -> Option<DocId> {
        let s = slot as usize;
        if ((self.present[s >> 6] >> (s & 63)) & 1) == 0 {
            return None;
        }
        Some(self.doc_ids[s])
    }

    #[inline]
    pub fn set(&mut self, slot: u8, doc_id: DocId, leaf_id: NodeId, local_id: LocalId) {
        let s = slot as usize;
        self.present[s >> 6] |= 1u64 << (s & 63);
        self.doc_ids[s] = doc_id;
        self.leaf_ids[s] = leaf_id;
        self.local_ids[s] = local_id;
    }

    #[inline]
    pub fn remove(&mut self, slot: u8) {
        let s = slot as usize;
        self.present[s >> 6] &= !(1u64 << (s & 63));
        self.doc_ids[s] = 0;
        self.leaf_ids[s] = 0;
        self.local_ids[s] = 0;
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(FTS_REV_BLOCK_LEN);
        for w in self.present {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for v in self.doc_ids {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in self.leaf_ids {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in self.local_ids {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != FTS_REV_BLOCK_LEN {
            return Err(TreeError::CorruptedData(format!(
                "FtsRevBlock expected {FTS_REV_BLOCK_LEN} bytes, got {}",
                bytes.len()
            )));
        }
        let mut present = [0u64; 4];
        for (i, out) in present.iter_mut().enumerate() {
            let off = i * 8;
            *out = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        }
        let mut off = FTS_REV_BLOCK_HEADER_LEN;
        let mut doc_ids = [0u64; FTS_BLOCK_SIZE];
        for out in &mut doc_ids {
            *out = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            off += 8;
        }
        let mut leaf_ids = [0u32; FTS_BLOCK_SIZE];
        for out in &mut leaf_ids {
            *out = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            off += 4;
        }
        let mut local_ids = [0u32; FTS_BLOCK_SIZE];
        for out in &mut local_ids {
            *out = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            off += 4;
        }
        Ok(Self {
            present,
            doc_ids,
            leaf_ids,
            local_ids,
        })
    }
}

pub type AttrZoneMap = BTreeMap<NodeId, (Vec<u8>, Vec<u8>)>;
type AttrZoneTable = Arc<HashMap<(AttrId, u8), AttrZoneMap>>;
type DocLenBlockTable = Arc<Vec<Option<Box<[u8]>>>>;
type FtsRevBlockTable = Arc<Vec<Option<Arc<FtsRevBlock>>>>;

/// RocksDB storage wrapper managing packed node, document, and attribute index column families.
pub struct Storage {
    db: DB,
    _block_cache: Cache,
    dim: usize,
    padded_dim: usize,
    rabitq_words: usize,
    fts_doc_lens_cache: RwLock<HashMap<u16, DocLenBlockTable>>,
    fts_rev_map_cache: RwLock<Option<FtsRevBlockTable>>,
    attr_zones_cache: RwLock<Option<AttrZoneTable>>,
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
            ColumnFamilyDescriptor::new(CF_ATTR_CLUSTERS, prefix_cf_opts(&block_cache, 3, false)),
            ColumnFamilyDescriptor::new(CF_ATTR_POSTINGS, prefix_cf_opts(&block_cache, 7, true)),
            ColumnFamilyDescriptor::new(CF_ATTR_ZONES, point_lookup_cf_opts(&block_cache, false)),
            ColumnFamilyDescriptor::new(
                CF_FTS_REV_DOC_MAP,
                point_lookup_cf_opts(&block_cache, true),
            ),
            ColumnFamilyDescriptor::new(CF_FTS_TERMS, prefix_cf_opts(&block_cache, 2, false)),
            ColumnFamilyDescriptor::new(CF_FTS_SKIPS, prefix_cf_opts(&block_cache, 4, true)),
            ColumnFamilyDescriptor::new(CF_FTS_POSTINGS, prefix_cf_opts(&block_cache, 4, true)),
            ColumnFamilyDescriptor::new(CF_FTS_POSITIONS, prefix_cf_opts(&block_cache, 4, false)),
            ColumnFamilyDescriptor::new(CF_FTS_DOC_LENS, point_lookup_cf_opts(&block_cache, true)),
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
            fts_doc_lens_cache: RwLock::new(HashMap::new()),
            fts_rev_map_cache: RwLock::new(None),
            attr_zones_cache: RwLock::new(None),
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

        let cf = self.cf(CF_ATTR_ZONES)?;
        let mut map: HashMap<(AttrId, u8), AttrZoneMap> = HashMap::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            if k.len() != 3 {
                return Err(TreeError::CorruptedData(format!(
                    "cf_attr_zones key expected 3 bytes, got {}",
                    k.len()
                )));
            }
            let attr_id = u16::from_be_bytes([k[0], k[1]]);
            let type_tag = k[2];
            map.insert((attr_id, type_tag), decode_zone_map(&v)?);
        }
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
    #[allow(clippy::too_many_arguments)]
    pub fn scan_attr_posting_range_into_words(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        leaf_id: NodeId,
        lo_val: Option<(&[u8], bool)>,
        hi_val: Option<(&[u8], bool)>,
        dst_words: &mut [u64],
    ) -> Result<()> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let prefix = encode_posting_prefix(attr_id, type_tag, leaf_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);

        let start_key = match lo_val {
            Some((v, _)) => {
                let mut k = Vec::with_capacity(7 + v.len());
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
            let val_bytes = &k[7..];
            if let Some((lo_b, lo_inc)) = lo_val
                && !lo_inc
                && val_bytes == lo_b
            {
                iter.next();
                continue;
            }
            if let Some((hi_b, hi_inc)) = hi_val {
                let cmp = val_bytes.cmp(hi_b);
                if if hi_inc {
                    cmp == std::cmp::Ordering::Greater
                } else {
                    cmp != std::cmp::Ordering::Less
                } {
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

    pub fn all_attr_postings(&self) -> Result<Vec<(Vec<u8>, LeafPosting)>> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((k.to_vec(), LeafPosting::decode(&v)?));
        }
        Ok(out)
    }

    pub fn all_attr_zones(&self) -> Result<Vec<(Vec<u8>, AttrZoneMap)>> {
        let cf = self.cf(CF_ATTR_ZONES)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((k.to_vec(), decode_zone_map(&v)?));
        }
        Ok(out)
    }

    pub fn get_fts_meta_bytes(&self) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_META)?;
        self.db.get_cf(cf, FTS_META_KEY).map_err(map_db_err)
    }

    pub fn get_fts_free_ids(&self) -> Result<RoaringBitmap> {
        let cf = self.cf(CF_META)?;
        match self.db.get_pinned_cf(cf, FTS_FREE_IDS_KEY).map_err(map_db_err)? {
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

        let cf = self.cf(CF_FTS_REV_DOC_MAP)?;
        let mut blocks: Vec<Option<Arc<FtsRevBlock>>> = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            let block_id = decode_u32_key(&k)? as usize;
            let blk = Arc::new(FtsRevBlock::decode(&v)?);
            if blocks.len() <= block_id {
                blocks.resize_with(block_id + 1, || None);
            }
            blocks[block_id] = Some(blk);
        }
        let table = Arc::new(blocks);
        *write_guard = Some(Arc::clone(&table));
        Ok(table)
    }

    pub fn get_fts_rev_block(&self, block_id: FtsBlockId) -> Result<Option<FtsRevBlock>> {
        let cf = self.cf(CF_FTS_REV_DOC_MAP)?;
        match self
            .db
            .get_pinned_cf(cf, encode_fts_block_key(block_id))
            .map_err(map_db_err)?
        {
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
        let cf = self.cf(CF_FTS_DOC_LENS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn open_fts_skip_cursor(&self, record: TermRecord) -> Result<RocksSkipCursor<'_>> {
        if let Some((bid, bytes)) = record.inline_block {
            let hdr = PostingBlockHeader::parse(&bytes)?;
            return Ok(RocksSkipCursor {
                skip_db: &self.db,
                skip_pin: None,
                skip_iter: None,
                chunk_layout: None,
                posting_iter: None,
                posting_pos: None,
                term_id: record.term_id,
                inline: Some((bid, bytes, (hdr.max_tf, hdr.min_dl_over_tf))),
                inline_done: false,
                chunk_id: 0,
                entry_idx: 0,
                entry_count: 0,
                current_block: Some(bid),
                current_bounds: (hdr.max_tf, hdr.min_dl_over_tf),
            });
        }
        let cf = self.cf(CF_FTS_SKIPS)?;
        if let Some(pin) = self
            .db
            .get_pinned_cf(cf, encode_fts_term_id_key(record.term_id, 0))
            .map_err(map_db_err)?
        {
            return RocksSkipCursor::from_chunk0_pin(&self.db, record.term_id, pin);
        }
        let mut cursor = RocksSkipCursor::empty_out_of_line(&self.db, record.term_id);
        cursor.ensure_skip_iter_seek(0, 0)?;
        Ok(cursor)
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
            if let Some((bid, ref bytes)) = rec.inline_block {
                let hdr = PostingBlockHeader::parse(bytes)?;
                out.push(RocksSkipCursor {
                    skip_db: &self.db,
                    skip_pin: None,
                    skip_iter: None,
                    chunk_layout: None,
                    posting_iter: None,
                    posting_pos: None,
                    term_id: rec.term_id,
                    inline: Some((bid, bytes.clone(), (hdr.max_tf, hdr.min_dl_over_tf))),
                    inline_done: false,
                    chunk_id: 0,
                    entry_idx: 0,
                    entry_count: 0,
                    current_block: Some(bid),
                    current_bounds: (hdr.max_tf, hdr.min_dl_over_tf),
                });
            } else if let Some(pin) = pin_opt {
                out.push(RocksSkipCursor::from_chunk0_pin(
                    &self.db,
                    rec.term_id,
                    pin,
                )?);
            } else {
                let mut cursor = RocksSkipCursor::empty_out_of_line(&self.db, rec.term_id);
                cursor.ensure_skip_iter_seek(0, 0)?;
                out.push(cursor);
            }
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

        let cf = self.cf(CF_FTS_DOC_LENS)?;
        let prefix = attr_id.to_be_bytes();
        let mut iter = self.db.raw_iterator_cf(cf);
        iter.seek(&prefix);
        let mut blocks: Vec<Option<Box<[u8]>>> = Vec::new();
        while let Some(k) = iter.key() {
            if k.len() != 6 || k[0] != prefix[0] || k[1] != prefix[1] {
                break;
            }
            let block_id = u32::from_be_bytes(k[2..6].try_into().unwrap()) as usize;
            if let Some(val) = iter.value() {
                if blocks.len() <= block_id {
                    blocks.resize_with(block_id + 1, || None);
                }
                blocks[block_id] = Some(val.to_vec().into_boxed_slice());
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)?;
        let table = Arc::new(blocks);
        write_guard.insert(attr_id, Arc::clone(&table));
        Ok(RocksDocLenCursor { blocks: table })
    }

    pub fn all_fts_rev_blocks(&self) -> Result<Vec<(FtsBlockId, FtsRevBlock)>> {
        let cf = self.cf(CF_FTS_REV_DOC_MAP)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((decode_u32_key(&k)?, FtsRevBlock::decode(&v)?));
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

    pub fn all_fts_skips(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf = self.cf(CF_FTS_SKIPS)?;
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

    pub fn all_fts_doc_lens(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf = self.cf(CF_FTS_DOC_LENS)?;
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

    pub fn put_attr_zone(&mut self, key: &[u8; 3], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_ATTR_ZONES).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_attr_zone(&mut self, key: &[u8; 3]) {
        let cf = self.storage.db.cf_handle(CF_ATTR_ZONES).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_fts_meta_bytes(&mut self, bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_META).unwrap();
        self.batch.put_cf(cf, FTS_META_KEY, bytes);
    }

    pub fn put_fts_free_ids(&mut self, bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_META).unwrap();
        self.batch.put_cf(cf, FTS_FREE_IDS_KEY, bytes);
    }

    pub fn delete_fts_free_ids(&mut self) {
        let cf = self.storage.db.cf_handle(CF_META).unwrap();
        self.batch.delete_cf(cf, FTS_FREE_IDS_KEY);
    }

    pub fn put_fts_rev_block(&mut self, block_id: FtsBlockId, bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_REV_DOC_MAP).unwrap();
        self.batch.put_cf(cf, encode_fts_block_key(block_id), bytes);
    }

    pub fn delete_fts_rev_block(&mut self, block_id: FtsBlockId) {
        let cf = self.storage.db.cf_handle(CF_FTS_REV_DOC_MAP).unwrap();
        self.batch.delete_cf(cf, encode_fts_block_key(block_id));
    }

    pub fn put_fts_term(&mut self, key: &[u8], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_TERMS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_fts_term(&mut self, key: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_TERMS).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_fts_skip(&mut self, key: &[u8; 8], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_SKIPS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_fts_skip(&mut self, key: &[u8; 8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_SKIPS).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_fts_posting(&mut self, key: &[u8; 8], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_POSTINGS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_fts_posting(&mut self, key: &[u8; 8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_POSTINGS).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_fts_position(&mut self, key: &[u8; 8], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_POSITIONS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_fts_position(&mut self, key: &[u8; 8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_POSITIONS).unwrap();
        self.batch.delete_cf(cf, key);
    }

    pub fn put_fts_doc_len(&mut self, key: &[u8; 6], bytes: &[u8]) {
        let cf = self.storage.db.cf_handle(CF_FTS_DOC_LENS).unwrap();
        self.batch.put_cf(cf, key, bytes);
    }

    pub fn delete_fts_doc_len(&mut self, key: &[u8; 6]) {
        let cf = self.storage.db.cf_handle(CF_FTS_DOC_LENS).unwrap();
        self.batch.delete_cf(cf, key);
    }
}

/// Cursor over a term's Block-Max skip chunks (`cf_fts_skips`) or inline posting block,
/// with a batched/pinned chunk-0 fast path (no skip iterator for `docs < 65,536`) and a lazy
/// sequential `cf_fts_postings` iterator opened only when a block survives pruning.
pub struct RocksSkipCursor<'a> {
    skip_db: &'a DB,
    skip_pin: Option<DBPinnableSlice<'a>>,
    skip_iter: Option<DBRawIteratorWithThreadMode<'a, DB>>,
    chunk_layout: Option<SkipBlockLayout>,
    posting_iter: Option<DBRawIteratorWithThreadMode<'a, DB>>,
    posting_pos: Option<FtsBlockId>,
    term_id: TermId,
    inline: Option<(FtsBlockId, Vec<u8>, (u32, f32))>,
    inline_done: bool,
    chunk_id: u32,
    entry_idx: usize,
    entry_count: usize,
    current_block: Option<FtsBlockId>,
    current_bounds: (u32, f32),
}

impl<'a> RocksSkipCursor<'a> {
    fn from_chunk0_pin(db: &'a DB, term_id: TermId, pin: DBPinnableSlice<'a>) -> Result<Self> {
        let layout = SkipBlockLayout::parse(pin.as_ref())?;
        let view = layout.view(pin.as_ref());
        let rel = view.rel_block_id(0);
        let bounds = (view.max_tf(0), view.min_dl_over_tf(0));
        let count = layout.count;
        Ok(Self {
            skip_db: db,
            skip_pin: Some(pin),
            skip_iter: None,
            chunk_layout: Some(layout),
            posting_iter: None,
            posting_pos: None,
            term_id,
            inline: None,
            inline_done: true,
            chunk_id: 0,
            entry_idx: 0,
            entry_count: count,
            current_block: Some(make_fts_block_id(0, rel)),
            current_bounds: bounds,
        })
    }

    fn empty_out_of_line(db: &'a DB, term_id: TermId) -> Self {
        Self {
            skip_db: db,
            skip_pin: None,
            skip_iter: None,
            chunk_layout: None,
            posting_iter: None,
            posting_pos: None,
            term_id,
            inline: None,
            inline_done: true,
            chunk_id: 0,
            entry_idx: 0,
            entry_count: 0,
            current_block: None,
            current_bounds: (0, f32::INFINITY),
        }
    }

    #[inline]
    fn current_chunk_view(&self) -> Option<SkipBlockView<'_>> {
        let layout = self.chunk_layout?;
        let bytes: &[u8] = if let Some(ref pin) = self.skip_pin {
            pin.as_ref()
        } else {
            self.skip_iter.as_ref()?.value()?
        };
        Some(layout.view(bytes))
    }

    #[inline]
    pub fn current_block_id(&self) -> Option<FtsBlockId> {
        self.current_block
    }

    #[inline]
    pub fn current_block_bounds(&self) -> (u32, f32) {
        self.current_bounds
    }

    #[inline]
    pub fn inline_posting_bytes(&self) -> Option<&[u8]> {
        if self.inline_done {
            return None;
        }
        self.inline.as_ref().map(|(_, bytes, _)| bytes.as_slice())
    }

    /// Return the encoded `PackedPostingBlock` bytes for `current_block`, initializing or advancing
    /// a lazy sequential iterator over `cf_fts_postings` only when called.
    pub fn current_posting_bytes(&mut self) -> Result<Option<&[u8]>> {
        if self.inline.is_some() {
            return Ok(self.inline_posting_bytes());
        }
        let Some(target) = self.current_block else {
            return Ok(None);
        };
        let prefix = encode_fts_term_id_prefix(self.term_id);
        let iter = match self.posting_iter {
            Some(ref mut it) => {
                if self.posting_pos.is_some_and(|b| b < target) {
                    let mut cur_bid = self.posting_pos;
                    let mut steps = 0usize;
                    while let Some(b) = cur_bid
                        && b < target
                        && steps < 4
                    {
                        it.next();
                        steps += 1;
                        cur_bid = match it.key() {
                            Some(k) if k.len() == 8 && k[0..4] == prefix => {
                                Some(u32::from_be_bytes(k[4..8].try_into().unwrap()))
                            }
                            _ => None,
                        };
                    }
                    if cur_bid.is_some_and(|b| b < target) {
                        it.seek(encode_fts_term_id_key(self.term_id, target));
                        cur_bid = match it.key() {
                            Some(k) if k.len() == 8 && k[0..4] == prefix => {
                                Some(u32::from_be_bytes(k[4..8].try_into().unwrap()))
                            }
                            _ => None,
                        };
                    }
                    self.posting_pos = cur_bid;
                }
                it
            }
            None => {
                let cf = self.skip_db.cf_handle(CF_FTS_POSTINGS).ok_or_else(|| {
                    TreeError::CorruptedData(format!("missing column family {CF_FTS_POSTINGS}"))
                })?;
                let mut ro = ReadOptions::default();
                ro.set_prefix_same_as_start(true);
                let mut it = self.skip_db.raw_iterator_cf_opt(cf, ro);
                it.seek(encode_fts_term_id_key(self.term_id, target));
                self.posting_pos = match it.key() {
                    Some(k) if k.len() == 8 && k[0..4] == prefix => {
                        Some(u32::from_be_bytes(k[4..8].try_into().unwrap()))
                    }
                    _ => None,
                };
                self.posting_iter.insert(it)
            }
        };
        if self.posting_pos == Some(target) {
            Ok(iter.value())
        } else {
            iter.status().map_err(map_db_err)?;
            Ok(None)
        }
    }

    fn ensure_skip_iter_seek(&mut self, seek_chunk: u32, min_rel: u8) -> Result<()> {
        self.skip_pin = None;
        self.chunk_layout = None;
        let key = encode_fts_term_id_key(self.term_id, seek_chunk);
        match self.skip_iter {
            Some(ref mut it) => it.seek(key),
            None => {
                let cf = self.skip_db.cf_handle(CF_FTS_SKIPS).ok_or_else(|| {
                    TreeError::CorruptedData(format!("missing column family {CF_FTS_SKIPS}"))
                })?;
                let mut ro = ReadOptions::default();
                ro.set_prefix_same_as_start(true);
                let mut it = self.skip_db.raw_iterator_cf_opt(cf, ro);
                it.seek(key);
                self.skip_iter = Some(it);
            }
        }
        self.chunk_id = seek_chunk;
        self.load_current_chunk(min_rel)
    }

    fn load_current_chunk(&mut self, min_rel: u8) -> Result<()> {
        let Some(ref mut iter) = self.skip_iter else {
            self.current_block = None;
            return Ok(());
        };
        let prefix = encode_fts_term_id_prefix(self.term_id);
        loop {
            let Some(k) = iter.key() else {
                iter.status().map_err(map_db_err)?;
                self.current_block = None;
                self.chunk_layout = None;
                return Ok(());
            };
            if k.len() != 8 || k[0..4] != prefix {
                self.current_block = None;
                self.chunk_layout = None;
                return Ok(());
            }
            let cid = u32::from_be_bytes(k[4..8].try_into().unwrap());
            let Some(val) = iter.value() else {
                iter.next();
                continue;
            };
            let layout = SkipBlockLayout::parse(val)?;
            let view = layout.view(val);
            let idx = if min_rel == 0 || cid > self.chunk_id {
                Some(0)
            } else {
                view.find_ge(min_rel)
            };
            if let Some(i) = idx {
                self.chunk_id = cid;
                self.entry_idx = i;
                self.entry_count = layout.count;
                let rel = view.rel_block_id(i);
                let bounds = (view.max_tf(i), view.min_dl_over_tf(i));
                self.chunk_layout = Some(layout);
                self.current_block = Some(make_fts_block_id(cid, rel));
                self.current_bounds = bounds;
                return Ok(());
            }
            iter.next();
        }
    }

    #[inline]
    pub fn next_block(&mut self) -> Result<()> {
        if self.inline.is_some() {
            self.inline_done = true;
            self.current_block = None;
            return Ok(());
        }
        if self.current_block.is_none() {
            return Ok(());
        }
        let next_idx = self.entry_idx + 1;
        if next_idx < self.entry_count {
            let view = self.current_chunk_view().unwrap();
            let rel = view.rel_block_id(next_idx);
            let bounds = (view.max_tf(next_idx), view.min_dl_over_tf(next_idx));
            self.entry_idx = next_idx;
            self.current_block = Some(make_fts_block_id(self.chunk_id, rel));
            self.current_bounds = bounds;
            return Ok(());
        }
        if self.skip_pin.is_some() {
            return self.ensure_skip_iter_seek(self.chunk_id + 1, 0);
        }
        self.skip_iter.as_mut().unwrap().next();
        self.load_current_chunk(0)
    }

    #[inline]
    pub fn seek_block(&mut self, target: FtsBlockId) -> Result<()> {
        let Some(cur) = self.current_block else {
            return Ok(());
        };
        if cur >= target {
            return Ok(());
        }
        if self.inline.is_some() {
            self.inline_done = true;
            self.current_block = None;
            return Ok(());
        }
        let target_chunk = target >> 8;
        let target_rel = (target & 0xFF) as u8;
        if target_chunk == self.chunk_id {
            let view = self.current_chunk_view().unwrap();
            let rel_slice = &view.rel_block_ids[self.entry_idx..];
            let pos = rel_slice.partition_point(|&b| b < target_rel);
            if pos < rel_slice.len() {
                let i = self.entry_idx + pos;
                let rel = view.rel_block_id(i);
                let bounds = (view.max_tf(i), view.min_dl_over_tf(i));
                self.entry_idx = i;
                self.current_block = Some(make_fts_block_id(self.chunk_id, rel));
                self.current_bounds = bounds;
                return Ok(());
            }
            if self.skip_pin.is_some() {
                return self.ensure_skip_iter_seek(self.chunk_id + 1, 0);
            }
            self.skip_iter.as_mut().unwrap().next();
            return self.load_current_chunk(0);
        }

        if self.skip_pin.is_some() || target_chunk > self.chunk_id + 1 {
            return self.ensure_skip_iter_seek(target_chunk, target_rel);
        }
        self.skip_iter.as_mut().unwrap().next();
        self.chunk_id = target_chunk;
        self.load_current_chunk(target_rel)
    }
}

/// Cached per-attribute cursor over `cf_fts_doc_lens` document-length blocks.
pub struct RocksDocLenCursor {
    blocks: DocLenBlockTable,
}

impl RocksDocLenCursor {
    #[inline]
    pub fn seek_block_bytes(&mut self, block_id: FtsBlockId) -> Result<Option<&[u8]>> {
        Ok(self
            .blocks
            .get(block_id as usize)
            .and_then(|entry| entry.as_deref()))
    }
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
    dirty_doc_pos: HashMap<DocId, Option<(NodeId, LocalId)>>,
    dirty_fts_doc_map: HashMap<DocId, Option<FtsDocId>>,
    removed_fts_doc_ids: Vec<FtsDocId>,
    dirty_docs: HashMap<DocId, Option<Vec<u8>>>,
    dirty_attr_clusters: HashMap<Vec<u8>, RoaringBitmap>,
    dirty_attr_postings: HashMap<Vec<u8>, LeafPosting>,
    fts_meta_bytes: Option<Vec<u8>>,
    fts_meta_dirty: bool,
    fts_free_ids: Option<RoaringBitmap>,
    fts_free_ids_dirty: bool,
    dirty_fts_terms: HashMap<Vec<u8>, Option<Vec<u8>>>,
    dirty_fts_skips: HashMap<[u8; 8], Option<Vec<u8>>>,
    dirty_fts_postings: HashMap<[u8; 8], Option<Vec<u8>>>,
    dirty_fts_positions: HashMap<[u8; 8], Option<Vec<u8>>>,
    dirty_fts_doc_lens: HashMap<[u8; 6], Option<Vec<u8>>>,
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
            dirty_doc_pos: HashMap::new(),
            dirty_fts_doc_map: HashMap::new(),
            removed_fts_doc_ids: Vec::new(),
            dirty_docs: HashMap::new(),
            dirty_attr_clusters: HashMap::new(),
            dirty_attr_postings: HashMap::new(),
            fts_meta_bytes: None,
            fts_meta_dirty: false,
            fts_free_ids: None,
            fts_free_ids_dirty: false,
            dirty_fts_terms: HashMap::new(),
            dirty_fts_skips: HashMap::new(),
            dirty_fts_postings: HashMap::new(),
            dirty_fts_positions: HashMap::new(),
            dirty_fts_doc_lens: HashMap::new(),
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

    pub fn try_get_leaf_node(&mut self, leaf_id: NodeId) -> Result<Option<PackedLeafNode>> {
        if let Some(opt) = self.leaf_nodes.get(&leaf_id) {
            return Ok(opt.clone());
        }
        let Some(leaf) = self.storage.get_leaf_node(leaf_id)? else {
            return Ok(None);
        };
        self.leaf_nodes.insert(leaf_id, Some(leaf.clone()));
        Ok(Some(leaf))
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
        let pos_opt = match self.dirty_doc_pos.get(&doc_id) {
            Some(Some((leaf_id, local_id))) => Some((*leaf_id, *local_id)),
            Some(None) => return Ok(None),
            None => match self.storage.get_doc_loc(doc_id)? {
                Some(loc) => {
                    let fts = match self.dirty_fts_doc_map.get(&doc_id) {
                        Some(&f) => f,
                        None => loc.fts_doc_id,
                    };
                    return Ok(Some(DocLoc {
                        leaf_id: loc.leaf_id,
                        local_id: loc.local_id,
                        fts_doc_id: fts,
                    }));
                }
                None => None,
            },
        };
        let Some((leaf_id, local_id)) = pos_opt else {
            return Ok(None);
        };
        let fts_doc_id = self.get_fts_doc_id(doc_id)?;
        Ok(Some(DocLoc {
            leaf_id,
            local_id,
            fts_doc_id,
        }))
    }

    pub fn multi_get_doc_locs(&self, doc_ids: &[DocId]) -> Result<Vec<Option<DocLoc>>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![None; doc_ids.len()];
        let mut need_db: Vec<(DocId, usize)> = Vec::new();
        for (slot, &doc_id) in doc_ids.iter().enumerate() {
            match self.dirty_doc_pos.get(&doc_id) {
                Some(None) => {}
                Some(Some((leaf_id, local_id))) => {
                    if let Some(&fts_opt) = self.dirty_fts_doc_map.get(&doc_id) {
                        out[slot] = Some(DocLoc {
                            leaf_id: *leaf_id,
                            local_id: *local_id,
                            fts_doc_id: fts_opt,
                        });
                    } else {
                        need_db.push((doc_id, slot));
                    }
                }
                None => need_db.push((doc_id, slot)),
            }
        }
        if !need_db.is_empty() {
            let ids: Vec<DocId> = need_db.iter().map(|&(d, _)| d).collect();
            let db_locs = self.storage.multi_get_doc_locs(&ids)?;
            for ((doc_id, slot), db_opt) in need_db.into_iter().zip(db_locs) {
                let pos = match self.dirty_doc_pos.get(&doc_id) {
                    Some(Some(p)) => Some(*p),
                    Some(None) => None,
                    None => db_opt.map(|l| (l.leaf_id, l.local_id)),
                };
                if let Some((leaf_id, local_id)) = pos {
                    let fts_doc_id = match self.dirty_fts_doc_map.get(&doc_id) {
                        Some(&f) => f,
                        None => db_opt.and_then(|l| l.fts_doc_id),
                    };
                    out[slot] = Some(DocLoc {
                        leaf_id,
                        local_id,
                        fts_doc_id,
                    });
                }
            }
        }
        Ok(out)
    }

    pub fn put_doc_pos(&mut self, doc_id: DocId, leaf_id: NodeId, local_id: LocalId) {
        self.dirty_doc_pos.insert(doc_id, Some((leaf_id, local_id)));
    }

    pub fn delete_doc_loc(&mut self, doc_id: DocId) {
        self.dirty_doc_pos.insert(doc_id, None);
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

    /// Mutate the posting at `(cluster_key, cluster_id)` and automatically maintain
    /// `cf_attr_clusters[cluster_key]` on `empty <-> non-empty` transitions.
    pub fn mutate_attr_posting(
        &mut self,
        cluster_key: &[u8],
        cluster_id: NodeId,
        mutate: impl FnOnce(&mut LeafPosting),
    ) -> Result<()> {
        let posting_key = make_posting_key(cluster_key, cluster_id);
        if !self.dirty_attr_postings.contains_key(&posting_key) {
            let existing = self
                .storage
                .get_attr_posting(&posting_key)?
                .unwrap_or_default();
            self.dirty_attr_postings.insert(posting_key.clone(), existing);
        }
        let posting = self.dirty_attr_postings.get_mut(&posting_key).unwrap();
        let was_empty = posting.is_empty();
        mutate(posting);
        let is_empty = posting.is_empty();

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

    pub fn get_fts_free_ids(&mut self) -> Result<&RoaringBitmap> {
        if self.fts_free_ids.is_none() {
            self.fts_free_ids = Some(self.storage.get_fts_free_ids()?);
        }
        Ok(self.fts_free_ids.as_ref().unwrap())
    }

    pub fn get_fts_free_ids_mut(&mut self) -> Result<&mut RoaringBitmap> {
        if self.fts_free_ids.is_none() {
            self.fts_free_ids = Some(self.storage.get_fts_free_ids()?);
        }
        self.fts_free_ids_dirty = true;
        Ok(self.fts_free_ids.as_mut().unwrap())
    }

    /// Visit all live `cf_fts_skips` chunks for `term_id`, merging RocksDB state with `dirty_fts_skips`.
    pub fn visit_fts_skip_blocks(
        &self,
        term_id: TermId,
        mut f: impl FnMut(u32, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.storage.cf(CF_FTS_SKIPS)?;
        let prefix = encode_fts_term_id_prefix(term_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self
            .storage
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&prefix, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if k.len() != 8 || k[0..4] != prefix {
                break;
            }
            let key_arr: [u8; 8] = k.as_ref().try_into().unwrap();
            if !self.dirty_fts_skips.contains_key(&key_arr) {
                let chunk_id = u32::from_be_bytes(k[4..8].try_into().unwrap());
                if !f(chunk_id, &v)? {
                    return Ok(());
                }
            }
        }
        for (key, opt_bytes) in &self.dirty_fts_skips {
            if key[0..4] == prefix
                && let Some(bytes) = opt_bytes
            {
                let chunk_id = u32::from_be_bytes(key[4..8].try_into().unwrap());
                if !f(chunk_id, bytes)? {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    pub fn get_fts_doc_id(&self, doc_id: DocId) -> Result<Option<FtsDocId>> {
        if let Some(opt) = self.dirty_fts_doc_map.get(&doc_id) {
            return Ok(*opt);
        }
        self.storage.get_fts_doc_id(doc_id)
    }

    pub fn put_fts_doc_mapping(&mut self, doc_id: DocId, fts_doc_id: FtsDocId) {
        self.dirty_fts_doc_map.insert(doc_id, Some(fts_doc_id));
    }

    pub fn delete_fts_doc_mapping(&mut self, doc_id: DocId, fts_doc_id: FtsDocId) {
        self.dirty_fts_doc_map.insert(doc_id, None);
        self.removed_fts_doc_ids.push(fts_doc_id);
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

    pub fn get_fts_skip_bytes(&self, key: &[u8; 8]) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_skips.get(key) {
            return Ok(opt.clone());
        }
        self.storage.get_fts_skip_bytes(key)
    }

    pub fn put_fts_skip_bytes(&mut self, key: [u8; 8], val: Vec<u8>) {
        self.dirty_fts_skips.insert(key, Some(val));
    }

    pub fn delete_fts_skip(&mut self, key: [u8; 8]) {
        self.dirty_fts_skips.insert(key, None);
    }

    pub fn get_fts_posting_bytes(&self, key: &[u8; 8]) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_postings.get(key) {
            return Ok(opt.clone());
        }
        self.storage.get_fts_posting_bytes(key)
    }

    pub fn put_fts_posting_bytes(&mut self, key: [u8; 8], val: Vec<u8>) {
        self.dirty_fts_postings.insert(key, Some(val));
    }

    pub fn delete_fts_posting(&mut self, key: [u8; 8]) {
        self.dirty_fts_postings.insert(key, None);
    }

    pub fn get_fts_position_bytes(&self, key: &[u8; 8]) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_positions.get(key) {
            return Ok(opt.clone());
        }
        self.storage.get_fts_position_bytes(key)
    }

    pub fn put_fts_position_bytes(&mut self, key: [u8; 8], val: Vec<u8>) {
        self.dirty_fts_positions.insert(key, Some(val));
    }

    pub fn delete_fts_position(&mut self, key: [u8; 8]) {
        self.dirty_fts_positions.insert(key, None);
    }

    pub fn get_fts_doc_len_bytes(&self, key: &[u8; 6]) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_doc_lens.get(key) {
            return Ok(opt.clone());
        }
        self.storage.get_fts_doc_len_bytes(key)
    }

    pub fn put_fts_doc_len_bytes(&mut self, key: [u8; 6], val: Vec<u8>) {
        self.dirty_fts_doc_lens.insert(key, Some(val));
    }

    pub fn delete_fts_doc_len(&mut self, key: [u8; 6]) {
        self.dirty_fts_doc_lens.insert(key, None);
    }

    /// Recompute exact `(min_val, max_val)` for `(attr_id, type_tag, leaf_id)` from `cf_attr_postings`
    /// merged with `self.dirty_attr_postings` when a boundary value is removed.
    fn recompute_leaf_attr_zone(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        leaf_id: NodeId,
    ) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let cf = self.storage.cf(CF_ATTR_POSTINGS)?;
        let prefix = encode_posting_prefix(attr_id, type_tag, leaf_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);

        let mut min_val: Option<Vec<u8>> = None;
        let mut max_val: Option<Vec<u8>> = None;

        // Single forward prefix scan over `(attr_id, type_tag, leaf_id)`: bounded by leaf size.
        let mut iter = self.storage.db.raw_iterator_cf_opt(cf, ro);
        iter.seek(&prefix);
        while iter.valid() {
            let Some(k) = iter.key() else { break };
            if !k.starts_with(&prefix) {
                break;
            }
            let live = match self.dirty_attr_postings.get(k) {
                Some(p) => !p.is_empty(),
                None => true,
            };
            if live {
                let v = &k[7..];
                if min_val.is_none() {
                    min_val = Some(v.to_vec());
                }
                max_val = Some(v.to_vec());
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)?;

        // Merge with newly added live keys in dirty_attr_postings
        for (k, p) in &self.dirty_attr_postings {
            if !p.is_empty() && k.starts_with(&prefix) {
                let v = &k[7..];
                if min_val.as_ref().is_none_or(|m| v < m.as_slice()) {
                    min_val = Some(v.to_vec());
                }
                if max_val.as_ref().is_none_or(|m| v > m.as_slice()) {
                    max_val = Some(v.to_vec());
                }
            }
        }

        match (min_val, max_val) {
            (Some(lo), Some(hi)) => Ok(Some((lo, hi))),
            _ => Ok(None),
        }
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
        for (&doc_id, opt) in &self.dirty_vectors {
            match opt {
                Some(vec) => batch.put_vector(doc_id, vec),
                None => batch.delete_vector(doc_id),
            }
        }

        // Reconcile unified cf_doc_loc + block-packed cf_fts_rev_doc_map
        let mut doc_loc_ids: Vec<DocId> = Vec::with_capacity(
            self.dirty_doc_pos.len() + self.dirty_fts_doc_map.len(),
        );
        doc_loc_ids.extend(self.dirty_doc_pos.keys().copied());
        doc_loc_ids.extend(self.dirty_fts_doc_map.keys().copied());
        doc_loc_ids.sort_unstable();
        doc_loc_ids.dedup();

        let existing_locs: HashMap<DocId, Option<DocLoc>> = if !doc_loc_ids.is_empty() {
            let fetched = self.storage.multi_get_doc_locs(&doc_loc_ids)?;
            doc_loc_ids.iter().copied().zip(fetched).collect()
        } else {
            HashMap::new()
        };

        let mut dirty_rev_blocks: HashMap<FtsBlockId, FtsRevBlock> = HashMap::new();
        for &fid in &self.removed_fts_doc_ids {
            let bid = fts_block_id(fid);
            if !dirty_rev_blocks.contains_key(&bid) {
                let blk = self.storage.get_fts_rev_block(bid)?.unwrap_or_default();
                dirty_rev_blocks.insert(bid, blk);
            }
            dirty_rev_blocks.get_mut(&bid).unwrap().remove(fts_slot(fid));
        }

        for doc_id in doc_loc_ids {
            let old_loc = existing_locs.get(&doc_id).and_then(|o| *o);
            let new_pos = match self.dirty_doc_pos.get(&doc_id) {
                Some(opt) => *opt,
                None => old_loc.map(|l| (l.leaf_id, l.local_id)),
            };
            match new_pos {
                None => {
                    batch.delete_doc_loc(doc_id);
                }
                Some((leaf_id, local_id)) => {
                    let fts_doc_id = match self.dirty_fts_doc_map.get(&doc_id) {
                        Some(&opt) => opt,
                        None => old_loc.and_then(|l| l.fts_doc_id),
                    };
                    let loc = DocLoc {
                        leaf_id,
                        local_id,
                        fts_doc_id,
                    };
                    batch.put_doc_loc(doc_id, loc);
                    if let Some(fid) = fts_doc_id {
                        let bid = fts_block_id(fid);
                        if !dirty_rev_blocks.contains_key(&bid) {
                            let blk = self.storage.get_fts_rev_block(bid)?.unwrap_or_default();
                            dirty_rev_blocks.insert(bid, blk);
                        }
                        dirty_rev_blocks
                            .get_mut(&bid)
                            .unwrap()
                            .set(fts_slot(fid), doc_id, leaf_id, local_id);
                    }
                }
            }
        }

        for (&bid, blk) in &dirty_rev_blocks {
            if blk.is_empty() {
                batch.delete_fts_rev_block(bid);
            } else {
                batch.put_fts_rev_block(bid, &blk.encode());
            }
        }

        for (&doc_id, opt) in &self.dirty_docs {
            match opt {
                Some(bytes) => batch.put_doc_bytes(doc_id, bytes),
                None => batch.delete_doc(doc_id),
            }
        }
        for (key, bm) in &self.dirty_attr_clusters {
            if bm.is_empty() {
                batch.delete_attr_cluster(key);
            } else {
                let bytes = encode_roaring_bitmap(bm)?;
                batch.put_attr_cluster(key, &bytes);
            }
        }

        // Maintain per-leaf AttrZoneMap atomically inside the same WriteBatch
        let mut updated_zone_attrs: HashMap<(AttrId, u8), AttrZoneMap> = HashMap::new();
        if !self.dirty_attr_postings.is_empty() {
            let zones_arc = self.storage.get_attr_zones_cached()?;
            let mut need_recompute: HashSet<(AttrId, u8, NodeId)> = HashSet::new();

            for (key, posting) in &self.dirty_attr_postings {
                let (attr_id, type_tag, leaf_id, val_bytes) = split_posting_key(key)?;
                if type_tag == crate::codec::TAG_PRESENT || type_tag == crate::codec::TAG_BOOL {
                    continue;
                }
                let attr_key = (attr_id, type_tag);
                if !posting.is_empty() {
                    let existing_bounds = updated_zone_attrs
                        .get(&attr_key)
                        .and_then(|zm| zm.get(&leaf_id))
                        .or_else(|| zones_arc.get(&attr_key).and_then(|zm| zm.get(&leaf_id)));
                    let needs_update = match existing_bounds {
                        Some((min_b, max_b)) => {
                            val_bytes < min_b.as_slice() || val_bytes > max_b.as_slice()
                        }
                        None => true,
                    };
                    if needs_update {
                        let zmap = updated_zone_attrs.entry(attr_key).or_insert_with(|| {
                            zones_arc.get(&attr_key).cloned().unwrap_or_default()
                        });
                        match zmap.get_mut(&leaf_id) {
                            Some((min_b, max_b)) => {
                                if val_bytes < min_b.as_slice() {
                                    *min_b = val_bytes.to_vec();
                                }
                                if val_bytes > max_b.as_slice() {
                                    *max_b = val_bytes.to_vec();
                                }
                            }
                            None => {
                                zmap.insert(leaf_id, (val_bytes.to_vec(), val_bytes.to_vec()));
                            }
                        }
                    }
                } else {
                    let existing_bounds = updated_zone_attrs
                        .get(&attr_key)
                        .and_then(|zm| zm.get(&leaf_id))
                        .or_else(|| zones_arc.get(&attr_key).and_then(|zm| zm.get(&leaf_id)));
                    if let Some((min_b, max_b)) = existing_bounds
                        && (val_bytes == min_b.as_slice() || val_bytes == max_b.as_slice())
                    {
                        need_recompute.insert((attr_id, type_tag, leaf_id));
                    }
                }
            }

            for (attr_id, type_tag, leaf_id) in need_recompute {
                let attr_key = (attr_id, type_tag);
                let recomputed = self.recompute_leaf_attr_zone(attr_id, type_tag, leaf_id)?;
                let zmap = updated_zone_attrs
                    .entry(attr_key)
                    .or_insert_with(|| zones_arc.get(&attr_key).cloned().unwrap_or_default());
                match recomputed {
                    Some(bounds) => {
                        zmap.insert(leaf_id, bounds);
                    }
                    None => {
                        zmap.remove(&leaf_id);
                    }
                }
            }

            for (&(attr_id, type_tag), zmap) in &updated_zone_attrs {
                let zkey = encode_attr_prefix(attr_id, type_tag);
                if zmap.is_empty() {
                    batch.delete_attr_zone(&zkey);
                } else {
                    batch.put_attr_zone(&zkey, &encode_zone_map(zmap));
                }
            }

            for (key, posting) in self.dirty_attr_postings {
                if posting.is_empty() {
                    batch.delete_attr_posting(&key);
                } else {
                    let bytes = posting.encode()?;
                    batch.put_attr_posting(&key, &bytes);
                }
            }
        }

        if self.fts_meta_dirty
            && let Some(ref bytes) = self.fts_meta_bytes
        {
            batch.put_fts_meta_bytes(bytes);
        }
        if self.fts_free_ids_dirty
            && let Some(ref bm) = self.fts_free_ids
        {
            if bm.is_empty() {
                batch.delete_fts_free_ids();
            } else {
                let bytes = encode_roaring_bitmap(bm)?;
                batch.put_fts_free_ids(&bytes);
            }
        }
        for (key, opt) in self.dirty_fts_terms {
            match opt {
                Some(bytes) => batch.put_fts_term(&key, &bytes),
                None => batch.delete_fts_term(&key),
            }
        }
        for (key, opt) in self.dirty_fts_skips {
            match opt {
                Some(bytes) => batch.put_fts_skip(&key, &bytes),
                None => batch.delete_fts_skip(&key),
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
        for (key, opt) in &self.dirty_fts_doc_lens {
            match opt {
                Some(bytes) => batch.put_fts_doc_len(key, bytes),
                None => batch.delete_fts_doc_len(key),
            }
        }

        self.storage.write_batch(batch)?;

        // Write-through cache updates after RocksDB commit succeeds.
        if !updated_zone_attrs.is_empty() {
            let mut zguard = self
                .storage
                .attr_zones_cache
                .write()
                .expect("attr_zones_cache poisoned");
            if let Some(ref mut table_arc) = *zguard {
                let map = Arc::make_mut(table_arc);
                for (k, zmap) in updated_zone_attrs {
                    if zmap.is_empty() {
                        map.remove(&k);
                    } else {
                        map.insert(k, zmap);
                    }
                }
            }
        }

        if !dirty_rev_blocks.is_empty() {
            let mut rguard = self
                .storage
                .fts_rev_map_cache
                .write()
                .expect("fts_rev_map_cache poisoned");
            if let Some(ref mut table_arc) = *rguard {
                let blocks = Arc::make_mut(table_arc);
                for (bid, blk) in dirty_rev_blocks {
                    let b_idx = bid as usize;
                    if blk.is_empty() {
                        if b_idx < blocks.len() {
                            blocks[b_idx] = None;
                        }
                    } else {
                        if blocks.len() <= b_idx {
                            blocks.resize_with(b_idx + 1, || None);
                        }
                        blocks[b_idx] = Some(Arc::new(blk));
                    }
                }
                while blocks.last().is_some_and(Option::is_none) {
                    blocks.pop();
                }
            }
        }

        if !self.dirty_fts_doc_lens.is_empty() {
            let mut cache = self
                .storage
                .fts_doc_lens_cache
                .write()
                .expect("fts_doc_lens_cache poisoned");
            let mut touched_attrs: HashSet<u16> = HashSet::new();
            for (key, opt) in self.dirty_fts_doc_lens {
                let attr_id = u16::from_be_bytes([key[0], key[1]]);
                if let Some(table) = cache.get_mut(&attr_id) {
                    touched_attrs.insert(attr_id);
                    let blocks = Arc::make_mut(table);
                    let block_id = u32::from_be_bytes([key[2], key[3], key[4], key[5]]) as usize;
                    match opt {
                        Some(bytes) => {
                            if blocks.len() <= block_id {
                                blocks.resize_with(block_id + 1, || None);
                            }
                            blocks[block_id] = Some(bytes.into_boxed_slice());
                        }
                        None => {
                            if block_id < blocks.len() {
                                blocks[block_id] = None;
                            }
                        }
                    }
                }
            }
            for attr_id in touched_attrs {
                if let Some(table) = cache.get_mut(&attr_id)
                    && table.last().is_some_and(Option::is_none)
                {
                    let blocks = Arc::make_mut(table);
                    while blocks.last().is_some_and(Option::is_none) {
                        blocks.pop();
                    }
                    if blocks.is_empty() {
                        cache.remove(&attr_id);
                    }
                }
            }
        }

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
