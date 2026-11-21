use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use rocksdb::{
    BlockBasedOptions, ColumnFamily, ColumnFamilyDescriptor, DB, Direction, IteratorMode, Options,
    ReadOptions, SliceTransform, WriteBatch, DEFAULT_COLUMN_FAMILY_NAME,
};

use crate::config::{DocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::node::{
    borrow_f32_slice, decode_f32_slice, encode_f32_slice, InternalEntry, NodeHeader, TreeMeta,
};

pub const CF_META: &str = "cf_meta";
pub const CF_NODE_HEADER: &str = "cf_node_header";
pub const CF_INTERNAL_ENTRIES: &str = "cf_internal_entries";
pub const CF_LEAF_ENTRIES: &str = "cf_leaf_entries";
pub const CF_DOC_LOC: &str = "cf_doc_loc";

const META_KEY: &[u8] = b"tree_meta";

/// RocksDB storage wrapper managing column families, prefix scans, and atomic batches.
pub struct Storage {
    db: DB,
    dim: usize,
}

impl Storage {
    /// Open or initialize the RocksDB instance and validate/create `TreeMeta`.
    pub fn open(path: impl AsRef<Path>, config: &TreeConfig) -> Result<(Self, TreeMeta)> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);

        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(DEFAULT_COLUMN_FAMILY_NAME, Options::default()),
            ColumnFamilyDescriptor::new(CF_META, Options::default()),
            ColumnFamilyDescriptor::new(CF_NODE_HEADER, point_lookup_cf_opts()),
            ColumnFamilyDescriptor::new(CF_INTERNAL_ENTRIES, prefix_scan_cf_opts()),
            ColumnFamilyDescriptor::new(CF_LEAF_ENTRIES, prefix_scan_cf_opts()),
            ColumnFamilyDescriptor::new(CF_DOC_LOC, point_lookup_cf_opts()),
        ];

        let db = DB::open_cf_descriptors(&db_opts, path, cf_descriptors)?;
        let storage = Self {
            db,
            dim: config.dim,
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
                };
                let init_root = NodeHeader {
                    level: 0,
                    parent_id: None,
                    entry_count: 0,
                    subtree_count: 0,
                    centroid: vec![0.0; config.dim],
                };
                let mut batch = storage.new_batch();
                batch.put_meta(&init_meta);
                batch.put_node_header(1, &init_root);
                storage.write_batch(batch)?;
                init_meta
            }
        };

        Ok((storage, meta))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn new_batch(&self) -> StorageBatch<'_> {
        StorageBatch {
            storage: self,
            batch: WriteBatch::default(),
        }
    }

    pub fn write_batch(&self, batch: StorageBatch<'_>) -> Result<()> {
        self.db.write(batch.batch)?;
        Ok(())
    }

    pub fn get_meta(&self) -> Result<Option<TreeMeta>> {
        let cf = self.cf(CF_META)?;
        match self.db.get_cf(cf, META_KEY)? {
            Some(bytes) => Ok(Some(TreeMeta::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn get_node_header(&self, node_id: NodeId) -> Result<Option<NodeHeader>> {
        let cf = self.cf(CF_NODE_HEADER)?;
        match self.db.get_cf(cf, node_key(node_id))? {
            Some(bytes) => Ok(Some(NodeHeader::decode(&bytes, self.dim)?)),
            None => Ok(None),
        }
    }

    pub fn get_doc_loc(&self, doc_id: DocId) -> Result<Option<NodeId>> {
        let cf = self.cf(CF_DOC_LOC)?;
        match self.db.get_cf(cf, node_key(doc_id))? {
            Some(bytes) => Ok(Some(decode_u64_key(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn get_internal_entry(
        &self,
        node_id: NodeId,
        child_id: NodeId,
    ) -> Result<Option<InternalEntry>> {
        let cf = self.cf(CF_INTERNAL_ENTRIES)?;
        match self.db.get_cf(cf, entry_key(node_id, child_id))? {
            Some(bytes) => Ok(Some(InternalEntry::decode(&bytes, self.dim)?)),
            None => Ok(None),
        }
    }

    pub fn scan_internal_entries(&self, node_id: NodeId) -> Result<Vec<(NodeId, InternalEntry)>> {
        let cf = self.cf(CF_INTERNAL_ENTRIES)?;
        let prefix = node_key(node_id);
        let mut read_opts = ReadOptions::default();
        read_opts.set_prefix_same_as_start(true);

        let mut out = Vec::new();
        let iter = self.db.iterator_cf_opt(
            cf,
            read_opts,
            IteratorMode::From(&prefix, Direction::Forward),
        );
        for item in iter {
            let (key, val) = item?;
            if !key.starts_with(&prefix) {
                break;
            }
            let (_, child_id) = decode_entry_key(&key)?;
            out.push((child_id, InternalEntry::decode(&val, self.dim)?));
        }
        Ok(out)
    }

    pub fn scan_leaf_entries(&self, leaf_id: NodeId) -> Result<Vec<(DocId, Vec<f32>)>> {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        let prefix = node_key(leaf_id);
        let mut read_opts = ReadOptions::default();
        read_opts.set_prefix_same_as_start(true);

        let mut out = Vec::new();
        let iter = self.db.iterator_cf_opt(
            cf,
            read_opts,
            IteratorMode::From(&prefix, Direction::Forward),
        );
        for item in iter {
            let (key, val) = item?;
            if !key.starts_with(&prefix) {
                break;
            }
            let (_, doc_id) = decode_entry_key(&key)?;
            out.push((doc_id, decode_f32_slice(&val, self.dim)?));
        }
        Ok(out)
    }

    /// Scan all vectors in `leaf_id`, invoking `f(doc_id, vector_slice)` with minimal copying.
    pub fn scan_leaf_with<F>(&self, leaf_id: NodeId, mut f: F) -> Result<()>
    where
        F: FnMut(DocId, &[f32]),
    {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        let prefix = node_key(leaf_id);
        let mut read_opts = ReadOptions::default();
        read_opts.set_prefix_same_as_start(true);

        let iter = self.db.iterator_cf_opt(
            cf,
            read_opts,
            IteratorMode::From(&prefix, Direction::Forward),
        );
        for item in iter {
            let (key, val) = item?;
            if !key.starts_with(&prefix) {
                break;
            }
            let (_, doc_id) = decode_entry_key(&key)?;
            let vec = borrow_f32_slice(&val, self.dim)?;
            f(doc_id, &vec);
        }
        Ok(())
    }

    /// Scan all internal entries of `node_id`, invoking `f(child_id, subtree_count, centroid)`.
    pub fn scan_internal_with<F>(&self, node_id: NodeId, mut f: F) -> Result<()>
    where
        F: FnMut(NodeId, u64, &[f32]),
    {
        let cf = self.cf(CF_INTERNAL_ENTRIES)?;
        let prefix = node_key(node_id);
        let mut read_opts = ReadOptions::default();
        read_opts.set_prefix_same_as_start(true);

        let iter = self.db.iterator_cf_opt(
            cf,
            read_opts,
            IteratorMode::From(&prefix, Direction::Forward),
        );
        for item in iter {
            let (key, val) = item?;
            if !key.starts_with(&prefix) {
                break;
            }
            let (_, child_id) = decode_entry_key(&key)?;
            if val.len() != 8 + self.dim * 4 {
                return Err(TreeError::CorruptedData(format!(
                    "InternalEntry expected {} bytes, got {}",
                    8 + self.dim * 4,
                    val.len()
                )));
            }
            let subtree_count = u64::from_le_bytes(val[0..8].try_into().unwrap());
            let centroid = borrow_f32_slice(&val[8..], self.dim)?;
            f(child_id, subtree_count, &centroid);
        }
        Ok(())
    }

    pub fn all_node_headers(&self) -> Result<Vec<(NodeId, NodeHeader)>> {
        let cf = self.cf(CF_NODE_HEADER)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item?;
            out.push((decode_u64_key(&k)?, NodeHeader::decode(&v, self.dim)?));
        }
        Ok(out)
    }

    pub fn all_doc_locs(&self) -> Result<Vec<(DocId, NodeId)>> {
        let cf = self.cf(CF_DOC_LOC)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item?;
            out.push((decode_u64_key(&k)?, decode_u64_key(&v)?));
        }
        Ok(out)
    }

    pub fn all_internal_entry_keys(&self) -> Result<Vec<(NodeId, NodeId)>> {
        let cf = self.cf(CF_INTERNAL_ENTRIES)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, _) = item?;
            out.push(decode_entry_key(&k)?);
        }
        Ok(out)
    }

    pub fn all_leaf_entry_keys(&self) -> Result<Vec<(NodeId, DocId)>> {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, _) = item?;
            out.push(decode_entry_key(&k)?);
        }
        Ok(out)
    }

    fn cf(&self, name: &str) -> Result<&ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| TreeError::CorruptedData(format!("missing column family {name}")))
    }
}

/// Typed wrapper around `rocksdb::WriteBatch` for atomic tree mutations.
pub struct StorageBatch<'a> {
    storage: &'a Storage,
    batch: WriteBatch,
}

impl<'a> StorageBatch<'a> {
    pub fn put_meta(&mut self, meta: &TreeMeta) {
        let cf = self.storage.db.cf_handle(CF_META).unwrap();
        self.batch.put_cf(cf, META_KEY, meta.encode());
    }

    pub fn put_node_header(&mut self, node_id: NodeId, header: &NodeHeader) {
        let cf = self.storage.db.cf_handle(CF_NODE_HEADER).unwrap();
        self.batch.put_cf(cf, node_key(node_id), header.encode());
    }

    pub fn delete_node_header(&mut self, node_id: NodeId) {
        let cf = self.storage.db.cf_handle(CF_NODE_HEADER).unwrap();
        self.batch.delete_cf(cf, node_key(node_id));
    }

    pub fn put_internal_entry(&mut self, node_id: NodeId, child_id: NodeId, entry: &InternalEntry) {
        let cf = self.storage.db.cf_handle(CF_INTERNAL_ENTRIES).unwrap();
        self.batch
            .put_cf(cf, entry_key(node_id, child_id), entry.encode());
    }

    pub fn delete_internal_entry(&mut self, node_id: NodeId, child_id: NodeId) {
        let cf = self.storage.db.cf_handle(CF_INTERNAL_ENTRIES).unwrap();
        self.batch.delete_cf(cf, entry_key(node_id, child_id));
    }

    pub fn put_leaf_entry(&mut self, leaf_id: NodeId, doc_id: DocId, vec: &[f32]) {
        let cf = self.storage.db.cf_handle(CF_LEAF_ENTRIES).unwrap();
        self.batch
            .put_cf(cf, entry_key(leaf_id, doc_id), encode_f32_slice(vec));
    }

    pub fn delete_leaf_entry(&mut self, leaf_id: NodeId, doc_id: DocId) {
        let cf = self.storage.db.cf_handle(CF_LEAF_ENTRIES).unwrap();
        self.batch.delete_cf(cf, entry_key(leaf_id, doc_id));
    }

    pub fn put_doc_loc(&mut self, doc_id: DocId, leaf_id: NodeId) {
        let cf = self.storage.db.cf_handle(CF_DOC_LOC).unwrap();
        self.batch.put_cf(cf, node_key(doc_id), node_key(leaf_id));
    }

    pub fn delete_doc_loc(&mut self, doc_id: DocId) {
        let cf = self.storage.db.cf_handle(CF_DOC_LOC).unwrap();
        self.batch.delete_cf(cf, node_key(doc_id));
    }
}

#[inline]
fn node_key(id: u64) -> [u8; 8] {
    id.to_be_bytes()
}

#[inline]
fn entry_key(prefix_id: u64, item_id: u64) -> [u8; 16] {
    let mut buf = [0u8; 16];
    buf[0..8].copy_from_slice(&prefix_id.to_be_bytes());
    buf[8..16].copy_from_slice(&item_id.to_be_bytes());
    buf
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

#[inline]
fn decode_entry_key(bytes: &[u8]) -> Result<(u64, u64)> {
    if bytes.len() != 16 {
        return Err(TreeError::CorruptedData(format!(
            "expected 16-byte composite key, got {}",
            bytes.len()
        )));
    }
    let a = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
    let b = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
    Ok((a, b))
}

fn prefix_scan_cf_opts() -> Options {
    let mut opts = Options::default();
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
    opts.set_memtable_prefix_bloom_ratio(0.2);

    let mut bbo = BlockBasedOptions::default();
    bbo.set_bloom_filter(10.0, false);
    bbo.set_whole_key_filtering(false);
    opts.set_block_based_table_factory(&bbo);
    opts
}

fn point_lookup_cf_opts() -> Options {
    let mut opts = Options::default();
    let mut bbo = BlockBasedOptions::default();
    bbo.set_bloom_filter(10.0, false);
    opts.set_block_based_table_factory(&bbo);
    opts
}

/// Read-your-own-writes transaction overlay that commits all mutations in one atomic `WriteBatch`.
pub struct TxContext<'a> {
    storage: &'a Storage,
    meta: TreeMeta,
    meta_dirty: bool,
    headers: HashMap<NodeId, Option<NodeHeader>>,
    internal_entries: HashMap<NodeId, BTreeMap<NodeId, InternalEntry>>,
    dirty_internal: HashMap<(NodeId, NodeId), Option<InternalEntry>>,
    leaf_entries: HashMap<NodeId, BTreeMap<DocId, Vec<f32>>>,
    dirty_leaf: HashMap<(NodeId, DocId), Option<Vec<f32>>>,
    dirty_doc_loc: HashMap<DocId, Option<NodeId>>,
}

impl<'a> TxContext<'a> {
    pub fn new(storage: &'a Storage, meta: TreeMeta) -> Self {
        Self {
            storage,
            meta,
            meta_dirty: false,
            headers: HashMap::new(),
            internal_entries: HashMap::new(),
            dirty_internal: HashMap::new(),
            leaf_entries: HashMap::new(),
            dirty_leaf: HashMap::new(),
            dirty_doc_loc: HashMap::new(),
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
        self.meta.next_node_id += 1;
        self.meta_dirty = true;
        id
    }

    pub fn get_header(&mut self, node_id: NodeId) -> Result<NodeHeader> {
        if let Some(opt) = self.headers.get(&node_id) {
            return opt.clone().ok_or_else(|| {
                TreeError::CorruptedData(format!("node {node_id} was deleted in transaction"))
            });
        }
        let header = self
            .storage
            .get_node_header(node_id)?
            .ok_or_else(|| TreeError::CorruptedData(format!("missing NodeHeader for {node_id}")))?;
        self.headers.insert(node_id, Some(header.clone()));
        Ok(header)
    }

    pub fn put_header(&mut self, node_id: NodeId, header: NodeHeader) {
        self.headers.insert(node_id, Some(header));
    }

    pub fn delete_header(&mut self, node_id: NodeId) {
        self.headers.insert(node_id, None);
    }

    pub fn get_internal_entries(
        &mut self,
        node_id: NodeId,
    ) -> Result<BTreeMap<NodeId, InternalEntry>> {
        if !self.internal_entries.contains_key(&node_id) {
            let mut map: BTreeMap<NodeId, InternalEntry> = self
                .storage
                .scan_internal_entries(node_id)?
                .into_iter()
                .collect();
            for (&(nid, cid), val) in &self.dirty_internal {
                if nid == node_id {
                    match val {
                        Some(entry) => {
                            map.insert(cid, entry.clone());
                        }
                        None => {
                            map.remove(&cid);
                        }
                    }
                }
            }
            self.internal_entries.insert(node_id, map);
        }
        Ok(self.internal_entries.get(&node_id).unwrap().clone())
    }

    pub fn put_internal_entry(&mut self, node_id: NodeId, child_id: NodeId, entry: InternalEntry) {
        if let Some(map) = self.internal_entries.get_mut(&node_id) {
            map.insert(child_id, entry.clone());
        }
        self.dirty_internal.insert((node_id, child_id), Some(entry));
    }

    pub fn delete_internal_entry(&mut self, node_id: NodeId, child_id: NodeId) {
        if let Some(map) = self.internal_entries.get_mut(&node_id) {
            map.remove(&child_id);
        }
        self.dirty_internal.insert((node_id, child_id), None);
    }

    pub fn get_leaf_entries(&mut self, leaf_id: NodeId) -> Result<BTreeMap<DocId, Vec<f32>>> {
        if !self.leaf_entries.contains_key(&leaf_id) {
            let mut map: BTreeMap<DocId, Vec<f32>> = self
                .storage
                .scan_leaf_entries(leaf_id)?
                .into_iter()
                .collect();
            for (&(lid, did), val) in &self.dirty_leaf {
                if lid == leaf_id {
                    match val {
                        Some(vec) => {
                            map.insert(did, vec.clone());
                        }
                        None => {
                            map.remove(&did);
                        }
                    }
                }
            }
            self.leaf_entries.insert(leaf_id, map);
        }
        Ok(self.leaf_entries.get(&leaf_id).unwrap().clone())
    }

    pub fn put_leaf_entry(&mut self, leaf_id: NodeId, doc_id: DocId, vec: Vec<f32>) {
        if let Some(map) = self.leaf_entries.get_mut(&leaf_id) {
            map.insert(doc_id, vec.clone());
        }
        self.dirty_leaf.insert((leaf_id, doc_id), Some(vec));
    }

    pub fn delete_leaf_entry(&mut self, leaf_id: NodeId, doc_id: DocId) {
        if let Some(map) = self.leaf_entries.get_mut(&leaf_id) {
            map.remove(&doc_id);
        }
        self.dirty_leaf.insert((leaf_id, doc_id), None);
    }

    pub fn put_doc_loc(&mut self, doc_id: DocId, leaf_id: NodeId) {
        self.dirty_doc_loc.insert(doc_id, Some(leaf_id));
    }

    pub fn delete_doc_loc(&mut self, doc_id: DocId) {
        self.dirty_doc_loc.insert(doc_id, None);
    }

    pub fn commit(self) -> Result<TreeMeta> {
        let mut batch = self.storage.new_batch();
        if self.meta_dirty {
            batch.put_meta(&self.meta);
        }
        for (node_id, opt) in self.headers {
            match opt {
                Some(header) => batch.put_node_header(node_id, &header),
                None => batch.delete_node_header(node_id),
            }
        }
        for ((node_id, child_id), opt) in self.dirty_internal {
            match opt {
                Some(entry) => batch.put_internal_entry(node_id, child_id, &entry),
                None => batch.delete_internal_entry(node_id, child_id),
            }
        }
        for ((leaf_id, doc_id), opt) in self.dirty_leaf {
            match opt {
                Some(vec) => batch.put_leaf_entry(leaf_id, doc_id, &vec),
                None => batch.delete_leaf_entry(leaf_id, doc_id),
            }
        }
        for (doc_id, opt) in self.dirty_doc_loc {
            match opt {
                Some(leaf_id) => batch.put_doc_loc(doc_id, leaf_id),
                None => batch.delete_doc_loc(doc_id),
            }
        }
        self.storage.write_batch(batch)?;
        Ok(self.meta)
    }
}
