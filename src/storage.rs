use std::collections::{HashMap, HashSet};
use std::path::Path;

use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamily, ColumnFamilyDescriptor, DBCompressionType,
    DBPinnableSlice, IteratorMode, Options, WriteBatch, DB, DEFAULT_COLUMN_FAMILY_NAME,
};

use crate::config::{DocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::node::{
    LeafVectorsView, PackedInternalNode, PackedLeafNode, PackedLeafVectors, TreeMeta,
    decode_f32_slice,
};

pub const CF_META: &str = "cf_meta";
pub const CF_INTERNAL_NODES: &str = "cf_internal_nodes";
pub const CF_LEAF_ENTRIES: &str = "cf_leaf_entries";
pub const CF_LEAF_VECTORS: &str = "cf_leaf_vectors";
pub const CF_DOC_LOC: &str = "cf_doc_loc";

const META_KEY: &[u8] = b"tree_meta";
const BLOCK_CACHE_BYTES: usize = 128 * 1024 * 1024;

/// RocksDB storage wrapper managing packed node column families, block cache, and atomic batches.
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
            ColumnFamilyDescriptor::new(CF_LEAF_VECTORS, point_lookup_cf_opts(&block_cache, true)),
            ColumnFamilyDescriptor::new(CF_DOC_LOC, point_lookup_cf_opts(&block_cache, false)),
        ];

        let db = DB::open_cf_descriptors(&db_opts, path, cf_descriptors)?;
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
                let init_vecs = PackedLeafVectors::empty();
                let mut batch = storage.new_batch();
                batch.put_meta(&init_meta);
                batch.put_leaf_node(1, &init_leaf);
                batch.put_leaf_vectors(1, &init_vecs);
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

    pub fn get_doc_loc(&self, doc_id: DocId) -> Result<Option<NodeId>> {
        let cf = self.cf(CF_DOC_LOC)?;
        match self.db.get_pinned_cf(cf, node_key(doc_id))? {
            Some(bytes) => Ok(Some(decode_u64_key(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn get_vector(&self, doc_id: DocId) -> Result<Option<Vec<f32>>> {
        let Some(leaf_id) = self.get_doc_loc(doc_id)? else {
            return Ok(None);
        };
        let cf = self.cf(CF_LEAF_VECTORS)?;
        let Some(slice) = self.db.get_pinned_cf(cf, node_key(leaf_id))? else {
            return Err(TreeError::CorruptedData(format!(
                "missing cf_leaf_vectors for leaf {leaf_id}"
            )));
        };
        let view = LeafVectorsView::parse(&slice, self.dim)?;
        match view.find_vec_bytes(doc_id, self.dim) {
            Some(bytes) => Ok(Some(decode_f32_slice(bytes, self.dim)?)),
            None => Err(TreeError::CorruptedData(format!(
                "doc {doc_id} missing from leaf {leaf_id} vector blob"
            ))),
        }
    }

    pub fn get_internal_node(&self, node_id: NodeId) -> Result<Option<PackedInternalNode>> {
        let cf = self.cf(CF_INTERNAL_NODES)?;
        match self.db.get_pinned_cf(cf, node_key(node_id))? {
            Some(bytes) => Ok(Some(PackedInternalNode::decode(&bytes, self.dim)?)),
            None => Ok(None),
        }
    }

    pub fn get_leaf_node(&self, leaf_id: NodeId) -> Result<Option<PackedLeafNode>> {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        match self.db.get_pinned_cf(cf, node_key(leaf_id))? {
            Some(bytes) => Ok(Some(PackedLeafNode::decode(
                &bytes,
                self.dim,
                self.padded_dim,
                self.rabitq_words,
            )?)),
            None => Ok(None),
        }
    }

    pub fn get_leaf_vectors(&self, leaf_id: NodeId) -> Result<Option<PackedLeafVectors>> {
        let cf = self.cf(CF_LEAF_VECTORS)?;
        match self.db.get_pinned_cf(cf, node_key(leaf_id))? {
            Some(bytes) => Ok(Some(PackedLeafVectors::decode(&bytes, self.dim)?)),
            None => Ok(None),
        }
    }

    /// Zero-copy batched lookup of quantized leaf blobs, assuming `sorted_leaf_ids` is sorted ascending.
    pub fn multi_get_leaf_nodes_pinned(
        &self,
        sorted_leaf_ids: &[NodeId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        self.batched_multi_get_pinned(CF_LEAF_ENTRIES, sorted_leaf_ids, true)
    }

    /// Zero-copy batched lookup of full-vector leaf blobs, assuming `sorted_leaf_ids` is sorted ascending.
    pub fn multi_get_leaf_vectors_pinned(
        &self,
        sorted_leaf_ids: &[NodeId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        self.batched_multi_get_pinned(CF_LEAF_VECTORS, sorted_leaf_ids, true)
    }

    fn batched_multi_get_pinned(
        &self,
        cf_name: &str,
        leaf_ids: &[NodeId],
        sorted_input: bool,
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if leaf_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(cf_name)?;
        let keys: Vec<[u8; 8]> = leaf_ids.iter().map(|&id| node_key(id)).collect();
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted_input);
        let mut out = Vec::with_capacity(leaf_ids.len());
        for res in results {
            out.push(res?);
        }
        Ok(out)
    }

    pub fn all_internal_nodes(&self) -> Result<Vec<(NodeId, PackedInternalNode)>> {
        let cf = self.cf(CF_INTERNAL_NODES)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item?;
            out.push((
                decode_u64_key(&k)?,
                PackedInternalNode::decode(&v, self.dim)?,
            ));
        }
        Ok(out)
    }

    pub fn all_leaf_nodes(&self) -> Result<Vec<(NodeId, PackedLeafNode)>> {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item?;
            out.push((
                decode_u64_key(&k)?,
                PackedLeafNode::decode(&v, self.dim, self.padded_dim, self.rabitq_words)?,
            ));
        }
        Ok(out)
    }

    pub fn all_leaf_vectors(&self) -> Result<Vec<(NodeId, PackedLeafVectors)>> {
        let cf = self.cf(CF_LEAF_VECTORS)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item?;
            out.push((
                decode_u64_key(&k)?,
                PackedLeafVectors::decode(&v, self.dim)?,
            ));
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

    pub fn put_leaf_vectors(&mut self, leaf_id: NodeId, vecs: &PackedLeafVectors) {
        let cf = self.storage.db.cf_handle(CF_LEAF_VECTORS).unwrap();
        self.batch
            .put_cf(cf, node_key(leaf_id), vecs.encode(self.storage.dim));
    }

    pub fn delete_leaf_vectors(&mut self, leaf_id: NodeId) {
        let cf = self.storage.db.cf_handle(CF_LEAF_VECTORS).unwrap();
        self.batch.delete_cf(cf, node_key(leaf_id));
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
pub(crate) fn node_key(id: u64) -> [u8; 8] {
    id.to_be_bytes()
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

/// Read-your-own-writes transaction overlay that commits all mutations in one atomic `WriteBatch`
/// and updates the in-memory internal node cache only after RocksDB persistence succeeds.
pub struct TxContext<'a> {
    storage: &'a Storage,
    internal_cache: &'a mut HashMap<NodeId, PackedInternalNode>,
    meta: TreeMeta,
    meta_dirty: bool,
    internal_nodes: HashMap<NodeId, Option<PackedInternalNode>>,
    dirty_internal: HashSet<NodeId>,
    leaf_nodes: HashMap<NodeId, Option<PackedLeafNode>>,
    dirty_leaf: HashSet<NodeId>,
    leaf_vectors: HashMap<NodeId, Option<PackedLeafVectors>>,
    dirty_leaf_vecs: HashSet<NodeId>,
    dirty_doc_loc: HashMap<DocId, Option<NodeId>>,
}

impl<'a> TxContext<'a> {
    pub fn new(
        storage: &'a Storage,
        internal_cache: &'a mut HashMap<NodeId, PackedInternalNode>,
        meta: TreeMeta,
    ) -> Self {
        Self {
            storage,
            internal_cache,
            meta,
            meta_dirty: false,
            internal_nodes: HashMap::new(),
            dirty_internal: HashSet::new(),
            leaf_nodes: HashMap::new(),
            dirty_leaf: HashSet::new(),
            leaf_vectors: HashMap::new(),
            dirty_leaf_vecs: HashSet::new(),
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

    pub fn is_internal_node(&self, node_id: NodeId) -> bool {
        if let Some(opt) = self.internal_nodes.get(&node_id) {
            return opt.is_some();
        }
        self.internal_cache.contains_key(&node_id)
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

    pub fn get_leaf_vectors(&mut self, leaf_id: NodeId) -> Result<PackedLeafVectors> {
        if let Some(opt) = self.leaf_vectors.get(&leaf_id) {
            return opt.clone().ok_or_else(|| {
                TreeError::CorruptedData(format!("leaf vectors {leaf_id} deleted in tx"))
            });
        }
        let vecs = self
            .storage
            .get_leaf_vectors(leaf_id)?
            .ok_or_else(|| TreeError::CorruptedData(format!("missing leaf vectors {leaf_id}")))?;
        self.leaf_vectors.insert(leaf_id, Some(vecs.clone()));
        Ok(vecs)
    }

    pub fn put_leaf_vectors(&mut self, leaf_id: NodeId, vecs: PackedLeafVectors) {
        self.leaf_vectors.insert(leaf_id, Some(vecs));
        self.dirty_leaf_vecs.insert(leaf_id);
    }

    pub fn delete_leaf_vectors(&mut self, leaf_id: NodeId) {
        self.leaf_vectors.insert(leaf_id, None);
        self.dirty_leaf_vecs.insert(leaf_id);
    }

    pub fn put_doc_loc(&mut self, doc_id: DocId, leaf_id: NodeId) {
        self.dirty_doc_loc.insert(doc_id, Some(leaf_id));
    }

    pub fn delete_doc_loc(&mut self, doc_id: DocId) {
        self.dirty_doc_loc.insert(doc_id, None);
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
        for &leaf_id in &self.dirty_leaf_vecs {
            match self.leaf_vectors.get(&leaf_id).and_then(|o| o.as_ref()) {
                Some(vecs) => batch.put_leaf_vectors(leaf_id, vecs),
                None => batch.delete_leaf_vectors(leaf_id),
            }
        }
        for (doc_id, opt) in self.dirty_doc_loc {
            match opt {
                Some(leaf_id) => batch.put_doc_loc(doc_id, leaf_id),
                None => batch.delete_doc_loc(doc_id),
            }
        }

        self.storage.write_batch(batch)?;

        // Update in-memory internal node cache only after RocksDB commit succeeds.
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
        Ok(self.meta)
    }
}
