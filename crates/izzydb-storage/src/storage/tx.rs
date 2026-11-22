use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use roaring::RoaringBitmap;
use rocksdb::{Direction, IteratorMode, ReadOptions};

use izzydb_core::config::{
    AttrId, DocId, FtsBlockId, FtsDocId, LocalId, NodeId, TermId, fts_block_id, fts_slot,
};
use izzydb_core::error::{Result, TreeError};

use crate::codec::{
    LeafPosting, encode_attr_prefix, encode_fts_term_id_prefix, encode_posting_prefix,
    encode_roaring_bitmap, encode_zone_map, make_posting_key, split_posting_key,
};
use crate::node::{PackedInternalNode, PackedLeafNode, TreeMeta, decode_f32_into};

use super::batch::StorageBatch;
use super::types::{AttrZoneMap, DocLoc, FtsRevBlock, map_db_err};
use super::{ATTR_COUNT_TAG, ATTR_POSTING_PREFIX_LEN, CF_ATTR_POSTINGS, CF_FTS_SKIPS, Storage};

/// Read-your-own-writes key/value overlay mapping `K -> Option<V>` (`None` = deleted).
struct OverlayMap<K, V>(HashMap<K, Option<V>>);

impl<K: Eq + Hash, V> Default for OverlayMap<K, V> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

impl<K: Eq + Hash, V: Clone> OverlayMap<K, V> {
    fn get_with<Q>(&self, key: &Q, fallback: impl FnOnce() -> Result<Option<V>>) -> Result<Option<V>>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        if let Some(opt) = self.0.get(key) {
            return Ok(opt.clone());
        }
        fallback()
    }

    fn put(&mut self, key: K, val: V) {
        self.0.insert(key, Some(val));
    }

    fn delete(&mut self, key: K) {
        self.0.insert(key, None);
    }
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
    dirty_vectors: OverlayMap<DocId, Vec<f32>>,
    dirty_doc_pos: HashMap<DocId, Option<(NodeId, LocalId)>>,
    dirty_fts_doc_map: HashMap<DocId, Option<FtsDocId>>,
    removed_fts_doc_ids: Vec<FtsDocId>,
    dirty_docs: OverlayMap<DocId, Vec<u8>>,
    dirty_attr_clusters: HashMap<Vec<u8>, RoaringBitmap>,
    dirty_attr_postings: HashMap<Vec<u8>, LeafPosting>,
    dirty_attr_count_deltas: HashMap<Vec<u8>, i64>,
    fts_meta_bytes: Option<Vec<u8>>,
    fts_meta_dirty: bool,
    fts_free_ids: Option<RoaringBitmap>,
    fts_free_ids_dirty: bool,
    dirty_fts_terms: OverlayMap<Vec<u8>, Vec<u8>>,
    dirty_fts_skips: OverlayMap<[u8; 8], Vec<u8>>,
    dirty_fts_postings: OverlayMap<[u8; 8], Vec<u8>>,
    dirty_fts_positions: OverlayMap<[u8; 8], Vec<u8>>,
    dirty_fts_doc_lens: OverlayMap<[u8; 6], Vec<u8>>,
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
            dirty_vectors: OverlayMap::default(),
            dirty_doc_pos: HashMap::new(),
            dirty_fts_doc_map: HashMap::new(),
            removed_fts_doc_ids: Vec::new(),
            dirty_docs: OverlayMap::default(),
            dirty_attr_clusters: HashMap::new(),
            dirty_attr_postings: HashMap::new(),
            dirty_attr_count_deltas: HashMap::new(),
            fts_meta_bytes: None,
            fts_meta_dirty: false,
            fts_free_ids: None,
            fts_free_ids_dirty: false,
            dirty_fts_terms: OverlayMap::default(),
            dirty_fts_skips: OverlayMap::default(),
            dirty_fts_postings: OverlayMap::default(),
            dirty_fts_positions: OverlayMap::default(),
            dirty_fts_doc_lens: OverlayMap::default(),
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
        self.dirty_vectors.put(doc_id, vec);
    }

    pub fn delete_vector(&mut self, doc_id: DocId) {
        self.dirty_vectors.delete(doc_id);
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
            match self.dirty_vectors.0.get(&doc_id) {
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
        self.dirty_docs
            .get_with(&doc_id, || self.storage.get_doc_bytes(doc_id))
    }

    pub fn multi_get_doc_bytes(&self, doc_ids: &[DocId]) -> Result<Vec<Option<Vec<u8>>>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![None; doc_ids.len()];
        let mut missing: Vec<(DocId, usize)> = Vec::new();
        for (slot, &doc_id) in doc_ids.iter().enumerate() {
            if let Some(opt) = self.dirty_docs.0.get(&doc_id) {
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
        self.dirty_docs.put(doc_id, bytes);
    }

    pub fn delete_doc(&mut self, doc_id: DocId) {
        self.dirty_docs.delete(doc_id);
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

    /// Record a signed change to the global document count of index key `cluster_key`
    /// (`default[0x06]`). Deltas coalesce within the transaction and are applied at commit.
    ///
    /// Only document-level changes (insert / delete / attribute edits) call this; moving
    /// documents between leaves (`ClusterDelta`) never changes global counts.
    pub fn adjust_attr_count(&mut self, cluster_key: &[u8], delta: i64) {
        if delta == 0 {
            return;
        }
        *self
            .dirty_attr_count_deltas
            .entry(cluster_key.to_vec())
            .or_insert(0) += delta;
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
            if !self.dirty_fts_skips.0.contains_key(&key_arr) {
                let chunk_id = u32::from_be_bytes(k[4..8].try_into().unwrap());
                if !f(chunk_id, &v)? {
                    return Ok(());
                }
            }
        }
        for (key, opt_bytes) in &self.dirty_fts_skips.0 {
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
        self.dirty_fts_terms
            .get_with(key, || self.storage.get_fts_term_bytes(key))
    }

    pub fn put_fts_term_bytes(&mut self, key: Vec<u8>, val: Vec<u8>) {
        self.dirty_fts_terms.put(key, val);
    }

    pub fn delete_fts_term(&mut self, key: Vec<u8>) {
        self.dirty_fts_terms.delete(key);
    }

    pub fn get_fts_skip_bytes(&self, key: &[u8; 8]) -> Result<Option<Vec<u8>>> {
        self.dirty_fts_skips
            .get_with(key, || self.storage.get_fts_skip_bytes(key))
    }

    pub fn put_fts_skip_bytes(&mut self, key: [u8; 8], val: Vec<u8>) {
        self.dirty_fts_skips.put(key, val);
    }

    pub fn delete_fts_skip(&mut self, key: [u8; 8]) {
        self.dirty_fts_skips.delete(key);
    }

    pub fn get_fts_posting_bytes(&self, key: &[u8; 8]) -> Result<Option<Vec<u8>>> {
        self.dirty_fts_postings
            .get_with(key, || self.storage.get_fts_posting_bytes(key))
    }

    pub fn put_fts_posting_bytes(&mut self, key: [u8; 8], val: Vec<u8>) {
        self.dirty_fts_postings.put(key, val);
    }

    pub fn delete_fts_posting(&mut self, key: [u8; 8]) {
        self.dirty_fts_postings.delete(key);
    }

    pub fn get_fts_position_bytes(&self, key: &[u8; 8]) -> Result<Option<Vec<u8>>> {
        self.dirty_fts_positions
            .get_with(key, || self.storage.get_fts_position_bytes(key))
    }

    pub fn put_fts_position_bytes(&mut self, key: [u8; 8], val: Vec<u8>) {
        self.dirty_fts_positions.put(key, val);
    }

    pub fn delete_fts_position(&mut self, key: [u8; 8]) {
        self.dirty_fts_positions.delete(key);
    }

    pub fn get_fts_doc_len_bytes(&self, key: &[u8; 6]) -> Result<Option<Vec<u8>>> {
        self.dirty_fts_doc_lens
            .get_with(key, || self.storage.get_fts_doc_len_bytes(key))
    }

    pub fn put_fts_doc_len_bytes(&mut self, key: [u8; 6], val: Vec<u8>) {
        self.dirty_fts_doc_lens.put(key, val);
    }

    pub fn delete_fts_doc_len(&mut self, key: [u8; 6]) {
        self.dirty_fts_doc_lens.delete(key);
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
        iter.seek(prefix);
        while iter.valid() {
            let Some(k) = iter.key() else { break };
            if !k.starts_with(&prefix) {
                break;
            }
            let live = self
                .dirty_attr_postings
                .get(k)
                .is_none_or(|p| !p.is_empty());
            if live {
                let v = &k[ATTR_POSTING_PREFIX_LEN..];
                if min_val.is_none() {
                    min_val = Some(v.to_vec());
                }
                max_val = Some(v.to_vec());
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)?;

        for (k, p) in &self.dirty_attr_postings {
            if !p.is_empty() && k.starts_with(&prefix) {
                let v = &k[ATTR_POSTING_PREFIX_LEN..];
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
        self.stage_tree_and_docs(&mut batch);
        let dirty_rev_blocks = self.stage_doc_locs_and_rev_blocks(&mut batch)?;
        let updated_zone_attrs = self.stage_attr_indexes(&mut batch)?;
        self.stage_attr_counts(&mut batch)?;
        self.stage_fts_writes(&mut batch)?;

        self.storage.write_batch(batch)?;

        self.flush_caches_after_commit(updated_zone_attrs, dirty_rev_blocks);
        Ok(self.meta)
    }

    fn stage_tree_and_docs(&self, batch: &mut StorageBatch<'_>) {
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
        for (&doc_id, opt) in &self.dirty_vectors.0 {
            match opt {
                Some(vec) => batch.put_vector(doc_id, vec),
                None => batch.delete_vector(doc_id),
            }
        }
        for (&doc_id, opt) in &self.dirty_docs.0 {
            match opt {
                Some(bytes) => batch.put_doc_bytes(doc_id, bytes),
                None => batch.delete_doc(doc_id),
            }
        }
    }

    fn stage_doc_locs_and_rev_blocks(
        &self,
        batch: &mut StorageBatch<'_>,
    ) -> Result<HashMap<FtsBlockId, FtsRevBlock>> {
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
            load_dirty_rev_block_mut(self.storage, &mut dirty_rev_blocks, bid)?
                .remove(fts_slot(fid));
        }

        for doc_id in doc_loc_ids {
            let old_loc = existing_locs.get(&doc_id).and_then(|o| *o);
            let new_pos = match self.dirty_doc_pos.get(&doc_id) {
                Some(opt) => *opt,
                None => old_loc.map(|l| (l.leaf_id, l.local_id)),
            };
            let Some((leaf_id, local_id)) = new_pos else {
                batch.delete_doc_loc(doc_id);
                continue;
            };
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
                load_dirty_rev_block_mut(self.storage, &mut dirty_rev_blocks, bid)?
                    .set(fts_slot(fid), doc_id, leaf_id, local_id);
            }
        }

        for (&bid, blk) in &dirty_rev_blocks {
            if blk.is_empty() {
                batch.delete_fts_rev_block(bid);
            } else {
                batch.put_fts_rev_block(bid, &blk.encode());
            }
        }
        Ok(dirty_rev_blocks)
    }

    fn stage_attr_counts(&self, batch: &mut StorageBatch<'_>) -> Result<()> {
        let mut deltas: Vec<(&Vec<u8>, i64)> = self
            .dirty_attr_count_deltas
            .iter()
            .filter(|&(_, &d)| d != 0)
            .map(|(k, &d)| (k, d))
            .collect();
        if deltas.is_empty() {
            return Ok(());
        }
        deltas.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let db_keys: Vec<Vec<u8>> = deltas
            .iter()
            .map(|(k, _)| {
                let mut key = Vec::with_capacity(1 + k.len());
                key.push(ATTR_COUNT_TAG);
                key.extend_from_slice(k);
                key
            })
            .collect();
        let current = self.storage.db.multi_get(db_keys.iter());
        for ((cluster_key, delta), cur) in deltas.iter().zip(current) {
            let cur = match cur.map_err(map_db_err)? {
                Some(bytes) => {
                    let arr: [u8; 8] = bytes.as_slice().try_into().map_err(|_| {
                        TreeError::CorruptedData(format!(
                            "attr count expected 8 bytes, got {}",
                            bytes.len()
                        ))
                    })?;
                    u64::from_le_bytes(arr) as i128
                }
                None => 0,
            };
            let next = cur + *delta as i128;
            if next < 0 {
                return Err(TreeError::InvariantViolation(format!(
                    "attr count underflow for key {cluster_key:?}: {cur} + {delta}"
                )));
            }
            if next == 0 {
                batch.delete_attr_count(cluster_key);
            } else {
                batch.put_attr_count(cluster_key, next as u64);
            }
        }
        Ok(())
    }

    fn stage_attr_indexes(
        &mut self,
        batch: &mut StorageBatch<'_>,
    ) -> Result<HashMap<(AttrId, u8), AttrZoneMap>> {
        for (key, bm) in &self.dirty_attr_clusters {
            if bm.is_empty() {
                batch.delete_attr_cluster(key);
            } else {
                let bytes = encode_roaring_bitmap(bm)?;
                batch.put_attr_cluster(key, &bytes);
            }
        }

        let mut updated_zone_attrs: HashMap<(AttrId, u8), AttrZoneMap> = HashMap::new();
        if self.dirty_attr_postings.is_empty() {
            return Ok(updated_zone_attrs);
        }

        let zones_arc = self.storage.get_attr_zones_cached()?;
        let mut need_recompute: HashSet<(AttrId, u8, NodeId)> = HashSet::new();

        for (key, posting) in &self.dirty_attr_postings {
            let (attr_id, type_tag, leaf_id, val_bytes) = split_posting_key(key)?;
            if type_tag == crate::codec::TAG_PRESENT || type_tag == crate::codec::TAG_BOOL {
                continue;
            }
            let attr_key = (attr_id, type_tag);
            let existing_bounds = updated_zone_attrs
                .get(&attr_key)
                .and_then(|zm| zm.get(&leaf_id))
                .or_else(|| zones_arc.get(&attr_key).and_then(|zm| zm.get(&leaf_id)));

            if !posting.is_empty() {
                let widens = match existing_bounds {
                    Some((min_b, max_b)) => {
                        val_bytes < min_b.as_slice() || val_bytes > max_b.as_slice()
                    }
                    None => true,
                };
                if widens {
                    let zmap = updated_zone_attrs
                        .entry(attr_key)
                        .or_insert_with(|| zones_arc.get(&attr_key).cloned().unwrap_or_default());
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
            } else if let Some((min_b, max_b)) = existing_bounds
                && (val_bytes == min_b.as_slice() || val_bytes == max_b.as_slice())
            {
                need_recompute.insert((attr_id, type_tag, leaf_id));
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

        for (key, posting) in std::mem::take(&mut self.dirty_attr_postings) {
            if posting.is_empty() {
                batch.delete_attr_posting(&key);
            } else {
                let bytes = posting.encode()?;
                batch.put_attr_posting(&key, &bytes);
            }
        }
        Ok(updated_zone_attrs)
    }

    fn stage_fts_writes(&self, batch: &mut StorageBatch<'_>) -> Result<()> {
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
        for (key, opt) in &self.dirty_fts_terms.0 {
            match opt {
                Some(bytes) => batch.put_fts_term(key, bytes),
                None => batch.delete_fts_term(key),
            }
        }
        for (key, opt) in &self.dirty_fts_skips.0 {
            match opt {
                Some(bytes) => batch.put_fts_skip(key, bytes),
                None => batch.delete_fts_skip(key),
            }
        }
        for (key, opt) in &self.dirty_fts_postings.0 {
            match opt {
                Some(bytes) => batch.put_fts_posting(key, bytes),
                None => batch.delete_fts_posting(key),
            }
        }
        for (key, opt) in &self.dirty_fts_positions.0 {
            match opt {
                Some(bytes) => batch.put_fts_position(key, bytes),
                None => batch.delete_fts_position(key),
            }
        }
        for (key, opt) in &self.dirty_fts_doc_lens.0 {
            match opt {
                Some(bytes) => batch.put_fts_doc_len(key, bytes),
                None => batch.delete_fts_doc_len(key),
            }
        }
        Ok(())
    }

    fn flush_caches_after_commit(
        &mut self,
        updated_zone_attrs: HashMap<(AttrId, u8), AttrZoneMap>,
        dirty_rev_blocks: HashMap<FtsBlockId, FtsRevBlock>,
    ) {
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

        if !self.dirty_fts_doc_lens.0.is_empty() {
            let mut cache = self
                .storage
                .fts_doc_lens_cache
                .write()
                .expect("fts_doc_lens_cache poisoned");
            let mut touched_attrs: HashSet<u16> = HashSet::new();
            for (key, opt) in std::mem::take(&mut self.dirty_fts_doc_lens.0) {
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

        for &node_id in &self.dirty_internal {
            match self.internal_nodes.remove(&node_id).flatten() {
                Some(node) => {
                    self.internal_cache.insert(node_id, node);
                }
                None => {
                    self.internal_cache.remove(&node_id);
                }
            }
        }
        for &leaf_id in &self.dirty_leaf {
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
    }
}

fn load_dirty_rev_block_mut<'m>(
    storage: &Storage,
    dirty_rev_blocks: &'m mut HashMap<FtsBlockId, FtsRevBlock>,
    bid: FtsBlockId,
) -> Result<&'m mut FtsRevBlock> {
    match dirty_rev_blocks.entry(bid) {
        std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.into_mut()),
        std::collections::hash_map::Entry::Vacant(entry) => {
            let blk = storage.get_fts_rev_block(bid)?.unwrap_or_default();
            Ok(entry.insert(blk))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izzydb_core::config::{DistanceMetric, TreeConfig};
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
