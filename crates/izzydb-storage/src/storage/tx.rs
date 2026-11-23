use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use roaring::RoaringBitmap;
use rocksdb::{Direction, IteratorMode, ReadOptions};

use izzydb_core::config::{
    AttrId, DocId, LocalId, NodeId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentId, SkipChunk, TermId,
};
use izzydb_core::error::{Result, TreeError};

use crate::codec::{
    ATTR_COUNT_TAG, Manifest, SegPosting, SegmentMeta, SegmentState, ZoneEntry, column_key,
    encode_fts_term_id_prefix, encode_roaring_bitmap, fts_chunk_key_coord, split_posting_key_parts,
};
use crate::node::{AnnMeta, PackedInternalNode, PackedLeafNode, decode_f32_into};

use super::batch::StorageBatch;
use super::segments::SegmentEntry;
use super::types::map_db_err;
use super::{CF_ATTR_POSTINGS, CF_FTS_SKIPS, Storage};

/// Read-your-own-writes transaction overlay across segment store, id map, ANN tree,
/// attribute postings, zones, counts, and FTS indexes.
pub struct TxContext<'a> {
    storage: &'a Storage,
    internal_cache: &'a mut HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &'a mut HashMap<NodeId, NodeId>,
    manifest: Manifest,
    manifest_dirty: bool,
    ann_meta: Option<AnnMeta>,
    ann_meta_dirty: bool,
    internal_nodes: HashMap<NodeId, Option<PackedInternalNode>>,
    dirty_internal: HashSet<NodeId>,
    leaf_nodes: HashMap<NodeId, Option<PackedLeafNode>>,
    dirty_leaf: HashSet<NodeId>,
    dirty_rows: BTreeMap<RowId, Option<Vec<u8>>>,
    dirty_vectors: BTreeMap<RowId, Option<Vec<f32>>>,
    dirty_id_map: BTreeMap<DocId, Option<RowId>>,
    dirty_doc_id_blocks: BTreeMap<(SegmentId, u32), Box<[DocId; ROW_BLOCK_SIZE]>>,
    dirty_seg_meta: BTreeMap<SegmentId, Option<SegmentMeta>>,
    dirty_seg_deletes: BTreeMap<SegmentId, RoaringBitmap>,
    dirty_dropped_segments: Option<RoaringBitmap>,
    dirty_postings: BTreeMap<Vec<u8>, RoaringBitmap>,
    dirty_zone: BTreeMap<(SegmentId, AttrId, u8), ZoneEntry>,
    dirty_attr_count_deltas: BTreeMap<Vec<u8>, i64>,
    fts_meta_bytes: Option<Vec<u8>>,
    fts_meta_dirty: bool,
    dirty_fts_terms: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    dirty_fts_skips: BTreeMap<(TermId, SkipChunk), Option<Vec<u8>>>,
    dirty_fts_postings: BTreeMap<(TermId, RowBlock), Option<Vec<u8>>>,
    dirty_fts_positions: BTreeMap<(TermId, RowBlock), Option<Vec<u8>>>,
    dirty_fts_doc_lens: BTreeMap<(AttrId, RowBlock), Option<Vec<u8>>>,
    dirty_columns: BTreeMap<[u8; 10], Vec<u8>>,
}

impl<'a> TxContext<'a> {
    pub fn new(
        storage: &'a Storage,
        internal_cache: &'a mut HashMap<NodeId, PackedInternalNode>,
        leaf_parent: &'a mut HashMap<NodeId, NodeId>,
        manifest: Manifest,
        ann_meta: Option<AnnMeta>,
    ) -> Self {
        Self {
            storage,
            internal_cache,
            leaf_parent,
            manifest,
            manifest_dirty: false,
            ann_meta,
            ann_meta_dirty: false,
            internal_nodes: HashMap::new(),
            dirty_internal: HashSet::new(),
            leaf_nodes: HashMap::new(),
            dirty_leaf: HashSet::new(),
            dirty_rows: BTreeMap::new(),
            dirty_vectors: BTreeMap::new(),
            dirty_id_map: BTreeMap::new(),
            dirty_doc_id_blocks: BTreeMap::new(),
            dirty_seg_meta: BTreeMap::new(),
            dirty_seg_deletes: BTreeMap::new(),
            dirty_dropped_segments: None,
            dirty_postings: BTreeMap::new(),
            dirty_zone: BTreeMap::new(),
            dirty_attr_count_deltas: BTreeMap::new(),
            fts_meta_bytes: None,
            fts_meta_dirty: false,
            dirty_fts_terms: BTreeMap::new(),
            dirty_fts_skips: BTreeMap::new(),
            dirty_fts_postings: BTreeMap::new(),
            dirty_fts_positions: BTreeMap::new(),
            dirty_fts_doc_lens: BTreeMap::new(),
            dirty_columns: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn storage(&self) -> &'a Storage {
        self.storage
    }

    #[inline]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    #[inline]
    pub fn manifest_mut(&mut self) -> &mut Manifest {
        self.manifest_dirty = true;
        &mut self.manifest
    }

    #[inline]
    pub fn ann_meta(&self) -> Option<&AnnMeta> {
        self.ann_meta.as_ref()
    }

    #[inline]
    pub fn ann_meta_mut(&mut self) -> Result<&mut AnnMeta> {
        self.ann_meta_dirty = true;
        self.ann_meta
            .as_mut()
            .ok_or_else(|| TreeError::InvalidConfig("collection has no vector index".into()))
    }

    pub fn alloc_node_id(&mut self) -> Result<NodeId> {
        let meta = self.ann_meta_mut()?;
        let id = meta.next_node_id;
        meta.next_node_id = id.checked_add(1).expect("NodeId u32 overflow");
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Segment table & tombstones
    // -----------------------------------------------------------------------

    pub fn get_seg_meta(&self, seg: SegmentId) -> Option<SegmentMeta> {
        if let Some(opt) = self.dirty_seg_meta.get(&seg) {
            return opt.clone();
        }
        self.storage
            .segments
            .snapshot()
            .get(seg)
            .map(|e| e.meta.clone())
    }

    pub fn put_seg_meta(&mut self, seg: SegmentId, meta: SegmentMeta) {
        self.dirty_seg_meta.insert(seg, Some(meta));
    }

    pub fn delete_seg_meta(&mut self, seg: SegmentId) {
        self.dirty_seg_meta.insert(seg, None);
    }

    pub fn tombstone_row(&mut self, row: RowId) -> Result<()> {
        let seg = row.seg();
        let local = row.local();
        let mut meta = self.get_seg_meta(seg).ok_or_else(|| {
            TreeError::CorruptedData(format!("tombstone for unknown segment {seg}"))
        })?;
        if local >= meta.row_count {
            return Err(TreeError::CorruptedData(format!(
                "tombstone local {local} >= segment {seg} row_count {}",
                meta.row_count
            )));
        }

        if !self.dirty_seg_deletes.contains_key(&seg) {
            let existing = self
                .storage
                .segments
                .snapshot()
                .get(seg)
                .and_then(|e| e.deleted.as_ref().map(|arc| (**arc).clone()))
                .unwrap_or_default();
            self.dirty_seg_deletes.insert(seg, existing);
        }
        let bm = self.dirty_seg_deletes.get_mut(&seg).unwrap();
        if !bm.insert(local) {
            return Err(TreeError::InvariantViolation(format!(
                "double tombstone on {row:?}"
            )));
        }
        meta.deleted_count += 1;
        self.dirty_seg_meta.insert(seg, Some(meta));
        Ok(())
    }

    pub fn mark_segment_dropped(&mut self, seg: SegmentId) {
        if self.dirty_dropped_segments.is_none() {
            self.dirty_dropped_segments =
                Some((*self.storage.dropped_segments_snapshot()).clone());
        }
        self.dirty_dropped_segments.as_mut().unwrap().insert(seg);
        self.dirty_seg_meta.insert(seg, None);
        self.dirty_seg_deletes.remove(&seg);
    }

    /// Resolve a [`RowId`] against uncommitted `TxContext` mutations and the base `SegmentTable`,
    /// returning `None` if tombstoned, dropped, retired, or out of range, or the forwarded live [`RowId`].
    pub fn resolve_row(&self, mut row: RowId) -> Option<RowId> {
        let snap = self.storage.segments.snapshot();
        for _ in 0..8 {
            let seg = row.seg();
            if let Some(ref dropped) = self.dirty_dropped_segments
                && dropped.contains(seg)
            {
                return None;
            }
            let meta = match self.dirty_seg_meta.get(&seg) {
                Some(Some(m)) => m,
                Some(None) => return None,
                None => &snap.get(seg)?.meta,
            };
            let local = row.local();
            if local >= meta.row_count {
                return None;
            }
            let deleted_bm = self
                .dirty_seg_deletes
                .get(&seg)
                .or_else(|| snap.get(seg).and_then(|e| e.deleted.as_deref()));
            if deleted_bm.is_some_and(|bm| bm.contains(local)) {
                return None;
            }
            match meta.state {
                SegmentState::Open | SegmentState::Sealed => return Some(row),
                SegmentState::Forwarding => {
                    let del_rank = deleted_bm.map_or(0, |bm| bm.rank(local) as u32);
                    let rank = local - del_rank;
                    row = RowId::new(meta.forward_target, meta.forward_base + rank);
                }
                SegmentState::Building | SegmentState::Retired => return None,
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Rows, vectors, id map, and row->doc block map
    // -----------------------------------------------------------------------

    pub fn get_id_map(&self, doc_id: DocId) -> Result<Option<RowId>> {
        if let Some(opt) = self.dirty_id_map.get(&doc_id) {
            return Ok(*opt);
        }
        self.storage.get_id_map(doc_id)
    }

    pub fn put_id_map(&mut self, doc_id: DocId, row: RowId) {
        self.dirty_id_map.insert(doc_id, Some(row));
    }

    pub fn delete_id_map(&mut self, doc_id: DocId) {
        self.dirty_id_map.insert(doc_id, None);
    }

    pub fn get_row_bytes(&self, row: RowId) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_rows.get(&row) {
            return Ok(opt.clone());
        }
        self.storage.get_row_bytes(row)
    }

    pub fn put_row_bytes(&mut self, row: RowId, bytes: Vec<u8>) {
        self.dirty_rows.insert(row, Some(bytes));
    }

    pub fn delete_row(&mut self, row: RowId) {
        self.dirty_rows.insert(row, None);
    }

    pub fn put_column_block(
        &mut self,
        seg: SegmentId,
        attr_id: AttrId,
        block: u32,
        bytes: Vec<u8>,
    ) {
        self.dirty_columns.insert(column_key(seg, attr_id, block), bytes);
    }

    /// Visit every live (non-tombstoned, non-deleted) row in `seg` in ascending `RowId` order,
    /// overlaying uncommitted `dirty_rows` and `dirty_seg_deletes` inside this transaction.
    pub fn visit_segment_live_rows_in_tx(
        &self,
        seg: SegmentId,
        mut visit: impl FnMut(RowId, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let mut last_db_row: Option<RowId> = None;
        let snap = self.storage.segments.snapshot();
        let deleted_bm = self
            .dirty_seg_deletes
            .get(&seg)
            .or_else(|| snap.get(seg).and_then(|e| e.deleted.as_deref()));

        self.storage.visit_segment_rows(seg, |row, db_bytes| {
            last_db_row = Some(row);
            if deleted_bm.is_some_and(|bm| bm.contains(row.local())) {
                return Ok(());
            }
            if let Some(opt) = self.dirty_rows.get(&row) {
                if let Some(bytes) = opt {
                    visit(row, bytes)?;
                }
            } else {
                visit(row, db_bytes)?;
            }
            Ok(())
        })?;

        let start_dirty = match last_db_row {
            Some(r) => RowId::new(seg, r.local().saturating_add(1)),
            None => RowId::new(seg, 0),
        };
        let end_dirty = RowId::new(seg, u32::MAX);
        for (&row, opt) in self.dirty_rows.range(start_dirty..=end_dirty) {
            if deleted_bm.is_some_and(|bm| bm.contains(row.local())) {
                continue;
            }
            if let Some(bytes) = opt {
                visit(row, bytes)?;
            }
        }
        Ok(())
    }

    pub fn set_row_doc_id(&mut self, row: RowId, doc_id: DocId) -> Result<()> {
        let key = (row.seg(), row.block_idx());
        if !self.dirty_doc_id_blocks.contains_key(&key) {
            let block = match self.storage.get_doc_id_block(key.0, key.1)? {
                Some(arc) => Box::new(*arc),
                None => Box::new([0u64; ROW_BLOCK_SIZE]),
            };
            self.dirty_doc_id_blocks.insert(key, block);
        }
        self.dirty_doc_id_blocks.get_mut(&key).unwrap()[row.slot() as usize] = doc_id;
        Ok(())
    }

    pub fn put_vector(&mut self, row: RowId, vec: Vec<f32>) {
        debug_assert_eq!(vec.len(), self.storage.dim);
        self.dirty_vectors.insert(row, Some(vec));
    }

    pub fn delete_vector(&mut self, row: RowId) {
        self.dirty_vectors.insert(row, None);
    }

    pub fn get_vector(&self, row: RowId) -> Result<Option<Vec<f32>>> {
        if let Some(opt) = self.dirty_vectors.get(&row) {
            return Ok(opt.clone());
        }
        self.storage.get_vector(row)
    }

    /// Fetch full-precision vectors for `rows` into a flat `rows.len() * dim` buffer,
    /// honoring read-your-own-writes mutations in `dirty_vectors`.
    pub fn get_vectors_flat(&self, rows: &[RowId]) -> Result<Vec<f32>> {
        let dim = self.storage.dim;
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![0.0f32; rows.len() * dim];
        let mut missing: Vec<(RowId, usize)> = Vec::new();
        for (slot, &row) in rows.iter().enumerate() {
            match self.dirty_vectors.get(&row) {
                Some(Some(vec)) => {
                    out[slot * dim..(slot + 1) * dim].copy_from_slice(vec);
                }
                Some(None) => {
                    return Err(TreeError::CorruptedData(format!(
                        "vector {row:?} was deleted in tx"
                    )));
                }
                None => missing.push((row, slot)),
            }
        }
        if !missing.is_empty() {
            missing.sort_unstable_by_key(|&(row, _)| row);
            let sorted_ids: Vec<RowId> = missing.iter().map(|&(row, _)| row).collect();
            let slices = self.storage.multi_get_vectors_pinned(&sorted_ids)?;
            let expected_bytes = dim * 4;
            for (&(row, slot), opt_slice) in missing.iter().zip(slices.iter()) {
                let slice = opt_slice.as_ref().ok_or_else(|| {
                    TreeError::CorruptedData(format!("missing vector {row:?} in cf_vectors"))
                })?;
                if slice.len() != expected_bytes {
                    return Err(TreeError::CorruptedData(format!(
                        "vector {row:?} expected {expected_bytes} bytes, got {}",
                        slice.len()
                    )));
                }
                decode_f32_into(slice, &mut out[slot * dim..(slot + 1) * dim])?;
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // ANN tree nodes
    // -----------------------------------------------------------------------

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
            .ok_or_else(|| TreeError::CorruptedData(format!("missing internal node {node_id}")))?;
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

    // -----------------------------------------------------------------------
    // Attribute postings, zones, and counts (insert-only deltas)
    // -----------------------------------------------------------------------

    pub fn add_posting(&mut self, posting_key: Vec<u8>, local: LocalId) {
        self.dirty_postings
            .entry(posting_key)
            .or_default()
            .insert(local);
    }

    pub fn widen_zone(
        &mut self,
        seg: SegmentId,
        attr_id: AttrId,
        tag: u8,
        encoded_val: &[u8],
    ) -> Result<()> {
        let key = (seg, attr_id, tag);
        if let Some(entry) = self.dirty_zone.get_mut(&key) {
            entry.widen(encoded_val);
            return Ok(());
        }
        let zones = self.storage.get_zones_cached()?;
        let mut entry = zones
            .get(&(attr_id, tag))
            .and_then(|m| m.get(&seg))
            .cloned()
            .unwrap_or_else(|| ZoneEntry::singleton(encoded_val));
        entry.widen(encoded_val);
        self.dirty_zone.insert(key, entry);
        Ok(())
    }

    pub fn adjust_attr_count(&mut self, attr_key: &[u8], delta: i64) {
        if delta == 0 {
            return;
        }
        *self
            .dirty_attr_count_deltas
            .entry(attr_key.to_vec())
            .or_insert(0) += delta;
    }

    // -----------------------------------------------------------------------
    // FTS overlay (keyed by RowBlock / SkipChunk)
    // -----------------------------------------------------------------------

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

    pub fn get_fts_skip_bytes(&self, term_id: TermId, chunk: SkipChunk) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_skips.get(&(term_id, chunk)) {
            return Ok(opt.clone());
        }
        self.storage
            .get_fts_skip_bytes(&fts_chunk_key_coord(term_id, chunk))
    }

    pub fn put_fts_skip_bytes(&mut self, term_id: TermId, chunk: SkipChunk, val: Vec<u8>) {
        self.dirty_fts_skips.insert((term_id, chunk), Some(val));
    }

    pub fn delete_fts_skip(&mut self, term_id: TermId, chunk: SkipChunk) {
        self.dirty_fts_skips.insert((term_id, chunk), None);
    }

    pub fn get_fts_posting_bytes(
        &self,
        term_id: TermId,
        block: RowBlock,
    ) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_postings.get(&(term_id, block)) {
            return Ok(opt.clone());
        }
        self.storage
            .get_fts_posting_bytes(&crate::codec::fts_block_key_coord(term_id, block))
    }

    pub fn put_fts_posting_bytes(&mut self, term_id: TermId, block: RowBlock, val: Vec<u8>) {
        self.dirty_fts_postings.insert((term_id, block), Some(val));
    }

    pub fn delete_fts_posting(&mut self, term_id: TermId, block: RowBlock) {
        self.dirty_fts_postings.insert((term_id, block), None);
    }

    pub fn get_fts_position_bytes(
        &self,
        term_id: TermId,
        block: RowBlock,
    ) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_positions.get(&(term_id, block)) {
            return Ok(opt.clone());
        }
        self.storage
            .get_fts_position_bytes(&crate::codec::fts_block_key_coord(term_id, block))
    }

    pub fn put_fts_position_bytes(&mut self, term_id: TermId, block: RowBlock, val: Vec<u8>) {
        self.dirty_fts_positions.insert((term_id, block), Some(val));
    }

    pub fn delete_fts_position(&mut self, term_id: TermId, block: RowBlock) {
        self.dirty_fts_positions.insert((term_id, block), None);
    }

    pub fn get_fts_doc_len_bytes(&self, attr_id: AttrId, block: RowBlock) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_doc_lens.get(&(attr_id, block)) {
            return Ok(opt.clone());
        }
        self.storage
            .get_fts_doc_len_bytes(&crate::codec::doc_len_key_coord(attr_id, block))
    }

    pub fn put_fts_doc_len_bytes(&mut self, attr_id: AttrId, block: RowBlock, val: Vec<u8>) {
        self.dirty_fts_doc_lens.insert((attr_id, block), Some(val));
    }

    pub fn delete_fts_doc_len(&mut self, attr_id: AttrId, block: RowBlock) {
        self.dirty_fts_doc_lens.insert((attr_id, block), None);
    }

    pub fn visit_fts_skip_blocks(
        &self,
        term_id: TermId,
        mut f: impl FnMut(SkipChunk, &[u8]) -> Result<bool>,
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
            if k.len() != 12 || k[0..4] != prefix {
                break;
            }
            let seg = u32::from_be_bytes(k[4..8].try_into().unwrap());
            let cid = u32::from_be_bytes(k[8..12].try_into().unwrap());
            let chunk = SkipChunk::new(seg, cid);
            if !self.dirty_fts_skips.contains_key(&(term_id, chunk)) && !f(chunk, &v)? {
                return Ok(());
            }
        }
        for (&(tid, chunk), opt_bytes) in &self.dirty_fts_skips {
            if tid == term_id
                && let Some(bytes) = opt_bytes
                && !f(chunk, bytes)?
            {
                return Ok(());
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Commit
    // -----------------------------------------------------------------------

    pub fn commit(mut self) -> Result<(Manifest, Option<AnnMeta>)> {
        let mut batch = self.storage.new_batch();
        self.stage_manifest_and_segments(&mut batch)?;
        self.stage_rows_and_tree(&mut batch);
        self.stage_postings_and_zones(&mut batch)?;
        self.stage_attr_counts(&mut batch)?;
        self.stage_fts_writes(&mut batch);
        self.stage_columns(&mut batch);

        self.storage.write_batch(batch)?;

        self.flush_caches_after_commit();
        Ok((self.manifest, self.ann_meta))
    }

    fn stage_manifest_and_segments(&self, batch: &mut StorageBatch<'_>) -> Result<()> {
        if self.manifest_dirty {
            batch.put_manifest(&self.manifest);
        }
        if let Some(ref dropped) = self.dirty_dropped_segments {
            batch.put_dropped_segments(&encode_roaring_bitmap(dropped)?);
        }
        for (&seg, opt_meta) in &self.dirty_seg_meta {
            match opt_meta {
                Some(meta) => batch.put_seg_meta(seg, meta),
                None => batch.delete_seg_meta(seg),
            }
        }
        for (&seg, bm) in &self.dirty_seg_deletes {
            if bm.is_empty() {
                batch.delete_seg_deletes(seg);
            } else {
                batch.put_seg_deletes(seg, &encode_roaring_bitmap(bm)?);
            }
        }
        Ok(())
    }

    fn stage_rows_and_tree(&self, batch: &mut StorageBatch<'_>) {
        if self.ann_meta_dirty
            && let Some(ref meta) = self.ann_meta
        {
            batch.put_ann_meta(meta);
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
        for (&doc_id, &opt_row) in &self.dirty_id_map {
            match opt_row {
                Some(row) => batch.put_id_map(doc_id, row),
                None => batch.delete_id_map(doc_id),
            }
        }
        for (&row, opt_bytes) in &self.dirty_rows {
            match opt_bytes {
                Some(bytes) => batch.put_row_bytes(row, bytes),
                None => batch.delete_row(row),
            }
        }
        for (&(seg, block), ids) in &self.dirty_doc_id_blocks {
            batch.put_doc_id_block(seg, block, ids);
        }
        for (&row, opt_vec) in &self.dirty_vectors {
            match opt_vec {
                Some(vec) => batch.put_vector(row, vec),
                None => batch.delete_vector(row),
            }
        }
    }

    fn stage_postings_and_zones(&self, batch: &mut StorageBatch<'_>) -> Result<()> {
        if !self.dirty_postings.is_empty() {
            let keys: Vec<Vec<u8>> = self.dirty_postings.keys().cloned().collect();
            let existing = self.storage.multi_get_attr_posting_slices(&keys)?;
            let cf = self.storage.cf(CF_ATTR_POSTINGS)?;
            let _ = cf;
            for ((key, delta_bm), opt_slice) in self.dirty_postings.iter().zip(existing) {
                let mut merged = match opt_slice {
                    Some(slice) => SegPosting::decode(&slice)?,
                    None => SegPosting::default(),
                };
                merged.or_bitmap(delta_bm);
                batch.put_attr_posting(key, &merged.encode()?);
            }
        }
        for (&(seg, attr_id, tag), entry) in &self.dirty_zone {
            batch.put_zone_entry(seg, attr_id, tag, entry);
        }
        Ok(())
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
        for ((attr_key, delta), cur) in deltas.iter().zip(current) {
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
                    "attr count underflow for key {attr_key:?}: {cur} + {delta}"
                )));
            }
            if next == 0 {
                batch.delete_attr_count(attr_key);
            } else {
                batch.put_attr_count(attr_key, next as u64);
            }
        }
        Ok(())
    }

    fn stage_fts_writes(&self, batch: &mut StorageBatch<'_>) {
        if self.fts_meta_dirty
            && let Some(ref bytes) = self.fts_meta_bytes
        {
            batch.put_fts_meta_bytes(bytes);
        }
        for (key, opt) in &self.dirty_fts_terms {
            match opt {
                Some(bytes) => batch.put_fts_term(key, bytes),
                None => batch.delete_fts_term(key),
            }
        }
        for (&(term_id, chunk), opt) in &self.dirty_fts_skips {
            match opt {
                Some(bytes) => batch.put_fts_skip(term_id, chunk, bytes),
                None => batch.delete_fts_skip(term_id, chunk),
            }
        }
        for (&(term_id, block), opt) in &self.dirty_fts_postings {
            match opt {
                Some(bytes) => batch.put_fts_posting(term_id, block, bytes),
                None => batch.delete_fts_posting(term_id, block),
            }
        }
        for (&(term_id, block), opt) in &self.dirty_fts_positions {
            match opt {
                Some(bytes) => batch.put_fts_position(term_id, block, bytes),
                None => batch.delete_fts_position(term_id, block),
            }
        }
        for (&(attr_id, block), opt) in &self.dirty_fts_doc_lens {
            match opt {
                Some(bytes) => batch.put_fts_doc_len(attr_id, block, bytes),
                None => batch.delete_fts_doc_len(attr_id, block),
            }
        }
    }

    fn stage_columns(&self, batch: &mut StorageBatch<'_>) {
        for (key, bytes) in &self.dirty_columns {
            batch.put_column_block(key, bytes);
        }
    }

    fn flush_caches_after_commit(&mut self) {
        if let Some(dropped) = self.dirty_dropped_segments.take() {
            *self
                .storage
                .dropped_segments
                .write()
                .expect("dropped_segments poisoned") = Arc::new(dropped);
        }

        if !self.dirty_seg_meta.is_empty() || !self.dirty_seg_deletes.is_empty() {
            let dirty_meta = std::mem::take(&mut self.dirty_seg_meta);
            let mut dirty_del = std::mem::take(&mut self.dirty_seg_deletes);
            let mut retired_segs = Vec::new();
            self.storage.segments.mutate_and_publish(|snap| {
                for (seg, opt_meta) in dirty_meta {
                    match opt_meta {
                        Some(meta) => {
                            let del_arc = dirty_del
                                .remove(&seg)
                                .map(|bm| (!bm.is_empty()).then(|| Arc::new(bm)))
                                .unwrap_or_else(|| {
                                    snap.segments.get(&seg).and_then(|e| e.deleted.clone())
                                });
                            if meta.state == SegmentState::Retired {
                                retired_segs.push(seg);
                            }
                            snap.segments.insert(
                                seg,
                                SegmentEntry {
                                    meta,
                                    deleted: del_arc,
                                },
                            );
                        }
                        None => {
                            retired_segs.push(seg);
                            snap.segments.remove(&seg);
                        }
                    }
                }
                for (seg, bm) in dirty_del {
                    if let Some(entry) = snap.segments.get_mut(&seg) {
                        entry.deleted = (!bm.is_empty()).then(|| Arc::new(bm));
                    }
                }
            });
            if !retired_segs.is_empty() {
                let mut cache = self
                    .storage
                    .doc_id_block_cache
                    .write()
                    .expect("doc_id_block_cache poisoned");
                for seg in retired_segs {
                    cache.remove_segment(seg);
                }
            }
        }

        if !self.dirty_doc_id_blocks.is_empty() {
            let mut cache = self
                .storage
                .doc_id_block_cache
                .write()
                .expect("doc_id_block_cache poisoned");
            for (key, blk) in std::mem::take(&mut self.dirty_doc_id_blocks) {
                cache.insert(key, Arc::from(blk));
            }
        }

        if !self.dirty_zone.is_empty() {
            let mut zguard = self
                .storage
                .zone_cache
                .write()
                .expect("zone_cache poisoned");
            if let Some(ref mut table_arc) = *zguard {
                let map = Arc::make_mut(table_arc);
                for ((seg, attr_id, tag), entry) in std::mem::take(&mut self.dirty_zone) {
                    map.entry((attr_id, tag)).or_default().insert(seg, entry);
                }
            }
        }

        if !self.dirty_fts_doc_lens.is_empty() {
            let mut cache = self
                .storage
                .fts_doc_lens_cache
                .write()
                .expect("fts_doc_lens_cache poisoned");
            for ((attr_id, block), opt) in std::mem::take(&mut self.dirty_fts_doc_lens) {
                if let Some(table) = cache.get_mut(&attr_id) {
                    let blocks = Arc::make_mut(table);
                    match opt {
                        Some(bytes) => {
                            blocks.insert(block, bytes.into_boxed_slice());
                        }
                        None => {
                            blocks.remove(&block);
                        }
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

// Re-check split_posting_key_parts import stays used cleanly (used in tests/later phases).
#[allow(dead_code)]
fn _keep_posting_split_import(k: &[u8]) -> Result<(AttrId, u8, &[u8], SegmentId)> {
    split_posting_key_parts(k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use izzydb_core::config::{DistanceMetric, TreeConfig};
    use izzydb_core::rowset::Liveness;
    use izzydb_core::schema::CollectionConfig;
    use tempfile::TempDir;

    #[test]
    fn test_tx_vectors_id_map_and_segment_snapshot_isolation() {
        let dir = TempDir::new().unwrap();
        let cfg = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared));
        let (storage, manifest, ann) = Storage::open(dir.path(), &cfg).unwrap();
        let mut cache = HashMap::new();
        let mut leaf_parent = HashMap::new();

        let r0 = RowId::new(0, 0);
        let r1 = RowId::new(0, 1);
        let r2 = RowId::new(0, 2);

        let (manifest, ann) = {
            let mut tx = TxContext::new(&storage, &mut cache, &mut leaf_parent, manifest, ann);
            let mut seg0 = SegmentMeta::new_open(0);
            seg0.row_count = 2;
            tx.put_seg_meta(0, seg0);
            tx.manifest_mut().open_segment = Some(0);
            tx.manifest_mut().next_segment_id = 1;
            tx.manifest_mut().live_rows = 2;
            tx.manifest_mut().total_rows = 2;

            tx.put_id_map(100, r0);
            tx.put_id_map(200, r1);
            tx.set_row_doc_id(r0, 100).unwrap();
            tx.set_row_doc_id(r1, 200).unwrap();
            tx.put_vector(r0, vec![1.0, 1.1, 1.2, 1.3]);
            tx.put_vector(r1, vec![2.0, 2.1, 2.2, 2.3]);
            tx.commit().unwrap()
        };

        let snap1 = storage.segment_snapshot();
        assert!(snap1.is_live(r0));
        assert!(snap1.is_live(r1));
        assert_eq!(storage.resolve_doc_ids(&[r0, r1]).unwrap(), vec![Some(100), Some(200)]);

        let mut tx = TxContext::new(&storage, &mut cache, &mut leaf_parent, manifest, ann);
        tx.tombstone_row(r0).unwrap();
        let mut seg0 = tx.get_seg_meta(0).unwrap();
        seg0.row_count = 3;
        tx.put_seg_meta(0, seg0);
        tx.put_id_map(100, r2);
        tx.set_row_doc_id(r2, 100).unwrap();
        tx.put_vector(r2, vec![9.0, 9.1, 9.2, 9.3]);

        let flat = tx.get_vectors_flat(&[r1, r2]).unwrap();
        assert_eq!(flat, vec![2.0, 2.1, 2.2, 2.3, 9.0, 9.1, 9.2, 9.3]);

        // Snap1 still sees r0 live before commit, and even after commit!
        tx.commit().unwrap();
        assert!(snap1.is_live(r0));

        let snap2 = storage.segment_snapshot();
        assert!(!snap2.is_live(r0));
        assert!(snap2.is_live(r1));
        assert!(snap2.is_live(r2));
        assert_eq!(storage.get_id_map(100).unwrap(), Some(r2));
    }
}
