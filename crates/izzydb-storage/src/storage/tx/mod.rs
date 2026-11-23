mod commit;
mod fts;
mod nodes;
mod postings;
mod rows;
mod segments;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap, HashSet};

use roaring::RoaringBitmap;

use izzydb_core::config::{
    AttrId, ColumnGen, DocId, NodeId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentId,
    SkipChunk, TermId,
};
use izzydb_core::error::{Result, TreeError};

use crate::codec::{
    DetachedMeta,
    Manifest, SegmentMeta, ZoneEntry,
};
use crate::node::{AnnMeta, PackedInternalNode, PackedLeafNode};

use super::Storage;

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
    dirty_zone_deletes: HashSet<(SegmentId, AttrId, u8)>,
    dirty_attr_count_deltas: BTreeMap<Vec<u8>, i64>,
    fts_meta_bytes: Option<Vec<u8>>,
    fts_meta_dirty: bool,
    dirty_fts_terms: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    dirty_fts_skips: BTreeMap<(TermId, SkipChunk), Option<Vec<u8>>>,
    dirty_fts_postings: BTreeMap<(TermId, RowBlock), Option<Vec<u8>>>,
    dirty_fts_positions: BTreeMap<(TermId, RowBlock), Option<Vec<u8>>>,
    dirty_fts_doc_lens: BTreeMap<(AttrId, RowBlock), Option<Vec<u8>>>,
    dirty_columns: BTreeMap<[u8; 10], Vec<u8>>,
    dirty_detached_meta: BTreeMap<AttrId, DetachedMeta>,
    dirty_detached_blocks: BTreeMap<(AttrId, ColumnGen, SegmentId, u32), Option<Vec<u8>>>,
    dirty_detached_postings: BTreeMap<Vec<u8>, RoaringBitmap>,
    dirty_detached_count_deltas: BTreeMap<Vec<u8>, i64>,
    dirty_detached_overwritten: BTreeMap<(AttrId, ColumnGen, SegmentId), RoaringBitmap>,
    retire_detached_gens: Vec<(AttrId, ColumnGen)>,
    #[allow(clippy::type_complexity)]
    retire_segments: Vec<(SegmentId, Vec<(AttrId, ColumnGen)>, Vec<AttrId>)>,
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
            dirty_zone_deletes: HashSet::new(),
            dirty_attr_count_deltas: BTreeMap::new(),
            fts_meta_bytes: None,
            fts_meta_dirty: false,
            dirty_fts_terms: BTreeMap::new(),
            dirty_fts_skips: BTreeMap::new(),
            dirty_fts_postings: BTreeMap::new(),
            dirty_fts_positions: BTreeMap::new(),
            dirty_fts_doc_lens: BTreeMap::new(),
            dirty_columns: BTreeMap::new(),
            dirty_detached_meta: BTreeMap::new(),
            dirty_detached_blocks: BTreeMap::new(),
            dirty_detached_postings: BTreeMap::new(),
            dirty_detached_count_deltas: BTreeMap::new(),
            dirty_detached_overwritten: BTreeMap::new(),
            retire_detached_gens: Vec::new(),
            retire_segments: Vec::new(),
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
}


