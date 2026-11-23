//! Global ANN clustering tree (~100x fanout) over `RowId`s with lazy tombstones,
//! vacuum-before-split, SPFresh LIRE boundary reassignment, and resumable `vacuum_ann`.

mod invariants;
mod maintenance;
mod rebalance;
mod split;

pub use invariants::*;
pub use maintenance::*;
pub(crate) use rebalance::*;
pub(crate) use split::*;


use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use izzydb_core::config::{NodeId, RowId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::metric::{
    cluster_distance, compute_mean_centroid_flat, incremental_mean,
    prepare_vector, validate_vector,
};
use izzydb_core::schema::{CollectionConfig, CollectionSchema};
use izzydb_rabitq::{EncodeScratch, RabitqQuantizer};
use izzydb_storage::codec::{Manifest, RowView, SegmentMeta};
use izzydb_storage::node::{AnnMeta, LeafNodeView, PackedInternalNode, PackedLeafNode};
use izzydb_storage::{SegmentTableSnapshot, Storage, TxContext};

use crate::TreeStats;
use crate::lire;
use crate::search::{self, SearchOptions};
use crate::segments::RowIndexHook;

const SMALL_LEAF_REENCODE_THRESHOLD: usize = 16;
const MIN_COMPACT_SLOT_CAP: usize = 32;
const VACUUM_TX_BATCH: usize = 64;

/// Summary of a child node passed during root creation, parent attachment, and redistribution.
#[derive(Clone, Copy)]
struct ChildSummary<'a> {
    id: NodeId,
    subtree_count: u64,
    centroid: &'a [f32],
}

/// Metadata about a completed leaf split passed to SPFresh LIRE.
#[derive(Debug, Clone)]
pub(crate) struct SplitPair {
    pub left_id: NodeId,
    pub right_id: NodeId,
    pub old_centroid: Vec<f32>,
}

/// Stateless global ANN hook operating inside `Collection`'s [`TxContext`] over `RowId`s.
#[derive(Debug, Clone)]
pub struct AnnIndex {
    pub(crate) config: TreeConfig,
    pub(crate) quantizer: RabitqQuantizer,
}

impl AnnIndex {
    pub fn new(config: TreeConfig, rabitq_seed: u64) -> Result<Self> {
        config.validate()?;
        let quantizer = RabitqQuantizer::new(config.dim, rabitq_seed);
        Ok(Self { config, quantizer })
    }
}

impl RowIndexHook for AnnIndex {
    fn on_insert(
        &self,
        tx: &mut TxContext<'_>,
        _schema: &CollectionSchema,
        row: RowId,
        _view: &RowView<'_>,
        vector: Option<&[f32]>,
        _prev_row: Option<RowId>,
    ) -> Result<()> {
        let Some(vec) = vector else {
            return Ok(());
        };
        let split_opt = insert_in_tx(tx, row, vec, &self.config, &self.quantizer)?;
        if let Some(split_pair) = split_opt
            && self.config.enable_lire
            && tx.ann_meta().is_some_and(|m| m.root_level > 0)
        {
            lire::run_lire_in_tx(tx, split_pair, &self.config, &self.quantizer)?;
        }
        Ok(())
    }

    fn on_delete(
        &self,
        tx: &mut TxContext<'_>,
        _schema: &CollectionSchema,
        _row: RowId,
        _view: &RowView<'_>,
        had_vector: bool,
    ) -> Result<()> {
        if had_vector {
            let meta = tx.ann_meta_mut()?;
            meta.stale_rows += 1;
            meta.num_vectors = meta.num_vectors.saturating_sub(1);
        }
        Ok(())
    }
}

/// Standalone `RowId`-indexed ANN tree wrapper (used by `tree_tests` without documents).
pub struct ClusteringTree {
    pub(crate) storage: Arc<Storage>,
    pub(crate) ann: AnnIndex,
    pub(crate) manifest: Manifest,
    pub(crate) meta: AnnMeta,
    pub(crate) internal_cache: HashMap<NodeId, PackedInternalNode>,
    pub(crate) leaf_parent: HashMap<NodeId, NodeId>,
}

impl ClusteringTree {
    /// Open or create a standalone [`ClusteringTree`] at `path` with the given `TreeConfig`.
    pub fn open(path: impl AsRef<Path>, config: TreeConfig) -> Result<Self> {
        config.validate()?;
        let col_cfg = CollectionConfig::new(config.clone());
        let (storage, manifest, ann_meta) = Storage::open(path, &col_cfg)?;
        let meta = ann_meta.ok_or_else(|| {
            TreeError::InvalidConfig("ClusteringTree requires vector index".into())
        })?;
        let storage = Arc::new(storage);
        let ann = AnnIndex::new(config, meta.rabitq_seed)?;
        let (internal_cache, leaf_parent) = load_internal_caches(&storage)?;
        Ok(Self {
            storage,
            ann,
            manifest,
            meta,
            internal_cache,
            leaf_parent,
        })
    }

    pub fn config(&self) -> &TreeConfig {
        &self.ann.config
    }

    pub fn meta(&self) -> &AnnMeta {
        &self.meta
    }

    pub fn len(&self) -> u64 {
        self.meta.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.meta.num_vectors == 0
    }

    pub fn get_vector(&self, row: RowId) -> Result<Option<Vec<f32>>> {
        if !self.storage.segment_snapshot().is_local_live_row(row) {
            return Ok(None);
        }
        self.storage.get_vector(row)
    }

    /// Insert a vector at an explicit `RowId`, registering its segment metadata as Open if needed.
    pub fn insert_row(&mut self, row: RowId, embedding: &[f32]) -> Result<()> {
        let vec = prepare_vector(embedding, self.ann.config.dim, self.ann.config.metric)?;
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            Some(self.meta.clone()),
        );
        let seg = row.seg();
        let mut seg_meta = tx
            .get_seg_meta(seg)
            .unwrap_or_else(|| SegmentMeta::new_open(0));
        if seg_meta.row_count <= row.local() {
            seg_meta.row_count = row.local() + 1;
        }
        tx.put_seg_meta(seg, seg_meta);
        tx.put_vector(row, vec.clone());

        let split_opt = insert_in_tx(&mut tx, row, &vec, &self.ann.config, &self.ann.quantizer)?;
        if let Some(split_pair) = split_opt
            && self.ann.config.enable_lire
            && tx.ann_meta().is_some_and(|m| m.root_level > 0)
        {
            lire::run_lire_in_tx(&mut tx, split_pair, &self.ann.config, &self.ann.quantizer)?;
        }

        let (manifest, ann_meta) = tx.commit()?;
        self.manifest = manifest;
        self.meta = ann_meta.expect("ann_meta");
        Ok(())
    }

    /// Lazily tombstone a `RowId` (no leaf I/O; increments `stale_rows`, decrements `num_vectors`).
    pub fn tombstone_row(&mut self, row: RowId) -> Result<bool> {
        if !self.storage.segment_snapshot().is_local_live_row(row) {
            return Ok(false);
        }
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            Some(self.meta.clone()),
        );
        tx.tombstone_row(row)?;
        let meta = tx.ann_meta_mut()?;
        meta.stale_rows += 1;
        meta.num_vectors = meta.num_vectors.saturating_sub(1);
        let (manifest, ann_meta) = tx.commit()?;
        self.manifest = manifest;
        self.meta = ann_meta.expect("ann_meta");
        Ok(true)
    }

    /// Validate `embedding` dimensionality/finiteness and tombstone `row`.
    pub fn tombstone_row_checked(&mut self, row: RowId, embedding: &[f32]) -> Result<bool> {
        validate_vector(embedding, self.ann.config.dim)?;
        self.tombstone_row(row)
    }

    /// Vacuum stale leaf entries and rebalance underflowing leaves up to `budget`.
    pub fn vacuum_ann(&mut self, budget: VacuumBudget) -> Result<VacuumStats> {
        let mut ann_meta = Some(self.meta.clone());
        let stats = vacuum_ann_storage(
            &self.storage,
            &self.ann,
            &mut self.manifest,
            &mut ann_meta,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            budget,
        )?;
        self.meta = ann_meta.expect("ann_meta");
        Ok(stats)
    }

    pub fn search(&self, query: &[f32], k: usize, beam_width: usize) -> Result<Vec<(RowId, f32)>> {
        search::search(
            &self.storage,
            &self.internal_cache,
            &self.meta,
            &self.ann.quantizer,
            query,
            SearchOptions::new(k, beam_width),
            &self.ann.config,
        )
    }

    pub fn search_with_rerank(
        &self,
        query: &[f32],
        k: usize,
        beam_width: usize,
        rerank_candidates: usize,
    ) -> Result<Vec<(RowId, f32)>> {
        search::search(
            &self.storage,
            &self.internal_cache,
            &self.meta,
            &self.ann.quantizer,
            query,
            SearchOptions::new(k, beam_width).with_rerank(rerank_candidates),
            &self.ann.config,
        )
    }

    pub fn verify_invariants(&self) -> Result<TreeStats> {
        verify_tree_invariants(
            &self.storage,
            &self.internal_cache,
            &self.leaf_parent,
            &self.meta,
            &self.ann.config,
        )
    }
}

trait SegmentSnapshotExt {
    fn is_local_live_row(&self, row: RowId) -> bool;
}

impl SegmentSnapshotExt for SegmentTableSnapshot {
    fn is_local_live_row(&self, row: RowId) -> bool {
        self.resolve_row(row) == Some(row)
    }
}

/// Load `internal_cache` and `leaf_parent` from `storage`.
pub(crate) fn load_internal_caches(
    storage: &Storage,
) -> Result<(HashMap<NodeId, PackedInternalNode>, HashMap<NodeId, NodeId>)> {
    let internal_cache: HashMap<NodeId, PackedInternalNode> =
        storage.all_internal_nodes()?.into_iter().collect();
    let mut leaf_parent = HashMap::new();
    for (&nid, node) in &internal_cache {
        if node.level == 1 {
            for &child_id in &node.child_ids {
                leaf_parent.insert(child_id, nid);
            }
        }
    }
    Ok((internal_cache, leaf_parent))
}

/// Returns true when a leaf should compact its slots to `0..live_count`.
#[inline]
pub(crate) fn needs_leaf_compaction(leaf: &PackedLeafNode, config: &TreeConfig) -> bool {
    leaf.slot_cap > config.max_leaf_size
        || (leaf.slot_cap > MIN_COMPACT_SLOT_CAP
            && leaf.live_count.saturating_mul(2) <= leaf.slot_cap)
}

#[inline]
fn needs_leaf_compaction_view(view: &LeafNodeView<'_>, config: &TreeConfig) -> bool {
    view.slot_cap > config.max_leaf_size
        || (view.slot_cap > MIN_COMPACT_SLOT_CAP
            && view.live_count.saturating_mul(2) <= view.slot_cap)
}

/// Drop stale slots and remap forwarded `RowId`s in `leaf` against `tx.resolve_row`.
/// Returns `(stale_dropped, rows_remapped)`.
pub(crate) fn prune_and_remap_leaf(
    tx: &mut TxContext<'_>,
    leaf: &mut PackedLeafNode,
    rabitq_words: usize,
) -> Result<(u64, u64)> {
    let pairs = leaf.live_slot_row_pairs();
    let mut dropped = 0u64;
    let mut remapped = 0u64;
    for (slot, row_id) in pairs {
        match tx.resolve_row(row_id) {
            None => {
                leaf.remove_slot(slot as usize, rabitq_words);
                dropped += 1;
            }
            Some(new_row) if new_row != row_id => {
                leaf.row_ids[slot as usize] = new_row.raw();
                remapped += 1;
            }
            Some(_) => {}
        }
    }
    if dropped > 0 {
        let meta = tx.ann_meta_mut()?;
        meta.stale_rows = meta.stale_rows.saturating_sub(dropped);
    }
    leaf.stale_hint = 0;
    Ok((dropped, remapped))
}

/// Check on a borrowed [`LeafNodeView`] if any slot needs pruning or remapping.
#[inline]
fn leaf_view_has_stale_or_forwarded(
    view: &LeafNodeView<'_>,
    seg_snap: &SegmentTableSnapshot,
    dropped: &roaring::RoaringBitmap,
) -> bool {
    for (w_idx, &chunk) in view.live_mask_bytes.as_chunks::<8>().0.iter().enumerate() {
        let mut word = u64::from_le_bytes(chunk);
        while word != 0 {
            let b = word.trailing_zeros() as usize;
            word &= word - 1;
            let slot = (w_idx << 6) | b;
            if slot < view.slot_cap {
                let row = view.row_id(slot);
                match seg_snap.resolve_row(row) {
                    Some(resolved) if !dropped.contains(resolved.seg()) && resolved == row => {}
                    _ => return true,
                }
            }
        }
    }
    false
}

/// Insert a prepared vector for `row` (already persisted in `cf_vectors` via `tx`) into the ANN tree.
/// Vacuums stale slots from an overflowing target leaf before splitting.
pub(crate) fn insert_in_tx(
    tx: &mut TxContext<'_>,
    row: RowId,
    vec: &[f32],
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<Option<SplitPair>> {
    let target_leaf = {
        let meta = tx
            .ann_meta()
            .ok_or_else(|| TreeError::InvalidConfig("collection has no vector index".into()))?;
        if meta.root_level == 0 {
            meta.root_id
        } else {
            search::route_to_leaf_in_tx(tx, vec, config.insert_beam_width, config)?
        }
    };

    let mut leaf = tx.get_leaf_node(target_leaf)?;
    let old_centroid = leaf.centroid.clone();
    tx.ann_meta_mut()?.num_vectors += 1;

    let words = quantizer.num_words();
    let mut dropped_on_overflow = 0u64;
    let mut remapped_on_overflow = 0u64;
    if leaf.entry_count() + 1 > config.max_leaf_size {
        let (d, r) = prune_and_remap_leaf(tx, &mut leaf, words)?;
        dropped_on_overflow = d;
        remapped_on_overflow = r;
    }

    let new_count = leaf.entry_count() + 1;
    if new_count <= config.max_leaf_size {
        let needs_compact = leaf.slot_cap > config.max_leaf_size
            || (dropped_on_overflow > 0 && needs_leaf_compaction(&leaf, config));
        if new_count == 1 {
            leaf.centroid = vec.to_vec();
            let mut scratch = EncodeScratch::default();
            leaf.reset_and_encode(&[row], vec, quantizer, config.dim, &mut scratch);
        } else if needs_compact || dropped_on_overflow > 0 {
            leaf.insert_slot_for_reencode(row, words);
            let live_rows = leaf.live_row_ids();
            let flat_vecs = tx.get_vectors_flat(&live_rows)?;
            leaf.centroid = compute_mean_centroid_flat(
                &flat_vecs,
                new_count,
                config.dim,
                config.metric,
            );
            let mut scratch = EncodeScratch::default();
            if needs_compact {
                leaf.reset_and_encode(&live_rows, &flat_vecs, quantizer, config.dim, &mut scratch);
            } else {
                leaf.re_encode_existing(&flat_vecs, quantizer, config.dim, &mut scratch);
            }
        } else if new_count.is_multiple_of(SMALL_LEAF_REENCODE_THRESHOLD) {
            leaf.insert_slot_for_reencode(row, words);
            let live_rows = leaf.live_row_ids();
            let flat_vecs = tx.get_vectors_flat(&live_rows)?;
            leaf.centroid = compute_mean_centroid_flat(
                &flat_vecs,
                new_count,
                config.dim,
                config.metric,
            );
            let mut scratch = EncodeScratch::default();
            leaf.re_encode_existing(&flat_vecs, quantizer, config.dim, &mut scratch);
        } else {
            leaf.centroid = incremental_mean(&leaf.centroid, vec, new_count, config.metric);
            let drifted = cluster_distance(config.metric, &old_centroid, &leaf.centroid)
                > config.centroid_drift_eps;
            if new_count <= SMALL_LEAF_REENCODE_THRESHOLD || drifted {
                leaf.insert_slot_for_reencode(row, words);
                let live_rows = leaf.live_row_ids();
                let flat_vecs = tx.get_vectors_flat(&live_rows)?;
                let mut scratch = EncodeScratch::default();
                leaf.re_encode_existing(&flat_vecs, quantizer, config.dim, &mut scratch);
            } else {
                quantizer.rotate_into(&leaf.centroid, &mut leaf.rotated_centroid);
                let mut scratch = EncodeScratch::default();
                let mut code = vec![0u64; words];
                let (res_norm_sq, scale, popcnt) =
                    quantizer.encode_into_with_popcnt(vec, &leaf.centroid, &mut scratch, &mut code);
                leaf.upsert_entry(row, None, res_norm_sq, scale, &code, popcnt, words);
            }
        }

        let _ = remapped_on_overflow;
        let parent_id = leaf.parent_id;
        let new_centroid = leaf.centroid.clone();
        tx.put_leaf_node(target_leaf, leaf);

        let root_id = tx.ann_meta().unwrap().root_id;
        if target_leaf != root_id && new_count < config.min_leaf_size {
            handle_underflow(tx, target_leaf, config, quantizer)?;
        } else if let Some(pid) = parent_id {
            propagate_centroid_upward(
                tx,
                target_leaf,
                pid,
                new_count as u64,
                &old_centroid,
                &new_centroid,
                config,
            )?;
        }
        Ok(None)
    } else {
        leaf.insert_slot_for_reencode(row, words);
        tx.put_leaf_node(target_leaf, leaf);
        let split_pair = split_node(tx, target_leaf, config, quantizer)?;
        Ok(Some(split_pair))
    }
}
