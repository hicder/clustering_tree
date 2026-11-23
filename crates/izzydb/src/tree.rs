//! Global ANN clustering tree (~100x fanout) over `RowId`s with lazy tombstones,
//! vacuum-before-split, SPFresh LIRE boundary reassignment, and resumable `vacuum_ann`.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use izzydb_core::config::{NodeId, RowId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::kmeans::balanced_two_means;
use izzydb_core::metric::{
    cluster_distance, compute_mean_centroid_flat, compute_weighted_centroid, incremental_mean,
    prepare_vector, validate_vector,
};
use izzydb_core::schema::{CollectionConfig, CollectionSchema};
use izzydb_rabitq::{EncodeScratch, RabitqQuantizer};
use izzydb_storage::codec::{Manifest, RowView, SegmentMeta};
use izzydb_storage::node::{AnnMeta, LeafNodeView, PackedInternalNode, PackedLeafNode};
use izzydb_storage::{LeafVisitAction, SegmentTableSnapshot, Storage, TxContext};

use crate::TreeStats;
use crate::lire;
use crate::search::{self, SearchOptions};
use crate::segments::RowIndexHook;

const SMALL_LEAF_REENCODE_THRESHOLD: usize = 16;
const MIN_COMPACT_SLOT_CAP: usize = 32;
const VACUUM_TX_BATCH: usize = 64;

/// Resource budget for [`Collection::vacuum_ann`] / [`ClusteringTree::vacuum_ann`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VacuumBudget {
    /// Maximum number of leaf blobs to scan (`None` = scan until full pass finishes).
    pub max_leaves: Option<usize>,
    /// Maximum number of encoded leaf bytes to scan (`None` = unlimited).
    pub max_bytes: Option<usize>,
}

impl VacuumBudget {
    pub fn unlimited() -> Self {
        Self::default()
    }

    pub fn leaves(max_leaves: usize) -> Self {
        Self {
            max_leaves: Some(max_leaves),
            max_bytes: None,
        }
    }

    pub fn bytes(max_bytes: usize) -> Self {
        Self {
            max_leaves: None,
            max_bytes: Some(max_bytes),
        }
    }
}

/// Statistics returned by `vacuum_ann`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VacuumStats {
    pub leaves_scanned: usize,
    pub leaves_rewritten: usize,
    pub leaves_merged_or_deleted: usize,
    pub stale_slots_dropped: u64,
    pub rows_remapped: u64,
    pub completed_pass: bool,
}

/// Policy knobs for [`Collection::maintenance`].
#[derive(Debug, Clone, Copy)]
pub struct MaintenanceOptions {
    /// Trigger `vacuum_ann` when `stale_rows / (num_vectors + stale_rows) > vacuum_stale_ratio`.
    pub vacuum_stale_ratio: f64,
    pub vacuum_budget: VacuumBudget,
}

impl Default for MaintenanceOptions {
    fn default() -> Self {
        Self {
            vacuum_stale_ratio: 0.2,
            vacuum_budget: VacuumBudget::unlimited(),
        }
    }
}

/// Outcome of a [`Collection::maintenance`] pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaintenanceStats {
    pub vacuum: Option<VacuumStats>,
    pub columns_built: usize,
}

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
    for (w_idx, chunk) in view.live_mask_bytes.chunks_exact(8).enumerate() {
        let mut word = u64::from_le_bytes(chunk.try_into().unwrap());
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

/// Incrementally vacuum stale slots and handle underflowing leaves across bounded transactions.
pub(crate) fn vacuum_ann_storage(
    storage: &Storage,
    ann: &AnnIndex,
    manifest: &mut Manifest,
    ann_meta: &mut Option<AnnMeta>,
    internal_cache: &mut HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &mut HashMap<NodeId, NodeId>,
    budget: VacuumBudget,
) -> Result<VacuumStats> {
    let Some(ref mut meta_ref) = *ann_meta else {
        return Ok(VacuumStats::default());
    };

    let config = &ann.config;
    let quantizer = &ann.quantizer;
    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();
    let words = quantizer.num_words();

    let mut stats = VacuumStats::default();
    let start_cursor = meta_ref.vacuum_cursor;
    let mut cursor = start_cursor;
    let mut wrapped = false;
    let mut bytes_scanned = 0usize;

    loop {
        if budget
            .max_leaves
            .is_some_and(|max_l| stats.leaves_scanned >= max_l)
            || budget.max_bytes.is_some_and(|max_b| bytes_scanned >= max_b)
        {
            break;
        }

        let seg_snap = storage.segment_snapshot();
        let dropped = storage.dropped_segments_snapshot();
        let has_any_stale_segs =
            !seg_snap.has_deletes_or_forwarding.is_empty() || !dropped.is_empty();

        let mut dirty_batch: Vec<NodeId> = Vec::new();
        let mut hit_wrap_boundary = false;
        let root_id = meta_ref.root_id;
        let is_unlimited = budget.max_leaves.is_none() && budget.max_bytes.is_none();

        let next_cursor_opt = storage.visit_leaf_nodes_from(cursor, |leaf_id, bytes| {
            if wrapped && leaf_id >= start_cursor {
                hit_wrap_boundary = true;
                return Ok(LeafVisitAction::StopBefore);
            }
            if budget
                .max_leaves
                .is_some_and(|max_l| stats.leaves_scanned >= max_l)
                || budget
                    .max_bytes
                    .is_some_and(|max_b| bytes_scanned > 0 && bytes_scanned >= max_b)
            {
                return Ok(LeafVisitAction::StopBefore);
            }
            stats.leaves_scanned += 1;
            bytes_scanned += bytes.len();

            let view = LeafNodeView::parse(bytes, dim, padded_dim, words)?;
            let needs_work = view.stale_hint > 0
                || needs_leaf_compaction_view(&view, config)
                || (leaf_id != root_id && view.live_count < config.min_leaf_size)
                || (has_any_stale_segs && leaf_view_has_stale_or_forwarded(&view, &seg_snap, &dropped));
            let hit_budget_now = budget
                .max_leaves
                .is_some_and(|max_l| stats.leaves_scanned >= max_l)
                || budget.max_bytes.is_some_and(|max_b| bytes_scanned >= max_b);
            if needs_work {
                dirty_batch.push(leaf_id);
                if dirty_batch.len() >= VACUUM_TX_BATCH || hit_budget_now {
                    return Ok(LeafVisitAction::StopAfter);
                }
            } else if hit_budget_now {
                return Ok(LeafVisitAction::StopAfter);
            }
            Ok(LeafVisitAction::Continue)
        })?;

        let (next_cursor, pass_finished_here) = if hit_wrap_boundary {
            (0, true)
        } else {
            match next_cursor_opt {
                Some(next_id) => (next_id, false),
                None => {
                    if is_unlimited && start_cursor > 0 && !wrapped {
                        wrapped = true;
                        (0, false)
                    } else {
                        (0, true)
                    }
                }
            }
        };

        if !dirty_batch.is_empty() || next_cursor != meta_ref.vacuum_cursor || pass_finished_here {
            let mut tx = TxContext::new(
                storage,
                internal_cache,
                leaf_parent,
                manifest.clone(),
                Some(meta_ref.clone()),
            );
            let mut scratch = EncodeScratch::default();
            for leaf_id in dirty_batch {
                let Some(mut leaf) = tx.try_get_leaf_node(leaf_id)? else {
                    continue;
                };
                let (dropped_slots, remapped_slots) = prune_and_remap_leaf(&mut tx, &mut leaf, words)?;
                stats.stale_slots_dropped += dropped_slots;
                stats.rows_remapped += remapped_slots;

                let cur_root_id = tx.ann_meta().unwrap().root_id;
                let count = leaf.entry_count();
                if count == 0 {
                    if leaf_id == cur_root_id {
                        leaf.centroid = vec![0.0; dim];
                        leaf.rotated_centroid.fill(0.0);
                        leaf.reset_and_encode(&[], &[], quantizer, dim, &mut scratch);
                        tx.put_leaf_node(leaf_id, leaf);
                        stats.leaves_rewritten += 1;
                    } else {
                        tx.put_leaf_node(leaf_id, leaf);
                        handle_underflow(&mut tx, leaf_id, config, quantizer)?;
                        stats.leaves_merged_or_deleted += 1;
                    }
                } else if leaf_id != cur_root_id && count < config.min_leaf_size {
                    tx.put_leaf_node(leaf_id, leaf);
                    handle_underflow(&mut tx, leaf_id, config, quantizer)?;
                    stats.leaves_merged_or_deleted += 1;
                } else if dropped_slots > 0 || remapped_slots > 0 || needs_leaf_compaction(&leaf, config) {
                    let old_c = leaf.centroid.clone();
                    if dropped_slots > 0 || needs_leaf_compaction(&leaf, config) {
                        let live_rows = leaf.live_row_ids();
                        let flat_vecs = tx.get_vectors_flat(&live_rows)?;
                        leaf.centroid = compute_mean_centroid_flat(
                            &flat_vecs,
                            count,
                            dim,
                            config.metric,
                        );
                        leaf.reset_and_encode(&live_rows, &flat_vecs, quantizer, dim, &mut scratch);
                    }
                    let parent_id = leaf.parent_id;
                    let new_c = leaf.centroid.clone();
                    tx.put_leaf_node(leaf_id, leaf);
                    if let Some(pid) = parent_id {
                        propagate_centroid_upward(
                            &mut tx,
                            leaf_id,
                            pid,
                            count as u64,
                            &old_c,
                            &new_c,
                            config,
                        )?;
                    }
                    stats.leaves_rewritten += 1;
                }
            }

            {
                let meta = tx.ann_meta_mut()?;
                meta.vacuum_cursor = next_cursor;
                if pass_finished_here {
                    meta.vacuum_epoch += 1;
                }
            }
            let (new_manifest, new_ann_meta) = tx.commit()?;
            *manifest = new_manifest;
            if let Some(m) = new_ann_meta {
                *meta_ref = m;
            }
        }

        cursor = next_cursor;
        if pass_finished_here {
            stats.completed_pass = true;
            break;
        }
    }

    Ok(stats)
}

/// Propagate centroid and `subtree_count` changes upward while centroid drift exceeds `centroid_drift_eps`.
pub(crate) fn propagate_centroid_upward(
    tx: &mut TxContext<'_>,
    start_node_id: NodeId,
    start_parent_id: NodeId,
    start_subtree_count: u64,
    old_start_centroid: &[f32],
    new_start_centroid: &[f32],
    config: &TreeConfig,
) -> Result<()> {
    if cluster_distance(config.metric, old_start_centroid, new_start_centroid)
        <= config.centroid_drift_eps
    {
        return Ok(());
    }

    let mut child_id = start_node_id;
    let mut parent_id_opt = Some(start_parent_id);
    let mut child_sub = start_subtree_count;
    let mut child_centroid = new_start_centroid.to_vec();

    while let Some(parent_id) = parent_id_opt {
        let mut parent = tx.get_internal_node(parent_id)?;
        let old_parent_centroid = parent.centroid.clone();
        parent.upsert_child(child_id, child_sub, &child_centroid, config.dim);
        parent.recompute_stats(config.dim, config.metric);

        let next_parent = parent.parent_id;
        let next_sub = parent.subtree_count;
        let next_centroid = parent.centroid.clone();
        tx.put_internal_node(parent_id, parent);

        if cluster_distance(config.metric, &old_parent_centroid, &next_centroid)
            <= config.centroid_drift_eps
        {
            break;
        }

        child_id = parent_id;
        parent_id_opt = next_parent;
        child_sub = next_sub;
        child_centroid = next_centroid;
    }

    Ok(())
}

/// Split an overflowing leaf or internal node into two balanced nodes, cascading upward as needed.
pub(crate) fn split_node(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<SplitPair> {
    let right_id = tx.alloc_node_id()?;
    if !tx.is_internal_node(node_id) {
        split_leaf(tx, node_id, right_id, config, quantizer)
    } else {
        split_internal(tx, node_id, right_id, config, quantizer)
    }
}

fn split_leaf(
    tx: &mut TxContext<'_>,
    left_id: NodeId,
    right_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<SplitPair> {
    let mut leaf = tx.get_leaf_node(left_id)?;
    let _ = prune_and_remap_leaf(tx, &mut leaf, quantizer.num_words())?;
    let live_row_ids = leaf.live_row_ids();

    let flat_vecs = tx.get_vectors_flat(&live_row_ids)?;
    let parent_id = leaf.parent_id;
    let old_centroid = leaf.centroid;
    let dim = config.dim;
    let padded_dim = quantizer.padded_dim();

    let weighted: Vec<(&[f32], u64)> = flat_vecs
        .chunks_exact(dim)
        .map(|v| (v, 1u64))
        .collect();
    let mut split = balanced_two_means(
        &weighted,
        config.min_leaf_size,
        config.max_leaf_size,
        dim,
        config.metric,
    );
    split.part1.sort_unstable();
    split.part2.sort_unstable();

    let build_half = |indices: &[usize]| -> (Vec<RowId>, Vec<f32>) {
        let mut ids = Vec::with_capacity(indices.len());
        let mut vecs = Vec::with_capacity(indices.len() * dim);
        for &idx in indices {
            ids.push(live_row_ids[idx]);
            vecs.extend_from_slice(&flat_vecs[idx * dim..(idx + 1) * dim]);
        }
        (ids, vecs)
    };
    let (left_rows, left_vecs) = build_half(&split.part1);
    let (right_rows, right_vecs) = build_half(&split.part2);

    let mut scratch = EncodeScratch::default();
    let mut encode_leaf = |centroid: Vec<f32>, ids: &[RowId], vecs: &[f32]| {
        let mut half = PackedLeafNode::empty(parent_id, dim, padded_dim);
        half.centroid = centroid;
        half.reset_and_encode(ids, vecs, quantizer, dim, &mut scratch);
        half
    };
    let mut left_leaf = encode_leaf(split.centroid1, &left_rows, &left_vecs);
    let mut right_leaf = encode_leaf(split.centroid2, &right_rows, &right_vecs);

    let left_c = left_leaf.centroid.clone();
    let right_c = right_leaf.centroid.clone();
    let left_summary = ChildSummary {
        id: left_id,
        subtree_count: left_leaf.entry_count() as u64,
        centroid: &left_c,
    };
    let right_summary = ChildSummary {
        id: right_id,
        subtree_count: right_leaf.entry_count() as u64,
        centroid: &right_c,
    };

    match parent_id {
        None => {
            let new_root_id = tx.alloc_node_id()?;
            left_leaf.parent_id = Some(new_root_id);
            right_leaf.parent_id = Some(new_root_id);
            tx.put_leaf_node(left_id, left_leaf);
            tx.put_leaf_node(right_id, right_leaf);
            create_new_root(tx, new_root_id, 1, left_summary, right_summary, config)?;
        }
        Some(pid) => {
            tx.put_leaf_node(left_id, left_leaf);
            tx.put_leaf_node(right_id, right_leaf);
            attach_to_parent(tx, pid, left_summary, right_summary, config, quantizer)?;
        }
    }

    Ok(SplitPair {
        left_id,
        right_id,
        old_centroid,
    })
}

fn split_internal(
    tx: &mut TxContext<'_>,
    left_id: NodeId,
    right_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<SplitPair> {
    let node = tx.get_internal_node(left_id)?;
    let level = node.level;
    let parent_id = node.parent_id;
    let old_centroid = node.centroid.clone();
    let dim = config.dim;

    let weighted: Vec<(&[f32], u64)> = node
        .child_centroids
        .chunks_exact(dim)
        .zip(node.child_subtree_counts.iter().copied())
        .collect();
    let mut split = balanced_two_means(
        &weighted,
        config.min_internal_fanout,
        config.max_internal_fanout,
        dim,
        config.metric,
    );
    split.part1.sort_unstable();
    split.part2.sort_unstable();

    let build_half = |indices: &[usize], centroid: Vec<f32>| -> PackedInternalNode {
        let mut half = PackedInternalNode {
            level,
            parent_id,
            subtree_count: 0,
            centroid,
            child_ids: Vec::with_capacity(indices.len()),
            child_subtree_counts: Vec::with_capacity(indices.len()),
            child_centroids: Vec::with_capacity(indices.len() * dim),
        };
        for &idx in indices {
            let sub = node.child_subtree_counts[idx];
            half.subtree_count += sub;
            half.child_ids.push(node.child_ids[idx]);
            half.child_subtree_counts.push(sub);
            half.child_centroids
                .extend_from_slice(node.child_centroid(idx, dim));
        }
        half
    };

    let mut left_node = build_half(&split.part1, split.centroid1);
    let mut right_node = build_half(&split.part2, split.centroid2);
    for &child_id in &right_node.child_ids {
        set_child_parent(tx, child_id, Some(right_id))?;
    }

    let left_c = left_node.centroid.clone();
    let right_c = right_node.centroid.clone();
    let left_summary = ChildSummary {
        id: left_id,
        subtree_count: left_node.subtree_count,
        centroid: &left_c,
    };
    let right_summary = ChildSummary {
        id: right_id,
        subtree_count: right_node.subtree_count,
        centroid: &right_c,
    };

    match parent_id {
        None => {
            let new_root_id = tx.alloc_node_id()?;
            left_node.parent_id = Some(new_root_id);
            right_node.parent_id = Some(new_root_id);
            tx.put_internal_node(left_id, left_node);
            tx.put_internal_node(right_id, right_node);
            create_new_root(
                tx,
                new_root_id,
                level + 1,
                left_summary,
                right_summary,
                config,
            )?;
        }
        Some(pid) => {
            tx.put_internal_node(left_id, left_node);
            tx.put_internal_node(right_id, right_node);
            attach_to_parent(tx, pid, left_summary, right_summary, config, quantizer)?;
        }
    }

    Ok(SplitPair {
        left_id,
        right_id,
        old_centroid,
    })
}

fn create_new_root(
    tx: &mut TxContext<'_>,
    new_root_id: NodeId,
    new_root_level: u8,
    left: ChildSummary<'_>,
    right: ChildSummary<'_>,
    config: &TreeConfig,
) -> Result<()> {
    let root_centroid = compute_weighted_centroid(
        &[
            (left.centroid, left.subtree_count),
            (right.centroid, right.subtree_count),
        ],
        config.dim,
        config.metric,
    );
    let mut root_node = PackedInternalNode {
        level: new_root_level,
        parent_id: None,
        subtree_count: left.subtree_count + right.subtree_count,
        centroid: root_centroid,
        child_ids: Vec::with_capacity(2),
        child_subtree_counts: Vec::with_capacity(2),
        child_centroids: Vec::with_capacity(2 * config.dim),
    };
    root_node.upsert_child(left.id, left.subtree_count, left.centroid, config.dim);
    root_node.upsert_child(right.id, right.subtree_count, right.centroid, config.dim);
    tx.put_internal_node(new_root_id, root_node);
    let meta = tx.ann_meta_mut()?;
    meta.root_id = new_root_id;
    meta.root_level = new_root_level;
    Ok(())
}

fn attach_to_parent(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    left: ChildSummary<'_>,
    right: ChildSummary<'_>,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let mut parent = tx.get_internal_node(parent_id)?;
    let old_parent_centroid = parent.centroid.clone();
    parent.upsert_child(left.id, left.subtree_count, left.centroid, config.dim);
    parent.upsert_child(right.id, right.subtree_count, right.centroid, config.dim);
    parent.recompute_stats(config.dim, config.metric);
    let fanout = parent.entry_count();
    let parent_pid = parent.parent_id;
    let parent_sub = parent.subtree_count;
    let new_parent_centroid = parent.centroid.clone();
    tx.put_internal_node(parent_id, parent);

    if fanout > config.max_internal_fanout {
        split_node(tx, parent_id, config, quantizer)?;
    } else if let Some(gp_id) = parent_pid {
        propagate_centroid_upward(
            tx,
            parent_id,
            gp_id,
            parent_sub,
            &old_parent_centroid,
            &new_parent_centroid,
            config,
        )?;
    }
    Ok(())
}

fn set_child_parent(
    tx: &mut TxContext<'_>,
    child_id: NodeId,
    new_parent: Option<NodeId>,
) -> Result<()> {
    if tx.is_internal_node(child_id) {
        let mut ch = tx.get_internal_node(child_id)?;
        ch.parent_id = new_parent;
        tx.put_internal_node(child_id, ch);
    } else {
        let mut leaf = tx.get_leaf_node(child_id)?;
        leaf.parent_id = new_parent;
        tx.put_leaf_node(child_id, leaf);
    }
    Ok(())
}

/// Resolve an underflowing non-root node via merge, redistribution, or empty-node deletion.
pub(crate) fn handle_underflow(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let root_id = tx
        .ann_meta()
        .ok_or_else(|| TreeError::InvalidConfig("collection has no vector index".into()))?
        .root_id;
    if node_id == root_id {
        collapse_root(tx)?;
        return Ok(());
    }

    let is_internal = tx.is_internal_node(node_id);
    let (entry_count, parent_id_opt, node_centroid) = if is_internal {
        let n = tx.get_internal_node(node_id)?;
        (n.entry_count(), n.parent_id, n.centroid)
    } else {
        let l = tx.get_leaf_node(node_id)?;
        (l.entry_count(), l.parent_id, l.centroid)
    };

    let min_size = if is_internal {
        config.min_internal_fanout
    } else {
        config.min_leaf_size
    };
    if entry_count >= min_size {
        return Ok(());
    }

    let parent_id = parent_id_opt.ok_or_else(|| {
        TreeError::InvariantViolation(format!("non-root node {node_id} missing parent_id"))
    })?;

    if entry_count == 0 {
        return delete_empty_node(tx, node_id, parent_id, is_internal, config, quantizer);
    }

    let Some(sib_id) = find_closest_sibling(tx, parent_id, node_id, &node_centroid, config)? else {
        if parent_id == root_id {
            collapse_root(tx)?;
            return Ok(());
        }
        return Err(TreeError::InvariantViolation(format!(
            "non-root internal node {parent_id} has only 1 child {node_id}"
        )));
    };

    if !is_internal {
        rebalance_underflowing_leaf(tx, node_id, sib_id, parent_id, config, quantizer)
    } else {
        rebalance_underflowing_internal(tx, node_id, sib_id, parent_id, config, quantizer)
    }
}

fn delete_empty_node(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    parent_id: NodeId,
    is_internal: bool,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    if is_internal {
        tx.delete_internal_node(node_id);
    } else {
        tx.delete_leaf_node(node_id);
    }
    let mut parent = tx.get_internal_node(parent_id)?;
    parent.remove_child(node_id, config.dim);
    tx.put_internal_node(parent_id, parent);
    refresh_parent_after_removal(tx, parent_id, config, quantizer)
}

fn find_closest_sibling(
    tx: &TxContext<'_>,
    parent_id: NodeId,
    node_id: NodeId,
    node_centroid: &[f32],
    config: &TreeConfig,
) -> Result<Option<NodeId>> {
    let parent = tx.get_internal_node_ref(parent_id)?;
    let dim = config.dim;
    Ok(parent
        .child_ids
        .iter()
        .enumerate()
        .filter(|&(_, &cid)| cid != node_id)
        .min_by(|&(ia, &id_a), &(ib, &id_b)| {
            let da = cluster_distance(config.metric, node_centroid, parent.child_centroid(ia, dim));
            let db = cluster_distance(config.metric, node_centroid, parent.child_centroid(ib, dim));
            da.total_cmp(&db).then_with(|| id_a.cmp(&id_b))
        })
        .map(|(_, &cid)| cid))
}

fn rebalance_underflowing_leaf(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    sib_id: NodeId,
    parent_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let dim = config.dim;
    let words = quantizer.num_words();
    let mut left_leaf = tx.get_leaf_node(node_id)?;
    let mut sib_leaf = tx.get_leaf_node(sib_id)?;
    let _ = prune_and_remap_leaf(tx, &mut left_leaf, words)?;
    let _ = prune_and_remap_leaf(tx, &mut sib_leaf, words)?;

    let left_ids = left_leaf.live_row_ids();
    let sib_ids = sib_leaf.live_row_ids();
    let total = left_ids.len() + sib_ids.len();

    if total == 0 {
        tx.put_leaf_node(sib_id, sib_leaf);
        delete_empty_node(tx, node_id, parent_id, false, config, quantizer)?;
        let cur_root = tx.ann_meta().unwrap().root_id;
        if sib_id != cur_root && tx.try_get_leaf_node(sib_id)?.is_some() {
            handle_underflow(tx, sib_id, config, quantizer)?;
        }
        return Ok(());
    }

    let left_vecs = tx.get_vectors_flat(&left_ids)?;
    let sib_vecs = tx.get_vectors_flat(&sib_ids)?;

    let mut combined: Vec<(RowId, &[f32])> = Vec::with_capacity(total);
    for (i, &rid) in left_ids.iter().enumerate() {
        combined.push((rid, &left_vecs[i * dim..(i + 1) * dim]));
    }
    for (i, &rid) in sib_ids.iter().enumerate() {
        combined.push((rid, &sib_vecs[i * dim..(i + 1) * dim]));
    }
    combined.sort_unstable_by_key(|&(rid, _)| rid);
    combined.dedup_by_key(|(rid, _)| *rid);
    let total = combined.len();

    let mut scratch = EncodeScratch::default();
    if total <= config.max_leaf_size {
        let mut merged_ids = Vec::with_capacity(total);
        let mut merged_vecs = Vec::with_capacity(total * dim);
        for (rid, v) in combined {
            merged_ids.push(rid);
            merged_vecs.extend_from_slice(v);
        }

        sib_leaf.centroid = compute_mean_centroid_flat(&merged_vecs, total, dim, config.metric);
        sib_leaf.reset_and_encode(&merged_ids, &merged_vecs, quantizer, dim, &mut scratch);

        let sib_sub = sib_leaf.entry_count() as u64;
        let sib_c = sib_leaf.centroid.clone();
        tx.put_leaf_node(sib_id, sib_leaf);
        tx.delete_leaf_node(node_id);

        let mut parent = tx.get_internal_node(parent_id)?;
        parent.remove_child(node_id, dim);
        parent.upsert_child(sib_id, sib_sub, &sib_c, dim);
        tx.put_internal_node(parent_id, parent);

        refresh_parent_after_removal(tx, parent_id, config, quantizer)?;

        let cur_root = tx.ann_meta().unwrap().root_id;
        if sib_id != cur_root
            && total < config.min_leaf_size
            && tx.try_get_leaf_node(sib_id)?.is_some()
        {
            handle_underflow(tx, sib_id, config, quantizer)?;
        }
        Ok(())
    } else {
        let weighted: Vec<(&[f32], u64)> = combined.iter().map(|&(_, v)| (v, 1u64)).collect();
        let mut split = balanced_two_means(
            &weighted,
            config.min_leaf_size,
            config.max_leaf_size,
            dim,
            config.metric,
        );
        split.part1.sort_unstable();
        split.part2.sort_unstable();

        let mut reencode_half = |leaf: &mut PackedLeafNode, indices: &[usize], centroid: Vec<f32>| {
            let mut ids = Vec::with_capacity(indices.len());
            let mut vecs = Vec::with_capacity(indices.len() * dim);
            for &idx in indices {
                let (rid, v) = combined[idx];
                ids.push(rid);
                vecs.extend_from_slice(v);
            }
            leaf.centroid = centroid;
            leaf.reset_and_encode(&ids, &vecs, quantizer, dim, &mut scratch);
        };
        reencode_half(&mut left_leaf, &split.part1, split.centroid1);
        reencode_half(&mut sib_leaf, &split.part2, split.centroid2);

        let left_c = left_leaf.centroid.clone();
        let sib_c = sib_leaf.centroid.clone();
        let left_summary = ChildSummary {
            id: node_id,
            subtree_count: left_leaf.entry_count() as u64,
            centroid: &left_c,
        };
        let sib_summary = ChildSummary {
            id: sib_id,
            subtree_count: sib_leaf.entry_count() as u64,
            centroid: &sib_c,
        };
        tx.put_leaf_node(node_id, left_leaf);
        tx.put_leaf_node(sib_id, sib_leaf);

        update_parent_pair(tx, parent_id, left_summary, sib_summary, config)
    }
}

fn rebalance_underflowing_internal(
    tx: &mut TxContext<'_>,
    node_id: NodeId,
    sib_id: NodeId,
    parent_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let dim = config.dim;
    let mut left_node = tx.get_internal_node(node_id)?;
    let mut sib_node = tx.get_internal_node(sib_id)?;
    let total = left_node.entry_count() + sib_node.entry_count();

    if total <= config.max_internal_fanout {
        for (i, &child_id) in left_node.child_ids.iter().enumerate() {
            sib_node.upsert_child(
                child_id,
                left_node.child_subtree_counts[i],
                left_node.child_centroid(i, dim),
                dim,
            );
            set_child_parent(tx, child_id, Some(sib_id))?;
        }
        sib_node.recompute_stats(dim, config.metric);

        let sib_sub = sib_node.subtree_count;
        let sib_c = sib_node.centroid.clone();
        tx.put_internal_node(sib_id, sib_node);
        tx.delete_internal_node(node_id);

        let mut parent = tx.get_internal_node(parent_id)?;
        parent.remove_child(node_id, dim);
        parent.upsert_child(sib_id, sib_sub, &sib_c, dim);
        tx.put_internal_node(parent_id, parent);

        refresh_parent_after_removal(tx, parent_id, config, quantizer)
    } else {
        struct ChildItem {
            id: NodeId,
            sub: u64,
            centroid: Vec<f32>,
            was_left: bool,
        }
        let mut combined: Vec<ChildItem> = Vec::with_capacity(total);
        for (i, &cid) in left_node.child_ids.iter().enumerate() {
            combined.push(ChildItem {
                id: cid,
                sub: left_node.child_subtree_counts[i],
                centroid: left_node.child_centroid(i, dim).to_vec(),
                was_left: true,
            });
        }
        for (i, &cid) in sib_node.child_ids.iter().enumerate() {
            combined.push(ChildItem {
                id: cid,
                sub: sib_node.child_subtree_counts[i],
                centroid: sib_node.child_centroid(i, dim).to_vec(),
                was_left: false,
            });
        }
        combined.sort_unstable_by_key(|item| item.id);

        let weighted: Vec<(&[f32], u64)> = combined
            .iter()
            .map(|c| (c.centroid.as_slice(), c.sub))
            .collect();
        let mut split = balanced_two_means(
            &weighted,
            config.min_internal_fanout,
            config.max_internal_fanout,
            dim,
            config.metric,
        );
        split.part1.sort_unstable();
        split.part2.sort_unstable();

        let mut refill = |target_id: NodeId,
                          node: &mut PackedInternalNode,
                          indices: &[usize],
                          centroid: Vec<f32>,
                          expect_left: bool|
         -> Result<()> {
            node.child_ids.clear();
            node.child_subtree_counts.clear();
            node.child_centroids.clear();
            node.subtree_count = 0;
            node.centroid = centroid;
            for &idx in indices {
                let item = &combined[idx];
                node.subtree_count += item.sub;
                node.child_ids.push(item.id);
                node.child_subtree_counts.push(item.sub);
                node.child_centroids.extend_from_slice(&item.centroid);
                if item.was_left != expect_left {
                    set_child_parent(tx, item.id, Some(target_id))?;
                }
            }
            Ok(())
        };
        refill(node_id, &mut left_node, &split.part1, split.centroid1, true)?;
        refill(sib_id, &mut sib_node, &split.part2, split.centroid2, false)?;

        let left_c = left_node.centroid.clone();
        let sib_c = sib_node.centroid.clone();
        let left_summary = ChildSummary {
            id: node_id,
            subtree_count: left_node.subtree_count,
            centroid: &left_c,
        };
        let sib_summary = ChildSummary {
            id: sib_id,
            subtree_count: sib_node.subtree_count,
            centroid: &sib_c,
        };
        tx.put_internal_node(node_id, left_node);
        tx.put_internal_node(sib_id, sib_node);

        update_parent_pair(tx, parent_id, left_summary, sib_summary, config)
    }
}

fn update_parent_pair(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    left: ChildSummary<'_>,
    right: ChildSummary<'_>,
    config: &TreeConfig,
) -> Result<()> {
    let mut parent = tx.get_internal_node(parent_id)?;
    let old_parent_centroid = parent.centroid.clone();
    parent.upsert_child(left.id, left.subtree_count, left.centroid, config.dim);
    parent.upsert_child(right.id, right.subtree_count, right.centroid, config.dim);
    parent.recompute_stats(config.dim, config.metric);

    let gp_id_opt = parent.parent_id;
    let parent_sub = parent.subtree_count;
    let new_parent_centroid = parent.centroid.clone();
    tx.put_internal_node(parent_id, parent);

    if let Some(gp_id) = gp_id_opt {
        propagate_centroid_upward(
            tx,
            parent_id,
            gp_id,
            parent_sub,
            &old_parent_centroid,
            &new_parent_centroid,
            config,
        )?;
    }
    Ok(())
}

fn refresh_parent_after_removal(
    tx: &mut TxContext<'_>,
    parent_id: NodeId,
    config: &TreeConfig,
    quantizer: &RabitqQuantizer,
) -> Result<()> {
    let mut parent = tx.get_internal_node(parent_id)?;
    let old_parent_centroid = parent.centroid.clone();
    parent.recompute_stats(config.dim, config.metric);
    let fanout = parent.entry_count();
    let gp_id_opt = parent.parent_id;
    let parent_sub = parent.subtree_count;
    let new_parent_centroid = parent.centroid.clone();
    tx.put_internal_node(parent_id, parent);

    let root_id = tx.ann_meta().unwrap().root_id;
    if parent_id == root_id {
        if fanout == 1 {
            collapse_root(tx)?;
        }
    } else if fanout < config.min_internal_fanout {
        handle_underflow(tx, parent_id, config, quantizer)?;
    } else if let Some(gp_id) = gp_id_opt {
        propagate_centroid_upward(
            tx,
            parent_id,
            gp_id,
            parent_sub,
            &old_parent_centroid,
            &new_parent_centroid,
            config,
        )?;
    }
    Ok(())
}

fn collapse_root(tx: &mut TxContext<'_>) -> Result<()> {
    while tx.ann_meta().is_some_and(|m| m.root_level > 0) {
        let root_id = tx.ann_meta().unwrap().root_id;
        let root = tx.get_internal_node(root_id)?;
        if root.entry_count() != 1 {
            break;
        }
        let only_child_id = root.child_ids[0];
        let child_level = if tx.is_internal_node(only_child_id) {
            let mut ch = tx.get_internal_node(only_child_id)?;
            ch.parent_id = None;
            let lvl = ch.level;
            tx.put_internal_node(only_child_id, ch);
            lvl
        } else {
            let mut leaf = tx.get_leaf_node(only_child_id)?;
            leaf.parent_id = None;
            tx.put_leaf_node(only_child_id, leaf);
            0
        };
        tx.delete_internal_node(root_id);
        let meta = tx.ann_meta_mut()?;
        meta.root_id = only_child_id;
        meta.root_level = child_level;
    }
    Ok(())
}

/// Exhaustively verify ANN structural invariants over `RowId`s and `SegmentTableSnapshot`.
pub fn verify_tree_invariants(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &HashMap<NodeId, NodeId>,
    mem_meta: &AnnMeta,
    config: &TreeConfig,
) -> Result<TreeStats> {
    let disk_meta = storage
        .get_ann_meta()?
        .ok_or_else(|| TreeError::InvariantViolation("missing \\x01ann_meta on disk".into()))?;
    if &disk_meta != mem_meta {
        return Err(TreeError::InvariantViolation(
            "in-memory AnnMeta differs from disk".into(),
        ));
    }
    if disk_meta.dim as usize != config.dim {
        return Err(TreeError::InvariantViolation(format!(
            "AnnMeta dim {} != config.dim {}",
            disk_meta.dim, config.dim
        )));
    }
    if disk_meta.metric != config.metric {
        return Err(TreeError::InvariantViolation(format!(
            "AnnMeta metric {:?} != config.metric {:?}",
            disk_meta.metric, config.metric
        )));
    }
    if disk_meta.root_id >= disk_meta.next_node_id {
        return Err(TreeError::InvariantViolation(format!(
            "root_id {} >= next_node_id {}",
            disk_meta.root_id, disk_meta.next_node_id
        )));
    }

    let disk_internals: HashMap<NodeId, PackedInternalNode> =
        storage.all_internal_nodes()?.into_iter().collect();
    if &disk_internals != internal_cache {
        return Err(TreeError::InvariantViolation(
            "in-memory internal_cache differs from cf_internal_nodes on disk".into(),
        ));
    }

    let disk_leaves: HashMap<NodeId, PackedLeafNode> =
        storage.all_leaf_nodes()?.into_iter().collect();
    let disk_vectors: HashMap<RowId, Vec<f32>> = storage.all_vectors()?.into_iter().collect();
    let seg_snap = storage.segment_snapshot();
    let dropped = storage.dropped_segments_snapshot();

    let mut visited_internal = HashSet::new();
    let mut visited_leaves = HashSet::new();
    let mut expected_leaf_parent = HashMap::new();
    let mut seen_live_rows: HashSet<RowId> = HashSet::new();
    let mut stale_count = 0u64;
    let mut min_leaf = usize::MAX;
    let mut max_leaf = 0usize;
    let mut min_internal = usize::MAX;
    let mut max_internal = 0usize;

    let mut stack = vec![(disk_meta.root_id, disk_meta.root_level, None)];
    while let Some((node_id, expected_level, expected_parent)) = stack.pop() {
        if visited_internal.contains(&node_id) || visited_leaves.contains(&node_id) {
            return Err(TreeError::InvariantViolation(format!(
                "cycle or shared node detected at {node_id}"
            )));
        }
        let is_root = node_id == disk_meta.root_id;
        if expected_level == 0 {
            visited_leaves.insert(node_id);
            if let Some(pid) = expected_parent {
                expected_leaf_parent.insert(node_id, pid);
            }
            if disk_internals.contains_key(&node_id) {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} present in internal nodes"
                )));
            }
            let leaf = disk_leaves.get(&node_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!("missing leaf {node_id}"))
            })?;
            if leaf.parent_id != expected_parent {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} parent_id {:?} != expected {expected_parent:?}",
                    leaf.parent_id
                )));
            }
            if leaf.centroid.len() != config.dim || leaf.centroid.iter().any(|x| !x.is_finite()) {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} invalid centroid"
                )));
            }
            let count = leaf.entry_count();
            if leaf.slot_cap > config.max_leaf_size {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} slot_cap {} > max_leaf_size {}",
                    leaf.slot_cap, config.max_leaf_size
                )));
            }
            if is_root {
                if count > config.max_leaf_size {
                    return Err(TreeError::InvariantViolation(format!(
                        "root leaf {node_id} count {count} > max_leaf_size {}",
                        config.max_leaf_size
                    )));
                }
            } else if count < config.min_leaf_size || count > config.max_leaf_size {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} count {count} not in {}..={}",
                    config.min_leaf_size, config.max_leaf_size
                )));
            }
            min_leaf = min_leaf.min(count);
            max_leaf = max_leaf.max(count);

            let mut slot_live = 0usize;
            for slot in 0..leaf.slot_cap {
                if !leaf.is_live(slot) {
                    continue;
                }
                slot_live += 1;
                let raw_row = leaf.row_id(slot);
                match seg_snap.resolve_row(raw_row) {
                    Some(resolved) if !dropped.contains(resolved.seg()) => {
                        let vec = disk_vectors.get(&resolved).ok_or_else(|| {
                            TreeError::InvariantViolation(format!(
                                "missing vector for live {resolved:?} in cf_vectors"
                            ))
                        })?;
                        if vec.len() != config.dim || vec.iter().any(|x| !x.is_finite()) {
                            return Err(TreeError::InvariantViolation(format!(
                                "invalid vector for {resolved:?}"
                            )));
                        }
                        if !seen_live_rows.insert(resolved) {
                            return Err(TreeError::InvariantViolation(format!(
                                "duplicate live row {resolved:?} in ANN leaves"
                            )));
                        }
                    }
                    _ => {
                        stale_count += 1;
                    }
                }
                let res_norm_sq = leaf.res_norm_sqs[slot];
                let scale = leaf.scales[slot];
                if !res_norm_sq.is_finite()
                    || res_norm_sq < 0.0
                    || !scale.is_finite()
                    || scale < 0.0
                    || (res_norm_sq > 0.0 && scale <= 0.0)
                {
                    return Err(TreeError::InvariantViolation(format!(
                        "row {raw_row:?} in leaf {node_id} has invalid quantized entry"
                    )));
                }
            }
            if slot_live != count {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {node_id} live_mask popcount {slot_live} != live_count {count}"
                )));
            }
        } else {
            visited_internal.insert(node_id);
            if disk_leaves.contains_key(&node_id) {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} present in cf_leaf_entries"
                )));
            }
            let node = disk_internals.get(&node_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!("missing internal node {node_id}"))
            })?;
            if node.level != expected_level || node.parent_id != expected_parent {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} level/parent mismatch"
                )));
            }
            if node.centroid.len() != config.dim || node.centroid.iter().any(|x| !x.is_finite()) {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} invalid centroid"
                )));
            }
            let count = node.entry_count();
            if is_root {
                if count < 2 || count > config.max_internal_fanout {
                    return Err(TreeError::InvariantViolation(format!(
                        "internal root {node_id} has {count} children"
                    )));
                }
            } else if count < config.min_internal_fanout || count > config.max_internal_fanout {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} has {count} children"
                )));
            }
            min_internal = min_internal.min(count);
            max_internal = max_internal.max(count);
            let sub_sum: u64 = node.child_subtree_counts.iter().sum();
            if node.subtree_count != sub_sum {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} subtree_count {} != child sum {sub_sum}",
                    node.subtree_count
                )));
            }
            for &cid in &node.child_ids {
                stack.push((cid, expected_level - 1, Some(node_id)));
            }
        }
    }

    if leaf_parent != &expected_leaf_parent {
        return Err(TreeError::InvariantViolation(
            "in-memory leaf_parent cache differs from disk".into(),
        ));
    }
    if visited_internals_or_leaves_orphaned(&disk_internals, &disk_leaves, &visited_internal, &visited_leaves) {
        return Err(TreeError::InvariantViolation(
            "orphaned tree nodes unreachable from root".into(),
        ));
    }
    if seen_live_rows.len() as u64 != disk_meta.num_vectors {
        return Err(TreeError::InvariantViolation(format!(
            "live ANN rows {} != AnnMeta.num_vectors {}",
            seen_live_rows.len(),
            disk_meta.num_vectors
        )));
    }
    if stale_count != disk_meta.stale_rows {
        return Err(TreeError::InvariantViolation(format!(
            "stale ANN slots {stale_count} != AnnMeta.stale_rows {}",
            disk_meta.stale_rows
        )));
    }

    Ok(TreeStats {
        num_vectors: disk_meta.num_vectors,
        stale_rows: disk_meta.stale_rows,
        root_id: disk_meta.root_id,
        root_level: disk_meta.root_level,
        total_nodes: visited_internal.len() + visited_leaves.len(),
        internal_nodes: visited_internal.len(),
        leaf_nodes: visited_leaves.len(),
        min_leaf_entries: if min_leaf == usize::MAX { 0 } else { min_leaf },
        max_leaf_entries: max_leaf,
        min_internal_fanout: if min_internal == usize::MAX { 0 } else { min_internal },
        max_internal_fanout: max_internal,
    })
}

fn visited_internals_or_leaves_orphaned(
    disk_internals: &HashMap<NodeId, PackedInternalNode>,
    disk_leaves: &HashMap<NodeId, PackedLeafNode>,
    visited_internal: &HashSet<NodeId>,
    visited_leaves: &HashSet<NodeId>,
) -> bool {
    disk_internals.keys().any(|id| !visited_internal.contains(id))
        || disk_leaves.keys().any(|id| !visited_leaves.contains(id))
}
