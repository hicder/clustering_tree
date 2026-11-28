//! Top-level collection invariant verification.

use std::collections::{BTreeMap, HashMap, HashSet};
use roaring::RoaringBitmap;
use izzydb_core::config::{
    AttrId, DocId, NodeId, RowId, SegmentId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{CollectionConfig, CollectionSchema};
use izzydb_storage::codec::{
    Manifest, SEG_DELETES_TAG, SEG_META_TAG, SegmentMeta, SegmentState, ZoneEntry, decode_doc_id_block, decode_doc_id_block_key, decode_row_key, RowView,
};
use izzydb_storage::node::{AnnMeta, PackedInternalNode};
use izzydb_storage::{
    CF_ROW_DOC_IDS, CF_ROWS, CF_VECTORS, Storage,
};
use crate::tree::verify_tree_invariants;
use crate::{CollectionStats, TreeStats};

use super::*;

/// Exhaustively verify every storage, index, and metadata invariant of a v3 [`crate::Collection`].
#[allow(clippy::too_many_arguments)]
pub fn verify_collection_invariants(
    storage: &Storage,
    internal_cache: &HashMap<NodeId, PackedInternalNode>,
    leaf_parent: &HashMap<NodeId, NodeId>,
    mem_manifest: &Manifest,
    mem_ann_meta: Option<&AnnMeta>,
    config: &CollectionConfig,
    mem_schema: &CollectionSchema,
) -> Result<CollectionStats> {
    let disk_schema = storage.get_schema()?.ok_or_else(|| {
        TreeError::InvariantViolation("missing \\x01schema on disk".into())
    })?;
    if &disk_schema != mem_schema {
        return Err(TreeError::InvariantViolation(
            "in-memory CollectionSchema differs from disk".into(),
        ));
    }

    let disk_manifest = storage.get_manifest()?.ok_or_else(|| {
        TreeError::InvariantViolation("missing \\x01manifest on disk".into())
    })?;
    if &disk_manifest != mem_manifest {
        return Err(TreeError::InvariantViolation(
            "in-memory Manifest differs from disk".into(),
        ));
    }
    if disk_manifest.seal_rows != config.segment.seal_rows {
        return Err(TreeError::InvariantViolation(format!(
            "Manifest seal_rows {} != config {}",
            disk_manifest.seal_rows, config.segment.seal_rows
        )));
    }
    if disk_manifest.live_rows > disk_manifest.total_rows {
        return Err(TreeError::InvariantViolation(format!(
            "Manifest live_rows {} > total_rows {}",
            disk_manifest.live_rows, disk_manifest.total_rows
        )));
    }

    // 1. SegmentTable <-> 0x11 <-> 0x10 <-> Manifest consistency + dropped segments exclusion.
    let seg_snap = storage.segment_snapshot();
    let dropped = storage.dropped_segments_snapshot();

    let mut disk_seg_metas: BTreeMap<SegmentId, SegmentMeta> = BTreeMap::new();
    storage.visit_default(&[SEG_META_TAG], |suffix, val| {
        if suffix.len() != 4 {
            return Err(TreeError::InvariantViolation(
                "0x11 SegmentMeta key suffix expected 4 bytes".into(),
            ));
        }
        let seg = SegmentId::from_be_bytes(suffix.try_into().unwrap());
        let meta = SegmentMeta::decode(val)?;
        disk_seg_metas.insert(seg, meta);
        Ok(())
    })?;

    if disk_seg_metas.len() != seg_snap.segments.len() {
        return Err(TreeError::InvariantViolation(format!(
            "disk 0x11 count {} != SegmentTableSnapshot count {}",
            disk_seg_metas.len(),
            seg_snap.segments.len()
        )));
    }

    let mut open_segs = Vec::new();
    let mut visible_live_total = 0u64;
    let mut max_seg_id: Option<SegmentId> = None;

    for (&seg, disk_meta) in &disk_seg_metas {
        if dropped.contains(seg) {
            return Err(TreeError::InvariantViolation(format!(
                "dropped segment {seg} still present in 0x11 / SegmentTable"
            )));
        }
        max_seg_id = Some(max_seg_id.map_or(seg, |m| m.max(seg)));
        let Some(entry) = seg_snap.segments.get(&seg) else {
            return Err(TreeError::InvariantViolation(format!(
                "segment {seg} present on disk 0x11 but missing in SegmentTable"
            )));
        };
        if &entry.meta != disk_meta {
            return Err(TreeError::InvariantViolation(format!(
                "segment {seg} SegmentMeta in-memory differs from disk"
            )));
        }
        if disk_meta.state == SegmentState::Building {
            return Err(TreeError::InvariantViolation(format!(
                "segment {seg} left in SegmentState::Building"
            )));
        }
        if disk_meta.state == SegmentState::Open {
            open_segs.push(seg);
        }
        if entry.is_visible() {
            visible_live_total += (disk_meta.row_count - disk_meta.deleted_count) as u64;
        }
    }

    if let Some(m) = max_seg_id
        && disk_manifest.next_segment_id <= m
    {
        return Err(TreeError::InvariantViolation(format!(
            "Manifest next_segment_id {} <= max segment id {m}",
            disk_manifest.next_segment_id
        )));
    }
    if open_segs.len() > 1 {
        return Err(TreeError::InvariantViolation(format!(
            "multiple Open segments found: {open_segs:?}"
        )));
    }
    if open_segs.first().copied() != disk_manifest.open_segment {
        return Err(TreeError::InvariantViolation(format!(
            "open_segment mismatch: disk 0x11 {:?} vs Manifest {:?}",
            open_segs.first().copied(),
            disk_manifest.open_segment
        )));
    }
    if visible_live_total != disk_manifest.live_rows {
        return Err(TreeError::InvariantViolation(format!(
            "visible segment live rows {visible_live_total} != Manifest.live_rows {}",
            disk_manifest.live_rows
        )));
    }

    let mut disk_deletes: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    storage.visit_default(&[SEG_DELETES_TAG], |suffix, val| {
        if suffix.len() != 4 {
            return Err(TreeError::InvariantViolation(
                "0x10 deletes key suffix expected 4 bytes".into(),
            ));
        }
        let seg = SegmentId::from_be_bytes(suffix.try_into().unwrap());
        let bm = RoaringBitmap::deserialize_from(val)
            .map_err(|e| TreeError::CorruptedData(format!("seg {seg} 0x10 bitmap: {e}")))?;
        disk_deletes.insert(seg, bm);
        Ok(())
    })?;

    let mut live_by_seg: HashMap<SegmentId, RoaringBitmap> = HashMap::new();
    for (&seg, entry) in &seg_snap.segments {
        let bm_opt = disk_deletes.get(&seg);
        let mem_bm = entry.deleted.as_deref();
        if bm_opt != mem_bm {
            return Err(TreeError::InvariantViolation(format!(
                "segment {seg} 0x10 bitmap differs from SegmentTable in-memory"
            )));
        }
        let del_len = bm_opt.map_or(0, |b| b.len());
        if del_len != entry.meta.deleted_count as u64 {
            return Err(TreeError::InvariantViolation(format!(
                "segment {seg} 0x10 bitmap cardinality {del_len} != deleted_count {}",
                entry.meta.deleted_count
            )));
        }
        if let Some(max_del) = bm_opt.and_then(|b| b.max())
            && max_del >= entry.meta.row_count
        {
            return Err(TreeError::InvariantViolation(format!(
                "segment {seg} 0x10 contains local {max_del} >= row_count {}",
                entry.meta.row_count
            )));
        }
        if entry.is_visible() {
            let mut live_bm = RoaringBitmap::new();
            live_bm.insert_range(0..entry.meta.row_count);
            if let Some(del) = bm_opt {
                live_bm -= del;
            }
            live_by_seg.insert(seg, live_bm);
        }
    }

    // 2. ANN invariants (when configured).
    let tree: Option<TreeStats> = match (&config.vector, mem_ann_meta) {
        (Some(vcfg), Some(am)) => Some(verify_tree_invariants(
            storage,
            internal_cache,
            leaf_parent,
            am,
            vcfg,
        )?),
        (Some(_), None) => {
            return Err(TreeError::InvariantViolation(
                "missing in-memory AnnMeta for vector collection".into(),
            ));
        }
        (None, Some(_)) => {
            return Err(TreeError::InvariantViolation(
                "unexpected in-memory AnnMeta for vectorless collection".into(),
            ));
        }
        (None, None) => {
            if storage.get_ann_meta()?.is_some() {
                return Err(TreeError::InvariantViolation(
                    "unexpected \\x01ann_meta on disk for vectorless collection".into(),
                ));
            }
            let vec_keys = storage.count_keys_in_cf(Some(CF_VECTORS), |_| true)?;
            if vec_keys != 0 {
                return Err(TreeError::InvariantViolation(format!(
                    "unexpected {vec_keys} keys in cf_vectors for vectorless collection"
                )));
            }
            None
        }
    };

    // 3. cf_rows <-> cf_row_doc_ids <-> cf_id_map bijection & vector presence.
    let mut cf_rows_visible: HashMap<RowId, Vec<u8>> = HashMap::new();
    let mut row_counts_per_seg: HashMap<SegmentId, u32> = HashMap::new();
    storage.visit_cf_raw(CF_ROWS, |k, v| {
        let row = decode_row_key(k)?;
        let seg = row.seg();
        if let Some(entry) = seg_snap.segments.get(&seg) {
            if row.local() >= entry.meta.row_count {
                return Err(TreeError::InvariantViolation(format!(
                    "cf_rows key {row:?} >= row_count {}",
                    entry.meta.row_count
                )));
            }
            if entry.is_visible() {
                *row_counts_per_seg.entry(seg).or_insert(0) += 1;
                cf_rows_visible.insert(row, v.to_vec());
            }
        } else if !dropped.contains(seg) {
            return Err(TreeError::InvariantViolation(format!(
                "cf_rows key {row:?} belongs to unknown non-dropped segment"
            )));
        }
        Ok(())
    })?;

    for (&seg, entry) in &seg_snap.segments {
        if entry.is_visible() {
            let cnt = row_counts_per_seg.get(&seg).copied().unwrap_or(0);
            if cnt != entry.meta.row_count {
                return Err(TreeError::InvariantViolation(format!(
                    "visible segment {seg} cf_rows count {cnt} != row_count {}",
                    entry.meta.row_count
                )));
            }
        }
    }

    let mut row_to_doc_id: HashMap<RowId, DocId> = HashMap::new();
    storage.visit_cf_raw(CF_ROW_DOC_IDS, |k, v| {
        let (seg, blk) = decode_doc_id_block_key(k)?;
        let Some(entry) = seg_snap.segments.get(&seg) else {
            if dropped.contains(seg) {
                return Ok(());
            }
            return Err(TreeError::InvariantViolation(format!(
                "cf_row_doc_ids block ({seg}, {blk}) belongs to unknown non-dropped segment"
            )));
        };
        if !entry.is_visible() {
            return Ok(());
        }
        let ids = decode_doc_id_block(v)?;
        let base_local = blk * 256;
        let max_slot = (entry.meta.row_count.saturating_sub(base_local)).min(256);
        for slot in 0..max_slot {
            let r = RowId::new(seg, base_local + slot);
            if !entry.deleted.as_ref().is_some_and(|bm| bm.contains(r.local())) {
                row_to_doc_id.insert(r, ids[slot as usize]);
            }
        }
        Ok(())
    })?;

    let mut live_rows: HashMap<RowId, (DocId, Vec<u8>)> =
        HashMap::with_capacity(disk_manifest.live_rows as usize);
    for (&seg, live_bm) in &live_by_seg {
        for local in live_bm.iter() {
            let row = RowId::new(seg, local);
            let bytes = cf_rows_visible.remove(&row).ok_or_else(|| {
                TreeError::InvariantViolation(format!("live row {row:?} missing in cf_rows"))
            })?;
            let doc_id = *row_to_doc_id.get(&row).ok_or_else(|| {
                TreeError::InvariantViolation(format!(
                    "live row {row:?} missing in cf_row_doc_ids"
                ))
            })?;
            live_rows.insert(row, (doc_id, bytes));
        }
    }

    let id_map = storage.all_id_map()?;
    if id_map.len() != live_rows.len() {
        return Err(TreeError::InvariantViolation(format!(
            "cf_id_map count {} != live_rows count {}",
            id_map.len(),
            live_rows.len()
        )));
    }
    let mut seen_rows_from_id_map: HashSet<RowId> = HashSet::with_capacity(id_map.len());
    for (doc_id, row) in &id_map {
        let Some(&(exp_doc_id, _)) = live_rows.get(row) else {
            return Err(TreeError::InvariantViolation(format!(
                "cf_id_map ({doc_id} -> {row:?}) points to dead, forwarding, or non-existent row"
            )));
        };
        if exp_doc_id != *doc_id {
            return Err(TreeError::InvariantViolation(format!(
                "cf_id_map ({doc_id} -> {row:?}) disagrees with cf_row_doc_ids ({exp_doc_id})"
            )));
        }
        if !seen_rows_from_id_map.insert(*row) {
            return Err(TreeError::InvariantViolation(format!(
                "duplicate RowId {row:?} target in cf_id_map"
            )));
        }
    }

    let vectors_map: HashMap<RowId, Vec<f32>> = if config.vector.is_some() {
        storage.all_vectors()?.into_iter().collect()
    } else {
        HashMap::new()
    };
    for (&row, (_, raw_bytes)) in &live_rows {
        let view = RowView::parse(raw_bytes)?;
        if view.has_vector() != vectors_map.contains_key(&row) {
            return Err(TreeError::InvariantViolation(format!(
                "live row {row:?} RowView.has_vector={} != cf_vectors presence",
                view.has_vector()
            )));
        }
    }

    // 3b. Leaf-colocated scalar columns and cached leaf/internal filter summaries.
    if config.vector.is_some() {
        verify_leaf_attrs(storage, mem_schema)?;
    }

    // 4. Attribute postings, counts (0x06), zones (0x12), and cf_columns.
    let mut expected_zones: HashMap<(SegmentId, AttrId, u8), ZoneEntry> = HashMap::new();
    let num_posting_bitmaps = verify_inline_attributes_and_columns(
        storage,
        mem_schema,
        &seg_snap,
        &dropped,
        &live_by_seg,
        &live_rows,
        &mut expected_zones,
    )?;

    // 5. Detached columns, counts (0x07), and detached zones.
    verify_detached_columns(
        storage,
        mem_schema,
        &seg_snap,
        &dropped,
        &live_by_seg,
        &live_rows,
        &mut expected_zones,
    )?;

    verify_zones(storage, &seg_snap, &dropped, &expected_zones)?;

    // 6. Full-text index.
    let (num_fts_docs, num_fts_terms, num_fts_postings, num_fts_positions) =
        verify_fts_invariants(storage, mem_schema, &seg_snap, &dropped, &live_by_seg, &live_rows)?;

    Ok(CollectionStats {
        tree,
        num_docs: live_rows.len(),
        num_segments: seg_snap.segments.len(),
        num_posting_bitmaps,
        num_fts_docs,
        num_fts_terms,
        num_fts_postings,
        num_fts_positions,
    })
}
