//! Exhaustive storage-format, ANN, attribute-index, columnar, detached, and FTS invariant verification.

use std::collections::{BTreeMap, HashMap, HashSet};

use roaring::RoaringBitmap;

use izzydb_core::config::{
    AttrId, DocId, NodeId, RowBlock, RowId, SegmentId, TermId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{CollectionConfig, CollectionSchema, ColumnStorage};
use izzydb_fts::{
    AttrFtsStats, DocLenBlock, FtsAnalyzer, FtsMeta, INLINE_TERM_MAX_DF, PackedPositionBlock,
    PackedPostingBlock, PostingBlockHeader, ReusableBuffer, SkipBlock, TermRecord,
    dl_over_tf_lower_bound,
};
use izzydb_storage::codec::{
    ATTR_COUNT_TAG, DETACHED_COUNT_TAG, FTS_DOC_LEN_TAG, Manifest, SEG_DELETES_TAG,
    SEG_FLAG_HAS_COLUMNS, SEG_META_TAG, SegPosting, SegmentMeta, SegmentState, TAG_BOOL,
    TAG_PRESENT, ZONE_TAG, ZoneEntry, decode_doc_id_block, decode_doc_id_block_key,
    decode_doc_len_key, decode_row_entries, decode_row_key, detached_count_suffix,
    extract_index_keys_for_attr, extract_index_keys_from_entries, make_posting_key,
    split_detached_posting_key, split_posting_key, RowView,
};
use izzydb_storage::column::ColumnBlockView;
use izzydb_storage::node::{AnnMeta, PackedInternalNode};
use izzydb_storage::{
    CF_ATTR_POSTINGS, CF_COLUMNS, CF_DETACHED, CF_DETACHED_POSTINGS, CF_FTS_POSITIONS,
    CF_FTS_POSTINGS, CF_FTS_SKIPS, CF_ROW_DOC_IDS, CF_ROWS, CF_VECTORS, Storage,
};

use crate::tree::verify_tree_invariants;
use crate::{CollectionStats, TreeStats};

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

fn verify_inline_attributes_and_columns(
    storage: &Storage,
    schema: &CollectionSchema,
    seg_snap: &izzydb_storage::SegmentTableSnapshot,
    dropped: &RoaringBitmap,
    live_by_seg: &HashMap<SegmentId, RoaringBitmap>,
    live_rows: &HashMap<RowId, (DocId, Vec<u8>)>,
    expected_zones: &mut HashMap<(SegmentId, AttrId, u8), ZoneEntry>,
) -> Result<usize> {
    let mut expected_postings: HashMap<Vec<u8>, RoaringBitmap> = HashMap::new();
    let mut expected_counts: HashMap<Vec<u8>, u64> = HashMap::new();

    for (&row, (_, raw_bytes)) in live_rows {
        let entries = decode_row_entries(raw_bytes)?;
        let keys = extract_index_keys_from_entries(schema, &entries)?;
        let seg = row.seg();
        let local = row.local();
        for ak in keys {
            *expected_counts.entry(ak.clone()).or_insert(0) += 1;
            let pkey = make_posting_key(&ak, seg);
            expected_postings.entry(pkey).or_default().insert(local);
            let aid = u16::from_be_bytes([ak[0], ak[1]]);
            let tag = ak[2];
            if ak.len() > 3
                && tag != TAG_PRESENT
                && tag != TAG_BOOL
                && schema.get_by_id(aid).is_some_and(|e| e.attr_type.is_ordered())
            {
                let z = expected_zones
                    .entry((seg, aid, tag))
                    .or_insert_with(|| ZoneEntry::singleton(&ak[3..]));
                z.widen(&ak[3..]);
            }
        }
    }

    let mut visible_disk_postings: HashMap<Vec<u8>, RoaringBitmap> = HashMap::new();
    storage.visit_cf_raw(CF_ATTR_POSTINGS, |k, v| {
        let (_, seg) = split_posting_key(k)?;
        if dropped.contains(seg) {
            return Ok(());
        }
        let Some(entry) = seg_snap.segments.get(&seg) else {
            return Err(TreeError::InvariantViolation(format!(
                "cf_attr_postings references unknown non-dropped segment {seg}"
            )));
        };
        if !entry.is_visible() {
            return Ok(());
        }
        let sp = SegPosting::decode(v)?;
        let mut live_bm = sp.into_bitmap();
        if let Some(live_seg) = live_by_seg.get(&seg) {
            live_bm &= live_seg;
        } else {
            live_bm.clear();
        }
        if !live_bm.is_empty() {
            visible_disk_postings.insert(k.to_vec(), live_bm);
        }
        Ok(())
    })?;

    if visible_disk_postings != expected_postings {
        return Err(TreeError::InvariantViolation(format!(
            "cf_attr_postings (∩ live) mismatch: disk visible non-empty={} vs expected={}",
            visible_disk_postings.len(),
            expected_postings.len()
        )));
    }

    let mut disk_counts: HashMap<Vec<u8>, u64> = HashMap::new();
    storage.visit_default(&[ATTR_COUNT_TAG], |k, v| {
        if v.len() != 8 {
            return Err(TreeError::CorruptedData("0x06 count expected 8 bytes".into()));
        }
        let cnt = u64::from_le_bytes(v.try_into().unwrap());
        if cnt > 0 {
            disk_counts.insert(k.to_vec(), cnt);
        }
        Ok(())
    })?;
    if disk_counts != expected_counts {
        return Err(TreeError::InvariantViolation(format!(
            "0x06 inline attribute counts mismatch: disk={} vs expected={}",
            disk_counts.len(),
            expected_counts.len()
        )));
    }

    // Verify cf_columns blocks against live cf_rows values for segments with SEG_FLAG_HAS_COLUMNS.
    let col_attrs: Vec<(AttrId, izzydb_core::schema::AttributeType)> = schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Inline && e.is_columnar())
        .map(|e| (e.id, e.attr_type))
        .collect();
    if !col_attrs.is_empty() {
        let mut col_blocks: HashMap<(SegmentId, AttrId, u32), Vec<u8>> = HashMap::new();
        storage.visit_cf_raw(CF_COLUMNS, |k, v| {
            if k.len() != 10 {
                return Err(TreeError::InvariantViolation(
                    "cf_columns key expected 10 bytes".into(),
                ));
            }
            let seg = SegmentId::from_be_bytes(k[0..4].try_into().unwrap());
            if dropped.contains(seg) {
                return Ok(());
            }
            let Some(entry) = seg_snap.segments.get(&seg) else {
                return Err(TreeError::InvariantViolation(format!(
                    "cf_columns key references unknown segment {seg}"
                )));
            };
            if entry.is_visible() {
                let aid = u16::from_be_bytes([k[4], k[5]]);
                let blk = u32::from_be_bytes(k[6..10].try_into().unwrap());
                col_blocks.insert((seg, aid, blk), v.to_vec());
            }
            Ok(())
        })?;

        for (&row, (_, raw_bytes)) in live_rows {
            let Some(seg_entry) = seg_snap.segments.get(&row.seg()) else {
                continue;
            };
            if (seg_entry.meta.flags & SEG_FLAG_HAS_COLUMNS) == 0 {
                continue;
            }
            let view = RowView::parse(raw_bytes)?;
            let blk = row.block_idx();
            let slot = row.slot();
            for &(aid, _) in &col_attrs {
                let row_val = view.decode_field(aid)?;
                let col_val = if let Some(bbytes) = col_blocks.get(&(row.seg(), aid, blk)) {
                    let bv = ColumnBlockView::parse(bbytes)?;
                    if bv.is_present(slot) {
                        Some(bv.decode_attr_value(slot)?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if row_val != col_val {
                    return Err(TreeError::InvariantViolation(format!(
                        "cf_columns mismatch at {row:?} attr_id={aid}: row={row_val:?} vs col={col_val:?}"
                    )));
                }
            }
        }
    }

    Ok(visible_disk_postings.len())
}

fn verify_detached_columns(
    storage: &Storage,
    schema: &CollectionSchema,
    seg_snap: &izzydb_storage::SegmentTableSnapshot,
    dropped: &RoaringBitmap,
    live_by_seg: &HashMap<SegmentId, RoaringBitmap>,
    live_rows: &HashMap<RowId, (DocId, Vec<u8>)>,
    expected_zones: &mut HashMap<(SegmentId, AttrId, u8), ZoneEntry>,
) -> Result<()> {
    let metas: HashMap<AttrId, _> = storage.all_detached_metas()?.into_iter().collect();
    for (aid, meta) in &metas {
        if meta.state != 0 || meta.pending_gen != 0 {
            return Err(TreeError::InvariantViolation(format!(
                "detached attr {aid} has non-idle state={} pending_gen={}",
                meta.state, meta.pending_gen
            )));
        }
    }

    let detached_entries: Vec<_> = schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
        .collect();
    if detached_entries.is_empty() {
        return Ok(());
    }

    let mut gen_by_attr: HashMap<AttrId, izzydb_core::config::ColumnGen> = HashMap::new();
    for entry in &detached_entries {
        let g = metas.get(&entry.id).map_or(0, |m| m.current_gen);
        gen_by_attr.insert(entry.id, g);
    }

    let mut disk_blocks: HashMap<(AttrId, SegmentId, u32), Vec<u8>> = HashMap::new();
    storage.visit_cf_raw(CF_DETACHED, |k, v| {
        if k.len() != 14 {
            return Err(TreeError::InvariantViolation(
                "cf_detached key expected 14 bytes".into(),
            ));
        }
        let aid = u16::from_be_bytes([k[0], k[1]]);
        let g = u32::from_be_bytes(k[2..6].try_into().unwrap());
        let seg = SegmentId::from_be_bytes(k[6..10].try_into().unwrap());
        let blk = u32::from_be_bytes(k[10..14].try_into().unwrap());
        if dropped.contains(seg) || gen_by_attr.get(&aid).copied() != Some(g) {
            return Ok(());
        }
        if seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            disk_blocks.insert((aid, seg, blk), v.to_vec());
        }
        Ok(())
    })?;

    let mut expected_det_postings: HashMap<Vec<u8>, RoaringBitmap> = HashMap::new();
    let mut expected_det_counts: HashMap<Vec<u8>, u64> = HashMap::new();

    for entry in &detached_entries {
        let col_gen = gen_by_attr[&entry.id];
        for &row in live_rows.keys() {
            let Some(raw_blk) = disk_blocks.get(&(entry.id, row.seg(), row.block_idx())) else {
                continue;
            };
            let bv = ColumnBlockView::parse(raw_blk)?;
            if !bv.is_present(row.slot()) {
                continue;
            }
            let val = bv.decode_attr_value(row.slot())?;
            if entry.filterable {
                for ak in extract_index_keys_for_attr(entry.id, &val)? {
                    let count_suffix = detached_count_suffix(col_gen, &ak);
                    *expected_det_counts.entry(count_suffix).or_insert(0) += 1;

                    let pkey = izzydb_storage::codec::detached_posting_key(
                        entry.id,
                        col_gen,
                        ak[2],
                        &ak[3..],
                        row.seg(),
                    );
                    expected_det_postings
                        .entry(pkey)
                        .or_default()
                        .insert(row.local());

                    let tag = ak[2];
                    if ak.len() > 3
                        && tag != TAG_PRESENT
                        && tag != TAG_BOOL
                        && entry.attr_type.is_ordered()
                    {
                        let z = expected_zones
                            .entry((row.seg(), entry.id, tag))
                            .or_insert_with(|| ZoneEntry::singleton(&ak[3..]));
                        z.widen(&ak[3..]);
                    }
                }
            }
        }
    }

    let mut overwritten_map: HashMap<(AttrId, SegmentId), RoaringBitmap> = HashMap::new();
    for entry in &detached_entries {
        let col_gen = gen_by_attr[&entry.id];
        for (seg, bm) in storage.all_detached_overwritten(entry.id, col_gen)? {
            if dropped.contains(seg) {
                continue;
            }
            if let Some(se) = seg_snap.segments.get(&seg) {
                if let Some(max_l) = bm.max()
                    && max_l >= se.meta.row_count
                {
                    return Err(TreeError::InvariantViolation(format!(
                        "0x14 detached overwritten attr={} seg={seg} local {max_l} >= row_count {}",
                        entry.id, se.meta.row_count
                    )));
                }
                if se.is_visible() && !bm.is_empty() {
                    overwritten_map.insert((entry.id, seg), bm);
                }
            }
        }
    }

    let mut disk_det_postings: HashMap<Vec<u8>, RoaringBitmap> = HashMap::new();
    storage.visit_cf_raw(CF_DETACHED_POSTINGS, |k, v| {
        let (prefix, seg) = split_detached_posting_key(k)?;
        if dropped.contains(seg) {
            return Ok(());
        }
        let aid = u16::from_be_bytes([prefix[0], prefix[1]]);
        let g = u32::from_be_bytes(prefix[2..6].try_into().unwrap());
        if gen_by_attr.get(&aid).copied() != Some(g) {
            return Ok(());
        }
        if !seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            return Ok(());
        }
        let sp = SegPosting::decode(v)?;
        let mut live_bm = sp.into_bitmap();
        if let Some(live_seg) = live_by_seg.get(&seg) {
            live_bm &= live_seg;
        } else {
            live_bm.clear();
        }
        if !live_bm.is_empty() {
            disk_det_postings.insert(k.to_vec(), live_bm);
        }
        Ok(())
    })?;

    // Check every expected posting is a subset of disk_det_postings, and any extra bits on disk
    // belong exclusively to 0x14 overwritten cells.
    for (pkey, exp_bm) in &expected_det_postings {
        let (prefix, seg) = split_detached_posting_key(pkey)?;
        let aid = u16::from_be_bytes([prefix[0], prefix[1]]);
        let disk_bm = disk_det_postings.get(pkey).ok_or_else(|| {
            TreeError::InvariantViolation(format!(
                "missing expected cf_detached_postings key for attr={aid} seg={seg}"
            ))
        })?;
        if !exp_bm.is_subset(disk_bm) {
            return Err(TreeError::InvariantViolation(format!(
                "cf_detached_postings missing live rows for attr={aid} seg={seg}"
            )));
        }
        let extra = disk_bm - exp_bm;
        if !extra.is_empty() {
            let ow = overwritten_map.get(&(aid, seg));
            if !ow.is_some_and(|b| extra.is_subset(b)) {
                return Err(TreeError::InvariantViolation(format!(
                    "cf_detached_postings has unexplained extra bits for attr={aid} seg={seg}"
                )));
            }
        }
    }
    for (pkey, disk_bm) in &disk_det_postings {
        if expected_det_postings.contains_key(pkey) {
            continue;
        }
        let (prefix, seg) = split_detached_posting_key(pkey)?;
        let aid = u16::from_be_bytes([prefix[0], prefix[1]]);
        let ow = overwritten_map.get(&(aid, seg));
        if !ow.is_some_and(|b| disk_bm.is_subset(b)) {
            return Err(TreeError::InvariantViolation(format!(
                "cf_detached_postings has unindexed stale posting for attr={aid} seg={seg}"
            )));
        }
    }

    let mut disk_det_counts: HashMap<Vec<u8>, u64> = HashMap::new();
    storage.visit_default(&[DETACHED_COUNT_TAG], |k, v| {
        if k.len() < 7 || v.len() != 8 {
            return Ok(());
        }
        let aid = u16::from_be_bytes([k[0], k[1]]);
        let g = u32::from_be_bytes(k[2..6].try_into().unwrap());
        if gen_by_attr.get(&aid).copied() != Some(g) {
            return Ok(());
        }
        let cnt = u64::from_le_bytes(v.try_into().unwrap());
        if cnt > 0 {
            disk_det_counts.insert(k.to_vec(), cnt);
        }
        Ok(())
    })?;
    if disk_det_counts != expected_det_counts {
        let missing: Vec<_> = expected_det_counts
            .iter()
            .filter(|(k, v)| disk_det_counts.get(*k) != Some(*v))
            .map(|(k, v)| (k, *v, disk_det_counts.get(k).copied()))
            .take(5)
            .collect();
        let extra: Vec<_> = disk_det_counts
            .iter()
            .filter(|(k, _)| !expected_det_counts.contains_key(*k))
            .take(5)
            .collect();
        return Err(TreeError::InvariantViolation(format!(
            "0x07 detached counts mismatch: disk={} vs expected={}, sample missing/diff={missing:?}, extra={extra:?}",
            disk_det_counts.len(),
            expected_det_counts.len()
        )));
    }

    Ok(())
}

fn verify_zones(
    storage: &Storage,
    seg_snap: &izzydb_storage::SegmentTableSnapshot,
    dropped: &RoaringBitmap,
    expected_zones: &HashMap<(SegmentId, AttrId, u8), ZoneEntry>,
) -> Result<()> {
    let mut disk_zones: HashMap<(SegmentId, AttrId, u8), ZoneEntry> = HashMap::new();
    storage.visit_default(&[ZONE_TAG], |k, v| {
        if k.len() != 7 {
            return Err(TreeError::InvariantViolation(
                "0x12 zone key suffix expected 7 bytes".into(),
            ));
        }
        let seg = SegmentId::from_be_bytes(k[0..4].try_into().unwrap());
        if dropped.contains(seg) {
            return Ok(());
        }
        if seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            let aid = u16::from_be_bytes([k[4], k[5]]);
            let tag = k[6];
            disk_zones.insert((seg, aid, tag), ZoneEntry::decode(v)?);
        }
        Ok(())
    })?;

    for (k, exp) in expected_zones {
        let Some(disk) = disk_zones.get(k) else {
            return Err(TreeError::InvariantViolation(format!(
                "missing 0x12 zone entry for {k:?}"
            )));
        };
        if disk.min > exp.min || disk.max < exp.max {
            return Err(TreeError::InvariantViolation(format!(
                "zone {k:?} does not cover live bounds"
            )));
        }
    }
    Ok(())
}

fn verify_fts_invariants(
    storage: &Storage,
    schema: &CollectionSchema,
    seg_snap: &izzydb_storage::SegmentTableSnapshot,
    dropped: &RoaringBitmap,
    live_by_seg: &HashMap<SegmentId, RoaringBitmap>,
    live_rows: &HashMap<RowId, (DocId, Vec<u8>)>,
) -> Result<(usize, usize, usize, usize)> {
    if !schema.has_fts() {
        if storage.get_fts_meta_bytes()?.is_some() {
            return Err(TreeError::InvariantViolation(
                "unexpected \\x01fts_meta for schema without FTS".into(),
            ));
        }
        return Ok((0, 0, 0, 0));
    }

    let fts_meta = match storage.get_fts_meta_bytes()? {
        Some(b) => FtsMeta::decode(&b)?,
        None => FtsMeta::default(),
    };

    let fts_entries: Vec<_> = schema
        .fts_entries()
        .filter(|e| e.storage == ColumnStorage::Inline)
        .collect();
    let analyzers: Vec<FtsAnalyzer> = fts_entries
        .iter()
        .map(|e| FtsAnalyzer::new(e.full_text_search.clone().unwrap()))
        .collect();
    let mut buf = ReusableBuffer::new();

    let mut exp_attr_stats: BTreeMap<AttrId, AttrFtsStats> = BTreeMap::new();
    let mut exp_dls: HashMap<(AttrId, RowBlock), HashMap<u8, u32>> = HashMap::new();
    // term_key -> (df, max_tf, min_dl_over_tf, RowBlock -> [(slot, tf)])
    type ExpTerm = (u64, u32, f32, BTreeMap<RowBlock, Vec<(u8, u32)>>);
    let mut exp_terms: HashMap<Vec<u8>, ExpTerm> = HashMap::new();
    // term_key -> RowBlock -> [(slot, positions)]
    type ExpPosMap = HashMap<Vec<u8>, BTreeMap<RowBlock, Vec<(u8, Vec<u32>)>>>;
    let mut exp_positions: ExpPosMap = HashMap::new();
    let mut fts_doc_count = 0usize;

    for (&row, (_, raw_bytes)) in live_rows {
        let view = RowView::parse(raw_bytes)?;
        let rblk = RowBlock::new(row.seg(), row.block_idx());
        let slot = row.slot();
        let mut doc_had_fts = false;

        for (i, entry) in fts_entries.iter().enumerate() {
            let Some(val) = view.decode_field(entry.id)? else {
                continue;
            };
            let Some(analyzed) = analyzers[i].analyze_attribute_value(&val, &mut buf)? else {
                continue;
            };
            doc_had_fts = true;
            let st = exp_attr_stats.entry(entry.id).or_default();
            st.doc_count += 1;
            st.total_tokens += analyzed.doc_len as u64;

            if analyzed.doc_len > 0 {
                exp_dls
                    .entry((entry.id, rblk))
                    .or_default()
                    .insert(slot, analyzed.doc_len);
                let idx_pos = entry.full_text_search.as_ref().unwrap().index_positions;
                for (tok, term) in analyzed.terms {
                    let tkey = izzydb_storage::codec::encode_fts_term_key(entry.id, &tok);
                    let te = exp_terms
                        .entry(tkey.clone())
                        .or_insert_with(|| (0, 0, f32::INFINITY, BTreeMap::new()));
                    te.0 += 1;
                    te.1 = te.1.max(term.tf);
                    te.2 = te.2.min(dl_over_tf_lower_bound(analyzed.doc_len, term.tf));
                    te.3.entry(rblk).or_default().push((slot, term.tf));
                    if idx_pos {
                        exp_positions
                            .entry(tkey)
                            .or_default()
                            .entry(rblk)
                            .or_default()
                            .push((slot, term.positions));
                    }
                }
            }
        }
        if doc_had_fts {
            fts_doc_count += 1;
        }
    }

    for entry in &fts_entries {
        let disk_st = fts_meta
            .per_attr_stats
            .get(&entry.id)
            .copied()
            .unwrap_or_default();
        let exp_st = exp_attr_stats.get(&entry.id).copied().unwrap_or_default();
        if disk_st != exp_st {
            return Err(TreeError::InvariantViolation(format!(
                "FTS attr_stats mismatch for '{}': disk {disk_st:?} != expected {exp_st:?}",
                entry.name
            )));
        }
    }

    // Check doc lens (0x05).
    let mut disk_dls: HashMap<(AttrId, RowBlock), DocLenBlock> = HashMap::new();
    storage.visit_default(&[FTS_DOC_LEN_TAG], |k, v| {
        let (aid, rblk) = decode_doc_len_key(k)?;
        if dropped.contains(rblk.seg()) {
            return Ok(());
        }
        if seg_snap.segments.get(&rblk.seg()).is_some_and(|e| e.is_visible()) {
            disk_dls.insert((aid, rblk), DocLenBlock::decode(v)?);
        }
        Ok(())
    })?;
    for ((aid, rblk), slot_map) in &exp_dls {
        let Some(disk_blk) = disk_dls.get(&(*aid, *rblk)) else {
            return Err(TreeError::InvariantViolation(format!(
                "missing 0x05 DocLenBlock for attr={aid} {rblk:?}"
            )));
        };
        for (&slot, &exp_dl) in slot_map {
            if disk_blk.get(slot) != exp_dl {
                return Err(TreeError::InvariantViolation(format!(
                    "DocLenBlock mismatch at attr={aid} {rblk:?} slot={slot}"
                )));
            }
        }
    }

    // Check FTS terms, skips, postings, positions (filtered to live slots in visible segments).
    let disk_terms: HashMap<Vec<u8>, Vec<u8>> = storage.all_fts_terms()?.into_iter().collect();
    let mut disk_skips: HashMap<(TermId, SegmentId, u32), Vec<u8>> = HashMap::new();
    storage.visit_cf_raw(CF_FTS_SKIPS, |k, v| {
        if k.len() != 12 {
            return Err(TreeError::InvariantViolation("12B skip key expected".into()));
        }
        let tid = u32::from_be_bytes(k[0..4].try_into().unwrap());
        let seg = SegmentId::from_be_bytes(k[4..8].try_into().unwrap());
        let chunk = u32::from_be_bytes(k[8..12].try_into().unwrap());
        if !dropped.contains(seg) && seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            disk_skips.insert((tid, seg, chunk), v.to_vec());
        }
        Ok(())
    })?;

    let mut disk_postings_by_tid: HashMap<TermId, BTreeMap<RowBlock, Vec<u8>>> = HashMap::new();
    let mut total_disk_postings = 0usize;
    storage.visit_cf_raw(CF_FTS_POSTINGS, |k, v| {
        if k.len() != 12 {
            return Err(TreeError::InvariantViolation("12B posting key expected".into()));
        }
        let tid = u32::from_be_bytes(k[0..4].try_into().unwrap());
        let seg = SegmentId::from_be_bytes(k[4..8].try_into().unwrap());
        let blk = u32::from_be_bytes(k[8..12].try_into().unwrap());
        if !dropped.contains(seg) && seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            total_disk_postings += 1;
            disk_postings_by_tid
                .entry(tid)
                .or_default()
                .insert(RowBlock::new(seg, blk), v.to_vec());
        }
        Ok(())
    })?;

    let mut disk_positions_by_tid: HashMap<TermId, BTreeMap<RowBlock, Vec<u8>>> = HashMap::new();
    let mut total_disk_positions = 0usize;
    storage.visit_cf_raw(CF_FTS_POSITIONS, |k, v| {
        if k.len() != 12 {
            return Err(TreeError::InvariantViolation("12B position key expected".into()));
        }
        let tid = u32::from_be_bytes(k[0..4].try_into().unwrap());
        let seg = SegmentId::from_be_bytes(k[4..8].try_into().unwrap());
        let blk = u32::from_be_bytes(k[8..12].try_into().unwrap());
        if !dropped.contains(seg) && seg_snap.segments.get(&seg).is_some_and(|e| e.is_visible()) {
            total_disk_positions += 1;
            disk_positions_by_tid
                .entry(tid)
                .or_default()
                .insert(RowBlock::new(seg, blk), v.to_vec());
        }
        Ok(())
    })?;

    let live_slots_for_block = |rblk: RowBlock| -> Vec<u8> {
        let Some(bm) = live_by_seg.get(&rblk.seg()) else {
            return Vec::new();
        };
        let base = rblk.block() * 256;
        bm.range(base..base + 256)
            .map(|l| (l - base) as u8)
            .collect()
    };

    let mut seen_tids = HashSet::new();
    for (tkey, (exp_df, exp_max_tf, exp_min_ratio, exp_blocks)) in &exp_terms {
        let Some(raw_rec) = disk_terms.get(tkey) else {
            return Err(TreeError::InvariantViolation(
                "missing live FTS term in cf_fts_terms".into(),
            ));
        };
        let rec = TermRecord::decode(raw_rec)?;
        if rec.term_id >= fts_meta.next_term_id || !seen_tids.insert(rec.term_id) {
            return Err(TreeError::InvariantViolation(format!(
                "invalid or duplicate term_id {}",
                rec.term_id
            )));
        }
        if rec.stats.doc_freq != *exp_df
            || rec.stats.max_tf < *exp_max_tf
            || rec.stats.min_dl_over_tf > *exp_min_ratio
        {
            return Err(TreeError::InvariantViolation(format!(
                "FTS TermRecord stats mismatch for tid={}: disk {:?} vs expected (df={exp_df}, max_tf>={exp_max_tf}, min_ratio<={exp_min_ratio})",
                rec.term_id, rec.stats
            )));
        }

        // Collect all visible posting blocks for this term (inline or out-of-line).
        let mut live_actual_blocks: BTreeMap<RowBlock, Vec<(u8, u32)>> = BTreeMap::new();
        if let Some((ib_rblk, ref ib_bytes)) = rec.inline_block {
            if *exp_df > INLINE_TERM_MAX_DF as u64 {
                return Err(TreeError::InvariantViolation(format!(
                    "inlined term {} has doc_freq={} > INLINE_TERM_MAX_DF",
                    rec.term_id, exp_df
                )));
            }
            if seg_snap.segments.get(&ib_rblk.seg()).is_some_and(|e| e.is_visible()) {
                let pblk = PackedPostingBlock::decode(ib_bytes)?;
                let base = ib_rblk.block() * 256;
                let live_seg = live_by_seg.get(&ib_rblk.seg());
                let live_entries: Vec<(u8, u32)> = pblk
                    .slots
                    .iter()
                    .zip(pblk.tfs.iter())
                    .filter(|&(&s, _)| live_seg.is_some_and(|bm| bm.contains(base + s as u32)))
                    .map(|(&s, &tf)| (s, tf))
                    .collect();
                if !live_entries.is_empty() {
                    live_actual_blocks.insert(ib_rblk, live_entries);
                }
            }
        } else if let Some(out_blocks) = disk_postings_by_tid.get(&rec.term_id) {
            for (&rblk, raw_p) in out_blocks {
                let hdr = PostingBlockHeader::parse(raw_p)?;
                let chunk = rblk.chunk();
                let raw_s = disk_skips.get(&(rec.term_id, chunk.seg(), chunk.chunk())).ok_or_else(|| {
                    TreeError::InvariantViolation(format!(
                        "missing SkipBlock for tid={} {chunk:?}",
                        rec.term_id
                    ))
                })?;
                let sb = SkipBlock::decode(raw_s)?;
                let Some(skip_idx) = sb.rel_block_ids.binary_search(&rblk.rel_block()).ok() else {
                    return Err(TreeError::InvariantViolation(format!(
                        "SkipBlock missing rel_block for tid={} {rblk:?}",
                        rec.term_id
                    )));
                };
                if sb.max_tfs[skip_idx] < hdr.max_tf
                    || sb.min_dl_over_tfs[skip_idx] > hdr.min_dl_over_tf
                {
                    return Err(TreeError::InvariantViolation(format!(
                        "SkipBlock bounds looser than PostingBlockHeader for tid={} {rblk:?}",
                        rec.term_id
                    )));
                }
                let pblk = PackedPostingBlock::decode(raw_p)?;
                let base = rblk.block() * 256;
                let live_seg = live_by_seg.get(&rblk.seg());
                let live_entries: Vec<(u8, u32)> = pblk
                    .slots
                    .iter()
                    .zip(pblk.tfs.iter())
                    .filter(|&(&s, _)| live_seg.is_some_and(|bm| bm.contains(base + s as u32)))
                    .map(|(&s, &tf)| (s, tf))
                    .collect();
                if !live_entries.is_empty() {
                    live_actual_blocks.insert(rblk, live_entries);
                }
            }
        }

        let mut exp_sorted = exp_blocks.clone();
        for v in exp_sorted.values_mut() {
            v.sort_unstable_by_key(|&(s, _)| s);
        }
        if live_actual_blocks != exp_sorted {
            return Err(TreeError::InvariantViolation(format!(
                "FTS postings (∩ live) mismatch for tid={}",
                rec.term_id
            )));
        }

        if let Some(exp_pos_blocks) = exp_positions.get(tkey) {
            let disk_pos_blocks = disk_positions_by_tid.get(&rec.term_id).ok_or_else(|| {
                TreeError::InvariantViolation(format!(
                    "missing cf_fts_positions for tid={}",
                    rec.term_id
                ))
            })?;
            for (rblk, exp_slots_pos) in exp_pos_blocks {
                let raw_pos = disk_pos_blocks.get(rblk).ok_or_else(|| {
                    TreeError::InvariantViolation(format!(
                        "missing cf_fts_positions block {rblk:?} for tid={}",
                        rec.term_id
                    ))
                })?;
                let pos_blk = PackedPositionBlock::decode(raw_pos)?;
                let mut exp_sorted = exp_slots_pos.clone();
                exp_sorted.sort_unstable_by_key(|(s, _)| *s);
                let live_set: HashSet<u8> = live_slots_for_block(*rblk).into_iter().collect();
                let mut disk_live: Vec<(u8, Vec<u32>)> = pos_blk
                    .slots
                    .iter()
                    .enumerate()
                    .filter(|&(_, s)| live_set.contains(s))
                    .map(|(idx, &s)| (s, pos_blk.positions[idx].clone()))
                    .collect();
                disk_live.sort_unstable_by_key(|(s, _)| *s);
                if disk_live != exp_sorted {
                    return Err(TreeError::InvariantViolation(format!(
                        "cf_fts_positions (∩ live) mismatch at {rblk:?} for tid={}",
                        rec.term_id
                    )));
                }
            }
        }
    }

    Ok((
        fts_doc_count,
        disk_terms.len(),
        total_disk_postings,
        total_disk_positions,
    ))
}
