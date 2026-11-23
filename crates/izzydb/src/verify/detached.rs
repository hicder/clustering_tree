//! Detached column verification.

use std::collections::HashMap;
use roaring::RoaringBitmap;
use izzydb_core::config::{
    AttrId, DocId, RowId, SegmentId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{CollectionSchema, ColumnStorage};
use izzydb_storage::codec::{
    DETACHED_COUNT_TAG, SegPosting, TAG_BOOL,
    TAG_PRESENT, ZoneEntry, detached_count_suffix,
    extract_index_keys_for_attr,
    split_detached_posting_key,
};
use izzydb_storage::column::ColumnBlockView;
use izzydb_storage::{
    CF_DETACHED, CF_DETACHED_POSTINGS, Storage,
};


pub(super) fn verify_detached_columns(
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
