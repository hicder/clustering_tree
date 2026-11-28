//! Inline attribute and column verification.

use std::collections::HashMap;
use roaring::RoaringBitmap;
use izzydb_core::config::{
    AttrId, DocId, RowId, SegmentId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{CollectionSchema, ColumnStorage};
use izzydb_storage::codec::{
    ATTR_COUNT_TAG,
    SEG_FLAG_HAS_COLUMNS, SegPosting, TAG_BOOL,
    TAG_PRESENT, ZoneEntry, decode_row_entries, extract_index_keys_from_entries, make_posting_key, split_posting_key, RowView,
};
use izzydb_storage::column::ColumnBlockView;
use izzydb_storage::leaf_attrs::{
    LeafAttrSpec, LeafAttrValue, LeafAttrView, NodeSummary,
};
use izzydb_storage::{
    CF_ATTR_POSTINGS, CF_COLUMNS, Storage,
};


pub(super) fn verify_inline_attributes_and_columns(
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

/// Verify leaf-colocated scalar columns and the in-memory leaf/internal filter summaries.
///
/// For every leaf slot, the value stored in the trailing leaf scalar block must equal the value
/// decoded from `cf_rows`, and every cached summary must equal the rollup of its children.
pub(super) fn verify_leaf_attrs(storage: &Storage, schema: &CollectionSchema) -> Result<()> {
    let specs: Vec<LeafAttrSpec> = schema
        .entries()
        .iter()
        .filter(|e| {
            e.storage == ColumnStorage::Inline
                && e.filterable
                && LeafAttrSpec::is_supported(e.attr_type)
        })
        .map(|e| LeafAttrSpec {
            attr_id: e.id,
            attr_type: e.attr_type,
        })
        .collect();
    if specs.is_empty() {
        return Ok(());
    }

    let leaves = storage.all_leaf_nodes()?;
    let seg_snap = storage.segment_snapshot();
    for (leaf_id, leaf) in &leaves {
        if leaf.slot_cap == 0 {
            continue;
        }
        if leaf.attr_block.is_empty() {
            return Err(TreeError::InvariantViolation(format!(
                "missing colocated attr block for non-empty leaf {leaf_id}"
            )));
        }
        let block = izzydb_storage::leaf_attrs::LeafAttrBlock::decode(&leaf.attr_block)?;
        for spec in &specs {
            let Some(col) = block.column(spec.attr_id) else {
                // Older leaves may predate a newly declared attribute; treat it as absent.
                continue;
            };
            for slot in 0..leaf.slot_cap {
                if !leaf.is_live(slot) {
                    continue;
                }
                let row = leaf.row_id(slot);
                let resolved = seg_snap.resolve_row(row).unwrap_or(row);
                let expected = match storage.get_row_bytes(resolved)? {
                    Some(row_bytes) => {
                        let view = RowView::parse(&row_bytes)?;
                        view.decode_field(spec.attr_id)?
                    }
                    None => None,
                };
                let actual = col.value_at(slot);
                let matches = match (&expected, actual) {
                    (None, None) => true,
                    (Some(izzydb_core::schema::AttributeValue::Bool(v)), Some(LeafAttrValue::Bool(a))) => {
                        *v == a
                    }
                    (Some(izzydb_core::schema::AttributeValue::Int(v)), Some(LeafAttrValue::Int(a))) => {
                        *v == a
                    }
                    (Some(izzydb_core::schema::AttributeValue::Uint(v)), Some(LeafAttrValue::Uint(a))) => {
                        *v == a
                    }
                    (Some(izzydb_core::schema::AttributeValue::Float(v)), Some(LeafAttrValue::Float(a))) => {
                        !v.is_nan() && !a.is_nan() && *v == a
                    }
                    _ => false,
                };
                if !matches {
                    return Err(TreeError::InvariantViolation(format!(
                        "leaf scalar mismatch at leaf {leaf_id} slot {slot} attr {}: row={expected:?} leaf={actual:?}",
                        spec.attr_id
                    )));
                }
            }
        }

        let view = LeafAttrView::parse(&leaf.attr_block)?;
        let expected_summary = NodeSummary::from_leaf(&view, leaf.slot_cap as u32);
        match storage.get_node_summary(*leaf_id) {
            Some(actual) if actual.as_ref() == &expected_summary => {}
            Some(actual) => {
                return Err(TreeError::InvariantViolation(format!(
                    "leaf {leaf_id} summary mismatch: cached={actual:?} expected={expected_summary:?}"
                )));
            }
            None => {
                return Err(TreeError::InvariantViolation(format!(
                    "missing node summary for leaf {leaf_id} (parent={:?}, slot_cap={}, live={})",
                    leaf.parent_id, leaf.slot_cap, leaf.live_count
                )));
            }
        }
    }

    for (node_id, node) in storage.all_internal_nodes()? {
        let mut expected = NodeSummary {
            row_count: 0,
            attrs: Vec::new(),
            complete: true,
        };
        for &child in &node.child_ids {
            match storage.get_node_summary(child) {
                Some(child_summary) => expected.merge_child(&child_summary),
                None => expected.complete = false,
            }
        }
        match storage.get_node_summary(node_id) {
            Some(actual) if actual.as_ref() == &expected => {}
            Some(actual) => {
                return Err(TreeError::InvariantViolation(format!(
                    "internal node {node_id} summary mismatch: cached={actual:?} expected={expected:?}"
                )));
            }
            None => {
                return Err(TreeError::InvariantViolation(format!(
                    "missing node summary for internal node {node_id}"
                )));
            }
        }
    }
    Ok(())
}
