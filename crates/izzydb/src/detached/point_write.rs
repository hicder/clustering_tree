//! Point writes, CoW carry-over, deletion, reads, and fragmentation rebuild.

use std::collections::HashMap;


use izzydb_core::config::{AttrId, ColumnGen, DocId, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rowset::Liveness;
use izzydb_core::schema::{
    AttributeValue, Attributes, CollectionSchema, ColumnStorage,
    SchemaEntry,
};
use izzydb_storage::codec::{
    TAG_FLOAT, TAG_INT, TAG_STRING, TAG_UINT, detached_posting_key, extract_index_keys_for_attr,
};
use izzydb_storage::column::ColumnBlockBuilder;
use izzydb_storage::{ColumnBlockView, Storage, TxContext};

use crate::collection::Collection;


use super::*;

// ---------------------------------------------------------------------------
// Point writes, CoW carry-over, deletion, get/get_batch, and >5% rebuild
// ---------------------------------------------------------------------------

/// Split `attrs` into `(inline_attributes, sorted detached (AttrId, AttributeValue))`.
pub(crate) fn partition_attributes(
    schema: &CollectionSchema,
    attrs: &Attributes,
) -> (Attributes, Vec<(AttrId, AttributeValue)>) {
    let mut inline = Attributes::new();
    let mut detached = Vec::new();
    for (name, val) in attrs {
        if let Some(entry) = schema.get_by_name(name)
            && entry.storage == ColumnStorage::Detached
        {
            detached.push((entry.id, val.clone()));
        } else {
            inline.insert(name.clone(), val.clone());
        }
    }
    detached.sort_unstable_by_key(|&(id, _)| id);
    (inline, detached)
}

#[inline]
pub(super) fn validate_attr_value(
    schema: &CollectionSchema,
    entry: &SchemaEntry,
    val: &AttributeValue,
) -> Result<()> {
    let mut single = Attributes::new();
    single.insert(entry.name.clone(), val.clone());
    schema.validate_attributes(&single)
}

#[inline]
pub(super) fn attr_values_exact_eq(a: &AttributeValue, b: &AttributeValue) -> bool {
    match (a, b) {
        (AttributeValue::Float(x), AttributeValue::Float(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Apply a single cell update at `(entry.id, col_gen, row)` inside `tx`.
/// Returns `Ok(true)` if the stored value changed.
pub(super) fn update_detached_cell_in_tx(
    tx: &mut TxContext<'_>,
    entry: &SchemaEntry,
    col_gen: ColumnGen,
    row: RowId,
    new_val: &AttributeValue,
    record_overwrite_on_replace: bool,
) -> Result<bool> {
    let seg = row.seg();
    let blk = row.block_idx();
    let slot = row.slot();
    let cur_bytes = tx.get_detached_block_bytes(entry.id, col_gen, seg, blk)?;
    let cur_view = cur_bytes
        .as_deref()
        .map(ColumnBlockView::parse)
        .transpose()?;
    let old_val = match cur_view.as_ref() {
        Some(v) if v.is_present(slot) => v.decode_attr_value(slot)?,
        _ => AttributeValue::Null,
    };

    if attr_values_exact_eq(&old_val, new_val) {
        return Ok(false);
    }

    let merged =
        ColumnBlockBuilder::merge_block(cur_view.as_ref(), entry.attr_type, &[(slot, new_val.clone())])?;
    tx.put_detached_block_bytes(entry.id, col_gen, seg, blk, merged);

    if entry.filterable {
        if !matches!(old_val, AttributeValue::Null) {
            let old_keys = extract_index_keys_for_attr(entry.id, &old_val)?;
            for k in &old_keys {
                tx.adjust_detached_count(col_gen, k, -1);
            }
            if record_overwrite_on_replace {
                tx.record_detached_overwritten(entry.id, col_gen, seg, row.local())?;
            }
        }
        if !matches!(new_val, AttributeValue::Null) {
            let new_keys = extract_index_keys_for_attr(entry.id, new_val)?;
            for k in &new_keys {
                let tag = k[2];
                tx.adjust_detached_count(col_gen, k, 1);
                tx.add_detached_posting(
                    detached_posting_key(entry.id, col_gen, tag, &k[3..], seg),
                    row.local(),
                );
                if k.len() > 3
                    && matches!(tag, TAG_INT | TAG_UINT | TAG_FLOAT | TAG_STRING)
                    && entry.attr_type.is_ordered()
                {
                    tx.widen_zone(seg, entry.id, tag, &k[3..])?;
                }
            }
        }
    }
    Ok(true)
}

/// Apply detached writes in-place on an existing `row` without changing its `RowId`.
/// When `reset_omitted` is true (`upsert`), every detached column in `schema` omitted from `updates`
/// is cleared to `Null`.
pub(crate) fn apply_detached_write_on_existing_row(
    tx: &mut TxContext<'_>,
    schema: &CollectionSchema,
    row: RowId,
    updates: &[(AttrId, AttributeValue)],
    reset_omitted: bool,
) -> Result<bool> {
    let mut changed = false;
    for entry in schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
    {
        let target_val = match updates.binary_search_by_key(&entry.id, |&(id, _)| id) {
            Ok(idx) => &updates[idx].1,
            Err(_) if reset_omitted => &AttributeValue::Null,
            Err(_) => continue,
        };
        let col_gen = tx.get_detached_meta(entry.id)?.current_gen;
        if update_detached_cell_in_tx(tx, entry, col_gen, row, target_val, true)? {
            changed = true;
        }
    }
    Ok(changed)
}

/// Write detached columns onto a newly-allocated `new_row`, optionally carrying over untouched
/// detached columns from `old_row` (`patch` sets `carry_over = true`; `upsert` sets `false`).
pub(crate) fn write_detached_on_new_row(
    tx: &mut TxContext<'_>,
    schema: &CollectionSchema,
    old_row: Option<RowId>,
    new_row: RowId,
    updates: &[(AttrId, AttributeValue)],
    carry_over: bool,
) -> Result<()> {
    for entry in schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
    {
        let col_gen = tx.get_detached_meta(entry.id)?.current_gen;

        let mut old_val = AttributeValue::Null;
        if let Some(prev) = old_row
            && let Some(prev_bytes) =
                tx.get_detached_block_bytes(entry.id, col_gen, prev.seg(), prev.block_idx())?
        {
            let v = ColumnBlockView::parse(&prev_bytes)?;
            if v.is_present(prev.slot()) {
                old_val = v.decode_attr_value(prev.slot())?;
            }
        }

        // Decrement counts for tombstoned old_row.
        if entry.filterable && !matches!(old_val, AttributeValue::Null) {
            let old_keys = extract_index_keys_for_attr(entry.id, &old_val)?;
            for k in &old_keys {
                tx.adjust_detached_count(col_gen, k, -1);
            }
        }

        let target_val = match updates.binary_search_by_key(&entry.id, |&(id, _)| id) {
            Ok(idx) => updates[idx].1.clone(),
            Err(_) if carry_over => old_val,
            Err(_) => continue,
        };

        if !matches!(target_val, AttributeValue::Null) {
            let _ =
                update_detached_cell_in_tx(tx, entry, col_gen, new_row, &target_val, false)?;
        }
    }
    Ok(())
}

/// Decrement `0x07` detached counts when deleting live row `old_row`.
pub(crate) fn delete_detached_in_tx(
    tx: &mut TxContext<'_>,
    schema: &CollectionSchema,
    old_row: RowId,
) -> Result<()> {
    for entry in schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached && e.filterable)
    {
        let col_gen = tx.get_detached_meta(entry.id)?.current_gen;
        let Some(bytes) = tx.get_detached_block_bytes(
            entry.id,
            col_gen,
            old_row.seg(),
            old_row.block_idx(),
        )?
        else {
            continue;
        };
        let v = ColumnBlockView::parse(&bytes)?;
        if !v.is_present(old_row.slot()) {
            continue;
        }
        let val = v.decode_attr_value(old_row.slot())?;
        let keys = extract_index_keys_for_attr(entry.id, &val)?;
        for k in &keys {
            tx.adjust_detached_count(col_gen, k, -1);
        }
    }
    Ok(())
}

/// Point write `set_detached` / `set_detached_batch` against live documents in place.
pub(crate) fn apply_point_detached_batch(
    coll: &mut Collection,
    updates: &[(DocId, Attributes)],
) -> Result<()> {
    if updates.is_empty() {
        return Ok(());
    }
    let mut doc_ids = Vec::with_capacity(updates.len());
    let mut parsed: Vec<(DocId, Vec<(AttrId, AttributeValue)>)> = Vec::with_capacity(updates.len());
    for (doc_id, attrs) in updates {
        coll.schema().validate_attributes(attrs)?;
        let mut items = Vec::with_capacity(attrs.len());
        for (name, val) in attrs {
            let entry = coll.schema().get_by_name(name).ok_or_else(|| {
                TreeError::InvalidSchema(format!("unknown attribute '{name}'"))
            })?;
            if entry.storage != ColumnStorage::Detached {
                return Err(TreeError::InvalidSchema(format!(
                    "set_detached only accepts Detached attributes; '{name}' is Inline"
                )));
            }
            items.push((entry.id, val.clone()));
        }
        items.sort_unstable_by_key(|&(id, _)| id);
        doc_ids.push(*doc_id);
        parsed.push((*doc_id, items));
    }

    let mapped = coll.storage().multi_get_id_map(&doc_ids)?;
    let mut resolved = Vec::with_capacity(parsed.len());
    for ((doc_id, items), opt_row) in parsed.into_iter().zip(mapped) {
        let row = opt_row.ok_or(TreeError::DocumentNotFound(doc_id))?;
        resolved.push((row, items));
    }
    resolved.sort_unstable_by_key(|&(r, _)| r);

    let schema = coll.schema().clone();
    let (any_changed, committed) = {
        let mut tx = TxContext::new(
            &coll.storage,
            &mut coll.internal_cache,
            &mut coll.leaf_parent,
            coll.manifest.clone(),
            coll.ann_meta.clone(),
        );
        let mut any_changed = false;
        for (row, items) in resolved {
            if apply_detached_write_on_existing_row(&mut tx, &schema, row, &items, false)? {
                any_changed = true;
            }
        }
        if any_changed {
            let (m, am) = tx.commit()?;
            (true, Some((m, am)))
        } else {
            (false, None)
        }
    };
    if any_changed && let Some((m, am)) = committed {
        coll.apply_tx_commit(m, am);
    }
    Ok(())
}

/// Read all non-`Null` detached attributes for a single `row` (for `Collection::get`).
pub(crate) fn read_detached_attrs_for_row(
    storage: &Storage,
    schema: &CollectionSchema,
    row: RowId,
    out: &mut Attributes,
) -> Result<()> {
    let detached_entries: Vec<&SchemaEntry> = schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
        .collect();
    if detached_entries.is_empty() {
        return Ok(());
    }
    let gens = all_detached_gens(storage)?;
    let keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = detached_entries
        .iter()
        .map(|e| {
            let g = gens.get(&e.id).copied().unwrap_or(0);
            (e.id, g, row.seg(), row.block_idx())
        })
        .collect();
    let slices = storage.multi_get_detached_blocks_pinned(&keys, true)?;
    for (entry, opt_slice) in detached_entries.iter().zip(slices.iter()) {
        let Some(slice) = opt_slice.as_ref() else {
            continue;
        };
        let view = ColumnBlockView::parse(slice)?;
        if view.is_present(row.slot()) {
            let val = view.decode_attr_value(row.slot())?;
            if !matches!(val, AttributeValue::Null) {
                out.insert(entry.name.clone(), val);
            }
        }
    }
    Ok(())
}

/// Populate non-`Null` detached attributes for `sorted_rows` into `per_row_attrs` (aligned by index).
pub(crate) fn read_detached_attrs_for_rows_sorted(
    storage: &Storage,
    schema: &CollectionSchema,
    sorted_rows: &[RowId],
    per_row_attrs: &mut [Attributes],
) -> Result<()> {
    if sorted_rows.is_empty() {
        return Ok(());
    }
    let detached_entries: Vec<&SchemaEntry> = schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
        .collect();
    if detached_entries.is_empty() {
        return Ok(());
    }
    let gens = all_detached_gens(storage)?;
    let mut keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = Vec::new();
    for entry in &detached_entries {
        let g = gens.get(&entry.id).copied().unwrap_or(0);
        for &row in sorted_rows {
            let k = (entry.id, g, row.seg(), row.block_idx());
            if keys.last().copied() != Some(k) {
                keys.push(k);
            }
        }
    }
    let slices = storage.multi_get_detached_blocks_pinned(&keys, true)?;
    let mut block_map: HashMap<(AttrId, SegmentId, u32), &ColumnBlockView<'_>> = HashMap::new();
    let mut parsed_views: Vec<Option<ColumnBlockView<'_>>> = Vec::with_capacity(slices.len());
    for s_opt in &slices {
        parsed_views.push(s_opt.as_ref().map(|s| ColumnBlockView::parse(s)).transpose()?);
    }
    for (k, view_opt) in keys.iter().zip(parsed_views.iter()) {
        if let Some(v) = view_opt.as_ref() {
            block_map.insert((k.0, k.2, k.3), v);
        }
    }
    for (&row, attrs) in sorted_rows.iter().zip(per_row_attrs.iter_mut()) {
        for entry in &detached_entries {
            if let Some(view) = block_map.get(&(entry.id, row.seg(), row.block_idx()))
                && view.is_present(row.slot())
            {
                let val = view.decode_attr_value(row.slot())?;
                if !matches!(val, AttributeValue::Null) {
                    attrs.insert(entry.name.clone(), val);
                }
            }
        }
    }
    Ok(())
}

/// Rebuild any detached column whose live `0x14` overwritten set exceeds 5% of `live_rows`.
pub(crate) fn rebuild_fragmented_detached_columns(coll: &mut Collection) -> Result<usize> {
    let snap = coll.storage().segment_snapshot();
    let live_rows: u64 = snap
        .visible_segments()
        .into_iter()
        .map(|s| {
            snap.get(s)
                .map_or(0, |e| e.meta.row_count.saturating_sub(e.meta.deleted_count) as u64)
        })
        .sum();
    if live_rows == 0 {
        return Ok(0);
    }

    let detached_names: Vec<(AttrId, String)> = coll
        .schema()
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached && e.filterable)
        .map(|e| (e.id, e.name.clone()))
        .collect();

    let mut to_rebuild: Vec<String> = Vec::new();
    for (attr_id, name) in detached_names {
        let cur_gen = current_detached_gen(coll.storage(), attr_id)?;
        let ow_segs = coll.storage().all_detached_overwritten(attr_id, cur_gen)?;
        if ow_segs.is_empty() {
            continue;
        }
        let mut live_ow = 0u64;
        for (seg, ow_bm) in ow_segs {
            if let Some(live_bm) = snap.live_bitmap(seg) {
                live_ow += (ow_bm & live_bm).len();
            }
        }
        if live_ow.saturating_mul(20) > live_rows {
            to_rebuild.push(name);
        }
    }

    let count = to_rebuild.len();
    for name in to_rebuild {
        let job = coll
            .detached_ingest()
            .column(name)
            .mode(DetachedWriteMode::Merge)
            .build()?;
        coll.publish_detached(job)?;
    }
    Ok(count)
}
