//! Query helpers: generation map, overwritten correction, and mixed filter evaluation.

use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, ColumnGen, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::rowset::Liveness;
use izzydb_core::schema::{
    AttributeValue, CollectionSchema, ColumnStorage, ScalarValue,
    SchemaEntry,
};
use izzydb_storage::codec::RowView;
use izzydb_storage::{ColumnBlockView, SegmentTableSnapshot, Storage};

use crate::fts_index::FtsIndex;



// ---------------------------------------------------------------------------
// Query helpers: generation map, overwritten correction, & mixed filter eval
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn current_detached_gen(storage: &Storage, attr_id: AttrId) -> Result<ColumnGen> {
    Ok(storage.get_detached_meta(attr_id)?.map_or(0, |m| m.current_gen))
}

pub(crate) fn all_detached_gens(storage: &Storage) -> Result<HashMap<AttrId, ColumnGen>> {
    let mut map = HashMap::new();
    for (attr_id, meta) in storage.all_detached_metas()? {
        map.insert(attr_id, meta.current_gen);
    }
    Ok(map)
}

pub(crate) fn filter_mentions_detached(schema: &CollectionSchema, filter: &Filter) -> bool {
    let is_det = |attr: &str| {
        schema
            .get_by_name(attr)
            .is_some_and(|e| e.storage == ColumnStorage::Detached)
    };
    match filter {
        Filter::And(ch) | Filter::Or(ch) => {
            ch.iter().any(|c| filter_mentions_detached(schema, c))
        }
        Filter::Not(c) => filter_mentions_detached(schema, c),
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Contains(a, _)
        | Filter::ContainsAny(a, _)
        | Filter::ContainsAllTokens { attr: a, .. }
        | Filter::ContainsAnyToken { attr: a, .. }
        | Filter::ContainsTokenSequence { attr: a, .. } => is_det(a),
    }
}

pub(crate) fn filter_mentions_inline_or_fts(schema: &CollectionSchema, filter: &Filter) -> bool {
    let is_inline = |attr: &str| {
        schema
            .get_by_name(attr)
            .is_some_and(|e| e.storage == ColumnStorage::Inline)
    };
    match filter {
        Filter::And(ch) | Filter::Or(ch) => ch
            .iter()
            .any(|c| filter_mentions_inline_or_fts(schema, c)),
        Filter::Not(c) => filter_mentions_inline_or_fts(schema, c),
        Filter::ContainsAllTokens { .. }
        | Filter::ContainsAnyToken { .. }
        | Filter::ContainsTokenSequence { .. } => true,
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Contains(a, _)
        | Filter::ContainsAny(a, _) => is_inline(a),
    }
}

pub(crate) fn collect_detached_filter_attr_ids(
    schema: &CollectionSchema,
    filter: &Filter,
    out: &mut Vec<AttrId>,
) {
    let mut push = |attr: &str| {
        if let Some(e) = schema.get_by_name(attr)
            && e.storage == ColumnStorage::Detached
            && !out.contains(&e.id)
        {
            out.push(e.id);
        }
    };
    match filter {
        Filter::And(ch) | Filter::Or(ch) => {
            for c in ch {
                collect_detached_filter_attr_ids(schema, c, out);
            }
        }
        Filter::Not(c) => collect_detached_filter_attr_ids(schema, c, out),
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Contains(a, _)
        | Filter::ContainsAny(a, _) => push(a),
        Filter::ContainsAllTokens { .. }
        | Filter::ContainsAnyToken { .. }
        | Filter::ContainsTokenSequence { .. } => {}
    }
}

/// Evaluate `cf_detached` blocks for `(seg, ow_bm)` across all overwritten rows that are live
/// in `snap`, inserting `local` into `out` if `slot_matches(val)` is `true`.
pub(crate) fn recheck_detached_overwritten_in_seg(
    storage: &Storage,
    entry: &SchemaEntry,
    col_gen: ColumnGen,
    seg: SegmentId,
    ow_live: &RoaringBitmap,
    out: &mut RoaringBitmap,
    mut slot_matches: impl FnMut(&ColumnBlockView<'_>, u8) -> Result<bool>,
) -> Result<()> {
    if ow_live.is_empty() {
        return Ok(());
    }
    let _ = entry;
    let mut blocks: Vec<(u32, Vec<u8>)> = Vec::new();
    for local in ow_live.iter() {
        let blk = local >> 8;
        let slot = (local & 0xFF) as u8;
        if let Some(last) = blocks.last_mut()
            && last.0 == blk
        {
            last.1.push(slot);
        } else {
            blocks.push((blk, vec![slot]));
        }
    }
    let keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = blocks
        .iter()
        .map(|&(blk, _)| (entry.id, col_gen, seg, blk))
        .collect();
    let slices = storage.multi_get_detached_blocks_pinned(&keys, true)?;
    for ((blk, slots), opt_slice) in blocks.iter().zip(slices.iter()) {
        let Some(slice) = opt_slice.as_ref() else {
            continue;
        };
        let view = ColumnBlockView::parse(slice)?;
        for &slot in slots {
            if view.is_present(slot) && slot_matches(&view, slot)? {
                out.insert((blk << 8) | (slot as u32));
            }
        }
    }
    Ok(())
}

/// Build the overwritten-set map (`seg -> ow & live`) for `(attr_id, col_gen)` and subtract it
/// from `by_seg` bitmaps, then insert matching overwritten rows via `slot_matches`.
pub(crate) fn apply_overwritten_correction_for_filter(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    entry: &SchemaEntry,
    col_gen: ColumnGen,
    by_seg: &mut BTreeMap<SegmentId, RoaringBitmap>,
    mut slot_matches: impl FnMut(&ColumnBlockView<'_>, u8) -> Result<bool>,
) -> Result<()> {
    let ow_list = storage.all_detached_overwritten(entry.id, col_gen)?;
    if ow_list.is_empty() {
        return Ok(());
    }
    for (seg, ow_bm) in ow_list {
        let Some(live_bm) = snap.live_bitmap(seg) else {
            continue;
        };
        let ow_live = ow_bm & live_bm;
        if ow_live.is_empty() {
            continue;
        }
        let dst = by_seg.entry(seg).or_default();
        *dst -= &ow_live;
        recheck_detached_overwritten_in_seg(
            storage,
            entry,
            col_gen,
            seg,
            &ow_live,
            dst,
            &mut slot_matches,
        )?;
        if dst.is_empty() {
            by_seg.remove(&seg);
        }
    }
    Ok(())
}

/// Evaluate a detached-aware `Filter` for a single `(row, optional RowView)` using `detached_views`
/// preloaded from `cf_detached` (`attr_id -> Option<ColumnBlockView>`).
pub(crate) fn matches_filter_mixed(
    schema: &CollectionSchema,
    filter: &Filter,
    row_view: Option<&RowView<'_>>,
    detached_views: &HashMap<AttrId, Option<ColumnBlockView<'_>>>,
    slot: u8,
    fts: &FtsIndex,
) -> Result<bool> {
    eval_mixed_node(schema, filter, row_view, detached_views, slot, fts)
}

pub(super) fn eval_mixed_node(
    schema: &CollectionSchema,
    filter: &Filter,
    row_view: Option<&RowView<'_>>,
    detached_views: &HashMap<AttrId, Option<ColumnBlockView<'_>>>,
    slot: u8,
    fts: &FtsIndex,
) -> Result<bool> {
    match filter {
        Filter::And(ch) => {
            for c in ch {
                if !eval_mixed_node(schema, c, row_view, detached_views, slot, fts)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Filter::Or(ch) => {
            for c in ch {
                if eval_mixed_node(schema, c, row_view, detached_views, slot, fts)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Filter::Not(c) => Ok(!eval_mixed_node(
            schema,
            c,
            row_view,
            detached_views,
            slot,
            fts,
        )?),
        leaf => {
            if let Some((kind, attr, query, last_as_prefix)) = leaf.as_fts_predicate() {
                let rv = row_view.ok_or_else(|| {
                    TreeError::InvariantViolation("missing RowView for FTS predicate".into())
                })?;
                return rv.matches_filter(schema, leaf, &mut |_k, _a, _q, _lp, val| {
                    fts.matches_fts_attr(schema, kind, attr, query, last_as_prefix, val)
                });
            }
            let attr_name = leaf_attr_name(leaf).ok_or_else(|| {
                TreeError::InvariantViolation("expected leaf Filter predicate".into())
            })?;
            let entry = schema.get_by_name(attr_name).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr_name}'"))
            })?;
            if entry.storage == ColumnStorage::Inline {
                let rv = row_view.ok_or_else(|| {
                    TreeError::InvariantViolation("missing RowView for Inline predicate".into())
                })?;
                return rv.matches_filter(
                    schema,
                    leaf,
                    &mut |kind, a, q, lp, val| {
                        fts.matches_fts_attr(schema, kind, a, q, lp, val)
                    },
                );
            }
            let view_opt = detached_views.get(&entry.id).and_then(|o| o.as_ref());
            eval_detached_leaf_in_block(view_opt, slot, leaf)
        }
    }
}

#[inline]
pub(super) fn leaf_attr_name(f: &Filter) -> Option<&str> {
    match f {
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Contains(a, _)
        | Filter::ContainsAny(a, _) => Some(a.as_str()),
        _ => None,
    }
}

pub(super) fn eval_detached_leaf_in_block(
    view_opt: Option<&ColumnBlockView<'_>>,
    slot: u8,
    leaf: &Filter,
) -> Result<bool> {
    let Some(view) = view_opt.filter(|v| v.is_present(slot)) else {
        return Ok(match leaf {
            Filter::IsNull(_) => true,
            Filter::Eq(_, ScalarValue::Null) => true,
            Filter::In(_, vals) => vals.iter().any(|v| matches!(v, ScalarValue::Null)),
            Filter::NotEq(_, v) => !matches!(v, ScalarValue::Null),
            Filter::NotIn(_, vals) => !vals.iter().any(|v| matches!(v, ScalarValue::Null)),
            _ => false,
        });
    };

    if matches!(leaf, Filter::IsNotNull(_)) {
        return Ok(true);
    }
    if matches!(leaf, Filter::IsNull(_)) {
        return Ok(false);
    }

    match leaf {
        Filter::Eq(_, rhs) => {
            if matches!(rhs, ScalarValue::Null) {
                Ok(false)
            } else {
                let s = view.get_scalar(slot)?;
                Ok(scalars_eq(&s, rhs))
            }
        }
        Filter::NotEq(_, rhs) => {
            if matches!(rhs, ScalarValue::Null) {
                Ok(true)
            } else {
                let s = view.get_scalar(slot)?;
                Ok(!scalars_eq(&s, rhs))
            }
        }
        Filter::In(_, vals) => {
            let s = view.get_scalar(slot)?;
            Ok(vals.iter().any(|rhs| scalars_eq(&s, rhs)))
        }
        Filter::NotIn(_, vals) => {
            let s = view.get_scalar(slot)?;
            Ok(!vals.iter().any(|rhs| scalars_eq(&s, rhs)))
        }
        Filter::Gt(_, rhs) => Ok(view
            .get_scalar(slot)?
            .cmp_ordered(rhs)
            .is_some_and(|o| o == std::cmp::Ordering::Greater)),
        Filter::Gte(_, rhs) => Ok(view
            .get_scalar(slot)?
            .cmp_ordered(rhs)
            .is_some_and(|o| o != std::cmp::Ordering::Less)),
        Filter::Lt(_, rhs) => Ok(view
            .get_scalar(slot)?
            .cmp_ordered(rhs)
            .is_some_and(|o| o == std::cmp::Ordering::Less)),
        Filter::Lte(_, rhs) => Ok(view
            .get_scalar(slot)?
            .cmp_ordered(rhs)
            .is_some_and(|o| o != std::cmp::Ordering::Greater)),
        Filter::Contains(_, rhs) => {
            let val = view.decode_attr_value(slot)?;
            Ok(attr_array_contains(&val, rhs))
        }
        Filter::ContainsAny(_, rhs_list) => {
            let val = view.decode_attr_value(slot)?;
            Ok(rhs_list.iter().any(|r| attr_array_contains(&val, r)))
        }
        _ => Ok(false),
    }
}

#[inline]
pub(super) fn scalars_eq(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Null, ScalarValue::Null) => true,
        (ScalarValue::Bool(x), ScalarValue::Bool(y)) => x == y,
        (ScalarValue::Int(x), ScalarValue::Int(y)) => x == y,
        (ScalarValue::Uint(x), ScalarValue::Uint(y)) => x == y,
        (ScalarValue::Float(x), ScalarValue::Float(y)) => !x.is_nan() && !y.is_nan() && x == y,
        (ScalarValue::String(x), ScalarValue::String(y)) => x == y,
        _ => false,
    }
}

pub(super) trait ScalarCmpExt {
    fn cmp_ordered(&self, other: &ScalarValue) -> Option<std::cmp::Ordering>;
}

impl ScalarCmpExt for ScalarValue {
    fn cmp_ordered(&self, other: &ScalarValue) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (ScalarValue::Int(x), ScalarValue::Int(y)) => Some(x.cmp(y)),
            (ScalarValue::Uint(x), ScalarValue::Uint(y)) => Some(x.cmp(y)),
            (ScalarValue::Float(x), ScalarValue::Float(y)) => {
                if x.is_nan() || y.is_nan() {
                    None
                } else {
                    let xn = if *x == 0.0 { 0.0 } else { *x };
                    let yn = if *y == 0.0 { 0.0 } else { *y };
                    Some(xn.total_cmp(&yn))
                }
            }
            (ScalarValue::String(x), ScalarValue::String(y)) => Some(x.cmp(y)),
            _ => None,
        }
    }
}

#[inline]
pub(super) fn attr_array_contains(val: &AttributeValue, scalar: &ScalarValue) -> bool {
    match (val, scalar) {
        (AttributeValue::IntArray(arr), ScalarValue::Int(v)) => arr.contains(v),
        (AttributeValue::UintArray(arr), ScalarValue::Uint(v)) => arr.contains(v),
        (AttributeValue::StringArray(arr), ScalarValue::String(v)) => arr.contains(v),
        _ => false,
    }
}

/// Test if a `ColumnBlockView` slot's encoded values match `encoded_targets` (ordered val bytes).
pub(crate) fn block_slot_matches_any_encoded(
    view: &ColumnBlockView<'_>,
    slot: u8,
    encoded_targets: &[Vec<u8>],
) -> Result<bool> {
    let mut matched = false;
    let mut buf = Vec::new();
    view.for_each_encoded_value(slot, &mut buf, |enc| {
        if !matched && encoded_targets.binary_search_by(|k| k.as_slice().cmp(enc)).is_ok() {
            matched = true;
        }
    })?;
    Ok(matched)
}

pub(crate) type OverwrittenTrueValues = (
    BTreeMap<SegmentId, RoaringBitmap>,
    BTreeMap<Vec<u8>, BTreeMap<SegmentId, RoaringBitmap>>,
);

/// Collect corrected detached value -> `(SegmentId -> RoaringBitmap)` overlay for `0x14` rows
/// across all segments for `(entry, col_gen)` (used by `order_by` streaming when `0x14` non-empty).
pub(crate) fn collect_overwritten_true_values(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    entry: &SchemaEntry,
    col_gen: ColumnGen,
) -> Result<OverwrittenTrueValues> {
    let ow_list = storage.all_detached_overwritten(entry.id, col_gen)?;
    let mut ow_live_by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    let mut true_val_by_seg: BTreeMap<Vec<u8>, BTreeMap<SegmentId, RoaringBitmap>> =
        BTreeMap::new();
    if ow_list.is_empty() {
        return Ok((ow_live_by_seg, true_val_by_seg));
    }

    let mut keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = Vec::new();
    let mut seg_blocks: Vec<(SegmentId, u32, Vec<u8>)> = Vec::new();
    for (seg, ow_bm) in ow_list {
        let Some(live_bm) = snap.live_bitmap(seg) else {
            continue;
        };
        let ow_live = ow_bm & live_bm;
        if ow_live.is_empty() {
            continue;
        }
        for local in ow_live.iter() {
            let blk = local >> 8;
            let slot = (local & 0xFF) as u8;
            if let Some(last) = seg_blocks.last_mut()
                && last.0 == seg
                && last.1 == blk
            {
                last.2.push(slot);
            } else {
                seg_blocks.push((seg, blk, vec![slot]));
                keys.push((entry.id, col_gen, seg, blk));
            }
        }
        ow_live_by_seg.insert(seg, ow_live);
    }

    let slices = storage.multi_get_detached_blocks_pinned(&keys, true)?;
    let mut buf = Vec::new();
    for ((seg, blk, slots), opt_slice) in seg_blocks.iter().zip(slices.iter()) {
        let Some(slice) = opt_slice.as_ref() else {
            continue;
        };
        let view = ColumnBlockView::parse(slice)?;
        for &slot in slots {
            let local = (blk << 8) | (slot as u32);
            view.for_each_encoded_value(slot, &mut buf, |enc| {
                true_val_by_seg
                    .entry(enc.to_vec())
                    .or_default()
                    .entry(*seg)
                    .or_default()
                    .insert(local);
            })?;
        }
    }
    Ok((ow_live_by_seg, true_val_by_seg))
}

/// Evaluate a mixed `Filter` for a list of sorted `RowId`s in batches (loading `cf_detached` and,
/// only if inline/FTS predicates exist, `cf_rows`).
pub(crate) fn filter_rows_mixed_batch(
    storage: &Storage,
    schema: &CollectionSchema,
    fts: &FtsIndex,
    filter: &Filter,
    rows: &[RowId],
) -> Result<Vec<bool>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let need_rows = filter_mentions_inline_or_fts(schema, filter);
    let mut det_attrs = Vec::new();
    collect_detached_filter_attr_ids(schema, filter, &mut det_attrs);
    det_attrs.sort_unstable();
    let gens = all_detached_gens(storage)?;

    let row_slices = if need_rows {
        storage.multi_get_row_bytes_pinned(rows)?
    } else {
        Vec::new()
    };

    let mut det_keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = Vec::new();
    for &attr_id in &det_attrs {
        let g = gens.get(&attr_id).copied().unwrap_or(0);
        for &row in rows {
            let k = (attr_id, g, row.seg(), row.block_idx());
            if det_keys.last().copied() != Some(k) {
                det_keys.push(k);
            }
        }
    }
    let det_slices = storage.multi_get_detached_blocks_pinned(&det_keys, true)?;
    let mut det_views_all: HashMap<(AttrId, SegmentId, u32), ColumnBlockView<'_>> = HashMap::new();
    for (k, opt_slice) in det_keys.iter().zip(det_slices.iter()) {
        if let Some(s) = opt_slice.as_ref() {
            det_views_all.insert((k.0, k.2, k.3), ColumnBlockView::parse(s)?);
        }
    }

    let mut results = Vec::with_capacity(rows.len());
    let mut block_views: HashMap<AttrId, Option<ColumnBlockView<'_>>> =
        HashMap::with_capacity(det_attrs.len());
    for (idx, &row) in rows.iter().enumerate() {
        block_views.clear();
        for &attr_id in &det_attrs {
            block_views.insert(
                attr_id,
                det_views_all
                    .get(&(attr_id, row.seg(), row.block_idx()))
                    .copied(),
            );
        }
        let rv = if need_rows {
            let Some(slice) = row_slices[idx].as_ref() else {
                results.push(false);
                continue;
            };
            Some(RowView::parse(slice)?)
        } else {
            None
        };
        results.push(matches_filter_mixed(
            schema,
            filter,
            rv.as_ref(),
            &block_views,
            row.slot(),
            fts,
        )?);
    }
    Ok(results)
}
