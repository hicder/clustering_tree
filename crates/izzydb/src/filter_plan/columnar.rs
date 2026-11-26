//! Columnar block filter evaluation.

use std::collections::HashMap;
use std::ops::Range;

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, RowId, SegmentId, for_each_set_bit_256};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::schema::{CollectionSchema, ColumnStorage, ScalarValue};
use izzydb_storage::codec::{
    SEG_FLAG_HAS_COLUMNS, SegmentState,
};
use izzydb_storage::column::ColumnBlockView;
use izzydb_storage::{SegmentTableSnapshot, Storage};

use crate::columns::force_row_fallback_enabled;

pub(crate) struct ColumnFilter<'a> {
    schema: &'a CollectionSchema,
    filter: &'a Filter,
    attrs: Vec<AttrId>,
}

impl<'a> ColumnFilter<'a> {
    pub(crate) fn new(schema: &'a CollectionSchema, filter: &'a Filter) -> Option<Self> {
        if force_row_fallback_enabled() || !filter_is_columnar_scalar_only(schema, filter) {
            return None;
        }
        let mut attrs = Vec::new();
        collect_filter_attr_ids(schema, filter, &mut attrs);
        attrs.sort_unstable();
        (!attrs.is_empty()).then_some(Self {
            schema,
            filter,
            attrs,
        })
    }

    /// Evaluate candidates sorted by RowId. Missing results need row-store verification.
    pub(crate) fn match_candidates(
        &self,
        storage: &Storage,
        snap: &SegmentTableSnapshot,
        rows: &[RowId],
    ) -> Result<Option<Vec<Option<bool>>>> {
        let mut blocks: Vec<(SegmentId, u32, Range<usize>)> = Vec::new();
        let mut column_rows = 0;
        let mut offset = 0;
        for segment_rows in rows.chunk_by(|a, b| a.seg() == b.seg()) {
            let seg = segment_rows[0].seg();
            let mut start = offset;
            offset += segment_rows.len();
            let Some(entry) = snap.get(seg) else {
                continue;
            };
            if entry.meta.state != SegmentState::Sealed
                || entry.meta.flags & SEG_FLAG_HAS_COLUMNS == 0
            {
                continue;
            }
            column_rows += segment_rows.len();
            for block_rows in segment_rows.chunk_by(|a, b| a.block_idx() == b.block_idx()) {
                let end = start + block_rows.len();
                blocks.push((seg, block_rows[0].block_idx(), start..end));
                start = end;
            }
        }

        // One row lookup supplies all attributes. Compare logical key counts before choosing columns.
        let key_count = blocks.len() * self.attrs.len();
        if key_count == 0 || key_count > column_rows {
            return Ok(None);
        }
        let keys: Vec<_> = blocks
            .iter()
            .flat_map(|&(seg, block, _)| self.attrs.iter().map(move |&attr| (seg, attr, block)))
            .collect();
        let slices = storage.multi_get_column_blocks_pinned(&keys, self.attrs.len() == 1)?;
        let mut matches = vec![None; rows.len()];
        let mut views = HashMap::with_capacity(self.attrs.len());
        for ((_, _, indices), block_slices) in
            blocks.into_iter().zip(slices.chunks_exact(self.attrs.len()))
        {
            for (&attr, slice) in self.attrs.iter().zip(block_slices) {
                let view = slice
                    .as_ref()
                    .map(|s| ColumnBlockView::parse(s))
                    .transpose()?;
                views.insert(attr, view);
            }
            for i in indices {
                matches[i] = Some(eval_column_filter_block(
                    self.schema,
                    self.filter,
                    &views,
                    rows[i].slot(),
                )?);
            }
        }
        Ok(Some(matches))
    }
}


pub(super) fn filter_is_columnar_scalar_only(schema: &CollectionSchema, filter: &Filter) -> bool {
    let is_col_scalar = |attr: &str| {
        schema.get_by_name(attr).is_some_and(|e| {
            e.storage == ColumnStorage::Inline
                && e.is_columnar()
                && !e.attr_type.is_array()
        })
    };
    match filter {
        Filter::And(ch) | Filter::Or(ch) => {
            ch.iter().all(|c| filter_is_columnar_scalar_only(schema, c))
        }
        Filter::Not(c) => filter_is_columnar_scalar_only(schema, c),
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _) => is_col_scalar(a),
        _ => false,
    }
}

pub(super) fn collect_filter_attr_ids(schema: &CollectionSchema, filter: &Filter, out: &mut Vec<AttrId>) {
    let push_attr = |attr: &str, out: &mut Vec<AttrId>| {
        if let Some(e) = schema.get_by_name(attr)
            && !out.contains(&e.id)
        {
            out.push(e.id);
        }
    };
    match filter {
        Filter::And(ch) | Filter::Or(ch) => {
            for c in ch {
                collect_filter_attr_ids(schema, c, out);
            }
        }
        Filter::Not(c) => collect_filter_attr_ids(schema, c, out),
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _) => push_attr(a, out),
        _ => {}
    }
}

pub(super) fn col_scalars_eq(a: &ScalarValue, b: &ScalarValue) -> bool {
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

pub(super) fn col_cmp_ordered(a: &ScalarValue, b: &ScalarValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
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

pub(super) fn eval_column_filter_block(
    schema: &CollectionSchema,
    filter: &Filter,
    views: &HashMap<AttrId, Option<ColumnBlockView<'_>>>,
    slot: u8,
) -> Result<bool> {
    let read_val = |attr: &str| -> Result<Option<ScalarValue>> {
        let Some(entry) = schema.get_by_name(attr) else {
            return Ok(None);
        };
        match views.get(&entry.id).and_then(|o| o.as_ref()) {
            Some(v) => {
                let s = v.get_scalar(slot)?;
                Ok((!matches!(s, ScalarValue::Null)).then_some(s))
            }
            None => Ok(None),
        }
    };
    Ok(match filter {
        Filter::And(ch) => {
            for c in ch {
                if !eval_column_filter_block(schema, c, views, slot)? {
                    return Ok(false);
                }
            }
            true
        }
        Filter::Or(ch) => {
            for c in ch {
                if eval_column_filter_block(schema, c, views, slot)? {
                    return Ok(true);
                }
            }
            false
        }
        Filter::Not(c) => !eval_column_filter_block(schema, c, views, slot)?,
        Filter::IsNull(a) => read_val(a)?.is_none(),
        Filter::IsNotNull(a) => read_val(a)?.is_some(),
        Filter::Eq(a, rhs) => {
            if matches!(rhs, ScalarValue::Null) {
                read_val(a)?.is_none()
            } else {
                read_val(a)?.is_some_and(|v| col_scalars_eq(&v, rhs))
            }
        }
        Filter::NotEq(a, rhs) => {
            if matches!(rhs, ScalarValue::Null) {
                read_val(a)?.is_some()
            } else {
                !read_val(a)?.is_some_and(|v| col_scalars_eq(&v, rhs))
            }
        }
        Filter::In(a, vals) => {
            let v = read_val(a)?;
            vals.iter().any(|rhs| match (v.as_ref(), rhs) {
                (None, ScalarValue::Null) => true,
                (Some(s), _) => col_scalars_eq(s, rhs),
                _ => false,
            })
        }
        Filter::NotIn(a, vals) => {
            let v = read_val(a)?;
            !vals.iter().any(|rhs| match (v.as_ref(), rhs) {
                (None, ScalarValue::Null) => true,
                (Some(s), _) => col_scalars_eq(s, rhs),
                _ => false,
            })
        }
        Filter::Gt(a, rhs) => read_val(a)?
            .and_then(|v| col_cmp_ordered(&v, rhs))
            .is_some_and(|o| o == std::cmp::Ordering::Greater),
        Filter::Gte(a, rhs) => read_val(a)?
            .and_then(|v| col_cmp_ordered(&v, rhs))
            .is_some_and(|o| o != std::cmp::Ordering::Less),
        Filter::Lt(a, rhs) => read_val(a)?
            .and_then(|v| col_cmp_ordered(&v, rhs))
            .is_some_and(|o| o == std::cmp::Ordering::Less),
        Filter::Lte(a, rhs) => read_val(a)?
            .and_then(|v| col_cmp_ordered(&v, rhs))
            .is_some_and(|o| o != std::cmp::Ordering::Greater),
        _ => unreachable!(),
    })
}

pub(super) fn try_eval_column_filter_segment(
    storage: &Storage,
    schema: &CollectionSchema,
    snap: &SegmentTableSnapshot,
    seg: SegmentId,
    cand_bm: &RoaringBitmap,
    filter: &Filter,
) -> Result<Option<RoaringBitmap>> {
    if force_row_fallback_enabled() {
        return Ok(None);
    }
    let Some(entry) = snap.get(seg) else {
        return Ok(Some(RoaringBitmap::new()));
    };
    if entry.meta.state != SegmentState::Sealed
        || (entry.meta.flags & SEG_FLAG_HAS_COLUMNS) == 0
        || !filter_is_columnar_scalar_only(schema, filter)
    {
        return Ok(None);
    }

    let mut attrs = Vec::new();
    collect_filter_attr_ids(schema, filter, &mut attrs);
    attrs.sort_unstable();

    let mut by_block: Vec<(u32, [u64; 4])> = Vec::new();
    for local in cand_bm.iter() {
        let blk = local >> 8;
        let slot = (local & 0xFF) as usize;
        if let Some(last) = by_block.last_mut()
            && last.0 == blk
        {
            last.1[slot >> 6] |= 1u64 << (slot & 63);
        } else {
            let mut m = [0u64; 4];
            m[slot >> 6] |= 1u64 << (slot & 63);
            by_block.push((blk, m));
        }
    }

    let mut keys: Vec<(SegmentId, AttrId, u32)> =
        Vec::with_capacity(by_block.len() * attrs.len().max(1));
    for &attr in &attrs {
        for &(blk, _) in &by_block {
            keys.push((seg, attr, blk));
        }
    }
    let slices = storage.multi_get_column_blocks_pinned(&keys, true)?;

    let num_blocks = by_block.len();
    let mut out = RoaringBitmap::new();
    let mut views: HashMap<AttrId, Option<ColumnBlockView<'_>>> =
        HashMap::with_capacity(attrs.len());

    for (bi, &(blk, mask)) in by_block.iter().enumerate() {
        views.clear();
        for (ai, &attr) in attrs.iter().enumerate() {
            let opt_slice = &slices[ai * num_blocks + bi];
            let parsed = opt_slice
                .as_ref()
                .map(|s| ColumnBlockView::parse(s))
                .transpose()?;
            views.insert(attr, parsed);
        }
        let mut block_err: Option<TreeError> = None;
        for_each_set_bit_256(&mask, |slot| {
            if block_err.is_some() {
                return;
            }
            match eval_column_filter_block(schema, filter, &views, slot) {
                Ok(true) => {
                    out.insert((blk << 8) | (slot as u32));
                }
                Ok(false) => {}
                Err(e) => block_err = Some(e),
            }
        });
        if let Some(e) = block_err {
            return Err(e);
        }
    }

    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AttributeType, Attributes, Collection, CollectionConfig, Document};

    #[test]
    fn candidate_filter_batches_blocks_and_leaves_open_rows_for_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let config = CollectionConfig::without_vectors()
            .with_seal_rows(512)
            .with_attribute("score", AttributeType::Float, true)
            .with_attribute("enabled", AttributeType::Bool, true);
        let mut collection = Collection::open(dir.path(), config).unwrap();
        let docs: Vec<_> = (0..=512)
            .map(|id| {
                let mut attributes = Attributes::new();
                // Every score in the second block is null, so that score block has no key.
                if id < 256 {
                    attributes.insert("score".into(), (id as f64).into());
                }
                attributes.insert("enabled".into(), (id % 2 == 0).into());
                Document {
                    id,
                    embedding: None,
                    attributes,
                }
            })
            .collect();
        collection.upsert_batch(&docs).unwrap();

        let filter = Filter::And(vec![
            Filter::Eq("enabled".into(), true.into()),
            Filter::Not(Box::new(Filter::Gte("score".into(), 100.0.into()))),
        ]);
        let column_filter = ColumnFilter::new(&collection.schema, &filter).unwrap();
        let snap = collection.segment_snapshot();
        let rows = [
            RowId::new(0, 0),
            RowId::new(0, 100),
            RowId::new(0, 256),
            RowId::new(0, 257),
            RowId::new(1, 0),
        ];
        assert_eq!(
            column_filter
                .match_candidates(&collection.storage, &snap, &rows)
                .unwrap(),
            Some(vec![Some(true), Some(false), Some(true), Some(false), None]),
        );
        let sparse_rows = [rows[0], rows[2]];
        assert_eq!(
            column_filter
                .match_candidates(&collection.storage, &snap, &sparse_rows)
                .unwrap(),
            None,
        );
    }
}
