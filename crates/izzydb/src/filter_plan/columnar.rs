//! Columnar block filter evaluation.

use std::ops::Range;

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, RowId, SegmentId, for_each_set_bit_256};
use izzydb_core::error::Result;
use izzydb_core::filter::Filter;
use izzydb_core::schema::{AttributeType, CollectionSchema, ColumnStorage, ScalarValue};
use izzydb_storage::codec::{SEG_FLAG_HAS_COLUMNS, SegmentState};
use izzydb_storage::column::ColumnBlockView;
use izzydb_storage::{SegmentTableSnapshot, Storage};

use crate::columns::force_row_fallback_enabled;

/// A filter compiled for direct evaluation against `cf_columns` blocks.
///
/// Attribute names are resolved to dense `attr_slot` indices once, and scalar operands are
/// normalized to typed Rust primitives, so per-row evaluation performs no name lookup,
/// `HashMap` probe, or `ScalarValue` allocation.
///
/// ```text
/// Filter::Lt("score".into(), 60.0.into())
///   -> ColumnPredicate::Cmp(0, CmpOp::Lt, ColRhs::Float(60.0))
/// ```
pub(crate) struct CompiledColumnFilter {
    attrs: Vec<AttrId>,
    predicate: ColumnPredicate,
}

impl CompiledColumnFilter {
    /// Compile `filter` against `schema`, returning `None` when the filter is not a pure
    /// columnar scalar predicate or row fallback is forced.
    pub(crate) fn compile(schema: &CollectionSchema, filter: &Filter) -> Option<Self> {
        if force_row_fallback_enabled() || !filter_is_columnar_scalar_only(schema, filter) {
            return None;
        }
        let mut attrs = Vec::new();
        collect_filter_attr_ids(schema, filter, &mut attrs);
        attrs.sort_unstable();
        if attrs.is_empty() {
            return None;
        }
        let predicate = compile_predicate(schema, &attrs, filter);
        Some(Self { attrs, predicate })
    }

    #[inline]
    fn attrs(&self) -> &[AttrId] {
        &self.attrs
    }

    /// Evaluate the compiled predicate at `slot` against block views aligned with
    /// [`Self::attrs`](Self::attrs).
    #[inline]
    pub(crate) fn eval(&self, views: &[Option<ColumnBlockView<'_>>], slot: u8) -> bool {
        debug_assert_eq!(views.len(), self.attrs.len());
        self.predicate.eval(views, slot)
    }
}

/// Query-local column filter that batches `cf_columns` lookups for sorted candidate rows.
pub(crate) struct ColumnFilter<'a> {
    compiled: CompiledColumnFilter,
    _filter: &'a Filter,
}

impl<'a> ColumnFilter<'a> {
    pub(crate) fn new(schema: &'a CollectionSchema, filter: &'a Filter) -> Option<Self> {
        Some(Self {
            compiled: CompiledColumnFilter::compile(schema, filter)?,
            _filter: filter,
        })
    }

    pub(crate) fn attrs(&self) -> &[AttrId] {
        self.compiled.attrs()
    }

    #[inline]
    pub(crate) fn eval(&self, views: &[Option<ColumnBlockView<'_>>], slot: u8) -> bool {
        self.compiled.eval(views, slot)
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
        let attrs = self.attrs();
        let key_count = blocks.len() * attrs.len();
        if key_count == 0 || key_count > column_rows {
            return Ok(None);
        }
        let keys: Vec<_> = blocks
            .iter()
            .flat_map(|&(seg, block, _)| attrs.iter().map(move |&attr| (seg, attr, block)))
            .collect();
        let slices = storage.multi_get_column_blocks_pinned(&keys, attrs.len() == 1)?;
        let mut matches = vec![None; rows.len()];
        let mut views: Vec<Option<ColumnBlockView<'_>>> = vec![None; attrs.len()];
        for ((_, _, indices), block_slices) in
            blocks.into_iter().zip(slices.chunks_exact(attrs.len()))
        {
            for (view, slice) in views.iter_mut().zip(block_slices) {
                *view = slice
                    .as_ref()
                    .map(|s| ColumnBlockView::parse(s))
                    .transpose()?;
            }
            for i in indices {
                matches[i] = Some(self.eval(&views, rows[i].slot()));
            }
        }
        Ok(Some(matches))
    }
}

/// Comparison operator for compiled ordered predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmpOp {
    Lt,
    Lte,
    Gt,
    Gte,
}

/// Pre-normalized right-hand side of a compiled column predicate.
#[derive(Debug, Clone)]
enum ColRhs {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    Str(Vec<u8>),
}

/// A predicate referencing dense attribute slots and pre-normalized operands.
#[derive(Debug, Clone)]
enum ColumnPredicate {
    True,
    False,
    And(Vec<Self>),
    Or(Vec<Self>),
    Not(Box<Self>),
    IsNull(usize),
    IsNotNull(usize),
    Eq(usize, ColRhs),
    NotEq(usize, ColRhs),
    In(usize, Vec<ColRhs>),
    Cmp(usize, CmpOp, ColRhs),
}

impl ColumnPredicate {
    fn eval(&self, views: &[Option<ColumnBlockView<'_>>], slot: u8) -> bool {
        match self {
            Self::True => true,
            Self::False => false,
            Self::And(children) => children.iter().all(|c| c.eval(views, slot)),
            Self::Or(children) => children.iter().any(|c| c.eval(views, slot)),
            Self::Not(child) => !child.eval(views, slot),
            Self::IsNull(idx) => view_at(views, *idx).is_none_or(|v| !v.is_present(slot)),
            Self::IsNotNull(idx) => view_at(views, *idx).is_some_and(|v| v.is_present(slot)),
            Self::Eq(idx, rhs) => rhs_eq(views, *idx, slot, rhs),
            Self::NotEq(idx, rhs) => !rhs_eq(views, *idx, slot, rhs),
            Self::In(idx, vals) => vals.iter().any(|rhs| rhs_eq(views, *idx, slot, rhs)),
            Self::Cmp(idx, op, rhs) => {
                rhs_cmp(views, *idx, slot, rhs).is_some_and(|ord| match op {
                    CmpOp::Lt => ord == std::cmp::Ordering::Less,
                    CmpOp::Lte => ord != std::cmp::Ordering::Greater,
                    CmpOp::Gt => ord == std::cmp::Ordering::Greater,
                    CmpOp::Gte => ord != std::cmp::Ordering::Less,
                })
            }
        }
    }
}

#[inline]
fn view_at<'v, 'a>(
    views: &'v [Option<ColumnBlockView<'a>>],
    idx: usize,
) -> Option<&'v ColumnBlockView<'a>> {
    views.get(idx).and_then(|v| v.as_ref())
}

/// Resolve `attr` to its dense slot and declared type.
fn slot_for(
    schema: &CollectionSchema,
    attrs: &[AttrId],
    attr: &str,
) -> Option<(usize, AttributeType)> {
    let entry = schema.get_by_name(attr)?;
    let idx = attrs.iter().position(|&id| id == entry.id)?;
    Some((idx, entry.attr_type))
}

/// Normalize a non-null [`ScalarValue`] to `ty`, returning `None` on a type mismatch.
fn normalize_rhs(value: &ScalarValue, ty: AttributeType) -> Option<ColRhs> {
    match (ty, value) {
        (AttributeType::Bool, ScalarValue::Bool(v)) => Some(ColRhs::Bool(*v)),
        (AttributeType::Int, ScalarValue::Int(v)) => Some(ColRhs::Int(*v)),
        (AttributeType::Uint, ScalarValue::Uint(v)) => Some(ColRhs::Uint(*v)),
        (AttributeType::Float, ScalarValue::Float(v)) => Some(ColRhs::Float(*v)),
        (AttributeType::String, ScalarValue::String(v)) => Some(ColRhs::Str(v.as_bytes().to_vec())),
        _ => None,
    }
}

fn compile_predicate(
    schema: &CollectionSchema,
    attrs: &[AttrId],
    filter: &Filter,
) -> ColumnPredicate {
    match filter {
        Filter::And(children) => ColumnPredicate::And(
            children
                .iter()
                .map(|c| compile_predicate(schema, attrs, c))
                .collect(),
        ),
        Filter::Or(children) => ColumnPredicate::Or(
            children
                .iter()
                .map(|c| compile_predicate(schema, attrs, c))
                .collect(),
        ),
        Filter::Not(child) => {
            ColumnPredicate::Not(Box::new(compile_predicate(schema, attrs, child)))
        }
        Filter::IsNull(attr) => slot_for(schema, attrs, attr)
            .map_or(ColumnPredicate::False, |(idx, _)| {
                ColumnPredicate::IsNull(idx)
            }),
        Filter::IsNotNull(attr) => slot_for(schema, attrs, attr)
            .map_or(ColumnPredicate::False, |(idx, _)| {
                ColumnPredicate::IsNotNull(idx)
            }),
        Filter::Eq(attr, rhs) | Filter::NotEq(attr, rhs) => {
            let is_not = matches!(filter, Filter::NotEq(_, _));
            let Some((idx, ty)) = slot_for(schema, attrs, attr) else {
                return ColumnPredicate::False;
            };
            if matches!(rhs, ScalarValue::Null) {
                return if is_not {
                    ColumnPredicate::IsNotNull(idx)
                } else {
                    ColumnPredicate::IsNull(idx)
                };
            }
            match normalize_rhs(rhs, ty) {
                Some(rhs) if is_not => ColumnPredicate::NotEq(idx, rhs),
                Some(rhs) => ColumnPredicate::Eq(idx, rhs),
                // A type-mismatched operand can never compare equal; `NotEq` is then always true.
                None if is_not => ColumnPredicate::True,
                None => ColumnPredicate::False,
            }
        }
        Filter::In(attr, values) | Filter::NotIn(attr, values) => {
            let is_not = matches!(filter, Filter::NotIn(_, _));
            let Some((idx, ty)) = slot_for(schema, attrs, attr) else {
                return ColumnPredicate::False;
            };
            let has_null = values.iter().any(|v| matches!(v, ScalarValue::Null));
            let normalized: Vec<ColRhs> = values
                .iter()
                .filter(|v| !matches!(v, ScalarValue::Null))
                .filter_map(|v| normalize_rhs(v, ty))
                .collect();
            let pred = if has_null {
                ColumnPredicate::Or(vec![
                    ColumnPredicate::IsNull(idx),
                    ColumnPredicate::In(idx, normalized),
                ])
            } else {
                ColumnPredicate::In(idx, normalized)
            };
            if is_not {
                ColumnPredicate::Not(Box::new(pred))
            } else {
                pred
            }
        }
        Filter::Lt(attr, rhs)
        | Filter::Lte(attr, rhs)
        | Filter::Gt(attr, rhs)
        | Filter::Gte(attr, rhs) => {
            let op = match filter {
                Filter::Lt(..) => CmpOp::Lt,
                Filter::Lte(..) => CmpOp::Lte,
                Filter::Gt(..) => CmpOp::Gt,
                _ => CmpOp::Gte,
            };
            let Some((idx, ty)) = slot_for(schema, attrs, attr) else {
                return ColumnPredicate::False;
            };
            normalize_rhs(rhs, ty).map_or(ColumnPredicate::False, |rhs| {
                ColumnPredicate::Cmp(idx, op, rhs)
            })
        }
        _ => ColumnPredicate::False,
    }
}

#[inline]
fn rhs_eq(views: &[Option<ColumnBlockView<'_>>], idx: usize, slot: u8, rhs: &ColRhs) -> bool {
    let Some(view) = view_at(views, idx) else {
        return false;
    };
    match rhs {
        ColRhs::Bool(r) => view.get_bool(slot) == Some(*r),
        ColRhs::Int(r) => view.get_i64(slot) == Some(*r),
        ColRhs::Uint(r) => view.get_u64(slot) == Some(*r),
        ColRhs::Float(r) => view
            .get_f64(slot)
            .is_some_and(|v| !v.is_nan() && !r.is_nan() && v == *r),
        ColRhs::Str(r) => view.get_str(slot) == Some(r.as_slice()),
    }
}

#[inline]
fn rhs_cmp(
    views: &[Option<ColumnBlockView<'_>>],
    idx: usize,
    slot: u8,
    rhs: &ColRhs,
) -> Option<std::cmp::Ordering> {
    let view = view_at(views, idx)?;
    match rhs {
        ColRhs::Int(r) => view.get_i64(slot).map(|v| v.cmp(r)),
        ColRhs::Uint(r) => view.get_u64(slot).map(|v| v.cmp(r)),
        ColRhs::Float(r) => {
            let v = view.get_f64(slot)?;
            if v.is_nan() || r.is_nan() {
                None
            } else {
                let vn = if v == 0.0 { 0.0 } else { v };
                let rn = if *r == 0.0 { 0.0 } else { *r };
                Some(vn.total_cmp(&rn))
            }
        }
        ColRhs::Str(r) => view.get_str(slot).map(|v| v.cmp(r.as_slice())),
        ColRhs::Bool(_) => None,
    }
}

pub(super) fn filter_is_columnar_scalar_only(schema: &CollectionSchema, filter: &Filter) -> bool {
    let is_col_scalar = |attr: &str| {
        schema.get_by_name(attr).is_some_and(|e| {
            e.storage == ColumnStorage::Inline && e.is_columnar() && !e.attr_type.is_array()
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

pub(super) fn collect_filter_attr_ids(
    schema: &CollectionSchema,
    filter: &Filter,
    out: &mut Vec<AttrId>,
) {
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
    let Some(compiled) = CompiledColumnFilter::compile(schema, filter) else {
        return Ok(None);
    };
    let attrs = compiled.attrs();

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
    for &attr in attrs {
        for &(blk, _) in &by_block {
            keys.push((seg, attr, blk));
        }
    }
    let slices = storage.multi_get_column_blocks_pinned(&keys, attrs.len() == 1)?;

    let num_blocks = by_block.len();
    let mut out = RoaringBitmap::new();
    let mut views: Vec<Option<ColumnBlockView<'_>>> = vec![None; attrs.len()];

    for (bi, &(blk, mask)) in by_block.iter().enumerate() {
        for (ai, view) in views.iter_mut().enumerate() {
            let opt_slice = &slices[ai * num_blocks + bi];
            *view = opt_slice
                .as_ref()
                .map(|s| ColumnBlockView::parse(s))
                .transpose()?;
        }
        for_each_set_bit_256(&mask, |slot| {
            if compiled.eval(&views, slot) {
                out.insert((blk << 8) | (slot as u32));
            }
        });
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
