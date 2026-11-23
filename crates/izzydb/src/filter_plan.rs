//! `RowSet` filter planner, cardinality estimation via `0x06`, zone pruning, and lazy row evaluation.

use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_core::schema::{CollectionSchema, ScalarValue};
use izzydb_storage::codec::{
    RowView, SegPostingView, ZoneEntry, encode_f64_ordered, encode_i64_ordered,
    encode_present_attr_key, encode_scalar_attr_key, encode_str_ordered, ordered_type_tag,
};
use izzydb_storage::{SegmentTableSnapshot, Storage, ZoneTable};

const DEFAULT_MIN_MATERIALIZE_LIMIT: u64 = 65_536;

/// Search regime selected by [`crate::search::search_filtered_with_trace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilteredSearchRegime {
    Unfiltered,
    Empty,
    ExactScan,
    MaskedTraversal,
    PostFilter,
}

/// Diagnostic trace emitted by [`plan_filter_with_trace`] and [`crate::Collection::search_with_trace`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterPlanTrace {
    pub estimated_cardinality: u64,
    pub materialize_limit: u64,
    pub materialized_rows: Option<u64>,
    pub is_lazy: bool,
    pub has_seed: bool,
    pub range_zones_checked: usize,
    pub range_segments_pruned: usize,
    pub range_segments_visited: usize,
    pub regime: Option<FilteredSearchRegime>,
}

/// Logical execution plan produced by the v3 filter planner.
#[derive(Debug, Clone)]
pub(crate) enum FilterPlan {
    /// Matches every currently live row.
    All,
    /// Matches zero live rows.
    Empty,
    /// Exact set of matching live rows (already ANDed with `SegmentTableSnapshot`).
    Materialized(RowSet),
    /// Residual predicate evaluated via `RowView::matches_filter` against optional live `seed`.
    Lazy {
        residual: Filter,
        seed: Option<RowSet>,
    },
}

#[inline]
pub(crate) fn total_live_rows(snap: &SegmentTableSnapshot) -> u64 {
    snap.segments
        .values()
        .filter(|e| e.is_visible())
        .map(|e| (e.meta.row_count as u64).saturating_sub(e.meta.deleted_count as u64))
        .sum()
}

#[inline]
pub(crate) fn default_materialize_limit(live_rows: u64) -> u64 {
    DEFAULT_MIN_MATERIALIZE_LIMIT.max(live_rows / 8)
}

/// Plan `filter` into [`FilterPlan`] using `0x06` counts and cached `0x12` zone maps.
#[allow(dead_code)]
pub(crate) fn plan_filter(
    storage: &Storage,
    schema: &CollectionSchema,
    filter: &Filter,
) -> Result<(FilterPlan, u64)> {
    let mut trace = FilterPlanTrace::default();
    let plan = plan_filter_with_trace(storage, schema, filter, &mut trace)?;
    Ok((plan, trace.estimated_cardinality))
}

/// Plan `filter` with diagnostic trace population.
pub(crate) fn plan_filter_with_trace(
    storage: &Storage,
    schema: &CollectionSchema,
    filter: &Filter,
    trace: &mut FilterPlanTrace,
) -> Result<FilterPlan> {
    filter.validate(schema)?;
    let snap = storage.segment_snapshot();
    let live_rows = total_live_rows(&snap);
    let materialize_limit = default_materialize_limit(live_rows);
    trace.materialize_limit = materialize_limit;

    if live_rows == 0 {
        trace.estimated_cardinality = 0;
        return Ok(FilterPlan::Empty);
    }

    let zones = storage.get_zones_cached()?;
    let mut ctx = PlanCtx {
        storage,
        schema,
        snap: &snap,
        zones: &zones,
        live_rows,
        materialize_limit,
        trace,
    };

    let compiled = compile_expr(filter)?;
    let (plan, est) = eval_plan(&mut ctx, &compiled)?;
    ctx.trace.estimated_cardinality = est;
    match &plan {
        FilterPlan::Empty => {
            ctx.trace.materialized_rows = Some(0);
        }
        FilterPlan::Materialized(rs) => {
            ctx.trace.materialized_rows = Some(rs.len());
        }
        FilterPlan::Lazy { seed, .. } => {
            ctx.trace.is_lazy = true;
            ctx.trace.has_seed = seed.is_some();
            ctx.trace.materialized_rows = seed.as_ref().map(|s| s.len());
        }
        FilterPlan::All => {}
    }
    Ok(plan)
}

/// Fully materialize `filter` into an exact live [`RowSet`] (used by `resolve_filter_doc_ids`,
/// `update_by_filter`, `delete_by_filter`, and Phase 6 facets/order-by).
pub(crate) fn materialize_filter(
    storage: &Storage,
    schema: &CollectionSchema,
    filter: &Filter,
) -> Result<RowSet> {
    let mut trace = FilterPlanTrace::default();
    let plan = plan_filter_with_trace(storage, schema, filter, &mut trace)?;
    materialize_plan(storage, schema, plan)
}

/// Materialize an already-planned [`FilterPlan`] into an exact live [`RowSet`].
pub(crate) fn materialize_plan(
    storage: &Storage,
    schema: &CollectionSchema,
    plan: FilterPlan,
) -> Result<RowSet> {
    let snap = storage.segment_snapshot();
    match plan {
        FilterPlan::All => Ok(RowSet::from_all_live(&*snap)),
        FilterPlan::Empty => Ok(RowSet::empty()),
        FilterPlan::Materialized(rs) => Ok(rs),
        FilterPlan::Lazy { residual, seed } => {
            materialize_lazy(storage, schema, &snap, &residual, seed.as_ref())
        }
    }
}

/// Evaluate `residual` via `RowView::matches_filter` over `seed` (or all live rows if `seed` is `None`).
pub(crate) fn materialize_lazy(
    storage: &Storage,
    schema: &CollectionSchema,
    snap: &SegmentTableSnapshot,
    residual: &Filter,
    seed: Option<&RowSet>,
) -> Result<RowSet> {
    reject_fts_in_filter(residual)?;
    let mut matched_rows: Vec<RowId> = Vec::new();
    match seed {
        Some(seed_rs) => {
            let mut batch: Vec<RowId> = Vec::with_capacity(1024);
            let mut cursor = seed_rs.iter().peekable();
            while cursor.peek().is_some() {
                batch.clear();
                for _ in 0..1024 {
                    if let Some(r) = cursor.next() {
                        if snap.is_live(r) {
                            batch.push(r);
                        }
                    } else {
                        break;
                    }
                }
                if batch.is_empty() {
                    continue;
                }
                let slices = storage.multi_get_row_bytes_pinned(&batch)?;
                for (&row, opt_slice) in batch.iter().zip(slices.iter()) {
                    let slice = opt_slice.as_ref().ok_or_else(|| {
                        TreeError::InvariantViolation(format!("row {row:?} missing in cf_rows"))
                    })?;
                    let view = RowView::parse(slice)?;
                    if view.matches_filter(schema, residual, &mut |_, _, _, _, _| false)? {
                        matched_rows.push(row);
                    }
                }
            }
            Ok(RowSet::from_rows(matched_rows))
        }
        None => {
            let mut segs: Vec<(SegmentId, RoaringBitmap)> = Vec::new();
            for seg in snap.visible_segments() {
                let mut bm = RoaringBitmap::new();
                storage.visit_segment_rows(seg, |row, bytes| {
                    if !snap.is_live(row) {
                        return Ok(());
                    }
                    let view = RowView::parse(bytes)?;
                    if view.matches_filter(schema, residual, &mut |_, _, _, _, _| false)? {
                        bm.insert(row.local());
                    }
                    Ok(())
                })?;
                if !bm.is_empty() {
                    segs.push((seg, bm));
                }
            }
            Ok(RowSet::from_sorted_segs(segs))
        }
    }
}

fn reject_fts_in_filter(filter: &Filter) -> Result<()> {
    match filter {
        Filter::And(ch) | Filter::Or(ch) => {
            for c in ch {
                reject_fts_in_filter(c)?;
            }
        }
        Filter::Not(c) => reject_fts_in_filter(c)?,
        f if f.as_fts_predicate().is_some() => {
            return Err(TreeError::Unsupported("v3: phase 5 (FTS filter)".into()));
        }
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Compiled normalized filter AST (combines same-attr range bounds inside And)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum CompiledNode {
    All,
    Empty,
    Eq {
        attr: String,
        val: ScalarValue,
        orig: Filter,
    },
    In {
        attr: String,
        vals: Vec<ScalarValue>,
        orig: Filter,
    },
    Contains {
        attr: String,
        val: ScalarValue,
        orig: Filter,
    },
    ContainsAny {
        attr: String,
        vals: Vec<ScalarValue>,
        orig: Filter,
    },
    IsNull {
        attr: String,
        orig: Filter,
    },
    IsNotNull {
        attr: String,
        orig: Filter,
    },
    Range {
        attr: String,
        lo: Option<(ScalarValue, bool)>,
        hi: Option<(ScalarValue, bool)>,
        orig: Filter,
    },
    And(Vec<CompiledNode>, Filter),
    Or(Vec<CompiledNode>, Filter),
    Not(Box<CompiledNode>, Filter),
}

impl CompiledNode {
    fn orig_filter(&self) -> &Filter {
        match self {
            Self::All => unreachable!(),
            Self::Empty => unreachable!(),
            Self::Eq { orig, .. }
            | Self::In { orig, .. }
            | Self::Contains { orig, .. }
            | Self::ContainsAny { orig, .. }
            | Self::IsNull { orig, .. }
            | Self::IsNotNull { orig, .. }
            | Self::Range { orig, .. }
            | Self::And(_, orig)
            | Self::Or(_, orig)
            | Self::Not(_, orig) => orig,
        }
    }
}

fn compile_expr(filter: &Filter) -> Result<CompiledNode> {
    if filter.as_fts_predicate().is_some() {
        return Err(TreeError::Unsupported("v3: phase 5 (FTS filter)".into()));
    }
    Ok(match filter {
        Filter::Eq(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else if matches!(val, ScalarValue::Null) {
                CompiledNode::IsNull {
                    attr: attr.clone(),
                    orig: filter.clone(),
                }
            } else {
                CompiledNode::Eq {
                    attr: attr.clone(),
                    val: val.clone(),
                    orig: filter.clone(),
                }
            }
        }
        Filter::NotEq(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::All
            } else if matches!(val, ScalarValue::Null) {
                CompiledNode::IsNotNull {
                    attr: attr.clone(),
                    orig: filter.clone(),
                }
            } else {
                CompiledNode::Not(
                    Box::new(CompiledNode::Eq {
                        attr: attr.clone(),
                        val: val.clone(),
                        orig: Filter::Eq(attr.clone(), val.clone()),
                    }),
                    filter.clone(),
                )
            }
        }
        Filter::In(attr, vals) => {
            if vals.is_empty() {
                CompiledNode::Empty
            } else {
                CompiledNode::In {
                    attr: attr.clone(),
                    vals: vals.clone(),
                    orig: filter.clone(),
                }
            }
        }
        Filter::NotIn(attr, vals) => {
            if vals.is_empty() {
                CompiledNode::All
            } else {
                CompiledNode::Not(
                    Box::new(CompiledNode::In {
                        attr: attr.clone(),
                        vals: vals.clone(),
                        orig: Filter::In(attr.clone(), vals.clone()),
                    }),
                    filter.clone(),
                )
            }
        }
        Filter::Contains(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Contains {
                    attr: attr.clone(),
                    val: val.clone(),
                    orig: filter.clone(),
                }
            }
        }
        Filter::ContainsAny(attr, vals) => {
            if vals.is_empty() {
                CompiledNode::Empty
            } else {
                CompiledNode::ContainsAny {
                    attr: attr.clone(),
                    vals: vals.clone(),
                    orig: filter.clone(),
                }
            }
        }
        Filter::IsNull(attr) => CompiledNode::IsNull {
            attr: attr.clone(),
            orig: filter.clone(),
        },
        Filter::IsNotNull(attr) => CompiledNode::IsNotNull {
            attr: attr.clone(),
            orig: filter.clone(),
        },
        Filter::Gt(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Range {
                    attr: attr.clone(),
                    lo: Some((val.clone(), false)),
                    hi: None,
                    orig: filter.clone(),
                }
            }
        }
        Filter::Gte(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Range {
                    attr: attr.clone(),
                    lo: Some((val.clone(), true)),
                    hi: None,
                    orig: filter.clone(),
                }
            }
        }
        Filter::Lt(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Range {
                    attr: attr.clone(),
                    lo: None,
                    hi: Some((val.clone(), false)),
                    orig: filter.clone(),
                }
            }
        }
        Filter::Lte(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Range {
                    attr: attr.clone(),
                    lo: None,
                    hi: Some((val.clone(), true)),
                    orig: filter.clone(),
                }
            }
        }
        Filter::And(children) => {
            if children.is_empty() {
                return Ok(CompiledNode::All);
            }
            let mut compiled_children = Vec::new();
            let mut ranges_by_attr: BTreeMap<String, (Option<(ScalarValue, bool)>, Option<(ScalarValue, bool)>, Vec<Filter>)> =
                BTreeMap::new();
            for ch in children {
                let c = compile_expr(ch)?;
                match c {
                    CompiledNode::Empty => return Ok(CompiledNode::Empty),
                    CompiledNode::All => continue,
                    CompiledNode::Range { attr, lo, hi, orig } => {
                        let entry = ranges_by_attr.entry(attr).or_insert((None, None, Vec::new()));
                        if let Some((new_lo, new_incl)) = lo {
                            entry.0 = Some(tighten_lo(entry.0.take(), new_lo, new_incl));
                        }
                        if let Some((new_hi, new_incl)) = hi {
                            entry.1 = Some(tighten_hi(entry.1.take(), new_hi, new_incl));
                        }
                        entry.2.push(orig);
                    }
                    other => compiled_children.push(other),
                }
            }
            for (attr, (lo, hi, origs)) in ranges_by_attr {
                let orig = if origs.len() == 1 {
                    origs.into_iter().next().unwrap()
                } else {
                    Filter::And(origs)
                };
                compiled_children.push(CompiledNode::Range { attr, lo, hi, orig });
            }
            if compiled_children.is_empty() {
                CompiledNode::All
            } else if compiled_children.len() == 1 {
                compiled_children.into_iter().next().unwrap()
            } else {
                CompiledNode::And(compiled_children, filter.clone())
            }
        }
        Filter::Or(children) => {
            if children.is_empty() {
                return Ok(CompiledNode::Empty);
            }
            let mut compiled_children = Vec::new();
            for ch in children {
                let c = compile_expr(ch)?;
                match c {
                    CompiledNode::All => return Ok(CompiledNode::All),
                    CompiledNode::Empty => continue,
                    other => compiled_children.push(other),
                }
            }
            if compiled_children.is_empty() {
                CompiledNode::Empty
            } else if compiled_children.len() == 1 {
                compiled_children.into_iter().next().unwrap()
            } else {
                CompiledNode::Or(compiled_children, filter.clone())
            }
        }
        Filter::Not(child) => match compile_expr(child)? {
            CompiledNode::All => CompiledNode::Empty,
            CompiledNode::Empty => CompiledNode::All,
            other => CompiledNode::Not(Box::new(other), filter.clone()),
        },
        other => {
            if other.as_fts_predicate().is_some() {
                return Err(TreeError::Unsupported("v3: phase 5 (FTS filter)".into()));
            }
            unreachable!("all Filter variants handled")
        }
    })
}

fn tighten_lo(
    cur: Option<(ScalarValue, bool)>,
    new_v: ScalarValue,
    new_incl: bool,
) -> (ScalarValue, bool) {
    let Some((cur_v, cur_incl)) = cur else {
        return (new_v, new_incl);
    };
    match cmp_scalar_ordered(&cur_v, &new_v) {
        std::cmp::Ordering::Less => (new_v, new_incl),
        std::cmp::Ordering::Greater => (cur_v, cur_incl),
        std::cmp::Ordering::Equal => (cur_v, cur_incl && new_incl),
    }
}

fn tighten_hi(
    cur: Option<(ScalarValue, bool)>,
    new_v: ScalarValue,
    new_incl: bool,
) -> (ScalarValue, bool) {
    let Some((cur_v, cur_incl)) = cur else {
        return (new_v, new_incl);
    };
    match cmp_scalar_ordered(&cur_v, &new_v) {
        std::cmp::Ordering::Greater => (new_v, new_incl),
        std::cmp::Ordering::Less => (cur_v, cur_incl),
        std::cmp::Ordering::Equal => (cur_v, cur_incl && new_incl),
    }
}

fn cmp_scalar_ordered(a: &ScalarValue, b: &ScalarValue) -> std::cmp::Ordering {
    match (a, b) {
        (ScalarValue::Int(x), ScalarValue::Int(y)) => x.cmp(y),
        (ScalarValue::Uint(x), ScalarValue::Uint(y)) => x.cmp(y),
        (ScalarValue::Float(x), ScalarValue::Float(y)) => {
            let xn = if *x == 0.0 { 0.0 } else { *x };
            let yn = if *y == 0.0 { 0.0 } else { *y };
            xn.total_cmp(&yn)
        }
        (ScalarValue::String(x), ScalarValue::String(y)) => x.cmp(y),
        _ => std::cmp::Ordering::Equal,
    }
}

// ---------------------------------------------------------------------------
// Estimation + materialization
// ---------------------------------------------------------------------------

struct PlanCtx<'a> {
    storage: &'a Storage,
    schema: &'a CollectionSchema,
    snap: &'a SegmentTableSnapshot,
    zones: &'a ZoneTable,
    live_rows: u64,
    materialize_limit: u64,
    trace: &'a mut FilterPlanTrace,
}

fn estimate_node(ctx: &PlanCtx<'_>, node: &CompiledNode) -> Result<u64> {
    match node {
        CompiledNode::All => Ok(ctx.live_rows),
        CompiledNode::Empty => Ok(0),
        CompiledNode::Eq { attr, val, .. } | CompiledNode::Contains { attr, val, .. } => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                return Ok(0);
            }
            let entry = ctx.schema.get_by_name(attr).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr}'"))
            })?;
            let key = encode_scalar_attr_key(entry.id, val)?;
            Ok(ctx.storage.get_attr_count(&key)?.min(ctx.live_rows))
        }
        CompiledNode::In { attr, vals, .. } | CompiledNode::ContainsAny { attr, vals, .. } => {
            let entry = ctx.schema.get_by_name(attr).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr}'"))
            })?;
            let mut sum = 0u64;
            let mut has_null = false;
            let mut seen_keys: Vec<Vec<u8>> = Vec::with_capacity(vals.len());
            for v in vals {
                if matches!(v, ScalarValue::Null) {
                    has_null = true;
                    continue;
                }
                if matches!(v, ScalarValue::Float(f) if f.is_nan()) {
                    continue;
                }
                seen_keys.push(encode_scalar_attr_key(entry.id, v)?);
            }
            seen_keys.sort_unstable();
            seen_keys.dedup();
            for key in seen_keys {
                sum = sum.saturating_add(ctx.storage.get_attr_count(&key)?);
            }
            if has_null {
                let present = ctx
                    .storage
                    .get_attr_count(&encode_present_attr_key(entry.id))?;
                sum = sum.saturating_add(ctx.live_rows.saturating_sub(present));
            }
            Ok(sum.min(ctx.live_rows))
        }
        CompiledNode::IsNotNull { attr, .. } => {
            let entry = ctx.schema.get_by_name(attr).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr}'"))
            })?;
            let present = ctx
                .storage
                .get_attr_count(&encode_present_attr_key(entry.id))?;
            Ok(present.min(ctx.live_rows))
        }
        CompiledNode::IsNull { attr, .. } => {
            let entry = ctx.schema.get_by_name(attr).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr}'"))
            })?;
            let present = ctx
                .storage
                .get_attr_count(&encode_present_attr_key(entry.id))?;
            Ok(ctx.live_rows.saturating_sub(present))
        }
        CompiledNode::Range { attr, lo, hi, .. } => {
            let (attr_id, tag, lo_bytes, hi_bytes) =
                encode_range_bounds(ctx.schema, attr, lo.as_ref(), hi.as_ref())?;
            if range_is_empty(lo_bytes.as_ref(), hi_bytes.as_ref()) {
                return Ok(0);
            }
            let start = lo_bytes.as_ref().map_or(&[][..], |(b, _)| b.as_slice());
            let mut sum = 0u64;
            let cap = ctx.materialize_limit.saturating_add(1);
            ctx.storage
                .visit_attr_counts(attr_id, tag, start, |val_bytes, cnt| {
                    if let Some((ref lo_b, incl)) = lo_bytes
                        && (val_bytes < lo_b.as_slice()
                            || (!incl && val_bytes == lo_b.as_slice()))
                    {
                        return Ok(true);
                    }
                    if let Some((ref hi_b, incl)) = hi_bytes
                        && (val_bytes > hi_b.as_slice()
                            || (!incl && val_bytes == hi_b.as_slice()))
                    {
                        return Ok(false);
                    }
                    sum = sum.saturating_add(cnt);
                    Ok(sum < cap)
                })?;
            Ok(sum.min(ctx.live_rows))
        }
        CompiledNode::And(children, _) => {
            let mut min_est = ctx.live_rows;
            for ch in children {
                let e = estimate_node(ctx, ch)?;
                if e == 0 {
                    return Ok(0);
                }
                min_est = min_est.min(e);
            }
            Ok(min_est)
        }
        CompiledNode::Or(children, _) => {
            let mut sum = 0u64;
            for ch in children {
                sum = sum.saturating_add(estimate_node(ctx, ch)?);
                if sum >= ctx.live_rows {
                    return Ok(ctx.live_rows);
                }
            }
            Ok(sum)
        }
        CompiledNode::Not(child, _) => {
            let inner = estimate_node(ctx, child)?;
            Ok(ctx.live_rows.saturating_sub(inner))
        }
    }
}

fn eval_plan(ctx: &mut PlanCtx<'_>, node: &CompiledNode) -> Result<(FilterPlan, u64)> {
    match node {
        CompiledNode::All => Ok((FilterPlan::All, ctx.live_rows)),
        CompiledNode::Empty => Ok((FilterPlan::Empty, 0)),
        CompiledNode::Eq { attr, val, orig } | CompiledNode::Contains { attr, val, orig } => {
            let est = estimate_node(ctx, node)?;
            if est == 0 {
                return Ok((FilterPlan::Empty, 0));
            }
            if est > ctx.materialize_limit {
                return Ok((
                    FilterPlan::Lazy {
                        residual: orig.clone(),
                        seed: None,
                    },
                    est,
                ));
            }
            let entry = ctx.schema.get_by_name(attr).unwrap();
            let attr_key = encode_scalar_attr_key(entry.id, val)?;
            let rs = load_postings_for_key(ctx.storage, ctx.snap, &attr_key)?;
            let actual = rs.len();
            if actual == 0 {
                Ok((FilterPlan::Empty, 0))
            } else {
                Ok((FilterPlan::Materialized(rs), actual))
            }
        }
        CompiledNode::In { attr, vals, orig } | CompiledNode::ContainsAny { attr, vals, orig } => {
            let est = estimate_node(ctx, node)?;
            if est == 0 {
                return Ok((FilterPlan::Empty, 0));
            }
            if est > ctx.materialize_limit {
                return Ok((
                    FilterPlan::Lazy {
                        residual: orig.clone(),
                        seed: None,
                    },
                    est,
                ));
            }
            let entry = ctx.schema.get_by_name(attr).unwrap();
            let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
            let mut has_null = false;
            let mut keys: Vec<Vec<u8>> = Vec::with_capacity(vals.len());
            for v in vals {
                if matches!(v, ScalarValue::Null) {
                    has_null = true;
                    continue;
                }
                if matches!(v, ScalarValue::Float(f) if f.is_nan()) {
                    continue;
                }
                keys.push(encode_scalar_attr_key(entry.id, v)?);
            }
            keys.sort_unstable();
            keys.dedup();
            for key in keys {
                merge_postings_for_key(ctx.storage, ctx.snap, &key, &mut by_seg)?;
            }
            let mut rs = finalize_seg_map(ctx.snap, by_seg);
            if has_null {
                let present = load_postings_for_key(
                    ctx.storage,
                    ctx.snap,
                    &encode_present_attr_key(entry.id),
                )?;
                let null_rows = RowSet::from_all_live(ctx.snap).and_not(&present);
                rs = rs.or(&null_rows);
            }
            let actual = rs.len();
            if actual == 0 {
                Ok((FilterPlan::Empty, 0))
            } else {
                Ok((FilterPlan::Materialized(rs), actual))
            }
        }
        CompiledNode::IsNotNull { attr, orig } => {
            let est = estimate_node(ctx, node)?;
            if est == 0 {
                return Ok((FilterPlan::Empty, 0));
            }
            if est > ctx.materialize_limit {
                return Ok((
                    FilterPlan::Lazy {
                        residual: orig.clone(),
                        seed: None,
                    },
                    est,
                ));
            }
            let entry = ctx.schema.get_by_name(attr).unwrap();
            let rs = load_postings_for_key(
                ctx.storage,
                ctx.snap,
                &encode_present_attr_key(entry.id),
            )?;
            let actual = rs.len();
            if actual == 0 {
                Ok((FilterPlan::Empty, 0))
            } else {
                Ok((FilterPlan::Materialized(rs), actual))
            }
        }
        CompiledNode::IsNull { attr, orig } => {
            let est = estimate_node(ctx, node)?;
            if est == 0 {
                return Ok((FilterPlan::Empty, 0));
            }
            if est > ctx.materialize_limit {
                return Ok((
                    FilterPlan::Lazy {
                        residual: orig.clone(),
                        seed: None,
                    },
                    est,
                ));
            }
            let entry = ctx.schema.get_by_name(attr).unwrap();
            let present = load_postings_for_key(
                ctx.storage,
                ctx.snap,
                &encode_present_attr_key(entry.id),
            )?;
            let rs = RowSet::from_all_live(ctx.snap).and_not(&present);
            let actual = rs.len();
            if actual == 0 {
                Ok((FilterPlan::Empty, 0))
            } else {
                Ok((FilterPlan::Materialized(rs), actual))
            }
        }
        CompiledNode::Range { attr, lo, hi, orig } => {
            let est = estimate_node(ctx, node)?;
            if est == 0 {
                return Ok((FilterPlan::Empty, 0));
            }
            if est > ctx.materialize_limit {
                return Ok((
                    FilterPlan::Lazy {
                        residual: orig.clone(),
                        seed: None,
                    },
                    est,
                ));
            }
            let (attr_id, tag, lo_bytes, hi_bytes) =
                encode_range_bounds(ctx.schema, attr, lo.as_ref(), hi.as_ref())?;
            if range_is_empty(lo_bytes.as_ref(), hi_bytes.as_ref()) {
                return Ok((FilterPlan::Empty, 0));
            }
            let rs = load_postings_range(ctx, attr_id, tag, lo_bytes.as_ref(), hi_bytes.as_ref())?;
            let actual = rs.len();
            if actual == 0 {
                Ok((FilterPlan::Empty, 0))
            } else {
                Ok((FilterPlan::Materialized(rs), actual))
            }
        }
        CompiledNode::Not(child, orig) => {
            let est = estimate_node(ctx, node)?;
            if est == 0 {
                return Ok((FilterPlan::Empty, 0));
            }
            if est > ctx.materialize_limit {
                return Ok((
                    FilterPlan::Lazy {
                        residual: orig.clone(),
                        seed: None,
                    },
                    est,
                ));
            }
            let (inner_plan, _) = eval_plan(ctx, child)?;
            match inner_plan {
                FilterPlan::Empty => Ok((FilterPlan::All, ctx.live_rows)),
                FilterPlan::All => Ok((FilterPlan::Empty, 0)),
                FilterPlan::Materialized(rs) => {
                    let comp = RowSet::from_all_live(ctx.snap).and_not(&rs);
                    let len = comp.len();
                    if len == 0 {
                        Ok((FilterPlan::Empty, 0))
                    } else {
                        Ok((FilterPlan::Materialized(comp), len))
                    }
                }
                FilterPlan::Lazy { .. } => Ok((
                    FilterPlan::Lazy {
                        residual: orig.clone(),
                        seed: None,
                    },
                    est,
                )),
            }
        }
        CompiledNode::And(children, orig) => {
            let mut estimated: Vec<(u64, &CompiledNode)> = Vec::with_capacity(children.len());
            for ch in children {
                let est = estimate_node(ctx, ch)?;
                if est == 0 {
                    return Ok((FilterPlan::Empty, 0));
                }
                estimated.push((est, ch));
            }
            estimated.sort_unstable_by_key(|&(est, _)| est);

            let mut acc: Option<RowSet> = None;
            let mut residual_filters: Vec<Filter> = Vec::new();
            let mut min_est = ctx.live_rows;

            for (est, ch) in estimated {
                min_est = min_est.min(est);
                let should_try = est <= ctx.materialize_limit || acc.is_none();
                if !should_try {
                    residual_filters.push(ch.orig_filter().clone());
                    continue;
                }
                let (sub_plan, _) = eval_plan(ctx, ch)?;
                match sub_plan {
                    FilterPlan::Empty => return Ok((FilterPlan::Empty, 0)),
                    FilterPlan::All => {}
                    FilterPlan::Materialized(rs) => {
                        let next = match acc.take() {
                            Some(prev) => prev.and(&rs),
                            None => rs,
                        };
                        if next.is_empty() {
                            return Ok((FilterPlan::Empty, 0));
                        }
                        acc = Some(next);
                    }
                    FilterPlan::Lazy {
                        residual,
                        seed: sub_seed,
                    } => {
                        if let Some(ss) = sub_seed {
                            let next = match acc.take() {
                                Some(prev) => prev.and(&ss),
                                None => ss,
                            };
                            if next.is_empty() {
                                return Ok((FilterPlan::Empty, 0));
                            }
                            acc = Some(next);
                        }
                        residual_filters.push(residual);
                    }
                }
            }

            if residual_filters.is_empty() {
                match acc {
                    Some(rs) => {
                        let len = rs.len();
                        if len == 0 {
                            Ok((FilterPlan::Empty, 0))
                        } else {
                            Ok((FilterPlan::Materialized(rs), len))
                        }
                    }
                    None => Ok((FilterPlan::All, ctx.live_rows)),
                }
            } else {
                let residual = if residual_filters.len() == 1 {
                    residual_filters.into_iter().next().unwrap()
                } else if residual_filters.len() == children.len() {
                    orig.clone()
                } else {
                    Filter::And(residual_filters)
                };
                let final_est = acc.as_ref().map_or(min_est, |s| s.len().min(min_est));
                Ok((FilterPlan::Lazy { residual, seed: acc }, final_est))
            }
        }
        CompiledNode::Or(children, orig) => {
            let est = estimate_node(ctx, node)?;
            if est == 0 {
                return Ok((FilterPlan::Empty, 0));
            }
            if est > ctx.materialize_limit {
                return Ok((
                    FilterPlan::Lazy {
                        residual: orig.clone(),
                        seed: None,
                    },
                    est,
                ));
            }
            let mut acc = RowSet::empty();
            for ch in children {
                let (sub_plan, _) = eval_plan(ctx, ch)?;
                match sub_plan {
                    FilterPlan::All => return Ok((FilterPlan::All, ctx.live_rows)),
                    FilterPlan::Empty => {}
                    FilterPlan::Materialized(rs) => {
                        acc = acc.or(&rs);
                    }
                    FilterPlan::Lazy { .. } => {
                        return Ok((
                            FilterPlan::Lazy {
                                residual: orig.clone(),
                                seed: None,
                            },
                            est,
                        ));
                    }
                }
            }
            let actual = acc.len();
            if actual == 0 {
                Ok((FilterPlan::Empty, 0))
            } else {
                Ok((FilterPlan::Materialized(acc), actual))
            }
        }
    }
}

fn load_postings_for_key(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    attr_key: &[u8],
) -> Result<RowSet> {
    let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    merge_postings_for_key(storage, snap, attr_key, &mut by_seg)?;
    Ok(finalize_seg_map(snap, by_seg))
}

fn merge_postings_for_key(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    attr_key: &[u8],
    by_seg: &mut BTreeMap<SegmentId, RoaringBitmap>,
) -> Result<()> {
    storage.visit_attr_postings_for_value(attr_key, |seg, bytes| {
        if !snap.is_visible_segment(seg) {
            return Ok(());
        }
        let dst = by_seg.entry(seg).or_default();
        SegPostingView::parse(bytes)?.or_into(dst)?;
        Ok(())
    })
}

fn finalize_seg_map(
    snap: &SegmentTableSnapshot,
    by_seg: BTreeMap<SegmentId, RoaringBitmap>,
) -> RowSet {
    let pairs: Vec<(SegmentId, RoaringBitmap)> = by_seg.into_iter().collect();
    RowSet::from_sorted_segs(pairs).and_live(snap)
}

fn encode_range_bounds(
    schema: &CollectionSchema,
    attr: &str,
    lo: Option<&(ScalarValue, bool)>,
    hi: Option<&(ScalarValue, bool)>,
) -> Result<(AttrId, u8, Option<(Vec<u8>, bool)>, Option<(Vec<u8>, bool)>)> {
    let entry = schema
        .get_by_name(attr)
        .ok_or_else(|| TreeError::InvalidFilter(format!("unknown attribute '{attr}'")))?;
    let tag = ordered_type_tag(entry.attr_type)?;
    let enc = |v: &ScalarValue| -> Result<Vec<u8>> {
        match v {
            ScalarValue::Int(x) => Ok(encode_i64_ordered(*x).to_vec()),
            ScalarValue::Uint(x) => Ok(x.to_be_bytes().to_vec()),
            ScalarValue::Float(x) => Ok(encode_f64_ordered(*x)?.to_vec()),
            ScalarValue::String(s) => {
                let mut out = Vec::with_capacity(s.len() + 2);
                encode_str_ordered(s.as_bytes(), &mut out);
                Ok(out)
            }
            other => Err(TreeError::InvalidFilter(format!(
                "invalid range bound value {other:?}"
            ))),
        }
    };
    let lo_b = lo
        .map(|(v, incl)| Ok::<_, TreeError>((enc(v)?, *incl)))
        .transpose()?;
    let hi_b = hi
        .map(|(v, incl)| Ok::<_, TreeError>((enc(v)?, *incl)))
        .transpose()?;
    Ok((entry.id, tag, lo_b, hi_b))
}

fn range_is_empty(lo: Option<&(Vec<u8>, bool)>, hi: Option<&(Vec<u8>, bool)>) -> bool {
    if let (Some((lo_b, lo_incl)), Some((hi_b, hi_incl))) = (lo, hi) {
        match lo_b.cmp(hi_b) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Equal => !(*lo_incl && *hi_incl),
            std::cmp::Ordering::Less => false,
        }
    } else {
        false
    }
}

fn zone_overlaps(
    zone: &ZoneEntry,
    lo: Option<&(Vec<u8>, bool)>,
    hi: Option<&(Vec<u8>, bool)>,
) -> bool {
    if let Some((lo_b, lo_incl)) = lo {
        match zone.max.as_slice().cmp(lo_b.as_slice()) {
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal if !*lo_incl => return false,
            _ => {}
        }
    }
    if let Some((hi_b, hi_incl)) = hi {
        match zone.min.as_slice().cmp(hi_b.as_slice()) {
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal if !*hi_incl => return false,
            _ => {}
        }
    }
    true
}

fn load_postings_range(
    ctx: &mut PlanCtx<'_>,
    attr_id: AttrId,
    tag: u8,
    lo: Option<&(Vec<u8>, bool)>,
    hi: Option<&(Vec<u8>, bool)>,
) -> Result<RowSet> {
    let seg_zones: Option<&HashMap<SegmentId, ZoneEntry>> = ctx.zones.get(&(attr_id, tag));
    let mut eligible_segs = RoaringBitmap::new();
    for seg in ctx.snap.visible_segments() {
        ctx.trace.range_zones_checked += 1;
        let overlaps = seg_zones
            .and_then(|m| m.get(&seg))
            .is_some_and(|z| zone_overlaps(z, lo, hi));
        if overlaps {
            eligible_segs.insert(seg);
        } else {
            ctx.trace.range_segments_pruned += 1;
        }
    }
    if eligible_segs.is_empty() {
        return Ok(RowSet::empty());
    }

    let start_val = lo.map_or(&[][..], |(b, _)| b.as_slice());
    let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    let mut visited_segs = RoaringBitmap::new();

    ctx.storage
        .visit_attr_postings_range(attr_id, tag, start_val, |val_bytes, seg, bytes| {
            if let Some((lo_b, lo_incl)) = lo
                && (val_bytes < lo_b.as_slice() || (!*lo_incl && val_bytes == lo_b.as_slice()))
            {
                return Ok(true);
            }
            if let Some((hi_b, hi_incl)) = hi
                && (val_bytes > hi_b.as_slice() || (!*hi_incl && val_bytes == hi_b.as_slice()))
            {
                return Ok(false);
            }
            if !eligible_segs.contains(seg) {
                return Ok(true);
            }
            visited_segs.insert(seg);
            let dst = by_seg.entry(seg).or_default();
            SegPostingView::parse(bytes)?.or_into(dst)?;
            Ok(true)
        })?;

    ctx.trace.range_segments_visited += visited_segs.len() as usize;
    Ok(finalize_seg_map(ctx.snap, by_seg))
}
