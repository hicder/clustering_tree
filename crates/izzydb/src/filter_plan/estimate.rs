//! Cardinality estimation and plan construction.



use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::{CollectionSchema, ColumnStorage, ScalarValue};
use izzydb_storage::codec::{
    encode_present_attr_key, encode_scalar_attr_key,
};
use izzydb_storage::{SegmentTableSnapshot, Storage, ZoneTable};

use crate::detached::current_detached_gen;
use crate::fts_index::FtsIndex;


use super::*;

// ---------------------------------------------------------------------------
// Estimation + materialization
// ---------------------------------------------------------------------------

pub(super) struct PlanCtx<'a> {
    pub(super) storage: &'a Storage,
    pub(super) schema: &'a CollectionSchema,
    pub(super) fts: &'a FtsIndex,
    pub(super) snap: &'a SegmentTableSnapshot,
    pub(super) zones: &'a ZoneTable,
    pub(super) live_rows: u64,
    pub(super) materialize_limit: u64,
    pub(super) trace: &'a mut FilterPlanTrace,
}

pub(super) fn estimate_node(ctx: &PlanCtx<'_>, node: &CompiledNode) -> Result<u64> {
    match node {
        CompiledNode::All => Ok(ctx.live_rows),
        CompiledNode::Empty => Ok(0),
        CompiledNode::Fts {
            kind,
            attr,
            query,
            last_as_prefix,
            ..
        } => ctx.fts.estimate_fts_predicate(
            ctx.storage,
            ctx.schema,
            *kind,
            attr,
            query,
            *last_as_prefix,
            ctx.live_rows,
        ),
        CompiledNode::Eq { attr, val, .. } | CompiledNode::Contains { attr, val, .. } => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                return Ok(0);
            }
            let entry = ctx.schema.get_by_name(attr).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr}'"))
            })?;
            let key = encode_scalar_attr_key(entry.id, val)?;
            let cnt = get_attr_or_detached_count(ctx.storage, entry, &key)?;
            Ok(cnt.min(ctx.live_rows))
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
                sum = sum.saturating_add(get_attr_or_detached_count(ctx.storage, entry, &key)?);
            }
            if has_null {
                let present = get_attr_or_detached_count(
                    ctx.storage,
                    entry,
                    &encode_present_attr_key(entry.id),
                )?;
                sum = sum.saturating_add(ctx.live_rows.saturating_sub(present));
            }
            Ok(sum.min(ctx.live_rows))
        }
        CompiledNode::IsNotNull { attr, .. } => {
            let entry = ctx.schema.get_by_name(attr).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr}'"))
            })?;
            let present = get_attr_or_detached_count(
                ctx.storage,
                entry,
                &encode_present_attr_key(entry.id),
            )?;
            Ok(present.min(ctx.live_rows))
        }
        CompiledNode::IsNull { attr, .. } => {
            let entry = ctx.schema.get_by_name(attr).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr}'"))
            })?;
            let present = get_attr_or_detached_count(
                ctx.storage,
                entry,
                &encode_present_attr_key(entry.id),
            )?;
            Ok(ctx.live_rows.saturating_sub(present))
        }
        CompiledNode::Range { attr, lo, hi, .. } => {
            let entry = ctx.schema.get_by_name(attr).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr}'"))
            })?;
            let (attr_id, tag, lo_bytes, hi_bytes) =
                encode_range_bounds(ctx.schema, attr, lo.as_ref(), hi.as_ref())?;
            if range_is_empty(lo_bytes.as_ref(), hi_bytes.as_ref()) {
                return Ok(0);
            }
            let start = lo_bytes.as_ref().map_or(&[][..], |(b, _)| b.as_slice());
            let mut sum = 0u64;
            let cap = ctx.materialize_limit.saturating_add(1);
            let visit_cb = |val_bytes: &[u8], cnt: u64| -> Result<bool> {
                if let Some((ref lo_b, incl)) = lo_bytes
                    && (val_bytes < lo_b.as_slice() || (!incl && val_bytes == lo_b.as_slice()))
                {
                    return Ok(true);
                }
                if let Some((ref hi_b, incl)) = hi_bytes
                    && (val_bytes > hi_b.as_slice() || (!incl && val_bytes == hi_b.as_slice()))
                {
                    return Ok(false);
                }
                sum = sum.saturating_add(cnt);
                Ok(sum < cap)
            };
            if entry.storage == ColumnStorage::Detached {
                let col_gen = current_detached_gen(ctx.storage, attr_id)?;
                ctx.storage
                    .visit_detached_counts(attr_id, col_gen, tag, start, visit_cb)?;
            } else {
                ctx.storage
                    .visit_attr_counts(attr_id, tag, start, visit_cb)?;
            }
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

pub(super) fn eval_plan(ctx: &mut PlanCtx<'_>, node: &CompiledNode) -> Result<(FilterPlan, u64)> {
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
            let rs = load_postings_for_keys(ctx.storage, ctx.snap, entry, &[attr_key])?;
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
            let mut rs = load_postings_for_keys(ctx.storage, ctx.snap, entry, &keys)?;
            if has_null {
                let present = load_postings_for_keys(
                    ctx.storage,
                    ctx.snap,
                    entry,
                    &[encode_present_attr_key(entry.id)],
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
            let rs = load_postings_for_keys(
                ctx.storage,
                ctx.snap,
                entry,
                &[encode_present_attr_key(entry.id)],
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
            let present = load_postings_for_keys(
                ctx.storage,
                ctx.snap,
                entry,
                &[encode_present_attr_key(entry.id)],
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
            let entry = ctx.schema.get_by_name(attr).unwrap();
            let (attr_id, tag, lo_bytes, hi_bytes) =
                encode_range_bounds(ctx.schema, attr, lo.as_ref(), hi.as_ref())?;
            if range_is_empty(lo_bytes.as_ref(), hi_bytes.as_ref()) {
                return Ok((FilterPlan::Empty, 0));
            }
            let rs = load_postings_range(ctx, entry, attr_id, tag, lo_bytes.as_ref(), hi_bytes.as_ref())?;
            let actual = rs.len();
            if actual == 0 {
                Ok((FilterPlan::Empty, 0))
            } else {
                Ok((FilterPlan::Materialized(rs), actual))
            }
        }
        CompiledNode::Fts {
            kind,
            attr,
            query,
            last_as_prefix,
            orig,
        } => {
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
            let rs = ctx.fts.eval_fts_predicate(
                ctx.storage,
                ctx.schema,
                *kind,
                attr,
                query,
                *last_as_prefix,
            )?;
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
