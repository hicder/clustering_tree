//! Strategy dispatch and complement evaluation.

use izzydb_core::config::RowId;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::rowset::RowSet;
use crate::filter_plan::{self, FilterPlan, FilterPlanTrace};

use super::*;

pub(super) fn complement_inner(filter: &Filter) -> Option<Filter> {
    match filter {
        Filter::Not(inner) => Some((**inner).clone()),
        Filter::NotEq(a, v) => Some(Filter::Eq(a.clone(), v.clone())),
        Filter::NotIn(a, vs) => Some(Filter::In(a.clone(), vs.clone())),
        Filter::And(children) if children.len() == 1 => complement_inner(&children[0]),
        _ => None,
    }
}

pub(super) fn filter_has_complement_or_null(filter: &Filter) -> bool {
    match filter {
        Filter::Not(_) | Filter::NotEq(_, _) | Filter::NotIn(_, _) | Filter::IsNull(_) => true,
        Filter::And(ch) | Filter::Or(ch) => ch.iter().any(filter_has_complement_or_null),
        _ => false,
    }
}

pub(super) fn eval_group(
    env: &FacetEnv<'_>,
    specs: &[FacetSpec],
    idxs: &[usize],
    filter: Option<&Filter>,
    rank_bm: Option<&RowSet>,
    out: &mut [Option<FacetResult>],
) -> Result<u64> {
    let mut accs = AttrAccs::build(env, specs, idxs)?;
    let finish = |accs: &mut AttrAccs, total: u64, out: &mut [Option<FacetResult>]| -> Result<u64> {
        accs.finalize_into(specs, total, out)?;
        Ok(total)
    };

    let live_rows = env.total_live_rows();
    if live_rows == 0 {
        return finish(&mut accs, 0, out);
    }
    if let Some(rs) = rank_bm
        && rs.is_empty()
    {
        return finish(&mut accs, 0, out);
    }

    // Strategy 1: unfiltered global counts.
    if filter.is_none() && rank_bm.is_none() && !accs.has_array_ranges() {
        trace_strategy("global");
        accs.load_global_counts(env)?;
        return finish(&mut accs, live_rows, out);
    }

    // Shortcut: small BM25 match set with no attribute filter.
    if filter.is_none()
        && let Some(rs) = rank_bm
        && rs.len() as usize <= FORWARD_PASS_MAX_DOCS
    {
        let rows: Vec<RowId> = rs.iter().collect();
        trace_strategy("docs");
        accs.accumulate_rows(env, &rows)?;
        return finish(&mut accs, rows.len() as u64, out);
    }

    // Strategy 2: complement of a selective positive filter against global counts.
    if rank_bm.is_none()
        && !accs.has_array_ranges()
        && let Some(inner) = filter.and_then(complement_inner)
    {
        let mut trace = FilterPlanTrace::default();
        let inner_plan = filter_plan::plan_filter_with_trace(
            env.storage,
            env.schema,
            &inner,
            env.fts_index,
            &mut trace,
        )?;
        let small_inner = match inner_plan {
            FilterPlan::Empty => Some(RowSet::empty()),
            FilterPlan::Materialized(rs) if rs.len() as usize <= FORWARD_PASS_MAX_DOCS => Some(rs),
            FilterPlan::Lazy {
                residual,
                seed: Some(s),
            } if s.len() as usize <= FORWARD_PASS_MAX_DOCS => {
                let snap = env.storage.segment_snapshot();
                let rs = filter_plan::materialize_lazy(
                    env.storage,
                    env.schema,
                    env.fts_index,
                    &snap,
                    &residual,
                    Some(&s),
                )?;
                (rs.len() as usize <= FORWARD_PASS_MAX_DOCS).then_some(rs)
            }
            _ => None,
        };
        if let Some(inner_rs) = small_inner {
            let inner_rows: Vec<RowId> = inner_rs.iter().collect();
            trace_strategy("complement");
            let mut inner_accs = AttrAccs::build(env, specs, idxs)?;
            inner_accs.accumulate_rows(env, &inner_rows)?;
            accs.load_global_counts(env)?;
            for (acc, sub) in accs.accs.iter_mut().zip(&inner_accs.accs) {
                for (k, &c) in &sub.values {
                    let slot = acc.values.get_mut(k.as_slice()).ok_or_else(|| {
                        TreeError::InvariantViolation(format!(
                            "complement facet: value {k:?} missing from global counts"
                        ))
                    })?;
                    *slot = slot.checked_sub(c).ok_or_else(|| {
                        TreeError::InvariantViolation(
                            "complement facet: inner count exceeds global count".into(),
                        )
                    })?;
                }
                if acc.need_present {
                    acc.present = acc.present.checked_sub(sub.present).ok_or_else(|| {
                        TreeError::InvariantViolation(
                            "complement facet: inner presence exceeds global presence".into(),
                        )
                    })?;
                }
            }
            let total = live_rows.saturating_sub(inner_rows.len() as u64);
            return finish(&mut accs, total, out);
        }
    }

    // General path: materialize the group's exact live `RowSet`.
    let snap = env.storage.segment_snapshot();
    let pop_rs = match (filter, rank_bm) {
        (Some(f), Some(bm)) => {
            let f_rs =
                filter_plan::materialize_filter(env.storage, env.schema, f, env.fts_index)?;
            f_rs.and(bm)
        }
        (Some(f), None) => {
            filter_plan::materialize_filter(env.storage, env.schema, f, env.fts_index)?
        }
        (None, Some(bm)) => bm.clone().and_live(&*snap),
        (None, None) => RowSet::from_all_live(&*snap),
    };

    if pop_rs.is_empty() {
        return finish(&mut accs, 0, out);
    }

    if pop_rs.len() as usize <= FORWARD_PASS_MAX_DOCS {
        let rows: Vec<RowId> = pop_rs.iter().collect();
        trace_strategy("docs");
        accs.accumulate_rows(env, &rows)?;
        return finish(&mut accs, rows.len() as u64, out);
    }

    if filter.is_some_and(filter_has_complement_or_null) {
        trace_strategy("leaf_masks_views");
    } else {
        trace_strategy("leaf_masks_no_views");
    }
    accs.accumulate_rowset_postings(env, &pop_rs)?;
    finish(&mut accs, pop_rs.len(), out)
}
