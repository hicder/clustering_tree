//! `RowSet` filter planner, cardinality estimation via `0x06`, zone pruning, and lazy row evaluation.


mod columnar;
mod compile;
mod estimate;
mod postings;

use columnar::*;
use compile::*;
use estimate::*;
use postings::*;


use roaring::RoaringBitmap;

use izzydb_core::config::{RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_core::schema::CollectionSchema;
use izzydb_storage::codec::RowView;
use izzydb_storage::{SegmentTableSnapshot, Storage};

use crate::detached::{
    filter_mentions_detached, filter_rows_mixed_batch,
};
use crate::fts_index::FtsIndex;

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

/// Plan `filter` into [`FilterPlan`] using `0x06` counts, FTS term stats, and cached `0x12` zone maps.
#[allow(dead_code)]
pub(crate) fn plan_filter(
    storage: &Storage,
    schema: &CollectionSchema,
    filter: &Filter,
    fts: &FtsIndex,
) -> Result<(FilterPlan, u64)> {
    let mut trace = FilterPlanTrace::default();
    let plan = plan_filter_with_trace(storage, schema, filter, fts, &mut trace)?;
    Ok((plan, trace.estimated_cardinality))
}

/// Plan `filter` with diagnostic trace population.
pub(crate) fn plan_filter_with_trace(
    storage: &Storage,
    schema: &CollectionSchema,
    filter: &Filter,
    fts: &FtsIndex,
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
        fts,
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
    fts: &FtsIndex,
) -> Result<RowSet> {
    let mut trace = FilterPlanTrace::default();
    let plan = plan_filter_with_trace(storage, schema, filter, fts, &mut trace)?;
    materialize_plan(storage, schema, fts, plan)
}

/// Materialize an already-planned [`FilterPlan`] into an exact live [`RowSet`].
pub(crate) fn materialize_plan(
    storage: &Storage,
    schema: &CollectionSchema,
    fts: &FtsIndex,
    plan: FilterPlan,
) -> Result<RowSet> {
    let snap = storage.segment_snapshot();
    match plan {
        FilterPlan::All => Ok(RowSet::from_all_live(&*snap)),
        FilterPlan::Empty => Ok(RowSet::empty()),
        FilterPlan::Materialized(rs) => Ok(rs),
        FilterPlan::Lazy { residual, seed } => {
            materialize_lazy(storage, schema, fts, &snap, &residual, seed.as_ref())
        }
    }
}

/// Evaluate `residual` via columnar blocks (`cf_columns`) when sealed & eligible, or
/// `RowView::matches_filter` otherwise, over `seed` (or all live rows if `seed` is `None`).
pub(crate) fn materialize_lazy(
    storage: &Storage,
    schema: &CollectionSchema,
    fts: &FtsIndex,
    snap: &SegmentTableSnapshot,
    residual: &Filter,
    seed: Option<&RowSet>,
) -> Result<RowSet> {
    let mut segs: Vec<(SegmentId, RoaringBitmap)> = Vec::new();
    let has_detached = filter_mentions_detached(schema, residual);
    match seed {
        Some(seed_rs) => {
            let mut batch: Vec<RowId> = Vec::with_capacity(1024);
            for &(seg, ref seed_bm) in seed_rs.segments() {
                if !snap.is_visible_segment(seg) {
                    continue;
                }
                let live_cand = if snap.segment_has_deletes(seg) {
                    if let Some(live_bm) = snap.live_bitmap(seg) {
                        seed_bm & live_bm
                    } else {
                        continue;
                    }
                } else {
                    let count = snap.segment_row_count(seg).unwrap_or(0);
                    if seed_bm.max().is_some_and(|m| m >= count) {
                        let mut valid = RoaringBitmap::new();
                        valid.insert_range(0..count);
                        seed_bm & valid
                    } else {
                        seed_bm.clone()
                    }
                };
                if live_cand.is_empty() {
                    continue;
                }
                if !has_detached
                    && let Some(bm) = try_eval_column_filter_segment(
                        storage, schema, snap, seg, &live_cand, residual,
                    )?
                {
                    if !bm.is_empty() {
                        segs.push((seg, bm));
                    }
                    continue;
                }

                let mut bm = RoaringBitmap::new();
                let mut cursor = live_cand.iter().peekable();
                while cursor.peek().is_some() {
                    batch.clear();
                    for _ in 0..1024 {
                        if let Some(local) = cursor.next() {
                            batch.push(RowId::new(seg, local));
                        } else {
                            break;
                        }
                    }
                    if has_detached {
                        let matched =
                            filter_rows_mixed_batch(storage, schema, fts, residual, &batch)?;
                        for (&row, ok) in batch.iter().zip(matched) {
                            if ok {
                                bm.insert(row.local());
                            }
                        }
                    } else {
                        let slices = storage.multi_get_row_bytes_pinned(&batch)?;
                        for (&row, opt_slice) in batch.iter().zip(slices.iter()) {
                            let slice = opt_slice.as_ref().ok_or_else(|| {
                                TreeError::InvariantViolation(format!(
                                    "row {row:?} missing in cf_rows"
                                ))
                            })?;
                            let view = RowView::parse(slice)?;
                            if view.matches_filter(
                                schema,
                                residual,
                                &mut |kind, attr, query, last_as_prefix, val| {
                                    fts.matches_fts_attr(
                                        schema,
                                        kind,
                                        attr,
                                        query,
                                        last_as_prefix,
                                        val,
                                    )
                                },
                            )? {
                                bm.insert(row.local());
                            }
                        }
                    }
                }
                if !bm.is_empty() {
                    segs.push((seg, bm));
                }
            }
            Ok(RowSet::from_sorted_segs(segs))
        }
        None => {
            let mut batch: Vec<RowId> = Vec::with_capacity(1024);
            for seg in snap.visible_segments() {
                if !has_detached
                    && let Some(live_bm) = snap.live_bitmap(seg)
                    && let Some(bm) = try_eval_column_filter_segment(
                        storage, schema, snap, seg, &live_bm, residual,
                    )?
                {
                    if !bm.is_empty() {
                        segs.push((seg, bm));
                    }
                    continue;
                }

                let mut bm = RoaringBitmap::new();
                if has_detached {
                    let Some(live_bm) = snap.live_bitmap(seg) else {
                        continue;
                    };
                    let mut cursor = live_bm.iter().peekable();
                    while cursor.peek().is_some() {
                        batch.clear();
                        for _ in 0..1024 {
                            if let Some(local) = cursor.next() {
                                batch.push(RowId::new(seg, local));
                            } else {
                                break;
                            }
                        }
                        let matched =
                            filter_rows_mixed_batch(storage, schema, fts, residual, &batch)?;
                        for (&row, ok) in batch.iter().zip(matched) {
                            if ok {
                                bm.insert(row.local());
                            }
                        }
                    }
                } else {
                    storage.visit_segment_rows(seg, |row, bytes| {
                        if !snap.is_live(row) {
                            return Ok(());
                        }
                        let view = RowView::parse(bytes)?;
                        if view.matches_filter(
                            schema,
                            residual,
                            &mut |kind, attr, query, last_as_prefix, val| {
                                fts.matches_fts_attr(
                                    schema,
                                    kind,
                                    attr,
                                    query,
                                    last_as_prefix,
                                    val,
                                )
                            },
                        )? {
                            bm.insert(row.local());
                        }
                        Ok(())
                    })?;
                }
                if !bm.is_empty() {
                    segs.push((seg, bm));
                }
            }
            Ok(RowSet::from_sorted_segs(segs))
        }
    }
}
