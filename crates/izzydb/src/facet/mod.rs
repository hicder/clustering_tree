//! Exact faceting engine on v3 segments: adaptive strategy dispatch over live global counts
//! (`default[0x06]`), forward [`AttrReader`] passes (`"docs"`), and inverted per-segment posting
//! intersections (`cf_attr_postings`).

mod accum;
mod dispatch;
mod finalize;
mod strat_columnar;
mod strat_global;
mod strat_postings;

use accum::*;
use dispatch::*;
use finalize::*;

pub use izzydb_core::facet::*;
use izzydb_core::config::{DocId, RowId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_core::schema::CollectionSchema;
use izzydb_storage::Storage;
use crate::filter_plan::total_live_rows;
use crate::fts_index::FtsIndex;

thread_local! {
    static STRATEGY_TRACE: std::cell::RefCell<Vec<&'static str>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn trace_strategy(name: &'static str) {
    STRATEGY_TRACE.with(|t| t.borrow_mut().push(name));
}

/// Drain the strategies chosen by facet evaluation on this thread.
#[doc(hidden)]
pub fn take_strategy_trace() -> Vec<&'static str> {
    STRATEGY_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()))
}

/// Maximum number of rows for the forward `AttrReader` pass to be preferred.
pub const FORWARD_PASS_MAX_DOCS: usize = 2048;

pub(crate) struct FacetEnv<'a> {
    pub storage: &'a Storage,
    pub schema: &'a CollectionSchema,
    pub fts_index: &'a FtsIndex,
}

impl FacetEnv<'_> {
    fn total_live_rows(&self) -> u64 {
        total_live_rows(&self.storage.segment_snapshot())
    }
}

/// The set of documents facets are computed over.
pub(crate) enum Population<'a> {
    /// Rows matching `filter` (each facet may rewrite it via its `FacetFilterMode`) and, when
    /// `rank_bm` is set, also present in `rank_bm`.
    Query {
        filter: Option<&'a Filter>,
        rank_bm: Option<&'a RowSet>,
    },
    /// An explicit live [`RowSet`] (filter modes are ignored).
    RowSet(&'a RowSet),
    /// Explicit `DocId`s (filter modes are ignored; resolved via `cf_id_map`).
    #[allow(dead_code)]
    Docs(&'a [DocId]),
}

/// Compute all `specs` over `pop`. Returns the results and the primary population size.
pub(crate) fn compute_facets(
    env: &FacetEnv<'_>,
    specs: &[FacetSpec],
    pop: Population<'_>,
) -> Result<(FacetResults, u64)> {
    validate_facet_specs(env.schema, specs)?;
    let mut out: Vec<Option<FacetResult>> = vec![None; specs.len()];

    let total = match pop {
        Population::RowSet(rs) => {
            let idx: Vec<usize> = (0..specs.len()).collect();
            let mut accs = AttrAccs::build(env, specs, &idx)?;
            let count = rs.len();
            if count > 0 {
                if count as usize <= FORWARD_PASS_MAX_DOCS {
                    let rows: Vec<RowId> = rs.iter().collect();
                    trace_strategy("docs");
                    accs.accumulate_rows(env, &rows)?;
                } else {
                    trace_strategy("leaf_masks_no_views");
                    accs.accumulate_rowset_postings(env, rs)?;
                }
            }
            accs.finalize_into(specs, count, &mut out)?;
            count
        }
        Population::Docs(docs) => {
            let mut ids = docs.to_vec();
            ids.sort_unstable();
            ids.dedup();
            let raw_rows = env.storage.multi_get_id_map(&ids)?;
            let snap = env.storage.segment_snapshot();
            let mut rows: Vec<RowId> = raw_rows
                .into_iter()
                .flatten()
                .filter(|&r| snap.is_live(r))
                .collect();
            rows.sort_unstable();
            let idx: Vec<usize> = (0..specs.len()).collect();
            let mut accs = AttrAccs::build(env, specs, &idx)?;
            accs.accumulate_rows(env, &rows)?;
            accs.finalize_into(specs, rows.len() as u64, &mut out)?;
            rows.len() as u64
        }
        Population::Query { filter, rank_bm } => {
            let mut groups: Vec<(Option<Filter>, Vec<usize>)> = vec![(filter.cloned(), Vec::new())];
            for (i, spec) in specs.iter().enumerate() {
                let eff = spec.effective_filter(filter);
                match groups.iter_mut().find(|(f, _)| *f == eff) {
                    Some((_, idxs)) => idxs.push(i),
                    None => groups.push((eff, vec![i])),
                }
            }
            let mut primary_total = 0;
            for (gi, (gfilter, idxs)) in groups.iter().enumerate() {
                let g_total = eval_group(env, specs, idxs, gfilter.as_ref(), rank_bm, &mut out)?;
                if gi == 0 {
                    primary_total = g_total;
                }
            }
            primary_total
        }
    };

    let mut results = FacetResults::default();
    for (spec, res) in specs.iter().zip(out) {
        let res = res.ok_or_else(|| {
            TreeError::InvariantViolation(format!("facet '{}' was not computed", spec.output_name()))
        })?;
        results.facets.insert(spec.output_name().to_owned(), res);
    }
    Ok((results, total))
}

// ---------------------------------------------------------------------------------------------
// Accumulators
// ---------------------------------------------------------------------------------------------
