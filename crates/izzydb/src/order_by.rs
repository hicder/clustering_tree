//! Attribute-ordered top-k queries (`ORDER BY`) over v3 `cf_attr_postings`
//! and [`AttrReader`].
//!
//! * Broad / unfiltered queries stream `cf_attr_postings` in `(value, seg)` key order
//!   (forward for `Asc`, reverse for `Desc`), merge segments per value into a [`RowSet`],
//!   AND with segment liveness + filter, and stop after `top_k` rows at the end of a value group.
//! * Selective filters (materialized `<= 8,192` rows) read sort keys via [`AttrReader`]
//!   and sort in memory.
//! * Documents whose primary key is absent / `Null` are placed first or last per [`NullsOrder`].
//!
//! Ties are broken by the next `order_by` key, then by ascending `DocId`.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, DocId, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::facet::{FacetSpec, FacetedSearchResponse};
use izzydb_core::filter::Filter;
use izzydb_core::rank::{NullsOrder, OrderBySpec, OrderedResult, SortDirection, validate_order_by};
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_core::schema::{ColumnStorage, ScalarValue};
use izzydb_storage::SegmentTableSnapshot;
use izzydb_storage::codec::{
    SegPostingView, decode_scalar_val_bytes, ordered_type_tag,
};

use crate::attr_reader::{AttrReader, ColumnReader};
use crate::collection::Collection;
use crate::facet::Population;
use crate::filter_plan::{self, FilterPlan, FilterPlanTrace};
use crate::segments::RowResolver;

/// Materialized `RowSet` cardinality at or below which `query_ordered` sorts in memory instead of
/// streaming `cf_attr_postings`.
const ORDER_BY_IN_MEMORY_THRESHOLD: u64 = 8_192;

fn scalar_cmp(a: &ScalarValue, b: &ScalarValue) -> Ordering {
    match (a, b) {
        (ScalarValue::Int(x), ScalarValue::Int(y)) => x.cmp(y),
        (ScalarValue::Uint(x), ScalarValue::Uint(y)) => x.cmp(y),
        (ScalarValue::Float(x), ScalarValue::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (ScalarValue::String(x), ScalarValue::String(y)) => x.as_bytes().cmp(y.as_bytes()),
        (ScalarValue::Bool(x), ScalarValue::Bool(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

/// Compare two key tuples under `specs` (nulls placed per spec regardless of direction).
fn cmp_keys(specs: &[OrderBySpec], a: &[ScalarValue], b: &[ScalarValue]) -> Ordering {
    for (spec, (x, y)) in specs.iter().zip(a.iter().zip(b)) {
        let ord = match (matches!(x, ScalarValue::Null), matches!(y, ScalarValue::Null)) {
            (true, true) => Ordering::Equal,
            (true, false) => match spec.nulls {
                NullsOrder::First => Ordering::Less,
                NullsOrder::Last => Ordering::Greater,
            },
            (false, true) => match spec.nulls {
                NullsOrder::First => Ordering::Greater,
                NullsOrder::Last => Ordering::Less,
            },
            (false, false) => {
                let o = scalar_cmp(x, y);
                match spec.direction {
                    SortDirection::Asc => o,
                    SortDirection::Desc => o.reverse(),
                }
            }
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

pub(crate) enum OrderPopulation {
    AllLive,
    RowSet(RowSet),
    Lazy {
        seed: Option<RowSet>,
        residual: Filter,
    },
}

impl Collection {
    /// Top-`top_k` documents ordered by one or more attributes, optionally restricted by `filter`.
    ///
    /// The primary key must be a `filterable` ordered scalar (`Int` / `Uint` / `Float` / `String`);
    /// secondary keys only need to be ordered scalars. Equal key tuples tie-break by ascending `DocId`.
    pub fn query_ordered(
        &self,
        order_by: &[OrderBySpec],
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<OrderedResult>> {
        validate_order_by(&self.schema, order_by)?;
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let pop = match filter {
            None => OrderPopulation::AllLive,
            Some(f) => {
                let mut trace = FilterPlanTrace::default();
                match filter_plan::plan_filter_with_trace(
                    &self.storage,
                    &self.schema,
                    f,
                    &self.fts_index,
                    &mut trace,
                )? {
                    FilterPlan::Empty => return Ok(Vec::new()),
                    FilterPlan::All => OrderPopulation::AllLive,
                    FilterPlan::Materialized(rs) => {
                        if rs.is_empty() {
                            return Ok(Vec::new());
                        }
                        if rs.len() <= ORDER_BY_IN_MEMORY_THRESHOLD {
                            let rows: Vec<RowId> = rs.iter().collect();
                            return Ok(to_results(self.sort_rows(order_by, &rows, top_k)?));
                        }
                        OrderPopulation::RowSet(rs)
                    }
                    FilterPlan::Lazy { residual, seed } => {
                        if let Some(ref s) = seed
                            && s.len() <= ORDER_BY_IN_MEMORY_THRESHOLD
                        {
                            let snap = self.storage.segment_snapshot();
                            let rs = filter_plan::materialize_lazy(
                                &self.storage,
                                &self.schema,
                                &self.fts_index,
                                &snap,
                                &residual,
                                Some(s),
                            )?;
                            let rows: Vec<RowId> = rs.iter().collect();
                            return Ok(to_results(self.sort_rows(order_by, &rows, top_k)?));
                        }
                        OrderPopulation::Lazy { seed, residual }
                    }
                }
            }
        };

        let primary = &order_by[0];
        let mut results: Vec<(DocId, Vec<ScalarValue>)> = Vec::new();

        if primary.nulls == NullsOrder::First {
            let null_rows = self.null_primary_rows(primary, filter)?;
            results.extend(self.sort_rows(order_by, &null_rows, top_k)?);
        }

        if results.len() < top_k {
            let reader = ColumnReader::new(&self.storage, &self.schema)?;
            self.stream_value_groups(primary, &pop, &mut |value, group_rows| {
                let remaining = top_k - results.len();
                if order_by.len() == 1 {
                    let doc_ids = RowResolver::resolve_all(&self.storage, &group_rows)?;
                    let mut pairs: Vec<(DocId, Vec<ScalarValue>)> = doc_ids
                        .into_iter()
                        .map(|id| (id, vec![value.clone()]))
                        .collect();
                    if pairs.len() > remaining {
                        pairs.select_nth_unstable_by(remaining - 1, |a, b| a.0.cmp(&b.0));
                        pairs.truncate(remaining);
                    }
                    pairs.sort_unstable_by_key(|a| a.0);
                    results.extend(pairs);
                } else {
                    let sorted = sort_rows_with_reader(
                        &self.storage,
                        &self.schema,
                        &reader,
                        order_by,
                        &group_rows,
                        remaining,
                    )?;
                    results.extend(sorted);
                }
                Ok(results.len() < top_k)
            })?;
        }

        if results.len() < top_k && primary.nulls == NullsOrder::Last {
            let null_rows = self.null_primary_rows(primary, filter)?;
            results.extend(self.sort_rows(order_by, &null_rows, top_k - results.len())?);
        }
        results.truncate(top_k);
        Ok(to_results(results))
    }

    /// [`Collection::query_ordered`] plus facets over every document matching `filter`.
    pub fn query_ordered_with_facets(
        &self,
        order_by: &[OrderBySpec],
        filter: Option<&Filter>,
        top_k: usize,
        facets: &[FacetSpec],
    ) -> Result<FacetedSearchResponse<OrderedResult>> {
        validate_order_by(&self.schema, order_by)?;
        let hits = self.query_ordered(order_by, filter, top_k)?;
        let (facets, total_hits) = crate::facet::compute_facets(
            &self.facet_env(),
            facets,
            Population::Query {
                filter,
                rank_bm: None,
            },
        )?;
        Ok(FacetedSearchResponse {
            hits,
            facets,
            total_hits,
        })
    }

    fn null_primary_rows(&self, primary: &OrderBySpec, filter: Option<&Filter>) -> Result<Vec<RowId>> {
        let is_null = Filter::IsNull(primary.attr.clone());
        let combined = match filter {
            Some(f) => Filter::And(vec![is_null, f.clone()]),
            None => is_null,
        };
        let rs = filter_plan::materialize_filter(
            &self.storage,
            &self.schema,
            &combined,
            &self.fts_index,
        )?;
        Ok(rs.iter().collect())
    }

    fn sort_rows(
        &self,
        order_by: &[OrderBySpec],
        rows: &[RowId],
        limit: usize,
    ) -> Result<Vec<(DocId, Vec<ScalarValue>)>> {
        let reader = ColumnReader::new(&self.storage, &self.schema)?;
        sort_rows_with_reader(&self.storage, &self.schema, &reader, order_by, rows, limit)
    }

    /// Stream `(value, sorted RowIds)` groups of documents matching `pop` in the order of `spec`
    /// by walking `cf_attr_postings` (`reverse` for `Desc`). `f` returns `false` to stop.
    pub(crate) fn stream_value_groups(
        &self,
        spec: &OrderBySpec,
        pop: &OrderPopulation,
        f: &mut dyn FnMut(ScalarValue, Vec<RowId>) -> Result<bool>,
    ) -> Result<()> {
        let entry = self.schema.get_by_name(&spec.attr).ok_or_else(|| {
            TreeError::InvalidQuery(format!("unknown attribute '{}'", spec.attr))
        })?;
        let tag = ordered_type_tag(entry.attr_type)?;
        let reverse = spec.direction == SortDirection::Desc;
        let snap = self.storage.segment_snapshot();

        let mut cur_val: Option<Vec<u8>> = None;
        let mut cur_segs: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
        let mut stopped = false;

        let mut flush_group = |val_bytes: &[u8],
                           segs: BTreeMap<SegmentId, RoaringBitmap>,
                           snap: &SegmentTableSnapshot|
         -> Result<bool> {
            let mut rs = RowSet::from_sorted_segs(segs.into_iter().collect()).and_live(snap);
            match pop {
                OrderPopulation::AllLive => {}
                OrderPopulation::RowSet(mask) => {
                    rs = rs.and(mask);
                }
                OrderPopulation::Lazy { seed, residual } => {
                    if let Some(s) = seed {
                        rs = rs.and(s);
                    }
                    if !rs.is_empty() {
                        rs = filter_plan::materialize_lazy(
                            &self.storage,
                            &self.schema,
                            &self.fts_index,
                            snap,
                            residual,
                            Some(&rs),
                        )?;
                    }
                }
            }
            if rs.is_empty() {
                return Ok(true);
            }
            let group_rows: Vec<RowId> = rs.iter().collect();
            let value = decode_scalar_val_bytes(tag, val_bytes)?;
            f(value, group_rows)
        };

        if entry.storage == ColumnStorage::Detached {
            let col_gen = crate::detached::current_detached_gen(&self.storage, entry.id)?;
            let (ow_live_by_seg, mut ow_true_vals) =
                crate::detached::collect_overwritten_true_values(
                    &self.storage,
                    &snap,
                    entry,
                    col_gen,
                )?;

            let pop_allows_seg = |seg: SegmentId| -> bool {
                if !snap.is_visible_segment(seg) {
                    return false;
                }
                match pop {
                    OrderPopulation::RowSet(mask) => mask.segment(seg).is_some(),
                    OrderPopulation::Lazy { seed: Some(s), .. } => s.segment(seg).is_some(),
                    _ => true,
                }
            };

            type FlushGroupFn<'a> =
                &'a mut dyn FnMut(&[u8], BTreeMap<SegmentId, RoaringBitmap>, &SegmentTableSnapshot) -> Result<bool>;
            let flush_ow_before = |target_val: Option<&[u8]>,
                                   ow_true_vals: &mut BTreeMap<Vec<u8>, BTreeMap<SegmentId, RoaringBitmap>>,
                                   flush_group: FlushGroupFn<'_>|
             -> Result<bool> {
                loop {
                    let next_key = if reverse {
                        ow_true_vals.keys().next_back().cloned()
                    } else {
                        ow_true_vals.keys().next().cloned()
                    };
                    let Some(k) = next_key else {
                        return Ok(true);
                    };
                    let comes_before = match target_val {
                        None => true,
                        Some(tv) => {
                            if reverse {
                                k.as_slice() > tv
                            } else {
                                k.as_slice() < tv
                            }
                        }
                    };
                    if !comes_before {
                        return Ok(true);
                    }
                    let segs = ow_true_vals.remove(&k).unwrap();
                    if !flush_group(&k, segs, &snap)? {
                        return Ok(false);
                    }
                }
            };

            self.storage.visit_detached_postings_prefix(
                entry.id,
                col_gen,
                tag,
                reverse,
                |val_bytes, seg, posting| {
                    if !pop_allows_seg(seg) {
                        return Ok(true);
                    }
                    if cur_val.as_deref() != Some(val_bytes) {
                        if let Some(prev_val) = cur_val.take() {
                            if !flush_ow_before(Some(&prev_val), &mut ow_true_vals, &mut flush_group)? {
                                stopped = true;
                                return Ok(false);
                            }
                            let mut prev_segs = std::mem::take(&mut cur_segs);
                            if let Some(extra) = ow_true_vals.remove(&prev_val) {
                                for (eseg, ebm) in extra {
                                    if pop_allows_seg(eseg) {
                                        *prev_segs.entry(eseg).or_default() |= ebm;
                                    }
                                }
                            }
                            if !flush_group(&prev_val, prev_segs, &snap)? {
                                stopped = true;
                                return Ok(false);
                            }
                        }
                        if !flush_ow_before(Some(val_bytes), &mut ow_true_vals, &mut flush_group)? {
                            stopped = true;
                            return Ok(false);
                        }
                        cur_val = Some(val_bytes.to_vec());
                    }
                    let dst = cur_segs.entry(seg).or_default();
                    SegPostingView::parse(posting)?.or_into(dst)?;
                    if let Some(ow_bm) = ow_live_by_seg.get(&seg) {
                        *dst -= ow_bm;
                    }
                    Ok(true)
                },
            )?;

            if !stopped && let Some(last_val) = cur_val.take() {
                if flush_ow_before(Some(&last_val), &mut ow_true_vals, &mut flush_group)? {
                    let mut last_segs = std::mem::take(&mut cur_segs);
                    if let Some(extra) = ow_true_vals.remove(&last_val) {
                        for (eseg, ebm) in extra {
                            if pop_allows_seg(eseg) {
                                *last_segs.entry(eseg).or_default() |= ebm;
                            }
                        }
                    }
                    if !flush_group(&last_val, last_segs, &snap)? {
                        stopped = true;
                    }
                } else {
                    stopped = true;
                }
            }
            if !stopped {
                let _ = flush_ow_before(None, &mut ow_true_vals, &mut flush_group)?;
            }
        } else {
            self.storage
                .visit_attr_postings_prefix(entry.id, tag, reverse, |val_bytes, seg, posting| {
                    if !snap.is_visible_segment(seg) {
                        return Ok(true);
                    }
                    let allowed_seg = match pop {
                        OrderPopulation::RowSet(mask) => mask.segment(seg).is_some(),
                        OrderPopulation::Lazy { seed: Some(s), .. } => s.segment(seg).is_some(),
                        _ => true,
                    };
                    if !allowed_seg {
                        return Ok(true);
                    }

                    if cur_val.as_deref() != Some(val_bytes) {
                        if let Some(prev_val) = cur_val.take() {
                            let keep_going =
                                flush_group(&prev_val, std::mem::take(&mut cur_segs), &snap)?;
                            if !keep_going {
                                stopped = true;
                                return Ok(false);
                            }
                        }
                        cur_val = Some(val_bytes.to_vec());
                    }

                    let dst = cur_segs.entry(seg).or_default();
                    SegPostingView::parse(posting)?.or_into(dst)?;
                    Ok(true)
                })?;

            if !stopped && let Some(last_val) = cur_val.take() {
                let _ = flush_group(&last_val, std::mem::take(&mut cur_segs), &snap)?;
            }
        }
        Ok(())
    }
}

fn sort_rows_with_reader(
    storage: &izzydb_storage::Storage,
    schema: &izzydb_core::schema::CollectionSchema,
    reader: &dyn AttrReader,
    order_by: &[OrderBySpec],
    rows: &[RowId],
    limit: usize,
) -> Result<Vec<(DocId, Vec<ScalarValue>)>> {
    if limit == 0 || rows.is_empty() {
        return Ok(Vec::new());
    }
    let mut sorted_rows = rows.to_vec();
    sorted_rows.sort_unstable();
    sorted_rows.dedup();

    let doc_ids = RowResolver::resolve_all(storage, &sorted_rows)?;
    let attr_ids: Vec<AttrId> = order_by
        .iter()
        .map(|s| schema.get_by_name(&s.attr).map(|e| e.id).unwrap_or_default())
        .collect();

    let mut per_attr_vals: Vec<Vec<Option<ScalarValue>>> =
        vec![Vec::with_capacity(sorted_rows.len()); attr_ids.len()];
    for (col_buf, &attr_id) in per_attr_vals.iter_mut().zip(&attr_ids) {
        reader.read_scalars(attr_id, &sorted_rows, col_buf)?;
    }

    let mut items: Vec<(DocId, Vec<ScalarValue>)> = Vec::with_capacity(sorted_rows.len());
    for (idx, doc_id) in doc_ids.into_iter().enumerate() {
        let keys = per_attr_vals
            .iter()
            .map(|col| col[idx].clone().unwrap_or(ScalarValue::Null))
            .collect();
        items.push((doc_id, keys));
    }

    let cmp = |a: &(DocId, Vec<ScalarValue>), b: &(DocId, Vec<ScalarValue>)| {
        cmp_keys(order_by, &a.1, &b.1).then_with(|| a.0.cmp(&b.0))
    };
    if items.len() > limit {
        items.select_nth_unstable_by(limit - 1, cmp);
        items.truncate(limit);
    }
    items.sort_by(cmp);
    Ok(items)
}

fn to_results(rows: Vec<(DocId, Vec<ScalarValue>)>) -> Vec<OrderedResult> {
    rows.into_iter()
        .map(|(id, sort_values)| OrderedResult { id, sort_values })
        .collect()
}
