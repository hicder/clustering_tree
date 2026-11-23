//! Exact faceting engine on v3 segments: adaptive strategy dispatch over live global counts
//! (`default[0x06]`), forward [`AttrReader`] passes (`"docs"`), and inverted per-segment posting
//! intersections (`cf_attr_postings`).

use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, DocId, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
pub use izzydb_core::facet::*;
use izzydb_core::filter::Filter;
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_core::schema::{AttributeType, CollectionSchema, ScalarValue};
use izzydb_storage::codec::{
    RowFieldRef, SegPostingView, attr_type_value_tag, decode_f64_ordered, decode_i64_ordered,
    decode_scalar_val_bytes, encode_present_attr_key, encode_scalar_val_bytes,
    encode_str_prefix_ordered,
};
use izzydb_storage::Storage;

use crate::attr_reader::{AttrReader, RowStoreReader};
use crate::filter_plan::{self, FilterPlan, FilterPlanTrace, total_live_rows};
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

#[derive(Debug, Clone, PartialEq)]
enum ScanSel {
    Full,
    Prefix(Vec<u8>),
    Span {
        lo: Option<Vec<u8>>,
        hi: Option<Vec<u8>>,
    },
}

impl ScanSel {
    fn start(&self) -> &[u8] {
        match self {
            ScanSel::Full => &[],
            ScanSel::Prefix(p) => p,
            ScanSel::Span { lo, .. } => lo.as_deref().unwrap_or(&[]),
        }
    }

    fn past_end(&self, val: &[u8]) -> bool {
        match self {
            ScanSel::Full => false,
            ScanSel::Prefix(p) => !val.starts_with(p),
            ScanSel::Span { hi, .. } => hi.as_deref().is_some_and(|h| val > h),
        }
    }

    fn merge(parts: Vec<ScanSel>) -> ScanSel {
        let mut iter = parts.into_iter();
        let Some(mut acc) = iter.next() else {
            return ScanSel::Full;
        };
        for next in iter {
            acc = match (acc, next) {
                (ScanSel::Prefix(a), ScanSel::Prefix(b)) if a == b => ScanSel::Prefix(a),
                (
                    ScanSel::Span { lo: l1, hi: h1 },
                    ScanSel::Span { lo: l2, hi: h2 },
                ) => ScanSel::Span {
                    lo: match (l1, l2) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        _ => None,
                    },
                    hi: match (h1, h2) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        _ => None,
                    },
                },
                _ => ScanSel::Full,
            };
        }
        acc
    }
}

#[derive(Debug, Clone)]
struct EncRange {
    lo: Option<(Vec<u8>, bool)>,
    hi: Option<(Vec<u8>, bool)>,
}

impl EncRange {
    #[inline]
    fn contains(&self, v: &[u8]) -> bool {
        if let Some((lo, inc)) = &self.lo {
            match v.cmp(lo.as_slice()) {
                std::cmp::Ordering::Less => return false,
                std::cmp::Ordering::Equal if !*inc => return false,
                _ => {}
            }
        }
        if let Some((hi, inc)) = &self.hi {
            match v.cmp(hi.as_slice()) {
                std::cmp::Ordering::Greater => return false,
                std::cmp::Ordering::Equal if !*inc => return false,
                _ => {}
            }
        }
        true
    }
}

struct ArrayRangeAcc {
    ranges: Vec<EncRange>,
    counts: Vec<u64>,
    /// Per-segment union of matching local ids (`seg_idx -> bucket_idx -> bitmap`).
    scratch: BTreeMap<SegmentId, Vec<RoaringBitmap>>,
}

struct AttrAcc {
    attr_id: AttrId,
    type_tag: u8,
    need_values: bool,
    need_present: bool,
    sel: ScanSel,
    values: HashMap<Vec<u8>, u64>,
    present: u64,
    array_ranges: Vec<ArrayRangeAcc>,
}

struct FacetPlan {
    acc_idx: usize,
    array_range: Option<usize>,
    attr_type: AttributeType,
}

struct AttrAccs {
    accs: Vec<AttrAcc>,
    plans: Vec<FacetPlan>,
    spec_idxs: Vec<usize>,
}

fn encode_bound(v: &ScalarValue) -> Result<Vec<u8>> {
    let mut b = Vec::with_capacity(16);
    encode_scalar_val_bytes(v, &mut b)?;
    Ok(b)
}

impl AttrAccs {
    fn build(env: &FacetEnv<'_>, specs: &[FacetSpec], idxs: &[usize]) -> Result<Self> {
        let mut accs: Vec<AttrAcc> = Vec::new();
        let mut sel_parts: Vec<Vec<ScanSel>> = Vec::new();
        let mut by_attr: HashMap<AttrId, usize> = HashMap::new();
        let mut plans = Vec::with_capacity(idxs.len());

        for &i in idxs {
            let spec = &specs[i];
            let entry = spec.entry(env.schema)?;
            let acc_idx = *by_attr.entry(entry.id).or_insert_with(|| {
                accs.push(AttrAcc {
                    attr_id: entry.id,
                    type_tag: attr_type_value_tag(entry.attr_type),
                    need_values: false,
                    need_present: false,
                    sel: ScanSel::Full,
                    values: HashMap::new(),
                    present: 0,
                    array_ranges: Vec::new(),
                });
                sel_parts.push(Vec::new());
                accs.len() - 1
            });
            let is_array = entry.attr_type.is_array();
            let acc = &mut accs[acc_idx];
            acc.need_present |= spec.include_missing;
            let mut array_range = None;
            match &spec.kind {
                FacetKind::Ranges { ranges } => {
                    let enc = encode_ranges(ranges)?;
                    let sel = ScanSel::Span {
                        lo: span_lo(&enc),
                        hi: span_hi(&enc),
                    };
                    sel_parts[acc_idx].push(sel);
                    if is_array {
                        let n = enc.len();
                        acc.array_ranges.push(ArrayRangeAcc {
                            ranges: enc,
                            counts: vec![0; n],
                            scratch: BTreeMap::new(),
                        });
                        array_range = Some(acc.array_ranges.len() - 1);
                    } else {
                        acc.need_values = true;
                    }
                }
                FacetKind::Terms {
                    prefix: Some(p), ..
                } => {
                    acc.need_values = true;
                    sel_parts[acc_idx]
                        .push(ScanSel::Prefix(encode_str_prefix_ordered(p.as_bytes())));
                }
                FacetKind::Terms { .. } | FacetKind::Stats | FacetKind::Histogram { .. } => {
                    acc.need_values = true;
                    sel_parts[acc_idx].push(ScanSel::Full);
                }
            }
            plans.push(FacetPlan {
                acc_idx,
                array_range,
                attr_type: entry.attr_type,
            });
        }
        for (acc, parts) in accs.iter_mut().zip(sel_parts) {
            acc.sel = ScanSel::merge(parts);
        }
        Ok(Self {
            accs,
            plans,
            spec_idxs: idxs.to_vec(),
        })
    }

    fn has_array_ranges(&self) -> bool {
        self.plans.iter().any(|p| p.array_range.is_some())
    }

    // ---- Strategy 1: global counts (`default[0x06]`) -------------------------------------

    fn load_global_counts(&mut self, env: &FacetEnv<'_>) -> Result<()> {
        for acc in &mut self.accs {
            if acc.need_values {
                let sel = acc.sel.clone();
                let values = &mut acc.values;
                env.storage
                    .visit_attr_counts(acc.attr_id, acc.type_tag, sel.start(), |val, c| {
                        if sel.past_end(val) {
                            return Ok(false);
                        }
                        if c > 0 {
                            values.insert(val.to_vec(), c);
                        }
                        Ok(true)
                    })?;
            }
            if acc.need_present {
                acc.present = env
                    .storage
                    .get_attr_count(&encode_present_attr_key(acc.attr_id))?;
            }
        }
        Ok(())
    }

    // ---- Strategy 3: forward `AttrReader` pass -------------------------------------------

    fn accumulate_rows(&mut self, env: &FacetEnv<'_>, sorted_rows: &[RowId]) -> Result<()> {
        if self.accs.is_empty() || sorted_rows.is_empty() {
            return Ok(());
        }
        let reader = RowStoreReader::new(env.storage);
        let mut key_buf: Vec<u8> = Vec::with_capacity(32);
        let mut elems: Vec<Vec<u8>> = Vec::new();
        reader.read_rows(sorted_rows, &mut |_, view| {
            for acc in &mut self.accs {
                let Some(field) = view.get_raw_field(acc.attr_id)? else {
                    continue;
                };
                if field.is_null() {
                    continue;
                }
                acc.present += 1;
                let is_array = !matches!(field, RowFieldRef::Scalar(_));
                if !is_array {
                    if acc.need_values {
                        field.for_each_scalar(|v| {
                            key_buf.clear();
                            v.encode_into(&mut key_buf);
                            bump(&mut acc.values, &key_buf, 1);
                        })?;
                    }
                    continue;
                }
                elems.clear();
                field.for_each_scalar(|v| {
                    let mut k = Vec::with_capacity(16);
                    v.encode_into(&mut k);
                    elems.push(k);
                })?;
                for ar in &mut acc.array_ranges {
                    for (b, r) in ar.ranges.iter().enumerate() {
                        if elems.iter().any(|e| r.contains(e)) {
                            ar.counts[b] += 1;
                        }
                    }
                }
                if acc.need_values {
                    elems.sort_unstable();
                    elems.dedup();
                    for e in &elems {
                        bump(&mut acc.values, e, 1);
                    }
                }
            }
            Ok(())
        })
    }

    // ---- Strategy 4: per-segment posting intersection over `RowSet` ----------------------

    fn accumulate_rowset_postings(&mut self, env: &FacetEnv<'_>, pop: &RowSet) -> Result<()> {
        if self.accs.is_empty() || pop.is_empty() {
            return Ok(());
        }
        for acc in &mut self.accs {
            if acc.need_present {
                let key = encode_present_attr_key(acc.attr_id);
                env.storage
                    .visit_attr_postings_for_value(&key, |seg, bytes| {
                        let Some(pop_bm) = pop.segment(seg) else {
                            return Ok(());
                        };
                        acc.present += SegPostingView::parse(bytes)?.and_count(pop_bm)?;
                        Ok(())
                    })?;
            }
            if !acc.need_values && acc.array_ranges.is_empty() {
                continue;
            }
            for ar in &mut acc.array_ranges {
                ar.scratch.clear();
            }
            let sel = acc.sel.clone();
            let need_values = acc.need_values;
            let AttrAcc {
                values,
                array_ranges,
                attr_id,
                type_tag,
                ..
            } = acc;
            env.storage.visit_attr_postings_range(
                *attr_id,
                *type_tag,
                sel.start(),
                |val, seg, bytes| {
                    if sel.past_end(val) {
                        return Ok(false);
                    }
                    let Some(pop_bm) = pop.segment(seg) else {
                        return Ok(true);
                    };
                    let view = SegPostingView::parse(bytes)?;
                    if need_values {
                        let c = view.and_count(pop_bm)?;
                        if c > 0 {
                            bump(values, val, c);
                        }
                    }
                    for ar in array_ranges.iter_mut() {
                        let n_buckets = ar.ranges.len();
                        for (b, r) in ar.ranges.iter().enumerate() {
                            if r.contains(val) {
                                let sb = ar
                                    .scratch
                                    .entry(seg)
                                    .or_insert_with(|| vec![RoaringBitmap::new(); n_buckets]);
                                view.or_into_masked(pop_bm, &mut sb[b])?;
                            }
                        }
                    }
                    Ok(true)
                },
            )?;
            for ar in &mut acc.array_ranges {
                for seg_buckets in ar.scratch.values() {
                    for (b, bm) in seg_buckets.iter().enumerate() {
                        ar.counts[b] += bm.len();
                    }
                }
            }
        }
        Ok(())
    }

    // ---- Finalization --------------------------------------------------------------------

    fn finalize_into(
        &self,
        specs: &[FacetSpec],
        total: u64,
        out: &mut [Option<FacetResult>],
    ) -> Result<()> {
        let sorted: Vec<Vec<(&[u8], u64)>> = self
            .accs
            .iter()
            .map(|acc| {
                let mut v: Vec<(&[u8], u64)> = acc
                    .values
                    .iter()
                    .filter(|&(_, &c)| c > 0)
                    .map(|(k, &c)| (k.as_slice(), c))
                    .collect();
                v.sort_unstable_by(|a, b| a.0.cmp(b.0));
                v
            })
            .collect();
        for (plan, &si) in self.plans.iter().zip(&self.spec_idxs) {
            let spec = &specs[si];
            let acc = &self.accs[plan.acc_idx];
            let missing = if spec.include_missing {
                total.saturating_sub(acc.present)
            } else {
                0
            };
            out[si] = Some(finalize_facet(
                spec,
                plan,
                acc,
                &sorted[plan.acc_idx],
                missing,
            )?);
        }
        Ok(())
    }
}

#[inline]
fn bump(map: &mut HashMap<Vec<u8>, u64>, key: &[u8], by: u64) {
    if let Some(c) = map.get_mut(key) {
        *c += by;
    } else {
        map.insert(key.to_vec(), by);
    }
}

fn encode_ranges(ranges: &[FacetRange]) -> Result<Vec<EncRange>> {
    ranges
        .iter()
        .map(|r| {
            Ok(EncRange {
                lo: r
                    .from
                    .as_ref()
                    .map(|v| encode_bound(v).map(|b| (b, r.from_inclusive)))
                    .transpose()?,
                hi: r
                    .to
                    .as_ref()
                    .map(|v| encode_bound(v).map(|b| (b, r.to_inclusive)))
                    .transpose()?,
            })
        })
        .collect()
}

fn span_lo(ranges: &[EncRange]) -> Option<Vec<u8>> {
    let mut lo: Option<Vec<u8>> = None;
    for r in ranges {
        let (b, _) = r.lo.as_ref()?;
        lo = Some(match lo {
            Some(cur) => cur.min(b.clone()),
            None => b.clone(),
        });
    }
    lo
}

fn span_hi(ranges: &[EncRange]) -> Option<Vec<u8>> {
    let mut hi: Option<Vec<u8>> = None;
    for r in ranges {
        let (b, _) = r.hi.as_ref()?;
        hi = Some(match hi {
            Some(cur) => cur.max(b.clone()),
            None => b.clone(),
        });
    }
    hi
}

fn decode_numeric(type_tag: u8, bytes: &[u8]) -> f64 {
    let arr: [u8; 8] = bytes.try_into().unwrap_or([0; 8]);
    match type_tag {
        izzydb_storage::codec::TAG_INT => decode_i64_ordered(arr) as f64,
        izzydb_storage::codec::TAG_UINT => u64::from_be_bytes(arr) as f64,
        _ => decode_f64_ordered(arr),
    }
}

fn finalize_facet(
    spec: &FacetSpec,
    plan: &FacetPlan,
    acc: &AttrAcc,
    sorted: &[(&[u8], u64)],
    missing: u64,
) -> Result<FacetResult> {
    let tag = acc.type_tag;
    match &spec.kind {
        FacetKind::Terms {
            limit,
            sort,
            min_count,
            prefix,
        } => {
            let pre = prefix
                .as_ref()
                .map(|p| encode_str_prefix_ordered(p.as_bytes()));
            let min = (*min_count).max(1);
            let mut kept: Vec<(&[u8], u64)> = sorted
                .iter()
                .copied()
                .filter(|(k, c)| *c >= min && pre.as_ref().is_none_or(|p| k.starts_with(p)))
                .collect();
            let total_values = kept.len() as u64;
            match sort {
                FacetSort::ValueAsc => {}
                FacetSort::ValueDesc => kept.reverse(),
                FacetSort::CountDesc => kept.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0))),
                FacetSort::CountAsc => kept.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(b.0))),
            }
            kept.truncate(*limit);
            let buckets = kept
                .into_iter()
                .map(|(k, count)| {
                    Ok(FacetBucket {
                        value: decode_scalar_val_bytes(tag, k)?,
                        count,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(FacetResult::Terms(TermFacetResult {
                buckets,
                total_values,
                missing_count: missing,
            }))
        }
        FacetKind::Ranges { ranges } => {
            let counts: Vec<u64> = match plan.array_range {
                Some(ai) => acc.array_ranges[ai].counts.clone(),
                None => {
                    let enc = encode_ranges(ranges)?;
                    enc.iter()
                        .map(|r| sorted.iter().filter(|(k, _)| r.contains(k)).map(|e| e.1).sum())
                        .collect()
                }
            };
            let buckets = ranges
                .iter()
                .zip(counts)
                .map(|(r, count)| RangeBucket {
                    label: r.label.clone(),
                    from: r.from.clone(),
                    to: r.to.clone(),
                    count,
                })
                .collect();
            Ok(FacetResult::Ranges(RangeFacetResult {
                buckets,
                missing_count: missing,
            }))
        }
        FacetKind::Stats => {
            let count: u64 = sorted.iter().map(|e| e.1).sum();
            let sum: f64 = sorted
                .iter()
                .map(|(k, c)| decode_numeric(tag, k) * *c as f64)
                .sum();
            let min = sorted
                .first()
                .map(|(k, _)| decode_scalar_val_bytes(tag, k))
                .transpose()?;
            let max = sorted
                .last()
                .map(|(k, _)| decode_scalar_val_bytes(tag, k))
                .transpose()?;
            debug_assert!(matches!(
                plan.attr_type,
                AttributeType::Int | AttributeType::Uint | AttributeType::Float
            ));
            Ok(FacetResult::Stats(StatsFacetResult {
                count,
                missing_count: missing,
                min,
                max,
                sum,
                avg: (count > 0).then(|| sum / count as f64),
            }))
        }
        FacetKind::Histogram {
            interval,
            offset,
            min_count,
        } => {
            let mut buckets: BTreeMap<i64, u64> = BTreeMap::new();
            for (k, c) in sorted {
                let v = decode_numeric(tag, k);
                let idx = ((v - offset) / interval).floor() as i64;
                *buckets.entry(idx).or_insert(0) += *c;
            }
            let min = (*min_count).max(1);
            let buckets = buckets
                .into_iter()
                .filter(|&(_, c)| c >= min)
                .map(|(idx, count)| {
                    let start = offset + idx as f64 * interval;
                    HistogramBucket {
                        start,
                        end: start + interval,
                        count,
                    }
                })
                .collect();
            Ok(FacetResult::Histogram(HistogramFacetResult {
                buckets,
                missing_count: missing,
            }))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Strategy dispatch
// ---------------------------------------------------------------------------------------------

fn complement_inner(filter: &Filter) -> Option<Filter> {
    match filter {
        Filter::Not(inner) => Some((**inner).clone()),
        Filter::NotEq(a, v) => Some(Filter::Eq(a.clone(), v.clone())),
        Filter::NotIn(a, vs) => Some(Filter::In(a.clone(), vs.clone())),
        Filter::And(children) if children.len() == 1 => complement_inner(&children[0]),
        _ => None,
    }
}

fn filter_has_complement_or_null(filter: &Filter) -> bool {
    match filter {
        Filter::Not(_) | Filter::NotEq(_, _) | Filter::NotIn(_, _) | Filter::IsNull(_) => true,
        Filter::And(ch) | Filter::Or(ch) => ch.iter().any(filter_has_complement_or_null),
        _ => false,
    }
}

fn eval_group(
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
