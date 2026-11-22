//! Exact faceting engine: adaptive strategy dispatch over global counts (`default[0x06]`),
//! forward `cf_docs` passes, and inverted per-leaf mask intersections (`cf_attr_postings`).
//!
//! Every facet kind (`Terms`, scalar `Ranges`, `Stats`, `Histogram`) is finalized from a sorted
//! `(encoded_value, doc_count)` list, so all strategies only need to produce per-value document
//! counts. Array-attribute `Ranges` are the exception: a document with several elements inside
//! one bucket must count once, so they accumulate per-bucket document counts directly.

use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, DocId, NodeId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
pub use izzydb_core::facet::*;
use izzydb_core::filter::Filter;
use izzydb_core::schema::{AttributeType, CollectionSchema, ScalarValue};
use izzydb_storage::codec::{
    DocView, LeafPostingView, TAG_PRESENT, attr_type_value_tag, decode_f64_ordered,
    decode_i64_ordered, decode_scalar_val_bytes, encode_present_cluster_key,
    encode_scalar_val_bytes, encode_str_prefix_ordered,
};
use izzydb_storage::node::TreeMeta;
use izzydb_storage::Storage;

use crate::attr_index::{AttributeIndex, ClusterCandidates, LeafMask, PreparedFilter};
use crate::fts_index::FtsIndex;
use crate::rabitq::RabitqQuantizer;
use crate::search::{collect_matching_doc_ids_in_leaves, parse_leaf_views};

thread_local! {
    static STRATEGY_TRACE: std::cell::RefCell<Vec<&'static str>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn trace_strategy(name: &'static str) {
    STRATEGY_TRACE.with(|t| t.borrow_mut().push(name));
}

/// Drain the strategies chosen by facet evaluation on this thread (one entry per evaluated
/// `FacetGroup`): `"global"`, `"complement"`, `"docs"`, `"leaf_masks_no_views"`,
/// `"leaf_masks_views"`. Intended for tests and diagnostics.
#[doc(hidden)]
pub fn take_strategy_trace() -> Vec<&'static str> {
    STRATEGY_TRACE.with(|t| std::mem::take(&mut *t.borrow_mut()))
}

/// Maximum number of documents for the forward `cf_docs` pass to be preferred.
pub const FORWARD_PASS_MAX_DOCS: usize = 2048;
/// Maximum number of candidate leaves resolved eagerly to decide between strategies 3 and 4.
const SMALL_CANDIDATE_LEAVES: usize = 16;
const DOC_CHUNK: usize = 512;
const LEAF_CHUNK: usize = 256;

/// Everything the engine needs from a [`Collection`](crate::Collection).
pub(crate) struct FacetEnv<'a> {
    pub storage: &'a Storage,
    pub schema: &'a CollectionSchema,
    pub attr_index: &'a AttributeIndex,
    pub fts_index: &'a FtsIndex,
    pub config: &'a TreeConfig,
    pub quantizer: &'a RabitqQuantizer,
    pub meta: &'a TreeMeta,
    pub leaf_parent: &'a HashMap<NodeId, NodeId>,
}

impl FacetEnv<'_> {
    fn num_vectors(&self) -> u64 {
        self.meta.num_vectors
    }

    fn all_leaf_ids(&self) -> Vec<NodeId> {
        if self.meta.root_level == 0 {
            vec![self.meta.root_id]
        } else {
            let mut ids: Vec<NodeId> = self.leaf_parent.keys().copied().collect();
            ids.sort_unstable();
            ids
        }
    }
}

/// The set of documents facets are computed over.
pub(crate) enum Population<'a> {
    /// Documents matching `filter` (each facet may rewrite it via its `FacetFilterMode`) and,
    /// when `rank_bm` is set, having BM25 score `> 0` (a set of `FtsDocId`s).
    Query {
        filter: Option<&'a Filter>,
        rank_bm: Option<&'a RoaringBitmap>,
        /// Pre-built plan for `filter` (shared with the primary search) when `rank_bm` is `None`.
        prepared: Option<&'a PreparedFilter>,
    },
    /// An explicit document set (filter modes are ignored).
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
        Population::Docs(docs) => {
            let mut docs = docs.to_vec();
            docs.sort_unstable();
            docs.dedup();
            let idx: Vec<usize> = (0..specs.len()).collect();
            let mut accs = AttrAccs::build(env, specs, &idx)?;
            accs.accumulate_docs(env, &docs)?;
            accs.finalize_into(specs, docs.len() as u64, &mut out)?;
            docs.len() as u64
        }
        Population::Query {
            filter,
            rank_bm,
            prepared,
        } => {
            // Groups share one population evaluation per distinct effective filter. The primary
            // group (effective filter == query filter) always exists so `total` is known.
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
                let shared = (gi == 0 && rank_bm.is_none()).then_some(prepared).flatten();
                let g_total = eval_group(env, specs, idxs, gfilter.as_ref(), rank_bm, shared, &mut out)?;
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

/// Which part of an attribute's value space a scan must cover.
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

    /// Whether `val` (visited in ascending order from `start`) is beyond the selection.
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

/// Per-bucket document counts for `Ranges` over array attributes (documents counted once).
struct ArrayRangeAcc {
    ranges: Vec<EncRange>,
    counts: Vec<u64>,
    /// Per-leaf scratch masks (strategy 4).
    scratch: Vec<LeafMask>,
}

/// Accumulated per-attribute state shared by every facet on that attribute.
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

/// Per-facet binding into [`AttrAccs`].
struct FacetPlan {
    acc_idx: usize,
    /// Index into `AttrAcc::array_ranges` for array-attribute `Ranges`.
    array_range: Option<usize>,
    attr_type: AttributeType,
}

struct AttrAccs {
    accs: Vec<AttrAcc>,
    /// Plans aligned with `spec_idxs`.
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
                            scratch: Vec::new(),
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

    // ---- Strategy 1: global counts -------------------------------------------------------

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
                        values.insert(val.to_vec(), c);
                        Ok(true)
                    })?;
            }
            if acc.need_present {
                acc.present = env
                    .storage
                    .get_attr_count(&encode_present_cluster_key(acc.attr_id))?;
            }
        }
        Ok(())
    }

    // ---- Strategy 3: forward DocView pass ------------------------------------------------

    fn accumulate_docs(&mut self, env: &FacetEnv<'_>, sorted_docs: &[DocId]) -> Result<()> {
        if self.accs.is_empty() {
            return Ok(());
        }
        let mut key_buf: Vec<u8> = Vec::with_capacity(32);
        let mut elems: Vec<Vec<u8>> = Vec::new();
        for chunk in sorted_docs.chunks(DOC_CHUNK) {
            let slices = env.storage.multi_get_doc_bytes_pinned(chunk)?;
            for (&doc_id, slice) in chunk.iter().zip(slices.iter()) {
                let slice = slice.as_ref().ok_or_else(|| {
                    TreeError::InvariantViolation(format!("facet doc {doc_id} missing in cf_docs"))
                })?;
                let view = DocView::parse(slice)?;
                for acc in &mut self.accs {
                    let Some(field) = view.get_raw_field(acc.attr_id)? else {
                        continue;
                    };
                    if field.is_null() {
                        continue;
                    }
                    acc.present += 1;
                    let is_array = !matches!(field, izzydb_storage::codec::DocFieldRef::Scalar(_));
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
            }
        }
        Ok(())
    }

    // ---- Strategy 4: inverted per-leaf mask intersection ---------------------------------

    fn accumulate_leaf(
        &mut self,
        env: &FacetEnv<'_>,
        leaf_id: NodeId,
        mask: &LeafMask,
    ) -> Result<()> {
        let words = mask.as_slice().len();
        for acc in &mut self.accs {
            if acc.need_present {
                env.storage.visit_leaf_attr_postings(
                    acc.attr_id,
                    TAG_PRESENT,
                    leaf_id,
                    &[],
                    |_, posting| {
                        acc.present +=
                            LeafPostingView::parse(posting)?.count_and_mask(mask.as_slice())?;
                        Ok(false)
                    },
                )?;
            }
            if !acc.need_values && acc.array_ranges.is_empty() {
                continue;
            }
            for ar in &mut acc.array_ranges {
                ar.scratch.clear();
                ar.scratch
                    .resize_with(ar.ranges.len(), || LeafMask::empty(words));
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
            env.storage
                .visit_leaf_attr_postings(*attr_id, *type_tag, leaf_id, sel.start(), |val, posting| {
                    if sel.past_end(val) {
                        return Ok(false);
                    }
                    let view = LeafPostingView::parse(posting)?;
                    if need_values {
                        let c = view.count_and_mask(mask.as_slice())?;
                        if c > 0 {
                            bump(values, val, c);
                        }
                    }
                    for ar in array_ranges.iter_mut() {
                        for (b, r) in ar.ranges.iter().enumerate() {
                            if r.contains(val) {
                                view.or_into_words(ar.scratch[b].as_mut_slice())?;
                            }
                        }
                    }
                    Ok(true)
                })?;
            for ar in &mut acc.array_ranges {
                for (b, scratch) in ar.scratch.iter().enumerate() {
                    ar.counts[b] += scratch.and_count(mask);
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
        // Sorted (encoded_value, count) lists, built once per attribute.
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

fn encode_ranges(ranges: &[izzydb_core::facet::FacetRange]) -> Result<Vec<EncRange>> {
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
                // `kept` is already ascending by encoded value (== value order).
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

/// If `filter` is a pure complement `Not(inner)` / `NotEq` / `NotIn`, return the positive `inner`.
fn complement_inner(filter: &Filter) -> Option<Filter> {
    match filter {
        Filter::Not(inner) => Some((**inner).clone()),
        Filter::NotEq(a, v) => Some(Filter::Eq(a.clone(), v.clone())),
        Filter::NotIn(a, vs) => Some(Filter::In(a.clone(), vs.clone())),
        Filter::And(children) if children.len() == 1 => complement_inner(&children[0]),
        _ => None,
    }
}

/// Evaluate one `FacetGroup` (all `idxs` share `filter`) writing results into `out`.
/// Returns the group's population size.
fn eval_group(
    env: &FacetEnv<'_>,
    specs: &[FacetSpec],
    idxs: &[usize],
    filter: Option<&Filter>,
    rank_bm: Option<&RoaringBitmap>,
    shared_prepared: Option<&PreparedFilter>,
    out: &mut [Option<FacetResult>],
) -> Result<u64> {
    let mut accs = AttrAccs::build(env, specs, idxs)?;
    let finish = |accs: &mut AttrAccs, total: u64, out: &mut [Option<FacetResult>]| -> Result<u64> {
        accs.finalize_into(specs, total, out)?;
        Ok(total)
    };

    if env.num_vectors() == 0 {
        return finish(&mut accs, 0, out);
    }
    if let Some(bm) = rank_bm
        && bm.is_empty()
    {
        return finish(&mut accs, 0, out);
    }

    // Strategy 1: unfiltered global counts.
    if filter.is_none() && rank_bm.is_none() && !accs.has_array_ranges() {
        trace_strategy("global");
        accs.load_global_counts(env)?;
        return finish(&mut accs, env.num_vectors(), out);
    }

    // Shortcut: a small BM25 match set with no attribute filter resolves straight to doc ids.
    if filter.is_none()
        && let Some(bm) = rank_bm
        && bm.len() as usize <= FORWARD_PASS_MAX_DOCS
    {
        let fts_ids: Vec<_> = bm.iter().collect();
        let mut docs: Vec<DocId> = env
            .storage
            .multi_get_doc_ids_from_fts(&fts_ids)?
            .into_iter()
            .flatten()
            .collect();
        docs.sort_unstable();
        docs.dedup();
        trace_strategy("docs");
        accs.accumulate_docs(env, &docs)?;
        return finish(&mut accs, docs.len() as u64, out);
    }

    // Strategy 2: complement of a selective positive filter against global counts.
    if rank_bm.is_none()
        && !accs.has_array_ranges()
        && let Some(inner) = filter.and_then(complement_inner)
        && let Some(inner_docs) = resolve_small_docs(env, &inner)?
    {
        trace_strategy("complement");
        let mut inner_accs = AttrAccs::build(env, specs, idxs)?;
        inner_accs.accumulate_docs(env, &inner_docs)?;
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
            // Global presence is only loaded when some facet needs `missing_count`.
            if acc.need_present {
                acc.present = acc.present.checked_sub(sub.present).ok_or_else(|| {
                    TreeError::InvariantViolation(
                        "complement facet: inner presence exceeds global presence".into(),
                    )
                })?;
            }
        }
        let total = env.num_vectors().saturating_sub(inner_docs.len() as u64);
        return finish(&mut accs, total, out);
    }

    // General path: resolve the population via a prepared filter.
    let owned;
    let prepared: &PreparedFilter = match shared_prepared {
        Some(p) => p,
        None => {
            owned = env.attr_index.prepare_filter_with_fts_and_rank_bm(
                env.storage,
                env.schema,
                filter,
                Some(env.fts_index),
                rank_bm,
            )?;
            &owned
        }
    };
    if prepared.cluster_candidates.is_empty() {
        return finish(&mut accs, 0, out);
    }
    let leaf_ids: Vec<NodeId> = match &prepared.cluster_candidates {
        ClusterCandidates::Subset(bm) => bm.iter().collect(),
        ClusterCandidates::All => env.all_leaf_ids(),
    };

    // Strategy 3 when the population is small and cheap to resolve.
    if matches!(prepared.cluster_candidates, ClusterCandidates::Subset(_))
        && leaf_ids.len() <= SMALL_CANDIDATE_LEAVES
    {
        let mut docs = Vec::new();
        collect_matching_doc_ids_in_leaves(
            env.storage,
            env.attr_index,
            prepared,
            &leaf_ids,
            env.config,
            env.quantizer,
            &mut docs,
        )?;
        docs.sort_unstable();
        docs.dedup();
        if docs.len() <= FORWARD_PASS_MAX_DOCS {
            trace_strategy("docs");
            accs.accumulate_docs(env, &docs)?;
            return finish(&mut accs, docs.len() as u64, out);
        }
    }

    // Strategy 4: per-leaf mask intersection over inverted postings.
    let total = run_leaf_masks(env, prepared, &leaf_ids, &mut accs)?;
    finish(&mut accs, total, out)
}

/// Resolve `inner`'s matching docs when it is cheap (few candidate leaves, few docs).
fn resolve_small_docs(env: &FacetEnv<'_>, inner: &Filter) -> Result<Option<Vec<DocId>>> {
    let prepared = env.attr_index.prepare_filter_with_fts(
        env.storage,
        env.schema,
        inner,
        Some(env.fts_index),
    )?;
    let leaf_ids: Vec<NodeId> = match &prepared.cluster_candidates {
        ClusterCandidates::Subset(bm) if bm.is_empty() => return Ok(Some(Vec::new())),
        ClusterCandidates::Subset(bm) if bm.len() as usize <= SMALL_CANDIDATE_LEAVES => {
            bm.iter().collect()
        }
        _ => return Ok(None),
    };
    let mut docs = Vec::new();
    collect_matching_doc_ids_in_leaves(
        env.storage,
        env.attr_index,
        &prepared,
        &leaf_ids,
        env.config,
        env.quantizer,
        &mut docs,
    )?;
    docs.sort_unstable();
    docs.dedup();
    Ok((docs.len() <= FORWARD_PASS_MAX_DOCS).then_some(docs))
}

/// Strategy 4: evaluate per-leaf masks once and fold every facet attribute through them.
fn run_leaf_masks(
    env: &FacetEnv<'_>,
    prepared: &PreparedFilter,
    leaf_ids: &[NodeId],
    accs: &mut AttrAccs,
) -> Result<u64> {
    let mut total = 0u64;
    let no_views = !prepared.needs_live_mask();
    trace_strategy(if no_views {
        "leaf_masks_no_views"
    } else {
        "leaf_masks_views"
    });
    let word_count = env.config.max_leaf_size.div_ceil(64);
    for chunk in leaf_ids.chunks(LEAF_CHUNK) {
        let masks: Vec<LeafMask> = if no_views {
            env.attr_index
                .eval_leaves_masks_no_views(env.storage, prepared, chunk, word_count)?
        } else {
            let slices = env.storage.multi_get_leaf_nodes_pinned(chunk)?;
            let views = parse_leaf_views(chunk, &slices, env.config, env.quantizer)?;
            let pairs: Vec<_> = chunk
                .iter()
                .copied()
                .zip(views.iter().map(|v| v.as_ref()))
                .collect();
            let mut masks = env
                .attr_index
                .eval_leaves_masks(env.storage, prepared, &pairs)?;
            for (view, mask) in views.iter().zip(masks.iter_mut()) {
                if let Some(view) = view
                    && view.has_tombstones()
                {
                    mask.and_live_mask_bytes(view.live_mask_bytes);
                }
            }
            masks
        };
        for (&leaf_id, mask) in chunk.iter().zip(masks.iter()) {
            if mask.is_empty() {
                continue;
            }
            total += mask.count_ones();
            accs.accumulate_leaf(env, leaf_id, mask)?;
        }
    }
    Ok(total)
}
