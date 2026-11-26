//! Facet accumulator types and construction.

use std::collections::{BTreeMap, HashMap};
use roaring::RoaringBitmap;
use izzydb_core::config::{AttrId, SegmentId};
use izzydb_core::error::Result;
use izzydb_core::schema::{AttributeType, ScalarValue};
use izzydb_storage::codec::{
    attr_type_value_tag, encode_scalar_val_bytes,
    encode_str_prefix_ordered,
};

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum ScanSel {
    Full,
    Prefix(Vec<u8>),
    Span {
        lo: Option<Vec<u8>>,
        hi: Option<Vec<u8>>,
    },
}

impl ScanSel {
    pub(super) fn start(&self) -> &[u8] {
        match self {
            ScanSel::Full => &[],
            ScanSel::Prefix(p) => p,
            ScanSel::Span { lo, .. } => lo.as_deref().unwrap_or(&[]),
        }
    }

    pub(super) fn past_end(&self, val: &[u8]) -> bool {
        match self {
            ScanSel::Full => false,
            ScanSel::Prefix(p) => !val.starts_with(p),
            ScanSel::Span { hi, .. } => hi.as_deref().is_some_and(|h| val > h),
        }
    }

    pub(super) fn merge(parts: Vec<ScanSel>) -> ScanSel {
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

/// One bucket of a range facet, with its bounds encoded as ordered value bytes.
///
/// Stored values are compared as raw bytes. The encoding preserves order, so
/// [`EncRange::contains`] can test a value without decoding it.
///
/// # Example
///
/// The bucket `10 <= price < 20` on an int attribute becomes:
///
/// ```text
/// EncRange {
///     lo: Some((encode_bound(10), true)),  // inclusive
///     hi: Some((encode_bound(20), false)), // exclusive
/// }
/// ```
///
/// This contains the encodings of 10 and 19, but not those of 9 or 20.
#[derive(Debug, Clone)]
pub(super) struct EncRange {
    /// Lower bound as `(encoded_value, is_inclusive)`. `None` means no lower bound.
    pub(super) lo: Option<(Vec<u8>, bool)>,
    /// Upper bound as `(encoded_value, is_inclusive)`. `None` means no upper bound.
    pub(super) hi: Option<(Vec<u8>, bool)>,
}

impl EncRange {
    #[inline]
    pub(super) fn contains(&self, v: &[u8]) -> bool {
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

/// Running bucket counts for one range facet on an array attribute.
///
/// A range facet on a scalar attribute can be computed at the end by summing the
/// per-value counts that fall in each bucket. That does not work for arrays,
/// because one document can have several elements in the same bucket and would
/// be counted once per element. This type counts each document at most once per
/// bucket. A document can still count in more than one bucket.
///
/// # Example
///
/// With buckets `[0, 10)` and `[10, 20)` over these documents:
///
/// ```text
/// doc A: sizes = [1, 2, 15]
/// doc B: sizes = [3]
/// ```
///
/// `counts` ends as `[2, 1]`. A matches the first bucket through both 1 and 2,
/// but is counted there once. Summing per-value counts would give 3.
pub(super) struct ArrayRangeAcc {
    /// The facet's buckets, in the order the request listed them.
    pub(super) ranges: Vec<EncRange>,
    /// Matching document count per bucket. `counts[i]` belongs to `ranges[i]`.
    pub(super) counts: Vec<u64>,
    /// Temporary state for the posting-list strategy
    /// (`segment id -> bucket index -> bitmap of matching local ids`).
    ///
    /// Postings are visited one element value at a time. OR-ing them into a
    /// bitmap removes documents that match a bucket through several values.
    /// After the scan, each bitmap's size is added to `counts`. The row-by-row
    /// pass does not use this field; it increments `counts` directly.
    pub(super) scratch: BTreeMap<SegmentId, Vec<RoaringBitmap>>,
}

/// Accumulator for one attribute, shared by every facet on that attribute.
///
/// It records both what has to be collected and what has been collected so far,
/// so each attribute is read only once however many facets use it. A counting
/// strategy (global counts, forward row pass or posting intersection) fills it,
/// and finalization turns it into facet results.
///
/// # Example
///
/// A terms facet and a stats facet on `price`, where the stats facet also sets
/// `include_missing`, share one `AttrAcc` with `need_values = true`,
/// `need_present = true` and `sel = ScanSel::Full`. If 3 documents are counted,
/// with prices 5, 5 and none, it ends with:
///
/// ```text
/// values  = { encode_bound(5) => 2 }
/// present = 2   // so the missing count is 3 - 2 = 1
/// ```
pub(super) struct AttrAcc {
    /// Schema id of the attribute being counted.
    pub(super) attr_id: AttrId,
    /// Codec tag for the attribute's value type (the element type for arrays).
    /// Used to scan storage keys and to decode the bytes in `values`.
    pub(super) type_tag: u8,
    /// Whether any facet needs per-value counts. This is true for terms, stats,
    /// histogram and scalar range facets. Array range facets use `array_ranges`
    /// instead.
    pub(super) need_values: bool,
    /// Whether any facet sets `include_missing`, and so needs `present`.
    pub(super) need_present: bool,
    /// The part of the value space to scan, merged from what each facet needs.
    /// It limits the global-count and posting scans. The forward row pass does
    /// not apply it, so finalization still filters by prefix and range.
    pub(super) sel: ScanSel,
    /// Document count per encoded value. For arrays, each element is a separate
    /// value, and repeats within one document count once.
    pub(super) values: HashMap<Vec<u8>, u64>,
    /// Number of counted documents that have this attribute. The missing count
    /// is the population total minus this. Only rely on it when `need_present`
    /// is true, because not every strategy fills it otherwise.
    pub(super) present: u64,
    /// One entry for each range facet on this attribute, if it is an array.
    /// Empty otherwise.
    pub(super) array_ranges: Vec<ArrayRangeAcc>,
}

/// Tells finalization where to find the data for one requested facet.
///
/// Accumulators are shared per attribute but results are produced per facet, so
/// each facet needs a pointer back to its data.
///
/// # Example
///
/// For a range facet that is the second one on the array attribute `sizes`,
/// where `sizes` owns the first accumulator:
///
/// ```text
/// FacetPlan {
///     acc_idx: 0,
///     array_range: Some(1),
///     attr_type: AttributeType::IntArray,
/// }
/// ```
///
/// Its bucket counts are in `accs[0].array_ranges[1].counts`.
pub(super) struct FacetPlan {
    /// Index into [`AttrAccs::accs`] of this facet's attribute accumulator.
    pub(super) acc_idx: usize,
    /// For a range facet on an array attribute, the index into that
    /// accumulator's [`AttrAcc::array_ranges`]. `None` for all other facets.
    pub(super) array_range: Option<usize>,
    /// Schema type of the attribute. Currently used by a debug assertion that
    /// stats facets are numeric.
    pub(super) attr_type: AttributeType,
}

/// All accumulation state for a group of facets computed over the same documents.
///
/// Created by [`AttrAccs::build`], filled by one of the counting strategies, then
/// turned into results by `finalize_into`.
///
/// # Example
///
/// If `build` is given `idxs = [0, 2, 3]`, and those specs are terms on `color`,
/// stats on `price` and terms on `price`:
///
/// ```text
/// accs      = [color, price]  // one per distinct attribute
/// plans     = [acc_idx 0, acc_idx 1, acc_idx 1]
/// spec_idxs = [0, 2, 3]
/// ```
pub(super) struct AttrAccs {
    /// One accumulator per distinct attribute, in order of first use.
    pub(super) accs: Vec<AttrAcc>,
    /// One plan per facet in this group.
    pub(super) plans: Vec<FacetPlan>,
    /// Each facet's position in the full spec list. `spec_idxs[i]` goes with
    /// `plans[i]`, and says which output slot receives that facet's result.
    pub(super) spec_idxs: Vec<usize>,
}

pub(super) fn encode_bound(v: &ScalarValue) -> Result<Vec<u8>> {
    let mut b = Vec::with_capacity(16);
    encode_scalar_val_bytes(v, &mut b)?;
    Ok(b)
}

impl AttrAccs {
    pub(super) fn build(env: &FacetEnv<'_>, specs: &[FacetSpec], idxs: &[usize]) -> Result<Self> {
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

    pub(super) fn has_array_ranges(&self) -> bool {
        self.plans.iter().any(|p| p.array_range.is_some())
    }
}
