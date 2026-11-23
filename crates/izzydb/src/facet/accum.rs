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

#[derive(Debug, Clone)]
pub(super) struct EncRange {
    pub(super) lo: Option<(Vec<u8>, bool)>,
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

pub(super) struct ArrayRangeAcc {
    pub(super) ranges: Vec<EncRange>,
    pub(super) counts: Vec<u64>,
    /// Per-segment union of matching local ids (`seg_idx -> bucket_idx -> bitmap`).
    pub(super) scratch: BTreeMap<SegmentId, Vec<RoaringBitmap>>,
}

pub(super) struct AttrAcc {
    pub(super) attr_id: AttrId,
    pub(super) type_tag: u8,
    pub(super) need_values: bool,
    pub(super) need_present: bool,
    pub(super) sel: ScanSel,
    pub(super) values: HashMap<Vec<u8>, u64>,
    pub(super) present: u64,
    pub(super) array_ranges: Vec<ArrayRangeAcc>,
}

pub(super) struct FacetPlan {
    pub(super) acc_idx: usize,
    pub(super) array_range: Option<usize>,
    pub(super) attr_type: AttributeType,
}

pub(super) struct AttrAccs {
    pub(super) accs: Vec<AttrAcc>,
    pub(super) plans: Vec<FacetPlan>,
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
