//! Facet finalization.

use std::collections::{BTreeMap, HashMap};
use izzydb_core::error::Result;
use izzydb_core::schema::AttributeType;
use izzydb_storage::codec::{
    decode_f64_ordered, decode_i64_ordered,
    decode_scalar_val_bytes,
    encode_str_prefix_ordered,
};

use super::*;

impl AttrAccs {
    // ---- Finalization --------------------------------------------------------------------

    pub(super) fn finalize_into(
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
pub(super) fn bump(map: &mut HashMap<Vec<u8>, u64>, key: &[u8], by: u64) {
    if let Some(c) = map.get_mut(key) {
        *c += by;
    } else {
        map.insert(key.to_vec(), by);
    }
}

pub(super) fn encode_ranges(ranges: &[FacetRange]) -> Result<Vec<EncRange>> {
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

pub(super) fn span_lo(ranges: &[EncRange]) -> Option<Vec<u8>> {
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

pub(super) fn span_hi(ranges: &[EncRange]) -> Option<Vec<u8>> {
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

pub(super) fn decode_numeric(type_tag: u8, bytes: &[u8]) -> f64 {
    let arr: [u8; 8] = bytes.try_into().unwrap_or([0; 8]);
    match type_tag {
        izzydb_storage::codec::TAG_INT => decode_i64_ordered(arr) as f64,
        izzydb_storage::codec::TAG_UINT => u64::from_be_bytes(arr) as f64,
        _ => decode_f64_ordered(arr),
    }
}

pub(super) fn finalize_facet(
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
