//! Posting and range loading for plan evaluation.

use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_core::schema::{CollectionSchema, ColumnStorage, ScalarValue, SchemaEntry};
use izzydb_storage::codec::{
    SegPostingView, ZoneEntry, encode_f64_ordered,
    encode_i64_ordered, encode_str_ordered,
    ordered_type_tag,
};
use izzydb_storage::{SegmentTableSnapshot, Storage};

use crate::detached::{
    apply_overwritten_correction_for_filter, block_slot_matches_any_encoded, current_detached_gen,
};


use super::*;

#[inline]
pub(super) fn get_attr_or_detached_count(
    storage: &Storage,
    entry: &SchemaEntry,
    attr_key: &[u8],
) -> Result<u64> {
    if entry.storage == ColumnStorage::Detached {
        let col_gen = current_detached_gen(storage, entry.id)?;
        storage.get_detached_count(col_gen, attr_key)
    } else {
        storage.get_attr_count(attr_key)
    }
}

pub(super) fn load_postings_for_keys(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    entry: &SchemaEntry,
    attr_keys: &[Vec<u8>],
) -> Result<RowSet> {
    let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    if entry.storage == ColumnStorage::Detached {
        if attr_keys.is_empty() {
            return Ok(RowSet::empty());
        }
        let col_gen = current_detached_gen(storage, entry.id)?;
        let is_present_only = attr_keys.iter().any(|k| k.len() == 3);
        let mut enc_targets: Vec<Vec<u8>> = if is_present_only {
            Vec::new()
        } else {
            attr_keys.iter().map(|k| k[3..].to_vec()).collect()
        };
        enc_targets.sort_unstable();
        enc_targets.dedup();
        for key in attr_keys {
            let tag = key[2];
            storage.visit_detached_postings_for_value(
                entry.id,
                col_gen,
                tag,
                &key[3..],
                |seg, bytes| {
                    if !snap.is_visible_segment(seg) {
                        return Ok(());
                    }
                    let dst = by_seg.entry(seg).or_default();
                    SegPostingView::parse(bytes)?.or_into(dst)?;
                    Ok(())
                },
            )?;
        }
        apply_overwritten_correction_for_filter(
            storage,
            snap,
            entry,
            col_gen,
            &mut by_seg,
            |view, slot| {
                if is_present_only {
                    Ok(true)
                } else {
                    block_slot_matches_any_encoded(view, slot, &enc_targets)
                }
            },
        )?;
        Ok(finalize_seg_map(snap, by_seg))
    } else {
        for key in attr_keys {
            merge_postings_for_key(storage, snap, key, &mut by_seg)?;
        }
        Ok(finalize_seg_map(snap, by_seg))
    }
}

pub(super) fn merge_postings_for_key(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    attr_key: &[u8],
    by_seg: &mut BTreeMap<SegmentId, RoaringBitmap>,
) -> Result<()> {
    storage.visit_attr_postings_for_value(attr_key, |seg, bytes| {
        if !snap.is_visible_segment(seg) {
            return Ok(());
        }
        let dst = by_seg.entry(seg).or_default();
        SegPostingView::parse(bytes)?.or_into(dst)?;
        Ok(())
    })
}

pub(super) fn finalize_seg_map(
    snap: &SegmentTableSnapshot,
    by_seg: BTreeMap<SegmentId, RoaringBitmap>,
) -> RowSet {
    let pairs: Vec<(SegmentId, RoaringBitmap)> = by_seg.into_iter().collect();
    RowSet::from_sorted_segs(pairs).and_live(snap)
}

pub(super) type EncodedBounds = (AttrId, u8, Option<(Vec<u8>, bool)>, Option<(Vec<u8>, bool)>);

pub(super) fn encode_range_bounds(
    schema: &CollectionSchema,
    attr: &str,
    lo: Option<&(ScalarValue, bool)>,
    hi: Option<&(ScalarValue, bool)>,
) -> Result<EncodedBounds> {
    let entry = schema
        .get_by_name(attr)
        .ok_or_else(|| TreeError::InvalidFilter(format!("unknown attribute '{attr}'")))?;
    let tag = ordered_type_tag(entry.attr_type)?;
    let enc = |v: &ScalarValue| -> Result<Vec<u8>> {
        match v {
            ScalarValue::Int(x) => Ok(encode_i64_ordered(*x).to_vec()),
            ScalarValue::Uint(x) => Ok(x.to_be_bytes().to_vec()),
            ScalarValue::Float(x) => Ok(encode_f64_ordered(*x)?.to_vec()),
            ScalarValue::String(s) => {
                let mut out = Vec::with_capacity(s.len() + 2);
                encode_str_ordered(s.as_bytes(), &mut out);
                Ok(out)
            }
            other => Err(TreeError::InvalidFilter(format!(
                "invalid range bound value {other:?}"
            ))),
        }
    };
    let lo_b = lo
        .map(|(v, incl)| Ok::<_, TreeError>((enc(v)?, *incl)))
        .transpose()?;
    let hi_b = hi
        .map(|(v, incl)| Ok::<_, TreeError>((enc(v)?, *incl)))
        .transpose()?;
    Ok((entry.id, tag, lo_b, hi_b))
}

pub(super) fn range_is_empty(lo: Option<&(Vec<u8>, bool)>, hi: Option<&(Vec<u8>, bool)>) -> bool {
    if let (Some((lo_b, lo_incl)), Some((hi_b, hi_incl))) = (lo, hi) {
        match lo_b.cmp(hi_b) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Equal => !(*lo_incl && *hi_incl),
            std::cmp::Ordering::Less => false,
        }
    } else {
        false
    }
}

pub(super) fn zone_overlaps(
    zone: &ZoneEntry,
    lo: Option<&(Vec<u8>, bool)>,
    hi: Option<&(Vec<u8>, bool)>,
) -> bool {
    if let Some((lo_b, lo_incl)) = lo {
        match zone.max.as_slice().cmp(lo_b.as_slice()) {
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal if !*lo_incl => return false,
            _ => {}
        }
    }
    if let Some((hi_b, hi_incl)) = hi {
        match zone.min.as_slice().cmp(hi_b.as_slice()) {
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal if !*hi_incl => return false,
            _ => {}
        }
    }
    true
}

pub(super) fn load_postings_range(
    ctx: &mut PlanCtx<'_>,
    entry: &SchemaEntry,
    attr_id: AttrId,
    tag: u8,
    lo: Option<&(Vec<u8>, bool)>,
    hi: Option<&(Vec<u8>, bool)>,
) -> Result<RowSet> {
    let seg_zones: Option<&HashMap<SegmentId, ZoneEntry>> = ctx.zones.get(&(attr_id, tag));
    let mut eligible_segs = RoaringBitmap::new();
    for seg in ctx.snap.visible_segments() {
        ctx.trace.range_zones_checked += 1;
        let overlaps = seg_zones
            .and_then(|m| m.get(&seg))
            .is_some_and(|z| zone_overlaps(z, lo, hi));
        if overlaps {
            eligible_segs.insert(seg);
        } else {
            ctx.trace.range_segments_pruned += 1;
        }
    }
    if eligible_segs.is_empty() {
        return Ok(RowSet::empty());
    }

    let start_val = lo.map_or(&[][..], |(b, _)| b.as_slice());
    let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    let mut visited_segs = RoaringBitmap::new();

    if entry.storage == ColumnStorage::Detached {
        let col_gen = current_detached_gen(ctx.storage, attr_id)?;
        ctx.storage.visit_detached_postings_range(
            attr_id,
            col_gen,
            tag,
            start_val,
            |val_bytes, seg, bytes| {
                if let Some((lo_b, lo_incl)) = lo
                    && (val_bytes < lo_b.as_slice()
                        || (!*lo_incl && val_bytes == lo_b.as_slice()))
                {
                    return Ok(true);
                }
                if let Some((hi_b, hi_incl)) = hi
                    && (val_bytes > hi_b.as_slice()
                        || (!*hi_incl && val_bytes == hi_b.as_slice()))
                {
                    return Ok(false);
                }
                if !eligible_segs.contains(seg) {
                    return Ok(true);
                }
                visited_segs.insert(seg);
                let dst = by_seg.entry(seg).or_default();
                SegPostingView::parse(bytes)?.or_into(dst)?;
                Ok(true)
            },
        )?;
        let mut buf = Vec::new();
        apply_overwritten_correction_for_filter(
            ctx.storage,
            ctx.snap,
            entry,
            col_gen,
            &mut by_seg,
            |view, slot| {
                let mut matched = false;
                view.for_each_encoded_value(slot, &mut buf, |val_bytes| {
                    let ok_lo = lo.is_none_or(|(lo_b, incl)| {
                        val_bytes > lo_b.as_slice() || (*incl && val_bytes == lo_b.as_slice())
                    });
                    let ok_hi = hi.is_none_or(|(hi_b, incl)| {
                        val_bytes < hi_b.as_slice() || (*incl && val_bytes == hi_b.as_slice())
                    });
                    if ok_lo && ok_hi {
                        matched = true;
                    }
                })?;
                Ok(matched)
            },
        )?;
    } else {
        ctx.storage
            .visit_attr_postings_range(attr_id, tag, start_val, |val_bytes, seg, bytes| {
                if let Some((lo_b, lo_incl)) = lo
                    && (val_bytes < lo_b.as_slice()
                        || (!*lo_incl && val_bytes == lo_b.as_slice()))
                {
                    return Ok(true);
                }
                if let Some((hi_b, hi_incl)) = hi
                    && (val_bytes > hi_b.as_slice()
                        || (!*hi_incl && val_bytes == hi_b.as_slice()))
                {
                    return Ok(false);
                }
                if !eligible_segs.contains(seg) {
                    return Ok(true);
                }
                visited_segs.insert(seg);
                let dst = by_seg.entry(seg).or_default();
                SegPostingView::parse(bytes)?.or_into(dst)?;
                Ok(true)
            })?;
    }

    ctx.trace.range_segments_visited += visited_segs.len() as usize;
    Ok(finalize_seg_map(ctx.snap, by_seg))
}
