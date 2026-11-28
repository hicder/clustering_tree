//! Posting and range loading for plan evaluation.

use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_core::schema::{CollectionSchema, ColumnStorage, ScalarValue, SchemaEntry};
use izzydb_storage::codec::{
    SegBitsetAccum, SegPostingView, ZoneEntry, encode_f64_ordered, encode_i64_ordered,
    encode_str_ordered, ordered_type_tag,
};
use izzydb_storage::bsi::{
    BSI_BASE_PLANE, BSI_CHUNK_ROWS, BSI_PRESENT_PLANE, BSI_WORDS, BsiChunk, decode_base,
    decode_plane,
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
        let mut accum: BTreeMap<SegmentId, SegBitsetAccum> = BTreeMap::new();
        for key in attr_keys {
            merge_postings_for_key(storage, snap, key, &mut accum)?;
        }
        Ok(finalize_accum_map(snap, accum))
    }
}

pub(super) fn merge_postings_for_key(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    attr_key: &[u8],
    by_seg: &mut BTreeMap<SegmentId, SegBitsetAccum>,
) -> Result<()> {
    storage.visit_attr_postings_for_value(attr_key, |seg, bytes| {
        if !snap.is_visible_segment(seg) {
            return Ok(());
        }
        let view = SegPostingView::parse(bytes)?;
        let acc = by_seg.entry(seg).or_default();
        view.or_into_accum(acc)?;
        Ok(())
    })
}

pub(super) fn finalize_seg_map(
    snap: &SegmentTableSnapshot,
    by_seg: BTreeMap<SegmentId, RoaringBitmap>,
) -> RowSet {
    let pairs: Vec<(SegmentId, RoaringBitmap)> = by_seg.into_iter().collect();
    RowSet::from_sorted_segs(pairs).into_live(snap)
}

pub(super) fn finalize_accum_map(
    snap: &SegmentTableSnapshot,
    by_seg: BTreeMap<SegmentId, SegBitsetAccum>,
) -> RowSet {
    let pairs: Vec<(SegmentId, RoaringBitmap)> = by_seg
        .into_iter()
        .map(|(seg, acc)| (seg, acc.into_roaring()))
        .collect();
    RowSet::from_sorted_segs(pairs).into_live(snap)
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

    let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();

    // Bit-sliced index fast path: only worthwhile when the range spans enough distinct
    // values that unioning value-major postings would dominate the plane reads.
    let mut resolved_segs = RoaringBitmap::new();
    if entry.storage == ColumnStorage::Inline
        && matches!(
            entry.attr_type,
            izzydb_core::schema::AttributeType::Int
                | izzydb_core::schema::AttributeType::Uint
                | izzydb_core::schema::AttributeType::Float
        )
        && bsi_worth_using(ctx, attr_id, tag, lo, hi)?
    {
        for seg in eligible_segs.iter() {
            if let Some(bm) = load_bsi_segment(ctx.storage, ctx.snap, seg, attr_id, lo, hi)? {
                resolved_segs.insert(seg);
                if !bm.is_empty() {
                    by_seg.insert(seg, bm);
                }
            }
        }
    }

    let start_val = lo.map_or(&[][..], |(b, _)| b.as_slice());
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
                    && (val_bytes < lo_b.as_slice() || (!*lo_incl && val_bytes == lo_b.as_slice()))
                {
                    return Ok(true);
                }
                if let Some((hi_b, hi_incl)) = hi
                    && (val_bytes > hi_b.as_slice() || (!*hi_incl && val_bytes == hi_b.as_slice()))
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
        let mut accum: BTreeMap<SegmentId, SegBitsetAccum> = BTreeMap::new();
        ctx.storage.visit_attr_postings_range(
            attr_id,
            tag,
            start_val,
            |val_bytes, seg, bytes| {
                if let Some((lo_b, lo_incl)) = lo
                    && (val_bytes < lo_b.as_slice() || (!*lo_incl && val_bytes == lo_b.as_slice()))
                {
                    return Ok(true);
                }
                if let Some((hi_b, hi_incl)) = hi
                    && (val_bytes > hi_b.as_slice() || (!*hi_incl && val_bytes == hi_b.as_slice()))
                {
                    return Ok(false);
                }
                if !eligible_segs.contains(seg) || resolved_segs.contains(seg) {
                    return Ok(true);
                }
                visited_segs.insert(seg);
                let view = SegPostingView::parse(bytes)?;
                let acc = accum.entry(seg).or_default();
                view.or_into_accum(acc)?;
                Ok(true)
            },
        )?;
        for (seg, acc) in accum {
            let rb = acc.into_roaring();
            let entry = by_seg.entry(seg).or_default();
            *entry |= rb;
        }
    }

    ctx.trace.range_segments_visited += visited_segs.len() as usize;
    let pairs: Vec<(SegmentId, RoaringBitmap)> = by_seg.into_iter().collect();
    Ok(RowSet::from_sorted_segs(pairs).into_live(ctx.snap))
}

/// Minimum number of distinct values a range must span before BSI plane reads beat a
/// value-major posting union.
const BSI_MIN_DISTINCT_VALUES: usize = 1_024;

/// Probe the value counts cheaply and decide whether the bit-sliced index is worthwhile.
fn bsi_worth_using(
    ctx: &PlanCtx<'_>,
    attr_id: AttrId,
    tag: u8,
    lo: Option<&(Vec<u8>, bool)>,
    hi: Option<&(Vec<u8>, bool)>,
) -> Result<bool> {
    let start_val = lo.map_or(&[][..], |(b, _)| b.as_slice());
    let mut distinct = 0usize;
    let mut reached_threshold = false;
    ctx.storage
        .visit_attr_counts(attr_id, tag, start_val, |val_bytes, _cnt| -> Result<bool> {
            if let Some((lo_b, lo_incl)) = lo
                && (val_bytes < lo_b.as_slice() || (!*lo_incl && val_bytes == lo_b.as_slice()))
            {
                return Ok(true);
            }
            if let Some((hi_b, hi_incl)) = hi
                && (val_bytes > hi_b.as_slice() || (!*hi_incl && val_bytes == hi_b.as_slice()))
            {
                return Ok(false);
            }
            distinct += 1;
            if distinct >= BSI_MIN_DISTINCT_VALUES {
                reached_threshold = true;
                return Ok(false);
            }
            Ok(true)
        })?;
    Ok(reached_threshold)
}

/// Load and evaluate the bit-sliced index for one segment.
///
/// Returns `None` when the segment has no complete BSI for `attr_id`, in which case the caller
/// falls back to value-major postings.
fn load_bsi_segment(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    seg: SegmentId,
    attr_id: AttrId,
    lo: Option<&(Vec<u8>, bool)>,
    hi: Option<&(Vec<u8>, bool)>,
) -> Result<Option<RoaringBitmap>> {
    let ordered_bound = |bound: &Option<&(Vec<u8>, bool)>| -> Option<Option<(u64, bool)>> {
        match bound {
            None => Some(None),
            Some((bytes, incl)) if bytes.len() == 8 => {
                Some(Some((u64::from_be_bytes(bytes.as_slice().try_into().unwrap()), *incl)))
            }
            Some(_) => None,
        }
    };
    let Some(lo_bound) = ordered_bound(&lo) else {
        return Ok(None);
    };
    let Some(hi_bound) = ordered_bound(&hi) else {
        return Ok(None);
    };
    let row_count = snap.segment_row_count(seg).unwrap_or(0);
    if row_count == 0 {
        return Ok(Some(RoaringBitmap::new()));
    }
    let chunks = row_count.div_ceil(BSI_CHUNK_ROWS);
    let mut out = RoaringBitmap::new();
    for chunk in 0..chunks {
        let Some(base_bytes) = storage.get_bsi_plane(seg, attr_id, chunk, BSI_BASE_PLANE)? else {
            return Ok(None);
        };
        let Some(present_bytes) = storage.get_bsi_plane(seg, attr_id, chunk, BSI_PRESENT_PLANE)?
        else {
            return Ok(None);
        };
        let Some((base, width)) = decode_base(&base_bytes) else {
            return Err(TreeError::CorruptedData(format!(
                "invalid cf_bsi base record for seg {seg} attr {attr_id} chunk {chunk}"
            )));
        };
        let Some(present) = decode_plane(&present_bytes) else {
            return Err(TreeError::CorruptedData(format!(
                "invalid cf_bsi presence plane for seg {seg} attr {attr_id} chunk {chunk}"
            )));
        };
        let keys: Vec<(SegmentId, AttrId, u32, u8)> =
            (0..width).map(|plane| (seg, attr_id, chunk, plane)).collect();
        let slices = storage.multi_get_bsi_planes_pinned(&keys)?;
        let planes: Vec<(u8, Option<Box<[u64; BSI_WORDS]>>)> = keys
            .iter()
            .zip(slices.iter())
            .map(|(&(_, _, _, plane), slice)| {
                let decoded = slice.as_ref().and_then(|bytes| decode_plane(bytes));
                (plane, decoded)
            })
            .collect();
        let chunk_bsi = BsiChunk::new(base, width, present, planes);
        let mut mask = match lo_bound {
            Some((lo, true)) => chunk_bsi.greater_equal(lo),
            Some((lo, false)) => chunk_bsi.greater_than(lo),
            None => *chunk_bsi.present,
        };
        if let Some((hi, incl)) = hi_bound {
            let hi_mask = if incl {
                chunk_bsi.less_equal(hi)
            } else {
                chunk_bsi.less_than(hi)
            };
            BsiChunk::intersect(&mut mask, &hi_mask);
        }
        BsiChunk::to_bitmap(&mask, chunk, &mut out);
    }
    Ok(Some(out))
}
