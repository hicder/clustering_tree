//! Strategy 4: per-segment posting intersection over `RowSet`.

use std::collections::HashMap;
use roaring::RoaringBitmap;
use izzydb_core::config::{AttrId, SegmentId};
use izzydb_core::error::Result;
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::{AttributeType, ColumnStorage};
use izzydb_storage::codec::{
    SegPostingView, encode_present_attr_key,
};
use izzydb_storage::column::ColumnBlockView;

use super::*;

impl AttrAccs {
    // ---- Strategy 4: per-segment posting intersection over `RowSet` ----------------------

    pub(super) fn accumulate_rowset_postings(&mut self, env: &FacetEnv<'_>, pop: &RowSet) -> Result<()> {
        if self.accs.is_empty() || pop.is_empty() {
            return Ok(());
        }
        let mut key_buf: Vec<u8> = Vec::with_capacity(32);
        let mut elems: Vec<Vec<u8>> = Vec::new();
        for acc in &mut self.accs {
            let is_detached = env
                .schema
                .get_by_id(acc.attr_id)
                .is_some_and(|e| e.storage == ColumnStorage::Detached);
            let col_gen = if is_detached {
                crate::detached::current_detached_gen(env.storage, acc.attr_id)?
            } else {
                0
            };

            let mut ow_pop_by_seg: HashMap<SegmentId, RoaringBitmap> = HashMap::new();
            let mut clean_pop_by_seg: HashMap<SegmentId, RoaringBitmap> = HashMap::new();
            if is_detached {
                for (seg, ow_bm) in env.storage.all_detached_overwritten(acc.attr_id, col_gen)? {
                    if let Some(pop_bm) = pop.segment(seg) {
                        let ow_in_pop = &ow_bm & pop_bm;
                        if !ow_in_pop.is_empty() {
                            clean_pop_by_seg.insert(seg, pop_bm - &ow_in_pop);
                            ow_pop_by_seg.insert(seg, ow_in_pop);
                        }
                    }
                }
            }
            let effective_mask = |seg: SegmentId| -> Option<&RoaringBitmap> {
                if let Some(clean) = clean_pop_by_seg.get(&seg) {
                    Some(clean)
                } else {
                    pop.segment(seg)
                }
            };

            if acc.need_present {
                let key = encode_present_attr_key(acc.attr_id);
                if is_detached {
                    env.storage.visit_detached_postings_for_value(
                        acc.attr_id,
                        col_gen,
                        key[2],
                        &key[3..],
                        |seg, bytes| {
                            let Some(mask) = effective_mask(seg) else {
                                return Ok(());
                            };
                            acc.present += SegPostingView::parse(bytes)?.and_count(mask)?;
                            Ok(())
                        },
                    )?;
                } else {
                    env.storage
                        .visit_attr_postings_for_value(&key, |seg, bytes| {
                            let Some(pop_bm) = pop.segment(seg) else {
                                return Ok(());
                            };
                            acc.present += SegPostingView::parse(bytes)?.and_count(pop_bm)?;
                            Ok(())
                        })?;
                }
            }

            if acc.need_values || !acc.array_ranges.is_empty() {
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
                let visit_posting = |val: &[u8], seg: SegmentId, bytes: &[u8]| -> Result<bool> {
                    if sel.past_end(val) {
                        return Ok(false);
                    }
                    let Some(mask) = effective_mask(seg) else {
                        return Ok(true);
                    };
                    let view = SegPostingView::parse(bytes)?;
                    if need_values {
                        let c = view.and_count(mask)?;
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
                                view.or_into_masked(mask, &mut sb[b])?;
                            }
                        }
                    }
                    Ok(true)
                };
                if is_detached {
                    env.storage.visit_detached_postings_range(
                        *attr_id,
                        col_gen,
                        *type_tag,
                        sel.start(),
                        visit_posting,
                    )?;
                } else {
                    env.storage.visit_attr_postings_range(
                        *attr_id,
                        *type_tag,
                        sel.start(),
                        visit_posting,
                    )?;
                }
                for ar in &mut acc.array_ranges {
                    for seg_buckets in ar.scratch.values() {
                        for (b, bm) in seg_buckets.iter().enumerate() {
                            ar.counts[b] += bm.len();
                        }
                    }
                }
            }

            // Re-evaluate overwritten rows in `pop` directly from `cf_detached`.
            if !ow_pop_by_seg.is_empty() {
                let mut ow_keys: Vec<(AttrId, u32, SegmentId, u32)> = Vec::new();
                let mut seg_blocks: Vec<(SegmentId, u32, Vec<u8>)> = Vec::new();
                let mut seg_list: Vec<(SegmentId, RoaringBitmap)> = ow_pop_by_seg.into_iter().collect();
                seg_list.sort_unstable_by_key(|(s, _)| *s);
                for (seg, bm) in seg_list {
                    for local in bm.iter() {
                        let blk = local >> 8;
                        let slot = (local & 0xFF) as u8;
                        if let Some(last) = seg_blocks.last_mut()
                            && last.0 == seg
                            && last.1 == blk
                        {
                            last.2.push(slot);
                        } else {
                            seg_blocks.push((seg, blk, vec![slot]));
                            ow_keys.push((acc.attr_id, col_gen, seg, blk));
                        }
                    }
                }
                let slices = env
                    .storage
                    .multi_get_detached_blocks_pinned(&ow_keys, true)?;
                let is_array = acc.array_ranges.is_empty()
                    && matches!(
                        env.schema.get_by_id(acc.attr_id).map(|e| e.attr_type),
                        Some(
                            AttributeType::IntArray
                                | AttributeType::UintArray
                                | AttributeType::StringArray
                        )
                    );
                let collect_elems = !acc.array_ranges.is_empty() || is_array;
                for ((_, _, slots), opt_slice) in seg_blocks.iter().zip(slices.iter()) {
                    let Some(slice) = opt_slice.as_ref() else {
                        continue;
                    };
                    let view = ColumnBlockView::parse(slice)?;
                    for &slot in slots {
                        if !view.is_present(slot) {
                            continue;
                        }
                        if acc.need_present {
                            acc.present += 1;
                        }
                        if !acc.need_values && acc.array_ranges.is_empty() {
                            continue;
                        }
                        if !collect_elems {
                            if acc.need_values {
                                view.for_each_encoded_value(slot, &mut key_buf, |enc| {
                                    bump(&mut acc.values, enc, 1);
                                })?;
                            }
                            continue;
                        }
                        elems.clear();
                        view.for_each_encoded_value(slot, &mut key_buf, |enc| {
                            elems.push(enc.to_vec());
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
        }
        Ok(())
    }
}
