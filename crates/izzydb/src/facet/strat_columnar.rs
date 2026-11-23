//! Strategy 3: forward `ColumnReader` pass with `RowStoreReader` fallback.

use std::collections::HashMap;
use izzydb_core::config::{RowId, SegmentId};
use izzydb_core::error::Result;
use izzydb_core::schema::AttributeType;
use izzydb_storage::codec::RowFieldRef;
use izzydb_storage::column::ColumnBlockView;
use crate::attr_reader::{AttrReader, ColumnReader, RowStoreReader};

use super::*;

impl AttrAccs {
    // ---- Strategy 3: forward `ColumnReader` (sealed) + `RowStoreReader` fallback pass ----

    pub(super) fn accumulate_rows(&mut self, env: &FacetEnv<'_>, sorted_rows: &[RowId]) -> Result<()> {
        if self.accs.is_empty() || sorted_rows.is_empty() {
            return Ok(());
        }
        let col_reader = ColumnReader::new(env.storage, env.schema)?;
        let mut key_buf: Vec<u8> = Vec::with_capacity(32);
        let mut elems: Vec<Vec<u8>> = Vec::new();
        let mut need_row_store = vec![false; sorted_rows.len()];
        let mut row_store_attrs = vec![false; self.accs.len()];

        for (ai, acc) in self.accs.iter_mut().enumerate() {
            let mut col_rows: Vec<usize> = Vec::new();
            let mut block_coords: Vec<(SegmentId, u32)> = Vec::new();
            for (ri, &row) in sorted_rows.iter().enumerate() {
                if col_reader.segment_has_columns(row.seg(), acc.attr_id) {
                    col_rows.push(ri);
                    let k = (row.seg(), row.block_idx());
                    if block_coords.last().copied() != Some(k) {
                        block_coords.push(k);
                    }
                } else {
                    need_row_store[ri] = true;
                    row_store_attrs[ai] = true;
                }
            }
            if col_rows.is_empty() {
                continue;
            }
            let block_slices = col_reader.fetch_column_slices(acc.attr_id, &block_coords)?;
            let mut block_views: HashMap<(SegmentId, u32), Option<ColumnBlockView<'_>>> =
                HashMap::with_capacity(block_coords.len());
            for (&(seg, block), opt_slice) in block_coords.iter().zip(block_slices.iter()) {
                let parsed = opt_slice
                    .as_ref()
                    .map(|s| ColumnBlockView::parse(s))
                    .transpose()?;
                block_views.insert((seg, block), parsed);
            }

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

            for ri in col_rows {
                let row = sorted_rows[ri];
                let Some(Some(view)) = block_views.get(&(row.seg(), row.block_idx())) else {
                    continue;
                };
                let slot = row.slot();
                if !view.is_present(slot) {
                    continue;
                }
                acc.present += 1;
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

        let fallback_rows: Vec<RowId> = sorted_rows
            .iter()
            .zip(need_row_store.iter())
            .filter_map(|(&r, &need)| need.then_some(r))
            .collect();
        if fallback_rows.is_empty() {
            return Ok(());
        }

        let reader = RowStoreReader::new(env.storage);
        reader.read_rows(&fallback_rows, &mut |row, view| {
            for (ai, acc) in self.accs.iter_mut().enumerate() {
                if !row_store_attrs[ai] || col_reader.segment_has_columns(row.seg(), acc.attr_id) {
                    continue;
                }
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
}
