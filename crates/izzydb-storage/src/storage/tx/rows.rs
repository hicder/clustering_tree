//! Rows, vectors, id map, and row-to-doc block map overlay.



use izzydb_core::config::{
    AttrId, DocId, ROW_BLOCK_SIZE, RowId, SegmentId,
};
use izzydb_core::error::{Result, TreeError};

use crate::codec::column_key;
use crate::node::decode_f32_into;


use super::*;

impl <'a> TxContext<'a> {
    // -----------------------------------------------------------------------
    // Rows, vectors, id map, and row->doc block map
    // -----------------------------------------------------------------------

    pub fn get_id_map(&self, doc_id: DocId) -> Result<Option<RowId>> {
        if let Some(opt) = self.dirty_id_map.get(&doc_id) {
            return Ok(*opt);
        }
        self.storage.get_id_map(doc_id)
    }

    pub fn put_id_map(&mut self, doc_id: DocId, row: RowId) {
        self.dirty_id_map.insert(doc_id, Some(row));
    }

    pub fn delete_id_map(&mut self, doc_id: DocId) {
        self.dirty_id_map.insert(doc_id, None);
    }

    pub fn get_row_bytes(&self, row: RowId) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_rows.get(&row) {
            return Ok(opt.clone());
        }
        self.storage.get_row_bytes(row)
    }

    pub fn put_row_bytes(&mut self, row: RowId, bytes: Vec<u8>) {
        self.dirty_rows.insert(row, Some(bytes));
    }

    pub fn delete_row(&mut self, row: RowId) {
        self.dirty_rows.insert(row, None);
    }

    pub fn put_column_block(
        &mut self,
        seg: SegmentId,
        attr_id: AttrId,
        block: u32,
        bytes: Vec<u8>,
    ) {
        self.dirty_columns.insert(column_key(seg, attr_id, block), bytes);
    }

    /// Visit every live (non-tombstoned, non-deleted) row in `seg` in ascending `RowId` order,
    /// overlaying uncommitted `dirty_rows` and `dirty_seg_deletes` inside this transaction.
    pub fn visit_segment_live_rows_in_tx(
        &self,
        seg: SegmentId,
        mut visit: impl FnMut(RowId, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let mut last_db_row: Option<RowId> = None;
        let snap = self.storage.segments.snapshot();
        let deleted_bm = self
            .dirty_seg_deletes
            .get(&seg)
            .or_else(|| snap.get(seg).and_then(|e| e.deleted.as_deref()));

        self.storage.visit_segment_rows(seg, |row, db_bytes| {
            last_db_row = Some(row);
            if deleted_bm.is_some_and(|bm| bm.contains(row.local())) {
                return Ok(());
            }
            if let Some(opt) = self.dirty_rows.get(&row) {
                if let Some(bytes) = opt {
                    visit(row, bytes)?;
                }
            } else {
                visit(row, db_bytes)?;
            }
            Ok(())
        })?;

        let start_dirty = match last_db_row {
            Some(r) => RowId::new(seg, r.local().saturating_add(1)),
            None => RowId::new(seg, 0),
        };
        let end_dirty = RowId::new(seg, u32::MAX);
        for (&row, opt) in self.dirty_rows.range(start_dirty..=end_dirty) {
            if deleted_bm.is_some_and(|bm| bm.contains(row.local())) {
                continue;
            }
            if let Some(bytes) = opt {
                visit(row, bytes)?;
            }
        }
        Ok(())
    }

    pub fn set_row_doc_id(&mut self, row: RowId, doc_id: DocId) -> Result<()> {
        let key = (row.seg(), row.block_idx());
        if !self.dirty_doc_id_blocks.contains_key(&key) {
            let block = match self.storage.get_doc_id_block(key.0, key.1)? {
                Some(arc) => Box::new(*arc),
                None => Box::new([0u64; ROW_BLOCK_SIZE]),
            };
            self.dirty_doc_id_blocks.insert(key, block);
        }
        self.dirty_doc_id_blocks.get_mut(&key).unwrap()[row.slot() as usize] = doc_id;
        Ok(())
    }

    pub fn put_vector(&mut self, row: RowId, vec: Vec<f32>) {
        debug_assert_eq!(vec.len(), self.storage.dim);
        self.dirty_vectors.insert(row, Some(vec));
    }

    pub fn delete_vector(&mut self, row: RowId) {
        self.dirty_vectors.insert(row, None);
    }

    pub fn get_vector(&self, row: RowId) -> Result<Option<Vec<f32>>> {
        if let Some(opt) = self.dirty_vectors.get(&row) {
            return Ok(opt.clone());
        }
        self.storage.get_vector(row)
    }

    /// Fetch full-precision vectors for `rows` into a flat `rows.len() * dim` buffer,
    /// honoring read-your-own-writes mutations in `dirty_vectors`.
    pub fn get_vectors_flat(&self, rows: &[RowId]) -> Result<Vec<f32>> {
        let dim = self.storage.dim;
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![0.0f32; rows.len() * dim];
        let mut missing: Vec<(RowId, usize)> = Vec::new();
        for (slot, &row) in rows.iter().enumerate() {
            match self.dirty_vectors.get(&row) {
                Some(Some(vec)) => {
                    out[slot * dim..(slot + 1) * dim].copy_from_slice(vec);
                }
                Some(None) => {
                    return Err(TreeError::CorruptedData(format!(
                        "vector {row:?} was deleted in tx"
                    )));
                }
                None => missing.push((row, slot)),
            }
        }
        if !missing.is_empty() {
            missing.sort_unstable_by_key(|&(row, _)| row);
            let sorted_ids: Vec<RowId> = missing.iter().map(|&(row, _)| row).collect();
            let slices = self.storage.multi_get_vectors_pinned(&sorted_ids)?;
            let expected_bytes = dim * 4;
            for (&(row, slot), opt_slice) in missing.iter().zip(slices.iter()) {
                let slice = opt_slice.as_ref().ok_or_else(|| {
                    TreeError::CorruptedData(format!("missing vector {row:?} in cf_vectors"))
                })?;
                if slice.len() != expected_bytes {
                    return Err(TreeError::CorruptedData(format!(
                        "vector {row:?} expected {expected_bytes} bytes, got {}",
                        slice.len()
                    )));
                }
                decode_f32_into(slice, &mut out[slot * dim..(slot + 1) * dim])?;
            }
        }
        Ok(out)
    }
}
