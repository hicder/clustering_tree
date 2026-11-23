//! Batched, `RowId`-ordered attribute readers: [`RowStoreReader`] (`cf_rows`) and
//! [`ColumnReader`] (`cf_columns` for sealed segments with automatic `cf_rows` fallback).

use std::collections::HashMap;
use std::sync::Arc;

use izzydb_core::config::{AttrId, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{CollectionSchema, ColumnStorage, ScalarValue};
use izzydb_storage::codec::{RowView, SEG_FLAG_HAS_COLUMNS, SegmentState};
use izzydb_storage::column::ColumnBlockView;
use izzydb_storage::{SegmentTableSnapshot, Storage};

use crate::columns::force_row_fallback_enabled;

const READER_CHUNK_SIZE: usize = 1024;

/// Batched, `RowId`-sorted attribute accessor used by ranking, `ORDER BY`, and faceting.
pub(crate) trait AttrReader {
    /// Read scalar attribute `attr` for `rows` (missing/`Null` pushed as `None`), in order.
    fn read_scalars(
        &self,
        attr: AttrId,
        rows: &[RowId],
        out: &mut Vec<Option<ScalarValue>>,
    ) -> Result<()>;

    /// Read numeric attribute `attr` as `f64` for `rows` (missing/`Null`/non-numeric as `None`), in order.
    fn read_numeric_batch(
        &self,
        attr: AttrId,
        rows: &[RowId],
        out: &mut Vec<Option<f64>>,
    ) -> Result<()>;

    /// Stream parsed [`RowView`]s for a batch of `RowId`s without cloning row payloads.
    fn read_rows(
        &self,
        rows: &[RowId],
        visit: &mut dyn FnMut(RowId, RowView<'_>) -> Result<()>,
    ) -> Result<()>;
}

/// Phase 6 `AttrReader` backed directly by `cf_rows` via pinned RocksDB `multi_get`.
pub(crate) struct RowStoreReader<'a> {
    storage: &'a Storage,
}

impl<'a> RowStoreReader<'a> {
    #[inline]
    pub(crate) fn new(storage: &'a Storage) -> Self {
        Self { storage }
    }
}

impl AttrReader for RowStoreReader<'_> {
    fn read_scalars(
        &self,
        attr: AttrId,
        rows: &[RowId],
        out: &mut Vec<Option<ScalarValue>>,
    ) -> Result<()> {
        out.clear();
        out.reserve(rows.len());
        self.read_rows(rows, &mut |_, view| {
            let v = view.get_scalar(attr)?;
            out.push((!matches!(v, ScalarValue::Null)).then_some(v));
            Ok(())
        })
    }

    fn read_numeric_batch(
        &self,
        attr: AttrId,
        rows: &[RowId],
        out: &mut Vec<Option<f64>>,
    ) -> Result<()> {
        out.clear();
        out.reserve(rows.len());
        self.read_rows(rows, &mut |_, view| {
            out.push(view.get_numeric_f64(attr)?);
            Ok(())
        })
    }

    fn read_rows(
        &self,
        rows: &[RowId],
        visit: &mut dyn FnMut(RowId, RowView<'_>) -> Result<()>,
    ) -> Result<()> {
        for chunk in rows.chunks(READER_CHUNK_SIZE) {
            let slices = self.storage.multi_get_row_bytes_pinned(chunk)?;
            for (&row, opt_slice) in chunk.iter().zip(slices.iter()) {
                let slice = opt_slice.as_ref().ok_or_else(|| {
                    TreeError::InvariantViolation(format!("row {row:?} missing in cf_rows"))
                })?;
                let view = RowView::parse(slice)?;
                visit(row, view)?;
            }
        }
        Ok(())
    }
}

/// Phase 7 `AttrReader`: reads sealed segments with `SEG_FLAG_HAS_COLUMNS` from `cf_columns`
/// in sorted `(seg, attr, block)` batches, automatically falling back to `cf_rows` for the open
/// segment or segments whose column build is deferred.
pub(crate) struct ColumnReader<'a> {
    storage: &'a Storage,
    schema: &'a CollectionSchema,
    snap: Arc<SegmentTableSnapshot>,
    row_fallback: RowStoreReader<'a>,
    force_rows: bool,
}

impl<'a> ColumnReader<'a> {
    pub(crate) fn new(storage: &'a Storage, schema: &'a CollectionSchema) -> Self {
        Self {
            storage,
            schema,
            snap: storage.segment_snapshot(),
            row_fallback: RowStoreReader::new(storage),
            force_rows: force_row_fallback_enabled(),
        }
    }

    #[inline]
    pub(crate) fn segment_has_columns(&self, seg: SegmentId, attr: AttrId) -> bool {
        if self.force_rows {
            return false;
        }
        let Some(entry) = self.schema.get_by_id(attr) else {
            return false;
        };
        if entry.storage != ColumnStorage::Inline || !entry.is_columnar() {
            return false;
        }
        self.snap
            .get(seg)
            .is_some_and(|s| {
                s.meta.state == SegmentState::Sealed
                    && (s.meta.flags & SEG_FLAG_HAS_COLUMNS) != 0
            })
    }

    /// Collect sorted `(seg, block)` pairs needed from `cf_columns`, plus fallback `RowId`s.
    fn split_rows_for_attr(
        &self,
        attr: AttrId,
        rows: &[RowId],
    ) -> (Vec<(SegmentId, AttrId, u32)>, Vec<RowId>) {
        let mut block_keys: Vec<(SegmentId, AttrId, u32)> = Vec::new();
        let mut fallback_rows: Vec<RowId> = Vec::new();
        for &row in rows {
            if self.segment_has_columns(row.seg(), attr) {
                let k = (row.seg(), attr, row.block_idx());
                if block_keys.last().copied() != Some(k) {
                    block_keys.push(k);
                }
            } else {
                fallback_rows.push(row);
            }
        }
        (block_keys, fallback_rows)
    }
}

impl AttrReader for ColumnReader<'_> {
    fn read_scalars(
        &self,
        attr: AttrId,
        rows: &[RowId],
        out: &mut Vec<Option<ScalarValue>>,
    ) -> Result<()> {
        out.clear();
        if rows.is_empty() {
            return Ok(());
        }
        let (block_keys, fallback_rows) = self.split_rows_for_attr(attr, rows);
        if block_keys.is_empty() {
            return self.row_fallback.read_scalars(attr, rows, out);
        }

        let block_slices = self
            .storage
            .multi_get_column_blocks_pinned(&block_keys, true)?;
        let mut block_views: HashMap<(SegmentId, u32), Option<ColumnBlockView<'_>>> =
            HashMap::with_capacity(block_keys.len());
        for (&(seg, _, block), opt_slice) in block_keys.iter().zip(block_slices.iter()) {
            let parsed = opt_slice
                .as_ref()
                .map(|s| ColumnBlockView::parse(s))
                .transpose()?;
            block_views.insert((seg, block), parsed);
        }

        let mut fb_vals: Vec<Option<ScalarValue>> = Vec::new();
        if !fallback_rows.is_empty() {
            self.row_fallback
                .read_scalars(attr, &fallback_rows, &mut fb_vals)?;
        }
        let mut fb_idx = 0usize;

        out.reserve(rows.len());
        for &row in rows {
            if let Some(opt_view) = block_views.get(&(row.seg(), row.block_idx())) {
                match opt_view {
                    Some(view) => {
                        let v = view.get_scalar(row.slot())?;
                        out.push((!matches!(v, ScalarValue::Null)).then_some(v));
                    }
                    None => out.push(None),
                }
            } else {
                out.push(fb_vals[fb_idx].take());
                fb_idx += 1;
            }
        }
        Ok(())
    }

    fn read_numeric_batch(
        &self,
        attr: AttrId,
        rows: &[RowId],
        out: &mut Vec<Option<f64>>,
    ) -> Result<()> {
        out.clear();
        if rows.is_empty() {
            return Ok(());
        }
        let (block_keys, fallback_rows) = self.split_rows_for_attr(attr, rows);
        if block_keys.is_empty() {
            return self.row_fallback.read_numeric_batch(attr, rows, out);
        }

        let block_slices = self
            .storage
            .multi_get_column_blocks_pinned(&block_keys, true)?;
        let mut block_views: HashMap<(SegmentId, u32), Option<ColumnBlockView<'_>>> =
            HashMap::with_capacity(block_keys.len());
        for (&(seg, _, block), opt_slice) in block_keys.iter().zip(block_slices.iter()) {
            let parsed = opt_slice
                .as_ref()
                .map(|s| ColumnBlockView::parse(s))
                .transpose()?;
            block_views.insert((seg, block), parsed);
        }

        let mut fb_vals: Vec<Option<f64>> = Vec::new();
        if !fallback_rows.is_empty() {
            self.row_fallback
                .read_numeric_batch(attr, &fallback_rows, &mut fb_vals)?;
        }
        let mut fb_idx = 0usize;

        out.reserve(rows.len());
        for &row in rows {
            if let Some(opt_view) = block_views.get(&(row.seg(), row.block_idx())) {
                match opt_view {
                    Some(view) => out.push(view.get_numeric_f64(row.slot())),
                    None => out.push(None),
                }
            } else {
                out.push(fb_vals[fb_idx]);
                fb_idx += 1;
            }
        }
        Ok(())
    }

    fn read_rows(
        &self,
        rows: &[RowId],
        visit: &mut dyn FnMut(RowId, RowView<'_>) -> Result<()>,
    ) -> Result<()> {
        self.row_fallback.read_rows(rows, visit)
    }
}
