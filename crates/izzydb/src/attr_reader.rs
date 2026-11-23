//! Batched, `RowId`-ordered attribute accessor trait and v3 row-store reader.
//!
//! Phase 6 implements [`RowStoreReader`] (`cf_rows` multi-get). Phase 7 swaps sealed-segment
//! scalar reads to [`ColumnReader`] behind the same [`AttrReader`] interface, and Phase 8 plugs
//! in detached column reads without touching callers in `rank_eval.rs`, `order_by.rs`, or `facet.rs`.

use izzydb_core::config::{AttrId, RowId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::ScalarValue;
use izzydb_storage::Storage;
use izzydb_storage::codec::RowView;

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
    #[allow(dead_code)]
    fn read_numeric_batch(&self, attr: AttrId, rows: &[RowId], out: &mut Vec<Option<f64>>) -> Result<()>;

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
