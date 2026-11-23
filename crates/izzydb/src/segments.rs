use izzydb_core::config::{DocId, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::CollectionSchema;
use izzydb_storage::codec::{
    RowView, SEG_FLAG_HAS_VECTORS, SegmentMeta, SegmentState,
};
use izzydb_storage::{Storage, TxContext};

/// Index maintenance hook invoked inside the same `TxContext` as row mutations.
pub(crate) trait RowIndexHook {
    fn on_insert(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        row: RowId,
        view: &RowView<'_>,
        vector: Option<&[f32]>,
        prev_row: Option<RowId>,
    ) -> Result<()>;

    fn on_delete(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        row: RowId,
        view: &RowView<'_>,
        had_vector: bool,
    ) -> Result<()>;

    fn on_seal(&self, _tx: &mut TxContext<'_>, _seg: SegmentId) -> Result<()> {
        Ok(())
    }
}

/// Write-path manager for append-only segments, id map, and tombstones.
pub(crate) struct SegmentStore {
    pub(crate) seal_rows: u32,
    pub(crate) defer_column_build: bool,
}

impl SegmentStore {
    pub(crate) fn new(seal_rows: u32, defer_column_build: bool) -> Self {
        Self {
            seal_rows,
            defer_column_build,
        }
    }

    /// Allocate the next [`RowId`] in the open segment (creating one if none / sealed).
    pub(crate) fn alloc_row(&self, tx: &mut TxContext<'_>) -> Result<RowId> {
        let open_seg = match tx.manifest().open_segment {
            Some(seg) => seg,
            None => {
                let seq = tx.storage().latest_sequence_number();
                let seg = tx.manifest().next_segment_id;
                tx.manifest_mut().next_segment_id = seg
                    .checked_add(1)
                    .ok_or_else(|| TreeError::Storage("SegmentId u32 overflow".into()))?;
                tx.manifest_mut().open_segment = Some(seg);
                tx.put_seg_meta(seg, SegmentMeta::new_open(seq));
                seg
            }
        };

        let mut seg_meta = tx.get_seg_meta(open_seg).ok_or_else(|| {
            TreeError::CorruptedData(format!("missing SegmentMeta for open segment {open_seg}"))
        })?;
        if seg_meta.state != SegmentState::Open {
            return Err(TreeError::InvariantViolation(format!(
                "open_segment {open_seg} has state {:?}",
                seg_meta.state
            )));
        }
        let local = seg_meta.row_count;
        seg_meta.row_count = local
            .checked_add(1)
            .ok_or_else(|| TreeError::Storage("LocalId overflow".into()))?;
        tx.put_seg_meta(open_seg, seg_meta);
        Ok(RowId::new(open_seg, local))
    }

    /// Persist a newly allocated row's bytes, optional vector, `cf_id_map`, and `cf_row_doc_ids`.
    pub(crate) fn write_row(
        &self,
        tx: &mut TxContext<'_>,
        row: RowId,
        doc_id: DocId,
        row_bytes: Vec<u8>,
        vector: Option<&[f32]>,
    ) -> Result<()> {
        tx.put_row_bytes(row, row_bytes);
        if let Some(vec) = vector {
            tx.put_vector(row, vec.to_vec());
            if let Some(mut seg_meta) = tx.get_seg_meta(row.seg())
                && (seg_meta.flags & SEG_FLAG_HAS_VECTORS) == 0
            {
                seg_meta.flags |= SEG_FLAG_HAS_VECTORS;
                tx.put_seg_meta(row.seg(), seg_meta);
            }
        }
        tx.put_id_map(doc_id, row);
        tx.set_row_doc_id(row, doc_id)?;
        let m = tx.manifest_mut();
        m.live_rows += 1;
        m.total_rows += 1;
        Ok(())
    }

    /// Tombstone an existing live row (keeping `cf_rows` / `cf_vectors` until compaction).
    pub(crate) fn tombstone(&self, tx: &mut TxContext<'_>, row: RowId) -> Result<()> {
        tx.tombstone_row(row)?;
        tx.manifest_mut().live_rows -= 1;
        Ok(())
    }

    /// Seal the open segment if its `row_count >= self.seal_rows`.
    pub(crate) fn maybe_seal(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        hooks: &[&dyn RowIndexHook],
    ) -> Result<Option<SegmentId>> {
        let Some(seg) = tx.manifest().open_segment else {
            return Ok(None);
        };
        let Some(meta) = tx.get_seg_meta(seg) else {
            return Err(TreeError::CorruptedData(format!(
                "missing SegmentMeta for open segment {seg}"
            )));
        };
        if meta.row_count >= self.seal_rows {
            self.seal(tx, schema, seg, hooks)?;
            return Ok(Some(seg));
        }
        Ok(None)
    }

    /// Unconditionally seal `seg`, build columns unless `defer_column_build`, and invoke `on_seal` hooks.
    pub(crate) fn seal(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        seg: SegmentId,
        hooks: &[&dyn RowIndexHook],
    ) -> Result<()> {
        let mut meta = tx.get_seg_meta(seg).ok_or_else(|| {
            TreeError::CorruptedData(format!("missing SegmentMeta for segment {seg}"))
        })?;
        if meta.state != SegmentState::Open {
            return Err(TreeError::InvariantViolation(format!(
                "cannot seal non-open segment {seg} ({:?})",
                meta.state
            )));
        }
        meta.state = SegmentState::Sealed;
        meta.sealed_seq = tx.storage().latest_sequence_number();
        tx.put_seg_meta(seg, meta);
        tx.manifest_mut().open_segment = None;
        if !self.defer_column_build {
            crate::columns::build_segment_columns_in_tx(tx, schema, seg)?;
        }
        for hook in hooks {
            hook.on_seal(tx, seg)?;
        }
        Ok(())
    }
}

/// Single translator from internal [`RowId`]s to external [`DocId`]s via `doc_id_block_cache`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RowResolver;

#[allow(dead_code)]
impl RowResolver {
    pub(crate) fn resolve(storage: &Storage, rows: &[RowId]) -> Result<Vec<Option<DocId>>> {
        storage.resolve_doc_ids(rows)
    }

    pub(crate) fn resolve_all(storage: &Storage, rows: &[RowId]) -> Result<Vec<DocId>> {
        let opts = Self::resolve(storage, rows)?;
        let mut out = Vec::with_capacity(opts.len());
        for (&row, opt) in rows.iter().zip(opts) {
            let doc_id = opt.ok_or_else(|| {
                TreeError::InvariantViolation(format!(
                    "missing cf_row_doc_ids entry for {row:?}"
                ))
            })?;
            out.push(doc_id);
        }
        Ok(out)
    }
}
