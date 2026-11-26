//! Seal-time and maintenance transposition of inline `cf_rows` into 256-row `cf_columns` blocks.

use std::sync::atomic::{AtomicBool, Ordering};

use izzydb_core::config::{AttrId, ROW_BLOCK_SIZE, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{AttributeType, CollectionSchema, ColumnStorage};
use izzydb_storage::TxContext;
use izzydb_storage::codec::{RowView, SEG_FLAG_HAS_COLUMNS, SegmentState};
use izzydb_storage::column::ColumnBlockBuilder;

static FORCE_ROW_FALLBACK: AtomicBool = AtomicBool::new(false);

/// Test and benchmark switch to force inline column readers to fall back to `cf_rows`.
#[doc(hidden)]
pub fn set_force_column_row_fallback(force: bool) {
    FORCE_ROW_FALLBACK.store(force, Ordering::Relaxed);
}

#[inline]
pub(crate) fn force_row_fallback_enabled() -> bool {
    FORCE_ROW_FALLBACK.load(Ordering::Relaxed)
}

/// Columnar attributes eligible for `cf_columns` within an inline segment.
pub(crate) fn columnar_inline_attrs(schema: &CollectionSchema) -> Vec<(AttrId, AttributeType)> {
    schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Inline && e.is_columnar())
        .map(|e| (e.id, e.attr_type))
        .collect()
}

/// Build `cf_columns` blocks for all live rows in `seg` inside `tx`, and set
/// `SegmentMeta.flags |= SEG_FLAG_HAS_COLUMNS`. Idempotent: if `SEG_FLAG_HAS_COLUMNS` is already
/// set, returns `Ok(0)` without rewriting.
pub(crate) fn build_segment_columns_in_tx(
    tx: &mut TxContext<'_>,
    schema: &CollectionSchema,
    seg: SegmentId,
) -> Result<usize> {
    let mut meta = tx.get_seg_meta(seg).ok_or_else(|| {
        TreeError::CorruptedData(format!("build_segment_columns: missing SegmentMeta for {seg}"))
    })?;
    if meta.state != SegmentState::Sealed {
        return Err(TreeError::InvariantViolation(format!(
            "cannot build columns for non-sealed segment {seg} ({:?})",
            meta.state
        )));
    }
    if (meta.flags & SEG_FLAG_HAS_COLUMNS) != 0 {
        return Ok(0);
    }

    let col_attrs = columnar_inline_attrs(schema);
    let mut blocks_written = 0usize;

    if !col_attrs.is_empty() {
        let attr_idx_map = hashbrown_or_vec::AttrSlotMap::new(&col_attrs);

        let mut cur_block: Option<u32> = None;
        let mut builders: Vec<ColumnBlockBuilder> = Vec::new();
        let mut staged: Vec<(AttrId, u32, Vec<u8>)> = Vec::new();

        let flush_block = |block: u32,
                           builders: Vec<ColumnBlockBuilder>,
                           staged: &mut Vec<(AttrId, u32, Vec<u8>)>| {
            for (&(attr_id, _), b) in col_attrs.iter().zip(builders) {
                if b.present_count() > 0 {
                    staged.push((attr_id, block, b.encode()));
                }
            }
        };

        tx.visit_segment_live_rows_in_tx(seg, |row, bytes| {
            let block = row.block_idx();
            let slot = row.slot();
            if cur_block != Some(block) {
                if let Some(prev) = cur_block.replace(block) {
                    let prev_builders = std::mem::take(&mut builders);
                    flush_block(prev, prev_builders, &mut staged);
                }
                builders = col_attrs
                    .iter()
                    .map(|&(_, ty)| ColumnBlockBuilder::new(ty))
                    .collect();
            }
            let view = RowView::parse(bytes)?;
            for i in 0..view.len() {
                let (attr_id, _, _, _) = view.dir_entry(i);
                if let Some(pos) = attr_idx_map.get(attr_id)
                    && let Some(field) = view.get_raw_field(attr_id)?
                {
                    builders[pos].set_row_field(slot, field)?;
                }
            }
            Ok(())
        })?;

        if let Some(last_block) = cur_block.take() {
            flush_block(last_block, builders, &mut staged);
        }

        blocks_written = staged.len();
        for (attr_id, block, bytes) in staged {
            tx.put_column_block(seg, attr_id, block, bytes);
        }
    }

    let _ = ROW_BLOCK_SIZE;
    meta.flags |= SEG_FLAG_HAS_COLUMNS;
    tx.put_seg_meta(seg, meta);
    Ok(blocks_written)
}

mod hashbrown_or_vec {
    use izzydb_core::config::AttrId;
    use izzydb_core::schema::AttributeType;

    pub(super) struct AttrSlotMap {
        entries: Vec<(AttrId, usize)>,
    }

    impl AttrSlotMap {
        pub(super) fn new(col_attrs: &[(AttrId, AttributeType)]) -> Self {
            let mut entries: Vec<(AttrId, usize)> = col_attrs
                .iter()
                .enumerate()
                .map(|(i, &(id, _))| (id, i))
                .collect();
            entries.sort_unstable_by_key(|&(id, _)| id);
            Self { entries }
        }

        #[inline]
        pub(super) fn get(&self, id: AttrId) -> Option<usize> {
            let idx = self.entries.binary_search_by_key(&id, |&(k, _)| k).ok()?;
            Some(self.entries[idx].1)
        }
    }
}
