//! Per-segment value-major attribute index (`cf_attr_postings`, `default[0x06]`, `default[0x12]`).

use izzydb_core::config::RowId;
use izzydb_core::error::Result;
use izzydb_core::schema::CollectionSchema;
use izzydb_storage::codec::{
    RowView, TAG_FLOAT, TAG_INT, TAG_STRING, TAG_UINT, decode_row_entries,
    extract_index_keys_from_entries, make_posting_key,
};
use izzydb_storage::TxContext;

use crate::segments::RowIndexHook;

/// Per-segment inverted attribute index hook.
///
/// On insert: writes segment-scoped postings (`cf_attr_postings`), increments live counts
/// (`default[0x06]`), and widens per-segment zone maps (`default[0x12]`).
/// On delete: decrements live counts (`default[0x06]`) only; postings and zones stay immutable
/// until segment compaction drops dead segments.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttributeIndex;

impl RowIndexHook for AttributeIndex {
    fn on_insert(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        row: RowId,
        view: &RowView<'_>,
        _vector: Option<&[f32]>,
        _prev_row: Option<RowId>,
    ) -> Result<()> {
        let entries = decode_row_entries(view.raw_bytes())?;
        let keys = extract_index_keys_from_entries(schema, &entries)?;
        let seg = row.seg();
        let local = row.local();

        for attr_key in &keys {
            let attr_id = u16::from_be_bytes([attr_key[0], attr_key[1]]);
            let tag = attr_key[2];
            tx.add_posting(make_posting_key(attr_key, seg), local);
            tx.adjust_attr_count(attr_key, 1);
            if attr_key.len() > 3
                && matches!(tag, TAG_INT | TAG_UINT | TAG_FLOAT | TAG_STRING)
                && schema
                    .get_by_id(attr_id)
                    .is_some_and(|e| e.attr_type.is_ordered())
            {
                tx.widen_zone(seg, attr_id, tag, &attr_key[3..])?;
            }
        }
        Ok(())
    }

    fn on_delete(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        _row: RowId,
        view: &RowView<'_>,
        _had_vector: bool,
    ) -> Result<()> {
        let entries = decode_row_entries(view.raw_bytes())?;
        let keys = extract_index_keys_from_entries(schema, &entries)?;
        for attr_key in &keys {
            tx.adjust_attr_count(attr_key, -1);
        }
        Ok(())
    }
}
