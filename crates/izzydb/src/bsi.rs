//! Seal-time construction of per-segment bit-sliced indexes.
//!
//! `cf_bsi` stores one presence bitmap and up to eight value planes per
//! `(segment, attribute, 65,536-row chunk)`. The index is built when a segment is sealed
//! (online writes) or emitted directly by the bulk importer, so it never adds per-insert
//! read-modify-write amplification.

use izzydb_core::config::{AttrId, SegmentId};
use izzydb_core::error::Result;
use izzydb_core::schema::{AttributeType, CollectionSchema, ColumnStorage};
use izzydb_storage::bsi::{BsiChunkBuilder, BSI_CHUNK_ROWS};
use izzydb_storage::codec::RowView;
use izzydb_storage::TxContext;

/// Ordered numeric inline attributes eligible for a bit-sliced index.
///
/// ```text
/// BsiSpec { attr_id: 4, attr_type: AttributeType::Float }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BsiSpec {
    /// Schema attribute id.
    pub attr_id: AttrId,
    /// Declared numeric type.
    pub attr_type: AttributeType,
}

/// Return the ordered numeric inline attributes of `schema` that get a BSI.
pub(crate) fn bsi_specs(schema: &CollectionSchema) -> Vec<BsiSpec> {
    let mut specs: Vec<BsiSpec> = schema
        .entries()
        .iter()
        .filter(|e| {
            e.storage == ColumnStorage::Inline
                && e.filterable
                && matches!(
                    e.attr_type,
                    AttributeType::Int | AttributeType::Uint | AttributeType::Float
                )
        })
        .map(|e| BsiSpec {
            attr_id: e.id,
            attr_type: e.attr_type,
        })
        .collect();
    specs.sort_unstable_by_key(|s| s.attr_id);
    specs
}

/// Build `cf_bsi` for all eligible attributes of `seg` from live rows visible inside `tx`.
///
/// `seg` must already be sealed; rows are read in ascending `RowId` order so chunk boundaries
/// and plane keys are emitted deterministically.
pub(crate) fn build_segment_bsi_in_tx(
    tx: &mut TxContext<'_>,
    schema: &CollectionSchema,
    seg: SegmentId,
) -> Result<usize> {
    let specs = bsi_specs(schema);
    if specs.is_empty() {
        return Ok(0);
    }

    let mut chunk = u32::MAX;
    let mut builders: Vec<BsiChunkBuilder> = Vec::with_capacity(specs.len());
    let mut pending: Vec<(AttrId, u32, u8, Vec<u8>)> = Vec::new();

    let flush = |chunk: u32,
                 builders: &mut Vec<BsiChunkBuilder>,
                 pending: &mut Vec<(AttrId, u32, u8, Vec<u8>)>| {
        if chunk == u32::MAX {
            return;
        }
        for (spec, builder) in specs.iter().zip(builders.iter_mut()) {
            for (plane, bytes) in builder.encode_planes() {
                pending.push((spec.attr_id, chunk, plane, bytes));
            }
            *builder = BsiChunkBuilder::new();
        }
    };

    tx.visit_segment_live_rows_in_tx(seg, |row, bytes| {
        let row_chunk = row.local() / BSI_CHUNK_ROWS;
        if row_chunk != chunk {
            flush(chunk, &mut builders, &mut pending);
            chunk = row_chunk;
            builders.clear();
            builders.resize_with(specs.len(), BsiChunkBuilder::new);
        }
        let view = RowView::parse(bytes)?;
        let in_chunk = row.local() % BSI_CHUNK_ROWS;
        for (spec, builder) in specs.iter().zip(builders.iter_mut()) {
            let value = view.get_scalar(spec.attr_id)?;
            if let Some(ordered) = izzydb_storage::bsi::ordered_value(&value) {
                builder.insert(in_chunk, ordered);
            }
        }
        Ok(())
    })?;

    flush(chunk, &mut builders, &mut pending);
    let mut emitted = 0usize;
    for (attr_id, chunk, plane, bytes) in pending {
        tx.put_bsi_plane_bytes(seg, attr_id, chunk, plane, bytes);
        emitted += 1;
    }
    Ok(emitted)
}
