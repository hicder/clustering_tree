//! `RowIndexHook` write path and posting upserts.



use izzydb_core::config::{RowBlock, RowId, TermId};
use izzydb_core::error::Result;
use izzydb_core::schema::{CollectionSchema, ColumnStorage};
use izzydb_fts::{
    DocLenBlock, FtsAnalyzer, FtsChunk, FtsChunkView, FtsMeta, INLINE_TERM_MAX_DF,
    PackedPositionBlock, PackedPostingBlock, PostingBlockHeader, TermRecord,
};
use izzydb_storage::codec::{
    RowView, encode_fts_term_key,
};
use izzydb_storage::TxContext;

use crate::segments::RowIndexHook;

use super::*;

impl RowIndexHook for FtsIndex {
    fn on_insert(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        row: RowId,
        view: &RowView<'_>,
        _vector: Option<&[f32]>,
        _prev_row: Option<RowId>,
    ) -> Result<()> {
        if !schema.has_fts() {
            return Ok(());
        }

        let mut meta = match tx.get_fts_meta_bytes()? {
            Some(bytes) => FtsMeta::decode(&bytes)?,
            None => FtsMeta::default(),
        };
        let block_id = row.block();
        let slot = row.slot();
        let mut buf = self.acquire_buffer();
        let mut modified = false;

        for entry in schema.fts_entries() {
            if entry.storage != ColumnStorage::Inline {
                continue;
            }
            let Some(val) = view.decode_field(entry.id)? else {
                continue;
            };
            let fts_cfg = entry.full_text_search.as_ref().unwrap();
            let analyzer = FtsAnalyzer::new(fts_cfg.clone());
            let Some(analyzed) = analyzer.analyze_attribute_value(&val, &mut buf)? else {
                continue;
            };
            modified = true;
            {
                let stats = meta.per_attr_stats.entry(entry.id).or_default();
                stats.doc_count = stats.doc_count.saturating_add(1);
                stats.total_tokens = stats.total_tokens.saturating_add(analyzed.doc_len as u64);
            }

            let mut dl_blk = match tx.get_fts_doc_len_bytes(entry.id, block_id)? {
                Some(bytes) => DocLenBlock::decode(&bytes)?,
                None => DocLenBlock::default(),
            };
            dl_blk.set(slot, analyzed.doc_len);
            tx.put_fts_doc_len_bytes(entry.id, block_id, dl_blk.encode());

            for (tok, term) in &analyzed.terms {
                let term_key = encode_fts_term_key(entry.id, tok);
                insert_new_term_posting(
                    tx,
                    &mut meta,
                    FtsDocSlot {
                        block_id,
                        slot,
                        index_positions: fts_cfg.index_positions,
                    },
                    term_key,
                    term.tf,
                    analyzed.doc_len,
                    &term.positions,
                    &dl_blk,
                )?;
            }
        }

        if modified {
            tx.put_fts_meta_bytes(meta.encode());
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
        if !schema.has_fts() {
            return Ok(());
        }
        let Some(meta_bytes) = tx.get_fts_meta_bytes()? else {
            return Ok(());
        };
        let mut meta = FtsMeta::decode(&meta_bytes)?;
        let mut buf = self.acquire_buffer();
        let mut modified = false;

        for entry in schema.fts_entries() {
            if entry.storage != ColumnStorage::Inline {
                continue;
            }
            let Some(val) = view.decode_field(entry.id)? else {
                continue;
            };
            let fts_cfg = entry.full_text_search.as_ref().unwrap();
            let analyzer = FtsAnalyzer::new(fts_cfg.clone());
            let Some(analyzed) = analyzer.analyze_attribute_value(&val, &mut buf)? else {
                continue;
            };
            modified = true;
            if let Some(stats) = meta.per_attr_stats.get_mut(&entry.id) {
                stats.doc_count = stats.doc_count.saturating_sub(1);
                stats.total_tokens = stats
                    .total_tokens
                    .saturating_sub(analyzed.doc_len as u64);
            }
            for tok in analyzed.terms.keys() {
                let term_key = encode_fts_term_key(entry.id, tok);
                if let Some(tbytes) = tx.get_fts_term_bytes(&term_key)? {
                    let mut rec = TermRecord::decode(&tbytes)?;
                    rec.stats.doc_freq = rec.stats.doc_freq.saturating_sub(1);
                    tx.put_fts_term_bytes(term_key, rec.encode());
                }
            }
        }

        if modified {
            tx.put_fts_meta_bytes(meta.encode());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct FtsDocSlot {
    pub(super) block_id: RowBlock,
    pub(super) slot: u8,
    pub(super) index_positions: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn insert_new_term_posting(
    tx: &mut TxContext<'_>,
    meta: &mut FtsMeta,
    loc: FtsDocSlot,
    term_key: Vec<u8>,
    new_tf: u32,
    new_dl: u32,
    new_positions: &[u32],
    doc_lens: &DocLenBlock,
) -> Result<()> {
    let (mut rec, is_brand_new) = match tx.get_fts_term_bytes(&term_key)? {
        Some(b) => (TermRecord::decode(&b)?, false),
        None => (TermRecord::new(meta.alloc_term_id()?), true),
    };
    rec.stats.doc_freq = rec.stats.doc_freq.saturating_add(1);
    rec.stats.observe(new_tf, new_dl);

    match rec.inline_block.take() {
        None if is_brand_new => {
            let mut blk = PackedPostingBlock::default();
            blk.upsert(loc.slot, new_tf, doc_lens);
            rec.inline_block = Some((loc.block_id, blk.encode()));
        }
        Some((ib_id, ib_bytes)) => {
            let hdr = PostingBlockHeader::parse(&ib_bytes)?;
            if ib_id == loc.block_id && (hdr.count as usize) < INLINE_TERM_MAX_DF {
                let mut blk = PackedPostingBlock::decode(&ib_bytes)?;
                blk.upsert(loc.slot, new_tf, doc_lens);
                rec.inline_block = Some((loc.block_id, blk.encode()));
            } else {
                materialize_inline_block(tx, rec.term_id, ib_id, &ib_bytes)?;
                upsert_posting_and_skip(tx, rec.term_id, loc.block_id, loc.slot, new_tf, doc_lens)?;
            }
        }
        None => {
            upsert_posting_and_skip(tx, rec.term_id, loc.block_id, loc.slot, new_tf, doc_lens)?;
        }
    }
    tx.put_fts_term_bytes(term_key, rec.encode());
    if loc.index_positions {
        upsert_position_block(tx, rec.term_id, loc.block_id, loc.slot, new_positions.to_vec())?;
    }
    Ok(())
}

pub(super) fn materialize_inline_block(
    tx: &mut TxContext<'_>,
    term_id: TermId,
    block_id: RowBlock,
    encoded: &[u8],
) -> Result<()> {
    let pblk = PackedPostingBlock::decode(encoded)?;
    let compact = pblk.encode_compact();
    let chunk = block_id.chunk();
    let existing = tx.get_fts_skip_bytes(term_id, chunk)?;
    let new_chunk = FtsChunk::upsert_block(
        existing.as_deref(),
        block_id.rel_block(),
        pblk.max_tf,
        pblk.min_dl_over_tf,
        &compact,
    )?;
    tx.put_fts_skip_bytes(term_id, chunk, new_chunk);
    Ok(())
}

pub(super) fn upsert_posting_and_skip(
    tx: &mut TxContext<'_>,
    term_id: TermId,
    block_id: RowBlock,
    slot: u8,
    tf: u32,
    doc_lens: &DocLenBlock,
) -> Result<()> {
    let chunk = block_id.chunk();
    let rel = block_id.rel_block();
    let existing = tx.get_fts_skip_bytes(term_id, chunk)?;
    let mut blk = match existing.as_deref() {
        Some(bytes) => {
            let view = FtsChunkView::parse(bytes)?;
            if let Some(idx) = view.find_exact(rel) {
                PackedPostingBlock::decode_compact(
                    view.block_bytes(idx),
                    view.max_tf(idx),
                    view.min_dl_over_tf(idx),
                )?
            } else {
                PackedPostingBlock::default()
            }
        }
        None => PackedPostingBlock::default(),
    };
    blk.upsert(slot, tf, doc_lens);
    let compact = blk.encode_compact();
    let new_chunk = FtsChunk::upsert_block(
        existing.as_deref(),
        rel,
        blk.max_tf,
        blk.min_dl_over_tf,
        &compact,
    )?;
    tx.put_fts_skip_bytes(term_id, chunk, new_chunk);
    Ok(())
}

pub(super) fn upsert_position_block(
    tx: &mut TxContext<'_>,
    term_id: TermId,
    block_id: RowBlock,
    slot: u8,
    positions: Vec<u32>,
) -> Result<()> {
    let mut pblk = match tx.get_fts_position_bytes(term_id, block_id)? {
        Some(bytes) => PackedPositionBlock::decode(&bytes)?,
        None => PackedPositionBlock::default(),
    };
    pblk.upsert(slot, positions);
    tx.put_fts_position_bytes(term_id, block_id, pblk.encode());
    Ok(())
}

// ---------------------------------------------------------------------------
// Token & phrase predicates -> RowSet
// ---------------------------------------------------------------------------
