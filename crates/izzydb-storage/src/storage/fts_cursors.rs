use rocksdb::{DB, DBPinnableSlice, DBRawIteratorWithThreadMode, ReadOptions};

use izzydb_core::config::{RowBlock, SkipChunk, TermId};
use izzydb_core::error::{Result, TreeError};
use izzydb_fts::{PostingBlockHeader, SkipBlockLayout, SkipBlockView};

use crate::codec::{encode_fts_term_id_prefix, fts_block_key_coord, fts_chunk_key_coord};

use super::types::{DocLenBlockTable, map_db_err};
use super::{CF_FTS_POSTINGS, CF_FTS_SKIPS};

const SEQUENTIAL_POSTING_STEP_LIMIT: usize = 4;

enum CursorSource<'a> {
    Inline {
        bytes: Vec<u8>,
        exhausted: bool,
    },
    OutOfLine(Box<OutOfLineState<'a>>),
}

struct OutOfLineState<'a> {
    skip_iter: Option<DBRawIteratorWithThreadMode<'a, DB>>,
    chunk_layout: Option<SkipBlockLayout>,
    chunk: SkipChunk,
    entry_idx: usize,
    entry_count: usize,
    posting_iter: Option<DBRawIteratorWithThreadMode<'a, DB>>,
    posting_pos: Option<RowBlock>,
}

/// Cursor over a term's Block-Max skip chunks (`cf_fts_skips`, keyed `term_id || seg || chunk`)
/// or inline posting block, iterating in global [`RowBlock`] order across all segments.
pub struct RocksSkipCursor<'a> {
    db: &'a DB,
    term_id: TermId,
    source: CursorSource<'a>,
    current_block: Option<RowBlock>,
    current_bounds: (u32, f32),
}

impl<'a> RocksSkipCursor<'a> {
    pub(super) fn from_inline(
        db: &'a DB,
        term_id: TermId,
        block_id: RowBlock,
        bytes: Vec<u8>,
    ) -> Result<Self> {
        let hdr = PostingBlockHeader::parse(&bytes)?;
        let bounds = (hdr.max_tf, hdr.min_dl_over_tf);
        Ok(Self {
            db,
            term_id,
            source: CursorSource::Inline {
                bytes,
                exhausted: false,
            },
            current_block: Some(block_id),
            current_bounds: bounds,
        })
    }

    pub(super) fn open_out_of_line(db: &'a DB, term_id: TermId) -> Result<Self> {
        let mut cursor = Self {
            db,
            term_id,
            source: CursorSource::OutOfLine(Box::new(OutOfLineState {
                skip_iter: None,
                chunk_layout: None,
                chunk: SkipChunk::default(),
                entry_idx: 0,
                entry_count: 0,
                posting_iter: None,
                posting_pos: None,
            })),
            current_block: None,
            current_bounds: (0, f32::INFINITY),
        };
        cursor.seek_skip_iter_prefix()?;
        Ok(cursor)
    }

    #[inline]
    pub fn current_block_id(&self) -> Option<RowBlock> {
        self.current_block
    }

    #[inline]
    pub fn current_block_bounds(&self) -> (u32, f32) {
        self.current_bounds
    }

    #[inline]
    pub fn inline_posting_bytes(&self) -> Option<&[u8]> {
        match &self.source {
            CursorSource::Inline { bytes, exhausted } if !*exhausted => Some(bytes.as_slice()),
            _ => None,
        }
    }

    /// Return the encoded `PackedPostingBlock` bytes for `current_block`, initializing or advancing
    /// a lazy sequential iterator over `cf_fts_postings` only when called.
    pub fn current_posting_bytes(&mut self) -> Result<Option<&[u8]>> {
        let Some(target) = self.current_block else {
            return Ok(None);
        };
        let term_id = self.term_id;
        let db = self.db;
        match &mut self.source {
            CursorSource::Inline { bytes, exhausted } => {
                Ok((!*exhausted).then_some(bytes.as_slice()))
            }
            CursorSource::OutOfLine(state) => {
                let prefix = encode_fts_term_id_prefix(term_id);
                if let Some(it) = state.posting_iter.as_mut() {
                    advance_posting_iter_to(it, &mut state.posting_pos, term_id, &prefix, target);
                } else {
                    let cf = db.cf_handle(CF_FTS_POSTINGS).ok_or_else(|| {
                        TreeError::CorruptedData(format!(
                            "missing column family {CF_FTS_POSTINGS}"
                        ))
                    })?;
                    let mut ro = ReadOptions::default();
                    ro.set_prefix_same_as_start(true);
                    let mut it = db.raw_iterator_cf_opt(cf, ro);
                    it.seek(fts_block_key_coord(term_id, target));
                    state.posting_pos = iter_block_coord(&it, &prefix);
                    state.posting_iter = Some(it);
                }
                let iter = state
                    .posting_iter
                    .as_ref()
                    .expect("posting_iter initialized");
                if state.posting_pos == Some(target) {
                    Ok(iter.value())
                } else {
                    iter.status().map_err(map_db_err)?;
                    Ok(None)
                }
            }
        }
    }

    #[inline]
    pub fn next_block(&mut self) -> Result<()> {
        if self.current_block.is_none() {
            return Ok(());
        }
        match &mut self.source {
            CursorSource::Inline { exhausted, .. } => {
                *exhausted = true;
                self.current_block = None;
                Ok(())
            }
            CursorSource::OutOfLine(state) => {
                let next_idx = state.entry_idx + 1;
                if next_idx < state.entry_count {
                    let view = chunk_view(state.chunk_layout, state.skip_iter.as_ref())
                        .expect("active skip chunk must have view");
                    let rel = view.rel_block_id(next_idx);
                    self.current_bounds = (view.max_tf(next_idx), view.min_dl_over_tf(next_idx));
                    self.current_block = Some(state.chunk.block(rel));
                    state.entry_idx = next_idx;
                    return Ok(());
                }
                state.skip_iter.as_mut().unwrap().next();
                self.load_current_chunk(None)
            }
        }
    }

    #[inline]
    pub fn seek_block(&mut self, target: RowBlock) -> Result<()> {
        let Some(cur) = self.current_block else {
            return Ok(());
        };
        if cur >= target {
            return Ok(());
        }
        match &mut self.source {
            CursorSource::Inline { exhausted, .. } => {
                *exhausted = true;
                self.current_block = None;
                Ok(())
            }
            CursorSource::OutOfLine(state) => {
                let target_chunk = target.chunk();
                let target_rel = target.rel_block();
                if target_chunk == state.chunk {
                    let view = chunk_view(state.chunk_layout, state.skip_iter.as_ref())
                        .expect("active skip chunk must have view");
                    let rel_slice = &view.rel_block_ids[state.entry_idx..];
                    let pos = rel_slice.partition_point(|&b| b < target_rel);
                    if pos < rel_slice.len() {
                        let i = state.entry_idx + pos;
                        let rel = view.rel_block_id(i);
                        self.current_bounds = (view.max_tf(i), view.min_dl_over_tf(i));
                        self.current_block = Some(state.chunk.block(rel));
                        state.entry_idx = i;
                        return Ok(());
                    }
                    state.skip_iter.as_mut().unwrap().next();
                    return self.load_current_chunk(Some((target_chunk, target_rel)));
                }

                if target_chunk.seg() == state.chunk.seg()
                    && target_chunk.chunk() == state.chunk.chunk() + 1
                {
                    state.skip_iter.as_mut().unwrap().next();
                    return self.load_current_chunk(Some((target_chunk, target_rel)));
                }
                self.seek_skip_iter(target_chunk, target_rel)
            }
        }
    }

    fn seek_skip_iter_prefix(&mut self) -> Result<()> {
        let db = self.db;
        let term_id = self.term_id;
        let CursorSource::OutOfLine(state) = &mut self.source else {
            return Ok(());
        };
        state.chunk_layout = None;
        let prefix = encode_fts_term_id_prefix(term_id);
        let cf = db.cf_handle(CF_FTS_SKIPS).ok_or_else(|| {
            TreeError::CorruptedData(format!("missing column family {CF_FTS_SKIPS}"))
        })?;
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let mut it = db.raw_iterator_cf_opt(cf, ro);
        it.seek(prefix);
        state.skip_iter = Some(it);
        self.load_current_chunk(None)
    }

    fn seek_skip_iter(&mut self, seek_chunk: SkipChunk, min_rel: u8) -> Result<()> {
        let db = self.db;
        let term_id = self.term_id;
        let CursorSource::OutOfLine(state) = &mut self.source else {
            return Ok(());
        };
        state.chunk_layout = None;
        let key = fts_chunk_key_coord(term_id, seek_chunk);
        match &mut state.skip_iter {
            Some(it) => it.seek(key),
            None => {
                let cf = db.cf_handle(CF_FTS_SKIPS).ok_or_else(|| {
                    TreeError::CorruptedData(format!("missing column family {CF_FTS_SKIPS}"))
                })?;
                let mut ro = ReadOptions::default();
                ro.set_prefix_same_as_start(true);
                let mut it = db.raw_iterator_cf_opt(cf, ro);
                it.seek(key);
                state.skip_iter = Some(it);
            }
        }
        self.load_current_chunk(Some((seek_chunk, min_rel)))
    }

    fn load_current_chunk(&mut self, min_pos: Option<(SkipChunk, u8)>) -> Result<()> {
        let term_id = self.term_id;
        let CursorSource::OutOfLine(state) = &mut self.source else {
            return Ok(());
        };
        let Some(iter) = &mut state.skip_iter else {
            self.current_block = None;
            return Ok(());
        };
        let prefix = encode_fts_term_id_prefix(term_id);
        loop {
            let Some(k) = iter.key() else {
                iter.status().map_err(map_db_err)?;
                self.current_block = None;
                state.chunk_layout = None;
                return Ok(());
            };
            if k.len() != 12 || k[0..4] != prefix {
                self.current_block = None;
                state.chunk_layout = None;
                return Ok(());
            }
            let seg = u32::from_be_bytes(k[4..8].try_into().unwrap());
            let cid = u32::from_be_bytes(k[8..12].try_into().unwrap());
            let chunk = SkipChunk::new(seg, cid);
            if let Some((target_chunk, _)) = min_pos
                && chunk < target_chunk
            {
                iter.next();
                continue;
            }
            let Some(val) = iter.value() else {
                iter.next();
                continue;
            };
            let layout = SkipBlockLayout::parse(val)?;
            let view = layout.view(val);
            let idx = match min_pos {
                Some((target_chunk, min_rel)) if chunk == target_chunk && min_rel > 0 => {
                    view.find_ge(min_rel)
                }
                _ => Some(0),
            };
            if let Some(i) = idx {
                state.chunk = chunk;
                state.entry_idx = i;
                state.entry_count = layout.count;
                let rel = view.rel_block_id(i);
                let bounds = (view.max_tf(i), view.min_dl_over_tf(i));
                state.chunk_layout = Some(layout);
                self.current_block = Some(chunk.block(rel));
                self.current_bounds = bounds;
                return Ok(());
            }
            iter.next();
        }
    }
}

#[inline]
fn chunk_view<'a>(
    layout: Option<SkipBlockLayout>,
    skip_iter: Option<&'a DBRawIteratorWithThreadMode<'a, DB>>,
) -> Option<SkipBlockView<'a>> {
    let layout = layout?;
    let bytes: &[u8] = skip_iter?.value()?;
    Some(layout.view(bytes))
}

#[inline]
fn iter_block_coord(
    iter: &DBRawIteratorWithThreadMode<'_, DB>,
    prefix: &[u8; 4],
) -> Option<RowBlock> {
    match iter.key() {
        Some(k) if k.len() == 12 && k[0..4] == *prefix => {
            let seg = u32::from_be_bytes(k[4..8].try_into().unwrap());
            let block = u32::from_be_bytes(k[8..12].try_into().unwrap());
            Some(RowBlock::new(seg, block))
        }
        _ => None,
    }
}

fn advance_posting_iter_to(
    iter: &mut DBRawIteratorWithThreadMode<'_, DB>,
    posting_pos: &mut Option<RowBlock>,
    term_id: TermId,
    prefix: &[u8; 4],
    target: RowBlock,
) {
    if !posting_pos.is_some_and(|b| b < target) {
        return;
    }
    let mut cur_bid = *posting_pos;
    let mut steps = 0usize;
    while let Some(b) = cur_bid
        && b < target
        && steps < SEQUENTIAL_POSTING_STEP_LIMIT
    {
        iter.next();
        steps += 1;
        cur_bid = iter_block_coord(iter, prefix);
    }
    if cur_bid.is_some_and(|b| b < target) {
        iter.seek(fts_block_key_coord(term_id, target));
        cur_bid = iter_block_coord(iter, prefix);
    }
    *posting_pos = cur_bid;
}

/// Cached per-attribute cursor over `default[0x05]` document-length blocks.
pub struct RocksDocLenCursor {
    pub(super) blocks: DocLenBlockTable,
}

impl RocksDocLenCursor {
    #[inline]
    pub fn seek_block_bytes(&mut self, block_id: RowBlock) -> Result<Option<&[u8]>> {
        Ok(self.blocks.get(&block_id).map(|b| b.as_ref()))
    }
}

// Keep DBPinnableSlice import used for future chunk-0 fast paths if desired.
#[allow(dead_code)]
type _UnusedPin<'a> = DBPinnableSlice<'a>;
