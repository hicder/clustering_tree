use rocksdb::{DB, DBPinnableSlice, DBRawIteratorWithThreadMode, ReadOptions};

use izzydb_core::config::{
    FtsBlockId, TermId, fts_skip_chunk_id, fts_skip_slot, make_fts_block_id,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_fts::{PostingBlockHeader, SkipBlockLayout, SkipBlockView};

use crate::codec::{encode_fts_term_id_key, encode_fts_term_id_prefix};

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
    skip_pin: Option<DBPinnableSlice<'a>>,
    skip_iter: Option<DBRawIteratorWithThreadMode<'a, DB>>,
    chunk_layout: Option<SkipBlockLayout>,
    chunk_id: u32,
    entry_idx: usize,
    entry_count: usize,
    posting_iter: Option<DBRawIteratorWithThreadMode<'a, DB>>,
    posting_pos: Option<FtsBlockId>,
}

/// Cursor over a term's Block-Max skip chunks (`cf_fts_skips`) or inline posting block,
/// with a batched/pinned chunk-0 fast path (no skip iterator for `docs < 65,536`) and a lazy
/// sequential `cf_fts_postings` iterator opened only when a block survives pruning.
pub struct RocksSkipCursor<'a> {
    db: &'a DB,
    term_id: TermId,
    source: CursorSource<'a>,
    current_block: Option<FtsBlockId>,
    current_bounds: (u32, f32),
}

impl<'a> RocksSkipCursor<'a> {
    pub(super) fn from_inline(
        db: &'a DB,
        term_id: TermId,
        block_id: FtsBlockId,
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

    pub(super) fn from_chunk0_pin(
        db: &'a DB,
        term_id: TermId,
        pin: DBPinnableSlice<'a>,
    ) -> Result<Self> {
        let layout = SkipBlockLayout::parse(pin.as_ref())?;
        let view = layout.view(pin.as_ref());
        let rel = view.rel_block_id(0);
        let bounds = (view.max_tf(0), view.min_dl_over_tf(0));
        let count = layout.count;
        Ok(Self {
            db,
            term_id,
            source: CursorSource::OutOfLine(Box::new(OutOfLineState {
                skip_pin: Some(pin),
                skip_iter: None,
                chunk_layout: Some(layout),
                chunk_id: 0,
                entry_idx: 0,
                entry_count: count,
                posting_iter: None,
                posting_pos: None,
            })),
            current_block: Some(make_fts_block_id(0, rel)),
            current_bounds: bounds,
        })
    }

    pub(super) fn open_out_of_line(db: &'a DB, term_id: TermId) -> Result<Self> {
        let mut cursor = Self {
            db,
            term_id,
            source: CursorSource::OutOfLine(Box::new(OutOfLineState {
                skip_pin: None,
                skip_iter: None,
                chunk_layout: None,
                chunk_id: 0,
                entry_idx: 0,
                entry_count: 0,
                posting_iter: None,
                posting_pos: None,
            })),
            current_block: None,
            current_bounds: (0, f32::INFINITY),
        };
        cursor.seek_skip_iter(0, 0)?;
        Ok(cursor)
    }

    #[inline]
    pub fn current_block_id(&self) -> Option<FtsBlockId> {
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
                    it.seek(encode_fts_term_id_key(term_id, target));
                    state.posting_pos = iter_block_id(&it, &prefix);
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
                    let view = chunk_view(
                        state.chunk_layout,
                        state.skip_pin.as_ref(),
                        state.skip_iter.as_ref(),
                    )
                    .expect("active skip chunk must have view");
                    let rel = view.rel_block_id(next_idx);
                    self.current_bounds = (view.max_tf(next_idx), view.min_dl_over_tf(next_idx));
                    self.current_block = Some(make_fts_block_id(state.chunk_id, rel));
                    state.entry_idx = next_idx;
                    return Ok(());
                }
                if state.skip_pin.is_some() {
                    let next_chunk = state.chunk_id + 1;
                    return self.seek_skip_iter(next_chunk, 0);
                }
                state.skip_iter.as_mut().unwrap().next();
                self.load_current_chunk(0)
            }
        }
    }

    #[inline]
    pub fn seek_block(&mut self, target: FtsBlockId) -> Result<()> {
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
                let target_chunk = fts_skip_chunk_id(target);
                let target_rel = fts_skip_slot(target);
                if target_chunk == state.chunk_id {
                    let view = chunk_view(
                        state.chunk_layout,
                        state.skip_pin.as_ref(),
                        state.skip_iter.as_ref(),
                    )
                    .expect("active skip chunk must have view");
                    let rel_slice = &view.rel_block_ids[state.entry_idx..];
                    let pos = rel_slice.partition_point(|&b| b < target_rel);
                    if pos < rel_slice.len() {
                        let i = state.entry_idx + pos;
                        let rel = view.rel_block_id(i);
                        self.current_bounds = (view.max_tf(i), view.min_dl_over_tf(i));
                        self.current_block = Some(make_fts_block_id(state.chunk_id, rel));
                        state.entry_idx = i;
                        return Ok(());
                    }
                    if state.skip_pin.is_some() {
                        let next_chunk = state.chunk_id + 1;
                        return self.seek_skip_iter(next_chunk, 0);
                    }
                    state.skip_iter.as_mut().unwrap().next();
                    return self.load_current_chunk(0);
                }

                if state.skip_pin.is_some() || target_chunk > state.chunk_id + 1 {
                    return self.seek_skip_iter(target_chunk, target_rel);
                }
                state.skip_iter.as_mut().unwrap().next();
                state.chunk_id = target_chunk;
                self.load_current_chunk(target_rel)
            }
        }
    }

    fn seek_skip_iter(&mut self, seek_chunk: u32, min_rel: u8) -> Result<()> {
        let db = self.db;
        let term_id = self.term_id;
        let CursorSource::OutOfLine(state) = &mut self.source else {
            return Ok(());
        };
        state.skip_pin = None;
        state.chunk_layout = None;
        let key = encode_fts_term_id_key(term_id, seek_chunk);
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
        state.chunk_id = seek_chunk;
        self.load_current_chunk(min_rel)
    }

    fn load_current_chunk(&mut self, min_rel: u8) -> Result<()> {
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
            if k.len() != 8 || k[0..4] != prefix {
                self.current_block = None;
                state.chunk_layout = None;
                return Ok(());
            }
            let cid = u32::from_be_bytes(k[4..8].try_into().unwrap());
            let Some(val) = iter.value() else {
                iter.next();
                continue;
            };
            let layout = SkipBlockLayout::parse(val)?;
            let view = layout.view(val);
            let idx = if min_rel == 0 || cid > state.chunk_id {
                Some(0)
            } else {
                view.find_ge(min_rel)
            };
            if let Some(i) = idx {
                state.chunk_id = cid;
                state.entry_idx = i;
                state.entry_count = layout.count;
                let rel = view.rel_block_id(i);
                let bounds = (view.max_tf(i), view.min_dl_over_tf(i));
                state.chunk_layout = Some(layout);
                self.current_block = Some(make_fts_block_id(cid, rel));
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
    skip_pin: Option<&'a DBPinnableSlice<'a>>,
    skip_iter: Option<&'a DBRawIteratorWithThreadMode<'a, DB>>,
) -> Option<SkipBlockView<'a>> {
    let layout = layout?;
    let bytes: &[u8] = if let Some(pin) = skip_pin {
        pin.as_ref()
    } else {
        skip_iter?.value()?
    };
    Some(layout.view(bytes))
}

#[inline]
fn iter_block_id(
    iter: &DBRawIteratorWithThreadMode<'_, DB>,
    prefix: &[u8; 4],
) -> Option<FtsBlockId> {
    match iter.key() {
        Some(k) if k.len() == 8 && k[0..4] == *prefix => {
            Some(u32::from_be_bytes(k[4..8].try_into().unwrap()))
        }
        _ => None,
    }
}

fn advance_posting_iter_to(
    iter: &mut DBRawIteratorWithThreadMode<'_, DB>,
    posting_pos: &mut Option<FtsBlockId>,
    term_id: TermId,
    prefix: &[u8; 4],
    target: FtsBlockId,
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
        cur_bid = iter_block_id(iter, prefix);
    }
    if cur_bid.is_some_and(|b| b < target) {
        iter.seek(encode_fts_term_id_key(term_id, target));
        cur_bid = iter_block_id(iter, prefix);
    }
    *posting_pos = cur_bid;
}

/// Cached per-attribute cursor over `cf_fts_doc_lens` document-length blocks.
pub struct RocksDocLenCursor {
    pub(super) blocks: DocLenBlockTable,
}

impl RocksDocLenCursor {
    #[inline]
    pub fn seek_block_bytes(&mut self, block_id: FtsBlockId) -> Result<Option<&[u8]>> {
        Ok(self
            .blocks
            .get(block_id as usize)
            .and_then(|entry| entry.as_deref()))
    }
}
