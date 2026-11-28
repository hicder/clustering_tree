use rocksdb::{DB, DBPinnableSlice, DBRawIteratorWithThreadMode, ReadOptions};

use izzydb_core::config::{RowBlock, SegmentId, SkipChunk, TermId};
use izzydb_core::error::{Result, TreeError};
use izzydb_fts::{
    EncodedPosting, FtsChunkLayout, PostingBlockHeader, PostingBlockView,
};

use crate::codec::{encode_fts_term_id_prefix, fts_chunk_key, fts_chunk_key_coord};

use super::types::{DocLenBlockTable, map_db_err};
use super::CF_FTS_SKIPS;

const SEQUENTIAL_CHUNK_STEP_LIMIT: usize = 4;

enum CursorSource<'a> {
    Inline {
        bytes: Vec<u8>,
        exhausted: bool,
    },
    OutOfLine(Box<OutOfLineState<'a>>),
}

struct OutOfLineState<'a> {
    skip_iter: Option<DBRawIteratorWithThreadMode<'a, DB>>,
    chunk_buf: Vec<u8>,
    chunk_layout: Option<FtsChunkLayout>,
    chunk: SkipChunk,
    entry_idx: usize,
    entry_count: usize,
    seg_start: SegmentId,
    seg_end: SegmentId,
}

/// Cursor over a term's unified Block-Max chunks (`cf_fts_skips`, keyed `term_id || seg || chunk`)
/// or inline posting block, iterating in global [`RowBlock`] order across all (or a range of) segments.
///
/// Serves both skip bounds and compact posting payloads directly from the active chunk buffer
/// without separate `cf_fts_postings` iterator lookups.
///
/// ```text
/// let mut cur = RocksSkipCursor::open_out_of_line_range(&db, term_id, 0, 16)?;
/// while let Some(bid) = cur.current_block_id() {
///     let view = cur.current_posting_view()?.unwrap();
///     cur.next_block()?;
/// }
/// ```
pub struct RocksSkipCursor<'a> {
    /// Underlying RocksDB instance.
    db: &'a DB,
    /// Numeric term identifier.
    term_id: TermId,
    /// Inline posting buffer or out-of-line `cf_fts_skips` chunk iterator.
    source: CursorSource<'a>,
    /// Current `RowBlock` coordinate (`None` when exhausted).
    current_block: Option<RowBlock>,
    /// `(max_tf, min_dl_over_tf)` bounds for `current_block`.
    current_bounds: (u32, f32),
}

impl<'a> RocksSkipCursor<'a> {
    pub(super) fn from_inline(
        db: &'a DB,
        term_id: TermId,
        block_id: RowBlock,
        bytes: Vec<u8>,
        seg_start: SegmentId,
        seg_end: SegmentId,
    ) -> Result<Self> {
        if block_id.seg() < seg_start || block_id.seg() >= seg_end {
            return Ok(Self {
                db,
                term_id,
                source: CursorSource::Inline {
                    bytes,
                    exhausted: true,
                },
                current_block: None,
                current_bounds: (0, f32::INFINITY),
            });
        }
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

    pub(super) fn open_out_of_line_range(
        db: &'a DB,
        term_id: TermId,
        seg_start: SegmentId,
        seg_end: SegmentId,
    ) -> Result<Self> {
        let mut cursor = Self {
            db,
            term_id,
            source: CursorSource::OutOfLine(Box::new(OutOfLineState {
                skip_iter: None,
                chunk_buf: Vec::new(),
                chunk_layout: None,
                chunk: SkipChunk::default(),
                entry_idx: 0,
                entry_count: 0,
                seg_start,
                seg_end,
            })),
            current_block: None,
            current_bounds: (0, f32::INFINITY),
        };
        if seg_start < seg_end {
            cursor.seek_skip_iter_prefix()?;
        }
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
    pub fn current_chunk_bounds(&self) -> Option<(SkipChunk, u32, f32)> {
        match &self.source {
            CursorSource::Inline { .. } => None,
            CursorSource::OutOfLine(state) => {
                let layout = state.chunk_layout?;
                Some((
                    state.chunk,
                    layout.chunk_max_tf,
                    layout.chunk_min_dl_over_tf,
                ))
            }
        }
    }

    #[inline]
    pub fn inline_posting_bytes(&self) -> Option<&[u8]> {
        match &self.source {
            CursorSource::Inline { bytes, exhausted } if !*exhausted => Some(bytes.as_slice()),
            _ => None,
        }
    }

    /// Borrow the raw encoded posting payload for `current_block` directly from the inline
    /// buffer or the cached active chunk slice (zero RocksDB calls, no decoding).
    #[inline]
    pub fn current_posting_encoded(&mut self) -> Result<Option<EncodedPosting<'_>>> {
        if self.current_block.is_none() {
            return Ok(None);
        }
        match &self.source {
            CursorSource::Inline { bytes, exhausted } => {
                if *exhausted {
                    return Ok(None);
                }
                Ok(Some(EncodedPosting::Standalone(bytes.as_slice())))
            }
            CursorSource::OutOfLine(state) => {
                let Some(layout) = state.chunk_layout else {
                    return Ok(None);
                };
                let view = layout.view(&state.chunk_buf);
                Ok(Some(EncodedPosting::Compact(
                    view.block_bytes(state.entry_idx),
                )))
            }
        }
    }

    /// Parse the [`PostingBlockView`] for `current_block` directly from the inline buffer
    /// or the cached active chunk slice (zero RocksDB calls).
    #[inline]
    pub fn current_posting_view(&mut self) -> Result<Option<PostingBlockView<'_>>> {
        match self.current_posting_encoded()? {
            Some(encoded) => Ok(Some(encoded.parse()?)),
            None => Ok(None),
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
                    let view = state
                        .chunk_layout
                        .expect("active skip chunk must have layout")
                        .view(&state.chunk_buf);
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
                if target_chunk.seg() >= state.seg_end {
                    self.current_block = None;
                    state.chunk_layout = None;
                    return Ok(());
                }
                if target_chunk == state.chunk {
                    let view = state
                        .chunk_layout
                        .expect("active skip chunk must have layout")
                        .view(&state.chunk_buf);
                    let next_idx = state.entry_idx + 1;
                    if next_idx < state.entry_count {
                        if view.rel_block_ids[next_idx] >= target_rel {
                            let rel = view.rel_block_id(next_idx);
                            self.current_bounds =
                                (view.max_tf(next_idx), view.min_dl_over_tf(next_idx));
                            self.current_block = Some(state.chunk.block(rel));
                            state.entry_idx = next_idx;
                            return Ok(());
                        }
                        let rel_slice = &view.rel_block_ids[next_idx..];
                        let pos = rel_slice.partition_point(|&b| b < target_rel);
                        if pos < rel_slice.len() {
                            let i = next_idx + pos;
                            let rel = view.rel_block_id(i);
                            self.current_bounds = (view.max_tf(i), view.min_dl_over_tf(i));
                            self.current_block = Some(state.chunk.block(rel));
                            state.entry_idx = i;
                            return Ok(());
                        }
                    }
                    state.skip_iter.as_mut().unwrap().next();
                    return self.load_current_chunk(Some((target_chunk, target_rel)));
                }

                // Step sequentially first before falling back to RocksDB Seek.
                state.skip_iter.as_mut().unwrap().next();
                self.load_current_chunk(Some((target_chunk, target_rel)))
            }
        }
    }

    #[inline]
    pub fn seek_past_chunk(&mut self, chunk: SkipChunk) -> Result<()> {
        let Some(cur) = self.current_block else {
            return Ok(());
        };
        if cur.chunk() != chunk {
            return Ok(());
        }
        match &mut self.source {
            CursorSource::Inline { exhausted, .. } => {
                *exhausted = true;
                self.current_block = None;
                Ok(())
            }
            CursorSource::OutOfLine(state) => {
                state.skip_iter.as_mut().unwrap().next();
                self.load_current_chunk(None)
            }
        }
    }

    fn init_read_options(term_id: TermId, seg_end: SegmentId) -> ReadOptions {
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        if seg_end < SegmentId::MAX {
            ro.set_iterate_upper_bound(fts_chunk_key(term_id, seg_end, 0));
        } else if let Some(next_t) = term_id.checked_add(1) {
            ro.set_iterate_upper_bound(next_t.to_be_bytes());
        }
        ro
    }

    fn seek_skip_iter_prefix(&mut self) -> Result<()> {
        let db = self.db;
        let term_id = self.term_id;
        let CursorSource::OutOfLine(state) = &mut self.source else {
            return Ok(());
        };
        state.chunk_layout = None;
        let start_key = fts_chunk_key(term_id, state.seg_start, 0);
        let cf = db.cf_handle(CF_FTS_SKIPS).ok_or_else(|| {
            TreeError::CorruptedData(format!("missing column family {CF_FTS_SKIPS}"))
        })?;
        let ro = Self::init_read_options(term_id, state.seg_end);
        let mut it = db.raw_iterator_cf_opt(cf, ro);
        it.seek(start_key);
        state.skip_iter = Some(it);
        self.load_current_chunk(None)
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
        let mut steps = 0usize;
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
            if seg >= state.seg_end {
                self.current_block = None;
                state.chunk_layout = None;
                return Ok(());
            }
            let cid = u32::from_be_bytes(k[8..12].try_into().unwrap());
            let chunk = SkipChunk::new(seg, cid);
            if let Some((target_chunk, _)) = min_pos
                && chunk < target_chunk
            {
                if steps < SEQUENTIAL_CHUNK_STEP_LIMIT {
                    iter.next();
                    steps += 1;
                } else {
                    state.chunk_layout = None;
                    iter.seek(fts_chunk_key_coord(term_id, target_chunk));
                    steps = 0;
                }
                continue;
            }
            let Some(val) = iter.value() else {
                iter.next();
                continue;
            };
            state.chunk_buf.clear();
            state.chunk_buf.extend_from_slice(val);
            let layout = FtsChunkLayout::parse(&state.chunk_buf)?;
            let view = layout.view(&state.chunk_buf);
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

#[allow(dead_code)]
type _UnusedPin<'a> = DBPinnableSlice<'a>;
