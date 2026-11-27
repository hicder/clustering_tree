//! Unified per-chunk skip + compact posting container stored in `cf_fts_skips`.

use izzydb_core::config::FTS_SKIP_CHUNK_SIZE;
use izzydb_core::error::{Result, TreeError};

use super::*;

/// Pre-validated offsets and chunk-level bounds for zero-copy traversal of an `FtsChunk`.
///
/// Computed once when a `RocksSkipCursor` steps onto a new chunk so every intra-chunk
/// `next_block()`, `seek_block()`, and posting payload slice is O(1) without re-parsing headers.
///
/// ```text
/// Header (12B): [version_flags: u8][skip_tf_width: u8][count_minus_1: u8][reserved: u8]
///               [chunk_max_tf: u32 LE][chunk_min_dl_over_tf: f32 LE]
/// Body:         [min_dl_over_tfs: f32 * count]
///               [block_ends: (u16 | u32) * count]
///               [max_tfs: skip_tf_width * count]
///               [rel_block_ids: u8 * count]
///               [compact posting payloads...]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FtsChunkLayout {
    /// Width of each entry in the `max_tfs` skip array.
    pub skip_tf_width: IntWidth,
    /// True when `block_ends` stores 16-bit cumulative offsets; false for 32-bit offsets.
    pub u16_offsets: bool,
    /// Number of active 256-row blocks in this chunk (`1..=256`).
    pub count: usize,
    /// Maximum term frequency across all blocks in this chunk.
    pub chunk_max_tf: u32,
    /// Minimum `dl / tf` lower bound across all blocks in this chunk.
    pub chunk_min_dl_over_tf: f32,
    /// Byte offset of `block_ends` within the chunk value.
    pub ends_off: usize,
    /// Byte offset of `max_tfs` within the chunk value.
    pub max_tf_off: usize,
    /// Byte offset of `rel_block_ids` within the chunk value.
    pub rel_off: usize,
    /// Byte offset of the concatenated compact posting payloads.
    pub payloads_off: usize,
}

impl FtsChunkLayout {
    /// Validate an encoded `FtsChunk` buffer and compute its section offsets.
    #[inline]
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < FTS_CHUNK_HEADER_LEN {
            return Err(TreeError::CorruptedData(format!(
                "FtsChunk header too short: {}",
                bytes.len()
            )));
        }
        let vflags = bytes[0];
        if (vflags & FTS_CHUNK_VERSION_MASK) != FTS_CHUNK_VERSION_V4 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported FtsChunk version/flags: {vflags:#04x}"
            )));
        }
        if (vflags & !(FTS_CHUNK_VERSION_MASK | FTS_CHUNK_FLAG_U16_OFFSETS)) != 0 {
            return Err(TreeError::CorruptedData(format!(
                "unknown FtsChunk flag bits: {vflags:#04x}"
            )));
        }
        let u16_offsets = (vflags & FTS_CHUNK_FLAG_U16_OFFSETS) != 0;
        let skip_tf_width = IntWidth::from_tag(bytes[1])?;
        let count = bytes[2] as usize + 1;
        if bytes[3] != 0 {
            return Err(TreeError::CorruptedData(format!(
                "unknown FtsChunk reserved byte: {}",
                bytes[3]
            )));
        }
        let chunk_max_tf = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let chunk_min_dl_over_tf = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if chunk_min_dl_over_tf.is_nan() || chunk_min_dl_over_tf < 0.0 {
            return Err(TreeError::CorruptedData(format!(
                "invalid FtsChunk min_dl_over_tf: {chunk_min_dl_over_tf}"
            )));
        }

        let ends_elem = if u16_offsets { 2 } else { 4 };
        let ends_off = FTS_CHUNK_HEADER_LEN + count * 4;
        let max_tf_off = ends_off + count * ends_elem;
        let rel_off = max_tf_off + count * skip_tf_width.byte_size();
        let payloads_off = rel_off + count;
        if bytes.len() < payloads_off {
            return Err(TreeError::CorruptedData(format!(
                "FtsChunk truncated: expected >= {payloads_off}, got {}",
                bytes.len()
            )));
        }

        let payloads_len = bytes.len() - payloads_off;
        let ends_bytes = &bytes[ends_off..max_tf_off];
        let last_end = read_end_at(u16_offsets, ends_bytes, count - 1);
        if last_end != payloads_len {
            return Err(TreeError::CorruptedData(format!(
                "FtsChunk payload length mismatch: last_end={last_end} vs actual={payloads_len}"
            )));
        }

        Ok(Self {
            skip_tf_width,
            u16_offsets,
            count,
            chunk_max_tf,
            chunk_min_dl_over_tf,
            ends_off,
            max_tf_off,
            rel_off,
            payloads_off,
        })
    }

    /// Bind a validated [`FtsChunkLayout`] to the same `bytes` slice for zero-copy access.
    #[inline]
    pub fn view(self, bytes: &[u8]) -> FtsChunkView<'_> {
        FtsChunkView {
            skip_tf_width: self.skip_tf_width,
            u16_offsets: self.u16_offsets,
            count: self.count,
            chunk_max_tf: self.chunk_max_tf,
            chunk_min_dl_over_tf: self.chunk_min_dl_over_tf,
            min_dl_over_tfs_bytes: &bytes[FTS_CHUNK_HEADER_LEN..self.ends_off],
            ends_bytes: &bytes[self.ends_off..self.max_tf_off],
            max_tfs_bytes: &bytes[self.max_tf_off..self.rel_off],
            rel_block_ids: &bytes[self.rel_off..self.payloads_off],
            payloads: &bytes[self.payloads_off..],
        }
    }
}

/// Zero-copy view over an encoded `FtsChunk` value in `cf_fts_skips`.
///
/// Exposes both the Block-Max skip bounds and the compact posting payload slice for each
/// active block in the 256-block chunk.
///
/// ```text
/// let view = FtsChunkView::parse(bytes)?;
/// if let Some(idx) = view.find_ge(target_rel) {
///     let posting = PostingBlockView::parse_compact(view.block_bytes(idx))?;
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FtsChunkView<'a> {
    /// Width of each entry in `max_tfs_bytes`.
    pub skip_tf_width: IntWidth,
    /// True when `ends_bytes` stores `u16` offsets; false for `u32`.
    pub u16_offsets: bool,
    /// Number of active blocks in this chunk (`1..=256`).
    pub count: usize,
    /// Maximum `tf` across all blocks in this chunk.
    pub chunk_max_tf: u32,
    /// Minimum `dl / tf` lower bound across all blocks in this chunk.
    pub chunk_min_dl_over_tf: f32,
    /// Packed `[f32; count]` little-endian `min_dl_over_tf` bounds.
    pub min_dl_over_tfs_bytes: &'a [u8],
    /// Packed cumulative payload end offsets (`[u16; count]` or `[u32; count]`).
    pub ends_bytes: &'a [u8],
    /// Packed `[skip_tf_width; count]` `max_tf` bounds.
    pub max_tfs_bytes: &'a [u8],
    /// Strictly ascending `rel_block_id` (`0..=255`) bytes.
    pub rel_block_ids: &'a [u8],
    /// Concatenated compact posting payloads.
    pub payloads: &'a [u8],
}

impl<'a> FtsChunkView<'a> {
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let layout = FtsChunkLayout::parse(bytes)?;
        Ok(layout.view(bytes))
    }

    #[inline]
    pub fn rel_block_id(&self, idx: usize) -> u8 {
        self.rel_block_ids[idx]
    }

    #[inline]
    pub fn max_tf(&self, idx: usize) -> u32 {
        read_tf_at(self.skip_tf_width, self.max_tfs_bytes, idx)
    }

    #[inline]
    pub fn min_dl_over_tf(&self, idx: usize) -> f32 {
        let off = idx * 4;
        f32::from_le_bytes(self.min_dl_over_tfs_bytes[off..off + 4].try_into().unwrap())
    }

    #[inline]
    pub fn block_bytes(&self, idx: usize) -> &'a [u8] {
        let start = if idx == 0 {
            0
        } else {
            read_end_at(self.u16_offsets, self.ends_bytes, idx - 1)
        };
        let end = read_end_at(self.u16_offsets, self.ends_bytes, idx);
        &self.payloads[start..end]
    }

    #[inline]
    pub fn find_ge(&self, target_rel: u8) -> Option<usize> {
        let idx = self.rel_block_ids.partition_point(|&b| b < target_rel);
        (idx < self.count).then_some(idx)
    }

    #[inline]
    pub fn find_exact(&self, target_rel: u8) -> Option<usize> {
        self.rel_block_ids.binary_search(&target_rel).ok()
    }
}

#[inline]
fn read_end_at(u16_offsets: bool, ends_bytes: &[u8], idx: usize) -> usize {
    if u16_offsets {
        let off = idx * 2;
        u16::from_le_bytes([ends_bytes[off], ends_bytes[off + 1]]) as usize
    } else {
        let off = idx * 4;
        u32::from_le_bytes(ends_bytes[off..off + 4].try_into().unwrap()) as usize
    }
}

/// Encoder and splicer for unified v4 per-chunk values in `cf_fts_skips`.
///
/// Combines skip bounds and compact posting block payloads for up to 256 consecutive
/// `RowBlock`s into a single RocksDB value keyed by `(term_id, seg, chunk)`.
///
/// ```text
/// let bytes = FtsChunk::encode_from_blocks(&[(0, 2, 7.5, &compact_b0)])?;
/// ```
pub struct FtsChunk;

impl FtsChunk {
    /// Encode strictly sorted `(rel_block_id, max_tf, min_dl_over_tf, compact_block_bytes)`
    /// entries into a unified v4 chunk value.
    pub fn encode_from_blocks(blocks: &[(u8, u32, f32, &[u8])]) -> Vec<u8> {
        debug_assert!(!blocks.is_empty() && blocks.len() <= FTS_SKIP_CHUNK_SIZE);
        debug_assert!(blocks.windows(2).all(|w| w[0].0 < w[1].0));
        let count = blocks.len();
        let mut chunk_max_tf = 0u32;
        let mut chunk_min_ratio = f32::INFINITY;
        let mut total_payload_len = 0usize;
        for &(_, mtf, min_r, bytes) in blocks {
            chunk_max_tf = chunk_max_tf.max(mtf);
            chunk_min_ratio = chunk_min_ratio.min(min_r);
            total_payload_len += bytes.len();
        }
        let skip_tf_width = IntWidth::min_for_max(chunk_max_tf);
        let u16_offsets = total_payload_len <= u16::MAX as usize;
        let ends_elem = if u16_offsets { 2 } else { 4 };
        let total_len = FTS_CHUNK_HEADER_LEN
            + count * (4 + ends_elem + skip_tf_width.byte_size() + 1)
            + total_payload_len;
        let mut buf = Vec::with_capacity(total_len);

        let vflags = FTS_CHUNK_VERSION_V4 | if u16_offsets { FTS_CHUNK_FLAG_U16_OFFSETS } else { 0 };
        buf.push(vflags);
        buf.push(skip_tf_width as u8);
        buf.push((count - 1) as u8);
        buf.push(0);
        buf.extend_from_slice(&chunk_max_tf.to_le_bytes());
        buf.extend_from_slice(&chunk_min_ratio.to_le_bytes());

        for &(_, _, min_r, _) in blocks {
            buf.extend_from_slice(&min_r.to_le_bytes());
        }

        let mut running = 0usize;
        if u16_offsets {
            for &(_, _, _, bytes) in blocks {
                running += bytes.len();
                buf.extend_from_slice(&(running as u16).to_le_bytes());
            }
        } else {
            for &(_, _, _, bytes) in blocks {
                running += bytes.len();
                buf.extend_from_slice(&(running as u32).to_le_bytes());
            }
        }

        match skip_tf_width {
            IntWidth::U8 => {
                for &(_, mtf, _, _) in blocks {
                    buf.push(mtf as u8);
                }
            }
            IntWidth::U16 => {
                for &(_, mtf, _, _) in blocks {
                    buf.extend_from_slice(&(mtf as u16).to_le_bytes());
                }
            }
            IntWidth::U32 => {
                for &(_, mtf, _, _) in blocks {
                    buf.extend_from_slice(&mtf.to_le_bytes());
                }
            }
        }

        for &(rel, _, _, _) in blocks {
            buf.push(rel);
        }

        for &(_, _, _, bytes) in blocks {
            buf.extend_from_slice(bytes);
        }

        buf
    }

    /// Upsert a single block into an optional existing encoded chunk without decoding untouched blocks.
    pub fn upsert_block(
        existing: Option<&[u8]>,
        rel_block: u8,
        max_tf: u32,
        min_dl_over_tf: f32,
        compact_block: &[u8],
    ) -> Result<Vec<u8>> {
        let Some(existing_bytes) = existing else {
            return Ok(Self::encode_from_blocks(&[(
                rel_block,
                max_tf,
                min_dl_over_tf,
                compact_block,
            )]));
        };
        let view = FtsChunkView::parse(existing_bytes)?;
        let mut entries: Vec<(u8, u32, f32, &[u8])> = Vec::with_capacity(view.count + 1);
        let mut inserted = false;
        for i in 0..view.count {
            let r = view.rel_block_id(i);
            if !inserted && rel_block <= r {
                entries.push((rel_block, max_tf, min_dl_over_tf, compact_block));
                inserted = true;
                if rel_block == r {
                    continue;
                }
            }
            entries.push((
                r,
                view.max_tf(i),
                view.min_dl_over_tf(i),
                view.block_bytes(i),
            ));
        }
        if !inserted {
            entries.push((rel_block, max_tf, min_dl_over_tf, compact_block));
        }
        Ok(Self::encode_from_blocks(&entries))
    }

    /// Merge a sorted batch of `(rel_block, max_tf, min_dl_over_tf, compact_block)` updates into an
    /// optional existing encoded chunk.
    pub fn upsert_blocks(
        existing: Option<&[u8]>,
        updates: &[(u8, u32, f32, Vec<u8>)],
    ) -> Result<Vec<u8>> {
        debug_assert!(!updates.is_empty());
        debug_assert!(updates.windows(2).all(|w| w[0].0 < w[1].0));
        let Some(existing_bytes) = existing else {
            let refs: Vec<(u8, u32, f32, &[u8])> = updates
                .iter()
                .map(|(r, mtf, mr, b)| (*r, *mtf, *mr, b.as_slice()))
                .collect();
            return Ok(Self::encode_from_blocks(&refs));
        };
        let view = FtsChunkView::parse(existing_bytes)?;
        let mut entries: Vec<(u8, u32, f32, &[u8])> =
            Vec::with_capacity(view.count + updates.len());
        let mut i = 0usize;
        let mut j = 0usize;
        while i < view.count && j < updates.len() {
            let er = view.rel_block_id(i);
            let ur = updates[j].0;
            if er < ur {
                entries.push((
                    er,
                    view.max_tf(i),
                    view.min_dl_over_tf(i),
                    view.block_bytes(i),
                ));
                i += 1;
            } else if ur < er {
                entries.push((ur, updates[j].1, updates[j].2, updates[j].3.as_slice()));
                j += 1;
            } else {
                entries.push((ur, updates[j].1, updates[j].2, updates[j].3.as_slice()));
                i += 1;
                j += 1;
            }
        }
        while i < view.count {
            entries.push((
                view.rel_block_id(i),
                view.max_tf(i),
                view.min_dl_over_tf(i),
                view.block_bytes(i),
            ));
            i += 1;
        }
        while j < updates.len() {
            entries.push((
                updates[j].0,
                updates[j].1,
                updates[j].2,
                updates[j].3.as_slice(),
            ));
            j += 1;
        }
        Ok(Self::encode_from_blocks(&entries))
    }
}
