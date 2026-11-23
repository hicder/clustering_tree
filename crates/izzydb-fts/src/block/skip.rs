//! Skip blocks.

use izzydb_core::config::FTS_SKIP_CHUNK_SIZE;
use izzydb_core::error::{Result, TreeError};

use super::*;

/// Pre-validated offsets for a [`SkipBlockView`] so intra-chunk steps avoid re-parsing headers.
#[derive(Debug, Clone, Copy)]
pub struct SkipBlockLayout {
    pub width: IntWidth,
    pub count: usize,
    pub max_tf_off: usize,
    pub rel_off: usize,
}

impl SkipBlockLayout {
    #[inline]
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < SKIP_BLOCK_HEADER_LEN {
            return Err(TreeError::CorruptedData(format!(
                "SkipBlock header too short: {}",
                bytes.len()
            )));
        }
        let width = IntWidth::from_tag(bytes[0])?;
        if bytes[1] != 0 {
            return Err(TreeError::CorruptedData(format!(
                "unknown SkipBlock reserved byte: {}",
                bytes[1]
            )));
        }
        let count = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        if count == 0 || count > FTS_SKIP_CHUNK_SIZE {
            return Err(TreeError::CorruptedData(format!(
                "invalid SkipBlock count: {count}"
            )));
        }
        let expected = SKIP_BLOCK_HEADER_LEN + count * (4 + width.byte_size() + 1);
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "SkipBlock expected {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let max_tf_off = SKIP_BLOCK_HEADER_LEN + count * 4;
        let rel_off = max_tf_off + count * width.byte_size();
        Ok(Self {
            width,
            count,
            max_tf_off,
            rel_off,
        })
    }

    #[inline]
    pub fn view(self, bytes: &[u8]) -> SkipBlockView<'_> {
        SkipBlockView {
            width: self.width,
            count: self.count,
            min_dl_over_tfs_bytes: &bytes[SKIP_BLOCK_HEADER_LEN..self.max_tf_off],
            max_tfs_bytes: &bytes[self.max_tf_off..self.rel_off],
            rel_block_ids: &bytes[self.rel_off..],
        }
    }
}

/// Zero-copy view over an encoded [`SkipBlock`] in `cf_fts_skips`.
#[derive(Debug, Clone, Copy)]
pub struct SkipBlockView<'a> {
    pub width: IntWidth,
    pub count: usize,
    pub min_dl_over_tfs_bytes: &'a [u8],
    pub max_tfs_bytes: &'a [u8],
    pub rel_block_ids: &'a [u8],
}

impl<'a> SkipBlockView<'a> {
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let layout = SkipBlockLayout::parse(bytes)?;
        Ok(layout.view(bytes))
    }

    #[inline]
    pub fn rel_block_id(&self, idx: usize) -> u8 {
        self.rel_block_ids[idx]
    }

    #[inline]
    pub fn max_tf(&self, idx: usize) -> u32 {
        read_tf_at(self.width, self.max_tfs_bytes, idx)
    }

    #[inline]
    pub fn min_dl_over_tf(&self, idx: usize) -> f32 {
        let off = idx * 4;
        f32::from_le_bytes(self.min_dl_over_tfs_bytes[off..off + 4].try_into().unwrap())
    }

    #[inline]
    pub fn find_ge(&self, target_rel: u8) -> Option<usize> {
        let idx = self.rel_block_ids.partition_point(|&b| b < target_rel);
        if idx < self.count {
            Some(idx)
        } else {
            None
        }
    }
}

/// Mutable 256-block Block-Max skip summary chunk for `cf_fts_skips`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SkipBlock {
    pub rel_block_ids: Vec<u8>,
    pub max_tfs: Vec<u32>,
    pub min_dl_over_tfs: Vec<f32>,
}

impl SkipBlock {
    /// Construct a [`SkipBlock`] directly from strictly sorted `(rel_block_id, max_tf, min_dl_over_tf)` entries.
    pub fn from_sorted_bounds(entries: &[(u8, u32, f32)]) -> Self {
        debug_assert!(!entries.is_empty() && entries.len() <= FTS_SKIP_CHUNK_SIZE);
        debug_assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
        let mut rel_block_ids = Vec::with_capacity(entries.len());
        let mut max_tfs = Vec::with_capacity(entries.len());
        let mut min_dl_over_tfs = Vec::with_capacity(entries.len());
        for &(rel, max_tf, min_ratio) in entries {
            rel_block_ids.push(rel);
            max_tfs.push(max_tf);
            min_dl_over_tfs.push(min_ratio);
        }
        Self {
            rel_block_ids,
            max_tfs,
            min_dl_over_tfs,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.rel_block_ids.is_empty()
    }

    pub fn upsert(&mut self, rel_slot: u8, max_tf: u32, min_dl_over_tf: f32) {
        match self.rel_block_ids.binary_search(&rel_slot) {
            Ok(idx) => {
                self.max_tfs[idx] = max_tf;
                self.min_dl_over_tfs[idx] = min_dl_over_tf;
            }
            Err(idx) => {
                self.rel_block_ids.insert(idx, rel_slot);
                self.max_tfs.insert(idx, max_tf);
                self.min_dl_over_tfs.insert(idx, min_dl_over_tf);
            }
        }
    }

    pub fn remove(&mut self, rel_slot: u8) -> bool {
        let Ok(idx) = self.rel_block_ids.binary_search(&rel_slot) else {
            return false;
        };
        self.rel_block_ids.remove(idx);
        self.max_tfs.remove(idx);
        self.min_dl_over_tfs.remove(idx);
        true
    }

    pub fn encode(&self) -> Vec<u8> {
        debug_assert!(
            !self.rel_block_ids.is_empty() && self.rel_block_ids.len() <= FTS_SKIP_CHUNK_SIZE
        );
        let count = self.rel_block_ids.len();
        let max_tf = self.max_tfs.iter().copied().max().unwrap_or(0);
        let width = IntWidth::min_for_max(max_tf);
        let mut buf = Vec::with_capacity(
            SKIP_BLOCK_HEADER_LEN + count * (4 + width.byte_size() + 1),
        );
        buf.push(width as u8);
        buf.push(0);
        buf.extend_from_slice(&(count as u16).to_le_bytes());
        for &r in &self.min_dl_over_tfs {
            buf.extend_from_slice(&r.to_le_bytes());
        }
        match width {
            IntWidth::U8 => {
                for &tf in &self.max_tfs {
                    buf.push(tf as u8);
                }
            }
            IntWidth::U16 => {
                for &tf in &self.max_tfs {
                    buf.extend_from_slice(&(tf as u16).to_le_bytes());
                }
            }
            IntWidth::U32 => {
                for &tf in &self.max_tfs {
                    buf.extend_from_slice(&tf.to_le_bytes());
                }
            }
        }
        buf.extend_from_slice(&self.rel_block_ids);
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let view = SkipBlockView::parse(bytes)?;
        let mut max_tfs = Vec::with_capacity(view.count);
        let mut min_dl_over_tfs = Vec::with_capacity(view.count);
        for i in 0..view.count {
            max_tfs.push(view.max_tf(i));
            min_dl_over_tfs.push(view.min_dl_over_tf(i));
        }
        Ok(Self {
            rel_block_ids: view.rel_block_ids.to_vec(),
            max_tfs,
            min_dl_over_tfs,
        })
    }
}
