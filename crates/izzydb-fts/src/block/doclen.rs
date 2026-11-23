//! Document-length blocks.

use izzydb_core::config::FTS_BLOCK_SIZE;
use izzydb_core::error::{Result, TreeError};

use super::*;

/// Zero-copy view over an encoded 256-slot [`DocLenBlock`] in `cf_fts_doc_lens`.
#[derive(Debug, Clone, Copy)]
pub struct DocLenBlockView<'a> {
    pub width: IntWidth,
    pub count: u16,
    pub dls_bytes: &'a [u8],
}

impl<'a> DocLenBlockView<'a> {
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < DOC_LEN_BLOCK_HEADER_LEN {
            return Err(TreeError::CorruptedData(format!(
                "DocLenBlock header too short: {}",
                bytes.len()
            )));
        }
        let width = IntWidth::from_tag(bytes[0])?;
        if bytes[1] != 0 {
            return Err(TreeError::CorruptedData(format!(
                "unknown DocLenBlock reserved byte: {}",
                bytes[1]
            )));
        }
        let count = u16::from_le_bytes([bytes[2], bytes[3]]);
        if count == 0 || count as usize > FTS_BLOCK_SIZE {
            return Err(TreeError::CorruptedData(format!(
                "invalid DocLenBlock count: {count}"
            )));
        }
        let expected = DOC_LEN_BLOCK_HEADER_LEN + FTS_BLOCK_SIZE * width.byte_size();
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "DocLenBlock expected {expected} bytes, got {}",
                bytes.len()
            )));
        }
        Ok(Self {
            width,
            count,
            dls_bytes: &bytes[DOC_LEN_BLOCK_HEADER_LEN..],
        })
    }

    #[inline]
    pub fn dl(&self, slot: u8) -> u32 {
        let idx = slot as usize;
        match self.width {
            IntWidth::U8 => self.dls_bytes[idx] as u32,
            IntWidth::U16 => {
                let off = idx * 2;
                u16::from_le_bytes([self.dls_bytes[off], self.dls_bytes[off + 1]]) as u32
            }
            IntWidth::U32 => {
                let off = idx * 4;
                u32::from_le_bytes(self.dls_bytes[off..off + 4].try_into().unwrap())
            }
        }
    }
}

/// Mutable 256-slot document length block for `(attr_id, block_id)` in `cf_fts_doc_lens`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocLenBlock {
    pub dls: [u32; FTS_BLOCK_SIZE],
    pub count: u16,
}

impl Default for DocLenBlock {
    fn default() -> Self {
        Self {
            dls: [0u32; FTS_BLOCK_SIZE],
            count: 0,
        }
    }
}

impl DocLenBlock {
    /// Construct a [`DocLenBlock`] directly from a slice of up to 256 per-slot document lengths.
    pub fn from_dls(slot_dls: &[u32]) -> Self {
        debug_assert!(slot_dls.len() <= FTS_BLOCK_SIZE);
        let mut dls = [0u32; FTS_BLOCK_SIZE];
        let mut count = 0u16;
        for (i, &dl) in slot_dls.iter().enumerate() {
            if dl > 0 {
                dls[i] = dl;
                count += 1;
            }
        }
        Self { dls, count }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn get(&self, slot: u8) -> u32 {
        self.dls[slot as usize]
    }

    #[inline]
    pub fn set(&mut self, slot: u8, dl: u32) {
        if dl == 0 {
            self.remove(slot);
            return;
        }
        let prev = self.dls[slot as usize];
        if prev == 0 {
            self.count += 1;
        }
        self.dls[slot as usize] = dl;
    }

    #[inline]
    pub fn remove(&mut self, slot: u8) {
        if self.dls[slot as usize] != 0 {
            self.dls[slot as usize] = 0;
            self.count -= 1;
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        debug_assert!(self.count > 0 && self.count as usize <= FTS_BLOCK_SIZE);
        let max_dl = self.dls.iter().copied().max().unwrap_or(0);
        let width = IntWidth::min_for_max(max_dl);
        let mut buf = Vec::with_capacity(DOC_LEN_BLOCK_HEADER_LEN + FTS_BLOCK_SIZE * width.byte_size());
        buf.push(width as u8);
        buf.push(0);
        buf.extend_from_slice(&self.count.to_le_bytes());
        match width {
            IntWidth::U8 => {
                for &dl in &self.dls {
                    buf.push(dl as u8);
                }
            }
            IntWidth::U16 => {
                for &dl in &self.dls {
                    buf.extend_from_slice(&(dl as u16).to_le_bytes());
                }
            }
            IntWidth::U32 => {
                for &dl in &self.dls {
                    buf.extend_from_slice(&dl.to_le_bytes());
                }
            }
        }
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let view = DocLenBlockView::parse(bytes)?;
        let mut dls = [0u32; FTS_BLOCK_SIZE];
        let mut actual_count = 0u16;
        for (i, out) in dls.iter_mut().enumerate() {
            let v = view.dl(i as u8);
            if v != 0 {
                actual_count += 1;
            }
            *out = v;
        }
        if actual_count != view.count {
            return Err(TreeError::CorruptedData(format!(
                "DocLenBlock non-zero count mismatch: header {} vs actual {actual_count}",
                view.count
            )));
        }
        Ok(Self {
            dls,
            count: actual_count,
        })
    }
}
