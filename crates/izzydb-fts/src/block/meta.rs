//! Integer widths, FTS meta, term stats, and term records.

use std::collections::BTreeMap;
use izzydb_core::config::{AttrId, RowBlock, TermId};
use izzydb_core::error::{Result, TreeError};

use super::*;

/// Compact integer width tag shared by `PackedPostingBlock` (`tf`), `DocLenBlock` (`dl`), and `SkipBlock` (`max_tf`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IntWidth {
    U8 = 0,
    U16 = 1,
    U32 = 2,
}

impl IntWidth {
    #[inline]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }

    #[inline]
    pub fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::U8),
            1 => Ok(Self::U16),
            2 => Ok(Self::U32),
            other => Err(TreeError::CorruptedData(format!(
                "unknown integer width tag: {other}"
            ))),
        }
    }

    #[inline]
    pub fn min_for_max(max_val: u32) -> Self {
        if max_val <= u8::MAX as u32 {
            Self::U8
        } else if max_val <= u16::MAX as u32 {
            Self::U16
        } else {
            Self::U32
        }
    }
}

/// Sound lower bound on `dl / tf` in `f32` (never exceeds `(dl as f64) / (tf as f64)`).
#[inline]
pub fn dl_over_tf_lower_bound(dl: u32, tf: u32) -> f32 {
    if tf == 0 || dl == 0 {
        return f32::INFINITY;
    }
    let exact = (dl as f64) / (tf as f64);
    let r = exact as f32;
    if (r as f64) > exact {
        r.next_down()
    } else {
        r
    }
}

/// Per-attribute corpus statistics stored in `cf_meta["fts_meta"]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttrFtsStats {
    pub doc_count: u64,
    pub total_tokens: u64,
}

/// Global FTS metadata stored in `default[\x01fts_meta]` (v4).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FtsMeta {
    pub next_term_id: TermId,
    pub per_attr_stats: BTreeMap<AttrId, AttrFtsStats>,
}

impl FtsMeta {
    pub fn new(next_term_id: TermId, per_attr_stats: BTreeMap<AttrId, AttrFtsStats>) -> Self {
        Self {
            next_term_id,
            per_attr_stats,
        }
    }

    pub fn alloc_term_id(&mut self) -> Result<TermId> {
        let id = self.next_term_id;
        self.next_term_id = self
            .next_term_id
            .checked_add(1)
            .ok_or_else(|| TreeError::Storage("TermId u32 space exhausted".into()))?;
        Ok(id)
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.per_attr_stats.len() * 18);
        buf.extend_from_slice(&self.next_term_id.to_le_bytes());
        buf.extend_from_slice(&(self.per_attr_stats.len() as u32).to_le_bytes());
        for (&attr_id, stats) in &self.per_attr_stats {
            buf.extend_from_slice(&attr_id.to_le_bytes());
            buf.extend_from_slice(&stats.doc_count.to_le_bytes());
            buf.extend_from_slice(&stats.total_tokens.to_le_bytes());
        }
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(TreeError::CorruptedData(
                "FtsMeta payload too short".into(),
            ));
        }
        let next_term_id = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if bytes.len() != 8 + count * 18 {
            return Err(TreeError::CorruptedData(format!(
                "FtsMeta expected {} bytes, got {}",
                8 + count * 18,
                bytes.len()
            )));
        }
        let mut per_attr_stats = BTreeMap::new();
        let mut off = 8;
        for _ in 0..count {
            let attr_id = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
            let doc_count = u64::from_le_bytes(bytes[off + 2..off + 10].try_into().unwrap());
            let total_tokens = u64::from_le_bytes(bytes[off + 10..off + 18].try_into().unwrap());
            per_attr_stats.insert(
                attr_id,
                AttrFtsStats {
                    doc_count,
                    total_tokens,
                },
            );
            off += 18;
        }
        Ok(Self {
            next_term_id,
            per_attr_stats,
        })
    }
}

/// Term-level BM25 statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TermStats {
    pub doc_freq: u64,
    pub max_tf: u32,
    pub min_dl_over_tf: f32,
}

impl Default for TermStats {
    #[inline]
    fn default() -> Self {
        Self {
            doc_freq: 0,
            max_tf: 0,
            min_dl_over_tf: f32::INFINITY,
        }
    }
}

impl TermStats {
    #[inline]
    pub fn observe(&mut self, tf: u32, dl: u32) {
        self.max_tf = self.max_tf.max(tf);
        self.min_dl_over_tf = self.min_dl_over_tf.min(dl_over_tf_lower_bound(dl, tf));
    }

    #[inline]
    pub fn encode(self) -> [u8; TERM_STATS_LEN] {
        let mut out = [0u8; TERM_STATS_LEN];
        out[0..8].copy_from_slice(&self.doc_freq.to_le_bytes());
        out[8..12].copy_from_slice(&self.max_tf.to_le_bytes());
        out[12..16].copy_from_slice(&self.min_dl_over_tf.to_le_bytes());
        out
    }

    #[inline]
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != TERM_STATS_LEN {
            return Err(TreeError::CorruptedData(format!(
                "TermStats expected {TERM_STATS_LEN} bytes, got {}",
                bytes.len()
            )));
        }
        let min_dl_over_tf = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if min_dl_over_tf.is_nan() || min_dl_over_tf < 0.0 {
            return Err(TreeError::CorruptedData(format!(
                "invalid TermStats min_dl_over_tf: {min_dl_over_tf}"
            )));
        }
        Ok(Self {
            doc_freq: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            max_tf: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            min_dl_over_tf,
        })
    }
}

/// Term dictionary entry stored in `cf_fts_terms` (v4) with numeric `TermId`, `TermStats`,
/// and optional inline single-block posting payload for rare terms (`doc_freq <= INLINE_TERM_MAX_DF`).
#[derive(Debug, Clone, PartialEq)]
pub struct TermRecord {
    pub term_id: TermId,
    pub stats: TermStats,
    pub inline_block: Option<(RowBlock, Vec<u8>)>,
}

impl TermRecord {
    pub fn new(term_id: TermId) -> Self {
        Self {
            term_id,
            stats: TermStats::default(),
            inline_block: None,
        }
    }

    pub fn with_inline(
        term_id: TermId,
        stats: TermStats,
        block_id: RowBlock,
        block_bytes: Vec<u8>,
    ) -> Self {
        Self {
            term_id,
            stats,
            inline_block: Some((block_id, block_bytes)),
        }
    }

    pub fn out_of_line(term_id: TermId, stats: TermStats) -> Self {
        Self {
            term_id,
            stats,
            inline_block: None,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let inline_len = self
            .inline_block
            .as_ref()
            .map_or(0, |(_, bytes)| 8 + bytes.len());
        let mut buf = Vec::with_capacity(TERM_RECORD_HEADER_LEN + inline_len);
        buf.extend_from_slice(&self.term_id.to_le_bytes());
        buf.extend_from_slice(&self.stats.encode());
        match &self.inline_block {
            Some((block_id, posting_bytes)) => {
                buf.push(TERM_FLAG_INLINE);
                buf.extend_from_slice(&block_id.seg().to_le_bytes());
                buf.extend_from_slice(&block_id.block().to_le_bytes());
                buf.extend_from_slice(posting_bytes);
            }
            None => {
                buf.push(0);
            }
        }
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < TERM_RECORD_HEADER_LEN {
            return Err(TreeError::CorruptedData(format!(
                "TermRecord expected >= {TERM_RECORD_HEADER_LEN} bytes, got {}",
                bytes.len()
            )));
        }
        let term_id = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let stats = TermStats::decode(&bytes[4..20])?;
        let flags = bytes[20];
        let inline_block = match flags {
            0 => {
                if bytes.len() != TERM_RECORD_HEADER_LEN {
                    return Err(TreeError::CorruptedData(format!(
                        "non-inline TermRecord expected {TERM_RECORD_HEADER_LEN} bytes, got {}",
                        bytes.len()
                    )));
                }
                None
            }
            TERM_FLAG_INLINE => {
                if bytes.len() < TERM_RECORD_HEADER_LEN + 8 + POSTING_BLOCK_HEADER_LEN {
                    return Err(TreeError::CorruptedData(
                        "inline TermRecord payload truncated".into(),
                    ));
                }
                let seg = u32::from_le_bytes(bytes[21..25].try_into().unwrap());
                let block = u32::from_le_bytes(bytes[25..29].try_into().unwrap());
                let block_id = RowBlock::new(seg, block);
                let posting_bytes = bytes[29..].to_vec();
                let _ = PostingBlockView::parse(&posting_bytes)?;
                Some((block_id, posting_bytes))
            }
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "unknown TermRecord flags: {other:#04x}"
                )));
            }
        };
        Ok(Self {
            term_id,
            stats,
            inline_block,
        })
    }
}
