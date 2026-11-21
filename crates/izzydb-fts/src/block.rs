use std::collections::BTreeMap;

use izzydb_core::config::{AttrId, FTS_BLOCK_SIZE, FtsDocId};
use izzydb_core::error::{Result, TreeError};

pub const POSTING_BLOCK_HEADER_LEN: usize = 12;
pub const TERM_STATS_LEN: usize = 16;
const FLAG_U16: u16 = 0;
const FLAG_U32: u16 = 1;

/// Per-attribute corpus statistics stored in `cf_meta["fts_meta"]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttrFtsStats {
    pub doc_count: u64,
    pub total_tokens: u64,
}

/// Global FTS metadata stored in `cf_meta["fts_meta"]`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FtsMeta {
    pub next_fts_doc_id: FtsDocId,
    pub per_attr_stats: BTreeMap<AttrId, AttrFtsStats>,
}

impl FtsMeta {
    pub fn alloc_doc_id(&mut self) -> Result<FtsDocId> {
        let id = self.next_fts_doc_id;
        self.next_fts_doc_id = self
            .next_fts_doc_id
            .checked_add(1)
            .ok_or_else(|| TreeError::Storage("FtsDocId u32 space exhausted".into()))?;
        Ok(id)
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.per_attr_stats.len() * 18);
        buf.extend_from_slice(&self.next_fts_doc_id.to_le_bytes());
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
        let next_fts_doc_id = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
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
            next_fts_doc_id,
            per_attr_stats,
        })
    }
}

/// Term-level statistics stored in `cf_fts_terms`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TermStats {
    pub doc_freq: u64,
    pub max_tf: u32,
    pub min_dl: u32,
}

impl TermStats {
    #[inline]
    pub fn encode(self) -> [u8; TERM_STATS_LEN] {
        let mut out = [0u8; TERM_STATS_LEN];
        out[0..8].copy_from_slice(&self.doc_freq.to_le_bytes());
        out[8..12].copy_from_slice(&self.max_tf.to_le_bytes());
        out[12..16].copy_from_slice(&self.min_dl.to_le_bytes());
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
        Ok(Self {
            doc_freq: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            max_tf: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            min_dl: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        })
    }
}

/// 12-byte header of a [`PackedPostingBlock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostingBlockHeader {
    pub count: u16,
    pub flags: u16,
    pub max_tf: u32,
    pub min_dl: u32,
}

impl PostingBlockHeader {
    #[inline]
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < POSTING_BLOCK_HEADER_LEN {
            return Err(TreeError::CorruptedData(format!(
                "PostingBlock header too short: {}",
                bytes.len()
            )));
        }
        let count = u16::from_le_bytes(bytes[0..2].try_into().unwrap());
        let flags = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let max_tf = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let min_dl = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if count == 0 || count as usize > FTS_BLOCK_SIZE {
            return Err(TreeError::CorruptedData(format!(
                "invalid PostingBlock count: {count}"
            )));
        }
        if flags != FLAG_U16 && flags != FLAG_U32 {
            return Err(TreeError::CorruptedData(format!(
                "unknown PostingBlock flags: {flags}"
            )));
        }
        Ok(Self {
            count,
            flags,
            max_tf,
            min_dl,
        })
    }
}

/// Zero-copy view over an encoded [`PackedPostingBlock`].
#[derive(Debug, Clone, Copy)]
pub struct PostingBlockView<'a> {
    pub header: PostingBlockHeader,
    pub slots: &'a [u8],
    pub tfs_bytes: &'a [u8],
    pub dls_bytes: &'a [u8],
}

impl<'a> PostingBlockView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let header = PostingBlockHeader::parse(bytes)?;
        let n = header.count as usize;
        let width = if header.flags == FLAG_U16 { 2 } else { 4 };
        let expected = POSTING_BLOCK_HEADER_LEN + n + 2 * n * width;
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "PostingBlock expected {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let slots_start = POSTING_BLOCK_HEADER_LEN;
        let tfs_start = slots_start + n;
        let dls_start = tfs_start + n * width;
        Ok(Self {
            header,
            slots: &bytes[slots_start..tfs_start],
            tfs_bytes: &bytes[tfs_start..dls_start],
            dls_bytes: &bytes[dls_start..],
        })
    }

    #[inline]
    pub fn is_u16(&self) -> bool {
        self.header.flags == FLAG_U16
    }

    #[inline]
    pub fn tf(&self, idx: usize) -> u32 {
        if self.header.flags == FLAG_U16 {
            let off = idx * 2;
            u16::from_le_bytes([self.tfs_bytes[off], self.tfs_bytes[off + 1]]) as u32
        } else {
            let off = idx * 4;
            u32::from_le_bytes(self.tfs_bytes[off..off + 4].try_into().unwrap())
        }
    }

    #[inline]
    pub fn dl(&self, idx: usize) -> u32 {
        if self.header.flags == FLAG_U16 {
            let off = idx * 2;
            u16::from_le_bytes([self.dls_bytes[off], self.dls_bytes[off + 1]]) as u32
        } else {
            let off = idx * 4;
            u32::from_le_bytes(self.dls_bytes[off..off + 4].try_into().unwrap())
        }
    }

    #[inline]
    pub fn find_slot(&self, slot: u8) -> Option<usize> {
        self.slots.binary_search(&slot).ok()
    }
}

/// Mutable 256-document Structure-of-Arrays posting block for `cf_fts_postings`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PackedPostingBlock {
    pub slots: Vec<u8>,
    pub tfs: Vec<u32>,
    pub dls: Vec<u32>,
    pub max_tf: u32,
    pub min_dl: u32,
}

impl PackedPostingBlock {
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn upsert(&mut self, slot: u8, tf: u32, dl: u32) {
        match self.slots.binary_search(&slot) {
            Ok(idx) => {
                self.tfs[idx] = tf;
                self.dls[idx] = dl;
            }
            Err(idx) => {
                self.slots.insert(idx, slot);
                self.tfs.insert(idx, tf);
                self.dls.insert(idx, dl);
            }
        }
        self.recompute_bounds();
    }

    pub fn remove(&mut self, slot: u8) -> bool {
        let Ok(idx) = self.slots.binary_search(&slot) else {
            return false;
        };
        self.slots.remove(idx);
        self.tfs.remove(idx);
        self.dls.remove(idx);
        self.recompute_bounds();
        true
    }

    pub fn recompute_bounds(&mut self) {
        if self.slots.is_empty() {
            self.max_tf = 0;
            self.min_dl = 0;
            return;
        }
        let mut max_tf = 0u32;
        let mut min_dl = u32::MAX;
        for (&tf, &dl) in self.tfs.iter().zip(self.dls.iter()) {
            if tf > max_tf {
                max_tf = tf;
            }
            if dl < min_dl {
                min_dl = dl;
            }
        }
        self.max_tf = max_tf;
        self.min_dl = min_dl;
    }

    pub fn encode(&self) -> Vec<u8> {
        debug_assert!(!self.slots.is_empty() && self.slots.len() <= FTS_BLOCK_SIZE);
        let n = self.slots.len();
        let needs_u32 = self
            .tfs
            .iter()
            .chain(self.dls.iter())
            .any(|&v| v > u16::MAX as u32);
        let flags = if needs_u32 { FLAG_U32 } else { FLAG_U16 };
        let width = if needs_u32 { 4 } else { 2 };

        let mut buf = Vec::with_capacity(POSTING_BLOCK_HEADER_LEN + n + 2 * n * width);
        buf.extend_from_slice(&(n as u16).to_le_bytes());
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&self.max_tf.to_le_bytes());
        buf.extend_from_slice(&self.min_dl.to_le_bytes());
        buf.extend_from_slice(&self.slots);
        if needs_u32 {
            for &tf in &self.tfs {
                buf.extend_from_slice(&tf.to_le_bytes());
            }
            for &dl in &self.dls {
                buf.extend_from_slice(&dl.to_le_bytes());
            }
        } else {
            for &tf in &self.tfs {
                buf.extend_from_slice(&(tf as u16).to_le_bytes());
            }
            for &dl in &self.dls {
                buf.extend_from_slice(&(dl as u16).to_le_bytes());
            }
        }
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let view = PostingBlockView::parse(bytes)?;
        let n = view.header.count as usize;
        let mut tfs = Vec::with_capacity(n);
        let mut dls = Vec::with_capacity(n);
        for i in 0..n {
            tfs.push(view.tf(i));
            dls.push(view.dl(i));
        }
        Ok(Self {
            slots: view.slots.to_vec(),
            tfs,
            dls,
            max_tf: view.header.max_tf,
            min_dl: view.header.min_dl,
        })
    }
}

/// Zero-copy view over an encoded [`PackedPositionBlock`].
#[derive(Debug, Clone, Copy)]
pub struct PositionBlockView<'a> {
    pub slots: &'a [u8],
    pub pos_offsets_bytes: &'a [u8],
    pub delta_varints: &'a [u8],
}

impl<'a> PositionBlockView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 2 {
            return Err(TreeError::CorruptedData(
                "PositionBlock header too short".into(),
            ));
        }
        let count = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        if count == 0 || count > FTS_BLOCK_SIZE {
            return Err(TreeError::CorruptedData(format!(
                "invalid PositionBlock count: {count}"
            )));
        }
        let slots_start = 2;
        let offsets_start = slots_start + count;
        let varints_start = offsets_start + (count + 1) * 4;
        if bytes.len() < varints_start {
            return Err(TreeError::CorruptedData(
                "PositionBlock offsets truncated".into(),
            ));
        }
        let slots = &bytes[slots_start..offsets_start];
        let pos_offsets_bytes = &bytes[offsets_start..varints_start];
        let delta_varints = &bytes[varints_start..];
        let last_off = u32::from_le_bytes(
            pos_offsets_bytes[count * 4..(count + 1) * 4]
                .try_into()
                .unwrap(),
        ) as usize;
        if last_off != delta_varints.len() {
            return Err(TreeError::CorruptedData(format!(
                "PositionBlock varint length mismatch: header {last_off} vs actual {}",
                delta_varints.len()
            )));
        }
        Ok(Self {
            slots,
            pos_offsets_bytes,
            delta_varints,
        })
    }

    pub fn slot_positions(&self, slot: u8) -> Result<Option<Vec<u32>>> {
        let Ok(idx) = self.slots.binary_search(&slot) else {
            return Ok(None);
        };
        let start = u32::from_le_bytes(
            self.pos_offsets_bytes[idx * 4..(idx + 1) * 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let end = u32::from_le_bytes(
            self.pos_offsets_bytes[(idx + 1) * 4..(idx + 2) * 4]
                .try_into()
                .unwrap(),
        ) as usize;
        if end < start || end > self.delta_varints.len() {
            return Err(TreeError::CorruptedData(
                "PositionBlock offset out of bounds".into(),
            ));
        }
        decode_delta_varints(&self.delta_varints[start..end]).map(Some)
    }
}

/// Mutable 256-document delta-varint position block for `cf_fts_positions`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PackedPositionBlock {
    pub slots: Vec<u8>,
    pub positions: Vec<Vec<u32>>,
}

impl PackedPositionBlock {
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn upsert(&mut self, slot: u8, pos: Vec<u32>) {
        match self.slots.binary_search(&slot) {
            Ok(idx) => {
                self.positions[idx] = pos;
            }
            Err(idx) => {
                self.slots.insert(idx, slot);
                self.positions.insert(idx, pos);
            }
        }
    }

    pub fn remove(&mut self, slot: u8) -> bool {
        let Ok(idx) = self.slots.binary_search(&slot) else {
            return false;
        };
        self.slots.remove(idx);
        self.positions.remove(idx);
        true
    }

    pub fn encode(&self) -> Vec<u8> {
        debug_assert!(!self.slots.is_empty() && self.slots.len() <= FTS_BLOCK_SIZE);
        let count = self.slots.len();
        let mut offsets: Vec<u32> = Vec::with_capacity(count + 1);
        let mut varints: Vec<u8> = Vec::with_capacity(count * 4);
        offsets.push(0);
        for pos_list in &self.positions {
            encode_delta_varints(pos_list, &mut varints);
            offsets.push(varints.len() as u32);
        }

        let mut buf = Vec::with_capacity(2 + count + (count + 1) * 4 + varints.len());
        buf.extend_from_slice(&(count as u16).to_le_bytes());
        buf.extend_from_slice(&self.slots);
        for off in offsets {
            buf.extend_from_slice(&off.to_le_bytes());
        }
        buf.extend_from_slice(&varints);
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let view = PositionBlockView::parse(bytes)?;
        let count = view.slots.len();
        let mut positions = Vec::with_capacity(count);
        for idx in 0..count {
            let start = u32::from_le_bytes(
                view.pos_offsets_bytes[idx * 4..(idx + 1) * 4]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let end = u32::from_le_bytes(
                view.pos_offsets_bytes[(idx + 1) * 4..(idx + 2) * 4]
                    .try_into()
                    .unwrap(),
            ) as usize;
            if end < start || end > view.delta_varints.len() {
                return Err(TreeError::CorruptedData(
                    "PositionBlock offset out of bounds".into(),
                ));
            }
            positions.push(decode_delta_varints(&view.delta_varints[start..end])?);
        }
        Ok(Self {
            slots: view.slots.to_vec(),
            positions,
        })
    }
}

fn encode_delta_varints(positions: &[u32], out: &mut Vec<u8>) {
    let mut prev = 0u32;
    for &p in positions {
        debug_assert!(p >= prev, "positions must be non-decreasing");
        let delta = p - prev;
        prev = p;
        encode_varint_u32(delta, out);
    }
}

fn decode_delta_varints(bytes: &[u8]) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    let mut off = 0usize;
    let mut acc = 0u32;
    while off < bytes.len() {
        let delta = decode_varint_u32(bytes, &mut off)?;
        acc = acc
            .checked_add(delta)
            .ok_or_else(|| TreeError::CorruptedData("position varint overflow".into()))?;
        out.push(acc);
    }
    Ok(out)
}

#[inline]
fn encode_varint_u32(mut v: u32, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
fn decode_varint_u32(bytes: &[u8], off: &mut usize) -> Result<u32> {
    let mut result = 0u32;
    let mut shift = 0u32;
    loop {
        if *off >= bytes.len() || shift >= 35 {
            return Err(TreeError::CorruptedData(
                "truncated or oversized varint in PositionBlock".into(),
            ));
        }
        let b = bytes[*off];
        *off += 1;
        result |= ((b & 0x7F) as u32) << shift;
        if (b & 0x80) == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fts_meta_and_term_stats_roundtrip() {
        let mut meta = FtsMeta::default();
        assert_eq!(meta.alloc_doc_id().unwrap(), 0);
        assert_eq!(meta.alloc_doc_id().unwrap(), 1);
        meta.per_attr_stats.insert(
            2,
            AttrFtsStats {
                doc_count: 100,
                total_tokens: 4500,
            },
        );
        let bytes = meta.encode();
        assert_eq!(FtsMeta::decode(&bytes).unwrap(), meta);

        let ts = TermStats {
            doc_freq: 42,
            max_tf: 9,
            min_dl: 3,
        };
        assert_eq!(TermStats::decode(&ts.encode()).unwrap(), ts);
    }

    #[test]
    fn test_posting_block_u16_and_u32_roundtrip() {
        let mut blk = PackedPostingBlock::default();
        blk.upsert(10, 3, 25);
        blk.upsert(2, 5, 12);
        blk.upsert(255, 1, 100);
        assert_eq!(blk.slots, vec![2, 10, 255]);
        assert_eq!(blk.max_tf, 5);
        assert_eq!(blk.min_dl, 12);

        let encoded = blk.encode();
        assert_eq!(PostingBlockHeader::parse(&encoded).unwrap().flags, FLAG_U16);
        let decoded = PackedPostingBlock::decode(&encoded).unwrap();
        assert_eq!(decoded, blk);

        // Overflow into u32 mode
        blk.upsert(10, 70_000, 120_000);
        let enc_u32 = blk.encode();
        assert_eq!(PostingBlockHeader::parse(&enc_u32).unwrap().flags, FLAG_U32);
        let dec_u32 = PackedPostingBlock::decode(&enc_u32).unwrap();
        assert_eq!(dec_u32, blk);

        assert!(blk.remove(10));
        assert!(!blk.remove(99));
        assert_eq!(blk.max_tf, 5);
        assert_eq!(blk.min_dl, 12);
    }

    #[test]
    fn test_position_block_roundtrip() {
        let mut pblk = PackedPositionBlock::default();
        pblk.upsert(5, vec![0, 3, 128, 300]);
        pblk.upsert(1, vec![2, 4]);
        let enc = pblk.encode();
        let view = PositionBlockView::parse(&enc).unwrap();
        assert_eq!(view.slot_positions(1).unwrap(), Some(vec![2, 4]));
        assert_eq!(view.slot_positions(5).unwrap(), Some(vec![0, 3, 128, 300]));
        assert_eq!(view.slot_positions(9).unwrap(), None);
        assert_eq!(PackedPositionBlock::decode(&enc).unwrap(), pblk);
    }
}
