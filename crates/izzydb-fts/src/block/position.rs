//! Position blocks and varint helpers.

use izzydb_core::config::FTS_BLOCK_SIZE;
use izzydb_core::error::{Result, TreeError};

use super::*;

/// Zero-copy view over an encoded [`PackedPositionBlock`].
#[derive(Debug, Clone, Copy)]
pub struct PositionBlockView<'a> {
    pub u32_offsets: bool,
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
        let flags = bytes[0];
        if (flags & !POS_FLAG_U32_OFFSETS) != 0 {
            return Err(TreeError::CorruptedData(format!(
                "unknown PositionBlock flags: {flags:#04x}"
            )));
        }
        let u32_offsets = (flags & POS_FLAG_U32_OFFSETS) != 0;
        let off_width = if u32_offsets { 4usize } else { 2usize };
        let count = (bytes[1] as usize) + 1;
        let slots_start = 2;
        let offsets_start = slots_start + count;
        let varints_start = offsets_start + (count + 1) * off_width;
        if bytes.len() < varints_start {
            return Err(TreeError::CorruptedData(
                "PositionBlock offsets truncated".into(),
            ));
        }
        let slots = &bytes[slots_start..offsets_start];
        let pos_offsets_bytes = &bytes[offsets_start..varints_start];
        let delta_varints = &bytes[varints_start..];
        let last_off = read_pos_offset(u32_offsets, pos_offsets_bytes, count);
        if last_off != delta_varints.len() {
            return Err(TreeError::CorruptedData(format!(
                "PositionBlock varint length mismatch: header {last_off} vs actual {}",
                delta_varints.len()
            )));
        }
        Ok(Self {
            u32_offsets,
            slots,
            pos_offsets_bytes,
            delta_varints,
        })
    }

    #[inline]
    pub fn slot_positions(&self, slot: u8) -> Result<Option<Vec<u32>>> {
        let Ok(idx) = self.slots.binary_search(&slot) else {
            return Ok(None);
        };
        let start = read_pos_offset(self.u32_offsets, self.pos_offsets_bytes, idx);
        let end = read_pos_offset(self.u32_offsets, self.pos_offsets_bytes, idx + 1);
        if end < start || end > self.delta_varints.len() {
            return Err(TreeError::CorruptedData(
                "PositionBlock offset out of bounds".into(),
            ));
        }
        decode_delta_varints(&self.delta_varints[start..end]).map(Some)
    }

    /// Check whether `slot` contains `target_pos` without allocating a `Vec<u32>`.
    #[inline]
    pub fn slot_contains_pos(&self, slot: u8, target_pos: u32) -> Result<bool> {
        let Ok(idx) = self.slots.binary_search(&slot) else {
            return Ok(false);
        };
        let start = read_pos_offset(self.u32_offsets, self.pos_offsets_bytes, idx);
        let end = read_pos_offset(self.u32_offsets, self.pos_offsets_bytes, idx + 1);
        if end < start || end > self.delta_varints.len() {
            return Err(TreeError::CorruptedData(
                "PositionBlock offset out of bounds".into(),
            ));
        }
        let slice = &self.delta_varints[start..end];
        let mut off = 0usize;
        let mut acc = 0u32;
        while off < slice.len() {
            let delta = decode_varint_u32(slice, &mut off)?;
            acc = acc
                .checked_add(delta)
                .ok_or_else(|| TreeError::CorruptedData("position varint overflow".into()))?;
            if acc == target_pos {
                return Ok(true);
            }
            if acc > target_pos {
                return Ok(false);
            }
        }
        Ok(false)
    }
}

#[inline]
pub(super) fn read_pos_offset(u32_offsets: bool, bytes: &[u8], idx: usize) -> usize {
    if u32_offsets {
        let off = idx * 4;
        u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize
    } else {
        let off = idx * 2;
        u16::from_le_bytes([bytes[off], bytes[off + 1]]) as usize
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

    /// Encode sorted `(slot, positions)` slices directly into `cf_fts_positions` bytes.
    pub fn encode_from_sorted_slices(entries: &[(u8, &[u32])]) -> Vec<u8> {
        debug_assert!(!entries.is_empty() && entries.len() <= FTS_BLOCK_SIZE);
        debug_assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
        let count = entries.len();
        let mut offsets: Vec<u32> = Vec::with_capacity(count + 1);
        let mut varints: Vec<u8> = Vec::with_capacity(count * 4);
        offsets.push(0);
        for &(_, pos_list) in entries {
            encode_delta_varints(pos_list, &mut varints);
            offsets.push(varints.len() as u32);
        }
        finish_position_block_encode(entries.iter().map(|&(s, _)| s), &offsets, &varints)
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
        finish_position_block_encode(self.slots.iter().copied(), &offsets, &varints)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let view = PositionBlockView::parse(bytes)?;
        let count = view.slots.len();
        let mut positions = Vec::with_capacity(count);
        for idx in 0..count {
            let start = read_pos_offset(view.u32_offsets, view.pos_offsets_bytes, idx);
            let end = read_pos_offset(view.u32_offsets, view.pos_offsets_bytes, idx + 1);
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

pub(super) fn finish_position_block_encode(
    slots: impl ExactSizeIterator<Item = u8>,
    offsets: &[u32],
    varints: &[u8],
) -> Vec<u8> {
    let count = slots.len();
    let u32_offsets = varints.len() > u16::MAX as usize;
    let off_width = if u32_offsets { 4usize } else { 2usize };
    let flags = if u32_offsets { POS_FLAG_U32_OFFSETS } else { 0 };
    let mut buf = Vec::with_capacity(2 + count + (count + 1) * off_width + varints.len());
    buf.push(flags);
    buf.push((count - 1) as u8);
    buf.extend(slots);
    if u32_offsets {
        for &off in offsets {
            buf.extend_from_slice(&off.to_le_bytes());
        }
    } else {
        for &off in offsets {
            buf.extend_from_slice(&(off as u16).to_le_bytes());
        }
    }
    buf.extend_from_slice(varints);
    buf
}

pub(super) fn encode_delta_varints(positions: &[u32], out: &mut Vec<u8>) {
    let mut prev = 0u32;
    for &p in positions {
        debug_assert!(p >= prev, "positions must be non-decreasing");
        let delta = p - prev;
        prev = p;
        encode_varint_u32(delta, out);
    }
}

pub(super) fn decode_delta_varints(bytes: &[u8]) -> Result<Vec<u32>> {
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
pub(super) fn encode_varint_u32(mut v: u32, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
pub(super) fn decode_varint_u32(bytes: &[u8], off: &mut usize) -> Result<u32> {
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
