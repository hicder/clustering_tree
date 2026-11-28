//! Zero-copy leaf node view.

use std::borrow::Cow;
use izzydb_core::config::{NodeId, RowId};
use izzydb_core::error::{Result, TreeError};

use super::*;

/// Zero-copy borrowed view over serialized [`PackedLeafNode`] bytes (e.g. from `DBPinnableSlice`).
#[derive(Debug, Clone, Copy)]
pub struct LeafNodeView<'a> {
    pub parent_id: Option<NodeId>,
    pub popcount_u32: bool,
    pub slot_cap: usize,
    pub live_count: usize,
    pub stale_hint: u32,
    pub padded_slots: usize,
    pub live_mask_bytes: &'a [u8],
    pub row_ids_bytes: &'a [u8],
    pub codes_bytes: &'a [u8],
    pub res_norm_sqs_bytes: &'a [u8],
    pub scales_bytes: &'a [u8],
    pub centroid_bytes: &'a [u8],
    pub rotated_centroid_bytes: &'a [u8],
    pub code_popcounts_bytes: &'a [u8],
    /// Encoded colocated scalar attribute block (empty when the leaf has no eligible attrs).
    pub attr_bytes: &'a [u8],
}

impl<'a> LeafNodeView<'a> {
    pub fn parse(
        bytes: &'a [u8],
        dim: usize,
        padded_dim: usize,
        rabitq_words: usize,
    ) -> Result<Self> {
        if bytes.len() < LEAF_HEADER_LEN {
            return Err(TreeError::CorruptedData(format!(
                "PackedLeafNode header too short: {}",
                bytes.len()
            )));
        }
        let parent_id = decode_parent_flag(bytes[0], &bytes[8..12])?;
        let popcount_u32 = match bytes[1] {
            0 => false,
            1 => true,
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "invalid popcount_u32 flag: {other}"
                )));
            }
        };
        let mask_words = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        let slot_cap = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let live_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let stale_hint = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        if bytes[20] != LEAF_FORMAT_VERSION {
            return Err(TreeError::CorruptedData(format!(
                "unsupported PackedLeafNode format version {} (expected {LEAF_FORMAT_VERSION}); recreate the collection",
                bytes[20]
            )));
        }
        let attr_len =
            u32::from_le_bytes([bytes[21], bytes[22], bytes[23], 0]) as usize;
        if mask_words != slot_cap.div_ceil(64) || live_count > slot_cap {
            return Err(TreeError::CorruptedData(format!(
                "invalid PackedLeafNode header: slot_cap={slot_cap}, live_count={live_count}, mask_words={mask_words}"
            )));
        }

        let padded_slots = slot_cap.next_multiple_of(8);
        let pop_bytes = padded_slots * if popcount_u32 { 4 } else { 2 };
        let expected = LEAF_HEADER_LEN
            + mask_words * 8
            + padded_slots * (8 + rabitq_words * 8 + 8)
            + (dim + padded_dim) * 4
            + pop_bytes
            + attr_len;
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "PackedLeafNode expected {expected} bytes, got {}",
                bytes.len()
            )));
        }

        let mut off = LEAF_HEADER_LEN;
        let live_mask_bytes = &bytes[off..off + mask_words * 8];
        off += mask_words * 8;
        let row_ids_bytes = &bytes[off..off + padded_slots * 8];
        off += padded_slots * 8;
        let codes_bytes = &bytes[off..off + padded_slots * rabitq_words * 8];
        off += padded_slots * rabitq_words * 8;
        let res_norm_sqs_bytes = &bytes[off..off + padded_slots * 4];
        off += padded_slots * 4;
        let scales_bytes = &bytes[off..off + padded_slots * 4];
        off += padded_slots * 4;
        let centroid_bytes = &bytes[off..off + dim * 4];
        off += dim * 4;
        let rotated_centroid_bytes = &bytes[off..off + padded_dim * 4];
        off += padded_dim * 4;
        let code_popcounts_bytes = &bytes[off..off + pop_bytes];
        off += pop_bytes;
        let attr_bytes = &bytes[off..off + attr_len];

        Ok(Self {
            parent_id,
            popcount_u32,
            slot_cap,
            live_count,
            stale_hint,
            padded_slots,
            live_mask_bytes,
            row_ids_bytes,
            codes_bytes,
            res_norm_sqs_bytes,
            scales_bytes,
            centroid_bytes,
            rotated_centroid_bytes,
            code_popcounts_bytes,
            attr_bytes,
        })
    }

    #[inline]
    pub fn has_tombstones(&self) -> bool {
        self.live_count < self.slot_cap
    }

    #[inline]
    pub fn mask_word_count(&self) -> usize {
        self.live_mask_bytes.len() / 8
    }

    #[inline]
    pub fn copy_live_mask_into(&self, dst: &mut Vec<u64>) {
        let words = self.mask_word_count();
        dst.clear();
        dst.reserve(words);
        for &chunk in self.live_mask_bytes.as_chunks::<8>().0 {
            dst.push(u64::from_le_bytes(chunk));
        }
    }

    #[inline]
    pub fn and_live_mask_into(&self, words_buf: &mut [u64]) {
        for (dst, &chunk) in words_buf
            .iter_mut()
            .zip(self.live_mask_bytes.as_chunks::<8>().0)
        {
            *dst &= u64::from_le_bytes(chunk);
        }
    }

    #[inline]
    pub fn is_live(&self, slot: usize) -> bool {
        if slot >= self.slot_cap {
            return false;
        }
        let w = slot >> 6;
        let word = u64::from_le_bytes(
            self.live_mask_bytes[w * 8..(w + 1) * 8]
                .try_into()
                .unwrap(),
        );
        ((word >> (slot & 63)) & 1) != 0
    }

    #[inline]
    pub fn row_id(&self, slot: usize) -> RowId {
        RowId(u64::from_le_bytes(
            self.row_ids_bytes[slot * 8..(slot + 1) * 8]
                .try_into()
                .unwrap(),
        ))
    }

    pub fn centroid_cow(&self, dim: usize) -> Result<Cow<'a, [f32]>> {
        borrow_f32_slice(self.centroid_bytes, dim)
    }
}

#[inline]
pub(super) fn decode_parent_flag(flag: u8, pid_bytes: &[u8]) -> Result<Option<NodeId>> {
    let raw = u32::from_le_bytes(pid_bytes.try_into().unwrap());
    match flag {
        0 => Ok(None),
        1 => Ok(Some(raw)),
        other => Err(TreeError::CorruptedData(format!(
            "invalid has_parent flag: {other}"
        ))),
    }
}
