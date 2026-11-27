//! Posting blocks (standalone v3 for inline `TermRecord`s and compact v4 for `FtsChunk`s).

use izzydb_core::config::FTS_BLOCK_SIZE;
use izzydb_core::error::{Result, TreeError};

use super::*;

/// 12-byte header of a standalone v3 [`PackedPostingBlock`].
///
/// Stores block cardinality, layout/width flags, and Block-Max bounds for inline blocks
/// embedded directly inside `TermRecord`.
///
/// ```text
/// [count: u16 LE][flags: u16 LE][max_tf: u32 LE][min_dl_over_tf: f32 LE]
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostingBlockHeader {
    /// Number of postings in this block (`1..=256`).
    pub count: u16,
    /// Version (`0x0300`), dense bit (`0x0001`), and `IntWidth` tag bits (`0x0006`).
    pub flags: u16,
    /// Maximum term frequency across all postings in this block.
    pub max_tf: u32,
    /// Sound lower bound on `dl / tf` across all postings in this block.
    pub min_dl_over_tf: f32,
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
        let min_dl_over_tf = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if count == 0 || count as usize > FTS_BLOCK_SIZE {
            return Err(TreeError::CorruptedData(format!(
                "invalid PostingBlock count: {count}"
            )));
        }
        if (flags & POSTING_VERSION_MASK) != POSTING_VERSION_V3 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported PostingBlock flags/version: {flags:#06x}"
            )));
        }
        let unknown_bits = flags & !(POSTING_VERSION_MASK | FLAG_DENSE | WIDTH_MASK);
        if unknown_bits != 0 {
            return Err(TreeError::CorruptedData(format!(
                "unknown PostingBlock flag bits: {flags:#06x}"
            )));
        }
        let _ = IntWidth::from_tag(((flags & WIDTH_MASK) >> WIDTH_SHIFT) as u8)?;
        if min_dl_over_tf.is_nan() || min_dl_over_tf < 0.0 {
            return Err(TreeError::CorruptedData(format!(
                "invalid PostingBlock min_dl_over_tf: {min_dl_over_tf}"
            )));
        }
        Ok(Self {
            count,
            flags,
            max_tf,
            min_dl_over_tf,
        })
    }

    #[inline]
    pub fn is_dense(&self) -> bool {
        (self.flags & FLAG_DENSE) != 0
    }

    #[inline]
    pub fn tf_width(&self) -> IntWidth {
        match (self.flags & WIDTH_MASK) >> WIDTH_SHIFT {
            0 => IntWidth::U8,
            1 => IntWidth::U16,
            _ => IntWidth::U32,
        }
    }
}

/// Decoded payload slice of a [`PostingBlockView`].
#[derive(Debug, Clone, Copy)]
pub enum PostingBlockPayload<'a> {
    Sparse {
        slots: &'a [u8],
        tfs_bytes: &'a [u8],
    },
    Dense {
        bitmap: [u64; 4],
        tfs_bytes: &'a [u8],
    },
}

/// Zero-copy view over an encoded posting block (v3 standalone or v4 chunk-compact).
///
/// Used by BM25 SIMD scorers and FTS filter bitmask evaluation without heap allocation.
///
/// ```text
/// v3:      [12B PostingBlockHeader][Sparse or Dense payload]
/// compact: [1B flags][optional 1B count_minus_1][Sparse or Dense payload]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PostingBlockView<'a> {
    /// Standalone 12-byte v3 header (`None` for chunk-compact blocks whose bounds live in `FtsChunk`).
    pub header: Option<PostingBlockHeader>,
    /// Number of postings in this block (`1..=256`).
    pub count: u16,
    /// Term-frequency payload encoding (`TfOne`, `U2`, `U4`, `U8`, `U16`, or `U32`).
    pub tf_enc: TfEncoding,
    /// Zero-copy sparse slot/tf slices or dense 256-bit bitmap/tf slice.
    pub payload: PostingBlockPayload<'a>,
}

impl<'a> PostingBlockView<'a> {
    /// Parse a standalone 12-byte-header v3 posting block.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let header = PostingBlockHeader::parse(bytes)?;
        let tf_width = header.tf_width();
        let tf_enc = TfEncoding::from_int_width(tf_width);
        let wbytes = tf_width.byte_size();
        let payload = if header.is_dense() {
            let expected = POSTING_BLOCK_HEADER_LEN + 32 + FTS_BLOCK_SIZE * wbytes;
            if bytes.len() != expected {
                return Err(TreeError::CorruptedData(format!(
                    "Dense PostingBlock expected {expected} bytes, got {}",
                    bytes.len()
                )));
            }
            let bm_bytes = &bytes[POSTING_BLOCK_HEADER_LEN..POSTING_BLOCK_HEADER_LEN + 32];
            let bitmap = [
                u64::from_le_bytes(bm_bytes[0..8].try_into().unwrap()),
                u64::from_le_bytes(bm_bytes[8..16].try_into().unwrap()),
                u64::from_le_bytes(bm_bytes[16..24].try_into().unwrap()),
                u64::from_le_bytes(bm_bytes[24..32].try_into().unwrap()),
            ];
            let pop: u32 = bitmap.iter().map(|w| w.count_ones()).sum();
            if pop != header.count as u32 {
                return Err(TreeError::CorruptedData(format!(
                    "Dense PostingBlock bitmap popcount {pop} != header count {}",
                    header.count
                )));
            }
            PostingBlockPayload::Dense {
                bitmap,
                tfs_bytes: &bytes[POSTING_BLOCK_HEADER_LEN + 32..],
            }
        } else {
            let n = header.count as usize;
            let expected = POSTING_BLOCK_HEADER_LEN + n * wbytes + n;
            if bytes.len() != expected {
                return Err(TreeError::CorruptedData(format!(
                    "Sparse PostingBlock expected {expected} bytes, got {}",
                    bytes.len()
                )));
            }
            let tfs_start = POSTING_BLOCK_HEADER_LEN;
            let slots_start = tfs_start + n * wbytes;
            PostingBlockPayload::Sparse {
                tfs_bytes: &bytes[tfs_start..slots_start],
                slots: &bytes[slots_start..],
            }
        };
        Ok(Self {
            header: Some(header),
            count: header.count,
            tf_enc,
            payload,
        })
    }

    /// Parse a compact chunk-embedded v4 posting payload (no duplicated `max_tf` / `min_dl_over_tf`).
    #[inline]
    pub fn parse_compact(bytes: &'a [u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Err(TreeError::CorruptedData("empty compact posting block".into()));
        }
        let flags = bytes[0];
        if (flags & !(COMPACT_FLAG_DENSE | COMPACT_TF_MODE_MASK)) != 0 {
            return Err(TreeError::CorruptedData(format!(
                "unknown compact posting flag bits: {flags:#04x}"
            )));
        }
        let dense = (flags & COMPACT_FLAG_DENSE) != 0;
        let tf_enc = TfEncoding::from_tag((flags & COMPACT_TF_MODE_MASK) >> COMPACT_TF_MODE_SHIFT)?;

        if dense {
            let tf_len = tf_enc.payload_len(FTS_BLOCK_SIZE);
            let expected = 1 + 32 + tf_len;
            if bytes.len() != expected {
                return Err(TreeError::CorruptedData(format!(
                    "Dense compact PostingBlock expected {expected} bytes, got {}",
                    bytes.len()
                )));
            }
            let bm_bytes = &bytes[1..33];
            let bitmap = [
                u64::from_le_bytes(bm_bytes[0..8].try_into().unwrap()),
                u64::from_le_bytes(bm_bytes[8..16].try_into().unwrap()),
                u64::from_le_bytes(bm_bytes[16..24].try_into().unwrap()),
                u64::from_le_bytes(bm_bytes[24..32].try_into().unwrap()),
            ];
            let pop: u32 = bitmap.iter().map(|w| w.count_ones()).sum();
            if pop == 0 || pop as usize > FTS_BLOCK_SIZE {
                return Err(TreeError::CorruptedData(format!(
                    "Dense compact PostingBlock invalid popcount {pop}"
                )));
            }
            return Ok(Self {
                header: None,
                count: pop as u16,
                tf_enc,
                payload: PostingBlockPayload::Dense {
                    bitmap,
                    tfs_bytes: &bytes[33..],
                },
            });
        }

        // Fast path for the dominant cold-term case: sparse `tf == 1` everywhere.
        if tf_enc == TfEncoding::TfOne {
            let n = bytes.len() - 1;
            if n == 0 || n > FTS_BLOCK_SIZE {
                return Err(TreeError::CorruptedData(format!(
                    "Sparse TfOne PostingBlock invalid count {n}"
                )));
            }
            return Ok(Self {
                header: None,
                count: n as u16,
                tf_enc,
                payload: PostingBlockPayload::Sparse {
                    slots: &bytes[1..],
                    tfs_bytes: &[],
                },
            });
        }

        let (n, tfs_start) = match tf_enc {
            TfEncoding::U2 | TfEncoding::U4 => {
                if bytes.len() < 3 {
                    return Err(TreeError::CorruptedData(
                        "Sparse packed compact PostingBlock truncated".into(),
                    ));
                }
                (bytes[1] as usize + 1, 2usize)
            }
            TfEncoding::U8 => {
                let rem = bytes.len() - 1;
                if rem == 0 || rem % 2 != 0 {
                    return Err(TreeError::CorruptedData(format!(
                        "Sparse U8 compact PostingBlock invalid len {}",
                        bytes.len()
                    )));
                }
                (rem / 2, 1usize)
            }
            TfEncoding::U16 => {
                let rem = bytes.len() - 1;
                if rem == 0 || rem % 3 != 0 {
                    return Err(TreeError::CorruptedData(format!(
                        "Sparse U16 compact PostingBlock invalid len {}",
                        bytes.len()
                    )));
                }
                (rem / 3, 1usize)
            }
            TfEncoding::U32 => {
                let rem = bytes.len() - 1;
                if rem == 0 || rem % 5 != 0 {
                    return Err(TreeError::CorruptedData(format!(
                        "Sparse U32 compact PostingBlock invalid len {}",
                        bytes.len()
                    )));
                }
                (rem / 5, 1usize)
            }
            TfEncoding::TfOne => unreachable!(),
        };

        if n == 0 || n > FTS_BLOCK_SIZE {
            return Err(TreeError::CorruptedData(format!(
                "Sparse compact PostingBlock invalid count {n}"
            )));
        }
        let tf_bytes_len = tf_enc.payload_len(n);
        let expected = tfs_start + tf_bytes_len + n;
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "Sparse compact PostingBlock expected {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let slots_start = tfs_start + tf_bytes_len;
        Ok(Self {
            header: None,
            count: n as u16,
            tf_enc,
            payload: PostingBlockPayload::Sparse {
                tfs_bytes: &bytes[tfs_start..slots_start],
                slots: &bytes[slots_start..],
            },
        })
    }

    #[inline]
    pub fn or_slots_into_mask(&self, mask: &mut [u64; 4]) {
        match &self.payload {
            PostingBlockPayload::Sparse { slots, .. } => {
                for &slot in *slots {
                    mask[(slot >> 6) as usize] |= 1u64 << (slot & 63);
                }
            }
            PostingBlockPayload::Dense { bitmap, .. } => {
                for w in 0..4 {
                    mask[w] |= bitmap[w];
                }
            }
        }
    }

    #[inline]
    pub fn for_each_slot(&self, mut f: impl FnMut(u8)) {
        match &self.payload {
            PostingBlockPayload::Sparse { slots, .. } => {
                for &slot in *slots {
                    f(slot);
                }
            }
            PostingBlockPayload::Dense { bitmap, .. } => {
                for (w, &bits) in bitmap.iter().enumerate() {
                    let mut b = bits;
                    while b != 0 {
                        let bit = b.trailing_zeros() as usize;
                        b &= b - 1;
                        f(((w << 6) | bit) as u8);
                    }
                }
            }
        }
    }

    /// Iterate over all `(slot, tf)` pairs in ascending slot order.
    #[inline]
    pub fn for_each_posting(&self, mut f: impl FnMut(u8, u32)) {
        match &self.payload {
            PostingBlockPayload::Sparse { slots, tfs_bytes } => {
                for (i, &slot) in slots.iter().enumerate() {
                    f(slot, read_tf_enc_at(self.tf_enc, tfs_bytes, i));
                }
            }
            PostingBlockPayload::Dense { bitmap, tfs_bytes } => {
                for (w, &bits) in bitmap.iter().enumerate() {
                    let mut b = bits;
                    while b != 0 {
                        let bit = b.trailing_zeros() as usize;
                        b &= b - 1;
                        let slot = ((w << 6) | bit) as u8;
                        f(slot, read_tf_enc_at(self.tf_enc, tfs_bytes, slot as usize));
                    }
                }
            }
        }
    }
}

#[inline]
pub(crate) fn read_tf_at(width: IntWidth, tfs_bytes: &[u8], idx: usize) -> u32 {
    match width {
        IntWidth::U8 => tfs_bytes[idx] as u32,
        IntWidth::U16 => {
            let off = idx * 2;
            u16::from_le_bytes([tfs_bytes[off], tfs_bytes[off + 1]]) as u32
        }
        IntWidth::U32 => {
            let off = idx * 4;
            u32::from_le_bytes(tfs_bytes[off..off + 4].try_into().unwrap())
        }
    }
}

#[inline]
pub(crate) fn read_tf_enc_at(enc: TfEncoding, tfs_bytes: &[u8], idx: usize) -> u32 {
    match enc {
        TfEncoding::TfOne => 1,
        TfEncoding::U2 => (((tfs_bytes[idx >> 2] >> ((idx & 3) * 2)) & 0x03) as u32) + 1,
        TfEncoding::U4 => (((tfs_bytes[idx >> 1] >> ((idx & 1) * 4)) & 0x0F) as u32) + 1,
        TfEncoding::U8 => tfs_bytes[idx] as u32,
        TfEncoding::U16 => {
            let off = idx * 2;
            u16::from_le_bytes([tfs_bytes[off], tfs_bytes[off + 1]]) as u32
        }
        TfEncoding::U32 => {
            let off = idx * 4;
            u32::from_le_bytes(tfs_bytes[off..off + 4].try_into().unwrap())
        }
    }
}

/// Mutable 256-document TF-only posting block for FTS inverted postings.
///
/// Used when building, mutating, compacting, or verifying 256-row posting blocks.
///
/// ```text
/// PackedPostingBlock { slots: [2, 10], tfs: [5, 3], max_tf: 5, min_dl_over_tf: 2.4 }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PackedPostingBlock {
    /// Strictly ascending 0..=255 slot indices within the 256-row block.
    pub slots: Vec<u8>,
    /// Parallel term frequencies for each slot in `slots`.
    pub tfs: Vec<u32>,
    /// Maximum term frequency in `tfs`.
    pub max_tf: u32,
    /// Sound lower bound on `dl / tf` across `(slots, tfs)`.
    pub min_dl_over_tf: f32,
}

impl Default for PackedPostingBlock {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            tfs: Vec::new(),
            max_tf: 0,
            min_dl_over_tf: f32::INFINITY,
        }
    }
}

impl PackedPostingBlock {
    /// Construct a [`PackedPostingBlock`] directly from strictly sorted `(slot, tf)` pairs
    /// using `slot_dl` to compute exact `max_tf` and `min_dl_over_tf` bounds.
    pub fn from_sorted_with_dl(entries: &[(u8, u32)], slot_dl: impl Fn(u8) -> u32) -> Self {
        debug_assert!(entries.len() <= FTS_BLOCK_SIZE);
        debug_assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
        let mut slots = Vec::with_capacity(entries.len());
        let mut tfs = Vec::with_capacity(entries.len());
        for &(s, tf) in entries {
            slots.push(s);
            tfs.push(tf);
        }
        let mut blk = Self {
            slots,
            tfs,
            max_tf: 0,
            min_dl_over_tf: f32::INFINITY,
        };
        blk.recompute_bounds_with(slot_dl);
        blk
    }

    /// Encode sorted `(slot, tf)` pairs in standalone v3 format.
    pub fn encode_from_sorted(entries: &[(u8, u32)], doc_lens: &DocLenBlock) -> Vec<u8> {
        Self::from_sorted_with_dl(entries, |s| doc_lens.get(s)).encode()
    }

    /// Encode sorted `(slot, tf)` pairs directly in compact v4 payload format and return
    /// `(max_tf, min_dl_over_tf, compact_bytes)` without allocating intermediate vectors.
    pub fn encode_compact_from_sorted(
        entries: &[(u8, u32)],
        slot_dl: impl Fn(u8) -> u32,
    ) -> (u32, f32, Vec<u8>) {
        debug_assert!(!entries.is_empty() && entries.len() <= FTS_BLOCK_SIZE);
        debug_assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
        let mut min_tf = u32::MAX;
        let mut max_tf = 0u32;
        let mut min_ratio = f32::INFINITY;
        for &(slot, tf) in entries {
            min_tf = min_tf.min(tf);
            max_tf = max_tf.max(tf);
            let dl = slot_dl(slot);
            min_ratio = min_ratio.min(dl_over_tf_lower_bound(dl, tf));
        }
        let tf_enc = TfEncoding::select(min_tf, max_tf);
        let dense = use_dense_compact(entries.len(), tf_enc);
        let buf = encode_compact_slices(
            entries.len(),
            tf_enc,
            dense,
            entries.iter().map(|&(s, tf)| (s, tf)),
        );
        (max_tf, min_ratio, buf)
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn upsert(&mut self, slot: u8, tf: u32, doc_lens: &DocLenBlock) {
        match self.slots.binary_search(&slot) {
            Ok(idx) => {
                self.tfs[idx] = tf;
            }
            Err(idx) => {
                self.slots.insert(idx, slot);
                self.tfs.insert(idx, tf);
            }
        }
        self.recompute_bounds(doc_lens);
    }

    pub fn remove(&mut self, slot: u8, doc_lens: &DocLenBlock) -> bool {
        let Ok(idx) = self.slots.binary_search(&slot) else {
            return false;
        };
        self.slots.remove(idx);
        self.tfs.remove(idx);
        self.recompute_bounds(doc_lens);
        true
    }

    pub fn recompute_bounds(&mut self, doc_lens: &DocLenBlock) {
        self.recompute_bounds_with(|slot| doc_lens.get(slot));
    }

    pub fn recompute_bounds_with(&mut self, slot_dl: impl Fn(u8) -> u32) {
        if self.slots.is_empty() {
            self.max_tf = 0;
            self.min_dl_over_tf = f32::INFINITY;
            return;
        }
        let mut max_tf = 0u32;
        let mut min_ratio = f32::INFINITY;
        for (&slot, &tf) in self.slots.iter().zip(self.tfs.iter()) {
            if tf > max_tf {
                max_tf = tf;
            }
            let dl = slot_dl(slot);
            let ratio = dl_over_tf_lower_bound(dl, tf);
            if ratio < min_ratio {
                min_ratio = ratio;
            }
        }
        self.max_tf = max_tf;
        self.min_dl_over_tf = min_ratio;
    }

    /// Encode in standalone v3 format (12-byte header + payload) for inline `TermRecord`s.
    pub fn encode(&self) -> Vec<u8> {
        debug_assert!(!self.slots.is_empty() && self.slots.len() <= FTS_BLOCK_SIZE);
        let n = self.slots.len();
        let dense = n >= POSTING_DENSE_THRESHOLD;
        let max_tf = self.tfs.iter().copied().max().unwrap_or(0);
        let tf_width = IntWidth::min_for_max(max_tf);
        let flags = POSTING_VERSION_V3
            | ((tf_width as u16) << WIDTH_SHIFT)
            | if dense { FLAG_DENSE } else { 0 };

        if dense {
            let mut bitmap = [0u64; 4];
            let mut dense_tfs = [0u32; FTS_BLOCK_SIZE];
            for (&slot, &tf) in self.slots.iter().zip(self.tfs.iter()) {
                bitmap[(slot >> 6) as usize] |= 1u64 << (slot & 63);
                dense_tfs[slot as usize] = tf;
            }
            let mut buf = Vec::with_capacity(
                POSTING_BLOCK_HEADER_LEN + 32 + FTS_BLOCK_SIZE * tf_width.byte_size(),
            );
            buf.extend_from_slice(&(n as u16).to_le_bytes());
            buf.extend_from_slice(&flags.to_le_bytes());
            buf.extend_from_slice(&self.max_tf.to_le_bytes());
            buf.extend_from_slice(&self.min_dl_over_tf.to_le_bytes());
            for w in bitmap {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            match tf_width {
                IntWidth::U8 => {
                    for &tf in &dense_tfs {
                        buf.push(tf as u8);
                    }
                }
                IntWidth::U16 => {
                    for &tf in &dense_tfs {
                        buf.extend_from_slice(&(tf as u16).to_le_bytes());
                    }
                }
                IntWidth::U32 => {
                    for &tf in &dense_tfs {
                        buf.extend_from_slice(&tf.to_le_bytes());
                    }
                }
            }
            buf
        } else {
            let mut buf =
                Vec::with_capacity(POSTING_BLOCK_HEADER_LEN + n * tf_width.byte_size() + n);
            buf.extend_from_slice(&(n as u16).to_le_bytes());
            buf.extend_from_slice(&flags.to_le_bytes());
            buf.extend_from_slice(&self.max_tf.to_le_bytes());
            buf.extend_from_slice(&self.min_dl_over_tf.to_le_bytes());
            match tf_width {
                IntWidth::U8 => {
                    for &tf in &self.tfs {
                        buf.push(tf as u8);
                    }
                }
                IntWidth::U16 => {
                    for &tf in &self.tfs {
                        buf.extend_from_slice(&(tf as u16).to_le_bytes());
                    }
                }
                IntWidth::U32 => {
                    for &tf in &self.tfs {
                        buf.extend_from_slice(&tf.to_le_bytes());
                    }
                }
            }
            buf.extend_from_slice(&self.slots);
            buf
        }
    }

    /// Encode in compact v4 chunk-embedded format (omitting bounds stored in the chunk skip table).
    pub fn encode_compact(&self) -> Vec<u8> {
        debug_assert!(!self.slots.is_empty() && self.slots.len() <= FTS_BLOCK_SIZE);
        let mut min_tf = u32::MAX;
        let mut max_tf = 0u32;
        for &tf in &self.tfs {
            min_tf = min_tf.min(tf);
            max_tf = max_tf.max(tf);
        }
        let tf_enc = TfEncoding::select(min_tf, max_tf);
        let dense = use_dense_compact(self.slots.len(), tf_enc);
        encode_compact_slices(
            self.slots.len(),
            tf_enc,
            dense,
            self.slots.iter().copied().zip(self.tfs.iter().copied()),
        )
    }

    /// Decode a standalone v3 posting block.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let view = PostingBlockView::parse(bytes)?;
        let hdr = view.header.expect("v3 PostingBlockView has header");
        Self::from_view(&view, hdr.max_tf, hdr.min_dl_over_tf)
    }

    /// Decode a compact v4 chunk-embedded posting block given its bounds from the chunk skip table.
    pub fn decode_compact(bytes: &[u8], max_tf: u32, min_dl_over_tf: f32) -> Result<Self> {
        let view = PostingBlockView::parse_compact(bytes)?;
        Self::from_view(&view, max_tf, min_dl_over_tf)
    }

    fn from_view(view: &PostingBlockView<'_>, max_tf: u32, min_dl_over_tf: f32) -> Result<Self> {
        let n = view.count as usize;
        let mut slots = Vec::with_capacity(n);
        let mut tfs = Vec::with_capacity(n);
        match view.payload {
            PostingBlockPayload::Sparse {
                slots: s,
                tfs_bytes,
            } => {
                slots.extend_from_slice(s);
                for i in 0..n {
                    tfs.push(read_tf_enc_at(view.tf_enc, tfs_bytes, i));
                }
            }
            PostingBlockPayload::Dense { bitmap, tfs_bytes } => {
                for (w, &bits) in bitmap.iter().enumerate() {
                    let mut b = bits;
                    while b != 0 {
                        let bit = b.trailing_zeros() as usize;
                        b &= b - 1;
                        let slot = ((w << 6) | bit) as u8;
                        slots.push(slot);
                        tfs.push(read_tf_enc_at(view.tf_enc, tfs_bytes, slot as usize));
                    }
                }
            }
        }
        Ok(Self {
            slots,
            tfs,
            max_tf,
            min_dl_over_tf,
        })
    }
}

#[inline]
fn use_dense_compact(n: usize, tf_enc: TfEncoding) -> bool {
    match tf_enc {
        TfEncoding::TfOne => n >= 33,
        TfEncoding::U2 => n >= 76,
        _ => n >= POSTING_DENSE_THRESHOLD,
    }
}

fn encode_compact_slices(
    n: usize,
    tf_enc: TfEncoding,
    dense: bool,
    entries: impl Iterator<Item = (u8, u32)>,
) -> Vec<u8> {
    let flags = ((tf_enc as u8) << COMPACT_TF_MODE_SHIFT)
        | if dense { COMPACT_FLAG_DENSE } else { 0 };

    if dense {
        let mut bitmap = [0u64; 4];
        let mut dense_tfs = [0u32; FTS_BLOCK_SIZE];
        for (slot, tf) in entries {
            bitmap[(slot >> 6) as usize] |= 1u64 << (slot & 63);
            dense_tfs[slot as usize] = tf;
        }
        let tf_len = tf_enc.payload_len(FTS_BLOCK_SIZE);
        let mut buf = Vec::with_capacity(1 + 32 + tf_len);
        buf.push(flags);
        for w in bitmap {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        match tf_enc {
            TfEncoding::TfOne => {}
            TfEncoding::U2 => {
                for chunk in dense_tfs.chunks_exact(4) {
                    let a = chunk[0].saturating_sub(1) & 0x03;
                    let b = chunk[1].saturating_sub(1) & 0x03;
                    let c = chunk[2].saturating_sub(1) & 0x03;
                    let d = chunk[3].saturating_sub(1) & 0x03;
                    buf.push((a | (b << 2) | (c << 4) | (d << 6)) as u8);
                }
            }
            TfEncoding::U4 => {
                for chunk in dense_tfs.chunks_exact(2) {
                    let a = chunk[0].saturating_sub(1) & 0x0F;
                    let b = chunk[1].saturating_sub(1) & 0x0F;
                    buf.push((a | (b << 4)) as u8);
                }
            }
            TfEncoding::U8 => {
                for &tf in &dense_tfs {
                    buf.push(tf as u8);
                }
            }
            TfEncoding::U16 => {
                for &tf in &dense_tfs {
                    buf.extend_from_slice(&(tf as u16).to_le_bytes());
                }
            }
            TfEncoding::U32 => {
                for &tf in &dense_tfs {
                    buf.extend_from_slice(&tf.to_le_bytes());
                }
            }
        }
        buf
    } else {
        let tf_len = tf_enc.payload_len(n);
        let extra = usize::from(matches!(tf_enc, TfEncoding::U2 | TfEncoding::U4));
        let mut buf = Vec::with_capacity(1 + extra + tf_len + n);
        buf.push(flags);
        if extra == 1 {
            buf.push((n - 1) as u8);
        }
        let mut slots = Vec::with_capacity(n);
        match tf_enc {
            TfEncoding::TfOne => {
                for (slot, _) in entries {
                    slots.push(slot);
                }
            }
            TfEncoding::U2 => {
                let mut acc = 0u8;
                let mut shift = 0u8;
                for (slot, tf) in entries {
                    slots.push(slot);
                    acc |= (((tf - 1) & 0x03) as u8) << shift;
                    shift += 2;
                    if shift == 8 {
                        buf.push(acc);
                        acc = 0;
                        shift = 0;
                    }
                }
                if shift > 0 {
                    buf.push(acc);
                }
            }
            TfEncoding::U4 => {
                let mut acc = 0u8;
                let mut high = false;
                for (slot, tf) in entries {
                    slots.push(slot);
                    let nib = ((tf - 1) & 0x0F) as u8;
                    if !high {
                        acc = nib;
                        high = true;
                    } else {
                        buf.push(acc | (nib << 4));
                        acc = 0;
                        high = false;
                    }
                }
                if high {
                    buf.push(acc);
                }
            }
            TfEncoding::U8 => {
                for (slot, tf) in entries {
                    slots.push(slot);
                    buf.push(tf as u8);
                }
            }
            TfEncoding::U16 => {
                for (slot, tf) in entries {
                    slots.push(slot);
                    buf.extend_from_slice(&(tf as u16).to_le_bytes());
                }
            }
            TfEncoding::U32 => {
                for (slot, tf) in entries {
                    slots.push(slot);
                    buf.extend_from_slice(&tf.to_le_bytes());
                }
            }
        }
        buf.extend_from_slice(&slots);
        buf
    }
}
