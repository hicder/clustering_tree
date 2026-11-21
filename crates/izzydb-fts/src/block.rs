use std::collections::BTreeMap;

use izzydb_core::config::{AttrId, FTS_BLOCK_SIZE, FtsDocId};
use izzydb_core::error::{Result, TreeError};

pub const POSTING_BLOCK_HEADER_LEN: usize = 12;
pub const DOC_LEN_BLOCK_HEADER_LEN: usize = 4;
pub const TERM_STATS_LEN: usize = 16;

/// Minimum posting count in a 256-doc block at which v2 encoding switches from Sparse to Dense.
pub const POSTING_DENSE_THRESHOLD: usize = 96;

const POSTING_VERSION_V2: u16 = 0x0200;
const POSTING_VERSION_MASK: u16 = 0xFF00;
const FLAG_DENSE: u16 = 0x0001;
const WIDTH_SHIFT: u16 = 1;
const WIDTH_MASK: u16 = 0x0006;

/// Compact integer width tag shared by `PackedPostingBlock` (`tf`) and `DocLenBlock` (`dl`).
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

/// 12-byte header of a v2 [`PackedPostingBlock`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostingBlockHeader {
    pub count: u16,
    pub flags: u16,
    pub max_tf: u32,
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
        if (flags & POSTING_VERSION_MASK) != POSTING_VERSION_V2 {
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

/// Decoded payload slice of a v2 [`PostingBlockView`].
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

/// Zero-copy view over an encoded v2 [`PackedPostingBlock`].
#[derive(Debug, Clone, Copy)]
pub struct PostingBlockView<'a> {
    pub header: PostingBlockHeader,
    pub tf_width: IntWidth,
    pub payload: PostingBlockPayload<'a>,
}

impl<'a> PostingBlockView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let header = PostingBlockHeader::parse(bytes)?;
        let tf_width = header.tf_width();
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
            let expected = POSTING_BLOCK_HEADER_LEN + n + n * wbytes;
            if bytes.len() != expected {
                return Err(TreeError::CorruptedData(format!(
                    "Sparse PostingBlock expected {expected} bytes, got {}",
                    bytes.len()
                )));
            }
            let slots_start = POSTING_BLOCK_HEADER_LEN;
            let tfs_start = slots_start + n;
            PostingBlockPayload::Sparse {
                slots: &bytes[slots_start..tfs_start],
                tfs_bytes: &bytes[tfs_start..],
            }
        };
        Ok(Self {
            header,
            tf_width,
            payload,
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

/// Mutable 256-document TF-only posting block for `cf_fts_postings`.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedPostingBlock {
    pub slots: Vec<u8>,
    pub tfs: Vec<u32>,
    pub max_tf: u32,
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
            let dl = doc_lens.get(slot);
            let ratio = dl_over_tf_lower_bound(dl, tf);
            if ratio < min_ratio {
                min_ratio = ratio;
            }
        }
        self.max_tf = max_tf;
        self.min_dl_over_tf = min_ratio;
    }

    pub fn encode(&self) -> Vec<u8> {
        debug_assert!(!self.slots.is_empty() && self.slots.len() <= FTS_BLOCK_SIZE);
        let n = self.slots.len();
        let dense = n >= POSTING_DENSE_THRESHOLD;
        let max_tf = self.tfs.iter().copied().max().unwrap_or(0);
        let tf_width = IntWidth::min_for_max(max_tf);
        let flags = POSTING_VERSION_V2
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
                Vec::with_capacity(POSTING_BLOCK_HEADER_LEN + n + n * tf_width.byte_size());
            buf.extend_from_slice(&(n as u16).to_le_bytes());
            buf.extend_from_slice(&flags.to_le_bytes());
            buf.extend_from_slice(&self.max_tf.to_le_bytes());
            buf.extend_from_slice(&self.min_dl_over_tf.to_le_bytes());
            buf.extend_from_slice(&self.slots);
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
            buf
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let view = PostingBlockView::parse(bytes)?;
        let n = view.header.count as usize;
        let mut slots = Vec::with_capacity(n);
        let mut tfs = Vec::with_capacity(n);
        match view.payload {
            PostingBlockPayload::Sparse {
                slots: s,
                tfs_bytes,
            } => {
                slots.extend_from_slice(s);
                for i in 0..n {
                    tfs.push(read_tf_at(view.tf_width, tfs_bytes, i));
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
                        tfs.push(read_tf_at(view.tf_width, tfs_bytes, slot as usize));
                    }
                }
            }
        }
        Ok(Self {
            slots,
            tfs,
            max_tf: view.header.max_tf,
            min_dl_over_tf: view.header.min_dl_over_tf,
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

        let mut ts = TermStats::default();
        assert_eq!(ts.min_dl_over_tf, f32::INFINITY);
        ts.doc_freq = 42;
        ts.observe(9, 27);
        ts.observe(3, 12);
        assert_eq!(ts.max_tf, 9);
        assert!((ts.min_dl_over_tf - 3.0).abs() < 1e-6);
        assert_eq!(TermStats::decode(&ts.encode()).unwrap(), ts);
    }

    #[test]
    fn test_doc_len_block_widths_roundtrip() {
        for max_dl in [10u32, 300, 100_000] {
            let mut dls = DocLenBlock::default();
            dls.set(0, 1);
            dls.set(17, max_dl);
            dls.set(255, 5);
            assert_eq!(dls.count, 3);

            let enc = dls.encode();
            let dec = DocLenBlock::decode(&enc).unwrap();
            assert_eq!(dec, dls);

            dls.remove(17);
            assert_eq!(dls.count, 2);
            assert_eq!(dls.get(17), 0);
        }
    }

    #[test]
    fn test_posting_block_sparse_dense_and_widths_roundtrip() {
        let mut dls = DocLenBlock::default();
        dls.set(2, 12);
        dls.set(10, 25);
        dls.set(255, 100);

        let mut blk = PackedPostingBlock::default();
        blk.upsert(10, 3, &dls);
        blk.upsert(2, 5, &dls);
        blk.upsert(255, 1, &dls);
        assert_eq!(blk.slots, vec![2, 10, 255]);
        assert_eq!(blk.max_tf, 5);
        assert!((blk.min_dl_over_tf - (12.0 / 5.0)).abs() < 1e-5);

        let enc_u8 = blk.encode();
        let hdr_u8 = PostingBlockHeader::parse(&enc_u8).unwrap();
        assert!(!hdr_u8.is_dense());
        assert_eq!(hdr_u8.tf_width(), IntWidth::U8);
        assert_eq!(PackedPostingBlock::decode(&enc_u8).unwrap(), blk);

        // Overflow into u16 and u32
        dls.set(10, 120_000);
        blk.upsert(10, 500, &dls);
        let enc_u16 = blk.encode();
        assert_eq!(PostingBlockHeader::parse(&enc_u16).unwrap().tf_width(), IntWidth::U16);
        assert_eq!(PackedPostingBlock::decode(&enc_u16).unwrap(), blk);

        blk.upsert(10, 70_000, &dls);
        let enc_u32 = blk.encode();
        assert_eq!(PostingBlockHeader::parse(&enc_u32).unwrap().tf_width(), IntWidth::U32);
        assert_eq!(PackedPostingBlock::decode(&enc_u32).unwrap(), blk);

        assert!(blk.remove(10, &dls));
        assert!(!blk.remove(99, &dls));
        assert_eq!(blk.max_tf, 5);
        assert!((blk.min_dl_over_tf - (12.0 / 5.0)).abs() < 1e-5);

        // Dense layout (>= POSTING_DENSE_THRESHOLD)
        let mut dense_dls = DocLenBlock::default();
        let mut dense_blk = PackedPostingBlock::default();
        for s in (0..=254u8).step_by(2) {
            dense_dls.set(s, 20 + s as u32);
            dense_blk.upsert(s, 1 + (s % 5) as u32, &dense_dls);
        }
        assert!(dense_blk.len() >= POSTING_DENSE_THRESHOLD);
        let enc_dense = dense_blk.encode();
        let hdr_dense = PostingBlockHeader::parse(&enc_dense).unwrap();
        assert!(hdr_dense.is_dense());
        assert_eq!(hdr_dense.tf_width(), IntWidth::U8);
        assert_eq!(PackedPostingBlock::decode(&enc_dense).unwrap(), dense_blk);
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
