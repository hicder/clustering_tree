use super::*;
use izzydb_core::config::LocalId;
use izzydb_core::error::{Result, TreeError};
use roaring::RoaringBitmap;

// Roaring serialization format constants (https://github.com/RoaringBitmap/RoaringFormatSpec).
const ROARING_COOKIE_NO_RUNCONTAINER: u32 = 12346;
const ROARING_COOKIE: u16 = 12347;
const ROARING_NO_OFFSET_THRESHOLD: usize = 4;
const ROARING_ARRAY_MAX_CARDINALITY: u32 = 4096;

/// Number of `u64` words in one 65,536-bit chunk of a [`SegBitsetAccum`].
pub const SEG_BITSET_WORDS: usize = 1024;
/// Number of local ids covered by one [`SegBitsetAccum`] chunk.
pub const SEG_BITSET_BITS: usize = SEG_BITSET_WORDS * 64;

/// Flat 65,536-bit-per-chunk accumulator for OR-ing segment postings.
///
/// Range, `In`, and `Or` filters union many small postings into one bitmap. Promoting directly
/// to a flat bitset avoids the `Vec<u16>` merge cost that Roaring pays before its containers
/// promote to a bitmap. Chunks are allocated lazily for larger `seal_rows` values.
///
/// ```text
/// let mut acc = SegBitsetAccum::new();
/// acc.or_view(&SegPostingView::parse(bytes)?)?;
/// let bitmap = acc.into_roaring();
/// ```
#[derive(Debug, Clone, Default)]
pub struct SegBitsetAccum {
    chunks: Vec<[u64; SEG_BITSET_WORDS]>,
}

impl SegBitsetAccum {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chunks.iter().all(|c| c.iter().all(|&w| w == 0))
    }

    #[inline]
    pub fn contains(&self, local: LocalId) -> bool {
        let key = (local as usize) / SEG_BITSET_BITS;
        let bit = (local as usize) % SEG_BITSET_BITS;
        self.chunks
            .get(key)
            .is_some_and(|c| (c[bit >> 6] >> (bit & 63)) & 1 != 0)
    }

    #[inline]
    fn chunk_mut(&mut self, key: usize) -> &mut [u64; SEG_BITSET_WORDS] {
        if key >= self.chunks.len() {
            self.chunks
                .resize_with(key + 1, || [0u64; SEG_BITSET_WORDS]);
        }
        &mut self.chunks[key]
    }

    #[inline]
    pub fn insert(&mut self, local: LocalId) {
        let key = (local as usize) / SEG_BITSET_BITS;
        let bit = (local as usize) % SEG_BITSET_BITS;
        let chunk = self.chunk_mut(key);
        chunk[bit >> 6] |= 1u64 << (bit & 63);
    }

    /// Set bits `[start, end)`.
    pub fn insert_range(&mut self, start: u32, end: u32) {
        if start >= end {
            return;
        }
        let mut start = start as usize;
        let end = end as usize;
        while start < end {
            let key = start / SEG_BITSET_BITS;
            let key_end = ((key + 1) * SEG_BITSET_BITS).min(end);
            let begin = start % SEG_BITSET_BITS;
            let stop = key_end - key * SEG_BITSET_BITS;
            let chunk = self.chunk_mut(key);
            let first_word = begin >> 6;
            let last_word = (stop - 1) >> 6;
            if first_word == last_word {
                chunk[first_word] |= mask_bits(begin & 63, stop & 63);
            } else {
                chunk[first_word] |= !0u64 << (begin & 63);
                for word in &mut chunk[first_word + 1..last_word] {
                    *word = !0u64;
                }
                let tail = stop & 63;
                chunk[last_word] |= if tail == 0 { !0u64 } else { (1u64 << tail) - 1 };
            }
            start = key_end;
        }
    }

    /// OR a borrowed (possibly raw Roaring) posting directly into the flat bitset.
    pub fn or_view(&mut self, view: &SegPostingView<'_>) -> Result<()> {
        match *view {
            SegPostingView::Sparse(slice) => {
                for &chunk in slice.as_chunks::<4>().0 {
                    self.insert(u32::from_le_bytes(chunk));
                }
                Ok(())
            }
            SegPostingView::Roaring(slice) => {
                let layout = RoaringLayout::parse(slice)?;
                for idx in 0..layout.size {
                    let (key, container) = layout.container(idx)?;
                    let base = key as usize * SEG_BITSET_BITS;
                    match container {
                        RoaringContainer::Run(payload) => {
                            let runs = payload.len() / 4;
                            for run in payload.as_chunks::<4>().0.iter().take(runs) {
                                let start = u16::from_le_bytes([run[0], run[1]]) as u32;
                                let len = u16::from_le_bytes([run[2], run[3]]) as u32;
                                let abs = (base + start as usize) as u32;
                                self.insert_range(abs, abs + len + 1);
                            }
                        }
                        RoaringContainer::Bitmap(payload) => {
                            let chunk = self.chunk_mut(key as usize);
                            for (dst, word) in
                                chunk.iter_mut().zip(payload.as_chunks::<8>().0.iter())
                            {
                                *dst |= u64::from_le_bytes(*word);
                            }
                        }
                        RoaringContainer::Array(payload) => {
                            let chunk = self.chunk_mut(key as usize);
                            for &value in payload.as_chunks::<2>().0 {
                                let bit = u16::from_le_bytes(value) as usize;
                                chunk[bit >> 6] |= 1u64 << (bit & 63);
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// OR an in-memory bitmap into the flat bitset.
    pub fn or_bitmap(&mut self, bitmap: &RoaringBitmap) {
        for local in bitmap.iter() {
            self.insert(local);
        }
    }

    /// Convert to a Roaring bitmap via a pre-built standard Roaring serialization buffer.
    ///
    /// Assembling the serialized form directly from the flat words avoids one `RoaringBitmap`
    /// container push (and array-to-bitmap promotion) per set bit. The bytes are produced by
    /// this crate on the read path, so the unchecked decoder is safe to use.
    pub fn into_roaring(self) -> RoaringBitmap {
        let containers: Vec<(u16, u32)> = self
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(idx, chunk)| {
                let card: u32 = chunk.iter().map(|word| word.count_ones()).sum();
                (card > 0).then_some((idx as u16, card))
            })
            .collect();
        if containers.is_empty() {
            return RoaringBitmap::new();
        }

        let size = containers.len();
        let data_len: usize = containers
            .iter()
            .map(|&(_, card)| {
                if card > ROARING_ARRAY_MAX_CARDINALITY {
                    8192
                } else {
                    card as usize * 2
                }
            })
            .sum();
        let mut out = Vec::with_capacity(8 + size * 8 + data_len);
        out.extend_from_slice(&ROARING_COOKIE_NO_RUNCONTAINER.to_le_bytes());
        out.extend_from_slice(&(size as u32).to_le_bytes());
        for &(key, card) in &containers {
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&((card - 1) as u16).to_le_bytes());
        }
        let mut offset = (8 + size * 8) as u32;
        for &(_, card) in &containers {
            out.extend_from_slice(&offset.to_le_bytes());
            offset += if card > ROARING_ARRAY_MAX_CARDINALITY {
                8192
            } else {
                card * 2
            };
        }
        for &(key, card) in &containers {
            let chunk = &self.chunks[key as usize];
            if card > ROARING_ARRAY_MAX_CARDINALITY {
                for word in chunk {
                    out.extend_from_slice(&word.to_le_bytes());
                }
            } else {
                for (word_idx, &word) in chunk.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as u16;
                        bits &= bits - 1;
                        out.extend_from_slice(&((word_idx as u16) * 64 + bit).to_le_bytes());
                    }
                }
            }
        }
        RoaringBitmap::deserialize_unchecked_from(&out[..])
            .expect("SegBitsetAccum serializes a valid Roaring bitmap")
    }
}

#[inline]
fn mask_bits(start: usize, end: usize) -> u64 {
    let tail = if end == 0 { 64 } else { end };
    if start == 0 && tail == 64 {
        !0u64
    } else {
        (!0u64 << start) & ((1u64 << tail) - 1)
    }
}

/// Borrowed zero-copy layout of a standard Roaring serialized bitmap.
struct RoaringLayout<'a> {
    size: usize,
    has_runs: bool,
    run_bits: &'a [u8],
    descriptions: &'a [u8],
    data: &'a [u8],
}

/// Borrowed view of one Roaring container payload.
enum RoaringContainer<'a> {
    Run(&'a [u8]),
    Array(&'a [u8]),
    Bitmap(&'a [u8]),
}

impl<'a> RoaringLayout<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(TreeError::CorruptedData("roaring header truncated".into()));
        }
        let cookie = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let (size, has_runs, mut off) = if cookie == ROARING_COOKIE_NO_RUNCONTAINER {
            let size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
            (size, false, 8)
        } else if cookie as u16 == ROARING_COOKIE {
            let size = ((cookie >> 16) + 1) as usize;
            (size, true, 4)
        } else {
            return Err(TreeError::CorruptedData(format!(
                "unknown roaring cookie: {cookie}"
            )));
        };
        if size > u16::MAX as usize + 1 {
            return Err(TreeError::CorruptedData(
                "roaring container count exceeds 65,536".into(),
            ));
        }
        let run_bits_len = if has_runs { size.div_ceil(8) } else { 0 };
        if off + run_bits_len + size * 4 > bytes.len() {
            return Err(TreeError::CorruptedData(
                "roaring container descriptions truncated".into(),
            ));
        }
        let run_bits = &bytes[off..off + run_bits_len];
        off += run_bits_len;
        let descriptions = &bytes[off..off + size * 4];
        off += size * 4;
        let has_offsets = !has_runs || size >= ROARING_NO_OFFSET_THRESHOLD;
        if has_offsets {
            if off + size * 4 > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "roaring container offsets truncated".into(),
                ));
            }
            off += size * 4;
        }
        Ok(Self {
            size,
            has_runs,
            run_bits,
            descriptions,
            data: &bytes[off..],
        })
    }

    #[inline]
    fn is_run_container(&self, idx: usize) -> bool {
        self.has_runs && (self.run_bits[idx / 8] >> (idx % 8)) & 1 != 0
    }

    #[inline]
    fn key_and_cardinality(&self, idx: usize) -> (u16, u32) {
        let d = &self.descriptions[idx * 4..idx * 4 + 4];
        let key = u16::from_le_bytes([d[0], d[1]]);
        let card = u32::from(u16::from_le_bytes([d[2], d[3]])) + 1;
        (key, card)
    }

    fn container_offset(&self, idx: usize) -> Result<usize> {
        let mut off = 0usize;
        for i in 0..idx {
            if self.is_run_container(i) {
                if off + 2 > self.data.len() {
                    return Err(TreeError::CorruptedData(
                        "roaring run header truncated".into(),
                    ));
                }
                let runs =
                    u16::from_le_bytes(self.data[off..off + 2].try_into().expect("two bytes"))
                        as usize;
                off += 2 + runs * 4;
            } else {
                let (_, card) = self.key_and_cardinality(i);
                off += if card > ROARING_ARRAY_MAX_CARDINALITY {
                    8192
                } else {
                    card as usize * 2
                };
            }
        }
        if off > self.data.len() {
            return Err(TreeError::CorruptedData(
                "roaring container data truncated".into(),
            ));
        }
        Ok(off)
    }

    fn container(&self, idx: usize) -> Result<(u16, RoaringContainer<'a>)> {
        let (key, card) = self.key_and_cardinality(idx);
        let off = self.container_offset(idx)?;
        if self.is_run_container(idx) {
            let runs_len = 2;
            if off + runs_len > self.data.len() {
                return Err(TreeError::CorruptedData(
                    "roaring run header truncated".into(),
                ));
            }
            let runs =
                u16::from_le_bytes(self.data[off..off + 2].try_into().expect("two bytes")) as usize;
            let len = 2 + runs * 4;
            if off + len > self.data.len() {
                return Err(TreeError::CorruptedData(
                    "roaring run payload truncated".into(),
                ));
            }
            Ok((key, RoaringContainer::Run(&self.data[off + 2..off + len])))
        } else if card > ROARING_ARRAY_MAX_CARDINALITY {
            if off + 8192 > self.data.len() {
                return Err(TreeError::CorruptedData(
                    "roaring bitmap payload truncated".into(),
                ));
            }
            Ok((key, RoaringContainer::Bitmap(&self.data[off..off + 8192])))
        } else {
            let len = card as usize * 2;
            if off + len > self.data.len() {
                return Err(TreeError::CorruptedData(
                    "roaring array payload truncated".into(),
                ));
            }
            Ok((key, RoaringContainer::Array(&self.data[off..off + len])))
        }
    }

    /// Test membership without materializing the bitmap.
    fn contains(&self, local: LocalId) -> Result<bool> {
        let target = (local >> 16) as u16;
        let mut lo = 0usize;
        let mut hi = self.size;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (key, _) = self.key_and_cardinality(mid);
            match key.cmp(&target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let offset = local & 0xFFFF;
                    return self.container_contains(mid, offset);
                }
            }
        }
        Ok(false)
    }

    fn container_contains(&self, idx: usize, offset: u32) -> Result<bool> {
        let (_, container) = self.container(idx)?;
        Ok(match container {
            RoaringContainer::Run(payload) => {
                let runs = payload.len() / 4;
                let mut lo = 0usize;
                let mut hi = runs;
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let word = &payload[mid * 4..mid * 4 + 4];
                    let start = u32::from(u16::from_le_bytes([word[0], word[1]]));
                    let len = u32::from(u16::from_le_bytes([word[2], word[3]]));
                    if offset < start {
                        hi = mid;
                    } else if offset > start + len {
                        lo = mid + 1;
                    } else {
                        return Ok(true);
                    }
                }
                false
            }
            RoaringContainer::Bitmap(payload) => {
                let word = (offset as usize) >> 6;
                let bit = offset & 63;
                let w = u64::from_le_bytes(
                    payload[word * 8..word * 8 + 8]
                        .try_into()
                        .expect("eight bytes"),
                );
                (w >> bit) & 1 != 0
            }
            RoaringContainer::Array(payload) => {
                let values = payload.as_chunks::<2>().0;
                let mut lo = 0usize;
                let mut hi = values.len();
                let target = offset as u16;
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let v = u16::from_le_bytes(values[mid]);
                    match v.cmp(&target) {
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                        std::cmp::Ordering::Equal => return Ok(true),
                    }
                }
                false
            }
        })
    }
}

// SegPosting (`cf_attr_postings`, `cf_detached_postings`)
// ---------------------------------------------------------------------------

/// Segment-local posting set (`LocalId`s within one segment).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SegPosting {
    bm: RoaringBitmap,
}

impl SegPosting {
    pub fn from_bitmap(bm: RoaringBitmap) -> Self {
        Self { bm }
    }

    pub fn from_locals(locals: &[LocalId]) -> Self {
        let mut bm = RoaringBitmap::new();
        for &l in locals {
            bm.insert(l);
        }
        Self { bm }
    }

    /// Encode strictly ascending `LocalId`s without allocating a `RoaringBitmap` in the sparse case.
    pub fn encode_sorted_locals(locals: &[LocalId]) -> Result<Vec<u8>> {
        debug_assert!(locals.windows(2).all(|w| w[0] < w[1]));
        if locals.len() <= SEG_POSTING_SPARSE_MAX {
            let mut buf = Vec::with_capacity(1 + locals.len() * 4);
            buf.push(SEG_POSTING_SPARSE);
            for &id in locals {
                buf.extend_from_slice(&id.to_le_bytes());
            }
            return Ok(buf);
        }
        let bm = RoaringBitmap::from_sorted_iter(locals.iter().copied())
            .map_err(|e| TreeError::Storage(format!("roaring build error: {e}")))?;
        Self::from_bitmap(bm).encode()
    }

    pub fn into_bitmap(self) -> RoaringBitmap {
        self.bm
    }

    pub fn as_bitmap(&self) -> &RoaringBitmap {
        &self.bm
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bm.is_empty()
    }

    #[inline]
    pub fn insert(&mut self, local_id: LocalId) -> bool {
        self.bm.insert(local_id)
    }

    #[inline]
    pub fn remove(&mut self, local_id: LocalId) -> bool {
        self.bm.remove(local_id)
    }

    pub fn or_bitmap(&mut self, other: &RoaringBitmap) {
        self.bm |= other;
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let len = self.bm.len() as usize;
        if len <= SEG_POSTING_SPARSE_MAX {
            let mut buf = Vec::with_capacity(1 + len * 4);
            buf.push(SEG_POSTING_SPARSE);
            for id in self.bm.iter() {
                buf.extend_from_slice(&id.to_le_bytes());
            }
            return Ok(buf);
        }
        let mut buf = Vec::with_capacity(1 + self.bm.serialized_size());
        buf.push(SEG_POSTING_ROARING);
        self.bm
            .serialize_into(&mut buf)
            .map_err(|e| TreeError::Storage(format!("roaring serialize error: {e}")))?;
        Ok(buf)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let view = SegPostingView::parse(bytes)?;
        let mut bm = RoaringBitmap::new();
        view.or_into(&mut bm)?;
        Ok(Self { bm })
    }
}

/// Zero-copy borrowed view over an encoded [`SegPosting`].
#[derive(Debug, Clone, Copy)]
pub enum SegPostingView<'a> {
    Sparse(&'a [u8]),
    Roaring(&'a [u8]),
}

impl<'a> SegPostingView<'a> {
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Err(TreeError::CorruptedData("empty SegPosting".into()));
        }
        match bytes[0] {
            SEG_POSTING_SPARSE => {
                let rest = &bytes[1..];
                if !rest.len().is_multiple_of(4) {
                    return Err(TreeError::CorruptedData(
                        "SegPosting sparse payload length not multiple of 4".into(),
                    ));
                }
                if rest.len() > SEG_POSTING_SPARSE_MAX * 4 {
                    return Err(TreeError::CorruptedData(format!(
                        "SegPosting sparse payload too large: {} entries",
                        rest.len() / 4
                    )));
                }
                let mut prev: Option<u32> = None;
                for &chunk in rest.as_chunks::<4>().0 {
                    let v = u32::from_le_bytes(chunk);
                    if let Some(p) = prev
                        && v <= p
                    {
                        return Err(TreeError::CorruptedData(
                            "SegPosting sparse payload not strictly ascending".into(),
                        ));
                    }
                    prev = Some(v);
                }
                Ok(Self::Sparse(rest))
            }
            SEG_POSTING_ROARING => Ok(Self::Roaring(&bytes[1..])),
            other => Err(TreeError::CorruptedData(format!(
                "unknown SegPosting tag: {other}"
            ))),
        }
    }

    /// OR this posting into `dst` without intermediate `RoaringBitmap` allocation for the sparse form.
    #[inline]
    pub fn or_into(&self, dst: &mut RoaringBitmap) -> Result<()> {
        match *self {
            Self::Sparse(slice) => {
                for &chunk in slice.as_chunks::<4>().0 {
                    dst.insert(u32::from_le_bytes(chunk));
                }
            }
            Self::Roaring(slice) => {
                // Internal postings are validated when written; skip redundant sortedness and
                // popcount validation on the read hot path.
                let bm = RoaringBitmap::deserialize_unchecked_from(slice).map_err(|e| {
                    TreeError::CorruptedData(format!("roaring deserialize error: {e}"))
                })?;
                *dst |= bm;
            }
        }
        Ok(())
    }

    /// OR this posting into a flat 65,536-bit accumulator without materializing a bitmap.
    #[inline]
    pub fn or_into_accum(&self, acc: &mut SegBitsetAccum) -> Result<()> {
        acc.or_view(self)
    }

    /// Test membership of a segment-local id without materializing a bitmap or allocating.
    #[inline]
    pub fn contains(&self, local: LocalId) -> Result<bool> {
        match *self {
            Self::Sparse(slice) => {
                let target = local;
                let values = slice.as_chunks::<4>().0;
                let mut lo = 0usize;
                let mut hi = values.len();
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    match u32::from_le_bytes(values[mid]).cmp(&target) {
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                        std::cmp::Ordering::Equal => return Ok(true),
                    }
                }
                Ok(false)
            }
            Self::Roaring(slice) => RoaringLayout::parse(slice)?.contains(local),
        }
    }

    pub fn count(&self) -> Result<u64> {
        match *self {
            Self::Sparse(slice) => Ok((slice.len() / 4) as u64),
            Self::Roaring(slice) => Ok(decode_roaring_bitmap(slice)?.len()),
        }
    }

    /// Count how many entries in this posting intersect with `mask`.
    #[inline]
    pub fn and_count(&self, mask: &RoaringBitmap) -> Result<u64> {
        match *self {
            Self::Sparse(slice) => {
                let mut cnt = 0u64;
                for &chunk in slice.as_chunks::<4>().0 {
                    if mask.contains(u32::from_le_bytes(chunk)) {
                        cnt += 1;
                    }
                }
                Ok(cnt)
            }
            Self::Roaring(slice) => Ok(decode_roaring_bitmap(slice)?.intersection_len(mask)),
        }
    }

    /// OR entries of this posting that are present in `mask` into `dst`.
    #[inline]
    pub fn or_into_masked(&self, mask: &RoaringBitmap, dst: &mut RoaringBitmap) -> Result<()> {
        match *self {
            Self::Sparse(slice) => {
                for &chunk in slice.as_chunks::<4>().0 {
                    let v = u32::from_le_bytes(chunk);
                    if mask.contains(v) {
                        dst.insert(v);
                    }
                }
            }
            Self::Roaring(slice) => {
                let bm = decode_roaring_bitmap(slice)?;
                *dst |= bm & mask;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_roundtrip(bitmap: RoaringBitmap) {
        let bytes = SegPosting::from_bitmap(bitmap.clone()).encode().unwrap();
        let view = SegPostingView::parse(&bytes).unwrap();

        let mut from_accum = SegBitsetAccum::new();
        from_accum.or_view(&view).unwrap();
        assert_eq!(from_accum.clone().into_roaring(), bitmap);

        let mut materialized = RoaringBitmap::new();
        view.or_into(&mut materialized).unwrap();
        assert_eq!(materialized, bitmap);

        for local in [
            0u32, 1, 31, 32, 63, 64, 255, 4095, 4096, 65_535, 65_536, 70_000, 131_071, 200_000,
        ] {
            assert_eq!(
                view.contains(local).unwrap(),
                bitmap.contains(local),
                "contains mismatch at {local}"
            );
        }
    }

    #[test]
    fn seg_posting_views_match_roaring_across_container_kinds() {
        let sparse: RoaringBitmap = (0..20u32).map(|i| i * 3).collect();
        assert_roundtrip(sparse);

        let arrays: RoaringBitmap = (0..10_000u32).map(|i| (i * 7) % 200_000).collect();
        assert_roundtrip(arrays);

        let bitmaps: RoaringBitmap = (0..65_000u32).map(|i| i * 2).collect();
        assert_roundtrip(bitmaps);

        let mut runs = RoaringBitmap::new();
        runs.insert_range(0..70_000);
        runs.insert_range(100_000..160_000);
        assert_roundtrip(runs);
    }
}
