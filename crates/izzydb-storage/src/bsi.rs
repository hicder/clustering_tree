//! Per-segment bit-sliced index (BSI) for ordered numeric range predicates.
//!
//! Each sealed segment stores, per filterable ordered numeric attribute and per 65,536-row
//! chunk, a presence bitmap, a frame-of-reference base/width, and up to 64 value bit-planes of
//! `value - base`. Range predicates are evaluated with wordwise `&`/`|`/`!` over
//! `[u64; 1024]` planes instead of unioning thousands of tiny value-major postings.
//!
//! ```text
//! key   = seg(4) || attr(2) || chunk(4) || plane(1)
//! plane = 0..63 value bit, 0xFE base+width, 0xFF presence
//! value = 1024 little-endian u64 words (8 KiB), or 9 bytes for the base record
//! ```
//!
//! Values use the unsigned order-preserving encoding from [`LeafAttrValue::ordered_u64`](crate::leaf_attrs::LeafAttrValue::ordered_u64).

use izzydb_core::config::{AttrId, SegmentId};
use izzydb_core::schema::ScalarValue;

use crate::leaf_attrs::f64_ordered_bits;

/// Number of rows represented by one bit-plane chunk.
pub const BSI_CHUNK_ROWS: u32 = 65_536;

/// Number of `u64` words in one bit-plane chunk.
pub const BSI_WORDS: usize = (BSI_CHUNK_ROWS / 64) as usize;

/// Encoded byte length of one bit-plane chunk.
pub const BSI_PLANE_BYTES: usize = BSI_WORDS * 8;

/// Plane key used for the frame-of-reference base and bit width (`[base u64 LE][width u8]`).
pub const BSI_BASE_PLANE: u8 = 0xFE;

/// Plane key used for the presence bitmap.
pub const BSI_PRESENT_PLANE: u8 = 0xFF;

/// Maximum number of value bit-planes.
pub const BSI_MAX_VALUE_PLANES: u8 = 64;

/// Key for one BSI plane: `seg(4) || attr(2) || chunk(4) || plane(1)`.
#[inline]
pub fn bsi_key(seg: SegmentId, attr_id: AttrId, chunk: u32, plane: u8) -> [u8; 11] {
    let mut key = [0u8; 11];
    key[0..4].copy_from_slice(&seg.to_be_bytes());
    key[4..6].copy_from_slice(&attr_id.to_be_bytes());
    key[6..10].copy_from_slice(&chunk.to_be_bytes());
    key[10] = plane;
    key
}

/// Decode a BSI plane key.
#[inline]
pub fn decode_bsi_key(key: &[u8]) -> Option<(SegmentId, AttrId, u32, u8)> {
    if key.len() != 11 {
        return None;
    }
    Some((
        SegmentId::from_be_bytes(key[0..4].try_into().ok()?),
        AttrId::from_be_bytes(key[4..6].try_into().ok()?),
        u32::from_be_bytes(key[6..10].try_into().ok()?),
        key[10],
    ))
}

/// Encode a non-null ordered numeric scalar into its unsigned order-preserving value.
#[inline]
pub fn ordered_value(value: &ScalarValue) -> Option<u64> {
    match value {
        ScalarValue::Int(v) => Some((*v as u64) ^ (1u64 << 63)),
        ScalarValue::Uint(v) => Some(*v),
        ScalarValue::Float(v) if !v.is_nan() => Some(f64_ordered_bits(*v)),
        _ => None,
    }
}

/// Encode the BSI base record.
#[inline]
pub fn encode_base(base: u64, width: u8) -> [u8; 9] {
    let mut out = [0u8; 9];
    out[0..8].copy_from_slice(&base.to_le_bytes());
    out[8] = width;
    out
}

/// Decode the BSI base record.
#[inline]
pub fn decode_base(bytes: &[u8]) -> Option<(u64, u8)> {
    if bytes.len() != 9 {
        return None;
    }
    let base = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
    let width = bytes[8];
    if width > BSI_MAX_VALUE_PLANES {
        return None;
    }
    Some((base, width))
}

/// Accumulates one chunk's values for a single attribute before compression.
///
/// ```text
/// let mut b = BsiChunkBuilder::new();
/// b.insert(0, ordered_score);
/// let planes = b.encode_planes();
/// ```
#[derive(Debug)]
pub struct BsiChunkBuilder {
    presence: Box<[u64; BSI_WORDS]>,
    values: Vec<u64>,
    count: usize,
}

impl Default for BsiChunkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BsiChunkBuilder {
    /// Create an empty chunk builder.
    pub fn new() -> Self {
        Self {
            presence: Box::new([0u64; BSI_WORDS]),
            values: vec![0; BSI_CHUNK_ROWS as usize],
            count: 0,
        }
    }

    /// Insert one row value at `local` within the chunk.
    #[inline]
    pub fn insert(&mut self, local: u32, value: u64) {
        let local = local as usize;
        if local >= BSI_CHUNK_ROWS as usize {
            return;
        }
        let word = local >> 6;
        let bit = 1u64 << (local & 63);
        self.presence[word] |= bit;
        self.values[local] = value;
        self.count += 1;
    }

    /// Returns `true` when no row was inserted.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Compress and emit `(plane, bytes)` pairs.
    ///
    /// The presence bitmap and base/width record are always emitted, so a chunk with no values
    /// still marks the segment as having a BSI (`present == 0` means no matches).
    pub fn encode_planes(&self) -> Vec<(u8, Vec<u8>)> {
        let mut base = u64::MAX;
        let mut max = 0u64;
        for (local, &value) in self.values.iter().enumerate() {
            if !is_present(self.presence.as_slice(), local) {
                continue;
            }
            base = base.min(value);
            max = max.max(value);
        }
        let (base, span) = if base == u64::MAX {
            (0, 0)
        } else {
            (base, max - base)
        };
        let width = if span == 0 {
            0
        } else {
            (64 - span.leading_zeros()) as u8
        };
        let mut out = Vec::with_capacity(width as usize + 2);
        out.push((BSI_BASE_PLANE, encode_base(base, width).to_vec()));
        out.push((BSI_PRESENT_PLANE, encode_words(&self.presence)));
        if width == 0 {
            return out;
        }
        let mut planes: Vec<Box<[u64; BSI_WORDS]>> = (0..width)
            .map(|_| Box::new([0u64; BSI_WORDS]))
            .collect();
        for (local, &value) in self.values.iter().enumerate() {
            if !is_present(self.presence.as_slice(), local) {
                continue;
            }
            let word = local >> 6;
            let bit = 1u64 << (local & 63);
            let mut delta = value - base;
            while delta != 0 {
                let plane = delta.trailing_zeros() as usize;
                if plane < width as usize {
                    planes[plane][word] |= bit;
                }
                delta &= delta - 1;
            }
        }
        for (plane, words) in planes.iter().enumerate() {
            out.push((plane as u8, encode_words(words)));
        }
        out
    }
}

#[inline]
fn is_present(presence: &[u64], local: usize) -> bool {
    ((presence[local >> 6] >> (local & 63)) & 1) != 0
}

/// Encode `[u64; BSI_WORDS]` little-endian.
fn encode_words(words: &[u64; BSI_WORDS]) -> Vec<u8> {
    let mut out = Vec::with_capacity(BSI_PLANE_BYTES);
    for &word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

/// Decode one plane into `[u64; BSI_WORDS]`.
pub fn decode_plane(bytes: &[u8]) -> Option<Box<[u64; BSI_WORDS]>> {
    if bytes.len() != BSI_PLANE_BYTES {
        return None;
    }
    let mut out = Box::new([0u64; BSI_WORDS]);
    for (i, chunk) in bytes.as_chunks::<8>().0.iter().enumerate() {
        out[i] = u64::from_le_bytes(*chunk);
    }
    Some(out)
}

/// Decoded planes for one BSI chunk.
///
/// ```text
/// let chunk = BsiChunk::new(base, width, present, planes);
/// let rows = chunk.less_than(bound);
/// ```
#[derive(Debug, Clone)]
pub struct BsiChunk {
    /// Frame-of-reference base.
    pub base: u64,
    /// Number of value bit-planes.
    pub width: u8,
    /// Presence bitmap.
    pub present: Box<[u64; BSI_WORDS]>,
    /// Value bit-planes `0..width`; missing planes are all zero.
    pub planes: Vec<Option<Box<[u64; BSI_WORDS]>>>,
}

impl BsiChunk {
    /// Create a chunk from the base record and value planes returned by multi-gets.
    pub fn new(
        base: u64,
        width: u8,
        present: Box<[u64; BSI_WORDS]>,
        planes: Vec<(u8, Option<Box<[u64; BSI_WORDS]>>)>,
    ) -> Self {
        let mut by_plane: Vec<Option<Box<[u64; BSI_WORDS]>>> =
            vec![None; width as usize];
        for (plane, words) in planes {
            if (plane as usize) < by_plane.len() {
                by_plane[plane as usize] = words;
            }
        }
        Self {
            base,
            width,
            present,
            planes: by_plane,
        }
    }

    #[inline]
    fn span_limit(&self) -> Option<u64> {
        if self.width == 0 {
            Some(0)
        } else if self.width >= 64 {
            None
        } else {
            Some((1u64 << self.width) - 1)
        }
    }

    /// Compute the rows with `value < bound` (present rows only).
    fn lt(&self, bound: u64) -> [u64; BSI_WORDS] {
        let mut lt = [0u64; BSI_WORDS];
        if bound <= self.base {
            return lt;
        }
        let delta_bound = bound - self.base;
        if self.span_limit().is_some_and(|limit| delta_bound > limit) {
            return *self.present;
        }
        let mut eq = *self.present;
        for bit in (0..self.width as usize).rev() {
            let plane = self.planes[bit].as_deref();
            if ((delta_bound >> bit) & 1) == 1 {
                for i in 0..BSI_WORDS {
                    let p = plane.map_or(0, |p| p[i]);
                    lt[i] |= eq[i] & !p;
                    eq[i] &= p;
                }
            } else {
                for i in 0..BSI_WORDS {
                    let p = plane.map_or(0, |p| p[i]);
                    eq[i] &= !p;
                }
            }
            if eq.iter().all(|&w| w == 0) {
                break;
            }
        }
        lt
    }

    /// Compute the rows with `value == bound` (present rows only).
    fn eq(&self, bound: u64) -> [u64; BSI_WORDS] {
        if bound < self.base {
            return [0u64; BSI_WORDS];
        }
        let delta = bound - self.base;
        if self.span_limit().is_some_and(|limit| delta > limit) {
            return [0u64; BSI_WORDS];
        }
        let mut eq = *self.present;
        for bit in 0..self.width as usize {
            let plane = self.planes[bit].as_deref();
            let want = (delta >> bit) & 1;
            for i in 0..BSI_WORDS {
                let p = plane.map_or(0, |p| p[i]);
                if want == 1 {
                    eq[i] &= p;
                } else {
                    eq[i] &= !p;
                }
            }
            if eq.iter().all(|&w| w == 0) {
                break;
            }
        }
        eq
    }

    /// Rows where `value < bound`.
    pub fn less_than(&self, bound: u64) -> [u64; BSI_WORDS] {
        self.lt(bound)
    }

    /// Rows where `value <= bound`.
    pub fn less_equal(&self, bound: u64) -> [u64; BSI_WORDS] {
        let mut lt = self.lt(bound);
        let eq = self.eq(bound);
        for i in 0..BSI_WORDS {
            lt[i] |= eq[i];
        }
        lt
    }

    /// Rows where `value >= bound`.
    pub fn greater_equal(&self, bound: u64) -> [u64; BSI_WORDS] {
        let lt = self.lt(bound);
        let mut out = *self.present;
        for i in 0..BSI_WORDS {
            out[i] &= !lt[i];
        }
        out
    }

    /// Rows where `value > bound`.
    pub fn greater_than(&self, bound: u64) -> [u64; BSI_WORDS] {
        let le = self.less_equal(bound);
        let mut out = *self.present;
        for i in 0..BSI_WORDS {
            out[i] &= !le[i];
        }
        out
    }

    /// Intersect `mask` with another mask.
    pub fn intersect(mask: &mut [u64; BSI_WORDS], other: &[u64; BSI_WORDS]) {
        for i in 0..BSI_WORDS {
            mask[i] &= other[i];
        }
    }

    /// Build a roaring bitmap from a chunk mask, offsetting by `chunk * BSI_CHUNK_ROWS`.
    pub fn to_bitmap(mask: &[u64; BSI_WORDS], chunk: u32, out: &mut roaring::RoaringBitmap) {
        let base = chunk as u64 * BSI_CHUNK_ROWS as u64;
        for (w, &word) in mask.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as u64;
                bits &= bits - 1;
                out.insert((base + (w as u64 * 64) + bit) as u32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(values: &[(u32, u64)]) -> BsiChunk {
        let mut builder = BsiChunkBuilder::new();
        for &(local, value) in values {
            builder.insert(local, value);
        }
        let encoded = builder.encode_planes();
        let base_record = encoded
            .iter()
            .find(|(p, _)| *p == BSI_BASE_PLANE)
            .unwrap();
        let (base, width) = decode_base(&base_record.1).unwrap();
        let present = decode_plane(
            &encoded
                .iter()
                .find(|(p, _)| *p == BSI_PRESENT_PLANE)
                .unwrap()
                .1,
        )
        .unwrap();
        let planes = (0u8..width)
            .map(|p| {
                let bytes = encoded.iter().find(|(pp, _)| *pp == p).map(|(_, b)| b);
                (p, bytes.and_then(|b| decode_plane(b)))
            })
            .collect();
        BsiChunk::new(base, width, present, planes)
    }

    #[test]
    fn bsi_range_masks() {
        let chunk = build(&[(0, 0), (1, 3), (2, 7), (3, 10), (4, 100), (5, 1000)]);

        let mut mask = chunk.less_than(10);
        let mut bitmap = roaring::RoaringBitmap::new();
        BsiChunk::to_bitmap(&mask, 0, &mut bitmap);
        assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![0, 1, 2]);

        mask = chunk.greater_equal(10);
        bitmap.clear();
        BsiChunk::to_bitmap(&mask, 0, &mut bitmap);
        assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![3, 4, 5]);

        mask = chunk.less_equal(10);
        bitmap.clear();
        BsiChunk::to_bitmap(&mask, 0, &mut bitmap);
        assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![0, 1, 2, 3]);

        let hi = chunk.less_than(1000);
        let lo = chunk.greater_equal(10);
        let mut both = hi;
        BsiChunk::intersect(&mut both, &lo);
        bitmap.clear();
        BsiChunk::to_bitmap(&both, 0, &mut bitmap);
        assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![3, 4]);

        // Bounds outside the stored range.
        mask = chunk.less_than(0);
        assert!(mask.iter().all(|&w| w == 0));
        mask = chunk.less_than(u64::MAX);
        bitmap.clear();
        BsiChunk::to_bitmap(&mask, 0, &mut bitmap);
        assert_eq!(bitmap.len(), 6);
    }

    #[test]
    fn bsi_float_large_values_use_for_range() {
        let base = f64_ordered_bits(0.25);
        let chunk = build(&[(0, base), (1, f64_ordered_bits(0.5)), (2, f64_ordered_bits(0.75))]);
        let mask = chunk.less_than(f64_ordered_bits(0.6));
        let mut bitmap = roaring::RoaringBitmap::new();
        BsiChunk::to_bitmap(&mask, 0, &mut bitmap);
        assert_eq!(bitmap.iter().collect::<Vec<_>>(), vec![0, 1]);
    }
}
