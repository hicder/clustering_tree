
use izzydb_core::config::LocalId;
use izzydb_core::error::{Result, TreeError};
use roaring::RoaringBitmap;
use super::*;

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
                let bm = decode_roaring_bitmap(slice)?;
                *dst |= bm;
            }
        }
        Ok(())
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
