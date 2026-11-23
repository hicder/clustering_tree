use std::cmp::Ordering;

use izzydb_core::config::{
    AttrId, ColumnGen, DocId, LocalId, ROW_BLOCK_SIZE, RowBlock, RowId, SegmentId, SkipChunk,
    TermId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::{Filter, FtsPredicateKind, TokenQuery};
use izzydb_core::schema::{
    AttributeType, AttributeValue, Attributes, CollectionSchema, ColumnStorage, ScalarValue,
};
use roaring::RoaringBitmap;

/// Reserved `type_tag` tracking non-null attribute presence for `IsNull` / `IsNotNull`.
pub const TAG_PRESENT: u8 = 0x00;
pub const TAG_BOOL: u8 = 0x01;
pub const TAG_INT: u8 = 0x02;
pub const TAG_UINT: u8 = 0x03;
pub const TAG_FLOAT: u8 = 0x04;
pub const TAG_STRING: u8 = 0x05;

// `default` CF keyspace tags.
pub const INTERNAL_NODE_TAG: u8 = 0x02;
pub const FTS_DOC_LEN_TAG: u8 = 0x05;
pub const ATTR_COUNT_TAG: u8 = 0x06;
pub const DETACHED_COUNT_TAG: u8 = 0x07;
pub const SEG_DELETES_TAG: u8 = 0x10;
pub const SEG_META_TAG: u8 = 0x11;
pub const ZONE_TAG: u8 = 0x12;
pub const DETACHED_META_TAG: u8 = 0x13;
pub const DETACHED_OVERWRITTEN_TAG: u8 = 0x14;

const ROW_FORMAT_V3: u8 = 0x03;
const ROW_FLAG_HAS_VECTOR: u8 = 0x01;
const ROW_HEADER_LEN: usize = 4;
const ROW_DIR_ENTRY_LEN: usize = 8;

const ROW_VAL_NULL: u8 = 0;
const ROW_VAL_BOOL: u8 = 1;
const ROW_VAL_INT: u8 = 2;
const ROW_VAL_UINT: u8 = 3;
const ROW_VAL_FLOAT: u8 = 4;
const ROW_VAL_STRING: u8 = 5;
const ROW_VAL_STRING_ARRAY: u8 = 6;
const ROW_VAL_INT_ARRAY: u8 = 7;
const ROW_VAL_UINT_ARRAY: u8 = 8;

const SEG_POSTING_SPARSE: u8 = 0x00;
const SEG_POSTING_ROARING: u8 = 0x01;
pub const SEG_POSTING_SPARSE_MAX: usize = 32;

pub const DOC_ID_BLOCK_BYTES: usize = ROW_BLOCK_SIZE * 8;
pub const MANIFEST_BYTES: usize = 32;
pub const SEGMENT_META_BYTES: usize = 44;
pub const DETACHED_META_BYTES: usize = 24;

pub const SEG_FLAG_HAS_COLUMNS: u8 = 0x01;
pub const SEG_FLAG_HAS_VECTORS: u8 = 0x02;

// ---------------------------------------------------------------------------
// Key builders & decoders
// ---------------------------------------------------------------------------

#[inline]
pub fn row_key(row: RowId) -> [u8; 8] {
    row.to_be_bytes()
}

#[inline]
pub fn decode_row_key(bytes: &[u8]) -> Result<RowId> {
    if bytes.len() != 8 {
        return Err(TreeError::CorruptedData(format!(
            "row_key expected 8 bytes, got {}",
            bytes.len()
        )));
    }
    Ok(RowId::from_be_bytes(bytes.try_into().unwrap()))
}

#[inline]
pub fn doc_id_block_key(seg: SegmentId, block: u32) -> [u8; 8] {
    let s = seg.to_be_bytes();
    let b = block.to_be_bytes();
    [s[0], s[1], s[2], s[3], b[0], b[1], b[2], b[3]]
}

#[inline]
pub fn decode_doc_id_block_key(bytes: &[u8]) -> Result<(SegmentId, u32)> {
    if bytes.len() != 8 {
        return Err(TreeError::CorruptedData(format!(
            "doc_id_block_key expected 8 bytes, got {}",
            bytes.len()
        )));
    }
    Ok((
        u32::from_be_bytes(bytes[0..4].try_into().unwrap()),
        u32::from_be_bytes(bytes[4..8].try_into().unwrap()),
    ))
}

/// 3-byte prefix (`attr_id BE || type_tag`) for `cf_attr_postings` and `default[0x06]`.
#[inline]
pub fn encode_attr_prefix(attr_id: AttrId, type_tag: u8) -> [u8; 3] {
    let id = attr_id.to_be_bytes();
    [id[0], id[1], type_tag]
}

/// Build an index value key (`attr_id BE || type_tag || encoded_val`) for `default[0x06]` counts
/// and as the value prefix of `cf_attr_postings`.
#[inline]
pub fn encode_present_attr_key(attr_id: AttrId) -> Vec<u8> {
    encode_attr_prefix(attr_id, TAG_PRESENT).to_vec()
}

/// Build an index value key: `attr_id (u16 BE) || type_tag (u8) || encoded_val`.
pub fn encode_scalar_attr_key(attr_id: AttrId, val: &ScalarValue) -> Result<Vec<u8>> {
    let mut key = Vec::with_capacity(16);
    key.extend_from_slice(&attr_id.to_be_bytes());
    key.push(0); // placeholder for type_tag
    let tag = encode_scalar_val_bytes(val, &mut key)?;
    key[2] = tag;
    Ok(key)
}

/// Build a value-major `cf_attr_postings` key:
/// `attr_id (u16 BE) || type_tag (u8) || encoded_val || seg (u32 BE)`.
#[inline]
pub fn posting_key(attr_id: AttrId, type_tag: u8, encoded_val: &[u8], seg: SegmentId) -> Vec<u8> {
    let mut key = Vec::with_capacity(7 + encoded_val.len());
    key.extend_from_slice(&encode_attr_prefix(attr_id, type_tag));
    key.extend_from_slice(encoded_val);
    key.extend_from_slice(&seg.to_be_bytes());
    key
}

/// Build a `cf_attr_postings` key from an `attr_key` (`attr_id BE || tag || encoded_val`) and `seg`.
#[inline]
pub fn make_posting_key(attr_key: &[u8], seg: SegmentId) -> Vec<u8> {
    debug_assert!(attr_key.len() >= 3);
    let mut key = Vec::with_capacity(attr_key.len() + 4);
    key.extend_from_slice(attr_key);
    key.extend_from_slice(&seg.to_be_bytes());
    key
}

/// Split a `cf_attr_postings` key into `(prefix = attr_id || tag || encoded_val, seg)`.
///
/// Valid even when `encoded_val` contains embedded `0x00` bytes because `seg` is fixed-width 4 bytes
/// at the tail and `encoded_val` is self-delimiting.
#[inline]
pub fn split_posting_key(key: &[u8]) -> Result<(&[u8], SegmentId)> {
    if key.len() < 7 {
        return Err(TreeError::CorruptedData(format!(
            "cf_attr_postings key too short: {}",
            key.len()
        )));
    }
    let split = key.len() - 4;
    let seg = u32::from_be_bytes(key[split..].try_into().unwrap());
    Ok((&key[..split], seg))
}

/// Split a `cf_attr_postings` key into `(attr_id, type_tag, encoded_val, seg)`.
#[inline]
pub fn split_posting_key_parts(key: &[u8]) -> Result<(AttrId, u8, &[u8], SegmentId)> {
    let (prefix, seg) = split_posting_key(key)?;
    let attr_id = u16::from_be_bytes([prefix[0], prefix[1]]);
    let type_tag = prefix[2];
    Ok((attr_id, type_tag, &prefix[3..], seg))
}

/// 7-byte prefix (`attr_id BE || gen BE || type_tag`) for `cf_detached_postings`.
#[inline]
pub fn detached_posting_prefix(attr_id: AttrId, col_gen: ColumnGen, type_tag: u8) -> [u8; 7] {
    let a = attr_id.to_be_bytes();
    let g = col_gen.to_be_bytes();
    [a[0], a[1], g[0], g[1], g[2], g[3], type_tag]
}

/// Build a `cf_detached_postings` key: `attr_id BE || gen BE || tag || encoded_val || seg BE`.
#[inline]
pub fn detached_posting_key(
    attr_id: AttrId,
    col_gen: ColumnGen,
    type_tag: u8,
    encoded_val: &[u8],
    seg: SegmentId,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(11 + encoded_val.len());
    key.extend_from_slice(&detached_posting_prefix(attr_id, col_gen, type_tag));
    key.extend_from_slice(encoded_val);
    key.extend_from_slice(&seg.to_be_bytes());
    key
}

/// Split a `cf_detached_postings` key into `(prefix = attr || gen || tag || val, seg)`.
#[inline]
pub fn split_detached_posting_key(key: &[u8]) -> Result<(&[u8], SegmentId)> {
    if key.len() < 11 {
        return Err(TreeError::CorruptedData(format!(
            "cf_detached_postings key too short: {}",
            key.len()
        )));
    }
    let split = key.len() - 4;
    let seg = u32::from_be_bytes(key[split..].try_into().unwrap());
    Ok((&key[..split], seg))
}

/// Build a 12-byte `cf_fts_postings` / `cf_fts_positions` key: `term_id BE || seg BE || block BE`.
#[inline]
pub fn fts_block_key(term: TermId, seg: SegmentId, block: u32) -> [u8; 12] {
    let t = term.to_be_bytes();
    let s = seg.to_be_bytes();
    let b = block.to_be_bytes();
    [
        t[0], t[1], t[2], t[3], s[0], s[1], s[2], s[3], b[0], b[1], b[2], b[3],
    ]
}

#[inline]
pub fn fts_block_key_coord(term: TermId, block: RowBlock) -> [u8; 12] {
    fts_block_key(term, block.seg(), block.block())
}

#[inline]
pub fn decode_fts_block_key(bytes: &[u8]) -> Result<(TermId, RowBlock)> {
    if bytes.len() != 12 {
        return Err(TreeError::CorruptedData(format!(
            "fts_block_key expected 12 bytes, got {}",
            bytes.len()
        )));
    }
    let term = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    let seg = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
    let block = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    Ok((term, RowBlock::new(seg, block)))
}

/// Build a 12-byte `cf_fts_skips` key: `term_id BE || seg BE || chunk BE`.
#[inline]
pub fn fts_chunk_key(term: TermId, seg: SegmentId, chunk: u32) -> [u8; 12] {
    fts_block_key(term, seg, chunk)
}

#[inline]
pub fn fts_chunk_key_coord(term: TermId, chunk: SkipChunk) -> [u8; 12] {
    fts_chunk_key(term, chunk.seg(), chunk.chunk())
}

#[inline]
pub fn decode_fts_chunk_key(bytes: &[u8]) -> Result<(TermId, SkipChunk)> {
    if bytes.len() != 12 {
        return Err(TreeError::CorruptedData(format!(
            "fts_chunk_key expected 12 bytes, got {}",
            bytes.len()
        )));
    }
    let term = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    let seg = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
    let chunk = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    Ok((term, SkipChunk::new(seg, chunk)))
}

/// Build a 10-byte doc-len payload key (without the `0x05` `default` CF tag): `attr BE || seg BE || block BE`.
#[inline]
pub fn doc_len_key(attr: AttrId, seg: SegmentId, block: u32) -> [u8; 10] {
    let a = attr.to_be_bytes();
    let s = seg.to_be_bytes();
    let b = block.to_be_bytes();
    [a[0], a[1], s[0], s[1], s[2], s[3], b[0], b[1], b[2], b[3]]
}

#[inline]
pub fn doc_len_key_coord(attr: AttrId, block: RowBlock) -> [u8; 10] {
    doc_len_key(attr, block.seg(), block.block())
}

#[inline]
pub fn decode_doc_len_key(bytes: &[u8]) -> Result<(AttrId, RowBlock)> {
    if bytes.len() != 10 {
        return Err(TreeError::CorruptedData(format!(
            "doc_len_key expected 10 bytes, got {}",
            bytes.len()
        )));
    }
    let attr = u16::from_be_bytes([bytes[0], bytes[1]]);
    let seg = u32::from_be_bytes(bytes[2..6].try_into().unwrap());
    let block = u32::from_be_bytes(bytes[6..10].try_into().unwrap());
    Ok((attr, RowBlock::new(seg, block)))
}

#[inline]
pub fn seg_deletes_key(seg: SegmentId) -> [u8; 5] {
    let s = seg.to_be_bytes();
    [SEG_DELETES_TAG, s[0], s[1], s[2], s[3]]
}

#[inline]
pub fn seg_meta_key(seg: SegmentId) -> [u8; 5] {
    let s = seg.to_be_bytes();
    [SEG_META_TAG, s[0], s[1], s[2], s[3]]
}

#[inline]
pub fn zone_key(seg: SegmentId, attr: AttrId, tag: u8) -> [u8; 8] {
    let s = seg.to_be_bytes();
    let a = attr.to_be_bytes();
    [ZONE_TAG, s[0], s[1], s[2], s[3], a[0], a[1], tag]
}

#[inline]
pub fn column_key(seg: SegmentId, attr: AttrId, block: u32) -> [u8; 10] {
    let s = seg.to_be_bytes();
    let a = attr.to_be_bytes();
    let b = block.to_be_bytes();
    [s[0], s[1], s[2], s[3], a[0], a[1], b[0], b[1], b[2], b[3]]
}

#[inline]
pub fn detached_key(attr: AttrId, col_gen: ColumnGen, seg: SegmentId, block: u32) -> [u8; 14] {
    let a = attr.to_be_bytes();
    let g = col_gen.to_be_bytes();
    let s = seg.to_be_bytes();
    let b = block.to_be_bytes();
    [
        a[0], a[1], g[0], g[1], g[2], g[3], s[0], s[1], s[2], s[3], b[0], b[1], b[2], b[3],
    ]
}

#[inline]
pub fn detached_meta_key(attr: AttrId) -> [u8; 3] {
    let a = attr.to_be_bytes();
    [DETACHED_META_TAG, a[0], a[1]]
}

#[inline]
pub fn detached_overwritten_key(attr: AttrId, col_gen: ColumnGen, seg: SegmentId) -> [u8; 11] {
    let a = attr.to_be_bytes();
    let g = col_gen.to_be_bytes();
    let s = seg.to_be_bytes();
    [
        DETACHED_OVERWRITTEN_TAG,
        a[0],
        a[1],
        g[0],
        g[1],
        g[2],
        g[3],
        s[0],
        s[1],
        s[2],
        s[3],
    ]
}

// ---------------------------------------------------------------------------
// Order-preserving scalar encodings
// ---------------------------------------------------------------------------

/// Encode a scalar value in order-preserving bytes and return its `type_tag`.
pub fn encode_scalar_val_bytes(val: &ScalarValue, out: &mut Vec<u8>) -> Result<u8> {
    match val {
        ScalarValue::Null => Ok(TAG_PRESENT),
        ScalarValue::Bool(b) => {
            out.push(if *b { 0x01 } else { 0x00 });
            Ok(TAG_BOOL)
        }
        ScalarValue::Int(v) => {
            out.extend_from_slice(&encode_i64_ordered(*v));
            Ok(TAG_INT)
        }
        ScalarValue::Uint(v) => {
            out.extend_from_slice(&v.to_be_bytes());
            Ok(TAG_UINT)
        }
        ScalarValue::Float(v) => {
            out.extend_from_slice(&encode_f64_ordered(*v)?);
            Ok(TAG_FLOAT)
        }
        ScalarValue::String(s) => {
            encode_str_ordered(s.as_bytes(), out);
            Ok(TAG_STRING)
        }
    }
}

/// Return the `type_tag` used by range queries for an ordered [`AttributeType`].
pub fn ordered_type_tag(attr_type: AttributeType) -> Result<u8> {
    match attr_type {
        AttributeType::Bool => Ok(TAG_BOOL),
        AttributeType::Int => Ok(TAG_INT),
        AttributeType::Uint => Ok(TAG_UINT),
        AttributeType::Float => Ok(TAG_FLOAT),
        AttributeType::String => Ok(TAG_STRING),
        other => Err(TreeError::InvalidFilter(format!(
            "attribute type {other:?} has no scalar type_tag"
        ))),
    }
}

#[inline]
pub fn encode_i64_ordered(v: i64) -> [u8; 8] {
    ((v as u64) ^ (1u64 << 63)).to_be_bytes()
}

#[inline]
pub fn encode_f64_ordered(v: f64) -> Result<[u8; 8]> {
    if v.is_nan() {
        return Err(TreeError::InvalidFilter(
            "NaN float value is not allowed".into(),
        ));
    }
    let normalized = if v == 0.0 { 0.0f64 } else { v };
    let bits = normalized.to_bits();
    let ordered = if (bits >> 63) != 0 {
        !bits
    } else {
        bits ^ (1u64 << 63)
    };
    Ok(ordered.to_be_bytes())
}

/// Escaped-null-terminated UTF-8 encoding (`0x00 -> 0x00 0xFF`, terminated with `0x00 0x00`).
#[inline]
pub fn encode_str_ordered(bytes: &[u8], out: &mut Vec<u8>) {
    out.reserve(bytes.len() + 2);
    for &b in bytes {
        if b == 0x00 {
            out.push(0x00);
            out.push(0xFF);
        } else {
            out.push(b);
        }
    }
    out.push(0x00);
    out.push(0x00);
}

/// Escaped prefix encoding for typeahead scans: `0x00 -> 0x00 0xFF` with **no** terminator.
#[inline]
pub fn encode_str_prefix_ordered(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for &b in bytes {
        if b == 0x00 {
            out.push(0x00);
            out.push(0xFF);
        } else {
            out.push(b);
        }
    }
    out
}

#[inline]
pub fn decode_i64_ordered(bytes: [u8; 8]) -> i64 {
    (u64::from_be_bytes(bytes) ^ (1u64 << 63)) as i64
}

#[inline]
pub fn decode_f64_ordered(bytes: [u8; 8]) -> f64 {
    let ordered = u64::from_be_bytes(bytes);
    let bits = if (ordered >> 63) != 0 {
        ordered ^ (1u64 << 63)
    } else {
        !ordered
    };
    f64::from_bits(bits)
}

pub fn decode_str_ordered(bytes: &[u8]) -> Result<String> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    loop {
        match bytes.get(i) {
            None => {
                return Err(TreeError::CorruptedData(
                    "ordered string missing terminator".into(),
                ));
            }
            Some(0x00) => match bytes.get(i + 1) {
                Some(0x00) => {
                    if i + 2 != bytes.len() {
                        return Err(TreeError::CorruptedData(
                            "ordered string has trailing bytes".into(),
                        ));
                    }
                    break;
                }
                Some(0xFF) => {
                    out.push(0x00);
                    i += 2;
                }
                _ => {
                    return Err(TreeError::CorruptedData(
                        "ordered string has invalid escape".into(),
                    ));
                }
            },
            Some(&b) => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out)
        .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in ordered string: {e}")))
}

pub fn decode_scalar_val_bytes(type_tag: u8, bytes: &[u8]) -> Result<ScalarValue> {
    let arr8 = |b: &[u8]| -> Result<[u8; 8]> {
        b.try_into().map_err(|_| {
            TreeError::CorruptedData(format!("expected 8 encoded bytes, got {}", b.len()))
        })
    };
    match type_tag {
        TAG_PRESENT => {
            if !bytes.is_empty() {
                return Err(TreeError::CorruptedData(
                    "TAG_PRESENT value must be empty".into(),
                ));
            }
            Ok(ScalarValue::Null)
        }
        TAG_BOOL => match bytes {
            [0] => Ok(ScalarValue::Bool(false)),
            [1] => Ok(ScalarValue::Bool(true)),
            _ => Err(TreeError::CorruptedData("invalid encoded bool".into())),
        },
        TAG_INT => Ok(ScalarValue::Int(decode_i64_ordered(arr8(bytes)?))),
        TAG_UINT => Ok(ScalarValue::Uint(u64::from_be_bytes(arr8(bytes)?))),
        TAG_FLOAT => Ok(ScalarValue::Float(decode_f64_ordered(arr8(bytes)?))),
        TAG_STRING => Ok(ScalarValue::String(decode_str_ordered(bytes)?)),
        other => Err(TreeError::CorruptedData(format!(
            "unknown index type_tag {other}"
        ))),
    }
}

pub fn attr_type_value_tag(attr_type: AttributeType) -> u8 {
    match attr_type.scalar_type() {
        AttributeType::Bool => TAG_BOOL,
        AttributeType::Int => TAG_INT,
        AttributeType::Uint => TAG_UINT,
        AttributeType::Float => TAG_FLOAT,
        _ => TAG_STRING,
    }
}

// ---------------------------------------------------------------------------
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
                for chunk in rest.chunks_exact(4) {
                    let v = u32::from_le_bytes(chunk.try_into().unwrap());
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
                for chunk in slice.chunks_exact(4) {
                    dst.insert(u32::from_le_bytes(chunk.try_into().unwrap()));
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
}

// ---------------------------------------------------------------------------
// DocIdBlock (`cf_row_doc_ids`)
// ---------------------------------------------------------------------------

pub fn encode_doc_id_block(ids: &[DocId; ROW_BLOCK_SIZE]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(DOC_ID_BLOCK_BYTES);
    #[cfg(target_endian = "little")]
    {
        buf.extend_from_slice(bytemuck::cast_slice(ids));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &id in ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }
    buf
}

pub fn decode_doc_id_block(bytes: &[u8]) -> Result<Box<[DocId; ROW_BLOCK_SIZE]>> {
    if bytes.len() != DOC_ID_BLOCK_BYTES {
        return Err(TreeError::CorruptedData(format!(
            "DocIdBlock expected {DOC_ID_BLOCK_BYTES} bytes, got {}",
            bytes.len()
        )));
    }
    let mut out = Box::new([0u64; ROW_BLOCK_SIZE]);
    #[cfg(target_endian = "little")]
    if let Ok(src) = bytemuck::try_cast_slice::<u8, u64>(bytes) {
        out.copy_from_slice(src);
        return Ok(out);
    }
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(8)) {
        *dst = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Segment metadata, Manifest, ZoneEntry, DetachedMeta
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SegmentState {
    Open = 0,
    Sealed = 1,
    Building = 2,
    Forwarding = 3,
    Retired = 4,
}

impl SegmentState {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::Open),
            1 => Ok(Self::Sealed),
            2 => Ok(Self::Building),
            3 => Ok(Self::Forwarding),
            4 => Ok(Self::Retired),
            other => Err(TreeError::CorruptedData(format!(
                "unknown SegmentState byte: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentMeta {
    pub state: SegmentState,
    pub flags: u8,
    pub row_count: u32,
    pub deleted_count: u32,
    pub forward_target: SegmentId,
    pub forward_base: LocalId,
    pub forwarded_at_epoch: u64,
    pub created_seq: u64,
    pub sealed_seq: u64,
}

impl SegmentMeta {
    pub fn new_open(created_seq: u64) -> Self {
        Self {
            state: SegmentState::Open,
            flags: 0,
            row_count: 0,
            deleted_count: 0,
            forward_target: 0,
            forward_base: 0,
            forwarded_at_epoch: 0,
            created_seq,
            sealed_seq: 0,
        }
    }

    pub fn encode(&self) -> [u8; SEGMENT_META_BYTES] {
        let mut buf = [0u8; SEGMENT_META_BYTES];
        buf[0] = 1; // version
        buf[1] = self.state as u8;
        buf[2] = self.flags;
        buf[3] = 0;
        buf[4..8].copy_from_slice(&self.row_count.to_le_bytes());
        buf[8..12].copy_from_slice(&self.deleted_count.to_le_bytes());
        buf[12..16].copy_from_slice(&self.forward_target.to_le_bytes());
        buf[16..20].copy_from_slice(&self.forward_base.to_le_bytes());
        buf[20..28].copy_from_slice(&self.forwarded_at_epoch.to_le_bytes());
        buf[28..36].copy_from_slice(&self.created_seq.to_le_bytes());
        buf[36..44].copy_from_slice(&self.sealed_seq.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != SEGMENT_META_BYTES {
            return Err(TreeError::CorruptedData(format!(
                "SegmentMeta expected {SEGMENT_META_BYTES} bytes, got {}",
                bytes.len()
            )));
        }
        if bytes[0] != 1 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported SegmentMeta version: {}",
                bytes[0]
            )));
        }
        if bytes[3] != 0 {
            return Err(TreeError::CorruptedData(format!(
                "invalid SegmentMeta reserved byte: {}",
                bytes[3]
            )));
        }
        let state = SegmentState::from_u8(bytes[1])?;
        let flags = bytes[2];
        let row_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let deleted_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if deleted_count > row_count {
            return Err(TreeError::CorruptedData(format!(
                "SegmentMeta deleted_count ({deleted_count}) > row_count ({row_count})"
            )));
        }
        Ok(Self {
            state,
            flags,
            row_count,
            deleted_count,
            forward_target: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            forward_base: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            forwarded_at_epoch: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
            created_seq: u64::from_le_bytes(bytes[28..36].try_into().unwrap()),
            sealed_seq: u64::from_le_bytes(bytes[36..44].try_into().unwrap()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub open_segment: Option<SegmentId>,
    pub next_segment_id: SegmentId,
    pub seal_rows: u32,
    pub live_rows: u64,
    pub total_rows: u64,
}

impl Manifest {
    pub fn new(seal_rows: u32) -> Self {
        Self {
            open_segment: None,
            next_segment_id: 0,
            seal_rows,
            live_rows: 0,
            total_rows: 0,
        }
    }

    pub fn encode(&self) -> [u8; MANIFEST_BYTES] {
        let mut buf = [0u8; MANIFEST_BYTES];
        buf[0] = 1; // version
        match self.open_segment {
            Some(seg) => {
                buf[1] = 1;
                buf[8..12].copy_from_slice(&seg.to_le_bytes());
            }
            None => {
                buf[1] = 0;
            }
        }
        buf[4..8].copy_from_slice(&self.next_segment_id.to_le_bytes());
        buf[12..16].copy_from_slice(&self.seal_rows.to_le_bytes());
        buf[16..24].copy_from_slice(&self.live_rows.to_le_bytes());
        buf[24..32].copy_from_slice(&self.total_rows.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != MANIFEST_BYTES {
            return Err(TreeError::CorruptedData(format!(
                "Manifest expected {MANIFEST_BYTES} bytes, got {}",
                bytes.len()
            )));
        }
        if bytes[0] != 1 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported Manifest version: {}",
                bytes[0]
            )));
        }
        if bytes[2] != 0 || bytes[3] != 0 {
            return Err(TreeError::CorruptedData(
                "invalid Manifest reserved bytes".into(),
            ));
        }
        let open_raw = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let open_segment = match bytes[1] {
            0 => None,
            1 => Some(open_raw),
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "invalid Manifest has_open byte: {other}"
                )));
            }
        };
        let next_segment_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let seal_rows = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let live_rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let total_rows = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        if live_rows > total_rows {
            return Err(TreeError::CorruptedData(format!(
                "Manifest live_rows ({live_rows}) > total_rows ({total_rows})"
            )));
        }
        Ok(Self {
            open_segment,
            next_segment_id,
            seal_rows,
            live_rows,
            total_rows,
        })
    }
}

/// Per-segment min/max ordered scalar bounds stored in `default[0x12 || seg || attr || tag]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneEntry {
    pub min: Vec<u8>,
    pub max: Vec<u8>,
}

impl ZoneEntry {
    pub fn singleton(val: &[u8]) -> Self {
        Self {
            min: val.to_vec(),
            max: val.to_vec(),
        }
    }

    /// Widen bounds to include `val`; returns `true` if `min` or `max` changed.
    pub fn widen(&mut self, val: &[u8]) -> bool {
        let mut changed = false;
        if val < self.min.as_slice() {
            self.min.clear();
            self.min.extend_from_slice(val);
            changed = true;
        }
        if val > self.max.as_slice() {
            self.max.clear();
            self.max.extend_from_slice(val);
            changed = true;
        }
        changed
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.min.len() + self.max.len());
        buf.extend_from_slice(&(self.min.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(self.max.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.min);
        buf.extend_from_slice(&self.max);
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(TreeError::CorruptedData(format!(
                "ZoneEntry too short: {}",
                bytes.len()
            )));
        }
        let min_len = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        let max_len = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        let expected = 4usize
            .checked_add(min_len)
            .and_then(|x| x.checked_add(max_len))
            .ok_or_else(|| TreeError::CorruptedData("ZoneEntry length overflow".into()))?;
        if bytes.len() != expected || min_len == 0 || max_len == 0 {
            return Err(TreeError::CorruptedData(format!(
                "ZoneEntry expected {expected} bytes (non-empty bounds), got {}",
                bytes.len()
            )));
        }
        let min = bytes[4..4 + min_len].to_vec();
        let max = bytes[4 + min_len..].to_vec();
        if min > max {
            return Err(TreeError::CorruptedData(
                "ZoneEntry min > max".into(),
            ));
        }
        Ok(Self { min, max })
    }
}

/// Per-detached-column metadata stored in `default[0x13 || attr_id BE]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetachedMeta {
    pub state: u8,
    pub current_gen: ColumnGen,
    pub pending_gen: ColumnGen,
    pub retired_below: ColumnGen,
    pub built_against_watermark: RowId,
}

impl Default for DetachedMeta {
    fn default() -> Self {
        Self {
            state: 0,
            current_gen: 0,
            pending_gen: 0,
            retired_below: 0,
            built_against_watermark: RowId(0),
        }
    }
}

impl DetachedMeta {
    pub fn encode(&self) -> [u8; DETACHED_META_BYTES] {
        let mut buf = [0u8; DETACHED_META_BYTES];
        buf[0] = 1; // version
        buf[1] = self.state;
        buf[2..4].copy_from_slice(&0u16.to_le_bytes());
        buf[4..8].copy_from_slice(&self.current_gen.to_le_bytes());
        buf[8..12].copy_from_slice(&self.pending_gen.to_le_bytes());
        buf[12..16].copy_from_slice(&self.retired_below.to_le_bytes());
        buf[16..24].copy_from_slice(&self.built_against_watermark.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != DETACHED_META_BYTES {
            return Err(TreeError::CorruptedData(format!(
                "DetachedMeta expected {DETACHED_META_BYTES} bytes, got {}",
                bytes.len()
            )));
        }
        if bytes[0] != 1 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported DetachedMeta version: {}",
                bytes[0]
            )));
        }
        if bytes[2] != 0 || bytes[3] != 0 {
            return Err(TreeError::CorruptedData(
                "invalid DetachedMeta reserved bytes".into(),
            ));
        }
        Ok(Self {
            state: bytes[1],
            current_gen: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            pending_gen: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            retired_below: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            built_against_watermark: RowId::from_le_bytes(bytes[16..24].try_into().unwrap()),
        })
    }
}

/// Stub for 256-row columnar block codec (implemented in Phase 7).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColumnBlock;

// ---------------------------------------------------------------------------
// RowView v3 (`cf_rows`)
// ---------------------------------------------------------------------------

struct ByteReader<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> ByteReader<'a> {
    #[inline]
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    #[inline]
    fn read_bytes(&mut self, len: usize, label: &str) -> Result<&'a [u8]> {
        let end = self.off.checked_add(len).ok_or_else(|| {
            TreeError::CorruptedData(format!("{label}: length overflow"))
        })?;
        if end > self.bytes.len() {
            return Err(TreeError::CorruptedData(format!("{label}: truncated")));
        }
        let slice = &self.bytes[self.off..end];
        self.off = end;
        Ok(slice)
    }

    #[inline]
    fn read_u32_le(&mut self, label: &str) -> Result<u32> {
        let s = self.read_bytes(4, label)?;
        Ok(u32::from_le_bytes(s.try_into().unwrap()))
    }

    #[inline]
    fn read_len_prefixed_str(&mut self, label: &str) -> Result<String> {
        let len = self.read_u32_le(label)? as usize;
        let s = self.read_bytes(len, label)?;
        std::str::from_utf8(s)
            .map(str::to_owned)
            .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in {label}: {e}")))
    }

    #[inline]
    fn expect_eof(&self, label: &str) -> Result<()> {
        if self.off != self.bytes.len() {
            return Err(TreeError::CorruptedData(format!("{label}: trailing bytes")));
        }
        Ok(())
    }
}

/// Encode inline document attributes into compact v3 binary form (`cf_rows`) with an
/// `AttrId`-sorted offset directory for O(log F) zero-copy field access.
///
/// Detached attributes in `attrs` are skipped (they live in `cf_detached`).
pub fn encode_row(
    schema: &CollectionSchema,
    attrs: &Attributes,
    has_vector: bool,
) -> Result<Vec<u8>> {
    let mut items: Vec<(AttrId, &AttributeValue)> = Vec::with_capacity(attrs.len());
    for (name, val) in attrs {
        let entry = schema
            .get_by_name(name)
            .ok_or_else(|| TreeError::InvalidAttribute {
                name: name.clone(),
                reason: "attribute is not declared in collection schema".into(),
            })?;
        if entry.storage == ColumnStorage::Detached {
            continue;
        }
        items.push((entry.id, val));
    }
    items.sort_unstable_by_key(|&(id, _)| id);

    let count = items.len();
    let dir_end = ROW_HEADER_LEN + count * ROW_DIR_ENTRY_LEN;
    let mut buf = Vec::with_capacity(dir_end + count * 12);
    buf.push(ROW_FORMAT_V3);
    buf.push(if has_vector { ROW_FLAG_HAS_VECTOR } else { 0 });
    buf.extend_from_slice(&(count as u16).to_le_bytes());
    buf.resize(dir_end, 0);

    for (idx, (attr_id, val)) in items.into_iter().enumerate() {
        let offset = buf.len() as u32;
        let kind = match val {
            AttributeValue::Null => ROW_VAL_NULL,
            AttributeValue::Bool(b) => {
                buf.push(if *b { 1 } else { 0 });
                ROW_VAL_BOOL
            }
            AttributeValue::Int(v) => {
                buf.extend_from_slice(&v.to_le_bytes());
                ROW_VAL_INT
            }
            AttributeValue::Uint(v) => {
                buf.extend_from_slice(&v.to_le_bytes());
                ROW_VAL_UINT
            }
            AttributeValue::Float(v) => {
                if v.is_nan() {
                    return Err(TreeError::InvalidAttribute {
                        name: schema
                            .get_by_id(attr_id)
                            .map(|e| e.name.clone())
                            .unwrap_or_default(),
                        reason: "NaN float attribute values are not allowed".into(),
                    });
                }
                buf.extend_from_slice(&v.to_le_bytes());
                ROW_VAL_FLOAT
            }
            AttributeValue::String(s) => {
                buf.extend_from_slice(s.as_bytes());
                ROW_VAL_STRING
            }
            AttributeValue::StringArray(arr) => {
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for s in arr {
                    let sb = s.as_bytes();
                    buf.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                    buf.extend_from_slice(sb);
                }
                ROW_VAL_STRING_ARRAY
            }
            AttributeValue::IntArray(arr) => {
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for &v in arr {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                ROW_VAL_INT_ARRAY
            }
            AttributeValue::UintArray(arr) => {
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for &v in arr {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                ROW_VAL_UINT_ARRAY
            }
        };
        let entry_off = ROW_HEADER_LEN + idx * ROW_DIR_ENTRY_LEN;
        buf[entry_off..entry_off + 2].copy_from_slice(&attr_id.to_le_bytes());
        buf[entry_off + 2] = kind;
        buf[entry_off + 3] = 0;
        buf[entry_off + 4..entry_off + 8].copy_from_slice(&offset.to_le_bytes());
    }
    Ok(buf)
}

/// Zero-copy view over a v3 `cf_rows` blob supporting O(log F) single-attribute reads.
#[derive(Debug, Clone, Copy)]
pub struct RowView<'a> {
    bytes: &'a [u8],
    flags: u8,
    count: usize,
}

impl<'a> RowView<'a> {
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < ROW_HEADER_LEN {
            return Err(TreeError::CorruptedData(
                "cf_rows payload too short".into(),
            ));
        }
        if bytes[0] != ROW_FORMAT_V3 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported cf_rows format version: {}",
                bytes[0]
            )));
        }
        let flags = bytes[1];
        if (flags & !ROW_FLAG_HAS_VECTOR) != 0 {
            return Err(TreeError::CorruptedData(format!(
                "unknown cf_rows flags: {flags:#04x}"
            )));
        }
        let count = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        let dir_end = ROW_HEADER_LEN + count * ROW_DIR_ENTRY_LEN;
        if bytes.len() < dir_end {
            return Err(TreeError::CorruptedData(
                "cf_rows directory truncated".into(),
            ));
        }
        Ok(Self {
            bytes,
            flags,
            count,
        })
    }

    #[inline]
    pub fn raw_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    #[inline]
    pub fn has_vector(&self) -> bool {
        (self.flags & ROW_FLAG_HAS_VECTOR) != 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn dir_entry(&self, idx: usize) -> (AttrId, u8, usize, usize) {
        let off = ROW_HEADER_LEN + idx * ROW_DIR_ENTRY_LEN;
        let attr_id = u16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]);
        let kind = self.bytes[off + 2];
        let start = u32::from_le_bytes(self.bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        let end = if idx + 1 < self.count {
            let next_off = ROW_HEADER_LEN + (idx + 1) * ROW_DIR_ENTRY_LEN;
            u32::from_le_bytes(
                self.bytes[next_off + 4..next_off + 8]
                    .try_into()
                    .unwrap(),
            ) as usize
        } else {
            self.bytes.len()
        };
        (attr_id, kind, start, end)
    }

    #[inline]
    fn find_entry(&self, target_id: AttrId) -> Option<(u8, &'a [u8])> {
        let mut lo = 0usize;
        let mut hi = self.count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let off = ROW_HEADER_LEN + mid * ROW_DIR_ENTRY_LEN;
            let id = u16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]);
            match id.cmp(&target_id) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => {
                    let (_, kind, start, end) = self.dir_entry(mid);
                    if start <= end && end <= self.bytes.len() {
                        return Some((kind, &self.bytes[start..end]));
                    }
                    return None;
                }
            }
        }
        None
    }

    pub fn decode_field(&self, target_id: AttrId) -> Result<Option<AttributeValue>> {
        let Some((kind, payload)) = self.find_entry(target_id) else {
            return Ok(None);
        };
        decode_payload_value(kind, payload).map(Some)
    }

    pub fn get_raw_field(&self, attr_id: AttrId) -> Result<Option<RowFieldRef<'a>>> {
        let Some((kind, payload)) = self.find_entry(attr_id) else {
            return Ok(None);
        };
        RowFieldRef::from_raw(kind, payload).map(Some)
    }

    #[inline]
    pub fn get_numeric_f64(&self, attr_id: AttrId) -> Result<Option<f64>> {
        Ok(match self.get_raw_field(attr_id)? {
            Some(RowFieldRef::Scalar(RawScalar::Int(v))) => Some(v as f64),
            Some(RowFieldRef::Scalar(RawScalar::Uint(v))) => Some(v as f64),
            Some(RowFieldRef::Scalar(RawScalar::Float(v))) => Some(v),
            _ => None,
        })
    }

    pub fn get_scalar(&self, attr_id: AttrId) -> Result<ScalarValue> {
        Ok(match self.get_raw_field(attr_id)? {
            Some(RowFieldRef::Scalar(RawScalar::Bool(v))) => ScalarValue::Bool(v),
            Some(RowFieldRef::Scalar(RawScalar::Int(v))) => ScalarValue::Int(v),
            Some(RowFieldRef::Scalar(RawScalar::Uint(v))) => ScalarValue::Uint(v),
            Some(RowFieldRef::Scalar(RawScalar::Float(v))) => ScalarValue::Float(v),
            Some(RowFieldRef::Scalar(RawScalar::Str(b))) => ScalarValue::String(
                String::from_utf8(b.to_vec())
                    .map_err(|_| TreeError::CorruptedData("cf_rows string not utf-8".into()))?,
            ),
            _ => ScalarValue::Null,
        })
    }

    /// Evaluate a [`Filter`] directly against this `RowView` without decoding untouched fields.
    pub fn matches_filter<F>(
        &self,
        schema: &CollectionSchema,
        filter: &Filter,
        eval_fts: &mut F,
    ) -> Result<bool>
    where
        F: FnMut(FtsPredicateKind, &str, &TokenQuery, bool, Option<&AttributeValue>) -> bool,
    {
        match filter {
            Filter::And(children) => {
                for c in children {
                    if !self.matches_filter(schema, c, eval_fts)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Filter::Or(children) => {
                for c in children {
                    if self.matches_filter(schema, c, eval_fts)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Filter::Not(child) => Ok(!self.matches_filter(schema, child, eval_fts)?),
            Filter::Eq(attr, target) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(false);
                };
                match (self.find_entry(entry.id), target) {
                    (None | Some((ROW_VAL_NULL, _)), ScalarValue::Null) => Ok(true),
                    (Some((kind, payload)), target) => raw_eq_scalar(kind, payload, target),
                    (None, _) => Ok(false),
                }
            }
            Filter::NotEq(attr, target) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(true);
                };
                match (self.find_entry(entry.id), target) {
                    (None | Some((ROW_VAL_NULL, _)), ScalarValue::Null) => Ok(false),
                    (Some((kind, payload)), target) => Ok(!raw_eq_scalar(kind, payload, target)?),
                    (None, _) => Ok(true),
                }
            }
            Filter::In(attr, targets) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(false);
                };
                match self.find_entry(entry.id) {
                    None | Some((ROW_VAL_NULL, _)) => {
                        Ok(targets.iter().any(|t| matches!(t, ScalarValue::Null)))
                    }
                    Some((kind, payload)) => {
                        for t in targets {
                            if raw_eq_scalar(kind, payload, t)? {
                                return Ok(true);
                            }
                        }
                        Ok(false)
                    }
                }
            }
            Filter::NotIn(attr, targets) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(true);
                };
                match self.find_entry(entry.id) {
                    None | Some((ROW_VAL_NULL, _)) => {
                        Ok(!targets.iter().any(|t| matches!(t, ScalarValue::Null)))
                    }
                    Some((kind, payload)) => {
                        for t in targets {
                            if raw_eq_scalar(kind, payload, t)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                }
            }
            Filter::Lt(attr, target) => {
                Ok(self.cmp_attr(schema, attr, target)?.is_some_and(|o| o == Ordering::Less))
            }
            Filter::Lte(attr, target) => {
                Ok(self.cmp_attr(schema, attr, target)?.is_some_and(|o| o != Ordering::Greater))
            }
            Filter::Gt(attr, target) => {
                Ok(self.cmp_attr(schema, attr, target)?.is_some_and(|o| o == Ordering::Greater))
            }
            Filter::Gte(attr, target) => {
                Ok(self.cmp_attr(schema, attr, target)?.is_some_and(|o| o != Ordering::Less))
            }
            Filter::Contains(attr, target) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(false);
                };
                let Some((kind, payload)) = self.find_entry(entry.id) else {
                    return Ok(false);
                };
                raw_contains_scalar(kind, payload, target)
            }
            Filter::ContainsAny(attr, targets) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(false);
                };
                let Some((kind, payload)) = self.find_entry(entry.id) else {
                    return Ok(false);
                };
                for t in targets {
                    if raw_contains_scalar(kind, payload, t)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Filter::IsNull(attr) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(true);
                };
                Ok(matches!(
                    self.find_entry(entry.id),
                    None | Some((ROW_VAL_NULL, _))
                ))
            }
            Filter::IsNotNull(attr) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(false);
                };
                Ok(!matches!(
                    self.find_entry(entry.id),
                    None | Some((ROW_VAL_NULL, _))
                ))
            }
            other => {
                if let Some((kind, attr, query, last_as_prefix)) = other.as_fts_predicate() {
                    let val = match schema.get_by_name(attr) {
                        Some(e) => self.decode_field(e.id)?,
                        None => None,
                    };
                    Ok(eval_fts(kind, attr, query, last_as_prefix, val.as_ref()))
                } else {
                    unreachable!("all non-FTS Filter variants handled above")
                }
            }
        }
    }

    fn cmp_attr(
        &self,
        schema: &CollectionSchema,
        attr: &str,
        target: &ScalarValue,
    ) -> Result<Option<Ordering>> {
        let Some(entry) = schema.get_by_name(attr) else {
            return Ok(None);
        };
        let Some((kind, payload)) = self.find_entry(entry.id) else {
            return Ok(None);
        };
        raw_cmp_scalar(kind, payload, target)
    }
}

/// One scalar element borrowed from a `cf_rows` blob.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RawScalar<'a> {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    Str(&'a [u8]),
}

impl RawScalar<'_> {
    #[inline]
    pub fn encode_into(&self, out: &mut Vec<u8>) -> u8 {
        match *self {
            RawScalar::Bool(b) => {
                out.push(b as u8);
                TAG_BOOL
            }
            RawScalar::Int(v) => {
                out.extend_from_slice(&encode_i64_ordered(v));
                TAG_INT
            }
            RawScalar::Uint(v) => {
                out.extend_from_slice(&v.to_be_bytes());
                TAG_UINT
            }
            RawScalar::Float(v) => {
                let bytes = encode_f64_ordered(v).unwrap_or([0xFF; 8]);
                out.extend_from_slice(&bytes);
                TAG_FLOAT
            }
            RawScalar::Str(s) => {
                encode_str_ordered(s, out);
                TAG_STRING
            }
        }
    }
}

/// Zero-copy borrowed attribute value from a [`RowView`].
#[derive(Debug, Clone, Copy)]
pub enum RowFieldRef<'a> {
    Null,
    Scalar(RawScalar<'a>),
    StringArray(&'a [u8]),
    IntArray(&'a [u8]),
    UintArray(&'a [u8]),
}

impl<'a> RowFieldRef<'a> {
    fn from_raw(kind: u8, payload: &'a [u8]) -> Result<Self> {
        let fixed8 = |what: &str| -> Result<[u8; 8]> {
            payload
                .try_into()
                .map_err(|_| TreeError::CorruptedData(format!("cf_rows {what} truncated")))
        };
        Ok(match kind {
            ROW_VAL_NULL => Self::Null,
            ROW_VAL_BOOL => match payload {
                [0] => Self::Scalar(RawScalar::Bool(false)),
                [1] => Self::Scalar(RawScalar::Bool(true)),
                _ => return Err(TreeError::CorruptedData("cf_rows bool invalid".into())),
            },
            ROW_VAL_INT => Self::Scalar(RawScalar::Int(i64::from_le_bytes(fixed8("i64")?))),
            ROW_VAL_UINT => Self::Scalar(RawScalar::Uint(u64::from_le_bytes(fixed8("u64")?))),
            ROW_VAL_FLOAT => Self::Scalar(RawScalar::Float(f64::from_le_bytes(fixed8("f64")?))),
            ROW_VAL_STRING => Self::Scalar(RawScalar::Str(payload)),
            ROW_VAL_STRING_ARRAY => Self::StringArray(payload),
            ROW_VAL_INT_ARRAY => Self::IntArray(payload),
            ROW_VAL_UINT_ARRAY => Self::UintArray(payload),
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "unknown cf_rows value kind: {other}"
                )));
            }
        })
    }

    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn for_each_scalar(&self, mut f: impl FnMut(RawScalar<'a>)) -> Result<()> {
        match *self {
            Self::Null => {}
            Self::Scalar(v) => f(v),
            Self::StringArray(payload) => {
                let mut r = ByteReader::new(payload);
                let len = r.read_u32_le("cf_rows string array len")? as usize;
                for _ in 0..len {
                    let n = r.read_u32_le("cf_rows string elem len")? as usize;
                    f(RawScalar::Str(r.read_bytes(n, "cf_rows string elem")?));
                }
                r.expect_eof("cf_rows string array")?;
            }
            Self::IntArray(payload) => {
                let mut r = ByteReader::new(payload);
                let len = r.read_u32_le("cf_rows int array len")? as usize;
                let body = r.read_bytes(len * 8, "cf_rows int array")?;
                r.expect_eof("cf_rows int array")?;
                for c in body.chunks_exact(8) {
                    f(RawScalar::Int(i64::from_le_bytes(c.try_into().unwrap())));
                }
            }
            Self::UintArray(payload) => {
                let mut r = ByteReader::new(payload);
                let len = r.read_u32_le("cf_rows uint array len")? as usize;
                let body = r.read_bytes(len * 8, "cf_rows uint array")?;
                r.expect_eof("cf_rows uint array")?;
                for c in body.chunks_exact(8) {
                    f(RawScalar::Uint(u64::from_le_bytes(c.try_into().unwrap())));
                }
            }
        }
        Ok(())
    }
}

fn raw_eq_scalar(kind: u8, payload: &[u8], target: &ScalarValue) -> Result<bool> {
    match (kind, target) {
        (ROW_VAL_NULL, ScalarValue::Null) => Ok(true),
        (ROW_VAL_BOOL, ScalarValue::Bool(b)) => {
            if payload.len() != 1 {
                return Err(TreeError::CorruptedData("cf_rows bool len".into()));
            }
            Ok((payload[0] != 0) == *b)
        }
        (ROW_VAL_INT, ScalarValue::Int(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_rows i64 len".into()));
            }
            Ok(i64::from_le_bytes(payload.try_into().unwrap()) == *b)
        }
        (ROW_VAL_UINT, ScalarValue::Uint(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_rows u64 len".into()));
            }
            Ok(u64::from_le_bytes(payload.try_into().unwrap()) == *b)
        }
        (ROW_VAL_FLOAT, ScalarValue::Float(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_rows f64 len".into()));
            }
            let a = f64::from_le_bytes(payload.try_into().unwrap());
            Ok(!a.is_nan() && !b.is_nan() && a == *b)
        }
        (ROW_VAL_STRING, ScalarValue::String(b)) => Ok(payload == b.as_bytes()),
        _ => Ok(false),
    }
}

fn raw_cmp_scalar(kind: u8, payload: &[u8], target: &ScalarValue) -> Result<Option<Ordering>> {
    match (kind, target) {
        (ROW_VAL_INT, ScalarValue::Int(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_rows i64 len".into()));
            }
            Ok(Some(i64::from_le_bytes(payload.try_into().unwrap()).cmp(b)))
        }
        (ROW_VAL_UINT, ScalarValue::Uint(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_rows u64 len".into()));
            }
            Ok(Some(u64::from_le_bytes(payload.try_into().unwrap()).cmp(b)))
        }
        (ROW_VAL_FLOAT, ScalarValue::Float(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_rows f64 len".into()));
            }
            let a = f64::from_le_bytes(payload.try_into().unwrap());
            if a.is_nan() || b.is_nan() {
                Ok(None)
            } else {
                let an = if a == 0.0 { 0.0 } else { a };
                let bn = if *b == 0.0 { 0.0 } else { *b };
                Ok(Some(an.total_cmp(&bn)))
            }
        }
        (ROW_VAL_STRING, ScalarValue::String(b)) => {
            let s = std::str::from_utf8(payload)
                .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in cf_rows: {e}")))?;
            Ok(Some(s.cmp(b.as_str())))
        }
        _ => Ok(None),
    }
}

fn raw_contains_scalar(kind: u8, payload: &[u8], target: &ScalarValue) -> Result<bool> {
    match (kind, target) {
        (ROW_VAL_STRING_ARRAY, ScalarValue::String(t)) => {
            if payload.len() < 4 {
                return Err(TreeError::CorruptedData("cf_rows str array len".into()));
            }
            let arr_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            let mut off = 4usize;
            let tb = t.as_bytes();
            for _ in 0..arr_len {
                if off + 4 > payload.len() {
                    return Err(TreeError::CorruptedData("cf_rows str elem len".into()));
                }
                let slen = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
                off += 4;
                if off + slen > payload.len() {
                    return Err(TreeError::CorruptedData("cf_rows str elem".into()));
                }
                if &payload[off..off + slen] == tb {
                    return Ok(true);
                }
                off += slen;
            }
            Ok(false)
        }
        (ROW_VAL_INT_ARRAY, ScalarValue::Int(t)) => {
            if payload.len() < 4 {
                return Err(TreeError::CorruptedData("cf_rows int array len".into()));
            }
            let arr_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            if payload.len() != 4 + arr_len * 8 {
                return Err(TreeError::CorruptedData("cf_rows int array".into()));
            }
            for chunk in payload[4..].chunks_exact(8) {
                if i64::from_le_bytes(chunk.try_into().unwrap()) == *t {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        (ROW_VAL_UINT_ARRAY, ScalarValue::Uint(t)) => {
            if payload.len() < 4 {
                return Err(TreeError::CorruptedData("cf_rows uint array len".into()));
            }
            let arr_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            if payload.len() != 4 + arr_len * 8 {
                return Err(TreeError::CorruptedData("cf_rows uint array".into()));
            }
            for chunk in payload[4..].chunks_exact(8) {
                if u64::from_le_bytes(chunk.try_into().unwrap()) == *t {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

fn decode_payload_value(kind: u8, payload: &[u8]) -> Result<AttributeValue> {
    match kind {
        ROW_VAL_NULL => {
            if !payload.is_empty() {
                return Err(TreeError::CorruptedData("cf_rows null non-empty".into()));
            }
            Ok(AttributeValue::Null)
        }
        ROW_VAL_BOOL => {
            if payload.len() != 1 {
                return Err(TreeError::CorruptedData("cf_rows bool truncated".into()));
            }
            let b = match payload[0] {
                0 => false,
                1 => true,
                other => {
                    return Err(TreeError::CorruptedData(format!(
                        "invalid bool byte in cf_rows: {other}"
                    )));
                }
            };
            Ok(AttributeValue::Bool(b))
        }
        ROW_VAL_INT => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_rows i64 truncated".into()));
            }
            Ok(AttributeValue::Int(i64::from_le_bytes(
                payload.try_into().unwrap(),
            )))
        }
        ROW_VAL_UINT => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_rows u64 truncated".into()));
            }
            Ok(AttributeValue::Uint(u64::from_le_bytes(
                payload.try_into().unwrap(),
            )))
        }
        ROW_VAL_FLOAT => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_rows f64 truncated".into()));
            }
            Ok(AttributeValue::Float(f64::from_le_bytes(
                payload.try_into().unwrap(),
            )))
        }
        ROW_VAL_STRING => {
            let s = std::str::from_utf8(payload)
                .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in cf_rows: {e}")))?
                .to_owned();
            Ok(AttributeValue::String(s))
        }
        ROW_VAL_STRING_ARRAY => {
            let mut r = ByteReader::new(payload);
            let len = r.read_u32_le("cf_rows string array len")? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(r.read_len_prefixed_str("cf_rows string elem")?);
            }
            r.expect_eof("cf_rows string array")?;
            Ok(AttributeValue::StringArray(arr))
        }
        ROW_VAL_INT_ARRAY => {
            let mut r = ByteReader::new(payload);
            let len = r.read_u32_le("cf_rows int array len")? as usize;
            let body = r.read_bytes(len * 8, "cf_rows int array")?;
            r.expect_eof("cf_rows int array")?;
            let arr = body
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(AttributeValue::IntArray(arr))
        }
        ROW_VAL_UINT_ARRAY => {
            let mut r = ByteReader::new(payload);
            let len = r.read_u32_le("cf_rows uint array len")? as usize;
            let body = r.read_bytes(len * 8, "cf_rows uint array")?;
            r.expect_eof("cf_rows uint array")?;
            let arr = body
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(AttributeValue::UintArray(arr))
        }
        other => Err(TreeError::CorruptedData(format!(
            "unknown cf_rows value kind: {other}"
        ))),
    }
}

pub fn decode_row_entries(bytes: &[u8]) -> Result<Vec<(AttrId, AttributeValue)>> {
    let view = RowView::parse(bytes)?;
    let mut out = Vec::with_capacity(view.count);
    for idx in 0..view.count {
        let (attr_id, kind, start, end) = view.dir_entry(idx);
        if end < start || end > bytes.len() {
            return Err(TreeError::CorruptedData(
                "cf_rows directory offset out of bounds".into(),
            ));
        }
        let val = decode_payload_value(kind, &bytes[start..end])?;
        out.push((attr_id, val));
    }
    Ok(out)
}

pub fn decode_row_attributes(schema: &CollectionSchema, bytes: &[u8]) -> Result<Attributes> {
    let entries = decode_row_entries(bytes)?;
    let mut map = Attributes::new();
    for (attr_id, val) in entries {
        let entry = schema.get_by_id(attr_id).ok_or_else(|| {
            TreeError::CorruptedData(format!("cf_rows references unknown attr_id {attr_id}"))
        })?;
        map.insert(entry.name.clone(), val);
    }
    Ok(map)
}

/// Extract all distinct `attr_key`s (`attr_id BE || tag || encoded_val`) to index for a row's
/// `(AttrId, AttributeValue)` list (filterable inline attributes only).
pub fn extract_index_keys_from_entries(
    schema: &CollectionSchema,
    entries: &[(AttrId, AttributeValue)],
) -> Result<Vec<Vec<u8>>> {
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(entries.len() * 2);
    for (attr_id, val) in entries {
        let Some(schema_entry) = schema.get_by_id(*attr_id) else {
            return Err(TreeError::CorruptedData(format!(
                "unknown attr_id {attr_id} in row"
            )));
        };
        if !schema_entry.filterable || schema_entry.storage != ColumnStorage::Inline {
            continue;
        }
        append_index_keys_for_value(*attr_id, val, &mut keys)?;
    }
    keys.sort_unstable();
    keys.dedup();
    Ok(keys)
}

pub fn extract_index_keys(schema: &CollectionSchema, attrs: &Attributes) -> Result<Vec<Vec<u8>>> {
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(attrs.len() * 2);
    for (name, val) in attrs {
        let Some(schema_entry) = schema.get_by_name(name) else {
            return Err(TreeError::InvalidAttribute {
                name: name.clone(),
                reason: "attribute is not declared in collection schema".into(),
            });
        };
        if !schema_entry.filterable || schema_entry.storage != ColumnStorage::Inline {
            continue;
        }
        append_index_keys_for_value(schema_entry.id, val, &mut keys)?;
    }
    keys.sort_unstable();
    keys.dedup();
    Ok(keys)
}

/// Extract distinct `attr_key`s (`attr_id BE || tag || encoded_val`) for a single attribute value.
pub fn extract_index_keys_for_attr(attr_id: AttrId, val: &AttributeValue) -> Result<Vec<Vec<u8>>> {
    let mut keys = Vec::new();
    append_index_keys_for_value(attr_id, val, &mut keys)?;
    keys.sort_unstable();
    keys.dedup();
    Ok(keys)
}

fn append_index_keys_for_value(
    attr_id: AttrId,
    val: &AttributeValue,
    keys: &mut Vec<Vec<u8>>,
) -> Result<()> {
    if matches!(val, AttributeValue::Null) {
        return Ok(());
    }
    keys.push(encode_present_attr_key(attr_id));
    match val {
        AttributeValue::Null => unreachable!(),
        AttributeValue::Bool(b) => {
            keys.push(encode_scalar_attr_key(attr_id, &ScalarValue::Bool(*b))?);
        }
        AttributeValue::Int(v) => {
            keys.push(encode_scalar_attr_key(attr_id, &ScalarValue::Int(*v))?);
        }
        AttributeValue::Uint(v) => {
            keys.push(encode_scalar_attr_key(attr_id, &ScalarValue::Uint(*v))?);
        }
        AttributeValue::Float(v) => {
            keys.push(encode_scalar_attr_key(attr_id, &ScalarValue::Float(*v))?);
        }
        AttributeValue::String(s) => {
            let mut key = Vec::with_capacity(5 + s.len());
            key.extend_from_slice(&attr_id.to_be_bytes());
            key.push(TAG_STRING);
            encode_str_ordered(s.as_bytes(), &mut key);
            keys.push(key);
        }
        AttributeValue::StringArray(arr) => {
            for s in arr {
                let mut key = Vec::with_capacity(5 + s.len());
                key.extend_from_slice(&attr_id.to_be_bytes());
                key.push(TAG_STRING);
                encode_str_ordered(s.as_bytes(), &mut key);
                keys.push(key);
            }
        }
        AttributeValue::IntArray(arr) => {
            for &v in arr {
                let mut key = Vec::with_capacity(11);
                key.extend_from_slice(&attr_id.to_be_bytes());
                key.push(TAG_INT);
                key.extend_from_slice(&encode_i64_ordered(v));
                keys.push(key);
            }
        }
        AttributeValue::UintArray(arr) => {
            for &v in arr {
                let mut key = Vec::with_capacity(11);
                key.extend_from_slice(&attr_id.to_be_bytes());
                key.push(TAG_UINT);
                key.extend_from_slice(&v.to_be_bytes());
                keys.push(key);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Roaring & FTS term dictionary keys
// ---------------------------------------------------------------------------

pub fn encode_roaring_bitmap(bm: &RoaringBitmap) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(bm.serialized_size());
    bm.serialize_into(&mut buf)
        .map_err(|e| TreeError::Storage(format!("roaring serialize error: {e}")))?;
    Ok(buf)
}

pub fn decode_roaring_bitmap(bytes: &[u8]) -> Result<RoaringBitmap> {
    RoaringBitmap::deserialize_from(bytes)
        .map_err(|e| TreeError::CorruptedData(format!("roaring deserialize error: {e}")))
}

#[inline]
pub fn encode_fts_term_key(attr_id: AttrId, token: &str) -> Vec<u8> {
    let tb = token.as_bytes();
    let mut key = Vec::with_capacity(2 + tb.len());
    key.extend_from_slice(&attr_id.to_be_bytes());
    key.extend_from_slice(tb);
    key
}

pub fn decode_fts_term_key(key: &[u8]) -> Result<(AttrId, String)> {
    if key.len() < 2 {
        return Err(TreeError::CorruptedData(
            "cf_fts_terms key too short".into(),
        ));
    }
    let attr_id = u16::from_be_bytes([key[0], key[1]]);
    let token = std::str::from_utf8(&key[2..])
        .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in cf_fts_terms key: {e}")))?
        .to_owned();
    Ok((attr_id, token))
}

#[inline]
pub fn encode_fts_term_id_prefix(term_id: TermId) -> [u8; 4] {
    term_id.to_be_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use izzydb_core::config::{DistanceMetric, TreeConfig};
    use izzydb_core::schema::CollectionConfig;

    #[test]
    fn key_ordering_and_splitting() {
        // row_key order == RowId order
        let rows = [
            RowId::new(0, 0),
            RowId::new(0, 255),
            RowId::new(0, 256),
            RowId::new(1, 0),
            RowId::new(u32::MAX, 10),
        ];
        let rkeys: Vec<[u8; 8]> = rows.iter().map(|&r| row_key(r)).collect();
        for w in rkeys.windows(2) {
            assert!(w[0] < w[1]);
        }
        for (&r, &k) in rows.iter().zip(rkeys.iter()) {
            assert_eq!(decode_row_key(&k).unwrap(), r);
        }

        // posting_key for (attr, tag, val) scans segments ascending, even with embedded 0x00 in strings
        let s = "a\0b\0c";
        let mut val_bytes = Vec::new();
        encode_str_ordered(s.as_bytes(), &mut val_bytes);
        let pk1 = posting_key(7, TAG_STRING, &val_bytes, 2);
        let pk2 = posting_key(7, TAG_STRING, &val_bytes, 10);
        assert!(pk1 < pk2);
        let (pre1, seg1) = split_posting_key(&pk1).unwrap();
        let (pre2, seg2) = split_posting_key(&pk2).unwrap();
        assert_eq!(seg1, 2);
        assert_eq!(seg2, 10);
        assert_eq!(pre1, pre2);
        let (a, tag, vb, seg) = split_posting_key_parts(&pk1).unwrap();
        assert_eq!((a, tag, seg), (7, TAG_STRING, 2));
        assert_eq!(decode_str_ordered(vb).unwrap(), s);

        // detached posting key
        let dpk = detached_posting_key(3, 99, TAG_STRING, &val_bytes, 5);
        let (dpre, dseg) = split_detached_posting_key(&dpk).unwrap();
        assert_eq!(dseg, 5);
        assert!(dpre.starts_with(&detached_posting_prefix(3, 99, TAG_STRING)));
    }

    #[test]
    fn fixed_codecs_roundtrip_and_corruption() {
        let m = Manifest {
            open_segment: Some(3),
            next_segment_id: 4,
            seal_rows: 65_536,
            live_rows: 1000,
            total_rows: 1200,
        };
        assert_eq!(Manifest::decode(&m.encode()).unwrap(), m);
        let m2 = Manifest::new(1024);
        assert_eq!(Manifest::decode(&m2.encode()).unwrap(), m2);

        let sm = SegmentMeta {
            state: SegmentState::Forwarding,
            flags: SEG_FLAG_HAS_COLUMNS | SEG_FLAG_HAS_VECTORS,
            row_count: 500,
            deleted_count: 100,
            forward_target: 9,
            forward_base: 42,
            forwarded_at_epoch: 7,
            created_seq: 111,
            sealed_seq: 222,
        };
        assert_eq!(SegmentMeta::decode(&sm.encode()).unwrap(), sm);

        let mut ze = ZoneEntry::singleton(b"hello");
        assert!(ze.widen(b"alpha"));
        assert!(ze.widen(b"zeta"));
        assert!(!ze.widen(b"mid"));
        assert_eq!(ZoneEntry::decode(&ze.encode()).unwrap(), ze);

        let dm = DetachedMeta {
            state: 1,
            current_gen: 4,
            pending_gen: 5,
            retired_below: 3,
            built_against_watermark: RowId::new(2, 99),
        };
        assert_eq!(DetachedMeta::decode(&dm.encode()).unwrap(), dm);

        let mut block = [0u64; ROW_BLOCK_SIZE];
        block[0] = 11;
        block[255] = u64::MAX;
        let enc_b = encode_doc_id_block(&block);
        assert_eq!(enc_b.len(), DOC_ID_BLOCK_BYTES);
        assert_eq!(*decode_doc_id_block(&enc_b).unwrap(), block);

        // Random / invalid lengths rejected
        for bad_len in [0usize, 1, 5, 31, 33, 43, 45, 2047, 2049] {
            let buf = vec![0u8; bad_len];
            assert!(Manifest::decode(&buf).is_err());
            assert!(SegmentMeta::decode(&buf).is_err());
            assert!(DetachedMeta::decode(&buf).is_err());
            assert!(decode_doc_id_block(&buf).is_err());
        }
        assert!(ZoneEntry::decode(&[0u8; 3]).is_err());
        assert!(ZoneEntry::decode(&ZoneEntry { min: vec![b'z'], max: vec![b'a'] }.encode()).is_err());
    }

    #[test]
    fn seg_posting_roundtrip_sparse_and_roaring() {
        let mut sp = SegPosting::default();
        for i in 0..32u32 {
            sp.insert(i * 1000);
        }
        let b1 = sp.encode().unwrap();
        assert_eq!(b1[0], SEG_POSTING_SPARSE);
        assert_eq!(SegPosting::decode(&b1).unwrap(), sp);

        let mut bm = RoaringBitmap::new();
        SegPostingView::parse(&b1).unwrap().or_into(&mut bm).unwrap();
        assert_eq!(&bm, sp.as_bitmap());

        sp.insert(99_999);
        let b2 = sp.encode().unwrap();
        assert_eq!(b2[0], SEG_POSTING_ROARING);
        assert_eq!(SegPosting::decode(&b2).unwrap(), sp);

        // Unsorted sparse payload rejected
        let mut bad = vec![SEG_POSTING_SPARSE];
        bad.extend_from_slice(&5u32.to_le_bytes());
        bad.extend_from_slice(&5u32.to_le_bytes());
        assert!(SegPostingView::parse(&bad).is_err());
    }

    #[test]
    fn row_view_v3_roundtrip_and_detached_skip() {
        let cfg = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared))
            .with_attribute("tag", AttributeType::String, true)
            .with_attribute("score", AttributeType::Float, true)
            .with_detached_attribute("pop", AttributeType::Float, true);
        let schema = CollectionSchema::from_config(&cfg).unwrap();

        let mut attrs = Attributes::new();
        attrs.insert("tag".into(), "alpha".into());
        attrs.insert("score".into(), 3.5f64.into());
        attrs.insert("pop".into(), 99.0f64.into());

        let bytes = encode_row(&schema, &attrs, true).unwrap();
        let view = RowView::parse(&bytes).unwrap();
        assert!(view.has_vector());
        assert_eq!(view.len(), 2); // detached 'pop' omitted from row

        let decoded = decode_row_attributes(&schema, &bytes).unwrap();
        assert!(!decoded.contains_key("pop"));
        assert_eq!(decoded.get("tag"), Some(&AttributeValue::String("alpha".into())));
        assert_eq!(
            view.get_numeric_f64(schema.get_by_name("score").unwrap().id).unwrap(),
            Some(3.5)
        );
        assert!(
            view.matches_filter(
                &schema,
                &Filter::Eq("tag".into(), "alpha".into()),
                &mut |_, _, _, _, _| false
            )
            .unwrap()
        );
    }
}
