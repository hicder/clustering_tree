
use izzydb_core::config::{
    AttrId, ColumnGen, RowBlock, RowId, SegmentId, SkipChunk,
    TermId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::ScalarValue;
use super::*;

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

/// Build `default[0x07]` key suffix (`attr_id BE || gen BE || tag || encoded_val`)
/// given `attr_key` (`attr_id BE || tag || encoded_val`).
#[inline]
pub fn detached_count_suffix(col_gen: ColumnGen, attr_key: &[u8]) -> Vec<u8> {
    debug_assert!(attr_key.len() >= 3);
    let mut out = Vec::with_capacity(4 + attr_key.len());
    out.extend_from_slice(&attr_key[0..2]);
    out.extend_from_slice(&col_gen.to_be_bytes());
    out.extend_from_slice(&attr_key[2..]);
    out
}

/// 6-byte `(attr_id BE || gen BE)` prefix and exclusive upper bound for `DeleteRange` GC.
#[inline]
pub fn detached_gen_bounds(attr_id: AttrId, col_gen: ColumnGen) -> ([u8; 6], [u8; 6]) {
    let mut lo = [0u8; 6];
    lo[0..2].copy_from_slice(&attr_id.to_be_bytes());
    lo[2..6].copy_from_slice(&col_gen.to_be_bytes());
    let mut hi = lo;
    for b in hi.iter_mut().rev() {
        if *b < 0xFF {
            *b += 1;
            break;
        }
        *b = 0;
    }
    (lo, hi)
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
