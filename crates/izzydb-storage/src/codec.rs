use std::cmp::Ordering;
use std::collections::BTreeMap;

use izzydb_core::config::{AttrId, FtsBlockId, LocalId, NodeId, TermId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::{Filter, FtsPredicateKind, TokenQuery};
use izzydb_core::schema::{
    AttributeType, AttributeValue, Attributes, CollectionSchema, ScalarValue,
};
use roaring::RoaringBitmap;

/// Reserved `type_tag` tracking non-null attribute presence for `IsNull` / `IsNotNull`.
pub const TAG_PRESENT: u8 = 0x00;
pub const TAG_BOOL: u8 = 0x01;
pub const TAG_INT: u8 = 0x02;
pub const TAG_UINT: u8 = 0x03;
pub const TAG_FLOAT: u8 = 0x04;
pub const TAG_STRING: u8 = 0x05;

const DOC_FORMAT_V2: u8 = 0x02;
const DOC_HEADER_LEN: usize = 4;
const DOC_DIR_ENTRY_LEN: usize = 8;

const DOC_VAL_NULL: u8 = 0;
const DOC_VAL_BOOL: u8 = 1;
const DOC_VAL_INT: u8 = 2;
const DOC_VAL_UINT: u8 = 3;
const DOC_VAL_FLOAT: u8 = 4;
const DOC_VAL_STRING: u8 = 5;
const DOC_VAL_STRING_ARRAY: u8 = 6;
const DOC_VAL_INT_ARRAY: u8 = 7;
const DOC_VAL_UINT_ARRAY: u8 = 8;

const LEAF_POSTING_SPARSE: u8 = 0x00;
const LEAF_POSTING_DENSE: u8 = 0x01;
const LEAF_POSTING_ROARING: u8 = 0x02;
const LEAF_POSTING_SPARSE_MAX: usize = 16;
const LEAF_POSTING_DENSE_MAX_LOCAL_ID: LocalId = 4095;

/// 3-byte RocksDB prefix (`attr_id BE || type_tag`) for `cf_attr_clusters` and `cf_attr_zones`.
#[inline]
pub fn encode_attr_prefix(attr_id: AttrId, type_tag: u8) -> [u8; 3] {
    let id = attr_id.to_be_bytes();
    [id[0], id[1], type_tag]
}

/// 7-byte RocksDB prefix (`attr_id BE || type_tag || cluster_id BE`) for cluster-first `cf_attr_postings`.
#[inline]
pub fn encode_posting_prefix(attr_id: AttrId, type_tag: u8, cluster_id: NodeId) -> [u8; 7] {
    let a = attr_id.to_be_bytes();
    let c = cluster_id.to_be_bytes();
    [a[0], a[1], type_tag, c[0], c[1], c[2], c[3]]
}

/// Build the `cf_attr_clusters` key for tracking presence (`TAG_PRESENT`) of `attr_id`.
#[inline]
pub fn encode_present_cluster_key(attr_id: AttrId) -> Vec<u8> {
    encode_attr_prefix(attr_id, TAG_PRESENT).to_vec()
}

/// Build a `cf_attr_clusters` key: `attr_id (u16 BE) || type_tag (u8) || encoded_val`.
pub fn encode_scalar_cluster_key(attr_id: AttrId, val: &ScalarValue) -> Result<Vec<u8>> {
    let mut key = Vec::with_capacity(16);
    key.extend_from_slice(&attr_id.to_be_bytes());
    key.push(0); // placeholder for type_tag
    let tag = encode_scalar_val_bytes(val, &mut key)?;
    key[2] = tag;
    Ok(key)
}

/// Build a cluster-first `cf_attr_postings` key from a `cf_attr_clusters` key and `cluster_id`:
/// `attr_id (u16 BE) || type_tag (u8) || cluster_id (u32 BE) || encoded_val`.
#[inline]
pub fn make_posting_key(cluster_key: &[u8], cluster_id: NodeId) -> Vec<u8> {
    debug_assert!(cluster_key.len() >= 3);
    let mut key = Vec::with_capacity(cluster_key.len() + 4);
    key.extend_from_slice(&cluster_key[..3]);
    key.extend_from_slice(&cluster_id.to_be_bytes());
    key.extend_from_slice(&cluster_key[3..]);
    key
}

/// Build a cluster-first `cf_attr_postings` key from `(attr_id, type_tag, cluster_id, encoded_val)`.
#[inline]
pub fn encode_posting_key_parts(
    attr_id: AttrId,
    type_tag: u8,
    cluster_id: NodeId,
    encoded_val: &[u8],
) -> Vec<u8> {
    let mut key = Vec::with_capacity(7 + encoded_val.len());
    key.extend_from_slice(&encode_posting_prefix(attr_id, type_tag, cluster_id));
    key.extend_from_slice(encoded_val);
    key
}

/// Split a cluster-first `cf_attr_postings` key into `(attr_id, type_tag, cluster_id, encoded_val)`.
#[inline]
pub fn split_posting_key(posting_key: &[u8]) -> Result<(AttrId, u8, NodeId, &[u8])> {
    if posting_key.len() < 7 {
        return Err(TreeError::CorruptedData(format!(
            "cf_attr_postings key too short: {}",
            posting_key.len()
        )));
    }
    let attr_id = u16::from_be_bytes([posting_key[0], posting_key[1]]);
    let type_tag = posting_key[2];
    let cluster_id = u32::from_be_bytes(posting_key[3..7].try_into().unwrap());
    Ok((attr_id, type_tag, cluster_id, &posting_key[7..]))
}

/// Reconstruct the `cf_attr_clusters` key from a cluster-first `cf_attr_postings` key.
#[inline]
pub fn cluster_key_from_posting_key(posting_key: &[u8]) -> Result<Vec<u8>> {
    if posting_key.len() < 7 {
        return Err(TreeError::CorruptedData(format!(
            "cf_attr_postings key too short: {}",
            posting_key.len()
        )));
    }
    let mut out = Vec::with_capacity(posting_key.len() - 4);
    out.extend_from_slice(&posting_key[..3]);
    out.extend_from_slice(&posting_key[7..]);
    Ok(out)
}

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

/// Compact per-leaf posting set stored in `cf_attr_postings` using sparse `u16`s,
/// dense `[u64; words]` bitsets, or a `RoaringBitmap` fallback for very large `LocalId`s.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LeafPosting {
    bm: RoaringBitmap,
}

impl LeafPosting {
    pub fn from_bitmap(bm: RoaringBitmap) -> Self {
        Self { bm }
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

    pub fn encode(&self) -> Result<Vec<u8>> {
        let len = self.bm.len() as usize;
        let max_id = self.bm.max().unwrap_or(0);
        if len <= LEAF_POSTING_SPARSE_MAX && max_id <= u16::MAX as u32 {
            let mut buf = Vec::with_capacity(1 + len * 2);
            buf.push(LEAF_POSTING_SPARSE);
            for id in self.bm.iter() {
                buf.extend_from_slice(&(id as u16).to_le_bytes());
            }
            return Ok(buf);
        }
        if max_id <= LEAF_POSTING_DENSE_MAX_LOCAL_ID {
            let num_words = (max_id as usize / 64) + 1;
            let mut words = vec![0u64; num_words];
            for id in self.bm.iter() {
                words[(id as usize) >> 6] |= 1u64 << (id & 63);
            }
            let mut buf = Vec::with_capacity(3 + num_words * 8);
            buf.push(LEAF_POSTING_DENSE);
            buf.extend_from_slice(&(num_words as u16).to_le_bytes());
            for w in words {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            return Ok(buf);
        }
        let mut buf = Vec::with_capacity(1 + self.bm.serialized_size());
        buf.push(LEAF_POSTING_ROARING);
        self.bm
            .serialize_into(&mut buf)
            .map_err(|e| TreeError::Storage(format!("roaring serialize error: {e}")))?;
        Ok(buf)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let view = LeafPostingView::parse(bytes)?;
        match view {
            LeafPostingView::Sparse(slice) => {
                let mut bm = RoaringBitmap::new();
                for chunk in slice.chunks_exact(2) {
                    bm.insert(u16::from_le_bytes([chunk[0], chunk[1]]) as u32);
                }
                Ok(Self { bm })
            }
            LeafPostingView::Dense(slice) => {
                let mut bm = RoaringBitmap::new();
                for (w_idx, chunk) in slice.chunks_exact(8).enumerate() {
                    let mut w = u64::from_le_bytes(chunk.try_into().unwrap());
                    while w != 0 {
                        let bit = w.trailing_zeros();
                        w &= w - 1;
                        bm.insert(((w_idx as u32) << 6) | bit);
                    }
                }
                Ok(Self { bm })
            }
            LeafPostingView::Roaring(slice) => {
                let bm = decode_roaring_bitmap(slice)?;
                Ok(Self { bm })
            }
        }
    }
}

/// Zero-copy borrowed view over an encoded [`LeafPosting`] in `cf_attr_postings`.
#[derive(Debug, Clone, Copy)]
pub enum LeafPostingView<'a> {
    Sparse(&'a [u8]),
    Dense(&'a [u8]),
    Roaring(&'a [u8]),
}

impl<'a> LeafPostingView<'a> {
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Err(TreeError::CorruptedData("empty LeafPosting".into()));
        }
        match bytes[0] {
            LEAF_POSTING_SPARSE => {
                let rest = &bytes[1..];
                if !rest.len().is_multiple_of(2) {
                    return Err(TreeError::CorruptedData(
                        "LeafPosting sparse payload odd byte length".into(),
                    ));
                }
                Ok(Self::Sparse(rest))
            }
            LEAF_POSTING_DENSE => {
                if bytes.len() < 3 {
                    return Err(TreeError::CorruptedData(
                        "LeafPosting dense header truncated".into(),
                    ));
                }
                let num_words = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
                let rest = &bytes[3..];
                if rest.len() != num_words * 8 {
                    return Err(TreeError::CorruptedData(format!(
                        "LeafPosting dense expected {} bytes, got {}",
                        num_words * 8,
                        rest.len()
                    )));
                }
                Ok(Self::Dense(rest))
            }
            LEAF_POSTING_ROARING => Ok(Self::Roaring(&bytes[1..])),
            other => Err(TreeError::CorruptedData(format!(
                "unknown LeafPosting tag: {other}"
            ))),
        }
    }

    #[inline]
    pub fn or_into_words(&self, dst: &mut [u64]) -> Result<()> {
        match self {
            Self::Sparse(slice) => {
                for chunk in slice.chunks_exact(2) {
                    let id = u16::from_le_bytes([chunk[0], chunk[1]]) as usize;
                    let w = id >> 6;
                    if w < dst.len() {
                        dst[w] |= 1u64 << (id & 63);
                    }
                }
            }
            Self::Dense(slice) => {
                let words = (slice.len() / 8).min(dst.len());
                #[cfg(target_endian = "little")]
                if let Ok(src) = bytemuck::try_cast_slice::<u8, u64>(slice) {
                    for (d, &s) in dst.iter_mut().zip(src[..words].iter()) {
                        *d |= s;
                    }
                    return Ok(());
                }
                for (w_idx, chunk) in slice.chunks_exact(8).take(words).enumerate() {
                    dst[w_idx] |= u64::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            Self::Roaring(slice) => {
                let bm = decode_roaring_bitmap(slice)?;
                for id in bm.iter() {
                    let idx = id as usize;
                    let w = idx >> 6;
                    if w < dst.len() {
                        dst[w] |= 1u64 << (idx & 63);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Encode per-leaf `(min_bytes, max_bytes)` zone map for an ordered attribute.
pub fn encode_zone_map(map: &BTreeMap<NodeId, (Vec<u8>, Vec<u8>)>) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + map.len() * 24);
    buf.extend_from_slice(&(map.len() as u32).to_le_bytes());
    for (&leaf_id, (min_b, max_b)) in map {
        buf.extend_from_slice(&leaf_id.to_le_bytes());
        buf.extend_from_slice(&(min_b.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(max_b.len() as u16).to_le_bytes());
        buf.extend_from_slice(min_b);
        buf.extend_from_slice(max_b);
    }
    buf
}

/// Decode per-leaf `(min_bytes, max_bytes)` zone map for an ordered attribute.
pub fn decode_zone_map(bytes: &[u8]) -> Result<BTreeMap<NodeId, (Vec<u8>, Vec<u8>)>> {
    if bytes.len() < 4 {
        return Err(TreeError::CorruptedData("zone map payload too short".into()));
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut off = 4usize;
    let mut map = BTreeMap::new();
    for _ in 0..count {
        if off + 8 > bytes.len() {
            return Err(TreeError::CorruptedData("zone map entry truncated".into()));
        }
        let leaf_id = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let min_len = u16::from_le_bytes([bytes[off + 4], bytes[off + 5]]) as usize;
        let max_len = u16::from_le_bytes([bytes[off + 6], bytes[off + 7]]) as usize;
        off += 8;
        if off + min_len + max_len > bytes.len() {
            return Err(TreeError::CorruptedData("zone map bounds truncated".into()));
        }
        let min_b = bytes[off..off + min_len].to_vec();
        off += min_len;
        let max_b = bytes[off..off + max_len].to_vec();
        off += max_len;
        map.insert(leaf_id, (min_b, max_b));
    }
    if off != bytes.len() {
        return Err(TreeError::CorruptedData(
            "trailing bytes in zone map payload".into(),
        ));
    }
    Ok(map)
}

/// Encode forward document attributes into compact v2 binary form with an `AttrId`-sorted
/// offset directory for O(log F) zero-copy field access.
pub fn encode_attributes(schema: &CollectionSchema, attrs: &Attributes) -> Result<Vec<u8>> {
    let mut items: Vec<(AttrId, &AttributeValue)> = Vec::with_capacity(attrs.len());
    for (name, val) in attrs {
        let entry = schema
            .get_by_name(name)
            .ok_or_else(|| TreeError::InvalidAttribute {
                name: name.clone(),
                reason: "attribute is not declared in collection schema".into(),
            })?;
        items.push((entry.id, val));
    }
    items.sort_unstable_by_key(|&(id, _)| id);

    let count = items.len();
    let dir_end = DOC_HEADER_LEN + count * DOC_DIR_ENTRY_LEN;
    let mut buf = Vec::with_capacity(dir_end + count * 12);
    buf.push(DOC_FORMAT_V2);
    buf.push(0);
    buf.extend_from_slice(&(count as u16).to_le_bytes());
    buf.resize(dir_end, 0);

    for (idx, (attr_id, val)) in items.into_iter().enumerate() {
        let offset = buf.len() as u32;
        let kind = match val {
            AttributeValue::Null => DOC_VAL_NULL,
            AttributeValue::Bool(b) => {
                buf.push(if *b { 1 } else { 0 });
                DOC_VAL_BOOL
            }
            AttributeValue::Int(v) => {
                buf.extend_from_slice(&v.to_le_bytes());
                DOC_VAL_INT
            }
            AttributeValue::Uint(v) => {
                buf.extend_from_slice(&v.to_le_bytes());
                DOC_VAL_UINT
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
                DOC_VAL_FLOAT
            }
            AttributeValue::String(s) => {
                buf.extend_from_slice(s.as_bytes());
                DOC_VAL_STRING
            }
            AttributeValue::StringArray(arr) => {
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for s in arr {
                    let sb = s.as_bytes();
                    buf.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                    buf.extend_from_slice(sb);
                }
                DOC_VAL_STRING_ARRAY
            }
            AttributeValue::IntArray(arr) => {
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for &v in arr {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                DOC_VAL_INT_ARRAY
            }
            AttributeValue::UintArray(arr) => {
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for &v in arr {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                DOC_VAL_UINT_ARRAY
            }
        };
        let entry_off = DOC_HEADER_LEN + idx * DOC_DIR_ENTRY_LEN;
        buf[entry_off..entry_off + 2].copy_from_slice(&attr_id.to_le_bytes());
        buf[entry_off + 2] = kind;
        buf[entry_off + 3] = 0;
        buf[entry_off + 4..entry_off + 8].copy_from_slice(&offset.to_le_bytes());
    }
    Ok(buf)
}

/// Zero-copy view over a v2 `cf_docs` blob supporting O(log F) single-attribute reads
/// without allocating a `BTreeMap` or decoding untouched fields.
#[derive(Debug, Clone, Copy)]
pub struct DocView<'a> {
    bytes: &'a [u8],
    count: usize,
}

impl<'a> DocView<'a> {
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < DOC_HEADER_LEN {
            return Err(TreeError::CorruptedData(
                "cf_docs payload too short".into(),
            ));
        }
        if bytes[0] != DOC_FORMAT_V2 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported cf_docs format version: {}",
                bytes[0]
            )));
        }
        let count = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        let dir_end = DOC_HEADER_LEN + count * DOC_DIR_ENTRY_LEN;
        if bytes.len() < dir_end {
            return Err(TreeError::CorruptedData(
                "cf_docs directory truncated".into(),
            ));
        }
        Ok(Self { bytes, count })
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
    fn dir_entry(&self, idx: usize) -> (AttrId, u8, usize, usize) {
        let off = DOC_HEADER_LEN + idx * DOC_DIR_ENTRY_LEN;
        let attr_id = u16::from_le_bytes([self.bytes[off], self.bytes[off + 1]]);
        let kind = self.bytes[off + 2];
        let start = u32::from_le_bytes(self.bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        let end = if idx + 1 < self.count {
            let next_off = DOC_HEADER_LEN + (idx + 1) * DOC_DIR_ENTRY_LEN;
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
            let off = DOC_HEADER_LEN + mid * DOC_DIR_ENTRY_LEN;
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

    /// Decode a single attribute by `AttrId` if present (allocates only for that field).
    pub fn decode_field(&self, target_id: AttrId) -> Result<Option<AttributeValue>> {
        let Some((kind, payload)) = self.find_entry(target_id) else {
            return Ok(None);
        };
        decode_payload_value(kind, payload).map(Some)
    }

    /// Evaluate a [`Filter`] directly against this `DocView` without decoding untouched fields.
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
                    (None | Some((DOC_VAL_NULL, _)), ScalarValue::Null) => Ok(true),
                    (Some((kind, payload)), target) => raw_eq_scalar(kind, payload, target),
                    (None, _) => Ok(false),
                }
            }
            Filter::NotEq(attr, target) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(true);
                };
                match (self.find_entry(entry.id), target) {
                    (None | Some((DOC_VAL_NULL, _)), ScalarValue::Null) => Ok(false),
                    (Some((kind, payload)), target) => Ok(!raw_eq_scalar(kind, payload, target)?),
                    (None, _) => Ok(true),
                }
            }
            Filter::In(attr, targets) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(false);
                };
                match self.find_entry(entry.id) {
                    None | Some((DOC_VAL_NULL, _)) => {
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
                    None | Some((DOC_VAL_NULL, _)) => {
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
                    None | Some((DOC_VAL_NULL, _))
                ))
            }
            Filter::IsNotNull(attr) => {
                let Some(entry) = schema.get_by_name(attr) else {
                    return Ok(false);
                };
                Ok(!matches!(
                    self.find_entry(entry.id),
                    None | Some((DOC_VAL_NULL, _))
                ))
            }
            Filter::ContainsAllTokens {
                attr,
                query,
                last_as_prefix,
            } => {
                let val = match schema.get_by_name(attr) {
                    Some(e) => self.decode_field(e.id)?,
                    None => None,
                };
                Ok(eval_fts(
                    FtsPredicateKind::AllTokens,
                    attr,
                    query,
                    *last_as_prefix,
                    val.as_ref(),
                ))
            }
            Filter::ContainsAnyToken {
                attr,
                query,
                last_as_prefix,
            } => {
                let val = match schema.get_by_name(attr) {
                    Some(e) => self.decode_field(e.id)?,
                    None => None,
                };
                Ok(eval_fts(
                    FtsPredicateKind::AnyToken,
                    attr,
                    query,
                    *last_as_prefix,
                    val.as_ref(),
                ))
            }
            Filter::ContainsTokenSequence {
                attr,
                query,
                last_as_prefix,
            } => {
                let val = match schema.get_by_name(attr) {
                    Some(e) => self.decode_field(e.id)?,
                    None => None,
                };
                Ok(eval_fts(
                    FtsPredicateKind::TokenSequence,
                    attr,
                    query,
                    *last_as_prefix,
                    val.as_ref(),
                ))
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

fn raw_eq_scalar(kind: u8, payload: &[u8], target: &ScalarValue) -> Result<bool> {
    match (kind, target) {
        (DOC_VAL_NULL, ScalarValue::Null) => Ok(true),
        (DOC_VAL_BOOL, ScalarValue::Bool(b)) => {
            if payload.len() != 1 {
                return Err(TreeError::CorruptedData("cf_docs bool len".into()));
            }
            Ok((payload[0] != 0) == *b)
        }
        (DOC_VAL_INT, ScalarValue::Int(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_docs i64 len".into()));
            }
            Ok(i64::from_le_bytes(payload.try_into().unwrap()) == *b)
        }
        (DOC_VAL_UINT, ScalarValue::Uint(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_docs u64 len".into()));
            }
            Ok(u64::from_le_bytes(payload.try_into().unwrap()) == *b)
        }
        (DOC_VAL_FLOAT, ScalarValue::Float(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_docs f64 len".into()));
            }
            let a = f64::from_le_bytes(payload.try_into().unwrap());
            Ok(!a.is_nan() && !b.is_nan() && a == *b)
        }
        (DOC_VAL_STRING, ScalarValue::String(b)) => Ok(payload == b.as_bytes()),
        _ => Ok(false),
    }
}

fn raw_cmp_scalar(kind: u8, payload: &[u8], target: &ScalarValue) -> Result<Option<Ordering>> {
    match (kind, target) {
        (DOC_VAL_INT, ScalarValue::Int(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_docs i64 len".into()));
            }
            Ok(Some(i64::from_le_bytes(payload.try_into().unwrap()).cmp(b)))
        }
        (DOC_VAL_UINT, ScalarValue::Uint(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_docs u64 len".into()));
            }
            Ok(Some(u64::from_le_bytes(payload.try_into().unwrap()).cmp(b)))
        }
        (DOC_VAL_FLOAT, ScalarValue::Float(b)) => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_docs f64 len".into()));
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
        (DOC_VAL_STRING, ScalarValue::String(b)) => {
            let s = std::str::from_utf8(payload)
                .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in cf_docs: {e}")))?;
            Ok(Some(s.cmp(b.as_str())))
        }
        _ => Ok(None),
    }
}

fn raw_contains_scalar(kind: u8, payload: &[u8], target: &ScalarValue) -> Result<bool> {
    match (kind, target) {
        (DOC_VAL_STRING_ARRAY, ScalarValue::String(t)) => {
            if payload.len() < 4 {
                return Err(TreeError::CorruptedData("cf_docs str array len".into()));
            }
            let arr_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            let mut off = 4usize;
            let tb = t.as_bytes();
            for _ in 0..arr_len {
                if off + 4 > payload.len() {
                    return Err(TreeError::CorruptedData("cf_docs str elem len".into()));
                }
                let slen = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
                off += 4;
                if off + slen > payload.len() {
                    return Err(TreeError::CorruptedData("cf_docs str elem".into()));
                }
                if &payload[off..off + slen] == tb {
                    return Ok(true);
                }
                off += slen;
            }
            Ok(false)
        }
        (DOC_VAL_INT_ARRAY, ScalarValue::Int(t)) => {
            if payload.len() < 4 {
                return Err(TreeError::CorruptedData("cf_docs int array len".into()));
            }
            let arr_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            if payload.len() != 4 + arr_len * 8 {
                return Err(TreeError::CorruptedData("cf_docs int array".into()));
            }
            for chunk in payload[4..].chunks_exact(8) {
                if i64::from_le_bytes(chunk.try_into().unwrap()) == *t {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        (DOC_VAL_UINT_ARRAY, ScalarValue::Uint(t)) => {
            if payload.len() < 4 {
                return Err(TreeError::CorruptedData("cf_docs uint array len".into()));
            }
            let arr_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            if payload.len() != 4 + arr_len * 8 {
                return Err(TreeError::CorruptedData("cf_docs uint array".into()));
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
        DOC_VAL_NULL => {
            if !payload.is_empty() {
                return Err(TreeError::CorruptedData("cf_docs null non-empty".into()));
            }
            Ok(AttributeValue::Null)
        }
        DOC_VAL_BOOL => {
            if payload.len() != 1 {
                return Err(TreeError::CorruptedData("cf_docs bool truncated".into()));
            }
            let b = match payload[0] {
                0 => false,
                1 => true,
                other => {
                    return Err(TreeError::CorruptedData(format!(
                        "invalid bool byte in cf_docs: {other}"
                    )));
                }
            };
            Ok(AttributeValue::Bool(b))
        }
        DOC_VAL_INT => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_docs i64 truncated".into()));
            }
            Ok(AttributeValue::Int(i64::from_le_bytes(
                payload.try_into().unwrap(),
            )))
        }
        DOC_VAL_UINT => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_docs u64 truncated".into()));
            }
            Ok(AttributeValue::Uint(u64::from_le_bytes(
                payload.try_into().unwrap(),
            )))
        }
        DOC_VAL_FLOAT => {
            if payload.len() != 8 {
                return Err(TreeError::CorruptedData("cf_docs f64 truncated".into()));
            }
            Ok(AttributeValue::Float(f64::from_le_bytes(
                payload.try_into().unwrap(),
            )))
        }
        DOC_VAL_STRING => {
            let s = std::str::from_utf8(payload)
                .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in cf_docs: {e}")))?
                .to_owned();
            Ok(AttributeValue::String(s))
        }
        DOC_VAL_STRING_ARRAY => {
            if payload.len() < 4 {
                return Err(TreeError::CorruptedData(
                    "cf_docs string array len truncated".into(),
                ));
            }
            let len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            let mut off = 4usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                let (s, next_off) = read_len_prefixed_str(payload, off)?;
                off = next_off;
                arr.push(s);
            }
            if off != payload.len() {
                return Err(TreeError::CorruptedData(
                    "cf_docs string array trailing bytes".into(),
                ));
            }
            Ok(AttributeValue::StringArray(arr))
        }
        DOC_VAL_INT_ARRAY => {
            if payload.len() < 4 {
                return Err(TreeError::CorruptedData(
                    "cf_docs int array len truncated".into(),
                ));
            }
            let len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            if payload.len() != 4 + len * 8 {
                return Err(TreeError::CorruptedData(
                    "cf_docs int array length mismatch".into(),
                ));
            }
            let mut arr = Vec::with_capacity(len);
            for chunk in payload[4..].chunks_exact(8) {
                arr.push(i64::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(AttributeValue::IntArray(arr))
        }
        DOC_VAL_UINT_ARRAY => {
            if payload.len() < 4 {
                return Err(TreeError::CorruptedData(
                    "cf_docs uint array len truncated".into(),
                ));
            }
            let len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
            if payload.len() != 4 + len * 8 {
                return Err(TreeError::CorruptedData(
                    "cf_docs uint array length mismatch".into(),
                ));
            }
            let mut arr = Vec::with_capacity(len);
            for chunk in payload[4..].chunks_exact(8) {
                arr.push(u64::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(AttributeValue::UintArray(arr))
        }
        other => Err(TreeError::CorruptedData(format!(
            "unknown cf_docs value kind: {other}"
        ))),
    }
}

/// Decode forward document attributes into `(AttrId, AttributeValue)` pairs without string lookups.
pub fn decode_attribute_entries(bytes: &[u8]) -> Result<Vec<(AttrId, AttributeValue)>> {
    let view = DocView::parse(bytes)?;
    let mut out = Vec::with_capacity(view.count);
    for idx in 0..view.count {
        let (attr_id, kind, start, end) = view.dir_entry(idx);
        if end < start || end > bytes.len() {
            return Err(TreeError::CorruptedData(
                "cf_docs directory offset out of bounds".into(),
            ));
        }
        let val = decode_payload_value(kind, &bytes[start..end])?;
        out.push((attr_id, val));
    }
    Ok(out)
}

/// Decode forward document attributes into an [`Attributes`] map using `schema`.
pub fn decode_attributes(schema: &CollectionSchema, bytes: &[u8]) -> Result<Attributes> {
    let entries = decode_attribute_entries(bytes)?;
    let mut map = Attributes::new();
    for (attr_id, val) in entries {
        let entry = schema.get_by_id(attr_id).ok_or_else(|| {
            TreeError::CorruptedData(format!("cf_docs references unknown attr_id {attr_id}"))
        })?;
        map.insert(entry.name.clone(), val);
    }
    Ok(map)
}

fn read_len_prefixed_str(bytes: &[u8], off: usize) -> Result<(String, usize)> {
    if off + 4 > bytes.len() {
        return Err(TreeError::CorruptedData(
            "cf_docs string len truncated".into(),
        ));
    }
    let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
    let start = off + 4;
    let end = start + len;
    if end > bytes.len() {
        return Err(TreeError::CorruptedData("cf_docs string truncated".into()));
    }
    let s = std::str::from_utf8(&bytes[start..end])
        .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in cf_docs: {e}")))?
        .to_owned();
    Ok((s, end))
}

/// Extract all distinct `cf_attr_clusters` keys to index for a document's `(AttrId, AttributeValue)` list.
///
/// Only `filterable` attributes are indexed. Every non-null attribute emits both its value key(s)
/// and its `TAG_PRESENT` presence key.
pub fn extract_index_keys_from_entries(
    schema: &CollectionSchema,
    entries: &[(AttrId, AttributeValue)],
) -> Result<Vec<Vec<u8>>> {
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(entries.len() * 2);
    for (attr_id, val) in entries {
        let Some(schema_entry) = schema.get_by_id(*attr_id) else {
            return Err(TreeError::CorruptedData(format!(
                "unknown attr_id {attr_id} in document"
            )));
        };
        if !schema_entry.filterable {
            continue;
        }
        append_index_keys_for_value(*attr_id, val, &mut keys)?;
    }
    keys.sort_unstable();
    keys.dedup();
    Ok(keys)
}

/// Extract all distinct `cf_attr_clusters` keys to index for an [`Attributes`] map.
pub fn extract_index_keys(schema: &CollectionSchema, attrs: &Attributes) -> Result<Vec<Vec<u8>>> {
    let mut keys: Vec<Vec<u8>> = Vec::with_capacity(attrs.len() * 2);
    for (name, val) in attrs {
        let Some(schema_entry) = schema.get_by_name(name) else {
            return Err(TreeError::InvalidAttribute {
                name: name.clone(),
                reason: "attribute is not declared in collection schema".into(),
            });
        };
        if !schema_entry.filterable {
            continue;
        }
        append_index_keys_for_value(schema_entry.id, val, &mut keys)?;
    }
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
    keys.push(encode_present_cluster_key(attr_id));
    match val {
        AttributeValue::Null => unreachable!(),
        AttributeValue::Bool(b) => {
            keys.push(encode_scalar_cluster_key(attr_id, &ScalarValue::Bool(*b))?);
        }
        AttributeValue::Int(v) => {
            keys.push(encode_scalar_cluster_key(attr_id, &ScalarValue::Int(*v))?);
        }
        AttributeValue::Uint(v) => {
            keys.push(encode_scalar_cluster_key(attr_id, &ScalarValue::Uint(*v))?);
        }
        AttributeValue::Float(v) => {
            keys.push(encode_scalar_cluster_key(attr_id, &ScalarValue::Float(*v))?);
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

/// Serialize a `RoaringBitmap` into bytes.
pub fn encode_roaring_bitmap(bm: &RoaringBitmap) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(bm.serialized_size());
    bm.serialize_into(&mut buf)
        .map_err(|e| TreeError::Storage(format!("roaring serialize error: {e}")))?;
    Ok(buf)
}

/// Deserialize a `RoaringBitmap` from bytes.
pub fn decode_roaring_bitmap(bytes: &[u8]) -> Result<RoaringBitmap> {
    RoaringBitmap::deserialize_from(bytes)
        .map_err(|e| TreeError::CorruptedData(format!("roaring deserialize error: {e}")))
}

/// Build a `cf_fts_terms` key: `attr_id (u16 BE) || token_utf8`.
#[inline]
pub fn encode_fts_term_key(attr_id: AttrId, token: &str) -> Vec<u8> {
    let tb = token.as_bytes();
    let mut key = Vec::with_capacity(2 + tb.len());
    key.extend_from_slice(&attr_id.to_be_bytes());
    key.extend_from_slice(tb);
    key
}

/// Decode a `cf_fts_terms` key into `(attr_id, token_utf8)`.
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

/// Build the 4-byte `term_id (u32 BE)` prefix used by `cf_fts_postings`, `cf_fts_positions`, and `cf_fts_skips`.
#[inline]
pub fn encode_fts_term_id_prefix(term_id: TermId) -> [u8; 4] {
    term_id.to_be_bytes()
}

/// Build a 4-byte `block_id` key for `cf_fts_rev_doc_map`.
#[inline]
pub fn encode_fts_block_key(block_id: u32) -> [u8; 4] {
    block_id.to_be_bytes()
}

/// Build an 8-byte key for `cf_fts_postings`, `cf_fts_positions`, or `cf_fts_skips`:
/// `term_id (u32 BE) || block_or_chunk_id (u32 BE)`.
#[inline]
pub fn encode_fts_term_id_key(term_id: TermId, block_or_chunk_id: u32) -> [u8; 8] {
    let t = term_id.to_be_bytes();
    let b = block_or_chunk_id.to_be_bytes();
    [t[0], t[1], t[2], t[3], b[0], b[1], b[2], b[3]]
}

/// Decode an 8-byte `term_id || block_or_chunk_id` key.
#[inline]
pub fn decode_fts_term_id_key(key: &[u8]) -> Result<(TermId, u32)> {
    if key.len() != 8 {
        return Err(TreeError::CorruptedData(format!(
            "FTS term_id key expected 8 bytes, got {}",
            key.len()
        )));
    }
    Ok((
        u32::from_be_bytes(key[0..4].try_into().unwrap()),
        u32::from_be_bytes(key[4..8].try_into().unwrap()),
    ))
}

/// Build a `cf_fts_doc_lens` key: `attr_id (u16 BE) || block_id (u32 BE)`.
#[inline]
pub fn encode_fts_doc_len_key(attr_id: AttrId, block_id: FtsBlockId) -> [u8; 6] {
    let a = attr_id.to_be_bytes();
    let b = block_id.to_be_bytes();
    [a[0], a[1], b[0], b[1], b[2], b[3]]
}

/// Decode a `cf_fts_doc_lens` key into `(attr_id, block_id)`.
#[inline]
pub fn decode_fts_doc_len_key(key: &[u8]) -> Result<(AttrId, FtsBlockId)> {
    if key.len() != 6 {
        return Err(TreeError::CorruptedData(format!(
            "cf_fts_doc_lens key expected 6 bytes, got {}",
            key.len()
        )));
    }
    let attr_id = u16::from_be_bytes([key[0], key[1]]);
    let block_id = u32::from_be_bytes([key[2], key[3], key[4], key[5]]);
    Ok((attr_id, block_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use izzydb_core::config::{DistanceMetric, TreeConfig};
    use izzydb_core::schema::CollectionConfig;

    #[test]
    fn test_order_preserving_encodings() {
        let ints = [i64::MIN, -1000, -1, 0, 1, 1000, i64::MAX];
        let enc_ints: Vec<Vec<u8>> = ints
            .iter()
            .map(|&v| encode_scalar_cluster_key(1, &ScalarValue::Int(v)).unwrap())
            .collect();
        for w in enc_ints.windows(2) {
            assert!(w[0] < w[1]);
        }

        let floats = [
            f64::NEG_INFINITY,
            -1e10,
            -1.5,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1.5,
            1e10,
            f64::INFINITY,
        ];
        let enc_floats: Vec<Vec<u8>> = floats
            .iter()
            .map(|&v| encode_scalar_cluster_key(1, &ScalarValue::Float(v)).unwrap())
            .collect();
        assert_eq!(enc_floats[4], enc_floats[5]);
        for i in 0..enc_floats.len() - 1 {
            if i == 4 {
                continue;
            }
            assert!(enc_floats[i] < enc_floats[i + 1], "failed at {i}");
        }

        let strs = ["", "\0", "\0\0", "\0a", "a", "ab", "ab\0", "ab\0c", "abc"];
        let enc_strs: Vec<Vec<u8>> = strs
            .iter()
            .map(|&s| encode_scalar_cluster_key(1, &ScalarValue::String(s.into())).unwrap())
            .collect();
        for w in enc_strs.windows(2) {
            assert!(w[0] < w[1], "{:?} not < {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn test_attributes_roundtrip_docview_and_index_keys() {
        let cfg = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared))
            .with_attribute("tag", AttributeType::String, true)
            .with_attribute("score", AttributeType::Float, true)
            .with_attribute("tags", AttributeType::StringArray, true)
            .with_attribute("raw_note", AttributeType::String, false);
        let schema = CollectionSchema::from_config(&cfg).unwrap();

        let mut attrs = Attributes::new();
        attrs.insert("tag".into(), "alpha".into());
        attrs.insert("score".into(), 3.5f64.into());
        attrs.insert(
            "tags".into(),
            vec!["a".to_string(), "b".to_string(), "a".to_string()].into(),
        );
        attrs.insert("raw_note".into(), "ignored_by_index".into());

        let bytes = encode_attributes(&schema, &attrs).unwrap();
        let decoded = decode_attributes(&schema, &bytes).unwrap();
        assert_eq!(attrs, decoded);

        let view = DocView::parse(&bytes).unwrap();
        assert!(
            view.matches_filter(
                &schema,
                &Filter::And(vec![
                    Filter::Eq("tag".into(), "alpha".into()),
                    Filter::Gt("score".into(), 3.0f64.into()),
                    Filter::Contains("tags".into(), "b".into()),
                ]),
                &mut |_, _, _, _, _| false,
            )
            .unwrap()
        );
        assert!(
            !view
                .matches_filter(
                    &schema,
                    &Filter::Eq("tag".into(), "beta".into()),
                    &mut |_, _, _, _, _| false,
                )
                .unwrap()
        );

        let keys1 = extract_index_keys(&schema, &attrs).unwrap();
        let entries = decode_attribute_entries(&bytes).unwrap();
        let keys2 = extract_index_keys_from_entries(&schema, &entries).unwrap();
        assert_eq!(keys1, keys2);
        assert_eq!(keys1.len(), 7);
    }

    #[test]
    fn test_leaf_posting_roundtrip() {
        let mut p1 = LeafPosting::default();
        p1.insert(7);
        let b1 = p1.encode().unwrap();
        assert_eq!(b1.len(), 3);
        assert_eq!(LeafPosting::decode(&b1).unwrap(), p1);

        let mut p2 = LeafPosting::default();
        for id in (0..80).step_by(2) {
            p2.insert(id);
        }
        let b2 = p2.encode().unwrap();
        assert_eq!(b2[0], LEAF_POSTING_DENSE);
        assert_eq!(LeafPosting::decode(&b2).unwrap(), p2);
        let mut words = [0u64; 2];
        LeafPostingView::parse(&b2).unwrap().or_into_words(&mut words).unwrap();
        assert_eq!(words[0].count_ones() + words[1].count_ones(), 40);
    }
}
