use izzydb_core::config::{AttrId, NodeId};
use izzydb_core::error::{Result, TreeError};
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

const DOC_VAL_NULL: u8 = 0;
const DOC_VAL_BOOL: u8 = 1;
const DOC_VAL_INT: u8 = 2;
const DOC_VAL_UINT: u8 = 3;
const DOC_VAL_FLOAT: u8 = 4;
const DOC_VAL_STRING: u8 = 5;
const DOC_VAL_STRING_ARRAY: u8 = 6;
const DOC_VAL_INT_ARRAY: u8 = 7;
const DOC_VAL_UINT_ARRAY: u8 = 8;

/// 3-byte RocksDB prefix (`attr_id BE || type_tag`) for `cf_attr_clusters` scans.
#[inline]
pub fn encode_attr_prefix(attr_id: AttrId, type_tag: u8) -> [u8; 3] {
    let id = attr_id.to_be_bytes();
    [id[0], id[1], type_tag]
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

/// Build a `cf_attr_postings` key from a `cf_attr_clusters` key and `cluster_id`.
#[inline]
pub fn make_posting_key(cluster_key: &[u8], cluster_id: NodeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(cluster_key.len() + 4);
    key.extend_from_slice(cluster_key);
    key.extend_from_slice(&cluster_id.to_be_bytes());
    key
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

/// Encode forward document attributes into compact binary form keyed by `AttrId`.
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

    let mut buf = Vec::with_capacity(2 + items.len() * 12);
    buf.extend_from_slice(&(items.len() as u16).to_le_bytes());
    for (attr_id, val) in items {
        buf.extend_from_slice(&attr_id.to_le_bytes());
        match val {
            AttributeValue::Null => buf.push(DOC_VAL_NULL),
            AttributeValue::Bool(b) => {
                buf.push(DOC_VAL_BOOL);
                buf.push(if *b { 1 } else { 0 });
            }
            AttributeValue::Int(v) => {
                buf.push(DOC_VAL_INT);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            AttributeValue::Uint(v) => {
                buf.push(DOC_VAL_UINT);
                buf.extend_from_slice(&v.to_le_bytes());
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
                buf.push(DOC_VAL_FLOAT);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            AttributeValue::String(s) => {
                buf.push(DOC_VAL_STRING);
                let sb = s.as_bytes();
                buf.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                buf.extend_from_slice(sb);
            }
            AttributeValue::StringArray(arr) => {
                buf.push(DOC_VAL_STRING_ARRAY);
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for s in arr {
                    let sb = s.as_bytes();
                    buf.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                    buf.extend_from_slice(sb);
                }
            }
            AttributeValue::IntArray(arr) => {
                buf.push(DOC_VAL_INT_ARRAY);
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for &v in arr {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            AttributeValue::UintArray(arr) => {
                buf.push(DOC_VAL_UINT_ARRAY);
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for &v in arr {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    Ok(buf)
}

/// Decode forward document attributes into `(AttrId, AttributeValue)` pairs without string lookups.
pub fn decode_attribute_entries(bytes: &[u8]) -> Result<Vec<(AttrId, AttributeValue)>> {
    if bytes.len() < 2 {
        return Err(TreeError::CorruptedData(
            "cf_docs payload too short".into(),
        ));
    }
    let count = u16::from_le_bytes(bytes[0..2].try_into().unwrap()) as usize;
    let mut off = 2usize;
    let mut out = Vec::with_capacity(count);

    for _ in 0..count {
        if off + 3 > bytes.len() {
            return Err(TreeError::CorruptedData(
                "cf_docs attribute header truncated".into(),
            ));
        }
        let attr_id = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
        let kind = bytes[off + 2];
        off += 3;

        let val = match kind {
            DOC_VAL_NULL => AttributeValue::Null,
            DOC_VAL_BOOL => {
                if off + 1 > bytes.len() {
                    return Err(TreeError::CorruptedData("cf_docs bool truncated".into()));
                }
                let b = match bytes[off] {
                    0 => false,
                    1 => true,
                    other => {
                        return Err(TreeError::CorruptedData(format!(
                            "invalid bool byte in cf_docs: {other}"
                        )));
                    }
                };
                off += 1;
                AttributeValue::Bool(b)
            }
            DOC_VAL_INT => {
                if off + 8 > bytes.len() {
                    return Err(TreeError::CorruptedData("cf_docs i64 truncated".into()));
                }
                let v = i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                off += 8;
                AttributeValue::Int(v)
            }
            DOC_VAL_UINT => {
                if off + 8 > bytes.len() {
                    return Err(TreeError::CorruptedData("cf_docs u64 truncated".into()));
                }
                let v = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                off += 8;
                AttributeValue::Uint(v)
            }
            DOC_VAL_FLOAT => {
                if off + 8 > bytes.len() {
                    return Err(TreeError::CorruptedData("cf_docs f64 truncated".into()));
                }
                let v = f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
                off += 8;
                AttributeValue::Float(v)
            }
            DOC_VAL_STRING => {
                let (s, next_off) = read_len_prefixed_str(bytes, off)?;
                off = next_off;
                AttributeValue::String(s)
            }
            DOC_VAL_STRING_ARRAY => {
                if off + 4 > bytes.len() {
                    return Err(TreeError::CorruptedData(
                        "cf_docs string array len truncated".into(),
                    ));
                }
                let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
                off += 4;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    let (s, next_off) = read_len_prefixed_str(bytes, off)?;
                    off = next_off;
                    arr.push(s);
                }
                AttributeValue::StringArray(arr)
            }
            DOC_VAL_INT_ARRAY => {
                if off + 4 > bytes.len() {
                    return Err(TreeError::CorruptedData(
                        "cf_docs int array len truncated".into(),
                    ));
                }
                let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
                off += 4;
                if off + len * 8 > bytes.len() {
                    return Err(TreeError::CorruptedData(
                        "cf_docs int array truncated".into(),
                    ));
                }
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(i64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
                    off += 8;
                }
                AttributeValue::IntArray(arr)
            }
            DOC_VAL_UINT_ARRAY => {
                if off + 4 > bytes.len() {
                    return Err(TreeError::CorruptedData(
                        "cf_docs uint array len truncated".into(),
                    ));
                }
                let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
                off += 4;
                if off + len * 8 > bytes.len() {
                    return Err(TreeError::CorruptedData(
                        "cf_docs uint array truncated".into(),
                    ));
                }
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
                    off += 8;
                }
                AttributeValue::UintArray(arr)
            }
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "unknown cf_docs value kind: {other}"
                )));
            }
        };
        out.push((attr_id, val));
    }

    if off != bytes.len() {
        return Err(TreeError::CorruptedData(
            "trailing bytes in cf_docs payload".into(),
        ));
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
        return Err(TreeError::CorruptedData("cf_docs string len truncated".into()));
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
        // -0.0 and +0.0 encode identically
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
    fn test_attributes_roundtrip_and_index_keys() {
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

        let keys1 = extract_index_keys(&schema, &attrs).unwrap();
        let entries = decode_attribute_entries(&bytes).unwrap();
        let keys2 = extract_index_keys_from_entries(&schema, &entries).unwrap();
        assert_eq!(keys1, keys2);
        // 3 filterable attrs => 3 present keys + 1 tag + 1 score + 2 unique tags = 7 keys; raw_note excluded.
        assert_eq!(keys1.len(), 7);
    }
}
