
use izzydb_core::config::AttrId;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{
    AttributeValue, Attributes, CollectionSchema, ColumnStorage, ScalarValue,
};
use super::super::*;
use super::*;

pub(super) fn decode_payload_value(kind: u8, payload: &[u8]) -> Result<AttributeValue> {
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
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&c| i64::from_le_bytes(c))
                .collect();
            Ok(AttributeValue::IntArray(arr))
        }
        ROW_VAL_UINT_ARRAY => {
            let mut r = ByteReader::new(payload);
            let len = r.read_u32_le("cf_rows uint array len")? as usize;
            let body = r.read_bytes(len * 8, "cf_rows uint array")?;
            r.expect_eof("cf_rows uint array")?;
            let arr = body
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&c| u64::from_le_bytes(c))
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

pub(super) fn append_index_keys_for_value(
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


