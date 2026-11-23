//! Staged document and detached-value encoding.

use std::fs::File;
use std::io::BufWriter;


use izzydb::config::AttrId;
use izzydb::error::{Result, TreeError};
use izzydb::schema::{
    AttributeValue, Attributes, CollectionSchema, ColumnStorage,
};
use izzydb::codec::encode_row;



pub(super) struct DocShardWriter {
    pub(super) shard_id: u32,
    pub(super) offset: u64,
    pub(super) writer: BufWriter<File>,
}

/// Encode staged document bytes:
/// - When `schema_has_detached == false`: exact `cf_rows` payload (`encode_row`).
/// - When `schema_has_detached == true`: `[inline_len: u32 LE][inline_row_bytes][detached_entries]`.
pub(super) fn encode_staged_doc(
    schema: &CollectionSchema,
    attrs: &Attributes,
    has_vector: bool,
    schema_has_detached: bool,
) -> Result<Vec<u8>> {
    let inline_bytes = encode_row(schema, attrs, has_vector)?;
    if !schema_has_detached {
        return Ok(inline_bytes);
    }

    let mut det_items: Vec<(AttrId, &AttributeValue)> = Vec::new();
    for (name, val) in attrs {
        if let Some(entry) = schema.get_by_name(name)
            && entry.storage == ColumnStorage::Detached
            && !matches!(val, AttributeValue::Null)
        {
            det_items.push((entry.id, val));
        }
    }
    det_items.sort_unstable_by_key(|&(id, _)| id);

    let mut out = Vec::with_capacity(4 + inline_bytes.len() + 2 + det_items.len() * 12);
    out.extend_from_slice(&(inline_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&inline_bytes);
    out.extend_from_slice(&(det_items.len() as u16).to_le_bytes());
    for (attr_id, val) in det_items {
        out.extend_from_slice(&attr_id.to_le_bytes());
        encode_detached_val(&mut out, val);
    }
    Ok(out)
}

pub type DecodedStagedDoc<'a> = (&'a [u8], Vec<(AttrId, AttributeValue)>);

/// Split staged document bytes into `(cf_rows_bytes, optional_detached_pairs)`.
#[inline]
pub fn decode_staged_doc(
    bytes: &[u8],
    schema_has_detached: bool,
) -> Result<DecodedStagedDoc<'_>> {
    if !schema_has_detached {
        return Ok((bytes, Vec::new()));
    }
    if bytes.len() < 6 {
        return Err(TreeError::CorruptedData("staged doc header truncated".into()));
    }
    let inline_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let inline_end = 4 + inline_len;
    if bytes.len() < inline_end + 2 {
        return Err(TreeError::CorruptedData("staged doc inline payload truncated".into()));
    }
    let inline_bytes = &bytes[4..inline_end];
    let det_count = u16::from_le_bytes(bytes[inline_end..inline_end + 2].try_into().unwrap()) as usize;
    if det_count == 0 {
        return Ok((inline_bytes, Vec::new()));
    }
    let mut off = inline_end + 2;
    let mut det = Vec::with_capacity(det_count);
    for _ in 0..det_count {
        if off + 3 > bytes.len() {
            return Err(TreeError::CorruptedData("staged detached attr truncated".into()));
        }
        let attr_id = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
        off += 2;
        let val = decode_detached_val(bytes, &mut off)?;
        det.push((attr_id, val));
    }
    Ok((inline_bytes, det))
}

pub(super) fn encode_detached_val(out: &mut Vec<u8>, val: &AttributeValue) {
    match val {
        AttributeValue::Null => out.push(0),
        AttributeValue::Bool(b) => {
            out.push(1);
            out.push(if *b { 1 } else { 0 });
        }
        AttributeValue::Int(v) => {
            out.push(2);
            out.extend_from_slice(&v.to_le_bytes());
        }
        AttributeValue::Uint(v) => {
            out.push(3);
            out.extend_from_slice(&v.to_le_bytes());
        }
        AttributeValue::Float(v) => {
            out.push(4);
            out.extend_from_slice(&v.to_le_bytes());
        }
        AttributeValue::String(s) => {
            out.push(5);
            let sb = s.as_bytes();
            out.extend_from_slice(&(sb.len() as u32).to_le_bytes());
            out.extend_from_slice(sb);
        }
        AttributeValue::StringArray(arr) => {
            out.push(6);
            out.extend_from_slice(&(arr.len() as u32).to_le_bytes());
            for s in arr {
                let sb = s.as_bytes();
                out.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                out.extend_from_slice(sb);
            }
        }
        AttributeValue::IntArray(arr) => {
            out.push(7);
            out.extend_from_slice(&(arr.len() as u32).to_le_bytes());
            for &v in arr {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        AttributeValue::UintArray(arr) => {
            out.push(8);
            out.extend_from_slice(&(arr.len() as u32).to_le_bytes());
            for &v in arr {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

pub(super) fn decode_detached_val(bytes: &[u8], off: &mut usize) -> Result<AttributeValue> {
    let tag = *bytes
        .get(*off)
        .ok_or_else(|| TreeError::CorruptedData("detached val tag EOF".into()))?;
    *off += 1;
    let take = |n: usize, off: &mut usize| -> Result<&[u8]> {
        let end = *off + n;
        if end > bytes.len() {
            return Err(TreeError::CorruptedData("detached val payload truncated".into()));
        }
        let s = &bytes[*off..end];
        *off = end;
        Ok(s)
    };
    match tag {
        0 => Ok(AttributeValue::Null),
        1 => Ok(AttributeValue::Bool(take(1, off)?[0] != 0)),
        2 => Ok(AttributeValue::Int(i64::from_le_bytes(
            take(8, off)?.try_into().unwrap(),
        ))),
        3 => Ok(AttributeValue::Uint(u64::from_le_bytes(
            take(8, off)?.try_into().unwrap(),
        ))),
        4 => Ok(AttributeValue::Float(f64::from_le_bytes(
            take(8, off)?.try_into().unwrap(),
        ))),
        5 => {
            let len = u32::from_le_bytes(take(4, off)?.try_into().unwrap()) as usize;
            let sb = take(len, off)?;
            let s = std::str::from_utf8(sb)
                .map_err(|e| TreeError::CorruptedData(format!("utf8 error: {e}")))?
                .to_owned();
            Ok(AttributeValue::String(s))
        }
        6 => {
            let len = u32::from_le_bytes(take(4, off)?.try_into().unwrap()) as usize;
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                let slen = u32::from_le_bytes(take(4, off)?.try_into().unwrap()) as usize;
                let sb = take(slen, off)?;
                let s = std::str::from_utf8(sb)
                    .map_err(|e| TreeError::CorruptedData(format!("utf8 error: {e}")))?
                    .to_owned();
                out.push(s);
            }
            Ok(AttributeValue::StringArray(out))
        }
        7 => {
            let len = u32::from_le_bytes(take(4, off)?.try_into().unwrap()) as usize;
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                out.push(i64::from_le_bytes(take(8, off)?.try_into().unwrap()));
            }
            Ok(AttributeValue::IntArray(out))
        }
        8 => {
            let len = u32::from_le_bytes(take(4, off)?.try_into().unwrap()) as usize;
            let mut out = Vec::with_capacity(len);
            for _ in 0..len {
                out.push(u64::from_le_bytes(take(8, off)?.try_into().unwrap()));
            }
            Ok(AttributeValue::UintArray(out))
        }
        other => Err(TreeError::CorruptedData(format!(
            "unknown detached staged tag {other}"
        ))),
    }
}
