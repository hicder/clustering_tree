
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{
    AttributeType, ScalarValue,
};
use super::*;

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
