use std::cmp::Ordering;

use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::ScalarValue;
use super::super::*;
use super::*;

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
    pub(super) fn from_raw(kind: u8, payload: &'a [u8]) -> Result<Self> {
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
                for &c in body.as_chunks::<8>().0 {
                    f(RawScalar::Int(i64::from_le_bytes(c)));
                }
            }
            Self::UintArray(payload) => {
                let mut r = ByteReader::new(payload);
                let len = r.read_u32_le("cf_rows uint array len")? as usize;
                let body = r.read_bytes(len * 8, "cf_rows uint array")?;
                r.expect_eof("cf_rows uint array")?;
                for &c in body.as_chunks::<8>().0 {
                    f(RawScalar::Uint(u64::from_le_bytes(c)));
                }
            }
        }
        Ok(())
    }
}

pub(super) fn raw_eq_scalar(kind: u8, payload: &[u8], target: &ScalarValue) -> Result<bool> {
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

pub(super) fn raw_cmp_scalar(kind: u8, payload: &[u8], target: &ScalarValue) -> Result<Option<Ordering>> {
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

pub(super) fn raw_contains_scalar(kind: u8, payload: &[u8], target: &ScalarValue) -> Result<bool> {
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
            for &chunk in payload[4..].as_chunks::<8>().0 {
                if i64::from_le_bytes(chunk) == *t {
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
            for &chunk in payload[4..].as_chunks::<8>().0 {
                if u64::from_le_bytes(chunk) == *t {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

