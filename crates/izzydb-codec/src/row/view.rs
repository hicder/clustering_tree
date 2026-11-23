use std::cmp::Ordering;

use izzydb_core::config::AttrId;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::{Filter, FtsPredicateKind, TokenQuery};
use izzydb_core::schema::{
    AttributeValue, CollectionSchema, ScalarValue,
};
use super::super::*;
use super::*;

/// Zero-copy view over a v3 `cf_rows` blob supporting O(log F) single-attribute reads.
#[derive(Debug, Clone, Copy)]
pub struct RowView<'a> {
    bytes: &'a [u8],
    flags: u8,
    pub(super) count: usize,
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
