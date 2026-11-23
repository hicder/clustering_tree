
use izzydb_core::config::AttrId;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{
    AttributeValue, Attributes, CollectionSchema, ColumnStorage,
};
use super::super::*;


// ---------------------------------------------------------------------------
// RowView v3 (`cf_rows`)
// ---------------------------------------------------------------------------

pub(super) struct ByteReader<'a> {
    bytes: &'a [u8],
    off: usize,
}

impl<'a> ByteReader<'a> {
    #[inline]
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, off: 0 }
    }

    #[inline]
    pub(super) fn read_bytes(&mut self, len: usize, label: &str) -> Result<&'a [u8]> {
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
    pub(super) fn read_u32_le(&mut self, label: &str) -> Result<u32> {
        let s = self.read_bytes(4, label)?;
        Ok(u32::from_le_bytes(s.try_into().unwrap()))
    }

    #[inline]
    pub(super) fn read_len_prefixed_str(&mut self, label: &str) -> Result<String> {
        let len = self.read_u32_le(label)? as usize;
        let s = self.read_bytes(len, label)?;
        std::str::from_utf8(s)
            .map(str::to_owned)
            .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in {label}: {e}")))
    }

    #[inline]
    pub(super) fn expect_eof(&self, label: &str) -> Result<()> {
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
