//! Zero-copy column block view.

use izzydb_core::config::ROW_BLOCK_SIZE;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{AttributeType, AttributeValue, ScalarValue};
use crate::codec::{encode_f64_ordered, encode_i64_ordered, encode_str_ordered};

use super::*;

/// Zero-copy borrowed view over an encoded 256-slot `ColumnBlock`.
#[derive(Debug, Clone, Copy)]
pub struct ColumnBlockView<'a> {
    pub attr_type: AttributeType,
    pub width: u8,
    pub flags: u8,
    pub present_count: u16,
    pub(super) present: [u64; 4],
    pub(super) payload: &'a [u8],
}

impl<'a> ColumnBlockView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < COL_HEADER_LEN {
            return Err(TreeError::CorruptedData("ColumnBlock header truncated".into()));
        }
        if bytes[0] != COLUMN_BLOCK_VERSION {
            return Err(TreeError::CorruptedData(format!(
                "unsupported ColumnBlock version: {}",
                bytes[0]
            )));
        }
        let attr_type = AttributeType::from_u8(bytes[1])?;
        let width = bytes[2];
        let flags = bytes[3];
        let present_count = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let reserved = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        if reserved != 0 {
            return Err(TreeError::CorruptedData(
                "invalid ColumnBlock reserved bytes".into(),
            ));
        }

        let all_present = (flags & COL_FLAG_ALL_PRESENT) != 0;
        let mut off = COL_HEADER_LEN;
        let present = if all_present {
            [u64::MAX; 4]
        } else {
            if off + 32 > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "ColumnBlock present bitmap truncated".into(),
                ));
            }
            let mut bm = [0u64; 4];
            for (i, w) in bm.iter_mut().enumerate() {
                let s = off + i * 8;
                *w = u64::from_le_bytes(bytes[s..s + 8].try_into().unwrap());
            }
            off += 32;
            bm
        };

        let payload = &bytes[off..];
        Self::validate_payload(attr_type, width, flags, present_count, payload)?;
        Ok(Self {
            attr_type,
            width,
            flags,
            present_count,
            present,
            payload,
        })
    }

    pub(super) fn validate_payload(
        attr_type: AttributeType,
        width: u8,
        flags: u8,
        present_count: u16,
        payload: &[u8],
    ) -> Result<()> {
        let is_for = (flags & COL_FLAG_FOR) != 0;
        let expect_len = match attr_type {
            AttributeType::Bool => 32,
            AttributeType::Int | AttributeType::Uint => {
                if present_count == 0 && !is_for && width == 0 {
                    0
                } else if is_for {
                    match width {
                        0 => 8,
                        1 | 2 | 4 => 8 + ROW_BLOCK_SIZE * width as usize,
                        _ => {
                            return Err(TreeError::CorruptedData(format!(
                                "invalid ColumnBlock FOR width: {width}"
                            )));
                        }
                    }
                } else if width == 8 {
                    ROW_BLOCK_SIZE * 8
                } else {
                    return Err(TreeError::CorruptedData(format!(
                        "invalid ColumnBlock integer width: {width}"
                    )));
                }
            }
            AttributeType::Float => ROW_BLOCK_SIZE * 8,
            AttributeType::String => {
                let hdr = (ROW_BLOCK_SIZE + 1) * 4;
                if payload.len() < hdr {
                    return Err(TreeError::CorruptedData(
                        "ColumnBlock String offsets truncated".into(),
                    ));
                }
                let data_len =
                    read_u32_at(payload, ROW_BLOCK_SIZE * 4) as usize;
                if payload.len() != hdr + data_len {
                    return Err(TreeError::CorruptedData(
                        "ColumnBlock String payload length mismatch".into(),
                    ));
                }
                return Ok(());
            }
            AttributeType::IntArray | AttributeType::UintArray => {
                let hdr = (ROW_BLOCK_SIZE + 1) * 4;
                if payload.len() < hdr {
                    return Err(TreeError::CorruptedData(
                        "ColumnBlock array offsets truncated".into(),
                    ));
                }
                let elems = read_u32_at(payload, ROW_BLOCK_SIZE * 4) as usize;
                if payload.len() != hdr + elems * 8 {
                    return Err(TreeError::CorruptedData(
                        "ColumnBlock numeric array payload length mismatch".into(),
                    ));
                }
                return Ok(());
            }
            AttributeType::StringArray => {
                let hdr = (ROW_BLOCK_SIZE + 1) * 4;
                if payload.len() < hdr {
                    return Err(TreeError::CorruptedData(
                        "ColumnBlock StringArray offsets truncated".into(),
                    ));
                }
                let total_elems = read_u32_at(payload, ROW_BLOCK_SIZE * 4) as usize;
                let str_offs_bytes = (total_elems + 1) * 4;
                if payload.len() < hdr + str_offs_bytes {
                    return Err(TreeError::CorruptedData(
                        "ColumnBlock StringArray string offsets truncated".into(),
                    ));
                }
                let data_len =
                    read_u32_at(payload, hdr + total_elems * 4) as usize;
                if payload.len() != hdr + str_offs_bytes + data_len {
                    return Err(TreeError::CorruptedData(
                        "ColumnBlock StringArray data length mismatch".into(),
                    ));
                }
                return Ok(());
            }
        };
        if payload.len() != expect_len {
            return Err(TreeError::CorruptedData(format!(
                "ColumnBlock {:?} payload length {} != expected {}",
                attr_type,
                payload.len(),
                expect_len
            )));
        }
        Ok(())
    }

    #[inline]
    pub fn present_mask(&self) -> [u64; 4] {
        self.present
    }

    #[inline]
    pub fn is_present(&self, slot: u8) -> bool {
        let s = slot as usize;
        ((self.present[s >> 6] >> (s & 63)) & 1) != 0
    }

    #[inline]
    pub fn get_bool(&self, slot: u8) -> Option<bool> {
        if !self.is_present(slot) || self.attr_type != AttributeType::Bool {
            return None;
        }
        let s = slot as usize;
        let off = (s >> 6) * 8;
        let w = u64::from_le_bytes(self.payload[off..off + 8].try_into().unwrap());
        Some(((w >> (s & 63)) & 1) != 0)
    }

    #[inline]
    pub fn get_i64(&self, slot: u8) -> Option<i64> {
        if !self.is_present(slot) || self.attr_type != AttributeType::Int {
            return None;
        }
        let s = slot as usize;
        let is_for = (self.flags & COL_FLAG_FOR) != 0;
        if is_for {
            let base = i64::from_le_bytes(self.payload[0..8].try_into().unwrap());
            let delta = match self.width {
                0 => 0i64,
                1 => self.payload[8 + s] as i64,
                2 => u16::from_le_bytes(self.payload[8 + s * 2..10 + s * 2].try_into().unwrap())
                    as i64,
                4 => u32::from_le_bytes(self.payload[8 + s * 4..12 + s * 4].try_into().unwrap())
                    as i64,
                _ => unreachable!(),
            };
            Some(base.wrapping_add(delta))
        } else {
            let off = s * 8;
            Some(i64::from_le_bytes(
                self.payload[off..off + 8].try_into().unwrap(),
            ))
        }
    }

    #[inline]
    pub fn get_u64(&self, slot: u8) -> Option<u64> {
        if !self.is_present(slot) || self.attr_type != AttributeType::Uint {
            return None;
        }
        let s = slot as usize;
        let is_for = (self.flags & COL_FLAG_FOR) != 0;
        if is_for {
            let base = u64::from_le_bytes(self.payload[0..8].try_into().unwrap());
            let delta = match self.width {
                0 => 0u64,
                1 => self.payload[8 + s] as u64,
                2 => u16::from_le_bytes(self.payload[8 + s * 2..10 + s * 2].try_into().unwrap())
                    as u64,
                4 => u32::from_le_bytes(self.payload[8 + s * 4..12 + s * 4].try_into().unwrap())
                    as u64,
                _ => unreachable!(),
            };
            Some(base.wrapping_add(delta))
        } else {
            let off = s * 8;
            Some(u64::from_le_bytes(
                self.payload[off..off + 8].try_into().unwrap(),
            ))
        }
    }

    #[inline]
    pub fn get_f64(&self, slot: u8) -> Option<f64> {
        if !self.is_present(slot) || self.attr_type != AttributeType::Float {
            return None;
        }
        let off = (slot as usize) * 8;
        let bits = u64::from_le_bytes(self.payload[off..off + 8].try_into().unwrap());
        Some(f64::from_bits(bits))
    }

    #[inline]
    pub fn get_str(&self, slot: u8) -> Option<&'a [u8]> {
        if !self.is_present(slot) || self.attr_type != AttributeType::String {
            return None;
        }
        let s = slot as usize;
        let start = read_u32_at(self.payload, s * 4) as usize;
        let end = read_u32_at(self.payload, (s + 1) * 4) as usize;
        let data = &self.payload[(ROW_BLOCK_SIZE + 1) * 4..];
        Some(&data[start..end])
    }

    #[inline]
    pub fn get_scalar(&self, slot: u8) -> Result<ScalarValue> {
        if !self.is_present(slot) {
            return Ok(ScalarValue::Null);
        }
        match self.attr_type {
            AttributeType::Bool => Ok(ScalarValue::Bool(self.get_bool(slot).unwrap())),
            AttributeType::Int => Ok(ScalarValue::Int(self.get_i64(slot).unwrap())),
            AttributeType::Uint => Ok(ScalarValue::Uint(self.get_u64(slot).unwrap())),
            AttributeType::Float => Ok(ScalarValue::Float(self.get_f64(slot).unwrap())),
            AttributeType::String => {
                let bytes = self.get_str(slot).unwrap();
                let st = std::str::from_utf8(bytes)
                    .map_err(|e| TreeError::CorruptedData(format!("column string utf8: {e}")))?;
                Ok(ScalarValue::String(st.to_owned()))
            }
            AttributeType::IntArray | AttributeType::UintArray | AttributeType::StringArray => {
                Ok(ScalarValue::Null)
            }
        }
    }

    #[inline]
    pub fn get_numeric_f64(&self, slot: u8) -> Option<f64> {
        match self.attr_type {
            AttributeType::Int => self.get_i64(slot).map(|v| v as f64),
            AttributeType::Uint => self.get_u64(slot).map(|v| v as f64),
            AttributeType::Float => self.get_f64(slot),
            _ => None,
        }
    }

    /// Decode all 256 numeric slots into `out` (`0.0` for absent slots), returning `present` mask.
    pub fn decode_numeric_f64_block(&self, out: &mut [f64; ROW_BLOCK_SIZE]) -> [u64; 4] {
        match self.attr_type {
            AttributeType::Int => {
                for (s, dst) in out.iter_mut().enumerate() {
                    *dst = self.get_i64(s as u8).map_or(0.0, |v| v as f64);
                }
            }
            AttributeType::Uint => {
                for (s, dst) in out.iter_mut().enumerate() {
                    *dst = self.get_u64(s as u8).map_or(0.0, |v| v as f64);
                }
            }
            AttributeType::Float => {
                for (s, dst) in out.iter_mut().enumerate() {
                    *dst = self.get_f64(s as u8).unwrap_or(0.0);
                }
            }
            _ => {
                out.fill(0.0);
                return [0u64; 4];
            }
        }
        self.present
    }

    /// Encode each scalar (or array element) at `slot` using ordered key encoding (for facets).
    /// Calls `f` once per scalar or array element with the encoded byte slice.
    pub fn for_each_encoded_value(
        &self,
        slot: u8,
        buf: &mut Vec<u8>,
        mut f: impl FnMut(&[u8]),
    ) -> Result<()> {
        if !self.is_present(slot) {
            return Ok(());
        }
        let s = slot as usize;
        match self.attr_type {
            AttributeType::Bool => {
                buf.clear();
                buf.push(u8::from(self.get_bool(slot).unwrap()));
                f(buf);
            }
            AttributeType::Int => {
                buf.clear();
                buf.extend_from_slice(&encode_i64_ordered(self.get_i64(slot).unwrap()));
                f(buf);
            }
            AttributeType::Uint => {
                buf.clear();
                buf.extend_from_slice(&self.get_u64(slot).unwrap().to_be_bytes());
                f(buf);
            }
            AttributeType::Float => {
                buf.clear();
                buf.extend_from_slice(
                    &encode_f64_ordered(self.get_f64(slot).unwrap()).unwrap_or([0xFF; 8]),
                );
                f(buf);
            }
            AttributeType::String => {
                buf.clear();
                encode_str_ordered(self.get_str(slot).unwrap(), buf);
                f(buf);
            }
            AttributeType::IntArray => {
                let start = read_u32_at(self.payload, s * 4) as usize;
                let end = read_u32_at(self.payload, (s + 1) * 4) as usize;
                let base = (ROW_BLOCK_SIZE + 1) * 4;
                for idx in start..end {
                    let off = base + idx * 8;
                    let v = i64::from_le_bytes(self.payload[off..off + 8].try_into().unwrap());
                    buf.clear();
                    buf.extend_from_slice(&encode_i64_ordered(v));
                    f(buf);
                }
            }
            AttributeType::UintArray => {
                let start = read_u32_at(self.payload, s * 4) as usize;
                let end = read_u32_at(self.payload, (s + 1) * 4) as usize;
                let base = (ROW_BLOCK_SIZE + 1) * 4;
                for idx in start..end {
                    let off = base + idx * 8;
                    let v = u64::from_le_bytes(self.payload[off..off + 8].try_into().unwrap());
                    buf.clear();
                    buf.extend_from_slice(&v.to_be_bytes());
                    f(buf);
                }
            }
            AttributeType::StringArray => {
                let hdr = (ROW_BLOCK_SIZE + 1) * 4;
                let total_elems = read_u32_at(self.payload, ROW_BLOCK_SIZE * 4) as usize;
                let data_base = hdr + (total_elems + 1) * 4;
                let elem_start = read_u32_at(self.payload, s * 4) as usize;
                let elem_end = read_u32_at(self.payload, (s + 1) * 4) as usize;
                for ei in elem_start..elem_end {
                    let ss = read_u32_at(self.payload, hdr + ei * 4) as usize;
                    let se = read_u32_at(self.payload, hdr + (ei + 1) * 4) as usize;
                    buf.clear();
                    encode_str_ordered(&self.payload[data_base + ss..data_base + se], buf);
                    f(buf);
                }
            }
        }
        Ok(())
    }

    /// Decode any `slot` into an owned [`AttributeValue`] (for `Collection::get` / detached reads).
    pub fn decode_attr_value(&self, slot: u8) -> Result<AttributeValue> {
        if !self.is_present(slot) {
            return Ok(AttributeValue::Null);
        }
        let s = slot as usize;
        match self.attr_type {
            AttributeType::Bool => Ok(AttributeValue::Bool(self.get_bool(slot).unwrap())),
            AttributeType::Int => Ok(AttributeValue::Int(self.get_i64(slot).unwrap())),
            AttributeType::Uint => Ok(AttributeValue::Uint(self.get_u64(slot).unwrap())),
            AttributeType::Float => Ok(AttributeValue::Float(self.get_f64(slot).unwrap())),
            AttributeType::String => {
                let st = std::str::from_utf8(self.get_str(slot).unwrap())
                    .map_err(|e| TreeError::CorruptedData(format!("column string utf8: {e}")))?;
                Ok(AttributeValue::String(st.to_owned()))
            }
            AttributeType::IntArray => {
                let start = read_u32_at(self.payload, s * 4) as usize;
                let end = read_u32_at(self.payload, (s + 1) * 4) as usize;
                let base = (ROW_BLOCK_SIZE + 1) * 4;
                let mut out = Vec::with_capacity(end - start);
                for idx in start..end {
                    let off = base + idx * 8;
                    out.push(i64::from_le_bytes(
                        self.payload[off..off + 8].try_into().unwrap(),
                    ));
                }
                Ok(AttributeValue::IntArray(out))
            }
            AttributeType::UintArray => {
                let start = read_u32_at(self.payload, s * 4) as usize;
                let end = read_u32_at(self.payload, (s + 1) * 4) as usize;
                let base = (ROW_BLOCK_SIZE + 1) * 4;
                let mut out = Vec::with_capacity(end - start);
                for idx in start..end {
                    let off = base + idx * 8;
                    out.push(u64::from_le_bytes(
                        self.payload[off..off + 8].try_into().unwrap(),
                    ));
                }
                Ok(AttributeValue::UintArray(out))
            }
            AttributeType::StringArray => {
                let hdr = (ROW_BLOCK_SIZE + 1) * 4;
                let total_elems = read_u32_at(self.payload, ROW_BLOCK_SIZE * 4) as usize;
                let data_base = hdr + (total_elems + 1) * 4;
                let elem_start = read_u32_at(self.payload, s * 4) as usize;
                let elem_end = read_u32_at(self.payload, (s + 1) * 4) as usize;
                let mut out = Vec::with_capacity(elem_end - elem_start);
                for ei in elem_start..elem_end {
                    let ss = read_u32_at(self.payload, hdr + ei * 4) as usize;
                    let se = read_u32_at(self.payload, hdr + (ei + 1) * 4) as usize;
                    let st = std::str::from_utf8(&self.payload[data_base + ss..data_base + se])
                        .map_err(|e| {
                            TreeError::CorruptedData(format!("column string_array utf8: {e}"))
                        })?;
                    out.push(st.to_owned());
                }
                Ok(AttributeValue::StringArray(out))
            }
        }
    }
}

#[inline]
pub(super) fn read_u32_at(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}
