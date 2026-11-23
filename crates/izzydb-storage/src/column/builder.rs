//! Typed 256-slot block builder and encoder.

use izzydb_core::config::ROW_BLOCK_SIZE;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{AttributeType, AttributeValue};
use crate::codec::RowFieldRef;

use super::*;

/// Typed 256-slot block builder + encoder for `cf_columns` and `cf_detached`.
#[derive(Debug, Clone)]
pub enum ColumnBlockBuilder {
    Bool {
        present: [u64; 4],
        values: [u64; 4],
        present_count: u16,
    },
    Int {
        present: [u64; 4],
        values: [i64; ROW_BLOCK_SIZE],
        present_count: u16,
    },
    Uint {
        present: [u64; 4],
        values: [u64; ROW_BLOCK_SIZE],
        present_count: u16,
    },
    Float {
        present: [u64; 4],
        values: [f64; ROW_BLOCK_SIZE],
        present_count: u16,
    },
    String {
        present: [u64; 4],
        offsets: [u32; ROW_BLOCK_SIZE + 1],
        data: Vec<u8>,
        present_count: u16,
    },
    IntArray {
        present: [u64; 4],
        offsets: [u32; ROW_BLOCK_SIZE + 1],
        elems: Vec<i64>,
        present_count: u16,
    },
    UintArray {
        present: [u64; 4],
        offsets: [u32; ROW_BLOCK_SIZE + 1],
        elems: Vec<u64>,
        present_count: u16,
    },
    StringArray {
        present: [u64; 4],
        elem_offsets: [u32; ROW_BLOCK_SIZE + 1],
        str_offsets: Vec<u32>,
        data: Vec<u8>,
        present_count: u16,
    },
}

impl ColumnBlockBuilder {
    pub fn new(attr_type: AttributeType) -> Self {
        match attr_type {
            AttributeType::Bool => Self::Bool {
                present: [0; 4],
                values: [0; 4],
                present_count: 0,
            },
            AttributeType::Int => Self::Int {
                present: [0; 4],
                values: [0; ROW_BLOCK_SIZE],
                present_count: 0,
            },
            AttributeType::Uint => Self::Uint {
                present: [0; 4],
                values: [0; ROW_BLOCK_SIZE],
                present_count: 0,
            },
            AttributeType::Float => Self::Float {
                present: [0; 4],
                values: [0.0; ROW_BLOCK_SIZE],
                present_count: 0,
            },
            AttributeType::String => Self::String {
                present: [0; 4],
                offsets: [0; ROW_BLOCK_SIZE + 1],
                data: Vec::new(),
                present_count: 0,
            },
            AttributeType::IntArray => Self::IntArray {
                present: [0; 4],
                offsets: [0; ROW_BLOCK_SIZE + 1],
                elems: Vec::new(),
                present_count: 0,
            },
            AttributeType::UintArray => Self::UintArray {
                present: [0; 4],
                offsets: [0; ROW_BLOCK_SIZE + 1],
                elems: Vec::new(),
                present_count: 0,
            },
            AttributeType::StringArray => Self::StringArray {
                present: [0; 4],
                elem_offsets: [0; ROW_BLOCK_SIZE + 1],
                str_offsets: vec![0],
                data: Vec::new(),
                present_count: 0,
            },
        }
    }

    #[inline]
    pub fn present_count(&self) -> u16 {
        match self {
            Self::Bool { present_count, .. }
            | Self::Int { present_count, .. }
            | Self::Uint { present_count, .. }
            | Self::Float { present_count, .. }
            | Self::String { present_count, .. }
            | Self::IntArray { present_count, .. }
            | Self::UintArray { present_count, .. }
            | Self::StringArray { present_count, .. } => *present_count,
        }
    }

    /// Overlay sparse `(slot, AttributeValue)` updates onto an optional existing [`ColumnBlockView`],
    /// returning `Ok(None)` if the resulting block has zero present values (caller should delete key).
    pub fn merge_block(
        existing: Option<&ColumnBlockView<'_>>,
        attr_type: AttributeType,
        updates: &[(u8, AttributeValue)],
    ) -> Result<Option<Vec<u8>>> {
        let mut by_slot: [Option<&AttributeValue>; ROW_BLOCK_SIZE] = [None; ROW_BLOCK_SIZE];
        for (slot, val) in updates {
            by_slot[*slot as usize] = Some(val);
        }
        let mut builder = Self::new(attr_type);
        for (slot_idx, slot_val) in by_slot.iter().enumerate() {
            let slot = slot_idx as u8;
            if let Some(new_val) = slot_val {
                if !matches!(new_val, AttributeValue::Null) {
                    builder.set_attr_value(slot, new_val)?;
                }
            } else if let Some(ev) = existing
                && ev.is_present(slot)
            {
                let old_val = ev.decode_attr_value(slot)?;
                builder.set_attr_value(slot, &old_val)?;
            }
        }
        if builder.present_count() == 0 {
            Ok(None)
        } else {
            Ok(Some(builder.encode()))
        }
    }

    /// Set `slot` (`0..=255`) from an [`AttributeValue`].
    ///
    /// Note: variable-length builders (`String`, `*Array`) require slots to be inserted in
    /// strictly ascending order (or finished via `seal_varlen_up_to`).
    pub fn set_attr_value(&mut self, slot: u8, val: &AttributeValue) -> Result<()> {
        if matches!(val, AttributeValue::Null) {
            return Ok(());
        }
        let s = slot as usize;
        let word = s >> 6;
        let bit = 1u64 << (s & 63);
        match (self, val) {
            (Self::Bool { present, values, present_count }, AttributeValue::Bool(b)) => {
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                if *b {
                    values[word] |= bit;
                } else {
                    values[word] &= !bit;
                }
                Ok(())
            }
            (Self::Int { present, values, present_count }, AttributeValue::Int(v)) => {
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                values[s] = *v;
                Ok(())
            }
            (Self::Uint { present, values, present_count }, AttributeValue::Uint(v)) => {
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                values[s] = *v;
                Ok(())
            }
            (Self::Float { present, values, present_count }, AttributeValue::Float(v)) => {
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                values[s] = *v;
                Ok(())
            }
            (
                Self::String {
                    present,
                    offsets,
                    data,
                    present_count,
                },
                AttributeValue::String(st),
            ) => {
                let start = data.len() as u32;
                fill_pending_offsets(offsets, s, start);
                data.extend_from_slice(st.as_bytes());
                offsets[s + 1] = data.len() as u32;
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                Ok(())
            }
            (
                Self::IntArray {
                    present,
                    offsets,
                    elems,
                    present_count,
                },
                AttributeValue::IntArray(arr),
            ) => {
                let start = elems.len() as u32;
                fill_pending_offsets(offsets, s, start);
                elems.extend_from_slice(arr);
                offsets[s + 1] = elems.len() as u32;
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                Ok(())
            }
            (
                Self::UintArray {
                    present,
                    offsets,
                    elems,
                    present_count,
                },
                AttributeValue::UintArray(arr),
            ) => {
                let start = elems.len() as u32;
                fill_pending_offsets(offsets, s, start);
                elems.extend_from_slice(arr);
                offsets[s + 1] = elems.len() as u32;
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                Ok(())
            }
            (
                Self::StringArray {
                    present,
                    elem_offsets,
                    str_offsets,
                    data,
                    present_count,
                },
                AttributeValue::StringArray(arr),
            ) => {
                let start = (str_offsets.len() - 1) as u32;
                fill_pending_offsets(elem_offsets, s, start);
                for elem in arr {
                    data.extend_from_slice(elem.as_bytes());
                    str_offsets.push(data.len() as u32);
                }
                elem_offsets[s + 1] = (str_offsets.len() - 1) as u32;
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                Ok(())
            }
            _ => Err(TreeError::InvalidAttribute {
                name: "column".into(),
                reason: "ColumnBlockBuilder value type mismatch".into(),
            }),
        }
    }

    /// Populate `slot` directly from a zero-copy [`RowFieldRef`] in ascending slot order.
    pub fn set_row_field(&mut self, slot: u8, field: RowFieldRef<'_>) -> Result<()> {
        if field.is_null() {
            return Ok(());
        }
        let s = slot as usize;
        let word = s >> 6;
        let bit = 1u64 << (s & 63);
        match (self, field) {
            (
                Self::Bool {
                    present,
                    values,
                    present_count,
                },
                RowFieldRef::Scalar(crate::codec::RawScalar::Bool(b)),
            ) => {
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                if b {
                    values[word] |= bit;
                } else {
                    values[word] &= !bit;
                }
                Ok(())
            }
            (
                Self::Int {
                    present,
                    values,
                    present_count,
                },
                RowFieldRef::Scalar(crate::codec::RawScalar::Int(v)),
            ) => {
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                values[s] = v;
                Ok(())
            }
            (
                Self::Uint {
                    present,
                    values,
                    present_count,
                },
                RowFieldRef::Scalar(crate::codec::RawScalar::Uint(v)),
            ) => {
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                values[s] = v;
                Ok(())
            }
            (
                Self::Float {
                    present,
                    values,
                    present_count,
                },
                RowFieldRef::Scalar(crate::codec::RawScalar::Float(v)),
            ) => {
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                values[s] = v;
                Ok(())
            }
            (
                Self::String {
                    present,
                    offsets,
                    data,
                    present_count,
                },
                RowFieldRef::Scalar(crate::codec::RawScalar::Str(bytes)),
            ) => {
                let start = data.len() as u32;
                fill_pending_offsets(offsets, s, start);
                data.extend_from_slice(bytes);
                offsets[s + 1] = data.len() as u32;
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                Ok(())
            }
            (
                Self::IntArray {
                    present,
                    offsets,
                    elems,
                    present_count,
                },
                RowFieldRef::IntArray(_),
            ) => {
                let start = elems.len() as u32;
                fill_pending_offsets(offsets, s, start);
                field.for_each_scalar(|raw| {
                    if let crate::codec::RawScalar::Int(v) = raw {
                        elems.push(v);
                    }
                })?;
                offsets[s + 1] = elems.len() as u32;
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                Ok(())
            }
            (
                Self::UintArray {
                    present,
                    offsets,
                    elems,
                    present_count,
                },
                RowFieldRef::UintArray(_),
            ) => {
                let start = elems.len() as u32;
                fill_pending_offsets(offsets, s, start);
                field.for_each_scalar(|raw| {
                    if let crate::codec::RawScalar::Uint(v) = raw {
                        elems.push(v);
                    }
                })?;
                offsets[s + 1] = elems.len() as u32;
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                Ok(())
            }
            (
                Self::StringArray {
                    present,
                    elem_offsets,
                    str_offsets,
                    data,
                    present_count,
                },
                RowFieldRef::StringArray(_),
            ) => {
                let start = (str_offsets.len() - 1) as u32;
                fill_pending_offsets(elem_offsets, s, start);
                field.for_each_scalar(|raw| {
                    if let crate::codec::RawScalar::Str(b) = raw {
                        data.extend_from_slice(b);
                        str_offsets.push(data.len() as u32);
                    }
                })?;
                elem_offsets[s + 1] = (str_offsets.len() - 1) as u32;
                if (present[word] & bit) == 0 {
                    present[word] |= bit;
                    *present_count += 1;
                }
                Ok(())
            }
            _ => Err(TreeError::CorruptedData(
                "ColumnBlockBuilder RowFieldRef type mismatch".into(),
            )),
        }
    }

    pub fn encode(mut self) -> Vec<u8> {
        let count = self.present_count();
        let all_present = count == ROW_BLOCK_SIZE as u16;
        let base_flags = if all_present { COL_FLAG_ALL_PRESENT } else { 0 };

        match &mut self {
            Self::Bool {
                present, values, ..
            } => {
                let mut buf = Vec::with_capacity(COL_HEADER_LEN + 64);
                push_header(&mut buf, AttributeType::Bool, 1, base_flags, count);
                if !all_present {
                    push_bitmap(&mut buf, present);
                }
                push_bitmap(&mut buf, values);
                buf
            }
            Self::Int {
                present, values, ..
            } => {
                if count == 0 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 32);
                    push_header(&mut buf, AttributeType::Int, 0, base_flags, 0);
                    push_bitmap(&mut buf, present);
                    return buf;
                }
                let (mut min_v, mut max_v) = (i64::MAX, i64::MIN);
                for (i, &v) in values.iter().enumerate() {
                    if ((present[i >> 6] >> (i & 63)) & 1) != 0 {
                        min_v = min_v.min(v);
                        max_v = max_v.max(v);
                    }
                }
                let span = max_v.abs_diff(min_v);
                if span == 0 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 40);
                    push_header(&mut buf, AttributeType::Int, 0, base_flags | COL_FLAG_FOR, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    buf.extend_from_slice(&min_v.to_le_bytes());
                    buf
                } else if span <= u8::MAX as u64 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 40 + ROW_BLOCK_SIZE);
                    push_header(&mut buf, AttributeType::Int, 1, base_flags | COL_FLAG_FOR, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    buf.extend_from_slice(&min_v.to_le_bytes());
                    for &v in values.iter() {
                        buf.push(v.wrapping_sub(min_v) as u8);
                    }
                    buf
                } else if span <= u16::MAX as u64 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 40 + ROW_BLOCK_SIZE * 2);
                    push_header(&mut buf, AttributeType::Int, 2, base_flags | COL_FLAG_FOR, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    buf.extend_from_slice(&min_v.to_le_bytes());
                    for &v in values.iter() {
                        buf.extend_from_slice(&(v.wrapping_sub(min_v) as u16).to_le_bytes());
                    }
                    buf
                } else if span <= u32::MAX as u64 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 40 + ROW_BLOCK_SIZE * 4);
                    push_header(&mut buf, AttributeType::Int, 4, base_flags | COL_FLAG_FOR, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    buf.extend_from_slice(&min_v.to_le_bytes());
                    for &v in values.iter() {
                        buf.extend_from_slice(&(v.wrapping_sub(min_v) as u32).to_le_bytes());
                    }
                    buf
                } else {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 32 + ROW_BLOCK_SIZE * 8);
                    push_header(&mut buf, AttributeType::Int, 8, base_flags, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    for &v in values.iter() {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    buf
                }
            }
            Self::Uint {
                present, values, ..
            } => {
                if count == 0 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 32);
                    push_header(&mut buf, AttributeType::Uint, 0, base_flags, 0);
                    push_bitmap(&mut buf, present);
                    return buf;
                }
                let (mut min_v, mut max_v) = (u64::MAX, 0u64);
                for (i, &v) in values.iter().enumerate() {
                    if ((present[i >> 6] >> (i & 63)) & 1) != 0 {
                        min_v = min_v.min(v);
                        max_v = max_v.max(v);
                    }
                }
                let span = max_v - min_v;
                if span == 0 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 40);
                    push_header(&mut buf, AttributeType::Uint, 0, base_flags | COL_FLAG_FOR, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    buf.extend_from_slice(&min_v.to_le_bytes());
                    buf
                } else if span <= u8::MAX as u64 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 40 + ROW_BLOCK_SIZE);
                    push_header(&mut buf, AttributeType::Uint, 1, base_flags | COL_FLAG_FOR, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    buf.extend_from_slice(&min_v.to_le_bytes());
                    for &v in values.iter() {
                        buf.push((v.wrapping_sub(min_v)) as u8);
                    }
                    buf
                } else if span <= u16::MAX as u64 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 40 + ROW_BLOCK_SIZE * 2);
                    push_header(&mut buf, AttributeType::Uint, 2, base_flags | COL_FLAG_FOR, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    buf.extend_from_slice(&min_v.to_le_bytes());
                    for &v in values.iter() {
                        buf.extend_from_slice(&((v.wrapping_sub(min_v)) as u16).to_le_bytes());
                    }
                    buf
                } else if span <= u32::MAX as u64 {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 40 + ROW_BLOCK_SIZE * 4);
                    push_header(&mut buf, AttributeType::Uint, 4, base_flags | COL_FLAG_FOR, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    buf.extend_from_slice(&min_v.to_le_bytes());
                    for &v in values.iter() {
                        buf.extend_from_slice(&((v.wrapping_sub(min_v)) as u32).to_le_bytes());
                    }
                    buf
                } else {
                    let mut buf = Vec::with_capacity(COL_HEADER_LEN + 32 + ROW_BLOCK_SIZE * 8);
                    push_header(&mut buf, AttributeType::Uint, 8, base_flags, count);
                    if !all_present {
                        push_bitmap(&mut buf, present);
                    }
                    for &v in values.iter() {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    buf
                }
            }
            Self::Float {
                present, values, ..
            } => {
                let mut buf = Vec::with_capacity(COL_HEADER_LEN + 32 + ROW_BLOCK_SIZE * 8);
                push_header(&mut buf, AttributeType::Float, 8, base_flags, count);
                if !all_present {
                    push_bitmap(&mut buf, present);
                }
                for &v in values.iter() {
                    buf.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                buf
            }
            Self::String {
                present,
                offsets,
                data,
                ..
            } => {
                fill_pending_offsets(offsets, ROW_BLOCK_SIZE, data.len() as u32);
                let mut buf =
                    Vec::with_capacity(COL_HEADER_LEN + 32 + (ROW_BLOCK_SIZE + 1) * 4 + data.len());
                push_header(&mut buf, AttributeType::String, 0, base_flags, count);
                if !all_present {
                    push_bitmap(&mut buf, present);
                }
                for &o in offsets.iter() {
                    buf.extend_from_slice(&o.to_le_bytes());
                }
                buf.extend_from_slice(data);
                buf
            }
            Self::IntArray {
                present,
                offsets,
                elems,
                ..
            } => {
                fill_pending_offsets(offsets, ROW_BLOCK_SIZE, elems.len() as u32);
                let mut buf = Vec::with_capacity(
                    COL_HEADER_LEN + 32 + (ROW_BLOCK_SIZE + 1) * 4 + elems.len() * 8,
                );
                push_header(&mut buf, AttributeType::IntArray, 8, base_flags, count);
                if !all_present {
                    push_bitmap(&mut buf, present);
                }
                for &o in offsets.iter() {
                    buf.extend_from_slice(&o.to_le_bytes());
                }
                for &e in elems.iter() {
                    buf.extend_from_slice(&e.to_le_bytes());
                }
                buf
            }
            Self::UintArray {
                present,
                offsets,
                elems,
                ..
            } => {
                fill_pending_offsets(offsets, ROW_BLOCK_SIZE, elems.len() as u32);
                let mut buf = Vec::with_capacity(
                    COL_HEADER_LEN + 32 + (ROW_BLOCK_SIZE + 1) * 4 + elems.len() * 8,
                );
                push_header(&mut buf, AttributeType::UintArray, 8, base_flags, count);
                if !all_present {
                    push_bitmap(&mut buf, present);
                }
                for &o in offsets.iter() {
                    buf.extend_from_slice(&o.to_le_bytes());
                }
                for &e in elems.iter() {
                    buf.extend_from_slice(&e.to_le_bytes());
                }
                buf
            }
            Self::StringArray {
                present,
                elem_offsets,
                str_offsets,
                data,
                ..
            } => {
                fill_pending_offsets(elem_offsets, ROW_BLOCK_SIZE, (str_offsets.len() - 1) as u32);
                let mut buf = Vec::with_capacity(
                    COL_HEADER_LEN
                        + 32
                        + (ROW_BLOCK_SIZE + 1) * 4
                        + str_offsets.len() * 4
                        + data.len(),
                );
                push_header(&mut buf, AttributeType::StringArray, 0, base_flags, count);
                if !all_present {
                    push_bitmap(&mut buf, present);
                }
                for &o in elem_offsets.iter() {
                    buf.extend_from_slice(&o.to_le_bytes());
                }
                for &o in str_offsets.iter() {
                    buf.extend_from_slice(&o.to_le_bytes());
                }
                buf.extend_from_slice(data);
                buf
            }
        }
    }
}

#[inline]
pub(super) fn fill_pending_offsets(offsets: &mut [u32; ROW_BLOCK_SIZE + 1], upto_slot: usize, cur: u32) {
    let mut i = upto_slot;
    while i > 0 && offsets[i] == 0 {
        offsets[i] = cur;
        i -= 1;
    }
    if upto_slot == ROW_BLOCK_SIZE {
        offsets[ROW_BLOCK_SIZE] = cur;
    } else if offsets[upto_slot] == 0 {
        offsets[upto_slot] = cur;
    }
}

#[inline]
pub(super) fn push_header(buf: &mut Vec<u8>, kind: AttributeType, width: u8, flags: u8, present_count: u16) {
    buf.push(COLUMN_BLOCK_VERSION);
    buf.push(kind.as_u8());
    buf.push(width);
    buf.push(flags);
    buf.extend_from_slice(&present_count.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
}

#[inline]
pub(super) fn push_bitmap(buf: &mut Vec<u8>, words: &[u64; 4]) {
    for &w in words {
        buf.extend_from_slice(&w.to_le_bytes());
    }
}
