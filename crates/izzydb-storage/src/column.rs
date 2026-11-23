//! 256-row columnar block codec (`cf_columns`, `cf_detached`).
//!
//! Wire layout (`storage-format.md` §4.4):
//! - Header (8 B): `[version u8 = 1][kind u8][width u8][flags u8][present_count u16 LE][reserved u16 = 0]`
//!   - `kind`: matches `AttributeType as u8` (`1=String, 2=Int, 3=Uint, 4=Float, 5=Bool, 6=StringArray, 7=IntArray, 8=UintArray`)
//!   - `flags & 0x01`: `ALL_PRESENT` (`present: [u64; 4]` LE bitmap omitted when all 256 slots present)
//!   - `flags & 0x02`: `FOR` (frame-of-reference base `u64`/`i64` LE preceding narrow deltas `width` in `0/1/2/4`)
//! - Present bitmap (32 B `[u64; 4]` LE, unless `ALL_PRESENT`)
//! - Payload per `kind` (slot-indexed over `0..256`).

use izzydb_core::config::ROW_BLOCK_SIZE;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{AttributeType, AttributeValue, ScalarValue};

use crate::codec::{RowFieldRef, encode_f64_ordered, encode_i64_ordered, encode_str_ordered};

const COLUMN_BLOCK_VERSION: u8 = 1;
const COL_HEADER_LEN: usize = 8;
const COL_FLAG_ALL_PRESENT: u8 = 0x01;
const COL_FLAG_FOR: u8 = 0x02;

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
fn fill_pending_offsets(offsets: &mut [u32; ROW_BLOCK_SIZE + 1], upto_slot: usize, cur: u32) {
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
fn push_header(buf: &mut Vec<u8>, kind: AttributeType, width: u8, flags: u8, present_count: u16) {
    buf.push(COLUMN_BLOCK_VERSION);
    buf.push(kind.as_u8());
    buf.push(width);
    buf.push(flags);
    buf.extend_from_slice(&present_count.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
}

#[inline]
fn push_bitmap(buf: &mut Vec<u8>, words: &[u64; 4]) {
    for &w in words {
        buf.extend_from_slice(&w.to_le_bytes());
    }
}

/// Zero-copy borrowed view over an encoded 256-slot `ColumnBlock`.
#[derive(Debug, Clone, Copy)]
pub struct ColumnBlockView<'a> {
    pub attr_type: AttributeType,
    pub width: u8,
    pub flags: u8,
    pub present_count: u16,
    present: [u64; 4],
    payload: &'a [u8],
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

    fn validate_payload(
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
fn read_u32_at(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_block_roundtrip_all_types_and_for_widths() {
        // Test Bool (ALL_PRESENT)
        let mut b = ColumnBlockBuilder::new(AttributeType::Bool);
        for i in 0..256u16 {
            b.set_attr_value(i as u8, &AttributeValue::Bool(i % 3 == 0))
                .unwrap();
        }
        let enc = b.encode();
        let view = ColumnBlockView::parse(&enc).unwrap();
        assert_eq!(view.present_count, 256);
        assert_ne!(view.flags & COL_FLAG_ALL_PRESENT, 0);
        for i in 0..256u16 {
            assert_eq!(view.get_bool(i as u8), Some(i % 3 == 0));
        }

        // Test Int FOR widths: 0 (const), 1, 2, 4, 8
        for (min_v, step) in [
            (-500i64, 0i64),
            (-100, 1),
            (10_000, 200),
            (-1_000_000, 100_000),
            (i64::MIN / 2, (i64::MAX / 256)),
        ] {
            let mut b = ColumnBlockBuilder::new(AttributeType::Int);
            for i in 0..256u16 {
                if i % 7 != 0 {
                    b.set_attr_value(i as u8, &AttributeValue::Int(min_v + (i as i64) * step))
                        .unwrap();
                }
            }
            let enc = b.encode();
            let view = ColumnBlockView::parse(&enc).unwrap();
            for i in 0..256u16 {
                let expect = (i % 7 != 0).then_some(min_v + (i as i64) * step);
                assert_eq!(view.get_i64(i as u8), expect);
            }
        }

        // Test Uint FOR widths
        for (min_v, step) in [(42u64, 0u64), (10, 1), (0, 250), (1, 100_000), (0, u64::MAX / 256)] {
            let mut b = ColumnBlockBuilder::new(AttributeType::Uint);
            for i in 0..200u16 {
                if i % 5 != 0 {
                    b.set_attr_value(i as u8, &AttributeValue::Uint(min_v + (i as u64) * step))
                        .unwrap();
                }
            }
            let enc = b.encode();
            let view = ColumnBlockView::parse(&enc).unwrap();
            for i in 0..256u16 {
                let expect = (i < 200 && i % 5 != 0).then_some(min_v + (i as u64) * step);
                assert_eq!(view.get_u64(i as u8), expect);
            }
        }

        // Test Float preserving -0.0 and NaN/arbitrary bits
        let mut b = ColumnBlockBuilder::new(AttributeType::Float);
        b.set_attr_value(0, &AttributeValue::Float(-0.0)).unwrap();
        b.set_attr_value(1, &AttributeValue::Float(0.0)).unwrap();
        b.set_attr_value(5, &AttributeValue::Float(3.5)).unwrap();
        let enc = b.encode();
        let view = ColumnBlockView::parse(&enc).unwrap();
        assert_eq!(view.get_f64(0).unwrap().to_bits(), (-0.0f64).to_bits());
        assert_eq!(view.get_f64(1).unwrap().to_bits(), 0.0f64.to_bits());
        assert_eq!(view.get_f64(2), None);
        assert_eq!(view.get_f64(5), Some(3.5));

        // Test String + arrays
        let mut bs = ColumnBlockBuilder::new(AttributeType::String);
        bs.set_attr_value(2, &AttributeValue::String("alpha".into()))
            .unwrap();
        bs.set_attr_value(10, &AttributeValue::String(String::new()))
            .unwrap();
        let enc = bs.encode();
        let vs = ColumnBlockView::parse(&enc).unwrap();
        assert_eq!(vs.get_str(0), None);
        assert_eq!(vs.get_str(2), Some(b"alpha".as_slice()));
        assert_eq!(vs.get_str(10), Some(b"".as_slice()));

        let mut bia = ColumnBlockBuilder::new(AttributeType::IntArray);
        bia.set_attr_value(3, &AttributeValue::IntArray(vec![-1, 0, 42]))
            .unwrap();
        bia.set_attr_value(7, &AttributeValue::IntArray(Vec::new()))
            .unwrap();
        let enc = bia.encode();
        let via = ColumnBlockView::parse(&enc).unwrap();
        assert_eq!(
            via.decode_attr_value(3).unwrap(),
            AttributeValue::IntArray(vec![-1, 0, 42])
        );
        assert_eq!(
            via.decode_attr_value(7).unwrap(),
            AttributeValue::IntArray(Vec::new())
        );

        let mut bsa = ColumnBlockBuilder::new(AttributeType::StringArray);
        bsa.set_attr_value(1, &AttributeValue::StringArray(vec!["x".into(), "".into(), "yz".into()]))
            .unwrap();
        let enc = bsa.encode();
        let vsa = ColumnBlockView::parse(&enc).unwrap();
        assert_eq!(
            vsa.decode_attr_value(1).unwrap(),
            AttributeValue::StringArray(vec!["x".into(), "".into(), "yz".into()])
        );
        assert_eq!(vsa.decode_attr_value(0).unwrap(), AttributeValue::Null);
    }
}
