//! Leaf-colocated fixed-width scalar attributes and aggregate node filter summaries.
//!
//! A [`LeafAttrBlock`] mirrors one `PackedLeafNode` slot layout (`0..slot_cap`) for every
//! eligible inline filterable scalar attribute, so a filtered ANN traversal can evaluate
//! attribute predicates on the leaf blob it already fetched instead of issuing random
//! `cf_columns` block reads.
//!
//! ```text
//! slot 0: score=7.5  in_stock=true
//! slot 1: score=null in_stock=false
//! slot 2: score=3.0  in_stock=true
//! ```
//!
//! A [`NodeSummary`] is the conservative min/max/presence/equality-bitset rollup of one leaf or
//! internal subtree. Traversal prunes a child when its summary cannot possibly satisfy the
//! compiled predicate.
//!
//! ```text
//! leaf summary: score in [3.0, 7.5], present=2/3, bits={...}
//! internal rollup: score in [0.5, 9.1], present=182/200, bits={...}
//! ```

use izzydb_core::config::AttrId;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::{AttributeType, AttributeValue};

/// Version byte stored at the start of every [`LeafAttrBlock`].
pub const LEAF_ATTR_BLOCK_VERSION: u8 = 2;

/// Version byte stored at the start of every [`NodeSummary`].
pub const NODE_SUMMARY_VERSION: u8 = 1;

/// Fixed-width scalar attribute types colocated with a leaf.
///
/// Strings, arrays, and detached columns stay in their existing storage; this type only
/// covers values that can be packed into a slot-aligned numeric/bool column.
///
/// ```text
/// LeafAttrSpec { attr_id: 3, attr_type: AttributeType::Float }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafAttrSpec {
    /// Schema attribute id.
    pub attr_id: AttrId,
    /// Declared scalar type (`Bool`, `Int`, `Uint`, or `Float`).
    pub attr_type: AttributeType,
}

impl LeafAttrSpec {
    /// Returns `true` when `attr_type` can be stored in a leaf attribute column.
    #[inline]
    pub fn is_supported(attr_type: AttributeType) -> bool {
        matches!(
            attr_type,
            AttributeType::Bool | AttributeType::Int | AttributeType::Uint | AttributeType::Float
        )
    }
}

/// Slot-aligned values for one leaf attribute column.
#[derive(Debug, Clone, PartialEq)]
pub enum LeafAttrValues {
    /// One bit per slot (`0` = false, `1` = true).
    Bool(Vec<u64>),
    /// Frame-of-reference compressed signed integers (`base + delta`).
    I64 {
        /// Minimum present value in the leaf.
        base: i64,
        /// Delta width in bytes (`0`, `1`, `2`, `4`, or `8`).
        width: u8,
        /// Per-slot little-endian deltas; absent slots store zero.
        deltas: Vec<u8>,
    },
    /// Frame-of-reference compressed unsigned integers (`base + delta`).
    U64 {
        /// Minimum present value in the leaf.
        base: u64,
        /// Delta width in bytes (`0`, `1`, `2`, `4`, or `8`).
        width: u8,
        /// Per-slot little-endian deltas; absent slots store zero.
        deltas: Vec<u8>,
    },
    /// One `f64` per slot (`0.0` for absent slots).
    F64(Vec<f64>),
    /// One losslessly-narrowed `f32` per slot (`0.0` for absent slots).
    F32(Vec<f32>),
}

/// One slot-aligned scalar column inside a [`LeafAttrBlock`].
///
/// ```text
/// LeafAttrColumn { attr_id: 2, attr_type: Float, present: [0b101], values: F64([1.5, 0.0, 3.0]) }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LeafAttrColumn {
    /// Schema attribute id.
    pub attr_id: AttrId,
    /// Declared scalar type.
    pub attr_type: AttributeType,
    /// Presence mask (`ceil(slot_cap / 64)` words).
    pub present: Vec<u64>,
    /// Slot-aligned values.
    pub values: LeafAttrValues,
}

impl LeafAttrColumn {
    /// Encode this column's payload after its header.
    fn encode_payload(&self, buf: &mut Vec<u8>) {
        for &word in &self.present {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        match &self.values {
            LeafAttrValues::Bool(words) => {
                for &word in words {
                    buf.extend_from_slice(&word.to_le_bytes());
                }
            }
            LeafAttrValues::I64 {
                base,
                width,
                deltas,
            } => {
                buf.extend_from_slice(&base.to_le_bytes());
                buf.push(*width);
                buf.extend_from_slice(deltas);
            }
            LeafAttrValues::U64 {
                base,
                width,
                deltas,
            } => {
                buf.extend_from_slice(&base.to_le_bytes());
                buf.push(*width);
                buf.extend_from_slice(deltas);
            }
            LeafAttrValues::F64(values) => {
                for &v in values {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            LeafAttrValues::F32(values) => {
                for &v in values {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }

    /// Return the scalar value at `slot`, or `None` when absent.
    #[inline]
    pub fn value_at(&self, slot: usize) -> Option<LeafAttrValue> {
        if !self.is_present(slot) {
            return None;
        }
        match &self.values {
            LeafAttrValues::Bool(words) => {
                let word = words[slot >> 6];
                Some(LeafAttrValue::Bool(((word >> (slot & 63)) & 1) != 0))
            }
            LeafAttrValues::I64 {
                base,
                width,
                deltas,
            } => {
                let delta = read_delta(deltas, *width, slot);
                Some(LeafAttrValue::Int(base.wrapping_add(delta as i64)))
            }
            LeafAttrValues::U64 {
                base,
                width,
                deltas,
            } => {
                let delta = read_delta(deltas, *width, slot);
                Some(LeafAttrValue::Uint(base.wrapping_add(delta)))
            }
            LeafAttrValues::F64(values) => Some(LeafAttrValue::Float(values[slot])),
            LeafAttrValues::F32(values) => Some(LeafAttrValue::Float(values[slot] as f64)),
        }
    }

    #[inline]
    fn is_present(&self, slot: usize) -> bool {
        let word = self.present[slot >> 6];
        ((word >> (slot & 63)) & 1) != 0
    }
}

/// A decoded leaf attribute value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeafAttrValue {
    /// Boolean value.
    Bool(bool),
    /// Signed integer value.
    Int(i64),
    /// Unsigned integer value.
    Uint(u64),
    /// Floating point value.
    Float(f64),
}

impl LeafAttrValue {
    /// Convert the value into the unsigned order-preserving representation used by summaries
    /// and bit-sliced indexes.
    ///
    /// Int uses sign-bit flipping; float uses the IEEE-754 total-order transform used by
    /// `izzydb_codec::encode_f64_ordered`.
    #[inline]
    pub fn ordered_u64(self) -> u64 {
        match self {
            Self::Bool(v) => u64::from(v),
            Self::Int(v) => (v as u64) ^ (1u64 << 63),
            Self::Uint(v) => v,
            Self::Float(v) => f64_ordered_bits(v),
        }
    }
}

/// Return the unsigned order-preserving bit pattern for an `f64`.
#[inline]
pub fn f64_ordered_bits(v: f64) -> u64 {
    let normalized = if v == 0.0 { 0.0 } else { v };
    let bits = normalized.to_bits();
    if (bits >> 63) != 0 {
        !bits
    } else {
        bits ^ (1u64 << 63)
    }
}

/// One slot-aligned leaf attribute block already decoded for evaluation.
///
/// ```text
/// LeafAttrBlock { slot_cap: 3, columns: [score, rank, active] }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LeafAttrBlock {
    /// Number of slots covered by every column.
    pub slot_cap: usize,
    /// Columns in ascending `attr_id` order.
    pub columns: Vec<LeafAttrColumn>,
}

impl LeafAttrBlock {
    /// Build an empty block for `specs` and `slot_cap`.
    pub fn empty(specs: &[LeafAttrSpec], slot_cap: usize) -> Self {
        let words = slot_cap.div_ceil(64);
        let columns = specs
            .iter()
            .map(|spec| LeafAttrColumn {
                attr_id: spec.attr_id,
                attr_type: spec.attr_type,
                present: vec![0; words],
                values: match spec.attr_type {
                    AttributeType::Bool => LeafAttrValues::Bool(vec![0; words]),
                    AttributeType::Int => LeafAttrValues::I64 {
                        base: 0,
                        width: 0,
                        deltas: Vec::new(),
                    },
                    AttributeType::Uint => LeafAttrValues::U64 {
                        base: 0,
                        width: 0,
                        deltas: Vec::new(),
                    },
                    AttributeType::Float => LeafAttrValues::F64(vec![0.0; slot_cap]),
                    other => {
                        debug_assert!(!LeafAttrSpec::is_supported(other));
                        LeafAttrValues::Bool(vec![0; words])
                    }
                },
            })
            .collect();
        Self { slot_cap, columns }
    }

    /// Encode this block.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(LEAF_ATTR_BLOCK_VERSION);
        buf.extend_from_slice(&(self.slot_cap as u32).to_le_bytes());
        buf.extend_from_slice(&(self.columns.len() as u16).to_le_bytes());
        for col in &self.columns {
            buf.extend_from_slice(&col.attr_id.to_le_bytes());
            buf.push(col.attr_type.as_u8());
            buf.push(match col.values {
                LeafAttrValues::F32(_) => 1,
                _ => 0,
            });
            let len_off = buf.len();
            buf.extend_from_slice(&0u32.to_le_bytes());
            let start = buf.len();
            col.encode_payload(&mut buf);
            let len = (buf.len() - start) as u32;
            buf[len_off..len_off + 4].copy_from_slice(&len.to_le_bytes());
        }
        buf
    }

    /// Parse an encoded block.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        LeafAttrView::parse(bytes).map(LeafAttrView::into_owned)
    }

    /// Look up a column by attribute id.
    #[inline]
    pub fn column(&self, attr_id: AttrId) -> Option<&LeafAttrColumn> {
        self.columns
            .binary_search_by_key(&attr_id, |c| c.attr_id)
            .ok()
            .map(|idx| &self.columns[idx])
    }
}

/// Zero-copy borrowed view over an encoded [`LeafAttrBlock`].
///
/// ```text
/// let view = LeafAttrView::parse(bytes)?;
/// let score = view.column(1).and_then(|c| c.value_at(slot));
/// ```
#[derive(Debug, Clone)]
pub struct LeafAttrView<'a> {
    /// Number of slots covered by every column.
    pub slot_cap: usize,
    columns: Vec<LeafAttrColumnView<'a>>,
}

impl<'a> LeafAttrView<'a> {
    /// Parse `bytes` into a borrowed view.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.is_empty() {
            return Err(TreeError::CorruptedData(
                "LeafAttrBlock payload is empty".into(),
            ));
        }
        if bytes[0] != LEAF_ATTR_BLOCK_VERSION {
            return Err(TreeError::CorruptedData(format!(
                "unsupported LeafAttrBlock version: {}",
                bytes[0]
            )));
        }
        if bytes.len() < 7 {
            return Err(TreeError::CorruptedData(
                "LeafAttrBlock header truncated".into(),
            ));
        }
        let slot_cap = u32::from_le_bytes(bytes[1..5].try_into().unwrap()) as usize;
        let count = u16::from_le_bytes(bytes[5..7].try_into().unwrap()) as usize;
        let words = slot_cap.div_ceil(64);
        let mut off = 7;
        let mut columns = Vec::with_capacity(count);
        for _ in 0..count {
            if off + 8 > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "LeafAttrBlock column header truncated".into(),
                ));
            }
            let attr_id = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
            let attr_type = AttributeType::from_u8(bytes[off + 2])?;
            let encoding = bytes[off + 3];
            let len = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
            off += 8;
            if off + len > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "LeafAttrBlock column payload truncated".into(),
                ));
            }
            columns.push(LeafAttrColumnView::parse(
                attr_id,
                attr_type,
                encoding,
                slot_cap,
                words,
                &bytes[off..off + len],
            )?);
            off += len;
        }
        if off != bytes.len() {
            return Err(TreeError::CorruptedData(
                "trailing bytes in LeafAttrBlock".into(),
            ));
        }
        Ok(Self { slot_cap, columns })
    }

    /// Number of slots in the block.
    #[inline]
    pub fn slot_cap(&self) -> usize {
        self.slot_cap
    }

    /// Look up a column by attribute id.
    #[inline]
    pub fn column(&self, attr_id: AttrId) -> Option<&LeafAttrColumnView<'a>> {
        self.columns
            .binary_search_by_key(&attr_id, |c| c.attr_id)
            .ok()
            .map(|idx| &self.columns[idx])
    }

    /// Iterate columns.
    #[inline]
    pub fn columns(&self) -> &[LeafAttrColumnView<'a>] {
        &self.columns
    }

    /// Convert into an owned block.
    pub fn into_owned(self) -> LeafAttrBlock {
        LeafAttrBlock {
            slot_cap: self.slot_cap,
            columns: self.columns.into_iter().map(|c| c.into_owned()).collect(),
        }
    }
}

/// Zero-copy view over one encoded leaf attribute column.
///
/// ```text
/// view.column(1).unwrap().value_at(7) // Some(LeafAttrValue::Float(12.5))
/// ```
#[derive(Debug, Clone)]
pub struct LeafAttrColumnView<'a> {
    /// Schema attribute id.
    pub attr_id: AttrId,
    /// Declared scalar type.
    pub attr_type: AttributeType,
    /// `1` for losslessly narrowed `f32` float columns, `0` otherwise.
    pub encoding: u8,
    slot_cap: usize,
    words: usize,
    present: &'a [u8],
    payload: &'a [u8],
}

impl<'a> LeafAttrColumnView<'a> {
    fn parse(
        attr_id: AttrId,
        attr_type: AttributeType,
        encoding: u8,
        slot_cap: usize,
        words: usize,
        payload: &'a [u8],
    ) -> Result<Self> {
        let present_len = words * 8;
        if payload.len() < present_len {
            return Err(TreeError::CorruptedData(
                "LeafAttrBlock present mask truncated".into(),
            ));
        }
        let present = &payload[..present_len];
        let values = &payload[present_len..];
        let expected = match attr_type {
            AttributeType::Bool => words * 8,
            AttributeType::Int | AttributeType::Uint => {
                if values.is_empty() {
                    0
                } else {
                    if values.len() < 9 {
                        return Err(TreeError::CorruptedData(
                            "LeafAttrBlock integer column truncated".into(),
                        ));
                    }
                    let width = values[8] as usize;
                    if !matches!(width, 0 | 1 | 2 | 4 | 8) {
                        return Err(TreeError::CorruptedData(format!(
                            "invalid LeafAttrBlock delta width: {width}"
                        )));
                    }
                    9 + slot_cap * width
                }
            }
            AttributeType::Float => match encoding {
                0 => slot_cap * 8,
                1 => slot_cap * 4,
                other => {
                    return Err(TreeError::CorruptedData(format!(
                        "invalid LeafAttrBlock float encoding: {other}"
                    )));
                }
            },
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "unsupported LeafAttrBlock type: {other:?}"
                )));
            }
        };
        if values.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "LeafAttrBlock {:?} payload length {} != expected {}",
                attr_type,
                values.len(),
                expected
            )));
        }
        Ok(Self {
            attr_id,
            attr_type,
            encoding,
            slot_cap,
            words,
            present,
            payload: values,
        })
    }

    /// Number of slots in the column.
    #[inline]
    pub fn slot_cap(&self) -> usize {
        self.slot_cap
    }

    /// Return `true` when `slot` has a value.
    #[inline]
    pub fn is_present(&self, slot: usize) -> bool {
        if slot >= self.slot_cap {
            return false;
        }
        let w = slot >> 6;
        let word = u64::from_le_bytes(
            self.present[w * 8..(w + 1) * 8]
                .try_into()
                .expect("checked present mask"),
        );
        ((word >> (slot & 63)) & 1) != 0
    }

    /// Copy the presence mask into `dst`.
    pub fn present_mask_into(&self, dst: &mut Vec<u64>) {
        dst.clear();
        for chunk in self.present.as_chunks::<8>().0 {
            dst.push(u64::from_le_bytes(*chunk));
        }
    }

    /// Return the value at `slot`, or `None` when absent.
    #[inline]
    pub fn value_at(&self, slot: usize) -> Option<LeafAttrValue> {
        if !self.is_present(slot) {
            return None;
        }
        Some(match self.attr_type {
            AttributeType::Bool => {
                let w = slot >> 6;
                let word = u64::from_le_bytes(
                    self.payload[w * 8..(w + 1) * 8]
                        .try_into()
                        .expect("checked bool payload"),
                );
                LeafAttrValue::Bool(((word >> (slot & 63)) & 1) != 0)
            }
            AttributeType::Int => {
                let base = i64::from_le_bytes(self.payload[0..8].try_into().unwrap());
                let width = self.payload[8] as usize;
                let delta = read_delta(&self.payload[9..], width as u8, slot);
                LeafAttrValue::Int(base.wrapping_add(delta as i64))
            }
            AttributeType::Uint => {
                let base = u64::from_le_bytes(self.payload[0..8].try_into().unwrap());
                let width = self.payload[8] as usize;
                let delta = read_delta(&self.payload[9..], width as u8, slot);
                LeafAttrValue::Uint(base.wrapping_add(delta))
            }
            AttributeType::Float if self.encoding == 1 => {
                let off = slot * 4;
                LeafAttrValue::Float(f32::from_le_bytes(
                    self.payload[off..off + 4].try_into().unwrap(),
                ) as f64)
            }
            AttributeType::Float => {
                let off = slot * 8;
                LeafAttrValue::Float(f64::from_le_bytes(
                    self.payload[off..off + 8].try_into().unwrap(),
                ))
            }
            other => {
                debug_assert!(!LeafAttrSpec::is_supported(other));
                return None;
            }
        })
    }

    fn into_owned(self) -> LeafAttrColumn {
        let mut present = Vec::with_capacity(self.words);
        for chunk in self.present.as_chunks::<8>().0 {
            present.push(u64::from_le_bytes(*chunk));
        }
        let values = match self.attr_type {
            AttributeType::Bool => {
                let mut words = Vec::with_capacity(self.words);
                for chunk in self.payload.as_chunks::<8>().0 {
                    words.push(u64::from_le_bytes(*chunk));
                }
                LeafAttrValues::Bool(words)
            }
            AttributeType::Int => {
                let base = i64::from_le_bytes(self.payload[0..8].try_into().unwrap());
                let width = self.payload[8];
                LeafAttrValues::I64 {
                    base,
                    width,
                    deltas: self.payload[9..].to_vec(),
                }
            }
            AttributeType::Uint => {
                let base = u64::from_le_bytes(self.payload[0..8].try_into().unwrap());
                let width = self.payload[8];
                LeafAttrValues::U64 {
                    base,
                    width,
                    deltas: self.payload[9..].to_vec(),
                }
            }
            AttributeType::Float if self.encoding == 1 => {
                let mut values = Vec::with_capacity(self.slot_cap);
                for chunk in self.payload.as_chunks::<4>().0 {
                    values.push(f32::from_le_bytes(*chunk));
                }
                LeafAttrValues::F32(values)
            }
            AttributeType::Float => {
                let mut values = Vec::with_capacity(self.slot_cap);
                for chunk in self.payload.as_chunks::<8>().0 {
                    values.push(f64::from_le_bytes(*chunk));
                }
                LeafAttrValues::F64(values)
            }
            other => {
                debug_assert!(!LeafAttrSpec::is_supported(other));
                LeafAttrValues::Bool(vec![0; self.words])
            }
        };
        LeafAttrColumn {
            attr_id: self.attr_id,
            attr_type: self.attr_type,
            present,
            values,
        }
    }
}

/// Builder that accumulates slot-aligned values for one leaf.
///
/// ```text
/// let mut b = LeafAttrBlockBuilder::new(&specs, leaf.slot_cap);
/// b.set_from_row_view(slot, &row_view)?;
/// let bytes = b.finish().encode();
/// ```
#[derive(Debug, Clone)]
pub struct LeafAttrBlockBuilder {
    slot_cap: usize,
    columns: Vec<LeafAttrColumn>,
    specs: Vec<LeafAttrSpec>,
}

impl LeafAttrBlockBuilder {
    /// Create a builder for `specs` covering `slot_cap` slots.
    pub fn new(specs: &[LeafAttrSpec], slot_cap: usize) -> Self {
        let base = LeafAttrBlock::empty(specs, slot_cap);
        Self {
            slot_cap,
            columns: base.columns,
            specs: specs.to_vec(),
        }
    }

    /// Record one value. Returns `false` when the attribute is not covered or the value type
    /// does not match the declared type.
    pub fn set(&mut self, slot: usize, spec_idx: usize, value: &AttributeValue) -> bool {
        if slot >= self.slot_cap {
            return false;
        }
        let spec = self.specs[spec_idx];
        let col = &mut self.columns[spec_idx];
        debug_assert_eq!(col.attr_id, spec.attr_id);
        match (spec.attr_type, value) {
            (AttributeType::Bool, AttributeValue::Bool(v)) => {
                col.present[slot >> 6] |= 1u64 << (slot & 63);
                if let LeafAttrValues::Bool(words) = &mut col.values
                    && *v
                {
                    words[slot >> 6] |= 1u64 << (slot & 63);
                }
                true
            }
            (AttributeType::Int, AttributeValue::Int(v)) => {
                col.present[slot >> 6] |= 1u64 << (slot & 63);
                // Values are materialized in `finish` so the width can be chosen from the
                // full leaf range.
                self.set_i64(spec_idx, slot, *v);
                true
            }
            (AttributeType::Uint, AttributeValue::Uint(v)) => {
                col.present[slot >> 6] |= 1u64 << (slot & 63);
                self.set_u64(spec_idx, slot, *v);
                true
            }
            (AttributeType::Float, AttributeValue::Float(v)) => {
                col.present[slot >> 6] |= 1u64 << (slot & 63);
                if let LeafAttrValues::F64(values) = &mut col.values {
                    values[slot] = *v;
                }
                true
            }
            _ => false,
        }
    }

    /// Copy all supported attribute values from `view` into `slot`.
    pub fn set_from_row_view(&mut self, slot: usize, view: &crate::codec::RowView<'_>) -> Result<()> {
        for i in 0..view.len() {
            let (attr_id, _, _, _) = view.dir_entry(i);
            let Some(spec_idx) = self
                .specs
                .binary_search_by_key(&attr_id, |s| s.attr_id)
                .ok()
            else {
                continue;
            };
            if let Some(value) = view.decode_field(attr_id)? {
                self.set(slot, spec_idx, &value);
            }
        }
        Ok(())
    }

    fn set_i64(&mut self, spec_idx: usize, slot: usize, value: i64) {
        if let LeafAttrValues::I64 { deltas, .. } = &mut self.columns[spec_idx].values {
            let need = (slot + 1) * 8;
            if deltas.len() < need {
                deltas.resize(need, 0);
            }
            deltas[slot * 8..(slot + 1) * 8].copy_from_slice(&value.to_le_bytes());
        }
    }

    fn set_u64(&mut self, spec_idx: usize, slot: usize, value: u64) {
        if let LeafAttrValues::U64 { deltas, .. } = &mut self.columns[spec_idx].values {
            let need = (slot + 1) * 8;
            if deltas.len() < need {
                deltas.resize(need, 0);
            }
            deltas[slot * 8..(slot + 1) * 8].copy_from_slice(&value.to_le_bytes());
        }
    }

    /// Finish compression and return the owned block.
    pub fn finish(mut self) -> LeafAttrBlock {
        for col in &mut self.columns {
            if let LeafAttrValues::F64(values) = &col.values {
                let lossless = (0..self.slot_cap).all(|slot| {
                    !is_present_mask(&col.present, slot)
                        || (values[slot] as f32 as f64 == values[slot])
                });
                if lossless {
                    let narrowed: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                    col.values = LeafAttrValues::F32(narrowed);
                    continue;
                }
            }
            match &mut col.values {
                LeafAttrValues::I64 {
                    base,
                    width,
                    deltas,
                } => {
                    // `deltas` currently holds raw i64 per slot when built via `set_from_row_view`.
                    compress_signed(col.present.as_slice(), self.slot_cap, base, width, deltas);
                }
                LeafAttrValues::U64 {
                    base,
                    width,
                    deltas,
                } => {
                    compress_unsigned(col.present.as_slice(), self.slot_cap, base, width, deltas);
                }
                _ => {}
            }
        }
        LeafAttrBlock {
            slot_cap: self.slot_cap,
            columns: self.columns,
        }
    }
}

fn compress_signed(
    present: &[u64],
    slot_cap: usize,
    base: &mut i64,
    width: &mut u8,
    deltas: &mut Vec<u8>,
) {
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    let mut any = false;
    for slot in 0..slot_cap {
        if !is_present_mask(present, slot) {
            continue;
        }
        any = true;
        let v = i64::from_le_bytes(
            deltas[slot * 8..(slot + 1) * 8]
                .try_into()
                .expect("raw i64 slot"),
        );
        min = min.min(v);
        max = max.max(v);
    }
    if !any {
        *base = 0;
        *width = 0;
        deltas.clear();
        return;
    }
    *base = min;
    let span = (max.wrapping_sub(min)) as u64;
    *width = delta_width(span);
    let width_usize = *width as usize;
    let mut out = vec![0u8; slot_cap * width_usize];
    for slot in 0..slot_cap {
        if !is_present_mask(present, slot) {
            continue;
        }
        let v = i64::from_le_bytes(
            deltas[slot * 8..(slot + 1) * 8]
                .try_into()
                .expect("raw i64 slot"),
        );
        let d = (v.wrapping_sub(min)) as u64;
        write_delta(&mut out, width_usize, slot, d);
    }
    *deltas = out;
}

fn compress_unsigned(
    present: &[u64],
    slot_cap: usize,
    base: &mut u64,
    width: &mut u8,
    deltas: &mut Vec<u8>,
) {
    let mut min = u64::MAX;
    let mut max = 0u64;
    let mut any = false;
    for slot in 0..slot_cap {
        if !is_present_mask(present, slot) {
            continue;
        }
        any = true;
        let v = u64::from_le_bytes(
            deltas[slot * 8..(slot + 1) * 8]
                .try_into()
                .expect("raw u64 slot"),
        );
        min = min.min(v);
        max = max.max(v);
    }
    if !any {
        *base = 0;
        *width = 0;
        deltas.clear();
        return;
    }
    *base = min;
    *width = delta_width(max - min);
    let width_usize = *width as usize;
    let mut out = vec![0u8; slot_cap * width_usize];
    for slot in 0..slot_cap {
        if !is_present_mask(present, slot) {
            continue;
        }
        let v = u64::from_le_bytes(
            deltas[slot * 8..(slot + 1) * 8]
                .try_into()
                .expect("raw u64 slot"),
        );
        write_delta(&mut out, width_usize, slot, v - min);
    }
    *deltas = out;
}

#[inline]
fn is_present_mask(mask: &[u64], slot: usize) -> bool {
    let word = mask[slot >> 6];
    ((word >> (slot & 63)) & 1) != 0
}

#[inline]
fn delta_width(span: u64) -> u8 {
    if span <= u8::MAX as u64 {
        1
    } else if span <= u16::MAX as u64 {
        2
    } else if span <= u32::MAX as u64 {
        4
    } else {
        8
    }
}

#[inline]
fn write_delta(out: &mut [u8], width: usize, slot: usize, delta: u64) {
    if width == 0 {
        return;
    }
    let bytes = delta.to_le_bytes();
    out[slot * width..(slot + 1) * width].copy_from_slice(&bytes[..width]);
}

#[inline]
fn read_delta(deltas: &[u8], width: u8, slot: usize) -> u64 {
    match width {
        0 => 0,
        1 => deltas[slot] as u64,
        2 => u16::from_le_bytes([deltas[slot * 2], deltas[slot * 2 + 1]]) as u64,
        4 => u32::from_le_bytes(
            deltas[slot * 4..slot * 4 + 4]
                .try_into()
                .expect("checked 4-byte delta"),
        ) as u64,
        8 => u64::from_le_bytes(
            deltas[slot * 8..slot * 8 + 8]
                .try_into()
                .expect("checked 8-byte delta"),
        ),
        _ => 0,
    }
}

/// Conservative per-attribute zone summary used to prune tree children.
///
/// ```text
/// AttrSummary { attr_id: 1, attr_type: Int, present_count: 91, min: 3, max: 97, bits: 0x... }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttrSummary {
    /// Schema attribute id.
    pub attr_id: AttrId,
    /// Declared scalar type.
    pub attr_type: AttributeType,
    /// Number of rows with a value.
    pub present_count: u32,
    /// Whether any row has a value.
    pub has_values: bool,
    /// Minimum order-preserving encoded value.
    pub min: u64,
    /// Maximum order-preserving encoded value.
    pub max: u64,
    /// 64-bit hash bloom of values, used to reject equality probes.
    pub bits: u64,
}

/// Conservative summary of one leaf or internal subtree.
///
/// ```text
/// NodeSummary { row_count: 160, attrs: [score: [0.0, 12.5]], complete: true }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSummary {
    /// Number of rows covered by the subtree (live + tombstoned leaf slots).
    pub row_count: u32,
    /// Per-attribute summaries in ascending `attr_id` order.
    pub attrs: Vec<AttrSummary>,
    /// `true` when every child contributed a summary; unknown summaries never prune.
    pub complete: bool,
}

impl NodeSummary {
    /// Build a summary for a leaf attribute block.
    pub fn from_leaf(block: &LeafAttrView<'_>, row_count: u32) -> Self {
        let mut attrs = Vec::with_capacity(block.columns().len());
        for col in block.columns() {
            let mut present_count = 0u32;
            let mut min = u64::MAX;
            let mut max = 0u64;
            let mut bits = 0u64;
            for slot in 0..block.slot_cap().min(col.slot_cap()) {
                let Some(value) = col.value_at(slot) else {
                    continue;
                };
                present_count += 1;
                let enc = value.ordered_u64();
                min = min.min(enc);
                max = max.max(enc);
                bits |= 1u64 << (mix64(enc) & 63);
            }
            let has_values = present_count > 0;
            attrs.push(AttrSummary {
                attr_id: col.attr_id,
                attr_type: col.attr_type,
                present_count,
                has_values,
                min: if has_values { min } else { 0 },
                max: if has_values { max } else { 0 },
                bits,
            });
        }
        attrs.sort_unstable_by_key(|a| a.attr_id);
        Self {
            row_count,
            attrs,
            complete: true,
        }
    }

    /// Merge `child` into this summary.
    pub fn merge_child(&mut self, child: &NodeSummary) {
        self.row_count = self.row_count.saturating_add(child.row_count);
        self.complete &= child.complete;
        for child_attr in &child.attrs {
            match self
                .attrs
                .binary_search_by_key(&child_attr.attr_id, |a| a.attr_id)
            {
                Ok(idx) => {
                    let dst = &mut self.attrs[idx];
                    if child_attr.has_values {
                        if dst.has_values {
                            dst.min = dst.min.min(child_attr.min);
                            dst.max = dst.max.max(child_attr.max);
                        } else {
                            dst.min = child_attr.min;
                            dst.max = child_attr.max;
                            dst.has_values = true;
                        }
                        dst.bits |= child_attr.bits;
                    }
                    dst.present_count = dst.present_count.saturating_add(child_attr.present_count);
                }
                Err(idx) => self.attrs.insert(idx, *child_attr),
            }
        }
    }

    /// Number of rows in the subtree.
    #[inline]
    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    /// Look up an attribute summary.
    #[inline]
    pub fn attr(&self, attr_id: AttrId) -> Option<&AttrSummary> {
        self.attrs
            .binary_search_by_key(&attr_id, |a| a.attr_id)
            .ok()
            .map(|idx| &self.attrs[idx])
    }

    /// Encode this summary.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9 + self.attrs.len() * 28);
        buf.push(NODE_SUMMARY_VERSION);
        buf.push(u8::from(self.complete));
        buf.extend_from_slice(&self.row_count.to_le_bytes());
        buf.extend_from_slice(&(self.attrs.len() as u16).to_le_bytes());
        for attr in &self.attrs {
            buf.extend_from_slice(&attr.attr_id.to_le_bytes());
            buf.push(attr.attr_type.as_u8());
            buf.extend_from_slice(&attr.present_count.to_le_bytes());
            buf.push(u8::from(attr.has_values));
            buf.extend_from_slice(&attr.min.to_le_bytes());
            buf.extend_from_slice(&attr.max.to_le_bytes());
            buf.extend_from_slice(&attr.bits.to_le_bytes());
        }
        buf
    }

    /// Decode a summary.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(TreeError::CorruptedData(
                "NodeSummary header truncated".into(),
            ));
        }
        if bytes[0] != NODE_SUMMARY_VERSION {
            return Err(TreeError::CorruptedData(format!(
                "unsupported NodeSummary version: {}",
                bytes[0]
            )));
        }
        let complete = match bytes[1] {
            0 => false,
            1 => true,
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "invalid NodeSummary complete flag: {other}"
                )));
            }
        };
        let row_count = u32::from_le_bytes(bytes[2..6].try_into().unwrap());
        let count = u16::from_le_bytes(bytes[6..8].try_into().unwrap()) as usize;
        let mut off = 8;
        let mut attrs = Vec::with_capacity(count);
        for _ in 0..count {
            if off + 28 > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "NodeSummary attribute truncated".into(),
                ));
            }
            let attr_id = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
            let attr_type = AttributeType::from_u8(bytes[off + 2])?;
            let present_count =
                u32::from_le_bytes(bytes[off + 3..off + 7].try_into().unwrap());
            let has_values = match bytes[off + 7] {
                0 => false,
                1 => true,
                other => {
                    return Err(TreeError::CorruptedData(format!(
                        "invalid NodeSummary has_values flag: {other}"
                    )));
                }
            };
            let min = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
            let max = u64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap());
            let bits = u64::from_le_bytes(bytes[off + 24..off + 32].try_into().unwrap());
            // Attribute record is 32 bytes (2 id + 1 type + 4 present + 1 flag + 8 + 8 + 8).
            off += 32;
            attrs.push(AttrSummary {
                attr_id,
                attr_type,
                present_count,
                has_values,
                min,
                max,
                bits,
            });
        }
        if off != bytes.len() {
            return Err(TreeError::CorruptedData(
                "trailing bytes in NodeSummary".into(),
            ));
        }
        Ok(Self {
            row_count,
            attrs,
            complete,
        })
    }
}

/// 64-bit avalanche hash used for the summary equality bloom.
#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_attr_roundtrip_and_summary() {
        let specs = [
            LeafAttrSpec {
                attr_id: 0,
                attr_type: AttributeType::Float,
            },
            LeafAttrSpec {
                attr_id: 1,
                attr_type: AttributeType::Int,
            },
            LeafAttrSpec {
                attr_id: 2,
                attr_type: AttributeType::Bool,
            },
        ];
        let mut builder = LeafAttrBlockBuilder::new(&specs, 6);
        builder.set(0, 0, &AttributeValue::Float(1.5));
        builder.set(3, 0, &AttributeValue::Float(4.5));
        builder.set(0, 1, &AttributeValue::Int(-7));
        builder.set(1, 1, &AttributeValue::Int(900));
        builder.set(5, 1, &AttributeValue::Int(3));
        builder.set(2, 2, &AttributeValue::Bool(true));
        let block = builder.finish();
        let bytes = block.encode();
        let decoded = LeafAttrBlock::decode(&bytes).unwrap();
        assert_eq!(decoded.slot_cap, 6);
        let float_col = decoded.column(0).unwrap();
        assert_eq!(float_col.value_at(0), Some(LeafAttrValue::Float(1.5)));
        assert_eq!(float_col.value_at(1), None);
        let int_col = decoded.column(1).unwrap();
        assert_eq!(int_col.value_at(0), Some(LeafAttrValue::Int(-7)));
        assert_eq!(int_col.value_at(1), Some(LeafAttrValue::Int(900)));
        assert_eq!(int_col.value_at(5), Some(LeafAttrValue::Int(3)));
        let bool_col = decoded.column(2).unwrap();
        assert_eq!(bool_col.value_at(2), Some(LeafAttrValue::Bool(true)));
        assert_eq!(bool_col.value_at(3), None);

        let view = LeafAttrView::parse(&bytes).unwrap();
        let summary = NodeSummary::from_leaf(&view, 6);
        assert_eq!(summary.row_count, 6);
        assert_eq!(summary.attr(0).unwrap().present_count, 2);
        assert_eq!(summary.attr(1).unwrap().present_count, 3);
        assert_eq!(summary.attr(2).unwrap().present_count, 1);
        let decoded_summary = NodeSummary::decode(&summary.encode()).unwrap();
        assert_eq!(decoded_summary, summary);
    }
}
