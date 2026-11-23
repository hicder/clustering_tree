
use izzydb_core::config::{
    ColumnGen, LocalId, RowId, SegmentId,
};
use izzydb_core::error::{Result, TreeError};
use super::*;

// ---------------------------------------------------------------------------
// Segment metadata, Manifest, ZoneEntry, DetachedMeta
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SegmentState {
    Open = 0,
    Sealed = 1,
    Building = 2,
    Forwarding = 3,
    Retired = 4,
}

impl SegmentState {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::Open),
            1 => Ok(Self::Sealed),
            2 => Ok(Self::Building),
            3 => Ok(Self::Forwarding),
            4 => Ok(Self::Retired),
            other => Err(TreeError::CorruptedData(format!(
                "unknown SegmentState byte: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentMeta {
    pub state: SegmentState,
    pub flags: u8,
    pub row_count: u32,
    pub deleted_count: u32,
    pub forward_target: SegmentId,
    pub forward_base: LocalId,
    pub forwarded_at_epoch: u64,
    pub created_seq: u64,
    pub sealed_seq: u64,
}

impl SegmentMeta {
    pub fn new_open(created_seq: u64) -> Self {
        Self {
            state: SegmentState::Open,
            flags: 0,
            row_count: 0,
            deleted_count: 0,
            forward_target: 0,
            forward_base: 0,
            forwarded_at_epoch: 0,
            created_seq,
            sealed_seq: 0,
        }
    }

    pub fn encode(&self) -> [u8; SEGMENT_META_BYTES] {
        let mut buf = [0u8; SEGMENT_META_BYTES];
        buf[0] = 1; // version
        buf[1] = self.state as u8;
        buf[2] = self.flags;
        buf[3] = 0;
        buf[4..8].copy_from_slice(&self.row_count.to_le_bytes());
        buf[8..12].copy_from_slice(&self.deleted_count.to_le_bytes());
        buf[12..16].copy_from_slice(&self.forward_target.to_le_bytes());
        buf[16..20].copy_from_slice(&self.forward_base.to_le_bytes());
        buf[20..28].copy_from_slice(&self.forwarded_at_epoch.to_le_bytes());
        buf[28..36].copy_from_slice(&self.created_seq.to_le_bytes());
        buf[36..44].copy_from_slice(&self.sealed_seq.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != SEGMENT_META_BYTES {
            return Err(TreeError::CorruptedData(format!(
                "SegmentMeta expected {SEGMENT_META_BYTES} bytes, got {}",
                bytes.len()
            )));
        }
        if bytes[0] != 1 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported SegmentMeta version: {}",
                bytes[0]
            )));
        }
        if bytes[3] != 0 {
            return Err(TreeError::CorruptedData(format!(
                "invalid SegmentMeta reserved byte: {}",
                bytes[3]
            )));
        }
        let state = SegmentState::from_u8(bytes[1])?;
        let flags = bytes[2];
        let row_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let deleted_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if deleted_count > row_count {
            return Err(TreeError::CorruptedData(format!(
                "SegmentMeta deleted_count ({deleted_count}) > row_count ({row_count})"
            )));
        }
        Ok(Self {
            state,
            flags,
            row_count,
            deleted_count,
            forward_target: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            forward_base: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            forwarded_at_epoch: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
            created_seq: u64::from_le_bytes(bytes[28..36].try_into().unwrap()),
            sealed_seq: u64::from_le_bytes(bytes[36..44].try_into().unwrap()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub open_segment: Option<SegmentId>,
    pub next_segment_id: SegmentId,
    pub seal_rows: u32,
    pub live_rows: u64,
    pub total_rows: u64,
}

impl Manifest {
    pub fn new(seal_rows: u32) -> Self {
        Self {
            open_segment: None,
            next_segment_id: 0,
            seal_rows,
            live_rows: 0,
            total_rows: 0,
        }
    }

    pub fn encode(&self) -> [u8; MANIFEST_BYTES] {
        let mut buf = [0u8; MANIFEST_BYTES];
        buf[0] = 1; // version
        match self.open_segment {
            Some(seg) => {
                buf[1] = 1;
                buf[8..12].copy_from_slice(&seg.to_le_bytes());
            }
            None => {
                buf[1] = 0;
            }
        }
        buf[4..8].copy_from_slice(&self.next_segment_id.to_le_bytes());
        buf[12..16].copy_from_slice(&self.seal_rows.to_le_bytes());
        buf[16..24].copy_from_slice(&self.live_rows.to_le_bytes());
        buf[24..32].copy_from_slice(&self.total_rows.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != MANIFEST_BYTES {
            return Err(TreeError::CorruptedData(format!(
                "Manifest expected {MANIFEST_BYTES} bytes, got {}",
                bytes.len()
            )));
        }
        if bytes[0] != 1 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported Manifest version: {}",
                bytes[0]
            )));
        }
        if bytes[2] != 0 || bytes[3] != 0 {
            return Err(TreeError::CorruptedData(
                "invalid Manifest reserved bytes".into(),
            ));
        }
        let open_raw = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let open_segment = match bytes[1] {
            0 => None,
            1 => Some(open_raw),
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "invalid Manifest has_open byte: {other}"
                )));
            }
        };
        let next_segment_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let seal_rows = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let live_rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let total_rows = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        if live_rows > total_rows {
            return Err(TreeError::CorruptedData(format!(
                "Manifest live_rows ({live_rows}) > total_rows ({total_rows})"
            )));
        }
        Ok(Self {
            open_segment,
            next_segment_id,
            seal_rows,
            live_rows,
            total_rows,
        })
    }
}

/// Per-segment min/max ordered scalar bounds stored in `default[0x12 || seg || attr || tag]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneEntry {
    pub min: Vec<u8>,
    pub max: Vec<u8>,
}

impl ZoneEntry {
    pub fn singleton(val: &[u8]) -> Self {
        Self {
            min: val.to_vec(),
            max: val.to_vec(),
        }
    }

    /// Widen bounds to include `val`; returns `true` if `min` or `max` changed.
    pub fn widen(&mut self, val: &[u8]) -> bool {
        let mut changed = false;
        if val < self.min.as_slice() {
            self.min.clear();
            self.min.extend_from_slice(val);
            changed = true;
        }
        if val > self.max.as_slice() {
            self.max.clear();
            self.max.extend_from_slice(val);
            changed = true;
        }
        changed
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + self.min.len() + self.max.len());
        buf.extend_from_slice(&(self.min.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(self.max.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.min);
        buf.extend_from_slice(&self.max);
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(TreeError::CorruptedData(format!(
                "ZoneEntry too short: {}",
                bytes.len()
            )));
        }
        let min_len = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        let max_len = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        let expected = 4usize
            .checked_add(min_len)
            .and_then(|x| x.checked_add(max_len))
            .ok_or_else(|| TreeError::CorruptedData("ZoneEntry length overflow".into()))?;
        if bytes.len() != expected || min_len == 0 || max_len == 0 {
            return Err(TreeError::CorruptedData(format!(
                "ZoneEntry expected {expected} bytes (non-empty bounds), got {}",
                bytes.len()
            )));
        }
        let min = bytes[4..4 + min_len].to_vec();
        let max = bytes[4 + min_len..].to_vec();
        if min > max {
            return Err(TreeError::CorruptedData(
                "ZoneEntry min > max".into(),
            ));
        }
        Ok(Self { min, max })
    }
}

/// Per-detached-column metadata stored in `default[0x13 || attr_id BE]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetachedMeta {
    pub state: u8,
    pub current_gen: ColumnGen,
    pub pending_gen: ColumnGen,
    pub retired_below: ColumnGen,
    pub built_against_watermark: RowId,
}

impl Default for DetachedMeta {
    fn default() -> Self {
        Self {
            state: 0,
            current_gen: 0,
            pending_gen: 0,
            retired_below: 0,
            built_against_watermark: RowId(0),
        }
    }
}

impl DetachedMeta {
    pub fn encode(&self) -> [u8; DETACHED_META_BYTES] {
        let mut buf = [0u8; DETACHED_META_BYTES];
        buf[0] = 1; // version
        buf[1] = self.state;
        buf[2..4].copy_from_slice(&0u16.to_le_bytes());
        buf[4..8].copy_from_slice(&self.current_gen.to_le_bytes());
        buf[8..12].copy_from_slice(&self.pending_gen.to_le_bytes());
        buf[12..16].copy_from_slice(&self.retired_below.to_le_bytes());
        buf[16..24].copy_from_slice(&self.built_against_watermark.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != DETACHED_META_BYTES {
            return Err(TreeError::CorruptedData(format!(
                "DetachedMeta expected {DETACHED_META_BYTES} bytes, got {}",
                bytes.len()
            )));
        }
        if bytes[0] != 1 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported DetachedMeta version: {}",
                bytes[0]
            )));
        }
        if bytes[2] != 0 || bytes[3] != 0 {
            return Err(TreeError::CorruptedData(
                "invalid DetachedMeta reserved bytes".into(),
            ));
        }
        Ok(Self {
            state: bytes[1],
            current_gen: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            pending_gen: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            retired_below: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            built_against_watermark: RowId::from_le_bytes(bytes[16..24].try_into().unwrap()),
        })
    }
}

/// Stub for 256-row columnar block codec (implemented in Phase 7).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColumnBlock;

