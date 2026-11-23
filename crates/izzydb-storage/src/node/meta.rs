//! Global ANN tree metadata.

use izzydb_core::config::{DistanceMetric, NodeId};
use izzydb_core::error::{Result, TreeError};

use super::*;

/// Global ANN tree metadata persisted in `default[\x01ann_meta]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnMeta {
    pub root_id: NodeId,
    pub root_level: u8,
    pub next_node_id: NodeId,
    pub dim: u32,
    pub metric: DistanceMetric,
    pub num_vectors: u64,
    pub rabitq_seed: u64,
    pub rabitq_bits: u8,
    pub stale_rows: u64,
    pub vacuum_epoch: u64,
    pub vacuum_cursor: NodeId,
}

/// Backward-compatible type alias for [`AnnMeta`].
pub type TreeMeta = AnnMeta;

impl AnnMeta {
    pub fn encode(&self) -> [u8; META_LEN] {
        let mut buf = [0u8; META_LEN];
        buf[0..4].copy_from_slice(&self.root_id.to_le_bytes());
        buf[4] = self.root_level;
        buf[5..9].copy_from_slice(&self.next_node_id.to_le_bytes());
        buf[9..13].copy_from_slice(&self.dim.to_le_bytes());
        buf[13] = self.metric.as_u8();
        buf[14..22].copy_from_slice(&self.num_vectors.to_le_bytes());
        buf[22..30].copy_from_slice(&self.rabitq_seed.to_le_bytes());
        buf[30] = self.rabitq_bits;
        buf[31..39].copy_from_slice(&self.stale_rows.to_le_bytes());
        buf[39..47].copy_from_slice(&self.vacuum_epoch.to_le_bytes());
        buf[47..51].copy_from_slice(&self.vacuum_cursor.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != META_LEN {
            return Err(TreeError::CorruptedData(format!(
                "AnnMeta expected {META_LEN} bytes, got {}",
                bytes.len()
            )));
        }
        Ok(Self {
            root_id: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            root_level: bytes[4],
            next_node_id: u32::from_le_bytes(bytes[5..9].try_into().unwrap()),
            dim: u32::from_le_bytes(bytes[9..13].try_into().unwrap()),
            metric: DistanceMetric::from_u8(bytes[13])?,
            num_vectors: u64::from_le_bytes(bytes[14..22].try_into().unwrap()),
            rabitq_seed: u64::from_le_bytes(bytes[22..30].try_into().unwrap()),
            rabitq_bits: bytes[30],
            stale_rows: u64::from_le_bytes(bytes[31..39].try_into().unwrap()),
            vacuum_epoch: u64::from_le_bytes(bytes[39..47].try_into().unwrap()),
            vacuum_cursor: u32::from_le_bytes(bytes[47..51].try_into().unwrap()),
        })
    }
}
