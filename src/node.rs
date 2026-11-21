use std::borrow::Cow;

use crate::config::{DistanceMetric, NodeId};
use crate::error::{Result, TreeError};

const META_LEN: usize = 30;
const HEADER_PREFIX_LEN: usize = 22;

/// Global tree metadata persisted in `cf_meta`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeMeta {
    pub root_id: NodeId,
    pub root_level: u8,
    pub next_node_id: NodeId,
    pub dim: u32,
    pub metric: DistanceMetric,
    pub num_vectors: u64,
}

impl TreeMeta {
    pub fn encode(&self) -> [u8; META_LEN] {
        let mut buf = [0u8; META_LEN];
        buf[0..8].copy_from_slice(&self.root_id.to_le_bytes());
        buf[8] = self.root_level;
        buf[9..17].copy_from_slice(&self.next_node_id.to_le_bytes());
        buf[17..21].copy_from_slice(&self.dim.to_le_bytes());
        buf[21] = self.metric.as_u8();
        buf[22..30].copy_from_slice(&self.num_vectors.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != META_LEN {
            return Err(TreeError::CorruptedData(format!(
                "TreeMeta expected {META_LEN} bytes, got {}",
                bytes.len()
            )));
        }
        Ok(Self {
            root_id: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            root_level: bytes[8],
            next_node_id: u64::from_le_bytes(bytes[9..17].try_into().unwrap()),
            dim: u32::from_le_bytes(bytes[17..21].try_into().unwrap()),
            metric: DistanceMetric::from_u8(bytes[21])?,
            num_vectors: u64::from_le_bytes(bytes[22..30].try_into().unwrap()),
        })
    }
}

/// Per-node metadata persisted in `cf_node_header`.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeHeader {
    pub level: u8,
    pub parent_id: Option<NodeId>,
    pub entry_count: u32,
    pub subtree_count: u64,
    pub centroid: Vec<f32>,
}

impl NodeHeader {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_PREFIX_LEN + self.centroid.len() * 4);
        buf.push(self.level);
        match self.parent_id {
            Some(pid) => {
                buf.push(1);
                buf.extend_from_slice(&pid.to_le_bytes());
            }
            None => {
                buf.push(0);
                buf.extend_from_slice(&0u64.to_le_bytes());
            }
        }
        buf.extend_from_slice(&self.entry_count.to_le_bytes());
        buf.extend_from_slice(&self.subtree_count.to_le_bytes());
        write_f32_le(&self.centroid, &mut buf);
        buf
    }

    pub fn decode(bytes: &[u8], dim: usize) -> Result<Self> {
        let expected = HEADER_PREFIX_LEN + dim * 4;
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "NodeHeader expected {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let level = bytes[0];
        let has_parent = bytes[1];
        let parent_raw = u64::from_le_bytes(bytes[2..10].try_into().unwrap());
        let parent_id = match has_parent {
            0 => None,
            1 => Some(parent_raw),
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "invalid has_parent flag in NodeHeader: {other}"
                )));
            }
        };
        let entry_count = u32::from_le_bytes(bytes[10..14].try_into().unwrap());
        let subtree_count = u64::from_le_bytes(bytes[14..22].try_into().unwrap());
        let centroid = read_f32_le(&bytes[HEADER_PREFIX_LEN..], dim)?;
        Ok(Self {
            level,
            parent_id,
            entry_count,
            subtree_count,
            centroid,
        })
    }
}

/// Child entry stored in `cf_internal_entries`.
#[derive(Debug, Clone, PartialEq)]
pub struct InternalEntry {
    pub subtree_count: u64,
    pub centroid: Vec<f32>,
}

impl InternalEntry {
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.centroid.len() * 4);
        buf.extend_from_slice(&self.subtree_count.to_le_bytes());
        write_f32_le(&self.centroid, &mut buf);
        buf
    }

    pub fn decode(bytes: &[u8], dim: usize) -> Result<Self> {
        let expected = 8 + dim * 4;
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "InternalEntry expected {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let subtree_count = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let centroid = read_f32_le(&bytes[8..], dim)?;
        Ok(Self {
            subtree_count,
            centroid,
        })
    }
}

/// Encode a slice of `f32` as little-endian bytes.
pub fn encode_f32_slice(slice: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(slice.len() * 4);
    write_f32_le(slice, &mut buf);
    buf
}

/// Decode `dim` little-endian `f32` values from `bytes`.
pub fn decode_f32_slice(bytes: &[u8], dim: usize) -> Result<Vec<f32>> {
    read_f32_le(bytes, dim)
}

/// Borrow `bytes` directly as `&[f32]` on little-endian hosts when 4-byte aligned,
/// falling back to decoding into an owned `Vec<f32>`.
#[inline]
pub fn borrow_f32_slice<'a>(bytes: &'a [u8], dim: usize) -> Result<Cow<'a, [f32]>> {
    let expected = dim * 4;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "vector payload expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        return Ok(Cow::Borrowed(slice));
    }
    Ok(Cow::Owned(read_f32_le(bytes, dim)?))
}

#[inline]
fn write_f32_le(slice: &[f32], out: &mut Vec<u8>) {
    #[cfg(target_endian = "little")]
    {
        out.extend_from_slice(bytemuck::cast_slice(slice));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &v in slice {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

#[inline]
fn read_f32_le(bytes: &[u8], dim: usize) -> Result<Vec<f32>> {
    let expected = dim * 4;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "f32 slice expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        return Ok(slice.to_vec());
    }
    let mut out = Vec::with_capacity(dim);
    let (chunks, _) = bytes.as_chunks::<4>();
    for &chunk in chunks {
        out.push(f32::from_le_bytes(chunk));
    }
    Ok(out)
}
