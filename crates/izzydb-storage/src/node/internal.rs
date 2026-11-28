//! Packed internal ANN nodes.

use izzydb_core::config::{DistanceMetric, NodeId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::metric::compute_weighted_centroid_flat;

use super::*;

/// Packed internal tree node stored as a single blob in `cf_internal_nodes` and cached in memory.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedInternalNode {
    pub level: u8,
    pub parent_id: Option<NodeId>,
    pub subtree_count: u64,
    pub centroid: Vec<f32>,
    pub child_ids: Vec<NodeId>,
    pub child_subtree_counts: Vec<u64>,
    pub child_centroids: Vec<f32>,
}

impl PackedInternalNode {
    /// Construct a [`PackedInternalNode`] from `(child_id, subtree_count, centroid)` entries,
    /// sorting children by `NodeId` and computing `subtree_count` and weighted `centroid`.
    pub fn from_children(
        level: u8,
        parent_id: Option<NodeId>,
        children: &[(NodeId, u64, &[f32])],
        dim: usize,
        metric: DistanceMetric,
    ) -> Self {
        let mut sorted: Vec<(NodeId, u64, &[f32])> = children.to_vec();
        sorted.sort_unstable_by_key(|&(id, _, _)| id);

        let n = sorted.len();
        let mut child_ids = Vec::with_capacity(n);
        let mut child_subtree_counts = Vec::with_capacity(n);
        let mut child_centroids = Vec::with_capacity(n * dim);
        let mut subtree_count = 0u64;

        for (cid, sub, cent) in sorted {
            debug_assert_eq!(cent.len(), dim);
            child_ids.push(cid);
            child_subtree_counts.push(sub);
            child_centroids.extend_from_slice(cent);
            subtree_count += sub;
        }

        let centroid = compute_weighted_centroid_flat(
            &child_centroids,
            &child_subtree_counts,
            dim,
            metric,
        );

        Self {
            level,
            parent_id,
            subtree_count,
            centroid,
            child_ids,
            child_subtree_counts,
            child_centroids,
        }
    }

    pub fn entry_count(&self) -> usize {
        self.child_ids.len()
    }

    #[inline]
    pub fn child_centroid(&self, idx: usize, dim: usize) -> &[f32] {
        &self.child_centroids[idx * dim..(idx + 1) * dim]
    }

    pub fn upsert_child(
        &mut self,
        child_id: NodeId,
        subtree_count: u64,
        centroid: &[f32],
        dim: usize,
    ) {
        debug_assert_eq!(centroid.len(), dim);
        match self.child_ids.binary_search(&child_id) {
            Ok(idx) => {
                self.child_subtree_counts[idx] = subtree_count;
                self.child_centroids[idx * dim..(idx + 1) * dim].copy_from_slice(centroid);
            }
            Err(idx) => {
                self.child_ids.insert(idx, child_id);
                self.child_subtree_counts.insert(idx, subtree_count);
                self.child_centroids
                    .splice(idx * dim..idx * dim, centroid.iter().copied());
            }
        }
    }

    pub fn remove_child(&mut self, child_id: NodeId, dim: usize) -> Option<(u64, Vec<f32>)> {
        let idx = self.child_ids.binary_search(&child_id).ok()?;
        self.child_ids.remove(idx);
        let sub = self.child_subtree_counts.remove(idx);
        let centroid = self
            .child_centroids
            .drain(idx * dim..(idx + 1) * dim)
            .collect();
        Some((sub, centroid))
    }

    pub fn recompute_stats(&mut self, dim: usize, metric: DistanceMetric) {
        self.subtree_count = self.child_subtree_counts.iter().sum();
        self.centroid = compute_weighted_centroid_flat(
            &self.child_centroids,
            &self.child_subtree_counts,
            dim,
            metric,
        );
    }

    pub fn encode(&self, dim: usize) -> Vec<u8> {
        let n = self.child_ids.len();
        debug_assert_eq!(self.centroid.len(), dim);
        debug_assert_eq!(self.child_subtree_counts.len(), n);
        debug_assert_eq!(self.child_centroids.len(), n * dim);

        let total = INTERNAL_HEADER_LEN + n * 12 + (1 + n) * dim * 4;
        let mut buf = Vec::with_capacity(total);
        buf.push(self.level);
        match self.parent_id {
            Some(pid) => {
                buf.push(1);
                buf.extend_from_slice(&0u16.to_le_bytes());
                buf.extend_from_slice(&(n as u32).to_le_bytes());
                buf.extend_from_slice(&pid.to_le_bytes());
                buf.extend_from_slice(&[INTERNAL_FORMAT_VERSION, 0, 0, 0]);
            }
            None => {
                buf.push(0);
                buf.extend_from_slice(&0u16.to_le_bytes());
                buf.extend_from_slice(&(n as u32).to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
                buf.extend_from_slice(&[INTERNAL_FORMAT_VERSION, 0, 0, 0]);
            }
        }
        buf.extend_from_slice(&self.subtree_count.to_le_bytes());
        write_u64_le(&self.child_subtree_counts, &mut buf);
        write_u32_le(&self.child_ids, &mut buf);
        write_f32_le(&self.centroid, &mut buf);
        write_f32_le(&self.child_centroids, &mut buf);
        buf
    }

    pub fn decode(bytes: &[u8], dim: usize) -> Result<Self> {
        if bytes.len() < INTERNAL_HEADER_LEN {
            return Err(TreeError::CorruptedData(format!(
                "PackedInternalNode header too short: {}",
                bytes.len()
            )));
        }
        let level = bytes[0];
        let parent_id = decode_parent_flag(bytes[1], &bytes[8..12])?;
        let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let subtree_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        if bytes[12] != INTERNAL_FORMAT_VERSION {
            return Err(TreeError::CorruptedData(format!(
                "unsupported PackedInternalNode format version {} (expected {INTERNAL_FORMAT_VERSION}); recreate the collection",
                bytes[12]
            )));
        }
        if bytes[13] != 0 || bytes[14] != 0 || bytes[15] != 0 {
            return Err(TreeError::CorruptedData(
                "invalid PackedInternalNode reserved bytes".into(),
            ));
        }

        let expected = INTERNAL_HEADER_LEN + n * 12 + (1 + n) * dim * 4;
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "PackedInternalNode expected {expected} bytes, got {}",
                bytes.len()
            )));
        }

        let mut off = INTERNAL_HEADER_LEN;
        let child_subtree_counts = read_u64_le(&bytes[off..off + n * 8], n)?;
        off += n * 8;
        let child_ids = read_u32_le(&bytes[off..off + n * 4], n)?;
        off += n * 4;
        let centroid = read_f32_le(&bytes[off..off + dim * 4], dim)?;
        off += dim * 4;
        let child_centroids = read_f32_le(&bytes[off..off + n * dim * 4], n * dim)?;

        Ok(Self {
            level,
            parent_id,
            subtree_count,
            centroid,
            child_ids,
            child_subtree_counts,
            child_centroids,
        })
    }
}
