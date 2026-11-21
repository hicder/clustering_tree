use std::borrow::Cow;

use izzydb_core::config::{DistanceMetric, DocId, NodeId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::metric::compute_weighted_centroid_flat;
use izzydb_rabitq::{EncodeScratch, QuantizedEntry, RabitqQuantizer};

const META_LEN: usize = 39;
const INTERNAL_HEADER_LEN: usize = 24;
const LEAF_HEADER_LEN: usize = 24;

/// Global tree metadata persisted in `cf_meta`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeMeta {
    pub root_id: NodeId,
    pub root_level: u8,
    pub next_node_id: NodeId,
    pub dim: u32,
    pub metric: DistanceMetric,
    pub num_vectors: u64,
    pub rabitq_seed: u64,
    pub rabitq_bits: u8,
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
        buf[30..38].copy_from_slice(&self.rabitq_seed.to_le_bytes());
        buf[38] = self.rabitq_bits;
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
            rabitq_seed: u64::from_le_bytes(bytes[30..38].try_into().unwrap()),
            rabitq_bits: bytes[38],
        })
    }
}

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

        let total = INTERNAL_HEADER_LEN + n * 16 + (1 + n) * dim * 4;
        let mut buf = Vec::with_capacity(total);
        buf.push(self.level);
        match self.parent_id {
            Some(pid) => {
                buf.push(1);
                buf.extend_from_slice(&0u16.to_le_bytes());
                buf.extend_from_slice(&(n as u32).to_le_bytes());
                buf.extend_from_slice(&pid.to_le_bytes());
            }
            None => {
                buf.push(0);
                buf.extend_from_slice(&0u16.to_le_bytes());
                buf.extend_from_slice(&(n as u32).to_le_bytes());
                buf.extend_from_slice(&0u64.to_le_bytes());
            }
        }
        buf.extend_from_slice(&self.subtree_count.to_le_bytes());
        write_u64_le(&self.child_ids, &mut buf);
        write_u64_le(&self.child_subtree_counts, &mut buf);
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
        let parent_id = decode_parent_flag(bytes[1], &bytes[8..16])?;
        let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let subtree_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

        let expected = INTERNAL_HEADER_LEN + n * 16 + (1 + n) * dim * 4;
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "PackedInternalNode expected {expected} bytes, got {}",
                bytes.len()
            )));
        }

        let mut off = INTERNAL_HEADER_LEN;
        let child_ids = read_u64_le(&bytes[off..off + n * 8], n)?;
        off += n * 8;
        let child_subtree_counts = read_u64_le(&bytes[off..off + n * 8], n)?;
        off += n * 8;
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

/// Packed leaf node stored as a Struct-of-Arrays (SoA) blob in `cf_leaf_entries`.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedLeafNode {
    pub parent_id: Option<NodeId>,
    pub centroid: Vec<f32>,
    pub rotated_centroid: Vec<f32>,
    pub doc_ids: Vec<DocId>,
    pub res_norm_sqs: Vec<f32>,
    pub scales: Vec<f32>,
    /// Word-major RaBitQ codes: `codes[w * n + i]` is word `w` of entry `i`.
    pub codes: Vec<u64>,
}

impl PackedLeafNode {
    pub fn empty(parent_id: Option<NodeId>, dim: usize, padded_dim: usize) -> Self {
        Self {
            parent_id,
            centroid: vec![0.0; dim],
            rotated_centroid: vec![0.0; padded_dim],
            doc_ids: Vec::new(),
            res_norm_sqs: Vec::new(),
            scales: Vec::new(),
            codes: Vec::new(),
        }
    }

    pub fn entry_count(&self) -> usize {
        self.doc_ids.len()
    }

    pub fn upsert_entry(
        &mut self,
        doc_id: DocId,
        res_norm_sq: f32,
        scale: f32,
        code: &[u64],
        rabitq_words: usize,
    ) {
        debug_assert_eq!(code.len(), rabitq_words);
        let n = self.doc_ids.len();
        match self.doc_ids.binary_search(&doc_id) {
            Ok(idx) => {
                self.res_norm_sqs[idx] = res_norm_sq;
                self.scales[idx] = scale;
                for (w, &word) in code.iter().enumerate() {
                    self.codes[w * n + idx] = word;
                }
            }
            Err(idx) => {
                self.doc_ids.insert(idx, doc_id);
                self.res_norm_sqs.insert(idx, res_norm_sq);
                self.scales.insert(idx, scale);

                let new_n = n + 1;
                self.codes.resize(new_n * rabitq_words, 0);
                for w in (0..rabitq_words).rev() {
                    let old_base = w * n;
                    let new_base = w * new_n;
                    if idx < n {
                        self.codes
                            .copy_within(old_base + idx..old_base + n, new_base + idx + 1);
                    }
                    if idx > 0 && new_base != old_base {
                        self.codes.copy_within(old_base..old_base + idx, new_base);
                    }
                    self.codes[new_base + idx] = code[w];
                }
            }
        }
    }

    pub fn remove_entry(&mut self, doc_id: DocId, rabitq_words: usize) -> bool {
        let Ok(idx) = self.doc_ids.binary_search(&doc_id) else {
            return false;
        };
        let n = self.doc_ids.len();
        let new_n = n - 1;
        self.doc_ids.remove(idx);
        self.res_norm_sqs.remove(idx);
        self.scales.remove(idx);

        for w in 0..rabitq_words {
            let old_base = w * n;
            let new_base = w * new_n;
            if idx > 0 && new_base != old_base {
                self.codes.copy_within(old_base..old_base + idx, new_base);
            }
            if idx + 1 < n {
                self.codes
                    .copy_within(old_base + idx + 1..old_base + n, new_base + idx);
            }
        }
        self.codes.truncate(new_n * rabitq_words);
        true
    }

    /// Re-encode all entries from `flat_vectors` (aligned with `doc_ids`) relative to
    /// `self.centroid` and refresh `self.rotated_centroid`.
    pub fn re_encode_all(
        &mut self,
        doc_ids: &[DocId],
        flat_vectors: &[f32],
        quantizer: &RabitqQuantizer,
        dim: usize,
        scratch: &mut EncodeScratch,
    ) {
        debug_assert_eq!(self.centroid.len(), dim);
        assert_eq!(flat_vectors.len(), doc_ids.len() * dim);
        let n = doc_ids.len();
        let words = quantizer.num_words();
        quantizer.rotate_into(&self.centroid, &mut self.rotated_centroid);

        self.doc_ids.clear();
        self.doc_ids.extend_from_slice(doc_ids);
        self.res_norm_sqs.resize(n, 0.0);
        self.scales.resize(n, 0.0);
        self.codes.resize(n * words, 0);

        let mut code_buf = vec![0u64; words];
        for (i, vec) in flat_vectors.chunks_exact(dim).enumerate() {
            let (res_norm_sq, scale) =
                quantizer.encode_into(vec, &self.centroid, scratch, &mut code_buf);
            self.res_norm_sqs[i] = res_norm_sq;
            self.scales[i] = scale;
            for (w, &word) in code_buf.iter().enumerate() {
                self.codes[w * n + i] = word;
            }
        }
    }

    pub fn entry_code_word(&self, idx: usize, word: usize) -> u64 {
        let n = self.doc_ids.len();
        self.codes[word * n + idx]
    }

    pub fn encode(&self, dim: usize, padded_dim: usize, rabitq_words: usize) -> Vec<u8> {
        let n = self.doc_ids.len();
        debug_assert_eq!(self.centroid.len(), dim);
        debug_assert_eq!(self.rotated_centroid.len(), padded_dim);
        debug_assert_eq!(self.res_norm_sqs.len(), n);
        debug_assert_eq!(self.scales.len(), n);
        debug_assert_eq!(self.codes.len(), n * rabitq_words);

        let total =
            LEAF_HEADER_LEN + (dim + padded_dim) * 4 + n * (16 + rabitq_words * 8);
        let mut buf = Vec::with_capacity(total);
        match self.parent_id {
            Some(pid) => {
                buf.push(1);
                buf.extend_from_slice(&[0u8; 3]);
                buf.extend_from_slice(&(n as u32).to_le_bytes());
                buf.extend_from_slice(&pid.to_le_bytes());
            }
            None => {
                buf.push(0);
                buf.extend_from_slice(&[0u8; 3]);
                buf.extend_from_slice(&(n as u32).to_le_bytes());
                buf.extend_from_slice(&0u64.to_le_bytes());
            }
        }
        buf.extend_from_slice(&0u64.to_le_bytes());
        write_u64_le(&self.doc_ids, &mut buf);
        write_u64_le(&self.codes, &mut buf);
        write_f32_le(&self.centroid, &mut buf);
        write_f32_le(&self.rotated_centroid, &mut buf);
        write_f32_le(&self.res_norm_sqs, &mut buf);
        write_f32_le(&self.scales, &mut buf);
        buf
    }

    pub fn decode(bytes: &[u8], dim: usize, padded_dim: usize, rabitq_words: usize) -> Result<Self> {
        let view = LeafNodeView::parse(bytes, dim, padded_dim, rabitq_words)?;
        let n = view.entry_count;
        Ok(Self {
            parent_id: view.parent_id,
            centroid: read_f32_le(view.centroid_bytes, dim)?,
            rotated_centroid: read_f32_le(view.rotated_centroid_bytes, padded_dim)?,
            doc_ids: read_u64_le(view.doc_ids_bytes, n)?,
            res_norm_sqs: read_f32_le(view.res_norm_sqs_bytes, n)?,
            scales: read_f32_le(view.scales_bytes, n)?,
            codes: read_u64_le(view.codes_bytes, n * rabitq_words)?,
        })
    }

    /// Build a [`QuantizedEntry`] for slot `idx` (used for diagnostics/verification).
    pub fn quantized_entry(&self, idx: usize, rabitq_words: usize) -> QuantizedEntry {
        let n = self.doc_ids.len();
        let code = (0..rabitq_words).map(|w| self.codes[w * n + idx]).collect();
        QuantizedEntry {
            res_norm_sq: self.res_norm_sqs[idx],
            scale: self.scales[idx],
            code,
        }
    }
}

/// Zero-copy borrowed view over serialized [`PackedLeafNode`] bytes (e.g. from `DBPinnableSlice`).
#[derive(Debug, Clone, Copy)]
pub struct LeafNodeView<'a> {
    pub parent_id: Option<NodeId>,
    pub entry_count: usize,
    pub doc_ids_bytes: &'a [u8],
    pub codes_bytes: &'a [u8],
    pub centroid_bytes: &'a [u8],
    pub rotated_centroid_bytes: &'a [u8],
    pub res_norm_sqs_bytes: &'a [u8],
    pub scales_bytes: &'a [u8],
}

impl<'a> LeafNodeView<'a> {
    pub fn parse(
        bytes: &'a [u8],
        dim: usize,
        padded_dim: usize,
        rabitq_words: usize,
    ) -> Result<Self> {
        if bytes.len() < LEAF_HEADER_LEN {
            return Err(TreeError::CorruptedData(format!(
                "PackedLeafNode header too short: {}",
                bytes.len()
            )));
        }
        let parent_id = decode_parent_flag(bytes[0], &bytes[8..16])?;
        let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let expected = LEAF_HEADER_LEN + (dim + padded_dim) * 4 + n * (16 + rabitq_words * 8);
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "PackedLeafNode expected {expected} bytes, got {}",
                bytes.len()
            )));
        }

        let mut off = LEAF_HEADER_LEN;
        let doc_ids_bytes = &bytes[off..off + n * 8];
        off += n * 8;
        let codes_bytes = &bytes[off..off + n * rabitq_words * 8];
        off += n * rabitq_words * 8;
        let centroid_bytes = &bytes[off..off + dim * 4];
        off += dim * 4;
        let rotated_centroid_bytes = &bytes[off..off + padded_dim * 4];
        off += padded_dim * 4;
        let res_norm_sqs_bytes = &bytes[off..off + n * 4];
        off += n * 4;
        let scales_bytes = &bytes[off..off + n * 4];

        Ok(Self {
            parent_id,
            entry_count: n,
            doc_ids_bytes,
            codes_bytes,
            centroid_bytes,
            rotated_centroid_bytes,
            res_norm_sqs_bytes,
            scales_bytes,
        })
    }

    pub fn centroid_cow(&self, dim: usize) -> Result<Cow<'a, [f32]>> {
        borrow_f32_slice(self.centroid_bytes, dim)
    }
}

#[inline]
fn decode_parent_flag(flag: u8, pid_bytes: &[u8]) -> Result<Option<NodeId>> {
    let raw = u64::from_le_bytes(pid_bytes.try_into().unwrap());
    match flag {
        0 => Ok(None),
        1 => Ok(Some(raw)),
        other => Err(TreeError::CorruptedData(format!(
            "invalid has_parent flag: {other}"
        ))),
    }
}

/// Encode `vals` as raw little-endian `f32` bytes (no header).
pub fn encode_f32_slice(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    write_f32_le(vals, &mut out);
    out
}

/// Decode `dim` little-endian `f32` values from `bytes`.
pub fn decode_f32_slice(bytes: &[u8], dim: usize) -> Result<Vec<f32>> {
    read_f32_le(bytes, dim)
}

/// Decode little-endian `f32` values from `bytes` directly into `dst`.
#[inline]
pub fn decode_f32_into(bytes: &[u8], dst: &mut [f32]) -> Result<()> {
    let expected = dst.len() * 4;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "f32 slice expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        dst.copy_from_slice(slice);
        return Ok(());
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    for (out, &chunk) in dst.iter_mut().zip(chunks.iter()) {
        *out = f32::from_le_bytes(chunk);
    }
    Ok(())
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
fn write_u64_le(slice: &[u64], out: &mut Vec<u8>) {
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
fn read_f32_le(bytes: &[u8], len: usize) -> Result<Vec<f32>> {
    let expected = len * 4;
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
    let mut out = Vec::with_capacity(len);
    let (chunks, _) = bytes.as_chunks::<4>();
    for &chunk in chunks {
        out.push(f32::from_le_bytes(chunk));
    }
    Ok(out)
}

#[inline]
fn read_u64_le(bytes: &[u8], len: usize) -> Result<Vec<u64>> {
    let expected = len * 8;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "u64 slice expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, u64>(bytes) {
        return Ok(slice.to_vec());
    }
    let mut out = Vec::with_capacity(len);
    let (chunks, _) = bytes.as_chunks::<8>();
    for &chunk in chunks {
        out.push(u64::from_le_bytes(chunk));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_leaf_node_insert_remove_roundtrip() {
        let dim = 5;
        let padded_dim = 64;
        let words = 1;
        let mut leaf = PackedLeafNode::empty(Some(9), dim, padded_dim);

        leaf.upsert_entry(30, 1.5, 2.0, &[0xAA], words);
        leaf.upsert_entry(10, 0.5, 1.0, &[0x11], words);
        leaf.upsert_entry(20, 2.5, 3.0, &[0x55], words);

        assert_eq!(leaf.doc_ids, vec![10, 20, 30]);
        assert_eq!(leaf.res_norm_sqs, vec![0.5, 2.5, 1.5]);
        assert_eq!(leaf.scales, vec![1.0, 3.0, 2.0]);
        assert_eq!(leaf.codes, vec![0x11, 0x55, 0xAA]);

        let bytes = leaf.encode(dim, padded_dim, words);
        let decoded = PackedLeafNode::decode(&bytes, dim, padded_dim, words).unwrap();
        assert_eq!(leaf, decoded);

        assert!(leaf.remove_entry(20, words));
        assert_eq!(leaf.doc_ids, vec![10, 30]);
        assert_eq!(leaf.codes, vec![0x11, 0xAA]);

        // Multi-word test
        let words2 = 3;
        let mut leaf2 = PackedLeafNode::empty(None, dim, 192);
        leaf2.upsert_entry(2, 1.0, 1.0, &[1, 2, 3], words2);
        leaf2.upsert_entry(4, 2.0, 2.0, &[4, 5, 6], words2);
        leaf2.upsert_entry(1, 3.0, 3.0, &[7, 8, 9], words2);
        leaf2.upsert_entry(3, 4.0, 4.0, &[10, 11, 12], words2);
        // Word-major: w0=[7,1,10,4], w1=[8,2,11,5], w2=[9,3,12,6]
        assert_eq!(leaf2.doc_ids, vec![1, 2, 3, 4]);
        assert_eq!(leaf2.codes, vec![7, 1, 10, 4, 8, 2, 11, 5, 9, 3, 12, 6]);
        assert!(leaf2.remove_entry(2, words2));
        assert_eq!(leaf2.doc_ids, vec![1, 3, 4]);
        assert_eq!(leaf2.codes, vec![7, 10, 4, 8, 11, 5, 9, 12, 6]);
    }
}
