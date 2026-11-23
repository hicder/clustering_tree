use std::borrow::Cow;

use izzydb_core::config::{DistanceMetric, LocalId, NodeId, RowId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::metric::compute_weighted_centroid_flat;
use izzydb_rabitq::{EncodeScratch, QuantizedEntry, RabitqQuantizer};

const META_LEN: usize = 51;
const INTERNAL_HEADER_LEN: usize = 24;
const LEAF_HEADER_LEN: usize = 24;

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
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
            None => {
                buf.push(0);
                buf.extend_from_slice(&0u16.to_le_bytes());
                buf.extend_from_slice(&(n as u32).to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
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

/// Packed slot-indexed leaf node (`slot` is leaf-local) stored in `cf_leaf_entries`.
///
/// Arrays (`row_ids`, `code_popcounts`, `res_norm_sqs`, `scales`, `codes`) are padded to
/// `slot_cap.next_multiple_of(8)` so 8-wide RaBitQ SIMD loops execute cleanly over 64-byte tiles:
/// `codes[((slot / 8) * rabitq_words + w) * 8 + (slot % 8)]`.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedLeafNode {
    pub parent_id: Option<NodeId>,
    pub slot_cap: usize,
    pub live_count: usize,
    pub stale_hint: u32,
    pub live_mask: Vec<u64>,
    pub centroid: Vec<f32>,
    pub rotated_centroid: Vec<f32>,
    pub row_ids: Vec<u64>,
    pub code_popcounts: Vec<u32>,
    pub res_norm_sqs: Vec<f32>,
    pub scales: Vec<f32>,
    pub codes: Vec<u64>,
}

impl PackedLeafNode {
    /// Construct a dense, encoded [`PackedLeafNode`] directly from `(centroid, row_ids, flat_vectors)`
    /// without a `TxContext`.
    pub fn from_entries(
        parent_id: Option<NodeId>,
        centroid: Vec<f32>,
        row_ids: &[RowId],
        flat_vectors: &[f32],
        quantizer: &RabitqQuantizer,
        dim: usize,
        scratch: &mut EncodeScratch,
    ) -> Self {
        let padded_dim = quantizer.padded_dim();
        let mut leaf = Self::empty(parent_id, dim, padded_dim);
        leaf.centroid = centroid;
        if row_ids.is_empty() {
            quantizer.rotate_into(&leaf.centroid, &mut leaf.rotated_centroid);
        } else {
            leaf.reset_and_encode(row_ids, flat_vectors, quantizer, dim, scratch);
        }
        leaf
    }

    pub fn empty(parent_id: Option<NodeId>, dim: usize, padded_dim: usize) -> Self {
        Self {
            parent_id,
            slot_cap: 0,
            live_count: 0,
            stale_hint: 0,
            live_mask: Vec::new(),
            centroid: vec![0.0; dim],
            rotated_centroid: vec![0.0; padded_dim],
            row_ids: Vec::new(),
            code_popcounts: Vec::new(),
            res_norm_sqs: Vec::new(),
            scales: Vec::new(),
            codes: Vec::new(),
        }
    }

    #[inline]
    pub fn entry_count(&self) -> usize {
        self.live_count
    }

    #[inline]
    pub fn is_live(&self, slot: usize) -> bool {
        slot < self.slot_cap && ((self.live_mask[slot >> 6] >> (slot & 63)) & 1) != 0
    }

    #[inline]
    pub fn row_id(&self, slot: usize) -> RowId {
        RowId(self.row_ids[slot])
    }

    /// Collect all live `RowId`s in slot order.
    pub fn live_row_ids(&self) -> Vec<RowId> {
        let mut out = Vec::with_capacity(self.live_count);
        for (w_idx, &word) in self.live_mask.iter().enumerate() {
            let mut b = word;
            while b != 0 {
                let bit = b.trailing_zeros() as usize;
                b &= b - 1;
                let slot = (w_idx << 6) | bit;
                if slot < self.slot_cap {
                    out.push(RowId(self.row_ids[slot]));
                }
            }
        }
        out
    }

    /// Collect all `(slot as LocalId, RowId)` pairs in ascending slot order.
    pub fn live_slot_row_pairs(&self) -> Vec<(LocalId, RowId)> {
        let mut out = Vec::with_capacity(self.live_count);
        for (w_idx, &word) in self.live_mask.iter().enumerate() {
            let mut b = word;
            while b != 0 {
                let bit = b.trailing_zeros() as usize;
                b &= b - 1;
                let slot = (w_idx << 6) | bit;
                if slot < self.slot_cap {
                    out.push((slot as LocalId, RowId(self.row_ids[slot])));
                }
            }
        }
        out
    }

    pub fn find_slot(&self, row_id: RowId) -> Option<usize> {
        let raw = row_id.raw();
        for (w_idx, &word) in self.live_mask.iter().enumerate() {
            let mut b = word;
            while b != 0 {
                let bit = b.trailing_zeros() as usize;
                b &= b - 1;
                let slot = (w_idx << 6) | bit;
                if slot < self.slot_cap && self.row_ids[slot] == raw {
                    return Some(slot);
                }
            }
        }
        None
    }

    fn ensure_slot_storage(&mut self, new_slot_cap: usize, rabitq_words: usize) {
        self.slot_cap = new_slot_cap;
        let padded = new_slot_cap.next_multiple_of(8);
        let mask_words = new_slot_cap.div_ceil(64);
        self.live_mask.resize(mask_words, 0);
        self.row_ids.resize(padded, 0);
        self.code_popcounts.resize(padded, 0);
        self.res_norm_sqs.resize(padded, 0.0);
        self.scales.resize(padded, 0.0);
        self.codes.resize(padded * rabitq_words, 0);
    }

    fn alloc_slot(&mut self, rabitq_words: usize) -> usize {
        if self.live_count < self.slot_cap {
            for (w_idx, &word) in self.live_mask.iter().enumerate() {
                let inv = !word;
                if inv != 0 {
                    let bit = inv.trailing_zeros() as usize;
                    let slot = (w_idx << 6) | bit;
                    if slot < self.slot_cap {
                        return slot;
                    }
                }
            }
        }
        let slot = self.slot_cap;
        self.ensure_slot_storage(slot + 1, rabitq_words);
        slot
    }

    #[inline]
    fn write_slot_code(
        &mut self,
        slot: usize,
        code: &[u64],
        popcnt: u32,
        rabitq_words: usize,
    ) {
        self.code_popcounts[slot] = popcnt;
        let tile = slot >> 3;
        let lane = slot & 7;
        for (w, &word) in code.iter().enumerate() {
            self.codes[(tile * rabitq_words + w) * 8 + lane] = word;
        }
    }

    /// Insert or update `row_id` at `known_slot` (or allocate/find a slot).
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_entry(
        &mut self,
        row_id: RowId,
        known_slot: Option<LocalId>,
        res_norm_sq: f32,
        scale: f32,
        code: &[u64],
        popcnt: u32,
        rabitq_words: usize,
    ) -> LocalId {
        debug_assert_eq!(code.len(), rabitq_words);
        let raw = row_id.raw();
        let existing_slot = known_slot
            .map(|s| s as usize)
            .filter(|&s| self.is_live(s) && self.row_ids[s] == raw)
            .or_else(|| self.find_slot(row_id));

        let slot = match existing_slot {
            Some(s) => s,
            None => {
                let s = self.alloc_slot(rabitq_words);
                self.live_mask[s >> 6] |= 1u64 << (s & 63);
                self.live_count += 1;
                self.row_ids[s] = raw;
                s
            }
        };
        self.res_norm_sqs[slot] = res_norm_sq;
        self.scales[slot] = scale;
        self.write_slot_code(slot, code, popcnt, rabitq_words);
        slot as LocalId
    }

    /// Allocate/find slot for `row_id` with zeroed quantized fields (used before `re_encode_existing`).
    pub fn insert_slot_for_reencode(&mut self, row_id: RowId, rabitq_words: usize) -> LocalId {
        if let Some(s) = self.find_slot(row_id) {
            return s as LocalId;
        }
        let s = self.alloc_slot(rabitq_words);
        self.live_mask[s >> 6] |= 1u64 << (s & 63);
        self.live_count += 1;
        self.row_ids[s] = row_id.raw();
        s as LocalId
    }

    /// Remove `row_id` (optionally at `known_slot`) while keeping all surviving entries' slots untouched.
    pub fn remove_entry(
        &mut self,
        row_id: RowId,
        known_slot: Option<LocalId>,
        rabitq_words: usize,
    ) -> Option<LocalId> {
        let raw = row_id.raw();
        let slot = known_slot
            .map(|s| s as usize)
            .filter(|&s| self.is_live(s) && self.row_ids[s] == raw)
            .or_else(|| self.find_slot(row_id))?;

        self.remove_slot(slot, rabitq_words);
        Some(slot as LocalId)
    }

    /// Remove a slot directly (used when pruning stale entries by slot index).
    pub fn remove_slot(&mut self, slot: usize, rabitq_words: usize) {
        if !self.is_live(slot) {
            return;
        }
        self.live_mask[slot >> 6] &= !(1u64 << (slot & 63));
        self.live_count -= 1;
        self.row_ids[slot] = 0;
        self.res_norm_sqs[slot] = 0.0;
        self.scales[slot] = 0.0;
        self.code_popcounts[slot] = 0;
        let tile = slot >> 3;
        let lane = slot & 7;
        for w in 0..rabitq_words {
            self.codes[(tile * rabitq_words + w) * 8 + lane] = 0;
        }

        let mut new_cap = self.slot_cap;
        while new_cap > 0 && !self.is_live(new_cap - 1) {
            new_cap -= 1;
        }
        if new_cap != self.slot_cap {
            self.ensure_slot_storage(new_cap, rabitq_words);
        }
    }

    /// Re-encode all live entries from `flat_vectors` (aligned with `self.live_slot_row_pairs()`)
    /// relative to `self.centroid` without altering any slot.
    pub fn re_encode_existing(
        &mut self,
        flat_vectors: &[f32],
        quantizer: &RabitqQuantizer,
        dim: usize,
        scratch: &mut EncodeScratch,
    ) {
        debug_assert_eq!(self.centroid.len(), dim);
        let words = quantizer.num_words();
        quantizer.rotate_into(&self.centroid, &mut self.rotated_centroid);

        let live_slots = self.live_slot_row_pairs();
        assert_eq!(flat_vectors.len(), live_slots.len() * dim);

        let mut code_buf = vec![0u64; words];
        for ((slot, _), vec) in live_slots.into_iter().zip(flat_vectors.chunks_exact(dim)) {
            let (res_norm_sq, scale, popcnt) =
                quantizer.encode_into_with_popcnt(vec, &self.centroid, scratch, &mut code_buf);
            let s = slot as usize;
            self.res_norm_sqs[s] = res_norm_sq;
            self.scales[s] = scale;
            self.write_slot_code(s, &code_buf, popcnt, words);
        }
    }

    /// Compact entries into dense slots `0..n` and encode from `flat_vectors`.
    pub fn reset_and_encode(
        &mut self,
        row_ids: &[RowId],
        flat_vectors: &[f32],
        quantizer: &RabitqQuantizer,
        dim: usize,
        scratch: &mut EncodeScratch,
    ) {
        let n = row_ids.len();
        let words = quantizer.num_words();
        self.live_mask.clear();
        self.row_ids.clear();
        self.code_popcounts.clear();
        self.res_norm_sqs.clear();
        self.scales.clear();
        self.codes.clear();

        self.live_count = n;
        self.stale_hint = 0;
        self.ensure_slot_storage(n, words);
        for (i, &rid) in row_ids.iter().enumerate() {
            self.live_mask[i >> 6] |= 1u64 << (i & 63);
            self.row_ids[i] = rid.raw();
        }
        self.re_encode_existing(flat_vectors, quantizer, dim, scratch);
    }

    pub fn entry_code_word(&self, slot: usize, word: usize, rabitq_words: usize) -> u64 {
        let tile = slot >> 3;
        let lane = slot & 7;
        self.codes[(tile * rabitq_words + word) * 8 + lane]
    }

    pub fn encode(&self, dim: usize, padded_dim: usize, rabitq_words: usize) -> Vec<u8> {
        let padded_slots = self.slot_cap.next_multiple_of(8);
        let mask_words = self.slot_cap.div_ceil(64);
        let pop_u32 = padded_dim >= 65536;
        let pop_bytes = padded_slots * if pop_u32 { 4 } else { 2 };

        debug_assert_eq!(self.centroid.len(), dim);
        debug_assert_eq!(self.rotated_centroid.len(), padded_dim);
        debug_assert_eq!(self.live_mask.len(), mask_words);
        debug_assert_eq!(self.row_ids.len(), padded_slots);
        debug_assert_eq!(self.code_popcounts.len(), padded_slots);
        debug_assert_eq!(self.res_norm_sqs.len(), padded_slots);
        debug_assert_eq!(self.scales.len(), padded_slots);
        debug_assert_eq!(self.codes.len(), padded_slots * rabitq_words);

        let total = LEAF_HEADER_LEN
            + mask_words * 8
            + padded_slots * (8 + rabitq_words * 8 + 8)
            + (dim + padded_dim) * 4
            + pop_bytes;
        let mut buf = Vec::with_capacity(total);
        match self.parent_id {
            Some(pid) => {
                buf.push(1);
                buf.push(u8::from(pop_u32));
                buf.extend_from_slice(&(mask_words as u16).to_le_bytes());
                buf.extend_from_slice(&(self.slot_cap as u32).to_le_bytes());
                buf.extend_from_slice(&pid.to_le_bytes());
            }
            None => {
                buf.push(0);
                buf.push(u8::from(pop_u32));
                buf.extend_from_slice(&(mask_words as u16).to_le_bytes());
                buf.extend_from_slice(&(self.slot_cap as u32).to_le_bytes());
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        buf.extend_from_slice(&(self.live_count as u32).to_le_bytes());
        buf.extend_from_slice(&self.stale_hint.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        write_u64_le(&self.live_mask, &mut buf);
        write_u64_le(&self.row_ids, &mut buf);
        write_u64_le(&self.codes, &mut buf);
        write_f32_le(&self.res_norm_sqs, &mut buf);
        write_f32_le(&self.scales, &mut buf);
        write_f32_le(&self.centroid, &mut buf);
        write_f32_le(&self.rotated_centroid, &mut buf);
        if pop_u32 {
            write_u32_le(&self.code_popcounts, &mut buf);
        } else {
            for &p in &self.code_popcounts {
                buf.extend_from_slice(&(p as u16).to_le_bytes());
            }
        }
        buf
    }

    pub fn decode(bytes: &[u8], dim: usize, padded_dim: usize, rabitq_words: usize) -> Result<Self> {
        let view = LeafNodeView::parse(bytes, dim, padded_dim, rabitq_words)?;
        let padded_slots = view.padded_slots;
        let mask_words = view.live_mask_bytes.len() / 8;
        let code_popcounts = if view.popcount_u32 {
            read_u32_le(view.code_popcounts_bytes, padded_slots)?
        } else {
            view.code_popcounts_bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| u16::from_le_bytes(c) as u32)
                .collect()
        };
        Ok(Self {
            parent_id: view.parent_id,
            slot_cap: view.slot_cap,
            live_count: view.live_count,
            stale_hint: view.stale_hint,
            live_mask: read_u64_le(view.live_mask_bytes, mask_words)?,
            centroid: read_f32_le(view.centroid_bytes, dim)?,
            rotated_centroid: read_f32_le(view.rotated_centroid_bytes, padded_dim)?,
            row_ids: read_u64_le(view.row_ids_bytes, padded_slots)?,
            code_popcounts,
            res_norm_sqs: read_f32_le(view.res_norm_sqs_bytes, padded_slots)?,
            scales: read_f32_le(view.scales_bytes, padded_slots)?,
            codes: read_u64_le(view.codes_bytes, padded_slots * rabitq_words)?,
        })
    }

    /// Build a [`QuantizedEntry`] for `slot` (used for diagnostics/verification).
    pub fn quantized_entry(&self, slot: usize, rabitq_words: usize) -> QuantizedEntry {
        let code = (0..rabitq_words)
            .map(|w| self.entry_code_word(slot, w, rabitq_words))
            .collect();
        QuantizedEntry {
            res_norm_sq: self.res_norm_sqs[slot],
            scale: self.scales[slot],
            code,
        }
    }
}

/// Zero-copy borrowed view over serialized [`PackedLeafNode`] bytes (e.g. from `DBPinnableSlice`).
#[derive(Debug, Clone, Copy)]
pub struct LeafNodeView<'a> {
    pub parent_id: Option<NodeId>,
    pub popcount_u32: bool,
    pub slot_cap: usize,
    pub live_count: usize,
    pub stale_hint: u32,
    pub padded_slots: usize,
    pub live_mask_bytes: &'a [u8],
    pub row_ids_bytes: &'a [u8],
    pub codes_bytes: &'a [u8],
    pub res_norm_sqs_bytes: &'a [u8],
    pub scales_bytes: &'a [u8],
    pub centroid_bytes: &'a [u8],
    pub rotated_centroid_bytes: &'a [u8],
    pub code_popcounts_bytes: &'a [u8],
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
        let parent_id = decode_parent_flag(bytes[0], &bytes[8..12])?;
        let popcount_u32 = match bytes[1] {
            0 => false,
            1 => true,
            other => {
                return Err(TreeError::CorruptedData(format!(
                    "invalid popcount_u32 flag: {other}"
                )));
            }
        };
        let mask_words = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        let slot_cap = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let live_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let stale_hint = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        if mask_words != slot_cap.div_ceil(64) || live_count > slot_cap {
            return Err(TreeError::CorruptedData(format!(
                "invalid PackedLeafNode header: slot_cap={slot_cap}, live_count={live_count}, mask_words={mask_words}"
            )));
        }

        let padded_slots = slot_cap.next_multiple_of(8);
        let pop_bytes = padded_slots * if popcount_u32 { 4 } else { 2 };
        let expected = LEAF_HEADER_LEN
            + mask_words * 8
            + padded_slots * (8 + rabitq_words * 8 + 8)
            + (dim + padded_dim) * 4
            + pop_bytes;
        if bytes.len() != expected {
            return Err(TreeError::CorruptedData(format!(
                "PackedLeafNode expected {expected} bytes, got {}",
                bytes.len()
            )));
        }

        let mut off = LEAF_HEADER_LEN;
        let live_mask_bytes = &bytes[off..off + mask_words * 8];
        off += mask_words * 8;
        let row_ids_bytes = &bytes[off..off + padded_slots * 8];
        off += padded_slots * 8;
        let codes_bytes = &bytes[off..off + padded_slots * rabitq_words * 8];
        off += padded_slots * rabitq_words * 8;
        let res_norm_sqs_bytes = &bytes[off..off + padded_slots * 4];
        off += padded_slots * 4;
        let scales_bytes = &bytes[off..off + padded_slots * 4];
        off += padded_slots * 4;
        let centroid_bytes = &bytes[off..off + dim * 4];
        off += dim * 4;
        let rotated_centroid_bytes = &bytes[off..off + padded_dim * 4];
        off += padded_dim * 4;
        let code_popcounts_bytes = &bytes[off..off + pop_bytes];

        Ok(Self {
            parent_id,
            popcount_u32,
            slot_cap,
            live_count,
            stale_hint,
            padded_slots,
            live_mask_bytes,
            row_ids_bytes,
            codes_bytes,
            res_norm_sqs_bytes,
            scales_bytes,
            centroid_bytes,
            rotated_centroid_bytes,
            code_popcounts_bytes,
        })
    }

    #[inline]
    pub fn has_tombstones(&self) -> bool {
        self.live_count < self.slot_cap
    }

    #[inline]
    pub fn mask_word_count(&self) -> usize {
        self.live_mask_bytes.len() / 8
    }

    #[inline]
    pub fn copy_live_mask_into(&self, dst: &mut Vec<u64>) {
        let words = self.mask_word_count();
        dst.clear();
        dst.reserve(words);
        for &chunk in self.live_mask_bytes.as_chunks::<8>().0 {
            dst.push(u64::from_le_bytes(chunk));
        }
    }

    #[inline]
    pub fn and_live_mask_into(&self, words_buf: &mut [u64]) {
        for (dst, &chunk) in words_buf
            .iter_mut()
            .zip(self.live_mask_bytes.as_chunks::<8>().0)
        {
            *dst &= u64::from_le_bytes(chunk);
        }
    }

    #[inline]
    pub fn is_live(&self, slot: usize) -> bool {
        if slot >= self.slot_cap {
            return false;
        }
        let w = slot >> 6;
        let word = u64::from_le_bytes(
            self.live_mask_bytes[w * 8..(w + 1) * 8]
                .try_into()
                .unwrap(),
        );
        ((word >> (slot & 63)) & 1) != 0
    }

    #[inline]
    pub fn row_id(&self, slot: usize) -> RowId {
        RowId(u64::from_le_bytes(
            self.row_ids_bytes[slot * 8..(slot + 1) * 8]
                .try_into()
                .unwrap(),
        ))
    }

    pub fn centroid_cow(&self, dim: usize) -> Result<Cow<'a, [f32]>> {
        borrow_f32_slice(self.centroid_bytes, dim)
    }
}

#[inline]
fn decode_parent_flag(flag: u8, pid_bytes: &[u8]) -> Result<Option<NodeId>> {
    let raw = u32::from_le_bytes(pid_bytes.try_into().unwrap());
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
fn write_u32_le(slice: &[u32], out: &mut Vec<u8>) {
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
fn read_u32_le(bytes: &[u8], len: usize) -> Result<Vec<u32>> {
    let expected = len * 4;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "u32 slice expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, u32>(bytes) {
        return Ok(slice.to_vec());
    }
    let mut out = Vec::with_capacity(len);
    let (chunks, _) = bytes.as_chunks::<4>();
    for &chunk in chunks {
        out.push(u32::from_le_bytes(chunk));
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

        let r30 = RowId::new(1, 30);
        let r10 = RowId::new(0, 10);
        let r20 = RowId::new(2, 20);
        let r40 = RowId::new(3, 40);

        assert_eq!(leaf.upsert_entry(r30, None, 1.5, 2.0, &[0xAA], 4, words), 0);
        assert_eq!(leaf.upsert_entry(r10, None, 0.5, 1.0, &[0x11], 2, words), 1);
        assert_eq!(leaf.upsert_entry(r20, None, 2.5, 3.0, &[0x55], 4, words), 2);
        assert_eq!(leaf.slot_cap, 3);
        assert_eq!(leaf.live_count, 3);

        let bytes = leaf.encode(dim, padded_dim, words);
        let decoded = PackedLeafNode::decode(&bytes, dim, padded_dim, words).unwrap();
        assert_eq!(leaf, decoded);

        // Removing slot 1 (r10) keeps slot 0 (r30) and slot 2 (r20) untouched!
        assert_eq!(leaf.remove_entry(r10, Some(1), words), Some(1));
        assert_eq!(leaf.live_count, 2);
        assert_eq!(leaf.slot_cap, 3);
        assert!(!leaf.is_live(1));
        assert!(leaf.is_live(0));
        assert!(leaf.is_live(2));
        assert_eq!(leaf.row_id(0), r30);
        assert_eq!(leaf.row_id(2), r20);

        // Next insert reuses free slot 1!
        assert_eq!(leaf.upsert_entry(r40, None, 3.5, 4.0, &[0xFF], 8, words), 1);
        assert_eq!(leaf.live_count, 3);
        assert_eq!(leaf.slot_cap, 3);
        assert_eq!(leaf.row_id(1), r40);

        // Multi-word test
        let words2 = 3;
        let mut leaf2 = PackedLeafNode::empty(None, dim, 192);
        let s2 = leaf2.upsert_entry(RowId::new(0, 2), None, 1.0, 1.0, &[1, 2, 3], 6, words2);
        let s4 = leaf2.upsert_entry(RowId::new(0, 4), None, 2.0, 2.0, &[4, 5, 6], 6, words2);
        let s1 = leaf2.upsert_entry(RowId::new(0, 1), None, 3.0, 3.0, &[7, 8, 9], 7, words2);
        let s3 = leaf2.upsert_entry(RowId::new(0, 3), None, 4.0, 4.0, &[10, 11, 12], 7, words2);
        assert_eq!((s2, s4, s1, s3), (0, 1, 2, 3));
        assert_eq!(leaf2.entry_code_word(2, 1, words2), 8);
        assert_eq!(leaf2.entry_code_word(3, 2, words2), 12);
        let bytes2 = leaf2.encode(dim, 192, words2);
        assert_eq!(
            PackedLeafNode::decode(&bytes2, dim, 192, words2).unwrap(),
            leaf2
        );

        // AnnMeta roundtrip
        let meta = AnnMeta {
            root_id: 1,
            root_level: 2,
            next_node_id: 99,
            dim: 128,
            metric: DistanceMetric::Cosine,
            num_vectors: 1000,
            rabitq_seed: 42,
            rabitq_bits: 1,
            stale_rows: 17,
            vacuum_epoch: 3,
            vacuum_cursor: 55,
        };
        assert_eq!(AnnMeta::decode(&meta.encode()).unwrap(), meta);
    }
}
