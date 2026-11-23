//! Packed leaf ANN nodes.

use izzydb_core::config::{LocalId, NodeId, RowId};
use izzydb_core::error::Result;
use izzydb_rabitq::{EncodeScratch, QuantizedEntry, RabitqQuantizer};

use super::*;

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

    pub(super) fn ensure_slot_storage(&mut self, new_slot_cap: usize, rabitq_words: usize) {
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

    pub(super) fn alloc_slot(&mut self, rabitq_words: usize) -> usize {
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
    pub(super) fn write_slot_code(
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
