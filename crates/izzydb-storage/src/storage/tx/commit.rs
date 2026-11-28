//! Commit of the overlay into a write batch.

use std::sync::Arc;

use izzydb_core::config::{AttrId, ColumnGen, LocalId, SegmentId};
use izzydb_core::error::{Result, TreeError};

use crate::codec::{
    ATTR_COUNT_TAG, DETACHED_COUNT_TAG, DETACHED_OVERWRITTEN_TAG, DetachedMeta, FTS_DOC_LEN_TAG,
    Manifest, SegPosting, SegmentState, ZONE_TAG, ZoneEntry, detached_count_suffix,
    detached_gen_bounds, detached_key, encode_roaring_bitmap,
};
use crate::node::AnnMeta;

use crate::storage::batch::StorageBatch;
use crate::storage::segments::SegmentEntry;
use crate::storage::types::map_db_err;
use crate::storage::{
    CF_ATTR_POSTINGS, CF_COLUMNS, CF_DETACHED, CF_DETACHED_POSTINGS, CF_ROW_DOC_IDS, CF_ROWS,
    CF_VECTORS,
};

use super::*;

impl<'a> TxContext<'a> {
    // -----------------------------------------------------------------------
    // Commit
    // -----------------------------------------------------------------------

    pub fn delete_zone_entry(&mut self, seg: SegmentId, attr_id: AttrId, tag: u8) {
        self.dirty_zone.remove(&(seg, attr_id, tag));
        self.dirty_zone_deletes.insert((seg, attr_id, tag));
    }

    pub fn set_zone_entry(&mut self, seg: SegmentId, attr_id: AttrId, tag: u8, entry: ZoneEntry) {
        self.dirty_zone_deletes.remove(&(seg, attr_id, tag));
        self.dirty_zone.insert((seg, attr_id, tag), entry);
    }

    pub fn get_detached_meta(&self, attr_id: AttrId) -> Result<DetachedMeta> {
        if let Some(m) = self.dirty_detached_meta.get(&attr_id) {
            return Ok(m.clone());
        }
        Ok(self.storage.get_detached_meta(attr_id)?.unwrap_or_default())
    }

    pub fn put_detached_meta(&mut self, attr_id: AttrId, meta: DetachedMeta) {
        self.dirty_detached_meta.insert(attr_id, meta);
    }

    pub fn get_detached_block_bytes(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        seg: SegmentId,
        block: u32,
    ) -> Result<Option<Vec<u8>>> {
        let key = (attr_id, col_gen, seg, block);
        if let Some(opt) = self.dirty_detached_blocks.get(&key) {
            return Ok(opt.clone());
        }
        Ok(self
            .storage
            .get_detached_block_pinned(attr_id, col_gen, seg, block)?
            .map(|s| s.to_vec()))
    }

    pub fn put_detached_block_bytes(
        &mut self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        seg: SegmentId,
        block: u32,
        bytes: Option<Vec<u8>>,
    ) {
        self.dirty_detached_blocks
            .insert((attr_id, col_gen, seg, block), bytes);
    }

    pub fn add_detached_posting(&mut self, key: Vec<u8>, local: LocalId) {
        self.dirty_detached_postings
            .entry(key)
            .or_default()
            .insert(local);
    }

    pub fn adjust_detached_count(&mut self, col_gen: ColumnGen, attr_key: &[u8], delta: i64) {
        if delta == 0 {
            return;
        }
        let suffix = detached_count_suffix(col_gen, attr_key);
        let entry = self.dirty_detached_count_deltas.entry(suffix).or_insert(0);
        *entry += delta;
    }

    pub fn record_detached_overwritten(
        &mut self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        seg: SegmentId,
        local: LocalId,
    ) -> Result<()> {
        let key = (attr_id, col_gen, seg);
        if !self.dirty_detached_overwritten.contains_key(&key) {
            let existing = self
                .storage
                .get_detached_overwritten(attr_id, col_gen, seg)?
                .unwrap_or_default();
            self.dirty_detached_overwritten.insert(key, existing);
        }
        self.dirty_detached_overwritten
            .get_mut(&key)
            .unwrap()
            .insert(local);
        Ok(())
    }

    pub fn retire_detached_gen(&mut self, attr_id: AttrId, col_gen: ColumnGen) {
        self.retire_detached_gens.push((attr_id, col_gen));
    }

    pub fn commit(mut self) -> Result<(Manifest, Option<AnnMeta>)> {
        let mut batch = self.storage.new_batch();
        self.stage_manifest_and_segments(&mut batch)?;
        self.stage_rows_and_tree(&mut batch);
        self.stage_postings_and_zones(&mut batch)?;
        self.stage_attr_counts(&mut batch)?;
        self.stage_fts_writes(&mut batch);
        self.stage_columns(&mut batch);
        self.stage_detached(&mut batch)?;

        self.storage.write_batch(batch)?;

        self.flush_caches_after_commit();
        Ok((self.manifest, self.ann_meta))
    }

    fn stage_manifest_and_segments(&self, batch: &mut StorageBatch<'_>) -> Result<()> {
        if self.manifest_dirty {
            batch.put_manifest(&self.manifest);
        }
        if let Some(ref dropped) = self.dirty_dropped_segments {
            batch.put_dropped_segments(&encode_roaring_bitmap(dropped)?);
        }
        for (&seg, opt_meta) in &self.dirty_seg_meta {
            match opt_meta {
                Some(meta) => batch.put_seg_meta(seg, meta),
                None => batch.delete_seg_meta(seg),
            }
        }
        for (&seg, bm) in &self.dirty_seg_deletes {
            if bm.is_empty() {
                batch.delete_seg_deletes(seg);
            } else {
                batch.put_seg_deletes(seg, &encode_roaring_bitmap(bm)?);
            }
        }
        Ok(())
    }

    fn stage_rows_and_tree(&self, batch: &mut StorageBatch<'_>) {
        if self.ann_meta_dirty
            && let Some(ref meta) = self.ann_meta
        {
            batch.put_ann_meta(meta);
        }
        for &node_id in &self.dirty_internal {
            match self.internal_nodes.get(&node_id).and_then(|o| o.as_ref()) {
                Some(node) => batch.put_internal_node(node_id, node),
                None => batch.delete_internal_node(node_id),
            }
        }
        for &leaf_id in &self.dirty_leaf {
            match self.leaf_nodes.get(&leaf_id).and_then(|o| o.as_ref()) {
                Some(leaf) => batch.put_leaf_node(leaf_id, leaf),
                None => batch.delete_leaf_node(leaf_id),
            }
        }
        for (&doc_id, &opt_row) in &self.dirty_id_map {
            match opt_row {
                Some(row) => batch.put_id_map(doc_id, row),
                None => batch.delete_id_map(doc_id),
            }
        }
        for (&row, opt_bytes) in &self.dirty_rows {
            match opt_bytes {
                Some(bytes) => batch.put_row_bytes(row, bytes),
                None => batch.delete_row(row),
            }
        }
        for (&(seg, block), ids) in &self.dirty_doc_id_blocks {
            batch.put_doc_id_block(seg, block, ids);
        }
        for (&row, opt_vec) in &self.dirty_vectors {
            match opt_vec {
                Some(vec) => batch.put_vector(row, vec),
                None => batch.delete_vector(row),
            }
        }
    }

    fn stage_postings_and_zones(&self, batch: &mut StorageBatch<'_>) -> Result<()> {
        if !self.dirty_postings.is_empty() {
            let keys: Vec<Vec<u8>> = self.dirty_postings.keys().cloned().collect();
            let existing = self.storage.multi_get_attr_posting_slices(&keys)?;
            let cf = self.storage.cf(CF_ATTR_POSTINGS)?;
            let _ = cf;
            for ((key, delta_bm), opt_slice) in self.dirty_postings.iter().zip(existing) {
                let mut merged = match opt_slice {
                    Some(slice) => SegPosting::decode(&slice)?,
                    None => SegPosting::default(),
                };
                merged.or_bitmap(delta_bm);
                batch.put_attr_posting(key, &merged.encode()?);
            }
        }
        for (&(seg, attr_id, tag), entry) in &self.dirty_zone {
            batch.put_zone_entry(seg, attr_id, tag, entry);
        }
        for &(seg, attr_id, tag) in &self.dirty_zone_deletes {
            batch.delete_zone_entry(seg, attr_id, tag);
        }
        Ok(())
    }

    fn stage_attr_counts(&self, batch: &mut StorageBatch<'_>) -> Result<()> {
        let mut deltas: Vec<(&Vec<u8>, i64)> = self
            .dirty_attr_count_deltas
            .iter()
            .filter(|&(_, &d)| d != 0)
            .map(|(k, &d)| (k, d))
            .collect();
        if deltas.is_empty() {
            return Ok(());
        }
        deltas.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let db_keys: Vec<Vec<u8>> = deltas
            .iter()
            .map(|(k, _)| {
                let mut key = Vec::with_capacity(1 + k.len());
                key.push(ATTR_COUNT_TAG);
                key.extend_from_slice(k);
                key
            })
            .collect();
        let current = self.storage.db.multi_get(db_keys.iter());
        for ((attr_key, delta), cur) in deltas.iter().zip(current) {
            let cur = match cur.map_err(map_db_err)? {
                Some(bytes) => {
                    let arr: [u8; 8] = bytes.as_slice().try_into().map_err(|_| {
                        TreeError::CorruptedData(format!(
                            "attr count expected 8 bytes, got {}",
                            bytes.len()
                        ))
                    })?;
                    u64::from_le_bytes(arr) as i128
                }
                None => 0,
            };
            let next = cur + *delta as i128;
            if next < 0 {
                return Err(TreeError::InvariantViolation(format!(
                    "attr count underflow for key {attr_key:?}: {cur} + {delta}"
                )));
            }
            if next == 0 {
                batch.delete_attr_count(attr_key);
            } else {
                batch.put_attr_count(attr_key, next as u64);
            }
        }
        Ok(())
    }

    fn stage_fts_writes(&self, batch: &mut StorageBatch<'_>) {
        if self.fts_meta_dirty
            && let Some(ref bytes) = self.fts_meta_bytes
        {
            batch.put_fts_meta_bytes(bytes);
        }
        for (key, opt) in &self.dirty_fts_terms {
            match opt {
                Some(bytes) => batch.put_fts_term(key, bytes),
                None => batch.delete_fts_term(key),
            }
        }
        for (&(term_id, chunk), opt) in &self.dirty_fts_skips {
            match opt {
                Some(bytes) => batch.put_fts_skip(term_id, chunk, bytes),
                None => batch.delete_fts_skip(term_id, chunk),
            }
        }
        for (&(term_id, block), opt) in &self.dirty_fts_postings {
            match opt {
                Some(bytes) => batch.put_fts_posting(term_id, block, bytes),
                None => batch.delete_fts_posting(term_id, block),
            }
        }
        for (&(term_id, block), opt) in &self.dirty_fts_positions {
            match opt {
                Some(bytes) => batch.put_fts_position(term_id, block, bytes),
                None => batch.delete_fts_position(term_id, block),
            }
        }
        for (&(attr_id, block), opt) in &self.dirty_fts_doc_lens {
            match opt {
                Some(bytes) => batch.put_fts_doc_len(attr_id, block, bytes),
                None => batch.delete_fts_doc_len(attr_id, block),
            }
        }
    }

    fn stage_columns(&self, batch: &mut StorageBatch<'_>) {
        for (key, bytes) in &self.dirty_columns {
            batch.put_column_block(key, bytes);
        }
    }

    fn stage_detached(&self, batch: &mut StorageBatch<'_>) -> Result<()> {
        for &(attr_id, col_gen) in &self.retire_detached_gens {
            let (lo, hi) = detached_gen_bounds(attr_id, col_gen);
            batch.delete_range_cf(CF_DETACHED, &lo, &hi);
            batch.delete_range_cf(CF_DETACHED_POSTINGS, &lo, &hi);

            let mut cnt_lo = [0u8; 7];
            cnt_lo[0] = DETACHED_COUNT_TAG;
            cnt_lo[1..7].copy_from_slice(&lo);
            let mut cnt_hi = [0u8; 7];
            cnt_hi[0] = DETACHED_COUNT_TAG;
            cnt_hi[1..7].copy_from_slice(&hi);
            batch.delete_range_default(&cnt_lo, &cnt_hi);

            let mut ow_lo = [0u8; 7];
            ow_lo[0] = DETACHED_OVERWRITTEN_TAG;
            ow_lo[1..7].copy_from_slice(&lo);
            let mut ow_hi = [0u8; 7];
            ow_hi[0] = DETACHED_OVERWRITTEN_TAG;
            ow_hi[1..7].copy_from_slice(&hi);
            batch.delete_range_default(&ow_lo, &ow_hi);
        }

        for (&attr_id, meta) in &self.dirty_detached_meta {
            batch.put_detached_meta(attr_id, meta);
        }
        for (&(attr_id, col_gen, seg, block), opt_bytes) in &self.dirty_detached_blocks {
            let key = detached_key(attr_id, col_gen, seg, block);
            match opt_bytes {
                Some(bytes) => batch.put_detached_block(&key, bytes),
                None => batch.delete_detached_block(&key),
            }
        }
        if !self.dirty_detached_postings.is_empty() {
            let keys: Vec<Vec<u8>> = self.dirty_detached_postings.keys().cloned().collect();
            let existing = self.storage.multi_get_detached_posting_slices(&keys)?;
            for ((key, delta_bm), opt_slice) in self.dirty_detached_postings.iter().zip(existing) {
                let mut merged = match opt_slice {
                    Some(slice) => SegPosting::decode(&slice)?,
                    None => SegPosting::default(),
                };
                merged.or_bitmap(delta_bm);
                batch.put_detached_posting(key, &merged.encode()?);
            }
        }
        if !self.dirty_detached_count_deltas.is_empty() {
            let mut deltas: Vec<(&Vec<u8>, i64)> = self
                .dirty_detached_count_deltas
                .iter()
                .filter(|&(_, &d)| d != 0)
                .map(|(k, &d)| (k, d))
                .collect();
            deltas.sort_unstable_by(|a, b| a.0.cmp(b.0));
            if !deltas.is_empty() {
                let db_keys: Vec<Vec<u8>> = deltas
                    .iter()
                    .map(|(suffix, _)| {
                        let mut k = Vec::with_capacity(1 + suffix.len());
                        k.push(DETACHED_COUNT_TAG);
                        k.extend_from_slice(suffix);
                        k
                    })
                    .collect();
                let current = self.storage.db.multi_get(db_keys.iter());
                for ((suffix, delta), cur) in deltas.iter().zip(current) {
                    let cur = match cur.map_err(map_db_err)? {
                        Some(bytes) => {
                            let arr: [u8; 8] = bytes.as_slice().try_into().map_err(|_| {
                                TreeError::CorruptedData(format!(
                                    "detached count expected 8 bytes, got {}",
                                    bytes.len()
                                ))
                            })?;
                            u64::from_le_bytes(arr) as i128
                        }
                        None => 0,
                    };
                    let next = cur + *delta as i128;
                    if next < 0 {
                        return Err(TreeError::InvariantViolation(format!(
                            "detached count underflow for suffix {suffix:?}: {cur} + {delta}"
                        )));
                    }
                    if next == 0 {
                        batch.delete_detached_count(suffix);
                    } else {
                        batch.put_detached_count(suffix, next as u64);
                    }
                }
            }
        }
        for (&(attr_id, col_gen, seg), bm) in &self.dirty_detached_overwritten {
            if bm.is_empty() {
                batch.delete_detached_overwritten(attr_id, col_gen, seg);
            } else {
                batch.put_detached_overwritten(attr_id, col_gen, seg, &encode_roaring_bitmap(bm)?);
            }
        }
        for &(seg, ref detached_gens, ref fts_attrs) in &self.retire_segments {
            if let Some(next_seg) = seg.checked_add(1) {
                let lo = seg.to_be_bytes();
                let hi = next_seg.to_be_bytes();
                for cf_name in [CF_ROWS, CF_ROW_DOC_IDS, CF_VECTORS, CF_COLUMNS] {
                    batch.delete_range_cf(cf_name, &lo, &hi);
                }
                let mut zlo = [0u8; 5];
                zlo[0] = ZONE_TAG;
                zlo[1..5].copy_from_slice(&lo);
                let mut zhi = [0u8; 5];
                zhi[0] = ZONE_TAG;
                zhi[1..5].copy_from_slice(&hi);
                batch.delete_range_default(&zlo, &zhi);

                for &(attr_id, col_gen) in detached_gens {
                    let a = attr_id.to_be_bytes();
                    let g = col_gen.to_be_bytes();
                    let mut dlo = [0u8; 10];
                    dlo[0..2].copy_from_slice(&a);
                    dlo[2..6].copy_from_slice(&g);
                    dlo[6..10].copy_from_slice(&lo);
                    let mut dhi = dlo;
                    dhi[6..10].copy_from_slice(&hi);
                    batch.delete_range_cf(CF_DETACHED, &dlo, &dhi);
                }

                for &attr_id in fts_attrs {
                    let a = attr_id.to_be_bytes();
                    let mut flo = [0u8; 7];
                    flo[0] = FTS_DOC_LEN_TAG;
                    flo[1..3].copy_from_slice(&a);
                    flo[3..7].copy_from_slice(&lo);
                    let mut fhi = flo;
                    fhi[3..7].copy_from_slice(&hi);
                    batch.delete_range_default(&flo, &fhi);
                }
            }
        }
        Ok(())
    }

    fn flush_caches_after_commit(&mut self) {
        if !self.dirty_detached_meta.is_empty() {
            let mut rguard = self
                .storage
                .retired_detached
                .write()
                .expect("retired_detached poisoned");
            let mut next_map = (**rguard).clone();
            for (&attr_id, meta) in &self.dirty_detached_meta {
                if meta.retired_below > 0 {
                    next_map.insert(attr_id, meta.retired_below);
                }
            }
            *rguard = Arc::new(next_map);
        }

        if let Some(dropped) = self.dirty_dropped_segments.take() {
            *self
                .storage
                .dropped_segments
                .write()
                .expect("dropped_segments poisoned") = Arc::new(dropped);
        }

        if !self.retire_segments.is_empty() {
            let mut zguard = self
                .storage
                .zone_cache
                .write()
                .expect("zone_cache poisoned");
            if let Some(ref mut table_arc) = *zguard {
                let map = Arc::make_mut(table_arc);
                for seg_map in map.values_mut() {
                    for &(seg, _, _) in &self.retire_segments {
                        seg_map.remove(&seg);
                    }
                }
            }
        }

        if !self.dirty_seg_meta.is_empty() || !self.dirty_seg_deletes.is_empty() {
            let dirty_meta = std::mem::take(&mut self.dirty_seg_meta);
            let mut dirty_del = std::mem::take(&mut self.dirty_seg_deletes);
            let mut retired_segs = Vec::new();
            self.storage.segments.mutate_and_publish(|snap| {
                for (seg, opt_meta) in dirty_meta {
                    match opt_meta {
                        Some(meta) => {
                            let del_arc = dirty_del
                                .remove(&seg)
                                .map(|bm| (!bm.is_empty()).then(|| Arc::new(bm)))
                                .unwrap_or_else(|| {
                                    snap.segments.get(&seg).and_then(|e| e.deleted.clone())
                                });
                            if meta.state == SegmentState::Retired {
                                retired_segs.push(seg);
                            }
                            snap.segments.insert(
                                seg,
                                SegmentEntry {
                                    meta,
                                    deleted: del_arc,
                                },
                            );
                        }
                        None => {
                            retired_segs.push(seg);
                            snap.segments.remove(&seg);
                        }
                    }
                }
                for (seg, bm) in dirty_del {
                    if let Some(entry) = snap.segments.get_mut(&seg) {
                        entry.deleted = (!bm.is_empty()).then(|| Arc::new(bm));
                    }
                }
            });
            if !retired_segs.is_empty() {
                let mut cache = self
                    .storage
                    .doc_id_block_cache
                    .write()
                    .expect("doc_id_block_cache poisoned");
                for &seg in &retired_segs {
                    cache.remove_segment(seg);
                }
                let mut posting_cache = self
                    .storage
                    .attr_posting_cache
                    .write()
                    .expect("attr_posting_cache poisoned");
                for seg in retired_segs {
                    posting_cache.remove_segment(seg);
                }
            }
        }

        if !self.dirty_doc_id_blocks.is_empty() {
            let mut cache = self
                .storage
                .doc_id_block_cache
                .write()
                .expect("doc_id_block_cache poisoned");
            for (key, blk) in std::mem::take(&mut self.dirty_doc_id_blocks) {
                cache.insert(key, Arc::from(blk));
            }
        }

        if !self.dirty_zone.is_empty() || !self.dirty_zone_deletes.is_empty() {
            let mut zguard = self
                .storage
                .zone_cache
                .write()
                .expect("zone_cache poisoned");
            if let Some(ref mut table_arc) = *zguard {
                let map = Arc::make_mut(table_arc);
                for (seg, attr_id, tag) in std::mem::take(&mut self.dirty_zone_deletes) {
                    if let Some(seg_map) = map.get_mut(&(attr_id, tag)) {
                        seg_map.remove(&seg);
                    }
                }
                for ((seg, attr_id, tag), entry) in std::mem::take(&mut self.dirty_zone) {
                    map.entry((attr_id, tag)).or_default().insert(seg, entry);
                }
            }
        }

        if !self.dirty_fts_doc_lens.is_empty() {
            let mut cache = self
                .storage
                .fts_doc_lens_cache
                .write()
                .expect("fts_doc_lens_cache poisoned");
            for ((attr_id, block), opt) in std::mem::take(&mut self.dirty_fts_doc_lens) {
                if let Some(table) = cache.get_mut(&attr_id) {
                    let blocks = Arc::make_mut(table);
                    match opt {
                        Some(bytes) => {
                            blocks.insert(block, bytes.into_boxed_slice());
                        }
                        None => {
                            blocks.remove(&block);
                        }
                    }
                }
            }
        }

        for &node_id in &self.dirty_internal {
            match self.internal_nodes.remove(&node_id).flatten() {
                Some(node) => {
                    self.internal_cache.insert(node_id, node);
                }
                None => {
                    self.internal_cache.remove(&node_id);
                }
            }
        }
        for &leaf_id in &self.dirty_leaf {
            match self.leaf_nodes.remove(&leaf_id).flatten() {
                Some(leaf) => {
                    if let Some(pid) = leaf.parent_id {
                        self.leaf_parent.insert(leaf_id, pid);
                    } else {
                        self.leaf_parent.remove(&leaf_id);
                    }
                }
                None => {
                    self.leaf_parent.remove(&leaf_id);
                }
            }
        }
    }
}
