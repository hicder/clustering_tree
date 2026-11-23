//! Segment table and tombstone overlay.


use roaring::RoaringBitmap;

use izzydb_core::config::{
    AttrId, ColumnGen, RowId, SegmentId,
};
use izzydb_core::error::{Result, TreeError};

use crate::codec::{
    SegmentMeta, SegmentState,
};


use super::*;

impl <'a> TxContext<'a> {
    // -----------------------------------------------------------------------
    // Segment table & tombstones
    // -----------------------------------------------------------------------

    pub fn get_seg_meta(&self, seg: SegmentId) -> Option<SegmentMeta> {
        if let Some(opt) = self.dirty_seg_meta.get(&seg) {
            return opt.clone();
        }
        self.storage
            .segments
            .snapshot()
            .get(seg)
            .map(|e| e.meta.clone())
    }

    pub fn put_seg_meta(&mut self, seg: SegmentId, meta: SegmentMeta) {
        self.dirty_seg_meta.insert(seg, Some(meta));
    }

    pub fn delete_seg_meta(&mut self, seg: SegmentId) {
        self.dirty_seg_meta.insert(seg, None);
    }

    pub fn tombstone_row(&mut self, row: RowId) -> Result<()> {
        let seg = row.seg();
        let local = row.local();
        let mut meta = self.get_seg_meta(seg).ok_or_else(|| {
            TreeError::CorruptedData(format!("tombstone for unknown segment {seg}"))
        })?;
        if local >= meta.row_count {
            return Err(TreeError::CorruptedData(format!(
                "tombstone local {local} >= segment {seg} row_count {}",
                meta.row_count
            )));
        }

        if !self.dirty_seg_deletes.contains_key(&seg) {
            let existing = self
                .storage
                .segments
                .snapshot()
                .get(seg)
                .and_then(|e| e.deleted.as_ref().map(|arc| (**arc).clone()))
                .unwrap_or_default();
            self.dirty_seg_deletes.insert(seg, existing);
        }
        let bm = self.dirty_seg_deletes.get_mut(&seg).unwrap();
        if !bm.insert(local) {
            return Err(TreeError::InvariantViolation(format!(
                "double tombstone on {row:?}"
            )));
        }
        meta.deleted_count += 1;
        self.dirty_seg_meta.insert(seg, Some(meta));
        Ok(())
    }

    pub fn mark_segment_dropped(&mut self, seg: SegmentId) {
        if self.dirty_dropped_segments.is_none() {
            self.dirty_dropped_segments =
                Some((*self.storage.dropped_segments_snapshot()).clone());
        }
        self.dirty_dropped_segments.as_mut().unwrap().insert(seg);
        self.dirty_seg_meta.insert(seg, None);
        self.dirty_seg_deletes.remove(&seg);
    }

    /// Fully retire a `Forwarding` segment: mark dropped in `\x01dropped_segments`, delete
    /// `0x10`/`0x11` metadata, and issue `DeleteRange` over every segment-major keyspace.
    pub fn retire_segment(
        &mut self,
        seg: SegmentId,
        detached_gens: Vec<(AttrId, ColumnGen)>,
        fts_attrs: Vec<AttrId>,
    ) {
        self.mark_segment_dropped(seg);
        for &(attr_id, col_gen) in &detached_gens {
            self.dirty_detached_overwritten
                .insert((attr_id, col_gen, seg), RoaringBitmap::new());
        }
        self.retire_segments.push((seg, detached_gens, fts_attrs));
    }

    /// Resolve a [`RowId`] against uncommitted `TxContext` mutations and the base `SegmentTable`,
    /// returning `None` if tombstoned, dropped, retired, or out of range, or the forwarded live [`RowId`].
    pub fn resolve_row(&self, mut row: RowId) -> Option<RowId> {
        let snap = self.storage.segments.snapshot();
        for _ in 0..8 {
            let seg = row.seg();
            if let Some(ref dropped) = self.dirty_dropped_segments
                && dropped.contains(seg)
            {
                return None;
            }
            let meta = match self.dirty_seg_meta.get(&seg) {
                Some(Some(m)) => m,
                Some(None) => return None,
                None => &snap.get(seg)?.meta,
            };
            let local = row.local();
            if local >= meta.row_count {
                return None;
            }
            let deleted_bm = self
                .dirty_seg_deletes
                .get(&seg)
                .or_else(|| snap.get(seg).and_then(|e| e.deleted.as_deref()));
            if deleted_bm.is_some_and(|bm| bm.contains(local)) {
                return None;
            }
            match meta.state {
                SegmentState::Open | SegmentState::Sealed => return Some(row),
                SegmentState::Forwarding => {
                    let del_rank = deleted_bm.map_or(0, |bm| bm.rank(local) as u32);
                    let rank = local - del_rank;
                    row = RowId::new(meta.forward_target, meta.forward_base + rank);
                }
                SegmentState::Building | SegmentState::Retired => return None,
            }
        }
        None
    }
}
