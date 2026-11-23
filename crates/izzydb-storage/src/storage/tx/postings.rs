//! Attribute postings, zones, and counts (insert-only deltas).



use izzydb_core::config::{
    AttrId, LocalId, SegmentId,
};
use izzydb_core::error::Result;

use crate::codec::ZoneEntry;


use super::*;

impl <'a> TxContext<'a> {
    // -----------------------------------------------------------------------
    // Attribute postings, zones, and counts (insert-only deltas)
    // -----------------------------------------------------------------------

    pub fn add_posting(&mut self, posting_key: Vec<u8>, local: LocalId) {
        self.dirty_postings
            .entry(posting_key)
            .or_default()
            .insert(local);
    }

    pub fn widen_zone(
        &mut self,
        seg: SegmentId,
        attr_id: AttrId,
        tag: u8,
        encoded_val: &[u8],
    ) -> Result<()> {
        let key = (seg, attr_id, tag);
        if let Some(entry) = self.dirty_zone.get_mut(&key) {
            entry.widen(encoded_val);
            return Ok(());
        }
        let zones = self.storage.get_zones_cached()?;
        let mut entry = zones
            .get(&(attr_id, tag))
            .and_then(|m| m.get(&seg))
            .cloned()
            .unwrap_or_else(|| ZoneEntry::singleton(encoded_val));
        entry.widen(encoded_val);
        self.dirty_zone.insert(key, entry);
        Ok(())
    }

    pub fn adjust_attr_count(&mut self, attr_key: &[u8], delta: i64) {
        if delta == 0 {
            return;
        }
        *self
            .dirty_attr_count_deltas
            .entry(attr_key.to_vec())
            .or_insert(0) += delta;
    }
}
