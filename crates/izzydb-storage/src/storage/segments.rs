use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use izzydb_core::config::{RowId, SegmentId};
use izzydb_core::rowset::Liveness;
use roaring::RoaringBitmap;

use crate::codec::{SegmentMeta, SegmentState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentEntry {
    pub meta: SegmentMeta,
    pub deleted: Option<Arc<RoaringBitmap>>,
}

impl SegmentEntry {
    #[inline]
    pub fn is_visible(&self) -> bool {
        matches!(self.meta.state, SegmentState::Open | SegmentState::Sealed)
    }

    #[inline]
    pub fn is_forwarding(&self) -> bool {
        self.meta.state == SegmentState::Forwarding
    }

    #[inline]
    pub fn is_local_live(&self, local: u32) -> bool {
        if local >= self.meta.row_count {
            return false;
        }
        match &self.deleted {
            Some(del) => !del.contains(local),
            None => true,
        }
    }
}

/// Immutable point-in-time snapshot of all non-retired segments and their deletion bitmaps.
#[derive(Debug, Clone, Default)]
pub struct SegmentTableSnapshot {
    pub segments: BTreeMap<SegmentId, SegmentEntry>,
    /// Segments that are visible with deletions OR are in `Forwarding` state.
    pub has_deletes_or_forwarding: RoaringBitmap,
}

impl SegmentTableSnapshot {
    pub fn from_entries(entries: BTreeMap<SegmentId, SegmentEntry>) -> Self {
        let mut snap = Self {
            segments: entries,
            has_deletes_or_forwarding: RoaringBitmap::new(),
        };
        snap.rebuild_flags();
        snap
    }

    pub fn rebuild_flags(&mut self) {
        self.has_deletes_or_forwarding.clear();
        for (&seg, entry) in &self.segments {
            if entry.is_forwarding()
                || (entry.is_visible() && entry.meta.deleted_count > 0)
            {
                self.has_deletes_or_forwarding.insert(seg);
            }
        }
    }

    #[inline]
    pub fn get(&self, seg: SegmentId) -> Option<&SegmentEntry> {
        self.segments.get(&seg)
    }

    /// Resolve a [`RowId`] (possibly in a `Forwarding` segment from Phase 9 compaction) to a
    /// currently live [`RowId`], or `None` if dead or dropped.
    #[inline]
    pub fn resolve_row(&self, mut row: RowId) -> Option<RowId> {
        // Bounded loop in case a segment was forwarded multiple times before ANN vacuum caught up.
        for _ in 0..8 {
            let entry = self.segments.get(&row.seg())?;
            match entry.meta.state {
                SegmentState::Open | SegmentState::Sealed => {
                    return entry.is_local_live(row.local()).then_some(row);
                }
                SegmentState::Forwarding => {
                    let local = row.local();
                    if local >= entry.meta.row_count {
                        return None;
                    }
                    let rank = match &entry.deleted {
                        Some(del) => {
                            if del.contains(local) {
                                return None;
                            }
                            local - (del.rank(local) as u32)
                        }
                        None => local,
                    };
                    row = RowId::new(entry.meta.forward_target, entry.meta.forward_base + rank);
                }
                SegmentState::Building | SegmentState::Retired => return None,
            }
        }
        None
    }
}

impl Liveness for SegmentTableSnapshot {
    #[inline]
    fn is_live(&self, row: RowId) -> bool {
        let Some(entry) = self.segments.get(&row.seg()) else {
            return false;
        };
        entry.is_visible() && entry.is_local_live(row.local())
    }

    #[inline]
    fn is_visible_segment(&self, seg: SegmentId) -> bool {
        self.segments.get(&seg).is_some_and(SegmentEntry::is_visible)
    }

    #[inline]
    fn segment_row_count(&self, seg: SegmentId) -> Option<u32> {
        let entry = self.segments.get(&seg)?;
        entry.is_visible().then_some(entry.meta.row_count)
    }

    #[inline]
    fn segment_has_deletes(&self, seg: SegmentId) -> bool {
        self.segments
            .get(&seg)
            .is_some_and(|e| e.is_visible() && e.meta.deleted_count > 0)
    }

    #[inline]
    fn deleted_bitmap(&self, seg: SegmentId) -> Option<&RoaringBitmap> {
        let entry = self.segments.get(&seg)?;
        if !entry.is_visible() {
            return None;
        }
        entry.deleted.as_deref()
    }

    fn visible_segments(&self) -> Vec<SegmentId> {
        self.segments
            .iter()
            .filter_map(|(&seg, e)| e.is_visible().then_some(seg))
            .collect()
    }
}

/// Thread-safe, snapshot-isolated segment table held by `Storage`.
pub struct SegmentTable {
    inner: RwLock<Arc<SegmentTableSnapshot>>,
}

impl SegmentTable {
    pub fn new(snapshot: SegmentTableSnapshot) -> Self {
        Self {
            inner: RwLock::new(Arc::new(snapshot)),
        }
    }

    #[inline]
    pub fn snapshot(&self) -> Arc<SegmentTableSnapshot> {
        Arc::clone(&self.inner.read().expect("SegmentTable poisoned"))
    }

    pub fn publish(&self, snapshot: SegmentTableSnapshot) {
        let mut guard = self.inner.write().expect("SegmentTable poisoned");
        *guard = Arc::new(snapshot);
    }

    pub fn mutate_and_publish(&self, f: impl FnOnce(&mut SegmentTableSnapshot)) {
        let mut guard = self.inner.write().expect("SegmentTable poisoned");
        let mut next = (**guard).clone();
        f(&mut next);
        next.rebuild_flags();
        *guard = Arc::new(next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::SegmentMeta;

    #[test]
    fn snapshot_isolation_and_forwarding_resolution() {
        let mut entries = BTreeMap::new();
        let mut m0 = SegmentMeta::new_open(1);
        m0.state = SegmentState::Sealed;
        m0.row_count = 5;
        entries.insert(
            0,
            SegmentEntry {
                meta: m0,
                deleted: None,
            },
        );
        let table = SegmentTable::new(SegmentTableSnapshot::from_entries(entries));

        // Reader takes snapshot before mutation.
        let snap1 = table.snapshot();
        assert!(snap1.is_live(RowId::new(0, 1)));
        assert!(!snap1.segment_has_deletes(0));

        // Writer publishes a tombstone on (0, 1) and forwards seg 0 -> seg 1 (base 10).
        table.mutate_and_publish(|s| {
            let mut del = RoaringBitmap::new();
            del.insert(1);
            del.insert(3);
            let e0 = s.segments.get_mut(&0).unwrap();
            e0.meta.state = SegmentState::Forwarding;
            e0.meta.deleted_count = 2;
            e0.meta.forward_target = 1;
            e0.meta.forward_base = 10;
            e0.deleted = Some(Arc::new(del));

            let mut m1 = SegmentMeta::new_open(2);
            m1.state = SegmentState::Sealed;
            m1.row_count = 13;
            s.segments.insert(
                1,
                SegmentEntry {
                    meta: m1,
                    deleted: None,
                },
            );
        });

        // Old snapshot is unaffected.
        assert!(snap1.is_live(RowId::new(0, 1)));
        assert!(!snap1.has_deletes_or_forwarding.contains(0));

        // New snapshot resolves forwarding accurately:
        // seg 0 live rows: 0 -> (1, 10), 2 -> (1, 11), 4 -> (1, 12); 1 & 3 -> None.
        let snap2 = table.snapshot();
        assert!(snap2.has_deletes_or_forwarding.contains(0));
        assert_eq!(snap2.resolve_row(RowId::new(0, 0)), Some(RowId::new(1, 10)));
        assert_eq!(snap2.resolve_row(RowId::new(0, 1)), None);
        assert_eq!(snap2.resolve_row(RowId::new(0, 2)), Some(RowId::new(1, 11)));
        assert_eq!(snap2.resolve_row(RowId::new(0, 3)), None);
        assert_eq!(snap2.resolve_row(RowId::new(0, 4)), Some(RowId::new(1, 12)));
    }
}
