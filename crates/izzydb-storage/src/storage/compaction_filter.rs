//! Orphan-cleanup compaction filters for segment-scoped keys.


use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, RwLock};

use roaring::RoaringBitmap;
use rocksdb::compaction_filter::{CompactionFilter, Decision};
use rocksdb::compaction_filter_factory::{CompactionFilterContext, CompactionFilterFactory};

use izzydb_core::config::{
    AttrId, ColumnGen, SegmentId,
};

use crate::codec::{
    DETACHED_COUNT_TAG, DETACHED_OVERWRITTEN_TAG,
    FTS_DOC_LEN_TAG, ZONE_TAG,
    split_detached_posting_key, split_posting_key,
};



pub(super) type FilterHandles<'a> = (
    &'a Arc<RwLock<Arc<RoaringBitmap>>>,
    &'a Arc<RwLock<Arc<HashMap<AttrId, ColumnGen>>>>,
);

#[derive(Clone, Copy)]
pub(super) enum OrphanFilterKind {
    DefaultCf,
    LeadingSeg,
    AttrPostings,
    DetachedBlocks,
    DetachedPostings,
    FtsTermSeg,
}

pub(super) struct OrphanCompactionFilter {
    kind: OrphanFilterKind,
    dropped: Arc<RoaringBitmap>,
    retired_detached: Arc<HashMap<AttrId, ColumnGen>>,
}

impl CompactionFilter for OrphanCompactionFilter {
    fn filter(&mut self, _level: u32, key: &[u8], _value: &[u8]) -> Decision {
        let remove = match self.kind {
            OrphanFilterKind::LeadingSeg => {
                key.len() >= 4
                    && self
                        .dropped
                        .contains(SegmentId::from_be_bytes([key[0], key[1], key[2], key[3]]))
            }
            OrphanFilterKind::AttrPostings => split_posting_key(key)
                .map(|(_, seg)| self.dropped.contains(seg))
                .unwrap_or(false),
            OrphanFilterKind::DetachedBlocks => {
                if key.len() != 14 {
                    false
                } else {
                    let attr_id = AttrId::from_be_bytes([key[0], key[1]]);
                    let col_gen = ColumnGen::from_be_bytes([key[2], key[3], key[4], key[5]]);
                    let seg = SegmentId::from_be_bytes([key[6], key[7], key[8], key[9]]);
                    self.retired_detached
                        .get(&attr_id)
                        .is_some_and(|&r| col_gen < r)
                        || self.dropped.contains(seg)
                }
            }
            OrphanFilterKind::DetachedPostings => {
                if key.len() < 11 {
                    false
                } else {
                    let attr_id = AttrId::from_be_bytes([key[0], key[1]]);
                    let col_gen = ColumnGen::from_be_bytes([key[2], key[3], key[4], key[5]]);
                    let retired = self
                        .retired_detached
                        .get(&attr_id)
                        .is_some_and(|&r| col_gen < r);
                    retired
                        || split_detached_posting_key(key)
                            .map(|(_, seg)| self.dropped.contains(seg))
                            .unwrap_or(false)
                }
            }
            OrphanFilterKind::FtsTermSeg => {
                key.len() >= 8
                    && self
                        .dropped
                        .contains(SegmentId::from_be_bytes([key[4], key[5], key[6], key[7]]))
            }
            OrphanFilterKind::DefaultCf => match key.first().copied() {
                Some(FTS_DOC_LEN_TAG) => {
                    key.len() >= 7
                        && self
                            .dropped
                            .contains(SegmentId::from_be_bytes([key[3], key[4], key[5], key[6]]))
                }
                Some(DETACHED_COUNT_TAG) => {
                    if key.len() < 7 {
                        false
                    } else {
                        let attr_id = AttrId::from_be_bytes([key[1], key[2]]);
                        let col_gen = ColumnGen::from_be_bytes([key[3], key[4], key[5], key[6]]);
                        self.retired_detached
                            .get(&attr_id)
                            .is_some_and(|&r| col_gen < r)
                    }
                }
                Some(ZONE_TAG) => {
                    key.len() >= 5
                        && self
                            .dropped
                            .contains(SegmentId::from_be_bytes([key[1], key[2], key[3], key[4]]))
                }
                Some(DETACHED_OVERWRITTEN_TAG) => {
                    if key.len() != 11 {
                        false
                    } else {
                        let attr_id = AttrId::from_be_bytes([key[1], key[2]]);
                        let col_gen = ColumnGen::from_be_bytes([key[3], key[4], key[5], key[6]]);
                        let seg = SegmentId::from_be_bytes([key[7], key[8], key[9], key[10]]);
                        self.retired_detached
                            .get(&attr_id)
                            .is_some_and(|&r| col_gen < r)
                            || self.dropped.contains(seg)
                    }
                }
                _ => false,
            },
        };
        if remove {
            Decision::Remove
        } else {
            Decision::Keep
        }
    }

    fn name(&self) -> &CStr {
        c"izzydb_orphan_filter"
    }
}

pub(super) struct OrphanCompactionFilterFactory {
    pub(super) kind: OrphanFilterKind,
    pub(super) dropped: Arc<RwLock<Arc<RoaringBitmap>>>,
    pub(super) retired_detached: Arc<RwLock<Arc<HashMap<AttrId, ColumnGen>>>>,
}

impl CompactionFilterFactory for OrphanCompactionFilterFactory {
    type Filter = OrphanCompactionFilter;

    fn create(&mut self, _context: CompactionFilterContext) -> Self::Filter {
        OrphanCompactionFilter {
            kind: self.kind,
            dropped: Arc::clone(&*self.dropped.read().expect("dropped_segments poisoned")),
            retired_detached: Arc::clone(
                &*self
                    .retired_detached
                    .read()
                    .expect("retired_detached poisoned"),
            ),
        }
    }

    fn name(&self) -> &CStr {
        c"izzydb_orphan_filter_factory"
    }
}
