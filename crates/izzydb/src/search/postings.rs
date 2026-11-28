use std::sync::Arc;

use izzydb_core::config::{RowId, SegmentId};
use izzydb_core::error::Result;
use izzydb_core::filter::Filter;
use izzydb_core::schema::{CollectionSchema, ColumnStorage, ScalarValue};
use izzydb_storage::Storage;
use izzydb_storage::codec::{
    SegBitsetAccum, SegPostingView, SegmentState, encode_present_attr_key, encode_scalar_attr_key,
    make_posting_key,
};
use roaring::RoaringBitmap;

use crate::simd::LeafCandidate;

/// Query-local lookup for an inline equality predicate. Negation tests the same
/// posting and includes missing values without constructing a complement.
///
/// Sealed-segment postings come from the bounded [`Storage`] posting cache when possible;
/// open-segment postings are decoded once and held in a dense `Vec` indexed by segment.
pub(super) struct PostingMembership {
    key: Vec<u8>,
    negated: bool,
    by_segment: Vec<Option<Arc<RoaringBitmap>>>,
}

impl PostingMembership {
    pub(super) fn for_filter(schema: &CollectionSchema, filter: &Filter) -> Result<Option<Self>> {
        let (attr, value, negated) = match filter {
            Filter::Eq(attr, value) => (attr, value, false),
            Filter::NotEq(attr, value) => (attr, value, true),
            Filter::Not(child) => match child.as_ref() {
                Filter::Eq(attr, value) => (attr, value, true),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        let Some(entry) = schema.get_by_name(attr) else {
            return Ok(None);
        };
        if entry.storage != ColumnStorage::Inline {
            return Ok(None);
        }
        let (key, negated) = match value {
            ScalarValue::Null => (encode_present_attr_key(entry.id), !negated),
            _ => (encode_scalar_attr_key(entry.id, value)?, negated),
        };
        Ok(Some(Self {
            key,
            negated,
            by_segment: Vec::new(),
        }))
    }

    /// Candidates must already be live and resolved through segment forwarding.
    pub(super) fn retain_matches(
        &mut self,
        storage: &Storage,
        candidates: &mut Vec<LeafCandidate>,
    ) -> Result<()> {
        let mut max_seg = 0usize;
        let mut missing: Vec<SegmentId> = Vec::new();
        for candidate in candidates.iter() {
            let seg = RowId(candidate.doc_id).seg();
            max_seg = max_seg.max(seg as usize);
            if self
                .by_segment
                .get(seg as usize)
                .is_none_or(Option::is_none)
            {
                missing.push(seg);
            }
        }
        missing.sort_unstable();
        missing.dedup();

        if !missing.is_empty() {
            self.by_segment.resize(max_seg + 1, None);
            let snap = storage.segment_snapshot();

            // Serve immutable sealed-segment postings from the bounded cache.
            let mut uncached: Vec<SegmentId> = Vec::new();
            for &seg in &missing {
                let posting_key = make_posting_key(&self.key, seg);
                if let Some(bitmap) = storage.cached_attr_posting(&posting_key) {
                    self.by_segment[seg as usize] = Some(bitmap);
                } else {
                    uncached.push(seg);
                }
            }

            // Fetch all remaining keys with one flat scratch buffer.
            let slices = storage.multi_get_attr_posting_slices_for_segs(&self.key, &uncached)?;
            for (&seg, slice) in uncached.iter().zip(slices) {
                let bitmap = match slice {
                    Some(bytes) => {
                        let view = SegPostingView::parse(&bytes)?;
                        let mut acc = SegBitsetAccum::new();
                        view.or_into_accum(&mut acc)?;
                        Arc::new(acc.into_roaring())
                    }
                    None => Arc::new(RoaringBitmap::new()),
                };
                let sealed = snap
                    .get(seg)
                    .is_some_and(|entry| entry.meta.state == SegmentState::Sealed);
                if sealed {
                    storage
                        .cache_attr_posting(make_posting_key(&self.key, seg), Arc::clone(&bitmap));
                }
                self.by_segment[seg as usize] = Some(bitmap);
            }
        }

        candidates.retain(|candidate| {
            let row = RowId(candidate.doc_id);
            self.by_segment
                .get(row.seg() as usize)
                .and_then(|entry| entry.as_ref())
                .is_some_and(|bitmap| bitmap.contains(row.local()))
                ^ self.negated
        });
        Ok(())
    }
}
