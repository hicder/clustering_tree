use std::collections::HashMap;

use izzydb_core::config::{RowId, SegmentId};
use izzydb_core::error::Result;
use izzydb_core::filter::Filter;
use izzydb_core::schema::{CollectionSchema, ColumnStorage, ScalarValue};
use izzydb_storage::Storage;
use izzydb_storage::codec::{
    SegPosting, encode_present_attr_key, encode_scalar_attr_key, make_posting_key,
};
use roaring::RoaringBitmap;

use crate::simd::LeafCandidate;

/// Query-local lookup for an inline equality predicate. Negation tests the same
/// posting and includes missing values without constructing a complement.
pub(super) struct PostingMembership {
    key: Vec<u8>,
    negated: bool,
    by_segment: HashMap<SegmentId, RoaringBitmap>,
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
            by_segment: HashMap::new(),
        }))
    }

    /// Candidates must already be live and resolved through segment forwarding.
    pub(super) fn retain_matches(
        &mut self,
        storage: &Storage,
        candidates: &mut Vec<LeafCandidate>,
    ) -> Result<()> {
        let mut missing: Vec<SegmentId> = candidates
            .iter()
            .map(|candidate| RowId(candidate.doc_id).seg())
            .filter(|seg| !self.by_segment.contains_key(seg))
            .collect();
        missing.sort_unstable();
        missing.dedup();
        if !missing.is_empty() {
            let keys: Vec<_> = missing
                .iter()
                .map(|&seg| make_posting_key(&self.key, seg))
                .collect();
            let slices = storage.multi_get_attr_posting_slices(&keys)?;
            for (seg, slice) in missing.into_iter().zip(slices) {
                let posting = match slice {
                    Some(bytes) => SegPosting::decode(&bytes)?.into_bitmap(),
                    None => RoaringBitmap::new(),
                };
                self.by_segment.insert(seg, posting);
            }
        }
        candidates.retain(|candidate| {
            let row = RowId(candidate.doc_id);
            self.by_segment[&row.seg()].contains(row.local()) ^ self.negated
        });
        Ok(())
    }
}
