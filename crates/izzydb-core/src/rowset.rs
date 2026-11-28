use std::cmp::Ordering;
use std::collections::BTreeMap;

use roaring::RoaringBitmap;

use crate::config::{ROW_BLOCK_SIZE, RowId, SegmentId};

/// Segment-level liveness oracle used by filters, FTS, ANN masking, facets, and ranking.
pub trait Liveness {
    /// Returns `true` if `row` is in a visible (`Open` or `Sealed`) segment, `< row_count`,
    /// and not tombstoned.
    fn is_live(&self, row: RowId) -> bool;

    /// Returns `true` if `seg` is visible (`Open` or `Sealed`).
    fn is_visible_segment(&self, seg: SegmentId) -> bool;

    /// Number of allocated rows (`high-water LocalId + 1`) in `seg`, if visible.
    fn segment_row_count(&self, seg: SegmentId) -> Option<u32>;

    /// Returns `true` if `seg` has any tombstones (fast path: `false` skips bitmap checks).
    fn segment_has_deletes(&self, seg: SegmentId) -> bool;

    /// Tombstone bitmap of `seg`, if `seg` is visible and has deletions.
    fn deleted_bitmap(&self, seg: SegmentId) -> Option<&RoaringBitmap>;

    /// Full bitmap of live `LocalId`s in `seg`, if `seg` is visible.
    fn live_bitmap(&self, seg: SegmentId) -> Option<RoaringBitmap> {
        let count = self.segment_row_count(seg)?;
        let mut bm = RoaringBitmap::new();
        bm.insert_range(0..count);
        if let Some(del) = self.deleted_bitmap(seg) {
            bm -= del;
        }
        Some(bm)
    }

    /// 256-bit liveness mask for `(seg, block)` where bit `slot` is 1 iff `RowId(seg, block*256 + slot)`
    /// is live. Returns `None` when `seg` is visible, has no deletions, and the entire block is within
    /// `row_count` (fast path: all 256 slots are live).
    fn block_live_mask(&self, seg: SegmentId, block: u32) -> Option<[u64; 4]> {
        let Some(count) = self.segment_row_count(seg) else {
            return Some([0u64; 4]);
        };
        let base = block.saturating_mul(ROW_BLOCK_SIZE as u32);
        if base >= count {
            return Some([0u64; 4]);
        }
        let valid = (count - base).min(ROW_BLOCK_SIZE as u32);
        let has_deletes = self.segment_has_deletes(seg);
        if !has_deletes && valid == ROW_BLOCK_SIZE as u32 {
            return None;
        }
        let mut mask = if valid == ROW_BLOCK_SIZE as u32 {
            [u64::MAX; 4]
        } else {
            let mut m = [0u64; 4];
            let full_words = (valid / 64) as usize;
            for w in m.iter_mut().take(full_words) {
                *w = u64::MAX;
            }
            let rem = valid % 64;
            if rem > 0 {
                m[full_words] = (1u64 << rem) - 1;
            }
            m
        };
        if let Some(del) = self.deleted_bitmap(seg) {
            let end = base + valid;
            for local in del.range(base..end) {
                let slot = (local - base) as usize;
                mask[slot >> 6] &= !(1u64 << (slot & 63));
            }
        }
        Some(mask)
    }

    /// Visible segment ids in ascending order.
    fn visible_segments(&self) -> Vec<SegmentId>;
}

/// Segment-partitioned set of [`RowId`]s represented as sorted `(SegmentId, RoaringBitmap)` pairs
/// with no empty bitmaps.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowSet {
    segs: Vec<(SegmentId, RoaringBitmap)>,
}

impl RowSet {
    #[inline]
    pub fn empty() -> Self {
        Self { segs: Vec::new() }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    #[inline]
    pub fn num_segments(&self) -> usize {
        self.segs.len()
    }

    pub fn len(&self) -> u64 {
        self.segs.iter().map(|(_, bm)| bm.len()).sum()
    }

    #[inline]
    pub fn segments(&self) -> &[(SegmentId, RoaringBitmap)] {
        &self.segs
    }

    #[inline]
    pub fn into_segments(self) -> Vec<(SegmentId, RoaringBitmap)> {
        self.segs
    }

    pub fn segment(&self, seg: SegmentId) -> Option<&RoaringBitmap> {
        let idx = self.segs.binary_search_by_key(&seg, |(s, _)| *s).ok()?;
        Some(&self.segs[idx].1)
    }

    /// Build a [`RowSet`] from `(SegmentId, RoaringBitmap)` pairs, sorting by `SegmentId`,
    /// merging duplicate segment entries, and dropping empty bitmaps.
    pub fn from_segments(mut pairs: Vec<(SegmentId, RoaringBitmap)>) -> Self {
        pairs.retain(|(_, bm)| !bm.is_empty());
        if pairs.is_empty() {
            return Self::empty();
        }
        pairs.sort_by_key(|(seg, _)| *seg);
        let mut out: Vec<(SegmentId, RoaringBitmap)> = Vec::with_capacity(pairs.len());
        for (seg, bm) in pairs {
            if let Some((last_seg, last_bm)) = out.last_mut()
                && *last_seg == seg
            {
                *last_bm |= bm;
            } else {
                out.push((seg, bm));
            }
        }
        Self { segs: out }
    }

    /// Build a [`RowSet`] directly from already strictly ascending, non-empty `(SegmentId, RoaringBitmap)` pairs.
    pub fn from_sorted_segs(segs: Vec<(SegmentId, RoaringBitmap)>) -> Self {
        let segs = segs
            .into_iter()
            .filter(|(_, bm)| !bm.is_empty())
            .collect::<Vec<_>>();
        debug_assert!(segs.windows(2).all(|w| w[0].0 < w[1].0));
        Self { segs }
    }

    /// Build a [`RowSet`] from an arbitrary iterator of [`RowId`]s.
    pub fn from_rows(rows: impl IntoIterator<Item = RowId>) -> Self {
        let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
        for row in rows {
            by_seg.entry(row.seg()).or_default().insert(row.local());
        }
        Self {
            segs: by_seg
                .into_iter()
                .filter(|(_, bm)| !bm.is_empty())
                .collect(),
        }
    }

    /// Build a [`RowSet`] containing every live row in `liveness`.
    pub fn from_all_live<L: Liveness + ?Sized>(liveness: &L) -> Self {
        let mut segs = Vec::new();
        for seg in liveness.visible_segments() {
            if let Some(bm) = liveness.live_bitmap(seg)
                && !bm.is_empty()
            {
                segs.push((seg, bm));
            }
        }
        Self { segs }
    }

    pub fn insert(&mut self, row: RowId) -> bool {
        let seg = row.seg();
        let local = row.local();
        match self.segs.binary_search_by_key(&seg, |(s, _)| *s) {
            Ok(idx) => self.segs[idx].1.insert(local),
            Err(idx) => {
                let mut bm = RoaringBitmap::new();
                bm.insert(local);
                self.segs.insert(idx, (seg, bm));
                true
            }
        }
    }

    pub fn remove(&mut self, row: RowId) -> bool {
        let seg = row.seg();
        let Ok(idx) = self.segs.binary_search_by_key(&seg, |(s, _)| *s) else {
            return false;
        };
        let removed = self.segs[idx].1.remove(row.local());
        if self.segs[idx].1.is_empty() {
            self.segs.remove(idx);
        }
        removed
    }

    pub fn contains(&self, row: RowId) -> bool {
        let Ok(idx) = self.segs.binary_search_by_key(&row.seg(), |(s, _)| *s) else {
            return false;
        };
        self.segs[idx].1.contains(row.local())
    }

    pub fn and(&self, other: &RowSet) -> RowSet {
        let mut out = Vec::new();
        let mut i = 0;
        let mut j = 0;
        while i < self.segs.len() && j < other.segs.len() {
            match self.segs[i].0.cmp(&other.segs[j].0) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    let inter = &self.segs[i].1 & &other.segs[j].1;
                    if !inter.is_empty() {
                        out.push((self.segs[i].0, inter));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        RowSet { segs: out }
    }

    pub fn or(&self, other: &RowSet) -> RowSet {
        let mut out = Vec::with_capacity(self.segs.len().max(other.segs.len()));
        let mut i = 0;
        let mut j = 0;
        while i < self.segs.len() && j < other.segs.len() {
            match self.segs[i].0.cmp(&other.segs[j].0) {
                Ordering::Less => {
                    out.push(self.segs[i].clone());
                    i += 1;
                }
                Ordering::Greater => {
                    out.push(other.segs[j].clone());
                    j += 1;
                }
                Ordering::Equal => {
                    let union = &self.segs[i].1 | &other.segs[j].1;
                    if !union.is_empty() {
                        out.push((self.segs[i].0, union));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < self.segs.len() {
            out.push(self.segs[i].clone());
            i += 1;
        }
        while j < other.segs.len() {
            out.push(other.segs[j].clone());
            j += 1;
        }
        RowSet { segs: out }
    }

    pub fn and_not(&self, other: &RowSet) -> RowSet {
        let mut out = Vec::with_capacity(self.segs.len());
        let mut i = 0;
        let mut j = 0;
        while i < self.segs.len() && j < other.segs.len() {
            match self.segs[i].0.cmp(&other.segs[j].0) {
                Ordering::Less => {
                    out.push(self.segs[i].clone());
                    i += 1;
                }
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    let diff = &self.segs[i].1 - &other.segs[j].1;
                    if !diff.is_empty() {
                        out.push((self.segs[i].0, diff));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < self.segs.len() {
            out.push(self.segs[i].clone());
            i += 1;
        }
        RowSet { segs: out }
    }

    pub fn and_live<L: Liveness + ?Sized>(&self, liveness: &L) -> RowSet {
        let mut out = Vec::with_capacity(self.segs.len());
        for (seg, bm) in &self.segs {
            let Some(row_count) = liveness.segment_row_count(*seg) else {
                continue;
            };
            if row_count == 0 {
                continue;
            }
            let mut filtered = if let Some(del) = liveness.deleted_bitmap(*seg) {
                bm - del
            } else {
                bm.clone()
            };
            if filtered.max().is_some_and(|m| m >= row_count) {
                filtered.remove_range(row_count..);
            }
            if !filtered.is_empty() {
                out.push((*seg, filtered));
            }
        }
        RowSet { segs: out }
    }

    /// Consuming liveness filter that moves bitmaps in place when no tombstones apply.
    pub fn into_live<L: Liveness + ?Sized>(self, liveness: &L) -> RowSet {
        let mut out = Vec::with_capacity(self.segs.len());
        for (seg, mut bm) in self.segs {
            let Some(row_count) = liveness.segment_row_count(seg) else {
                continue;
            };
            if row_count == 0 {
                continue;
            }
            if let Some(del) = liveness.deleted_bitmap(seg) {
                bm -= del;
            }
            if bm.max().is_some_and(|m| m >= row_count) {
                bm.remove_range(row_count..);
            }
            if !bm.is_empty() {
                out.push((seg, bm));
            }
        }
        RowSet { segs: out }
    }

    pub fn iter(&self) -> impl Iterator<Item = RowId> + '_ {
        self.segs
            .iter()
            .flat_map(|(seg, bm)| bm.iter().map(move |local| RowId::new(*seg, local)))
    }

    /// 256-slot bitmask for `(seg, block)` where bit `slot` is 1 iff `RowId(seg, block*256 + slot)`
    /// is in this set.
    pub fn block_mask(&self, seg: SegmentId, block: u32) -> [u64; 4] {
        let Some(bm) = self.segment(seg) else {
            return [0u64; 4];
        };
        let Some(base) = block.checked_mul(ROW_BLOCK_SIZE as u32) else {
            return [0u64; 4];
        };
        let end = base.saturating_add(ROW_BLOCK_SIZE as u32);
        let mut mask = [0u64; 4];
        for local in bm.range(base..end) {
            let slot = (local - base) as usize;
            mask[slot >> 6] |= 1u64 << (slot & 63);
        }
        mask
    }

    /// Build a cursor for probing sorted [`RowId`]s with galloping over segment entries.
    #[inline]
    pub fn cursor(&self) -> RowSetCursor<'_> {
        RowSetCursor {
            segs: &self.segs,
            idx: 0,
        }
    }
}

/// Dense segment-indexed lookup over a [`RowSet`] for repeated [`RowId`] probes.
///
/// Avoids the 153-way binary search in [`RowSet::contains`] during leaf-slot masking.
///
/// ```text
/// let lookup = FastRowSetLookup::new(&rowset);
/// assert!(lookup.contains(RowId::new(7, 42)) == rowset.contains(RowId::new(7, 42)));
/// ```
pub struct FastRowSetLookup<'a> {
    by_seg: Vec<Option<&'a RoaringBitmap>>,
}

impl<'a> FastRowSetLookup<'a> {
    pub fn new(rowset: &'a RowSet) -> Self {
        let max_seg = rowset.segments().last().map_or(0, |(seg, _)| *seg as usize);
        let mut by_seg = vec![None; max_seg + 1];
        for (seg, bm) in rowset.segments() {
            by_seg[*seg as usize] = Some(bm);
        }
        Self { by_seg }
    }

    #[inline]
    pub fn contains(&self, row: RowId) -> bool {
        self.by_seg
            .get(row.seg() as usize)
            .and_then(|entry| *entry)
            .is_some_and(|bm| bm.contains(row.local()))
    }
}

/// Galloping cursor over a [`RowSet`] for sorted [`RowId`] probes.
pub struct RowSetCursor<'a> {
    segs: &'a [(SegmentId, RoaringBitmap)],
    idx: usize,
}

impl<'a> RowSetCursor<'a> {
    #[inline]
    pub fn contains_sorted(&mut self, row: RowId) -> bool {
        let target_seg = row.seg();
        if self.idx >= self.segs.len() {
            return false;
        }
        if self.segs[self.idx].0 < target_seg {
            // Exponential gallop, then binary search.
            let mut step = 1usize;
            let mut lo = self.idx + 1;
            while lo < self.segs.len() && self.segs[lo].0 < target_seg {
                self.idx = lo;
                lo = (lo + step).min(self.segs.len());
                step <<= 1;
            }
            let hi = lo.min(self.segs.len());
            let rel = self.segs[self.idx..hi].partition_point(|(s, _)| *s < target_seg);
            self.idx += rel;
            if self.idx >= self.segs.len() {
                return false;
            }
        }
        if self.segs[self.idx].0 != target_seg {
            return false;
        }
        self.segs[self.idx].1.contains(row.local())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    struct MockLiveness {
        segments: BTreeMap<SegmentId, (u32, RoaringBitmap)>,
    }

    impl Liveness for MockLiveness {
        fn is_live(&self, row: RowId) -> bool {
            let Some((count, del)) = self.segments.get(&row.seg()) else {
                return false;
            };
            row.local() < *count && !del.contains(row.local())
        }

        fn is_visible_segment(&self, seg: SegmentId) -> bool {
            self.segments.contains_key(&seg)
        }

        fn segment_row_count(&self, seg: SegmentId) -> Option<u32> {
            self.segments.get(&seg).map(|(c, _)| *c)
        }

        fn segment_has_deletes(&self, seg: SegmentId) -> bool {
            self.segments
                .get(&seg)
                .is_some_and(|(_, del)| !del.is_empty())
        }

        fn deleted_bitmap(&self, seg: SegmentId) -> Option<&RoaringBitmap> {
            self.segments
                .get(&seg)
                .and_then(|(_, del)| (!del.is_empty()).then_some(del))
        }

        fn visible_segments(&self) -> Vec<SegmentId> {
            self.segments.keys().copied().collect()
        }
    }

    fn model_to_rowset(model: &BTreeSet<RowId>) -> RowSet {
        RowSet::from_rows(model.iter().copied())
    }

    fn rowset_to_model(rs: &RowSet) -> BTreeSet<RowId> {
        rs.iter().collect()
    }

    #[test]
    fn rowset_algebra_matches_btreeset_model() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(0x5EED_CAFE);
        let mut live_segs = BTreeMap::new();
        for seg in [0u32, 2, 5, 6] {
            let count = rng.random_range(100u32..700);
            let mut del = RoaringBitmap::new();
            for _ in 0..(count / 4) {
                del.insert(rng.random_range(0..count));
            }
            live_segs.insert(seg, (count, del));
        }
        let liveness = MockLiveness {
            segments: live_segs,
        };

        for _ in 0..50 {
            let mut m_a = BTreeSet::new();
            let mut m_b = BTreeSet::new();
            for _ in 0..200 {
                let seg = rng.random_range(0u32..8);
                let local = rng.random_range(0u32..700);
                m_a.insert(RowId::new(seg, local));
            }
            for _ in 0..200 {
                let seg = rng.random_range(0u32..8);
                let local = rng.random_range(0u32..700);
                m_b.insert(RowId::new(seg, local));
            }

            let rs_a = model_to_rowset(&m_a);
            let rs_b = model_to_rowset(&m_b);

            assert_eq!(rs_a.len(), m_a.len() as u64);
            assert_eq!(rowset_to_model(&rs_a), m_a);
            assert_eq!(
                rowset_to_model(&rs_a.and(&rs_b)),
                m_a.intersection(&m_b).copied().collect()
            );
            assert_eq!(
                rowset_to_model(&rs_a.or(&rs_b)),
                m_a.union(&m_b).copied().collect()
            );
            assert_eq!(
                rowset_to_model(&rs_a.and_not(&rs_b)),
                m_a.difference(&m_b).copied().collect()
            );

            let live_model: BTreeSet<RowId> = m_a
                .iter()
                .copied()
                .filter(|&r| liveness.is_live(r))
                .collect();
            assert_eq!(rowset_to_model(&rs_a.and_live(&liveness)), live_model);
            assert_eq!(
                rowset_to_model(&rs_a.clone().into_live(&liveness)),
                live_model
            );

            // Check block_mask and cursor
            let lookup = FastRowSetLookup::new(&rs_a);
            let mut cursor = rs_a.cursor();
            for seg in 0u32..8 {
                for blk in 0u32..3 {
                    let mask = rs_a.block_mask(seg, blk);
                    for slot in 0u32..256 {
                        let row = RowId::new(seg, blk * 256 + slot);
                        let bit = ((mask[(slot >> 6) as usize] >> (slot & 63)) & 1) != 0;
                        assert_eq!(bit, m_a.contains(&row));
                        assert_eq!(rs_a.contains(row), m_a.contains(&row));
                        assert_eq!(lookup.contains(row), m_a.contains(&row));
                        assert_eq!(cursor.contains_sorted(row), m_a.contains(&row));
                    }
                }
            }
        }
    }

    #[test]
    fn liveness_block_mask_fast_and_slow_paths() {
        let mut segs = BTreeMap::new();
        segs.insert(0, (512, RoaringBitmap::new())); // full blocks, no deletes
        let mut del1 = RoaringBitmap::new();
        del1.insert(5);
        del1.insert(260);
        segs.insert(1, (300, del1)); // partial tail + deletes
        segs.insert(2, (300, RoaringBitmap::new())); // partial tail, no deletes
        let liveness = MockLiveness { segments: segs };

        assert_eq!(liveness.block_live_mask(0, 0), None);
        assert_eq!(liveness.block_live_mask(0, 1), None);
        assert_eq!(liveness.block_live_mask(0, 2), Some([0u64; 4]));

        let m1_0 = liveness.block_live_mask(1, 0).unwrap();
        assert_eq!(m1_0[0] & (1 << 5), 0);
        assert_ne!(m1_0[0] & (1 << 4), 0);
        let m1_1 = liveness.block_live_mask(1, 1).unwrap();
        assert_eq!(m1_1[0] & (1 << 4), 0); // 256 + 4 = 260 deleted
        assert_ne!(m1_1[0] & (1 << 43), 0); // 256 + 43 = 299 < 300
        assert_eq!(m1_1[0] & (1 << 44), 0); // 256 + 44 = 300 >= 300

        let m2_0 = liveness.block_live_mask(2, 0);
        assert_eq!(m2_0, None);
        let m2_1 = liveness.block_live_mask(2, 1).unwrap();
        assert_ne!(m2_1[0] & (1 << 43), 0);
        assert_eq!(m2_1[0] & (1 << 44), 0);

        let all_live = RowSet::from_all_live(&liveness);
        let expected_total: u64 = (512 + 298 + 300) as u64;
        assert_eq!(all_live.len(), expected_total);
        let _ = HashMap::<SegmentId, u32>::new();
    }
}
