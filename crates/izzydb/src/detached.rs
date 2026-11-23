//! Phase 8: Detached columns (`cf_detached`, `cf_detached_postings`, `default[0x07/0x13/0x14]`),
//! atomic whole-column build/publish ingest, point overwrites, and CoW carry-over.

use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, ColumnGen, DocId, ROW_BLOCK_SIZE, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::rowset::Liveness;
use izzydb_core::schema::{
    AttributeType, AttributeValue, Attributes, CollectionSchema, ColumnStorage, ScalarValue,
    SchemaEntry,
};
use izzydb_storage::codec::{
    RowView, SegPosting, TAG_FLOAT, TAG_INT, TAG_STRING, TAG_UINT, ZoneEntry,
    detached_count_suffix, detached_key, detached_posting_key, extract_index_keys_for_attr,
    ordered_type_tag,
};
use izzydb_storage::column::ColumnBlockBuilder;
use izzydb_storage::{ColumnBlockView, SegmentTableSnapshot, Storage, TxContext};

use crate::collection::Collection;
use crate::fts_index::FtsIndex;
use crate::segments::RowResolver;

const DEFAULT_SST_BLOCK_THRESHOLD: usize = 64;

/// How whole-column [`DetachedIngestBuilder`] treats existing values on documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetachedWriteMode {
    /// Replace the entire column: live documents omitted from input become `Null`.
    #[default]
    Replace,
    /// Overlay input onto the existing column: omitted live documents retain their prior value.
    Merge,
}

/// How duplicate `DocId`s in a detached ingest input stream are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetachedDuplicatePolicy {
    /// Keep the last value emitted for each `DocId`.
    #[default]
    LastWins,
    /// Keep the first value emitted for each `DocId`.
    FirstWins,
    /// Reject duplicate `DocId`s with `TreeError::InvalidFilter`.
    Error,
}

/// Statistics produced by [`DetachedIngestBuilder::build`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DetachedBuildStats {
    pub columns: usize,
    pub input_rows: u64,
    pub matched_rows: u64,
    pub unknown_ids: u64,
    pub blocks_written: u64,
    pub used_sst: bool,
    pub watermark: RowId,
}

/// Statistics produced by [`Collection::publish_detached`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DetachedPublishStats {
    pub columns: usize,
    pub catchup_rows: u64,
    pub post_snapshot_deletes: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct DetachedColPlan {
    pub attr_id: AttrId,
    pub name: String,
    pub attr_type: AttributeType,
    pub filterable: bool,
    pub old_gen: ColumnGen,
    pub next_gen: ColumnGen,
    pub ordered_tag: Option<u8>,
    pub zones: BTreeMap<SegmentId, ZoneEntry>,
}

/// Staged whole-column ingest ready for atomic swap via [`Collection::publish_detached`].
///
/// Concurrent writes against `Collection` remain safe between `build()` and `publish_detached()`:
/// `publish_detached` resolves all rows allocated at or above `watermark` against `sorted_input`
/// and adjusts counts for rows `< watermark` deleted after the build snapshot.
#[derive(Debug)]
pub struct DetachedJob {
    pub stats: DetachedBuildStats,
    pub(crate) cols: Vec<DetachedColPlan>,
    pub(crate) mode: DetachedWriteMode,
    pub(crate) watermark: RowId,
    pub(crate) snap_open_end: Option<(SegmentId, u32)>,
    pub(crate) snap_next_segment_id: SegmentId,
    pub(crate) deleted_at_snapshot: BTreeMap<SegmentId, Option<RoaringBitmap>>,
    pub(crate) sorted_input: Vec<(DocId, Vec<AttributeValue>)>,
}

impl DetachedJob {
    #[inline]
    fn snapshot_limit_local(&self, seg: SegmentId, row_count: u32) -> u32 {
        if seg >= self.snap_next_segment_id {
            0
        } else if let Some((open_seg, open_rows)) = self.snap_open_end
            && seg == open_seg
        {
            open_rows.min(row_count)
        } else {
            row_count
        }
    }
}

/// Builder for staging an atomic whole-column detached ingest on `&Collection`.
pub struct DetachedIngestBuilder<'a> {
    coll: &'a Collection,
    columns: Vec<String>,
    mode: DetachedWriteMode,
    dup_policy: DetachedDuplicatePolicy,
    sst_threshold: usize,
    raw_rows: Vec<(DocId, Vec<AttributeValue>)>,
}

impl<'a> DetachedIngestBuilder<'a> {
    pub(crate) fn new(coll: &'a Collection) -> Self {
        Self {
            coll,
            columns: Vec::new(),
            mode: DetachedWriteMode::Replace,
            dup_policy: DetachedDuplicatePolicy::LastWins,
            sst_threshold: DEFAULT_SST_BLOCK_THRESHOLD,
            raw_rows: Vec::new(),
        }
    }

    pub fn column(mut self, name: impl Into<String>) -> Self {
        self.columns.push(name.into());
        self
    }

    pub fn columns<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for n in names {
            self.columns.push(n.into());
        }
        self
    }

    pub fn mode(mut self, mode: DetachedWriteMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn duplicate_policy(mut self, policy: DetachedDuplicatePolicy) -> Self {
        self.dup_policy = policy;
        self
    }

    /// Override the block count threshold above which `cf_detached` blocks are ingested via
    /// `SstFileWriter` instead of a `WriteBatch` (useful for testing both paths deterministically).
    pub fn sst_threshold(mut self, min_blocks: usize) -> Self {
        self.sst_threshold = min_blocks;
        self
    }

    pub fn from_iter<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = (DocId, Vec<AttributeValue>)>,
    {
        self.raw_rows.extend(iter);
        self
    }

    /// Convenience helper for single-column whole-column ingest.
    pub fn from_pairs<I>(mut self, iter: I) -> Self
    where
        I: IntoIterator<Item = (DocId, AttributeValue)>,
    {
        self.raw_rows
            .extend(iter.into_iter().map(|(id, val)| (id, vec![val])));
        self
    }

    pub fn build(self) -> Result<DetachedJob> {
        if self.columns.is_empty() {
            return Err(TreeError::InvalidSchema(
                "detached_ingest requires at least one column".into(),
            ));
        }

        let mut schema_entries: Vec<&SchemaEntry> = Vec::with_capacity(self.columns.len());
        let mut seen_ids: Vec<AttrId> = Vec::with_capacity(self.columns.len());
        for name in &self.columns {
            let entry = self.coll.schema().get_by_name(name).ok_or_else(|| {
                TreeError::InvalidSchema(format!("unknown detached attribute '{name}'"))
            })?;
            if entry.storage != ColumnStorage::Detached {
                return Err(TreeError::InvalidSchema(format!(
                    "attribute '{name}' is Inline, not Detached"
                )));
            }
            if seen_ids.contains(&entry.id) {
                return Err(TreeError::InvalidSchema(format!(
                    "duplicate detached column '{name}' in detached_ingest"
                )));
            }
            seen_ids.push(entry.id);
            schema_entries.push(entry);
        }

        let num_cols = schema_entries.len();
        let input_rows = self.raw_rows.len() as u64;
        let mut staged: Vec<(DocId, usize, Vec<AttributeValue>)> =
            Vec::with_capacity(self.raw_rows.len());
        for (seq, (doc_id, vals)) in self.raw_rows.into_iter().enumerate() {
            if vals.len() != num_cols {
                return Err(TreeError::InvalidSchema(format!(
                    "detached_ingest row for doc {doc_id} has {} values, expected {num_cols}",
                    vals.len()
                )));
            }
            for (val, entry) in vals.iter().zip(&schema_entries) {
                validate_attr_value(self.coll.schema(), entry, val)?;
            }
            staged.push((doc_id, seq, vals));
        }

        staged.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut sorted_input: Vec<(DocId, Vec<AttributeValue>)> = Vec::with_capacity(staged.len());
        for (doc_id, _seq, vals) in staged {
            if let Some(last) = sorted_input.last_mut()
                && last.0 == doc_id
            {
                match self.dup_policy {
                    DetachedDuplicatePolicy::LastWins => last.1 = vals,
                    DetachedDuplicatePolicy::FirstWins => {}
                    DetachedDuplicatePolicy::Error => {
                        return Err(TreeError::InvalidSchema(format!(
                            "duplicate doc_id {doc_id} in detached_ingest"
                        )));
                    }
                }
            } else {
                sorted_input.push((doc_id, vals));
            }
        }

        // Snapshot (Manifest, SegmentTable, cf_id_map) atomically.
        let (manifest, seg_snap, mut id_map) = self.coll.storage().snapshot_id_map_and_state()?;
        // cf_id_map is big-endian DocId, already ascending; ensure in case of unexpected state.
        debug_assert!(id_map.windows(2).all(|w| w[0].0 < w[1].0));
        if !id_map.windows(2).all(|w| w[0].0 < w[1].0) {
            id_map.sort_unstable_by_key(|&(doc_id, _)| doc_id);
        }

        let snap_open_end = manifest
            .open_segment
            .map(|seg| (seg, seg_snap.get(seg).map(|e| e.meta.row_count).unwrap_or(0)));
        let snap_next_segment_id = manifest.next_segment_id;
        let watermark = match snap_open_end {
            Some((seg, rows)) => RowId::new(seg, rows),
            None => RowId::new(snap_next_segment_id, 0),
        };

        let mut deleted_at_snapshot: BTreeMap<SegmentId, Option<RoaringBitmap>> = BTreeMap::new();
        for (&seg, entry) in &seg_snap.segments {
            deleted_at_snapshot.insert(seg, entry.deleted.as_deref().cloned());
        }

        // Merge-join sorted_input with id_map to map DocId -> (RowId, input_idx).
        let mut matched: Vec<(RowId, usize)> = Vec::new();
        let mut unknown_ids = 0u64;
        let mut i = 0usize;
        let mut j = 0usize;
        while i < sorted_input.len() && j < id_map.len() {
            match sorted_input[i].0.cmp(&id_map[j].0) {
                std::cmp::Ordering::Less => {
                    unknown_ids += 1;
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    matched.push((id_map[j].1, i));
                    i += 1;
                    j += 1;
                }
            }
        }
        unknown_ids += (sorted_input.len() - i) as u64;
        let matched_rows = matched.len() as u64;

        matched.sort_unstable_by_key(|&(row, _)| row);

        let mut col_plans = Vec::with_capacity(num_cols);
        let mut total_blocks_written = 0u64;
        let mut any_used_sst = false;

        for (col_idx, entry) in schema_entries.iter().enumerate() {
            let mut meta = self
                .coll
                .storage()
                .get_detached_meta(entry.id)?
                .unwrap_or_default();
            let old_gen = meta.current_gen;
            let next_gen = meta
                .current_gen
                .max(meta.pending_gen)
                .max(meta.retired_below)
                .saturating_add(1);
            if meta.pending_gen != 0 {
                meta.retired_below = meta.retired_below.max(meta.pending_gen.saturating_add(1));
            }
            meta.pending_gen = next_gen;
            meta.built_against_watermark = watermark;
            self.coll
                .storage()
                .put_detached_meta_sync(entry.id, &meta)?;

            let ordered_tag = if entry.filterable && entry.attr_type.is_ordered() {
                Some(ordered_type_tag(entry.attr_type)?)
            } else {
                None
            };

            let mut existing_blocks: Vec<(SegmentId, u32, Vec<u8>)> = Vec::new();
            if self.mode == DetachedWriteMode::Merge {
                self.coll
                    .storage()
                    .visit_detached_blocks_for_gen(entry.id, old_gen, |seg, blk, bytes| {
                        existing_blocks.push((seg, blk, bytes.to_vec()));
                        Ok(())
                    })?;
            }

            let built = build_column_generation(
                entry,
                col_idx,
                next_gen,
                ordered_tag,
                &deleted_at_snapshot,
                &existing_blocks,
                &matched,
                &sorted_input,
            )?;

            total_blocks_written += built.blocks.len() as u64;
            let use_sst = built.blocks.len() >= self.sst_threshold;
            if use_sst {
                any_used_sst = true;
                self.coll.storage().ingest_detached_sst(&built.blocks)?;
                self.coll
                    .storage()
                    .write_detached_stage(&built.counts, &built.postings, None)?;
            } else {
                self.coll.storage().write_detached_stage(
                    &built.counts,
                    &built.postings,
                    Some(&built.blocks),
                )?;
            }

            col_plans.push(DetachedColPlan {
                attr_id: entry.id,
                name: entry.name.clone(),
                attr_type: entry.attr_type,
                filterable: entry.filterable,
                old_gen,
                next_gen,
                ordered_tag,
                zones: built.zones,
            });
        }

        Ok(DetachedJob {
            stats: DetachedBuildStats {
                columns: col_plans.len(),
                input_rows,
                matched_rows,
                unknown_ids,
                blocks_written: total_blocks_written,
                used_sst: any_used_sst,
                watermark,
            },
            cols: col_plans,
            mode: self.mode,
            watermark,
            snap_open_end,
            snap_next_segment_id,
            deleted_at_snapshot,
            sorted_input,
        })
    }
}

struct BuiltColumnData {
    blocks: Vec<([u8; 14], Vec<u8>)>,
    counts: Vec<(Vec<u8>, u64)>,
    postings: Vec<(Vec<u8>, Vec<u8>)>,
    zones: BTreeMap<SegmentId, ZoneEntry>,
}

#[allow(clippy::too_many_arguments)]
fn build_column_generation(
    entry: &SchemaEntry,
    col_idx: usize,
    next_gen: ColumnGen,
    ordered_tag: Option<u8>,
    deleted_at_snapshot: &BTreeMap<SegmentId, Option<RoaringBitmap>>,
    existing_blocks: &[(SegmentId, u32, Vec<u8>)],
    matched: &[(RowId, usize)],
    sorted_input: &[(DocId, Vec<AttributeValue>)],
) -> Result<BuiltColumnData> {
    let mut blocks: Vec<([u8; 14], Vec<u8>)> = Vec::new();
    let mut counts_map: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut postings_map: BTreeMap<Vec<u8>, BTreeMap<SegmentId, RoaringBitmap>> = BTreeMap::new();
    let mut zones: BTreeMap<SegmentId, ZoneEntry> = BTreeMap::new();

    let mut ei = 0usize;
    let mut mi = 0usize;

    while ei < existing_blocks.len() || mi < matched.len() {
        let next_existing = existing_blocks.get(ei).map(|(s, b, _)| (*s, *b));
        let next_matched = matched
            .get(mi)
            .map(|(r, _)| (r.seg(), r.block_idx()));
        let (seg, blk) = match (next_existing, next_matched) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };

        let existing_bytes_opt = if next_existing == Some((seg, blk)) {
            let b = &existing_blocks[ei].2;
            ei += 1;
            Some(b.as_slice())
        } else {
            None
        };

        let mut builder = ColumnBlockBuilder::new(entry.attr_type);
        let dead_bm = deleted_at_snapshot.get(&seg).and_then(|o| o.as_ref());
        if let Some(ex_bytes) = existing_bytes_opt {
            let ex_view = ColumnBlockView::parse(ex_bytes)?;
            let base_local = blk << 8;
            for slot_u in 0..ROW_BLOCK_SIZE {
                let slot = slot_u as u8;
                if !ex_view.is_present(slot) {
                    continue;
                }
                let local = base_local | (slot as u32);
                if dead_bm.is_some_and(|bm| bm.contains(local)) {
                    continue;
                }
                let v = ex_view.decode_attr_value(slot)?;
                builder.set_attr_value(slot, &v)?;
            }
        }

        while mi < matched.len() && matched[mi].0.seg() == seg && matched[mi].0.block_idx() == blk
        {
            let row = matched[mi].0;
            let input_idx = matched[mi].1;
            mi += 1;
            if dead_bm.is_some_and(|bm| bm.contains(row.local())) {
                continue;
            }
            let val = &sorted_input[input_idx].1[col_idx];
            builder.set_attr_value(row.slot(), val)?;
        }

        if builder.present_count() == 0 {
            continue;
        }
        let encoded = builder.encode();
        if entry.filterable {
            let view = ColumnBlockView::parse(&encoded)?;
            let base_local = blk << 8;
            for slot_u in 0..ROW_BLOCK_SIZE {
                let slot = slot_u as u8;
                if !view.is_present(slot) {
                    continue;
                }
                let local = base_local | (slot as u32);
                let val = view.decode_attr_value(slot)?;
                let keys = extract_index_keys_for_attr(entry.id, &val)?;
                for attr_key in keys {
                    let suffix = detached_count_suffix(next_gen, &attr_key);
                    *counts_map.entry(suffix).or_insert(0) += 1;

                    let pkey = detached_posting_key(
                        entry.id,
                        next_gen,
                        attr_key[2],
                        &attr_key[3..],
                        seg,
                    );
                    postings_map
                        .entry(pkey)
                        .or_default()
                        .entry(seg)
                        .or_default()
                        .insert(local);

                    if ordered_tag == Some(attr_key[2]) && attr_key.len() > 3 {
                        let enc_val = &attr_key[3..];
                        match zones.get_mut(&seg) {
                            Some(z) => {
                                z.widen(enc_val);
                            }
                            None => {
                                zones.insert(seg, ZoneEntry::singleton(enc_val));
                            }
                        }
                    }
                }
            }
        }
        blocks.push((detached_key(entry.id, next_gen, seg, blk), encoded));
    }

    let counts: Vec<(Vec<u8>, u64)> = counts_map.into_iter().collect();
    let mut postings: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(postings_map.len());
    for (pkey, by_seg) in postings_map {
        for (_seg, bm) in by_seg {
            postings.push((pkey.clone(), SegPosting::from_bitmap(bm).encode()?));
        }
    }

    Ok(BuiltColumnData {
        blocks,
        counts,
        postings,
        zones,
    })
}

/// Atomically publish a staged [`DetachedJob`] onto `coll`, catching up on rows inserted or
/// updated after the build snapshot and correcting `0x07` counts for post-snapshot deletions.
pub(crate) fn publish_detached_job(
    coll: &mut Collection,
    mut job: DetachedJob,
) -> Result<DetachedPublishStats> {
    if job.cols.is_empty() {
        return Ok(DetachedPublishStats::default());
    }

    // Verify no pending_gen mismatch (e.g. if a newer build overwrote pending_gen).
    for plan in &job.cols {
        let meta = coll
            .storage()
            .get_detached_meta(plan.attr_id)?
            .ok_or_else(|| {
                TreeError::InvalidFilter(format!(
                    "detached publish aborted: missing DetachedMeta for attr {}",
                    plan.name
                ))
            })?;
        if meta.pending_gen != plan.next_gen {
            return Err(TreeError::InvalidFilter(format!(
                "detached publish for '{}' stale: expected pending_gen {}, found {}",
                plan.name, plan.next_gen, meta.pending_gen
            )));
        }
    }

    let snap = coll.storage().segment_snapshot();
    // 1. Identify rows existing at snapshot S that became deleted before publish.
    let mut post_delete_rows: Vec<RowId> = Vec::new();
    for (&seg, snap_entry) in &snap.segments {
        let limit_local = job.snapshot_limit_local(seg, snap_entry.meta.row_count);
        if limit_local == 0 {
            continue;
        }
        let Some(ref deleted_now) = snap_entry.deleted else {
            continue;
        };
        let deleted_then = job.deleted_at_snapshot.get(&seg).and_then(|o| o.as_ref());
        for local in deleted_now.iter() {
            if local >= limit_local {
                break;
            }
            if !deleted_then.is_some_and(|d| d.contains(local)) {
                post_delete_rows.push(RowId::new(seg, local));
            }
        }
    }

    // 2. Identify live rows allocated after snapshot S.
    let mut tail_rows: Vec<RowId> = Vec::new();
    for seg in snap.visible_segments() {
        let Some(entry) = snap.get(seg) else {
            continue;
        };
        let start_local = job.snapshot_limit_local(seg, entry.meta.row_count);
        if start_local >= entry.meta.row_count {
            continue;
        }
        for local in start_local..entry.meta.row_count {
            let row = RowId::new(seg, local);
            if snap.is_live(row) {
                tail_rows.push(row);
            }
        }
    }
    let tail_doc_ids = RowResolver::resolve_all(coll.storage(), &tail_rows)?;

    let existing_zones = coll.storage().get_zones_cached()?;
    let cols_count = job.cols.len();
    let catchup_rows = tail_rows.len() as u64;
    let post_snapshot_deletes = post_delete_rows.len() as u64;

    let (manifest, ann_meta) = {
        let mut tx = TxContext::new(
            &coll.storage,
            &mut coll.internal_cache,
            &mut coll.leaf_parent,
            coll.manifest.clone(),
            coll.ann_meta.clone(),
        );

        // Phase A: decrement 0x07 counts for rows `< watermark` deleted after build snapshot.
        for &row in &post_delete_rows {
            let seg = row.seg();
            let blk = row.block_idx();
            let slot = row.slot();
            for plan in &job.cols {
                if !plan.filterable {
                    continue;
                }
                let Some(bytes) =
                    tx.get_detached_block_bytes(plan.attr_id, plan.next_gen, seg, blk)?
                else {
                    continue;
                };
                let view = ColumnBlockView::parse(&bytes)?;
                if !view.is_present(slot) {
                    continue;
                }
                let val = view.decode_attr_value(slot)?;
                let keys = extract_index_keys_for_attr(plan.attr_id, &val)?;
                for k in keys {
                    tx.adjust_detached_count(plan.next_gen, &k, -1);
                }
            }
        }

        // Phase B: catch up live rows `>= watermark` grouped by (seg, block).
        if !tail_rows.is_empty() {
            let mut blk_idx = 0usize;
            while blk_idx < tail_rows.len() {
                let seg = tail_rows[blk_idx].seg();
                let blk = tail_rows[blk_idx].block_idx();
                let start = blk_idx;
                while blk_idx < tail_rows.len()
                    && tail_rows[blk_idx].seg() == seg
                    && tail_rows[blk_idx].block_idx() == blk
                {
                    blk_idx += 1;
                }

                for (col_idx, plan) in job.cols.iter_mut().enumerate() {
                    let mut updates: Vec<(u8, AttributeValue)> = Vec::new();
                    let mut old_block_bytes: Option<Vec<u8>> = None;
                    let mut old_view_loaded = false;

                    for pos in start..blk_idx {
                        let row = tail_rows[pos];
                        let doc_id = tail_doc_ids[pos];
                        let val =
                            match job.sorted_input.binary_search_by_key(&doc_id, |&(d, _)| d) {
                                Ok(idx) => job.sorted_input[idx].1[col_idx].clone(),
                                Err(_) => match job.mode {
                                    DetachedWriteMode::Replace => continue,
                                    DetachedWriteMode::Merge => {
                                        if !old_view_loaded {
                                            old_block_bytes = tx.get_detached_block_bytes(
                                                plan.attr_id,
                                                plan.old_gen,
                                                seg,
                                                blk,
                                            )?;
                                            old_view_loaded = true;
                                        }
                                        let Some(ref b) = old_block_bytes else {
                                            continue;
                                        };
                                        let v = ColumnBlockView::parse(b)?;
                                        if !v.is_present(row.slot()) {
                                            continue;
                                        }
                                        v.decode_attr_value(row.slot())?
                                    }
                                },
                            };
                        if !matches!(val, AttributeValue::Null) {
                            if plan.filterable {
                                let keys = extract_index_keys_for_attr(plan.attr_id, &val)?;
                                for k in keys {
                                    tx.adjust_detached_count(plan.next_gen, &k, 1);
                                    tx.add_detached_posting(
                                        detached_posting_key(
                                            plan.attr_id,
                                            plan.next_gen,
                                            k[2],
                                            &k[3..],
                                            seg,
                                        ),
                                        row.local(),
                                    );
                                    if plan.ordered_tag == Some(k[2]) && k.len() > 3 {
                                        let enc_val = &k[3..];
                                        match plan.zones.get_mut(&seg) {
                                            Some(z) => {
                                                z.widen(enc_val);
                                            }
                                            None => {
                                                plan.zones
                                                    .insert(seg, ZoneEntry::singleton(enc_val));
                                            }
                                        }
                                    }
                                }
                            }
                            updates.push((row.slot(), val));
                        }
                    }

                    if !updates.is_empty() {
                        let cur = tx.get_detached_block_bytes(
                            plan.attr_id,
                            plan.next_gen,
                            seg,
                            blk,
                        )?;
                        let cur_view = cur.as_deref().map(ColumnBlockView::parse).transpose()?;
                        let merged = ColumnBlockBuilder::merge_block(
                            cur_view.as_ref(),
                            plan.attr_type,
                            &updates,
                        )?;
                        tx.put_detached_block_bytes(plan.attr_id, plan.next_gen, seg, blk, merged);
                    }
                }
            }
        }

        // Phase C: swap zones, retire old generations, and flip DetachedMeta atomically.
        for plan in &job.cols {
            if let Some(tag) = plan.ordered_tag {
                if let Some(seg_map) = existing_zones.get(&(plan.attr_id, tag)) {
                    for &seg in seg_map.keys() {
                        if !plan.zones.contains_key(&seg) {
                            tx.delete_zone_entry(seg, plan.attr_id, tag);
                        }
                    }
                }
                for (&seg, zone) in &plan.zones {
                    tx.set_zone_entry(seg, plan.attr_id, tag, zone.clone());
                }
            }
            let mut meta = tx.get_detached_meta(plan.attr_id)?;
            meta.state = 0;
            meta.current_gen = plan.next_gen;
            meta.pending_gen = 0;
            meta.retired_below = meta.retired_below.max(plan.next_gen);
            meta.built_against_watermark = job.watermark;
            tx.put_detached_meta(plan.attr_id, meta);
            tx.retire_detached_gen(plan.attr_id, plan.old_gen);
        }

        tx.commit()?
    };

    coll.apply_tx_commit(manifest, ann_meta);
    Ok(DetachedPublishStats {
        columns: cols_count,
        catchup_rows,
        post_snapshot_deletes,
    })
}

// ---------------------------------------------------------------------------
// Point writes, CoW carry-over, deletion, get/get_batch, and >5% rebuild
// ---------------------------------------------------------------------------

/// Split `attrs` into `(inline_attributes, sorted detached (AttrId, AttributeValue))`.
pub(crate) fn partition_attributes(
    schema: &CollectionSchema,
    attrs: &Attributes,
) -> (Attributes, Vec<(AttrId, AttributeValue)>) {
    let mut inline = Attributes::new();
    let mut detached = Vec::new();
    for (name, val) in attrs {
        if let Some(entry) = schema.get_by_name(name)
            && entry.storage == ColumnStorage::Detached
        {
            detached.push((entry.id, val.clone()));
        } else {
            inline.insert(name.clone(), val.clone());
        }
    }
    detached.sort_unstable_by_key(|&(id, _)| id);
    (inline, detached)
}

#[inline]
fn validate_attr_value(
    schema: &CollectionSchema,
    entry: &SchemaEntry,
    val: &AttributeValue,
) -> Result<()> {
    let mut single = Attributes::new();
    single.insert(entry.name.clone(), val.clone());
    schema.validate_attributes(&single)
}

#[inline]
fn attr_values_exact_eq(a: &AttributeValue, b: &AttributeValue) -> bool {
    match (a, b) {
        (AttributeValue::Float(x), AttributeValue::Float(y)) => x.to_bits() == y.to_bits(),
        _ => a == b,
    }
}

/// Apply a single cell update at `(entry.id, col_gen, row)` inside `tx`.
/// Returns `Ok(true)` if the stored value changed.
fn update_detached_cell_in_tx(
    tx: &mut TxContext<'_>,
    entry: &SchemaEntry,
    col_gen: ColumnGen,
    row: RowId,
    new_val: &AttributeValue,
    record_overwrite_on_replace: bool,
) -> Result<bool> {
    let seg = row.seg();
    let blk = row.block_idx();
    let slot = row.slot();
    let cur_bytes = tx.get_detached_block_bytes(entry.id, col_gen, seg, blk)?;
    let cur_view = cur_bytes
        .as_deref()
        .map(ColumnBlockView::parse)
        .transpose()?;
    let old_val = match cur_view.as_ref() {
        Some(v) if v.is_present(slot) => v.decode_attr_value(slot)?,
        _ => AttributeValue::Null,
    };

    if attr_values_exact_eq(&old_val, new_val) {
        return Ok(false);
    }

    let merged =
        ColumnBlockBuilder::merge_block(cur_view.as_ref(), entry.attr_type, &[(slot, new_val.clone())])?;
    tx.put_detached_block_bytes(entry.id, col_gen, seg, blk, merged);

    if entry.filterable {
        if !matches!(old_val, AttributeValue::Null) {
            let old_keys = extract_index_keys_for_attr(entry.id, &old_val)?;
            for k in &old_keys {
                tx.adjust_detached_count(col_gen, k, -1);
            }
            if record_overwrite_on_replace {
                tx.record_detached_overwritten(entry.id, col_gen, seg, row.local())?;
            }
        }
        if !matches!(new_val, AttributeValue::Null) {
            let new_keys = extract_index_keys_for_attr(entry.id, new_val)?;
            for k in &new_keys {
                let tag = k[2];
                tx.adjust_detached_count(col_gen, k, 1);
                tx.add_detached_posting(
                    detached_posting_key(entry.id, col_gen, tag, &k[3..], seg),
                    row.local(),
                );
                if k.len() > 3
                    && matches!(tag, TAG_INT | TAG_UINT | TAG_FLOAT | TAG_STRING)
                    && entry.attr_type.is_ordered()
                {
                    tx.widen_zone(seg, entry.id, tag, &k[3..])?;
                }
            }
        }
    }
    Ok(true)
}

/// Apply detached writes in-place on an existing `row` without changing its `RowId`.
/// When `reset_omitted` is true (`upsert`), every detached column in `schema` omitted from `updates`
/// is cleared to `Null`.
pub(crate) fn apply_detached_write_on_existing_row(
    tx: &mut TxContext<'_>,
    schema: &CollectionSchema,
    row: RowId,
    updates: &[(AttrId, AttributeValue)],
    reset_omitted: bool,
) -> Result<bool> {
    let mut changed = false;
    for entry in schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
    {
        let target_val = match updates.binary_search_by_key(&entry.id, |&(id, _)| id) {
            Ok(idx) => &updates[idx].1,
            Err(_) if reset_omitted => &AttributeValue::Null,
            Err(_) => continue,
        };
        let col_gen = tx.get_detached_meta(entry.id)?.current_gen;
        if update_detached_cell_in_tx(tx, entry, col_gen, row, target_val, true)? {
            changed = true;
        }
    }
    Ok(changed)
}

/// Write detached columns onto a newly-allocated `new_row`, optionally carrying over untouched
/// detached columns from `old_row` (`patch` sets `carry_over = true`; `upsert` sets `false`).
pub(crate) fn write_detached_on_new_row(
    tx: &mut TxContext<'_>,
    schema: &CollectionSchema,
    old_row: Option<RowId>,
    new_row: RowId,
    updates: &[(AttrId, AttributeValue)],
    carry_over: bool,
) -> Result<()> {
    for entry in schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
    {
        let col_gen = tx.get_detached_meta(entry.id)?.current_gen;

        let mut old_val = AttributeValue::Null;
        if let Some(prev) = old_row
            && let Some(prev_bytes) =
                tx.get_detached_block_bytes(entry.id, col_gen, prev.seg(), prev.block_idx())?
        {
            let v = ColumnBlockView::parse(&prev_bytes)?;
            if v.is_present(prev.slot()) {
                old_val = v.decode_attr_value(prev.slot())?;
            }
        }

        // Decrement counts for tombstoned old_row.
        if entry.filterable && !matches!(old_val, AttributeValue::Null) {
            let old_keys = extract_index_keys_for_attr(entry.id, &old_val)?;
            for k in &old_keys {
                tx.adjust_detached_count(col_gen, k, -1);
            }
        }

        let target_val = match updates.binary_search_by_key(&entry.id, |&(id, _)| id) {
            Ok(idx) => updates[idx].1.clone(),
            Err(_) if carry_over => old_val,
            Err(_) => continue,
        };

        if !matches!(target_val, AttributeValue::Null) {
            let _ =
                update_detached_cell_in_tx(tx, entry, col_gen, new_row, &target_val, false)?;
        }
    }
    Ok(())
}

/// Decrement `0x07` detached counts when deleting live row `old_row`.
pub(crate) fn delete_detached_in_tx(
    tx: &mut TxContext<'_>,
    schema: &CollectionSchema,
    old_row: RowId,
) -> Result<()> {
    for entry in schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached && e.filterable)
    {
        let col_gen = tx.get_detached_meta(entry.id)?.current_gen;
        let Some(bytes) = tx.get_detached_block_bytes(
            entry.id,
            col_gen,
            old_row.seg(),
            old_row.block_idx(),
        )?
        else {
            continue;
        };
        let v = ColumnBlockView::parse(&bytes)?;
        if !v.is_present(old_row.slot()) {
            continue;
        }
        let val = v.decode_attr_value(old_row.slot())?;
        let keys = extract_index_keys_for_attr(entry.id, &val)?;
        for k in &keys {
            tx.adjust_detached_count(col_gen, k, -1);
        }
    }
    Ok(())
}

/// Point write `set_detached` / `set_detached_batch` against live documents in place.
pub(crate) fn apply_point_detached_batch(
    coll: &mut Collection,
    updates: &[(DocId, Attributes)],
) -> Result<()> {
    if updates.is_empty() {
        return Ok(());
    }
    let mut doc_ids = Vec::with_capacity(updates.len());
    let mut parsed: Vec<(DocId, Vec<(AttrId, AttributeValue)>)> = Vec::with_capacity(updates.len());
    for (doc_id, attrs) in updates {
        coll.schema().validate_attributes(attrs)?;
        let mut items = Vec::with_capacity(attrs.len());
        for (name, val) in attrs {
            let entry = coll.schema().get_by_name(name).ok_or_else(|| {
                TreeError::InvalidSchema(format!("unknown attribute '{name}'"))
            })?;
            if entry.storage != ColumnStorage::Detached {
                return Err(TreeError::InvalidSchema(format!(
                    "set_detached only accepts Detached attributes; '{name}' is Inline"
                )));
            }
            items.push((entry.id, val.clone()));
        }
        items.sort_unstable_by_key(|&(id, _)| id);
        doc_ids.push(*doc_id);
        parsed.push((*doc_id, items));
    }

    let mapped = coll.storage().multi_get_id_map(&doc_ids)?;
    let mut resolved = Vec::with_capacity(parsed.len());
    for ((doc_id, items), opt_row) in parsed.into_iter().zip(mapped) {
        let row = opt_row.ok_or(TreeError::DocumentNotFound(doc_id))?;
        resolved.push((row, items));
    }
    resolved.sort_unstable_by_key(|&(r, _)| r);

    let schema = coll.schema().clone();
    let (any_changed, committed) = {
        let mut tx = TxContext::new(
            &coll.storage,
            &mut coll.internal_cache,
            &mut coll.leaf_parent,
            coll.manifest.clone(),
            coll.ann_meta.clone(),
        );
        let mut any_changed = false;
        for (row, items) in resolved {
            if apply_detached_write_on_existing_row(&mut tx, &schema, row, &items, false)? {
                any_changed = true;
            }
        }
        if any_changed {
            let (m, am) = tx.commit()?;
            (true, Some((m, am)))
        } else {
            (false, None)
        }
    };
    if any_changed && let Some((m, am)) = committed {
        coll.apply_tx_commit(m, am);
    }
    Ok(())
}

/// Read all non-`Null` detached attributes for a single `row` (for `Collection::get`).
pub(crate) fn read_detached_attrs_for_row(
    storage: &Storage,
    schema: &CollectionSchema,
    row: RowId,
    out: &mut Attributes,
) -> Result<()> {
    let detached_entries: Vec<&SchemaEntry> = schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
        .collect();
    if detached_entries.is_empty() {
        return Ok(());
    }
    let gens = all_detached_gens(storage)?;
    let keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = detached_entries
        .iter()
        .map(|e| {
            let g = gens.get(&e.id).copied().unwrap_or(0);
            (e.id, g, row.seg(), row.block_idx())
        })
        .collect();
    let slices = storage.multi_get_detached_blocks_pinned(&keys, true)?;
    for (entry, opt_slice) in detached_entries.iter().zip(slices.iter()) {
        let Some(slice) = opt_slice.as_ref() else {
            continue;
        };
        let view = ColumnBlockView::parse(slice)?;
        if view.is_present(row.slot()) {
            let val = view.decode_attr_value(row.slot())?;
            if !matches!(val, AttributeValue::Null) {
                out.insert(entry.name.clone(), val);
            }
        }
    }
    Ok(())
}

/// Populate non-`Null` detached attributes for `sorted_rows` into `per_row_attrs` (aligned by index).
pub(crate) fn read_detached_attrs_for_rows_sorted(
    storage: &Storage,
    schema: &CollectionSchema,
    sorted_rows: &[RowId],
    per_row_attrs: &mut [Attributes],
) -> Result<()> {
    if sorted_rows.is_empty() {
        return Ok(());
    }
    let detached_entries: Vec<&SchemaEntry> = schema
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached)
        .collect();
    if detached_entries.is_empty() {
        return Ok(());
    }
    let gens = all_detached_gens(storage)?;
    let mut keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = Vec::new();
    for entry in &detached_entries {
        let g = gens.get(&entry.id).copied().unwrap_or(0);
        for &row in sorted_rows {
            let k = (entry.id, g, row.seg(), row.block_idx());
            if keys.last().copied() != Some(k) {
                keys.push(k);
            }
        }
    }
    let slices = storage.multi_get_detached_blocks_pinned(&keys, true)?;
    let mut block_map: HashMap<(AttrId, SegmentId, u32), &ColumnBlockView<'_>> = HashMap::new();
    let mut parsed_views: Vec<Option<ColumnBlockView<'_>>> = Vec::with_capacity(slices.len());
    for s_opt in &slices {
        parsed_views.push(s_opt.as_ref().map(|s| ColumnBlockView::parse(s)).transpose()?);
    }
    for (k, view_opt) in keys.iter().zip(parsed_views.iter()) {
        if let Some(v) = view_opt.as_ref() {
            block_map.insert((k.0, k.2, k.3), v);
        }
    }
    for (&row, attrs) in sorted_rows.iter().zip(per_row_attrs.iter_mut()) {
        for entry in &detached_entries {
            if let Some(view) = block_map.get(&(entry.id, row.seg(), row.block_idx()))
                && view.is_present(row.slot())
            {
                let val = view.decode_attr_value(row.slot())?;
                if !matches!(val, AttributeValue::Null) {
                    attrs.insert(entry.name.clone(), val);
                }
            }
        }
    }
    Ok(())
}

/// Rebuild any detached column whose live `0x14` overwritten set exceeds 5% of `live_rows`.
pub(crate) fn rebuild_fragmented_detached_columns(coll: &mut Collection) -> Result<usize> {
    let snap = coll.storage().segment_snapshot();
    let live_rows: u64 = snap
        .visible_segments()
        .into_iter()
        .map(|s| {
            snap.get(s)
                .map_or(0, |e| e.meta.row_count.saturating_sub(e.meta.deleted_count) as u64)
        })
        .sum();
    if live_rows == 0 {
        return Ok(0);
    }

    let detached_names: Vec<(AttrId, String)> = coll
        .schema()
        .entries()
        .iter()
        .filter(|e| e.storage == ColumnStorage::Detached && e.filterable)
        .map(|e| (e.id, e.name.clone()))
        .collect();

    let mut to_rebuild: Vec<String> = Vec::new();
    for (attr_id, name) in detached_names {
        let cur_gen = current_detached_gen(coll.storage(), attr_id)?;
        let ow_segs = coll.storage().all_detached_overwritten(attr_id, cur_gen)?;
        if ow_segs.is_empty() {
            continue;
        }
        let mut live_ow = 0u64;
        for (seg, ow_bm) in ow_segs {
            if let Some(live_bm) = snap.live_bitmap(seg) {
                live_ow += (ow_bm & live_bm).len();
            }
        }
        if live_ow.saturating_mul(20) > live_rows {
            to_rebuild.push(name);
        }
    }

    let count = to_rebuild.len();
    for name in to_rebuild {
        let job = coll
            .detached_ingest()
            .column(name)
            .mode(DetachedWriteMode::Merge)
            .build()?;
        coll.publish_detached(job)?;
    }
    Ok(count)
}

// ---------------------------------------------------------------------------
// Query helpers: generation map, overwritten correction, & mixed filter eval
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn current_detached_gen(storage: &Storage, attr_id: AttrId) -> Result<ColumnGen> {
    Ok(storage.get_detached_meta(attr_id)?.map_or(0, |m| m.current_gen))
}

pub(crate) fn all_detached_gens(storage: &Storage) -> Result<HashMap<AttrId, ColumnGen>> {
    let mut map = HashMap::new();
    for (attr_id, meta) in storage.all_detached_metas()? {
        map.insert(attr_id, meta.current_gen);
    }
    Ok(map)
}

pub(crate) fn filter_mentions_detached(schema: &CollectionSchema, filter: &Filter) -> bool {
    let is_det = |attr: &str| {
        schema
            .get_by_name(attr)
            .is_some_and(|e| e.storage == ColumnStorage::Detached)
    };
    match filter {
        Filter::And(ch) | Filter::Or(ch) => {
            ch.iter().any(|c| filter_mentions_detached(schema, c))
        }
        Filter::Not(c) => filter_mentions_detached(schema, c),
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Contains(a, _)
        | Filter::ContainsAny(a, _)
        | Filter::ContainsAllTokens { attr: a, .. }
        | Filter::ContainsAnyToken { attr: a, .. }
        | Filter::ContainsTokenSequence { attr: a, .. } => is_det(a),
    }
}

pub(crate) fn filter_mentions_inline_or_fts(schema: &CollectionSchema, filter: &Filter) -> bool {
    let is_inline = |attr: &str| {
        schema
            .get_by_name(attr)
            .is_some_and(|e| e.storage == ColumnStorage::Inline)
    };
    match filter {
        Filter::And(ch) | Filter::Or(ch) => ch
            .iter()
            .any(|c| filter_mentions_inline_or_fts(schema, c)),
        Filter::Not(c) => filter_mentions_inline_or_fts(schema, c),
        Filter::ContainsAllTokens { .. }
        | Filter::ContainsAnyToken { .. }
        | Filter::ContainsTokenSequence { .. } => true,
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Contains(a, _)
        | Filter::ContainsAny(a, _) => is_inline(a),
    }
}

pub(crate) fn collect_detached_filter_attr_ids(
    schema: &CollectionSchema,
    filter: &Filter,
    out: &mut Vec<AttrId>,
) {
    let mut push = |attr: &str| {
        if let Some(e) = schema.get_by_name(attr)
            && e.storage == ColumnStorage::Detached
            && !out.contains(&e.id)
        {
            out.push(e.id);
        }
    };
    match filter {
        Filter::And(ch) | Filter::Or(ch) => {
            for c in ch {
                collect_detached_filter_attr_ids(schema, c, out);
            }
        }
        Filter::Not(c) => collect_detached_filter_attr_ids(schema, c, out),
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Contains(a, _)
        | Filter::ContainsAny(a, _) => push(a),
        Filter::ContainsAllTokens { .. }
        | Filter::ContainsAnyToken { .. }
        | Filter::ContainsTokenSequence { .. } => {}
    }
}

/// Evaluate `cf_detached` blocks for `(seg, ow_bm)` across all overwritten rows that are live
/// in `snap`, inserting `local` into `out` if `slot_matches(val)` is `true`.
pub(crate) fn recheck_detached_overwritten_in_seg(
    storage: &Storage,
    entry: &SchemaEntry,
    col_gen: ColumnGen,
    seg: SegmentId,
    ow_live: &RoaringBitmap,
    out: &mut RoaringBitmap,
    mut slot_matches: impl FnMut(&ColumnBlockView<'_>, u8) -> Result<bool>,
) -> Result<()> {
    if ow_live.is_empty() {
        return Ok(());
    }
    let _ = entry;
    let mut blocks: Vec<(u32, Vec<u8>)> = Vec::new();
    for local in ow_live.iter() {
        let blk = local >> 8;
        let slot = (local & 0xFF) as u8;
        if let Some(last) = blocks.last_mut()
            && last.0 == blk
        {
            last.1.push(slot);
        } else {
            blocks.push((blk, vec![slot]));
        }
    }
    let keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = blocks
        .iter()
        .map(|&(blk, _)| (entry.id, col_gen, seg, blk))
        .collect();
    let slices = storage.multi_get_detached_blocks_pinned(&keys, true)?;
    for ((blk, slots), opt_slice) in blocks.iter().zip(slices.iter()) {
        let Some(slice) = opt_slice.as_ref() else {
            continue;
        };
        let view = ColumnBlockView::parse(slice)?;
        for &slot in slots {
            if view.is_present(slot) && slot_matches(&view, slot)? {
                out.insert((blk << 8) | (slot as u32));
            }
        }
    }
    Ok(())
}

/// Build the overwritten-set map (`seg -> ow & live`) for `(attr_id, col_gen)` and subtract it
/// from `by_seg` bitmaps, then insert matching overwritten rows via `slot_matches`.
pub(crate) fn apply_overwritten_correction_for_filter(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    entry: &SchemaEntry,
    col_gen: ColumnGen,
    by_seg: &mut BTreeMap<SegmentId, RoaringBitmap>,
    mut slot_matches: impl FnMut(&ColumnBlockView<'_>, u8) -> Result<bool>,
) -> Result<()> {
    let ow_list = storage.all_detached_overwritten(entry.id, col_gen)?;
    if ow_list.is_empty() {
        return Ok(());
    }
    for (seg, ow_bm) in ow_list {
        let Some(live_bm) = snap.live_bitmap(seg) else {
            continue;
        };
        let ow_live = ow_bm & live_bm;
        if ow_live.is_empty() {
            continue;
        }
        let dst = by_seg.entry(seg).or_default();
        *dst -= &ow_live;
        recheck_detached_overwritten_in_seg(
            storage,
            entry,
            col_gen,
            seg,
            &ow_live,
            dst,
            &mut slot_matches,
        )?;
        if dst.is_empty() {
            by_seg.remove(&seg);
        }
    }
    Ok(())
}

/// Evaluate a detached-aware `Filter` for a single `(row, optional RowView)` using `detached_views`
/// preloaded from `cf_detached` (`attr_id -> Option<ColumnBlockView>`).
pub(crate) fn matches_filter_mixed(
    schema: &CollectionSchema,
    filter: &Filter,
    row_view: Option<&RowView<'_>>,
    detached_views: &HashMap<AttrId, Option<ColumnBlockView<'_>>>,
    slot: u8,
    fts: &FtsIndex,
) -> Result<bool> {
    eval_mixed_node(schema, filter, row_view, detached_views, slot, fts)
}

fn eval_mixed_node(
    schema: &CollectionSchema,
    filter: &Filter,
    row_view: Option<&RowView<'_>>,
    detached_views: &HashMap<AttrId, Option<ColumnBlockView<'_>>>,
    slot: u8,
    fts: &FtsIndex,
) -> Result<bool> {
    match filter {
        Filter::And(ch) => {
            for c in ch {
                if !eval_mixed_node(schema, c, row_view, detached_views, slot, fts)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Filter::Or(ch) => {
            for c in ch {
                if eval_mixed_node(schema, c, row_view, detached_views, slot, fts)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Filter::Not(c) => Ok(!eval_mixed_node(
            schema,
            c,
            row_view,
            detached_views,
            slot,
            fts,
        )?),
        leaf => {
            if let Some((kind, attr, query, last_as_prefix)) = leaf.as_fts_predicate() {
                let rv = row_view.ok_or_else(|| {
                    TreeError::InvariantViolation("missing RowView for FTS predicate".into())
                })?;
                return rv.matches_filter(schema, leaf, &mut |_k, _a, _q, _lp, val| {
                    fts.matches_fts_attr(schema, kind, attr, query, last_as_prefix, val)
                });
            }
            let attr_name = leaf_attr_name(leaf).ok_or_else(|| {
                TreeError::InvariantViolation("expected leaf Filter predicate".into())
            })?;
            let entry = schema.get_by_name(attr_name).ok_or_else(|| {
                TreeError::InvalidFilter(format!("unknown attribute '{attr_name}'"))
            })?;
            if entry.storage == ColumnStorage::Inline {
                let rv = row_view.ok_or_else(|| {
                    TreeError::InvariantViolation("missing RowView for Inline predicate".into())
                })?;
                return rv.matches_filter(
                    schema,
                    leaf,
                    &mut |kind, a, q, lp, val| {
                        fts.matches_fts_attr(schema, kind, a, q, lp, val)
                    },
                );
            }
            let view_opt = detached_views.get(&entry.id).and_then(|o| o.as_ref());
            eval_detached_leaf_in_block(view_opt, slot, leaf)
        }
    }
}

#[inline]
fn leaf_attr_name(f: &Filter) -> Option<&str> {
    match f {
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Contains(a, _)
        | Filter::ContainsAny(a, _) => Some(a.as_str()),
        _ => None,
    }
}

fn eval_detached_leaf_in_block(
    view_opt: Option<&ColumnBlockView<'_>>,
    slot: u8,
    leaf: &Filter,
) -> Result<bool> {
    let Some(view) = view_opt.filter(|v| v.is_present(slot)) else {
        return Ok(match leaf {
            Filter::IsNull(_) => true,
            Filter::Eq(_, ScalarValue::Null) => true,
            Filter::In(_, vals) => vals.iter().any(|v| matches!(v, ScalarValue::Null)),
            Filter::NotEq(_, v) => !matches!(v, ScalarValue::Null),
            Filter::NotIn(_, vals) => !vals.iter().any(|v| matches!(v, ScalarValue::Null)),
            _ => false,
        });
    };

    if matches!(leaf, Filter::IsNotNull(_)) {
        return Ok(true);
    }
    if matches!(leaf, Filter::IsNull(_)) {
        return Ok(false);
    }

    match leaf {
        Filter::Eq(_, rhs) => {
            if matches!(rhs, ScalarValue::Null) {
                Ok(false)
            } else {
                let s = view.get_scalar(slot)?;
                Ok(scalars_eq(&s, rhs))
            }
        }
        Filter::NotEq(_, rhs) => {
            if matches!(rhs, ScalarValue::Null) {
                Ok(true)
            } else {
                let s = view.get_scalar(slot)?;
                Ok(!scalars_eq(&s, rhs))
            }
        }
        Filter::In(_, vals) => {
            let s = view.get_scalar(slot)?;
            Ok(vals.iter().any(|rhs| scalars_eq(&s, rhs)))
        }
        Filter::NotIn(_, vals) => {
            let s = view.get_scalar(slot)?;
            Ok(!vals.iter().any(|rhs| scalars_eq(&s, rhs)))
        }
        Filter::Gt(_, rhs) => Ok(view
            .get_scalar(slot)?
            .cmp_ordered(rhs)
            .is_some_and(|o| o == std::cmp::Ordering::Greater)),
        Filter::Gte(_, rhs) => Ok(view
            .get_scalar(slot)?
            .cmp_ordered(rhs)
            .is_some_and(|o| o != std::cmp::Ordering::Less)),
        Filter::Lt(_, rhs) => Ok(view
            .get_scalar(slot)?
            .cmp_ordered(rhs)
            .is_some_and(|o| o == std::cmp::Ordering::Less)),
        Filter::Lte(_, rhs) => Ok(view
            .get_scalar(slot)?
            .cmp_ordered(rhs)
            .is_some_and(|o| o != std::cmp::Ordering::Greater)),
        Filter::Contains(_, rhs) => {
            let val = view.decode_attr_value(slot)?;
            Ok(attr_array_contains(&val, rhs))
        }
        Filter::ContainsAny(_, rhs_list) => {
            let val = view.decode_attr_value(slot)?;
            Ok(rhs_list.iter().any(|r| attr_array_contains(&val, r)))
        }
        _ => Ok(false),
    }
}

#[inline]
fn scalars_eq(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Null, ScalarValue::Null) => true,
        (ScalarValue::Bool(x), ScalarValue::Bool(y)) => x == y,
        (ScalarValue::Int(x), ScalarValue::Int(y)) => x == y,
        (ScalarValue::Uint(x), ScalarValue::Uint(y)) => x == y,
        (ScalarValue::Float(x), ScalarValue::Float(y)) => !x.is_nan() && !y.is_nan() && x == y,
        (ScalarValue::String(x), ScalarValue::String(y)) => x == y,
        _ => false,
    }
}

trait ScalarCmpExt {
    fn cmp_ordered(&self, other: &ScalarValue) -> Option<std::cmp::Ordering>;
}

impl ScalarCmpExt for ScalarValue {
    fn cmp_ordered(&self, other: &ScalarValue) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (ScalarValue::Int(x), ScalarValue::Int(y)) => Some(x.cmp(y)),
            (ScalarValue::Uint(x), ScalarValue::Uint(y)) => Some(x.cmp(y)),
            (ScalarValue::Float(x), ScalarValue::Float(y)) => {
                if x.is_nan() || y.is_nan() {
                    None
                } else {
                    let xn = if *x == 0.0 { 0.0 } else { *x };
                    let yn = if *y == 0.0 { 0.0 } else { *y };
                    Some(xn.total_cmp(&yn))
                }
            }
            (ScalarValue::String(x), ScalarValue::String(y)) => Some(x.cmp(y)),
            _ => None,
        }
    }
}

#[inline]
fn attr_array_contains(val: &AttributeValue, scalar: &ScalarValue) -> bool {
    match (val, scalar) {
        (AttributeValue::IntArray(arr), ScalarValue::Int(v)) => arr.contains(v),
        (AttributeValue::UintArray(arr), ScalarValue::Uint(v)) => arr.contains(v),
        (AttributeValue::StringArray(arr), ScalarValue::String(v)) => arr.contains(v),
        _ => false,
    }
}

/// Test if a `ColumnBlockView` slot's encoded values match `encoded_targets` (ordered val bytes).
pub(crate) fn block_slot_matches_any_encoded(
    view: &ColumnBlockView<'_>,
    slot: u8,
    encoded_targets: &[Vec<u8>],
) -> Result<bool> {
    let mut matched = false;
    let mut buf = Vec::new();
    view.for_each_encoded_value(slot, &mut buf, |enc| {
        if !matched && encoded_targets.binary_search_by(|k| k.as_slice().cmp(enc)).is_ok() {
            matched = true;
        }
    })?;
    Ok(matched)
}

pub(crate) type OverwrittenTrueValues = (
    BTreeMap<SegmentId, RoaringBitmap>,
    BTreeMap<Vec<u8>, BTreeMap<SegmentId, RoaringBitmap>>,
);

/// Collect corrected detached value -> `(SegmentId -> RoaringBitmap)` overlay for `0x14` rows
/// across all segments for `(entry, col_gen)` (used by `order_by` streaming when `0x14` non-empty).
pub(crate) fn collect_overwritten_true_values(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    entry: &SchemaEntry,
    col_gen: ColumnGen,
) -> Result<OverwrittenTrueValues> {
    let ow_list = storage.all_detached_overwritten(entry.id, col_gen)?;
    let mut ow_live_by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    let mut true_val_by_seg: BTreeMap<Vec<u8>, BTreeMap<SegmentId, RoaringBitmap>> =
        BTreeMap::new();
    if ow_list.is_empty() {
        return Ok((ow_live_by_seg, true_val_by_seg));
    }

    let mut keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = Vec::new();
    let mut seg_blocks: Vec<(SegmentId, u32, Vec<u8>)> = Vec::new();
    for (seg, ow_bm) in ow_list {
        let Some(live_bm) = snap.live_bitmap(seg) else {
            continue;
        };
        let ow_live = ow_bm & live_bm;
        if ow_live.is_empty() {
            continue;
        }
        for local in ow_live.iter() {
            let blk = local >> 8;
            let slot = (local & 0xFF) as u8;
            if let Some(last) = seg_blocks.last_mut()
                && last.0 == seg
                && last.1 == blk
            {
                last.2.push(slot);
            } else {
                seg_blocks.push((seg, blk, vec![slot]));
                keys.push((entry.id, col_gen, seg, blk));
            }
        }
        ow_live_by_seg.insert(seg, ow_live);
    }

    let slices = storage.multi_get_detached_blocks_pinned(&keys, true)?;
    let mut buf = Vec::new();
    for ((seg, blk, slots), opt_slice) in seg_blocks.iter().zip(slices.iter()) {
        let Some(slice) = opt_slice.as_ref() else {
            continue;
        };
        let view = ColumnBlockView::parse(slice)?;
        for &slot in slots {
            let local = (blk << 8) | (slot as u32);
            view.for_each_encoded_value(slot, &mut buf, |enc| {
                true_val_by_seg
                    .entry(enc.to_vec())
                    .or_default()
                    .entry(*seg)
                    .or_default()
                    .insert(local);
            })?;
        }
    }
    Ok((ow_live_by_seg, true_val_by_seg))
}

/// Evaluate a mixed `Filter` for a list of sorted `RowId`s in batches (loading `cf_detached` and,
/// only if inline/FTS predicates exist, `cf_rows`).
pub(crate) fn filter_rows_mixed_batch(
    storage: &Storage,
    schema: &CollectionSchema,
    fts: &FtsIndex,
    filter: &Filter,
    rows: &[RowId],
) -> Result<Vec<bool>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let need_rows = filter_mentions_inline_or_fts(schema, filter);
    let mut det_attrs = Vec::new();
    collect_detached_filter_attr_ids(schema, filter, &mut det_attrs);
    det_attrs.sort_unstable();
    let gens = all_detached_gens(storage)?;

    let row_slices = if need_rows {
        storage.multi_get_row_bytes_pinned(rows)?
    } else {
        Vec::new()
    };

    let mut det_keys: Vec<(AttrId, ColumnGen, SegmentId, u32)> = Vec::new();
    for &attr_id in &det_attrs {
        let g = gens.get(&attr_id).copied().unwrap_or(0);
        for &row in rows {
            let k = (attr_id, g, row.seg(), row.block_idx());
            if det_keys.last().copied() != Some(k) {
                det_keys.push(k);
            }
        }
    }
    let det_slices = storage.multi_get_detached_blocks_pinned(&det_keys, true)?;
    let mut det_views_all: HashMap<(AttrId, SegmentId, u32), ColumnBlockView<'_>> = HashMap::new();
    for (k, opt_slice) in det_keys.iter().zip(det_slices.iter()) {
        if let Some(s) = opt_slice.as_ref() {
            det_views_all.insert((k.0, k.2, k.3), ColumnBlockView::parse(s)?);
        }
    }

    let mut results = Vec::with_capacity(rows.len());
    let mut block_views: HashMap<AttrId, Option<ColumnBlockView<'_>>> =
        HashMap::with_capacity(det_attrs.len());
    for (idx, &row) in rows.iter().enumerate() {
        block_views.clear();
        for &attr_id in &det_attrs {
            block_views.insert(
                attr_id,
                det_views_all
                    .get(&(attr_id, row.seg(), row.block_idx()))
                    .copied(),
            );
        }
        let rv = if need_rows {
            let Some(slice) = row_slices[idx].as_ref() else {
                results.push(false);
                continue;
            };
            Some(RowView::parse(slice)?)
        } else {
            None
        };
        results.push(matches_filter_mixed(
            schema,
            filter,
            rv.as_ref(),
            &block_views,
            row.slot(),
            fts,
        )?);
    }
    Ok(results)
}
