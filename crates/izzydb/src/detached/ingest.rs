//! Whole-column build and atomic publish ingest.

use std::collections::{BTreeMap, HashMap};

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, ColumnGen, DocId, ROW_BLOCK_SIZE, RowId, SegmentId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rowset::Liveness;
use izzydb_core::schema::{
    AttributeType, AttributeValue, ColumnStorage,
    SchemaEntry,
};
use izzydb_storage::codec::{
    SegPosting, ZoneEntry,
    detached_count_suffix, detached_key, detached_posting_key, extract_index_keys_for_attr,
    ordered_type_tag,
};
use izzydb_storage::column::ColumnBlockBuilder;
use izzydb_storage::{ColumnBlockView, TxContext};

use crate::collection::Collection;
use crate::segments::RowResolver;


use super::*;

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
