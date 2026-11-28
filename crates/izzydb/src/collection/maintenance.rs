//! ANN vacuum, compaction, and tree maintenance.


use izzydb_core::config::SegmentId;
use izzydb_core::error::Result;
use izzydb_storage::codec::{
    SEG_FLAG_HAS_COLUMNS, SegmentState,
};
use izzydb_storage::TxContext;

use crate::compaction::{
    self, CompactionHookStage, CompactionOptions, CompactionStats,
};
use crate::detached::{
    self,
};
use crate::tree::{
    MaintenanceOptions, MaintenanceStats, VacuumBudget, VacuumStats, vacuum_ann_storage, verify_tree_invariants,
};
use crate::TreeStats;

use super::*;

impl Collection {
    /// Incrementally vacuum stale ANN slots and merge/rebalance underflowing leaves up to `budget`,
    /// retiring any `Forwarding` segments whose forwarding epoch has completed.
    pub fn vacuum_ann(&mut self, budget: VacuumBudget) -> Result<VacuumStats> {
        let Some(ref ann) = self.ann else {
            return Ok(VacuumStats::default());
        };
        let stats = vacuum_ann_storage(
            &self.storage,
            ann,
            &mut self.manifest,
            &mut self.ann_meta,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            budget,
        )?;
        let _ = compaction::retire_eligible_forwarding_segments(self)?;
        Ok(stats)
    }

    /// Merge eligible sealed segments (high tombstone ratio or undersized), forward RowIds,
    /// sweep ANN references, and retire obsolete segments.
    pub fn compact(&mut self, opts: CompactionOptions) -> Result<CompactionStats> {
        compaction::run_compact(self, opts, &mut |_| Ok(()))
    }

    /// Test-only compaction entrypoint invoking `hook` at each phase boundary.
    #[doc(hidden)]
    pub fn compact_with_hook<F>(
        &mut self,
        opts: CompactionOptions,
        hook: &mut F,
    ) -> Result<CompactionStats>
    where
        F: FnMut(CompactionHookStage) -> Result<()>,
    {
        compaction::run_compact(self, opts, hook)
    }

    /// Run background maintenance: build deferred `cf_columns`, vacuum ANN if stale ratio exceeds
    /// threshold, compact eligible segments, rebuild fragmented detached columns, and tighten FTS
    /// stats when segments were forwarded/retired.
    pub fn maintenance(&mut self, opts: MaintenanceOptions) -> Result<MaintenanceStats> {
        let mut stats = MaintenanceStats::default();
        let pending_column_segs: Vec<SegmentId> = {
            let snap = self.storage.segment_snapshot();
            let mut segs: Vec<SegmentId> = snap
                .segments
                .iter()
                .filter(|(_, e)| {
                    e.meta.state == SegmentState::Sealed
                        && (e.meta.flags & SEG_FLAG_HAS_COLUMNS) == 0
                })
                .map(|(&seg, _)| seg)
                .collect();
            segs.sort_unstable();
            segs
        };
        if !pending_column_segs.is_empty() {
            let mut tx = TxContext::new(
                &self.storage,
                &mut self.internal_cache,
                &mut self.leaf_parent,
                self.manifest.clone(),
                self.ann_meta.clone(),
            );
            for seg in pending_column_segs {
                crate::columns::build_segment_columns_in_tx(&mut tx, &self.schema, seg)?;
                if self.config.bit_sliced_index {
                    crate::bsi::build_segment_bsi_in_tx(&mut tx, &self.schema, seg)?;
                }
                stats.columns_built += 1;
            }
            let (manifest, ann_meta) = tx.commit()?;
            self.manifest = manifest;
            self.ann_meta = ann_meta;
        }
        if let Some(ref meta) = self.ann_meta {
            let total = meta.num_vectors + meta.stale_rows;
            if total > 0 && (meta.stale_rows as f64 / total as f64) > opts.vacuum_stale_ratio {
                stats.vacuum = Some(self.vacuum_ann(opts.vacuum_budget)?);
            }
        }
        let has_pending_detached = self
            .storage
            .all_detached_metas()?
            .into_iter()
            .any(|(_, m)| m.pending_gen != 0);
        if !has_pending_detached {
            stats.compaction = self.compact(opts.compaction)?;
            if stats.compaction.segments_forwarded > 0 || stats.compaction.segments_retired > 0 {
                self.compact_fts_stats()?;
            }
        }
        stats.detached_rebuilt = detached::rebuild_fragmented_detached_columns(self)?;
        Ok(stats)
    }

    /// Verify ANN tree structural invariants against `SegmentTableSnapshot`.
    pub fn verify_ann_invariants(&self) -> Result<Option<TreeStats>> {
        match (&self.ann, &self.ann_meta) {
            (Some(ann), Some(meta)) => Ok(Some(verify_tree_invariants(
                &self.storage,
                &self.internal_cache,
                &self.leaf_parent,
                meta,
                &ann.config,
            )?)),
            _ => Ok(None),
        }
    }

    /// Recompute exact live `TermStats` (`doc_freq`, `max_tf`, `min_dl_over_tf`) across all FTS
    /// terms by scanning live posting slots against `SegmentTableSnapshot`.
    pub fn compact_fts_stats(&mut self) -> Result<()> {
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            self.ann_meta.clone(),
        );
        let updated = self.fts_index.compact_term_stats(&self.storage, &mut tx)?;
        if updated > 0 {
            let (manifest, ann_meta) = tx.commit()?;
            self.manifest = manifest;
            self.ann_meta = ann_meta;
        }
        Ok(())
    }
}
