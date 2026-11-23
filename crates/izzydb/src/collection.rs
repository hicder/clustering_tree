//! High-level `Collection` API coordinating the RocksDB [`Storage`],
//! [`SegmentStore`], and secondary indexes (ANN, attribute index, FTS).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use izzydb_core::config::{DocId, NodeId, RowId, SegmentId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::facet::{
    FacetResults, FacetSpec, FacetedSearchResponse, HybridFacetScope, VectorFacetScope,
    validate_facet_specs,
};
use izzydb_core::filter::Filter;
use izzydb_core::metric::prepare_vector;
use izzydb_core::rank::{OrderBySpec, OrderedResult, RankBy};
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::{AttributeValue, Attributes, CollectionConfig, CollectionSchema, ColumnStorage};
use izzydb_fts::Bm25Result;
use izzydb_storage::codec::{
    Manifest, RowView, decode_row_attributes, decode_row_entries, encode_row,
};
use izzydb_storage::node::{AnnMeta, PackedInternalNode, decode_f32_slice};
use izzydb_storage::{SegmentTableSnapshot, Storage, TxContext};

use crate::attr_index::AttributeIndex;
use crate::facet::{FacetEnv, Population};
use crate::filter_plan::{self, FilterPlanTrace};
use crate::fts_index::FtsIndex;
use crate::search;
use crate::segments::{RowIndexHook, RowResolver, SegmentStore};
use crate::tree::{
    AnnIndex, MaintenanceOptions, MaintenanceStats, VacuumBudget, VacuumStats,
    load_internal_caches, vacuum_ann_storage, verify_tree_invariants,
};
use crate::{CollectionStats, SearchOptions, TreeStats};

/// Partial document mutation applied to an existing document.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocumentPatch {
    /// `Some(vec)` replaces the stored vector; `None` leaves the vector untouched.
    pub embedding: Option<Vec<f32>>,
    /// Attribute changes merged onto the existing document.
    /// Setting an attribute to `AttributeValue::Null` clears it; omitted attributes are preserved.
    pub attributes: Attributes,
}

impl DocumentPatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_attribute(mut self, name: impl Into<String>, value: AttributeValue) -> Self {
        self.attributes.insert(name.into(), value);
        self
    }

    pub fn clear_attribute(mut self, name: impl Into<String>) -> Self {
        self.attributes.insert(name.into(), AttributeValue::Null);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.embedding.is_none() && self.attributes.is_empty()
    }
}

/// Stored document returned by [`Collection::get`] and accepted by [`Collection::upsert_batch`].
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub id: DocId,
    pub embedding: Option<Vec<f32>>,
    pub attributes: Attributes,
}

/// Vector search hit returned by [`Collection::search`].
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: DocId,
    pub distance: f32,
}

/// Options for hybrid vector + BM25 search fused via Reciprocal Rank Fusion (RRF).
#[derive(Debug, Clone, Copy)]
pub struct HybridSearchOptions {
    pub k: usize,
    pub vector_search: SearchOptions,
    pub bm25_candidates: Option<usize>,
    pub rrf_k: f32,
    pub vector_weight: f32,
    pub bm25_weight: f32,
}

impl HybridSearchOptions {
    pub fn new(k: usize, vector_search: SearchOptions) -> Self {
        Self {
            k,
            vector_search,
            bm25_candidates: None,
            rrf_k: 60.0,
            vector_weight: 1.0,
            bm25_weight: 1.0,
        }
    }

    pub fn with_bm25_candidates(mut self, bm25_candidates: usize) -> Self {
        self.bm25_candidates = Some(bm25_candidates);
        self
    }

    pub fn with_rrf_k(mut self, rrf_k: f32) -> Self {
        self.rrf_k = rrf_k;
        self
    }

    pub fn with_weights(mut self, vector_weight: f32, bm25_weight: f32) -> Self {
        self.vector_weight = vector_weight;
        self.bm25_weight = bm25_weight;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if !self.rrf_k.is_finite() || self.rrf_k < 0.0 {
            return Err(TreeError::InvalidQuery(format!(
                "rrf_k must be finite and >= 0.0, got {}",
                self.rrf_k
            )));
        }
        if !self.vector_weight.is_finite() || self.vector_weight < 0.0 {
            return Err(TreeError::InvalidQuery(format!(
                "vector_weight must be finite and >= 0.0, got {}",
                self.vector_weight
            )));
        }
        if !self.bm25_weight.is_finite() || self.bm25_weight < 0.0 {
            return Err(TreeError::InvalidQuery(format!(
                "bm25_weight must be finite and >= 0.0, got {}",
                self.bm25_weight
            )));
        }
        Ok(())
    }
}

/// Hybrid search hit returned by [`Collection::hybrid_search`].
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchResult {
    pub id: DocId,
    pub score: f32,
    pub vector_rank: Option<usize>,
    pub bm25_rank: Option<usize>,
}

/// Top-level document + segment + secondary index collection.
pub struct Collection {
    pub(crate) storage: Arc<Storage>,
    pub(crate) config: CollectionConfig,
    pub(crate) schema: CollectionSchema,
    pub(crate) segments: SegmentStore,
    pub(crate) manifest: Manifest,
    pub(crate) ann_meta: Option<AnnMeta>,
    pub(crate) ann: Option<AnnIndex>,
    pub(crate) attr_index: AttributeIndex,
    pub(crate) fts_index: FtsIndex,
    pub(crate) internal_cache: HashMap<NodeId, PackedInternalNode>,
    pub(crate) leaf_parent: HashMap<NodeId, NodeId>,
}

impl Collection {
    /// Open or create a [`Collection`] at `path` with the given `config`.
    pub fn open(path: impl AsRef<Path>, config: CollectionConfig) -> Result<Self> {
        config.validate()?;
        let (storage, manifest, ann_meta) = Storage::open(path, &config)?;
        let schema = match storage.get_schema()? {
            Some(mut existing) => {
                let changed = existing.reconcile(&config)?;
                if changed {
                    storage.put_schema(&existing)?;
                }
                existing
            }
            None => {
                if manifest.total_rows > 0 {
                    return Err(TreeError::InvalidSchema(
                        "existing non-empty database has no CollectionSchema".into(),
                    ));
                }
                let created = CollectionSchema::from_config(&config)?;
                storage.put_schema(&created)?;
                created
            }
        };

        let storage = Arc::new(storage);
        let segments = SegmentStore::new(config.segment.seal_rows);
        let (ann, internal_cache, leaf_parent) = match (&config.vector, &ann_meta) {
            (Some(vcfg), Some(meta)) => {
                let ann = AnnIndex::new(vcfg.clone(), meta.rabitq_seed)?;
                let (internal_cache, leaf_parent) = load_internal_caches(&storage)?;
                (Some(ann), internal_cache, leaf_parent)
            }
            _ => (None, HashMap::new(), HashMap::new()),
        };
        Ok(Self {
            storage,
            config,
            schema,
            segments,
            manifest,
            ann_meta,
            ann,
            attr_index: AttributeIndex,
            fts_index: FtsIndex::new(),
            internal_cache,
            leaf_parent,
        })
    }

    pub fn schema(&self) -> &CollectionSchema {
        &self.schema
    }

    pub fn collection_config(&self) -> &CollectionConfig {
        &self.config
    }

    pub fn vector_config(&self) -> Option<&TreeConfig> {
        self.config.vector.as_ref()
    }

    pub fn ann_meta(&self) -> Option<&AnnMeta> {
        self.ann_meta.as_ref()
    }

    pub fn len(&self) -> u64 {
        self.manifest.live_rows
    }

    pub fn is_empty(&self) -> bool {
        self.manifest.live_rows == 0
    }

    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

    pub fn segment_snapshot(&self) -> Arc<SegmentTableSnapshot> {
        self.storage.segment_snapshot()
    }

    fn prepare_optional_vector(&self, embedding: Option<&[f32]>) -> Result<Option<Vec<f32>>> {
        match (embedding, &self.config.vector) {
            (Some(emb), Some(vcfg)) => Ok(Some(prepare_vector(emb, vcfg.dim, vcfg.metric)?)),
            (Some(_), None) => Err(TreeError::InvalidConfig(
                "cannot store vector in a collection without a vector index".into(),
            )),
            (None, _) => Ok(None),
        }
    }

    fn reject_detached_attributes(&self, attrs: &Attributes) -> Result<()> {
        for name in attrs.keys() {
            if let Some(entry) = self.schema.get_by_name(name)
                && entry.storage == ColumnStorage::Detached
            {
                return Err(TreeError::Unsupported(
                    "v3: phase 8 (detached attributes)".into(),
                ));
            }
        }
        Ok(())
    }

    /// Force sealing the current open segment, if one exists and is non-empty.
    pub fn seal(&mut self) -> Result<Option<SegmentId>> {
        let Some(open_seg) = self.manifest.open_segment else {
            return Ok(None);
        };
        let hooks = collect_hooks(self.ann.as_ref(), &self.attr_index, &self.fts_index);
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            self.ann_meta.clone(),
        );
        self.segments.seal(&mut tx, open_seg, &hooks)?;
        let (manifest, ann_meta) = tx.commit()?;
        self.manifest = manifest;
        self.ann_meta = ann_meta;
        Ok(Some(open_seg))
    }

    /// Insert or update a single document atomically.
    pub fn upsert(
        &mut self,
        id: DocId,
        embedding: Option<&[f32]>,
        attributes: Attributes,
    ) -> Result<()> {
        self.schema.validate_attributes(&attributes)?;
        self.reject_detached_attributes(&attributes)?;
        let prepared_vec = self.prepare_optional_vector(embedding)?;
        let new_row_bytes = encode_row(&self.schema, &attributes, prepared_vec.is_some())?;

        let hooks = collect_hooks(self.ann.as_ref(), &self.attr_index, &self.fts_index);
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            self.ann_meta.clone(),
        );
        let changed = upsert_one_in_tx(
            &self.segments,
            &self.schema,
            &mut tx,
            id,
            prepared_vec.as_deref(),
            new_row_bytes,
            &hooks,
        )?;
        if changed {
            let (manifest, ann_meta) = tx.commit()?;
            self.manifest = manifest;
            self.ann_meta = ann_meta;
        }
        Ok(())
    }

    /// Insert or update a batch of documents atomically in a single transaction,
    /// rolling over segments across `seal_rows` boundaries as needed.
    pub fn upsert_batch(&mut self, docs: &[Document]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let mut prepared = Vec::with_capacity(docs.len());
        for doc in docs {
            self.schema.validate_attributes(&doc.attributes)?;
            self.reject_detached_attributes(&doc.attributes)?;
            let pvec = self.prepare_optional_vector(doc.embedding.as_deref())?;
            let row_bytes = encode_row(&self.schema, &doc.attributes, pvec.is_some())?;
            prepared.push((doc.id, pvec, row_bytes));
        }

        let hooks = collect_hooks(self.ann.as_ref(), &self.attr_index, &self.fts_index);
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            self.ann_meta.clone(),
        );
        let mut any_changed = false;
        for (id, pvec, row_bytes) in prepared {
            if upsert_one_in_tx(
                &self.segments,
                &self.schema,
                &mut tx,
                id,
                pvec.as_deref(),
                row_bytes,
                &hooks,
            )? {
                any_changed = true;
            }
        }
        if any_changed {
            let (manifest, ann_meta) = tx.commit()?;
            self.manifest = manifest;
            self.ann_meta = ann_meta;
        }
        Ok(())
    }

    /// Apply a partial update (`DocumentPatch`) to an existing document atomically.
    pub fn patch(&mut self, id: DocId, patch: DocumentPatch) -> Result<()> {
        self.schema.validate_attributes(&patch.attributes)?;
        self.reject_detached_attributes(&patch.attributes)?;
        let prepared_vec = self.prepare_optional_vector(patch.embedding.as_deref())?;

        let hooks = collect_hooks(self.ann.as_ref(), &self.attr_index, &self.fts_index);
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            self.ann_meta.clone(),
        );
        let changed = apply_patch_in_tx(
            &self.segments,
            &self.schema,
            &mut tx,
            id,
            prepared_vec.as_deref(),
            &patch.attributes,
            &hooks,
        )?;
        if changed {
            let (manifest, ann_meta) = tx.commit()?;
            self.manifest = manifest;
            self.ann_meta = ann_meta;
        }
        Ok(())
    }

    /// Validate all patches up front and apply them atomically in a single transaction.
    pub fn patch_batch(&mut self, patches: &[(DocId, DocumentPatch)]) -> Result<()> {
        if patches.is_empty() {
            return Ok(());
        }
        let mut prepared_vecs = Vec::with_capacity(patches.len());
        let mut doc_ids = Vec::with_capacity(patches.len());
        for &(id, ref patch) in patches {
            self.schema.validate_attributes(&patch.attributes)?;
            self.reject_detached_attributes(&patch.attributes)?;
            prepared_vecs.push(self.prepare_optional_vector(patch.embedding.as_deref())?);
            doc_ids.push(id);
        }

        let existing = self.storage.multi_get_id_map(&doc_ids)?;
        for (&id, opt) in doc_ids.iter().zip(existing) {
            if opt.is_none() {
                return Err(TreeError::DocumentNotFound(id));
            }
        }

        let hooks = collect_hooks(self.ann.as_ref(), &self.attr_index, &self.fts_index);
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            self.ann_meta.clone(),
        );
        let mut any_changed = false;
        for (&(id, ref patch), pv) in patches.iter().zip(prepared_vecs.iter()) {
            if apply_patch_in_tx(
                &self.segments,
                &self.schema,
                &mut tx,
                id,
                pv.as_deref(),
                &patch.attributes,
                &hooks,
            )? {
                any_changed = true;
            }
        }
        if any_changed {
            let (manifest, ann_meta) = tx.commit()?;
            self.manifest = manifest;
            self.ann_meta = ann_meta;
        }
        Ok(())
    }

    /// Delete a document atomically. Returns `true` if `id` existed.
    pub fn delete(&mut self, id: DocId) -> Result<bool> {
        let hooks = collect_hooks(self.ann.as_ref(), &self.attr_index, &self.fts_index);
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            self.ann_meta.clone(),
        );
        let Some(old_row) = tx.get_id_map(id)? else {
            return Ok(false);
        };
        let old_bytes = tx.get_row_bytes(old_row)?.ok_or_else(|| {
            TreeError::InvariantViolation(format!("row {old_row:?} missing in cf_rows"))
        })?;
        let old_view = RowView::parse(&old_bytes)?;
        self.segments.tombstone(&mut tx, old_row)?;
        for hook in &hooks {
            hook.on_delete(&mut tx, &self.schema, old_row, &old_view, old_view.has_vector())?;
        }
        tx.delete_id_map(id);
        let (manifest, ann_meta) = tx.commit()?;
        self.manifest = manifest;
        self.ann_meta = ann_meta;
        Ok(true)
    }

    /// Delete a batch of documents (deduplicating and ignoring missing IDs) atomically.
    pub fn delete_batch(&mut self, ids: &[DocId]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }
        let mut unique_ids = ids.to_vec();
        unique_ids.sort_unstable();
        unique_ids.dedup();

        let row_opts = self.storage.multi_get_id_map(&unique_ids)?;
        let mut targets: Vec<(DocId, RowId)> = unique_ids
            .into_iter()
            .zip(row_opts)
            .filter_map(|(doc_id, opt_row)| opt_row.map(|r| (doc_id, r)))
            .collect();
        if targets.is_empty() {
            return Ok(0);
        }

        let hooks = collect_hooks(self.ann.as_ref(), &self.attr_index, &self.fts_index);
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            self.ann_meta.clone(),
        );
        targets.sort_unstable_by_key(|&(_, r)| r);
        for &(doc_id, old_row) in &targets {
            let old_bytes = tx.get_row_bytes(old_row)?.ok_or_else(|| {
                TreeError::InvariantViolation(format!("row {old_row:?} missing in cf_rows"))
            })?;
            let old_view = RowView::parse(&old_bytes)?;
            self.segments.tombstone(&mut tx, old_row)?;
            for hook in &hooks {
                hook.on_delete(&mut tx, &self.schema, old_row, &old_view, old_view.has_vector())?;
            }
            tx.delete_id_map(doc_id);
        }
        let (manifest, ann_meta) = tx.commit()?;
        self.manifest = manifest;
        self.ann_meta = ann_meta;
        Ok(targets.len() as u64)
    }

    /// Retrieve a document's stored vector and inline attributes by `id`.
    pub fn get(&self, id: DocId) -> Result<Option<Document>> {
        let Some(row) = self.storage.get_id_map(id)? else {
            return Ok(None);
        };
        let row_bytes = self.storage.get_row_bytes_pinned(row)?.ok_or_else(|| {
            TreeError::InvariantViolation(format!("doc {id} mapped to {row:?} missing in cf_rows"))
        })?;
        let view = RowView::parse(&row_bytes)?;
        let embedding = if view.has_vector() {
            Some(self.storage.get_vector(row)?.ok_or_else(|| {
                TreeError::InvariantViolation(format!("row {row:?} has_vector=1 missing in cf_vectors"))
            })?)
        } else {
            None
        };
        let attributes = decode_row_attributes(&self.schema, &row_bytes)?;
        Ok(Some(Document {
            id,
            embedding,
            attributes,
        }))
    }

    /// Batched `get` sorting lookups by `RowId` for segment locality.
    pub fn get_batch(&self, ids: &[DocId]) -> Result<Vec<Option<Document>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mapped = self.storage.multi_get_id_map(ids)?;
        let mut to_fetch: Vec<(RowId, usize, DocId)> = Vec::with_capacity(ids.len());
        for (slot, (&doc_id, opt_row)) in ids.iter().zip(mapped.iter()).enumerate() {
            if let Some(row) = opt_row {
                to_fetch.push((*row, slot, doc_id));
            }
        }
        let mut out: Vec<Option<Document>> = vec![None; ids.len()];
        if to_fetch.is_empty() {
            return Ok(out);
        }
        to_fetch.sort_unstable_by_key(|&(row, _, _)| row);
        let sorted_rows: Vec<RowId> = to_fetch.iter().map(|&(r, _, _)| r).collect();
        let row_slices = self.storage.multi_get_row_bytes_pinned(&sorted_rows)?;

        let mut vec_rows: Vec<(RowId, usize)> = Vec::new();
        let mut decoded_attrs: Vec<Option<(usize, DocId, Attributes)>> =
            Vec::with_capacity(to_fetch.len());

        for (i, (&(row, slot, doc_id), opt_slice)) in
            to_fetch.iter().zip(row_slices.iter()).enumerate()
        {
            let slice = opt_slice.as_ref().ok_or_else(|| {
                TreeError::InvariantViolation(format!("row {row:?} missing in cf_rows"))
            })?;
            let view = RowView::parse(slice)?;
            if view.has_vector() {
                vec_rows.push((row, i));
            }
            let attributes = decode_row_attributes(&self.schema, slice)?;
            decoded_attrs.push(Some((slot, doc_id, attributes)));
        }

        let mut embeddings: Vec<Option<Vec<f32>>> = vec![None; to_fetch.len()];
        if !vec_rows.is_empty() {
            let dim = self.storage.dim();
            let vrows: Vec<RowId> = vec_rows.iter().map(|&(r, _)| r).collect();
            let vec_slices = self.storage.multi_get_vectors_pinned(&vrows)?;
            for (&(row, idx), opt_vec) in vec_rows.iter().zip(vec_slices.iter()) {
                let slice = opt_vec.as_ref().ok_or_else(|| {
                    TreeError::InvariantViolation(format!(
                        "row {row:?} has_vector=1 missing in cf_vectors"
                    ))
                })?;
                embeddings[idx] = Some(decode_f32_slice(slice, dim)?);
            }
        }

        for (idx, item) in decoded_attrs.into_iter().enumerate() {
            if let Some((slot, doc_id, attributes)) = item {
                out[slot] = Some(Document {
                    id: doc_id,
                    embedding: embeddings[idx].take(),
                    attributes,
                });
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Stubbed subsystems (ported in Phases 3-6, 11)
    // -----------------------------------------------------------------------

    pub fn search(
        &self,
        query: &[f32],
        opts: SearchOptions,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        let mut trace = FilterPlanTrace::default();
        self.search_with_trace(query, opts, filter, &mut trace)
    }

    pub fn search_with_trace(
        &self,
        query: &[f32],
        opts: SearchOptions,
        filter: Option<&Filter>,
        trace: &mut FilterPlanTrace,
    ) -> Result<Vec<SearchResult>> {
        let row_hits = self.search_row_hits(query, opts, filter, trace)?;
        self.resolve_search_results(row_hits)
    }

    pub(crate) fn search_row_hits(
        &self,
        query: &[f32],
        opts: SearchOptions,
        filter: Option<&Filter>,
        trace: &mut FilterPlanTrace,
    ) -> Result<Vec<(RowId, f32)>> {
        let (Some(ann), Some(ann_meta)) = (&self.ann, &self.ann_meta) else {
            return Err(TreeError::InvalidConfig(
                "collection has no vector index".into(),
            ));
        };
        match filter {
            None => {
                trace.regime = Some(filter_plan::FilteredSearchRegime::Unfiltered);
                search::search(
                    &self.storage,
                    &self.internal_cache,
                    ann_meta,
                    &ann.quantizer,
                    query,
                    opts,
                    &ann.config,
                )
            }
            Some(f) => search::search_filtered_with_trace(
                &self.storage,
                &self.internal_cache,
                &self.leaf_parent,
                ann_meta,
                &ann.quantizer,
                &self.schema,
                &self.fts_index,
                query,
                opts,
                f,
                &ann.config,
                trace,
            ),
        }
    }

    fn resolve_search_results(&self, row_hits: Vec<(RowId, f32)>) -> Result<Vec<SearchResult>> {
        if row_hits.is_empty() {
            return Ok(Vec::new());
        }
        let mut sorted = row_hits;
        sorted.sort_unstable_by_key(|&(row, _)| row);
        let rows: Vec<RowId> = sorted.iter().map(|&(r, _)| r).collect();
        let doc_ids = RowResolver::resolve_all(&self.storage, &rows)?;
        let mut results: Vec<SearchResult> = doc_ids
            .into_iter()
            .zip(sorted)
            .map(|(id, (_, distance))| SearchResult { id, distance })
            .collect();
        results.sort_unstable_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(results)
    }

    /// Incrementally vacuum stale ANN slots and merge/rebalance underflowing leaves up to `budget`.
    pub fn vacuum_ann(&mut self, budget: VacuumBudget) -> Result<VacuumStats> {
        let Some(ref ann) = self.ann else {
            return Ok(VacuumStats::default());
        };
        vacuum_ann_storage(
            &self.storage,
            ann,
            &mut self.manifest,
            &mut self.ann_meta,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            budget,
        )
    }

    /// Run background maintenance (ANN vacuum when stale ratio exceeds threshold; Phase 9 adds compaction).
    pub fn maintenance(&mut self, opts: MaintenanceOptions) -> Result<MaintenanceStats> {
        let mut stats = MaintenanceStats::default();
        if let Some(ref meta) = self.ann_meta {
            let total = meta.num_vectors + meta.stale_rows;
            if total > 0 && (meta.stale_rows as f64 / total as f64) > opts.vacuum_stale_ratio {
                stats.vacuum = Some(self.vacuum_ann(opts.vacuum_budget)?);
            }
        }
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

    /// Resolve all live `DocId`s matching `filter`, sorted ascending.
    pub fn resolve_filter_doc_ids(&self, filter: &Filter) -> Result<Vec<DocId>> {
        filter.validate(&self.schema)?;
        let rowset = filter_plan::materialize_filter(
            &self.storage,
            &self.schema,
            filter,
            &self.fts_index,
        )?;
        let rows: Vec<RowId> = rowset.iter().collect();
        let mut doc_ids = RowResolver::resolve_all(&self.storage, &rows)?;
        doc_ids.sort_unstable();
        Ok(doc_ids)
    }

    /// Patch every document matching `filter` over a point-in-time snapshot, chunked at 512 docs
    /// to avoid the Halloween problem when patches preserve filter membership.
    pub fn update_by_filter(&mut self, filter: &Filter, patch: &DocumentPatch) -> Result<u64> {
        filter.validate(&self.schema)?;
        self.schema.validate_attributes(&patch.attributes)?;
        self.reject_detached_attributes(&patch.attributes)?;
        let prepared_vec = self.prepare_optional_vector(patch.embedding.as_deref())?;

        let doc_ids = self.resolve_filter_doc_ids(filter)?;
        if doc_ids.is_empty() {
            return Ok(0);
        }

        let matched = doc_ids.len() as u64;
        for chunk in doc_ids.chunks(512) {
            let hooks = collect_hooks(self.ann.as_ref(), &self.attr_index, &self.fts_index);
            let mut tx = TxContext::new(
                &self.storage,
                &mut self.internal_cache,
                &mut self.leaf_parent,
                self.manifest.clone(),
                self.ann_meta.clone(),
            );
            let mut any_changed = false;
            for &id in chunk {
                if apply_patch_in_tx(
                    &self.segments,
                    &self.schema,
                    &mut tx,
                    id,
                    prepared_vec.as_deref(),
                    &patch.attributes,
                    &hooks,
                )? {
                    any_changed = true;
                }
            }
            if any_changed {
                let (manifest, ann_meta) = tx.commit()?;
                self.manifest = manifest;
                self.ann_meta = ann_meta;
            }
        }
        Ok(matched)
    }

    /// Delete every document matching `filter` over a point-in-time snapshot, chunked at 512 docs.
    pub fn delete_by_filter(&mut self, filter: &Filter) -> Result<u64> {
        filter.validate(&self.schema)?;
        let doc_ids = self.resolve_filter_doc_ids(filter)?;
        if doc_ids.is_empty() {
            return Ok(0);
        }
        let mut total_deleted = 0u64;
        for chunk in doc_ids.chunks(512) {
            total_deleted += self.delete_batch(chunk)?;
        }
        Ok(total_deleted)
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

    pub fn query_bm25(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<Bm25Result>> {
        self.rank_query(rank_by, filter, top_k)
    }

    pub fn query(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<Bm25Result>> {
        self.rank_query(rank_by, filter, top_k)
    }

    pub fn hybrid_search(
        &self,
        query_vec: &[f32],
        rank_by: &RankBy,
        opts: HybridSearchOptions,
        filter: Option<&Filter>,
    ) -> Result<Vec<HybridSearchResult>> {
        let (results, _, _) = self.hybrid_search_with_legs(query_vec, rank_by, opts, filter)?;
        Ok(results)
    }

    fn hybrid_search_with_legs(
        &self,
        query_vec: &[f32],
        rank_by: &RankBy,
        opts: HybridSearchOptions,
        filter: Option<&Filter>,
    ) -> Result<(Vec<HybridSearchResult>, Vec<SearchResult>, Vec<Bm25Result>)> {
        opts.validate()?;
        rank_by.validate(&self.schema)?;
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        if self.ann.is_none() || self.ann_meta.is_none() {
            return Err(TreeError::InvalidConfig(
                "collection has no vector index".into(),
            ));
        }

        let vec_hits = self.search(query_vec, opts.vector_search, filter)?;
        let bm25_k = opts
            .bm25_candidates
            .unwrap_or_else(|| opts.k.saturating_mul(2).max(20));
        let bm25_hits = self.query_bm25(rank_by, filter, bm25_k)?;
        if opts.k == 0 {
            return Ok((Vec::new(), vec_hits, bm25_hits));
        }

        let mut by_doc: HashMap<DocId, (f32, Option<usize>, Option<usize>)> =
            HashMap::with_capacity(vec_hits.len() + bm25_hits.len());

        for (idx, hit) in vec_hits.iter().enumerate() {
            let rank = idx + 1;
            let contrib = opts.vector_weight / (opts.rrf_k + rank as f32);
            let entry = by_doc.entry(hit.id).or_insert((0.0, None, None));
            entry.0 += contrib;
            entry.1 = Some(rank);
        }
        for (idx, hit) in bm25_hits.iter().enumerate() {
            let rank = idx + 1;
            let contrib = opts.bm25_weight / (opts.rrf_k + rank as f32);
            let entry = by_doc.entry(hit.id).or_insert((0.0, None, None));
            entry.0 += contrib;
            entry.2 = Some(rank);
        }

        let mut results: Vec<HybridSearchResult> = by_doc
            .into_iter()
            .map(|(id, (score, vector_rank, bm25_rank))| HybridSearchResult {
                id,
                score,
                vector_rank,
                bm25_rank,
            })
            .collect();
        results.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.id.cmp(&b.id))
        });
        results.truncate(opts.k);
        Ok((results, vec_hits, bm25_hits))
    }

    pub fn order_by(
        &self,
        order_by: &[OrderBySpec],
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<OrderedResult>> {
        self.query_ordered(order_by, filter, top_k)
    }

    pub(crate) fn facet_env(&self) -> FacetEnv<'_> {
        FacetEnv {
            storage: &self.storage,
            schema: &self.schema,
            fts_index: &self.fts_index,
        }
    }

    pub fn facet(&self, filter: Option<&Filter>, facets: &[FacetSpec]) -> Result<FacetResults> {
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        let (res, _) = crate::facet::compute_facets(
            &self.facet_env(),
            facets,
            Population::Query {
                filter,
                rank_bm: None,
            },
        )?;
        Ok(res)
    }

    pub fn search_with_facets(
        &self,
        query: &[f32],
        opts: SearchOptions,
        filter: Option<&Filter>,
        facets: &[FacetSpec],
        scope: VectorFacetScope,
    ) -> Result<FacetedSearchResponse<SearchResult>> {
        validate_facet_specs(&self.schema, facets)?;
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        let mut trace = FilterPlanTrace::default();
        let (hits, facet_res, total_hits) = match scope {
            VectorFacetScope::FilterMatches => {
                let row_hits = self.search_row_hits(query, opts, filter, &mut trace)?;
                let hits = self.resolve_search_results(row_hits)?;
                let (facet_res, total_hits) = crate::facet::compute_facets(
                    &self.facet_env(),
                    facets,
                    Population::Query {
                        filter,
                        rank_bm: None,
                    },
                )?;
                (hits, facet_res, total_hits)
            }
            VectorFacetScope::TopK => {
                let row_hits = self.search_row_hits(query, opts, filter, &mut trace)?;
                let rs = RowSet::from_rows(row_hits.iter().map(|&(r, _)| r));
                let hits = self.resolve_search_results(row_hits)?;
                let (facet_res, total_hits) = crate::facet::compute_facets(
                    &self.facet_env(),
                    facets,
                    Population::RowSet(&rs),
                )?;
                (hits, facet_res, total_hits)
            }
            VectorFacetScope::Candidates(n) => {
                let cand_opts = SearchOptions {
                    k: n.max(opts.k),
                    ..opts
                };
                let row_hits = self.search_row_hits(query, cand_opts, filter, &mut trace)?;
                let mut sorted_rows = row_hits;
                sorted_rows.sort_unstable_by_key(|&(r, _)| r);
                let rows: Vec<RowId> = sorted_rows.iter().map(|&(r, _)| r).collect();
                let doc_ids = if rows.is_empty() {
                    Vec::new()
                } else {
                    RowResolver::resolve_all(&self.storage, &rows)?
                };
                let mut paired: Vec<(RowId, DocId, f32)> = sorted_rows
                    .into_iter()
                    .zip(doc_ids)
                    .map(|((r, dist), id)| (r, id, dist))
                    .collect();
                paired.sort_unstable_by(|a, b| a.2.total_cmp(&b.2).then_with(|| a.1.cmp(&b.1)));
                let rs = RowSet::from_rows(paired.iter().take(n).map(|&(r, _, _)| r));
                let hits: Vec<SearchResult> = paired
                    .into_iter()
                    .take(opts.k)
                    .map(|(_, id, distance)| SearchResult { id, distance })
                    .collect();
                let (facet_res, total_hits) = crate::facet::compute_facets(
                    &self.facet_env(),
                    facets,
                    Population::RowSet(&rs),
                )?;
                (hits, facet_res, total_hits)
            }
        };
        Ok(FacetedSearchResponse {
            hits,
            facets: facet_res,
            total_hits,
        })
    }

    pub fn query_bm25_with_facets(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
        facets: &[FacetSpec],
    ) -> Result<FacetedSearchResponse<Bm25Result>> {
        validate_facet_specs(&self.schema, facets)?;
        let hits = self.query_bm25(rank_by, filter, top_k)?;
        let rank_rs = self.rank_matching_rowset(rank_by, None)?;
        let (facets, total_hits) = crate::facet::compute_facets(
            &self.facet_env(),
            facets,
            Population::Query {
                filter,
                rank_bm: Some(&rank_rs),
            },
        )?;
        Ok(FacetedSearchResponse {
            hits,
            facets,
            total_hits,
        })
    }

    pub fn hybrid_search_with_facets(
        &self,
        query_vec: &[f32],
        rank_by: &RankBy,
        opts: HybridSearchOptions,
        filter: Option<&Filter>,
        facets: &[FacetSpec],
        scope: HybridFacetScope,
    ) -> Result<FacetedSearchResponse<HybridSearchResult>> {
        validate_facet_specs(&self.schema, facets)?;
        let (hits, vec_hits, bm25_hits) =
            self.hybrid_search_with_legs(query_vec, rank_by, opts, filter)?;
        let (facets, total_hits) = match scope {
            HybridFacetScope::FilterMatches => crate::facet::compute_facets(
                &self.facet_env(),
                facets,
                Population::Query {
                    filter,
                    rank_bm: None,
                },
            )?,
            HybridFacetScope::FusedCandidates => {
                let mut union: Vec<DocId> = Vec::with_capacity(vec_hits.len() + bm25_hits.len());
                union.extend(vec_hits.iter().map(|h| h.id));
                union.extend(bm25_hits.iter().map(|h| h.id));
                crate::facet::compute_facets(
                    &self.facet_env(),
                    facets,
                    Population::Docs(&union),
                )?
            }
            HybridFacetScope::Bm25AndFilterMatches => {
                let rank_rs = self.rank_matching_rowset(rank_by, None)?;
                crate::facet::compute_facets(
                    &self.facet_env(),
                    facets,
                    Population::Query {
                        filter,
                        rank_bm: Some(&rank_rs),
                    },
                )?
            }
        };
        Ok(FacetedSearchResponse {
            hits,
            facets,
            total_hits,
        })
    }

    pub fn verify_invariants(&self) -> Result<CollectionStats> {
        Err(TreeError::Unsupported("v3: phase 11".into()))
    }
}

fn collect_hooks<'a>(
    ann: Option<&'a AnnIndex>,
    attr_index: &'a AttributeIndex,
    fts_index: &'a FtsIndex,
) -> Vec<&'a dyn RowIndexHook> {
    let mut hooks: Vec<&'a dyn RowIndexHook> = Vec::with_capacity(3);
    if let Some(a) = ann {
        hooks.push(a);
    }
    hooks.push(attr_index);
    hooks.push(fts_index);
    hooks
}

fn vectors_bitwise_equal(
    tx: &TxContext<'_>,
    old_row: RowId,
    old_had_vector: bool,
    new_vec: Option<&[f32]>,
) -> Result<bool> {
    match (old_had_vector, new_vec) {
        (false, None) => Ok(true),
        (true, None) | (false, Some(_)) => Ok(false),
        (true, Some(nv)) => {
            let old_vec = tx.get_vector(old_row)?.ok_or_else(|| {
                TreeError::InvariantViolation(format!(
                    "row {old_row:?} has_vector=1 missing in cf_vectors"
                ))
            })?;
            Ok(old_vec.len() == nv.len()
                && old_vec
                    .iter()
                    .zip(nv.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()))
        }
    }
}

fn upsert_one_in_tx(
    segments: &SegmentStore,
    schema: &CollectionSchema,
    tx: &mut TxContext<'_>,
    id: DocId,
    prepared_vec: Option<&[f32]>,
    new_row_bytes: Vec<u8>,
    hooks: &[&dyn RowIndexHook],
) -> Result<bool> {
    let prev_row = tx.get_id_map(id)?;
    if let Some(old_row) = prev_row {
        let old_bytes = tx.get_row_bytes(old_row)?.ok_or_else(|| {
            TreeError::InvariantViolation(format!("row {old_row:?} missing in cf_rows"))
        })?;
        let old_view = RowView::parse(&old_bytes)?;
        if old_bytes == new_row_bytes
            && vectors_bitwise_equal(tx, old_row, old_view.has_vector(), prepared_vec)?
        {
            return Ok(false);
        }
        segments.tombstone(tx, old_row)?;
        for hook in hooks {
            hook.on_delete(tx, schema, old_row, &old_view, old_view.has_vector())?;
        }
    }

    let new_row = segments.alloc_row(tx)?;
    segments.write_row(tx, new_row, id, new_row_bytes, prepared_vec)?;
    let new_bytes = tx.get_row_bytes(new_row)?.expect("just written");
    let new_view = RowView::parse(&new_bytes)?;
    for hook in hooks {
        hook.on_insert(tx, schema, new_row, &new_view, prepared_vec, prev_row)?;
    }
    segments.maybe_seal(tx, hooks)?;
    Ok(true)
}

fn apply_patch_in_tx(
    segments: &SegmentStore,
    schema: &CollectionSchema,
    tx: &mut TxContext<'_>,
    id: DocId,
    prepared_vec: Option<&[f32]>,
    patch_attrs: &Attributes,
    hooks: &[&dyn RowIndexHook],
) -> Result<bool> {
    let Some(old_row) = tx.get_id_map(id)? else {
        return Err(TreeError::DocumentNotFound(id));
    };
    let old_bytes = tx.get_row_bytes(old_row)?.ok_or_else(|| {
        TreeError::InvariantViolation(format!("row {old_row:?} missing in cf_rows"))
    })?;
    let old_view = RowView::parse(&old_bytes)?;

    let effective_vec: Option<Vec<f32>> = match prepared_vec {
        Some(nv) => Some(nv.to_vec()),
        None if old_view.has_vector() => Some(tx.get_vector(old_row)?.ok_or_else(|| {
            TreeError::InvariantViolation(format!(
                "row {old_row:?} has_vector=1 missing in cf_vectors"
            ))
        })?),
        None => None,
    };

    let new_row_bytes = if patch_attrs.is_empty() {
        if vectors_bitwise_equal(tx, old_row, old_view.has_vector(), effective_vec.as_deref())? {
            return Ok(false);
        }
        // Keep existing inline attributes, just potentially update has_vector flag if needed
        let mut attrs = Attributes::new();
        for (attr_id, val) in decode_row_entries(&old_bytes)? {
            if let Some(entry) = schema.get_by_id(attr_id) {
                attrs.insert(entry.name.clone(), val);
            }
        }
        encode_row(schema, &attrs, effective_vec.is_some())?
    } else {
        let old_entries = decode_row_entries(&old_bytes)?;
        let mut merged = Attributes::new();
        for (attr_id, val) in old_entries {
            if let Some(entry) = schema.get_by_id(attr_id) {
                merged.insert(entry.name.clone(), val);
            }
        }
        for (name, val) in patch_attrs {
            if matches!(val, AttributeValue::Null) {
                merged.remove(name);
            } else {
                merged.insert(name.clone(), val.clone());
            }
        }
        let encoded = encode_row(schema, &merged, effective_vec.is_some())?;
        if encoded == old_bytes
            && vectors_bitwise_equal(tx, old_row, old_view.has_vector(), effective_vec.as_deref())?
        {
            return Ok(false);
        }
        encoded
    };

    segments.tombstone(tx, old_row)?;
    for hook in hooks {
        hook.on_delete(tx, schema, old_row, &old_view, old_view.has_vector())?;
    }

    let new_row = segments.alloc_row(tx)?;
    segments.write_row(tx, new_row, id, new_row_bytes, effective_vec.as_deref())?;
    let new_bytes = tx.get_row_bytes(new_row)?.expect("just written");
    let new_view = RowView::parse(&new_bytes)?;
    for hook in hooks {
        hook.on_insert(
            tx,
            schema,
            new_row,
            &new_view,
            effective_vec.as_deref(),
            Some(old_row),
        )?;
    }
    segments.maybe_seal(tx, hooks)?;
    Ok(true)
}
