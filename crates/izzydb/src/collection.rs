//! High-level `Collection` API coordinating the RocksDB [`Storage`],
//! [`ClusteringTree`] vector index, forward document store (`cf_docs`),
//! [`AttributeIndex`], and [`FtsIndex`].

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use roaring::RoaringBitmap;

use crate::attr_index::{AttributeIndex, ClusterCandidates, PreparedFilter};
use crate::codec::{
    decode_attribute_entries, decode_attributes, encode_attributes,
    extract_index_keys_from_entries,
};
use crate::config::{AttrId, DocId, FtsDocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::facet::{
    self, FacetEnv, FacetResults, FacetSpec, FacetedSearchResponse, HybridFacetScope, Population,
    VectorFacetScope, validate_facet_specs,
};
use crate::filter::Filter;
use crate::fts_index::{FtsIndex, has_fts_attr_changes, has_fts_patch_changes};
use crate::lire;
use crate::metric::prepare_vector;
use crate::rabitq::RabitqQuantizer;
use crate::rank::RankBy;
use crate::schema::{AttributeValue, Attributes, CollectionConfig, CollectionSchema};
use crate::search::{self, SearchOptions};
use crate::storage::{DocLoc, Storage, TxContext};
use crate::tree::{
    ClusteringTree, SplitPair, VectorUpdateResult, insert_in_tx, remove_existing_in_tx,
    remove_many_in_tx, update_vector_in_tx,
};
use crate::verify::{self, CollectionStats};
use izzydb_fts::Bm25Result;

const MUTATION_CHUNK_SIZE: usize = 512;
const PUSHDOWN_MAX_CANDIDATE_CLUSTERS: u64 = 64;

pub(crate) enum FilterPushdown {
    None,
    Superset(RoaringBitmap),
    Exact(RoaringBitmap),
}

enum AttrDiffMode<'a> {
    Full(&'a Attributes),
    Patch(&'a Attributes),
}

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

/// Stored document returned by [`Collection::get`].
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub id: DocId,
    pub embedding: Vec<f32>,
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

/// Top-level document + vector + attribute + full-text collection coordinating a shared RocksDB
/// [`Storage`], a [`ClusteringTree`] vector index, forward document attributes (`cf_docs`),
/// a cluster-aware [`AttributeIndex`], and an inverted [`FtsIndex`].
pub struct Collection {
    pub(crate) storage: Arc<Storage>,
    pub(crate) schema: CollectionSchema,
    pub(crate) vector_index: ClusteringTree,
    pub(crate) attr_index: AttributeIndex,
    pub(crate) fts_index: FtsIndex,
}

impl Collection {
    /// Open or create a [`Collection`] at `path` with the given `config`.
    pub fn open(path: impl AsRef<Path>, config: CollectionConfig) -> Result<Self> {
        config.validate()?;
        let (storage, meta) = Storage::open(path, &config.tree)?;
        let (schema, reconciled) = match storage.get_schema()? {
            Some(mut existing) => {
                let changed = existing.reconcile(&config)?;
                if changed {
                    storage.put_schema(&existing)?;
                }
                (existing, changed)
            }
            None => {
                if meta.num_vectors > 0 {
                    return Err(TreeError::InvalidSchema(
                        "existing non-empty database has no CollectionSchema".into(),
                    ));
                }
                let created = CollectionSchema::from_config(&config)?;
                storage.put_schema(&created)?;
                (created, false)
            }
        };

        // One-time atomic backfill of `default[0x06]` global attribute value counts for databases
        // created before they existed (and the marker for fresh databases).
        if !storage.has_attr_counts_marker()? {
            let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
            if meta.num_vectors > 0 {
                for (_, raw_bytes) in storage.all_docs_bytes()? {
                    let entries = decode_attribute_entries(&raw_bytes)?;
                    for key in extract_index_keys_from_entries(&schema, &entries)? {
                        *counts.entry(key).or_insert(0) += 1;
                    }
                }
            }
            let mut batch = storage.new_batch();
            for (key, count) in &counts {
                batch.put_attr_count(key, *count);
            }
            batch.put_attr_counts_marker();
            storage.write_batch(batch)?;
        }

        let storage = Arc::new(storage);
        let mut vector_index = ClusteringTree::from_storage(Arc::clone(&storage), meta, config.tree)?;
        let fts_index = FtsIndex::new();

        if reconciled && schema.has_fts() && vector_index.meta.num_vectors > 0 {
            let docs = storage.all_docs_bytes()?;
            vector_index.in_tx(|tx, _, _| {
                for (doc_id, raw_bytes) in docs {
                    if tx.get_fts_doc_id(doc_id)?.is_none() {
                        let attrs = decode_attributes(&schema, &raw_bytes)?;
                        fts_index.upsert_doc(tx, &schema, doc_id, None, &attrs)?;
                    }
                }
                Ok(())
            })?;
        }

        Ok(Self {
            storage,
            schema,
            vector_index,
            attr_index: AttributeIndex,
            fts_index,
        })
    }

    pub fn schema(&self) -> &CollectionSchema {
        &self.schema
    }

    pub fn config(&self) -> &TreeConfig {
        self.vector_index.config()
    }

    pub fn len(&self) -> u64 {
        self.vector_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vector_index.is_empty()
    }

    /// Insert or update a document's vector, forward attributes, attribute index, and FTS index atomically.
    pub fn upsert(
        &mut self,
        id: DocId,
        embedding: &[f32],
        attributes: Attributes,
    ) -> Result<()> {
        self.schema.validate_attributes(&attributes)?;
        let vec = prepare_vector(
            embedding,
            self.vector_index.config.dim,
            self.vector_index.config.metric,
        )?;
        let new_doc_bytes = encode_attributes(&self.schema, &attributes)?;

        let (tree_config, quantizer, mut tx) = self.vector_index.begin_tx_parts();
        let mut splits = Vec::new();

        if let Some(old_loc) = tx.get_doc_loc(id)? {
            let effective_vec = if vectors_bitwise_equal(&tx, id, &vec)? {
                None
            } else {
                Some(vec.as_slice())
            };
            let old_entries = match tx.get_doc_bytes(id)? {
                Some(bytes) => decode_attribute_entries(&bytes)?,
                None => Vec::new(),
            };
            let fts_changed = has_fts_attr_changes(&self.schema, &old_entries, &attributes);
            update_existing_doc_in_tx(
                &DocUpdateCtx {
                    schema: &self.schema,
                    attr_index: &self.attr_index,
                    fts_index: &self.fts_index,
                    tree_config,
                    quantizer,
                },
                &mut tx,
                id,
                old_loc,
                effective_vec,
                &old_entries,
                &attributes,
                new_doc_bytes,
                AttrDiffMode::Full(&attributes),
                fts_changed,
                &mut splits,
            )?;
        } else {
            self.fts_index
                .upsert_doc(&mut tx, &self.schema, id, None, &attributes)?;
            tx.put_doc_bytes(id, new_doc_bytes);
            let (new_loc, ins_delta, split_opt) =
                insert_in_tx(&mut tx, id, &vec, tree_config, quantizer)?;
            self.attr_index
                .apply_cluster_delta(&mut tx, &self.schema, &ins_delta)?;
            self.attr_index.insert_doc(
                &mut tx,
                &self.schema,
                &attributes,
                new_loc.leaf_id,
                new_loc.local_id,
            )?;
            if let Some(sp) = split_opt {
                splits.push(sp);
            }
        }

        self.vector_index.meta = tx.commit()?;
        self.run_lire_for_splits(splits)?;
        Ok(())
    }

    /// Apply a partial update (`DocumentPatch`) to an existing document atomically.
    ///
    /// Returns [`TreeError::DocumentNotFound`] if `id` does not exist.
    pub fn patch(&mut self, id: DocId, patch: DocumentPatch) -> Result<()> {
        self.schema.validate_attributes(&patch.attributes)?;
        let prepared_vec = match patch.embedding {
            Some(ref emb) => Some(prepare_vector(
                emb,
                self.vector_index.config.dim,
                self.vector_index.config.metric,
            )?),
            None => None,
        };

        let (tree_config, quantizer, mut tx) = self.vector_index.begin_tx_parts();
        let ctx = DocUpdateCtx {
            schema: &self.schema,
            attr_index: &self.attr_index,
            fts_index: &self.fts_index,
            tree_config,
            quantizer,
        };
        let mut splits = Vec::new();
        apply_patch_in_tx(
            &ctx,
            &mut tx,
            id,
            prepared_vec.as_deref(),
            &patch.attributes,
            &mut splits,
        )?;
        self.vector_index.meta = tx.commit()?;
        self.run_lire_for_splits(splits)?;
        Ok(())
    }

    /// Validate all patches up front and apply them atomically in a single transaction.
    ///
    /// Returns [`TreeError::DocumentNotFound`] without writing anything if any target ID is absent.
    pub fn patch_batch(&mut self, patches: &[(DocId, DocumentPatch)]) -> Result<()> {
        if patches.is_empty() {
            return Ok(());
        }
        let mut prepared_vecs: Vec<Option<Vec<f32>>> = Vec::with_capacity(patches.len());
        let mut doc_ids: Vec<DocId> = Vec::with_capacity(patches.len());
        for &(id, ref patch) in patches {
            self.schema.validate_attributes(&patch.attributes)?;
            let pv = match patch.embedding {
                Some(ref emb) => Some(prepare_vector(
                    emb,
                    self.vector_index.config.dim,
                    self.vector_index.config.metric,
                )?),
                None => None,
            };
            prepared_vecs.push(pv);
            doc_ids.push(id);
        }

        let (tree_config, quantizer, mut tx) = self.vector_index.begin_tx_parts();
        let locs = tx.multi_get_doc_locs(&doc_ids)?;
        for (&id, opt_loc) in doc_ids.iter().zip(locs) {
            if opt_loc.is_none() {
                return Err(TreeError::DocumentNotFound(id));
            }
        }

        let ctx = DocUpdateCtx {
            schema: &self.schema,
            attr_index: &self.attr_index,
            fts_index: &self.fts_index,
            tree_config,
            quantizer,
        };
        let mut splits = Vec::new();
        for (&(id, ref patch), pv) in patches.iter().zip(prepared_vecs.iter()) {
            apply_patch_in_tx(
                &ctx,
                &mut tx,
                id,
                pv.as_deref(),
                &patch.attributes,
                &mut splits,
            )?;
        }
        self.vector_index.meta = tx.commit()?;
        self.run_lire_for_splits(splits)?;
        Ok(())
    }

    /// Resolve all documents matching `filter` and apply `patch` in bounded atomic chunks,
    /// returning the number of updated documents.
    pub fn update_by_filter(&mut self, filter: &Filter, patch: &DocumentPatch) -> Result<u64> {
        self.schema.validate_attributes(&patch.attributes)?;
        let prepared_vec = match patch.embedding {
            Some(ref emb) => Some(prepare_vector(
                emb,
                self.vector_index.config.dim,
                self.vector_index.config.metric,
            )?),
            None => None,
        };
        let doc_ids = self.resolve_filter_doc_ids(filter)?;
        if doc_ids.is_empty() {
            return Ok(0);
        }
        if patch.is_empty() {
            return Ok(doc_ids.len() as u64);
        }

        let mut updated = 0u64;
        for chunk in doc_ids.chunks(MUTATION_CHUNK_SIZE) {
            let (tree_config, quantizer, mut tx) = self.vector_index.begin_tx_parts();
            let ctx = DocUpdateCtx {
                schema: &self.schema,
                attr_index: &self.attr_index,
                fts_index: &self.fts_index,
                tree_config,
                quantizer,
            };
            let mut splits = Vec::new();
            for &id in chunk {
                apply_patch_in_tx(
                    &ctx,
                    &mut tx,
                    id,
                    prepared_vec.as_deref(),
                    &patch.attributes,
                    &mut splits,
                )?;
                updated += 1;
            }
            self.vector_index.meta = tx.commit()?;
            self.run_lire_for_splits(splits)?;
        }
        Ok(updated)
    }

    /// Delete a document's vector, forward attributes, attribute index, and FTS index entries atomically.
    ///
    /// Returns `true` if `id` existed and was deleted, or `false` if `id` was absent.
    pub fn delete(&mut self, id: DocId) -> Result<bool> {
        let (tree_config, quantizer, mut tx) = self.vector_index.begin_tx_parts();
        let Some((old_loc, delta)) = remove_existing_in_tx(&mut tx, id, tree_config, quantizer)?
        else {
            return Ok(false);
        };
        if let Some(old_bytes) = tx.get_doc_bytes(id)? {
            let entries = decode_attribute_entries(&old_bytes)?;
            self.attr_index.remove_doc_entries(
                &mut tx,
                &self.schema,
                &entries,
                old_loc.leaf_id,
                old_loc.local_id,
            )?;
            self.fts_index
                .remove_doc(&mut tx, &self.schema, id, &entries)?;
        }
        tx.delete_doc(id);
        self.attr_index
            .apply_cluster_delta(&mut tx, &self.schema, &delta)?;
        self.vector_index.meta = tx.commit()?;
        Ok(true)
    }

    /// Delete a batch of documents (deduplicating and ignoring missing IDs) atomically
    /// across all three indexes, returning the number of deleted documents.
    pub fn delete_batch(&mut self, ids: &[DocId]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }
        let (tree_config, quantizer, mut tx) = self.vector_index.begin_tx_parts();
        let (removed_docs, delta) = remove_many_in_tx(&mut tx, ids, tree_config, quantizer)?;
        if removed_docs.is_empty() {
            return Ok(0);
        }

        let removed_ids: Vec<DocId> = removed_docs.iter().map(|&(did, _)| did).collect();
        let docs_bytes = tx.multi_get_doc_bytes(&removed_ids)?;
        for (&(doc_id, old_loc), opt_bytes) in removed_docs.iter().zip(docs_bytes) {
            if let Some(old_bytes) = opt_bytes {
                let entries = decode_attribute_entries(&old_bytes)?;
                self.attr_index.remove_doc_entries(
                    &mut tx,
                    &self.schema,
                    &entries,
                    old_loc.leaf_id,
                    old_loc.local_id,
                )?;
                self.fts_index
                    .remove_doc(&mut tx, &self.schema, doc_id, &entries)?;
            }
            tx.delete_doc(doc_id);
        }

        self.attr_index
            .apply_cluster_delta(&mut tx, &self.schema, &delta)?;
        self.vector_index.meta = tx.commit()?;
        Ok(removed_docs.len() as u64)
    }

    /// Resolve all documents matching `filter` and delete them in bounded atomic chunks,
    /// returning the number of deleted documents.
    pub fn delete_by_filter(&mut self, filter: &Filter) -> Result<u64> {
        let doc_ids = self.resolve_filter_doc_ids(filter)?;
        if doc_ids.is_empty() {
            return Ok(0);
        }
        let mut total_deleted = 0u64;
        for chunk in doc_ids.chunks(MUTATION_CHUNK_SIZE) {
            total_deleted += self.delete_batch(chunk)?;
        }
        Ok(total_deleted)
    }

    /// Recompute exact `TermStats` upper bounds (`max_tf`, `min_dl_over_tf`) across all FTS terms.
    pub fn compact_fts_stats(&mut self) -> Result<()> {
        if !self.schema.has_fts() {
            return Ok(());
        }
        let (_, _, mut tx) = self.vector_index.begin_tx_parts();
        let updated = self.fts_index.compact_term_stats(&self.storage, &mut tx)?;
        if updated > 0 {
            self.vector_index.meta = tx.commit()?;
        }
        Ok(())
    }

    /// Resolve all `DocId`s matching `filter` (supporting attribute, FTS, and combined predicates).
    pub fn resolve_filter_doc_ids(&self, filter: &Filter) -> Result<Vec<DocId>> {
        filter.validate(&self.schema)?;
        if self.is_empty() {
            return Ok(Vec::new());
        }

        if filter.is_fts_only()
            && !filter_has_not(filter)
            && let Some(fts_bm) = self.eval_positive_fts_bitmap(filter)?
        {
            if fts_bm.is_empty() {
                return Ok(Vec::new());
            }
            let fts_ids: Vec<FtsDocId> = fts_bm.iter().collect();
            let mut doc_ids: Vec<DocId> = self
                .storage
                .multi_get_doc_ids_from_fts(&fts_ids)?
                .into_iter()
                .flatten()
                .collect();
            doc_ids.sort_unstable();
            doc_ids.dedup();
            return Ok(doc_ids);
        }

        let prepared = self.attr_index.prepare_filter_with_fts(
            &self.storage,
            &self.schema,
            filter,
            Some(&self.fts_index),
        )?;
        let leaf_ids: Vec<NodeId> = match &prepared.cluster_candidates {
            ClusterCandidates::Subset(bm) => {
                if bm.is_empty() {
                    return Ok(Vec::new());
                }
                bm.iter().collect()
            }
            ClusterCandidates::All => {
                if self.vector_index.meta.root_level == 0 {
                    vec![self.vector_index.meta.root_id]
                } else {
                    let mut ids: Vec<NodeId> =
                        self.vector_index.leaf_parent.keys().copied().collect();
                    ids.sort_unstable();
                    ids
                }
            }
        };

        let mut doc_ids = Vec::new();
        search::collect_matching_doc_ids_in_leaves(
            &self.storage,
            &self.attr_index,
            &prepared,
            &leaf_ids,
            &self.vector_index.config,
            &self.vector_index.quantizer,
            &mut doc_ids,
        )?;

        doc_ids.sort_unstable();
        doc_ids.dedup();
        Ok(doc_ids)
    }

    fn eval_positive_fts_bitmap(&self, filter: &Filter) -> Result<Option<RoaringBitmap>> {
        if let Some((kind, attr, query, last_as_prefix)) = filter.as_fts_predicate() {
            return Ok(Some(self.fts_index.eval_fts_predicate(
                &self.storage,
                &self.schema,
                kind,
                attr,
                query,
                last_as_prefix,
            )?));
        }

        match filter {
            Filter::And(children) => {
                let mut acc: Option<RoaringBitmap> = None;
                for child in children {
                    if let Some(sub) = self.eval_positive_fts_bitmap(child)? {
                        if sub.is_empty() {
                            return Ok(Some(RoaringBitmap::new()));
                        }
                        acc = Some(match acc {
                            Some(mut cur) => {
                                cur &= &sub;
                                if cur.is_empty() {
                                    return Ok(Some(RoaringBitmap::new()));
                                }
                                cur
                            }
                            None => sub,
                        });
                    }
                }
                Ok(acc)
            }
            Filter::Or(children) => {
                let mut acc = RoaringBitmap::new();
                for child in children {
                    match self.eval_positive_fts_bitmap(child)? {
                        Some(sub) => acc |= &sub,
                        None => return Ok(None),
                    }
                }
                Ok(Some(acc))
            }
            _ => Ok(None),
        }
    }

    fn run_lire_for_splits(&mut self, splits: Vec<SplitPair>) -> Result<()> {
        if splits.is_empty()
            || !self.vector_index.config.enable_lire
            || self.vector_index.meta.root_level == 0
        {
            return Ok(());
        }
        for split_pair in splits {
            if self.vector_index.meta.root_level == 0 {
                break;
            }
            lire::run_lire_with_hook(&mut self.vector_index, split_pair, |tx, delta| {
                self.attr_index
                    .apply_cluster_delta(tx, &self.schema, delta)
            })?;
        }
        Ok(())
    }

    /// Retrieve a document's stored vector and attributes by `id`.
    pub fn get(&self, id: DocId) -> Result<Option<Document>> {
        let Some(embedding) = self.storage.get_vector(id)? else {
            return Ok(None);
        };
        let doc_bytes = self.storage.get_doc_bytes(id)?.ok_or_else(|| {
            TreeError::InvariantViolation(format!("doc {id} present in cf_vectors but missing in cf_docs"))
        })?;
        let attributes = decode_attributes(&self.schema, &doc_bytes)?;
        Ok(Some(Document {
            id,
            embedding,
            attributes,
        }))
    }

    /// Search for the `k` nearest neighbors matching an optional [`Filter`].
    pub fn search(
        &self,
        query: &[f32],
        opts: SearchOptions,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        let hits = match filter {
            None => search::search(
                &self.storage,
                &self.vector_index.internal_cache,
                &self.vector_index.meta,
                &self.vector_index.quantizer,
                query,
                opts,
                &self.vector_index.config,
            )?,
            Some(f) => search::search_filtered(
                &self.storage,
                &self.vector_index.internal_cache,
                &self.vector_index.leaf_parent,
                &self.vector_index.meta,
                &self.vector_index.quantizer,
                &self.schema,
                &self.attr_index,
                Some(&self.fts_index),
                query,
                opts,
                f,
                &self.vector_index.config,
            )?,
        };
        Ok(hits
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    /// Execute a ranking query with an optional [`Filter`].
    ///
    /// `rank_by` may be plain BM25, or any composition of BM25, filter boosts, numeric attribute
    /// scores (`Attribute` / `Saturate` / `Decay`), constants, `Sum`, `Max`, and `Product`.
    /// Results are the exact top-`top_k` by `score DESC, DocId ASC`; documents scoring `<= 0`
    /// are excluded.
    pub fn query_bm25(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<Bm25Result>> {
        self.rank_query(rank_by, filter, top_k)
    }

    /// Alias for [`Collection::query_bm25`].
    pub fn query(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<Bm25Result>> {
        self.rank_query(rank_by, filter, top_k)
    }

    /// Run vector search and BM25 full-text search under `filter`, fusing rankings via RRF.
    pub fn hybrid_search(
        &self,
        query_vec: &[f32],
        rank_by: &RankBy,
        opts: HybridSearchOptions,
        filter: Option<&Filter>,
    ) -> Result<Vec<HybridSearchResult>> {
        Ok(self
            .hybrid_search_inner(query_vec, rank_by, opts, filter)?
            .0)
    }

    /// Fused RRF results plus every candidate `DocId` that entered fusion (sorted, deduplicated).
    fn hybrid_search_inner(
        &self,
        query_vec: &[f32],
        rank_by: &RankBy,
        opts: HybridSearchOptions,
        filter: Option<&Filter>,
    ) -> Result<(Vec<HybridSearchResult>, Vec<DocId>)> {
        opts.validate()?;
        rank_by.validate(&self.schema)?;
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }

        let vec_hits = self.search(query_vec, opts.vector_search, filter)?;
        let bm25_k = opts
            .bm25_candidates
            .unwrap_or_else(|| opts.vector_search.k.max(opts.k.saturating_mul(2)));
        let bm25_hits = self.query_bm25(rank_by, filter, bm25_k)?;

        let mut by_doc: HashMap<DocId, HybridSearchResult> =
            HashMap::with_capacity(vec_hits.len() + bm25_hits.len());

        for (idx, hit) in vec_hits.iter().enumerate() {
            let rank = idx + 1;
            let contrib = opts.vector_weight / (opts.rrf_k + rank as f32);
            let entry = by_doc.entry(hit.id).or_insert(HybridSearchResult {
                id: hit.id,
                score: 0.0,
                vector_rank: None,
                bm25_rank: None,
            });
            entry.score += contrib;
            entry.vector_rank = Some(rank);
        }

        for (idx, hit) in bm25_hits.iter().enumerate() {
            let rank = idx + 1;
            let contrib = opts.bm25_weight / (opts.rrf_k + rank as f32);
            let entry = by_doc.entry(hit.id).or_insert(HybridSearchResult {
                id: hit.id,
                score: 0.0,
                vector_rank: None,
                bm25_rank: None,
            });
            entry.score += contrib;
            entry.bm25_rank = Some(rank);
        }

        let mut candidates: Vec<DocId> = by_doc.keys().copied().collect();
        candidates.sort_unstable();

        if opts.k == 0 {
            return Ok((Vec::new(), candidates));
        }
        let mut results: Vec<HybridSearchResult> = by_doc.into_values().collect();
        results.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        results.truncate(opts.k);
        Ok((results, candidates))
    }

    // ------------------------------------------------------------------------------------
    // Faceting
    // ------------------------------------------------------------------------------------

    pub(crate) fn facet_env(&self) -> FacetEnv<'_> {
        FacetEnv {
            storage: &self.storage,
            schema: &self.schema,
            attr_index: &self.attr_index,
            fts_index: &self.fts_index,
            config: &self.vector_index.config,
            quantizer: &self.vector_index.quantizer,
            meta: &self.vector_index.meta,
            leaf_parent: &self.vector_index.leaf_parent,
        }
    }

    fn validate_facet_request(&self, filter: Option<&Filter>, facets: &[FacetSpec]) -> Result<()> {
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        validate_facet_specs(&self.schema, facets)
    }

    /// Exact facet aggregations over all documents matching `filter` (or the whole collection).
    ///
    /// Every faceted attribute must be `filterable`. Each facet's [`FacetFilterMode`]
    /// (`ExcludeSelf`, `ExcludeAttributes`, `Custom`) enables disjunctive / multi-select faceting.
    ///
    /// [`FacetFilterMode`]: crate::FacetFilterMode
    pub fn facet(&self, filter: Option<&Filter>, facets: &[FacetSpec]) -> Result<FacetResults> {
        self.validate_facet_request(filter, facets)?;
        let (results, _) = facet::compute_facets(
            &self.facet_env(),
            facets,
            Population::Query {
                filter,
                rank_bm: None,
                prepared: None,
            },
        )?;
        Ok(results)
    }

    /// Vector ANN search plus facets. `scope` selects the facet population:
    /// all `filter` matches (default), only the returned top-`k`, or the top-`n` ANN candidates.
    ///
    /// For `TopK` / `Candidates`, the population is a fixed document set, so per-facet
    /// [`FacetFilterMode`](crate::FacetFilterMode)s are ignored.
    pub fn search_with_facets(
        &self,
        query: &[f32],
        opts: SearchOptions,
        filter: Option<&Filter>,
        facets: &[FacetSpec],
        scope: VectorFacetScope,
    ) -> Result<FacetedSearchResponse<SearchResult>> {
        self.validate_facet_request(filter, facets)?;
        let env = self.facet_env();
        match scope {
            VectorFacetScope::FilterMatches => {
                let prepared = match filter {
                    Some(f) => Some(self.attr_index.prepare_filter_with_fts(
                        &self.storage,
                        &self.schema,
                        f,
                        Some(&self.fts_index),
                    )?),
                    None => None,
                };
                let hits = self.search_hits(query, opts, prepared.as_ref())?;
                let (facets, total_hits) = facet::compute_facets(
                    &env,
                    facets,
                    Population::Query {
                        filter,
                        rank_bm: None,
                        prepared: prepared.as_ref(),
                    },
                )?;
                Ok(FacetedSearchResponse {
                    hits,
                    facets,
                    total_hits,
                })
            }
            VectorFacetScope::TopK => {
                let hits = self.search(query, opts, filter)?;
                let ids: Vec<DocId> = hits.iter().map(|h| h.id).collect();
                let (facets, total_hits) =
                    facet::compute_facets(&env, facets, Population::Docs(&ids))?;
                Ok(FacetedSearchResponse {
                    hits,
                    facets,
                    total_hits,
                })
            }
            VectorFacetScope::Candidates(n) => {
                let mut wide = opts;
                wide.k = n.max(opts.k);
                let mut cands = self.search(query, wide, filter)?;
                let ids: Vec<DocId> = cands.iter().map(|h| h.id).collect();
                cands.truncate(opts.k);
                let (facets, total_hits) =
                    facet::compute_facets(&env, facets, Population::Docs(&ids))?;
                Ok(FacetedSearchResponse {
                    hits: cands,
                    facets,
                    total_hits,
                })
            }
        }
    }

    /// BM25 ranking plus facets over every document with BM25 score `> 0` for `rank_by` that also
    /// matches `filter`. Top-`k` pruning is unaffected: the exact match set is computed from
    /// posting unions without scoring.
    pub fn query_bm25_with_facets(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
        facets: &[FacetSpec],
    ) -> Result<FacetedSearchResponse<Bm25Result>> {
        self.validate_facet_request(filter, facets)?;
        rank_by.validate(&self.schema)?;
        let hits = self.query_bm25(rank_by, filter, top_k)?;
        let (facets, total_hits) = self.facet_rank_population(rank_by, filter, facets)?;
        Ok(FacetedSearchResponse {
            hits,
            facets,
            total_hits,
        })
    }

    /// Hybrid RRF search plus facets. `scope` selects the facet population: all `filter` matches
    /// (default), the fused vector + BM25 candidates, or all BM25 matches that also match `filter`.
    pub fn hybrid_search_with_facets(
        &self,
        query_vec: &[f32],
        rank_by: &RankBy,
        opts: HybridSearchOptions,
        filter: Option<&Filter>,
        facets: &[FacetSpec],
        scope: HybridFacetScope,
    ) -> Result<FacetedSearchResponse<HybridSearchResult>> {
        self.validate_facet_request(filter, facets)?;
        let (hits, candidates) = self.hybrid_search_inner(query_vec, rank_by, opts, filter)?;
        let env = self.facet_env();
        let (facets, total_hits) = match scope {
            HybridFacetScope::FilterMatches => facet::compute_facets(
                &env,
                facets,
                Population::Query {
                    filter,
                    rank_bm: None,
                    prepared: None,
                },
            )?,
            HybridFacetScope::FusedCandidates => {
                facet::compute_facets(&env, facets, Population::Docs(&candidates))?
            }
            HybridFacetScope::Bm25AndFilterMatches => {
                self.facet_rank_population(rank_by, filter, facets)?
            }
        };
        Ok(FacetedSearchResponse {
            hits,
            facets,
            total_hits,
        })
    }

    /// Facets over every document with `rank_by` score `> 0` that also matches `filter`.
    ///
    /// Pure lexical expressions keep the posting-union bitmap population (per-facet filter modes
    /// apply). Expressions with filter boosts / attribute scores / products are evaluated
    /// exactly into a fixed document set, so per-facet filter modes are ignored for them.
    fn facet_rank_population(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        facets: &[FacetSpec],
    ) -> Result<(FacetResults, u64)> {
        let norm = rank_by.normalize();
        if norm.is_pure_lexical() {
            let rank_bm = self.rank_match_bitmap(&norm)?;
            return facet::compute_facets(
                &self.facet_env(),
                facets,
                Population::Query {
                    filter,
                    rank_bm: Some(&rank_bm),
                    prepared: None,
                },
            );
        }
        let ids = self.rank_matching_doc_ids(rank_by, filter)?;
        facet::compute_facets(&self.facet_env(), facets, Population::Docs(&ids))
    }

    /// Exact `FtsDocId`s with score `> 0` for a *pure lexical* expression.
    pub(crate) fn rank_match_bitmap(&self, rank_by: &RankBy) -> Result<RoaringBitmap> {
        if self.is_empty() {
            return Ok(RoaringBitmap::new());
        }
        self.fts_index
            .eval_rank_by_match_bitmap(&self.storage, &self.schema, &rank_by.normalize())
    }

    /// ANN search over an optional pre-prepared (already validated) filter plan.
    fn search_hits(
        &self,
        query: &[f32],
        opts: SearchOptions,
        prepared: Option<&PreparedFilter>,
    ) -> Result<Vec<SearchResult>> {
        let hits = match prepared {
            None => search::search(
                &self.storage,
                &self.vector_index.internal_cache,
                &self.vector_index.meta,
                &self.vector_index.quantizer,
                query,
                opts,
                &self.vector_index.config,
            )?,
            Some(p) => search::search_filtered_prepared(
                &self.storage,
                &self.vector_index.internal_cache,
                &self.vector_index.leaf_parent,
                &self.vector_index.meta,
                &self.vector_index.quantizer,
                &self.attr_index,
                query,
                opts,
                p,
                &self.vector_index.config,
            )?,
        };
        Ok(hits
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    pub(crate) fn try_pushdown_filter(&self, filter: &Filter) -> Result<FilterPushdown> {
        if let Some((kind, attr, query, last_as_prefix)) = filter.as_fts_predicate() {
            let bm = self.fts_index.eval_fts_predicate(
                &self.storage,
                &self.schema,
                kind,
                attr,
                query,
                last_as_prefix,
            )?;
            return Ok(FilterPushdown::Exact(bm));
        }

        match filter {
            Filter::And(children) => {
                if children.is_empty() {
                    return Ok(FilterPushdown::None);
                }
                let mut combined: Option<RoaringBitmap> = None;
                let mut all_exact = true;
                for child in children {
                    let sub_bm = match self.try_pushdown_filter(child)? {
                        FilterPushdown::None => {
                            all_exact = false;
                            None
                        }
                        FilterPushdown::Superset(bm) => {
                            all_exact = false;
                            Some(bm)
                        }
                        FilterPushdown::Exact(bm) => Some(bm),
                    };
                    if let Some(bm) = sub_bm {
                        if bm.is_empty() {
                            return Ok(FilterPushdown::Exact(RoaringBitmap::new()));
                        }
                        combined = Some(match combined {
                            Some(mut cur) => {
                                cur &= &bm;
                                if cur.is_empty() {
                                    return Ok(FilterPushdown::Exact(RoaringBitmap::new()));
                                }
                                cur
                            }
                            None => bm,
                        });
                    }
                }
                match (combined, all_exact) {
                    (None, _) => Ok(FilterPushdown::None),
                    (Some(bm), true) => Ok(FilterPushdown::Exact(bm)),
                    (Some(bm), false) => Ok(FilterPushdown::Superset(bm)),
                }
            }
            Filter::Or(children) => {
                if children.is_empty() {
                    return Ok(FilterPushdown::Exact(RoaringBitmap::new()));
                }
                let mut union = RoaringBitmap::new();
                for child in children {
                    match self.try_pushdown_filter(child)? {
                        FilterPushdown::Exact(bm) => union |= &bm,
                        _ => return Ok(FilterPushdown::None),
                    }
                }
                Ok(FilterPushdown::Exact(union))
            }
            _ if !filter.has_fts() => {
                let prepared = self
                    .attr_index
                    .prepare_filter(&self.storage, &self.schema, filter)?;
                match prepared.cluster_candidates {
                    ClusterCandidates::Subset(ref bm) if bm.is_empty() => {
                        Ok(FilterPushdown::Exact(RoaringBitmap::new()))
                    }
                    ClusterCandidates::Subset(ref bm)
                        if bm.len() <= PUSHDOWN_MAX_CANDIDATE_CLUSTERS =>
                    {
                        let leaf_ids: Vec<_> = bm.iter().collect();
                        let mut doc_ids = Vec::new();
                        search::collect_matching_doc_ids_in_leaves(
                            &self.storage,
                            &self.attr_index,
                            &prepared,
                            &leaf_ids,
                            &self.vector_index.config,
                            &self.vector_index.quantizer,
                            &mut doc_ids,
                        )?;
                        if doc_ids.is_empty() {
                            return Ok(FilterPushdown::Exact(RoaringBitmap::new()));
                        }
                        doc_ids.sort_unstable();
                        doc_ids.dedup();
                        let fts_ids = self.storage.multi_get_fts_doc_ids(&doc_ids)?;
                        let mut fts_bm = RoaringBitmap::new();
                        for fid in fts_ids.into_iter().flatten() {
                            fts_bm.insert(fid);
                        }
                        Ok(FilterPushdown::Exact(fts_bm))
                    }
                    _ => Ok(FilterPushdown::None),
                }
            }
            _ => Ok(FilterPushdown::None),
        }
    }

    /// Exhaustively verify all structural, document, attribute index, and FTS index invariants.
    pub fn verify_invariants(&self) -> Result<CollectionStats> {
        verify::verify_collection_invariants(
            &self.storage,
            &self.vector_index.internal_cache,
            &self.vector_index.leaf_parent,
            &self.vector_index.meta,
            &self.vector_index.config,
            &self.schema,
        )
    }
}

struct DocUpdateCtx<'a> {
    schema: &'a CollectionSchema,
    attr_index: &'a AttributeIndex,
    fts_index: &'a FtsIndex,
    tree_config: &'a TreeConfig,
    quantizer: &'a RabitqQuantizer,
}

#[inline]
fn vectors_bitwise_equal(tx: &TxContext<'_>, id: DocId, vec: &[f32]) -> Result<bool> {
    let old_vec = tx.get_vectors_flat(&[id])?;
    Ok(old_vec.len() == vec.len()
        && old_vec
            .iter()
            .zip(vec.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()))
}

#[allow(clippy::too_many_arguments)]
fn update_existing_doc_in_tx(
    ctx: &DocUpdateCtx<'_>,
    tx: &mut TxContext<'_>,
    id: DocId,
    old_loc: DocLoc,
    new_vec: Option<&[f32]>,
    old_entries: &[(AttrId, AttributeValue)],
    new_attrs: &Attributes,
    new_doc_bytes: Vec<u8>,
    diff_mode: AttrDiffMode<'_>,
    fts_changed: bool,
    splits: &mut Vec<SplitPair>,
) -> Result<()> {
    let apply_in_place_attr_diff = |tx: &mut TxContext<'_>, leaf_id: NodeId, local_id: u32| {
        match diff_mode {
            AttrDiffMode::Full(attrs) => {
                ctx.attr_index
                    .diff_doc_in_place(tx, ctx.schema, old_entries, attrs, leaf_id, local_id)
            }
            AttrDiffMode::Patch(patch_attrs) => ctx.attr_index.diff_patched_in_place(
                tx,
                ctx.schema,
                old_entries,
                patch_attrs,
                leaf_id,
                local_id,
            ),
        }
    };

    let Some(vec) = new_vec else {
        if fts_changed {
            ctx.fts_index
                .upsert_doc(tx, ctx.schema, id, Some(old_entries), new_attrs)?;
        }
        apply_in_place_attr_diff(tx, old_loc.leaf_id, old_loc.local_id)?;
        tx.put_doc_bytes(id, new_doc_bytes);
        return Ok(());
    };

    match update_vector_in_tx(tx, id, old_loc, vec, ctx.tree_config, ctx.quantizer)? {
        VectorUpdateResult::SameLeaf { delta } => {
            if fts_changed {
                ctx.fts_index
                    .upsert_doc(tx, ctx.schema, id, Some(old_entries), new_attrs)?;
            }
            apply_in_place_attr_diff(tx, old_loc.leaf_id, old_loc.local_id)?;
            tx.put_doc_bytes(id, new_doc_bytes);
            ctx.attr_index.apply_cluster_delta(tx, ctx.schema, &delta)?;
        }
        VectorUpdateResult::Moved {
            old_loc,
            new_loc,
            rem_delta,
            ins_delta,
            split,
        } => {
            ctx.attr_index.remove_doc_entries(
                tx,
                ctx.schema,
                old_entries,
                old_loc.leaf_id,
                old_loc.local_id,
            )?;
            ctx.attr_index.apply_cluster_delta(tx, ctx.schema, &rem_delta)?;
            if fts_changed {
                ctx.fts_index
                    .upsert_doc(tx, ctx.schema, id, Some(old_entries), new_attrs)?;
            }
            tx.put_doc_bytes(id, new_doc_bytes);
            ctx.attr_index.apply_cluster_delta(tx, ctx.schema, &ins_delta)?;
            ctx.attr_index.insert_doc(
                tx,
                ctx.schema,
                new_attrs,
                new_loc.leaf_id,
                new_loc.local_id,
            )?;
            if let Some(sp) = split {
                splits.push(sp);
            }
        }
    }
    Ok(())
}

fn apply_patch_in_tx(
    ctx: &DocUpdateCtx<'_>,
    tx: &mut TxContext<'_>,
    id: DocId,
    prepared_vec: Option<&[f32]>,
    patch_attrs: &Attributes,
    splits: &mut Vec<SplitPair>,
) -> Result<()> {
    let Some(old_loc) = tx.get_doc_loc(id)? else {
        return Err(TreeError::DocumentNotFound(id));
    };

    let effective_vec = match prepared_vec {
        Some(new_vec) if !vectors_bitwise_equal(tx, id, new_vec)? => Some(new_vec),
        _ => None,
    };

    if patch_attrs.is_empty() {
        let Some(new_vec) = effective_vec else {
            return Ok(());
        };
        match update_vector_in_tx(tx, id, old_loc, new_vec, ctx.tree_config, ctx.quantizer)? {
            VectorUpdateResult::SameLeaf { delta } => {
                ctx.attr_index.apply_cluster_delta(tx, ctx.schema, &delta)?;
            }
            VectorUpdateResult::Moved {
                old_loc,
                new_loc,
                rem_delta,
                ins_delta,
                split,
            } => {
                if let Some(old_bytes) = tx.get_doc_bytes(id)? {
                    let entries = decode_attribute_entries(&old_bytes)?;
                    ctx.attr_index.remove_doc_entries(
                        tx,
                        ctx.schema,
                        &entries,
                        old_loc.leaf_id,
                        old_loc.local_id,
                    )?;
                    ctx.attr_index.apply_cluster_delta(tx, ctx.schema, &rem_delta)?;
                    ctx.attr_index.apply_cluster_delta(tx, ctx.schema, &ins_delta)?;
                    ctx.attr_index.insert_doc_entries(
                        tx,
                        ctx.schema,
                        &entries,
                        new_loc.leaf_id,
                        new_loc.local_id,
                    )?;
                } else {
                    ctx.attr_index.apply_cluster_delta(tx, ctx.schema, &rem_delta)?;
                    ctx.attr_index.apply_cluster_delta(tx, ctx.schema, &ins_delta)?;
                }
                if let Some(sp) = split {
                    splits.push(sp);
                }
            }
        }
        return Ok(());
    }

    let old_doc_bytes = tx.get_doc_bytes(id)?.ok_or_else(|| {
        TreeError::InvariantViolation(format!(
            "doc {id} present in cf_doc_loc but missing in cf_docs"
        ))
    })?;
    let old_entries = decode_attribute_entries(&old_doc_bytes)?;

    let mut merged_attrs = Attributes::new();
    for &(attr_id, ref val) in &old_entries {
        if let Some(entry) = ctx.schema.get_by_id(attr_id) {
            merged_attrs.insert(entry.name.clone(), val.clone());
        }
    }
    for (name, val) in patch_attrs {
        if matches!(val, AttributeValue::Null) {
            merged_attrs.remove(name);
        } else {
            merged_attrs.insert(name.clone(), val.clone());
        }
    }
    let new_doc_bytes = encode_attributes(ctx.schema, &merged_attrs)?;
    let fts_changed = has_fts_patch_changes(ctx.schema, &old_entries, patch_attrs);

    update_existing_doc_in_tx(
        ctx,
        tx,
        id,
        old_loc,
        effective_vec,
        &old_entries,
        &merged_attrs,
        new_doc_bytes,
        AttrDiffMode::Patch(patch_attrs),
        fts_changed,
        splits,
    )
}

fn filter_has_not(filter: &Filter) -> bool {
    match filter {
        Filter::And(children) | Filter::Or(children) => children.iter().any(filter_has_not),
        Filter::Not(_) => true,
        _ => false,
    }
}
