use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use roaring::RoaringBitmap;

use crate::attr_index::{AttributeIndex, ClusterCandidates};
use crate::codec::{DocView, decode_attribute_entries, decode_attributes, encode_attributes};
use crate::config::{DocId, FtsDocId, NodeId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::filter::Filter;
use crate::fts_index::{FtsIndex, has_fts_attr_changes, has_fts_patch_changes};
use crate::lire;
use crate::metric::prepare_vector;
use crate::node::LeafNodeView;
use crate::rabitq::RabitqQuantizer;
use crate::rank::RankBy;
use crate::schema::{AttributeValue, Attributes, CollectionConfig, CollectionSchema};
use crate::search::{self, SearchOptions};
use crate::storage::{Storage, TxContext};
use crate::tree::{
    ClusteringTree, SplitPair, VectorUpdateResult, insert_in_tx, remove_existing_in_tx,
    remove_many_in_tx, update_vector_in_tx,
};
use crate::verify::{self, CollectionStats};
use izzydb_fts::Bm25Result;

const MUTATION_CHUNK_SIZE: usize = 512;
const RESOLVE_LEAF_CHUNK_SIZE: usize = 64;

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
    storage: Arc<Storage>,
    schema: CollectionSchema,
    vector_index: ClusteringTree,
    attr_index: AttributeIndex,
    fts_index: FtsIndex,
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

        let storage = Arc::new(storage);
        let mut vector_index = ClusteringTree::from_storage(Arc::clone(&storage), meta, config.tree)?;
        let fts_index = FtsIndex::new();

        if reconciled && schema.has_fts() && vector_index.meta.num_vectors > 0 {
            let docs = storage.all_docs_bytes()?;
            let mut tx = TxContext::new(
                &vector_index.storage,
                &mut vector_index.internal_cache,
                &mut vector_index.leaf_parent,
                vector_index.meta.clone(),
            );
            for (doc_id, raw_bytes) in docs {
                if tx.get_fts_doc_id(doc_id)?.is_none() {
                    let attrs = decode_attributes(&schema, &raw_bytes)?;
                    fts_index.upsert_doc(&mut tx, &schema, doc_id, None, &attrs)?;
                }
            }
            vector_index.meta = tx.commit()?;
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

        let mut tx = TxContext::new(
            &self.vector_index.storage,
            &mut self.vector_index.internal_cache,
            &mut self.vector_index.leaf_parent,
            self.vector_index.meta.clone(),
        );
        let mut splits = Vec::new();

        if let Some(old_loc) = tx.get_doc_loc(id)? {
            let old_vec = tx.get_vectors_flat(&[id])?;
            let vec_unchanged = old_vec.len() == vec.len()
                && old_vec
                    .iter()
                    .zip(vec.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            let old_entries = match tx.get_doc_bytes(id)? {
                Some(bytes) => decode_attribute_entries(&bytes)?,
                None => Vec::new(),
            };
            let fts_changed = has_fts_attr_changes(&self.schema, &old_entries, &attributes);

            if vec_unchanged {
                if fts_changed {
                    self.fts_index.upsert_doc(
                        &mut tx,
                        &self.schema,
                        id,
                        Some(&old_entries),
                        &attributes,
                    )?;
                }
                self.attr_index.diff_doc_in_place(
                    &mut tx,
                    &self.schema,
                    &old_entries,
                    &attributes,
                    old_loc.leaf_id,
                    old_loc.local_id,
                )?;
                tx.put_doc_bytes(id, new_doc_bytes);
            } else {
                match update_vector_in_tx(
                    &mut tx,
                    id,
                    old_loc,
                    &vec,
                    &self.vector_index.config,
                    &self.vector_index.quantizer,
                )? {
                    VectorUpdateResult::SameLeaf { delta, .. } => {
                        if fts_changed {
                            self.fts_index.upsert_doc(
                                &mut tx,
                                &self.schema,
                                id,
                                Some(&old_entries),
                                &attributes,
                            )?;
                        }
                        self.attr_index.diff_doc_in_place(
                            &mut tx,
                            &self.schema,
                            &old_entries,
                            &attributes,
                            old_loc.leaf_id,
                            old_loc.local_id,
                        )?;
                        tx.put_doc_bytes(id, new_doc_bytes);
                        self.attr_index
                            .apply_cluster_delta(&mut tx, &self.schema, &delta)?;
                    }
                    VectorUpdateResult::Moved {
                        old_loc,
                        new_loc,
                        rem_delta,
                        ins_delta,
                        split,
                    } => {
                        self.attr_index.remove_doc_entries(
                            &mut tx,
                            &self.schema,
                            &old_entries,
                            old_loc.leaf_id,
                            old_loc.local_id,
                        )?;
                        self.attr_index
                            .apply_cluster_delta(&mut tx, &self.schema, &rem_delta)?;
                        if fts_changed {
                            self.fts_index.upsert_doc(
                                &mut tx,
                                &self.schema,
                                id,
                                Some(&old_entries),
                                &attributes,
                            )?;
                        }
                        tx.put_doc_bytes(id, new_doc_bytes);
                        self.attr_index
                            .apply_cluster_delta(&mut tx, &self.schema, &ins_delta)?;
                        self.attr_index.insert_doc(
                            &mut tx,
                            &self.schema,
                            &attributes,
                            new_loc.leaf_id,
                            new_loc.local_id,
                        )?;
                        if let Some(sp) = split {
                            splits.push(sp);
                        }
                    }
                }
            }
        } else {
            self.fts_index
                .upsert_doc(&mut tx, &self.schema, id, None, &attributes)?;
            tx.put_doc_bytes(id, new_doc_bytes);
            let (new_loc, ins_delta, split_opt) = insert_in_tx(
                &mut tx,
                id,
                &vec,
                &self.vector_index.config,
                &self.vector_index.quantizer,
            )?;
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

        let mut tx = TxContext::new(
            &self.vector_index.storage,
            &mut self.vector_index.internal_cache,
            &mut self.vector_index.leaf_parent,
            self.vector_index.meta.clone(),
        );
        let mut splits = Vec::new();
        apply_patch_in_tx(
            &self.schema,
            &self.attr_index,
            &self.fts_index,
            &self.vector_index.config,
            &self.vector_index.quantizer,
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

        let mut tx = TxContext::new(
            &self.vector_index.storage,
            &mut self.vector_index.internal_cache,
            &mut self.vector_index.leaf_parent,
            self.vector_index.meta.clone(),
        );
        let locs = tx.multi_get_doc_locs(&doc_ids)?;
        for (&id, opt_loc) in doc_ids.iter().zip(locs) {
            if opt_loc.is_none() {
                return Err(TreeError::DocumentNotFound(id));
            }
        }

        let mut splits = Vec::new();
        for (&(id, ref patch), pv) in patches.iter().zip(prepared_vecs.iter()) {
            apply_patch_in_tx(
                &self.schema,
                &self.attr_index,
                &self.fts_index,
                &self.vector_index.config,
                &self.vector_index.quantizer,
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
            let mut tx = TxContext::new(
                &self.vector_index.storage,
                &mut self.vector_index.internal_cache,
                &mut self.vector_index.leaf_parent,
                self.vector_index.meta.clone(),
            );
            let mut splits = Vec::new();
            for &id in chunk {
                apply_patch_in_tx(
                    &self.schema,
                    &self.attr_index,
                    &self.fts_index,
                    &self.vector_index.config,
                    &self.vector_index.quantizer,
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
        let mut tx = TxContext::new(
            &self.vector_index.storage,
            &mut self.vector_index.internal_cache,
            &mut self.vector_index.leaf_parent,
            self.vector_index.meta.clone(),
        );
        let Some((old_loc, delta)) = remove_existing_in_tx(
            &mut tx,
            id,
            &self.vector_index.config,
            &self.vector_index.quantizer,
        )?
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
        let mut tx = TxContext::new(
            &self.vector_index.storage,
            &mut self.vector_index.internal_cache,
            &mut self.vector_index.leaf_parent,
            self.vector_index.meta.clone(),
        );
        let (removed_docs, delta) = remove_many_in_tx(
            &mut tx,
            ids,
            &self.vector_index.config,
            &self.vector_index.quantizer,
        )?;
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
        let mut tx = TxContext::new(
            &self.vector_index.storage,
            &mut self.vector_index.internal_cache,
            &mut self.vector_index.leaf_parent,
            self.vector_index.meta.clone(),
        );
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

        let dim = self.vector_index.config.dim;
        let padded_dim = self.vector_index.quantizer.padded_dim();
        let rabitq_words = self.vector_index.quantizer.num_words();
        let mut doc_ids = Vec::new();

        for chunk in leaf_ids.chunks(RESOLVE_LEAF_CHUNK_SIZE) {
            let leaf_slices = self.storage.multi_get_leaf_nodes_pinned(chunk)?;
            let mut views: Vec<Option<LeafNodeView<'_>>> = Vec::with_capacity(chunk.len());
            for (&leaf_id, opt_slice) in chunk.iter().zip(leaf_slices.iter()) {
                let slice = opt_slice.as_ref().ok_or_else(|| {
                    TreeError::CorruptedData(format!("missing leaf blob for leaf {leaf_id}"))
                })?;
                views.push(Some(LeafNodeView::parse(
                    slice,
                    dim,
                    padded_dim,
                    rabitq_words,
                )?));
            }
            let leaf_pairs: Vec<(NodeId, Option<&LeafNodeView<'_>>)> = chunk
                .iter()
                .copied()
                .zip(views.iter().map(|v| v.as_ref()))
                .collect();
            let masks =
                self.attr_index
                    .eval_leaves_masks(&self.storage, &prepared, &leaf_pairs)?;
            for (view_opt, mask) in views.iter().zip(masks.iter()) {
                if mask.is_empty() {
                    continue;
                }
                if let Some(view) = view_opt {
                    mask.for_each_set_bit(view.slot_cap, |slot| {
                        if view.is_live(slot) {
                            doc_ids.push(view.doc_id(slot));
                        }
                    });
                }
            }
        }

        doc_ids.sort_unstable();
        doc_ids.dedup();
        Ok(doc_ids)
    }

    fn eval_positive_fts_bitmap(&self, filter: &Filter) -> Result<Option<RoaringBitmap>> {
        match filter {
            Filter::ContainsAllTokens {
                attr,
                query,
                last_as_prefix,
            } => Ok(Some(self.fts_index.eval_fts_predicate(
                &self.storage,
                &self.schema,
                izzydb_core::filter::FtsPredicateKind::AllTokens,
                attr,
                query,
                *last_as_prefix,
            )?)),
            Filter::ContainsAnyToken {
                attr,
                query,
                last_as_prefix,
            } => Ok(Some(self.fts_index.eval_fts_predicate(
                &self.storage,
                &self.schema,
                izzydb_core::filter::FtsPredicateKind::AnyToken,
                attr,
                query,
                *last_as_prefix,
            )?)),
            Filter::ContainsTokenSequence {
                attr,
                query,
                last_as_prefix,
            } => Ok(Some(self.fts_index.eval_fts_predicate(
                &self.storage,
                &self.schema,
                izzydb_core::filter::FtsPredicateKind::TokenSequence,
                attr,
                query,
                *last_as_prefix,
            )?)),
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

    /// Execute a BM25 full-text ranking query with an optional [`Filter`].
    pub fn query_bm25(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<Bm25Result>> {
        rank_by.validate(&self.schema)?;
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let Some(f) = filter else {
            return self
                .fts_index
                .query_bm25(&self.storage, &self.schema, rank_by, top_k, None, None);
        };

        let (pushdown_opt, is_exact) = self.try_pushdown_filter(f)?;
        if let Some(ref bm) = pushdown_opt
            && bm.is_empty()
        {
            return Ok(Vec::new());
        }

        if is_exact {
            self.fts_index.query_bm25(
                &self.storage,
                &self.schema,
                rank_by,
                top_k,
                pushdown_opt.as_ref(),
                None,
            )
        } else {
            let mut lazy = |doc_id: DocId| -> Result<bool> {
                let Some(slice) = self.storage.get_doc_bytes_pinned(doc_id)? else {
                    return Ok(false);
                };
                if !f.has_fts() {
                    DocView::parse(&slice)?
                        .matches_filter(&self.schema, f, &mut |_, _, _, _, _| false)
                } else {
                    let attrs = decode_attributes(&self.schema, &slice)?;
                    Ok(self.fts_index.matches_filter(&self.schema, f, &attrs))
                }
            };
            self.fts_index.query_bm25(
                &self.storage,
                &self.schema,
                rank_by,
                top_k,
                pushdown_opt.as_ref(),
                Some(&mut lazy),
            )
        }
    }

    /// Alias for [`Collection::query_bm25`].
    pub fn query(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<Bm25Result>> {
        self.query_bm25(rank_by, filter, top_k)
    }

    /// Run vector search and BM25 full-text search under `filter`, fusing rankings via RRF.
    pub fn hybrid_search(
        &self,
        query_vec: &[f32],
        rank_by: &RankBy,
        opts: HybridSearchOptions,
        filter: Option<&Filter>,
    ) -> Result<Vec<HybridSearchResult>> {
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

        if opts.k == 0 {
            return Ok(Vec::new());
        }

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

        let mut results: Vec<HybridSearchResult> = by_doc.into_values().collect();
        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.id.cmp(&b.id))
                .then(Ordering::Equal)
        });
        results.truncate(opts.k);
        Ok(results)
    }

    fn try_pushdown_filter(&self, filter: &Filter) -> Result<(Option<RoaringBitmap>, bool)> {
        match filter {
            Filter::ContainsAllTokens {
                attr,
                query,
                last_as_prefix,
            } => {
                let bm = self.fts_index.eval_fts_predicate(
                    &self.storage,
                    &self.schema,
                    izzydb_core::filter::FtsPredicateKind::AllTokens,
                    attr,
                    query,
                    *last_as_prefix,
                )?;
                Ok((Some(bm), true))
            }
            Filter::ContainsAnyToken {
                attr,
                query,
                last_as_prefix,
            } => {
                let bm = self.fts_index.eval_fts_predicate(
                    &self.storage,
                    &self.schema,
                    izzydb_core::filter::FtsPredicateKind::AnyToken,
                    attr,
                    query,
                    *last_as_prefix,
                )?;
                Ok((Some(bm), true))
            }
            Filter::ContainsTokenSequence {
                attr,
                query,
                last_as_prefix,
            } => {
                let bm = self.fts_index.eval_fts_predicate(
                    &self.storage,
                    &self.schema,
                    izzydb_core::filter::FtsPredicateKind::TokenSequence,
                    attr,
                    query,
                    *last_as_prefix,
                )?;
                Ok((Some(bm), true))
            }
            Filter::And(children) => {
                if children.is_empty() {
                    return Ok((None, true));
                }
                let mut combined: Option<RoaringBitmap> = None;
                let mut all_exact = true;
                for child in children {
                    let (sub_bm, sub_exact) = self.try_pushdown_filter(child)?;
                    if !sub_exact {
                        all_exact = false;
                    }
                    if let Some(bm) = sub_bm {
                        if bm.is_empty() {
                            return Ok((Some(RoaringBitmap::new()), true));
                        }
                        combined = Some(match combined {
                            Some(mut cur) => {
                                cur &= &bm;
                                if cur.is_empty() {
                                    return Ok((Some(RoaringBitmap::new()), true));
                                }
                                cur
                            }
                            None => bm,
                        });
                    }
                }
                Ok((combined, all_exact))
            }
            Filter::Or(children) => {
                if children.is_empty() {
                    return Ok((Some(RoaringBitmap::new()), true));
                }
                let mut union = RoaringBitmap::new();
                for child in children {
                    let (sub_bm, sub_exact) = self.try_pushdown_filter(child)?;
                    match (sub_bm, sub_exact) {
                        (Some(bm), true) => union |= &bm,
                        _ => return Ok((None, false)),
                    }
                }
                Ok((Some(union), true))
            }
            _ if !filter.has_fts() => {
                let prepared = self
                    .attr_index
                    .prepare_filter(&self.storage, &self.schema, filter)?;
                match prepared.cluster_candidates {
                    ClusterCandidates::Subset(ref bm) if bm.is_empty() => {
                        Ok((Some(RoaringBitmap::new()), true))
                    }
                    ClusterCandidates::Subset(ref bm) if bm.len() <= 64 => {
                        let leaf_ids: Vec<_> = bm.iter().collect();
                        let leaf_slices = self.storage.multi_get_leaf_nodes_pinned(&leaf_ids)?;
                        let dim = self.vector_index.config.dim;
                        let padded_dim = self.vector_index.quantizer.padded_dim();
                        let rabitq_words = self.vector_index.quantizer.num_words();
                        let mut views = Vec::with_capacity(leaf_ids.len());
                        for (lid, opt_slice) in leaf_ids.iter().copied().zip(leaf_slices.iter()) {
                            let slice = opt_slice.as_ref().ok_or_else(|| {
                                TreeError::CorruptedData(format!("missing leaf blob for leaf {lid}"))
                            })?;
                            views.push(Some(crate::node::LeafNodeView::parse(
                                slice,
                                dim,
                                padded_dim,
                                rabitq_words,
                            )?));
                        }
                        let leaf_pairs: Vec<_> = leaf_ids
                            .iter()
                            .copied()
                            .zip(views.iter().map(|v| v.as_ref()))
                            .collect();
                        let masks = self.attr_index.eval_leaves_masks(
                            &self.storage,
                            &prepared,
                            &leaf_pairs,
                        )?;
                        let mut doc_ids = Vec::new();
                        for (view_opt, mask) in views.iter().zip(masks.iter()) {
                            if mask.is_empty() {
                                continue;
                            }
                            if let Some(view) = view_opt {
                                mask.for_each_set_bit(view.slot_cap, |slot| {
                                    if view.is_live(slot) {
                                        doc_ids.push(view.doc_id(slot));
                                    }
                                });
                            }
                        }
                        if doc_ids.is_empty() {
                            return Ok((Some(RoaringBitmap::new()), true));
                        }
                        doc_ids.sort_unstable();
                        doc_ids.dedup();
                        let fts_ids = self.storage.multi_get_fts_doc_ids(&doc_ids)?;
                        let mut fts_bm = RoaringBitmap::new();
                        for fid in fts_ids.into_iter().flatten() {
                            fts_bm.insert(fid);
                        }
                        Ok((Some(fts_bm), true))
                    }
                    _ => Ok((None, false)),
                }
            }
            _ => Ok((None, false)),
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

#[allow(clippy::too_many_arguments)]
fn apply_patch_in_tx(
    schema: &CollectionSchema,
    attr_index: &AttributeIndex,
    fts_index: &FtsIndex,
    tree_config: &TreeConfig,
    quantizer: &RabitqQuantizer,
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
        Some(new_vec) => {
            let old_vec = tx.get_vectors_flat(&[id])?;
            let unchanged = old_vec.len() == new_vec.len()
                && old_vec
                    .iter()
                    .zip(new_vec.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            if unchanged { None } else { Some(new_vec) }
        }
        None => None,
    };

    if patch_attrs.is_empty() {
        let Some(new_vec) = effective_vec else {
            return Ok(());
        };
        match update_vector_in_tx(tx, id, old_loc, new_vec, tree_config, quantizer)? {
            VectorUpdateResult::SameLeaf { delta, .. } => {
                attr_index.apply_cluster_delta(tx, schema, &delta)?;
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
                    attr_index.remove_doc_entries(
                        tx,
                        schema,
                        &entries,
                        old_loc.leaf_id,
                        old_loc.local_id,
                    )?;
                    attr_index.apply_cluster_delta(tx, schema, &rem_delta)?;
                    attr_index.apply_cluster_delta(tx, schema, &ins_delta)?;
                    attr_index.insert_doc_entries(
                        tx,
                        schema,
                        &entries,
                        new_loc.leaf_id,
                        new_loc.local_id,
                    )?;
                } else {
                    attr_index.apply_cluster_delta(tx, schema, &rem_delta)?;
                    attr_index.apply_cluster_delta(tx, schema, &ins_delta)?;
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
        if let Some(entry) = schema.get_by_id(attr_id) {
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
    let new_doc_bytes = encode_attributes(schema, &merged_attrs)?;
    let fts_changed = has_fts_patch_changes(schema, &old_entries, patch_attrs);

    match effective_vec {
        None => {
            if fts_changed {
                fts_index.upsert_doc(tx, schema, id, Some(&old_entries), &merged_attrs)?;
            }
            attr_index.diff_patched_in_place(
                tx,
                schema,
                &old_entries,
                patch_attrs,
                old_loc.leaf_id,
                old_loc.local_id,
            )?;
            tx.put_doc_bytes(id, new_doc_bytes);
        }
        Some(new_vec) => {
            match update_vector_in_tx(tx, id, old_loc, new_vec, tree_config, quantizer)? {
                VectorUpdateResult::SameLeaf { delta, .. } => {
                    if fts_changed {
                        fts_index.upsert_doc(
                            tx,
                            schema,
                            id,
                            Some(&old_entries),
                            &merged_attrs,
                        )?;
                    }
                    attr_index.diff_patched_in_place(
                        tx,
                        schema,
                        &old_entries,
                        patch_attrs,
                        old_loc.leaf_id,
                        old_loc.local_id,
                    )?;
                    tx.put_doc_bytes(id, new_doc_bytes);
                    attr_index.apply_cluster_delta(tx, schema, &delta)?;
                }
                VectorUpdateResult::Moved {
                    old_loc,
                    new_loc,
                    rem_delta,
                    ins_delta,
                    split,
                } => {
                    attr_index.remove_doc_entries(
                        tx,
                        schema,
                        &old_entries,
                        old_loc.leaf_id,
                        old_loc.local_id,
                    )?;
                    attr_index.apply_cluster_delta(tx, schema, &rem_delta)?;
                    if fts_changed {
                        fts_index.upsert_doc(
                            tx,
                            schema,
                            id,
                            Some(&old_entries),
                            &merged_attrs,
                        )?;
                    }
                    tx.put_doc_bytes(id, new_doc_bytes);
                    attr_index.apply_cluster_delta(tx, schema, &ins_delta)?;
                    attr_index.insert_doc(
                        tx,
                        schema,
                        &merged_attrs,
                        new_loc.leaf_id,
                        new_loc.local_id,
                    )?;
                    if let Some(sp) = split {
                        splits.push(sp);
                    }
                }
            }
        }
    }
    Ok(())
}

fn filter_has_not(filter: &Filter) -> bool {
    match filter {
        Filter::And(children) | Filter::Or(children) => children.iter().any(filter_has_not),
        Filter::Not(_) => true,
        _ => false,
    }
}
