use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use roaring::RoaringBitmap;

use crate::attr_index::{AttributeIndex, ClusterCandidates};
use crate::codec::{decode_attribute_entries, decode_attributes, encode_attributes};
use crate::config::{DocId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::filter::Filter;
use crate::fts_index::FtsIndex;
use crate::lire;
use crate::metric::prepare_vector;
use crate::rank::RankBy;
use crate::schema::{AttributeValue, Attributes, CollectionConfig, CollectionSchema};
use crate::search::{self, SearchOptions};
use crate::storage::{Storage, TxContext};
use crate::tree::{ClusteringTree, insert_in_tx, remove_existing_in_tx};
use crate::verify::{self, CollectionStats};
use izzydb_fts::Bm25Result;

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
        let mut old_entries_opt: Option<Vec<(izzydb_core::config::AttrId, AttributeValue)>> = None;
        if let Some((old_loc, rem_delta)) = remove_existing_in_tx(
            &mut tx,
            id,
            &self.vector_index.config,
            &self.vector_index.quantizer,
        )? {
            if let Some(old_doc_bytes) = tx.get_doc_bytes(id)? {
                let old_entries = decode_attribute_entries(&old_doc_bytes)?;
                self.attr_index.remove_doc_entries(
                    &mut tx,
                    &self.schema,
                    &old_entries,
                    old_loc.leaf_id,
                    old_loc.local_id,
                )?;
                old_entries_opt = Some(old_entries);
            }
            self.attr_index
                .apply_cluster_delta(&mut tx, &self.schema, &rem_delta)?;
        }

        self.fts_index.upsert_doc(
            &mut tx,
            &self.schema,
            id,
            old_entries_opt.as_deref(),
            &attributes,
        )?;

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

        self.vector_index.meta = tx.commit()?;

        if let Some(split_pair) = split_opt
            && self.vector_index.config.enable_lire
            && self.vector_index.meta.root_level > 0
        {
            lire::run_lire_with_hook(&mut self.vector_index, split_pair, |tx, delta| {
                self.attr_index
                    .apply_cluster_delta(tx, &self.schema, delta)
            })?;
        }
        Ok(())
    }

    /// Delete a document's vector, forward attributes, attribute index, and FTS index entries atomically.
    ///
    /// If `id` does not exist, returns `Ok(())`.
    pub fn delete(&mut self, id: DocId) -> Result<()> {
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
            return Ok(());
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
                let Some(bytes) = self.storage.get_doc_bytes(doc_id)? else {
                    return Ok(false);
                };
                let attrs = decode_attributes(&self.schema, &bytes)?;
                Ok(self.fts_index.matches_filter(&self.schema, f, &attrs))
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
                                for slot in 0..view.entry_count {
                                    if mask.contains(view.local_id(slot)) {
                                        doc_ids.push(view.doc_id(slot));
                                    }
                                }
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
