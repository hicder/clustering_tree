use std::path::Path;
use std::sync::Arc;

use crate::attr_index::AttributeIndex;
use crate::config::{DocId, TreeConfig};
use crate::error::{Result, TreeError};
use crate::filter::Filter;
use crate::lire;
use crate::metric::prepare_vector;
use crate::schema::{Attributes, CollectionConfig, CollectionSchema};
use crate::search::{self, SearchOptions};
use crate::storage::{Storage, TxContext};
use crate::codec::{decode_attribute_entries, decode_attributes, encode_attributes};
use crate::tree::{ClusteringTree, insert_in_tx, remove_existing_in_tx};
use crate::verify::{self, CollectionStats};

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

/// Top-level document + vector + attribute collection coordinating a shared RocksDB [`Storage`],
/// a [`ClusteringTree`] vector index, forward document attributes (`cf_docs`),
/// and a cluster-aware [`AttributeIndex`].
pub struct Collection {
    storage: Arc<Storage>,
    schema: CollectionSchema,
    vector_index: ClusteringTree,
    attr_index: AttributeIndex,
}

impl Collection {
    /// Open or create a [`Collection`] at `path` with the given `config`.
    pub fn open(path: impl AsRef<Path>, config: CollectionConfig) -> Result<Self> {
        config.validate()?;
        let (storage, meta) = Storage::open(path, &config.tree)?;
        let schema = match storage.get_schema()? {
            Some(mut existing) => {
                if existing.reconcile(&config)? {
                    storage.put_schema(&existing)?;
                }
                existing
            }
            None => {
                if meta.num_vectors > 0 {
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
        let vector_index = ClusteringTree::from_storage(Arc::clone(&storage), meta, config.tree)?;
        Ok(Self {
            storage,
            schema,
            vector_index,
            attr_index: AttributeIndex,
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

    /// Insert or update a document's vector and forward attributes atomically.
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
            }
            self.attr_index
                .apply_cluster_delta(&mut tx, &self.schema, &rem_delta)?;
        }

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

    /// Delete a document's vector, forward attributes, and index entries atomically.
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

    /// Exhaustively verify all structural, document, and attribute index invariants.
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
