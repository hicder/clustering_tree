//! High-level `Collection` API coordinating the RocksDB [`Storage`],
//! [`SegmentStore`], and secondary indexes (ANN, attribute index, FTS).

mod facets;
mod maintenance;
mod query;
mod read;
mod types;
mod write;

pub use types::*;
pub use write::BulkTxHelper;
use write::collect_hooks;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use izzydb_core::config::{DocId, NodeId, SegmentId, TreeConfig};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::metric::prepare_vector;
use izzydb_core::schema::{AttributeValue, Attributes, CollectionConfig, CollectionSchema};
use izzydb_storage::codec::Manifest;
use izzydb_storage::node::{AnnMeta, PackedInternalNode};
use izzydb_storage::{SegmentTableSnapshot, Storage, TxContext};

use crate::attr_index::AttributeIndex;
use crate::detached::{
    self, DetachedIngestBuilder, DetachedJob, DetachedPublishStats,
};
use crate::fts_index::FtsIndex;
use crate::segments::SegmentStore;
use crate::tree::{
    AnnIndex,
    load_internal_caches,
};
use crate::CollectionStats;

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
        let leaf_specs: Vec<izzydb_storage::LeafAttrSpec> = schema
            .entries()
            .iter()
            .filter(|e| {
                e.storage == izzydb_core::schema::ColumnStorage::Inline
                    && e.filterable
                    && izzydb_storage::LeafAttrSpec::is_supported(e.attr_type)
            })
            .map(|e| izzydb_storage::LeafAttrSpec {
                attr_id: e.id,
                attr_type: e.attr_type,
            })
            .collect();
        storage.set_leaf_attr_specs(leaf_specs);
        let segments = SegmentStore::new(
            config.segment.seal_rows,
            config.segment.defer_column_build,
            config.bit_sliced_index,
        );
        let (ann, internal_cache, leaf_parent) = match (&config.vector, &ann_meta) {
            (Some(vcfg), Some(meta)) => {
                let ann = AnnIndex::new(vcfg.clone(), meta.rabitq_seed)?;
                let (internal_cache, leaf_parent) = load_internal_caches(&storage)?;
                (Some(ann), internal_cache, leaf_parent)
            }
            _ => (None, HashMap::new(), HashMap::new()),
        };
        let fts_index = FtsIndex::new().with_parallel_segments(config.fts_parallel_segments);
        Ok(Self {
            storage,
            config,
            schema,
            segments,
            manifest,
            ann_meta,
            ann,
            attr_index: AttributeIndex,
            fts_index,
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

    pub(crate) fn apply_tx_commit(&mut self, manifest: Manifest, ann_meta: Option<AnnMeta>) {
        self.manifest = manifest;
        self.ann_meta = ann_meta;
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
        self.segments.seal(&mut tx, &self.schema, open_seg, &hooks)?;
        let (manifest, ann_meta) = tx.commit()?;
        self.manifest = manifest;
        self.ann_meta = ann_meta;
        Ok(Some(open_seg))
    }

    /// Start building an atomic whole-column detached ingest job on `&self`.
    pub fn detached_ingest(&self) -> DetachedIngestBuilder<'_> {
        DetachedIngestBuilder::new(self)
    }

    /// Atomically publish a staged [`DetachedJob`] onto the collection, resolving concurrent
    /// writes above the build watermark.
    pub fn publish_detached(&mut self, job: DetachedJob) -> Result<DetachedPublishStats> {
        detached::publish_detached_job(self, job)
    }

    /// Point write a single detached attribute value onto an existing live document in place
    /// (without moving its `RowId`).
    pub fn set_detached(
        &mut self,
        id: DocId,
        attr: impl Into<String>,
        value: AttributeValue,
    ) -> Result<()> {
        let mut attrs = Attributes::new();
        attrs.insert(attr.into(), value);
        detached::apply_point_detached_batch(self, &[(id, attrs)])
    }

    /// Point write detached attribute updates onto existing live documents atomically in place.
    pub fn set_detached_batch(&mut self, updates: &[(DocId, Attributes)]) -> Result<()> {
        detached::apply_point_detached_batch(self, updates)
    }

    pub fn verify_invariants(&self) -> Result<CollectionStats> {
        crate::verify::verify_collection_invariants(
            &self.storage,
            &self.internal_cache,
            &self.leaf_parent,
            &self.manifest,
            self.ann_meta.as_ref(),
            &self.config,
            &self.schema,
        )
    }

    #[doc(hidden)]
    pub fn with_bulk_tx<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(&mut TxContext<'_>, &BulkTxHelper<'_>) -> Result<()>,
    {
        let helper = BulkTxHelper {
            segments: &self.segments,
            schema: &self.schema,
            ann: self.ann.as_ref(),
            attr_index: &self.attr_index,
            fts_index: &self.fts_index,
            ann_meta: self.ann_meta.as_ref(),
            old_internal_ids: self.internal_cache.keys().copied().collect(),
            old_leaf_ids: {
                let mut lids: Vec<NodeId> = self.leaf_parent.keys().copied().collect();
                if let Some(ref cur_meta) = self.ann_meta
                    && cur_meta.root_level == 0
                {
                    lids.push(cur_meta.root_id);
                }
                lids.sort_unstable();
                lids.dedup();
                lids
            },
        };
        let mut tx = TxContext::new(
            &self.storage,
            &mut self.internal_cache,
            &mut self.leaf_parent,
            self.manifest.clone(),
            self.ann_meta.clone(),
        );
        f(&mut tx, &helper)?;
        let (manifest, ann_meta) = tx.commit()?;
        self.manifest = manifest;
        self.ann_meta = ann_meta;
        Ok(())
    }
}
