//! Mutations: upsert, patch, delete, filter-based updates, and the transactional helpers behind them.


use izzydb_core::config::{AttrId, DocId, NodeId, RowId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::schema::{AttributeValue, Attributes, CollectionSchema};
use izzydb_storage::codec::{
    RowView,
    decode_row_entries, encode_row,
};
use izzydb_storage::node::{AnnMeta, PackedInternalNode};
use izzydb_storage::TxContext;

use crate::attr_index::AttributeIndex;
use crate::detached::{
    self,
};
use crate::filter_plan::{self};
use crate::fts_index::FtsIndex;
use crate::segments::{RowIndexHook, RowResolver, SegmentStore};
use crate::tree::AnnIndex;

use super::*;

impl Collection {
    /// Insert or update a single document atomically.
    pub fn upsert(
        &mut self,
        id: DocId,
        embedding: Option<&[f32]>,
        attributes: Attributes,
    ) -> Result<()> {
        self.schema.validate_attributes(&attributes)?;
        let (inline_attrs, detached_updates) =
            detached::partition_attributes(&self.schema, &attributes);
        let prepared_vec = self.prepare_optional_vector(embedding)?;
        let new_row_bytes = encode_row(&self.schema, &inline_attrs, prepared_vec.is_some())?;

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
            &detached_updates,
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
            let (inline_attrs, detached_updates) =
                detached::partition_attributes(&self.schema, &doc.attributes);
            let pvec = self.prepare_optional_vector(doc.embedding.as_deref())?;
            let row_bytes = encode_row(&self.schema, &inline_attrs, pvec.is_some())?;
            prepared.push((doc.id, pvec, row_bytes, detached_updates));
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
        for (id, pvec, row_bytes, detached_updates) in prepared {
            if upsert_one_in_tx(
                &self.segments,
                &self.schema,
                &mut tx,
                id,
                pvec.as_deref(),
                row_bytes,
                &detached_updates,
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
        let (inline_attrs, detached_updates) =
            detached::partition_attributes(&self.schema, &patch.attributes);
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
            &inline_attrs,
            &detached_updates,
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
        let mut split_attrs = Vec::with_capacity(patches.len());
        let mut doc_ids = Vec::with_capacity(patches.len());
        for &(id, ref patch) in patches {
            self.schema.validate_attributes(&patch.attributes)?;
            prepared_vecs.push(self.prepare_optional_vector(patch.embedding.as_deref())?);
            split_attrs.push(detached::partition_attributes(&self.schema, &patch.attributes));
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
        for ((&(id, _), pv), (inline_attrs, detached_updates)) in
            patches.iter().zip(prepared_vecs.iter()).zip(split_attrs.iter())
        {
            if apply_patch_in_tx(
                &self.segments,
                &self.schema,
                &mut tx,
                id,
                pv.as_deref(),
                inline_attrs,
                detached_updates,
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
        detached::delete_detached_in_tx(&mut tx, &self.schema, old_row)?;
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
            detached::delete_detached_in_tx(&mut tx, &self.schema, old_row)?;
            tx.delete_id_map(doc_id);
        }
        let (manifest, ann_meta) = tx.commit()?;
        self.manifest = manifest;
        self.ann_meta = ann_meta;
        Ok(targets.len() as u64)
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
        let (inline_attrs, detached_updates) =
            detached::partition_attributes(&self.schema, &patch.attributes);
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
                    &inline_attrs,
                    &detached_updates,
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
}

#[doc(hidden)]
pub struct BulkTxHelper<'a> {
    pub(super) segments: &'a SegmentStore,
    pub(super) schema: &'a CollectionSchema,
    pub(super) ann: Option<&'a AnnIndex>,
    pub(super) attr_index: &'a AttributeIndex,
    pub(super) fts_index: &'a FtsIndex,
    pub(super) ann_meta: Option<&'a AnnMeta>,
    pub(super) old_internal_ids: Vec<NodeId>,
    pub(super) old_leaf_ids: Vec<NodeId>,
}

impl<'a> BulkTxHelper<'a> {
    pub fn delete_row_for_replace(&self, tx: &mut TxContext<'_>, old_row: RowId) -> Result<()> {
        let old_bytes = tx.get_row_bytes(old_row)?.ok_or_else(|| {
            TreeError::InvariantViolation(format!("row {old_row:?} missing in cf_rows"))
        })?;
        let old_view = RowView::parse(&old_bytes)?;
        self.segments.tombstone(tx, old_row)?;
        let hooks = collect_hooks(self.ann, self.attr_index, self.fts_index);
        for hook in &hooks {
            hook.on_delete(tx, self.schema, old_row, &old_view, old_view.has_vector())?;
        }
        detached::delete_detached_in_tx(tx, self.schema, old_row)?;
        Ok(())
    }

    pub fn insert_vector(
        &self,
        tx: &mut TxContext<'_>,
        row: RowId,
        prepared_vec: &[f32],
    ) -> Result<()> {
        let Some(ann) = self.ann else {
            return Ok(());
        };
        let empty_bytes = encode_row(self.schema, &Attributes::new(), true)?;
        let empty_view = RowView::parse(&empty_bytes)?;
        ann.on_insert(tx, self.schema, row, &empty_view, Some(prepared_vec), None)
    }

    pub fn replace_ann_tree(
        &self,
        tx: &mut TxContext<'_>,
        new_meta: AnnMeta,
        internals: Vec<(NodeId, PackedInternalNode)>,
        leaves: Vec<(NodeId, izzydb_storage::node::PackedLeafNode)>,
    ) -> Result<()> {
        let _ = self.ann_meta;
        for &nid in &self.old_internal_ids {
            tx.delete_internal_node(nid);
        }
        for &lid in &self.old_leaf_ids {
            tx.delete_leaf_node(lid);
        }
        for (nid, node) in internals {
            tx.put_internal_node(nid, node);
        }
        for (lid, leaf) in leaves {
            tx.put_leaf_node(lid, leaf);
        }
        *tx.ann_meta_mut()? = new_meta;
        Ok(())
    }
}

pub(super) fn collect_hooks<'a>(
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

#[allow(clippy::too_many_arguments)]
fn upsert_one_in_tx(
    segments: &SegmentStore,
    schema: &CollectionSchema,
    tx: &mut TxContext<'_>,
    id: DocId,
    prepared_vec: Option<&[f32]>,
    new_row_bytes: Vec<u8>,
    detached_updates: &[(AttrId, AttributeValue)],
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
            return detached::apply_detached_write_on_existing_row(
                tx,
                schema,
                old_row,
                detached_updates,
                true,
            );
        }
        segments.tombstone(tx, old_row)?;
        for hook in hooks {
            hook.on_delete(tx, schema, old_row, &old_view, old_view.has_vector())?;
        }
        detached::delete_detached_in_tx(tx, schema, old_row)?;
    }

    let new_row = segments.alloc_row(tx)?;
    segments.write_row(tx, new_row, id, new_row_bytes, prepared_vec)?;
    let new_bytes = tx.get_row_bytes(new_row)?.expect("just written");
    let new_view = RowView::parse(&new_bytes)?;
    for hook in hooks {
        hook.on_insert(tx, schema, new_row, &new_view, prepared_vec, prev_row)?;
    }
    detached::write_detached_on_new_row(tx, schema, None, new_row, detached_updates, false)?;
    segments.maybe_seal(tx, schema, hooks)?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn apply_patch_in_tx(
    segments: &SegmentStore,
    schema: &CollectionSchema,
    tx: &mut TxContext<'_>,
    id: DocId,
    prepared_vec: Option<&[f32]>,
    inline_patch_attrs: &Attributes,
    detached_updates: &[(AttrId, AttributeValue)],
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

    let inline_changed_bytes: Option<Vec<u8>> = if inline_patch_attrs.is_empty() {
        if vectors_bitwise_equal(tx, old_row, old_view.has_vector(), effective_vec.as_deref())? {
            None
        } else {
            let mut attrs = Attributes::new();
            for (attr_id, val) in decode_row_entries(&old_bytes)? {
                if let Some(entry) = schema.get_by_id(attr_id) {
                    attrs.insert(entry.name.clone(), val);
                }
            }
            Some(encode_row(schema, &attrs, effective_vec.is_some())?)
        }
    } else {
        let old_entries = decode_row_entries(&old_bytes)?;
        let mut merged = Attributes::new();
        for (attr_id, val) in old_entries {
            if let Some(entry) = schema.get_by_id(attr_id) {
                merged.insert(entry.name.clone(), val);
            }
        }
        for (name, val) in inline_patch_attrs {
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
            None
        } else {
            Some(encoded)
        }
    };

    let Some(new_row_bytes) = inline_changed_bytes else {
        if detached_updates.is_empty() {
            return Ok(false);
        }
        return detached::apply_detached_write_on_existing_row(
            tx,
            schema,
            old_row,
            detached_updates,
            false,
        );
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
    detached::write_detached_on_new_row(
        tx,
        schema,
        Some(old_row),
        new_row,
        detached_updates,
        true,
    )?;
    segments.maybe_seal(tx, schema, hooks)?;
    Ok(true)
}
