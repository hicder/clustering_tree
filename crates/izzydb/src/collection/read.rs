//! Point reads and vector search.


use izzydb_core::config::{DocId, RowId};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::schema::Attributes;
use izzydb_storage::codec::{
    RowView, decode_row_attributes,
};
use izzydb_storage::node::decode_f32_slice;

use crate::detached::{
    self,
};
use crate::filter_plan::{self, FilterPlanTrace};
use crate::search;
use crate::segments::RowResolver;
use crate::SearchOptions;

use super::*;

impl Collection {
    /// Retrieve a document's stored vector and attributes (inline + detached) by `id`.
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
        let mut attributes = decode_row_attributes(&self.schema, &row_bytes)?;
        detached::read_detached_attrs_for_row(&self.storage, &self.schema, row, &mut attributes)?;
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
        let mut decoded_attrs: Vec<(usize, DocId, Attributes)> =
            Vec::with_capacity(to_fetch.len());
        let mut per_row_attrs: Vec<Attributes> = Vec::with_capacity(to_fetch.len());

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
            per_row_attrs.push(attributes);
            decoded_attrs.push((slot, doc_id, Attributes::new()));
        }
        detached::read_detached_attrs_for_rows_sorted(
            &self.storage,
            &self.schema,
            &sorted_rows,
            &mut per_row_attrs,
        )?;

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

        for (idx, ((slot, doc_id, _), attributes)) in
            decoded_attrs.into_iter().zip(per_row_attrs).enumerate()
        {
            out[slot] = Some(Document {
                id: doc_id,
                embedding: embeddings[idx].take(),
                attributes,
            });
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

    pub(super) fn resolve_search_results(&self, row_hits: Vec<(RowId, f32)>) -> Result<Vec<SearchResult>> {
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
}
