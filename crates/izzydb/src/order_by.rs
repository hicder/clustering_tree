//! Attribute-ordered top-k queries (`ORDER BY`) over the existing order-preserving
//! `cf_attr_clusters` / `cf_attr_postings` indexes (no additional storage formats).
//!
//! * Broad / unfiltered queries stream `cf_attr_clusters` in key order (forward for `Asc`,
//!   reverse for `Desc`) and stop as soon as `top_k` rows are produced.
//! * Selective filters (few candidate leaves) resolve the matching documents and sort in memory.
//! * Documents whose primary key is absent / `Null` are placed first or last per [`NullsOrder`].
//!
//! Ties are broken by the next `order_by` key, then by ascending `DocId`.

use std::cmp::Ordering;

use crate::attr_index::{ClusterCandidates, PreparedFilter};
use crate::codec::{DocView, decode_scalar_val_bytes, encode_attr_prefix, ordered_type_tag};
use crate::collection::Collection;
use crate::config::{DocId, NodeId};
use crate::error::Result;
use crate::facet::{FacetSpec, FacetedSearchResponse, Population};
use crate::filter::Filter;
use crate::rank::{NullsOrder, OrderBySpec, OrderedResult, SortDirection, validate_order_by};
use crate::schema::ScalarValue;
use crate::search;

/// Candidate-leaf count at or below which a filtered query sorts the matches in memory instead
/// of streaming the index (mirrors `SearchOptions::exhaustive_cluster_threshold`).
const ORDER_BY_EXHAUSTIVE_CLUSTER_THRESHOLD: usize = 8;

fn scalar_cmp(a: &ScalarValue, b: &ScalarValue) -> Ordering {
    match (a, b) {
        (ScalarValue::Int(x), ScalarValue::Int(y)) => x.cmp(y),
        (ScalarValue::Uint(x), ScalarValue::Uint(y)) => x.cmp(y),
        (ScalarValue::Float(x), ScalarValue::Float(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (ScalarValue::String(x), ScalarValue::String(y)) => x.as_bytes().cmp(y.as_bytes()),
        (ScalarValue::Bool(x), ScalarValue::Bool(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

/// Compare two key tuples under `specs` (nulls placed per spec regardless of direction).
fn cmp_keys(specs: &[OrderBySpec], a: &[ScalarValue], b: &[ScalarValue]) -> Ordering {
    for (spec, (x, y)) in specs.iter().zip(a.iter().zip(b)) {
        let ord = match (matches!(x, ScalarValue::Null), matches!(y, ScalarValue::Null)) {
            (true, true) => Ordering::Equal,
            (true, false) => match spec.nulls {
                NullsOrder::First => Ordering::Less,
                NullsOrder::Last => Ordering::Greater,
            },
            (false, true) => match spec.nulls {
                NullsOrder::First => Ordering::Greater,
                NullsOrder::Last => Ordering::Less,
            },
            (false, false) => {
                let o = scalar_cmp(x, y);
                match spec.direction {
                    SortDirection::Asc => o,
                    SortDirection::Desc => o.reverse(),
                }
            }
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

impl Collection {
    /// Top-`top_k` documents ordered by one or more attributes, optionally restricted by `filter`.
    ///
    /// The primary key must be a `filterable` ordered scalar (`Int` / `Uint` / `Float` / `String`);
    /// secondary keys only need to be ordered scalars. Equal key tuples tie-break by ascending `DocId`.
    pub fn query_ordered(
        &self,
        order_by: &[OrderBySpec],
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<OrderedResult>> {
        validate_order_by(&self.schema, order_by)?;
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let prepared = match filter {
            Some(f) => Some(self.attr_index.prepare_filter_with_fts(
                &self.storage,
                &self.schema,
                f,
                Some(&self.fts_index),
            )?),
            None => None,
        };
        if prepared
            .as_ref()
            .is_some_and(|p| p.cluster_candidates.is_empty())
        {
            return Ok(Vec::new());
        }

        // Selective-filter crossover: sort the few matching documents in memory.
        if let (Some(f), Some(p)) = (filter, prepared.as_ref())
            && let ClusterCandidates::Subset(bm) = &p.cluster_candidates
            && bm.len() as usize <= ORDER_BY_EXHAUSTIVE_CLUSTER_THRESHOLD
        {
            let ids = self.resolve_filter_doc_ids(f)?;
            return Ok(to_results(self.sort_docs(order_by, ids, top_k)?));
        }

        let primary = &order_by[0];
        let mut rows: Vec<(DocId, Vec<ScalarValue>)> = Vec::new();

        if primary.nulls == NullsOrder::First {
            let ids = self.null_primary_docs(primary, filter)?;
            rows.extend(self.sort_docs(order_by, ids, top_k)?);
        }

        if rows.len() < top_k {
            self.stream_value_groups(primary, prepared.as_ref(), &mut |value, ids| {
                let remaining = top_k - rows.len();
                if order_by.len() == 1 {
                    // `ids` is ascending: equal primary keys tie-break by `DocId`.
                    rows.extend(
                        ids.into_iter()
                            .take(remaining)
                            .map(|id| (id, vec![value.clone()])),
                    );
                } else {
                    rows.extend(self.sort_docs(order_by, ids, remaining)?);
                }
                Ok(rows.len() < top_k)
            })?;
        }

        if rows.len() < top_k && primary.nulls == NullsOrder::Last {
            let ids = self.null_primary_docs(primary, filter)?;
            rows.extend(self.sort_docs(order_by, ids, top_k - rows.len())?);
        }
        rows.truncate(top_k);
        Ok(to_results(rows))
    }

    /// [`Collection::query_ordered`] plus facets over every document matching `filter`.
    pub fn query_ordered_with_facets(
        &self,
        order_by: &[OrderBySpec],
        filter: Option<&Filter>,
        top_k: usize,
        facets: &[FacetSpec],
    ) -> Result<FacetedSearchResponse<OrderedResult>> {
        validate_order_by(&self.schema, order_by)?;
        let hits = self.query_ordered(order_by, filter, top_k)?;
        // Validates `filter` and `facets`, then aggregates over the full filter match set.
        let (facets, total_hits) = crate::facet::compute_facets(
            &self.facet_env(),
            facets,
            Population::Query {
                filter,
                rank_bm: None,
                prepared: None,
            },
        )?;
        Ok(FacetedSearchResponse {
            hits,
            facets,
            total_hits,
        })
    }

    /// Documents whose primary sort attribute is absent or `Null`, intersected with `filter`.
    fn null_primary_docs(&self, primary: &OrderBySpec, filter: Option<&Filter>) -> Result<Vec<DocId>> {
        let is_null = Filter::IsNull(primary.attr.clone());
        let combined = match filter {
            Some(f) => Filter::And(vec![is_null, f.clone()]),
            None => is_null,
        };
        self.resolve_filter_doc_ids(&combined)
    }

    /// Read the sort-key tuple of every document in `ids` and return the best `limit` rows
    /// ordered by `order_by` then `DocId`.
    fn sort_docs(
        &self,
        order_by: &[OrderBySpec],
        mut ids: Vec<DocId>,
        limit: usize,
    ) -> Result<Vec<(DocId, Vec<ScalarValue>)>> {
        if limit == 0 || ids.is_empty() {
            return Ok(Vec::new());
        }
        ids.sort_unstable();
        ids.dedup();
        let attr_ids: Vec<_> = order_by
            .iter()
            .map(|s| self.schema.get_by_name(&s.attr).map(|e| e.id).unwrap_or_default())
            .collect();
        let mut rows: Vec<(DocId, Vec<ScalarValue>)> = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(1024) {
            let slices = self.storage.multi_get_doc_bytes_pinned(chunk)?;
            for (&id, slice) in chunk.iter().zip(slices) {
                let Some(slice) = slice else { continue };
                let view = DocView::parse(&slice)?;
                let keys = attr_ids
                    .iter()
                    .map(|&a| view.get_scalar(a))
                    .collect::<Result<Vec<_>>>()?;
                rows.push((id, keys));
            }
        }
        let cmp = |a: &(DocId, Vec<ScalarValue>), b: &(DocId, Vec<ScalarValue>)| {
            cmp_keys(order_by, &a.1, &b.1).then_with(|| a.0.cmp(&b.0))
        };
        if rows.len() > limit {
            rows.select_nth_unstable_by(limit - 1, cmp);
            rows.truncate(limit);
        }
        rows.sort_by(cmp);
        Ok(rows)
    }

    /// Stream `(value, ascending DocIds)` groups of the filter-matching documents in the order of
    /// `spec` by walking `cf_attr_clusters` (reverse for `Desc`). `f` returns `false` to stop.
    /// Documents without a value for `spec.attr` are not visited.
    pub(crate) fn stream_value_groups(
        &self,
        spec: &OrderBySpec,
        prepared: Option<&PreparedFilter>,
        f: &mut dyn FnMut(ScalarValue, Vec<DocId>) -> Result<bool>,
    ) -> Result<()> {
        let entry = self.schema.get_by_name(&spec.attr).ok_or_else(|| {
            crate::error::TreeError::InvalidQuery(format!("unknown attribute '{}'", spec.attr))
        })?;
        let tag = ordered_type_tag(entry.attr_type)?;
        let prefix = encode_attr_prefix(entry.id, tag);
        let reverse = spec.direction == SortDirection::Desc;
        let candidates = prepared.map(|p| &p.cluster_candidates);

        for item in self.storage.iter_attr_clusters_prefix(prefix, reverse)? {
            let (vbytes, leaf_bm) = item?;
            let leaves = match candidates {
                Some(ClusterCandidates::Subset(c)) => &leaf_bm & c,
                _ => leaf_bm.clone(),
            };
            if leaves.is_empty() {
                continue;
            }
            let mut key = Vec::with_capacity(prefix.len() + vbytes.len());
            key.extend_from_slice(&prefix);
            key.extend_from_slice(&vbytes);
            let restricted = PreparedFilter::restricted_to_key(prepared, key, leaf_bm);
            let leaf_ids: Vec<NodeId> = leaves.iter().collect();
            let mut ids = Vec::new();
            search::collect_matching_doc_ids_in_leaves(
                &self.storage,
                &self.attr_index,
                &restricted,
                &leaf_ids,
                &self.vector_index.config,
                &self.vector_index.quantizer,
                &mut ids,
            )?;
            if ids.is_empty() {
                continue;
            }
            ids.sort_unstable();
            ids.dedup();
            let value = decode_scalar_val_bytes(tag, &vbytes)?;
            if !f(value, ids)? {
                break;
            }
        }
        Ok(())
    }
}

fn to_results(rows: Vec<(DocId, Vec<ScalarValue>)>) -> Vec<OrderedResult> {
    rows.into_iter()
        .map(|(id, sort_values)| OrderedResult { id, sort_values })
        .collect()
}
