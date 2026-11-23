//! Faceted search entry points.


use izzydb_core::config::{DocId, RowId};
use izzydb_core::error::Result;
use izzydb_core::facet::{
    FacetResults, FacetSpec, FacetedSearchResponse, HybridFacetScope, VectorFacetScope,
    validate_facet_specs,
};
use izzydb_core::filter::Filter;
use izzydb_core::rank::RankBy;
use izzydb_core::rowset::RowSet;
use izzydb_fts::Bm25Result;

use crate::facet::{FacetEnv, Population};
use crate::filter_plan::FilterPlanTrace;
use crate::segments::RowResolver;
use crate::SearchOptions;

use super::*;

impl Collection {
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
}
