//! Lexical, hybrid, and ordered queries.

use std::collections::HashMap;

use izzydb_core::config::DocId;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::Filter;
use izzydb_core::rank::{OrderBySpec, OrderedResult, RankBy};
use izzydb_fts::Bm25Result;


use super::*;

impl Collection {
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

    pub(super) fn hybrid_search_with_legs(
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
}
