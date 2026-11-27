//! BM25, predicate, and rank-by query evaluation.



use izzydb_core::config::RowId;
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::{Filter, FtsPredicateKind, TokenQuery};
use izzydb_core::rank::RankBy;
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::{AttributeValue, Attributes, CollectionSchema};
use izzydb_fts::{
    BlockMaskProvider, Bm25Result, DEFAULT_MAX_PREFIX_EXPANSIONS,
    FtsAnalyzer, FtsPostingSource, RankModifier, TermRecord, execute_rank_by_with_shards,
};
use izzydb_storage::{
    SegmentTableSnapshot, Storage,
};


use super::*;

impl FtsIndex {
    /// Execute a [`RankBy`] BM25 query with combined liveness + optional pushed-down [`RowSet`] mask
    /// and optional lazy filter callback.
    #[allow(clippy::too_many_arguments)]
    pub fn query_bm25(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        rank_by: &RankBy,
        top_k: usize,
        block_mask: Option<&dyn BlockMaskProvider>,
        lazy_filter: Option<&mut dyn FnMut(RowId) -> Result<bool>>,
    ) -> Result<Vec<Bm25Result>> {
        self.query_bm25_with(
            storage,
            schema,
            rank_by,
            top_k,
            block_mask,
            lazy_filter,
            None,
        )
    }

    /// [`FtsIndex::query_bm25`] with an optional [`RankModifier`].
    #[allow(clippy::too_many_arguments)]
    pub fn query_bm25_with(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        rank_by: &RankBy,
        top_k: usize,
        block_mask: Option<&dyn BlockMaskProvider>,
        lazy_filter: Option<&mut dyn FnMut(RowId) -> Result<bool>>,
        modifier: Option<RankModifier<'_>>,
    ) -> Result<Vec<Bm25Result>> {
        let source = StorageFtsSource::new(storage);
        let mut buf = self.acquire_buffer();
        execute_rank_by_with_shards(
            &source,
            schema,
            rank_by,
            top_k,
            block_mask,
            lazy_filter,
            modifier,
            self.max_parallel_segments,
            &mut buf,
        )
    }

    /// Estimate live cardinality of an FTS predicate from `TermRecord.stats.doc_freq`.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_fts_predicate(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        kind: FtsPredicateKind,
        attr: &str,
        query: &TokenQuery,
        last_as_prefix: bool,
        live_rows: u64,
    ) -> Result<u64> {
        let entry = schema
            .get_by_name(attr)
            .ok_or_else(|| TreeError::InvalidFilter(format!("unknown attribute '{attr}'")))?;
        let fts_cfg = entry.full_text_search.as_ref().ok_or_else(|| {
            TreeError::InvalidFilter(format!(
                "attribute '{attr}' does not have full_text_search enabled"
            ))
        })?;
        let qtokens = {
            let mut buf = self.acquire_buffer();
            FtsAnalyzer::new(fts_cfg.clone()).analyze_query(query, &mut buf)?
        };
        if qtokens.is_empty() {
            return Ok(0);
        }

        let source = StorageFtsSource::new(storage);
        let mut exact_tokens = exact_query_tokens(&qtokens, last_as_prefix);
        exact_tokens.sort_unstable();
        exact_tokens.dedup();
        let exact_recs = source.multi_get_term_records(entry.id, &exact_tokens)?;
        let prefix_recs: Vec<TermRecord> = if last_as_prefix {
            let prefix = &qtokens.last().unwrap().text;
            source
                .scan_prefix_terms(entry.id, prefix, DEFAULT_MAX_PREFIX_EXPANSIONS)?
                .into_iter()
                .map(|(_, r)| r)
                .filter(|r| r.stats.doc_freq > 0)
                .collect()
        } else {
            Vec::new()
        };

        let est = match kind {
            FtsPredicateKind::AnyToken => {
                let exact_sum: u64 = exact_recs
                    .iter()
                    .flatten()
                    .map(|r| r.stats.doc_freq)
                    .fold(0u64, u64::saturating_add);
                let pref_sum: u64 = prefix_recs
                    .iter()
                    .map(|r| r.stats.doc_freq)
                    .fold(0u64, u64::saturating_add);
                exact_sum.saturating_add(pref_sum)
            }
            FtsPredicateKind::AllTokens | FtsPredicateKind::TokenSequence => {
                let mut min_df = u64::MAX;
                for rec_opt in &exact_recs {
                    let df = rec_opt.as_ref().map_or(0, |r| r.stats.doc_freq);
                    if df == 0 {
                        return Ok(0);
                    }
                    min_df = min_df.min(df);
                }
                if last_as_prefix {
                    let pref_sum: u64 = prefix_recs
                        .iter()
                        .map(|r| r.stats.doc_freq)
                        .fold(0u64, u64::saturating_add);
                    if pref_sum == 0 {
                        return Ok(0);
                    }
                    min_df = min_df.min(pref_sum);
                }
                if min_df == u64::MAX { 0 } else { min_df }
            }
        };
        Ok(est.min(live_rows))
    }

    /// Evaluate a single FTS filter predicate directly into a live [`RowSet`].
    pub fn eval_fts_predicate(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        kind: FtsPredicateKind,
        attr: &str,
        query: &TokenQuery,
        last_as_prefix: bool,
    ) -> Result<RowSet> {
        let entry = schema
            .get_by_name(attr)
            .ok_or_else(|| TreeError::InvalidFilter(format!("unknown attribute '{attr}'")))?;
        let fts_cfg = entry.full_text_search.as_ref().ok_or_else(|| {
            TreeError::InvalidFilter(format!(
                "attribute '{attr}' does not have full_text_search enabled"
            ))
        })?;
        let qtokens = {
            let mut buf = self.acquire_buffer();
            FtsAnalyzer::new(fts_cfg.clone()).analyze_query(query, &mut buf)?
        };
        if qtokens.is_empty() {
            return Ok(RowSet::empty());
        }

        let snap = storage.segment_snapshot();
        let source = StorageFtsSource::new(storage);
        let prefix_expansions = if last_as_prefix {
            let prefix = &qtokens.last().unwrap().text;
            Some(source.scan_prefix_terms(entry.id, prefix, DEFAULT_MAX_PREFIX_EXPANSIONS)?)
        } else {
            None
        };

        match kind {
            FtsPredicateKind::AnyToken => eval_any_token_predicate(
                storage,
                &snap,
                &source,
                entry.id,
                &qtokens,
                prefix_expansions,
            ),
            FtsPredicateKind::AllTokens => eval_all_tokens_predicate(
                storage,
                &snap,
                &source,
                entry.id,
                &qtokens,
                prefix_expansions,
            ),
            FtsPredicateKind::TokenSequence => eval_token_sequence_predicate(
                storage,
                &snap,
                &source,
                entry.id,
                &qtokens,
                prefix_expansions,
            ),
        }
    }

    /// Exact live [`RowSet`] of documents whose BM25 score under pure-lexical `rank_by` is `> 0`.
    pub fn eval_rank_by_match_rowset(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        rank_by: &RankBy,
    ) -> Result<RowSet> {
        let snap = storage.segment_snapshot();
        let source = StorageFtsSource::new(storage);
        self.rank_by_match(storage, &snap, &source, schema, rank_by, 1.0)
    }

    fn rank_by_match(
        &self,
        storage: &Storage,
        snap: &SegmentTableSnapshot,
        source: &StorageFtsSource<'_>,
        schema: &CollectionSchema,
        expr: &RankBy,
        weight: f32,
    ) -> Result<RowSet> {
        if weight <= 0.0 {
            return Ok(RowSet::empty());
        }
        match expr {
            RankBy::Product(children) => {
                let mut w = weight;
                let mut lexical: Option<&RankBy> = None;
                for child in children {
                    match child {
                        RankBy::Const(c) => w *= *c,
                        other => {
                            if lexical.replace(other).is_some() {
                                return Err(TreeError::InvalidQuery(
                                    "rank_by_match requires a pure lexical expression".into(),
                                ));
                            }
                        }
                    }
                }
                match lexical {
                    Some(inner) => self.rank_by_match(storage, snap, source, schema, inner, w),
                    None => Err(TreeError::InvalidQuery(
                        "rank_by_match requires a pure lexical expression".into(),
                    )),
                }
            }
            RankBy::Const(_)
            | RankBy::Filter(_)
            | RankBy::Attribute { .. }
            | RankBy::Saturate { .. }
            | RankBy::Decay { .. } => Err(TreeError::InvalidQuery(
                "rank_by_match requires a pure lexical expression".into(),
            )),
            RankBy::Sum(children) | RankBy::Max(children) => {
                let mut acc = RowSet::empty();
                for child in children {
                    let sub = self.rank_by_match(storage, snap, source, schema, child, weight)?;
                    acc = acc.or(&sub);
                }
                Ok(acc)
            }
            RankBy::Bm25 {
                attr,
                query,
                params,
            } => {
                let entry = schema.get_by_name(attr).ok_or_else(|| {
                    TreeError::InvalidQuery(format!("unknown attribute '{attr}'"))
                })?;
                let fts_cfg = entry.full_text_search.as_ref().ok_or_else(|| {
                    TreeError::InvalidQuery(format!(
                        "attribute '{attr}' does not have full_text_search enabled"
                    ))
                })?;
                let qtokens = {
                    let mut buf = self.acquire_buffer();
                    FtsAnalyzer::new(fts_cfg.clone()).analyze_query(query, &mut buf)?
                };
                if qtokens.is_empty() {
                    return Ok(RowSet::empty());
                }
                let last_as_prefix = params
                    .as_ref()
                    .and_then(|p| p.last_as_prefix)
                    .unwrap_or(false);
                let max_expansions = params
                    .as_ref()
                    .and_then(|p| p.max_prefix_expansions)
                    .unwrap_or(DEFAULT_MAX_PREFIX_EXPANSIONS);

                let mut exact = exact_query_tokens(&qtokens, last_as_prefix);
                exact.sort_unstable();
                exact.dedup();
                let mut records: Vec<TermRecord> = source
                    .multi_get_term_records(entry.id, &exact)?
                    .into_iter()
                    .flatten()
                    .collect();
                if last_as_prefix {
                    let prefix = &qtokens.last().unwrap().text;
                    records.extend(
                        source
                            .scan_prefix_terms(entry.id, prefix, max_expansions)?
                            .into_iter()
                            .map(|(_, rec)| rec),
                    );
                }
                records.retain(|r| r.stats.doc_freq > 0);
                eval_any_token_records(storage, snap, &records)
            }
        }
    }

    /// Evaluate a single FTS predicate against a decoded attribute value (used for lazy filters).
    pub fn matches_fts_attr(
        &self,
        schema: &CollectionSchema,
        kind: FtsPredicateKind,
        attr: &str,
        query: &TokenQuery,
        last_as_prefix: bool,
        val_opt: Option<&AttributeValue>,
    ) -> bool {
        let Some(entry) = schema.get_by_name(attr) else {
            return false;
        };
        let Some(fts_cfg) = entry.full_text_search.as_ref() else {
            return false;
        };
        let Some(val) = val_opt else {
            return false;
        };
        let mut buf = self.acquire_buffer();
        let analyzer = FtsAnalyzer::new(fts_cfg.clone());
        let Ok(Some(doc)) = analyzer.analyze_attribute_value(val, &mut buf) else {
            return false;
        };
        let Ok(qtokens) = analyzer.analyze_query(query, &mut buf) else {
            return false;
        };
        if qtokens.is_empty() {
            return false;
        }
        match kind {
            FtsPredicateKind::AnyToken => qtokens.iter().enumerate().any(|(i, qt)| {
                if last_as_prefix && i == qtokens.len() - 1 {
                    doc.terms.keys().any(|t| t.starts_with(&qt.text))
                } else {
                    doc.terms.contains_key(&qt.text)
                }
            }),
            FtsPredicateKind::AllTokens => qtokens.iter().enumerate().all(|(i, qt)| {
                if last_as_prefix && i == qtokens.len() - 1 {
                    doc.terms.keys().any(|t| t.starts_with(&qt.text))
                } else {
                    doc.terms.contains_key(&qt.text)
                }
            }),
            FtsPredicateKind::TokenSequence => {
                let base = qtokens[0].position;
                let rel: Vec<u32> = qtokens.iter().map(|t| t.position - base).collect();
                if qtokens.len() == 1 {
                    let qt = &qtokens[0];
                    return if last_as_prefix {
                        doc.terms.keys().any(|t| t.starts_with(&qt.text))
                    } else {
                        doc.terms.contains_key(&qt.text)
                    };
                }
                let Some(first_term) = doc.terms.get(&qtokens[0].text) else {
                    return false;
                };
                'outer: for &p0 in &first_term.positions {
                    let exact_end = if last_as_prefix {
                        qtokens.len() - 1
                    } else {
                        qtokens.len()
                    };
                    for i in 1..exact_end {
                        let target = p0 + rel[i];
                        let Some(t) = doc.terms.get(&qtokens[i].text) else {
                            continue 'outer;
                        };
                        if t.positions.binary_search(&target).is_err() {
                            continue 'outer;
                        }
                    }
                    if last_as_prefix {
                        let target = p0 + rel[qtokens.len() - 1];
                        let prefix = &qtokens[qtokens.len() - 1].text;
                        let ok = doc.terms.iter().any(|(tok, term)| {
                            tok.starts_with(prefix) && term.positions.binary_search(&target).is_ok()
                        });
                        if !ok {
                            continue 'outer;
                        }
                    }
                    return true;
                }
                false
            }
        }
    }

    /// Evaluate a full [`Filter`] (supporting mixed attribute + FTS predicates) in memory
    /// against a document's attributes.
    pub fn matches_filter(
        &self,
        schema: &CollectionSchema,
        filter: &Filter,
        attrs: &Attributes,
    ) -> bool {
        filter.matches_with_fts(attrs, &mut |kind, attr, query, last_as_prefix| {
            self.matches_fts_attr(schema, kind, attr, query, last_as_prefix, attrs.get(attr))
        })
    }
}
