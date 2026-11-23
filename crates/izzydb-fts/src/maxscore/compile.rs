//! Rank expression compilation into clauses.


use alyze::analyze::ReusableBuffer;

use izzydb_core::config::{
    AttrId, TermId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rank::RankBy;
use izzydb_core::schema::CollectionSchema;

use crate::analyzer::FtsAnalyzer;
use crate::block::{FtsMeta, TermRecord};
use crate::bm25::Bm25TermScorer;

use super::*;

pub(super) fn not_lexical_err() -> TreeError {
    TreeError::InvalidQuery(
        "MAXSCORE executes only pure lexical RankBy expressions (BM25 combined with Sum / Max / \
         constant-weight Product)"
            .into(),
    )
}

pub(super) struct CompileContext<'a, 'b, S: FtsPostingSource + 'a> {
    pub(super) source: &'a S,
    pub(super) schema: &'a CollectionSchema,
    pub(super) meta: &'a FtsMeta,
    pub(super) buffer: &'b mut ReusableBuffer,
    pub(super) dl_cache: &'b mut DlNormCache<'a, S>,
}

/// Compile a `RankBy` AST into flat MAXSCORE clauses (flattening `Sum` and distributing `Product`).
pub(super) fn compile_clauses<'a, S: FtsPostingSource>(
    ctx: &mut CompileContext<'a, '_, S>,
    expr: &RankBy,
    weight: f32,
    out: &mut Vec<Clause<'a, S>>,
) -> Result<()> {
    if weight <= 0.0 {
        return Ok(());
    }
    match expr {
        RankBy::Product(children) => {
            // Callers guarantee `is_pure_lexical()`: constants plus exactly one lexical factor.
            let mut w = weight;
            let mut lexical: Option<&RankBy> = None;
            for child in children {
                match child {
                    RankBy::Const(c) => w *= *c,
                    other => {
                        if lexical.replace(other).is_some() {
                            return Err(not_lexical_err());
                        }
                    }
                }
            }
            match lexical {
                Some(inner) => compile_clauses(ctx, inner, w, out),
                None => Err(not_lexical_err()),
            }
        }
        RankBy::Const(_)
        | RankBy::Filter(_)
        | RankBy::Attribute { .. }
        | RankBy::Saturate { .. }
        | RankBy::Decay { .. } => Err(not_lexical_err()),
        RankBy::Sum(children) => {
            for child in children {
                compile_clauses(ctx, child, weight, out)?;
            }
            Ok(())
        }
        RankBy::Max(children) => {
            let mut branches = Vec::with_capacity(children.len());
            let mut max_branch_ub = 0.0f32;
            for child in children {
                let mut branch_clauses = Vec::new();
                compile_clauses(ctx, child, weight, &mut branch_clauses)?;
                if !branch_clauses.is_empty() {
                    let branch_ub: f32 = branch_clauses.iter().map(|c| c.term_max).sum();
                    max_branch_ub = max_branch_ub.max(branch_ub);
                    branches.push(branch_clauses);
                }
            }
            if !branches.is_empty() && max_branch_ub > 0.0 {
                if branches.len() == 1 {
                    out.extend(branches.into_iter().next().unwrap());
                } else {
                    out.push(Clause {
                        term_max: max_branch_ub,
                        kind: ClauseKind::MaxBranches(branches),
                    });
                }
            }
            Ok(())
        }
        RankBy::Bm25 {
            attr,
            query,
            params,
        } => {
            let entry = ctx.schema.get_by_name(attr).unwrap();
            let fts = entry.full_text_search.as_ref().unwrap();
            let Some(&attr_stats) = ctx
                .meta
                .per_attr_stats
                .get(&entry.id)
                .filter(|s| s.doc_count > 0 && s.total_tokens > 0)
            else {
                return Ok(());
            };
            let qtokens = FtsAnalyzer::new(fts.clone()).analyze_query(query, ctx.buffer)?;
            if qtokens.is_empty() {
                return Ok(());
            }

            let resolved = resolve_params(fts, params.as_ref());
            let (exact_tokens, prefix_token) = if resolved.last_as_prefix {
                let last_idx = qtokens.len() - 1;
                (&qtokens[..last_idx], Some(qtokens[last_idx].text.clone()))
            } else {
                (&qtokens[..], None)
            };

            if !exact_tokens.is_empty() {
                let mut token_refs: Vec<&str> =
                    exact_tokens.iter().map(|t| t.text.as_str()).collect();
                token_refs.sort_unstable();
                let mut keys: Vec<String> = Vec::with_capacity(token_refs.len());
                let mut qtfs: Vec<u32> = Vec::with_capacity(token_refs.len());
                for tok in token_refs {
                    if keys.last().is_some_and(|last| last == tok) {
                        *qtfs.last_mut().unwrap() += 1;
                    } else {
                        keys.push(tok.to_owned());
                        qtfs.push(1);
                    }
                }
                let records = ctx.source.multi_get_term_records(entry.id, &keys)?;
                let pairs = records.into_iter().zip(qtfs).filter_map(|(r, q)| r.map(|rec| (rec, q)));
                let streams = build_term_streams(
                    ctx.source,
                    ctx.dl_cache,
                    entry.id,
                    attr_stats,
                    weight,
                    &resolved,
                    pairs,
                )?;
                for (ub, stream) in streams {
                    out.push(Clause {
                        term_max: ub,
                        kind: ClauseKind::Single(stream),
                    });
                }
            }

            if let Some(prefix) = prefix_token {
                let expanded =
                    ctx.source
                        .scan_prefix_terms(entry.id, &prefix, resolved.max_prefix_expansions)?;
                let pairs = expanded.into_iter().map(|(_, rec)| (rec, 1));
                let streams_with_ub = build_term_streams(
                    ctx.source,
                    ctx.dl_cache,
                    entry.id,
                    attr_stats,
                    weight,
                    &resolved,
                    pairs,
                )?;
                let max_ub = streams_with_ub
                    .iter()
                    .map(|(ub, _)| *ub)
                    .fold(0.0f32, f32::max);
                let mut streams: Vec<TermStream<'a, S>> =
                    streams_with_ub.into_iter().map(|(_, s)| s).collect();
                if streams.len() == 1 {
                    out.push(Clause {
                        term_max: max_ub,
                        kind: ClauseKind::Single(streams.pop().unwrap()),
                    });
                } else if !streams.is_empty() {
                    out.push(Clause {
                        term_max: max_ub,
                        kind: ClauseKind::PrefixMax(streams),
                    });
                }
            }

            Ok(())
        }
    }
}

pub(super) fn build_term_streams<'a, S: FtsPostingSource>(
    source: &'a S,
    dl_cache: &mut DlNormCache<'a, S>,
    attr_id: AttrId,
    attr_stats: crate::block::AttrFtsStats,
    weight: f32,
    params: &ResolvedBm25Params,
    records: impl Iterator<Item = (TermRecord, u32)>,
) -> Result<Vec<(f32, TermStream<'a, S>)>> {
    let mut active_meta: Vec<(TermId, Bm25TermScorer, f32, usize)> = Vec::new();
    let mut active_recs: Vec<TermRecord> = Vec::new();
    for (rec, qtf) in records {
        if rec.stats.doc_freq == 0 {
            continue;
        }
        let scorer = Bm25TermScorer::new(
            weight,
            attr_stats.doc_count,
            attr_stats.total_tokens,
            rec.stats.doc_freq,
            qtf,
            params.k1,
            params.b,
            params.k3,
        );
        let ub = scorer.upper_bound(rec.stats.max_tf, rec.stats.min_dl_over_tf);
        if ub <= 0.0 {
            continue;
        }
        let dl_group = dl_cache.intern(attr_id, &scorer);
        active_meta.push((rec.term_id, scorer, ub, dl_group));
        active_recs.push(rec);
    }
    if active_recs.is_empty() {
        return Ok(Vec::new());
    }
    let cursors = source.open_skip_cursors(active_recs)?;
    let mut out = Vec::with_capacity(cursors.len());
    for ((term_id, scorer, ub, dl_group), cursor) in active_meta.into_iter().zip(cursors) {
        if cursor.current_block_id().is_some() {
            out.push((
                ub,
                TermStream {
                    term_id,
                    scorer,
                    dl_group,
                    cursor,
                },
            ));
        }
    }
    Ok(out)
}
