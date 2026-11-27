//! Rank expression compilation into reusable query plans and segment-sharded clauses.

use std::collections::HashSet;

use alyze::analyze::ReusableBuffer;

use izzydb_core::config::{AttrId, SegmentId};
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

/// Cursor-free compiled term entry produced once per query and instantiated per segment shard.
///
/// Captures the resolved [`TermRecord`], precomputed [`Bm25TermScorer`], and interned
/// doc-length group index so each worker shard opens its own cursors without repeating lookups.
///
/// ```text
/// CompiledTerm { record: rec, scorer: Bm25TermScorer { .. }, dl_group: 0 }
/// ```
#[derive(Clone)]
pub(super) struct CompiledTerm {
    /// Term dictionary record (`term_id`, stats, optional inline block).
    pub(super) record: TermRecord,
    /// Precomputed BM25 term scorer (`num_scale`, `k1_one_minus_b`, `k1_b_over_avgdl`).
    pub(super) scorer: Bm25TermScorer,
    /// Index into `CompiledPlan::dl_specs` for this term's attribute and `(k1, b)`.
    pub(super) dl_group: usize,
}

pub(super) enum CompiledClauseKind {
    Single(CompiledTerm),
    PrefixMax(Vec<CompiledTerm>),
    MaxBranches(Vec<Vec<CompiledClause>>),
}

/// Cursor-free compiled MAXSCORE clause with its global upper bound.
///
/// Represents a single term, a prefix-expanded `Max` over terms, or nested `Max` branches
/// prior to opening segment-scoped skip cursors.
///
/// ```text
/// CompiledClause { term_max: 4.12, kind: CompiledClauseKind::Single(term) }
/// ```
pub(super) struct CompiledClause {
    /// Global BM25 upper bound across all blocks for this clause.
    pub(super) term_max: f32,
    /// Structural clause kind (`Single`, `PrefixMax`, or `MaxBranches`).
    pub(super) kind: CompiledClauseKind,
}

/// Pre-resolved query plan (`CompiledClause`s and doc-length group specs) ready to instantiate
/// cursors globally or over disjoint segment ranges.
///
/// Built once per query so parallel shards share term dictionary lookups and doc-length prewarming.
///
/// ```text
/// let plan = compile_plan(source, schema, &meta, &rank_by, mult_ub, buffer)?;
/// ```
pub(super) struct CompiledPlan {
    /// Compiled clauses sorted ascending by global `term_max`.
    pub(super) clauses: Vec<CompiledClause>,
    /// Interned `(attr_id, c1, c2)` doc-length normalization specs.
    pub(super) dl_specs: Vec<DlGroupSpec>,
}

struct CompileContext<'a, 'b, S: FtsPostingSource + 'a> {
    source: &'a S,
    schema: &'a CollectionSchema,
    meta: &'a FtsMeta,
    buffer: &'b mut ReusableBuffer,
    dl_specs: &'b mut Vec<DlGroupSpec>,
}

pub(super) fn compile_plan<S: FtsPostingSource>(
    source: &S,
    schema: &CollectionSchema,
    meta: &FtsMeta,
    rank_by: &RankBy,
    weight: f32,
    buffer: &mut ReusableBuffer,
) -> Result<CompiledPlan> {
    let mut dl_specs = Vec::new();
    let mut clauses = Vec::new();
    {
        let mut ctx = CompileContext {
            source,
            schema,
            meta,
            buffer,
            dl_specs: &mut dl_specs,
        };
        compile_clauses(&mut ctx, rank_by, weight, &mut clauses)?;
    }
    clauses.sort_by(|a, b| a.term_max.total_cmp(&b.term_max));
    Ok(CompiledPlan { clauses, dl_specs })
}

impl CompiledPlan {
    /// Pre-warm doc-length block tables on `source` before fanning out to parallel workers.
    pub(super) fn prewarm_doc_lens<S: FtsPostingSource>(&self, source: &S) -> Result<()> {
        let mut seen: HashSet<AttrId> = HashSet::new();
        for spec in &self.dl_specs {
            if seen.insert(spec.attr_id) {
                let _ = source.open_doc_len_cursor(spec.attr_id)?;
            }
        }
        Ok(())
    }

    /// Instantiate live cursors over either the whole corpus (`range = None`) or `[seg_start, seg_end)`.
    pub(super) fn instantiate<'a, S: FtsPostingSource>(
        &self,
        source: &'a S,
        range: Option<(SegmentId, SegmentId)>,
    ) -> Result<(Vec<Clause<'a, S>>, Vec<f32>)> {
        let mut records = Vec::new();
        collect_records(&self.clauses, &mut records);
        if records.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let cursors = match range {
            Some((seg_start, seg_end)) => {
                source.open_skip_cursors_in_range(&records, seg_start, seg_end)?
            }
            None => source.open_skip_cursors(records)?,
        };
        let mut cursor_iter = cursors.into_iter();
        let mut clauses = Vec::with_capacity(self.clauses.len());
        for cc in &self.clauses {
            if let Some(c) = build_clause(cc, &mut cursor_iter) {
                clauses.push(c);
            }
        }
        let mut prefix_max_sum = Vec::with_capacity(clauses.len());
        let mut acc = 0.0f64;
        for c in &clauses {
            acc += c.term_max as f64;
            prefix_max_sum.push(acc as f32);
        }
        Ok((clauses, prefix_max_sum))
    }
}

fn collect_records(clauses: &[CompiledClause], out: &mut Vec<TermRecord>) {
    for c in clauses {
        match &c.kind {
            CompiledClauseKind::Single(t) => out.push(t.record.clone()),
            CompiledClauseKind::PrefixMax(ts) => {
                for t in ts {
                    out.push(t.record.clone());
                }
            }
            CompiledClauseKind::MaxBranches(branches) => {
                for b in branches {
                    collect_records(b, out);
                }
            }
        }
    }
}

fn build_clause<'a, S: FtsPostingSource + 'a>(
    cc: &CompiledClause,
    cursors: &mut impl Iterator<Item = S::SkipCursor<'a>>,
) -> Option<Clause<'a, S>> {
    match &cc.kind {
        CompiledClauseKind::Single(t) => {
            let cursor = cursors.next().expect("cursor count mismatch");
            cursor.current_block_id()?;
            Some(Clause {
                term_max: cc.term_max,
                kind: ClauseKind::Single(TermStream {
                    term_id: t.record.term_id,
                    scorer: t.scorer,
                    dl_group: t.dl_group,
                    cursor,
                }),
            })
        }
        CompiledClauseKind::PrefixMax(ts) => {
            let mut streams = Vec::with_capacity(ts.len());
            for t in ts {
                let cursor = cursors.next().expect("cursor count mismatch");
                if cursor.current_block_id().is_some() {
                    streams.push(TermStream {
                        term_id: t.record.term_id,
                        scorer: t.scorer,
                        dl_group: t.dl_group,
                        cursor,
                    });
                }
            }
            if streams.is_empty() {
                None
            } else if streams.len() == 1 {
                Some(Clause {
                    term_max: cc.term_max,
                    kind: ClauseKind::Single(streams.pop().unwrap()),
                })
            } else {
                Some(Clause {
                    term_max: cc.term_max,
                    kind: ClauseKind::PrefixMax(streams),
                })
            }
        }
        CompiledClauseKind::MaxBranches(branches) => {
            let mut built_branches = Vec::with_capacity(branches.len());
            for branch in branches {
                let mut b = Vec::with_capacity(branch.len());
                for sub in branch {
                    if let Some(c) = build_clause(sub, cursors) {
                        b.push(c);
                    }
                }
                if !b.is_empty() {
                    built_branches.push(b);
                }
            }
            if built_branches.is_empty() {
                None
            } else {
                Some(Clause {
                    term_max: cc.term_max,
                    kind: ClauseKind::MaxBranches(built_branches),
                })
            }
        }
    }
}

/// Compile a `RankBy` AST into cursor-free `CompiledClause`s (flattening `Sum` and distributing `Product`).
fn compile_clauses<S: FtsPostingSource>(
    ctx: &mut CompileContext<'_, '_, S>,
    expr: &RankBy,
    weight: f32,
    out: &mut Vec<CompiledClause>,
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
                    out.push(CompiledClause {
                        term_max: max_branch_ub,
                        kind: CompiledClauseKind::MaxBranches(branches),
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
                let pairs = records
                    .into_iter()
                    .zip(qtfs)
                    .filter_map(|(r, q)| r.map(|rec| (rec, q)));
                let terms = build_compiled_terms(
                    ctx.dl_specs,
                    entry.id,
                    attr_stats,
                    weight,
                    &resolved,
                    pairs,
                );
                for (ub, term) in terms {
                    out.push(CompiledClause {
                        term_max: ub,
                        kind: CompiledClauseKind::Single(term),
                    });
                }
            }

            if let Some(prefix) = prefix_token {
                let expanded =
                    ctx.source
                        .scan_prefix_terms(entry.id, &prefix, resolved.max_prefix_expansions)?;
                let pairs = expanded.into_iter().map(|(_, rec)| (rec, 1));
                let terms_with_ub = build_compiled_terms(
                    ctx.dl_specs,
                    entry.id,
                    attr_stats,
                    weight,
                    &resolved,
                    pairs,
                );
                let max_ub = terms_with_ub
                    .iter()
                    .map(|(ub, _)| *ub)
                    .fold(0.0f32, f32::max);
                let mut terms: Vec<CompiledTerm> =
                    terms_with_ub.into_iter().map(|(_, t)| t).collect();
                if terms.len() == 1 {
                    out.push(CompiledClause {
                        term_max: max_ub,
                        kind: CompiledClauseKind::Single(terms.pop().unwrap()),
                    });
                } else if !terms.is_empty() {
                    out.push(CompiledClause {
                        term_max: max_ub,
                        kind: CompiledClauseKind::PrefixMax(terms),
                    });
                }
            }

            Ok(())
        }
    }
}

fn build_compiled_terms(
    dl_specs: &mut Vec<DlGroupSpec>,
    attr_id: AttrId,
    attr_stats: crate::block::AttrFtsStats,
    weight: f32,
    params: &ResolvedBm25Params,
    records: impl Iterator<Item = (TermRecord, u32)>,
) -> Vec<(f32, CompiledTerm)> {
    let mut out = Vec::new();
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
        let dl_group = intern_dl_group(dl_specs, attr_id, &scorer);
        out.push((
            ub,
            CompiledTerm {
                record: rec,
                scorer,
                dl_group,
            },
        ));
    }
    out
}
