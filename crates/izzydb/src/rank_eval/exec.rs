//! `Collection` entry points: rank queries and lexical execution.

use std::sync::Arc;

use izzydb_core::config::{AttrId, RowId};
use izzydb_core::error::Result;
use izzydb_core::filter::Filter;
use izzydb_core::rank::RankBy;
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::{AttributeType, ColumnStorage, ScalarValue};
use izzydb_fts::{Bm25Result, RankModifier};
use izzydb_storage::SegmentTableSnapshot;
use izzydb_storage::codec::{RowView, decode_scalar_val_bytes, ordered_type_tag};

use crate::collection::Collection;
use crate::filter_plan::{self, FilterPlan, FilterPlanTrace};
use crate::fts_index::LivenessMaskProvider;


// ------------------------------------------------------------------------------------------
// Compiled per-row evaluation tree
// ------------------------------------------------------------------------------------------

use super::*;

impl Collection {
    /// Sound upper bound of a numeric `filterable` attribute using live counts in `default[0x06]`.
    pub(super) fn attr_max_numeric(&self, attr_id: AttrId) -> Result<Option<f32>> {
        let Some(entry) = self.schema.get_by_id(attr_id) else {
            return Ok(None);
        };
        if !entry.filterable
            || !matches!(
                entry.attr_type,
                AttributeType::Int | AttributeType::Uint | AttributeType::Float
            )
        {
            return Ok(None);
        }
        let tag = ordered_type_tag(entry.attr_type)?;
        let vbytes_opt = if entry.storage == ColumnStorage::Detached {
            let col_gen = crate::detached::current_detached_gen(&self.storage, attr_id)?;
            self.storage.max_detached_count_key(attr_id, col_gen, tag)?
        } else {
            self.storage.max_attr_count_key(attr_id, tag)?
        };
        let Some(vbytes) = vbytes_opt else {
            return Ok(Some(0.0));
        };
        let v = match decode_scalar_val_bytes(tag, &vbytes)? {
            ScalarValue::Int(v) => v as f64,
            ScalarValue::Uint(v) => v as f64,
            ScalarValue::Float(v) => v,
            _ => return Ok(None),
        };
        let ub = round_up_f32(v);
        Ok(ub.is_finite().then_some(ub))
    }

    pub(super) fn resolve_filter_sets(&self, node: &mut Node) -> Result<()> {
        const MAX_ROW_SET: u64 = 1 << 20;
        match node {
            Node::Filter(f) => {
                let mut trace = FilterPlanTrace::default();
                match filter_plan::plan_filter_with_trace(
                    &self.storage,
                    &self.schema,
                    f,
                    &self.fts_index,
                    &mut trace,
                )? {
                    FilterPlan::Empty => *node = Node::Const(0.0),
                    FilterPlan::All => *node = Node::Const(1.0),
                    FilterPlan::Materialized(rs) if rs.len() <= MAX_ROW_SET => {
                        *node = Node::RowSet(Arc::new(rs));
                    }
                    _ => {}
                }
            }
            Node::Sum(c) | Node::Max(c) | Node::Product(c) => {
                for n in c {
                    self.resolve_filter_sets(n)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Execute any validated [`RankBy`] with an optional filter; exact top-k by
    /// `score DESC, DocId ASC`.
    pub(crate) fn rank_query(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<Bm25Result>> {
        rank_by.validate(&self.schema)?;
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        if top_k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let norm = rank_by.normalize();
        if matches!(norm, RankBy::Const(c) if c <= 0.0) {
            return Ok(Vec::new());
        }
        match classify(&norm) {
            Plan::Lexical => self.lexical_query(&norm, filter, top_k, None),
            Plan::Rescored { lex, mult, add } => {
                self.rescored_query(&norm, lex, mult, add, filter, top_k)
            }
            Plan::General => {
                if !norm.has_bm25()
                    && let Some(res) = self.monotone_rank_query(&norm, filter, top_k)?
                {
                    return Ok(res);
                }
                let mut topk = TopK::new(top_k);
                self.general_eval(&norm, filter, &mut |_row, doc_id, s| topk.push(doc_id, s))?;
                Ok(topk.into_sorted())
            }
        }
    }

    /// Exact [`RowSet`] of all live rows with score `> 0` that match `filter` (used for faceting).
    pub(crate) fn rank_matching_rowset(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
    ) -> Result<RowSet> {
        rank_by.validate(&self.schema)?;
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        if self.is_empty() {
            return Ok(RowSet::empty());
        }
        let norm = rank_by.normalize();
        if norm.is_pure_lexical() {
            let lex_rs = self.fts_index.eval_rank_by_match_rowset(
                &self.storage,
                &self.schema,
                &norm,
            )?;
            if lex_rs.is_empty() {
                return Ok(RowSet::empty());
            }
            return match filter {
                Some(f) => {
                    let f_rs = filter_plan::materialize_filter(
                        &self.storage,
                        &self.schema,
                        f,
                        &self.fts_index,
                    )?;
                    Ok(lex_rs.and(&f_rs))
                }
                None => Ok(lex_rs),
            };
        }
        let mut rows = Vec::new();
        self.general_eval(&norm, filter, &mut |row, _doc_id, _s| rows.push(row))?;
        Ok(RowSet::from_rows(rows))
    }

    /// Run a pure lexical `rank_by` through MAXSCORE using `LivenessMaskProvider` + optional
    /// [`FilterPlan`] pushdown and [`LexModifier`].
    pub(crate) fn lexical_query(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
        modifier: Option<&LexModifier>,
    ) -> Result<Vec<Bm25Result>> {
        let snap = self.storage.segment_snapshot();
        match filter {
            None => self.run_lexical(rank_by, top_k, &snap, None, None, modifier),
            Some(f) => {
                let mut trace = FilterPlanTrace::default();
                match filter_plan::plan_filter_with_trace(
                    &self.storage,
                    &self.schema,
                    f,
                    &self.fts_index,
                    &mut trace,
                )? {
                    FilterPlan::Empty => Ok(Vec::new()),
                    FilterPlan::All => self.run_lexical(rank_by, top_k, &snap, None, None, modifier),
                    FilterPlan::Materialized(rs) => {
                        if rs.is_empty() {
                            return Ok(Vec::new());
                        }
                        self.run_lexical(rank_by, top_k, &snap, Some(&rs), None, modifier)
                    }
                    FilterPlan::Lazy { residual, seed } => self.run_lexical(
                        rank_by,
                        top_k,
                        &snap,
                        seed.as_ref(),
                        Some(&residual),
                        modifier,
                    ),
                }
            }
        }
    }

    pub(super) fn run_lexical(
        &self,
        rank_by: &RankBy,
        top_k: usize,
        snap: &SegmentTableSnapshot,
        filter_rs: Option<&RowSet>,
        post_filter: Option<&Filter>,
        modifier: Option<&LexModifier>,
    ) -> Result<Vec<Bm25Result>> {
        let mask = LivenessMaskProvider {
            snap,
            filter: filter_rs,
        };
        let Some(m) = modifier else {
            return match post_filter {
                None => self.fts_index.query_bm25(
                    &self.storage,
                    &self.schema,
                    rank_by,
                    top_k,
                    Some(&mask),
                    None,
                ),
                Some(f) => {
                    let mut lazy = |row: RowId| -> Result<bool> {
                        let Some(slice) = self.storage.get_row_bytes_pinned(row)? else {
                            return Ok(false);
                        };
                        let view = RowView::parse(&slice)?;
                        RowCtx::new(self, row, view).filter_matches(f)
                    };
                    self.fts_index.query_bm25(
                        &self.storage,
                        &self.schema,
                        rank_by,
                        top_k,
                        Some(&mask),
                        Some(&mut lazy),
                    )
                }
            };
        };

        let mut rescore = |row: RowId, s_ub: f32| -> Result<f32> {
            let Some(slice) = self.storage.get_row_bytes_pinned(row)? else {
                return Ok(0.0);
            };
            let view = RowView::parse(&slice)?;
            let ctx = RowCtx::new(self, row, view);
            if let Some(f) = post_filter
                && !ctx.filter_matches(f)?
            {
                return Ok(0.0);
            }
            let mv = match &m.mult {
                Some(n) => n.eval(&ctx, &[])?,
                None => m.mult_ub,
            };
            let av = match &m.add {
                Some(n) => n.eval(&ctx, &[])?,
                None => 0.0,
            };
            let lexical = if mv == m.mult_ub {
                s_ub
            } else {
                s_ub * (mv / m.mult_ub)
            };
            let total = lexical + av;
            Ok(if total.is_nan() {
                0.0
            } else {
                total.min(f32::MAX)
            })
        };
        self.fts_index.query_bm25_with(
            &self.storage,
            &self.schema,
            rank_by,
            top_k,
            Some(&mask),
            None,
            Some(RankModifier {
                mult_ub: m.mult_ub,
                add_ub: m.add_ub,
                rescore: &mut rescore,
            }),
        )
    }

    pub(super) fn rescored_query(
        &self,
        norm: &RankBy,
        lex: RankBy,
        mult: Vec<RankBy>,
        add: Vec<RankBy>,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Vec<Bm25Result>> {
        let mut no_lex = Vec::new();
        let mut mult_node = (!mult.is_empty())
            .then(|| Node::compile(&RankBy::Product(mult).normalize(), &self.schema, &mut no_lex));
        let add_expr = (!add.is_empty()).then(|| RankBy::Sum(add).normalize());
        let mut add_node = add_expr
            .as_ref()
            .map(|e| Node::compile(e, &self.schema, &mut no_lex));
        if let Some(n) = mult_node.as_mut() {
            self.resolve_filter_sets(n)?;
        }
        if let Some(n) = add_node.as_mut() {
            self.resolve_filter_sets(n)?;
        }
        let mult_ub = match &mult_node {
            Some(n) => n.upper_bound(self)?,
            None => 1.0,
        };
        let add_ub = match &add_node {
            Some(n) => n.upper_bound(self)?,
            None => 0.0,
        };
        if !(mult_ub.is_finite() && mult_ub > 0.0 && add_ub.is_finite()) {
            let mut topk = TopK::new(top_k);
            self.general_eval(norm, filter, &mut |_row, doc_id, s| topk.push(doc_id, s))?;
            return Ok(topk.into_sorted());
        }
        let modifier = LexModifier {
            mult: mult_node,
            add: add_node,
            mult_ub,
            add_ub,
        };
        let mut results = self.lexical_query(&lex, filter, top_k, Some(&modifier))?;

        if let Some(add_node) = modifier.add.as_ref()
            && add_ub > 0.0
        {
            let covered =
                results.len() == top_k && results.last().is_some_and(|r| r.score > add_ub);
            if !covered {
                let lex_rs =
                    self.fts_index
                        .eval_rank_by_match_rowset(&self.storage, &self.schema, &lex)?;
                let support = add_expr.as_ref().and_then(positive_support);
                let snap = self.storage.segment_snapshot();
                let cand_rs = match (support, filter) {
                    (Some(sup), Some(f)) => filter_plan::materialize_filter(
                        &self.storage,
                        &self.schema,
                        &Filter::And(vec![f.clone(), sup]),
                        &self.fts_index,
                    )?,
                    (Some(sup), None) => filter_plan::materialize_filter(
                        &self.storage,
                        &self.schema,
                        &sup,
                        &self.fts_index,
                    )?,
                    (None, Some(f)) => filter_plan::materialize_filter(
                        &self.storage,
                        &self.schema,
                        f,
                        &self.fts_index,
                    )?,
                    (None, None) => RowSet::from_all_live(&*snap),
                }
                .and_not(&lex_rs);

                let mut topk = TopK::new(top_k);
                self.scan_rowset(add_node, &cand_rs, &[], &mut |_row, doc_id, s| {
                    topk.push(doc_id, s)
                })?;
                results.extend(topk.into_sorted());
                sort_results(&mut results);
                results.truncate(top_k);
            }
        }
        Ok(results)
    }
}
