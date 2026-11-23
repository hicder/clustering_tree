//! Generalized [`RankBy`] execution on v3 `RowId` / `RowSet` / [`FilterPlan`].
//!
//! | normalized shape                                    | strategy                                              |
//! |-----------------------------------------------------|-------------------------------------------------------|
//! | pure lexical                                        | MAXSCORE with `LivenessMaskProvider`                  |
//! | `L * m(row) + a(row)` (one lexical factor + mods)   | MAXSCORE on `L * UB(m)` + exact `RowView` rescoring   |
//! | no BM25 (`Filter` / `Attribute` / `Saturate` / ...) | index-order `stream_value_groups` if monotone, else `RowSet` scan |
//! | anything else (e.g. `BM25 * BM25`)                  | exact evaluation over the positive `RowSet` universe  |
//!
//! Every path returns the exact top-k under `score DESC, DocId ASC`; scores `<= 0` are excluded.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use izzydb_core::config::{AttrId, DocId, ROW_BLOCK_SIZE, RowId, SegmentId, for_each_set_bit_256};
use izzydb_core::error::Result;
use izzydb_core::filter::Filter;
use izzydb_core::rank::{DecayMode, OrderBySpec, RankBy, eval_decay, eval_saturate};
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::{AttributeType, CollectionSchema, ColumnStorage, ScalarValue};
use izzydb_core::simd::{batch_eval_decay, batch_eval_saturate};
use izzydb_fts::{Bm25Result, RankModifier};
use izzydb_storage::SegmentTableSnapshot;
use izzydb_storage::codec::{RowView, decode_scalar_val_bytes, ordered_type_tag};
use izzydb_storage::column::ColumnBlockView;

use crate::attr_reader::{AttrReader, ColumnReader, RowStoreReader};
use crate::collection::Collection;
use crate::filter_plan::{self, FilterPlan, FilterPlanTrace};
use crate::fts_index::LivenessMaskProvider;
use crate::order_by::OrderPopulation;
use crate::segments::RowResolver;

const SCAN_BATCH: usize = 1024;

// ------------------------------------------------------------------------------------------
// Compiled per-row evaluation tree
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) enum Node {
    Const(f32),
    Lex(usize),
    Filter(Filter),
    RowSet(Arc<RowSet>),
    Attr {
        id: AttrId,
        missing: f32,
    },
    Sat {
        id: AttrId,
        midpoint: f64,
        exponent: f64,
        missing: f32,
    },
    Decay {
        id: AttrId,
        origin: f64,
        midpoint: f64,
        exponent: f64,
        mode: DecayMode,
        missing: f32,
    },
    Sum(Vec<Node>),
    Max(Vec<Node>),
    Product(Vec<Node>),
}

impl Node {
    pub(crate) fn compile(expr: &RankBy, schema: &CollectionSchema, lex: &mut Vec<RankBy>) -> Node {
        if expr.is_pure_lexical() {
            lex.push(expr.clone());
            return Node::Lex(lex.len() - 1);
        }
        let id_of = |attr: &str| schema.get_by_name(attr).map(|e| e.id).unwrap_or_default();
        match expr {
            RankBy::Const(c) => Node::Const(*c),
            RankBy::Bm25 { .. } => {
                lex.push(expr.clone());
                Node::Lex(lex.len() - 1)
            }
            RankBy::Filter(f) => Node::Filter((**f).clone()),
            RankBy::Attribute { attr, missing } => Node::Attr {
                id: id_of(attr),
                missing: *missing,
            },
            RankBy::Saturate {
                attr,
                midpoint,
                exponent,
                missing,
            } => Node::Sat {
                id: id_of(attr),
                midpoint: *midpoint,
                exponent: *exponent,
                missing: *missing,
            },
            RankBy::Decay {
                attr,
                origin,
                midpoint,
                exponent,
                mode,
                missing,
            } => Node::Decay {
                id: id_of(attr),
                origin: *origin,
                midpoint: *midpoint,
                exponent: *exponent,
                mode: *mode,
                missing: *missing,
            },
            RankBy::Sum(c) => Node::Sum(c.iter().map(|e| Self::compile(e, schema, lex)).collect()),
            RankBy::Max(c) => Node::Max(c.iter().map(|e| Self::compile(e, schema, lex)).collect()),
            RankBy::Product(c) => {
                Node::Product(c.iter().map(|e| Self::compile(e, schema, lex)).collect())
            }
        }
    }

    /// Score of this node for one row. `lex` holds the BM25 score of each `Lex` slot.
    pub(crate) fn eval(&self, ctx: &RowCtx<'_, '_>, lex: &[f32]) -> Result<f32> {
        Ok(match self {
            Node::Const(c) => *c,
            Node::Lex(i) => lex.get(*i).copied().unwrap_or(0.0),
            Node::Filter(f) => {
                if ctx.filter_matches(f)? {
                    1.0
                } else {
                    0.0
                }
            }
            Node::RowSet(set) => {
                if set.contains(ctx.row_id) {
                    1.0
                } else {
                    0.0
                }
            }
            Node::Attr { id, missing } => match ctx.get_numeric_f64(*id)? {
                Some(v) => clamp_f32(v),
                None => *missing,
            },
            Node::Sat {
                id,
                midpoint,
                exponent,
                missing,
            } => match ctx.get_numeric_f64(*id)? {
                Some(v) => eval_saturate(v, *midpoint, *exponent),
                None => *missing,
            },
            Node::Decay {
                id,
                origin,
                midpoint,
                exponent,
                mode,
                missing,
            } => match ctx.get_numeric_f64(*id)? {
                Some(v) => eval_decay(v, *origin, *midpoint, *exponent, *mode),
                None => *missing,
            },
            Node::Sum(c) => {
                let mut acc = 0.0f32;
                for n in c {
                    acc += n.eval(ctx, lex)?;
                }
                acc
            }
            Node::Max(c) => {
                let mut acc = 0.0f32;
                for n in c {
                    acc = acc.max(n.eval(ctx, lex)?);
                }
                acc
            }
            Node::Product(c) => {
                let mut acc = 1.0f32;
                for n in c {
                    acc *= n.eval(ctx, lex)?;
                    if acc == 0.0 {
                        break;
                    }
                }
                acc
            }
        })
    }

    /// Sound upper bound of this node over all live rows; `f32::INFINITY` if none is known.
    pub(crate) fn upper_bound(&self, coll: &Collection) -> Result<f32> {
        Ok(match self {
            Node::Const(c) => *c,
            Node::Lex(_) => f32::INFINITY,
            Node::Filter(_) | Node::RowSet(_) => 1.0,
            Node::Sat { missing, .. } | Node::Decay { missing, .. } => missing.max(1.0),
            Node::Attr { id, missing } => match coll.attr_max_numeric(*id)? {
                Some(m) => m.max(*missing),
                None => f32::INFINITY,
            },
            Node::Sum(c) => {
                let mut acc = 0.0f32;
                for n in c {
                    acc += n.upper_bound(coll)?;
                }
                acc
            }
            Node::Max(c) => {
                let mut acc = 0.0f32;
                for n in c {
                    acc = acc.max(n.upper_bound(coll)?);
                }
                acc
            }
            Node::Product(c) => {
                let mut acc = 1.0f32;
                for n in c {
                    let ub = n.upper_bound(coll)?;
                    if ub == 0.0 {
                        return Ok(0.0);
                    }
                    acc *= ub;
                }
                acc
            }
        })
    }

    fn has_filter_node(&self) -> bool {
        match self {
            Node::Filter(_) => true,
            Node::Sum(c) | Node::Max(c) | Node::Product(c) => {
                c.iter().any(Node::has_filter_node)
            }
            _ => false,
        }
    }

    fn collect_numeric_attrs(&self, out: &mut Vec<AttrId>) {
        match self {
            Node::Attr { id, .. } | Node::Sat { id, .. } | Node::Decay { id, .. } => {
                if !out.contains(id) {
                    out.push(*id);
                }
            }
            Node::Sum(c) | Node::Max(c) | Node::Product(c) => {
                for n in c {
                    n.collect_numeric_attrs(out);
                }
            }
            _ => {}
        }
    }

    fn eval_from_cols(
        &self,
        row_idx: usize,
        row_id: RowId,
        attrs: &[AttrId],
        cols: &[Vec<Option<f64>>],
        lex: &[f32],
    ) -> f32 {
        let lookup = |id: AttrId| -> Option<f64> {
            let pos = attrs.iter().position(|&a| a == id)?;
            cols[pos][row_idx]
        };
        match self {
            Node::Const(c) => *c,
            Node::Lex(i) => lex.get(*i).copied().unwrap_or(0.0),
            Node::Filter(_) => unreachable!("eval_from_cols called with Filter node"),
            Node::RowSet(set) => {
                if set.contains(row_id) {
                    1.0
                } else {
                    0.0
                }
            }
            Node::Attr { id, missing } => match lookup(*id) {
                Some(v) => clamp_f32(v),
                None => *missing,
            },
            Node::Sat {
                id,
                midpoint,
                exponent,
                missing,
            } => match lookup(*id) {
                Some(v) => eval_saturate(v, *midpoint, *exponent),
                None => *missing,
            },
            Node::Decay {
                id,
                origin,
                midpoint,
                exponent,
                mode,
                missing,
            } => match lookup(*id) {
                Some(v) => eval_decay(v, *origin, *midpoint, *exponent, *mode),
                None => *missing,
            },
            Node::Sum(c) => c
                .iter()
                .map(|n| n.eval_from_cols(row_idx, row_id, attrs, cols, lex))
                .sum(),
            Node::Max(c) => c
                .iter()
                .map(|n| n.eval_from_cols(row_idx, row_id, attrs, cols, lex))
                .fold(0.0f32, f32::max),
            Node::Product(c) => {
                let mut acc = 1.0f32;
                for n in c {
                    acc *= n.eval_from_cols(row_idx, row_id, attrs, cols, lex);
                    if acc == 0.0 {
                        break;
                    }
                }
                acc
            }
        }
    }

    /// Superset [`RowSet`] of rows with score `> 0`, or `None` when not restricted to lexical matches.
    fn positive_universe(&self, lex_sets: &[RowSet]) -> Option<RowSet> {
        match self {
            Node::Const(c) => (*c <= 0.0).then(RowSet::empty),
            Node::Lex(i) => Some(lex_sets[*i].clone()),
            Node::Filter(_)
            | Node::RowSet(_)
            | Node::Attr { .. }
            | Node::Sat { .. }
            | Node::Decay { .. } => None,
            Node::Sum(c) | Node::Max(c) => {
                let mut acc = RowSet::empty();
                for n in c {
                    let sub = n.positive_universe(lex_sets)?;
                    acc = acc.or(&sub);
                }
                Some(acc)
            }
            Node::Product(c) => {
                let mut acc: Option<RowSet> = None;
                for n in c {
                    if let Some(rs) = n.positive_universe(lex_sets) {
                        acc = Some(match acc {
                            Some(prev) => prev.and(&rs),
                            None => rs,
                        });
                    }
                }
                acc
            }
        }
    }
}

#[inline]
fn clamp_f32(v: f64) -> f32 {
    if v.is_nan() || v <= 0.0 {
        0.0
    } else {
        v.min(f32::MAX as f64) as f32
    }
}

/// Smallest `f32` that is `>= v` (for non-negative `v`).
fn round_up_f32(v: f64) -> f32 {
    let f = clamp_f32(v);
    if (f as f64) < v && f.is_finite() {
        f.next_up()
    } else {
        f
    }
}

/// Per-row evaluation context wrapping zero-copy [`RowView`].
pub(crate) struct RowCtx<'c, 'b> {
    coll: &'c Collection,
    row_id: RowId,
    pub(crate) view: RowView<'b>,
}

impl<'c, 'b> RowCtx<'c, 'b> {
    #[inline]
    pub(crate) fn new(coll: &'c Collection, row_id: RowId, view: RowView<'b>) -> Self {
        Self { coll, row_id, view }
    }

    pub(crate) fn filter_matches(&self, f: &Filter) -> Result<bool> {
        if crate::detached::filter_mentions_detached(&self.coll.schema, f) {
            return Ok(crate::detached::filter_rows_mixed_batch(
                &self.coll.storage,
                &self.coll.schema,
                &self.coll.fts_index,
                f,
                &[self.row_id],
            )?[0]);
        }
        self.view.matches_filter(
            &self.coll.schema,
            f,
            &mut |kind, attr, query, last_as_prefix, val| {
                self.coll.fts_index.matches_fts_attr(
                    &self.coll.schema,
                    kind,
                    attr,
                    query,
                    last_as_prefix,
                    val,
                )
            },
        )
    }

    pub(crate) fn get_numeric_f64(&self, id: AttrId) -> Result<Option<f64>> {
        if self
            .coll
            .schema
            .get_by_id(id)
            .is_some_and(|e| e.storage == ColumnStorage::Detached)
        {
            let col_gen = crate::detached::current_detached_gen(&self.coll.storage, id)?;
            let Some(slice) = self.coll.storage.get_detached_block_pinned(
                id,
                col_gen,
                self.row_id.seg(),
                self.row_id.block_idx(),
            )?
            else {
                return Ok(None);
            };
            let blk = ColumnBlockView::parse(&slice)?;
            return Ok(blk.get_numeric_f64(self.row_id.slot()));
        }
        self.view.get_numeric_f64(id)
    }
}

// ------------------------------------------------------------------------------------------
// Top-k collection (`score DESC, DocId ASC`)
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Ranked {
    score: f32,
    id: DocId,
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    /// "Greater" means *worse* (lower score, then higher id) so the max-heap root is evicted first.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

struct TopK {
    k: usize,
    heap: BinaryHeap<Ranked>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.min(4096) + 1),
        }
    }

    fn push(&mut self, id: DocId, score: f32) {
        let r = Ranked { score, id };
        if self.heap.len() < self.k {
            self.heap.push(r);
        } else if let Some(worst) = self.heap.peek()
            && r < *worst
        {
            self.heap.pop();
            self.heap.push(r);
        }
    }

    fn into_sorted(self) -> Vec<Bm25Result> {
        let mut v = self.heap.into_vec();
        v.sort_unstable();
        v.into_iter()
            .map(|r| Bm25Result {
                id: r.id,
                score: r.score,
            })
            .collect()
    }
}

fn sort_results(v: &mut [Bm25Result]) {
    v.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
}

// ------------------------------------------------------------------------------------------
// Planning
// ------------------------------------------------------------------------------------------

enum Plan {
    Lexical,
    Rescored {
        lex: RankBy,
        mult: Vec<RankBy>,
        add: Vec<RankBy>,
    },
    General,
}

fn classify(norm: &RankBy) -> Plan {
    if norm.is_pure_lexical() {
        return Plan::Lexical;
    }
    if !norm.has_bm25() {
        return Plan::General;
    }
    match norm {
        RankBy::Product(children) => {
            let (lex, rest): (Vec<&RankBy>, Vec<&RankBy>) =
                children.iter().partition(|c| c.has_bm25());
            if lex.len() == 1 && lex[0].is_pure_lexical() {
                Plan::Rescored {
                    lex: lex[0].clone(),
                    mult: rest.into_iter().cloned().collect(),
                    add: Vec::new(),
                }
            } else {
                Plan::General
            }
        }
        RankBy::Sum(children) => {
            let (lex, rest): (Vec<&RankBy>, Vec<&RankBy>) =
                children.iter().partition(|c| c.has_bm25());
            if !lex.is_empty() && lex.iter().all(|c| c.is_pure_lexical()) {
                let lex_expr = if lex.len() == 1 {
                    lex[0].clone()
                } else {
                    RankBy::Sum(lex.into_iter().cloned().collect())
                };
                Plan::Rescored {
                    lex: lex_expr,
                    mult: Vec::new(),
                    add: rest.into_iter().cloned().collect(),
                }
            } else {
                Plan::General
            }
        }
        _ => Plan::General,
    }
}

fn positive_support(expr: &RankBy) -> Option<Filter> {
    match expr {
        RankBy::Filter(f) => Some((**f).clone()),
        RankBy::Product(children) => {
            let parts: Vec<Filter> = children
                .iter()
                .filter(|c| !matches!(c, RankBy::Const(_)))
                .filter_map(positive_support)
                .collect();
            match parts.len() {
                0 => None,
                1 => parts.into_iter().next(),
                _ => Some(Filter::And(parts)),
            }
        }
        RankBy::Sum(children) | RankBy::Max(children) => {
            let parts: Option<Vec<Filter>> = children.iter().map(positive_support).collect();
            parts.map(|p| {
                if p.len() == 1 {
                    p.into_iter().next().unwrap()
                } else {
                    Filter::Or(p)
                }
            })
        }
        _ => None,
    }
}

pub(crate) struct LexModifier {
    mult: Option<Node>,
    add: Option<Node>,
    mult_ub: f32,
    add_ub: f32,
}

// ------------------------------------------------------------------------------------------
// Collection integration
// ------------------------------------------------------------------------------------------

impl Collection {
    /// Sound upper bound of a numeric `filterable` attribute using live counts in `default[0x06]`.
    fn attr_max_numeric(&self, attr_id: AttrId) -> Result<Option<f32>> {
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

    fn resolve_filter_sets(&self, node: &mut Node) -> Result<()> {
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

    fn run_lexical(
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

    fn rescored_query(
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

    fn scan_rowset(
        &self,
        node: &Node,
        rowset: &RowSet,
        lex_maps: &[HashMap<DocId, f32>],
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        if rowset.is_empty() {
            return Ok(());
        }
        let col_reader = ColumnReader::new(&self.storage, &self.schema)?;
        let row_reader = RowStoreReader::new(&self.storage);
        let mut lex_scores = vec![0.0f32; lex_maps.len()];
        let mut batch: Vec<RowId> = Vec::with_capacity(SCAN_BATCH);

        let no_filter = !node.has_filter_node();
        let mut ref_attrs = Vec::new();
        if no_filter {
            node.collect_numeric_attrs(&mut ref_attrs);
        }

        for &(seg, ref bm) in rowset.segments() {
            let seg_columnar =
                no_filter && ref_attrs.iter().all(|&a| col_reader.segment_has_columns(seg, a));
            if seg_columnar
                && lex_maps.is_empty()
                && let Some(block_eval) = Self::as_single_block_simd_leaf(node)
            {
                self.scan_segment_simd_blocks(seg, bm, block_eval, sink)?;
                continue;
            }
            if seg_columnar && !ref_attrs.is_empty() {
                batch.clear();
                let mut col_bufs: Vec<Vec<Option<f64>>> = vec![Vec::new(); ref_attrs.len()];
                for local in bm.iter() {
                    batch.push(RowId::new(seg, local));
                    if batch.len() >= SCAN_BATCH {
                        self.eval_batch_columnar(
                            &col_reader,
                            node,
                            &ref_attrs,
                            &mut col_bufs,
                            &batch,
                            lex_maps,
                            &mut lex_scores,
                            sink,
                        )?;
                        batch.clear();
                    }
                }
                if !batch.is_empty() {
                    self.eval_batch_columnar(
                        &col_reader,
                        node,
                        &ref_attrs,
                        &mut col_bufs,
                        &batch,
                        lex_maps,
                        &mut lex_scores,
                        sink,
                    )?;
                }
                continue;
            }

            for local in bm.iter() {
                batch.push(RowId::new(seg, local));
                if batch.len() >= SCAN_BATCH {
                    self.eval_batch(&row_reader, node, &batch, lex_maps, &mut lex_scores, sink)?;
                    batch.clear();
                }
            }
            if !batch.is_empty() {
                self.eval_batch(&row_reader, node, &batch, lex_maps, &mut lex_scores, sink)?;
                batch.clear();
            }
        }
        Ok(())
    }

    fn as_single_block_simd_leaf(node: &Node) -> Option<(AttrId, SimdLeafOp)> {
        match node {
            Node::Sat {
                id,
                midpoint,
                exponent,
                missing,
            } => Some((
                *id,
                SimdLeafOp::Saturate {
                    weight: 1.0,
                    midpoint: *midpoint,
                    exponent: *exponent,
                    missing: *missing,
                },
            )),
            Node::Decay {
                id,
                origin,
                midpoint,
                exponent,
                mode,
                missing,
            } => Some((
                *id,
                SimdLeafOp::Decay {
                    weight: 1.0,
                    origin: *origin,
                    midpoint: *midpoint,
                    exponent: *exponent,
                    mode: *mode,
                    missing: *missing,
                },
            )),
            Node::Product(ch) if ch.len() == 2 => match (&ch[0], &ch[1]) {
                (
                    Node::Const(w),
                    Node::Sat {
                        id,
                        midpoint,
                        exponent,
                        missing,
                    },
                ) => Some((
                    *id,
                    SimdLeafOp::Saturate {
                        weight: *w,
                        midpoint: *midpoint,
                        exponent: *exponent,
                        missing: *missing,
                    },
                )),
                (
                    Node::Const(w),
                    Node::Decay {
                        id,
                        origin,
                        midpoint,
                        exponent,
                        mode,
                        missing,
                    },
                ) => Some((
                    *id,
                    SimdLeafOp::Decay {
                        weight: *w,
                        origin: *origin,
                        midpoint: *midpoint,
                        exponent: *exponent,
                        mode: *mode,
                        missing: *missing,
                    },
                )),
                _ => None,
            },
            _ => None,
        }
    }

    fn scan_segment_simd_blocks(
        &self,
        seg: SegmentId,
        bm: &roaring::RoaringBitmap,
        (attr_id, op): (AttrId, SimdLeafOp),
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        let mut by_block: Vec<(u32, [u64; 4])> = Vec::new();
        for local in bm.iter() {
            let blk = local >> 8;
            let slot = (local & 0xFF) as usize;
            if let Some(last) = by_block.last_mut()
                && last.0 == blk
            {
                last.1[slot >> 6] |= 1u64 << (slot & 63);
            } else {
                let mut m = [0u64; 4];
                m[slot >> 6] |= 1u64 << (slot & 63);
                by_block.push((blk, m));
            }
        }
        let slices = if self
            .schema
            .get_by_id(attr_id)
            .is_some_and(|e| e.storage == ColumnStorage::Detached)
        {
            let col_gen = crate::detached::current_detached_gen(&self.storage, attr_id)?;
            let det_keys: Vec<(AttrId, izzydb_core::config::ColumnGen, SegmentId, u32)> = by_block
                .iter()
                .map(|&(blk, _)| (attr_id, col_gen, seg, blk))
                .collect();
            self.storage
                .multi_get_detached_blocks_pinned(&det_keys, true)?
        } else {
            let keys: Vec<(SegmentId, AttrId, u32)> = by_block
                .iter()
                .map(|&(blk, _)| (seg, attr_id, blk))
                .collect();
            self.storage.multi_get_column_blocks_pinned(&keys, true)?
        };

        let mut f64_buf = [0.0f64; ROW_BLOCK_SIZE];
        let mut out_scores = [0.0f32; ROW_BLOCK_SIZE];
        let mut pos_rows: Vec<RowId> = Vec::with_capacity(ROW_BLOCK_SIZE);
        let mut pos_scores: Vec<f32> = Vec::with_capacity(ROW_BLOCK_SIZE);

        for ((blk, req_mask), opt_slice) in by_block.into_iter().zip(slices.iter()) {
            let present_mask = match opt_slice {
                Some(slice) => ColumnBlockView::parse(slice)?.decode_numeric_f64_block(&mut f64_buf),
                None => {
                    f64_buf.fill(0.0);
                    [0u64; 4]
                }
            };
            let missing_val = match op {
                SimdLeafOp::Saturate {
                    midpoint,
                    exponent,
                    missing,
                    ..
                } => {
                    batch_eval_saturate(
                        &f64_buf,
                        &present_mask,
                        midpoint,
                        exponent,
                        missing,
                        &mut out_scores,
                    );
                    missing
                }
                SimdLeafOp::Decay {
                    origin,
                    midpoint,
                    exponent,
                    mode,
                    missing,
                    ..
                } => {
                    batch_eval_decay(
                        &f64_buf,
                        &present_mask,
                        origin,
                        midpoint,
                        exponent,
                        mode,
                        missing,
                        &mut out_scores,
                    );
                    missing
                }
            };
            let _ = missing_val;
            let weight = match op {
                SimdLeafOp::Saturate { weight, .. } | SimdLeafOp::Decay { weight, .. } => weight,
            };
            pos_rows.clear();
            pos_scores.clear();
            for_each_set_bit_256(&req_mask, |slot| {
                let s = out_scores[slot as usize] * weight;
                if s > 0.0 && !s.is_nan() {
                    pos_rows.push(RowId::new(seg, (blk << 8) | (slot as u32)));
                    pos_scores.push(s.min(f32::MAX));
                }
            });
            if !pos_rows.is_empty() {
                let doc_ids = RowResolver::resolve_all(&self.storage, &pos_rows)?;
                for ((&r, doc_id), &s) in pos_rows.iter().zip(doc_ids).zip(&pos_scores) {
                    sink(r, doc_id, s);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_batch_columnar(
        &self,
        reader: &ColumnReader<'_>,
        node: &Node,
        attrs: &[AttrId],
        col_bufs: &mut [Vec<Option<f64>>],
        batch: &[RowId],
        lex_maps: &[HashMap<DocId, f32>],
        lex_scores: &mut [f32],
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        for (&attr_id, buf) in attrs.iter().zip(col_bufs.iter_mut()) {
            reader.read_numeric_batch(attr_id, batch, buf)?;
        }
        let doc_ids = RowResolver::resolve_all(&self.storage, batch)?;
        for (idx, (&row, doc_id)) in batch.iter().zip(doc_ids).enumerate() {
            for (s, m) in lex_scores.iter_mut().zip(lex_maps) {
                *s = m.get(&doc_id).copied().unwrap_or(0.0);
            }
            let score = node.eval_from_cols(idx, row, attrs, col_bufs, lex_scores);
            if score > 0.0 && !score.is_nan() {
                sink(row, doc_id, score.min(f32::MAX));
            }
        }
        Ok(())
    }

    fn eval_batch(
        &self,
        reader: &dyn AttrReader,
        node: &Node,
        batch: &[RowId],
        lex_maps: &[HashMap<DocId, f32>],
        lex_scores: &mut [f32],
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        let doc_ids = RowResolver::resolve_all(&self.storage, batch)?;
        let mut idx = 0usize;
        reader.read_rows(batch, &mut |row, view| {
            let doc_id = doc_ids[idx];
            idx += 1;
            for (s, m) in lex_scores.iter_mut().zip(lex_maps) {
                *s = m.get(&doc_id).copied().unwrap_or(0.0);
            }
            let ctx = RowCtx::new(self, row, view);
            let score = node.eval(&ctx, lex_scores)?;
            if score > 0.0 && !score.is_nan() {
                sink(row, doc_id, score.min(f32::MAX));
            }
            Ok(())
        })
    }

    fn general_eval(
        &self,
        norm: &RankBy,
        filter: Option<&Filter>,
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        let mut lex = Vec::new();
        let mut node = Node::compile(norm, &self.schema, &mut lex);
        self.resolve_filter_sets(&mut node)?;
        let lex_sets: Vec<RowSet> = lex
            .iter()
            .map(|e| {
                self.fts_index
                    .eval_rank_by_match_rowset(&self.storage, &self.schema, e)
            })
            .collect::<Result<_>>()?;

        let snap = self.storage.segment_snapshot();
        let cand_rs = match node.positive_universe(&lex_sets) {
            Some(mut rs) => {
                if rs.is_empty() {
                    return Ok(());
                }
                if let Some(f) = filter {
                    let f_rs = filter_plan::materialize_filter(
                        &self.storage,
                        &self.schema,
                        f,
                        &self.fts_index,
                    )?;
                    rs = rs.and(&f_rs);
                }
                rs
            }
            None => match filter {
                Some(f) => filter_plan::materialize_filter(
                    &self.storage,
                    &self.schema,
                    f,
                    &self.fts_index,
                )?,
                None => RowSet::from_all_live(&*snap),
            },
        };
        if cand_rs.is_empty() {
            return Ok(());
        }

        let mut lex_maps = Vec::with_capacity(lex.len());
        if !lex.is_empty() {
            let mask = LivenessMaskProvider {
                snap: &snap,
                filter: Some(&cand_rs),
            };
            let k = (cand_rs.len() as usize).max(1);
            for expr in &lex {
                let hits = self.fts_index.query_bm25(
                    &self.storage,
                    &self.schema,
                    expr,
                    k,
                    Some(&mask),
                    None,
                )?;
                lex_maps.push(hits.into_iter().map(|h| (h.id, h.score)).collect());
            }
        }
        self.scan_rowset(&node, &cand_rs, &lex_maps, sink)
    }

    fn monotone_rank_query(
        &self,
        norm: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
    ) -> Result<Option<Vec<Bm25Result>>> {
        let (weight, leaf) = match norm {
            RankBy::Product(c) if c.len() == 2 => match (&c[0], &c[1]) {
                (RankBy::Const(w), leaf) => (Some(*w), leaf),
                _ => return Ok(None),
            },
            leaf => (None, leaf),
        };
        let (attr, descending, missing) = match leaf {
            RankBy::Attribute { attr, missing } => (attr, true, *missing),
            RankBy::Saturate { attr, missing, .. } => (attr, true, *missing),
            RankBy::Decay {
                attr,
                mode,
                missing,
                ..
            } => match mode {
                DecayMode::Forward => (attr, false, *missing),
                DecayMode::Backward => (attr, true, *missing),
                DecayMode::Symmetric => return Ok(None),
            },
            _ => return Ok(None),
        };
        let Some(entry) = self.schema.get_by_name(attr) else {
            return Ok(None);
        };
        if missing != 0.0 || !entry.filterable || weight.is_some_and(|w| w <= 0.0) {
            return Ok(None);
        }

        let mut no_lex = Vec::new();
        let node = Node::compile(norm, &self.schema, &mut no_lex);
        let leaf_node = Node::compile(leaf, &self.schema, &mut no_lex);
        let pop = match filter {
            None => OrderPopulation::AllLive,
            Some(f) => {
                let mut trace = FilterPlanTrace::default();
                match filter_plan::plan_filter_with_trace(
                    &self.storage,
                    &self.schema,
                    f,
                    &self.fts_index,
                    &mut trace,
                )? {
                    FilterPlan::Empty => return Ok(Some(Vec::new())),
                    FilterPlan::All => OrderPopulation::AllLive,
                    FilterPlan::Materialized(rs) => {
                        if rs.is_empty() {
                            return Ok(Some(Vec::new()));
                        }
                        OrderPopulation::RowSet(rs)
                    }
                    FilterPlan::Lazy { residual, seed } => OrderPopulation::Lazy { seed, residual },
                }
            }
        };

        let spec = if descending {
            OrderBySpec::desc(attr.clone())
        } else {
            OrderBySpec::asc(attr.clone())
        };
        let score_of = |value: &ScalarValue| -> Option<f32> {
            let v = match value {
                ScalarValue::Int(v) => *v as f64,
                ScalarValue::Uint(v) => *v as f64,
                ScalarValue::Float(v) => *v,
                _ => return None,
            };
            Some(eval_value(&node, &leaf_node, v))
        };

        let mut collected: Vec<Bm25Result> = Vec::new();
        let mut boundary: Option<f32> = None;
        self.stream_value_groups(&spec, &pop, &mut |value, group_rows| {
            let Some(score) = score_of(&value) else {
                return Ok(true);
            };
            if score <= 0.0 || score.is_nan() {
                return Ok(false);
            }
            if let Some(b) = boundary
                && score < b
            {
                return Ok(false);
            }
            let doc_ids = RowResolver::resolve_all(&self.storage, &group_rows)?;
            collected.extend(doc_ids.into_iter().map(|id| Bm25Result { id, score }));
            if boundary.is_none() && collected.len() >= top_k {
                boundary = Some(score);
            }
            Ok(true)
        })?;
        sort_results(&mut collected);
        collected.truncate(top_k);
        Ok(Some(collected))
    }
}

fn eval_value(node: &Node, leaf: &Node, v: f64) -> f32 {
    let leaf_score = match leaf {
        Node::Attr { .. } => clamp_f32(v),
        Node::Sat {
            midpoint, exponent, ..
        } => eval_saturate(v, *midpoint, *exponent),
        Node::Decay {
            origin,
            midpoint,
            exponent,
            mode,
            ..
        } => eval_decay(v, *origin, *midpoint, *exponent, *mode),
        _ => 0.0,
    };
    match node {
        Node::Product(c) => {
            let mut acc = 1.0f32;
            for n in c {
                acc *= match n {
                    Node::Const(w) => *w,
                    _ => leaf_score,
                };
            }
            acc
        }
        _ => leaf_score,
    }
}

#[derive(Clone, Copy)]
enum SimdLeafOp {
    Saturate {
        weight: f32,
        midpoint: f64,
        exponent: f64,
        missing: f32,
    },
    Decay {
        weight: f32,
        origin: f64,
        midpoint: f64,
        exponent: f64,
        mode: DecayMode,
        missing: f32,
    },
}
