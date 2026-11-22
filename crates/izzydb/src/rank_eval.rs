//! Generalized [`RankBy`] execution.
//!
//! The Block-Max MAXSCORE engine in `izzydb-fts` only understands *pure lexical* expressions
//! (BM25 leaves combined with `Sum` / `Max` / constant weights). Everything else is routed here:
//!
//! | normalized shape                                   | strategy                                              |
//! |----------------------------------------------------|-------------------------------------------------------|
//! | pure lexical                                       | MAXSCORE unchanged                                    |
//! | `L * m(doc)` (one lexical factor, others modifiers)| MAXSCORE on `L * UB(m)`, exact rescoring of survivors |
//! | `L + a(doc)` (lexical terms plus modifier terms)   | MAXSCORE with a lowered threshold + exact backfill    |
//! | no BM25 (`Filter` / `Attribute` / `Saturate` / ...)| index-order scan if monotone, else candidate scan     |
//! | anything else (e.g. `BM25 x BM25`)                 | exact evaluation over the positive candidate universe |
//!
//! Every path returns the exact top-k under `score DESC, DocId ASC`; scores `<= 0` are excluded.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

use roaring::RoaringBitmap;

use crate::codec::{
    DocView, decode_attributes, decode_scalar_val_bytes, encode_attr_prefix, ordered_type_tag,
};
use crate::collection::Collection;
use crate::config::{AttrId, DocId};
use crate::error::Result;
use crate::filter::Filter;
use crate::rank::{DecayMode, OrderBySpec, RankBy, eval_decay, eval_saturate};
use crate::schema::{AttributeType, Attributes, CollectionSchema, ScalarValue};
use izzydb_fts::{Bm25Result, RankModifier};

const SCAN_BATCH: usize = 1024;

// ------------------------------------------------------------------------------------------
// Compiled per-document evaluation tree
// ------------------------------------------------------------------------------------------

/// A [`RankBy`] resolved against the schema: attribute names become ids and every maximal pure
/// lexical subtree becomes a `Lex(i)` slot filled from precomputed BM25 scores.
#[derive(Debug, Clone)]
pub(crate) enum Node {
    Const(f32),
    Lex(usize),
    Filter(Filter),
    /// A filter resolved up-front to the exact set of matching documents.
    DocSet(Arc<HashSet<DocId>>),
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

    /// Score of this node for one document. `lex` holds the BM25 score of each `Lex` slot.
    pub(crate) fn eval(&self, ctx: &mut DocCtx<'_, '_>, lex: &[f32]) -> Result<f32> {
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
            Node::DocSet(set) => {
                if set.contains(&ctx.doc_id) {
                    1.0
                } else {
                    0.0
                }
            }
            Node::Attr { id, missing } => match ctx.view.get_numeric_f64(*id)? {
                Some(v) => clamp_f32(v),
                None => *missing,
            },
            Node::Sat {
                id,
                midpoint,
                exponent,
                missing,
            } => match ctx.view.get_numeric_f64(*id)? {
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
            } => match ctx.view.get_numeric_f64(*id)? {
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

    /// Sound upper bound of this node over all documents; `f32::INFINITY` if none is known.
    pub(crate) fn upper_bound(&self, coll: &Collection) -> Result<f32> {
        Ok(match self {
            Node::Const(c) => *c,
            Node::Lex(_) => f32::INFINITY,
            Node::Filter(_) | Node::DocSet(_) => 1.0,
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

    /// Superset of `FtsDocId`s with score `> 0`, or `None` when it is not restricted to
    /// documents reachable from the FTS index (i.e. "all documents").
    fn positive_universe(&self, lex_bms: &[RoaringBitmap]) -> Option<RoaringBitmap> {
        match self {
            Node::Const(c) => (*c <= 0.0).then(RoaringBitmap::new),
            Node::Lex(i) => Some(lex_bms[*i].clone()),
            Node::Filter(_) | Node::Attr { .. } | Node::Sat { .. } | Node::Decay { .. } => None,
            Node::DocSet(_) => None,
            Node::Sum(c) | Node::Max(c) => {
                let mut acc = RoaringBitmap::new();
                for n in c {
                    acc |= n.positive_universe(lex_bms)?;
                }
                Some(acc)
            }
            Node::Product(c) => {
                let mut acc: Option<RoaringBitmap> = None;
                for n in c {
                    if let Some(bm) = n.positive_universe(lex_bms) {
                        acc = Some(match acc {
                            Some(mut cur) => {
                                cur &= bm;
                                cur
                            }
                            None => bm,
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

/// Per-document evaluation context: zero-copy [`DocView`] plus lazily decoded attributes for
/// FTS filter predicates.
pub(crate) struct DocCtx<'c, 'b> {
    coll: &'c Collection,
    doc_id: DocId,
    bytes: &'b [u8],
    pub(crate) view: DocView<'b>,
    attrs: Option<Attributes>,
}

impl<'c, 'b> DocCtx<'c, 'b> {
    pub(crate) fn new(coll: &'c Collection, doc_id: DocId, bytes: &'b [u8]) -> Result<Self> {
        Ok(Self {
            coll,
            doc_id,
            bytes,
            view: DocView::parse(bytes)?,
            attrs: None,
        })
    }

    pub(crate) fn filter_matches(&mut self, f: &Filter) -> Result<bool> {
        if !f.has_fts() {
            return self
                .view
                .matches_filter(&self.coll.schema, f, &mut |_, _, _, _, _| false);
        }
        if self.attrs.is_none() {
            self.attrs = Some(decode_attributes(&self.coll.schema, self.bytes)?);
        }
        Ok(self
            .coll
            .fts_index
            .matches_filter(&self.coll.schema, f, self.attrs.as_ref().unwrap()))
    }
}

// ------------------------------------------------------------------------------------------
// Top-k collection
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
        v.sort_unstable(); // best first (`Ord` is "worse is greater")
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
    /// Pure lexical: plain MAXSCORE.
    Lexical,
    /// `lex * prod(mult) + sum(add)` with a pure lexical `lex`.
    Rescored {
        lex: RankBy,
        mult: Vec<RankBy>,
        add: Vec<RankBy>,
    },
    /// Exact evaluation over the positive candidate universe.
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

/// A filter whose matches are a superset of the documents where the (normalized) non-lexical
/// `expr` scores `> 0`, or `None` when no such restriction exists (e.g. `Decay`, `Const`).
fn positive_support(expr: &RankBy) -> Option<Filter> {
    match expr {
        RankBy::Filter(f) => Some((**f).clone()),
        RankBy::Product(children) => {
            let parts: Vec<Filter> = children
                .iter()
                .filter(|c| !matches!(c, RankBy::Const(_)))
                .filter_map(positive_support)
                .collect();
            // Any single factor that is zero outside its support zeroes the whole product.
            match parts.len() {
                0 => None,
                1 => parts.into_iter().next(),
                _ => Some(Filter::And(parts)),
            }
        }
        RankBy::Sum(children) | RankBy::Max(children) => {
            let parts: Option<Vec<Filter>> = children.iter().map(positive_support).collect();
            parts.map(|p| if p.len() == 1 { p.into_iter().next().unwrap() } else { Filter::Or(p) })
        }
        _ => None,
    }
}

/// Exact per-document modifier around a pure lexical expression:
/// `score = L * mult(doc) + add(doc)`.
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
    /// Largest value of a numeric `filterable` attribute according to `cf_attr_clusters`
    /// (a sound upper bound: stale keys can only over-estimate), clamped to `>= 0`.
    /// `None` when the attribute is not indexed or the bound is not representable.
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
        let mut it = self
            .storage
            .iter_attr_clusters_prefix(encode_attr_prefix(attr_id, tag), true)?;
        let Some(first) = it.next() else {
            return Ok(Some(0.0));
        };
        let (vbytes, _) = first?;
        let v = match decode_scalar_val_bytes(tag, &vbytes)? {
            ScalarValue::Int(v) => v as f64,
            ScalarValue::Uint(v) => v as f64,
            ScalarValue::Float(v) => v,
            _ => return Ok(None),
        };
        let ub = round_up_f32(v);
        Ok(ub.is_finite().then_some(ub))
    }

    /// Replace `Filter` leaves that can be resolved exactly from the indexes (FTS predicates and
    /// filters over few candidate clusters) with precomputed document sets, so per-document
    /// scoring is a set lookup instead of re-evaluating (and re-analyzing) the filter.
    fn resolve_filter_sets(&self, node: &mut Node) -> Result<()> {
        /// Do not materialize sets larger than this many documents.
        const MAX_DOC_SET: u64 = 1 << 20;
        match node {
            Node::Filter(f) => {
                if let crate::collection::FilterPushdown::Exact(bm) = self.try_pushdown_filter(f)?
                    && bm.len() <= MAX_DOC_SET
                {
                    let fts_ids: Vec<_> = bm.iter().collect();
                    let set: HashSet<DocId> = self
                        .storage
                        .multi_get_doc_ids_from_fts(&fts_ids)?
                        .into_iter()
                        .flatten()
                        .collect();
                    *node = Node::DocSet(Arc::new(set));
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
                self.general_eval(&norm, filter, &mut |id, s| topk.push(id, s))?;
                Ok(topk.into_sorted())
            }
        }
    }

    /// Exact ids of all documents with score `> 0` that match `filter` (used for faceting).
    pub(crate) fn rank_matching_doc_ids(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
    ) -> Result<Vec<DocId>> {
        rank_by.validate(&self.schema)?;
        if let Some(f) = filter {
            f.validate(&self.schema)?;
        }
        if self.is_empty() {
            return Ok(Vec::new());
        }
        let norm = rank_by.normalize();
        let mut ids = Vec::new();
        self.general_eval(&norm, filter, &mut |id, _| ids.push(id))?;
        ids.sort_unstable();
        Ok(ids)
    }

    /// Run a pure lexical `rank_by` through MAXSCORE, pushing `filter` down where possible and
    /// optionally applying an exact per-document [`LexModifier`].
    pub(crate) fn lexical_query(
        &self,
        rank_by: &RankBy,
        filter: Option<&Filter>,
        top_k: usize,
        modifier: Option<&LexModifier>,
    ) -> Result<Vec<Bm25Result>> {
        let Some(f) = filter else {
            return self.run_lexical(rank_by, top_k, None, None, modifier);
        };
        let (pushdown_bm, is_exact) = match self.try_pushdown_filter(f)? {
            crate::collection::FilterPushdown::None => (None, false),
            crate::collection::FilterPushdown::Superset(bm) => (Some(bm), false),
            crate::collection::FilterPushdown::Exact(bm) => (Some(bm), true),
        };
        if let Some(ref bm) = pushdown_bm
            && bm.is_empty()
        {
            return Ok(Vec::new());
        }
        let post_filter = if is_exact { None } else { Some(f) };
        self.run_lexical(rank_by, top_k, pushdown_bm.as_ref(), post_filter, modifier)
    }

    fn run_lexical(
        &self,
        rank_by: &RankBy,
        top_k: usize,
        doc_filter: Option<&RoaringBitmap>,
        post_filter: Option<&Filter>,
        modifier: Option<&LexModifier>,
    ) -> Result<Vec<Bm25Result>> {
        let Some(m) = modifier else {
            return match post_filter {
                None => self.fts_index.query_bm25(
                    &self.storage,
                    &self.schema,
                    rank_by,
                    top_k,
                    doc_filter,
                    None,
                ),
                Some(f) => {
                    let mut lazy = |doc_id: DocId| -> Result<bool> {
                        let Some(slice) = self.storage.get_doc_bytes_pinned(doc_id)? else {
                            return Ok(false);
                        };
                        DocCtx::new(self, doc_id, &slice)?.filter_matches(f)
                    };
                    self.fts_index.query_bm25(
                        &self.storage,
                        &self.schema,
                        rank_by,
                        top_k,
                        doc_filter,
                        Some(&mut lazy),
                    )
                }
            };
        };

        let mut rescore = |doc_id: DocId, s_ub: f32| -> Result<f32> {
            let Some(slice) = self.storage.get_doc_bytes_pinned(doc_id)? else {
                return Ok(0.0);
            };
            let mut ctx = DocCtx::new(self, doc_id, &slice)?;
            if let Some(f) = post_filter
                && !ctx.filter_matches(f)?
            {
                return Ok(0.0);
            }
            let mv = match &m.mult {
                Some(n) => n.eval(&mut ctx, &[])?,
                None => m.mult_ub,
            };
            let av = match &m.add {
                Some(n) => n.eval(&mut ctx, &[])?,
                None => 0.0,
            };
            let lexical = if mv == m.mult_ub {
                s_ub
            } else {
                s_ub * (mv / m.mult_ub)
            };
            let total = lexical + av;
            Ok(if total.is_nan() { 0.0 } else { total.min(f32::MAX) })
        };
        self.fts_index.query_bm25_with(
            &self.storage,
            &self.schema,
            rank_by,
            top_k,
            doc_filter,
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
        let mult_node = (!mult.is_empty())
            .then(|| Node::compile(&RankBy::Product(mult).normalize(), &self.schema, &mut no_lex));
        let add_expr = (!add.is_empty()).then(|| RankBy::Sum(add).normalize());
        let add_node = add_expr
            .as_ref()
            .map(|e| Node::compile(e, &self.schema, &mut no_lex));
        let (mut mult_node, mut add_node) = (mult_node, add_node);
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
        // Unbounded (or degenerate) modifiers cannot drive pruning: evaluate exactly instead.
        if !(mult_ub.is_finite() && mult_ub > 0.0 && add_ub.is_finite()) {
            let mut topk = TopK::new(top_k);
            self.general_eval(norm, filter, &mut |id, s| topk.push(id, s))?;
            return Ok(topk.into_sorted());
        }
        let modifier = LexModifier {
            mult: mult_node,
            add: add_node,
            mult_ub,
            add_ub,
        };
        let mut results = self.lexical_query(&lex, filter, top_k, Some(&modifier))?;

        // Documents without any lexical match score `add(doc)`; they can only matter if the
        // lexical top-k is not full or its weakest score does not strictly exceed `add_ub`.
        if let Some(add_node) = modifier.add.as_ref()
            && add_ub > 0.0
        {
            let covered = results.len() == top_k
                && results.last().is_some_and(|r| r.score > add_ub);
            if !covered {
                let lex_bm = self.rank_match_bitmap(&lex)?;
                // Restrict the backfill universe to the additive term's support when it has one
                // (e.g. `w * Filter(..)` only scores on filter matches).
                let support = add_expr.as_ref().and_then(positive_support);
                let docs = match (support, filter) {
                    (Some(sup), Some(f)) => {
                        self.resolve_filter_doc_ids(&Filter::And(vec![f.clone(), sup]))?
                    }
                    (Some(sup), None) => self.resolve_filter_doc_ids(&sup)?,
                    (None, _) => self.universe_docs(filter)?,
                };
                let mut topk = TopK::new(top_k);
                self.scan_docs(add_node, &docs, &[], Some(&lex_bm), &mut |id, s| {
                    topk.push(id, s)
                })?;
                results.extend(topk.into_sorted());
                sort_results(&mut results);
                results.truncate(top_k);
            }
        }
        Ok(results)
    }

    /// Sorted ids of all live documents matching `filter` (all documents when `None`).
    fn universe_docs(&self, filter: Option<&Filter>) -> Result<Vec<DocId>> {
        match filter {
            Some(f) => self.resolve_filter_doc_ids(f),
            None => self.storage.all_doc_ids(),
        }
    }

    /// Evaluate `node` over `docs` (sorted), skipping documents in the `exclude` FTS bitmap and
    /// reporting every document with a finite score `> 0`.
    fn scan_docs(
        &self,
        node: &Node,
        docs: &[DocId],
        lex_maps: &[HashMap<DocId, f32>],
        exclude: Option<&RoaringBitmap>,
        sink: &mut dyn FnMut(DocId, f32),
    ) -> Result<()> {
        let mut lex_scores = vec![0.0f32; lex_maps.len()];
        for batch in docs.chunks(SCAN_BATCH) {
            let skip: Vec<bool> = match exclude {
                Some(bm) => self
                    .storage
                    .multi_get_fts_doc_ids(batch)?
                    .into_iter()
                    .map(|f| f.is_some_and(|f| bm.contains(f)))
                    .collect(),
                None => vec![false; batch.len()],
            };
            let slices = self.storage.multi_get_doc_bytes_pinned(batch)?;
            for ((&doc_id, slice), skipped) in batch.iter().zip(slices).zip(skip) {
                let Some(slice) = slice else { continue };
                if skipped {
                    continue;
                }
                for (s, m) in lex_scores.iter_mut().zip(lex_maps) {
                    *s = m.get(&doc_id).copied().unwrap_or(0.0);
                }
                let mut ctx = DocCtx::new(self, doc_id, &slice)?;
                let score = node.eval(&mut ctx, &lex_scores)?;
                if score > 0.0 && !score.is_nan() {
                    sink(doc_id, score.min(f32::MAX));
                }
            }
        }
        Ok(())
    }

    /// Exact evaluation of an arbitrary normalized expression: derive a candidate universe from
    /// the lexical leaves, fetch their BM25 scores, then evaluate every candidate.
    fn general_eval(
        &self,
        norm: &RankBy,
        filter: Option<&Filter>,
        sink: &mut dyn FnMut(DocId, f32),
    ) -> Result<()> {
        let mut lex = Vec::new();
        let mut node = Node::compile(norm, &self.schema, &mut lex);
        self.resolve_filter_sets(&mut node)?;
        let lex_bms: Vec<RoaringBitmap> = lex
            .iter()
            .map(|e| self.rank_match_bitmap(e))
            .collect::<Result<_>>()?;

        let mut docs = match node.positive_universe(&lex_bms) {
            Some(bm) => {
                if bm.is_empty() {
                    return Ok(());
                }
                let fts_ids: Vec<_> = bm.iter().collect();
                let mut ids: Vec<DocId> = self
                    .storage
                    .multi_get_doc_ids_from_fts(&fts_ids)?
                    .into_iter()
                    .flatten()
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                if let Some(f) = filter {
                    ids = intersect_sorted(&ids, &self.resolve_filter_doc_ids(f)?);
                }
                ids
            }
            None => self.universe_docs(filter)?,
        };
        docs.dedup();
        if docs.is_empty() {
            return Ok(());
        }

        let mut lex_maps = Vec::with_capacity(lex.len());
        if !lex.is_empty() {
            let mut doc_fts = RoaringBitmap::new();
            for fid in self.storage.multi_get_fts_doc_ids(&docs)?.into_iter().flatten() {
                doc_fts.insert(fid);
            }
            for expr in &lex {
                let hits = if doc_fts.is_empty() {
                    Vec::new()
                } else {
                    self.fts_index.query_bm25(
                        &self.storage,
                        &self.schema,
                        expr,
                        (doc_fts.len() as usize).max(1),
                        Some(&doc_fts),
                        None,
                    )?
                };
                lex_maps.push(hits.into_iter().map(|h| (h.id, h.score)).collect());
            }
        }
        self.scan_docs(&node, &docs, &lex_maps, None, sink)
    }

    /// Fast path for a monotone transform of a single `filterable` numeric attribute
    /// (`Attribute` / `Saturate` / `Decay` with `Forward` or `Backward` mode, optionally scaled
    /// by a constant): stream the attribute's index in score order and stop after the top-k
    /// score tier instead of scanning every candidate.
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
        // Documents without the attribute must score 0 (so they never qualify).
        let Some(entry) = self.schema.get_by_name(attr) else {
            return Ok(None);
        };
        if missing != 0.0 || !entry.filterable || weight.is_some_and(|w| w <= 0.0) {
            return Ok(None);
        }

        let mut no_lex = Vec::new();
        let node = Node::compile(norm, &self.schema, &mut no_lex);
        let leaf_node = Node::compile(leaf, &self.schema, &mut no_lex);
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
            return Ok(Some(Vec::new()));
        }

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
        self.stream_value_groups(&spec, prepared.as_ref(), &mut |value, ids| {
            let Some(score) = score_of(&value) else {
                return Ok(true);
            };
            if score <= 0.0 || score.is_nan() {
                return Ok(false); // monotone: nothing later can score > 0
            }
            if let Some(b) = boundary
                && score < b
            {
                return Ok(false);
            }
            collected.extend(ids.into_iter().map(|id| Bm25Result { id, score }));
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

/// Score of `node` (a `Product([Const, leaf])` or a bare `leaf`) for a single attribute value.
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

fn intersect_sorted(a: &[DocId], b: &[DocId]) -> Vec<DocId> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}
