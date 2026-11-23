//! Compiled per-row evaluation tree.

use std::sync::Arc;

use izzydb_core::config::{AttrId, RowId};
use izzydb_core::error::Result;
use izzydb_core::filter::Filter;
use izzydb_core::rank::{DecayMode, RankBy, eval_decay, eval_saturate};
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::{CollectionSchema, ColumnStorage};
use izzydb_storage::codec::RowView;
use izzydb_storage::column::ColumnBlockView;

use crate::collection::Collection;


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

    pub(super) fn has_filter_node(&self) -> bool {
        match self {
            Node::Filter(_) => true,
            Node::Sum(c) | Node::Max(c) | Node::Product(c) => {
                c.iter().any(Node::has_filter_node)
            }
            _ => false,
        }
    }

    pub(super) fn collect_numeric_attrs(&self, out: &mut Vec<AttrId>) {
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

    pub(super) fn eval_from_cols(
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
    pub(super) fn positive_universe(&self, lex_sets: &[RowSet]) -> Option<RowSet> {
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
pub(super) fn clamp_f32(v: f64) -> f32 {
    if v.is_nan() || v <= 0.0 {
        0.0
    } else {
        v.min(f32::MAX as f64) as f32
    }
}

/// Smallest `f32` that is `>= v` (for non-negative `v`).
pub(super) fn round_up_f32(v: f64) -> f32 {
    let f = clamp_f32(v);
    if (f as f64) < v && f.is_finite() {
        f.next_up()
    } else {
        f
    }
}

/// Per-row evaluation context wrapping zero-copy [`RowView`].
pub(crate) struct RowCtx<'c, 'b> {
    pub(super) coll: &'c Collection,
    pub(super) row_id: RowId,
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
