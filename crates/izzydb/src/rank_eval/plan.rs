//! Rank-expression classification and planning.


use izzydb_core::filter::Filter;
use izzydb_core::rank::RankBy;



// ------------------------------------------------------------------------------------------
// Compiled per-row evaluation tree
// ------------------------------------------------------------------------------------------

use super::*;

pub(super) enum Plan {
    Lexical,
    Rescored {
        lex: RankBy,
        mult: Vec<RankBy>,
        add: Vec<RankBy>,
    },
    General,
}

pub(super) fn classify(norm: &RankBy) -> Plan {
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

pub(super) fn positive_support(expr: &RankBy) -> Option<Filter> {
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
    pub(super) mult: Option<Node>,
    pub(super) add: Option<Node>,
    pub(super) mult_ub: f32,
    pub(super) add_ub: f32,
}

// ------------------------------------------------------------------------------------------
// Collection integration
// ------------------------------------------------------------------------------------------
