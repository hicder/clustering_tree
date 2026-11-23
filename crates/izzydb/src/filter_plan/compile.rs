//! Normalized filter AST compilation.

use std::collections::BTreeMap;


use izzydb_core::error::Result;
use izzydb_core::filter::{Filter, FtsPredicateKind, TokenQuery};
use izzydb_core::schema::ScalarValue;




// ---------------------------------------------------------------------------
// Compiled normalized filter AST (combines same-attr range bounds inside And)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(super) enum CompiledNode {
    All,
    Empty,
    Eq {
        attr: String,
        val: ScalarValue,
        orig: Filter,
    },
    In {
        attr: String,
        vals: Vec<ScalarValue>,
        orig: Filter,
    },
    Contains {
        attr: String,
        val: ScalarValue,
        orig: Filter,
    },
    ContainsAny {
        attr: String,
        vals: Vec<ScalarValue>,
        orig: Filter,
    },
    IsNull {
        attr: String,
        orig: Filter,
    },
    IsNotNull {
        attr: String,
        orig: Filter,
    },
    Range {
        attr: String,
        lo: Option<(ScalarValue, bool)>,
        hi: Option<(ScalarValue, bool)>,
        orig: Filter,
    },
    Fts {
        kind: FtsPredicateKind,
        attr: String,
        query: TokenQuery,
        last_as_prefix: bool,
        orig: Filter,
    },
    And(Vec<CompiledNode>, Filter),
    Or(Vec<CompiledNode>, Filter),
    Not(Box<CompiledNode>, Filter),
}

impl CompiledNode {
    pub(super) fn orig_filter(&self) -> &Filter {
        match self {
            Self::All => unreachable!(),
            Self::Empty => unreachable!(),
            Self::Eq { orig, .. }
            | Self::In { orig, .. }
            | Self::Contains { orig, .. }
            | Self::ContainsAny { orig, .. }
            | Self::IsNull { orig, .. }
            | Self::IsNotNull { orig, .. }
            | Self::Range { orig, .. }
            | Self::Fts { orig, .. }
            | Self::And(_, orig)
            | Self::Or(_, orig)
            | Self::Not(_, orig) => orig,
        }
    }
}

pub(super) fn compile_expr(filter: &Filter) -> Result<CompiledNode> {
    if let Some((kind, attr, query, last_as_prefix)) = filter.as_fts_predicate() {
        return Ok(CompiledNode::Fts {
            kind,
            attr: attr.to_string(),
            query: query.clone(),
            last_as_prefix,
            orig: filter.clone(),
        });
    }
    Ok(match filter {
        Filter::Eq(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else if matches!(val, ScalarValue::Null) {
                CompiledNode::IsNull {
                    attr: attr.clone(),
                    orig: filter.clone(),
                }
            } else {
                CompiledNode::Eq {
                    attr: attr.clone(),
                    val: val.clone(),
                    orig: filter.clone(),
                }
            }
        }
        Filter::NotEq(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::All
            } else if matches!(val, ScalarValue::Null) {
                CompiledNode::IsNotNull {
                    attr: attr.clone(),
                    orig: filter.clone(),
                }
            } else {
                CompiledNode::Not(
                    Box::new(CompiledNode::Eq {
                        attr: attr.clone(),
                        val: val.clone(),
                        orig: Filter::Eq(attr.clone(), val.clone()),
                    }),
                    filter.clone(),
                )
            }
        }
        Filter::In(attr, vals) => {
            if vals.is_empty() {
                CompiledNode::Empty
            } else {
                CompiledNode::In {
                    attr: attr.clone(),
                    vals: vals.clone(),
                    orig: filter.clone(),
                }
            }
        }
        Filter::NotIn(attr, vals) => {
            if vals.is_empty() {
                CompiledNode::All
            } else {
                CompiledNode::Not(
                    Box::new(CompiledNode::In {
                        attr: attr.clone(),
                        vals: vals.clone(),
                        orig: Filter::In(attr.clone(), vals.clone()),
                    }),
                    filter.clone(),
                )
            }
        }
        Filter::Contains(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Contains {
                    attr: attr.clone(),
                    val: val.clone(),
                    orig: filter.clone(),
                }
            }
        }
        Filter::ContainsAny(attr, vals) => {
            if vals.is_empty() {
                CompiledNode::Empty
            } else {
                CompiledNode::ContainsAny {
                    attr: attr.clone(),
                    vals: vals.clone(),
                    orig: filter.clone(),
                }
            }
        }
        Filter::IsNull(attr) => CompiledNode::IsNull {
            attr: attr.clone(),
            orig: filter.clone(),
        },
        Filter::IsNotNull(attr) => CompiledNode::IsNotNull {
            attr: attr.clone(),
            orig: filter.clone(),
        },
        Filter::Gt(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Range {
                    attr: attr.clone(),
                    lo: Some((val.clone(), false)),
                    hi: None,
                    orig: filter.clone(),
                }
            }
        }
        Filter::Gte(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Range {
                    attr: attr.clone(),
                    lo: Some((val.clone(), true)),
                    hi: None,
                    orig: filter.clone(),
                }
            }
        }
        Filter::Lt(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Range {
                    attr: attr.clone(),
                    lo: None,
                    hi: Some((val.clone(), false)),
                    orig: filter.clone(),
                }
            }
        }
        Filter::Lte(attr, val) => {
            if matches!(val, ScalarValue::Float(v) if v.is_nan()) {
                CompiledNode::Empty
            } else {
                CompiledNode::Range {
                    attr: attr.clone(),
                    lo: None,
                    hi: Some((val.clone(), true)),
                    orig: filter.clone(),
                }
            }
        }
        Filter::And(children) => {
            if children.is_empty() {
                return Ok(CompiledNode::All);
            }
            let mut compiled_children = Vec::new();
            type RangeAccumulator =
                BTreeMap<String, (Option<(ScalarValue, bool)>, Option<(ScalarValue, bool)>, Vec<Filter>)>;
            let mut ranges_by_attr: RangeAccumulator = BTreeMap::new();
            for ch in children {
                let c = compile_expr(ch)?;
                match c {
                    CompiledNode::Empty => return Ok(CompiledNode::Empty),
                    CompiledNode::All => continue,
                    CompiledNode::Range { attr, lo, hi, orig } => {
                        let entry = ranges_by_attr.entry(attr).or_insert((None, None, Vec::new()));
                        if let Some((new_lo, new_incl)) = lo {
                            entry.0 = Some(tighten_lo(entry.0.take(), new_lo, new_incl));
                        }
                        if let Some((new_hi, new_incl)) = hi {
                            entry.1 = Some(tighten_hi(entry.1.take(), new_hi, new_incl));
                        }
                        entry.2.push(orig);
                    }
                    other => compiled_children.push(other),
                }
            }
            for (attr, (lo, hi, origs)) in ranges_by_attr {
                let orig = if origs.len() == 1 {
                    origs.into_iter().next().unwrap()
                } else {
                    Filter::And(origs)
                };
                compiled_children.push(CompiledNode::Range { attr, lo, hi, orig });
            }
            if compiled_children.is_empty() {
                CompiledNode::All
            } else if compiled_children.len() == 1 {
                compiled_children.into_iter().next().unwrap()
            } else {
                CompiledNode::And(compiled_children, filter.clone())
            }
        }
        Filter::Or(children) => {
            if children.is_empty() {
                return Ok(CompiledNode::Empty);
            }
            let mut compiled_children = Vec::new();
            for ch in children {
                let c = compile_expr(ch)?;
                match c {
                    CompiledNode::All => return Ok(CompiledNode::All),
                    CompiledNode::Empty => continue,
                    other => compiled_children.push(other),
                }
            }
            if compiled_children.is_empty() {
                CompiledNode::Empty
            } else if compiled_children.len() == 1 {
                compiled_children.into_iter().next().unwrap()
            } else {
                CompiledNode::Or(compiled_children, filter.clone())
            }
        }
        Filter::Not(child) => match compile_expr(child)? {
            CompiledNode::All => CompiledNode::Empty,
            CompiledNode::Empty => CompiledNode::All,
            other => CompiledNode::Not(Box::new(other), filter.clone()),
        },
        other => {
            let _ = other;
            unreachable!("all Filter variants handled")
        }
    })
}

pub(super) fn tighten_lo(
    cur: Option<(ScalarValue, bool)>,
    new_v: ScalarValue,
    new_incl: bool,
) -> (ScalarValue, bool) {
    let Some((cur_v, cur_incl)) = cur else {
        return (new_v, new_incl);
    };
    match cmp_scalar_ordered(&cur_v, &new_v) {
        std::cmp::Ordering::Less => (new_v, new_incl),
        std::cmp::Ordering::Greater => (cur_v, cur_incl),
        std::cmp::Ordering::Equal => (cur_v, cur_incl && new_incl),
    }
}

pub(super) fn tighten_hi(
    cur: Option<(ScalarValue, bool)>,
    new_v: ScalarValue,
    new_incl: bool,
) -> (ScalarValue, bool) {
    let Some((cur_v, cur_incl)) = cur else {
        return (new_v, new_incl);
    };
    match cmp_scalar_ordered(&cur_v, &new_v) {
        std::cmp::Ordering::Greater => (new_v, new_incl),
        std::cmp::Ordering::Less => (cur_v, cur_incl),
        std::cmp::Ordering::Equal => (cur_v, cur_incl && new_incl),
    }
}

pub(super) fn cmp_scalar_ordered(a: &ScalarValue, b: &ScalarValue) -> std::cmp::Ordering {
    match (a, b) {
        (ScalarValue::Int(x), ScalarValue::Int(y)) => x.cmp(y),
        (ScalarValue::Uint(x), ScalarValue::Uint(y)) => x.cmp(y),
        (ScalarValue::Float(x), ScalarValue::Float(y)) => {
            let xn = if *x == 0.0 { 0.0 } else { *x };
            let yn = if *y == 0.0 { 0.0 } else { *y };
            xn.total_cmp(&yn)
        }
        (ScalarValue::String(x), ScalarValue::String(y)) => x.cmp(y),
        _ => std::cmp::Ordering::Equal,
    }
}
