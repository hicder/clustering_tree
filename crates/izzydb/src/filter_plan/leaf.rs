//! Compiled predicates evaluated directly over leaf-colocated scalar columns and summaries.

use std::cmp::Ordering;

use izzydb_core::filter::Filter;
use izzydb_core::schema::{AttributeType, CollectionSchema, ColumnStorage, ScalarValue};
use izzydb_storage::leaf_attrs::{
    f64_ordered_bits, AttrSummary, LeafAttrSpec, LeafAttrValue, LeafAttrView, NodeSummary,
};

/// Evaluate a filter against leaf-colocated attribute columns.
///
/// The filter is compiled once into a dense predicate referencing positions in the storage
/// leaf-attribute spec list, so per-slot evaluation performs no name lookup and no allocation.
///
/// ```text
/// Filter::And([Lt("score", 10.0), Eq("in_stock", true)])
///   -> And([Cmp(0, Lt, Float(10.0)), Eq(1, Bool(true))])
/// ```
pub(crate) struct LeafAttrFilter {
    specs: Vec<LeafAttrSpec>,
    predicate: LeafPredicate,
}

impl LeafAttrFilter {
    /// Compile `filter` against `schema` and the storage spec list.
    pub(crate) fn new(
        schema: &CollectionSchema,
        specs: Vec<LeafAttrSpec>,
        filter: &Filter,
    ) -> Option<Self> {
        if crate::columns::force_row_fallback_enabled() || !leaf_filter_eligible(schema, filter) {
            return None;
        }
        let predicate = compile_predicate(schema, &specs, filter);
        Some(Self { specs, predicate })
    }

    /// Evaluate the predicate for `slot`.
    #[inline]
    pub(crate) fn eval(&self, view: &LeafAttrView<'_>, slot: usize) -> bool {
        self.predicate.eval(&self.specs, view, slot)
    }

    /// Conservative test: return `false` only when no row in the subtree can match.
    #[inline]
    pub(crate) fn may_match(&self, summary: &NodeSummary) -> bool {
        self.predicate.may_match(&self.specs, summary)
    }

    /// Returns `false` when a negation makes every summary test return `true`, so callers can
    /// avoid paying the traversal overhead for no possible pruning.
    #[inline]
    pub(crate) fn can_prune_summaries(&self) -> bool {
        self.predicate.prunable()
    }
}

/// Returns `true` when every attribute in `filter` is a fixed-width inline scalar.
pub(crate) fn leaf_filter_eligible(schema: &CollectionSchema, filter: &Filter) -> bool {
    let eligible = |attr: &str| {
        schema.get_by_name(attr).is_some_and(|e| {
            e.storage == ColumnStorage::Inline
                && e.filterable
                && LeafAttrSpec::is_supported(e.attr_type)
        })
    };
    match filter {
        Filter::And(children) | Filter::Or(children) => {
            !children.is_empty() && children.iter().all(|c| leaf_filter_eligible(schema, c))
        }
        Filter::Not(child) => leaf_filter_eligible(schema, child),
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _) => eligible(a),
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmpOp {
    Lt,
    Lte,
    Gt,
    Gte,
}

#[derive(Debug, Clone)]
enum LeafRhs {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
}

impl LeafRhs {
    #[inline]
    fn ordered_u64(&self) -> u64 {
        match self {
            Self::Bool(v) => u64::from(*v),
            Self::Int(v) => (*v as u64) ^ (1u64 << 63),
            Self::Uint(v) => *v,
            Self::Float(v) => f64_ordered_bits(*v),
        }
    }

    #[inline]
    fn matches(&self, value: LeafAttrValue) -> bool {
        match (self, value) {
            (Self::Bool(a), LeafAttrValue::Bool(b)) => *a == b,
            (Self::Int(a), LeafAttrValue::Int(b)) => *a == b,
            (Self::Uint(a), LeafAttrValue::Uint(b)) => *a == b,
            (Self::Float(a), LeafAttrValue::Float(b)) => {
                !a.is_nan() && !b.is_nan() && *a == b
            }
            _ => false,
        }
    }

    #[inline]
    fn compare_value(&self, value: LeafAttrValue) -> Option<Ordering> {
        match (self, value) {
            (Self::Int(a), LeafAttrValue::Int(b)) => Some(b.cmp(a)),
            (Self::Uint(a), LeafAttrValue::Uint(b)) => Some(b.cmp(a)),
            (Self::Float(a), LeafAttrValue::Float(b)) => {
                if a.is_nan() || b.is_nan() {
                    None
                } else {
                    let an = if *a == 0.0 { 0.0 } else { *a };
                    let bn = if b == 0.0 { 0.0 } else { b };
                    Some(bn.total_cmp(&an))
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
enum LeafPredicate {
    True,
    False,
    And(Vec<Self>),
    Or(Vec<Self>),
    Not(Box<Self>),
    IsNull(usize),
    IsNotNull(usize),
    Eq(usize, LeafRhs),
    NotEq(usize, LeafRhs),
    In(usize, Vec<LeafRhs>),
    Cmp(usize, CmpOp, LeafRhs),
}

impl LeafPredicate {
    fn prunable(&self) -> bool {
        match self {
            Self::True | Self::False | Self::IsNull(_) | Self::IsNotNull(_) | Self::Eq(_, _)
            | Self::NotEq(_, _) | Self::In(_, _) | Self::Cmp(_, _, _) => true,
            Self::And(children) | Self::Or(children) => {
                children.iter().all(|child| child.prunable())
            }
            Self::Not(_) => false,
        }
    }

    fn eval(&self, specs: &[LeafAttrSpec], view: &LeafAttrView<'_>, slot: usize) -> bool {
        match self {
            Self::True => true,
            Self::False => false,
            Self::And(children) => children.iter().all(|c| c.eval(specs, view, slot)),
            Self::Or(children) => children.iter().any(|c| c.eval(specs, view, slot)),
            Self::Not(child) => !child.eval(specs, view, slot),
            Self::IsNull(idx) => value_at(specs, view, *idx, slot).is_none(),
            Self::IsNotNull(idx) => value_at(specs, view, *idx, slot).is_some(),
            Self::Eq(idx, rhs) => value_at(specs, view, *idx, slot)
                .is_some_and(|value| rhs.matches(value)),
            Self::NotEq(idx, rhs) => !value_at(specs, view, *idx, slot)
                .is_some_and(|value| rhs.matches(value)),
            Self::In(idx, vals) => value_at(specs, view, *idx, slot)
                .is_some_and(|value| vals.iter().any(|rhs| rhs.matches(value))),
            Self::Cmp(idx, op, rhs) => value_at(specs, view, *idx, slot)
                .and_then(|value| rhs.compare_value(value))
                .is_some_and(|ord| match op {
                    CmpOp::Lt => ord == Ordering::Less,
                    CmpOp::Lte => ord != Ordering::Greater,
                    CmpOp::Gt => ord == Ordering::Greater,
                    CmpOp::Gte => ord != Ordering::Less,
                }),
        }
    }

    fn may_match(&self, specs: &[LeafAttrSpec], summary: &NodeSummary) -> bool {
        if !summary.complete {
            return true;
        }
        match self {
            Self::True => true,
            Self::False => false,
            Self::And(children) => children.iter().all(|c| c.may_match(specs, summary)),
            Self::Or(children) => children.iter().any(|c| c.may_match(specs, summary)),
            // A negation can match any subtree that contains a row failing the inner
            // predicate; without per-subtree counts it is never safe to prune.
            Self::Not(_) => true,
            Self::IsNull(idx) => {
                let Some(attr) = summary.attr(specs[*idx].attr_id) else {
                    return true;
                };
                attr.present_count < summary.row_count()
            }
            Self::IsNotNull(idx) => {
                let Some(attr) = summary.attr(specs[*idx].attr_id) else {
                    return true;
                };
                attr.present_count > 0
            }
            Self::Eq(idx, rhs) => eq_may_match(specs, summary, *idx, rhs),
            Self::NotEq(_, _) => true,
            Self::In(idx, vals) => vals
                .iter()
                .any(|rhs| eq_may_match(specs, summary, *idx, rhs)),
            Self::Cmp(idx, op, rhs) => cmp_may_match(specs, summary, *idx, *op, rhs),
        }
    }
}

#[inline]
fn value_at(
    specs: &[LeafAttrSpec],
    view: &LeafAttrView<'_>,
    idx: usize,
    slot: usize,
) -> Option<LeafAttrValue> {
    let spec = specs.get(idx)?;
    view.column(spec.attr_id)?.value_at(slot)
}

fn eq_may_match(
    specs: &[LeafAttrSpec],
    summary: &NodeSummary,
    idx: usize,
    rhs: &LeafRhs,
) -> bool {
    let Some(attr) = summary.attr(specs[idx].attr_id) else {
        return true;
    };
    if attr.present_count == 0 {
        return false;
    }
    let v = rhs.ordered_u64();
    if attr.has_values && (v < attr.min || v > attr.max) {
        return false;
    }
    (attr.bits & (1u64 << (izzydb_storage::leaf_attrs::mix64(v) & 63))) != 0
}

fn cmp_may_match(
    specs: &[LeafAttrSpec],
    summary: &NodeSummary,
    idx: usize,
    op: CmpOp,
    rhs: &LeafRhs,
) -> bool {
    let Some(attr): Option<&AttrSummary> = summary.attr(specs[idx].attr_id) else {
        return true;
    };
    if attr.present_count == 0 {
        return false;
    }
    if !attr.has_values {
        return true;
    }
    let v = rhs.ordered_u64();
    match op {
        CmpOp::Lt => attr.min < v,
        CmpOp::Lte => attr.min <= v,
        CmpOp::Gt => attr.max > v,
        CmpOp::Gte => attr.max >= v,
    }
}

fn compile_predicate(
    schema: &CollectionSchema,
    specs: &[LeafAttrSpec],
    filter: &Filter,
) -> LeafPredicate {
    match filter {
        Filter::And(children) => LeafPredicate::And(
            children
                .iter()
                .map(|c| compile_predicate(schema, specs, c))
                .collect(),
        ),
        Filter::Or(children) => LeafPredicate::Or(
            children
                .iter()
                .map(|c| compile_predicate(schema, specs, c))
                .collect(),
        ),
        Filter::Not(child) => {
            LeafPredicate::Not(Box::new(compile_predicate(schema, specs, child)))
        }
        Filter::IsNull(attr) => slot_for(schema, specs, attr)
            .map_or(LeafPredicate::True, |(idx, _)| LeafPredicate::IsNull(idx)),
        Filter::IsNotNull(attr) => slot_for(schema, specs, attr)
            .map_or(LeafPredicate::False, |(idx, _)| {
                LeafPredicate::IsNotNull(idx)
            }),
        Filter::Eq(attr, rhs) | Filter::NotEq(attr, rhs) => {
            let is_not = matches!(filter, Filter::NotEq(_, _));
            let Some((idx, ty)) = slot_for(schema, specs, attr) else {
                return if is_not {
                    LeafPredicate::True
                } else {
                    LeafPredicate::False
                };
            };
            if matches!(rhs, ScalarValue::Null) {
                return if is_not {
                    LeafPredicate::IsNotNull(idx)
                } else {
                    LeafPredicate::IsNull(idx)
                };
            }
            match normalize_rhs(rhs, ty) {
                Some(rhs) if is_not => LeafPredicate::NotEq(idx, rhs),
                Some(rhs) => LeafPredicate::Eq(idx, rhs),
                None if is_not => LeafPredicate::True,
                None => LeafPredicate::False,
            }
        }
        Filter::In(attr, values) | Filter::NotIn(attr, values) => {
            let is_not = matches!(filter, Filter::NotIn(_, _));
            let Some((idx, ty)) = slot_for(schema, specs, attr) else {
                return if is_not {
                    LeafPredicate::True
                } else {
                    LeafPredicate::False
                };
            };
            let has_null = values.iter().any(|v| matches!(v, ScalarValue::Null));
            let normalized: Vec<LeafRhs> = values
                .iter()
                .filter(|v| !matches!(v, ScalarValue::Null))
                .filter_map(|v| normalize_rhs(v, ty))
                .collect();
            let pred = if has_null {
                LeafPredicate::Or(vec![
                    LeafPredicate::IsNull(idx),
                    LeafPredicate::In(idx, normalized),
                ])
            } else {
                LeafPredicate::In(idx, normalized)
            };
            if is_not {
                LeafPredicate::Not(Box::new(pred))
            } else {
                pred
            }
        }
        Filter::Lt(attr, rhs)
        | Filter::Lte(attr, rhs)
        | Filter::Gt(attr, rhs)
        | Filter::Gte(attr, rhs) => {
            let op = match filter {
                Filter::Lt(..) => CmpOp::Lt,
                Filter::Lte(..) => CmpOp::Lte,
                Filter::Gt(..) => CmpOp::Gt,
                _ => CmpOp::Gte,
            };
            let Some((idx, ty)) = slot_for(schema, specs, attr) else {
                return LeafPredicate::False;
            };
            normalize_rhs(rhs, ty).map_or(LeafPredicate::False, |rhs| {
                LeafPredicate::Cmp(idx, op, rhs)
            })
        }
        _ => LeafPredicate::False,
    }
}

fn slot_for(
    schema: &CollectionSchema,
    specs: &[LeafAttrSpec],
    attr: &str,
) -> Option<(usize, AttributeType)> {
    let entry = schema.get_by_name(attr)?;
    let idx = specs
        .binary_search_by_key(&entry.id, |spec| spec.attr_id)
        .ok()?;
    Some((idx, entry.attr_type))
}

fn normalize_rhs(value: &ScalarValue, ty: AttributeType) -> Option<LeafRhs> {
    match (ty, value) {
        (AttributeType::Bool, ScalarValue::Bool(v)) => Some(LeafRhs::Bool(*v)),
        (AttributeType::Int, ScalarValue::Int(v)) => Some(LeafRhs::Int(*v)),
        (AttributeType::Uint, ScalarValue::Uint(v)) => Some(LeafRhs::Uint(*v)),
        (AttributeType::Float, ScalarValue::Float(v)) if !v.is_nan() => Some(LeafRhs::Float(*v)),
        _ => None,
    }
}
