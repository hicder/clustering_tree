use crate::config::DocId;
use crate::error::{Result, TreeError};
use crate::filter::{Filter, TokenQuery, require_fts, validate_token_query};
use crate::schema::{AttributeType, CollectionSchema, ScalarValue};

/// Optional query-time BM25 parameter overrides.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bm25Overrides {
    pub last_as_prefix: Option<bool>,
    pub k1: Option<f32>,
    pub b: Option<f32>,
    pub k3: Option<f32>,
    pub max_prefix_expansions: Option<usize>,
}

impl Bm25Overrides {
    pub fn with_last_as_prefix(mut self, last_as_prefix: bool) -> Self {
        self.last_as_prefix = Some(last_as_prefix);
        self
    }

    pub fn with_k1(mut self, k1: f32) -> Self {
        self.k1 = Some(k1);
        self
    }

    pub fn with_b(mut self, b: f32) -> Self {
        self.b = Some(b);
        self
    }

    pub fn with_k3(mut self, k3: f32) -> Self {
        self.k3 = Some(k3);
        self
    }

    pub fn with_max_prefix_expansions(mut self, max_prefix_expansions: usize) -> Self {
        self.max_prefix_expansions = Some(max_prefix_expansions);
        self
    }

    pub fn validate(&self) -> Result<()> {
        if let Some(k1) = self.k1
            && (!k1.is_finite() || k1 <= 0.0)
        {
            return Err(TreeError::InvalidQuery(format!(
                "BM25 k1 override must be finite and > 0.0, got {k1}"
            )));
        }
        if let Some(b) = self.b
            && (!b.is_finite() || !(0.0..=1.0).contains(&b))
        {
            return Err(TreeError::InvalidQuery(format!(
                "BM25 b override must be in 0.0..=1.0, got {b}"
            )));
        }
        if let Some(k3) = self.k3
            && (!k3.is_finite() || k3 < 0.0)
        {
            return Err(TreeError::InvalidQuery(format!(
                "BM25 k3 override must be finite and >= 0.0, got {k3}"
            )));
        }
        if let Some(max_exp) = self.max_prefix_expansions
            && max_exp == 0
        {
            return Err(TreeError::InvalidQuery(
                "max_prefix_expansions must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

/// Distance calculation mode for [`RankBy::Decay`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecayMode {
    /// `d = |x - origin|` (symmetric around `origin`, e.g. a target price).
    #[default]
    Symmetric,
    /// `d = (origin - x).max(0.0)`: penalizes values below `origin` (e.g. older timestamps);
    /// values `>= origin` score `1.0`.
    Backward,
    /// `d = (x - origin).max(0.0)`: penalizes values above `origin` (e.g. error counts);
    /// values `<= origin` score `1.0`.
    Forward,
}

/// Ranking expression AST for BM25, filter boosts, attribute transforms, and composite scoring.
///
/// Every expression evaluates to a finite score `>= 0.0`; documents scoring `<= 0.0` are excluded
/// from results. Ties break by ascending `DocId`.
#[derive(Debug, Clone, PartialEq)]
pub enum RankBy {
    /// Constant non-negative scalar (weights and additive bases such as `1.0 + w * Decay(..)`).
    Const(f32),
    /// BM25 full-text score over an FTS-indexed attribute.
    Bm25 {
        attr: String,
        query: TokenQuery,
        params: Option<Bm25Overrides>,
    },
    /// Filter-as-boost: `1.0` if the filter matches the document, `0.0` otherwise.
    Filter(Box<Filter>),
    /// Raw numeric attribute value (`Int` / `Uint` / `Float`), clamped to `>= 0.0`.
    Attribute {
        attr: String,
        /// Score used when the attribute is absent or `Null`.
        missing: f32,
    },
    /// Monotone increasing saturation in `[0, 1]`: `x^e / (x^e + midpoint^e)`, `x = val.max(0)`.
    Saturate {
        attr: String,
        midpoint: f64,
        exponent: f64,
        missing: f32,
    },
    /// Monotone decreasing decay in `[0, 1]`: `midpoint^e / (d^e + midpoint^e)` where `d` is the
    /// distance from `origin` per `mode`.
    Decay {
        attr: String,
        origin: f64,
        midpoint: f64,
        exponent: f64,
        mode: DecayMode,
        missing: f32,
    },
    Sum(Vec<RankBy>),
    Max(Vec<RankBy>),
    /// Product of 1+ sub-expressions.
    Product(Vec<RankBy>),
}

impl RankBy {
    pub fn constant(v: f32) -> Self {
        Self::Const(v)
    }

    pub fn bm25(attr: impl Into<String>, query: impl Into<TokenQuery>) -> Self {
        Self::Bm25 {
            attr: attr.into(),
            query: query.into(),
            params: None,
        }
    }

    pub fn bm25_with(
        attr: impl Into<String>,
        query: impl Into<TokenQuery>,
        params: Bm25Overrides,
    ) -> Self {
        Self::Bm25 {
            attr: attr.into(),
            query: query.into(),
            params: Some(params),
        }
    }

    pub fn sum(exprs: Vec<RankBy>) -> Self {
        Self::Sum(exprs)
    }

    pub fn max(exprs: Vec<RankBy>) -> Self {
        Self::Max(exprs)
    }

    /// `weight * expr`.
    pub fn product(weight: f32, expr: RankBy) -> Self {
        Self::Product(vec![Self::Const(weight), expr])
    }

    /// Product of all `exprs`.
    pub fn product_all(exprs: Vec<RankBy>) -> Self {
        Self::Product(exprs)
    }

    pub fn filter(filter: Filter) -> Self {
        Self::Filter(Box::new(filter))
    }

    pub fn attribute(attr: impl Into<String>) -> Self {
        Self::attribute_with_missing(attr, 0.0)
    }

    pub fn attribute_with_missing(attr: impl Into<String>, missing: f32) -> Self {
        Self::Attribute {
            attr: attr.into(),
            missing,
        }
    }

    pub fn saturate(attr: impl Into<String>, midpoint: f64) -> Self {
        Self::Saturate {
            attr: attr.into(),
            midpoint,
            exponent: 1.0,
            missing: 0.0,
        }
    }

    /// Distance from `0.0` penalizing larger values ([`DecayMode::Forward`], origin `0.0`).
    pub fn decay(attr: impl Into<String>, midpoint: f64) -> Self {
        Self::Decay {
            attr: attr.into(),
            origin: 0.0,
            midpoint,
            exponent: 1.0,
            mode: DecayMode::Forward,
            missing: 0.0,
        }
    }

    /// Symmetric decay around `origin`.
    pub fn decay_from(attr: impl Into<String>, origin: f64, midpoint: f64) -> Self {
        Self::Decay {
            attr: attr.into(),
            origin,
            midpoint,
            exponent: 1.0,
            mode: DecayMode::Symmetric,
            missing: 0.0,
        }
    }

    /// Override the exponent of a `Saturate` / `Decay` node (no-op for other variants).
    pub fn with_exponent(mut self, e: f64) -> Self {
        if let Self::Saturate { exponent, .. } | Self::Decay { exponent, .. } = &mut self {
            *exponent = e;
        }
        self
    }

    /// Override the `missing` score of an `Attribute` / `Saturate` / `Decay` node.
    pub fn with_missing(mut self, m: f32) -> Self {
        if let Self::Attribute { missing, .. }
        | Self::Saturate { missing, .. }
        | Self::Decay { missing, .. } = &mut self
        {
            *missing = m;
        }
        self
    }

    /// Override the mode of a `Decay` node (no-op for other variants).
    pub fn with_decay_mode(mut self, m: DecayMode) -> Self {
        if let Self::Decay { mode, .. } = &mut self {
            *mode = m;
        }
        self
    }

    /// `base * (1 + weight * signal)`.
    pub fn boost_multiplicative(base: RankBy, weight: f32, signal: RankBy) -> Self {
        Self::Product(vec![
            base,
            Self::Sum(vec![
                Self::Const(1.0),
                Self::Product(vec![Self::Const(weight), signal]),
            ]),
        ])
    }

    /// Whether any node of this expression is a [`RankBy::Bm25`].
    pub fn has_bm25(&self) -> bool {
        match self {
            Self::Bm25 { .. } => true,
            Self::Sum(c) | Self::Max(c) | Self::Product(c) => c.iter().any(Self::has_bm25),
            _ => false,
        }
    }

    /// Whether this expression is a "pure lexical" expression: BM25 leaves combined only with
    /// `Sum` / `Max` and constant weights. Such expressions run on the Block-Max MAXSCORE engine
    /// unchanged. A `Product` is pure lexical only when it has exactly one non-constant factor.
    pub fn is_pure_lexical(&self) -> bool {
        match self {
            Self::Bm25 { .. } => true,
            Self::Sum(c) | Self::Max(c) => !c.is_empty() && c.iter().all(Self::is_pure_lexical),
            Self::Product(c) => {
                let mut non_const = c.iter().filter(|e| !matches!(e, Self::Const(_)));
                match (non_const.next(), non_const.next()) {
                    (Some(e), None) => e.is_pure_lexical(),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Canonicalize the expression: flatten nested `Sum` / `Max` / `Product`, fold constants,
    /// prune zero terms, and distribute a constant weight over a `Sum`.
    ///
    /// The result is semantically identical (scores are bit-for-bit equal up to `f32`
    /// re-association of constant folding). Assumes the expression is valid (finite, `>= 0`).
    pub fn normalize(&self) -> RankBy {
        match self {
            Self::Sum(children) => {
                let mut flat = Vec::new();
                let mut konst = 0.0f32;
                for c in children {
                    match c.normalize() {
                        Self::Sum(inner) => {
                            for i in inner {
                                match i {
                                    Self::Const(v) => konst += v,
                                    other => flat.push(other),
                                }
                            }
                        }
                        Self::Const(v) => konst += v,
                        other => flat.push(other),
                    }
                }
                if konst > 0.0 {
                    flat.push(Self::Const(konst));
                }
                match flat.len() {
                    0 => Self::Const(0.0),
                    1 => flat.pop().unwrap(),
                    _ => Self::Sum(flat),
                }
            }
            Self::Max(children) => {
                let mut flat = Vec::new();
                let mut konst = 0.0f32;
                for c in children {
                    match c.normalize() {
                        Self::Max(inner) => {
                            for i in inner {
                                match i {
                                    Self::Const(v) => konst = konst.max(v),
                                    other => flat.push(other),
                                }
                            }
                        }
                        Self::Const(v) => konst = konst.max(v),
                        other => flat.push(other),
                    }
                }
                if konst > 0.0 {
                    flat.push(Self::Const(konst));
                }
                match flat.len() {
                    0 => Self::Const(0.0),
                    1 => flat.pop().unwrap(),
                    _ => Self::Max(flat),
                }
            }
            Self::Product(children) => {
                let mut flat = Vec::new();
                let mut scalar = 1.0f32;
                for c in children {
                    match c.normalize() {
                        Self::Product(inner) => {
                            for i in inner {
                                match i {
                                    Self::Const(v) => scalar *= v,
                                    other => flat.push(other),
                                }
                            }
                        }
                        Self::Const(v) => scalar *= v,
                        other => flat.push(other),
                    }
                }
                if scalar == 0.0 || children.is_empty() {
                    return Self::Const(0.0);
                }
                if flat.is_empty() {
                    return Self::Const(scalar);
                }
                // `w * Sum(xs)` -> `Sum(w * x)` so additive terms are visible at the top level.
                if flat.len() == 1 && scalar != 1.0 && matches!(flat[0], Self::Sum(_)) {
                    let Self::Sum(xs) = flat.pop().unwrap() else {
                        unreachable!()
                    };
                    let scaled: Vec<RankBy> = xs
                        .into_iter()
                        .map(|x| Self::Product(vec![Self::Const(scalar), x]).normalize())
                        .collect();
                    return Self::Sum(scaled).normalize();
                }
                if scalar == 1.0 && flat.len() == 1 {
                    return flat.pop().unwrap();
                }
                let mut out = Vec::with_capacity(flat.len() + 1);
                if scalar != 1.0 {
                    out.push(Self::Const(scalar));
                }
                out.extend(flat);
                Self::Product(out)
            }
            other => other.clone(),
        }
    }

    /// Validate all referenced attributes, token queries, filters, weights, and BM25 overrides.
    pub fn validate(&self, schema: &CollectionSchema) -> Result<()> {
        match self {
            Self::Const(v) => {
                if !v.is_finite() || *v < 0.0 {
                    return Err(TreeError::InvalidQuery(format!(
                        "RankBy::Const must be finite and >= 0.0, got {v}"
                    )));
                }
                Ok(())
            }
            Self::Bm25 {
                attr,
                query,
                params,
            } => {
                let (_, fts) = require_fts(schema, attr).map_err(filter_to_query_err)?;
                validate_token_query(attr, query, fts).map_err(filter_to_query_err)?;
                if let Some(p) = params {
                    p.validate()?;
                }
                Ok(())
            }
            Self::Filter(f) => f.validate(schema).map_err(filter_to_query_err),
            Self::Attribute { attr, missing } => {
                require_numeric(schema, attr)?;
                validate_missing(*missing)
            }
            Self::Saturate {
                attr,
                midpoint,
                exponent,
                missing,
            } => {
                require_numeric(schema, attr)?;
                validate_missing(*missing)?;
                validate_positive("Saturate midpoint", *midpoint)?;
                validate_positive("Saturate exponent", *exponent)
            }
            Self::Decay {
                attr,
                origin,
                midpoint,
                exponent,
                missing,
                ..
            } => {
                require_numeric(schema, attr)?;
                validate_missing(*missing)?;
                if !origin.is_finite() {
                    return Err(TreeError::InvalidQuery(format!(
                        "Decay origin must be finite, got {origin}"
                    )));
                }
                validate_positive("Decay midpoint", *midpoint)?;
                validate_positive("Decay exponent", *exponent)
            }
            Self::Sum(children) | Self::Max(children) => {
                for child in children {
                    child.validate(schema)?;
                }
                Ok(())
            }
            Self::Product(children) => {
                if children.is_empty() {
                    return Err(TreeError::InvalidQuery(
                        "RankBy::Product requires at least one factor".into(),
                    ));
                }
                for child in children {
                    child.validate(schema)?;
                }
                Ok(())
            }
        }
    }
}

fn filter_to_query_err(e: TreeError) -> TreeError {
    match e {
        TreeError::InvalidFilter(msg) => TreeError::InvalidQuery(msg),
        other => other,
    }
}

fn require_numeric(schema: &CollectionSchema, attr: &str) -> Result<()> {
    let entry = schema
        .get_by_name(attr)
        .ok_or_else(|| TreeError::InvalidQuery(format!("unknown attribute '{attr}'")))?;
    if !matches!(
        entry.attr_type,
        AttributeType::Int | AttributeType::Uint | AttributeType::Float
    ) {
        return Err(TreeError::InvalidQuery(format!(
            "attribute '{attr}' must be a scalar Int/Uint/Float for numeric ranking, got {:?}",
            entry.attr_type
        )));
    }
    Ok(())
}

fn validate_missing(missing: f32) -> Result<()> {
    if !missing.is_finite() || missing < 0.0 {
        return Err(TreeError::InvalidQuery(format!(
            "`missing` score must be finite and >= 0.0, got {missing}"
        )));
    }
    Ok(())
}

fn validate_positive(what: &str, v: f64) -> Result<()> {
    if !v.is_finite() || v <= 0.0 {
        return Err(TreeError::InvalidQuery(format!(
            "{what} must be finite and > 0.0, got {v}"
        )));
    }
    Ok(())
}

/// `x^e / (x^e + midpoint^e)` for `x = val.max(0)`, in `[0, 1]`. Overflow-safe for large `val`.
pub fn eval_saturate(val: f64, midpoint: f64, exponent: f64) -> f32 {
    if val.is_nan() || val <= 0.0 {
        return 0.0;
    }
    if exponent == 1.0 {
        return (val / (val + midpoint)) as f32;
    }
    let r = val / midpoint;
    (1.0 / (1.0 + r.powf(-exponent))) as f32
}

/// `midpoint^e / (d^e + midpoint^e)` where `d` is the distance of `val` from `origin` per `mode`,
/// in `[0, 1]`. Overflow-safe for large distances.
pub fn eval_decay(val: f64, origin: f64, midpoint: f64, exponent: f64, mode: DecayMode) -> f32 {
    let d = match mode {
        DecayMode::Symmetric => (val - origin).abs(),
        DecayMode::Backward => (origin - val).max(0.0),
        DecayMode::Forward => (val - origin).max(0.0),
    };
    if d.is_nan() {
        return 0.0;
    }
    if d <= 0.0 {
        return 1.0;
    }
    if exponent == 1.0 {
        return (midpoint / (d + midpoint)) as f32;
    }
    let r = d / midpoint;
    (1.0 / (1.0 + r.powf(exponent))) as f32
}

/// Sort direction for [`OrderBySpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Placement of documents whose sort attribute is absent or `Null`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NullsOrder {
    #[default]
    Last,
    First,
}

/// One sort key of an attribute-ordered query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderBySpec {
    pub attr: String,
    pub direction: SortDirection,
    pub nulls: NullsOrder,
}

impl OrderBySpec {
    pub fn asc(attr: impl Into<String>) -> Self {
        Self {
            attr: attr.into(),
            direction: SortDirection::Asc,
            nulls: NullsOrder::Last,
        }
    }

    pub fn desc(attr: impl Into<String>) -> Self {
        Self {
            attr: attr.into(),
            direction: SortDirection::Desc,
            nulls: NullsOrder::Last,
        }
    }

    pub fn with_nulls(mut self, nulls: NullsOrder) -> Self {
        self.nulls = nulls;
        self
    }
}

/// One result of an attribute-ordered query.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderedResult {
    pub id: DocId,
    /// Resolved sort key values in the order of the requested specs (`Null` if absent).
    pub sort_values: Vec<ScalarValue>,
}

/// Validate `order_by` specs: non-empty, attributes exist and are ordered scalars, and the
/// primary key is `filterable` (it drives the index-order scan).
pub fn validate_order_by(schema: &CollectionSchema, order_by: &[OrderBySpec]) -> Result<()> {
    if order_by.is_empty() {
        return Err(TreeError::InvalidQuery(
            "order_by must contain at least one key".into(),
        ));
    }
    for (i, spec) in order_by.iter().enumerate() {
        let entry = schema.get_by_name(&spec.attr).ok_or_else(|| {
            TreeError::InvalidQuery(format!("unknown order_by attribute '{}'", spec.attr))
        })?;
        if entry.attr_type.is_array() || !entry.attr_type.is_ordered() {
            return Err(TreeError::InvalidQuery(format!(
                "order_by attribute '{}' must be an ordered scalar (Int/Uint/Float/String), got {:?}",
                spec.attr, entry.attr_type
            )));
        }
        if i == 0 && !entry.filterable {
            return Err(TreeError::InvalidQuery(format!(
                "primary order_by attribute '{}' must be filterable",
                spec.attr
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bm(a: &str) -> RankBy {
        RankBy::bm25(a, "x")
    }

    #[test]
    fn saturate_curve() {
        assert_eq!(eval_saturate(-5.0, 10.0, 1.0), 0.0);
        assert_eq!(eval_saturate(0.0, 10.0, 1.0), 0.0);
        assert!((eval_saturate(10.0, 10.0, 1.0) - 0.5).abs() < 1e-6);
        assert!((eval_saturate(10.0, 10.0, 2.0) - 0.5).abs() < 1e-6);
        assert!((eval_saturate(30.0, 10.0, 2.0) - 0.9).abs() < 1e-6);
        // Unix-timestamp scale with a high exponent must not overflow to NaN.
        let v = eval_saturate(1.7e9, 1.0, 50.0);
        assert!(v.is_finite() && (v - 1.0).abs() < 1e-6);
        assert!(eval_saturate(1e300, 1e-300, 8.0) <= 1.0);
    }

    #[test]
    fn decay_modes() {
        assert_eq!(eval_decay(5.0, 10.0, 3.0, 1.0, DecayMode::Forward), 1.0);
        assert!((eval_decay(13.0, 10.0, 3.0, 1.0, DecayMode::Forward) - 0.5).abs() < 1e-6);
        assert_eq!(eval_decay(15.0, 10.0, 3.0, 1.0, DecayMode::Backward), 1.0);
        assert!((eval_decay(7.0, 10.0, 3.0, 1.0, DecayMode::Backward) - 0.5).abs() < 1e-6);
        assert!((eval_decay(7.0, 10.0, 3.0, 1.0, DecayMode::Symmetric) - 0.5).abs() < 1e-6);
        assert!((eval_decay(13.0, 10.0, 3.0, 2.0, DecayMode::Symmetric) - 0.5).abs() < 1e-6);
        assert_eq!(eval_decay(10.0, 10.0, 3.0, 2.0, DecayMode::Symmetric), 1.0);
        let far = eval_decay(1e18, 0.0, 1.0, 40.0, DecayMode::Forward);
        assert!(far.is_finite() && (0.0..=1.0).contains(&far));
    }

    #[test]
    fn normalize_folds_and_distributes() {
        let e = RankBy::Product(vec![
            RankBy::Const(2.0),
            RankBy::Product(vec![RankBy::Const(3.0), bm("a")]),
        ]);
        assert_eq!(e.normalize(), RankBy::Product(vec![RankBy::Const(6.0), bm("a")]));

        let e = RankBy::Product(vec![RankBy::Const(0.0), bm("a")]);
        assert_eq!(e.normalize(), RankBy::Const(0.0));

        let e = RankBy::Product(vec![RankBy::Const(2.0), RankBy::Sum(vec![bm("a"), bm("b")])]);
        assert_eq!(
            e.normalize(),
            RankBy::Sum(vec![
                RankBy::Product(vec![RankBy::Const(2.0), bm("a")]),
                RankBy::Product(vec![RankBy::Const(2.0), bm("b")]),
            ])
        );

        let e = RankBy::Sum(vec![
            RankBy::Const(1.0),
            RankBy::Sum(vec![bm("a"), RankBy::Const(2.0)]),
            RankBy::Const(0.0),
        ]);
        assert_eq!(e.normalize(), RankBy::Sum(vec![bm("a"), RankBy::Const(3.0)]));
        assert_eq!(RankBy::Product(vec![bm("a")]).normalize(), bm("a"));
    }

    #[test]
    fn pure_lexical_detection() {
        assert!(bm("a").is_pure_lexical());
        assert!(RankBy::product(2.0, RankBy::Sum(vec![bm("a"), bm("b")])).is_pure_lexical());
        assert!(!RankBy::Product(vec![bm("a"), bm("b")]).is_pure_lexical());
        assert!(!RankBy::Sum(vec![bm("a"), RankBy::Const(1.0)]).is_pure_lexical());
        assert!(!RankBy::attribute("n").is_pure_lexical());
    }
}
