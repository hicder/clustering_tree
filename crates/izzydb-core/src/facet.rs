//! Facet request/response types, validation, disjunctive filter rewriting, and scalar ordering.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};

use crate::error::{Result, TreeError};
use crate::filter::Filter;
use crate::schema::{AttributeType, CollectionSchema, ScalarValue, SchemaEntry};

/// Bucket ordering for [`FacetKind::Terms`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FacetSort {
    /// Descending by count, tie-breaking ascending by value.
    #[default]
    CountDesc,
    /// Ascending by count, tie-breaking ascending by value.
    CountAsc,
    /// Ascending by value order.
    ValueAsc,
    /// Descending by value order.
    ValueDesc,
}

/// One bucket of a [`FacetKind::Ranges`] facet.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetRange {
    pub label: Option<String>,
    /// Lower bound (`None` = -infinity).
    pub from: Option<ScalarValue>,
    /// Whether `from` itself belongs to the bucket (default: `true`).
    pub from_inclusive: bool,
    /// Upper bound (`None` = +infinity).
    pub to: Option<ScalarValue>,
    /// Whether `to` itself belongs to the bucket (default: `false`).
    pub to_inclusive: bool,
}

impl FacetRange {
    /// Half-open range `[from, to)`.
    pub fn between(from: impl Into<ScalarValue>, to: impl Into<ScalarValue>) -> Self {
        Self {
            label: None,
            from: Some(from.into()),
            from_inclusive: true,
            to: Some(to.into()),
            to_inclusive: false,
        }
    }

    /// Open-ended range `[from, +inf)`.
    pub fn from_value(from: impl Into<ScalarValue>) -> Self {
        Self {
            label: None,
            from: Some(from.into()),
            from_inclusive: true,
            to: None,
            to_inclusive: false,
        }
    }

    /// Open-ended range `(-inf, to)`.
    pub fn below(to: impl Into<ScalarValue>) -> Self {
        Self {
            label: None,
            from: None,
            from_inclusive: true,
            to: Some(to.into()),
            to_inclusive: false,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_inclusive(mut self, from_inclusive: bool, to_inclusive: bool) -> Self {
        self.from_inclusive = from_inclusive;
        self.to_inclusive = to_inclusive;
        self
    }
}

/// Which filter a facet is computed under.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum FacetFilterMode {
    /// Respect the query's full filter (conjunctive faceting).
    #[default]
    QueryFilter,
    /// Exclude top-level `Filter::And` clauses referencing only this facet's attribute
    /// (standard multi-select / disjunctive faceting).
    ExcludeSelf,
    /// Exclude top-level `Filter::And` clauses referencing only attributes in this list.
    ExcludeAttributes(Vec<String>),
    /// Override the filter completely for this facet (e.g. `Custom(None)` for global counts).
    Custom(Option<Filter>),
}

/// The aggregation computed by a [`FacetSpec`].
#[derive(Debug, Clone, PartialEq)]
pub enum FacetKind {
    Terms {
        limit: usize,
        sort: FacetSort,
        min_count: u64,
        /// String / StringArray typeahead facet search.
        prefix: Option<String>,
    },
    Ranges {
        ranges: Vec<FacetRange>,
    },
    /// count, missing_count, min, max, sum, avg on numeric scalar attributes (Int/Uint/Float).
    Stats,
    Histogram {
        interval: f64,
        offset: f64,
        min_count: u64,
    },
}

/// A single facet request.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetSpec {
    pub attr: String,
    /// Output key in [`FacetResults`]; defaults to `attr`. Allows multiple facets on one attribute.
    pub alias: Option<String>,
    pub kind: FacetKind,
    /// When `false`, `missing_count` is reported as `0` (and not computed).
    pub include_missing: bool,
    pub filter_mode: FacetFilterMode,
}

impl FacetSpec {
    fn new(attr: impl Into<String>, kind: FacetKind) -> Self {
        Self {
            attr: attr.into(),
            alias: None,
            kind,
            include_missing: false,
            filter_mode: FacetFilterMode::default(),
        }
    }

    /// Terms facet with `limit = 10`, `CountDesc`, `min_count = 1`.
    pub fn terms(attr: impl Into<String>) -> Self {
        Self::new(
            attr,
            FacetKind::Terms {
                limit: 10,
                sort: FacetSort::default(),
                min_count: 1,
                prefix: None,
            },
        )
    }

    pub fn ranges(attr: impl Into<String>, ranges: Vec<FacetRange>) -> Self {
        Self::new(attr, FacetKind::Ranges { ranges })
    }

    pub fn stats(attr: impl Into<String>) -> Self {
        Self::new(attr, FacetKind::Stats)
    }

    pub fn histogram(attr: impl Into<String>, interval: f64) -> Self {
        Self::new(
            attr,
            FacetKind::Histogram {
                interval,
                offset: 0.0,
                min_count: 1,
            },
        )
    }

    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    pub fn with_missing(mut self) -> Self {
        self.include_missing = true;
        self
    }

    pub fn with_filter_mode(mut self, mode: FacetFilterMode) -> Self {
        self.filter_mode = mode;
        self
    }

    pub fn exclude_self(self) -> Self {
        self.with_filter_mode(FacetFilterMode::ExcludeSelf)
    }

    /// Set the `Terms` bucket limit (no-op for other kinds).
    pub fn with_limit(mut self, new_limit: usize) -> Self {
        if let FacetKind::Terms { limit, .. } = &mut self.kind {
            *limit = new_limit;
        }
        self
    }

    /// Set the `Terms` sort (no-op for other kinds).
    pub fn with_sort(mut self, new_sort: FacetSort) -> Self {
        if let FacetKind::Terms { sort, .. } = &mut self.kind {
            *sort = new_sort;
        }
        self
    }

    /// Set the `Terms` / `Histogram` minimum bucket count (no-op for other kinds).
    pub fn with_min_count(mut self, new_min: u64) -> Self {
        match &mut self.kind {
            FacetKind::Terms { min_count, .. } | FacetKind::Histogram { min_count, .. } => {
                *min_count = new_min;
            }
            _ => {}
        }
        self
    }

    /// Set the `Terms` typeahead prefix (no-op for other kinds).
    pub fn with_prefix(mut self, new_prefix: impl Into<String>) -> Self {
        if let FacetKind::Terms { prefix, .. } = &mut self.kind {
            *prefix = Some(new_prefix.into());
        }
        self
    }

    /// Set the `Histogram` offset (no-op for other kinds).
    pub fn with_offset(mut self, new_offset: f64) -> Self {
        if let FacetKind::Histogram { offset, .. } = &mut self.kind {
            *offset = new_offset;
        }
        self
    }

    /// Key under which this facet appears in [`FacetResults`].
    pub fn output_name(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.attr)
    }

    /// Validate this spec against `schema` (does not check duplicate output names).
    pub fn validate(&self, schema: &CollectionSchema) -> Result<()> {
        let entry = self.entry(schema)?;
        let name = &self.attr;
        let ty = entry.attr_type;
        match &self.kind {
            FacetKind::Terms { prefix, .. } => {
                if prefix.is_some()
                    && !matches!(ty, AttributeType::String | AttributeType::StringArray)
                {
                    return Err(invalid(format!(
                        "Terms prefix is only valid on String/StringArray, '{name}' is {ty:?}"
                    )));
                }
            }
            FacetKind::Ranges { ranges } => {
                if ty == AttributeType::Bool {
                    return Err(invalid(format!(
                        "Ranges facet is not valid on Bool attribute '{name}'"
                    )));
                }
                let elem = ty.scalar_type();
                for r in ranges {
                    for bound in [&r.from, &r.to].into_iter().flatten() {
                        validate_bound(name, bound, elem)?;
                    }
                    if let (Some(from), Some(to)) = (&r.from, &r.to) {
                        match cmp_scalar_value(from, to) {
                            Ordering::Less => {}
                            Ordering::Equal if r.from_inclusive && r.to_inclusive => {}
                            _ => {
                                return Err(invalid(format!(
                                    "range on '{name}' must satisfy from < to \
                                     (or from == to with both bounds inclusive)"
                                )));
                            }
                        }
                    }
                }
            }
            FacetKind::Stats => {
                require_numeric_scalar(name, ty, "Stats")?;
            }
            FacetKind::Histogram {
                interval, offset, ..
            } => {
                require_numeric_scalar(name, ty, "Histogram")?;
                if !interval.is_finite() || *interval <= 0.0 {
                    return Err(invalid(format!(
                        "Histogram interval must be finite and > 0.0, got {interval}"
                    )));
                }
                if !offset.is_finite() {
                    return Err(invalid(format!(
                        "Histogram offset must be finite, got {offset}"
                    )));
                }
            }
        }
        match &self.filter_mode {
            FacetFilterMode::Custom(Some(f)) => f.validate(schema)?,
            FacetFilterMode::ExcludeAttributes(attrs) => {
                for a in attrs {
                    if schema.get_by_name(a).is_none() {
                        return Err(invalid(format!(
                            "ExcludeAttributes references unknown attribute '{a}'"
                        )));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Resolve this spec's schema entry, rejecting unknown or non-filterable attributes.
    pub fn entry<'a>(&self, schema: &'a CollectionSchema) -> Result<&'a SchemaEntry> {
        let entry = schema
            .get_by_name(&self.attr)
            .ok_or_else(|| invalid(format!("unknown facet attribute '{}'", self.attr)))?;
        if !entry.filterable {
            return Err(invalid(format!(
                "facet attribute '{}' is not filterable",
                self.attr
            )));
        }
        Ok(entry)
    }
}

/// Validate every spec and reject duplicate output names.
pub fn validate_facet_specs(schema: &CollectionSchema, specs: &[FacetSpec]) -> Result<()> {
    let mut seen: HashSet<&str> = HashSet::with_capacity(specs.len());
    for spec in specs {
        if !seen.insert(spec.output_name()) {
            return Err(invalid(format!(
                "duplicate facet output name '{}'",
                spec.output_name()
            )));
        }
        spec.validate(schema)?;
    }
    Ok(())
}

fn invalid(msg: String) -> TreeError {
    TreeError::InvalidQuery(msg)
}

fn require_numeric_scalar(name: &str, ty: AttributeType, what: &str) -> Result<()> {
    if matches!(
        ty,
        AttributeType::Int | AttributeType::Uint | AttributeType::Float
    ) {
        Ok(())
    } else {
        Err(invalid(format!(
            "{what} facet requires a numeric scalar attribute, '{name}' is {ty:?}"
        )))
    }
}

fn validate_bound(name: &str, bound: &ScalarValue, elem: AttributeType) -> Result<()> {
    if matches!(bound, ScalarValue::Null) {
        return Err(invalid(format!("range bound on '{name}' must not be Null")));
    }
    if !bound.matches_scalar_type(elem) {
        return Err(invalid(format!(
            "range bound {bound:?} does not match type {elem:?} of attribute '{name}'"
        )));
    }
    Ok(())
}

/// Scope of the facet population for vector ANN search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorFacetScope {
    /// All documents matching `filter` (or the whole collection).
    #[default]
    FilterMatches,
    /// Only the returned top-`k` hits.
    TopK,
    /// The top-`n` scored ANN candidates.
    Candidates(usize),
}

/// Scope of the facet population for hybrid RRF search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HybridFacetScope {
    /// All documents matching `filter` (or the whole collection).
    #[default]
    FilterMatches,
    /// Union of the vector and BM25 candidates entering RRF.
    FusedCandidates,
    /// All documents with BM25 score `> 0` for `rank_by` that also match `filter`.
    Bm25AndFilterMatches,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FacetBucket {
    pub value: ScalarValue,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeBucket {
    pub label: Option<String>,
    pub from: Option<ScalarValue>,
    pub to: Option<ScalarValue>,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBucket {
    pub start: f64,
    pub end: f64,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TermFacetResult {
    pub buckets: Vec<FacetBucket>,
    /// Number of distinct values passing `prefix` / `min_count` (before `limit`).
    pub total_values: u64,
    pub missing_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeFacetResult {
    pub buckets: Vec<RangeBucket>,
    pub missing_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatsFacetResult {
    pub count: u64,
    pub missing_count: u64,
    pub min: Option<ScalarValue>,
    pub max: Option<ScalarValue>,
    pub sum: f64,
    pub avg: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistogramFacetResult {
    pub buckets: Vec<HistogramBucket>,
    pub missing_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FacetResult {
    Terms(TermFacetResult),
    Ranges(RangeFacetResult),
    Stats(StatsFacetResult),
    Histogram(HistogramFacetResult),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FacetResults {
    pub facets: BTreeMap<String, FacetResult>,
}

impl FacetResults {
    pub fn get(&self, name: &str) -> Option<&FacetResult> {
        self.facets.get(name)
    }

    pub fn terms(&self, name: &str) -> Option<&TermFacetResult> {
        match self.facets.get(name) {
            Some(FacetResult::Terms(t)) => Some(t),
            _ => None,
        }
    }

    pub fn ranges(&self, name: &str) -> Option<&RangeFacetResult> {
        match self.facets.get(name) {
            Some(FacetResult::Ranges(r)) => Some(r),
            _ => None,
        }
    }

    pub fn stats(&self, name: &str) -> Option<&StatsFacetResult> {
        match self.facets.get(name) {
            Some(FacetResult::Stats(s)) => Some(s),
            _ => None,
        }
    }

    pub fn histogram(&self, name: &str) -> Option<&HistogramFacetResult> {
        match self.facets.get(name) {
            Some(FacetResult::Histogram(h)) => Some(h),
            _ => None,
        }
    }
}

/// Search hits plus facet aggregations.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetedSearchResponse<H> {
    pub hits: Vec<H>,
    pub facets: FacetResults,
    /// Size of the primary facet population (documents matching the query under its scope).
    pub total_hits: u64,
}

fn variant_rank(v: &ScalarValue) -> u8 {
    match v {
        ScalarValue::Null => 0,
        ScalarValue::Bool(_) => 1,
        ScalarValue::Int(_) => 2,
        ScalarValue::Uint(_) => 3,
        ScalarValue::Float(_) => 4,
        ScalarValue::String(_) => 5,
    }
}

/// Total order over [`ScalarValue`]s consistent with the order-preserving key encoding:
/// same-variant values compare naturally (`-0.0 == +0.0`, strings bytewise), and differing
/// variants order by variant (`Null < Bool < Int < Uint < Float < String`).
pub fn cmp_scalar_value(a: &ScalarValue, b: &ScalarValue) -> Ordering {
    match (a, b) {
        (ScalarValue::Bool(x), ScalarValue::Bool(y)) => x.cmp(y),
        (ScalarValue::Int(x), ScalarValue::Int(y)) => x.cmp(y),
        (ScalarValue::Uint(x), ScalarValue::Uint(y)) => x.cmp(y),
        (ScalarValue::Float(x), ScalarValue::Float(y)) => {
            let xn = if *x == 0.0 { 0.0 } else { *x };
            let yn = if *y == 0.0 { 0.0 } else { *y };
            xn.total_cmp(&yn)
        }
        (ScalarValue::String(x), ScalarValue::String(y)) => x.as_bytes().cmp(y.as_bytes()),
        _ => variant_rank(a).cmp(&variant_rank(b)),
    }
}

/// Collect every attribute name referenced by `filter` into `out`.
pub fn collect_filter_attrs<'a>(filter: &'a Filter, out: &mut HashSet<&'a str>) {
    match filter {
        Filter::And(children) | Filter::Or(children) => {
            for c in children {
                collect_filter_attrs(c, out);
            }
        }
        Filter::Not(child) => collect_filter_attrs(child, out),
        Filter::Eq(a, _)
        | Filter::NotEq(a, _)
        | Filter::In(a, _)
        | Filter::NotIn(a, _)
        | Filter::Lt(a, _)
        | Filter::Lte(a, _)
        | Filter::Gt(a, _)
        | Filter::Gte(a, _)
        | Filter::Contains(a, _)
        | Filter::ContainsAny(a, _)
        | Filter::IsNull(a)
        | Filter::IsNotNull(a) => {
            out.insert(a.as_str());
        }
        Filter::ContainsAllTokens { attr, .. }
        | Filter::ContainsAnyToken { attr, .. }
        | Filter::ContainsTokenSequence { attr, .. } => {
            out.insert(attr.as_str());
        }
    }
}

fn flatten_conjuncts<'a>(filter: &'a Filter, out: &mut Vec<&'a Filter>) {
    match filter {
        Filter::And(children) => {
            for c in children {
                flatten_conjuncts(c, out);
            }
        }
        other => out.push(other),
    }
}

/// Rewrite `filter` for disjunctive faceting: flatten top-level `And` conjuncts and drop every
/// conjunct whose referenced attribute set is a non-empty subset of `excluded`. Mixed
/// `Or` / `Not` conjuncts referencing any non-excluded attribute are preserved.
///
/// Returns `None` when no conjuncts remain (i.e. match everything).
pub fn strip_filter_attributes(filter: Option<&Filter>, excluded: &[&str]) -> Option<Filter> {
    let filter = filter?;
    let mut conjuncts = Vec::new();
    flatten_conjuncts(filter, &mut conjuncts);
    let mut kept: Vec<Filter> = Vec::with_capacity(conjuncts.len());
    for c in conjuncts {
        let mut attrs = HashSet::new();
        collect_filter_attrs(c, &mut attrs);
        let drop = !attrs.is_empty() && attrs.iter().all(|a| excluded.contains(a));
        if !drop {
            kept.push(c.clone());
        }
    }
    match kept.len() {
        0 => None,
        1 => kept.pop(),
        _ => Some(Filter::And(kept)),
    }
}

impl FacetSpec {
    /// Effective filter for this facet given the query's `filter`.
    pub fn effective_filter(&self, query_filter: Option<&Filter>) -> Option<Filter> {
        match &self.filter_mode {
            FacetFilterMode::QueryFilter => query_filter.cloned(),
            FacetFilterMode::ExcludeSelf => {
                strip_filter_attributes(query_filter, &[self.attr.as_str()])
            }
            FacetFilterMode::ExcludeAttributes(attrs) => {
                let excl: Vec<&str> = attrs.iter().map(String::as_str).collect();
                strip_filter_attributes(query_filter, &excl)
            }
            FacetFilterMode::Custom(f) => f.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DistanceMetric, TreeConfig};
    use crate::schema::CollectionConfig;

    fn schema() -> CollectionSchema {
        let cfg = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared))
            .with_attribute("cat", AttributeType::String, true)
            .with_attribute("n", AttributeType::Int, true)
            .with_attribute("f", AttributeType::Float, true)
            .with_attribute("b", AttributeType::Bool, true)
            .with_attribute("tags", AttributeType::StringArray, true)
            .with_attribute("hidden", AttributeType::String, false);
        CollectionSchema::from_config(&cfg).unwrap()
    }

    #[test]
    fn validation_rules() {
        let s = schema();
        assert!(FacetSpec::terms("cat").validate(&s).is_ok());
        assert!(FacetSpec::terms("tags").with_prefix("a").validate(&s).is_ok());
        assert!(FacetSpec::terms("n").with_prefix("a").validate(&s).is_err());
        assert!(FacetSpec::terms("nope").validate(&s).is_err());
        assert!(FacetSpec::terms("hidden").validate(&s).is_err());
        assert!(FacetSpec::stats("cat").validate(&s).is_err());
        assert!(FacetSpec::stats("n").validate(&s).is_ok());
        assert!(FacetSpec::histogram("n", 0.0).validate(&s).is_err());
        assert!(FacetSpec::histogram("n", f64::NAN).validate(&s).is_err());
        assert!(FacetSpec::histogram("n", 5.0).validate(&s).is_ok());
        assert!(
            FacetSpec::histogram("n", 5.0)
                .with_offset(f64::INFINITY)
                .validate(&s)
                .is_err()
        );
        assert!(
            FacetSpec::ranges("b", vec![FacetRange::from_value(false)])
                .validate(&s)
                .is_err()
        );
        assert!(
            FacetSpec::ranges("n", vec![FacetRange::between(1i64, 5i64)])
                .validate(&s)
                .is_ok()
        );
        assert!(
            FacetSpec::ranges("n", vec![FacetRange::between(5i64, 1i64)])
                .validate(&s)
                .is_err()
        );
        assert!(
            FacetSpec::ranges("n", vec![FacetRange::between(1i64, 1i64)])
                .validate(&s)
                .is_err()
        );
        assert!(
            FacetSpec::ranges(
                "n",
                vec![FacetRange::between(1i64, 1i64).with_inclusive(true, true)]
            )
            .validate(&s)
            .is_ok()
        );
        assert!(
            FacetSpec::ranges("n", vec![FacetRange::between("a", "b")])
                .validate(&s)
                .is_err()
        );
        assert!(
            FacetSpec::ranges("f", vec![FacetRange::from_value(f64::NAN)])
                .validate(&s)
                .is_err()
        );
        assert!(
            FacetSpec::terms("cat")
                .with_filter_mode(FacetFilterMode::ExcludeAttributes(vec!["nope".into()]))
                .validate(&s)
                .is_err()
        );
        assert!(
            FacetSpec::terms("cat")
                .with_filter_mode(FacetFilterMode::Custom(Some(Filter::Eq(
                    "nope".into(),
                    1i64.into()
                ))))
                .validate(&s)
                .is_err()
        );
    }

    #[test]
    fn duplicate_output_names_rejected() {
        let s = schema();
        let dup = [FacetSpec::terms("cat"), FacetSpec::stats("n").with_alias("cat")];
        assert!(validate_facet_specs(&s, &dup).is_err());
        let ok = [
            FacetSpec::terms("cat"),
            FacetSpec::terms("cat").with_alias("cat2"),
        ];
        assert!(validate_facet_specs(&s, &ok).is_ok());
    }

    #[test]
    fn strip_attributes() {
        let a = Filter::Eq("cat".into(), "x".into());
        let b = Filter::Gt("n".into(), 3i64.into());
        let mixed = Filter::Or(vec![a.clone(), b.clone()]);
        let q = Filter::And(vec![a.clone(), Filter::And(vec![b.clone(), mixed.clone()])]);

        assert_eq!(strip_filter_attributes(None, &["cat"]), None);
        assert_eq!(
            strip_filter_attributes(Some(&q), &["cat"]),
            Some(Filter::And(vec![b.clone(), mixed.clone()]))
        );
        assert_eq!(
            strip_filter_attributes(Some(&q), &["cat", "n"]),
            None
        );
        assert_eq!(strip_filter_attributes(Some(&a), &["cat"]), None);
        assert_eq!(strip_filter_attributes(Some(&a), &["n"]), Some(a.clone()));
        assert_eq!(
            strip_filter_attributes(Some(&q), &[]),
            Some(Filter::And(vec![a, b, mixed]))
        );
    }

    #[test]
    fn scalar_ordering() {
        use ScalarValue::*;
        assert_eq!(cmp_scalar_value(&Float(-0.0), &Float(0.0)), Ordering::Equal);
        assert_eq!(cmp_scalar_value(&Int(i64::MIN), &Int(i64::MAX)), Ordering::Less);
        assert_eq!(
            cmp_scalar_value(&String("a\0b".into()), &String("a".into())),
            Ordering::Greater
        );
        assert_eq!(cmp_scalar_value(&Null, &Bool(false)), Ordering::Less);
        assert_eq!(cmp_scalar_value(&Uint(1), &Int(100)), Ordering::Greater);
    }

    #[test]
    fn effective_filters() {
        let a = Filter::Eq("cat".into(), "x".into());
        let b = Filter::Gt("n".into(), 3i64.into());
        let q = Filter::And(vec![a.clone(), b.clone()]);
        assert_eq!(
            FacetSpec::terms("cat").effective_filter(Some(&q)),
            Some(q.clone())
        );
        assert_eq!(
            FacetSpec::terms("cat").exclude_self().effective_filter(Some(&q)),
            Some(b.clone())
        );
        assert_eq!(
            FacetSpec::terms("cat")
                .with_filter_mode(FacetFilterMode::Custom(None))
                .effective_filter(Some(&q)),
            None
        );
    }
}
