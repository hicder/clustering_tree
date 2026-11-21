use std::cmp::Ordering;

use crate::error::{Result, TreeError};
use crate::schema::{
    AttributeType, AttributeValue, Attributes, CollectionSchema, FullTextSearchConfig, ScalarValue,
    SchemaEntry, Tokenizer,
};

/// Query input for full-text search filters and `RankBy::Bm25`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenQuery {
    /// Raw text string to analyze using the attribute's configured FTS analyzer.
    Text(String),
    /// Pre-tokenized token list used verbatim (required for `Tokenizer::PreTokenizedArray`).
    PreTokenized(Vec<String>),
}

impl From<&str> for TokenQuery {
    fn from(s: &str) -> Self {
        Self::Text(s.to_owned())
    }
}

impl From<String> for TokenQuery {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<Vec<String>> for TokenQuery {
    fn from(tokens: Vec<String>) -> Self {
        Self::PreTokenized(tokens)
    }
}

/// FTS filter matching operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FtsPredicateKind {
    AllTokens,
    AnyToken,
    TokenSequence,
}

/// Boolean filter AST evaluated jointly over cluster summaries, per-leaf postings, and FTS indexes.
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
    Eq(String, ScalarValue),
    NotEq(String, ScalarValue),
    In(String, Vec<ScalarValue>),
    NotIn(String, Vec<ScalarValue>),
    Lt(String, ScalarValue),
    Lte(String, ScalarValue),
    Gt(String, ScalarValue),
    Gte(String, ScalarValue),
    Contains(String, ScalarValue),
    ContainsAny(String, Vec<ScalarValue>),
    IsNull(String),
    IsNotNull(String),
    ContainsAllTokens {
        attr: String,
        query: TokenQuery,
        last_as_prefix: bool,
    },
    ContainsAnyToken {
        attr: String,
        query: TokenQuery,
        last_as_prefix: bool,
    },
    ContainsTokenSequence {
        attr: String,
        query: TokenQuery,
        last_as_prefix: bool,
    },
}

impl Filter {
    /// Returns `true` if this filter tree contains any FTS token/sequence predicate.
    pub fn has_fts(&self) -> bool {
        match self {
            Self::And(children) | Self::Or(children) => children.iter().any(Self::has_fts),
            Self::Not(child) => child.has_fts(),
            Self::ContainsAllTokens { .. }
            | Self::ContainsAnyToken { .. }
            | Self::ContainsTokenSequence { .. } => true,
            _ => false,
        }
    }

    /// Returns `true` if this filter tree contains only FTS predicates and boolean combinators.
    pub fn is_fts_only(&self) -> bool {
        match self {
            Self::And(children) | Self::Or(children) => children.iter().all(Self::is_fts_only),
            Self::Not(child) => child.is_fts_only(),
            Self::ContainsAllTokens { .. }
            | Self::ContainsAnyToken { .. }
            | Self::ContainsTokenSequence { .. } => true,
            _ => false,
        }
    }

    /// Validate this filter AST against `schema`, checking attribute existence,
    /// `filterable` / `full_text_search` flags, and type compatibility.
    pub fn validate(&self, schema: &CollectionSchema) -> Result<()> {
        match self {
            Self::And(children) | Self::Or(children) => {
                for child in children {
                    child.validate(schema)?;
                }
                Ok(())
            }
            Self::Not(child) => child.validate(schema),
            Self::Eq(attr, val) | Self::NotEq(attr, val) => {
                let entry = require_filterable(schema, attr)?;
                if entry.attr_type.is_array() {
                    return Err(TreeError::InvalidFilter(format!(
                        "Eq/NotEq cannot be used on array attribute '{attr}'; use Contains"
                    )));
                }
                validate_scalar(attr, val, entry.attr_type, true)
            }
            Self::In(attr, vals) | Self::NotIn(attr, vals) => {
                let entry = require_filterable(schema, attr)?;
                if entry.attr_type.is_array() {
                    return Err(TreeError::InvalidFilter(format!(
                        "In/NotIn cannot be used on array attribute '{attr}'; use ContainsAny"
                    )));
                }
                for val in vals {
                    validate_scalar(attr, val, entry.attr_type, true)?;
                }
                Ok(())
            }
            Self::Lt(attr, val)
            | Self::Lte(attr, val)
            | Self::Gt(attr, val)
            | Self::Gte(attr, val) => {
                let entry = require_filterable(schema, attr)?;
                if !entry.attr_type.is_ordered() {
                    return Err(TreeError::InvalidFilter(format!(
                        "range filter cannot be used on attribute '{attr}' of type {:?}",
                        entry.attr_type
                    )));
                }
                validate_scalar(attr, val, entry.attr_type, false)
            }
            Self::Contains(attr, val) => {
                let entry = require_filterable(schema, attr)?;
                if !entry.attr_type.is_array() {
                    return Err(TreeError::InvalidFilter(format!(
                        "Contains requires array attribute, but '{attr}' is {:?}",
                        entry.attr_type
                    )));
                }
                validate_scalar(attr, val, entry.attr_type.scalar_type(), false)
            }
            Self::ContainsAny(attr, vals) => {
                let entry = require_filterable(schema, attr)?;
                if !entry.attr_type.is_array() {
                    return Err(TreeError::InvalidFilter(format!(
                        "ContainsAny requires array attribute, but '{attr}' is {:?}",
                        entry.attr_type
                    )));
                }
                let elem_type = entry.attr_type.scalar_type();
                for val in vals {
                    validate_scalar(attr, val, elem_type, false)?;
                }
                Ok(())
            }
            Self::IsNull(attr) | Self::IsNotNull(attr) => {
                require_filterable(schema, attr)?;
                Ok(())
            }
            Self::ContainsAllTokens { attr, query, .. }
            | Self::ContainsAnyToken { attr, query, .. } => {
                let (_, fts) = require_fts(schema, attr)?;
                validate_token_query(attr, query, fts)
            }
            Self::ContainsTokenSequence { attr, query, .. } => {
                let (_, fts) = require_fts(schema, attr)?;
                if !fts.index_positions {
                    return Err(TreeError::InvalidFilter(format!(
                        "ContainsTokenSequence requires index_positions = true on attribute '{attr}'"
                    )));
                }
                validate_token_query(attr, query, fts)
            }
        }
    }

    /// Evaluate this filter in memory against a document's forward attributes.
    ///
    /// Note: FTS predicates (`ContainsAllTokens`, `ContainsAnyToken`, `ContainsTokenSequence`)
    /// require an analyzer and schema; when evaluated via this method without an FTS callback,
    /// they return `false`. Use [`Filter::matches_with_fts`] when FTS predicates may be present.
    pub fn matches(&self, attrs: &Attributes) -> bool {
        self.matches_with_fts(attrs, &mut |_, _, _, _| false)
    }

    /// Evaluate this filter in memory against `attrs`, delegating FTS predicates to `eval_fts`.
    pub fn matches_with_fts<F>(&self, attrs: &Attributes, eval_fts: &mut F) -> bool
    where
        F: FnMut(FtsPredicateKind, &str, &TokenQuery, bool) -> bool,
    {
        match self {
            Self::And(children) => children
                .iter()
                .all(|c| c.matches_with_fts(attrs, eval_fts)),
            Self::Or(children) => children
                .iter()
                .any(|c| c.matches_with_fts(attrs, eval_fts)),
            Self::Not(child) => !child.matches_with_fts(attrs, eval_fts),
            Self::Eq(attr, target) => match (attrs.get(attr), target) {
                (None | Some(AttributeValue::Null), ScalarValue::Null) => true,
                (Some(val), target) => attr_eq_scalar(val, target),
                (None, _) => false,
            },
            Self::NotEq(attr, target) => !Self::Eq(attr.clone(), target.clone()).matches(attrs),
            Self::In(attr, targets) => match attrs.get(attr) {
                None | Some(AttributeValue::Null) => {
                    targets.iter().any(|t| matches!(t, ScalarValue::Null))
                }
                Some(val) => targets.iter().any(|t| attr_eq_scalar(val, t)),
            },
            Self::NotIn(attr, targets) => !Self::In(attr.clone(), targets.clone()).matches(attrs),
            Self::Lt(attr, target) => compare_attr_scalar(attrs.get(attr), target)
                .is_some_and(|ord| ord == Ordering::Less),
            Self::Lte(attr, target) => compare_attr_scalar(attrs.get(attr), target)
                .is_some_and(|ord| ord != Ordering::Greater),
            Self::Gt(attr, target) => compare_attr_scalar(attrs.get(attr), target)
                .is_some_and(|ord| ord == Ordering::Greater),
            Self::Gte(attr, target) => compare_attr_scalar(attrs.get(attr), target)
                .is_some_and(|ord| ord != Ordering::Less),
            Self::Contains(attr, target) => attr_contains_scalar(attrs.get(attr), target),
            Self::ContainsAny(attr, targets) => {
                targets.iter().any(|t| attr_contains_scalar(attrs.get(attr), t))
            }
            Self::IsNull(attr) => matches!(attrs.get(attr), None | Some(AttributeValue::Null)),
            Self::IsNotNull(attr) => !matches!(attrs.get(attr), None | Some(AttributeValue::Null)),
            Self::ContainsAllTokens {
                attr,
                query,
                last_as_prefix,
            } => eval_fts(FtsPredicateKind::AllTokens, attr, query, *last_as_prefix),
            Self::ContainsAnyToken {
                attr,
                query,
                last_as_prefix,
            } => eval_fts(FtsPredicateKind::AnyToken, attr, query, *last_as_prefix),
            Self::ContainsTokenSequence {
                attr,
                query,
                last_as_prefix,
            } => eval_fts(
                FtsPredicateKind::TokenSequence,
                attr,
                query,
                *last_as_prefix,
            ),
        }
    }
}

pub fn require_fts<'a>(
    schema: &'a CollectionSchema,
    attr: &str,
) -> Result<(&'a SchemaEntry, &'a FullTextSearchConfig)> {
    let entry = schema
        .get_by_name(attr)
        .ok_or_else(|| TreeError::InvalidFilter(format!("unknown attribute '{attr}'")))?;
    let fts = entry.full_text_search.as_ref().ok_or_else(|| {
        TreeError::InvalidFilter(format!(
            "attribute '{attr}' does not have full_text_search enabled"
        ))
    })?;
    Ok((entry, fts))
}

pub fn validate_token_query(
    attr: &str,
    query: &TokenQuery,
    fts: &FullTextSearchConfig,
) -> Result<()> {
    if fts.tokenizer == Tokenizer::PreTokenizedArray && matches!(query, TokenQuery::Text(_)) {
        return Err(TreeError::InvalidFilter(format!(
            "attribute '{attr}' uses PreTokenizedArray tokenizer and requires TokenQuery::PreTokenized"
        )));
    }
    Ok(())
}

fn require_filterable<'a>(
    schema: &'a CollectionSchema,
    attr: &str,
) -> Result<&'a SchemaEntry> {
    let entry = schema
        .get_by_name(attr)
        .ok_or_else(|| TreeError::InvalidFilter(format!("unknown attribute '{attr}'")))?;
    if !entry.filterable {
        return Err(TreeError::InvalidFilter(format!(
            "attribute '{attr}' is not filterable"
        )));
    }
    Ok(entry)
}

fn validate_scalar(
    attr: &str,
    val: &ScalarValue,
    expected: AttributeType,
    allow_null: bool,
) -> Result<()> {
    if matches!(val, ScalarValue::Null) {
        if allow_null {
            return Ok(());
        }
        return Err(TreeError::InvalidFilter(format!(
            "Null value not allowed in predicate on '{attr}'"
        )));
    }
    if !val.matches_scalar_type(expected) {
        return Err(TreeError::InvalidFilter(format!(
            "filter value {val:?} does not match type {expected:?} of attribute '{attr}'"
        )));
    }
    Ok(())
}

fn attr_eq_scalar(val: &AttributeValue, target: &ScalarValue) -> bool {
    match (val, target) {
        (AttributeValue::Null, ScalarValue::Null) => true,
        (AttributeValue::Bool(a), ScalarValue::Bool(b)) => a == b,
        (AttributeValue::Int(a), ScalarValue::Int(b)) => a == b,
        (AttributeValue::Uint(a), ScalarValue::Uint(b)) => a == b,
        (AttributeValue::Float(a), ScalarValue::Float(b)) => !a.is_nan() && !b.is_nan() && a == b,
        (AttributeValue::String(a), ScalarValue::String(b)) => a == b,
        _ => false,
    }
}

fn compare_attr_scalar(val: Option<&AttributeValue>, target: &ScalarValue) -> Option<Ordering> {
    let val = val?;
    match (val, target) {
        (AttributeValue::Int(a), ScalarValue::Int(b)) => Some(a.cmp(b)),
        (AttributeValue::Uint(a), ScalarValue::Uint(b)) => Some(a.cmp(b)),
        (AttributeValue::Float(a), ScalarValue::Float(b)) => {
            if a.is_nan() || b.is_nan() {
                None
            } else {
                // Normalize -0.0 == +0.0 to match IEEE-754 order-preserving key encoding.
                let an = if *a == 0.0 { 0.0 } else { *a };
                let bn = if *b == 0.0 { 0.0 } else { *b };
                Some(an.total_cmp(&bn))
            }
        }
        (AttributeValue::String(a), ScalarValue::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

fn attr_contains_scalar(val: Option<&AttributeValue>, target: &ScalarValue) -> bool {
    let Some(val) = val else {
        return false;
    };
    match (val, target) {
        (AttributeValue::StringArray(items), ScalarValue::String(t)) => items.iter().any(|x| x == t),
        (AttributeValue::IntArray(items), ScalarValue::Int(t)) => items.contains(t),
        (AttributeValue::UintArray(items), ScalarValue::Uint(t)) => items.contains(t),
        _ => false,
    }
}
