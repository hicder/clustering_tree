use crate::error::{Result, TreeError};
use crate::filter::{TokenQuery, require_fts, validate_token_query};
use crate::schema::CollectionSchema;

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

/// Ranking expression AST for single-field BM25 and composite multi-field scoring.
#[derive(Debug, Clone, PartialEq)]
pub enum RankBy {
    Bm25 {
        attr: String,
        query: TokenQuery,
        params: Option<Bm25Overrides>,
    },
    Sum(Vec<RankBy>),
    Max(Vec<RankBy>),
    Product(f32, Box<RankBy>),
}

impl RankBy {
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

    pub fn product(weight: f32, expr: RankBy) -> Self {
        Self::Product(weight, Box::new(expr))
    }

    /// Validate all referenced attributes, token queries, weights, and BM25 overrides.
    pub fn validate(&self, schema: &CollectionSchema) -> Result<()> {
        match self {
            Self::Bm25 {
                attr,
                query,
                params,
            } => {
                let (_, fts) = require_fts(schema, attr).map_err(|e| match e {
                    TreeError::InvalidFilter(msg) => TreeError::InvalidQuery(msg),
                    other => other,
                })?;
                validate_token_query(attr, query, fts).map_err(|e| match e {
                    TreeError::InvalidFilter(msg) => TreeError::InvalidQuery(msg),
                    other => other,
                })?;
                if let Some(p) = params {
                    p.validate()?;
                }
                Ok(())
            }
            Self::Sum(children) | Self::Max(children) => {
                for child in children {
                    child.validate(schema)?;
                }
                Ok(())
            }
            Self::Product(weight, child) => {
                if !weight.is_finite() || *weight < 0.0 {
                    return Err(TreeError::InvalidQuery(format!(
                        "RankBy::Product weight must be finite and >= 0.0, got {weight}"
                    )));
                }
                child.validate(schema)
            }
        }
    }
}
