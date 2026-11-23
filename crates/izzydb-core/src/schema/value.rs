//! Scalar and attribute values.

use std::collections::BTreeMap;

use super::*;

/// Scalar value used in filter predicates and scalar attributes.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    String(String),
}

impl ScalarValue {
    pub fn matches_scalar_type(&self, attr_type: AttributeType) -> bool {
        match (self, attr_type) {
            (Self::Null, _) => !attr_type.is_array(),
            (Self::Bool(_), AttributeType::Bool) => true,
            (Self::Int(_), AttributeType::Int) => true,
            (Self::Uint(_), AttributeType::Uint) => true,
            (Self::Float(v), AttributeType::Float) => !v.is_nan(),
            (Self::String(_), AttributeType::String) => true,
            _ => false,
        }
    }
}

impl From<bool> for ScalarValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<i64> for ScalarValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<i32> for ScalarValue {
    fn from(v: i32) -> Self {
        Self::Int(v as i64)
    }
}
impl From<u64> for ScalarValue {
    fn from(v: u64) -> Self {
        Self::Uint(v)
    }
}
impl From<u32> for ScalarValue {
    fn from(v: u32) -> Self {
        Self::Uint(v as u64)
    }
}
impl From<f64> for ScalarValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<&str> for ScalarValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}
impl From<String> for ScalarValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

/// Value stored in a document's forward attribute map.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    String(String),
    StringArray(Vec<String>),
    IntArray(Vec<i64>),
    UintArray(Vec<u64>),
}

impl From<bool> for AttributeValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<i64> for AttributeValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<i32> for AttributeValue {
    fn from(v: i32) -> Self {
        Self::Int(v as i64)
    }
}
impl From<u64> for AttributeValue {
    fn from(v: u64) -> Self {
        Self::Uint(v)
    }
}
impl From<u32> for AttributeValue {
    fn from(v: u32) -> Self {
        Self::Uint(v as u64)
    }
}
impl From<f64> for AttributeValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<&str> for AttributeValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}
impl From<String> for AttributeValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}
impl From<Vec<String>> for AttributeValue {
    fn from(v: Vec<String>) -> Self {
        Self::StringArray(v)
    }
}
impl From<Vec<i64>> for AttributeValue {
    fn from(v: Vec<i64>) -> Self {
        Self::IntArray(v)
    }
}
impl From<Vec<u64>> for AttributeValue {
    fn from(v: Vec<u64>) -> Self {
        Self::UintArray(v)
    }
}

/// Map of attribute name to attribute value for a document.
pub type Attributes = BTreeMap<String, AttributeValue>;
