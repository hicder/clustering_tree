use std::collections::{BTreeMap, HashMap, HashSet};

use crate::config::{AttrId, TreeConfig};
use crate::error::{Result, TreeError};

/// Supported attribute column types in a [`CollectionSchema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AttributeType {
    String = 1,
    Int = 2,
    Uint = 3,
    Float = 4,
    Bool = 5,
    StringArray = 6,
    IntArray = 7,
    UintArray = 8,
}

impl AttributeType {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            1 => Ok(Self::String),
            2 => Ok(Self::Int),
            3 => Ok(Self::Uint),
            4 => Ok(Self::Float),
            5 => Ok(Self::Bool),
            6 => Ok(Self::StringArray),
            7 => Ok(Self::IntArray),
            8 => Ok(Self::UintArray),
            other => Err(TreeError::CorruptedData(format!(
                "unknown AttributeType byte: {other}"
            ))),
        }
    }

    pub fn is_array(self) -> bool {
        matches!(self, Self::StringArray | Self::IntArray | Self::UintArray)
    }

    pub fn is_ordered(self) -> bool {
        matches!(self, Self::String | Self::Int | Self::Uint | Self::Float)
    }

    /// For array types, returns the scalar element type; for scalar types, returns `self`.
    pub fn scalar_type(self) -> Self {
        match self {
            Self::StringArray => Self::String,
            Self::IntArray => Self::Int,
            Self::UintArray => Self::Uint,
            other => other,
        }
    }
}

/// Configuration for a single attribute column in a collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeConfig {
    pub name: String,
    pub attr_type: AttributeType,
    pub filterable: bool,
}

impl AttributeConfig {
    /// Create a filterable attribute definition.
    pub fn new(name: impl Into<String>, attr_type: AttributeType) -> Self {
        Self {
            name: name.into(),
            attr_type,
            filterable: true,
        }
    }

    /// Set whether this attribute is indexed for filtering.
    pub fn filterable(mut self, filterable: bool) -> Self {
        self.filterable = filterable;
        self
    }
}

/// Top-level configuration for opening or creating a [`Collection`](crate::Collection) (in `izzydb`).
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub tree: TreeConfig,
    pub attributes: Vec<AttributeConfig>,
}

impl CollectionConfig {
    pub fn new(tree: TreeConfig) -> Self {
        Self {
            tree,
            attributes: Vec::new(),
        }
    }

    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        attr_type: AttributeType,
        filterable: bool,
    ) -> Self {
        self.attributes.push(AttributeConfig {
            name: name.into(),
            attr_type,
            filterable,
        });
        self
    }

    pub fn validate(&self) -> Result<()> {
        self.tree.validate()?;
        let mut seen = HashSet::with_capacity(self.attributes.len());
        for attr in &self.attributes {
            if attr.name.is_empty() {
                return Err(TreeError::InvalidSchema(
                    "attribute name cannot be empty".into(),
                ));
            }
            if !seen.insert(attr.name.as_str()) {
                return Err(TreeError::InvalidSchema(format!(
                    "duplicate attribute name '{}'",
                    attr.name
                )));
            }
        }
        if self.attributes.len() > AttrId::MAX as usize {
            return Err(TreeError::InvalidSchema(
                "too many attributes for u16 AttrId".into(),
            ));
        }
        Ok(())
    }
}

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

/// Persisted schema entry mapping attribute name <-> compact `AttrId`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaEntry {
    pub id: AttrId,
    pub name: String,
    pub attr_type: AttributeType,
    pub filterable: bool,
}

/// Persistent collection schema registry mapping attribute names to monotonic `AttrId`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionSchema {
    entries: Vec<SchemaEntry>,
    by_name: HashMap<String, usize>,
    by_id: HashMap<AttrId, usize>,
    next_attr_id: AttrId,
}

impl CollectionSchema {
    /// Build a fresh schema from `CollectionConfig`.
    pub fn from_config(config: &CollectionConfig) -> Result<Self> {
        config.validate()?;
        let mut schema = Self {
            entries: Vec::with_capacity(config.attributes.len()),
            by_name: HashMap::with_capacity(config.attributes.len()),
            by_id: HashMap::with_capacity(config.attributes.len()),
            next_attr_id: 0,
        };
        for attr in &config.attributes {
            schema.register(attr)?;
        }
        Ok(schema)
    }

    fn register(&mut self, attr: &AttributeConfig) -> Result<AttrId> {
        let id = self.next_attr_id;
        self.next_attr_id = self
            .next_attr_id
            .checked_add(1)
            .ok_or_else(|| TreeError::InvalidSchema("AttrId space exhausted".into()))?;
        let idx = self.entries.len();
        self.entries.push(SchemaEntry {
            id,
            name: attr.name.clone(),
            attr_type: attr.attr_type,
            filterable: attr.filterable,
        });
        self.by_name.insert(attr.name.clone(), idx);
        self.by_id.insert(id, idx);
        Ok(id)
    }

    /// Reconcile a reopened schema against `config`, checking existing column types/flags
    /// and registering any newly declared attributes. Returns `true` if new attributes were added.
    pub fn reconcile(&mut self, config: &CollectionConfig) -> Result<bool> {
        config.validate()?;
        let mut changed = false;
        for attr in &config.attributes {
            if let Some(existing) = self.get_by_name(&attr.name) {
                if existing.attr_type != attr.attr_type {
                    return Err(TreeError::ConfigMismatch {
                        field: "attribute_type",
                        expected: format!("{}:{:?}", existing.name, existing.attr_type),
                        actual: format!("{}:{:?}", attr.name, attr.attr_type),
                    });
                }
                if existing.filterable != attr.filterable {
                    return Err(TreeError::ConfigMismatch {
                        field: "attribute_filterable",
                        expected: format!("{}:{}", existing.name, existing.filterable),
                        actual: format!("{}:{}", attr.name, attr.filterable),
                    });
                }
            } else {
                self.register(attr)?;
                changed = true;
            }
        }
        Ok(changed)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&SchemaEntry> {
        self.by_name.get(name).map(|&idx| &self.entries[idx])
    }

    pub fn get_by_id(&self, id: AttrId) -> Option<&SchemaEntry> {
        self.by_id.get(&id).map(|&idx| &self.entries[idx])
    }

    pub fn entries(&self) -> &[SchemaEntry] {
        &self.entries
    }

    /// Validate document attributes against this schema.
    pub fn validate_attributes(&self, attrs: &Attributes) -> Result<()> {
        for (name, val) in attrs {
            let entry = self.get_by_name(name).ok_or_else(|| TreeError::InvalidAttribute {
                name: name.clone(),
                reason: "attribute is not declared in collection schema".into(),
            })?;
            match (val, entry.attr_type) {
                (AttributeValue::Null, _) => {}
                (AttributeValue::Bool(_), AttributeType::Bool) => {}
                (AttributeValue::Int(_), AttributeType::Int) => {}
                (AttributeValue::Uint(_), AttributeType::Uint) => {}
                (AttributeValue::Float(v), AttributeType::Float) => {
                    if v.is_nan() {
                        return Err(TreeError::InvalidAttribute {
                            name: name.clone(),
                            reason: "NaN float attribute values are not allowed".into(),
                        });
                    }
                }
                (AttributeValue::String(_), AttributeType::String) => {}
                (AttributeValue::StringArray(_), AttributeType::StringArray) => {}
                (AttributeValue::IntArray(_), AttributeType::IntArray) => {}
                (AttributeValue::UintArray(_), AttributeType::UintArray) => {}
                _ => {
                    return Err(TreeError::InvalidAttribute {
                        name: name.clone(),
                        reason: format!(
                            "value does not match declared type {:?}",
                            entry.attr_type
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Encode schema for persistence in `cf_meta["schema"]`.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(1u8); // version
        buf.extend_from_slice(&self.next_attr_id.to_le_bytes());
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in &self.entries {
            buf.extend_from_slice(&entry.id.to_le_bytes());
            buf.push(entry.attr_type.as_u8());
            buf.push(if entry.filterable { 1 } else { 0 });
            let name_bytes = entry.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);
        }
        buf
    }

    /// Decode schema from `cf_meta["schema"]`.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 7 {
            return Err(TreeError::CorruptedData(
                "CollectionSchema payload too short".into(),
            ));
        }
        let version = bytes[0];
        if version != 1 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported CollectionSchema version: {version}"
            )));
        }
        let next_attr_id = u16::from_le_bytes(bytes[1..3].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[3..7].try_into().unwrap()) as usize;
        let mut off = 7;

        let mut entries = Vec::with_capacity(count);
        let mut by_name = HashMap::with_capacity(count);
        let mut by_id = HashMap::with_capacity(count);

        for idx in 0..count {
            if off + 8 > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "CollectionSchema entry header truncated".into(),
                ));
            }
            let id = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
            let attr_type = AttributeType::from_u8(bytes[off + 2])?;
            let filterable = match bytes[off + 3] {
                0 => false,
                1 => true,
                other => {
                    return Err(TreeError::CorruptedData(format!(
                        "invalid schema filterable byte: {other}"
                    )));
                }
            };
            let name_len = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
            off += 8;
            if off + name_len > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "CollectionSchema entry name truncated".into(),
                ));
            }
            let name = std::str::from_utf8(&bytes[off..off + name_len])
                .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in schema: {e}")))?
                .to_owned();
            off += name_len;

            if by_name.contains_key(&name) || by_id.contains_key(&id) {
                return Err(TreeError::CorruptedData(
                    "duplicate entry in CollectionSchema".into(),
                ));
            }
            by_name.insert(name.clone(), idx);
            by_id.insert(id, idx);
            entries.push(SchemaEntry {
                id,
                name,
                attr_type,
                filterable,
            });
        }
        if off != bytes.len() {
            return Err(TreeError::CorruptedData(
                "trailing bytes in CollectionSchema".into(),
            ));
        }

        Ok(Self {
            entries,
            by_name,
            by_id,
            next_attr_id,
        })
    }
}
