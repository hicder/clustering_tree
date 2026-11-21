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

/// Supported languages for stemming and stopword removal in full-text search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Language {
    Arabic = 1,
    Danish = 2,
    Dutch = 3,
    #[default]
    English = 4,
    Finnish = 5,
    French = 6,
    German = 7,
    Greek = 8,
    Hungarian = 9,
    Italian = 10,
    Norwegian = 11,
    Portuguese = 12,
    Romanian = 13,
    Russian = 14,
    Spanish = 15,
    Swedish = 16,
    Tamil = 17,
    Turkish = 18,
}

impl Language {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            1 => Ok(Self::Arabic),
            2 => Ok(Self::Danish),
            3 => Ok(Self::Dutch),
            4 => Ok(Self::English),
            5 => Ok(Self::Finnish),
            6 => Ok(Self::French),
            7 => Ok(Self::German),
            8 => Ok(Self::Greek),
            9 => Ok(Self::Hungarian),
            10 => Ok(Self::Italian),
            11 => Ok(Self::Norwegian),
            12 => Ok(Self::Portuguese),
            13 => Ok(Self::Romanian),
            14 => Ok(Self::Russian),
            15 => Ok(Self::Spanish),
            16 => Ok(Self::Swedish),
            17 => Ok(Self::Tamil),
            18 => Ok(Self::Turkish),
            other => Err(TreeError::CorruptedData(format!(
                "unknown Language byte: {other}"
            ))),
        }
    }

    /// Whether stopword removal is supported for this language.
    pub fn supports_stopwords(self) -> bool {
        !matches!(
            self,
            Self::Arabic | Self::Greek | Self::Romanian | Self::Tamil | Self::Turkish
        )
    }
}

/// Tokenizer pipeline used to split text into FTS tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Tokenizer {
    /// UAX #29 Unicode word boundary segmentation (`alyze` DFA).
    #[default]
    WordV4 = 1,
    /// Split strictly on Unicode `White_Space`.
    Whitespace = 2,
    /// `StringArray` elements are used verbatim as tokens.
    PreTokenizedArray = 3,
    #[cfg(feature = "fts-cjk")]
    SmartCn = 4,
    #[cfg(feature = "fts-cjk")]
    Kuromoji = 5,
    #[cfg(feature = "fts-cjk")]
    Nori = 6,
}

impl Tokenizer {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            1 => Ok(Self::WordV4),
            2 => Ok(Self::Whitespace),
            3 => Ok(Self::PreTokenizedArray),
            #[cfg(feature = "fts-cjk")]
            4 => Ok(Self::SmartCn),
            #[cfg(feature = "fts-cjk")]
            5 => Ok(Self::Kuromoji),
            #[cfg(feature = "fts-cjk")]
            6 => Ok(Self::Nori),
            other => Err(TreeError::CorruptedData(format!(
                "unknown Tokenizer byte: {other}"
            ))),
        }
    }

    pub fn is_cjk(self) -> bool {
        match self {
            Self::WordV4 | Self::Whitespace | Self::PreTokenizedArray => false,
            #[cfg(feature = "fts-cjk")]
            Self::SmartCn | Self::Kuromoji | Self::Nori => true,
        }
    }
}

/// Per-attribute full-text search and BM25 scoring configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct FullTextSearchConfig {
    pub tokenizer: Tokenizer,
    pub language: Language,
    pub case_sensitive: bool,
    pub stemming: bool,
    pub remove_stopwords: bool,
    pub ascii_folding: bool,
    pub max_token_length: usize,
    pub k1: f32,
    pub b: f32,
    pub k3: f32,
    pub index_positions: bool,
}

impl Default for FullTextSearchConfig {
    fn default() -> Self {
        Self {
            tokenizer: Tokenizer::WordV4,
            language: Language::English,
            case_sensitive: false,
            stemming: false,
            remove_stopwords: false,
            ascii_folding: false,
            max_token_length: 39,
            k1: 1.2,
            b: 0.75,
            k3: 8.0,
            index_positions: true,
        }
    }
}

impl FullTextSearchConfig {
    pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    pub fn with_language(mut self, language: Language) -> Self {
        self.language = language;
        self
    }

    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    pub fn with_stemming(mut self, stemming: bool) -> Self {
        self.stemming = stemming;
        self
    }

    pub fn with_remove_stopwords(mut self, remove_stopwords: bool) -> Self {
        self.remove_stopwords = remove_stopwords;
        self
    }

    pub fn with_ascii_folding(mut self, ascii_folding: bool) -> Self {
        self.ascii_folding = ascii_folding;
        self
    }

    pub fn with_max_token_length(mut self, max_token_length: usize) -> Self {
        self.max_token_length = max_token_length;
        self
    }

    pub fn with_k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    pub fn with_b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    pub fn with_k3(mut self, k3: f32) -> Self {
        self.k3 = k3;
        self
    }

    pub fn with_index_positions(mut self, index_positions: bool) -> Self {
        self.index_positions = index_positions;
        self
    }

    /// Validate FTS configuration rules against the target attribute type.
    pub fn validate(&self, attr_name: &str, attr_type: AttributeType) -> Result<()> {
        if !matches!(attr_type, AttributeType::String | AttributeType::StringArray) {
            return Err(TreeError::InvalidSchema(format!(
                "full_text_search on '{attr_name}' is only valid for String or StringArray, got {attr_type:?}"
            )));
        }
        if self.max_token_length == 0 || self.max_token_length > 254 {
            return Err(TreeError::InvalidSchema(format!(
                "full_text_search on '{attr_name}': max_token_length must be in 1..=254, got {}",
                self.max_token_length
            )));
        }
        if !self.k1.is_finite() || self.k1 <= 0.0 {
            return Err(TreeError::InvalidSchema(format!(
                "full_text_search on '{attr_name}': k1 must be finite and > 0.0, got {}",
                self.k1
            )));
        }
        if !self.b.is_finite() || !(0.0..=1.0).contains(&self.b) {
            return Err(TreeError::InvalidSchema(format!(
                "full_text_search on '{attr_name}': b must be in 0.0..=1.0, got {}",
                self.b
            )));
        }
        if !self.k3.is_finite() || self.k3 < 0.0 {
            return Err(TreeError::InvalidSchema(format!(
                "full_text_search on '{attr_name}': k3 must be finite and >= 0.0, got {}",
                self.k3
            )));
        }

        if self.tokenizer == Tokenizer::PreTokenizedArray {
            if attr_type != AttributeType::StringArray {
                return Err(TreeError::InvalidSchema(format!(
                    "PreTokenizedArray tokenizer on '{attr_name}' requires AttributeType::StringArray, got {attr_type:?}"
                )));
            }
            if self.stemming || self.remove_stopwords || self.case_sensitive || self.ascii_folding {
                return Err(TreeError::InvalidSchema(format!(
                    "PreTokenizedArray tokenizer on '{attr_name}' does not allow stemming, remove_stopwords, case_sensitive, or ascii_folding"
                )));
            }
        } else if self.tokenizer == Tokenizer::Whitespace {
            if self.stemming || self.remove_stopwords {
                return Err(TreeError::InvalidSchema(format!(
                    "Whitespace tokenizer on '{attr_name}' does not support stemming or remove_stopwords"
                )));
            }
        } else if self.tokenizer.is_cjk()
            && (self.stemming || self.remove_stopwords || self.case_sensitive || self.ascii_folding)
        {
            return Err(TreeError::InvalidSchema(format!(
                "CJK tokenizer on '{attr_name}' does not allow custom stemming, remove_stopwords, case_sensitive, or ascii_folding overrides"
            )));
        }

        if (self.stemming || self.remove_stopwords) && self.case_sensitive {
            return Err(TreeError::InvalidSchema(format!(
                "full_text_search on '{attr_name}': stemming and remove_stopwords require case_sensitive == false"
            )));
        }

        if self.remove_stopwords && !self.language.supports_stopwords() {
            return Err(TreeError::InvalidSchema(format!(
                "full_text_search on '{attr_name}': stopword removal is not supported for language {:?}",
                self.language
            )));
        }

        Ok(())
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        buf.push(self.tokenizer.as_u8());
        buf.push(self.language.as_u8());
        buf.push(u8::from(self.case_sensitive));
        buf.push(u8::from(self.stemming));
        buf.push(u8::from(self.remove_stopwords));
        buf.push(u8::from(self.ascii_folding));
        buf.extend_from_slice(&(self.max_token_length as u16).to_le_bytes());
        buf.extend_from_slice(&self.k1.to_le_bytes());
        buf.extend_from_slice(&self.b.to_le_bytes());
        buf.extend_from_slice(&self.k3.to_le_bytes());
        buf.push(u8::from(self.index_positions));
    }

    fn decode(bytes: &[u8], off: &mut usize) -> Result<Self> {
        if *off + 21 > bytes.len() {
            return Err(TreeError::CorruptedData(
                "FullTextSearchConfig payload truncated".into(),
            ));
        }
        let tokenizer = Tokenizer::from_u8(bytes[*off])?;
        let language = Language::from_u8(bytes[*off + 1])?;
        let case_sensitive = decode_bool_byte(bytes[*off + 2], "fts case_sensitive")?;
        let stemming = decode_bool_byte(bytes[*off + 3], "fts stemming")?;
        let remove_stopwords = decode_bool_byte(bytes[*off + 4], "fts remove_stopwords")?;
        let ascii_folding = decode_bool_byte(bytes[*off + 5], "fts ascii_folding")?;
        let max_token_length = u16::from_le_bytes(bytes[*off + 6..*off + 8].try_into().unwrap()) as usize;
        let k1 = f32::from_le_bytes(bytes[*off + 8..*off + 12].try_into().unwrap());
        let b = f32::from_le_bytes(bytes[*off + 12..*off + 16].try_into().unwrap());
        let k3 = f32::from_le_bytes(bytes[*off + 16..*off + 20].try_into().unwrap());
        let index_positions = decode_bool_byte(bytes[*off + 20], "fts index_positions")?;
        *off += 21;
        Ok(Self {
            tokenizer,
            language,
            case_sensitive,
            stemming,
            remove_stopwords,
            ascii_folding,
            max_token_length,
            k1,
            b,
            k3,
            index_positions,
        })
    }
}

fn decode_bool_byte(byte: u8, field: &str) -> Result<bool> {
    match byte {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(TreeError::CorruptedData(format!(
            "invalid {field} byte: {other}"
        ))),
    }
}

/// Configuration for a single attribute column in a collection.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeConfig {
    pub name: String,
    pub attr_type: AttributeType,
    pub filterable: bool,
    pub full_text_search: Option<FullTextSearchConfig>,
}

impl AttributeConfig {
    /// Create a filterable attribute definition.
    pub fn new(name: impl Into<String>, attr_type: AttributeType) -> Self {
        Self {
            name: name.into(),
            attr_type,
            filterable: true,
            full_text_search: None,
        }
    }

    /// Set whether this attribute is indexed for filtering.
    pub fn filterable(mut self, filterable: bool) -> Self {
        self.filterable = filterable;
        self
    }

    /// Enable full-text search indexing and BM25 scoring on this attribute.
    pub fn full_text_search(mut self, config: FullTextSearchConfig) -> Self {
        self.full_text_search = Some(config);
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
            full_text_search: None,
        });
        self
    }

    pub fn with_attribute_config(mut self, attr: AttributeConfig) -> Self {
        self.attributes.push(attr);
        self
    }

    pub fn with_fts_attribute(
        mut self,
        name: impl Into<String>,
        attr_type: AttributeType,
        filterable: bool,
        fts: FullTextSearchConfig,
    ) -> Self {
        self.attributes.push(AttributeConfig {
            name: name.into(),
            attr_type,
            filterable,
            full_text_search: Some(fts),
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
            if let Some(fts) = &attr.full_text_search {
                fts.validate(&attr.name, attr.attr_type)?;
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
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaEntry {
    pub id: AttrId,
    pub name: String,
    pub attr_type: AttributeType,
    pub filterable: bool,
    pub full_text_search: Option<FullTextSearchConfig>,
}

/// Persistent collection schema registry mapping attribute names to monotonic `AttrId`s.
#[derive(Debug, Clone, PartialEq)]
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
            full_text_search: attr.full_text_search.clone(),
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
                if existing.full_text_search != attr.full_text_search {
                    return Err(TreeError::ConfigMismatch {
                        field: "attribute_full_text_search",
                        expected: format!("{}:{:?}", existing.name, existing.full_text_search),
                        actual: format!("{}:{:?}", attr.name, attr.full_text_search),
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

    /// Whether any attribute in this schema has full-text search enabled.
    pub fn has_fts(&self) -> bool {
        self.entries.iter().any(|e| e.full_text_search.is_some())
    }

    /// Iterate over all schema entries with full-text search enabled.
    pub fn fts_entries(&self) -> impl Iterator<Item = &SchemaEntry> {
        self.entries
            .iter()
            .filter(|e| e.full_text_search.is_some())
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
                (AttributeValue::StringArray(items), AttributeType::StringArray) => {
                    if let Some(fts) = &entry.full_text_search
                        && fts.tokenizer == Tokenizer::PreTokenizedArray
                    {
                        for tok in items {
                            if tok.is_empty() {
                                return Err(TreeError::InvalidAttribute {
                                    name: name.clone(),
                                    reason: "PreTokenizedArray tokens cannot be empty".into(),
                                });
                            }
                            if tok.chars().count() > fts.max_token_length {
                                return Err(TreeError::InvalidAttribute {
                                    name: name.clone(),
                                    reason: format!(
                                        "PreTokenizedArray token length exceeds max_token_length ({})",
                                        fts.max_token_length
                                    ),
                                });
                            }
                        }
                    }
                }
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

    /// Encode schema for persistence in `cf_meta["schema"]` (version 2).
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(2u8); // version
        buf.extend_from_slice(&self.next_attr_id.to_le_bytes());
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in &self.entries {
            buf.extend_from_slice(&entry.id.to_le_bytes());
            buf.push(entry.attr_type.as_u8());
            buf.push(u8::from(entry.filterable));
            match &entry.full_text_search {
                Some(fts) => {
                    buf.push(1);
                    fts.encode(&mut buf);
                }
                None => buf.push(0),
            }
            let name_bytes = entry.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);
        }
        buf
    }

    /// Decode schema from `cf_meta["schema"]` (version 2).
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 7 {
            return Err(TreeError::CorruptedData(
                "CollectionSchema payload too short".into(),
            ));
        }
        let version = bytes[0];
        if version != 2 {
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
            if off + 5 > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "CollectionSchema entry header truncated".into(),
                ));
            }
            let id = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
            let attr_type = AttributeType::from_u8(bytes[off + 2])?;
            let filterable = decode_bool_byte(bytes[off + 3], "schema filterable")?;
            let has_fts = decode_bool_byte(bytes[off + 4], "schema has_fts")?;
            off += 5;

            let full_text_search = if has_fts {
                let fts = FullTextSearchConfig::decode(bytes, &mut off)?;
                Some(fts)
            } else {
                None
            };

            if off + 4 > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "CollectionSchema entry name_len truncated".into(),
                ));
            }
            let name_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
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
                full_text_search,
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
