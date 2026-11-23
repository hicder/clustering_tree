//! Schema entries and the collection schema.

use std::collections::HashMap;
use crate::config::AttrId;
use crate::error::{Result, TreeError};

use super::*;

/// Persisted schema entry mapping attribute name <-> compact `AttrId`.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaEntry {
    pub id: AttrId,
    pub name: String,
    pub attr_type: AttributeType,
    pub filterable: bool,
    pub full_text_search: Option<FullTextSearchConfig>,
    pub storage: ColumnStorage,
    pub column: bool,
}

impl SchemaEntry {
    /// Returns `true` if this attribute is stored in `ColumnBlock`s (`cf_columns` for sealed
    /// inline segments, or `cf_detached` for detached columns).
    #[inline]
    pub fn is_columnar(&self) -> bool {
        self.storage == ColumnStorage::Detached
            || (self.storage == ColumnStorage::Inline
                && (self.filterable
                    || self.column
                    || matches!(
                        self.attr_type,
                        AttributeType::Bool
                            | AttributeType::Int
                            | AttributeType::Uint
                            | AttributeType::Float
                    )))
    }
}

/// Persistent collection schema registry mapping attribute names to monotonic `AttrId`s.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionSchema {
    pub(super) has_vectors: bool,
    pub(super) entries: Vec<SchemaEntry>,
    pub(super) by_name: HashMap<String, usize>,
    pub(super) by_id: HashMap<AttrId, usize>,
    pub(super) next_attr_id: AttrId,
}

impl CollectionSchema {
    /// Build a fresh schema from `CollectionConfig`.
    pub fn from_config(config: &CollectionConfig) -> Result<Self> {
        config.validate()?;
        let mut schema = Self {
            has_vectors: config.vector.is_some(),
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

    pub fn has_vectors(&self) -> bool {
        self.has_vectors
    }

    pub(super) fn register(&mut self, attr: &AttributeConfig) -> Result<AttrId> {
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
            storage: attr.storage,
            column: attr.column,
        });
        self.by_name.insert(attr.name.clone(), idx);
        self.by_id.insert(id, idx);
        Ok(id)
    }

    /// Reconcile a reopened schema against `config`, checking existing column types/flags
    /// and registering any newly declared attributes. Returns `true` if new attributes were added.
    pub fn reconcile(&mut self, config: &CollectionConfig) -> Result<bool> {
        config.validate()?;
        let cfg_has_vectors = config.vector.is_some();
        if self.has_vectors != cfg_has_vectors {
            return Err(TreeError::ConfigMismatch {
                field: "has_vectors",
                expected: self.has_vectors.to_string(),
                actual: cfg_has_vectors.to_string(),
            });
        }
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
                if existing.storage != attr.storage {
                    return Err(TreeError::ConfigMismatch {
                        field: "attribute_storage",
                        expected: format!("{}:{:?}", existing.name, existing.storage),
                        actual: format!("{}:{:?}", attr.name, attr.storage),
                    });
                }
                if existing.column != attr.column {
                    return Err(TreeError::ConfigMismatch {
                        field: "attribute_column",
                        expected: format!("{}:{}", existing.name, existing.column),
                        actual: format!("{}:{}", attr.name, attr.column),
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

    /// Encode schema for persistence in `default[\x01schema]` (version 4).
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(4u8); // version
        buf.push(u8::from(self.has_vectors));
        buf.extend_from_slice(&self.next_attr_id.to_le_bytes());
        buf.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in &self.entries {
            buf.extend_from_slice(&entry.id.to_le_bytes());
            buf.push(entry.attr_type.as_u8());
            buf.push(u8::from(entry.filterable) | (u8::from(entry.column) << 1));
            buf.push(entry.storage.as_u8());
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

    /// Decode schema from `default[\x01schema]` (version 4).
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(TreeError::CorruptedData(
                "CollectionSchema payload too short".into(),
            ));
        }
        let version = bytes[0];
        if version != 4 {
            return Err(TreeError::CorruptedData(format!(
                "unsupported CollectionSchema version: {version}"
            )));
        }
        let has_vectors = decode_bool_byte(bytes[1], "schema has_vectors")?;
        let next_attr_id = u16::from_le_bytes(bytes[2..4].try_into().unwrap());
        let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let mut off = 8;

        let mut entries = Vec::with_capacity(count);
        let mut by_name = HashMap::with_capacity(count);
        let mut by_id = HashMap::with_capacity(count);

        for idx in 0..count {
            if off + 6 > bytes.len() {
                return Err(TreeError::CorruptedData(
                    "CollectionSchema entry header truncated".into(),
                ));
            }
            let id = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
            let attr_type = AttributeType::from_u8(bytes[off + 2])?;
            let flag_byte = bytes[off + 3];
            if flag_byte > 3 {
                return Err(TreeError::CorruptedData(format!(
                    "invalid schema attribute flags: {flag_byte}"
                )));
            }
            let filterable = (flag_byte & 1) != 0;
            let column = (flag_byte & 2) != 0;
            let storage = ColumnStorage::from_u8(bytes[off + 4])?;
            let has_fts = decode_bool_byte(bytes[off + 5], "schema has_fts")?;
            off += 6;

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
                storage,
                column,
            });
        }
        if off != bytes.len() {
            return Err(TreeError::CorruptedData(
                "trailing bytes in CollectionSchema".into(),
            ));
        }

        Ok(Self {
            has_vectors,
            entries,
            by_name,
            by_id,
            next_attr_id,
        })
    }
}
