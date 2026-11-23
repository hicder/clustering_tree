//! Full-text search configuration: languages, tokenizers, and FTS config encoding.

use crate::error::{Result, TreeError};

use super::*;

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

    pub(super) fn encode(&self, buf: &mut Vec<u8>) {
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

    pub(super) fn decode(bytes: &[u8], off: &mut usize) -> Result<Self> {
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

pub(super) fn decode_bool_byte(byte: u8, field: &str) -> Result<bool> {
    match byte {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(TreeError::CorruptedData(format!(
            "invalid {field} byte: {other}"
        ))),
    }
}
