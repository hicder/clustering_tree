use std::collections::BTreeMap;

use alyze::analyze::{
    AnalysisOptions, Analyzer, LanguageWithStopwords, ReusableBuffer, StemmingLanguage,
    StopwordRemoval, TokenizerOptions,
};

use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::TokenQuery;
use izzydb_core::schema::{AttributeValue, FullTextSearchConfig, Language, Tokenizer};

/// Position gap inserted between consecutive elements of a `StringArray` attribute
/// (for non-`PreTokenizedArray` tokenizers) so phrase queries do not match across elements.
pub const STRING_ARRAY_POSITION_GAP: u32 = 2;

/// Per-token statistics extracted from analyzing a single document attribute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyzedTerm {
    pub tf: u32,
    pub positions: Vec<u32>,
}

/// Complete analysis result for a single document attribute.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnalyzedDoc {
    pub terms: BTreeMap<String, AnalyzedTerm>,
    pub doc_len: u32,
}

/// A single positioned token emitted from analyzing a query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryToken {
    pub text: String,
    pub position: u32,
}

/// Text analyzer configured from [`FullTextSearchConfig`] and backed by `alyze`.
#[derive(Debug, Clone)]
pub struct FtsAnalyzer {
    config: FullTextSearchConfig,
}

impl FtsAnalyzer {
    pub fn new(config: FullTextSearchConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &FullTextSearchConfig {
        &self.config
    }

    /// Analyze a document [`AttributeValue`] (`String`, `StringArray`, or `Null`) into
    /// term frequencies, positions, and total token length.
    ///
    /// Returns `Ok(None)` when `val` is `AttributeValue::Null`.
    pub fn analyze_attribute_value(
        &self,
        val: &AttributeValue,
        buffer: &mut ReusableBuffer,
    ) -> Result<Option<AnalyzedDoc>> {
        match val {
            AttributeValue::Null => Ok(None),
            AttributeValue::String(s) => Ok(Some(self.analyze_inputs([s.as_str()], buffer)?)),
            AttributeValue::StringArray(items) => Ok(Some(
                self.analyze_inputs(items.iter().map(String::as_str), buffer)?,
            )),
            other => Err(TreeError::InvalidSchema(format!(
                "FTS cannot analyze non-string attribute value {other:?}"
            ))),
        }
    }

    /// Analyze one or more string inputs into [`AnalyzedDoc`].
    ///
    /// For `PreTokenizedArray`, each input string is used verbatim as a single token at
    /// consecutive positions `0, 1, 2, ...`.
    /// For all other tokenizers, inputs are tokenized via `alyze` and consecutive array
    /// elements are separated by a `+2` position gap.
    pub fn analyze_inputs<'a>(
        &self,
        inputs: impl IntoIterator<Item = &'a str>,
        buffer: &mut ReusableBuffer,
    ) -> Result<AnalyzedDoc> {
        let mut doc = AnalyzedDoc::default();

        if self.config.tokenizer == Tokenizer::PreTokenizedArray {
            for (idx, tok) in inputs.into_iter().enumerate() {
                if tok.is_empty() {
                    return Err(TreeError::InvalidAttribute {
                        name: String::new(),
                        reason: "PreTokenizedArray tokens cannot be empty".into(),
                    });
                }
                if tok.chars().count() > self.config.max_token_length {
                    return Err(TreeError::InvalidAttribute {
                        name: String::new(),
                        reason: format!(
                            "PreTokenizedArray token length exceeds max_token_length ({})",
                            self.config.max_token_length
                        ),
                    });
                }
                doc.doc_len = doc.doc_len.saturating_add(1);
                let entry = doc.terms.entry(tok.to_owned()).or_insert_with(|| AnalyzedTerm {
                    tf: 0,
                    positions: Vec::new(),
                });
                entry.tf = entry.tf.saturating_add(1);
                if self.config.index_positions {
                    entry.positions.push(idx as u32);
                }
            }
            return Ok(doc);
        }

        let analyzer = build_alyze_analyzer(&self.config);
        let mut base_pos: u32 = 0;

        for input in inputs {
            let mut max_local_pos: Option<u32> = None;
            buffer.reset_keep_stemming_cache();
            analyzer.analyze(input, buffer, |token| {
                let local_pos = token.position as u32;
                let global_pos = base_pos.saturating_add(local_pos);
                max_local_pos = Some(max_local_pos.map_or(local_pos, |m| m.max(local_pos)));

                doc.doc_len = doc.doc_len.saturating_add(1);
                let entry = doc
                    .terms
                    .entry(token.text.to_owned())
                    .or_insert_with(|| AnalyzedTerm {
                        tf: 0,
                        positions: Vec::new(),
                    });
                entry.tf = entry.tf.saturating_add(1);
                if self.config.index_positions {
                    entry.positions.push(global_pos);
                }
                true
            });
            if let Some(max_local) = max_local_pos {
                base_pos = base_pos
                    .saturating_add(max_local)
                    .saturating_add(STRING_ARRAY_POSITION_GAP);
            }
        }

        Ok(doc)
    }

    /// Analyze a single input string into [`AnalyzedDoc`].
    pub fn analyze_str(&self, input: &str, buffer: &mut ReusableBuffer) -> Result<AnalyzedDoc> {
        self.analyze_inputs(std::iter::once(input), buffer)
    }

    /// Analyze a [`TokenQuery`] into an ordered sequence of [`QueryToken`]s.
    pub fn analyze_query(
        &self,
        query: &TokenQuery,
        buffer: &mut ReusableBuffer,
    ) -> Result<Vec<QueryToken>> {
        match query {
            TokenQuery::PreTokenized(tokens) => {
                let mut out = Vec::with_capacity(tokens.len());
                for (i, tok) in tokens.iter().enumerate() {
                    if !tok.is_empty() {
                        out.push(QueryToken {
                            text: tok.clone(),
                            position: i as u32,
                        });
                    }
                }
                Ok(out)
            }
            TokenQuery::Text(text) => {
                if self.config.tokenizer == Tokenizer::PreTokenizedArray {
                    return Err(TreeError::InvalidQuery(
                        "PreTokenizedArray attribute requires TokenQuery::PreTokenized".into(),
                    ));
                }
                let analyzer = build_alyze_analyzer(&self.config);
                let mut out = Vec::new();
                buffer.reset_keep_stemming_cache();
                analyzer.analyze(text, buffer, |token| {
                    out.push(QueryToken {
                        text: token.text.to_owned(),
                        position: token.position as u32,
                    });
                    true
                });
                Ok(out)
            }
        }
    }
}

fn build_alyze_analyzer(config: &FullTextSearchConfig) -> Analyzer<'static> {
    match config.tokenizer {
        Tokenizer::WordV4 => Analyzer::new(AnalysisOptions {
            tokenizer: TokenizerOptions::UAX29Word(Default::default()),
            maximum_token_length: Some(config.max_token_length),
            case_sensitive: config.case_sensitive,
            stopword_removal: if config.remove_stopwords {
                to_alyze_stopword_language(config.language).map(StopwordRemoval::ForLanguage)
            } else {
                None
            },
            stemming: if config.stemming {
                Some(to_alyze_stemming_language(config.language))
            } else {
                None
            },
            ascii_folding: config.ascii_folding,
        }),
        Tokenizer::Whitespace => Analyzer::new(AnalysisOptions {
            tokenizer: TokenizerOptions::Whitespace,
            maximum_token_length: Some(config.max_token_length),
            case_sensitive: config.case_sensitive,
            stopword_removal: None,
            stemming: None,
            ascii_folding: config.ascii_folding,
        }),
        Tokenizer::PreTokenizedArray => {
            unreachable!("PreTokenizedArray bypasses alyze analyzer")
        }
        #[cfg(feature = "fts-cjk")]
        Tokenizer::SmartCn => Analyzer::smartcn(Default::default()),
        #[cfg(feature = "fts-cjk")]
        Tokenizer::Kuromoji => Analyzer::kuromoji(Default::default()),
        #[cfg(feature = "fts-cjk")]
        Tokenizer::Nori => Analyzer::nori(Default::default()),
    }
}

fn to_alyze_stemming_language(lang: Language) -> StemmingLanguage {
    match lang {
        Language::Arabic => StemmingLanguage::Arabic,
        Language::Danish => StemmingLanguage::Danish,
        Language::Dutch => StemmingLanguage::Dutch,
        Language::English => StemmingLanguage::English,
        Language::Finnish => StemmingLanguage::Finnish,
        Language::French => StemmingLanguage::French,
        Language::German => StemmingLanguage::German,
        Language::Greek => StemmingLanguage::Greek,
        Language::Hungarian => StemmingLanguage::Hungarian,
        Language::Italian => StemmingLanguage::Italian,
        Language::Norwegian => StemmingLanguage::Norwegian,
        Language::Portuguese => StemmingLanguage::Portuguese,
        Language::Romanian => StemmingLanguage::Romanian,
        Language::Russian => StemmingLanguage::Russian,
        Language::Spanish => StemmingLanguage::Spanish,
        Language::Swedish => StemmingLanguage::Swedish,
        Language::Tamil => StemmingLanguage::Tamil,
        Language::Turkish => StemmingLanguage::Turkish,
    }
}

fn to_alyze_stopword_language(lang: Language) -> Option<LanguageWithStopwords> {
    match lang {
        Language::Danish => Some(LanguageWithStopwords::Danish),
        Language::Dutch => Some(LanguageWithStopwords::Dutch),
        Language::English => Some(LanguageWithStopwords::English),
        Language::Finnish => Some(LanguageWithStopwords::Finnish),
        Language::French => Some(LanguageWithStopwords::French),
        Language::German => Some(LanguageWithStopwords::German),
        Language::Hungarian => Some(LanguageWithStopwords::Hungarian),
        Language::Italian => Some(LanguageWithStopwords::Italian),
        Language::Norwegian => Some(LanguageWithStopwords::Norwegian),
        Language::Portuguese => Some(LanguageWithStopwords::Portuguese),
        Language::Russian => Some(LanguageWithStopwords::Russian),
        Language::Spanish => Some(LanguageWithStopwords::Spanish),
        Language::Swedish => Some(LanguageWithStopwords::Swedish),
        Language::Arabic
        | Language::Greek
        | Language::Romanian
        | Language::Tamil
        | Language::Turkish => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_v4_lowercasing_stemming_stopwords_ascii_folding() {
        let cfg = FullTextSearchConfig::default()
            .with_stemming(true)
            .with_remove_stopwords(true)
            .with_ascii_folding(true)
            .with_max_token_length(12);
        let analyzer = FtsAnalyzer::new(cfg);
        let mut buf = ReusableBuffer::new();

        let doc = analyzer
            .analyze_str(
                "The Café runners are running supercalifragilisticexpialidocious!",
                &mut buf,
            )
            .unwrap();
        // "The", "are" -> stopwords removed
        // "Café" -> "cafe" (ascii folded + lowercased)
        // "runners" -> "runner" (English Snowball)
        // "running" -> "run"
        // "supercalifragilisticexpialidocious" -> dropped (> 12 chars)
        assert_eq!(doc.doc_len, 3);
        assert!(doc.terms.contains_key("cafe"));
        assert!(doc.terms.contains_key("runner"));
        assert!(doc.terms.contains_key("run"));
        assert!(!doc.terms.contains_key("the"));
        assert!(!doc.terms.contains_key("are"));
        // Positions preserve gaps for dropped stopwords ("The" @ 0, "Café" @ 1, "runners" @ 2, "are" @ 3, "running" @ 4)
        assert_eq!(doc.terms["cafe"].positions, vec![1]);
        assert_eq!(doc.terms["runner"].positions, vec![2]);
        assert_eq!(doc.terms["run"].positions, vec![4]);
    }

    #[test]
    fn test_multilingual_stemming_and_stopwords() {
        let fr_cfg = FullTextSearchConfig::default()
            .with_language(Language::French)
            .with_stemming(true)
            .with_remove_stopwords(true);
        let fr = FtsAnalyzer::new(fr_cfg);
        let mut buf = ReusableBuffer::new();
        let doc = fr.analyze_str("Les chevaux mangent dans le jardin", &mut buf).unwrap();
        assert!(!doc.terms.contains_key("les"));
        assert!(!doc.terms.contains_key("dans"));
        assert!(!doc.terms.contains_key("le"));
        assert!(doc.doc_len >= 3);

        let de_cfg = FullTextSearchConfig::default()
            .with_language(Language::German)
            .with_stemming(true)
            .with_remove_stopwords(true);
        let de = FtsAnalyzer::new(de_cfg);
        let doc_de = de.analyze_str("Die Katzen schliefen ruhig", &mut buf).unwrap();
        assert!(!doc_de.terms.contains_key("die"));
        assert!(doc_de.terms.contains_key("katz"));
    }

    #[test]
    fn test_string_array_position_gaps_and_whitespace() {
        let cfg = FullTextSearchConfig::default();
        let analyzer = FtsAnalyzer::new(cfg);
        let mut buf = ReusableBuffer::new();
        let doc = analyzer
            .analyze_inputs(["quick brown", "fox jumps"], &mut buf)
            .unwrap();
        assert_eq!(doc.doc_len, 4);
        assert_eq!(doc.terms["quick"].positions, vec![0]);
        assert_eq!(doc.terms["brown"].positions, vec![1]);
        assert_eq!(doc.terms["fox"].positions, vec![3]);
        assert_eq!(doc.terms["jumps"].positions, vec![4]);

        let ws_cfg = FullTextSearchConfig::default()
            .with_tokenizer(Tokenizer::Whitespace)
            .with_case_sensitive(true);
        let ws = FtsAnalyzer::new(ws_cfg);
        let ws_doc = ws.analyze_str("Foo, Bar foo", &mut buf).unwrap();
        assert_eq!(ws_doc.doc_len, 3);
        assert!(ws_doc.terms.contains_key("Foo,"));
        assert!(ws_doc.terms.contains_key("Bar"));
        assert!(ws_doc.terms.contains_key("foo"));
    }

    #[test]
    fn test_pre_tokenized_array() {
        let cfg = FullTextSearchConfig::default().with_tokenizer(Tokenizer::PreTokenizedArray);
        let analyzer = FtsAnalyzer::new(cfg);
        let mut buf = ReusableBuffer::new();
        let val = AttributeValue::StringArray(vec![
            "CamelCase".into(),
            "foo-bar".into(),
            "CamelCase".into(),
        ]);
        let doc = analyzer
            .analyze_attribute_value(&val, &mut buf)
            .unwrap()
            .unwrap();
        assert_eq!(doc.doc_len, 3);
        assert_eq!(doc.terms["CamelCase"].tf, 2);
        assert_eq!(doc.terms["CamelCase"].positions, vec![0, 2]);
        assert_eq!(doc.terms["foo-bar"].positions, vec![1]);
    }
}
