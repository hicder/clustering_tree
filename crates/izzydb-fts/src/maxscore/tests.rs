use std::collections::HashMap;

use alyze::analyze::ReusableBuffer;

use izzydb_core::config::{
    AttrId, DocId, RowBlock, RowId, TermId,
};
use izzydb_core::error::Result;
use izzydb_core::rank::RankBy;
use izzydb_core::schema::{CollectionSchema, FullTextSearchConfig};

use crate::analyzer::FtsAnalyzer;
use crate::block::{FtsMeta, TermRecord};
use crate::bm25::Bm25TermScorer;
use super::*;
use crate::block::{DocLenBlock, EncodedPosting, PackedPostingBlock, PostingBlockHeader};
use izzydb_core::config::{DistanceMetric, TreeConfig};
use izzydb_core::schema::{AttributeType, CollectionConfig};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

struct MemSource {
    meta: FtsMeta,
    docs: HashMap<RowId, DocId>,
    terms: HashMap<(AttrId, String), TermRecord>,
    postings: HashMap<TermId, Vec<(RowBlock, Vec<u8>)>>,
    doc_lens: HashMap<(AttrId, RowBlock), Vec<u8>>,
}

struct MemSkipCursor {
    blocks: Vec<(RowBlock, u32, f32)>,
    inline: Option<(RowBlock, Vec<u8>)>,
    idx: usize,
}

impl FtsSkipCursor for MemSkipCursor {
    fn current_block_id(&self) -> Option<RowBlock> {
        if let Some((bid, _)) = &self.inline {
            if self.idx == 0 {
                return Some(*bid);
            }
            return None;
        }
        self.blocks.get(self.idx).map(|(id, _, _)| *id)
    }

    fn current_block_bounds(&self) -> (u32, f32) {
        if let Some((_, bytes)) = &self.inline {
            let hdr = PostingBlockHeader::parse(bytes).unwrap();
            return (hdr.max_tf, hdr.min_dl_over_tf);
        }
        let (_, max_tf, min_ratio) = self.blocks[self.idx];
        (max_tf, min_ratio)
    }

    fn inline_posting_bytes(&self) -> Option<&[u8]> {
        self.inline
            .as_ref()
            .filter(|_| self.idx == 0)
            .map(|(_, b)| b.as_slice())
    }

    fn current_posting_bytes(&mut self) -> Result<Option<&[u8]>> {
        Ok(self.inline_posting_bytes())
    }

    fn current_posting_encoded(&mut self) -> Result<Option<EncodedPosting<'_>>> {
        Ok(self.inline_posting_bytes().map(EncodedPosting::Standalone))
    }

    fn next_block(&mut self) -> Result<()> {
        if self.inline.is_some() {
            self.idx = 1;
            return Ok(());
        }
        if self.idx < self.blocks.len() {
            self.idx += 1;
        }
        Ok(())
    }

    fn seek_block(&mut self, target: RowBlock) -> Result<()> {
        if let Some((bid, _)) = &self.inline {
            if *bid < target {
                self.idx = 1;
            }
            return Ok(());
        }
        while self.idx < self.blocks.len() && self.blocks[self.idx].0 < target {
            self.idx += 1;
        }
        Ok(())
    }
}

struct MemDocLenCursor<'a> {
    source: &'a MemSource,
    attr_id: AttrId,
}

impl FtsDocLenCursor for MemDocLenCursor<'_> {
    fn seek_block_bytes(&mut self, block_id: RowBlock) -> Result<Option<&[u8]>> {
        Ok(self
            .source
            .doc_lens
            .get(&(self.attr_id, block_id))
            .map(|v| v.as_slice()))
    }
}

struct MemGuard<'a>(&'a [u8]);
impl AsRef<[u8]> for MemGuard<'_> {
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}

impl FtsPostingSource for MemSource {
    type SkipCursor<'a> = MemSkipCursor;
    type DocLenCursor<'a> = MemDocLenCursor<'a>;
    type PostingGuard<'a> = MemGuard<'a>;

    fn get_fts_meta(&self) -> Result<Option<FtsMeta>> {
        Ok(Some(self.meta.clone()))
    }

    fn multi_get_term_records(
        &self,
        attr_id: AttrId,
        tokens: &[String],
    ) -> Result<Vec<Option<TermRecord>>> {
        Ok(tokens
            .iter()
            .map(|t| self.terms.get(&(attr_id, t.clone())).cloned())
            .collect())
    }

    fn scan_prefix_terms(
        &self,
        attr_id: AttrId,
        prefix: &str,
        limit: usize,
    ) -> Result<Vec<(String, TermRecord)>> {
        let mut matches: Vec<(String, TermRecord)> = self
            .terms
            .iter()
            .filter(|((aid, tok), _)| *aid == attr_id && tok.starts_with(prefix))
            .map(|((_, tok), rec)| (tok.clone(), rec.clone()))
            .collect();
        matches.sort_by(|a, b| {
            (a.0 != prefix)
                .cmp(&(b.0 != prefix))
                .then_with(|| b.1.stats.doc_freq.cmp(&a.1.stats.doc_freq))
                .then_with(|| a.0.cmp(&b.0))
        });
        matches.truncate(limit);
        Ok(matches)
    }

    fn open_skip_cursor<'a>(&'a self, record: TermRecord) -> Result<Self::SkipCursor<'a>> {
        if let Some((bid, bytes)) = record.inline_block {
            return Ok(MemSkipCursor {
                blocks: Vec::new(),
                inline: Some((bid, bytes)),
                idx: 0,
            });
        }
        let blocks = self
            .postings
            .get(&record.term_id)
            .map(|v| {
                v.iter()
                    .map(|(bid, bytes)| {
                        let hdr = PostingBlockHeader::parse(bytes).unwrap();
                        (*bid, hdr.max_tf, hdr.min_dl_over_tf)
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(MemSkipCursor {
            blocks,
            inline: None,
            idx: 0,
        })
    }

    fn get_posting_block<'a>(
        &'a self,
        term_id: TermId,
        block_id: RowBlock,
    ) -> Result<Option<Self::PostingGuard<'a>>> {
        let Some(v) = self.postings.get(&term_id) else {
            return Ok(None);
        };
        Ok(v.iter()
            .find(|(bid, _)| *bid == block_id)
            .map(|(_, bytes)| MemGuard(bytes.as_slice())))
    }

    fn open_doc_len_cursor<'a>(&'a self, attr_id: AttrId) -> Result<Self::DocLenCursor<'a>> {
        Ok(MemDocLenCursor {
            source: self,
            attr_id,
        })
    }

    fn multi_get_doc_ids(&self, row_ids: &[RowId]) -> Result<Vec<Option<DocId>>> {
        Ok(row_ids
            .iter()
            .map(|id| self.docs.get(id).copied())
            .collect())
    }
}

#[test]
fn test_maxscore_matches_brute_force_on_long_queries_and_ties() {
    let cfg = CollectionConfig::new(TreeConfig::new(4, DistanceMetric::L2Squared))
        .with_fts_attribute("title", AttributeType::String, false, FullTextSearchConfig::default())
        .with_fts_attribute("body", AttributeType::String, false, FullTextSearchConfig::default());
    let schema = CollectionSchema::from_config(&cfg).unwrap();

    let vocab: Vec<String> = (0..60).map(|i| format!("w{i}")).collect();
    let mut rng = SmallRng::seed_from_u64(12345);
    let mut buf = ReusableBuffer::new();

    let n_docs = 650usize;
    let mut raw_docs: Vec<(DocId, String, String)> = Vec::new();
    let mut meta = FtsMeta::default();
    let mut docs_map = HashMap::new();
    let mut dl_acc: HashMap<(AttrId, RowBlock), DocLenBlock> = HashMap::new();
    let mut block_acc: HashMap<(AttrId, String, RowBlock), PackedPostingBlock> =
        HashMap::new();
    let mut term_records: HashMap<(AttrId, String), TermRecord> = HashMap::new();

    let title_analyzer = FtsAnalyzer::new(FullTextSearchConfig::default());
    let body_analyzer = FtsAnalyzer::new(FullTextSearchConfig::default());

    for i in 0..n_docs {
        let doc_id = (n_docs - i) as DocId * 3;
        let row_id = RowId::new((i / 300) as u32, (i % 300) as u32);
        docs_map.insert(row_id, doc_id);

        let title = if i < 20 {
            "w0 w1".to_string()
        } else {
            let len = rng.random_range(1..=4);
            (0..len)
                .map(|_| vocab[rng.random_range(0..vocab.len())].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let body = if i < 20 {
            "w2 w3 w4".to_string()
        } else {
            let len = rng.random_range(5..=30);
            (0..len)
                .map(|_| vocab[rng.random_range(0..vocab.len())].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };

        let td = title_analyzer.analyze_str(&title, &mut buf).unwrap();
        let bd = body_analyzer.analyze_str(&body, &mut buf).unwrap();
        let bid = row_id.block();
        let slot = row_id.slot();

        for (attr_id, ad) in [(0u16, &td), (1u16, &bd)] {
            let s = meta.per_attr_stats.entry(attr_id).or_default();
            s.doc_count += 1;
            s.total_tokens += ad.doc_len as u64;
            let dls = dl_acc.entry((attr_id, bid)).or_default();
            dls.set(slot, ad.doc_len);
            for (tok, term) in &ad.terms {
                block_acc
                    .entry((attr_id, tok.clone(), bid))
                    .or_default()
                    .upsert(slot, term.tf, dls);
                let rec = term_records
                    .entry((attr_id, tok.clone()))
                    .or_insert_with(|| TermRecord::new(meta.alloc_term_id().unwrap()));
                rec.stats.doc_freq += 1;
                rec.stats.observe(term.tf, ad.doc_len);
            }
        }
        raw_docs.push((doc_id, title, body));
    }

    let mut postings: HashMap<TermId, Vec<(RowBlock, Vec<u8>)>> = HashMap::new();
    for ((aid, tok, bid), mut blk) in block_acc {
        blk.recompute_bounds(&dl_acc[&(aid, bid)]);
        let tid = term_records[&(aid, tok)].term_id;
        postings.entry(tid).or_default().push((bid, blk.encode()));
    }
    for blocks in postings.values_mut() {
        blocks.sort_by_key(|(bid, _)| *bid);
    }
    let doc_lens: HashMap<(AttrId, RowBlock), Vec<u8>> = dl_acc
        .into_iter()
        .filter(|(_, d)| !d.is_empty())
        .map(|(k, d)| (k, d.encode()))
        .collect();

    let source = MemSource {
        meta: meta.clone(),
        docs: docs_map,
        terms: term_records.clone(),
        postings,
        doc_lens,
    };

    let query_str: String = (0..30)
        .map(|i| vocab[i % vocab.len()].as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let rank_by = RankBy::Sum(vec![
        RankBy::product(2.0, RankBy::bm25("title", query_str.as_str())),
        RankBy::bm25("body", query_str.as_str()),
    ]);

    let got = execute_rank_by(&source, &schema, &rank_by, 15, None, None, &mut buf).unwrap();

    let fts = FullTextSearchConfig::default();
    let q_tokens: Vec<String> = query_str.split_whitespace().map(str::to_owned).collect();
    let mut qtf_map: std::collections::BTreeMap<String, u32> =
        std::collections::BTreeMap::new();
    for t in q_tokens {
        *qtf_map.entry(t).or_insert(0) += 1;
    }
    let mut expected: Vec<Bm25Result> = raw_docs
        .iter()
        .map(|(doc_id, title, body)| {
            let td = title_analyzer.analyze_str(title, &mut buf).unwrap();
            let bd = body_analyzer.analyze_str(body, &mut buf).unwrap();
            let score_field = |aid: AttrId, ad: &crate::analyzer::AnalyzedDoc, w: f32| -> f32 {
                let st = meta.per_attr_stats[&aid];
                let mut sum = 0.0f32;
                for (tok, &qtf) in &qtf_map {
                    if let Some(at) = ad.terms.get(tok)
                        && let Some(rec) = term_records.get(&(aid, tok.clone()))
                    {
                        let scorer = Bm25TermScorer::new(
                            w,
                            st.doc_count,
                            st.total_tokens,
                            rec.stats.doc_freq,
                            qtf,
                            fts.k1,
                            fts.b,
                            fts.k3,
                        );
                        sum += scorer.score(at.tf, ad.doc_len);
                    }
                }
                sum
            };
            Bm25Result {
                id: *doc_id,
                score: score_field(0, &td, 2.0) + score_field(1, &bd, 1.0),
            }
        })
        .filter(|r| r.score > 0.0)
        .collect();
    expected.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    expected.truncate(15);

    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_eq!(g.id, e.id);
        assert!(
            (g.score - e.score).abs() <= 1e-5,
            "score mismatch for doc {}: {} vs {}",
            g.id,
            g.score,
            e.score
        );
    }
}
