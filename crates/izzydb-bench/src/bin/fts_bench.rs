use std::collections::HashMap;
use std::time::Instant;

use izzydb::fts::{Bm25TermScorer, FtsAnalyzer, ReusableBuffer, SimdTier};
use izzydb::{
    AttributeType, Attributes, Bm25Result, Collection, CollectionConfig, DistanceMetric, DocId,
    FullTextSearchConfig, RankBy, TreeConfig,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn run_suite(n: usize, vocab_size: usize) {
    let dim = 32usize;
    let mut rng = SmallRng::seed_from_u64(42);
    let vocab: Vec<String> = (0..vocab_size).map(|i| format!("term{i:04}")).collect();

    let dir = TempDir::new().unwrap();
    let cfg = CollectionConfig::new(TreeConfig::new(dim, DistanceMetric::L2Squared))
        .with_fts_attribute(
            "body",
            AttributeType::String,
            false,
            FullTextSearchConfig::default(),
        );
    let mut col = Collection::open(dir.path(), cfg).unwrap();

    let mut raw_texts = Vec::with_capacity(n);
    let t0 = Instant::now();
    for id in 0..n as DocId {
        let vec: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
        let doc_len = rng.random_range(15..=40);
        let text: String = (0..doc_len)
            .map(|_| {
                // Zipf-like skew: favor lower-index terms to generate dense blocks, mix rare terms
                let idx = if rng.random_bool(0.5) {
                    rng.random_range(0..50.min(vocab.len()))
                } else {
                    rng.random_range(0..vocab.len())
                };
                vocab[idx].as_str()
            })
            .collect::<Vec<_>>()
            .join(" ");
        let mut attrs = Attributes::new();
        attrs.insert("body".into(), text.clone().into());
        col.upsert(id, Some(&vec), attrs).unwrap();
        raw_texts.push((id, text));
    }
    println!(
        "\n[n={n}, tier={:?}] Indexed in {:.1}ms",
        SimdTier::detect(),
        t0.elapsed().as_secs_f64() * 1000.0
    );

    let analyzer = FtsAnalyzer::new(FullTextSearchConfig::default());
    let mut buf = ReusableBuffer::new();
    let mut analyzed_docs = Vec::with_capacity(n);
    let mut df_map: HashMap<String, u64> = HashMap::new();
    let mut total_tokens = 0u64;
    for (id, text) in &raw_texts {
        let ad = analyzer.analyze_str(text, &mut buf).unwrap();
        total_tokens += ad.doc_len as u64;
        for tok in ad.terms.keys() {
            *df_map.entry(tok.clone()).or_insert(0) += 1;
        }
        analyzed_docs.push((*id, ad));
    }

    for (label, num_terms) in [
        ("Short 3-term query", 3usize),
        ("Medium 8-term query", 8),
        ("Long 50-term LLM query", 50),
    ] {
        let q_terms: Vec<String> = (0..num_terms)
            .map(|i| vocab[(i * 19 + 5) % vocab.len()].clone())
            .collect();
        let q_str = q_terms.join(" ");
        let rank_by = RankBy::bm25("body", q_str.as_str());

        // Warmup
        let _ = col.query_bm25(&rank_by, None, 10).unwrap();

        let runs = 25usize;
        let t_maxscore = Instant::now();
        let mut last_got = Vec::new();
        for _ in 0..runs {
            last_got = col.query_bm25(&rank_by, None, 10).unwrap();
        }
        let maxscore_us = t_maxscore.elapsed().as_secs_f64() * 1e6 / runs as f64;

        let t_exh = Instant::now();
        let mut last_exh = Vec::new();
        let exh_runs = if n <= 20_000 { runs } else { 5 };
        for _ in 0..exh_runs {
            let mut scores = Vec::with_capacity(n);
            for (id, ad) in &analyzed_docs {
                let mut s = 0.0f32;
                for qt in &q_terms {
                    if let Some(at) = ad.terms.get(qt) {
                        let df = df_map[qt];
                        let scorer = Bm25TermScorer::new(
                            1.0,
                            n as u64,
                            total_tokens,
                            df,
                            1,
                            1.2,
                            0.75,
                            8.0,
                        );
                        s += scorer.score(at.tf, ad.doc_len);
                    }
                }
                if s > 0.0 {
                    scores.push(Bm25Result { id: *id, score: s });
                }
            }
            scores.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
            scores.truncate(10);
            last_exh = scores;
        }
        let exh_us = t_exh.elapsed().as_secs_f64() * 1e6 / exh_runs as f64;

        assert_eq!(
            last_got.iter().map(|r| r.id).collect::<Vec<_>>(),
            last_exh.iter().map(|r| r.id).collect::<Vec<_>>()
        );
        println!(
            "  {label:24} -> SIMD-MAXSCORE={maxscore_us:7.1}us, Exhaustive={exh_us:8.1}us ({:.1}x speedup)",
            exh_us / maxscore_us
        );
    }
}

fn main() {
    run_suite(10_000, 1_000);
}
