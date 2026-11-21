use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use alyze::analyze::ReusableBuffer;
use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, DocId, FTS_BLOCK_SIZE, FtsBlockId, FtsDocId, make_fts_doc_id};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rank::{Bm25Overrides, RankBy};
use izzydb_core::schema::{CollectionSchema, FullTextSearchConfig};

use crate::analyzer::FtsAnalyzer;
use crate::block::{DocLenBlockView, FtsMeta, PostingBlockHeader, PostingBlockView, TermStats};
use crate::bm25::Bm25TermScorer;
use crate::simd::{
    SimdTier, filter_ge_256, max_score_256, merge_add_256, merge_max_256, prepare_dl_norm_256,
};

pub const DEFAULT_MAX_PREFIX_EXPANSIONS: usize = 128;

/// BM25 search hit returned by `Collection::query_bm25` / `Collection::query`.
#[derive(Debug, Clone, PartialEq)]
pub struct Bm25Result {
    pub id: DocId,
    pub score: f32,
}

/// Cursor over a single term's posting blocks (`cf_fts_postings`) sorted by ascending `block_id`.
pub trait FtsBlockCursor {
    fn current_block_id(&self) -> Option<FtsBlockId>;
    fn current_block_bytes(&self) -> &[u8];
    fn next_block(&mut self) -> Result<()>;
    fn seek_block(&mut self, target: FtsBlockId) -> Result<()>;
}

/// Storage abstraction consumed by MAXSCORE and FTS filter evaluation.
pub trait FtsPostingSource {
    type Cursor<'a>: FtsBlockCursor + 'a
    where
        Self: 'a;

    fn get_fts_meta(&self) -> Result<Option<FtsMeta>>;
    fn multi_get_term_stats(
        &self,
        attr_id: AttrId,
        tokens: &[String],
    ) -> Result<Vec<Option<TermStats>>>;
    fn scan_prefix_terms(
        &self,
        attr_id: AttrId,
        prefix: &str,
        limit: usize,
    ) -> Result<Vec<(String, TermStats)>>;
    fn open_posting_cursor<'a>(
        &'a self,
        attr_id: AttrId,
        token: &str,
    ) -> Result<Self::Cursor<'a>>;
    fn read_doc_len_block(
        &self,
        attr_id: AttrId,
        block_id: FtsBlockId,
        f: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<bool>;
    fn multi_get_doc_ids(&self, fts_doc_ids: &[FtsDocId]) -> Result<Vec<Option<DocId>>>;
}

struct DlNormGroup {
    attr_id: AttrId,
    c1: f32,
    c2: f32,
    cached_block_id: Option<FtsBlockId>,
    dl_norm: [f32; FTS_BLOCK_SIZE],
}

struct DlNormCache {
    groups: Vec<DlNormGroup>,
}

impl DlNormCache {
    fn new() -> Self {
        Self { groups: Vec::new() }
    }

    fn intern(&mut self, attr_id: AttrId, scorer: &Bm25TermScorer) -> usize {
        let c1 = scorer.k1_one_minus_b;
        let c2 = scorer.k1_b_over_avgdl;
        if let Some(pos) = self.groups.iter().position(|g| {
            g.attr_id == attr_id && g.c1.to_bits() == c1.to_bits() && g.c2.to_bits() == c2.to_bits()
        }) {
            return pos;
        }
        let id = self.groups.len();
        self.groups.push(DlNormGroup {
            attr_id,
            c1,
            c2,
            cached_block_id: None,
            dl_norm: [f32::MIN_POSITIVE; FTS_BLOCK_SIZE],
        });
        id
    }

    #[inline]
    fn get_or_load<S: FtsPostingSource>(
        &mut self,
        source: &S,
        tier: SimdTier,
        group_id: usize,
        block_id: FtsBlockId,
    ) -> Result<&[f32; FTS_BLOCK_SIZE]> {
        let group = &mut self.groups[group_id];
        if group.cached_block_id != Some(block_id) {
            let attr_id = group.attr_id;
            let c1 = group.c1;
            let c2 = group.c2;
            let out = &mut group.dl_norm;
            let found = source.read_doc_len_block(attr_id, block_id, &mut |bytes| {
                let view = DocLenBlockView::parse(bytes)?;
                prepare_dl_norm_256(tier, view.width, view.dls_bytes, c1, c2, out);
                Ok(())
            })?;
            if !found {
                return Err(TreeError::CorruptedData(format!(
                    "missing cf_fts_doc_lens block for attr {attr_id} block {block_id}"
                )));
            }
            group.cached_block_id = Some(block_id);
        }
        Ok(&self.groups[group_id].dl_norm)
    }
}

struct TermStream<'a, S: FtsPostingSource + 'a> {
    scorer: Bm25TermScorer,
    dl_group: usize,
    cursor: S::Cursor<'a>,
}

impl<'a, S: FtsPostingSource + 'a> TermStream<'a, S> {
    #[inline]
    fn current_block_id(&self) -> Option<FtsBlockId> {
        self.cursor.current_block_id()
    }

    #[inline]
    fn block_upper_bound(&self) -> Result<f32> {
        let header = PostingBlockHeader::parse(self.cursor.current_block_bytes())?;
        Ok(self.scorer.upper_bound(header.max_tf, header.min_dl_over_tf))
    }

    #[inline]
    fn seek(&mut self, target: FtsBlockId) -> Result<()> {
        if let Some(cur) = self.cursor.current_block_id()
            && cur < target
        {
            self.cursor.seek_block(target)?;
        }
        Ok(())
    }

    #[inline]
    fn advance_past(&mut self, block_id: FtsBlockId) -> Result<()> {
        if self.cursor.current_block_id() == Some(block_id) {
            self.cursor.next_block()?;
        }
        Ok(())
    }
}

enum ClauseKind<'a, S: FtsPostingSource + 'a> {
    Single(TermStream<'a, S>),
    PrefixMax(Vec<TermStream<'a, S>>),
    MaxBranches(Vec<Vec<Clause<'a, S>>>),
}

struct Clause<'a, S: FtsPostingSource + 'a> {
    term_max: f32,
    kind: ClauseKind<'a, S>,
}

impl<'a, S: FtsPostingSource + 'a> Clause<'a, S> {
    fn next_block_id(&self) -> Option<FtsBlockId> {
        match &self.kind {
            ClauseKind::Single(t) => t.current_block_id(),
            ClauseKind::PrefixMax(ts) => ts.iter().filter_map(|t| t.current_block_id()).min(),
            ClauseKind::MaxBranches(branches) => branches
                .iter()
                .flat_map(|b| b.iter())
                .filter_map(|c| c.next_block_id())
                .min(),
        }
    }

    fn block_upper_bound(&self, target: FtsBlockId) -> Result<f32> {
        match &self.kind {
            ClauseKind::Single(t) => {
                if t.current_block_id() == Some(target) {
                    t.block_upper_bound()
                } else {
                    Ok(0.0)
                }
            }
            ClauseKind::PrefixMax(ts) => {
                let mut best = 0.0f32;
                for t in ts {
                    if t.current_block_id() == Some(target) {
                        best = best.max(t.block_upper_bound()?);
                    }
                }
                Ok(best)
            }
            ClauseKind::MaxBranches(branches) => {
                let mut best = 0.0f32;
                for branch in branches {
                    let mut sum = 0.0f32;
                    for c in branch {
                        sum += c.block_upper_bound(target)?;
                    }
                    if sum > best {
                        best = sum;
                    }
                }
                Ok(best)
            }
        }
    }

    fn seek_to(&mut self, target: FtsBlockId) -> Result<()> {
        match &mut self.kind {
            ClauseKind::Single(t) => t.seek(target),
            ClauseKind::PrefixMax(ts) => {
                for t in ts {
                    t.seek(target)?;
                }
                Ok(())
            }
            ClauseKind::MaxBranches(branches) => {
                for branch in branches {
                    for c in branch {
                        c.seek_to(target)?;
                    }
                }
                Ok(())
            }
        }
    }

    fn advance_past(&mut self, target: FtsBlockId) -> Result<()> {
        match &mut self.kind {
            ClauseKind::Single(t) => t.advance_past(target),
            ClauseKind::PrefixMax(ts) => {
                for t in ts {
                    t.advance_past(target)?;
                }
                Ok(())
            }
            ClauseKind::MaxBranches(branches) => {
                for branch in branches {
                    for c in branch {
                        c.advance_past(target)?;
                    }
                }
                Ok(())
            }
        }
    }

    fn accumulate_block(
        &self,
        target: FtsBlockId,
        scores: &mut [f32; FTS_BLOCK_SIZE],
        touched: &mut [u64; 4],
        dl_cache: &mut DlNormCache,
        source: &S,
        tier: SimdTier,
    ) -> Result<()> {
        match &self.kind {
            ClauseKind::Single(t) => {
                if t.current_block_id() == Some(target) && !t.scorer.is_zero() {
                    let view = PostingBlockView::parse(t.cursor.current_block_bytes())?;
                    let dl_norm = dl_cache.get_or_load(source, tier, t.dl_group, target)?;
                    t.scorer
                        .accumulate_block_simd(tier, &view, dl_norm, scores, touched);
                }
                Ok(())
            }
            ClauseKind::PrefixMax(ts) => {
                let matching = ts
                    .iter()
                    .filter(|t| t.current_block_id() == Some(target) && !t.scorer.is_zero())
                    .count();
                if matching == 0 {
                    return Ok(());
                }
                if matching == 1 {
                    let t = ts
                        .iter()
                        .find(|t| t.current_block_id() == Some(target) && !t.scorer.is_zero())
                        .unwrap();
                    let view = PostingBlockView::parse(t.cursor.current_block_bytes())?;
                    let dl_norm = dl_cache.get_or_load(source, tier, t.dl_group, target)?;
                    t.scorer
                        .accumulate_block_simd(tier, &view, dl_norm, scores, touched);
                    return Ok(());
                }

                let mut local = [0.0f32; FTS_BLOCK_SIZE];
                let mut tmp = [0.0f32; FTS_BLOCK_SIZE];
                let mut local_touched = [0u64; 4];
                let mut first = true;
                for t in ts {
                    if t.current_block_id() != Some(target) || t.scorer.is_zero() {
                        continue;
                    }
                    let view = PostingBlockView::parse(t.cursor.current_block_bytes())?;
                    let dl_norm = dl_cache.get_or_load(source, tier, t.dl_group, target)?;
                    if first {
                        t.scorer.accumulate_block_simd(
                            tier,
                            &view,
                            dl_norm,
                            &mut local,
                            &mut local_touched,
                        );
                        first = false;
                    } else {
                        tmp.fill(0.0);
                        t.scorer.accumulate_block_simd(
                            tier,
                            &view,
                            dl_norm,
                            &mut tmp,
                            &mut local_touched,
                        );
                        merge_max_256(tier, &mut local, &tmp);
                    }
                }
                merge_add_256(tier, scores, &local);
                for w in 0..4 {
                    touched[w] |= local_touched[w];
                }
                Ok(())
            }
            ClauseKind::MaxBranches(branches) => {
                let matching_branches = branches
                    .iter()
                    .filter(|b| b.iter().any(|c| c.next_block_id() == Some(target)))
                    .count();
                if matching_branches == 0 {
                    return Ok(());
                }
                if matching_branches == 1 {
                    let branch = branches
                        .iter()
                        .find(|b| b.iter().any(|c| c.next_block_id() == Some(target)))
                        .unwrap();
                    for c in branch {
                        if c.next_block_id() == Some(target) {
                            c.accumulate_block(target, scores, touched, dl_cache, source, tier)?;
                        }
                    }
                    return Ok(());
                }

                let mut best = [0.0f32; FTS_BLOCK_SIZE];
                let mut branch_scores = [0.0f32; FTS_BLOCK_SIZE];
                let mut any_touched = [0u64; 4];
                let mut first = true;
                for branch in branches {
                    if !branch.iter().any(|c| c.next_block_id() == Some(target)) {
                        continue;
                    }
                    if first {
                        for c in branch {
                            if c.next_block_id() == Some(target) {
                                c.accumulate_block(
                                    target,
                                    &mut best,
                                    &mut any_touched,
                                    dl_cache,
                                    source,
                                    tier,
                                )?;
                            }
                        }
                        first = false;
                    } else {
                        branch_scores.fill(0.0);
                        for c in branch {
                            if c.next_block_id() == Some(target) {
                                c.accumulate_block(
                                    target,
                                    &mut branch_scores,
                                    &mut any_touched,
                                    dl_cache,
                                    source,
                                    tier,
                                )?;
                            }
                        }
                        merge_max_256(tier, &mut best, &branch_scores);
                    }
                }
                merge_add_256(tier, scores, &best);
                for w in 0..4 {
                    touched[w] |= any_touched[w];
                }
                Ok(())
            }
        }
    }

    fn accumulate_block_masked(
        &self,
        target: FtsBlockId,
        mask: &[u64; 4],
        scores: &mut [f32; FTS_BLOCK_SIZE],
        dl_cache: &mut DlNormCache,
        source: &S,
        tier: SimdTier,
    ) -> Result<()> {
        match &self.kind {
            ClauseKind::Single(t) => {
                if t.current_block_id() == Some(target) && !t.scorer.is_zero() {
                    let view = PostingBlockView::parse(t.cursor.current_block_bytes())?;
                    let dl_norm = dl_cache.get_or_load(source, tier, t.dl_group, target)?;
                    t.scorer
                        .accumulate_block_masked_simd(tier, &view, dl_norm, mask, scores);
                }
                Ok(())
            }
            ClauseKind::PrefixMax(ts) => {
                let matching = ts
                    .iter()
                    .filter(|t| t.current_block_id() == Some(target) && !t.scorer.is_zero())
                    .count();
                if matching == 0 {
                    return Ok(());
                }
                if matching == 1 {
                    let t = ts
                        .iter()
                        .find(|t| t.current_block_id() == Some(target) && !t.scorer.is_zero())
                        .unwrap();
                    let view = PostingBlockView::parse(t.cursor.current_block_bytes())?;
                    let dl_norm = dl_cache.get_or_load(source, tier, t.dl_group, target)?;
                    t.scorer
                        .accumulate_block_masked_simd(tier, &view, dl_norm, mask, scores);
                    return Ok(());
                }

                let mut local = [0.0f32; FTS_BLOCK_SIZE];
                let mut tmp = [0.0f32; FTS_BLOCK_SIZE];
                let mut first = true;
                for t in ts {
                    if t.current_block_id() != Some(target) || t.scorer.is_zero() {
                        continue;
                    }
                    let view = PostingBlockView::parse(t.cursor.current_block_bytes())?;
                    let dl_norm = dl_cache.get_or_load(source, tier, t.dl_group, target)?;
                    if first {
                        t.scorer.accumulate_block_masked_simd(
                            tier,
                            &view,
                            dl_norm,
                            mask,
                            &mut local,
                        );
                        first = false;
                    } else {
                        tmp.fill(0.0);
                        t.scorer
                            .accumulate_block_masked_simd(tier, &view, dl_norm, mask, &mut tmp);
                        merge_max_256(tier, &mut local, &tmp);
                    }
                }
                merge_add_256(tier, scores, &local);
                Ok(())
            }
            ClauseKind::MaxBranches(branches) => {
                let matching_branches = branches
                    .iter()
                    .filter(|b| b.iter().any(|c| c.next_block_id() == Some(target)))
                    .count();
                if matching_branches == 0 {
                    return Ok(());
                }
                if matching_branches == 1 {
                    let branch = branches
                        .iter()
                        .find(|b| b.iter().any(|c| c.next_block_id() == Some(target)))
                        .unwrap();
                    for c in branch {
                        if c.next_block_id() == Some(target) {
                            c.accumulate_block_masked(
                                target, mask, scores, dl_cache, source, tier,
                            )?;
                        }
                    }
                    return Ok(());
                }

                let mut best = [0.0f32; FTS_BLOCK_SIZE];
                let mut branch_scores = [0.0f32; FTS_BLOCK_SIZE];
                let mut first = true;
                for branch in branches {
                    if !branch.iter().any(|c| c.next_block_id() == Some(target)) {
                        continue;
                    }
                    if first {
                        for c in branch {
                            if c.next_block_id() == Some(target) {
                                c.accumulate_block_masked(
                                    target, mask, &mut best, dl_cache, source, tier,
                                )?;
                            }
                        }
                        first = false;
                    } else {
                        branch_scores.fill(0.0);
                        for c in branch {
                            if c.next_block_id() == Some(target) {
                                c.accumulate_block_masked(
                                    target,
                                    mask,
                                    &mut branch_scores,
                                    dl_cache,
                                    source,
                                    tier,
                                )?;
                            }
                        }
                        merge_max_256(tier, &mut best, &branch_scores);
                    }
                }
                merge_add_256(tier, scores, &best);
                Ok(())
            }
        }
    }
}

/// Compile a `RankBy` AST into flat MAXSCORE clauses (flattening `Sum` and distributing `Product`).
#[allow(clippy::too_many_arguments)]
fn compile_clauses<'a, S: FtsPostingSource>(
    source: &'a S,
    schema: &CollectionSchema,
    meta: &FtsMeta,
    expr: &RankBy,
    weight: f32,
    buffer: &mut ReusableBuffer,
    dl_cache: &mut DlNormCache,
    out: &mut Vec<Clause<'a, S>>,
) -> Result<()> {
    if weight <= 0.0 {
        return Ok(());
    }
    match expr {
        RankBy::Product(w, inner) => {
            compile_clauses(source, schema, meta, inner, weight * *w, buffer, dl_cache, out)
        }
        RankBy::Sum(children) => {
            for child in children {
                compile_clauses(source, schema, meta, child, weight, buffer, dl_cache, out)?;
            }
            Ok(())
        }
        RankBy::Max(children) => {
            let mut branches = Vec::with_capacity(children.len());
            let mut max_branch_ub = 0.0f32;
            for child in children {
                let mut branch_clauses = Vec::new();
                compile_clauses(
                    source,
                    schema,
                    meta,
                    child,
                    weight,
                    buffer,
                    dl_cache,
                    &mut branch_clauses,
                )?;
                if !branch_clauses.is_empty() {
                    let branch_ub: f32 = branch_clauses.iter().map(|c| c.term_max).sum();
                    if branch_ub > max_branch_ub {
                        max_branch_ub = branch_ub;
                    }
                    branches.push(branch_clauses);
                }
            }
            if !branches.is_empty() && max_branch_ub > 0.0 {
                if branches.len() == 1 {
                    out.extend(branches.into_iter().next().unwrap());
                } else {
                    out.push(Clause {
                        term_max: max_branch_ub,
                        kind: ClauseKind::MaxBranches(branches),
                    });
                }
            }
            Ok(())
        }
        RankBy::Bm25 {
            attr,
            query,
            params,
        } => {
            let entry = schema.get_by_name(attr).unwrap();
            let fts = entry.full_text_search.as_ref().unwrap();
            let attr_stats = match meta.per_attr_stats.get(&entry.id) {
                Some(s) if s.doc_count > 0 && s.total_tokens > 0 => *s,
                _ => return Ok(()),
            };
            let analyzer = FtsAnalyzer::new(fts.clone());
            let qtokens = analyzer.analyze_query(query, buffer)?;
            if qtokens.is_empty() {
                return Ok(());
            }

            let (k1, b, k3, last_as_prefix, max_exp) = resolve_params(fts, params.as_ref());

            let (exact_tokens, prefix_token) = if last_as_prefix {
                let last_idx = qtokens.len() - 1;
                (&qtokens[..last_idx], Some(qtokens[last_idx].text.clone()))
            } else {
                (&qtokens[..], None)
            };

            let mut exact_counts: HashMap<String, u32> = HashMap::new();
            for qt in exact_tokens {
                *exact_counts.entry(qt.text.clone()).or_insert(0) += 1;
            }
            if !exact_counts.is_empty() {
                let mut token_list: Vec<(String, u32)> = exact_counts.into_iter().collect();
                token_list.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                let keys: Vec<String> = token_list.iter().map(|(t, _)| t.clone()).collect();
                let stats_list = source.multi_get_term_stats(entry.id, &keys)?;
                for ((token, qtf), opt_stats) in token_list.into_iter().zip(stats_list) {
                    let Some(ts) = opt_stats else {
                        continue;
                    };
                    if ts.doc_freq == 0 {
                        continue;
                    }
                    let scorer = Bm25TermScorer::new(
                        weight,
                        attr_stats.doc_count,
                        attr_stats.total_tokens,
                        ts.doc_freq,
                        qtf,
                        k1,
                        b,
                        k3,
                    );
                    let ub = scorer.upper_bound(ts.max_tf, ts.min_dl_over_tf);
                    if ub <= 0.0 {
                        continue;
                    }
                    let cursor = source.open_posting_cursor(entry.id, &token)?;
                    if cursor.current_block_id().is_some() {
                        let dl_group = dl_cache.intern(entry.id, &scorer);
                        out.push(Clause {
                            term_max: ub,
                            kind: ClauseKind::Single(TermStream {
                                scorer,
                                dl_group,
                                cursor,
                            }),
                        });
                    }
                }
            }

            if let Some(prefix) = prefix_token {
                let expanded = source.scan_prefix_terms(entry.id, &prefix, max_exp)?;
                let mut streams = Vec::with_capacity(expanded.len());
                let mut max_ub = 0.0f32;
                for (token, ts) in expanded {
                    if ts.doc_freq == 0 {
                        continue;
                    }
                    let scorer = Bm25TermScorer::new(
                        weight,
                        attr_stats.doc_count,
                        attr_stats.total_tokens,
                        ts.doc_freq,
                        1,
                        k1,
                        b,
                        k3,
                    );
                    let ub = scorer.upper_bound(ts.max_tf, ts.min_dl_over_tf);
                    if ub <= 0.0 {
                        continue;
                    }
                    let cursor = source.open_posting_cursor(entry.id, &token)?;
                    if cursor.current_block_id().is_some() {
                        if ub > max_ub {
                            max_ub = ub;
                        }
                        let dl_group = dl_cache.intern(entry.id, &scorer);
                        streams.push(TermStream {
                            scorer,
                            dl_group,
                            cursor,
                        });
                    }
                }
                if !streams.is_empty() {
                    if streams.len() == 1 {
                        out.push(Clause {
                            term_max: max_ub,
                            kind: ClauseKind::Single(streams.into_iter().next().unwrap()),
                        });
                    } else {
                        out.push(Clause {
                            term_max: max_ub,
                            kind: ClauseKind::PrefixMax(streams),
                        });
                    }
                }
            }

            Ok(())
        }
    }
}

fn resolve_params(
    fts: &FullTextSearchConfig,
    overrides: Option<&Bm25Overrides>,
) -> (f32, f32, f32, bool, usize) {
    match overrides {
        Some(o) => (
            o.k1.unwrap_or(fts.k1),
            o.b.unwrap_or(fts.b),
            o.k3.unwrap_or(fts.k3),
            o.last_as_prefix.unwrap_or(false),
            o.max_prefix_expansions
                .unwrap_or(DEFAULT_MAX_PREFIX_EXPANSIONS),
        ),
        None => (
            fts.k1,
            fts.b,
            fts.k3,
            false,
            DEFAULT_MAX_PREFIX_EXPANSIONS,
        ),
    }
}

/// Min-heap entry for top-$k$ BM25 (smallest score / largest DocId at the top so it is evicted first).
#[derive(Debug, Clone, Copy)]
struct TopKEntry {
    score: f32,
    fts_doc_id: FtsDocId,
    doc_id: Option<DocId>,
}

impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.fts_doc_id == other.fts_doc_id
    }
}

impl Eq for TopKEntry {}

impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| match (self.doc_id, other.doc_id) {
                (Some(a), Some(b)) => a.cmp(&b),
                _ => self.fts_doc_id.cmp(&other.fts_doc_id),
            })
    }
}

impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Precomputed per-block bitmasks from a `RoaringBitmap<FtsDocId>` filter pushdown.
pub struct BlockFilterMask {
    blocks: HashMap<FtsBlockId, [u64; 4]>,
}

impl BlockFilterMask {
    pub fn from_bitmap(bm: &RoaringBitmap) -> Self {
        let mut blocks: HashMap<FtsBlockId, [u64; 4]> = HashMap::new();
        for fts_doc_id in bm.iter() {
            let block_id = fts_doc_id >> 8;
            let slot = (fts_doc_id & 0xFF) as u8;
            let entry = blocks.entry(block_id).or_insert([0u64; 4]);
            entry[(slot >> 6) as usize] |= 1u64 << (slot & 63);
        }
        Self { blocks }
    }

    #[inline]
    pub fn get(&self, block_id: FtsBlockId) -> Option<&[u64; 4]> {
        self.blocks.get(&block_id)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Execute a validated [`RankBy`] query using SIMD Vectorized Block-Max MAXSCORE.
pub fn execute_rank_by<S: FtsPostingSource>(
    source: &S,
    schema: &CollectionSchema,
    rank_by: &RankBy,
    top_k: usize,
    doc_filter: Option<&RoaringBitmap>,
    mut lazy_filter: Option<&mut dyn FnMut(DocId) -> Result<bool>>,
    buffer: &mut ReusableBuffer,
) -> Result<Vec<Bm25Result>> {
    rank_by.validate(schema)?;
    if top_k == 0 {
        return Ok(Vec::new());
    }
    let Some(meta) = source.get_fts_meta()? else {
        return Ok(Vec::new());
    };
    let filter_mask = doc_filter.map(BlockFilterMask::from_bitmap);
    if let Some(ref fm) = filter_mask
        && fm.is_empty()
    {
        return Ok(Vec::new());
    }

    let tier = SimdTier::detect();
    let mut dl_cache = DlNormCache::new();
    let mut clauses = Vec::new();
    compile_clauses(
        source,
        schema,
        &meta,
        rank_by,
        1.0,
        buffer,
        &mut dl_cache,
        &mut clauses,
    )?;
    if clauses.is_empty() {
        return Ok(Vec::new());
    }

    // Sort clauses ascending by global upper bound `term_max` for MAXSCORE partitioning.
    clauses.sort_by(|a, b| a.term_max.total_cmp(&b.term_max));
    let n_clauses = clauses.len();
    let mut prefix_max_sum = Vec::with_capacity(n_clauses);
    let mut acc = 0.0f64;
    for c in &clauses {
        acc += c.term_max as f64;
        prefix_max_sum.push(acc as f32);
    }

    let mut essential_start: usize = 0;
    let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(top_k + 1);
    let mut theta: f32 = 0.0;

    let mut scores = [0.0f32; FTS_BLOCK_SIZE];
    let mut touched: [u64; 4];
    let mut candidates: Vec<(f32, FtsDocId)> = Vec::with_capacity(32);
    let mut non_ess_ubs: Vec<f32> = Vec::with_capacity(n_clauses);

    loop {
        // Find minimum block_id among essential clauses (`essential_start..n_clauses`).
        let mut target_block: Option<FtsBlockId> = None;
        for c in &clauses[essential_start..] {
            if let Some(bid) = c.next_block_id() {
                target_block = Some(target_block.map_or(bid, |m| m.min(bid)));
            }
        }
        let Some(block_id) = target_block else {
            break;
        };

        // Check pushdown block filter mask first.
        let allowed_mask = if let Some(ref fm) = filter_mask {
            let Some(mask) = fm.get(block_id) else {
                for c in &mut clauses[essential_start..] {
                    c.advance_past(block_id)?;
                }
                continue;
            };
            Some(*mask)
        } else {
            None
        };

        let non_ess_ub = if essential_start > 0 {
            prefix_max_sum[essential_start - 1]
        } else {
            0.0
        };

        // Block-Max header pruning before decoding any posting lists or doc_len blocks.
        if theta > 0.0 {
            let mut ess_block_ub = 0.0f64;
            for c in &clauses[essential_start..] {
                ess_block_ub += c.block_upper_bound(block_id)? as f64;
            }
            let block_ub = (ess_block_ub + non_ess_ub as f64) as f32;
            let eps = block_ub * 1e-6 + 1e-7;
            if block_ub + eps < theta {
                for c in &mut clauses[essential_start..] {
                    c.advance_past(block_id)?;
                }
                continue;
            }
        }

        // Accumulate essential clauses into L1-resident [f32; 256] buffer via SIMD.
        scores.fill(0.0);
        touched = [0u64; 4];
        for c in &clauses[essential_start..] {
            if c.next_block_id() == Some(block_id) {
                c.accumulate_block(block_id, &mut scores, &mut touched, &mut dl_cache, source, tier)?;
            }
        }

        if let Some(mask) = allowed_mask {
            for w in 0..4 {
                touched[w] &= mask[w];
            }
        }

        let mut active_mask = touched;
        if active_mask != [0u64; 4] {
            let max_ess = max_score_256(tier, &scores, &active_mask);

            if essential_start > 0 {
                let eps = (max_ess + non_ess_ub) * 1e-6 + 1e-7;
                if max_ess + non_ess_ub + eps >= theta {
                    non_ess_ubs.clear();
                    let mut non_ess_block_ub = 0.0f64;
                    for c in &mut clauses[..essential_start] {
                        c.seek_to(block_id)?;
                        let ub = c.block_upper_bound(block_id)?;
                        non_ess_ubs.push(ub);
                        non_ess_block_ub += ub as f64;
                    }
                    let tight_ub = (max_ess as f64 + non_ess_block_ub) as f32;
                    let tight_eps = tight_ub * 1e-6 + 1e-7;
                    if tight_ub + tight_eps >= theta {
                        let mut rem_ub = non_ess_block_ub;
                        // Iterate descending by global term_max so rem_ub shrinks fastest.
                        for j in (0..essential_start).rev() {
                            let ub_j = non_ess_ubs[j];
                            if theta > 0.0 {
                                let min_needed = (theta - rem_ub as f32) - tight_eps;
                                if min_needed > 0.0 {
                                    active_mask =
                                        filter_ge_256(tier, &scores, &active_mask, min_needed);
                                    if active_mask == [0u64; 4] {
                                        break;
                                    }
                                }
                            }
                            if ub_j > 0.0 && clauses[j].next_block_id() == Some(block_id) {
                                clauses[j].accumulate_block_masked(
                                    block_id,
                                    &active_mask,
                                    &mut scores,
                                    &mut dl_cache,
                                    source,
                                    tier,
                                )?;
                            }
                            rem_ub = (rem_ub - ub_j as f64).max(0.0);
                        }
                    } else {
                        active_mask = [0u64; 4];
                    }
                } else {
                    active_mask = [0u64; 4];
                }
            }
        }

        // SIMD threshold filtering for top-k candidates.
        if active_mask != [0u64; 4] {
            let cand_thresh = if heap.len() < top_k {
                f32::MIN_POSITIVE
            } else {
                theta
            };
            let hit_mask = filter_ge_256(tier, &scores, &active_mask, cand_thresh);
            if hit_mask != [0u64; 4] {
                candidates.clear();
                for (w, &bits) in hit_mask.iter().enumerate() {
                    let mut b = bits;
                    while b != 0 {
                        let bit = b.trailing_zeros() as usize;
                        b &= b - 1;
                        let slot = ((w << 6) | bit) as u8;
                        let s = scores[slot as usize];
                        candidates.push((s, make_fts_doc_id(block_id, slot)));
                    }
                }

                if !candidates.is_empty() {
                    let needs_doc_ids = lazy_filter.is_some()
                        || (heap.len() == top_k && candidates.iter().any(|&(s, _)| s == theta));

                    let resolved_ids: Vec<Option<DocId>> = if needs_doc_ids {
                        let fts_ids: Vec<FtsDocId> = candidates.iter().map(|&(_, id)| id).collect();
                        source.multi_get_doc_ids(&fts_ids)?
                    } else {
                        vec![None; candidates.len()]
                    };

                    for (&(s, fts_doc_id), &opt_doc_id) in
                        candidates.iter().zip(resolved_ids.iter())
                    {
                        if let Some(ref mut pred) = lazy_filter {
                            let Some(doc_id) = opt_doc_id else {
                                continue;
                            };
                            if !pred(doc_id)? {
                                continue;
                            }
                        }

                        if heap.len() < top_k {
                            heap.push(TopKEntry {
                                score: s,
                                fts_doc_id,
                                doc_id: opt_doc_id,
                            });
                            if heap.len() == top_k {
                                theta = heap.peek().unwrap().score;
                            }
                        } else if s > theta {
                            heap.pop();
                            heap.push(TopKEntry {
                                score: s,
                                fts_doc_id,
                                doc_id: opt_doc_id,
                            });
                            theta = heap.peek().unwrap().score;
                        } else if s == theta {
                            // Exact tie-breaking on ascending DocId.
                            let Some(new_doc_id) = opt_doc_id else {
                                continue;
                            };
                            let mut root = heap.pop().unwrap();
                            let root_doc_id = match root.doc_id {
                                Some(id) => id,
                                None => {
                                    let fetched = source.multi_get_doc_ids(&[root.fts_doc_id])?;
                                    let id = fetched[0].unwrap_or(DocId::MAX);
                                    root.doc_id = Some(id);
                                    id
                                }
                            };
                            if new_doc_id < root_doc_id {
                                heap.push(TopKEntry {
                                    score: s,
                                    fts_doc_id,
                                    doc_id: Some(new_doc_id),
                                });
                                theta = heap.peek().unwrap().score;
                            } else {
                                heap.push(root);
                            }
                        }
                    }
                }
            }
        }

        // Advance essential clauses processed at `block_id` BEFORE shrinking essential set.
        for c in &mut clauses[essential_start..] {
            c.advance_past(block_id)?;
        }

        // Dynamically shrink essential set as theta rises.
        if heap.len() == top_k && theta > 0.0 {
            while essential_start < n_clauses {
                let ub = prefix_max_sum[essential_start];
                let eps = ub * 1e-6 + 1e-7;
                if ub + eps < theta {
                    essential_start += 1;
                } else {
                    break;
                }
            }
            if essential_start >= n_clauses {
                break;
            }
        }
    }

    if heap.is_empty() {
        return Ok(Vec::new());
    }

    let mut entries: Vec<TopKEntry> = heap.into_vec();
    let missing_fts: Vec<FtsDocId> = entries
        .iter()
        .filter(|e| e.doc_id.is_none())
        .map(|e| e.fts_doc_id)
        .collect();
    if !missing_fts.is_empty() {
        let fetched = source.multi_get_doc_ids(&missing_fts)?;
        let map: HashMap<FtsDocId, DocId> = missing_fts
            .into_iter()
            .zip(fetched)
            .filter_map(|(fid, did)| did.map(|d| (fid, d)))
            .collect();
        for e in &mut entries {
            if e.doc_id.is_none() {
                e.doc_id = map.get(&e.fts_doc_id).copied();
            }
        }
    }

    let mut results: Vec<Bm25Result> = entries
        .into_iter()
        .filter_map(|e| {
            e.doc_id.map(|id| Bm25Result {
                id,
                score: e.score,
            })
        })
        .collect();
    results.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{DocLenBlock, PackedPostingBlock};
    use izzydb_core::config::{DistanceMetric, TreeConfig, fts_block_id, fts_slot};
    use izzydb_core::schema::{AttributeType, CollectionConfig};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    struct MemSource {
        meta: FtsMeta,
        docs: HashMap<FtsDocId, DocId>,
        terms: HashMap<(AttrId, String), TermStats>,
        postings: HashMap<(AttrId, String), Vec<(FtsBlockId, Vec<u8>)>>,
        doc_lens: HashMap<(AttrId, FtsBlockId), Vec<u8>>,
    }

    struct MemCursor {
        blocks: Vec<(FtsBlockId, Vec<u8>)>,
        idx: usize,
    }

    impl FtsBlockCursor for MemCursor {
        fn current_block_id(&self) -> Option<FtsBlockId> {
            self.blocks.get(self.idx).map(|(id, _)| *id)
        }

        fn current_block_bytes(&self) -> &[u8] {
            &self.blocks[self.idx].1
        }

        fn next_block(&mut self) -> Result<()> {
            if self.idx < self.blocks.len() {
                self.idx += 1;
            }
            Ok(())
        }

        fn seek_block(&mut self, target: FtsBlockId) -> Result<()> {
            while self.idx < self.blocks.len() && self.blocks[self.idx].0 < target {
                self.idx += 1;
            }
            Ok(())
        }
    }

    impl FtsPostingSource for MemSource {
        type Cursor<'a> = MemCursor;

        fn get_fts_meta(&self) -> Result<Option<FtsMeta>> {
            Ok(Some(self.meta.clone()))
        }

        fn multi_get_term_stats(
            &self,
            attr_id: AttrId,
            tokens: &[String],
        ) -> Result<Vec<Option<TermStats>>> {
            Ok(tokens
                .iter()
                .map(|t| self.terms.get(&(attr_id, t.clone())).copied())
                .collect())
        }

        fn scan_prefix_terms(
            &self,
            attr_id: AttrId,
            prefix: &str,
            limit: usize,
        ) -> Result<Vec<(String, TermStats)>> {
            let mut matches: Vec<(String, TermStats)> = self
                .terms
                .iter()
                .filter(|((aid, tok), _)| *aid == attr_id && tok.starts_with(prefix))
                .map(|((_, tok), &ts)| (tok.clone(), ts))
                .collect();
            matches.sort_by(|a, b| {
                (a.0 != prefix)
                    .cmp(&(b.0 != prefix))
                    .then_with(|| b.1.doc_freq.cmp(&a.1.doc_freq))
                    .then_with(|| a.0.cmp(&b.0))
            });
            matches.truncate(limit);
            Ok(matches)
        }

        fn open_posting_cursor<'a>(
            &'a self,
            attr_id: AttrId,
            token: &str,
        ) -> Result<Self::Cursor<'a>> {
            let blocks = self
                .postings
                .get(&(attr_id, token.to_owned()))
                .cloned()
                .unwrap_or_default();
            Ok(MemCursor { blocks, idx: 0 })
        }

        fn read_doc_len_block(
            &self,
            attr_id: AttrId,
            block_id: FtsBlockId,
            f: &mut dyn FnMut(&[u8]) -> Result<()>,
        ) -> Result<bool> {
            match self.doc_lens.get(&(attr_id, block_id)) {
                Some(bytes) => {
                    f(bytes)?;
                    Ok(true)
                }
                None => Ok(false),
            }
        }

        fn multi_get_doc_ids(&self, fts_doc_ids: &[FtsDocId]) -> Result<Vec<Option<DocId>>> {
            Ok(fts_doc_ids
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
        let mut dl_acc: HashMap<(AttrId, FtsBlockId), DocLenBlock> = HashMap::new();
        let mut block_acc: HashMap<(AttrId, String, FtsBlockId), PackedPostingBlock> =
            HashMap::new();
        let mut term_stats: HashMap<(AttrId, String), TermStats> = HashMap::new();

        let title_analyzer = FtsAnalyzer::new(FullTextSearchConfig::default());
        let body_analyzer = FtsAnalyzer::new(FullTextSearchConfig::default());

        for i in 0..n_docs {
            let doc_id = (n_docs - i) as DocId * 3; // non-monotonic DocIds to test tie-breaking!
            let fts_id = meta.alloc_doc_id().unwrap();
            docs_map.insert(fts_id, doc_id);

            let title = if i < 20 {
                // Force exact ties among first 20 docs
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
            let bid = fts_block_id(fts_id);
            let slot = fts_slot(fts_id);

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
                    let ts = term_stats.entry((attr_id, tok.clone())).or_default();
                    ts.doc_freq += 1;
                    ts.observe(term.tf, ad.doc_len);
                }
            }
            raw_docs.push((doc_id, title, body));
        }

        let mut postings: HashMap<(AttrId, String), Vec<(FtsBlockId, Vec<u8>)>> = HashMap::new();
        for ((aid, tok, bid), mut blk) in block_acc {
            blk.recompute_bounds(&dl_acc[&(aid, bid)]);
            postings
                .entry((aid, tok))
                .or_default()
                .push((bid, blk.encode()));
        }
        for blocks in postings.values_mut() {
            blocks.sort_by_key(|(bid, _)| *bid);
        }
        let doc_lens: HashMap<(AttrId, FtsBlockId), Vec<u8>> = dl_acc
            .into_iter()
            .filter(|(_, d)| !d.is_empty())
            .map(|(k, d)| (k, d.encode()))
            .collect();

        let source = MemSource {
            meta: meta.clone(),
            docs: docs_map,
            terms: term_stats.clone(),
            postings,
            doc_lens,
        };

        // Long 30-term LLM-style query + weighted multi-field Sum
        let query_str: String = (0..30)
            .map(|i| vocab[i % vocab.len()].as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let rank_by = RankBy::Sum(vec![
            RankBy::product(2.0, RankBy::bm25("title", query_str.as_str())),
            RankBy::bm25("body", query_str.as_str()),
        ]);

        let got = execute_rank_by(&source, &schema, &rank_by, 15, None, None, &mut buf).unwrap();

        // Brute-force reference
        let fts = FullTextSearchConfig::default();
        let q_tokens: Vec<String> = query_str.split_whitespace().map(str::to_owned).collect();
        let mut qtf_map: HashMap<String, u32> = HashMap::new();
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
                            && let Some(ts) = term_stats.get(&(aid, tok.clone()))
                        {
                            let scorer = Bm25TermScorer::new(
                                w,
                                st.doc_count,
                                st.total_tokens,
                                ts.doc_freq,
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
}
