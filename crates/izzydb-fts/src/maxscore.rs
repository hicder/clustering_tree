use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use alyze::analyze::ReusableBuffer;
use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, DocId, FTS_BLOCK_SIZE, FtsBlockId, FtsDocId, TermId, make_fts_doc_id};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::rank::{Bm25Overrides, RankBy};
use izzydb_core::schema::{CollectionSchema, FullTextSearchConfig};

use crate::analyzer::FtsAnalyzer;
use crate::block::{DocLenBlockView, FtsMeta, PostingBlockView, TermRecord};
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

/// Cursor over a single term's Block-Max skip summary (`cf_fts_skips` or inline `TermRecord`)
/// sorted by ascending `block_id`.
pub trait FtsSkipCursor {
    fn current_block_id(&self) -> Option<FtsBlockId>;
    fn current_block_bounds(&self) -> (u32, f32);
    fn inline_posting_bytes(&self) -> Option<&[u8]>;
    fn current_posting_bytes(&mut self) -> Result<Option<&[u8]>> {
        Ok(None)
    }
    fn next_block(&mut self) -> Result<()>;
    fn seek_block(&mut self, target: FtsBlockId) -> Result<()>;
}

/// Cursor over document-length blocks (`cf_fts_doc_lens`) for a single `attr_id`.
pub trait FtsDocLenCursor {
    fn seek_block_bytes(&mut self, block_id: FtsBlockId) -> Result<Option<&[u8]>>;
}

/// Storage abstraction consumed by MAXSCORE and FTS filter evaluation.
pub trait FtsPostingSource {
    type SkipCursor<'a>: FtsSkipCursor + 'a
    where
        Self: 'a;

    type DocLenCursor<'a>: FtsDocLenCursor + 'a
    where
        Self: 'a;

    type PostingGuard<'a>: AsRef<[u8]> + 'a
    where
        Self: 'a;

    fn get_fts_meta(&self) -> Result<Option<FtsMeta>>;
    fn multi_get_term_records(
        &self,
        attr_id: AttrId,
        tokens: &[String],
    ) -> Result<Vec<Option<TermRecord>>>;
    fn scan_prefix_terms(
        &self,
        attr_id: AttrId,
        prefix: &str,
        limit: usize,
    ) -> Result<Vec<(String, TermRecord)>>;
    fn open_skip_cursor<'a>(&'a self, record: TermRecord) -> Result<Self::SkipCursor<'a>>;
    fn open_skip_cursors<'a>(
        &'a self,
        records: Vec<TermRecord>,
    ) -> Result<Vec<Self::SkipCursor<'a>>> {
        records
            .into_iter()
            .map(|r| self.open_skip_cursor(r))
            .collect()
    }
    fn get_posting_block<'a>(
        &'a self,
        term_id: TermId,
        block_id: FtsBlockId,
    ) -> Result<Option<Self::PostingGuard<'a>>>;
    fn open_doc_len_cursor<'a>(&'a self, attr_id: AttrId) -> Result<Self::DocLenCursor<'a>>;
    fn multi_get_doc_ids(&self, fts_doc_ids: &[FtsDocId]) -> Result<Vec<Option<DocId>>>;
}

struct DlNormGroup<'a, S: FtsPostingSource + 'a> {
    attr_id: AttrId,
    c1: f32,
    c2: f32,
    cached_block_id: Option<FtsBlockId>,
    cursor: Option<S::DocLenCursor<'a>>,
    dl_norm: [f32; FTS_BLOCK_SIZE],
}

struct DlNormCache<'a, S: FtsPostingSource + 'a> {
    groups: Vec<DlNormGroup<'a, S>>,
}

impl<'a, S: FtsPostingSource + 'a> DlNormCache<'a, S> {
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
            cursor: None,
            dl_norm: [f32::MIN_POSITIVE; FTS_BLOCK_SIZE],
        });
        id
    }

    #[inline]
    fn get_or_load(
        &mut self,
        source: &'a S,
        tier: SimdTier,
        group_id: usize,
        block_id: FtsBlockId,
    ) -> Result<&[f32; FTS_BLOCK_SIZE]> {
        let group = &mut self.groups[group_id];
        if group.cached_block_id != Some(block_id) {
            let attr_id = group.attr_id;
            if group.cursor.is_none() {
                group.cursor = Some(source.open_doc_len_cursor(attr_id)?);
            }
            let cursor = group.cursor.as_mut().unwrap();
            let Some(bytes) = cursor.seek_block_bytes(block_id)? else {
                return Err(TreeError::CorruptedData(format!(
                    "missing cf_fts_doc_lens block for attr {attr_id} block {block_id}"
                )));
            };
            let view = DocLenBlockView::parse(bytes)?;
            prepare_dl_norm_256(
                tier,
                view.width,
                view.dls_bytes,
                group.c1,
                group.c2,
                &mut group.dl_norm,
            );
            group.cached_block_id = Some(block_id);
        }
        Ok(&self.groups[group_id].dl_norm)
    }
}

struct TermStream<'a, S: FtsPostingSource + 'a> {
    term_id: TermId,
    scorer: Bm25TermScorer,
    dl_group: usize,
    cursor: S::SkipCursor<'a>,
}

impl<'a, S: FtsPostingSource + 'a> TermStream<'a, S> {
    #[inline]
    fn current_block_id(&self) -> Option<FtsBlockId> {
        self.cursor.current_block_id()
    }

    #[inline]
    fn block_upper_bound(&self) -> f32 {
        let (max_tf, min_dl_over_tf) = self.cursor.current_block_bounds();
        self.scorer.upper_bound(max_tf, min_dl_over_tf)
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

    #[inline]
    fn accumulate(
        &mut self,
        target: FtsBlockId,
        scores: &mut [f32; FTS_BLOCK_SIZE],
        touched: &mut [u64; 4],
        dl_cache: &mut DlNormCache<'a, S>,
        source: &'a S,
        tier: SimdTier,
    ) -> Result<()> {
        let dl_norm = dl_cache.get_or_load(source, tier, self.dl_group, target)?;
        if let Some(bytes) = self.cursor.current_posting_bytes()? {
            let view = PostingBlockView::parse(bytes)?;
            self.scorer
                .accumulate_block_simd(tier, &view, dl_norm, scores, touched);
            return Ok(());
        }
        let Some(guard) = source.get_posting_block(self.term_id, target)? else {
            return Err(TreeError::CorruptedData(format!(
                "missing posting block for term_id {} block {target}",
                self.term_id
            )));
        };
        let view = PostingBlockView::parse(guard.as_ref())?;
        self.scorer
            .accumulate_block_simd(tier, &view, dl_norm, scores, touched);
        Ok(())
    }

    #[inline]
    fn accumulate_masked(
        &mut self,
        target: FtsBlockId,
        mask: &[u64; 4],
        scores: &mut [f32; FTS_BLOCK_SIZE],
        dl_cache: &mut DlNormCache<'a, S>,
        source: &'a S,
        tier: SimdTier,
    ) -> Result<()> {
        let dl_norm = dl_cache.get_or_load(source, tier, self.dl_group, target)?;
        if let Some(bytes) = self.cursor.current_posting_bytes()? {
            let view = PostingBlockView::parse(bytes)?;
            self.scorer
                .accumulate_block_masked_simd(tier, &view, dl_norm, mask, scores);
            return Ok(());
        }
        let Some(guard) = source.get_posting_block(self.term_id, target)? else {
            return Err(TreeError::CorruptedData(format!(
                "missing posting block for term_id {} block {target}",
                self.term_id
            )));
        };
        let view = PostingBlockView::parse(guard.as_ref())?;
        self.scorer
            .accumulate_block_masked_simd(tier, &view, dl_norm, mask, scores);
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

    fn block_upper_bound(&self, target: FtsBlockId) -> f32 {
        match &self.kind {
            ClauseKind::Single(t) => {
                if t.current_block_id() == Some(target) {
                    t.block_upper_bound()
                } else {
                    0.0
                }
            }
            ClauseKind::PrefixMax(ts) => {
                let mut best = 0.0f32;
                for t in ts {
                    if t.current_block_id() == Some(target) {
                        best = best.max(t.block_upper_bound());
                    }
                }
                best
            }
            ClauseKind::MaxBranches(branches) => {
                let mut best = 0.0f32;
                for branch in branches {
                    let mut sum = 0.0f32;
                    for c in branch {
                        sum += c.block_upper_bound(target);
                    }
                    if sum > best {
                        best = sum;
                    }
                }
                best
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
        &mut self,
        target: FtsBlockId,
        scores: &mut [f32; FTS_BLOCK_SIZE],
        touched: &mut [u64; 4],
        dl_cache: &mut DlNormCache<'a, S>,
        source: &'a S,
        tier: SimdTier,
    ) -> Result<()> {
        match &mut self.kind {
            ClauseKind::Single(t) => {
                if t.current_block_id() == Some(target) && !t.scorer.is_zero() {
                    t.accumulate(target, scores, touched, dl_cache, source, tier)?;
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
                        .iter_mut()
                        .find(|t| t.current_block_id() == Some(target) && !t.scorer.is_zero())
                        .unwrap();
                    return t.accumulate(target, scores, touched, dl_cache, source, tier);
                }

                let mut local = [0.0f32; FTS_BLOCK_SIZE];
                let mut tmp = [0.0f32; FTS_BLOCK_SIZE];
                let mut local_touched = [0u64; 4];
                let mut first = true;
                for t in ts.iter_mut() {
                    if t.current_block_id() != Some(target) || t.scorer.is_zero() {
                        continue;
                    }
                    if first {
                        t.accumulate(target, &mut local, &mut local_touched, dl_cache, source, tier)?;
                        first = false;
                    } else {
                        tmp.fill(0.0);
                        t.accumulate(target, &mut tmp, &mut local_touched, dl_cache, source, tier)?;
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
                        .iter_mut()
                        .find(|b| b.iter().any(|c| c.next_block_id() == Some(target)))
                        .unwrap();
                    for c in branch.iter_mut() {
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
                for branch in branches.iter_mut() {
                    if !branch.iter().any(|c| c.next_block_id() == Some(target)) {
                        continue;
                    }
                    if first {
                        for c in branch.iter_mut() {
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
                        for c in branch.iter_mut() {
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
        &mut self,
        target: FtsBlockId,
        mask: &[u64; 4],
        scores: &mut [f32; FTS_BLOCK_SIZE],
        dl_cache: &mut DlNormCache<'a, S>,
        source: &'a S,
        tier: SimdTier,
    ) -> Result<()> {
        match &mut self.kind {
            ClauseKind::Single(t) => {
                if t.current_block_id() == Some(target) && !t.scorer.is_zero() {
                    t.accumulate_masked(target, mask, scores, dl_cache, source, tier)?;
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
                        .iter_mut()
                        .find(|t| t.current_block_id() == Some(target) && !t.scorer.is_zero())
                        .unwrap();
                    return t.accumulate_masked(target, mask, scores, dl_cache, source, tier);
                }

                let mut local = [0.0f32; FTS_BLOCK_SIZE];
                let mut tmp = [0.0f32; FTS_BLOCK_SIZE];
                let mut first = true;
                for t in ts.iter_mut() {
                    if t.current_block_id() != Some(target) || t.scorer.is_zero() {
                        continue;
                    }
                    if first {
                        t.accumulate_masked(target, mask, &mut local, dl_cache, source, tier)?;
                        first = false;
                    } else {
                        tmp.fill(0.0);
                        t.accumulate_masked(target, mask, &mut tmp, dl_cache, source, tier)?;
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
                        .iter_mut()
                        .find(|b| b.iter().any(|c| c.next_block_id() == Some(target)))
                        .unwrap();
                    for c in branch.iter_mut() {
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
                for branch in branches.iter_mut() {
                    if !branch.iter().any(|c| c.next_block_id() == Some(target)) {
                        continue;
                    }
                    if first {
                        for c in branch.iter_mut() {
                            if c.next_block_id() == Some(target) {
                                c.accumulate_block_masked(
                                    target, mask, &mut best, dl_cache, source, tier,
                                )?;
                            }
                        }
                        first = false;
                    } else {
                        branch_scores.fill(0.0);
                        for c in branch.iter_mut() {
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
    dl_cache: &mut DlNormCache<'a, S>,
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

            if !exact_tokens.is_empty() {
                let mut token_refs: Vec<&str> =
                    exact_tokens.iter().map(|t| t.text.as_str()).collect();
                token_refs.sort_unstable();
                let mut keys: Vec<String> = Vec::with_capacity(token_refs.len());
                let mut qtfs: Vec<u32> = Vec::with_capacity(token_refs.len());
                for tok in token_refs {
                    if keys.last().is_some_and(|last| last == tok) {
                        *qtfs.last_mut().unwrap() += 1;
                    } else {
                        keys.push(tok.to_owned());
                        qtfs.push(1);
                    }
                }
                let records = source.multi_get_term_records(entry.id, &keys)?;
                let mut active_meta: Vec<(TermId, Bm25TermScorer, f32, usize)> =
                    Vec::with_capacity(keys.len());
                let mut active_recs: Vec<TermRecord> = Vec::with_capacity(keys.len());
                for (qtf, opt_rec) in qtfs.into_iter().zip(records) {
                    let Some(rec) = opt_rec else {
                        continue;
                    };
                    if rec.stats.doc_freq == 0 {
                        continue;
                    }
                    let scorer = Bm25TermScorer::new(
                        weight,
                        attr_stats.doc_count,
                        attr_stats.total_tokens,
                        rec.stats.doc_freq,
                        qtf,
                        k1,
                        b,
                        k3,
                    );
                    let ub = scorer.upper_bound(rec.stats.max_tf, rec.stats.min_dl_over_tf);
                    if ub <= 0.0 {
                        continue;
                    }
                    let dl_group = dl_cache.intern(entry.id, &scorer);
                    active_meta.push((rec.term_id, scorer, ub, dl_group));
                    active_recs.push(rec);
                }
                if !active_recs.is_empty() {
                    let cursors = source.open_skip_cursors(active_recs)?;
                    for ((term_id, scorer, ub, dl_group), cursor) in
                        active_meta.into_iter().zip(cursors)
                    {
                        if cursor.current_block_id().is_some() {
                            out.push(Clause {
                                term_max: ub,
                                kind: ClauseKind::Single(TermStream {
                                    term_id,
                                    scorer,
                                    dl_group,
                                    cursor,
                                }),
                            });
                        }
                    }
                }
            }

            if let Some(prefix) = prefix_token {
                let expanded = source.scan_prefix_terms(entry.id, &prefix, max_exp)?;
                let mut p_meta: Vec<(TermId, Bm25TermScorer, f32, usize)> =
                    Vec::with_capacity(expanded.len());
                let mut p_recs: Vec<TermRecord> = Vec::with_capacity(expanded.len());
                for (_token, rec) in expanded {
                    if rec.stats.doc_freq == 0 {
                        continue;
                    }
                    let scorer = Bm25TermScorer::new(
                        weight,
                        attr_stats.doc_count,
                        attr_stats.total_tokens,
                        rec.stats.doc_freq,
                        1,
                        k1,
                        b,
                        k3,
                    );
                    let ub = scorer.upper_bound(rec.stats.max_tf, rec.stats.min_dl_over_tf);
                    if ub <= 0.0 {
                        continue;
                    }
                    let dl_group = dl_cache.intern(entry.id, &scorer);
                    p_meta.push((rec.term_id, scorer, ub, dl_group));
                    p_recs.push(rec);
                }
                let mut streams = Vec::with_capacity(p_recs.len());
                let mut max_ub = 0.0f32;
                if !p_recs.is_empty() {
                    let cursors = source.open_skip_cursors(p_recs)?;
                    for ((term_id, scorer, ub, dl_group), cursor) in p_meta.into_iter().zip(cursors)
                    {
                        if cursor.current_block_id().is_some() {
                            if ub > max_ub {
                                max_ub = ub;
                            }
                            streams.push(TermStream {
                                term_id,
                                scorer,
                                dl_group,
                                cursor,
                            });
                        }
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

        // Block-Max skip pruning BEFORE reading any posting blocks or doc_len blocks.
        if theta > 0.0 {
            let mut ess_block_ub = 0.0f64;
            for c in &clauses[essential_start..] {
                ess_block_ub += c.block_upper_bound(block_id) as f64;
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

        // Accumulate essential clauses into L1-resident [f32; 256] buffer via SIMD,
        // and advance each processed cursor past `block_id` before shrinking essential_start.
        scores.fill(0.0);
        touched = [0u64; 4];
        for c in &mut clauses[essential_start..] {
            if c.next_block_id() == Some(block_id) {
                c.accumulate_block(block_id, &mut scores, &mut touched, &mut dl_cache, source, tier)?;
                c.advance_past(block_id)?;
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
                    let mut pruned_early = false;
                    // Seek descending by global term_max using out-of-line skip cursors only.
                    for j in (0..essential_start).rev() {
                        clauses[j].seek_to(block_id)?;
                        let ub = clauses[j].block_upper_bound(block_id);
                        non_ess_ubs.push(ub);
                        non_ess_block_ub += ub as f64;
                        if theta > 0.0 {
                            let rem_global = if j > 0 { prefix_max_sum[j - 1] } else { 0.0 };
                            let upper =
                                (max_ess as f64 + non_ess_block_ub + rem_global as f64) as f32;
                            let upper_eps = upper * 1e-6 + 1e-7;
                            if upper + upper_eps < theta {
                                pruned_early = true;
                                break;
                            }
                        }
                    }
                    if !pruned_early {
                        let tight_ub = (max_ess as f64 + non_ess_block_ub) as f32;
                        let tight_eps = tight_ub * 1e-6 + 1e-7;
                        let mut rem_ub = non_ess_block_ub;
                        for (idx, j) in (0..essential_start).rev().enumerate() {
                            let ub_j = non_ess_ubs[idx];
                            if ub_j <= 0.0 {
                                continue;
                            }
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
                            clauses[j].accumulate_block_masked(
                                block_id,
                                &active_mask,
                                &mut scores,
                                &mut dl_cache,
                                source,
                                tier,
                            )?;
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
    use crate::block::{DocLenBlock, PackedPostingBlock, PostingBlockHeader};
    use izzydb_core::config::{DistanceMetric, TreeConfig, fts_block_id, fts_slot};
    use izzydb_core::schema::{AttributeType, CollectionConfig};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    struct MemSource {
        meta: FtsMeta,
        docs: HashMap<FtsDocId, DocId>,
        terms: HashMap<(AttrId, String), TermRecord>,
        postings: HashMap<TermId, Vec<(FtsBlockId, Vec<u8>)>>,
        doc_lens: HashMap<(AttrId, FtsBlockId), Vec<u8>>,
    }

    struct MemSkipCursor {
        blocks: Vec<(FtsBlockId, u32, f32)>,
        inline: Option<(FtsBlockId, Vec<u8>)>,
        idx: usize,
    }

    impl FtsSkipCursor for MemSkipCursor {
        fn current_block_id(&self) -> Option<FtsBlockId> {
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

        fn seek_block(&mut self, target: FtsBlockId) -> Result<()> {
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
        fn seek_block_bytes(&mut self, block_id: FtsBlockId) -> Result<Option<&[u8]>> {
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
            block_id: FtsBlockId,
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
        let mut term_records: HashMap<(AttrId, String), TermRecord> = HashMap::new();

        let title_analyzer = FtsAnalyzer::new(FullTextSearchConfig::default());
        let body_analyzer = FtsAnalyzer::new(FullTextSearchConfig::default());

        for i in 0..n_docs {
            let doc_id = (n_docs - i) as DocId * 3;
            let fts_id = meta.alloc_doc_id().unwrap();
            docs_map.insert(fts_id, doc_id);

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
                    let rec = term_records
                        .entry((attr_id, tok.clone()))
                        .or_insert_with(|| TermRecord::new(meta.alloc_term_id().unwrap()));
                    rec.stats.doc_freq += 1;
                    rec.stats.observe(term.tf, ad.doc_len);
                }
            }
            raw_docs.push((doc_id, title, body));
        }

        let mut postings: HashMap<TermId, Vec<(FtsBlockId, Vec<u8>)>> = HashMap::new();
        for ((aid, tok, bid), mut blk) in block_acc {
            blk.recompute_bounds(&dl_acc[&(aid, bid)]);
            let tid = term_records[&(aid, tok)].term_id;
            postings.entry(tid).or_default().push((bid, blk.encode()));
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
}
