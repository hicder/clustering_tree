//! Inverted full-text index on `RowId` / `RowBlock` coordinates with segment-aware
//! liveness masks and Vectorized Block-Max MAXSCORE BM25 execution.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use roaring::RoaringBitmap;

use izzydb_core::config::{AttrId, DocId, RowBlock, RowId, SegmentId, TermId, for_each_set_bit_256};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::{Filter, FtsPredicateKind, TokenQuery};
use izzydb_core::rank::RankBy;
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_core::schema::{AttributeValue, Attributes, CollectionSchema, ColumnStorage};
use izzydb_fts::{
    BlockMaskProvider, Bm25Result, DEFAULT_MAX_PREFIX_EXPANSIONS, DocLenBlock, DocLenBlockView,
    FtsAnalyzer, FtsDocLenCursor, FtsMeta, FtsPostingSource, FtsSkipCursor, INLINE_TERM_MAX_DF,
    PackedPositionBlock, PackedPostingBlock, PositionBlockView, PostingBlockHeader,
    PostingBlockView, QueryToken, RankModifier, ReusableBuffer, SkipBlock, TermRecord,
    dl_over_tf_lower_bound, execute_rank_by_with,
};
use izzydb_storage::codec::{
    RowView, decode_fts_term_key, encode_fts_term_key, fts_block_key_coord,
};
use izzydb_storage::{
    DBPinnableSlice, RocksDocLenCursor, RocksSkipCursor, SegmentTableSnapshot, Storage, TxContext,
};

use crate::segments::RowIndexHook;

/// Combines segment-level liveness (`SegmentTableSnapshot`) with an optional pushed-down
/// filter [`RowSet`] for Block-Max MAXSCORE block masking.
pub struct LivenessMaskProvider<'a> {
    pub snap: &'a SegmentTableSnapshot,
    pub filter: Option<&'a RowSet>,
}

impl BlockMaskProvider for LivenessMaskProvider<'_> {
    #[inline]
    fn mask(&self, block: RowBlock) -> Option<[u64; 4]> {
        let live = self.snap.block_live_mask(block.seg(), block.block());
        match (live, self.filter) {
            (None, None) => None,
            (Some(m), None) => Some(m),
            (None, Some(rs)) => Some(rs.block_mask(block.seg(), block.block())),
            (Some(mut m), Some(rs)) => {
                let fm = rs.block_mask(block.seg(), block.block());
                for i in 0..4 {
                    m[i] &= fm[i];
                }
                Some(m)
            }
        }
    }
}

/// RocksDB-backed [`FtsPostingSource`] adapter over [`Storage`].
pub struct StorageFtsSource<'a> {
    storage: &'a Storage,
}

impl<'a> StorageFtsSource<'a> {
    pub fn new(storage: &'a Storage) -> Self {
        Self { storage }
    }
}

pub struct StorageSkipCursor<'a>(RocksSkipCursor<'a>);

impl FtsSkipCursor for StorageSkipCursor<'_> {
    #[inline]
    fn current_block_id(&self) -> Option<RowBlock> {
        self.0.current_block_id()
    }

    #[inline]
    fn current_block_bounds(&self) -> (u32, f32) {
        self.0.current_block_bounds()
    }

    #[inline]
    fn inline_posting_bytes(&self) -> Option<&[u8]> {
        self.0.inline_posting_bytes()
    }

    #[inline]
    fn current_posting_bytes(&mut self) -> Result<Option<&[u8]>> {
        self.0.current_posting_bytes()
    }

    #[inline]
    fn next_block(&mut self) -> Result<()> {
        self.0.next_block()
    }

    #[inline]
    fn seek_block(&mut self, target: RowBlock) -> Result<()> {
        self.0.seek_block(target)
    }
}

pub struct StorageDocLenCursor(RocksDocLenCursor);

impl FtsDocLenCursor for StorageDocLenCursor {
    #[inline]
    fn seek_block_bytes(&mut self, block_id: RowBlock) -> Result<Option<&[u8]>> {
        self.0.seek_block_bytes(block_id)
    }
}

impl FtsPostingSource for StorageFtsSource<'_> {
    type SkipCursor<'a>
        = StorageSkipCursor<'a>
    where
        Self: 'a;

    type DocLenCursor<'a>
        = StorageDocLenCursor
    where
        Self: 'a;

    type PostingGuard<'a>
        = DBPinnableSlice<'a>
    where
        Self: 'a;

    fn get_fts_meta(&self) -> Result<Option<FtsMeta>> {
        match self.storage.get_fts_meta_bytes()? {
            Some(bytes) => Ok(Some(FtsMeta::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    fn multi_get_term_records(
        &self,
        attr_id: AttrId,
        tokens: &[String],
    ) -> Result<Vec<Option<TermRecord>>> {
        let keys: Vec<Vec<u8>> = tokens
            .iter()
            .map(|t| encode_fts_term_key(attr_id, t))
            .collect();
        let mut out = Vec::with_capacity(keys.len());
        self.storage
            .multi_get_fts_term_bytes_sorted(&keys, |opt| {
                match opt {
                    Some(bytes) => out.push(Some(TermRecord::decode(bytes)?)),
                    None => out.push(None),
                }
                Ok(())
            })?;
        Ok(out)
    }

    fn scan_prefix_terms(
        &self,
        attr_id: AttrId,
        prefix: &str,
        limit: usize,
    ) -> Result<Vec<(String, TermRecord)>> {
        let prefix_key = encode_fts_term_key(attr_id, prefix);
        let scan_limit = limit.saturating_mul(16).max(2048);
        let rows = self
            .storage
            .scan_fts_term_prefix_bytes(&prefix_key, scan_limit)?;
        let mut matches = Vec::with_capacity(rows.len());
        for (k, v) in rows {
            let (_, token) = decode_fts_term_key(&k)?;
            let record = TermRecord::decode(&v)?;
            matches.push((token, record));
        }
        if !matches.iter().any(|(t, _)| t == prefix)
            && let Some(raw) = self.storage.get_fts_term_bytes(&prefix_key)?
        {
            matches.push((prefix.to_owned(), TermRecord::decode(&raw)?));
        }
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
        Ok(StorageSkipCursor(
            self.storage.open_fts_skip_cursor(record)?,
        ))
    }

    fn open_skip_cursors<'a>(
        &'a self,
        records: Vec<TermRecord>,
    ) -> Result<Vec<Self::SkipCursor<'a>>> {
        Ok(self
            .storage
            .open_fts_skip_cursors(&records)?
            .into_iter()
            .map(StorageSkipCursor)
            .collect())
    }

    fn get_posting_block<'a>(
        &'a self,
        term_id: TermId,
        block_id: RowBlock,
    ) -> Result<Option<Self::PostingGuard<'a>>> {
        self.storage.get_fts_posting_pinned(term_id, block_id)
    }

    fn open_doc_len_cursor<'a>(&'a self, attr_id: AttrId) -> Result<Self::DocLenCursor<'a>> {
        Ok(StorageDocLenCursor(
            self.storage.open_fts_doc_len_cursor(attr_id)?,
        ))
    }

    fn multi_get_doc_ids(&self, row_ids: &[RowId]) -> Result<Vec<Option<DocId>>> {
        self.storage.resolve_doc_ids(row_ids)
    }
}

/// Inverted full-text index manager operating directly on [`RowId`] / [`RowBlock`] coordinates.
pub struct FtsIndex {
    buffers: Mutex<Vec<ReusableBuffer>>,
}

struct PooledBuffer<'a> {
    pool: &'a Mutex<Vec<ReusableBuffer>>,
    buf: Option<ReusableBuffer>,
}

impl Drop for PooledBuffer<'_> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take()
            && let Ok(mut guard) = self.pool.lock()
            && guard.len() < 8
        {
            guard.push(buf);
        }
    }
}

impl std::ops::Deref for PooledBuffer<'_> {
    type Target = ReusableBuffer;

    fn deref(&self) -> &Self::Target {
        self.buf.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for PooledBuffer<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buf.as_mut().unwrap()
    }
}

impl Default for FtsIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl FtsIndex {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(vec![ReusableBuffer::new()]),
        }
    }

    fn acquire_buffer(&self) -> PooledBuffer<'_> {
        let buf = self
            .buffers
            .lock()
            .expect("fts analyzer buffer pool poisoned")
            .pop()
            .unwrap_or_else(ReusableBuffer::new);
        PooledBuffer {
            pool: &self.buffers,
            buf: Some(buf),
        }
    }

    /// Recompute exact live `TermStats` (`doc_freq`, `max_tf`, `min_dl_over_tf`) across all FTS
    /// terms by scanning live posting slots against `SegmentTableSnapshot`.
    pub fn compact_term_stats(&self, storage: &Storage, tx: &mut TxContext<'_>) -> Result<usize> {
        let snap = storage.segment_snapshot();
        let terms = storage.all_fts_terms()?;
        let mut dl_cursors: HashMap<AttrId, RocksDocLenCursor> = HashMap::new();
        let mut updated = 0usize;

        for (key, bytes) in terms {
            let (attr_id, _) = decode_fts_term_key(&key)?;
            let mut rec = TermRecord::decode(&bytes)?;
            let dl_cursor = match dl_cursors.entry(attr_id) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(storage.open_fts_doc_len_cursor(attr_id)?)
                }
            };

            let mut live_df = 0u64;
            let mut max_tf = 0u32;
            let mut min_ratio = f32::INFINITY;

            let mut accumulate_block = |block_id: RowBlock, pbytes: &[u8]| -> Result<()> {
                let pblk = PackedPostingBlock::decode(pbytes)?;
                let mask = snap.block_live_mask(block_id.seg(), block_id.block());
                if mask == Some([0u64; 4]) {
                    return Ok(());
                }
                let dl_view = dl_cursor
                    .seek_block_bytes(block_id)?
                    .map(DocLenBlockView::parse)
                    .transpose()?;
                for (&slot, &tf) in pblk.slots.iter().zip(pblk.tfs.iter()) {
                    if let Some(m) = mask
                        && ((m[(slot as usize) >> 6] >> (slot & 63)) & 1) == 0
                    {
                        continue;
                    }
                    live_df += 1;
                    max_tf = max_tf.max(tf);
                    let dl = dl_view.as_ref().map_or(tf, |dv| dv.dl(slot));
                    min_ratio = min_ratio.min(dl_over_tf_lower_bound(dl, tf));
                }
                Ok(())
            };

            if let Some((ib_id, ref ib_bytes)) = rec.inline_block {
                accumulate_block(ib_id, ib_bytes)?;
            } else {
                let mut cursor = storage.open_fts_skip_cursor(rec.clone())?;
                while let Some(bid) = cursor.current_block_id() {
                    if let Some(slice) = storage.get_fts_posting_pinned(rec.term_id, bid)? {
                        accumulate_block(bid, &slice)?;
                    }
                    cursor.next_block()?;
                }
            }

            if live_df == 0 {
                tx.delete_fts_term(key);
                updated += 1;
            } else if rec.stats.doc_freq != live_df
                || rec.stats.max_tf != max_tf
                || rec.stats.min_dl_over_tf.to_bits() != min_ratio.to_bits()
            {
                rec.stats.doc_freq = live_df;
                rec.stats.max_tf = max_tf;
                rec.stats.min_dl_over_tf = min_ratio;
                tx.put_fts_term_bytes(key, rec.encode());
                updated += 1;
            }
        }
        Ok(updated)
    }

    /// Execute a [`RankBy`] BM25 query with combined liveness + optional pushed-down [`RowSet`] mask
    /// and optional lazy filter callback.
    #[allow(clippy::too_many_arguments)]
    pub fn query_bm25(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        rank_by: &RankBy,
        top_k: usize,
        block_mask: Option<&dyn BlockMaskProvider>,
        lazy_filter: Option<&mut dyn FnMut(RowId) -> Result<bool>>,
    ) -> Result<Vec<Bm25Result>> {
        self.query_bm25_with(
            storage,
            schema,
            rank_by,
            top_k,
            block_mask,
            lazy_filter,
            None,
        )
    }

    /// [`FtsIndex::query_bm25`] with an optional [`RankModifier`].
    #[allow(clippy::too_many_arguments)]
    pub fn query_bm25_with(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        rank_by: &RankBy,
        top_k: usize,
        block_mask: Option<&dyn BlockMaskProvider>,
        lazy_filter: Option<&mut dyn FnMut(RowId) -> Result<bool>>,
        modifier: Option<RankModifier<'_>>,
    ) -> Result<Vec<Bm25Result>> {
        let source = StorageFtsSource::new(storage);
        let mut buf = self.acquire_buffer();
        execute_rank_by_with(
            &source,
            schema,
            rank_by,
            top_k,
            block_mask,
            lazy_filter,
            modifier,
            &mut buf,
        )
    }

    /// Estimate live cardinality of an FTS predicate from `TermRecord.stats.doc_freq`.
    pub fn estimate_fts_predicate(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        kind: FtsPredicateKind,
        attr: &str,
        query: &TokenQuery,
        last_as_prefix: bool,
        live_rows: u64,
    ) -> Result<u64> {
        let entry = schema
            .get_by_name(attr)
            .ok_or_else(|| TreeError::InvalidFilter(format!("unknown attribute '{attr}'")))?;
        let fts_cfg = entry.full_text_search.as_ref().ok_or_else(|| {
            TreeError::InvalidFilter(format!(
                "attribute '{attr}' does not have full_text_search enabled"
            ))
        })?;
        let qtokens = {
            let mut buf = self.acquire_buffer();
            FtsAnalyzer::new(fts_cfg.clone()).analyze_query(query, &mut buf)?
        };
        if qtokens.is_empty() {
            return Ok(0);
        }

        let source = StorageFtsSource::new(storage);
        let mut exact_tokens = exact_query_tokens(&qtokens, last_as_prefix);
        exact_tokens.sort_unstable();
        exact_tokens.dedup();
        let exact_recs = source.multi_get_term_records(entry.id, &exact_tokens)?;
        let prefix_recs: Vec<TermRecord> = if last_as_prefix {
            let prefix = &qtokens.last().unwrap().text;
            source
                .scan_prefix_terms(entry.id, prefix, DEFAULT_MAX_PREFIX_EXPANSIONS)?
                .into_iter()
                .map(|(_, r)| r)
                .filter(|r| r.stats.doc_freq > 0)
                .collect()
        } else {
            Vec::new()
        };

        let est = match kind {
            FtsPredicateKind::AnyToken => {
                let exact_sum: u64 = exact_recs
                    .iter()
                    .flatten()
                    .map(|r| r.stats.doc_freq)
                    .fold(0u64, u64::saturating_add);
                let pref_sum: u64 = prefix_recs
                    .iter()
                    .map(|r| r.stats.doc_freq)
                    .fold(0u64, u64::saturating_add);
                exact_sum.saturating_add(pref_sum)
            }
            FtsPredicateKind::AllTokens | FtsPredicateKind::TokenSequence => {
                let mut min_df = u64::MAX;
                for rec_opt in &exact_recs {
                    let df = rec_opt.as_ref().map_or(0, |r| r.stats.doc_freq);
                    if df == 0 {
                        return Ok(0);
                    }
                    min_df = min_df.min(df);
                }
                if last_as_prefix {
                    let pref_sum: u64 = prefix_recs
                        .iter()
                        .map(|r| r.stats.doc_freq)
                        .fold(0u64, u64::saturating_add);
                    if pref_sum == 0 {
                        return Ok(0);
                    }
                    min_df = min_df.min(pref_sum);
                }
                if min_df == u64::MAX { 0 } else { min_df }
            }
        };
        Ok(est.min(live_rows))
    }

    /// Evaluate a single FTS filter predicate directly into a live [`RowSet`].
    pub fn eval_fts_predicate(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        kind: FtsPredicateKind,
        attr: &str,
        query: &TokenQuery,
        last_as_prefix: bool,
    ) -> Result<RowSet> {
        let entry = schema
            .get_by_name(attr)
            .ok_or_else(|| TreeError::InvalidFilter(format!("unknown attribute '{attr}'")))?;
        let fts_cfg = entry.full_text_search.as_ref().ok_or_else(|| {
            TreeError::InvalidFilter(format!(
                "attribute '{attr}' does not have full_text_search enabled"
            ))
        })?;
        let qtokens = {
            let mut buf = self.acquire_buffer();
            FtsAnalyzer::new(fts_cfg.clone()).analyze_query(query, &mut buf)?
        };
        if qtokens.is_empty() {
            return Ok(RowSet::empty());
        }

        let snap = storage.segment_snapshot();
        let source = StorageFtsSource::new(storage);
        let prefix_expansions = if last_as_prefix {
            let prefix = &qtokens.last().unwrap().text;
            Some(source.scan_prefix_terms(entry.id, prefix, DEFAULT_MAX_PREFIX_EXPANSIONS)?)
        } else {
            None
        };

        match kind {
            FtsPredicateKind::AnyToken => eval_any_token_predicate(
                storage,
                &snap,
                &source,
                entry.id,
                &qtokens,
                prefix_expansions,
            ),
            FtsPredicateKind::AllTokens => eval_all_tokens_predicate(
                storage,
                &snap,
                &source,
                entry.id,
                &qtokens,
                prefix_expansions,
            ),
            FtsPredicateKind::TokenSequence => eval_token_sequence_predicate(
                storage,
                &snap,
                &source,
                entry.id,
                &qtokens,
                prefix_expansions,
            ),
        }
    }

    /// Exact live [`RowSet`] of documents whose BM25 score under pure-lexical `rank_by` is `> 0`.
    pub fn eval_rank_by_match_rowset(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        rank_by: &RankBy,
    ) -> Result<RowSet> {
        let snap = storage.segment_snapshot();
        let source = StorageFtsSource::new(storage);
        self.rank_by_match(storage, &snap, &source, schema, rank_by, 1.0)
    }

    fn rank_by_match(
        &self,
        storage: &Storage,
        snap: &SegmentTableSnapshot,
        source: &StorageFtsSource<'_>,
        schema: &CollectionSchema,
        expr: &RankBy,
        weight: f32,
    ) -> Result<RowSet> {
        if weight <= 0.0 {
            return Ok(RowSet::empty());
        }
        match expr {
            RankBy::Product(children) => {
                let mut w = weight;
                let mut lexical: Option<&RankBy> = None;
                for child in children {
                    match child {
                        RankBy::Const(c) => w *= *c,
                        other => {
                            if lexical.replace(other).is_some() {
                                return Err(TreeError::InvalidQuery(
                                    "rank_by_match requires a pure lexical expression".into(),
                                ));
                            }
                        }
                    }
                }
                match lexical {
                    Some(inner) => self.rank_by_match(storage, snap, source, schema, inner, w),
                    None => Err(TreeError::InvalidQuery(
                        "rank_by_match requires a pure lexical expression".into(),
                    )),
                }
            }
            RankBy::Const(_)
            | RankBy::Filter(_)
            | RankBy::Attribute { .. }
            | RankBy::Saturate { .. }
            | RankBy::Decay { .. } => Err(TreeError::InvalidQuery(
                "rank_by_match requires a pure lexical expression".into(),
            )),
            RankBy::Sum(children) | RankBy::Max(children) => {
                let mut acc = RowSet::empty();
                for child in children {
                    let sub = self.rank_by_match(storage, snap, source, schema, child, weight)?;
                    acc = acc.or(&sub);
                }
                Ok(acc)
            }
            RankBy::Bm25 {
                attr,
                query,
                params,
            } => {
                let entry = schema.get_by_name(attr).ok_or_else(|| {
                    TreeError::InvalidQuery(format!("unknown attribute '{attr}'"))
                })?;
                let fts_cfg = entry.full_text_search.as_ref().ok_or_else(|| {
                    TreeError::InvalidQuery(format!(
                        "attribute '{attr}' does not have full_text_search enabled"
                    ))
                })?;
                let qtokens = {
                    let mut buf = self.acquire_buffer();
                    FtsAnalyzer::new(fts_cfg.clone()).analyze_query(query, &mut buf)?
                };
                if qtokens.is_empty() {
                    return Ok(RowSet::empty());
                }
                let last_as_prefix = params
                    .as_ref()
                    .and_then(|p| p.last_as_prefix)
                    .unwrap_or(false);
                let max_expansions = params
                    .as_ref()
                    .and_then(|p| p.max_prefix_expansions)
                    .unwrap_or(DEFAULT_MAX_PREFIX_EXPANSIONS);

                let mut exact = exact_query_tokens(&qtokens, last_as_prefix);
                exact.sort_unstable();
                exact.dedup();
                let mut records: Vec<TermRecord> = source
                    .multi_get_term_records(entry.id, &exact)?
                    .into_iter()
                    .flatten()
                    .collect();
                if last_as_prefix {
                    let prefix = &qtokens.last().unwrap().text;
                    records.extend(
                        source
                            .scan_prefix_terms(entry.id, prefix, max_expansions)?
                            .into_iter()
                            .map(|(_, rec)| rec),
                    );
                }
                records.retain(|r| r.stats.doc_freq > 0);
                eval_any_token_records(storage, snap, &records)
            }
        }
    }

    /// Evaluate a single FTS predicate against a decoded attribute value (used for lazy filters).
    pub fn matches_fts_attr(
        &self,
        schema: &CollectionSchema,
        kind: FtsPredicateKind,
        attr: &str,
        query: &TokenQuery,
        last_as_prefix: bool,
        val_opt: Option<&AttributeValue>,
    ) -> bool {
        let Some(entry) = schema.get_by_name(attr) else {
            return false;
        };
        let Some(fts_cfg) = entry.full_text_search.as_ref() else {
            return false;
        };
        let Some(val) = val_opt else {
            return false;
        };
        let mut buf = self.acquire_buffer();
        let analyzer = FtsAnalyzer::new(fts_cfg.clone());
        let Ok(Some(doc)) = analyzer.analyze_attribute_value(val, &mut buf) else {
            return false;
        };
        let Ok(qtokens) = analyzer.analyze_query(query, &mut buf) else {
            return false;
        };
        if qtokens.is_empty() {
            return false;
        }
        match kind {
            FtsPredicateKind::AnyToken => qtokens.iter().enumerate().any(|(i, qt)| {
                if last_as_prefix && i == qtokens.len() - 1 {
                    doc.terms.keys().any(|t| t.starts_with(&qt.text))
                } else {
                    doc.terms.contains_key(&qt.text)
                }
            }),
            FtsPredicateKind::AllTokens => qtokens.iter().enumerate().all(|(i, qt)| {
                if last_as_prefix && i == qtokens.len() - 1 {
                    doc.terms.keys().any(|t| t.starts_with(&qt.text))
                } else {
                    doc.terms.contains_key(&qt.text)
                }
            }),
            FtsPredicateKind::TokenSequence => {
                let base = qtokens[0].position;
                let rel: Vec<u32> = qtokens.iter().map(|t| t.position - base).collect();
                if qtokens.len() == 1 {
                    let qt = &qtokens[0];
                    return if last_as_prefix {
                        doc.terms.keys().any(|t| t.starts_with(&qt.text))
                    } else {
                        doc.terms.contains_key(&qt.text)
                    };
                }
                let Some(first_term) = doc.terms.get(&qtokens[0].text) else {
                    return false;
                };
                'outer: for &p0 in &first_term.positions {
                    let exact_end = if last_as_prefix {
                        qtokens.len() - 1
                    } else {
                        qtokens.len()
                    };
                    for i in 1..exact_end {
                        let target = p0 + rel[i];
                        let Some(t) = doc.terms.get(&qtokens[i].text) else {
                            continue 'outer;
                        };
                        if t.positions.binary_search(&target).is_err() {
                            continue 'outer;
                        }
                    }
                    if last_as_prefix {
                        let target = p0 + rel[qtokens.len() - 1];
                        let prefix = &qtokens[qtokens.len() - 1].text;
                        let ok = doc.terms.iter().any(|(tok, term)| {
                            tok.starts_with(prefix) && term.positions.binary_search(&target).is_ok()
                        });
                        if !ok {
                            continue 'outer;
                        }
                    }
                    return true;
                }
                false
            }
        }
    }

    /// Evaluate a full [`Filter`] (supporting mixed attribute + FTS predicates) in memory
    /// against a document's attributes.
    pub fn matches_filter(
        &self,
        schema: &CollectionSchema,
        filter: &Filter,
        attrs: &Attributes,
    ) -> bool {
        filter.matches_with_fts(attrs, &mut |kind, attr, query, last_as_prefix| {
            self.matches_fts_attr(schema, kind, attr, query, last_as_prefix, attrs.get(attr))
        })
    }
}

impl RowIndexHook for FtsIndex {
    fn on_insert(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        row: RowId,
        view: &RowView<'_>,
        _vector: Option<&[f32]>,
        _prev_row: Option<RowId>,
    ) -> Result<()> {
        if !schema.has_fts() {
            return Ok(());
        }

        let mut meta = match tx.get_fts_meta_bytes()? {
            Some(bytes) => FtsMeta::decode(&bytes)?,
            None => FtsMeta::default(),
        };
        let block_id = row.block();
        let slot = row.slot();
        let mut buf = self.acquire_buffer();
        let mut modified = false;

        for entry in schema.fts_entries() {
            if entry.storage != ColumnStorage::Inline {
                continue;
            }
            let Some(val) = view.decode_field(entry.id)? else {
                continue;
            };
            let fts_cfg = entry.full_text_search.as_ref().unwrap();
            let analyzer = FtsAnalyzer::new(fts_cfg.clone());
            let Some(analyzed) = analyzer.analyze_attribute_value(&val, &mut buf)? else {
                continue;
            };
            modified = true;
            {
                let stats = meta.per_attr_stats.entry(entry.id).or_default();
                stats.doc_count = stats.doc_count.saturating_add(1);
                stats.total_tokens = stats.total_tokens.saturating_add(analyzed.doc_len as u64);
            }

            let mut dl_blk = match tx.get_fts_doc_len_bytes(entry.id, block_id)? {
                Some(bytes) => DocLenBlock::decode(&bytes)?,
                None => DocLenBlock::default(),
            };
            dl_blk.set(slot, analyzed.doc_len);
            tx.put_fts_doc_len_bytes(entry.id, block_id, dl_blk.encode());

            for (tok, term) in &analyzed.terms {
                let term_key = encode_fts_term_key(entry.id, tok);
                insert_new_term_posting(
                    tx,
                    &mut meta,
                    FtsDocSlot {
                        block_id,
                        slot,
                        index_positions: fts_cfg.index_positions,
                    },
                    term_key,
                    term.tf,
                    analyzed.doc_len,
                    &term.positions,
                    &dl_blk,
                )?;
            }
        }

        if modified {
            tx.put_fts_meta_bytes(meta.encode());
        }
        Ok(())
    }

    fn on_delete(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        _row: RowId,
        view: &RowView<'_>,
        _had_vector: bool,
    ) -> Result<()> {
        if !schema.has_fts() {
            return Ok(());
        }
        let Some(meta_bytes) = tx.get_fts_meta_bytes()? else {
            return Ok(());
        };
        let mut meta = FtsMeta::decode(&meta_bytes)?;
        let mut buf = self.acquire_buffer();
        let mut modified = false;

        for entry in schema.fts_entries() {
            if entry.storage != ColumnStorage::Inline {
                continue;
            }
            let Some(val) = view.decode_field(entry.id)? else {
                continue;
            };
            let fts_cfg = entry.full_text_search.as_ref().unwrap();
            let analyzer = FtsAnalyzer::new(fts_cfg.clone());
            let Some(analyzed) = analyzer.analyze_attribute_value(&val, &mut buf)? else {
                continue;
            };
            modified = true;
            if let Some(stats) = meta.per_attr_stats.get_mut(&entry.id) {
                stats.doc_count = stats.doc_count.saturating_sub(1);
                stats.total_tokens = stats
                    .total_tokens
                    .saturating_sub(analyzed.doc_len as u64);
            }
            for tok in analyzed.terms.keys() {
                let term_key = encode_fts_term_key(entry.id, tok);
                if let Some(tbytes) = tx.get_fts_term_bytes(&term_key)? {
                    let mut rec = TermRecord::decode(&tbytes)?;
                    rec.stats.doc_freq = rec.stats.doc_freq.saturating_sub(1);
                    tx.put_fts_term_bytes(term_key, rec.encode());
                }
            }
        }

        if modified {
            tx.put_fts_meta_bytes(meta.encode());
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct FtsDocSlot {
    block_id: RowBlock,
    slot: u8,
    index_positions: bool,
}

#[allow(clippy::too_many_arguments)]
fn insert_new_term_posting(
    tx: &mut TxContext<'_>,
    meta: &mut FtsMeta,
    loc: FtsDocSlot,
    term_key: Vec<u8>,
    new_tf: u32,
    new_dl: u32,
    new_positions: &[u32],
    doc_lens: &DocLenBlock,
) -> Result<()> {
    let (mut rec, is_brand_new) = match tx.get_fts_term_bytes(&term_key)? {
        Some(b) => (TermRecord::decode(&b)?, false),
        None => (TermRecord::new(meta.alloc_term_id()?), true),
    };
    rec.stats.doc_freq = rec.stats.doc_freq.saturating_add(1);
    rec.stats.observe(new_tf, new_dl);

    match rec.inline_block.take() {
        None if is_brand_new => {
            let mut blk = PackedPostingBlock::default();
            blk.upsert(loc.slot, new_tf, doc_lens);
            rec.inline_block = Some((loc.block_id, blk.encode()));
        }
        Some((ib_id, ib_bytes)) => {
            let hdr = PostingBlockHeader::parse(&ib_bytes)?;
            if ib_id == loc.block_id && (hdr.count as usize) < INLINE_TERM_MAX_DF {
                let mut blk = PackedPostingBlock::decode(&ib_bytes)?;
                blk.upsert(loc.slot, new_tf, doc_lens);
                rec.inline_block = Some((loc.block_id, blk.encode()));
            } else {
                materialize_inline_block(tx, rec.term_id, ib_id, &ib_bytes)?;
                upsert_posting_and_skip(tx, rec.term_id, loc.block_id, loc.slot, new_tf, doc_lens)?;
            }
        }
        None => {
            upsert_posting_and_skip(tx, rec.term_id, loc.block_id, loc.slot, new_tf, doc_lens)?;
        }
    }
    tx.put_fts_term_bytes(term_key, rec.encode());
    if loc.index_positions {
        upsert_position_block(tx, rec.term_id, loc.block_id, loc.slot, new_positions.to_vec())?;
    }
    Ok(())
}

fn materialize_inline_block(
    tx: &mut TxContext<'_>,
    term_id: TermId,
    block_id: RowBlock,
    encoded: &[u8],
) -> Result<()> {
    let hdr = PostingBlockHeader::parse(encoded)?;
    tx.put_fts_posting_bytes(term_id, block_id, encoded.to_vec());

    let chunk = block_id.chunk();
    let mut skip_blk = match tx.get_fts_skip_bytes(term_id, chunk)? {
        Some(sb) => SkipBlock::decode(&sb)?,
        None => SkipBlock::default(),
    };
    skip_blk.upsert(block_id.rel_block(), hdr.max_tf, hdr.min_dl_over_tf);
    tx.put_fts_skip_bytes(term_id, chunk, skip_blk.encode());
    Ok(())
}

fn upsert_posting_and_skip(
    tx: &mut TxContext<'_>,
    term_id: TermId,
    block_id: RowBlock,
    slot: u8,
    tf: u32,
    doc_lens: &DocLenBlock,
) -> Result<()> {
    let mut blk = match tx.get_fts_posting_bytes(term_id, block_id)? {
        Some(bytes) => PackedPostingBlock::decode(&bytes)?,
        None => PackedPostingBlock::default(),
    };
    blk.upsert(slot, tf, doc_lens);
    let encoded = blk.encode();
    let hdr = PostingBlockHeader::parse(&encoded)?;
    tx.put_fts_posting_bytes(term_id, block_id, encoded);

    let chunk = block_id.chunk();
    let mut skip_blk = match tx.get_fts_skip_bytes(term_id, chunk)? {
        Some(sb) => SkipBlock::decode(&sb)?,
        None => SkipBlock::default(),
    };
    skip_blk.upsert(block_id.rel_block(), hdr.max_tf, hdr.min_dl_over_tf);
    tx.put_fts_skip_bytes(term_id, chunk, skip_blk.encode());
    Ok(())
}

fn upsert_position_block(
    tx: &mut TxContext<'_>,
    term_id: TermId,
    block_id: RowBlock,
    slot: u8,
    positions: Vec<u32>,
) -> Result<()> {
    let mut pblk = match tx.get_fts_position_bytes(term_id, block_id)? {
        Some(bytes) => PackedPositionBlock::decode(&bytes)?,
        None => PackedPositionBlock::default(),
    };
    pblk.upsert(slot, positions);
    tx.put_fts_position_bytes(term_id, block_id, pblk.encode());
    Ok(())
}

// ---------------------------------------------------------------------------
// Token & phrase predicates -> RowSet
// ---------------------------------------------------------------------------

fn exact_query_tokens(qtokens: &[QueryToken], last_as_prefix: bool) -> Vec<String> {
    let slice = if last_as_prefix {
        &qtokens[..qtokens.len() - 1]
    } else {
        qtokens
    };
    slice.iter().map(|t| t.text.clone()).collect()
}

fn eval_any_token_predicate(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    qtokens: &[QueryToken],
    prefix_expansions: Option<Vec<(String, TermRecord)>>,
) -> Result<RowSet> {
    let mut exact_tokens = exact_query_tokens(qtokens, prefix_expansions.is_some());
    exact_tokens.sort_unstable();
    exact_tokens.dedup();
    let mut records: Vec<TermRecord> = source
        .multi_get_term_records(attr_id, &exact_tokens)?
        .into_iter()
        .flatten()
        .collect();
    if let Some(exp) = prefix_expansions {
        records.extend(exp.into_iter().map(|(_, rec)| rec));
    }
    eval_any_token_records(storage, snap, &records)
}

fn eval_all_tokens_predicate(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    qtokens: &[QueryToken],
    prefix_expansions: Option<Vec<(String, TermRecord)>>,
) -> Result<RowSet> {
    let last_as_prefix = prefix_expansions.is_some();
    let exact_tokens = exact_query_tokens(qtokens, last_as_prefix);
    let prefix_records = prefix_expansions.unwrap_or_default();
    if last_as_prefix && prefix_records.is_empty() {
        return Ok(RowSet::empty());
    }
    let Some(exact_map) = fetch_exact_term_map(source, attr_id, &exact_tokens)? else {
        return Ok(RowSet::empty());
    };
    let exact_recs: Vec<TermRecord> = exact_map.into_values().collect();
    let prefix_recs: Vec<TermRecord> = prefix_records.into_iter().map(|(_, r)| r).collect();
    let blocks = eval_all_tokens_blocks(
        storage,
        snap,
        &exact_recs,
        last_as_prefix.then_some(prefix_recs.as_slice()),
    )?;
    Ok(blocks_to_rowset(&blocks))
}

fn eval_token_sequence_predicate(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    qtokens: &[QueryToken],
    prefix_expansions: Option<Vec<(String, TermRecord)>>,
) -> Result<RowSet> {
    if qtokens.len() == 1 {
        let records: Vec<TermRecord> = if let Some(exp) = prefix_expansions {
            exp.into_iter().map(|(_, r)| r).collect()
        } else {
            source
                .multi_get_term_records(attr_id, std::slice::from_ref(&qtokens[0].text))?
                .into_iter()
                .flatten()
                .collect()
        };
        return eval_any_token_records(storage, snap, &records);
    }

    let last_as_prefix = prefix_expansions.is_some();
    let exact_tokens = exact_query_tokens(qtokens, last_as_prefix);
    let prefix_records = prefix_expansions.unwrap_or_default();
    if last_as_prefix && prefix_records.is_empty() {
        return Ok(RowSet::empty());
    }

    let Some(exact_map) = fetch_exact_term_map(source, attr_id, &exact_tokens)? else {
        return Ok(RowSet::empty());
    };
    let exact_recs: Vec<TermRecord> = exact_map.values().cloned().collect();
    let prefix_recs: Vec<TermRecord> = prefix_records.into_iter().map(|(_, r)| r).collect();

    let candidate_blocks = eval_all_tokens_blocks(
        storage,
        snap,
        &exact_recs,
        last_as_prefix.then_some(prefix_recs.as_slice()),
    )?;
    if candidate_blocks.is_empty() {
        return Ok(RowSet::empty());
    }

    let base_pos = qtokens[0].position;
    let rel_positions: Vec<u32> = qtokens.iter().map(|t| t.position - base_pos).collect();
    let exact_end = if last_as_prefix {
        qtokens.len() - 1
    } else {
        qtokens.len()
    };
    let query_term_ids: Vec<TermId> = qtokens[..exact_end]
        .iter()
        .map(|qt| exact_map[&qt.text].term_id)
        .collect();

    let mut unique_term_ids: Vec<TermId> = Vec::with_capacity(exact_end + prefix_recs.len());
    unique_term_ids.extend_from_slice(&query_term_ids);
    unique_term_ids.extend(prefix_recs.iter().map(|pr| pr.term_id));
    unique_term_ids.sort_unstable();
    unique_term_ids.dedup();

    let mut pos_keys: Vec<[u8; 12]> = Vec::with_capacity(unique_term_ids.len());
    let mut matched_rows: Vec<RowId> = Vec::new();
    for (block_id, cand_mask) in candidate_blocks {
        match_block_positions(
            storage,
            block_id,
            &cand_mask,
            &unique_term_ids,
            &query_term_ids,
            &prefix_recs,
            &rel_positions,
            last_as_prefix,
            &mut pos_keys,
            &mut matched_rows,
        )?;
    }
    Ok(RowSet::from_rows(matched_rows))
}

#[allow(clippy::too_many_arguments)]
fn match_block_positions(
    storage: &Storage,
    block_id: RowBlock,
    cand_mask: &[u64; 4],
    unique_term_ids: &[TermId],
    query_term_ids: &[TermId],
    prefix_recs: &[TermRecord],
    rel_positions: &[u32],
    last_as_prefix: bool,
    pos_keys: &mut Vec<[u8; 12]>,
    out_rows: &mut Vec<RowId>,
) -> Result<()> {
    pos_keys.clear();
    for &tid in unique_term_ids {
        pos_keys.push(fts_block_key_coord(tid, block_id));
    }
    let pos_slices = storage.multi_get_fts_positions_pinned(pos_keys)?;

    let mut views: Vec<Option<PositionBlockView<'_>>> = vec![None; unique_term_ids.len()];
    for (idx, opt_slice) in pos_slices.iter().enumerate() {
        if let Some(slice) = opt_slice {
            views[idx] = Some(PositionBlockView::parse(slice)?);
        }
    }

    for &tid in query_term_ids {
        let idx = unique_term_ids.binary_search(&tid).unwrap();
        if views[idx].is_none() {
            return Ok(());
        }
    }

    let prefix_view_indices: Vec<usize> = if last_as_prefix {
        prefix_recs
            .iter()
            .filter_map(|pr| {
                let idx = unique_term_ids.binary_search(&pr.term_id).unwrap();
                views[idx].as_ref().map(|_| idx)
            })
            .collect()
    } else {
        Vec::new()
    };
    if last_as_prefix && prefix_view_indices.is_empty() {
        return Ok(());
    }

    let first_view = views[unique_term_ids.binary_search(&query_term_ids[0]).unwrap()]
        .as_ref()
        .unwrap();
    let follow_view_indices: Vec<usize> = query_term_ids[1..]
        .iter()
        .map(|tid| unique_term_ids.binary_search(tid).unwrap())
        .collect();

    let mut match_err: Option<TreeError> = None;
    for_each_set_bit_256(cand_mask, |slot| {
        if match_err.is_some() {
            return;
        }
        let first_positions = match first_view.slot_positions(slot) {
            Ok(Some(p)) => p,
            Ok(None) => return,
            Err(e) => {
                match_err = Some(e);
                return;
            }
        };

        'start: for &p0 in &first_positions {
            for (offset_i, &v_idx) in follow_view_indices.iter().enumerate() {
                let target_pos = p0 + rel_positions[offset_i + 1];
                let pv = views[v_idx].as_ref().unwrap();
                match pv.slot_contains_pos(slot, target_pos) {
                    Ok(true) => {}
                    Ok(false) => continue 'start,
                    Err(e) => {
                        match_err = Some(e);
                        return;
                    }
                }
            }
            if last_as_prefix {
                let target_pos = p0 + *rel_positions.last().unwrap();
                let mut any_prefix_match = false;
                for &p_idx in &prefix_view_indices {
                    let pv = views[p_idx].as_ref().unwrap();
                    match pv.slot_contains_pos(slot, target_pos) {
                        Ok(true) => {
                            any_prefix_match = true;
                            break;
                        }
                        Ok(false) => {}
                        Err(e) => {
                            match_err = Some(e);
                            return;
                        }
                    }
                }
                if !any_prefix_match {
                    continue 'start;
                }
            }
            out_rows.push(block_id.row(slot));
            return;
        }
    });

    if let Some(e) = match_err {
        return Err(e);
    }
    Ok(())
}

fn eval_any_token_records(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    records: &[TermRecord],
) -> Result<RowSet> {
    let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    let mut add_block = |block_id: RowBlock, bytes: &[u8]| -> Result<()> {
        let Some(live_mask) = snap.block_live_mask(block_id.seg(), block_id.block()) else {
            let view = PostingBlockView::parse(bytes)?;
            let bm = by_seg.entry(block_id.seg()).or_default();
            let base = block_id.block() << 8;
            view.for_each_slot(|slot| {
                bm.insert(base | (slot as u32));
            });
            return Ok(());
        };
        if live_mask == [0u64; 4] {
            return Ok(());
        }
        let mut mask = [0u64; 4];
        PostingBlockView::parse(bytes)?.or_slots_into_mask(&mut mask);
        for w in 0..4 {
            mask[w] &= live_mask[w];
        }
        if mask != [0u64; 4] {
            let bm = by_seg.entry(block_id.seg()).or_default();
            let base = block_id.block() << 8;
            for_each_set_bit_256(&mask, |slot| {
                bm.insert(base | (slot as u32));
            });
        }
        Ok(())
    };

    for rec in records {
        if rec.stats.doc_freq == 0 {
            continue;
        }
        if let Some((block_id, ref ib_bytes)) = rec.inline_block {
            add_block(block_id, ib_bytes)?;
        } else {
            storage.visit_fts_postings_for_term(rec.term_id, |block_id, bytes| {
                add_block(block_id, bytes)
            })?;
        }
    }
    Ok(RowSet::from_sorted_segs(by_seg.into_iter().collect()))
}

fn eval_all_tokens_blocks(
    storage: &Storage,
    snap: &SegmentTableSnapshot,
    exact_records: &[TermRecord],
    prefix_records: Option<&[TermRecord]>,
) -> Result<Vec<(RowBlock, [u64; 4])>> {
    if exact_records.is_empty() {
        let Some(precs) = prefix_records else {
            return Ok(Vec::new());
        };
        let mut by_block: BTreeMap<RowBlock, [u64; 4]> = BTreeMap::new();
        for rec in precs {
            if rec.stats.doc_freq == 0 {
                continue;
            }
            if let Some((block_id, ref ib_bytes)) = rec.inline_block {
                let view = PostingBlockView::parse(ib_bytes)?;
                let mask = by_block.entry(block_id).or_insert([0u64; 4]);
                view.or_slots_into_mask(mask);
            } else {
                storage.visit_fts_postings_for_term(rec.term_id, |block_id, bytes| {
                    let view = PostingBlockView::parse(bytes)?;
                    let mask = by_block.entry(block_id).or_insert([0u64; 4]);
                    view.or_slots_into_mask(mask);
                    Ok(())
                })?;
            }
        }
        let mut out = Vec::with_capacity(by_block.len());
        for (bid, mut mask) in by_block {
            if let Some(live) = snap.block_live_mask(bid.seg(), bid.block()) {
                for w in 0..4 {
                    mask[w] &= live[w];
                }
            }
            if mask != [0u64; 4] {
                out.push((bid, mask));
            }
        }
        return Ok(out);
    }

    let mut ordered: Vec<TermRecord> = exact_records.to_vec();
    ordered.sort_unstable_by_key(|rec| rec.stats.doc_freq);

    let raw_cursors = storage.open_fts_skip_cursors(&ordered)?;
    let mut cursors: Vec<(TermId, RocksSkipCursor<'_>)> = Vec::with_capacity(ordered.len());
    for (rec, cur) in ordered.iter().zip(raw_cursors) {
        if cur.current_block_id().is_none() {
            return Ok(Vec::new());
        }
        cursors.push((rec.term_id, cur));
    }

    let mut prefix_cursors: Vec<(TermId, RocksSkipCursor<'_>)> = match prefix_records {
        Some(precs) => {
            let p_raw = storage.open_fts_skip_cursors(precs)?;
            let mut pcs = Vec::with_capacity(precs.len());
            for (rec, cur) in precs.iter().zip(p_raw) {
                if cur.current_block_id().is_some() {
                    pcs.push((rec.term_id, cur));
                }
            }
            if pcs.is_empty() {
                return Ok(Vec::new());
            }
            pcs
        }
        None => Vec::new(),
    };

    let mut results = Vec::new();
    'block_loop: loop {
        let Some(mut target) = cursors[0].1.current_block_id() else {
            break;
        };

        loop {
            let mut all_match = true;
            for (_, cur) in &mut cursors {
                match cur.current_block_id() {
                    None => break 'block_loop,
                    Some(bid) if bid < target => {
                        cur.seek_block(target)?;
                        match cur.current_block_id() {
                            None => break 'block_loop,
                            Some(nb) if nb > target => {
                                target = nb;
                                all_match = false;
                            }
                            _ => {}
                        }
                    }
                    Some(bid) if bid > target => {
                        target = bid;
                        all_match = false;
                    }
                    _ => {}
                }
            }
            if all_match {
                break;
            }
        }

        let mut mask = snap
            .block_live_mask(target.seg(), target.block())
            .unwrap_or([u64::MAX; 4]);
        if mask != [0u64; 4] {
            for (_, cur) in &mut cursors {
                let mut cur_mask = [0u64; 4];
                if let Some(bytes) = cur.current_posting_bytes()? {
                    PostingBlockView::parse(bytes)?.or_slots_into_mask(&mut cur_mask);
                }
                for w in 0..4 {
                    mask[w] &= cur_mask[w];
                }
                if mask == [0u64; 4] {
                    break;
                }
            }
        }

        if mask != [0u64; 4] && prefix_records.is_some() {
            let mut prefix_mask = [0u64; 4];
            for (_, pcur) in &mut prefix_cursors {
                if let Some(bid) = pcur.current_block_id()
                    && bid < target
                {
                    pcur.seek_block(target)?;
                }
                if pcur.current_block_id() == Some(target)
                    && let Some(bytes) = pcur.current_posting_bytes()?
                {
                    PostingBlockView::parse(bytes)?.or_slots_into_mask(&mut prefix_mask);
                }
            }
            for w in 0..4 {
                mask[w] &= prefix_mask[w];
            }
        }

        if mask != [0u64; 4] {
            results.push((target, mask));
        }

        for (_, cur) in &mut cursors {
            cur.next_block()?;
        }
    }

    Ok(results)
}

fn blocks_to_rowset(blocks: &[(RowBlock, [u64; 4])]) -> RowSet {
    let mut by_seg: BTreeMap<SegmentId, RoaringBitmap> = BTreeMap::new();
    for &(bid, ref mask) in blocks {
        let bm = by_seg.entry(bid.seg()).or_default();
        let base = bid.block() << 8;
        for_each_set_bit_256(mask, |slot| {
            bm.insert(base | (slot as u32));
        });
    }
    RowSet::from_sorted_segs(by_seg.into_iter().collect())
}

fn fetch_exact_term_map(
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    exact_tokens: &[String],
) -> Result<Option<HashMap<String, TermRecord>>> {
    let mut unique_exact: Vec<String> = exact_tokens.to_vec();
    unique_exact.sort_unstable();
    unique_exact.dedup();
    if unique_exact.is_empty() {
        return Ok(Some(HashMap::new()));
    }
    let recs = source.multi_get_term_records(attr_id, &unique_exact)?;
    let mut map = HashMap::with_capacity(unique_exact.len());
    for (tok, opt_rec) in unique_exact.into_iter().zip(recs) {
        let Some(rec) = opt_rec else {
            return Ok(None);
        };
        if rec.stats.doc_freq == 0 {
            return Ok(None);
        }
        map.insert(tok, rec);
    }
    Ok(Some(map))
}
