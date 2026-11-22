//! Inverted full-text index manager coordinating RocksDB writes, token/phrase filters,
//! and Vectorized Block-Max MAXSCORE BM25 execution.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use roaring::RoaringBitmap;

use izzydb_core::config::{
    AttrId, DocId, FtsBlockId, FtsDocId, TermId, for_each_set_bit_256, fts_block_id,
    fts_skip_chunk_id, fts_skip_slot, fts_slot, make_fts_doc_id,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::{Filter, FtsPredicateKind, TokenQuery};
use izzydb_core::rank::RankBy;
use izzydb_core::schema::{AttributeValue, Attributes, CollectionSchema};
use izzydb_fts::{
    AnalyzedDoc, QueryToken, Bm25Result, DEFAULT_MAX_PREFIX_EXPANSIONS, DocLenBlock,
    FtsAnalyzer, FtsDocLenCursor, FtsMeta, FtsPostingSource, FtsSkipCursor, INLINE_TERM_MAX_DF,
    PackedPositionBlock, PackedPostingBlock, PositionBlockView, PostingBlockHeader,
    PostingBlockView, ReusableBuffer, SkipBlock, SkipBlockView, TermRecord, TermStats,
    dl_over_tf_lower_bound, execute_rank_by,
};
use izzydb_storage::codec::{
    decode_fts_term_key, encode_fts_doc_len_key, encode_fts_term_id_key, encode_fts_term_key,
};
use izzydb_storage::{DBPinnableSlice, RocksDocLenCursor, RocksSkipCursor, Storage, TxContext};

/// Maximum posting blocks per term for which `TermStats` (`max_tf`, `min_dl_over_tf`)
/// are recomputed inline on deletes or frequency decreases.
pub(crate) const INLINE_TERM_STATS_RECOMPUTE_MAX_BLOCKS: usize = 64;

#[inline]
pub(crate) fn normalize_opt_attr_val(v: Option<&AttributeValue>) -> Option<&AttributeValue> {
    match v {
        None | Some(AttributeValue::Null) => None,
        Some(other) => Some(other),
    }
}

/// Returns `true` if any FTS-enabled attribute differs between `old_entries` and `new_attrs`.
pub(crate) fn has_fts_attr_changes(
    schema: &CollectionSchema,
    old_entries: &[(AttrId, AttributeValue)],
    new_attrs: &Attributes,
) -> bool {
    if !schema.has_fts() {
        return false;
    }
    let old_map: HashMap<AttrId, &AttributeValue> =
        old_entries.iter().map(|(id, v)| (*id, v)).collect();
    schema.fts_entries().any(|entry| {
        normalize_opt_attr_val(old_map.get(&entry.id).copied())
            != normalize_opt_attr_val(new_attrs.get(&entry.name))
    })
}

/// Returns `true` if any FTS-enabled attribute present in `patch_attrs` differs from `old_entries`.
pub(crate) fn has_fts_patch_changes(
    schema: &CollectionSchema,
    old_entries: &[(AttrId, AttributeValue)],
    patch_attrs: &Attributes,
) -> bool {
    if !schema.has_fts() || patch_attrs.is_empty() {
        return false;
    }
    let old_map: HashMap<AttrId, &AttributeValue> =
        old_entries.iter().map(|(id, v)| (*id, v)).collect();
    schema.fts_entries().any(|entry| {
        patch_attrs.get(&entry.name).is_some_and(|new_val| {
            normalize_opt_attr_val(old_map.get(&entry.id).copied())
                != normalize_opt_attr_val(Some(new_val))
        })
    })
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
    fn current_block_id(&self) -> Option<FtsBlockId> {
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
    fn seek_block(&mut self, target: FtsBlockId) -> Result<()> {
        self.0.seek_block(target)
    }
}

pub struct StorageDocLenCursor(RocksDocLenCursor);

impl FtsDocLenCursor for StorageDocLenCursor {
    #[inline]
    fn seek_block_bytes(&mut self, block_id: FtsBlockId) -> Result<Option<&[u8]>> {
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
        // Ensure exact match is present even if prefix has > scan_limit terms.
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
        block_id: FtsBlockId,
    ) -> Result<Option<Self::PostingGuard<'a>>> {
        self.storage.get_fts_posting_pinned(term_id, block_id)
    }

    fn open_doc_len_cursor<'a>(&'a self, attr_id: AttrId) -> Result<Self::DocLenCursor<'a>> {
        Ok(StorageDocLenCursor(
            self.storage.open_fts_doc_len_cursor(attr_id)?,
        ))
    }

    fn multi_get_doc_ids(&self, fts_doc_ids: &[FtsDocId]) -> Result<Vec<Option<DocId>>> {
        self.storage.multi_get_doc_ids_from_fts(fts_doc_ids)
    }
}

/// Inverted full-text index manager coordinating atomic RocksDB writes, token/phrase filters,
/// and Vectorized Block-Max MAXSCORE BM25 execution.
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

    /// Analyze and update all FTS-enabled attributes for `doc_id` inside `tx`.
    pub fn upsert_doc(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        doc_id: DocId,
        old_entries: Option<&[(AttrId, AttributeValue)]>,
        new_attrs: &Attributes,
    ) -> Result<()> {
        if !schema.has_fts() {
            return Ok(());
        }

        let mut meta = match tx.get_fts_meta_bytes()? {
            Some(bytes) => FtsMeta::decode(&bytes)?,
            None => FtsMeta::default(),
        };

        let fts_doc_id = match tx.get_fts_doc_id(doc_id)? {
            Some(existing) => existing,
            None => {
                let allocated = if let Some(min_free) = tx.get_fts_free_ids()?.min() {
                    tx.get_fts_free_ids_mut()?.remove(min_free);
                    min_free
                } else {
                    meta.alloc_doc_id()?
                };
                tx.put_fts_doc_mapping(doc_id, allocated);
                allocated
            }
        };
        let block_id = fts_block_id(fts_doc_id);
        let slot = fts_slot(fts_doc_id);

        let old_map: HashMap<AttrId, &AttributeValue> = match old_entries {
            Some(entries) => entries.iter().map(|(id, v)| (*id, v)).collect(),
            None => HashMap::new(),
        };

        let mut buf = self.acquire_buffer();
        for entry in schema.fts_entries() {
            let old_val = old_map.get(&entry.id).copied();
            let new_val = new_attrs.get(&entry.name);
            if old_entries.is_some()
                && normalize_opt_attr_val(old_val) == normalize_opt_attr_val(new_val)
            {
                continue;
            }

            let fts_cfg = entry.full_text_search.as_ref().unwrap();
            let analyzer = FtsAnalyzer::new(fts_cfg.clone());

            let old_analyzed = match old_val {
                Some(val) => analyzer.analyze_attribute_value(val, &mut buf)?,
                None => None,
            };
            let new_analyzed = match new_val {
                Some(val) => analyzer.analyze_attribute_value(val, &mut buf)?,
                None => None,
            };

            {
                let stats = meta.per_attr_stats.entry(entry.id).or_default();
                match (&old_analyzed, &new_analyzed) {
                    (None, None) => continue,
                    (None, Some(n)) => {
                        stats.doc_count = stats.doc_count.saturating_add(1);
                        stats.total_tokens = stats.total_tokens.saturating_add(n.doc_len as u64);
                    }
                    (Some(o), None) => {
                        stats.doc_count = stats.doc_count.saturating_sub(1);
                        stats.total_tokens = stats.total_tokens.saturating_sub(o.doc_len as u64);
                    }
                    (Some(o), Some(n)) => {
                        stats.total_tokens = stats
                            .total_tokens
                            .saturating_sub(o.doc_len as u64)
                            .saturating_add(n.doc_len as u64);
                    }
                }
            }

            apply_attr_token_diff(
                tx,
                &mut meta,
                FtsDocSlot {
                    attr_id: entry.id,
                    block_id,
                    slot,
                    index_positions: fts_cfg.index_positions,
                },
                old_analyzed.as_ref(),
                new_analyzed.as_ref(),
            )?;
        }

        tx.put_fts_meta_bytes(meta.encode());
        Ok(())
    }

    /// Remove all FTS postings and document mappings for `doc_id` inside `tx`.
    pub fn remove_doc(
        &self,
        tx: &mut TxContext<'_>,
        schema: &CollectionSchema,
        doc_id: DocId,
        old_entries: &[(AttrId, AttributeValue)],
    ) -> Result<()> {
        if !schema.has_fts() {
            return Ok(());
        }
        let Some(fts_doc_id) = tx.get_fts_doc_id(doc_id)? else {
            return Ok(());
        };
        let mut meta = match tx.get_fts_meta_bytes()? {
            Some(bytes) => FtsMeta::decode(&bytes)?,
            None => FtsMeta::default(),
        };
        let block_id = fts_block_id(fts_doc_id);
        let slot = fts_slot(fts_doc_id);

        let old_map: HashMap<AttrId, &AttributeValue> =
            old_entries.iter().map(|(id, v)| (*id, v)).collect();

        let mut buf = self.acquire_buffer();
        for entry in schema.fts_entries() {
            let fts_cfg = entry.full_text_search.as_ref().unwrap();
            let analyzer = FtsAnalyzer::new(fts_cfg.clone());
            let Some(val) = old_map.get(&entry.id) else {
                continue;
            };
            let Some(old_analyzed) = analyzer.analyze_attribute_value(val, &mut buf)? else {
                continue;
            };
            if let Some(stats) = meta.per_attr_stats.get_mut(&entry.id) {
                stats.doc_count = stats.doc_count.saturating_sub(1);
                stats.total_tokens = stats
                    .total_tokens
                    .saturating_sub(old_analyzed.doc_len as u64);
            }
            apply_attr_token_diff(
                tx,
                &mut meta,
                FtsDocSlot {
                    attr_id: entry.id,
                    block_id,
                    slot,
                    index_positions: fts_cfg.index_positions,
                },
                Some(&old_analyzed),
                None,
            )?;
        }

        tx.delete_fts_doc_mapping(doc_id, fts_doc_id);
        let free_ids = tx.get_fts_free_ids_mut()?;
        free_ids.insert(fts_doc_id);
        while meta.next_fts_doc_id > 0 && free_ids.remove(meta.next_fts_doc_id - 1) {
            meta.next_fts_doc_id -= 1;
        }
        tx.put_fts_meta_bytes(meta.encode());
        Ok(())
    }

    /// Recompute exact `TermStats` (`max_tf`, `min_dl_over_tf`) across all FTS terms by scanning
    /// posting block headers, tightening any high-DF terms excluded from inline recomputation.
    pub fn compact_term_stats(&self, storage: &Storage, tx: &mut TxContext<'_>) -> Result<usize> {
        let terms = storage.all_fts_terms()?;
        let mut updated = 0usize;
        for (key, bytes) in terms {
            let mut rec = TermRecord::decode(&bytes)?;
            let mut doc_freq = 0u64;
            let mut max_tf = 0u32;
            let mut min_ratio = f32::INFINITY;
            if let Some((_, ref ib_bytes)) = rec.inline_block {
                let header = PostingBlockHeader::parse(ib_bytes)?;
                doc_freq = header.count as u64;
                max_tf = header.max_tf;
                min_ratio = header.min_dl_over_tf;
            } else {
                let mut cursor = storage.open_fts_skip_cursor(rec.clone())?;
                while let Some(bid) = cursor.current_block_id() {
                    if let Some(slice) = storage.get_fts_posting_pinned(rec.term_id, bid)? {
                        let header = PostingBlockHeader::parse(&slice)?;
                        doc_freq = doc_freq.saturating_add(header.count as u64);
                        max_tf = max_tf.max(header.max_tf);
                        min_ratio = min_ratio.min(header.min_dl_over_tf);
                    }
                    cursor.next_block()?;
                }
            }
            if doc_freq == 0 {
                tx.delete_fts_term(key);
                updated += 1;
            } else if rec.stats.doc_freq != doc_freq
                || rec.stats.max_tf != max_tf
                || rec.stats.min_dl_over_tf.to_bits() != min_ratio.to_bits()
            {
                rec.stats.doc_freq = doc_freq;
                rec.stats.max_tf = max_tf;
                rec.stats.min_dl_over_tf = min_ratio;
                tx.put_fts_term_bytes(key, rec.encode());
                updated += 1;
            }
        }
        Ok(updated)
    }

    /// Execute a [`RankBy`] BM25 query with optional pushdown `doc_filter` and `lazy_filter`.
    pub fn query_bm25(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        rank_by: &RankBy,
        top_k: usize,
        doc_filter: Option<&RoaringBitmap>,
        lazy_filter: Option<&mut dyn FnMut(DocId) -> Result<bool>>,
    ) -> Result<Vec<Bm25Result>> {
        let source = StorageFtsSource::new(storage);
        let mut buf = self.acquire_buffer();
        execute_rank_by(
            &source,
            schema,
            rank_by,
            top_k,
            doc_filter,
            lazy_filter,
            &mut buf,
        )
    }

    /// Evaluate a single FTS filter predicate into a `RoaringBitmap` of matching `FtsDocId`s.
    pub fn eval_fts_predicate(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        kind: FtsPredicateKind,
        attr: &str,
        query: &TokenQuery,
        last_as_prefix: bool,
    ) -> Result<RoaringBitmap> {
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
            return Ok(RoaringBitmap::new());
        }

        let source = StorageFtsSource::new(storage);
        let prefix_expansions = if last_as_prefix {
            let prefix = &qtokens.last().unwrap().text;
            Some(source.scan_prefix_terms(entry.id, prefix, DEFAULT_MAX_PREFIX_EXPANSIONS)?)
        } else {
            None
        };

        match kind {
            FtsPredicateKind::AnyToken => {
                eval_any_token_predicate(storage, &source, entry.id, &qtokens, prefix_expansions)
            }
            FtsPredicateKind::AllTokens => {
                eval_all_tokens_predicate(storage, &source, entry.id, &qtokens, prefix_expansions)
            }
            FtsPredicateKind::TokenSequence => eval_token_sequence_predicate(
                storage,
                &source,
                entry.id,
                &qtokens,
                prefix_expansions,
            ),
        }
    }

    /// Exact set of `FtsDocId`s whose BM25 score under `rank_by` is `> 0`.
    ///
    /// BM25 IDF is strictly positive whenever `df > 0`, so a document scores `> 0` iff it contains
    /// at least one resolved term reachable through a positive-weight path of the `RankBy` AST
    /// (`Sum` / `Max` = union, `Product(w, e)` = `e` when the accumulated weight is `> 0`).
    /// Term resolution mirrors MAXSCORE compilation (query analysis, `last_as_prefix`, and
    /// `max_prefix_expansions`) but needs no float math or doc-length reads.
    pub fn eval_rank_by_match_bitmap(
        &self,
        storage: &Storage,
        schema: &CollectionSchema,
        rank_by: &RankBy,
    ) -> Result<RoaringBitmap> {
        let source = StorageFtsSource::new(storage);
        self.rank_by_match(storage, &source, schema, rank_by, 1.0)
    }

    fn rank_by_match(
        &self,
        storage: &Storage,
        source: &StorageFtsSource<'_>,
        schema: &CollectionSchema,
        expr: &RankBy,
        weight: f32,
    ) -> Result<RoaringBitmap> {
        if weight <= 0.0 {
            return Ok(RoaringBitmap::new());
        }
        match expr {
            RankBy::Product(w, inner) => {
                self.rank_by_match(storage, source, schema, inner, weight * *w)
            }
            RankBy::Sum(children) | RankBy::Max(children) => {
                let mut acc = RoaringBitmap::new();
                for child in children {
                    acc |= self.rank_by_match(storage, source, schema, child, weight)?;
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
                    return Ok(RoaringBitmap::new());
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
                eval_any_token_records(storage, &records)
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
        let mut buf = self.acquire_buffer();
        filter.matches_with_fts(attrs, &mut |kind, attr, query, last_as_prefix| {
            let Some(entry) = schema.get_by_name(attr) else {
                return false;
            };
            let Some(fts_cfg) = entry.full_text_search.as_ref() else {
                return false;
            };
            let Some(val) = attrs.get(attr) else {
                return false;
            };
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
        })
    }
}

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
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    qtokens: &[QueryToken],
    prefix_expansions: Option<Vec<(String, TermRecord)>>,
) -> Result<RoaringBitmap> {
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
    eval_any_token_records(storage, &records)
}

fn eval_all_tokens_predicate(
    storage: &Storage,
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    qtokens: &[QueryToken],
    prefix_expansions: Option<Vec<(String, TermRecord)>>,
) -> Result<RoaringBitmap> {
    let last_as_prefix = prefix_expansions.is_some();
    let exact_tokens = exact_query_tokens(qtokens, last_as_prefix);
    let prefix_records = prefix_expansions.unwrap_or_default();
    if last_as_prefix && prefix_records.is_empty() {
        return Ok(RoaringBitmap::new());
    }
    let Some(exact_map) = fetch_exact_term_map(source, attr_id, &exact_tokens)? else {
        return Ok(RoaringBitmap::new());
    };
    let exact_recs: Vec<TermRecord> = exact_map.into_values().collect();
    let prefix_recs: Vec<TermRecord> = prefix_records.into_iter().map(|(_, r)| r).collect();
    let blocks = eval_all_tokens_blocks(
        storage,
        &exact_recs,
        last_as_prefix.then_some(prefix_recs.as_slice()),
    )?;
    let mut bm = RoaringBitmap::new();
    for (block_id, mask) in blocks {
        insert_mask_into_bitmap(block_id, &mask, &mut bm);
    }
    Ok(bm)
}

fn eval_token_sequence_predicate(
    storage: &Storage,
    source: &StorageFtsSource<'_>,
    attr_id: AttrId,
    qtokens: &[QueryToken],
    prefix_expansions: Option<Vec<(String, TermRecord)>>,
) -> Result<RoaringBitmap> {
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
        return eval_any_token_records(storage, &records);
    }

    let last_as_prefix = prefix_expansions.is_some();
    let exact_tokens = exact_query_tokens(qtokens, last_as_prefix);
    let prefix_records = prefix_expansions.unwrap_or_default();
    if last_as_prefix && prefix_records.is_empty() {
        return Ok(RoaringBitmap::new());
    }

    let Some(exact_map) = fetch_exact_term_map(source, attr_id, &exact_tokens)? else {
        return Ok(RoaringBitmap::new());
    };
    let exact_recs: Vec<TermRecord> = exact_map.values().cloned().collect();
    let prefix_recs: Vec<TermRecord> = prefix_records.into_iter().map(|(_, r)| r).collect();

    let candidate_blocks = eval_all_tokens_blocks(
        storage,
        &exact_recs,
        last_as_prefix.then_some(prefix_recs.as_slice()),
    )?;
    if candidate_blocks.is_empty() {
        return Ok(RoaringBitmap::new());
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

    let mut pos_keys: Vec<[u8; 8]> = Vec::with_capacity(unique_term_ids.len());
    let mut result = RoaringBitmap::new();
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
            &mut result,
        )?;
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn match_block_positions(
    storage: &Storage,
    block_id: FtsBlockId,
    cand_mask: &[u64; 4],
    unique_term_ids: &[TermId],
    query_term_ids: &[TermId],
    prefix_recs: &[TermRecord],
    rel_positions: &[u32],
    last_as_prefix: bool,
    pos_keys: &mut Vec<[u8; 8]>,
    result: &mut RoaringBitmap,
) -> Result<()> {
    pos_keys.clear();
    for &tid in unique_term_ids {
        pos_keys.push(encode_fts_term_id_key(tid, block_id));
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
            result.insert(make_fts_doc_id(block_id, slot));
            return;
        }
    });

    if let Some(e) = match_err {
        return Err(e);
    }
    Ok(())
}

fn insert_mask_into_bitmap(block_id: FtsBlockId, mask: &[u64; 4], out: &mut RoaringBitmap) {
    for_each_set_bit_256(mask, |slot| {
        out.insert(make_fts_doc_id(block_id, slot));
    });
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

fn eval_any_token_records(storage: &Storage, records: &[TermRecord]) -> Result<RoaringBitmap> {
    let mut out = RoaringBitmap::new();
    for rec in records {
        if let Some((block_id, ref ib_bytes)) = rec.inline_block {
            let view = PostingBlockView::parse(ib_bytes)?;
            view.for_each_slot(|slot| {
                out.insert(make_fts_doc_id(block_id, slot));
            });
        } else {
            storage.visit_fts_postings_for_term(rec.term_id, |block_id, bytes| {
                let view = PostingBlockView::parse(bytes)?;
                view.for_each_slot(|slot| {
                    out.insert(make_fts_doc_id(block_id, slot));
                });
                Ok(())
            })?;
        }
    }
    Ok(out)
}

fn eval_all_tokens_blocks(
    storage: &Storage,
    exact_records: &[TermRecord],
    prefix_records: Option<&[TermRecord]>,
) -> Result<Vec<(FtsBlockId, [u64; 4])>> {
    if exact_records.is_empty() {
        let Some(precs) = prefix_records else {
            return Ok(Vec::new());
        };
        let mut by_block: BTreeMap<FtsBlockId, [u64; 4]> = BTreeMap::new();
        for rec in precs {
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
        return Ok(by_block.into_iter().collect());
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

        // Gallop / intersect block_ids using cf_fts_skips (and inline records) before touching cf_fts_postings.
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

        // Intersect intra-block slot bitmasks via lazy sequential posting iterators.
        let mut mask = [u64::MAX; 4];
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

#[derive(Clone, Copy)]
struct FtsDocSlot {
    attr_id: AttrId,
    block_id: FtsBlockId,
    slot: u8,
    index_positions: bool,
}

fn apply_attr_token_diff(
    tx: &mut TxContext<'_>,
    meta: &mut FtsMeta,
    loc: FtsDocSlot,
    old_doc: Option<&AnalyzedDoc>,
    new_doc: Option<&AnalyzedDoc>,
) -> Result<()> {
    let old_dl = old_doc.map_or(0, |d| d.doc_len);
    let new_dl = new_doc.map_or(0, |d| d.doc_len);

    let dl_key = encode_fts_doc_len_key(loc.attr_id, loc.block_id);
    let mut doc_len_block = match tx.get_fts_doc_len_bytes(&dl_key)? {
        Some(bytes) => DocLenBlock::decode(&bytes)?,
        None => DocLenBlock::default(),
    };
    if old_dl != new_dl {
        if new_dl > 0 {
            doc_len_block.set(loc.slot, new_dl);
        } else {
            doc_len_block.remove(loc.slot);
        }
        if doc_len_block.is_empty() {
            tx.delete_fts_doc_len(dl_key);
        } else {
            tx.put_fts_doc_len_bytes(dl_key, doc_len_block.encode());
        }
    }

    // 1. Removals
    if let Some(old) = old_doc {
        for (tok, old_term) in &old.terms {
            if new_doc.is_some_and(|n| n.terms.contains_key(tok)) {
                continue;
            }
            remove_token_posting(tx, loc, tok, old_term.tf, old_dl, &doc_len_block)?;
        }
    }

    // 2. Additions and updates
    if let Some(new) = new_doc {
        for (tok, new_term) in &new.terms {
            let term_key = encode_fts_term_key(loc.attr_id, tok);
            if let Some(old_term) = old_doc.and_then(|o| o.terms.get(tok)) {
                update_existing_term_posting(
                    tx,
                    meta,
                    loc,
                    term_key,
                    old_term.tf,
                    old_dl,
                    new_term.tf,
                    new_dl,
                    &old_term.positions,
                    &new_term.positions,
                    &doc_len_block,
                )?;
            } else {
                insert_new_term_posting(
                    tx,
                    meta,
                    loc,
                    term_key,
                    new_term.tf,
                    new_dl,
                    &new_term.positions,
                    &doc_len_block,
                )?;
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_existing_term_posting(
    tx: &mut TxContext<'_>,
    meta: &mut FtsMeta,
    loc: FtsDocSlot,
    term_key: Vec<u8>,
    old_tf: u32,
    old_dl: u32,
    new_tf: u32,
    new_dl: u32,
    old_positions: &[u32],
    new_positions: &[u32],
    doc_len_block: &DocLenBlock,
) -> Result<()> {
    let tf_or_dl_changed = old_tf != new_tf || old_dl != new_dl;
    let pos_changed = loc.index_positions && old_positions != new_positions;
    let mut rec = match tx.get_fts_term_bytes(&term_key)? {
        Some(b) => TermRecord::decode(&b)?,
        None => TermRecord::new(meta.alloc_term_id()?),
    };
    if tf_or_dl_changed {
        let mut was_inline_same_block = false;
        if let Some((ib_id, ref ib_bytes)) = rec.inline_block {
            if ib_id == loc.block_id {
                was_inline_same_block = true;
                let mut blk = PackedPostingBlock::decode(ib_bytes)?;
                blk.upsert(loc.slot, new_tf, doc_len_block);
                let encoded = blk.encode();
                let hdr = PostingBlockHeader::parse(&encoded)?;
                rec.stats.doc_freq = hdr.count as u64;
                rec.stats.max_tf = hdr.max_tf;
                rec.stats.min_dl_over_tf = hdr.min_dl_over_tf;
                rec.inline_block = Some((loc.block_id, encoded));
            } else {
                let (ib_id, ib_bytes) = rec.inline_block.take().unwrap();
                materialize_inline_block(tx, rec.term_id, ib_id, &ib_bytes)?;
                upsert_posting_and_skip(tx, rec.term_id, loc.block_id, loc.slot, new_tf, doc_len_block)?;
            }
        } else {
            upsert_posting_and_skip(tx, rec.term_id, loc.block_id, loc.slot, new_tf, doc_len_block)?;
        }

        if !was_inline_same_block {
            let lowered_max_tf = new_tf < old_tf && old_tf == rec.stats.max_tf;
            let old_ratio = dl_over_tf_lower_bound(old_dl, old_tf);
            let new_ratio = dl_over_tf_lower_bound(new_dl, new_tf);
            let raised_min_ratio = new_ratio > old_ratio && old_ratio <= rec.stats.min_dl_over_tf;
            rec.stats.observe(new_tf, new_dl);
            if lowered_max_tf || raised_min_ratio {
                maybe_recompute_term_stats_in_tx(tx, rec.term_id, &mut rec.stats)?;
            }
        }
        tx.put_fts_term_bytes(term_key, rec.encode());
    }
    if pos_changed {
        upsert_position_block(tx, rec.term_id, loc.block_id, loc.slot, new_positions.to_vec())?;
    }
    Ok(())
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
    doc_len_block: &DocLenBlock,
) -> Result<()> {
    let mut rec = match tx.get_fts_term_bytes(&term_key)? {
        Some(b) => TermRecord::decode(&b)?,
        None => TermRecord::new(meta.alloc_term_id()?),
    };
    rec.stats.doc_freq = rec.stats.doc_freq.saturating_add(1);
    rec.stats.observe(new_tf, new_dl);

    match rec.inline_block.take() {
        None if rec.stats.doc_freq == 1 => {
            let mut blk = PackedPostingBlock::default();
            blk.upsert(loc.slot, new_tf, doc_len_block);
            rec.inline_block = Some((loc.block_id, blk.encode()));
        }
        Some((ib_id, ib_bytes)) => {
            if ib_id == loc.block_id && rec.stats.doc_freq <= INLINE_TERM_MAX_DF as u64 {
                let mut blk = PackedPostingBlock::decode(&ib_bytes)?;
                blk.upsert(loc.slot, new_tf, doc_len_block);
                rec.inline_block = Some((loc.block_id, blk.encode()));
            } else {
                materialize_inline_block(tx, rec.term_id, ib_id, &ib_bytes)?;
                upsert_posting_and_skip(tx, rec.term_id, loc.block_id, loc.slot, new_tf, doc_len_block)?;
            }
        }
        None => {
            upsert_posting_and_skip(tx, rec.term_id, loc.block_id, loc.slot, new_tf, doc_len_block)?;
        }
    }
    tx.put_fts_term_bytes(term_key, rec.encode());
    if loc.index_positions {
        upsert_position_block(tx, rec.term_id, loc.block_id, loc.slot, new_positions.to_vec())?;
    }
    Ok(())
}

fn remove_token_posting(
    tx: &mut TxContext<'_>,
    loc: FtsDocSlot,
    token: &str,
    old_tf: u32,
    old_dl: u32,
    doc_lens: &DocLenBlock,
) -> Result<()> {
    let term_key = encode_fts_term_key(loc.attr_id, token);
    let Some(bytes) = tx.get_fts_term_bytes(&term_key)? else {
        return Ok(());
    };
    let mut rec = TermRecord::decode(&bytes)?;
    let term_id = rec.term_id;

    if loc.index_positions {
        let pos_key = encode_fts_term_id_key(term_id, loc.block_id);
        if let Some(pbytes) = tx.get_fts_position_bytes(&pos_key)? {
            let mut pblk = PackedPositionBlock::decode(&pbytes)?;
            if pblk.remove(loc.slot) {
                if pblk.is_empty() {
                    tx.delete_fts_position(pos_key);
                } else {
                    tx.put_fts_position_bytes(pos_key, pblk.encode());
                }
            }
        }
    }

    if let Some((ib_id, ib_bytes)) = rec.inline_block.take() {
        if ib_id == loc.block_id {
            let mut blk = PackedPostingBlock::decode(&ib_bytes)?;
            let _ = blk.remove(loc.slot, doc_lens);
            if blk.is_empty() {
                tx.delete_fts_term(term_key);
            } else {
                let encoded = blk.encode();
                let hdr = PostingBlockHeader::parse(&encoded)?;
                rec.stats.doc_freq = hdr.count as u64;
                rec.stats.max_tf = hdr.max_tf;
                rec.stats.min_dl_over_tf = hdr.min_dl_over_tf;
                rec.inline_block = Some((ib_id, encoded));
                tx.put_fts_term_bytes(term_key, rec.encode());
            }
        } else {
            rec.inline_block = Some((ib_id, ib_bytes));
            tx.put_fts_term_bytes(term_key, rec.encode());
        }
        return Ok(());
    }

    let pkey = encode_fts_term_id_key(term_id, loc.block_id);
    if let Some(pbytes) = tx.get_fts_posting_bytes(&pkey)? {
        let mut blk = PackedPostingBlock::decode(&pbytes)?;
        if blk.remove(loc.slot, doc_lens) {
            let skey = encode_fts_term_id_key(term_id, fts_skip_chunk_id(loc.block_id));
            let rel_slot = fts_skip_slot(loc.block_id);
            let mut skip_blk = match tx.get_fts_skip_bytes(&skey)? {
                Some(sb) => SkipBlock::decode(&sb)?,
                None => SkipBlock::default(),
            };
            if blk.is_empty() {
                tx.delete_fts_posting(pkey);
                skip_blk.remove(rel_slot);
            } else {
                let encoded = blk.encode();
                let hdr = PostingBlockHeader::parse(&encoded)?;
                tx.put_fts_posting_bytes(pkey, encoded);
                skip_blk.upsert(rel_slot, hdr.max_tf, hdr.min_dl_over_tf);
            }
            if skip_blk.is_empty() {
                tx.delete_fts_skip(skey);
            } else {
                tx.put_fts_skip_bytes(skey, skip_blk.encode());
            }
        }
    }

    rec.stats.doc_freq = rec.stats.doc_freq.saturating_sub(1);
    if rec.stats.doc_freq == 0 {
        tx.delete_fts_term(term_key);
    } else {
        let hit_max_tf = old_tf == rec.stats.max_tf;
        let hit_min_ratio = dl_over_tf_lower_bound(old_dl, old_tf) <= rec.stats.min_dl_over_tf;
        if hit_max_tf || hit_min_ratio {
            maybe_recompute_term_stats_in_tx(tx, term_id, &mut rec.stats)?;
        }
        tx.put_fts_term_bytes(term_key, rec.encode());
    }
    Ok(())
}

fn maybe_recompute_term_stats_in_tx(
    tx: &TxContext<'_>,
    term_id: TermId,
    ts: &mut TermStats,
) -> Result<bool> {
    let mut block_count = 0usize;
    let mut max_tf = 0u32;
    let mut min_ratio = f32::INFINITY;
    let mut aborted = false;

    tx.visit_fts_skip_blocks(term_id, |_chunk_id, bytes| {
        let view = SkipBlockView::parse(bytes)?;
        block_count += view.count;
        if block_count > INLINE_TERM_STATS_RECOMPUTE_MAX_BLOCKS {
            aborted = true;
            return Ok(false);
        }
        for i in 0..view.count {
            max_tf = max_tf.max(view.max_tf(i));
            min_ratio = min_ratio.min(view.min_dl_over_tf(i));
        }
        Ok(true)
    })?;

    if !aborted && block_count > 0 {
        ts.max_tf = max_tf;
        ts.min_dl_over_tf = min_ratio;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn materialize_inline_block(
    tx: &mut TxContext<'_>,
    term_id: TermId,
    block_id: FtsBlockId,
    encoded: &[u8],
) -> Result<()> {
    let hdr = PostingBlockHeader::parse(encoded)?;
    let pkey = encode_fts_term_id_key(term_id, block_id);
    tx.put_fts_posting_bytes(pkey, encoded.to_vec());

    let skey = encode_fts_term_id_key(term_id, fts_skip_chunk_id(block_id));
    let mut skip_blk = match tx.get_fts_skip_bytes(&skey)? {
        Some(sb) => SkipBlock::decode(&sb)?,
        None => SkipBlock::default(),
    };
    skip_blk.upsert(fts_skip_slot(block_id), hdr.max_tf, hdr.min_dl_over_tf);
    tx.put_fts_skip_bytes(skey, skip_blk.encode());
    Ok(())
}

fn upsert_posting_and_skip(
    tx: &mut TxContext<'_>,
    term_id: TermId,
    block_id: FtsBlockId,
    slot: u8,
    tf: u32,
    doc_lens: &DocLenBlock,
) -> Result<()> {
    let pkey = encode_fts_term_id_key(term_id, block_id);
    let mut blk = match tx.get_fts_posting_bytes(&pkey)? {
        Some(bytes) => PackedPostingBlock::decode(&bytes)?,
        None => PackedPostingBlock::default(),
    };
    blk.upsert(slot, tf, doc_lens);
    let encoded = blk.encode();
    let hdr = PostingBlockHeader::parse(&encoded)?;
    tx.put_fts_posting_bytes(pkey, encoded);

    let skey = encode_fts_term_id_key(term_id, fts_skip_chunk_id(block_id));
    let mut skip_blk = match tx.get_fts_skip_bytes(&skey)? {
        Some(sb) => SkipBlock::decode(&sb)?,
        None => SkipBlock::default(),
    };
    skip_blk.upsert(fts_skip_slot(block_id), hdr.max_tf, hdr.min_dl_over_tf);
    tx.put_fts_skip_bytes(skey, skip_blk.encode());
    Ok(())
}

fn upsert_position_block(
    tx: &mut TxContext<'_>,
    term_id: TermId,
    block_id: FtsBlockId,
    slot: u8,
    positions: Vec<u32>,
) -> Result<()> {
    let pkey = encode_fts_term_id_key(term_id, block_id);
    let mut pblk = match tx.get_fts_position_bytes(&pkey)? {
        Some(bytes) => PackedPositionBlock::decode(&bytes)?,
        None => PackedPositionBlock::default(),
    };
    pblk.upsert(slot, positions);
    tx.put_fts_position_bytes(pkey, pblk.encode());
    Ok(())
}
