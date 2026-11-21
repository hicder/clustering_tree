use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use roaring::RoaringBitmap;

use izzydb_core::config::{
    AttrId, DocId, FtsBlockId, FtsDocId, fts_block_id, fts_slot, make_fts_doc_id,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::filter::{Filter, FtsPredicateKind, TokenQuery};
use izzydb_core::rank::RankBy;
use izzydb_core::schema::{AttributeValue, Attributes, CollectionSchema};
use izzydb_fts::{
    AnalyzedDoc, Bm25Result, DEFAULT_MAX_PREFIX_EXPANSIONS, DocLenBlock, FtsAnalyzer,
    FtsBlockCursor, FtsDocLenCursor, FtsMeta, FtsPostingSource, PackedPositionBlock,
    PackedPostingBlock, PostingBlockHeader, PostingBlockView, ReusableBuffer, TermStats,
    dl_over_tf_lower_bound, execute_rank_by,
};
use izzydb_storage::codec::{
    decode_fts_term_key, encode_fts_doc_len_key, encode_fts_posting_key,
    encode_fts_posting_prefix, encode_fts_term_key,
};
use izzydb_storage::{RocksDocLenCursor, RocksPostingCursor, Storage, TxContext};

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

pub struct StorageCursor<'a>(RocksPostingCursor<'a>);

impl FtsBlockCursor for StorageCursor<'_> {
    #[inline]
    fn current_block_id(&self) -> Option<FtsBlockId> {
        self.0.current_block_id()
    }

    #[inline]
    fn current_block_bytes(&self) -> &[u8] {
        self.0.current_block_bytes()
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
    type Cursor<'a>
        = StorageCursor<'a>
    where
        Self: 'a;

    type DocLenCursor<'a>
        = StorageDocLenCursor
    where
        Self: 'a;

    fn get_fts_meta(&self) -> Result<Option<FtsMeta>> {
        match self.storage.get_fts_meta_bytes()? {
            Some(bytes) => Ok(Some(FtsMeta::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    fn multi_get_term_stats(
        &self,
        attr_id: AttrId,
        tokens: &[String],
    ) -> Result<Vec<Option<TermStats>>> {
        let keys: Vec<Vec<u8>> = tokens
            .iter()
            .map(|t| encode_fts_term_key(attr_id, t))
            .collect();
        let mut out = Vec::with_capacity(keys.len());
        self.storage
            .multi_get_fts_term_bytes_sorted(&keys, |opt| {
                match opt {
                    Some(bytes) => out.push(Some(TermStats::decode(bytes)?)),
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
    ) -> Result<Vec<(String, TermStats)>> {
        let prefix_key = encode_fts_term_key(attr_id, prefix);
        let scan_limit = limit.saturating_mul(16).max(2048);
        let rows = self
            .storage
            .scan_fts_term_prefix_bytes(&prefix_key, scan_limit)?;
        let mut matches = Vec::with_capacity(rows.len());
        for (k, v) in rows {
            let (_, token) = decode_fts_term_key(&k)?;
            let stats = TermStats::decode(&v)?;
            matches.push((token, stats));
        }
        // Ensure exact match is present even if prefix has > scan_limit terms.
        if !matches.iter().any(|(t, _)| t == prefix)
            && let Some(raw) = self.storage.get_fts_term_bytes(&prefix_key)?
        {
            matches.push((prefix.to_owned(), TermStats::decode(&raw)?));
        }
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
        let prefix = encode_fts_posting_prefix(attr_id, token);
        Ok(StorageCursor(self.storage.open_fts_posting_cursor(prefix)?))
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

            apply_attr_token_diff(
                tx,
                entry.id,
                block_id,
                slot,
                fts_cfg.index_positions,
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
                entry.id,
                block_id,
                slot,
                fts_cfg.index_positions,
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
            let (attr_id, token) = decode_fts_term_key(&key)?;
            let mut ts = TermStats::decode(&bytes)?;
            let prefix = encode_fts_posting_prefix(attr_id, &token);
            let mut cursor = storage.open_fts_posting_cursor(prefix)?;
            let mut doc_freq = 0u64;
            let mut max_tf = 0u32;
            let mut min_ratio = f32::INFINITY;
            while cursor.current_block_id().is_some() {
                let header = PostingBlockHeader::parse(cursor.current_block_bytes())?;
                doc_freq = doc_freq.saturating_add(header.count as u64);
                max_tf = max_tf.max(header.max_tf);
                min_ratio = min_ratio.min(header.min_dl_over_tf);
                cursor.next_block()?;
            }
            if doc_freq == 0 {
                tx.delete_fts_term(key);
                updated += 1;
            } else if ts.doc_freq != doc_freq
                || ts.max_tf != max_tf
                || ts.min_dl_over_tf.to_bits() != min_ratio.to_bits()
            {
                ts.doc_freq = doc_freq;
                ts.max_tf = max_tf;
                ts.min_dl_over_tf = min_ratio;
                tx.put_fts_term_bytes(key, ts.encode().to_vec());
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
            let expanded =
                source.scan_prefix_terms(entry.id, prefix, DEFAULT_MAX_PREFIX_EXPANSIONS)?;
            Some(expanded)
        } else {
            None
        };

        match kind {
            FtsPredicateKind::AnyToken => {
                let mut tokens: Vec<String> = if last_as_prefix {
                    qtokens[..qtokens.len() - 1]
                        .iter()
                        .map(|t| t.text.clone())
                        .collect()
                } else {
                    qtokens.iter().map(|t| t.text.clone()).collect()
                };
                if let Some(exp) = prefix_expansions {
                    for (t, _) in exp {
                        tokens.push(t);
                    }
                }
                tokens.sort_unstable();
                tokens.dedup();
                eval_any_token(storage, entry.id, &tokens)
            }
            FtsPredicateKind::AllTokens => {
                let exact_tokens: Vec<String> = if last_as_prefix {
                    qtokens[..qtokens.len() - 1]
                        .iter()
                        .map(|t| t.text.clone())
                        .collect()
                } else {
                    qtokens.iter().map(|t| t.text.clone()).collect()
                };
                let prefix_tokens: Vec<String> = prefix_expansions
                    .map(|exp| exp.into_iter().map(|(t, _)| t).collect())
                    .unwrap_or_default();
                if last_as_prefix && prefix_tokens.is_empty() {
                    return Ok(RoaringBitmap::new());
                }
                let blocks = eval_all_tokens_blocks(
                    &source,
                    storage,
                    entry.id,
                    &exact_tokens,
                    if last_as_prefix {
                        Some(&prefix_tokens)
                    } else {
                        None
                    },
                )?;
                let mut bm = RoaringBitmap::new();
                for (block_id, mask) in blocks {
                    insert_mask_into_bitmap(block_id, &mask, &mut bm);
                }
                Ok(bm)
            }
            FtsPredicateKind::TokenSequence => {
                if qtokens.len() == 1 {
                    let tokens: Vec<String> = if let Some(exp) = prefix_expansions {
                        exp.into_iter().map(|(t, _)| t).collect()
                    } else {
                        vec![qtokens[0].text.clone()]
                    };
                    return eval_any_token(storage, entry.id, &tokens);
                }

                let exact_tokens: Vec<String> = if last_as_prefix {
                    qtokens[..qtokens.len() - 1]
                        .iter()
                        .map(|t| t.text.clone())
                        .collect()
                } else {
                    qtokens.iter().map(|t| t.text.clone()).collect()
                };
                let prefix_tokens: Vec<String> = prefix_expansions
                    .map(|exp| exp.into_iter().map(|(t, _)| t).collect())
                    .unwrap_or_default();
                if last_as_prefix && prefix_tokens.is_empty() {
                    return Ok(RoaringBitmap::new());
                }

                let candidate_blocks = eval_all_tokens_blocks(
                    &source,
                    storage,
                    entry.id,
                    &exact_tokens,
                    if last_as_prefix {
                        Some(&prefix_tokens)
                    } else {
                        None
                    },
                )?;
                if candidate_blocks.is_empty() {
                    return Ok(RoaringBitmap::new());
                }

                let base_pos = qtokens[0].position;
                let rel_positions: Vec<u32> =
                    qtokens.iter().map(|t| t.position - base_pos).collect();
                let mut result = RoaringBitmap::new();

                let mut unique_exact: Vec<String> = exact_tokens.clone();
                unique_exact.sort_unstable();
                unique_exact.dedup();

                for (block_id, cand_mask) in candidate_blocks {
                    let mut exact_pos_blocks: HashMap<&str, PackedPositionBlock> =
                        HashMap::with_capacity(unique_exact.len());
                    let mut missing_pos = false;
                    for tok in &unique_exact {
                        let key = encode_fts_posting_key(entry.id, tok, block_id);
                        let Some(bytes) = storage.get_fts_position_bytes(&key)? else {
                            missing_pos = true;
                            break;
                        };
                        exact_pos_blocks.insert(tok.as_str(), PackedPositionBlock::decode(&bytes)?);
                    }
                    if missing_pos {
                        continue;
                    }

                    let mut prefix_pos_blocks: Vec<PackedPositionBlock> = Vec::new();
                    if last_as_prefix {
                        for tok in &prefix_tokens {
                            let key = encode_fts_posting_key(entry.id, tok, block_id);
                            if let Some(bytes) = storage.get_fts_position_bytes(&key)? {
                                prefix_pos_blocks.push(PackedPositionBlock::decode(&bytes)?);
                            }
                        }
                        if prefix_pos_blocks.is_empty() {
                            continue;
                        }
                    }

                    for (w, &bits) in cand_mask.iter().enumerate() {
                        let mut b = bits;
                        while b != 0 {
                            let bit = b.trailing_zeros() as usize;
                            b &= b - 1;
                            let slot = ((w << 6) | bit) as u8;

                            let first_tok = qtokens[0].text.as_str();
                            let Some(first_positions) =
                                slot_positions_in_block(&exact_pos_blocks[first_tok], slot)
                            else {
                                continue;
                            };

                            let mut slot_matched = false;
                            'start: for &p0 in first_positions {
                                // Check intermediate exact tokens
                                let exact_end = if last_as_prefix {
                                    qtokens.len() - 1
                                } else {
                                    qtokens.len()
                                };
                                for i in 1..exact_end {
                                    let target_pos = p0 + rel_positions[i];
                                    let tok = qtokens[i].text.as_str();
                                    let Some(pos_list) =
                                        slot_positions_in_block(&exact_pos_blocks[tok], slot)
                                    else {
                                        continue 'start;
                                    };
                                    if pos_list.binary_search(&target_pos).is_err() {
                                        continue 'start;
                                    }
                                }
                                if last_as_prefix {
                                    let target_pos = p0 + rel_positions[qtokens.len() - 1];
                                    let any_prefix_match = prefix_pos_blocks.iter().any(|pblk| {
                                        slot_positions_in_block(pblk, slot).is_some_and(|list| {
                                            list.binary_search(&target_pos).is_ok()
                                        })
                                    });
                                    if !any_prefix_match {
                                        continue 'start;
                                    }
                                }
                                slot_matched = true;
                                break;
                            }

                            if slot_matched {
                                result.insert(make_fts_doc_id(block_id, slot));
                            }
                        }
                    }
                }

                Ok(result)
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

fn slot_positions_in_block(block: &PackedPositionBlock, slot: u8) -> Option<&[u32]> {
    let idx = block.slots.binary_search(&slot).ok()?;
    Some(&block.positions[idx])
}

fn insert_mask_into_bitmap(block_id: FtsBlockId, mask: &[u64; 4], out: &mut RoaringBitmap) {
    for (w, &bits) in mask.iter().enumerate() {
        let mut b = bits;
        while b != 0 {
            let bit = b.trailing_zeros() as usize;
            b &= b - 1;
            let slot = ((w << 6) | bit) as u8;
            out.insert(make_fts_doc_id(block_id, slot));
        }
    }
}

fn eval_any_token(
    storage: &Storage,
    attr_id: AttrId,
    tokens: &[String],
) -> Result<RoaringBitmap> {
    let mut out = RoaringBitmap::new();
    for tok in tokens {
        let prefix = encode_fts_posting_prefix(attr_id, tok);
        let mut cursor = storage.open_fts_posting_cursor(prefix)?;
        while let Some(block_id) = cursor.current_block_id() {
            let view = PostingBlockView::parse(cursor.current_block_bytes())?;
            view.for_each_slot(|slot| {
                out.insert(make_fts_doc_id(block_id, slot));
            });
            cursor.next_block()?;
        }
    }
    Ok(out)
}

fn eval_all_tokens_blocks(
    source: &StorageFtsSource<'_>,
    storage: &Storage,
    attr_id: AttrId,
    exact_tokens: &[String],
    prefix_tokens: Option<&[String]>,
) -> Result<Vec<(FtsBlockId, [u64; 4])>> {
    if exact_tokens.is_empty() {
        let Some(ptoks) = prefix_tokens else {
            return Ok(Vec::new());
        };
        let mut by_block: BTreeMap<FtsBlockId, [u64; 4]> = BTreeMap::new();
        for tok in ptoks {
            let prefix = encode_fts_posting_prefix(attr_id, tok);
            let mut cursor = storage.open_fts_posting_cursor(prefix)?;
            while let Some(block_id) = cursor.current_block_id() {
                let view = PostingBlockView::parse(cursor.current_block_bytes())?;
                let mask = by_block.entry(block_id).or_insert([0u64; 4]);
                view.or_slots_into_mask(mask);
                cursor.next_block()?;
            }
        }
        return Ok(by_block.into_iter().collect());
    }

    let mut unique_exact: Vec<String> = exact_tokens.to_vec();
    unique_exact.sort_unstable();
    unique_exact.dedup();

    let stats_list = source.multi_get_term_stats(attr_id, &unique_exact)?;
    let mut ordered: Vec<(String, u64)> = Vec::with_capacity(unique_exact.len());
    for (tok, opt_st) in unique_exact.into_iter().zip(stats_list) {
        let Some(st) = opt_st else {
            return Ok(Vec::new());
        };
        if st.doc_freq == 0 {
            return Ok(Vec::new());
        }
        ordered.push((tok, st.doc_freq));
    }
    ordered.sort_unstable_by_key(|(_, df)| *df);

    let mut cursors: Vec<RocksPostingCursor<'_>> = Vec::with_capacity(ordered.len());
    for (tok, _) in ordered {
        let prefix = encode_fts_posting_prefix(attr_id, &tok);
        let cur = storage.open_fts_posting_cursor(prefix)?;
        if cur.current_block_id().is_none() {
            return Ok(Vec::new());
        }
        cursors.push(cur);
    }

    let mut prefix_cursors: Vec<RocksPostingCursor<'_>> = match prefix_tokens {
        Some(ptoks) => {
            let mut pcs = Vec::with_capacity(ptoks.len());
            for tok in ptoks {
                let prefix = encode_fts_posting_prefix(attr_id, tok);
                let cur = storage.open_fts_posting_cursor(prefix)?;
                if cur.current_block_id().is_some() {
                    pcs.push(cur);
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
        let Some(mut target) = cursors[0].current_block_id() else {
            break;
        };

        // Gallop / intersect block_ids across all exact cursors.
        loop {
            let mut all_match = true;
            for cur in &mut cursors {
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

        // Intersect intra-block slot bitmasks.
        let mut mask = [u64::MAX; 4];
        for cur in &cursors {
            let view = PostingBlockView::parse(cur.current_block_bytes())?;
            let mut cur_mask = [0u64; 4];
            view.or_slots_into_mask(&mut cur_mask);
            for w in 0..4 {
                mask[w] &= cur_mask[w];
            }
            if mask == [0u64; 4] {
                break;
            }
        }

        if mask != [0u64; 4] && prefix_tokens.is_some() {
            let mut prefix_mask = [0u64; 4];
            for pcur in &mut prefix_cursors {
                if let Some(bid) = pcur.current_block_id()
                    && bid < target
                {
                    pcur.seek_block(target)?;
                }
                if pcur.current_block_id() == Some(target) {
                    let view = PostingBlockView::parse(pcur.current_block_bytes())?;
                    view.or_slots_into_mask(&mut prefix_mask);
                }
            }
            for w in 0..4 {
                mask[w] &= prefix_mask[w];
            }
        }

        if mask != [0u64; 4] {
            results.push((target, mask));
        }

        for cur in &mut cursors {
            cur.next_block()?;
        }
    }

    Ok(results)
}

fn apply_attr_token_diff(
    tx: &mut TxContext<'_>,
    attr_id: AttrId,
    block_id: FtsBlockId,
    slot: u8,
    index_positions: bool,
    old_doc: Option<&AnalyzedDoc>,
    new_doc: Option<&AnalyzedDoc>,
) -> Result<()> {
    let old_dl = old_doc.map_or(0, |d| d.doc_len);
    let new_dl = new_doc.map_or(0, |d| d.doc_len);

    let dl_key = encode_fts_doc_len_key(attr_id, block_id);
    let mut doc_len_block = match tx.get_fts_doc_len_bytes(&dl_key)? {
        Some(bytes) => DocLenBlock::decode(&bytes)?,
        None => DocLenBlock::default(),
    };
    if old_dl != new_dl {
        if new_dl > 0 {
            doc_len_block.set(slot, new_dl);
        } else {
            doc_len_block.remove(slot);
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
            remove_token_posting(
                tx,
                attr_id,
                tok,
                block_id,
                slot,
                old_term.tf,
                old_dl,
                index_positions,
                &doc_len_block,
            )?;
        }
    }

    // 2. Additions and updates
    if let Some(new) = new_doc {
        for (tok, new_term) in &new.terms {
            if let Some(old_term) = old_doc.and_then(|o| o.terms.get(tok)) {
                let tf_or_dl_changed = old_term.tf != new_term.tf || old_dl != new_dl;
                let pos_changed = index_positions && old_term.positions != new_term.positions;
                if tf_or_dl_changed {
                    upsert_posting_block(
                        tx,
                        attr_id,
                        tok,
                        block_id,
                        slot,
                        new_term.tf,
                        &doc_len_block,
                    )?;
                    let term_key = encode_fts_term_key(attr_id, tok);
                    let mut ts = match tx.get_fts_term_bytes(&term_key)? {
                        Some(b) => TermStats::decode(&b)?,
                        None => TermStats {
                            doc_freq: 1,
                            ..TermStats::default()
                        },
                    };
                    let lowered_max_tf = new_term.tf < old_term.tf && old_term.tf == ts.max_tf;
                    let old_ratio = dl_over_tf_lower_bound(old_dl, old_term.tf);
                    let new_ratio = dl_over_tf_lower_bound(new_dl, new_term.tf);
                    let raised_min_ratio =
                        new_ratio > old_ratio && old_ratio <= ts.min_dl_over_tf;
                    ts.observe(new_term.tf, new_dl);
                    if lowered_max_tf || raised_min_ratio {
                        maybe_recompute_term_stats_in_tx(tx, attr_id, tok, &mut ts)?;
                    }
                    tx.put_fts_term_bytes(term_key, ts.encode().to_vec());
                }
                if pos_changed {
                    upsert_position_block(
                        tx,
                        attr_id,
                        tok,
                        block_id,
                        slot,
                        new_term.positions.clone(),
                    )?;
                }
            } else {
                upsert_posting_block(
                    tx,
                    attr_id,
                    tok,
                    block_id,
                    slot,
                    new_term.tf,
                    &doc_len_block,
                )?;
                if index_positions {
                    upsert_position_block(
                        tx,
                        attr_id,
                        tok,
                        block_id,
                        slot,
                        new_term.positions.clone(),
                    )?;
                }
                let term_key = encode_fts_term_key(attr_id, tok);
                let mut ts = match tx.get_fts_term_bytes(&term_key)? {
                    Some(b) => TermStats::decode(&b)?,
                    None => TermStats::default(),
                };
                ts.doc_freq = ts.doc_freq.saturating_add(1);
                ts.observe(new_term.tf, new_dl);
                tx.put_fts_term_bytes(term_key, ts.encode().to_vec());
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn remove_token_posting(
    tx: &mut TxContext<'_>,
    attr_id: AttrId,
    token: &str,
    block_id: FtsBlockId,
    slot: u8,
    old_tf: u32,
    old_dl: u32,
    index_positions: bool,
    doc_lens: &DocLenBlock,
) -> Result<()> {
    let pkey = encode_fts_posting_key(attr_id, token, block_id);
    if let Some(bytes) = tx.get_fts_posting_bytes(&pkey)? {
        let mut blk = PackedPostingBlock::decode(&bytes)?;
        if blk.remove(slot, doc_lens) {
            if blk.is_empty() {
                tx.delete_fts_posting(pkey.clone());
            } else {
                tx.put_fts_posting_bytes(pkey.clone(), blk.encode());
            }
        }
    }
    if index_positions && let Some(bytes) = tx.get_fts_position_bytes(&pkey)? {
        let mut pblk = PackedPositionBlock::decode(&bytes)?;
        if pblk.remove(slot) {
            if pblk.is_empty() {
                tx.delete_fts_position(pkey);
            } else {
                tx.put_fts_position_bytes(pkey, pblk.encode());
            }
        }
    }

    let term_key = encode_fts_term_key(attr_id, token);
    if let Some(bytes) = tx.get_fts_term_bytes(&term_key)? {
        let mut ts = TermStats::decode(&bytes)?;
        ts.doc_freq = ts.doc_freq.saturating_sub(1);
        if ts.doc_freq == 0 {
            tx.delete_fts_term(term_key);
        } else {
            let hit_max_tf = old_tf == ts.max_tf;
            let hit_min_ratio = dl_over_tf_lower_bound(old_dl, old_tf) <= ts.min_dl_over_tf;
            if hit_max_tf || hit_min_ratio {
                maybe_recompute_term_stats_in_tx(tx, attr_id, token, &mut ts)?;
            }
            tx.put_fts_term_bytes(term_key, ts.encode().to_vec());
        }
    }
    Ok(())
}

fn maybe_recompute_term_stats_in_tx(
    tx: &TxContext<'_>,
    attr_id: AttrId,
    token: &str,
    ts: &mut TermStats,
) -> Result<bool> {
    let prefix = encode_fts_posting_prefix(attr_id, token);
    let mut block_count = 0usize;
    let mut max_tf = 0u32;
    let mut min_ratio = f32::INFINITY;
    let mut aborted = false;

    tx.visit_fts_posting_blocks(&prefix, |_block_id, bytes| {
        block_count += 1;
        if block_count > INLINE_TERM_STATS_RECOMPUTE_MAX_BLOCKS {
            aborted = true;
            return Ok(false);
        }
        let header = PostingBlockHeader::parse(bytes)?;
        max_tf = max_tf.max(header.max_tf);
        min_ratio = min_ratio.min(header.min_dl_over_tf);
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

fn upsert_posting_block(
    tx: &mut TxContext<'_>,
    attr_id: AttrId,
    token: &str,
    block_id: FtsBlockId,
    slot: u8,
    tf: u32,
    doc_lens: &DocLenBlock,
) -> Result<()> {
    let pkey = encode_fts_posting_key(attr_id, token, block_id);
    let mut blk = match tx.get_fts_posting_bytes(&pkey)? {
        Some(bytes) => PackedPostingBlock::decode(&bytes)?,
        None => PackedPostingBlock::default(),
    };
    blk.upsert(slot, tf, doc_lens);
    tx.put_fts_posting_bytes(pkey, blk.encode());
    Ok(())
}

fn upsert_position_block(
    tx: &mut TxContext<'_>,
    attr_id: AttrId,
    token: &str,
    block_id: FtsBlockId,
    slot: u8,
    positions: Vec<u32>,
) -> Result<()> {
    let pkey = encode_fts_posting_key(attr_id, token, block_id);
    let mut pblk = match tx.get_fts_position_bytes(&pkey)? {
        Some(bytes) => PackedPositionBlock::decode(&bytes)?,
        None => PackedPositionBlock::default(),
    };
    pblk.upsert(slot, positions);
    tx.put_fts_position_bytes(pkey, pblk.encode());
    Ok(())
}
