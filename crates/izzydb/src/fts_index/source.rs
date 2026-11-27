//! Storage-backed FTS posting source and liveness mask provider.

use izzydb_core::config::{AttrId, DocId, RowBlock, RowId, SegmentId, SkipChunk, TermId};
use izzydb_core::error::Result;
use izzydb_core::rowset::{Liveness, RowSet};
use izzydb_fts::{
    BlockMaskProvider, FtsDocLenCursor, FtsMeta, FtsPostingSource, FtsSkipCursor,
    PostingBlockView, TermRecord,
};
use izzydb_storage::codec::{
    decode_fts_term_key, encode_fts_term_key,
};
use izzydb_storage::{
    DBPinnableSlice, RocksDocLenCursor, RocksSkipCursor, SegmentTableSnapshot, Storage,
};

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
    fn current_chunk_bounds(&self) -> Option<(SkipChunk, u32, f32)> {
        self.0.current_chunk_bounds()
    }

    #[inline]
    fn inline_posting_bytes(&self) -> Option<&[u8]> {
        self.0.inline_posting_bytes()
    }

    #[inline]
    fn current_posting_view(&mut self) -> Result<Option<PostingBlockView<'_>>> {
        self.0.current_posting_view()
    }

    #[inline]
    fn next_block(&mut self) -> Result<()> {
        self.0.next_block()
    }

    #[inline]
    fn seek_block(&mut self, target: RowBlock) -> Result<()> {
        self.0.seek_block(target)
    }

    #[inline]
    fn seek_past_chunk(&mut self, chunk: SkipChunk) -> Result<()> {
        self.0.seek_past_chunk(chunk)
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

    fn segment_ranges(&self, max_shards: usize) -> Option<Vec<(SegmentId, SegmentId)>> {
        if max_shards <= 1 {
            return None;
        }
        let snap = self.storage.segment_snapshot();
        let visible: Vec<SegmentId> = snap
            .segments
            .iter()
            .filter_map(|(&seg, entry)| entry.is_visible().then_some(seg))
            .collect();
        if visible.len() < 4 {
            return None;
        }
        let num_shards = max_shards.min(visible.len() / 2);
        if num_shards < 2 {
            return None;
        }
        let chunk_size = visible.len().div_ceil(num_shards);
        let chunks: Vec<&[SegmentId]> = visible.chunks(chunk_size).collect();
        let mut ranges = Vec::with_capacity(chunks.len());
        for (i, ch) in chunks.iter().enumerate() {
            let start = ch[0];
            let end = if i + 1 < chunks.len() {
                chunks[i + 1][0]
            } else {
                ch.last().copied().unwrap().saturating_add(1)
            };
            ranges.push((start, end));
        }
        Some(ranges)
    }

    fn open_skip_cursors_in_range<'a>(
        &'a self,
        records: &[TermRecord],
        seg_start: SegmentId,
        seg_end: SegmentId,
    ) -> Result<Vec<Self::SkipCursor<'a>>> {
        Ok(self
            .storage
            .open_fts_skip_cursors_in_range(records, seg_start, seg_end)?
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
