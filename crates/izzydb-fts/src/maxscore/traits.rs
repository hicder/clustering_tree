//! Posting source and cursor traits plus the BM25 result type.



use izzydb_core::config::{
    AttrId, DocId, RowBlock, RowId, TermId,
};
use izzydb_core::error::Result;

use crate::block::{FtsMeta, TermRecord};


/// BM25 search hit returned by `Collection::query_bm25` / `Collection::query`.
#[derive(Debug, Clone, PartialEq)]
pub struct Bm25Result {
    pub id: DocId,
    pub score: f32,
}

/// Cursor over a single term's Block-Max skip summary (`cf_fts_skips` or inline `TermRecord`)
/// sorted by ascending `RowBlock`.
pub trait FtsSkipCursor {
    fn current_block_id(&self) -> Option<RowBlock>;
    fn current_block_bounds(&self) -> (u32, f32);
    fn inline_posting_bytes(&self) -> Option<&[u8]>;
    fn current_posting_bytes(&mut self) -> Result<Option<&[u8]>> {
        Ok(None)
    }
    fn next_block(&mut self) -> Result<()>;
    fn seek_block(&mut self, target: RowBlock) -> Result<()>;
}

/// Cursor over document-length blocks (`default[0x05]`) for a single `attr_id`.
pub trait FtsDocLenCursor {
    fn seek_block_bytes(&mut self, block_id: RowBlock) -> Result<Option<&[u8]>>;
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
        block_id: RowBlock,
    ) -> Result<Option<Self::PostingGuard<'a>>>;
    fn open_doc_len_cursor<'a>(&'a self, attr_id: AttrId) -> Result<Self::DocLenCursor<'a>>;
    fn multi_get_doc_ids(&self, row_ids: &[RowId]) -> Result<Vec<Option<DocId>>>;
}
