//! FTS overlay keyed by `RowBlock` / `SkipChunk`.


use rocksdb::{Direction, IteratorMode, ReadOptions};

use izzydb_core::config::{
    AttrId, RowBlock,
    SkipChunk, TermId,
};
use izzydb_core::error::Result;

use crate::codec::{
    encode_fts_term_id_prefix, fts_chunk_key_coord,
};

use crate::storage::types::map_db_err;
use crate::storage::CF_FTS_SKIPS;

use super::*;

impl <'a> TxContext<'a> {
    // -----------------------------------------------------------------------
    // FTS overlay (keyed by RowBlock / SkipChunk)
    // -----------------------------------------------------------------------

    pub fn get_fts_meta_bytes(&self) -> Result<Option<Vec<u8>>> {
        if let Some(ref bytes) = self.fts_meta_bytes {
            return Ok(Some(bytes.clone()));
        }
        self.storage.get_fts_meta_bytes()
    }

    pub fn put_fts_meta_bytes(&mut self, bytes: Vec<u8>) {
        self.fts_meta_bytes = Some(bytes);
        self.fts_meta_dirty = true;
    }

    pub fn get_fts_term_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_terms.get(key) {
            return Ok(opt.clone());
        }
        self.storage.get_fts_term_bytes(key)
    }

    pub fn put_fts_term_bytes(&mut self, key: Vec<u8>, val: Vec<u8>) {
        self.dirty_fts_terms.insert(key, Some(val));
    }

    pub fn delete_fts_term(&mut self, key: Vec<u8>) {
        self.dirty_fts_terms.insert(key, None);
    }

    pub fn get_fts_skip_bytes(&self, term_id: TermId, chunk: SkipChunk) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_skips.get(&(term_id, chunk)) {
            return Ok(opt.clone());
        }
        self.storage
            .get_fts_skip_bytes(&fts_chunk_key_coord(term_id, chunk))
    }

    pub fn put_fts_skip_bytes(&mut self, term_id: TermId, chunk: SkipChunk, val: Vec<u8>) {
        self.dirty_fts_skips.insert((term_id, chunk), Some(val));
    }

    pub fn delete_fts_skip(&mut self, term_id: TermId, chunk: SkipChunk) {
        self.dirty_fts_skips.insert((term_id, chunk), None);
    }

    pub fn get_fts_posting_bytes(
        &self,
        term_id: TermId,
        block: RowBlock,
    ) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_postings.get(&(term_id, block)) {
            return Ok(opt.clone());
        }
        self.storage
            .get_fts_posting_bytes(&crate::codec::fts_block_key_coord(term_id, block))
    }

    pub fn put_fts_posting_bytes(&mut self, term_id: TermId, block: RowBlock, val: Vec<u8>) {
        self.dirty_fts_postings.insert((term_id, block), Some(val));
    }

    pub fn delete_fts_posting(&mut self, term_id: TermId, block: RowBlock) {
        self.dirty_fts_postings.insert((term_id, block), None);
    }

    pub fn get_fts_position_bytes(
        &self,
        term_id: TermId,
        block: RowBlock,
    ) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_positions.get(&(term_id, block)) {
            return Ok(opt.clone());
        }
        self.storage
            .get_fts_position_bytes(&crate::codec::fts_block_key_coord(term_id, block))
    }

    pub fn put_fts_position_bytes(&mut self, term_id: TermId, block: RowBlock, val: Vec<u8>) {
        self.dirty_fts_positions.insert((term_id, block), Some(val));
    }

    pub fn delete_fts_position(&mut self, term_id: TermId, block: RowBlock) {
        self.dirty_fts_positions.insert((term_id, block), None);
    }

    pub fn get_fts_doc_len_bytes(&self, attr_id: AttrId, block: RowBlock) -> Result<Option<Vec<u8>>> {
        if let Some(opt) = self.dirty_fts_doc_lens.get(&(attr_id, block)) {
            return Ok(opt.clone());
        }
        self.storage
            .get_fts_doc_len_bytes(&crate::codec::doc_len_key_coord(attr_id, block))
    }

    pub fn put_fts_doc_len_bytes(&mut self, attr_id: AttrId, block: RowBlock, val: Vec<u8>) {
        self.dirty_fts_doc_lens.insert((attr_id, block), Some(val));
    }

    pub fn delete_fts_doc_len(&mut self, attr_id: AttrId, block: RowBlock) {
        self.dirty_fts_doc_lens.insert((attr_id, block), None);
    }

    pub fn visit_fts_skip_blocks(
        &self,
        term_id: TermId,
        mut f: impl FnMut(SkipChunk, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.storage.cf(CF_FTS_SKIPS)?;
        let prefix = encode_fts_term_id_prefix(term_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self
            .storage
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&prefix, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if k.len() != 12 || k[0..4] != prefix {
                break;
            }
            let seg = u32::from_be_bytes(k[4..8].try_into().unwrap());
            let cid = u32::from_be_bytes(k[8..12].try_into().unwrap());
            let chunk = SkipChunk::new(seg, cid);
            if !self.dirty_fts_skips.contains_key(&(term_id, chunk)) && !f(chunk, &v)? {
                return Ok(());
            }
        }
        for (&(tid, chunk), opt_bytes) in &self.dirty_fts_skips {
            if tid == term_id
                && let Some(bytes) = opt_bytes
                && !f(chunk, bytes)?
            {
                return Ok(());
            }
        }
        Ok(())
    }
}
