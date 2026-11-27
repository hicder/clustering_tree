//! FTS term/skip/posting/position/doc-len reads and cursors.

use std::collections::HashMap;
use std::sync::Arc;

use rocksdb::{
    DBPinnableSlice, Direction,
    IteratorMode, ReadOptions,
};

use izzydb_core::config::{
    AttrId, RowBlock, SegmentId, SkipChunk, TermId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_fts::TermRecord;

use crate::codec::{
    FTS_DOC_LEN_TAG,
    decode_fts_block_key, decode_fts_chunk_key, encode_fts_term_id_prefix, fts_block_key_coord,
};


use super::*;

impl Storage {
    pub fn get_fts_meta_bytes(&self) -> Result<Option<Vec<u8>>> {
        self.db.get(FTS_META_KEY).map_err(map_db_err)
    }

    pub fn get_fts_term_bytes(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_TERMS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn multi_get_fts_term_bytes_sorted(
        &self,
        keys: &[Vec<u8>],
        mut on_item: impl FnMut(Option<&[u8]>) -> Result<()>,
    ) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let cf = self.cf(CF_FTS_TERMS)?;
        let sorted = keys.windows(2).all(|w| w[0] <= w[1]);
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted);
        for res in results {
            match res.map_err(map_db_err)? {
                Some(slice) => on_item(Some(&slice))?,
                None => on_item(None)?,
            }
        }
        Ok(())
    }

    pub fn scan_fts_term_prefix_bytes(
        &self,
        prefix: &[u8],
        scan_limit: usize,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let cf = self.cf(CF_FTS_TERMS)?;
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter =
            self.db
                .iterator_cf_opt(cf, ro, IteratorMode::From(prefix, Direction::Forward));
        let mut out = Vec::new();
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if !k.starts_with(prefix) {
                break;
            }
            out.push((k.to_vec(), v.to_vec()));
            if out.len() >= scan_limit {
                break;
            }
        }
        Ok(out)
    }

    pub fn get_fts_skip_bytes(&self, key: &[u8; 12]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_SKIPS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn get_fts_posting_bytes(&self, key: &[u8; 12]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn get_fts_posting_pinned(
        &self,
        term_id: TermId,
        block_id: RowBlock,
    ) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        let key = fts_block_key_coord(term_id, block_id);
        self.db.get_pinned_cf(cf, key).map_err(map_db_err)
    }

    pub fn all_fts_terms(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.scan_cf(CF_FTS_TERMS, |k| Ok(k.to_vec()), |v| Ok(v.to_vec()))
    }

    pub fn visit_fts_chunks_for_term(
        &self,
        term_id: TermId,
        mut visit: impl FnMut(SkipChunk, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_FTS_SKIPS)?;
        let prefix = encode_fts_term_id_prefix(term_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        if let Some(next_t) = term_id.checked_add(1) {
            ro.set_iterate_upper_bound(next_t.to_be_bytes());
        }
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&prefix, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if k.len() != 12 || k[0..4] != prefix {
                break;
            }
            let (tid, chunk) = decode_fts_chunk_key(&k)?;
            if tid != term_id {
                break;
            }
            visit(chunk, &v)?;
        }
        Ok(())
    }

    pub fn visit_fts_postings_for_term(
        &self,
        term_id: TermId,
        mut visit: impl FnMut(RowBlock, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_FTS_POSTINGS)?;
        let prefix = encode_fts_term_id_prefix(term_id);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&prefix, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if k.len() != 12 || k[0..4] != prefix {
                break;
            }
            let (tid, block) = decode_fts_block_key(&k)?;
            if tid != term_id {
                break;
            }
            visit(block, &v)?;
        }
        Ok(())
    }

    pub fn get_fts_position_bytes(&self, key: &[u8; 12]) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_FTS_POSITIONS)?;
        self.db.get_cf(cf, key).map_err(map_db_err)
    }

    pub fn multi_get_fts_positions_pinned(
        &self,
        keys: &[[u8; 12]],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_FTS_POSITIONS)?;
        let sorted = keys.windows(2).all(|w| w[0] <= w[1]);
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted);
        let mut out = Vec::with_capacity(keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn get_fts_doc_len_bytes(&self, key: &[u8; 10]) -> Result<Option<Vec<u8>>> {
        let key = tagged_key::<11>(FTS_DOC_LEN_TAG, key);
        self.db.get(key).map_err(map_db_err)
    }

    pub fn open_fts_skip_cursor(&self, record: TermRecord) -> Result<RocksSkipCursor<'_>> {
        self.open_fts_skip_cursor_in_range(record, 0, SegmentId::MAX)
    }

    pub fn open_fts_skip_cursor_in_range(
        &self,
        record: TermRecord,
        seg_start: SegmentId,
        seg_end: SegmentId,
    ) -> Result<RocksSkipCursor<'_>> {
        if let Some((bid, bytes)) = record.inline_block {
            return RocksSkipCursor::from_inline(
                &self.db,
                record.term_id,
                bid,
                bytes,
                seg_start,
                seg_end,
            );
        }
        RocksSkipCursor::open_out_of_line_range(&self.db, record.term_id, seg_start, seg_end)
    }

    pub fn open_fts_skip_cursors(&self, records: &[TermRecord]) -> Result<Vec<RocksSkipCursor<'_>>> {
        records
            .iter()
            .map(|r| self.open_fts_skip_cursor(r.clone()))
            .collect()
    }

    pub fn open_fts_skip_cursors_in_range(
        &self,
        records: &[TermRecord],
        seg_start: SegmentId,
        seg_end: SegmentId,
    ) -> Result<Vec<RocksSkipCursor<'_>>> {
        records
            .iter()
            .map(|r| self.open_fts_skip_cursor_in_range(r.clone(), seg_start, seg_end))
            .collect()
    }

    pub fn open_fts_doc_len_cursor(&self, attr_id: AttrId) -> Result<RocksDocLenCursor> {
        {
            let guard = self
                .fts_doc_lens_cache
                .read()
                .expect("fts_doc_lens_cache poisoned");
            if let Some(blocks) = guard.get(&attr_id) {
                return Ok(RocksDocLenCursor {
                    blocks: Arc::clone(blocks),
                });
            }
        }

        let mut write_guard = self
            .fts_doc_lens_cache
            .write()
            .expect("fts_doc_lens_cache poisoned");
        if let Some(blocks) = write_guard.get(&attr_id) {
            return Ok(RocksDocLenCursor {
                blocks: Arc::clone(blocks),
            });
        }

        let prefix = tagged_key::<3>(FTS_DOC_LEN_TAG, &attr_id.to_be_bytes());
        let mut blocks: HashMap<RowBlock, Box<[u8]>> = HashMap::new();
        self.visit_default(&prefix, |key, value| {
            if key.len() != 8 {
                return Err(TreeError::CorruptedData(format!(
                    "doc_len suffix expected 8 bytes, got {}",
                    key.len()
                )));
            }
            let seg = u32::from_be_bytes(key[0..4].try_into().unwrap());
            let blk = u32::from_be_bytes(key[4..8].try_into().unwrap());
            blocks.insert(RowBlock::new(seg, blk), value.into());
            Ok(())
        })?;
        let table = Arc::new(blocks);
        write_guard.insert(attr_id, Arc::clone(&table));
        Ok(RocksDocLenCursor { blocks: table })
    }
}
