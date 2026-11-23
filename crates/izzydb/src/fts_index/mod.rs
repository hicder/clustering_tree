//! Inverted full-text index on `RowId` / `RowBlock` coordinates with segment-aware
//! liveness masks and Vectorized Block-Max MAXSCORE BM25 execution.

mod hook;
mod predicates;
mod query;
mod source;

use predicates::*;
pub use source::*;

use std::collections::HashMap;
use std::sync::Mutex;


use izzydb_core::config::{AttrId, RowBlock};
use izzydb_core::error::Result;
use izzydb_core::rowset::Liveness;
use izzydb_fts::{
    DocLenBlockView, PackedPostingBlock, ReusableBuffer, TermRecord,
    dl_over_tf_lower_bound,
};
use izzydb_storage::codec::decode_fts_term_key;
use izzydb_storage::{
    RocksDocLenCursor, Storage, TxContext,
};


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
}
