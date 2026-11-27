//! Sorted key/value SSTs, count aggregation, and FTS stream emission.

use std::collections::HashMap;
use std::path::Path;

use izzydb::codec::{
    encode_fts_term_key, fts_block_key_coord, fts_chunk_key_coord,
};
use izzydb::config::{
    AttrId, ROW_BLOCK_SIZE, RowBlock, SkipChunk, TermId,
};
use izzydb::error::{Result, TreeError};
use izzydb::fts::{
    FtsChunk, INLINE_TERM_MAX_DF, PackedPositionBlock, PackedPostingBlock, TermRecord, TermStats,
    dl_over_tf_lower_bound,
};
use izzydb::storage::{
    cf_options_for, default_sst_block_cache,
};

use crate::common::sorter::{
    ExternalSorter, KvItem,
};
use crate::common::sst::{SstFileInfo, SstSink};

use super::*;

pub(super) fn emit_sorted_kv_sst(
    bundle_dir: &Path,
    cf_name: &str,
    target_sst_bytes: u64,
    sorter: ExternalSorter<KvItem>,
) -> Result<Vec<SstFileInfo>> {
    let block_cache = default_sst_block_cache();
    let opts = cf_options_for(cf_name, &block_cache)?;
    let mut sink = SstSink::new(
        bundle_dir.to_path_buf(),
        cf_name,
        &opts,
        target_sst_bytes,
        format!("{cf_name}/part-00"),
    );
    let mut iter = sorter.into_sorted_iter()?;
    while let Some(item) = iter.next_item()? {
        sink.put(&item.key, &item.val)?;
    }
    sink.finish()
}

pub(super) fn collect_summed_counts(sorter: ExternalSorter<CountDeltaItem>) -> Result<Vec<(Vec<u8>, u64)>> {
    let mut iter = sorter.into_sorted_iter()?;
    let mut out = Vec::new();
    let mut cur_key: Option<Vec<u8>> = None;
    let mut cur_sum: u64 = 0;
    while let Some(item) = iter.next_item()? {
        match &cur_key {
            Some(k) if k == &item.key => cur_sum += item.count,
            _ => {
                if let Some(prev) = cur_key.take() {
                    out.push((prev, cur_sum));
                }
                cur_key = Some(item.key);
                cur_sum = item.count;
            }
        }
    }
    if let Some(prev) = cur_key.take() {
        out.push((prev, cur_sum));
    }
    Ok(out)
}

pub(super) type KvEmitFn<'a> = &'a mut dyn FnMut(&[u8], &[u8]) -> Result<()>;
pub(super) type TermDeltaFn<'a> = &'a mut dyn FnMut(FtsTermAppendDelta) -> Result<()>;

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_fts_streams_inner(
    iter: &mut crate::common::sorter::SortedIter<FtsPostingItem>,
    attr_lookup: &HashMap<AttrId, bool>,
    existing_terms: &HashMap<(AttrId, String), TermRecord>,
    next_term_id_init: u32,
    is_append_mode: bool,
    emit_new_term: KvEmitFn<'_>,
    emit_updated_term: TermDeltaFn<'_>,
    emit_chunk: KvEmitFn<'_>,
    emit_position: KvEmitFn<'_>,
) -> Result<u32> {
    let mut next_term_id = next_term_id_init;

    let mut has_term = false;
    let mut cur_attr: AttrId = 0;
    let mut cur_token = String::new();
    let mut cur_term_id: TermId = 0;
    let mut cur_is_existing = false;
    let mut cur_index_pos = false;
    let mut cur_stats = TermStats::default();

    let mut cur_rblk: Option<RowBlock> = None;
    let mut blocks_flushed_for_term: u32 = 0;
    let mut block_tfs: Vec<(u8, u32)> = Vec::new();
    let mut block_dls = [0u32; ROW_BLOCK_SIZE];
    let mut block_pos: Vec<(u8, Vec<u32>)> = Vec::new();

    let mut deferred_first_block: Option<(RowBlock, Vec<u8>)> = None;
    let mut cur_skip_chunk: Option<SkipChunk> = None;
    let mut cur_chunk_blocks: Vec<(u8, u32, f32, Vec<u8>)> = Vec::new();

    let flush_chunk = |term_id: TermId,
                       chunk_opt: &mut Option<SkipChunk>,
                       blocks: &mut Vec<(u8, u32, f32, Vec<u8>)>,
                       emit_chunk: KvEmitFn<'_>|
     -> Result<()> {
        if let Some(chunk) = chunk_opt.take()
            && !blocks.is_empty()
        {
            let refs: Vec<(u8, u32, f32, &[u8])> = blocks
                .iter()
                .map(|(r, m, mr, b)| (*r, *m, *mr, b.as_slice()))
                .collect();
            let encoded = FtsChunk::encode_from_blocks(&refs);
            let skey = fts_chunk_key_coord(term_id, chunk);
            emit_chunk(&skey, &encoded)?;
            blocks.clear();
        }
        Ok(())
    };

    #[allow(clippy::too_many_arguments)]
    let flush_block = |term_id: TermId,
                       is_existing_term: bool,
                       index_pos: bool,
                       rblk_opt: &mut Option<RowBlock>,
                       blocks_flushed: &mut u32,
                       block_tfs: &mut Vec<(u8, u32)>,
                       block_dls: &mut [u32; ROW_BLOCK_SIZE],
                       block_pos: &mut Vec<(u8, Vec<u32>)>,
                       deferred: &mut Option<(RowBlock, Vec<u8>)>,
                       cur_skip_chunk: &mut Option<SkipChunk>,
                       cur_chunk_blocks: &mut Vec<(u8, u32, f32, Vec<u8>)>,
                       total_df: u64,
                       is_last_block: bool,
                       emit_chunk: KvEmitFn<'_>,
                       emit_position: KvEmitFn<'_>|
     -> Result<()> {
        let Some(rblk) = rblk_opt.take() else {
            return Ok(());
        };
        let is_first_block = *blocks_flushed == 0;
        *blocks_flushed += 1;

        if index_pos {
            let pos_slices: Vec<(u8, &[u32])> = block_pos
                .iter()
                .map(|(s, p)| (*s, p.as_slice()))
                .collect();
            let encoded_pos = PackedPositionBlock::encode_from_sorted_slices(&pos_slices);
            emit_position(&fts_block_key_coord(term_id, rblk), &encoded_pos)?;
        }

        if !is_existing_term
            && is_first_block
            && is_last_block
            && total_df <= INLINE_TERM_MAX_DF as u64
        {
            let pblk = PackedPostingBlock::from_sorted_with_dl(block_tfs, |slot| {
                block_dls[slot as usize]
            });
            let encoded_v3 = pblk.encode();
            block_tfs.clear();
            block_dls.fill(0);
            block_pos.clear();
            *deferred = Some((rblk, encoded_v3));
            return Ok(());
        }

        let (max_tf, min_dl_over_tf, compact_bytes) =
            PackedPostingBlock::encode_compact_from_sorted(block_tfs, |slot| {
                block_dls[slot as usize]
            });

        block_tfs.clear();
        block_dls.fill(0);
        block_pos.clear();

        let chunk = rblk.chunk();
        if *cur_skip_chunk != Some(chunk) {
            flush_chunk(term_id, cur_skip_chunk, cur_chunk_blocks, emit_chunk)?;
            *cur_skip_chunk = Some(chunk);
        }
        cur_chunk_blocks.push((rblk.rel_block(), max_tf, min_dl_over_tf, compact_bytes));
        Ok(())
    };

    let finish_term = |attr_id: AttrId,
                       token: &str,
                       term_id: TermId,
                       is_existing: bool,
                       stats: TermStats,
                       deferred: &mut Option<(RowBlock, Vec<u8>)>,
                       emit_new_term: KvEmitFn<'_>,
                       emit_updated_term: TermDeltaFn<'_>|
     -> Result<()> {
        let tk = encode_fts_term_key(attr_id, token);
        if is_existing {
            emit_updated_term(FtsTermAppendDelta {
                term_key: tk,
                add_doc_freq: stats.doc_freq,
                max_tf: stats.max_tf,
                min_dl_over_tf: stats.min_dl_over_tf,
            })?;
        } else {
            let rec = match deferred.take() {
                Some((rblk, bytes)) => TermRecord::with_inline(term_id, stats, rblk, bytes),
                None => TermRecord::out_of_line(term_id, stats),
            };
            emit_new_term(&tk, &rec.encode())?;
        }
        Ok(())
    };

    let _ = is_append_mode;
    while let Some(item) = iter.next_item()? {
        if has_term && (item.attr_id != cur_attr || item.token != cur_token) {
            flush_block(
                cur_term_id,
                cur_is_existing,
                cur_index_pos,
                &mut cur_rblk,
                &mut blocks_flushed_for_term,
                &mut block_tfs,
                &mut block_dls,
                &mut block_pos,
                &mut deferred_first_block,
                &mut cur_skip_chunk,
                &mut cur_chunk_blocks,
                cur_stats.doc_freq,
                true,
                emit_chunk,
                emit_position,
            )?;
            flush_chunk(
                cur_term_id,
                &mut cur_skip_chunk,
                &mut cur_chunk_blocks,
                emit_chunk,
            )?;
            finish_term(
                cur_attr,
                &cur_token,
                cur_term_id,
                cur_is_existing,
                cur_stats,
                &mut deferred_first_block,
                emit_new_term,
                emit_updated_term,
            )?;
            has_term = false;
        }

        if !has_term {
            has_term = true;
            cur_attr = item.attr_id;
            cur_token = item.token;
            cur_index_pos = *attr_lookup.get(&cur_attr).unwrap();
            if let Some(existing) = existing_terms.get(&(cur_attr, cur_token.clone())) {
                cur_term_id = existing.term_id;
                cur_is_existing = true;
            } else {
                cur_term_id = next_term_id;
                next_term_id = next_term_id
                    .checked_add(1)
                    .ok_or_else(|| TreeError::Storage("TermId u32 space exhausted".into()))?;
                cur_is_existing = false;
            }
            cur_stats = TermStats::default();
            cur_rblk = None;
            blocks_flushed_for_term = 0;
            deferred_first_block = None;
            cur_skip_chunk = None;
            cur_chunk_blocks.clear();
        }

        let rblk = RowBlock::new(item.seg, item.blk);
        if cur_rblk != Some(rblk) && cur_rblk.is_some() {
            flush_block(
                cur_term_id,
                cur_is_existing,
                cur_index_pos,
                &mut cur_rblk,
                &mut blocks_flushed_for_term,
                &mut block_tfs,
                &mut block_dls,
                &mut block_pos,
                &mut deferred_first_block,
                &mut cur_skip_chunk,
                &mut cur_chunk_blocks,
                cur_stats.doc_freq,
                false,
                emit_chunk,
                emit_position,
            )?;
        }
        cur_rblk = Some(rblk);

        cur_stats.doc_freq += 1;
        cur_stats.max_tf = cur_stats.max_tf.max(item.tf);
        cur_stats.min_dl_over_tf = cur_stats
            .min_dl_over_tf
            .min(dl_over_tf_lower_bound(item.dl, item.tf));
        block_tfs.push((item.slot, item.tf));
        block_dls[item.slot as usize] = item.dl;
        if cur_index_pos {
            block_pos.push((item.slot, item.positions));
        }
    }

    if has_term {
        flush_block(
            cur_term_id,
            cur_is_existing,
            cur_index_pos,
            &mut cur_rblk,
            &mut blocks_flushed_for_term,
            &mut block_tfs,
            &mut block_dls,
            &mut block_pos,
            &mut deferred_first_block,
            &mut cur_skip_chunk,
            &mut cur_chunk_blocks,
            cur_stats.doc_freq,
            true,
            emit_chunk,
            emit_position,
        )?;
        flush_chunk(
            cur_term_id,
            &mut cur_skip_chunk,
            &mut cur_chunk_blocks,
            emit_chunk,
        )?;
        finish_term(
            cur_attr,
            &cur_token,
            cur_term_id,
            cur_is_existing,
            cur_stats,
            &mut deferred_first_block,
            emit_new_term,
            emit_updated_term,
        )?;
    }

    Ok(next_term_id)
}
