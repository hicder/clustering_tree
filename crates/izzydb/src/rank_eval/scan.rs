//! `RowSet` scan evaluation, SIMD leaf paths, and general evaluation.

use std::collections::HashMap;

use izzydb_core::config::{AttrId, DocId, ROW_BLOCK_SIZE, RowId, SegmentId, for_each_set_bit_256};
use izzydb_core::error::Result;
use izzydb_core::filter::Filter;
use izzydb_core::rank::{DecayMode, RankBy, eval_decay, eval_saturate};
use izzydb_core::rowset::RowSet;
use izzydb_core::schema::ColumnStorage;
use izzydb_core::simd::{batch_eval_decay, batch_eval_saturate};
use izzydb_storage::column::ColumnBlockView;

use crate::attr_reader::{AttrReader, ColumnReader, RowStoreReader};
use crate::collection::Collection;
use crate::filter_plan::{self};
use crate::fts_index::LivenessMaskProvider;
use crate::segments::RowResolver;


// ------------------------------------------------------------------------------------------
// Compiled per-row evaluation tree
// ------------------------------------------------------------------------------------------

use super::*;

impl Collection {
    pub(super) fn scan_rowset(
        &self,
        node: &Node,
        rowset: &RowSet,
        lex_maps: &[HashMap<DocId, f32>],
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        if rowset.is_empty() {
            return Ok(());
        }
        let col_reader = ColumnReader::new(&self.storage, &self.schema)?;
        let row_reader = RowStoreReader::new(&self.storage);
        let mut lex_scores = vec![0.0f32; lex_maps.len()];
        let mut batch: Vec<RowId> = Vec::with_capacity(SCAN_BATCH);

        let no_filter = !node.has_filter_node();
        let mut ref_attrs = Vec::new();
        if no_filter {
            node.collect_numeric_attrs(&mut ref_attrs);
        }

        for &(seg, ref bm) in rowset.segments() {
            let seg_columnar =
                no_filter && ref_attrs.iter().all(|&a| col_reader.segment_has_columns(seg, a));
            if seg_columnar
                && lex_maps.is_empty()
                && let Some(block_eval) = Self::as_single_block_simd_leaf(node)
            {
                self.scan_segment_simd_blocks(seg, bm, block_eval, sink)?;
                continue;
            }
            if seg_columnar && !ref_attrs.is_empty() {
                batch.clear();
                let mut col_bufs: Vec<Vec<Option<f64>>> = vec![Vec::new(); ref_attrs.len()];
                for local in bm.iter() {
                    batch.push(RowId::new(seg, local));
                    if batch.len() >= SCAN_BATCH {
                        self.eval_batch_columnar(
                            &col_reader,
                            node,
                            &ref_attrs,
                            &mut col_bufs,
                            &batch,
                            lex_maps,
                            &mut lex_scores,
                            sink,
                        )?;
                        batch.clear();
                    }
                }
                if !batch.is_empty() {
                    self.eval_batch_columnar(
                        &col_reader,
                        node,
                        &ref_attrs,
                        &mut col_bufs,
                        &batch,
                        lex_maps,
                        &mut lex_scores,
                        sink,
                    )?;
                }
                continue;
            }

            for local in bm.iter() {
                batch.push(RowId::new(seg, local));
                if batch.len() >= SCAN_BATCH {
                    self.eval_batch(&row_reader, node, &batch, lex_maps, &mut lex_scores, sink)?;
                    batch.clear();
                }
            }
            if !batch.is_empty() {
                self.eval_batch(&row_reader, node, &batch, lex_maps, &mut lex_scores, sink)?;
                batch.clear();
            }
        }
        Ok(())
    }

    pub(super) fn as_single_block_simd_leaf(node: &Node) -> Option<(AttrId, SimdLeafOp)> {
        match node {
            Node::Sat {
                id,
                midpoint,
                exponent,
                missing,
            } => Some((
                *id,
                SimdLeafOp::Saturate {
                    weight: 1.0,
                    midpoint: *midpoint,
                    exponent: *exponent,
                    missing: *missing,
                },
            )),
            Node::Decay {
                id,
                origin,
                midpoint,
                exponent,
                mode,
                missing,
            } => Some((
                *id,
                SimdLeafOp::Decay {
                    weight: 1.0,
                    origin: *origin,
                    midpoint: *midpoint,
                    exponent: *exponent,
                    mode: *mode,
                    missing: *missing,
                },
            )),
            Node::Product(ch) if ch.len() == 2 => match (&ch[0], &ch[1]) {
                (
                    Node::Const(w),
                    Node::Sat {
                        id,
                        midpoint,
                        exponent,
                        missing,
                    },
                ) => Some((
                    *id,
                    SimdLeafOp::Saturate {
                        weight: *w,
                        midpoint: *midpoint,
                        exponent: *exponent,
                        missing: *missing,
                    },
                )),
                (
                    Node::Const(w),
                    Node::Decay {
                        id,
                        origin,
                        midpoint,
                        exponent,
                        mode,
                        missing,
                    },
                ) => Some((
                    *id,
                    SimdLeafOp::Decay {
                        weight: *w,
                        origin: *origin,
                        midpoint: *midpoint,
                        exponent: *exponent,
                        mode: *mode,
                        missing: *missing,
                    },
                )),
                _ => None,
            },
            _ => None,
        }
    }

    pub(super) fn scan_segment_simd_blocks(
        &self,
        seg: SegmentId,
        bm: &roaring::RoaringBitmap,
        (attr_id, op): (AttrId, SimdLeafOp),
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        let mut by_block: Vec<(u32, [u64; 4])> = Vec::new();
        for local in bm.iter() {
            let blk = local >> 8;
            let slot = (local & 0xFF) as usize;
            if let Some(last) = by_block.last_mut()
                && last.0 == blk
            {
                last.1[slot >> 6] |= 1u64 << (slot & 63);
            } else {
                let mut m = [0u64; 4];
                m[slot >> 6] |= 1u64 << (slot & 63);
                by_block.push((blk, m));
            }
        }
        let slices = if self
            .schema
            .get_by_id(attr_id)
            .is_some_and(|e| e.storage == ColumnStorage::Detached)
        {
            let col_gen = crate::detached::current_detached_gen(&self.storage, attr_id)?;
            let det_keys: Vec<(AttrId, izzydb_core::config::ColumnGen, SegmentId, u32)> = by_block
                .iter()
                .map(|&(blk, _)| (attr_id, col_gen, seg, blk))
                .collect();
            self.storage
                .multi_get_detached_blocks_pinned(&det_keys, true)?
        } else {
            let keys: Vec<(SegmentId, AttrId, u32)> = by_block
                .iter()
                .map(|&(blk, _)| (seg, attr_id, blk))
                .collect();
            self.storage.multi_get_column_blocks_pinned(&keys, true)?
        };

        let mut f64_buf = [0.0f64; ROW_BLOCK_SIZE];
        let mut out_scores = [0.0f32; ROW_BLOCK_SIZE];
        let mut pos_rows: Vec<RowId> = Vec::with_capacity(ROW_BLOCK_SIZE);
        let mut pos_scores: Vec<f32> = Vec::with_capacity(ROW_BLOCK_SIZE);

        for ((blk, req_mask), opt_slice) in by_block.into_iter().zip(slices.iter()) {
            let present_mask = match opt_slice {
                Some(slice) => ColumnBlockView::parse(slice)?.decode_numeric_f64_block(&mut f64_buf),
                None => {
                    f64_buf.fill(0.0);
                    [0u64; 4]
                }
            };
            let missing_val = match op {
                SimdLeafOp::Saturate {
                    midpoint,
                    exponent,
                    missing,
                    ..
                } => {
                    batch_eval_saturate(
                        &f64_buf,
                        &present_mask,
                        midpoint,
                        exponent,
                        missing,
                        &mut out_scores,
                    );
                    missing
                }
                SimdLeafOp::Decay {
                    origin,
                    midpoint,
                    exponent,
                    mode,
                    missing,
                    ..
                } => {
                    batch_eval_decay(
                        &f64_buf,
                        &present_mask,
                        origin,
                        midpoint,
                        exponent,
                        mode,
                        missing,
                        &mut out_scores,
                    );
                    missing
                }
            };
            let _ = missing_val;
            let weight = match op {
                SimdLeafOp::Saturate { weight, .. } | SimdLeafOp::Decay { weight, .. } => weight,
            };
            pos_rows.clear();
            pos_scores.clear();
            for_each_set_bit_256(&req_mask, |slot| {
                let s = out_scores[slot as usize] * weight;
                if s > 0.0 && !s.is_nan() {
                    pos_rows.push(RowId::new(seg, (blk << 8) | (slot as u32)));
                    pos_scores.push(s.min(f32::MAX));
                }
            });
            if !pos_rows.is_empty() {
                let doc_ids = RowResolver::resolve_all(&self.storage, &pos_rows)?;
                for ((&r, doc_id), &s) in pos_rows.iter().zip(doc_ids).zip(&pos_scores) {
                    sink(r, doc_id, s);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_batch_columnar(
        &self,
        reader: &ColumnReader<'_>,
        node: &Node,
        attrs: &[AttrId],
        col_bufs: &mut [Vec<Option<f64>>],
        batch: &[RowId],
        lex_maps: &[HashMap<DocId, f32>],
        lex_scores: &mut [f32],
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        for (&attr_id, buf) in attrs.iter().zip(col_bufs.iter_mut()) {
            reader.read_numeric_batch(attr_id, batch, buf)?;
        }
        let doc_ids = RowResolver::resolve_all(&self.storage, batch)?;
        for (idx, (&row, doc_id)) in batch.iter().zip(doc_ids).enumerate() {
            for (s, m) in lex_scores.iter_mut().zip(lex_maps) {
                *s = m.get(&doc_id).copied().unwrap_or(0.0);
            }
            let score = node.eval_from_cols(idx, row, attrs, col_bufs, lex_scores);
            if score > 0.0 && !score.is_nan() {
                sink(row, doc_id, score.min(f32::MAX));
            }
        }
        Ok(())
    }

    pub(super) fn eval_batch(
        &self,
        reader: &dyn AttrReader,
        node: &Node,
        batch: &[RowId],
        lex_maps: &[HashMap<DocId, f32>],
        lex_scores: &mut [f32],
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        let doc_ids = RowResolver::resolve_all(&self.storage, batch)?;
        let mut idx = 0usize;
        reader.read_rows(batch, &mut |row, view| {
            let doc_id = doc_ids[idx];
            idx += 1;
            for (s, m) in lex_scores.iter_mut().zip(lex_maps) {
                *s = m.get(&doc_id).copied().unwrap_or(0.0);
            }
            let ctx = RowCtx::new(self, row, view);
            let score = node.eval(&ctx, lex_scores)?;
            if score > 0.0 && !score.is_nan() {
                sink(row, doc_id, score.min(f32::MAX));
            }
            Ok(())
        })
    }

    pub(super) fn general_eval(
        &self,
        norm: &RankBy,
        filter: Option<&Filter>,
        sink: &mut dyn FnMut(RowId, DocId, f32),
    ) -> Result<()> {
        let mut lex = Vec::new();
        let mut node = Node::compile(norm, &self.schema, &mut lex);
        self.resolve_filter_sets(&mut node)?;
        let lex_sets: Vec<RowSet> = lex
            .iter()
            .map(|e| {
                self.fts_index
                    .eval_rank_by_match_rowset(&self.storage, &self.schema, e)
            })
            .collect::<Result<_>>()?;

        let snap = self.storage.segment_snapshot();
        let cand_rs = match node.positive_universe(&lex_sets) {
            Some(mut rs) => {
                if rs.is_empty() {
                    return Ok(());
                }
                if let Some(f) = filter {
                    let f_rs = filter_plan::materialize_filter(
                        &self.storage,
                        &self.schema,
                        f,
                        &self.fts_index,
                    )?;
                    rs = rs.and(&f_rs);
                }
                rs
            }
            None => match filter {
                Some(f) => filter_plan::materialize_filter(
                    &self.storage,
                    &self.schema,
                    f,
                    &self.fts_index,
                )?,
                None => RowSet::from_all_live(&*snap),
            },
        };
        if cand_rs.is_empty() {
            return Ok(());
        }

        let mut lex_maps = Vec::with_capacity(lex.len());
        if !lex.is_empty() {
            let mask = LivenessMaskProvider {
                snap: &snap,
                filter: Some(&cand_rs),
            };
            let k = (cand_rs.len() as usize).max(1);
            for expr in &lex {
                let hits = self.fts_index.query_bm25(
                    &self.storage,
                    &self.schema,
                    expr,
                    k,
                    Some(&mask),
                    None,
                )?;
                lex_maps.push(hits.into_iter().map(|h| (h.id, h.score)).collect());
            }
        }
        self.scan_rowset(&node, &cand_rs, &lex_maps, sink)
    }
}

pub(super) fn eval_value(node: &Node, leaf: &Node, v: f64) -> f32 {
    let leaf_score = match leaf {
        Node::Attr { .. } => clamp_f32(v),
        Node::Sat {
            midpoint, exponent, ..
        } => eval_saturate(v, *midpoint, *exponent),
        Node::Decay {
            origin,
            midpoint,
            exponent,
            mode,
            ..
        } => eval_decay(v, *origin, *midpoint, *exponent, *mode),
        _ => 0.0,
    };
    match node {
        Node::Product(c) => {
            let mut acc = 1.0f32;
            for n in c {
                acc *= match n {
                    Node::Const(w) => *w,
                    _ => leaf_score,
                };
            }
            acc
        }
        _ => leaf_score,
    }
}

#[derive(Clone, Copy)]
pub(super) enum SimdLeafOp {
    Saturate {
        weight: f32,
        midpoint: f64,
        exponent: f64,
        missing: f32,
    },
    Decay {
        weight: f32,
        origin: f64,
        midpoint: f64,
        exponent: f64,
        mode: DecayMode,
        missing: f32,
    },
}
