//! External-sort item types and staged-document helpers.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};


use izzydb::codec::{
    Manifest, SegmentMeta, ZoneEntry,
};
use izzydb::config::{
    AttrId, RowBlock,
    SegmentId,
};
use izzydb::fts::AttrFtsStats;
use izzydb::node::AnnMeta;

use crate::common::parquet_reader::MappedFile;
use crate::common::sorter::{
    SortItem, read_bytes_lp, read_bytes_req, read_u16_opt, read_u32_req,
    write_bytes_lp, write_u16, write_u32, write_u64,
};
use crate::common::sst::SstFileInfo;


pub(super) struct CountDeltaItem {
    pub(super) key: Vec<u8>,
    pub(super) count: u64,
}

impl SortItem for CountDeltaItem {
    #[inline]
    fn cmp_key(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.key.cmp(&b.key)
    }

    #[inline]
    fn estimated_bytes(&self) -> usize {
        self.key.len() + 24
    }

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        write_bytes_lp(w, &self.key)?;
        write_u64(w, self.count)
    }

    fn read_from(r: &mut impl Read) -> io::Result<Option<Self>> {
        let Some(key) = read_bytes_lp(r)? else {
            return Ok(None);
        };
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        Ok(Some(Self {
            key,
            count: u64::from_le_bytes(b),
        }))
    }
}

pub(super) struct FtsPostingItem {
    pub(super) attr_id: AttrId,
    pub(super) token: String,
    pub(super) seg: SegmentId,
    pub(super) blk: u32,
    pub(super) slot: u8,
    pub(super) tf: u32,
    pub(super) dl: u32,
    pub(super) positions: Vec<u32>,
}

impl SortItem for FtsPostingItem {
    #[inline]
    fn cmp_key(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.attr_id
            .cmp(&b.attr_id)
            .then_with(|| a.token.as_bytes().cmp(b.token.as_bytes()))
            .then_with(|| a.seg.cmp(&b.seg))
            .then_with(|| a.blk.cmp(&b.blk))
            .then_with(|| a.slot.cmp(&b.slot))
    }

    #[inline]
    fn estimated_bytes(&self) -> usize {
        self.token.len() + self.positions.len() * 4 + 56
    }

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        write_u16(w, self.attr_id)?;
        write_bytes_lp(w, self.token.as_bytes())?;
        write_u32(w, self.seg)?;
        write_u32(w, self.blk)?;
        w.write_all(&[self.slot])?;
        write_u32(w, self.tf)?;
        write_u32(w, self.dl)?;
        write_u32(w, self.positions.len() as u32)?;
        for &p in &self.positions {
            write_u32(w, p)?;
        }
        Ok(())
    }

    fn read_from(r: &mut impl Read) -> io::Result<Option<Self>> {
        let Some(attr_id) = read_u16_opt(r)? else {
            return Ok(None);
        };
        let tok_bytes = read_bytes_req(r)?;
        let token = String::from_utf8(tok_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let seg = read_u32_req(r)?;
        let blk = read_u32_req(r)?;
        let mut s = [0u8; 1];
        r.read_exact(&mut s)?;
        let slot = s[0];
        let tf = read_u32_req(r)?;
        let dl = read_u32_req(r)?;
        let pos_len = read_u32_req(r)? as usize;
        let mut positions = Vec::with_capacity(pos_len);
        for _ in 0..pos_len {
            positions.push(read_u32_req(r)?);
        }
        Ok(Some(Self {
            attr_id,
            token,
            seg,
            blk,
            slot,
            tf,
            dl,
            positions,
        }))
    }
}

pub struct SegmentPassOutputs {
    pub cf_files: BTreeMap<String, Vec<SstFileInfo>>,
    pub seg_metas: Vec<(SegmentId, SegmentMeta)>,
    pub zones: Vec<(SegmentId, AttrId, u8, ZoneEntry)>,
    pub doc_len_blocks: Vec<(AttrId, RowBlock, Vec<u8>)>,
    pub attr_count_deltas: Vec<(Vec<u8>, u64)>,
    pub detached_count_deltas: Vec<(Vec<u8>, u64)>,
    pub fts_attr_stats: BTreeMap<AttrId, AttrFtsStats>,
    pub new_fts_terms: Vec<(Vec<u8>, Vec<u8>)>,
    pub updated_fts_terms: Vec<FtsTermAppendDelta>,
    pub next_fts_term_id: u32,
}

pub struct FtsTermAppendDelta {
    pub term_key: Vec<u8>,
    pub add_doc_freq: u64,
    pub max_tf: u32,
    pub min_dl_over_tf: f32,
}

#[derive(Default)]
pub(super) struct WorkerSegAccum {
    pub(super) id_map_ssts: Vec<SstFileInfo>,
    pub(super) rows_ssts: Vec<SstFileInfo>,
    pub(super) vecs_ssts: Vec<SstFileInfo>,
    pub(super) doc_ids_ssts: Vec<SstFileInfo>,
    pub(super) cols_ssts: Vec<SstFileInfo>,
    pub(super) seg_metas: Vec<(SegmentId, SegmentMeta)>,
    pub(super) zones: Vec<(SegmentId, AttrId, u8, ZoneEntry)>,
    pub(super) doc_len_blocks: Vec<(AttrId, RowBlock, Vec<u8>)>,
    pub(super) fts_stats: Vec<AttrFtsStats>,
}

#[inline]
pub(super) fn read_doc_bytes<'a>(docs_idx: &[u8], docs_shards: &'a [MappedFile], row: u32) -> &'a [u8] {
    let off = (row as usize) * crate::common::parquet_reader::DocIndexEntry::SIZE;
    let entry = crate::common::parquet_reader::DocIndexEntry::decode(
        &docs_idx[off..off + crate::common::parquet_reader::DocIndexEntry::SIZE],
    );
    if entry.len == 0 {
        return &[];
    }
    let shard = docs_shards[entry.shard as usize].as_bytes();
    &shard[entry.offset as usize..entry.offset as usize + entry.len as usize]
}

pub type EmptyImportEmitResult = (
    BTreeMap<String, Vec<SstFileInfo>>,
    Manifest,
    Option<AnnMeta>,
    u32,
    u64,
);
