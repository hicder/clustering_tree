//! Detached column storage reads/writes.


use roaring::RoaringBitmap;
use rocksdb::{
    DBPinnableSlice, Direction,
    IteratorMode, PrefixRange, ReadOptions, SstFileWriter,
};

use izzydb_core::config::{
    AttrId, ColumnGen, SegmentId,
};
use izzydb_core::error::{Result, TreeError};

use crate::codec::{
    DETACHED_COUNT_TAG, DETACHED_META_TAG, DETACHED_OVERWRITTEN_TAG, DetachedMeta, decode_roaring_bitmap, detached_count_suffix,
    detached_key, detached_meta_key, detached_overwritten_key, detached_posting_prefix,
    split_detached_posting_key,
};


use super::*;

impl Storage {
    pub fn get_detached_meta(&self, attr_id: AttrId) -> Result<Option<DetachedMeta>> {
        match self.db.get_pinned(detached_meta_key(attr_id)).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(DetachedMeta::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn all_detached_metas(&self) -> Result<Vec<(AttrId, DetachedMeta)>> {
        self.scan_default(
            DETACHED_META_TAG,
            |k| {
                let arr: [u8; 2] = k.try_into().map_err(|_| {
                    TreeError::CorruptedData("DetachedMeta key expected 2 bytes".into())
                })?;
                Ok(AttrId::from_be_bytes(arr))
            },
            DetachedMeta::decode,
        )
    }

    pub fn put_detached_meta_sync(&self, attr_id: AttrId, meta: &DetachedMeta) -> Result<()> {
        let mut batch = self.new_batch();
        batch.put_detached_meta(attr_id, meta);
        self.write_batch_sync(batch)
    }

    pub fn get_detached_block_pinned(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        seg: SegmentId,
        block: u32,
    ) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_DETACHED)?;
        self.db
            .get_pinned_cf(cf, detached_key(attr_id, col_gen, seg, block))
            .map_err(map_db_err)
    }

    pub fn multi_get_detached_blocks_pinned(
        &self,
        keys: &[(AttrId, ColumnGen, SegmentId, u32)],
        sorted_input: bool,
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_DETACHED)?;
        let raw_keys: Vec<[u8; 14]> = keys
            .iter()
            .map(|&(attr, g, seg, block)| detached_key(attr, g, seg, block))
            .collect();
        let results = self.db.batched_multi_get_cf(cf, raw_keys.iter(), sorted_input);
        let mut out = Vec::with_capacity(keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn get_detached_overwritten(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        seg: SegmentId,
    ) -> Result<Option<RoaringBitmap>> {
        match self
            .db
            .get_pinned(detached_overwritten_key(attr_id, col_gen, seg))
            .map_err(map_db_err)?
        {
            Some(bytes) => Ok(Some(decode_roaring_bitmap(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn all_detached_overwritten(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
    ) -> Result<Vec<(SegmentId, RoaringBitmap)>> {
        let a = attr_id.to_be_bytes();
        let g = col_gen.to_be_bytes();
        let prefix = [
            DETACHED_OVERWRITTEN_TAG,
            a[0],
            a[1],
            g[0],
            g[1],
            g[2],
            g[3],
        ];
        let mut out = Vec::new();
        self.visit_default(&prefix, |k, v| {
            let arr: [u8; 4] = k.try_into().map_err(|_| {
                TreeError::CorruptedData("detached overwritten seg suffix expected 4 bytes".into())
            })?;
            out.push((SegmentId::from_be_bytes(arr), decode_roaring_bitmap(v)?));
            Ok(())
        })?;
        Ok(out)
    }

    pub fn get_detached_count(&self, col_gen: ColumnGen, attr_key: &[u8]) -> Result<u64> {
        let suffix = detached_count_suffix(col_gen, attr_key);
        let mut key = Vec::with_capacity(1 + suffix.len());
        key.push(DETACHED_COUNT_TAG);
        key.extend_from_slice(&suffix);
        match self.db.get_pinned(key).map_err(map_db_err)? {
            Some(bytes) => decode_count(&bytes),
            None => Ok(0),
        }
    }

    pub fn visit_detached_counts(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
        start_val: &[u8],
        mut visit: impl FnMut(&[u8], u64) -> Result<bool>,
    ) -> Result<()> {
        let p = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut prefix = [0u8; 8];
        prefix[0] = DETACHED_COUNT_TAG;
        prefix[1..8].copy_from_slice(&p);
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        options.set_total_order_seek(true);
        let mut iter = self.db.raw_iterator_opt(options);
        let mut seek_key = Vec::with_capacity(prefix.len() + start_val.len());
        seek_key.extend_from_slice(&prefix);
        seek_key.extend_from_slice(start_val);
        iter.seek(&seek_key);
        while let Some((key, value)) = iter.item() {
            if !key.starts_with(&prefix) {
                break;
            }
            if !visit(&key[prefix.len()..], decode_count(value)?)? {
                break;
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)
    }

    pub fn max_detached_count_key(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
    ) -> Result<Option<Vec<u8>>> {
        let p = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut prefix = [0u8; 8];
        prefix[0] = DETACHED_COUNT_TAG;
        prefix[1..8].copy_from_slice(&p);
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        options.set_total_order_seek(true);
        let mut iter = self.db.raw_iterator_opt(options);
        iter.seek_to_last();
        while let Some((key, value)) = iter.item() {
            if !key.starts_with(&prefix) {
                break;
            }
            if decode_count(value)? > 0 {
                return Ok(Some(key[prefix.len()..].to_vec()));
            }
            iter.prev();
        }
        iter.status().map_err(map_db_err)?;
        Ok(None)
    }

    pub fn multi_get_detached_posting_slices(
        &self,
        keys: &[Vec<u8>],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_DETACHED_POSTINGS)?;
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), true);
        let mut out = Vec::with_capacity(keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn visit_detached_postings_for_value(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
        encoded_val: &[u8],
        mut visit: impl FnMut(SegmentId, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_DETACHED_POSTINGS)?;
        let p = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut key_prefix = Vec::with_capacity(7 + encoded_val.len());
        key_prefix.extend_from_slice(&p);
        key_prefix.extend_from_slice(encoded_val);
        let mut ro = ReadOptions::default();
        ro.set_iterate_range(PrefixRange(&key_prefix[..]));
        ro.set_total_order_seek(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&key_prefix, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            let (prefix, seg) = split_detached_posting_key(&k)?;
            if prefix != key_prefix.as_slice() {
                break;
            }
            visit(seg, &v)?;
        }
        Ok(())
    }

    pub fn visit_detached_postings_range(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
        start_val: &[u8],
        mut visit: impl FnMut(&[u8], SegmentId, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.cf(CF_DETACHED_POSTINGS)?;
        let prefix = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut seek_key = Vec::with_capacity(7 + start_val.len());
        seek_key.extend_from_slice(&prefix);
        seek_key.extend_from_slice(start_val);
        let mut ro = ReadOptions::default();
        ro.set_iterate_range(PrefixRange(&prefix[..]));
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&seek_key, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            let (val_prefix, seg) = split_detached_posting_key(&k)?;
            if !visit(&val_prefix[7..], seg, &v)? {
                break;
            }
        }
        Ok(())
    }

    pub fn visit_detached_postings_prefix(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        type_tag: u8,
        reverse: bool,
        mut visit: impl FnMut(&[u8], SegmentId, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.cf(CF_DETACHED_POSTINGS)?;
        let prefix = detached_posting_prefix(attr_id, col_gen, type_tag);
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        options.set_total_order_seek(true);
        let mut iter = self.db.raw_iterator_cf_opt(cf, options);
        if reverse {
            iter.seek_to_last();
        } else {
            iter.seek(prefix);
        }
        while let Some((k, v)) = iter.item() {
            if !k.starts_with(&prefix) {
                break;
            }
            let (val_prefix, seg) = split_detached_posting_key(k)?;
            if !visit(&val_prefix[7..], seg, v)? {
                break;
            }
            if reverse {
                iter.prev();
            } else {
                iter.next();
            }
        }
        iter.status().map_err(map_db_err)
    }

    pub fn visit_detached_blocks_for_gen(
        &self,
        attr_id: AttrId,
        col_gen: ColumnGen,
        mut visit: impl FnMut(SegmentId, u32, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_DETACHED)?;
        let a = attr_id.to_be_bytes();
        let g = col_gen.to_be_bytes();
        let prefix = [a[0], a[1], g[0], g[1], g[2], g[3]];
        let mut ro = ReadOptions::default();
        ro.set_iterate_range(PrefixRange(&prefix[..]));
        ro.set_total_order_seek(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&prefix, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if !k.starts_with(&prefix) || k.len() != 14 {
                continue;
            }
            let seg = u32::from_be_bytes(k[6..10].try_into().unwrap());
            let block = u32::from_be_bytes(k[10..14].try_into().unwrap());
            visit(seg, block, &v)?;
        }
        Ok(())
    }

    pub fn ingest_detached_sst(&self, sorted_entries: &[([u8; 14], Vec<u8>)]) -> Result<()> {
        if sorted_entries.is_empty() {
            return Ok(());
        }
        let path = self.db.path().join(format!(
            ".detached_stage_{}_{}.sst",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let opts = self.cf_options(CF_DETACHED)?;
        let mut writer = SstFileWriter::create(&opts);
        let res = (|| -> Result<()> {
            writer.open(&path).map_err(map_db_err)?;
            for (k, v) in sorted_entries {
                writer.put(k, v).map_err(map_db_err)?;
            }
            writer.finish().map_err(map_db_err)?;
            self.ingest_cf(CF_DETACHED, std::slice::from_ref(&path))?;
            Ok(())
        })();
        let _ = std::fs::remove_file(&path);
        res
    }

    pub fn write_detached_stage(
        &self,
        counts: &[(Vec<u8>, u64)],
        postings: &[(Vec<u8>, Vec<u8>)],
        blocks: Option<&[([u8; 14], Vec<u8>)]>,
    ) -> Result<()> {
        let mut batch = self.new_batch();
        if let Some(blks) = blocks {
            for (k, v) in blks {
                batch.put_detached_block(k, v);
            }
        }
        for (suffix, cnt) in counts {
            if *cnt > 0 {
                batch.put_detached_count(suffix, *cnt);
            }
        }
        for (k, v) in postings {
            batch.put_detached_posting(k, v);
        }
        self.write_batch(batch)
    }
}
