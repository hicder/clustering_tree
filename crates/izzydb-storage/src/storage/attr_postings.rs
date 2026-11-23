//! Attribute postings, counts, and zone reads.

use std::collections::HashMap;
use std::sync::Arc;

use rocksdb::{
    DBPinnableSlice, Direction,
    IteratorMode, PrefixRange, ReadOptions,
};

use izzydb_core::config::{
    AttrId, RowId, SegmentId,
};
use izzydb_core::error::{Result, TreeError};

use crate::codec::{
    ATTR_COUNT_TAG, SegPosting, ZONE_TAG, ZoneEntry, decode_row_key, encode_attr_prefix, row_key, split_posting_key,
};


use super::*;

impl Storage {
    pub fn get_attr_posting(&self, key: &[u8]) -> Result<Option<SegPosting>> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        match self.db.get_pinned_cf(cf, key).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(SegPosting::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn multi_get_attr_posting_slices(
        &self,
        sorted_keys: &[Vec<u8>],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if sorted_keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let results = self.db.batched_multi_get_cf(cf, sorted_keys.iter(), true);
        let mut out = Vec::with_capacity(sorted_keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn visit_attr_postings_for_value(
        &self,
        attr_key: &[u8],
        mut visit: impl FnMut(SegmentId, &[u8]) -> Result<()>,
    ) -> Result<()> {
        debug_assert!(attr_key.len() >= 3);
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(attr_key, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            let (prefix, seg) = split_posting_key(&k)?;
            if prefix != attr_key {
                break;
            }
            visit(seg, &v)?;
        }
        Ok(())
    }

    pub fn visit_attr_postings_range(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        start_val: &[u8],
        mut visit: impl FnMut(&[u8], SegmentId, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let prefix = encode_attr_prefix(attr_id, type_tag);
        let mut seek_key = Vec::with_capacity(prefix.len() + start_val.len());
        seek_key.extend_from_slice(&prefix);
        seek_key.extend_from_slice(start_val);
        let mut ro = ReadOptions::default();
        ro.set_prefix_same_as_start(true);
        let iter = self
            .db
            .iterator_cf_opt(cf, ro, IteratorMode::From(&seek_key, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            let (val_prefix, seg) = split_posting_key(&k)?;
            if !visit(&val_prefix[3..], seg, &v)? {
                break;
            }
        }
        Ok(())
    }

    pub fn visit_segment_rows(
        &self,
        seg: SegmentId,
        mut visit: impl FnMut(RowId, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(CF_ROWS)?;
        let start = row_key(RowId::new(seg, 0));
        let iter = self
            .db
            .iterator_cf(cf, IteratorMode::From(&start, Direction::Forward));
        for item in iter {
            let (k, v) = item.map_err(map_db_err)?;
            let row = decode_row_key(&k)?;
            if row.seg() != seg {
                break;
            }
            visit(row, &v)?;
        }
        Ok(())
    }

    /// Visit all `(val_bytes, seg, posting_bytes)` for `(attr_id, type_tag)` in `cf_attr_postings`,
    /// ordered forward (`reverse == false`) or reverse (`reverse == true`) by `(val_bytes, seg)`.
    pub fn visit_attr_postings_prefix(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        reverse: bool,
        mut visit: impl FnMut(&[u8], SegmentId, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let cf = self.cf(CF_ATTR_POSTINGS)?;
        let prefix = encode_attr_prefix(attr_id, type_tag);
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
            let (val_prefix, seg) = split_posting_key(k)?;
            if !visit(&val_prefix[3..], seg, v)? {
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

    /// Largest encoded value in `default[0x06]` with live count `> 0` for `(attr_id, type_tag)`.
    pub fn max_attr_count_key(&self, attr_id: AttrId, type_tag: u8) -> Result<Option<Vec<u8>>> {
        let a = attr_id.to_be_bytes();
        let prefix = [ATTR_COUNT_TAG, a[0], a[1], type_tag];
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

    pub fn get_zones_cached(&self) -> Result<ZoneTable> {
        {
            let guard = self.zone_cache.read().expect("zone_cache poisoned");
            if let Some(ref table) = *guard {
                return Ok(Arc::clone(table));
            }
        }
        let mut write_guard = self.zone_cache.write().expect("zone_cache poisoned");
        if let Some(ref table) = *write_guard {
            return Ok(Arc::clone(table));
        }

        let mut map: HashMap<(AttrId, u8), HashMap<SegmentId, ZoneEntry>> = HashMap::new();
        self.visit_default(&[ZONE_TAG], |k, v| {
            if k.len() != 7 {
                return Err(TreeError::CorruptedData(format!(
                    "zone key expected 7 bytes after tag, got {}",
                    k.len()
                )));
            }
            let seg = u32::from_be_bytes(k[0..4].try_into().unwrap());
            let attr_id = u16::from_be_bytes([k[4], k[5]]);
            let tag = k[6];
            map.entry((attr_id, tag))
                .or_default()
                .insert(seg, ZoneEntry::decode(v)?);
            Ok(())
        })?;
        let table = Arc::new(map);
        *write_guard = Some(Arc::clone(&table));
        Ok(table)
    }

    pub fn get_attr_count(&self, attr_key: &[u8]) -> Result<u64> {
        let mut key = Vec::with_capacity(1 + attr_key.len());
        key.push(ATTR_COUNT_TAG);
        key.extend_from_slice(attr_key);
        match self.db.get_pinned(key).map_err(map_db_err)? {
            Some(bytes) => decode_count(&bytes),
            None => Ok(0),
        }
    }

    pub fn visit_attr_counts(
        &self,
        attr_id: AttrId,
        type_tag: u8,
        start_val: &[u8],
        mut visit: impl FnMut(&[u8], u64) -> Result<bool>,
    ) -> Result<()> {
        let a = attr_id.to_be_bytes();
        let prefix = [ATTR_COUNT_TAG, a[0], a[1], type_tag];
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(&prefix[..]));
        let mut iter = self.db.raw_iterator_opt(options);
        let mut seek_key = Vec::with_capacity(prefix.len() + start_val.len());
        seek_key.extend_from_slice(&prefix);
        seek_key.extend_from_slice(start_val);
        iter.seek(&seek_key);
        while let Some((key, value)) = iter.item() {
            if !visit(&key[prefix.len()..], decode_count(value)?)? {
                break;
            }
            iter.next();
        }
        iter.status().map_err(map_db_err)
    }

    pub fn has_attr_counts_marker(&self) -> Result<bool> {
        Ok(self
            .db
            .get_pinned(ATTR_COUNTS_VERSION_KEY)
            .map_err(map_db_err)?
            .is_some())
    }
}
