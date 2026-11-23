//! Point reads: manifest, schema, id map, rows, columns, vectors, doc ids, nodes.

use std::collections::BTreeMap;
use std::sync::Arc;

use rocksdb::{
    DBPinnableSlice,
    IteratorMode, PrefixRange, ReadOptions,
};

use izzydb_core::config::{
    AttrId, DocId, NodeId, ROW_BLOCK_SIZE, RowId, SegmentId,
};
use izzydb_core::error::{Result, TreeError};
use izzydb_core::schema::CollectionSchema;

use crate::codec::{
    INTERNAL_NODE_TAG, Manifest, SEG_DELETES_TAG, SEG_META_TAG,
    SegmentMeta, column_key, decode_doc_id_block, decode_roaring_bitmap,
    doc_id_block_key, row_key,
};
use crate::node::{AnnMeta, PackedInternalNode, PackedLeafNode, decode_f32_slice};


use super::*;

impl Storage {
    /// Take a point-in-time RocksDB snapshot and read `(Manifest, SegmentTableSnapshot, cf_id_map)`
    /// at that exact snapshot (used by `DetachedIngest::build`).
    #[allow(clippy::type_complexity)]
    pub fn snapshot_id_map_and_state(
        &self,
    ) -> Result<(Manifest, SegmentTableSnapshot, Vec<(DocId, RowId)>)> {
        let snap = self.db.snapshot();
        let mut ro_def = ReadOptions::default();
        ro_def.set_snapshot(&snap);

        let manifest_bytes = self
            .db
            .get_opt(MANIFEST_KEY, &ro_def)
            .map_err(map_db_err)?
            .ok_or_else(|| TreeError::CorruptedData("missing manifest in snapshot".into()))?;
        let manifest = Manifest::decode(&manifest_bytes)?;

        let mut entries: BTreeMap<SegmentId, SegmentEntry> = BTreeMap::new();
        {
            let mut ro = ReadOptions::default();
            ro.set_snapshot(&snap);
            ro.set_iterate_range(PrefixRange(&[SEG_META_TAG][..]));
            let mut it = self.db.raw_iterator_opt(ro);
            it.seek([SEG_META_TAG]);
            while let Some((k, v)) = it.item() {
                let seg = decode_u32_key(&k[1..])?;
                entries.insert(
                    seg,
                    SegmentEntry {
                        meta: SegmentMeta::decode(v)?,
                        deleted: None,
                    },
                );
                it.next();
            }
            it.status().map_err(map_db_err)?;
        }
        {
            let mut ro = ReadOptions::default();
            ro.set_snapshot(&snap);
            ro.set_iterate_range(PrefixRange(&[SEG_DELETES_TAG][..]));
            let mut it = self.db.raw_iterator_opt(ro);
            it.seek([SEG_DELETES_TAG]);
            while let Some((k, v)) = it.item() {
                let seg = decode_u32_key(&k[1..])?;
                let bm = decode_roaring_bitmap(v)?;
                if let Some(entry) = entries.get_mut(&seg) {
                    entry.deleted = (!bm.is_empty()).then(|| Arc::new(bm));
                }
                it.next();
            }
            it.status().map_err(map_db_err)?;
        }
        let seg_snap = SegmentTableSnapshot::from_entries(entries);

        let cf_id_map = self.cf(CF_ID_MAP)?;
        let mut ro_id = ReadOptions::default();
        ro_id.set_snapshot(&snap);
        let mut id_map = Vec::with_capacity(manifest.live_rows as usize);
        for item in self.db.iterator_cf_opt(cf_id_map, ro_id, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            let doc_id = decode_u64_key(&k)?;
            let r_arr: [u8; 8] = v.as_ref().try_into().map_err(|_| {
                TreeError::CorruptedData("cf_id_map value expected 8 bytes".into())
            })?;
            id_map.push((doc_id, RowId::from_be_bytes(r_arr)));
        }
        Ok((manifest, seg_snap, id_map))
    }

    pub fn get_manifest(&self) -> Result<Option<Manifest>> {
        match self.db.get_pinned(MANIFEST_KEY).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(Manifest::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn get_ann_meta(&self) -> Result<Option<AnnMeta>> {
        match self.db.get_pinned(ANN_META_KEY).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(AnnMeta::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn get_schema(&self) -> Result<Option<CollectionSchema>> {
        match self.db.get(SCHEMA_KEY).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(CollectionSchema::decode(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn put_schema(&self, schema: &CollectionSchema) -> Result<()> {
        let mut batch = self.new_batch();
        batch.put_schema(schema);
        self.write_batch(batch)
    }

    pub fn get_id_map(&self, doc_id: DocId) -> Result<Option<RowId>> {
        let cf = self.cf(CF_ID_MAP)?;
        match self
            .db
            .get_pinned_cf(cf, doc_key(doc_id))
            .map_err(map_db_err)?
        {
            Some(bytes) => {
                let arr: [u8; 8] = bytes.as_ref().try_into().map_err(|_| {
                    TreeError::CorruptedData(format!(
                        "cf_id_map expected 8 bytes, got {}",
                        bytes.len()
                    ))
                })?;
                Ok(Some(RowId::from_be_bytes(arr)))
            }
            None => Ok(None),
        }
    }

    pub fn multi_get_id_map(&self, doc_ids: &[DocId]) -> Result<Vec<Option<RowId>>> {
        if doc_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_ID_MAP)?;
        let keys: Vec<[u8; 8]> = doc_ids.iter().map(|&id| doc_key(id)).collect();
        let sorted = doc_ids.windows(2).all(|w| w[0] <= w[1]);
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted);
        let mut out = Vec::with_capacity(doc_ids.len());
        for res in results {
            match res.map_err(map_db_err)? {
                Some(slice) => {
                    let arr: [u8; 8] = slice.as_ref().try_into().map_err(|_| {
                        TreeError::CorruptedData(format!(
                            "cf_id_map expected 8 bytes, got {}",
                            slice.len()
                        ))
                    })?;
                    out.push(Some(RowId::from_be_bytes(arr)));
                }
                None => out.push(None),
            }
        }
        Ok(out)
    }

    pub fn get_row_bytes(&self, row: RowId) -> Result<Option<Vec<u8>>> {
        let cf = self.cf(CF_ROWS)?;
        self.db.get_cf(cf, row_key(row)).map_err(map_db_err)
    }

    pub fn get_row_bytes_pinned(&self, row: RowId) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_ROWS)?;
        self.db.get_pinned_cf(cf, row_key(row)).map_err(map_db_err)
    }

    pub fn multi_get_row_bytes_pinned(
        &self,
        sorted_rows: &[RowId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        self.batched_multi_get_row_pinned(CF_ROWS, sorted_rows, true)
    }

    pub fn get_column_block_pinned(
        &self,
        seg: SegmentId,
        attr_id: AttrId,
        block: u32,
    ) -> Result<Option<DBPinnableSlice<'_>>> {
        let cf = self.cf(CF_COLUMNS)?;
        self.db
            .get_pinned_cf(cf, column_key(seg, attr_id, block))
            .map_err(map_db_err)
    }

    /// Batch pinned `multi_get` against `cf_columns` for `(seg, attr_id, block)` keys.
    /// When `sorted_input` is true, `keys` must be in ascending `(seg, attr_id, block)` order.
    pub fn multi_get_column_blocks_pinned(
        &self,
        keys: &[(SegmentId, AttrId, u32)],
        sorted_input: bool,
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_COLUMNS)?;
        let raw_keys: Vec<[u8; 10]> = keys
            .iter()
            .map(|&(seg, attr_id, block)| column_key(seg, attr_id, block))
            .collect();
        let results = self.db.batched_multi_get_cf(cf, raw_keys.iter(), sorted_input);
        let mut out = Vec::with_capacity(keys.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn get_vector(&self, row: RowId) -> Result<Option<Vec<f32>>> {
        let cf = self.cf(CF_VECTORS)?;
        match self
            .db
            .get_pinned_cf(cf, row_key(row))
            .map_err(map_db_err)?
        {
            Some(slice) => Ok(Some(decode_f32_slice(&slice, self.dim)?)),
            None => Ok(None),
        }
    }

    pub fn multi_get_vectors_pinned(
        &self,
        sorted_rows: &[RowId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        self.batched_multi_get_row_pinned(CF_VECTORS, sorted_rows, true)
    }

    fn batched_multi_get_row_pinned(
        &self,
        cf_name: &str,
        rows: &[RowId],
        sorted_input: bool,
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(cf_name)?;
        let keys: Vec<[u8; 8]> = rows.iter().map(|&r| row_key(r)).collect();
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), sorted_input);
        let mut out = Vec::with_capacity(rows.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }

    pub fn get_doc_id_block(
        &self,
        seg: SegmentId,
        block: u32,
    ) -> Result<Option<Arc<[DocId; ROW_BLOCK_SIZE]>>> {
        if let Some(cached) = self
            .doc_id_block_cache
            .read()
            .expect("doc_id_block_cache poisoned")
            .get(&(seg, block))
        {
            return Ok(Some(cached));
        }
        let cf = self.cf(CF_ROW_DOC_IDS)?;
        let Some(bytes) = self
            .db
            .get_pinned_cf(cf, doc_id_block_key(seg, block))
            .map_err(map_db_err)?
        else {
            return Ok(None);
        };
        let decoded = decode_doc_id_block(&bytes)?;
        let arc: Arc<[DocId; ROW_BLOCK_SIZE]> = Arc::from(decoded);
        self.doc_id_block_cache
            .write()
            .expect("doc_id_block_cache poisoned")
            .insert((seg, block), Arc::clone(&arc));
        Ok(Some(arc))
    }

    /// Resolve a slice of [`RowId`]s to [`DocId`]s through the bounded `doc_id_block_cache`.
    pub fn resolve_doc_ids(&self, rows: &[RowId]) -> Result<Vec<Option<DocId>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(rows.len());
        let mut last_key: Option<(SegmentId, u32)> = None;
        let mut last_blk: Option<Arc<[DocId; ROW_BLOCK_SIZE]>> = None;
        for &row in rows {
            let key = (row.seg(), row.block_idx());
            if last_key != Some(key) {
                last_key = Some(key);
                last_blk = self.get_doc_id_block(key.0, key.1)?;
            }
            out.push(last_blk.as_ref().map(|blk| blk[row.slot() as usize]));
        }
        Ok(out)
    }

    pub fn get_internal_node(&self, node_id: NodeId) -> Result<Option<PackedInternalNode>> {
        let key = tagged_key::<5>(INTERNAL_NODE_TAG, &node_key(node_id));
        match self.db.get_pinned(key).map_err(map_db_err)? {
            Some(bytes) => Ok(Some(PackedInternalNode::decode(&bytes, self.dim)?)),
            None => Ok(None),
        }
    }

    pub fn get_leaf_node(&self, leaf_id: NodeId) -> Result<Option<PackedLeafNode>> {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        match self
            .db
            .get_pinned_cf(cf, node_key(leaf_id))
            .map_err(map_db_err)?
        {
            Some(bytes) => Ok(Some(PackedLeafNode::decode(
                &bytes,
                self.dim,
                self.padded_dim,
                self.rabitq_words,
            )?)),
            None => Ok(None),
        }
    }

    pub fn multi_get_leaf_nodes_pinned(
        &self,
        sorted_leaf_ids: &[NodeId],
    ) -> Result<Vec<Option<DBPinnableSlice<'_>>>> {
        if sorted_leaf_ids.is_empty() {
            return Ok(Vec::new());
        }
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        let keys: Vec<[u8; 4]> = sorted_leaf_ids.iter().map(|&id| node_key(id)).collect();
        let results = self.db.batched_multi_get_cf(cf, keys.iter(), true);
        let mut out = Vec::with_capacity(sorted_leaf_ids.len());
        for res in results {
            out.push(res.map_err(map_db_err)?);
        }
        Ok(out)
    }
}
