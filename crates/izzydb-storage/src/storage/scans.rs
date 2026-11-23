//! Whole-CF scans and visitors.


use rocksdb::{
    IteratorMode, PrefixRange, ReadOptions,
};

use izzydb_core::config::{
    DocId, NodeId, RowId,
};
use izzydb_core::error::{Result, TreeError};

use crate::codec::INTERNAL_NODE_TAG;
use crate::node::{PackedInternalNode, PackedLeafNode, decode_f32_slice};


use super::*;

impl Storage {
    pub fn all_internal_nodes(&self) -> Result<Vec<(NodeId, PackedInternalNode)>> {
        self.scan_default(INTERNAL_NODE_TAG, decode_u32_key, |v| {
            PackedInternalNode::decode(v, self.dim)
        })
    }

    pub fn all_leaf_nodes(&self) -> Result<Vec<(NodeId, PackedLeafNode)>> {
        self.scan_cf(CF_LEAF_ENTRIES, decode_u32_key, |v| {
            PackedLeafNode::decode(v, self.dim, self.padded_dim, self.rabitq_words)
        })
    }

    /// Iterate `cf_leaf_entries` in ascending `NodeId` order starting at `start_leaf_id`.
    /// When `visit` returns `Ok(false)`, stops immediately at `leaf_id` (if `stop_after` was not
    /// requested) or after `leaf_id`: passing `(bool_should_continue, bool_advance_past_current)`.
    pub fn visit_leaf_nodes_from(
        &self,
        start_leaf_id: NodeId,
        mut visit: impl FnMut(NodeId, &[u8]) -> Result<LeafVisitAction>,
    ) -> Result<Option<NodeId>> {
        let cf = self.cf(CF_LEAF_ENTRIES)?;
        let start_key = node_key(start_leaf_id);
        let mut iter = self
            .db
            .iterator_cf(cf, IteratorMode::From(&start_key, rocksdb::Direction::Forward));
        while let Some(item) = iter.next() {
            let (k, v) = item.map_err(map_db_err)?;
            let leaf_id = decode_u32_key(&k)?;
            match visit(leaf_id, &v)? {
                LeafVisitAction::Continue => {}
                LeafVisitAction::StopBefore => return Ok(Some(leaf_id)),
                LeafVisitAction::StopAfter => {
                    return match iter.next() {
                        Some(next_item) => {
                            let (nk, _) = next_item.map_err(map_db_err)?;
                            Ok(Some(decode_u32_key(&nk)?))
                        }
                        None => Ok(None),
                    };
                }
            }
        }
        Ok(None)
    }

    pub fn all_vectors(&self) -> Result<Vec<(RowId, Vec<f32>)>> {
        self.scan_cf(
            CF_VECTORS,
            |k| {
                let arr: [u8; 8] = k.try_into().map_err(|_| {
                    TreeError::CorruptedData(format!(
                        "cf_vectors key expected 8 bytes, got {}",
                        k.len()
                    ))
                })?;
                Ok(RowId::from_be_bytes(arr))
            },
            |v| decode_f32_slice(v, self.dim),
        )
    }

    pub fn all_id_map(&self) -> Result<Vec<(DocId, RowId)>> {
        self.scan_cf(CF_ID_MAP, decode_u64_key, |v| {
            let arr: [u8; 8] = v.try_into().map_err(|_| {
                TreeError::CorruptedData("cf_id_map value expected 8 bytes".into())
            })?;
            Ok(RowId::from_be_bytes(arr))
        })
    }

    pub fn visit_cf_raw(
        &self,
        cf_name: &str,
        mut visit: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        let cf = self.cf(cf_name)?;
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            visit(&k, &v)?;
        }
        Ok(())
    }

    pub(super) fn scan_cf<K, V>(
        &self,
        cf_name: &str,
        mut decode_key: impl FnMut(&[u8]) -> Result<K>,
        mut decode_val: impl FnMut(&[u8]) -> Result<V>,
    ) -> Result<Vec<(K, V)>> {
        let cf = self.cf(cf_name)?;
        let mut out = Vec::new();
        for item in self.db.iterator_cf(cf, IteratorMode::Start) {
            let (k, v) = item.map_err(map_db_err)?;
            out.push((decode_key(&k)?, decode_val(&v)?));
        }
        Ok(out)
    }

    pub fn visit_default(
        &self,
        prefix: &[u8],
        mut visit: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        let mut options = ReadOptions::default();
        options.set_iterate_range(PrefixRange(prefix));
        let mut iter = self.db.raw_iterator_opt(options);
        iter.seek(prefix);
        while let Some((key, value)) = iter.item() {
            visit(&key[prefix.len()..], value)?;
            iter.next();
        }
        iter.status().map_err(map_db_err)
    }

    pub(super) fn scan_default<K, V>(
        &self,
        tag: u8,
        mut decode_key: impl FnMut(&[u8]) -> Result<K>,
        mut decode_val: impl FnMut(&[u8]) -> Result<V>,
    ) -> Result<Vec<(K, V)>> {
        let mut out = Vec::new();
        self.visit_default(&[tag], |key, value| {
            out.push((decode_key(key)?, decode_val(value)?));
            Ok(())
        })?;
        Ok(out)
    }
}
