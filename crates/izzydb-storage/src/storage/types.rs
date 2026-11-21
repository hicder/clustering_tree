use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use izzydb_core::config::{AttrId, DocId, FTS_BLOCK_SIZE, FtsDocId, LocalId, NodeId};
use izzydb_core::error::{Result, TreeError};

const FTS_REV_BLOCK_HEADER_LEN: usize = 32;
const FTS_REV_BLOCK_LEN: usize =
    FTS_REV_BLOCK_HEADER_LEN + FTS_BLOCK_SIZE * (8 + 4 + 4);

#[inline]
pub(super) fn map_db_err(err: rocksdb::Error) -> TreeError {
    TreeError::Storage(err.to_string())
}

#[inline]
pub(crate) fn node_key(id: NodeId) -> [u8; 4] {
    id.to_be_bytes()
}

#[inline]
pub(crate) fn doc_key(id: DocId) -> [u8; 8] {
    id.to_be_bytes()
}

#[inline]
pub(super) fn decode_u32_key(bytes: &[u8]) -> Result<u32> {
    if bytes.len() != 4 {
        return Err(TreeError::CorruptedData(format!(
            "expected 4-byte key, got {}",
            bytes.len()
        )));
    }
    Ok(u32::from_be_bytes(bytes.try_into().unwrap()))
}

#[inline]
pub(super) fn decode_u64_key(bytes: &[u8]) -> Result<u64> {
    if bytes.len() != 8 {
        return Err(TreeError::CorruptedData(format!(
            "expected 8-byte key, got {}",
            bytes.len()
        )));
    }
    Ok(u64::from_be_bytes(bytes.try_into().unwrap()))
}

/// Physical location of a document in the clustering tree (plus optional `FtsDocId`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocLoc {
    pub leaf_id: NodeId,
    pub local_id: LocalId,
    pub fts_doc_id: Option<FtsDocId>,
}

impl DocLoc {
    #[inline]
    pub fn encode(self) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&self.leaf_id.to_be_bytes());
        out[4..8].copy_from_slice(&self.local_id.to_be_bytes());
        out[8..12].copy_from_slice(&self.fts_doc_id.unwrap_or(FtsDocId::MAX).to_be_bytes());
        out
    }

    #[inline]
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 12 {
            return Err(TreeError::CorruptedData(format!(
                "DocLoc expected 12 bytes, got {}",
                bytes.len()
            )));
        }
        let raw_fts = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
        Ok(Self {
            leaf_id: u32::from_be_bytes(bytes[0..4].try_into().unwrap()),
            local_id: u32::from_be_bytes(bytes[4..8].try_into().unwrap()),
            fts_doc_id: if raw_fts == FtsDocId::MAX {
                None
            } else {
                Some(raw_fts)
            },
        })
    }
}

/// 256-document block-packed reverse map in `cf_fts_rev_doc_map` mapping `slot -> (DocId, leaf_id, local_id)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FtsRevBlock {
    pub present: [u64; 4],
    pub doc_ids: [DocId; FTS_BLOCK_SIZE],
    pub leaf_ids: [NodeId; FTS_BLOCK_SIZE],
    pub local_ids: [LocalId; FTS_BLOCK_SIZE],
}

impl Default for FtsRevBlock {
    fn default() -> Self {
        Self {
            present: [0u64; 4],
            doc_ids: [0u64; FTS_BLOCK_SIZE],
            leaf_ids: [0u32; FTS_BLOCK_SIZE],
            local_ids: [0u32; FTS_BLOCK_SIZE],
        }
    }
}

impl FtsRevBlock {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.present == [0u64; 4]
    }

    #[inline]
    fn is_present(&self, slot: usize) -> bool {
        ((self.present[slot >> 6] >> (slot & 63)) & 1) != 0
    }

    #[inline]
    pub fn get(&self, slot: u8) -> Option<(DocId, NodeId, LocalId)> {
        let s = slot as usize;
        self.is_present(s)
            .then(|| (self.doc_ids[s], self.leaf_ids[s], self.local_ids[s]))
    }

    #[inline]
    pub fn get_doc_id(&self, slot: u8) -> Option<DocId> {
        let s = slot as usize;
        self.is_present(s).then(|| self.doc_ids[s])
    }

    #[inline]
    pub fn set(&mut self, slot: u8, doc_id: DocId, leaf_id: NodeId, local_id: LocalId) {
        let s = slot as usize;
        self.present[s >> 6] |= 1u64 << (s & 63);
        self.doc_ids[s] = doc_id;
        self.leaf_ids[s] = leaf_id;
        self.local_ids[s] = local_id;
    }

    #[inline]
    pub fn remove(&mut self, slot: u8) {
        let s = slot as usize;
        self.present[s >> 6] &= !(1u64 << (s & 63));
        self.doc_ids[s] = 0;
        self.leaf_ids[s] = 0;
        self.local_ids[s] = 0;
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(FTS_REV_BLOCK_LEN);
        for w in self.present {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        for v in self.doc_ids {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in self.leaf_ids {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in self.local_ids {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != FTS_REV_BLOCK_LEN {
            return Err(TreeError::CorruptedData(format!(
                "FtsRevBlock expected {FTS_REV_BLOCK_LEN} bytes, got {}",
                bytes.len()
            )));
        }
        let mut present = [0u64; 4];
        for (i, out) in present.iter_mut().enumerate() {
            let off = i * 8;
            *out = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        }
        let mut off = FTS_REV_BLOCK_HEADER_LEN;
        let mut doc_ids = [0u64; FTS_BLOCK_SIZE];
        for out in &mut doc_ids {
            *out = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            off += 8;
        }
        let mut leaf_ids = [0u32; FTS_BLOCK_SIZE];
        for out in &mut leaf_ids {
            *out = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            off += 4;
        }
        let mut local_ids = [0u32; FTS_BLOCK_SIZE];
        for out in &mut local_ids {
            *out = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            off += 4;
        }
        Ok(Self {
            present,
            doc_ids,
            leaf_ids,
            local_ids,
        })
    }
}

pub type AttrZoneMap = BTreeMap<NodeId, (Vec<u8>, Vec<u8>)>;
pub(super) type AttrZoneTable = Arc<HashMap<(AttrId, u8), AttrZoneMap>>;
pub(super) type DocLenBlockTable = Arc<Vec<Option<Box<[u8]>>>>;
pub(super) type FtsRevBlockTable = Arc<Vec<Option<Arc<FtsRevBlock>>>>;
