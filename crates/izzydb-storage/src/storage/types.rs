use std::collections::HashMap;
use std::sync::Arc;

use izzydb_core::config::{AttrId, RowBlock, SegmentId};
use izzydb_core::error::{Result, TreeError};

use crate::codec::ZoneEntry;

#[inline]
pub(super) fn map_db_err(err: rocksdb::Error) -> TreeError {
    TreeError::Storage(err.to_string())
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

/// Cached per-(segment, attr_id, type_tag) zone map entries (`default[0x12]`).
pub type ZoneTable = Arc<HashMap<(AttrId, u8), HashMap<SegmentId, ZoneEntry>>>;

/// Cached per-attribute document-length blocks (`default[0x05]`) keyed by [`RowBlock`].
pub type DocLenBlockTable = Arc<HashMap<RowBlock, Box<[u8]>>>;
