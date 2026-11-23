
use izzydb_core::config::{
    DocId, ROW_BLOCK_SIZE,
};
use izzydb_core::error::{Result, TreeError};
use super::*;

// DocIdBlock (`cf_row_doc_ids`)
// ---------------------------------------------------------------------------

pub fn encode_doc_id_block(ids: &[DocId; ROW_BLOCK_SIZE]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(DOC_ID_BLOCK_BYTES);
    #[cfg(target_endian = "little")]
    {
        buf.extend_from_slice(bytemuck::cast_slice(ids));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &id in ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }
    buf
}

pub fn decode_doc_id_block(bytes: &[u8]) -> Result<Box<[DocId; ROW_BLOCK_SIZE]>> {
    if bytes.len() != DOC_ID_BLOCK_BYTES {
        return Err(TreeError::CorruptedData(format!(
            "DocIdBlock expected {DOC_ID_BLOCK_BYTES} bytes, got {}",
            bytes.len()
        )));
    }
    let mut out = Box::new([0u64; ROW_BLOCK_SIZE]);
    #[cfg(target_endian = "little")]
    if let Ok(src) = bytemuck::try_cast_slice::<u8, u64>(bytes) {
        out.copy_from_slice(src);
        return Ok(out);
    }
    for (dst, &chunk) in out.iter_mut().zip(bytes.as_chunks::<8>().0) {
        *dst = u64::from_le_bytes(chunk);
    }
    Ok(out)
}

