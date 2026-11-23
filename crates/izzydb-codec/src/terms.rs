
use izzydb_core::config::{
    AttrId,
    TermId,
};
use izzydb_core::error::{Result, TreeError};
use roaring::RoaringBitmap;

// ---------------------------------------------------------------------------
// Roaring & FTS term dictionary keys
// ---------------------------------------------------------------------------

pub fn encode_roaring_bitmap(bm: &RoaringBitmap) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(bm.serialized_size());
    bm.serialize_into(&mut buf)
        .map_err(|e| TreeError::Storage(format!("roaring serialize error: {e}")))?;
    Ok(buf)
}

pub fn decode_roaring_bitmap(bytes: &[u8]) -> Result<RoaringBitmap> {
    RoaringBitmap::deserialize_from(bytes)
        .map_err(|e| TreeError::CorruptedData(format!("roaring deserialize error: {e}")))
}

#[inline]
pub fn encode_fts_term_key(attr_id: AttrId, token: &str) -> Vec<u8> {
    let tb = token.as_bytes();
    let mut key = Vec::with_capacity(2 + tb.len());
    key.extend_from_slice(&attr_id.to_be_bytes());
    key.extend_from_slice(tb);
    key
}

pub fn decode_fts_term_key(key: &[u8]) -> Result<(AttrId, String)> {
    if key.len() < 2 {
        return Err(TreeError::CorruptedData(
            "cf_fts_terms key too short".into(),
        ));
    }
    let attr_id = u16::from_be_bytes([key[0], key[1]]);
    let token = std::str::from_utf8(&key[2..])
        .map_err(|e| TreeError::CorruptedData(format!("invalid utf8 in cf_fts_terms key: {e}")))?
        .to_owned();
    Ok((attr_id, token))
}

#[inline]
pub fn encode_fts_term_id_prefix(term_id: TermId) -> [u8; 4] {
    term_id.to_be_bytes()
}

