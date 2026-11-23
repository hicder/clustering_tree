//! Little-endian f32/u32/u64 slice codecs.

use std::borrow::Cow;
use izzydb_core::error::{Result, TreeError};


/// Encode `vals` as raw little-endian `f32` bytes (no header).
pub fn encode_f32_slice(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    write_f32_le(vals, &mut out);
    out
}

/// Decode `dim` little-endian `f32` values from `bytes`.
pub fn decode_f32_slice(bytes: &[u8], dim: usize) -> Result<Vec<f32>> {
    read_f32_le(bytes, dim)
}

/// Decode little-endian `f32` values from `bytes` directly into `dst`.
#[inline]
pub fn decode_f32_into(bytes: &[u8], dst: &mut [f32]) -> Result<()> {
    let expected = dst.len() * 4;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "f32 slice expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        dst.copy_from_slice(slice);
        return Ok(());
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    for (out, &chunk) in dst.iter_mut().zip(chunks.iter()) {
        *out = f32::from_le_bytes(chunk);
    }
    Ok(())
}

/// Borrow `bytes` directly as `&[f32]` on little-endian hosts when 4-byte aligned,
/// falling back to decoding into an owned `Vec<f32>`.
#[inline]
pub fn borrow_f32_slice<'a>(bytes: &'a [u8], dim: usize) -> Result<Cow<'a, [f32]>> {
    let expected = dim * 4;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "vector payload expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        return Ok(Cow::Borrowed(slice));
    }
    Ok(Cow::Owned(read_f32_le(bytes, dim)?))
}

#[inline]
pub(super) fn write_f32_le(slice: &[f32], out: &mut Vec<u8>) {
    #[cfg(target_endian = "little")]
    {
        out.extend_from_slice(bytemuck::cast_slice(slice));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &v in slice {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

#[inline]
pub(super) fn write_u32_le(slice: &[u32], out: &mut Vec<u8>) {
    #[cfg(target_endian = "little")]
    {
        out.extend_from_slice(bytemuck::cast_slice(slice));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &v in slice {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

#[inline]
pub(super) fn write_u64_le(slice: &[u64], out: &mut Vec<u8>) {
    #[cfg(target_endian = "little")]
    {
        out.extend_from_slice(bytemuck::cast_slice(slice));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &v in slice {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

#[inline]
pub(super) fn read_f32_le(bytes: &[u8], len: usize) -> Result<Vec<f32>> {
    let expected = len * 4;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "f32 slice expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        return Ok(slice.to_vec());
    }
    let mut out = Vec::with_capacity(len);
    let (chunks, _) = bytes.as_chunks::<4>();
    for &chunk in chunks {
        out.push(f32::from_le_bytes(chunk));
    }
    Ok(out)
}

#[inline]
pub(super) fn read_u32_le(bytes: &[u8], len: usize) -> Result<Vec<u32>> {
    let expected = len * 4;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "u32 slice expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, u32>(bytes) {
        return Ok(slice.to_vec());
    }
    let mut out = Vec::with_capacity(len);
    let (chunks, _) = bytes.as_chunks::<4>();
    for &chunk in chunks {
        out.push(u32::from_le_bytes(chunk));
    }
    Ok(out)
}

#[inline]
pub(super) fn read_u64_le(bytes: &[u8], len: usize) -> Result<Vec<u64>> {
    let expected = len * 8;
    if bytes.len() != expected {
        return Err(TreeError::CorruptedData(format!(
            "u64 slice expected {expected} bytes, got {}",
            bytes.len()
        )));
    }
    #[cfg(target_endian = "little")]
    if let Ok(slice) = bytemuck::try_cast_slice::<u8, u64>(bytes) {
        return Ok(slice.to_vec());
    }
    let mut out = Vec::with_capacity(len);
    let (chunks, _) = bytes.as_chunks::<8>();
    for &chunk in chunks {
        out.push(u64::from_le_bytes(chunk));
    }
    Ok(out)
}
