//! Parquet scanning, validation, and staged-document encoding for bulk import.

mod decode_value;
mod scan;
mod staging;
mod validate;

pub use decode_value::*;
pub use scan::*;
pub use staging::*;
use validate::*;

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use izzydb::config::DocId;
use izzydb::error::{Result, TreeError};

use crate::common::sorter::{
    SortItem, read_u32_req, read_u64_opt, write_u32, write_u64,
};

/// Wrapper around `memmap2::Mmap` that handles empty (0-byte) files cleanly.
pub enum MappedFile {
    Empty,
    Mapped(Mmap),
}

impl MappedFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| TreeError::Storage(format!("open mmap file {}: {e}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|e| TreeError::Storage(format!("stat mmap file {}: {e}", path.display())))?
            .len();
        if len == 0 {
            return Ok(Self::Empty);
        }
        // Safety: staged files are immutable once Phase 1 completes and live on local disk.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| TreeError::Storage(format!("mmap {}: {e}", path.display())))?;
        Ok(Self::Mapped(mmap))
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::Mapped(m) => m,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DocIndexEntry {
    pub offset: u64,
    pub len: u32,
    pub shard: u32,
}

impl DocIndexEntry {
    pub const SIZE: usize = 16;

    #[inline]
    pub fn encode(self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..12].copy_from_slice(&self.len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.shard.to_le_bytes());
        buf
    }

    #[inline]
    pub fn decode(bytes: &[u8]) -> Self {
        Self {
            offset: u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            len: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            shard: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DocRowPair {
    doc_id: DocId,
    row: u32,
}

impl SortItem for DocRowPair {
    #[inline]
    fn cmp_key(a: &Self, b: &Self) -> std::cmp::Ordering {
        a.doc_id.cmp(&b.doc_id).then_with(|| a.row.cmp(&b.row))
    }

    #[inline]
    fn estimated_bytes(&self) -> usize {
        12
    }

    fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        write_u64(w, self.doc_id)?;
        write_u32(w, self.row)
    }

    fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Option<Self>> {
        let Some(doc_id) = read_u64_opt(r)? else {
            return Ok(None);
        };
        let row = read_u32_req(r)?;
        Ok(Some(Self { doc_id, row }))
    }
}

pub struct StagedData {
    pub rows_read: u64,
    pub docs_imported: u64,
    pub duplicates_dropped: u64,
    pub vectors_path: PathBuf,
    pub doc_ids_by_row_path: PathBuf,
    pub docs_idx_path: PathBuf,
    pub docs_shard_paths: Vec<PathBuf>,
    pub rows_by_doc_id_path: PathBuf,
    pub live_rows: Vec<u32>,
    pub has_detached_staging: bool,
}
