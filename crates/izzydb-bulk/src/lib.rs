//! Parallel offline Parquet bulk importer and appender for `izzydb` (v3 segmented storage).

pub mod common;
mod append;
mod config;
mod import;
mod ingest;
mod pipeline;
mod tree_builder;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use izzydb::Collection;
use izzydb::codec::{SegmentMeta, SegmentState};
use izzydb::config::{AttrId, ColumnGen, DocId, LocalId, NodeId, RowId, SegmentId};
use izzydb::error::{Result, TreeError};
use izzydb::fts::{FtsMeta, PostingBlockHeader, SkipBlock, TermRecord};
use izzydb::node::PackedLeafNode;
use izzydb::rabitq::{EncodeScratch, RabitqQuantizer};
use izzydb::schema::{CollectionConfig, CollectionSchema, ColumnStorage};
use izzydb::storage::{ALL_DATA_CF_NAMES, DEFAULT_CF_NAME, Storage};

pub use common::parquet_reader::scan_detached_from_parquet;
pub use common::sst::{
    ManifestAnnMeta, ManifestSegmentInfo, SST_BUNDLE_FORMAT_VERSION, SstBundle, SstManifest,
};

use common::parquet_reader::{MappedFile, read_doc_id_at, scan_and_stage};
use common::sst::{SstFileInfo, compute_config_hash};
use pipeline::{emit_empty_import_ssts, emit_segment_range_ssts};
use tree_builder::{VecEntry, build_tree_and_assign, read_vector_at};

pub use append::*;
pub use config::*;
pub use import::*;
pub use ingest::*;

fn same_filesystem(a: &Path, b: &Path) -> bool {
    let Ok(ma) = fs::metadata(a) else {
        return false;
    };
    let Ok(mb) = fs::metadata(b) else {
        return false;
    };
    ma.dev() == mb.dev()
}

#[inline]
fn report_progress(
    cb: Option<&dyn Fn(Progress)>,
    phase: &'static str,
    completed: u64,
    total: Option<u64>,
) {
    if let Some(f) = cb {
        f(Progress {
            phase,
            completed,
            total,
        });
    }
}

#[inline]
fn map_io(e: std::io::Error) -> TreeError {
    TreeError::Storage(e.to_string())
}
