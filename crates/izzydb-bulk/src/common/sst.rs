use std::collections::BTreeMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use izzydb::config::{DistanceMetric, NodeId, SegmentId};
use izzydb::error::{Result, TreeError};
use izzydb::node::AnnMeta;
use izzydb::schema::{CollectionConfig, CollectionSchema};
use izzydb::codec::Manifest;
use rocksdb::{Options, SstFileWriter};
use serde::{Deserialize, Serialize};

pub const SST_BUNDLE_FORMAT_VERSION: u32 = 3;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestAnnMeta {
    pub root_id: NodeId,
    pub root_level: u8,
    pub next_node_id: NodeId,
    pub dim: u32,
    pub metric: u8,
    pub num_vectors: u64,
    pub rabitq_seed: u64,
    pub rabitq_bits: u8,
    pub stale_rows: u64,
    pub vacuum_epoch: u64,
    pub vacuum_cursor: NodeId,
}

impl From<&AnnMeta> for ManifestAnnMeta {
    fn from(m: &AnnMeta) -> Self {
        Self {
            root_id: m.root_id,
            root_level: m.root_level,
            next_node_id: m.next_node_id,
            dim: m.dim,
            metric: m.metric.as_u8(),
            num_vectors: m.num_vectors,
            rabitq_seed: m.rabitq_seed,
            rabitq_bits: m.rabitq_bits,
            stale_rows: m.stale_rows,
            vacuum_epoch: m.vacuum_epoch,
            vacuum_cursor: m.vacuum_cursor,
        }
    }
}

impl ManifestAnnMeta {
    pub fn to_ann_meta(&self) -> Result<AnnMeta> {
        Ok(AnnMeta {
            root_id: self.root_id,
            root_level: self.root_level,
            next_node_id: self.next_node_id,
            dim: self.dim,
            metric: DistanceMetric::from_u8(self.metric)?,
            num_vectors: self.num_vectors,
            rabitq_seed: self.rabitq_seed,
            rabitq_bits: self.rabitq_bits,
            stale_rows: self.stale_rows,
            vacuum_epoch: self.vacuum_epoch,
            vacuum_cursor: self.vacuum_cursor,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestSegmentInfo {
    pub open_segment: Option<SegmentId>,
    pub next_segment_id: SegmentId,
    pub seal_rows: u32,
    pub live_rows: u64,
    pub total_rows: u64,
    pub num_segments: u32,
}

impl ManifestSegmentInfo {
    pub fn from_manifest(m: &Manifest, num_segments: u32) -> Self {
        Self {
            open_segment: m.open_segment,
            next_segment_id: m.next_segment_id,
            seal_rows: m.seal_rows,
            live_rows: m.live_rows,
            total_rows: m.total_rows,
            num_segments,
        }
    }

    pub fn to_manifest(&self) -> Manifest {
        Manifest {
            open_segment: self.open_segment,
            next_segment_id: self.next_segment_id,
            seal_rows: self.seal_rows,
            live_rows: self.live_rows,
            total_rows: self.total_rows,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SstManifest {
    pub format_version: u32,
    pub config_hash: u64,
    pub has_vectors: bool,
    pub has_fts: bool,
    pub schema_bytes: Vec<u8>,
    pub segment_info: ManifestSegmentInfo,
    pub ann_meta: Option<ManifestAnnMeta>,
    pub cf_files: BTreeMap<String, Vec<String>>,
    pub rows_read: u64,
    pub docs_imported: u64,
    pub duplicates_dropped: u64,
    pub leaves: u32,
    pub internal_nodes: u32,
    pub tree_height: u8,
    pub terms: u64,
    pub sst_files: usize,
    pub sst_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct SstBundle {
    pub dir: PathBuf,
    pub manifest: SstManifest,
}

impl SstBundle {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest_path = dir.join("MANIFEST.json");
        let file = File::open(&manifest_path).map_err(|e| {
            TreeError::Storage(format!(
                "open SST bundle manifest {}: {e}",
                manifest_path.display()
            ))
        })?;
        let manifest: SstManifest = serde_json::from_reader(BufReader::new(file)).map_err(|e| {
            TreeError::CorruptedData(format!(
                "parse SST bundle manifest {}: {e}",
                manifest_path.display()
            ))
        })?;
        if manifest.format_version != SST_BUNDLE_FORMAT_VERSION {
            return Err(TreeError::CorruptedData(format!(
                "unsupported SST bundle format_version {}, expected {SST_BUNDLE_FORMAT_VERSION}",
                manifest.format_version
            )));
        }
        Ok(Self { dir, manifest })
    }

    pub fn save_manifest(&self) -> Result<()> {
        let manifest_path = self.dir.join("MANIFEST.json");
        let file = File::create(&manifest_path).map_err(|e| {
            TreeError::Storage(format!(
                "create SST bundle manifest {}: {e}",
                manifest_path.display()
            ))
        })?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.manifest).map_err(|e| {
            TreeError::Storage(format!(
                "write SST bundle manifest {}: {e}",
                manifest_path.display()
            ))
        })?;
        Ok(())
    }

    pub fn files(&self, cf_name: &str) -> Vec<PathBuf> {
        self.manifest
            .cf_files
            .get(cf_name)
            .map(|list| list.iter().map(|rel| self.dir.join(rel)).collect())
            .unwrap_or_default()
    }

    pub fn validate_against(&self, config: &CollectionConfig) -> Result<()> {
        let expected_hash = compute_config_hash(config)?;
        if self.manifest.config_hash != expected_hash {
            return Err(TreeError::ConfigMismatch {
                field: "bulk_import_config",
                expected: format!("{expected_hash:#x}"),
                actual: format!("{:#x}", self.manifest.config_hash),
            });
        }
        let expected_schema = CollectionSchema::from_config(config)?;
        if self.manifest.schema_bytes != expected_schema.encode() {
            return Err(TreeError::ConfigMismatch {
                field: "schema",
                expected: "matching CollectionSchema".into(),
                actual: "mismatched CollectionSchema".into(),
            });
        }
        Ok(())
    }
}

pub fn compute_config_hash(config: &CollectionConfig) -> Result<u64> {
    let schema = CollectionSchema::from_config(config)?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match &config.vector {
        Some(v) => {
            1u8.hash(&mut h);
            v.dim.hash(&mut h);
            v.metric.as_u8().hash(&mut h);
            v.min_leaf_size.hash(&mut h);
            v.max_leaf_size.hash(&mut h);
            v.min_internal_fanout.hash(&mut h);
            v.max_internal_fanout.hash(&mut h);
        }
        None => 0u8.hash(&mut h),
    }
    config.segment.seal_rows.hash(&mut h);
    config.segment.defer_column_build.hash(&mut h);
    schema.encode().hash(&mut h);
    Ok(h.finish())
}

#[derive(Debug, Clone)]
pub struct SstFileInfo {
    pub cf_name: String,
    pub rel_path: String,
    pub bytes: u64,
    pub entries: u64,
}

/// Rolling RocksDB SST writer that splits cleanly into `<prefix>-NNNN.sst` parts
/// when the active SST approaches `target_sst_bytes`.
pub struct SstSink<'a> {
    bundle_dir: PathBuf,
    cf_name: &'a str,
    opts: &'a Options,
    target_bytes: u64,
    prefix: String,
    part_idx: u32,
    writer: Option<SstFileWriter<'a>>,
    active_rel_path: String,
    active_entries: u64,
    approx_bytes: u64,
    finished: Vec<SstFileInfo>,
}

impl<'a> SstSink<'a> {
    pub fn new(
        bundle_dir: PathBuf,
        cf_name: &'a str,
        opts: &'a Options,
        target_bytes: u64,
        prefix: impl Into<String>,
    ) -> Self {
        Self {
            bundle_dir,
            cf_name,
            opts,
            target_bytes: target_bytes.max(1024 * 1024),
            prefix: prefix.into(),
            part_idx: 0,
            writer: None,
            active_rel_path: String::new(),
            active_entries: 0,
            approx_bytes: 0,
            finished: Vec::new(),
        }
    }

    fn ensure_open(&mut self) -> Result<()> {
        if self.writer.is_some() {
            return Ok(());
        }
        let rel = format!("{}-{:04}.sst", self.prefix, self.part_idx);
        let abs = self.bundle_dir.join(&rel);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                TreeError::Storage(format!("create SST dir {}: {e}", parent.display()))
            })?;
        }
        let w = SstFileWriter::create(self.opts);
        w.open(&abs).map_err(|e| {
            TreeError::Storage(format!("open SstFileWriter {}: {e}", abs.display()))
        })?;
        self.part_idx += 1;
        self.active_rel_path = rel;
        self.active_entries = 0;
        self.approx_bytes = 0;
        self.writer = Some(w);
        Ok(())
    }

    pub fn put(&mut self, key: &[u8], val: &[u8]) -> Result<()> {
        self.ensure_open()?;
        let w = self.writer.as_mut().unwrap();
        w.put(key, val).map_err(|e| {
            TreeError::Storage(format!(
                "SstFileWriter({}) put ({} bytes): {e}",
                self.cf_name,
                key.len() + val.len()
            ))
        })?;
        self.active_entries += 1;
        self.approx_bytes += (key.len() + val.len() + 16) as u64;
        if self.approx_bytes >= self.target_bytes {
            self.rotate()?;
        }
        Ok(())
    }

    fn rotate(&mut self) -> Result<()> {
        let Some(mut w) = self.writer.take() else {
            return Ok(());
        };
        if self.active_entries == 0 {
            return Ok(());
        }
        w.finish().map_err(|e| {
            TreeError::Storage(format!(
                "finish SST {}: {e}",
                self.bundle_dir.join(&self.active_rel_path).display()
            ))
        })?;
        let abs = self.bundle_dir.join(&self.active_rel_path);
        let bytes = fs::metadata(&abs).map_or(0, |m| m.len());
        self.finished.push(SstFileInfo {
            cf_name: self.cf_name.to_owned(),
            rel_path: std::mem::take(&mut self.active_rel_path),
            bytes,
            entries: self.active_entries,
        });
        self.active_entries = 0;
        self.approx_bytes = 0;
        Ok(())
    }

    pub fn finish(mut self) -> Result<Vec<SstFileInfo>> {
        self.rotate()?;
        Ok(self.finished)
    }
}
