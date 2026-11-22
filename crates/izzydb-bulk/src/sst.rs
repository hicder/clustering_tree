use std::collections::BTreeMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use rocksdb::{Options, SstFileWriter};
use serde::{Deserialize, Serialize};

use izzydb::config::{DistanceMetric, NodeId};
use izzydb::error::{Result, TreeError};
use izzydb::node::TreeMeta;
use izzydb::schema::{CollectionConfig, CollectionSchema};

pub const SST_BUNDLE_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestTreeMeta {
    pub root_id: NodeId,
    pub root_level: u8,
    pub next_node_id: NodeId,
    pub dim: u32,
    pub metric: u8,
    pub num_vectors: u64,
    pub rabitq_seed: u64,
    pub rabitq_bits: u8,
}

impl From<&TreeMeta> for ManifestTreeMeta {
    fn from(m: &TreeMeta) -> Self {
        Self {
            root_id: m.root_id,
            root_level: m.root_level,
            next_node_id: m.next_node_id,
            dim: m.dim,
            metric: m.metric.as_u8(),
            num_vectors: m.num_vectors,
            rabitq_seed: m.rabitq_seed,
            rabitq_bits: m.rabitq_bits,
        }
    }
}

impl ManifestTreeMeta {
    pub fn to_tree_meta(&self) -> Result<TreeMeta> {
        Ok(TreeMeta {
            root_id: self.root_id,
            root_level: self.root_level,
            next_node_id: self.next_node_id,
            dim: self.dim,
            metric: DistanceMetric::from_u8(self.metric)?,
            num_vectors: self.num_vectors,
            rabitq_seed: self.rabitq_seed,
            rabitq_bits: self.rabitq_bits,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SstManifest {
    pub format_version: u32,
    pub config_hash: u64,
    pub dim: usize,
    pub metric: u8,
    pub schema_bytes: Vec<u8>,
    pub tree_meta: ManifestTreeMeta,
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

/// Directory containing sorted, non-overlapping external SST files per column family and `manifest.json`.
#[derive(Debug, Clone)]
pub struct SstBundle {
    pub dir: PathBuf,
    pub manifest: SstManifest,
}

impl SstBundle {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest_path = dir.join("manifest.json");
        let raw = fs::read_to_string(&manifest_path).map_err(|e| {
            TreeError::Storage(format!(
                "failed to read manifest {}: {e}",
                manifest_path.display()
            ))
        })?;
        let manifest: SstManifest = serde_json::from_str(&raw).map_err(|e| {
            TreeError::CorruptedData(format!(
                "invalid manifest {}: {e}",
                manifest_path.display()
            ))
        })?;
        if manifest.format_version != SST_BUNDLE_FORMAT_VERSION {
            return Err(TreeError::InvalidConfig(format!(
                "unsupported SST bundle format version: {}",
                manifest.format_version
            )));
        }
        Ok(Self { dir, manifest })
    }

    pub fn save_manifest(&self) -> Result<()> {
        let manifest_path = self.dir.join("manifest.json");
        let file = File::create(&manifest_path).map_err(|e| {
            TreeError::Storage(format!(
                "failed to create manifest {}: {e}",
                manifest_path.display()
            ))
        })?;
        serde_json::to_writer_pretty(BufWriter::new(file), &self.manifest).map_err(|e| {
            TreeError::Storage(format!(
                "failed to write manifest {}: {e}",
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

    /// Validate that this bundle matches `config` (dim, metric, schema, and format version).
    pub fn validate_against(&self, config: &CollectionConfig) -> Result<()> {
        if self.manifest.format_version != SST_BUNDLE_FORMAT_VERSION {
            return Err(TreeError::InvalidConfig(format!(
                "SST bundle format version mismatch: expected {SST_BUNDLE_FORMAT_VERSION}, got {}",
                self.manifest.format_version
            )));
        }
        if self.manifest.dim != config.tree.dim {
            return Err(TreeError::ConfigMismatch {
                field: "dim",
                expected: self.manifest.dim.to_string(),
                actual: config.tree.dim.to_string(),
            });
        }
        if self.manifest.metric != config.tree.metric.as_u8() {
            return Err(TreeError::ConfigMismatch {
                field: "metric",
                expected: self.manifest.metric.to_string(),
                actual: config.tree.metric.as_u8().to_string(),
            });
        }
        let bundle_schema = CollectionSchema::decode(&self.manifest.schema_bytes)?;
        let expected_schema = CollectionSchema::from_config(config)?;
        if bundle_schema != expected_schema {
            return Err(TreeError::InvalidSchema(
                "SST bundle schema does not match CollectionConfig".into(),
            ));
        }
        let expected_hash = compute_config_hash(config)?;
        if self.manifest.config_hash != expected_hash {
            return Err(TreeError::InvalidConfig(
                "SST bundle CollectionConfig hash mismatch".into(),
            ));
        }
        Ok(())
    }
}

/// Compute a deterministic 64-bit hash over `CollectionConfig` (`dim`, `metric`, and schema).
pub fn compute_config_hash(config: &CollectionConfig) -> Result<u64> {
    let schema = CollectionSchema::from_config(config)?;
    let mut hasher = Fnv1aHasher::default();
    config.tree.dim.hash(&mut hasher);
    config.tree.metric.as_u8().hash(&mut hasher);
    schema.encode().hash(&mut hasher);
    Ok(hasher.finish())
}

#[derive(Default)]
struct Fnv1aHasher(u64);

impl Hasher for Fnv1aHasher {
    fn finish(&self) -> u64 {
        self.0 ^ 0xcbf29ce484222325
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 {
            0xcbf29ce484222325
        } else {
            self.0
        };
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.0 = h;
    }
}

#[derive(Debug, Clone)]
pub struct SstFileInfo {
    pub rel_path: String,
    pub bytes: u64,
}

/// Streaming SST writer that enforces strictly ascending keys and rolls over at key boundaries
/// once `file_size() >= target_bytes`.
pub struct SstSink<'a> {
    bundle_dir: PathBuf,
    cf_name: &'static str,
    opts: &'a Options,
    target_bytes: u64,
    shard_prefix: String,
    seq: usize,
    writer: Option<(SstFileWriter<'a>, PathBuf, String)>,
    last_key: Vec<u8>,
    has_last: bool,
    finished: Vec<SstFileInfo>,
}

impl<'a> SstSink<'a> {
    pub fn new(
        bundle_dir: PathBuf,
        cf_name: &'static str,
        opts: &'a Options,
        target_bytes: u64,
        shard_prefix: String,
    ) -> Self {
        Self {
            bundle_dir,
            cf_name,
            opts,
            target_bytes: target_bytes.max(1024),
            shard_prefix,
            seq: 0,
            writer: None,
            last_key: Vec::new(),
            has_last: false,
            finished: Vec::new(),
        }
    }

    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if self.has_last && key <= self.last_key.as_slice() {
            debug_assert!(
                key > self.last_key.as_slice(),
                "SstSink({}): non-increasing key",
                self.cf_name
            );
            return Err(TreeError::CorruptedData(format!(
                "SstSink({}) received non-increasing key",
                self.cf_name
            )));
        }

        if self.writer.is_none() {
            let rel_path = format!("{}-{:05}.sst", self.shard_prefix, self.seq);
            let abs_path = self.bundle_dir.join(&rel_path);
            if let Some(parent) = abs_path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    TreeError::Storage(format!("create SST dir {}: {e}", parent.display()))
                })?;
            }
            let writer = SstFileWriter::create(self.opts);
            writer
                .open(&abs_path)
                .map_err(|e| TreeError::Storage(format!("SstFileWriter::open: {e}")))?;
            self.seq += 1;
            self.writer = Some((writer, abs_path, rel_path));
        }

        let (writer, _, _) = self.writer.as_mut().unwrap();
        writer
            .put(key, value)
            .map_err(|e| TreeError::Storage(format!("SstFileWriter::put: {e}")))?;

        self.last_key.clear();
        self.last_key.extend_from_slice(key);
        self.has_last = true;

        if writer.file_size() >= self.target_bytes {
            self.roll_current()?;
        }
        Ok(())
    }

    fn roll_current(&mut self) -> Result<()> {
        if let Some((mut writer, abs_path, rel_path)) = self.writer.take() {
            writer
                .finish()
                .map_err(|e| TreeError::Storage(format!("SstFileWriter::finish: {e}")))?;
            let bytes = fs::metadata(&abs_path)
                .map_err(|e| TreeError::Storage(format!("stat SST {}: {e}", abs_path.display())))?
                .len();
            self.finished.push(SstFileInfo { rel_path, bytes });
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<Vec<SstFileInfo>> {
        self.roll_current()?;
        Ok(self.finished)
    }
}
