//! Bulk import configuration, reports, and hook stages.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DuplicatePolicy {
    #[default]
    Error,
    LastWins,
}

#[derive(Debug, Clone)]
pub struct Progress {
    pub phase: &'static str,
    pub completed: u64,
    pub total: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct BulkImportConfig {
    pub collection: CollectionConfig,
    pub id_column: String,
    pub vector_column: Option<String>,
    /// Parquet column -> schema attribute name; `None` maps every schema attribute by the same name.
    pub column_map: Option<Vec<(String, String)>>,
    pub on_duplicate_id: DuplicatePolicy,
    pub fill_factor: f32,
    pub train_sample: usize,
    pub work_dir: PathBuf,
    pub keep_work_dir: bool,
    pub memory_budget_bytes: usize,
    pub target_sst_bytes: u64,
    pub threads: usize,
    pub verify_after: bool,
}

impl BulkImportConfig {
    pub fn new(
        collection: CollectionConfig,
        id_column: impl Into<String>,
        vector_column: impl Into<String>,
        work_dir: impl Into<PathBuf>,
    ) -> Self {
        let vec_col = if collection.vector.is_some() {
            Some(vector_column.into())
        } else {
            None
        };
        Self {
            collection,
            id_column: id_column.into(),
            vector_column: vec_col,
            column_map: None,
            on_duplicate_id: DuplicatePolicy::Error,
            fill_factor: 0.75,
            train_sample: 1_000_000,
            work_dir: work_dir.into(),
            keep_work_dir: false,
            memory_budget_bytes: 2 * 1024 * 1024 * 1024,
            target_sst_bytes: 256 * 1024 * 1024,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            verify_after: false,
        }
    }

    pub fn new_vectorless(
        collection: CollectionConfig,
        id_column: impl Into<String>,
        work_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            collection,
            id_column: id_column.into(),
            vector_column: None,
            column_map: None,
            on_duplicate_id: DuplicatePolicy::Error,
            fill_factor: 0.75,
            train_sample: 1_000_000,
            work_dir: work_dir.into(),
            keep_work_dir: false,
            memory_budget_bytes: 2 * 1024 * 1024 * 1024,
            target_sst_bytes: 256 * 1024 * 1024,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            verify_after: false,
        }
    }

    pub fn with_vector_column(mut self, col: impl Into<String>) -> Self {
        self.vector_column = Some(col.into());
        self
    }

    pub fn with_column_map(mut self, column_map: Vec<(String, String)>) -> Self {
        self.column_map = Some(column_map);
        self
    }

    pub fn with_on_duplicate_id(mut self, policy: DuplicatePolicy) -> Self {
        self.on_duplicate_id = policy;
        self
    }

    pub fn with_fill_factor(mut self, fill_factor: f32) -> Self {
        self.fill_factor = fill_factor;
        self
    }

    pub fn with_train_sample(mut self, train_sample: usize) -> Self {
        self.train_sample = train_sample;
        self
    }

    pub fn with_keep_work_dir(mut self, keep: bool) -> Self {
        self.keep_work_dir = keep;
        self
    }

    pub fn with_memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.memory_budget_bytes = bytes;
        self
    }

    pub fn with_target_sst_bytes(mut self, bytes: u64) -> Self {
        self.target_sst_bytes = bytes;
        self
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_verify_after(mut self, verify: bool) -> Self {
        self.verify_after = verify;
        self
    }

    pub fn validate(&self) -> Result<()> {
        self.collection.validate()?;
        if self.id_column.is_empty() {
            return Err(TreeError::InvalidConfig("id_column cannot be empty".into()));
        }
        if self.collection.vector.is_some() {
            let Some(ref vc) = self.vector_column else {
                return Err(TreeError::InvalidConfig(
                    "vector_column is required when CollectionConfig::vector is Some".into(),
                ));
            };
            if vc.is_empty() {
                return Err(TreeError::InvalidConfig(
                    "vector_column cannot be empty".into(),
                ));
            }
        }
        if !self.fill_factor.is_finite() || self.fill_factor <= 0.0 || self.fill_factor > 1.0 {
            return Err(TreeError::InvalidConfig(format!(
                "fill_factor must be in (0.0, 1.0], got {}",
                self.fill_factor
            )));
        }
        if self.target_sst_bytes == 0 {
            return Err(TreeError::InvalidConfig(
                "target_sst_bytes must be > 0".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BulkImportReport {
    pub rows_read: u64,
    pub docs_imported: u64,
    pub duplicates_dropped: u64,
    pub segments_created: u32,
    pub leaves: u32,
    pub internal_nodes: u32,
    pub tree_height: u8,
    pub terms: u64,
    pub sst_files: usize,
    pub sst_bytes: u64,
    pub phase_timings: Vec<(&'static str, Duration)>,
}

#[derive(Debug, Clone, Default)]
pub struct BulkAppendReport {
    pub rows_read: u64,
    pub docs_appended: u64,
    pub docs_replaced: u64,
    pub duplicates_dropped: u64,
    pub segments_created: u32,
    pub rebuilt_ann_tree: bool,
    pub sst_files: usize,
    pub sst_bytes: u64,
    pub phase_timings: Vec<(&'static str, Duration)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BulkAppendHookStage {
    AfterBuildingMeta,
    AfterIngestCf(&'static str),
    BeforeFinalCommit,
}
