use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, SchemaRef};
use memmap2::Mmap;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;

use izzydb::codec::encode_attributes;
use izzydb::config::{DistanceMetric, DocId};
use izzydb::error::{Result, TreeError};
use izzydb::metric::prepare_vector;
use izzydb::schema::{AttributeType, AttributeValue, Attributes, CollectionSchema, SchemaEntry};

use crate::DuplicatePolicy;
use crate::sorter::{ExternalSorter, SortItem, read_u32_req, read_u64_opt, write_u32, write_u64};

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
        // Safety: staged files are immutable once Phase 1/3 completes and live on local disk.
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
}

struct RowGroupTask {
    file_idx: usize,
    rg_idx: usize,
    global_row_start: u32,
    row_in_file_start: usize,
    projected_root_indices: Vec<usize>,
    attr_bindings: Vec<AttrColumnBinding>,
}

#[derive(Clone)]
struct AttrColumnBinding {
    parquet_col: String,
    schema_attr_name: String,
    attr_type: AttributeType,
}

struct DocShardWriter {
    shard_id: u32,
    offset: u64,
    writer: BufWriter<File>,
}

#[allow(clippy::too_many_arguments)]
pub fn scan_and_stage(
    parquet_files: &[PathBuf],
    schema: &CollectionSchema,
    dim: usize,
    metric: DistanceMetric,
    id_column: &str,
    vector_column: &str,
    column_map: Option<&[(String, String)]>,
    on_duplicate_id: DuplicatePolicy,
    stage_dir: &Path,
    memory_budget_bytes: usize,
    pool: &rayon::ThreadPool,
) -> Result<StagedData> {
    fs::create_dir_all(stage_dir).map_err(|e| {
        TreeError::Storage(format!("create staging dir {}: {e}", stage_dir.display()))
    })?;

    // Resolve Parquet column -> schema attribute mapping.
    let col_to_schema = build_column_mapping(schema, column_map)?;

    let mut tasks = Vec::new();
    let mut total_rows: u64 = 0;

    for (file_idx, path) in parquet_files.iter().enumerate() {
        let file = File::open(path).map_err(|e| {
            TreeError::Storage(format!("open parquet file {}: {e}", path.display()))
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            TreeError::CorruptedData(format!("read parquet metadata {}: {e}", path.display()))
        })?;
        let arrow_schema = builder.schema().clone();
        let parquet_meta = builder.metadata().clone();

        let (projected_root_indices, attr_bindings) = validate_and_bind_file_schema(
            path,
            &arrow_schema,
            id_column,
            vector_column,
            dim,
            &col_to_schema,
        )?;

        let mut row_in_file = 0usize;
        for rg_idx in 0..parquet_meta.num_row_groups() {
            let rg_rows = parquet_meta.row_group(rg_idx).num_rows() as usize;
            if total_rows + (rg_rows as u64) > u32::MAX as u64 {
                return Err(TreeError::InvalidConfig(
                    "bulk import exceeds u32 staging row capacity".into(),
                ));
            }
            if rg_rows > 0 {
                tasks.push(RowGroupTask {
                    file_idx,
                    rg_idx,
                    global_row_start: total_rows as u32,
                    row_in_file_start: row_in_file,
                    projected_root_indices: projected_root_indices.clone(),
                    attr_bindings: attr_bindings.clone(),
                });
                total_rows += rg_rows as u64;
                row_in_file += rg_rows;
            }
        }
    }

    let vectors_path = stage_dir.join("vectors.bin");
    let doc_ids_by_row_path = stage_dir.join("doc_ids_by_row.bin");
    let docs_idx_path = stage_dir.join("docs.idx");
    let rows_by_doc_id_path = stage_dir.join("rows_by_doc_id.bin");

    if total_rows == 0 {
        File::create(&vectors_path).map_err(map_io)?;
        File::create(&doc_ids_by_row_path).map_err(map_io)?;
        File::create(&docs_idx_path).map_err(map_io)?;
        File::create(&rows_by_doc_id_path).map_err(map_io)?;
        return Ok(StagedData {
            rows_read: 0,
            docs_imported: 0,
            duplicates_dropped: 0,
            vectors_path,
            doc_ids_by_row_path,
            docs_idx_path,
            docs_shard_paths: Vec::new(),
            rows_by_doc_id_path,
            live_rows: Vec::new(),
        });
    }

    let vectors_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(&vectors_path)
        .map_err(map_io)?;
    vectors_file
        .set_len(total_rows * (dim as u64) * 4)
        .map_err(map_io)?;

    let doc_ids_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(&doc_ids_by_row_path)
        .map_err(map_io)?;
    doc_ids_file.set_len(total_rows * 8).map_err(map_io)?;

    let docs_idx_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(&docs_idx_path)
        .map_err(map_io)?;
    docs_idx_file
        .set_len(total_rows * DocIndexEntry::SIZE as u64)
        .map_err(map_io)?;

    let num_shards = pool.current_num_threads().min(tasks.len()).clamp(1, 64);
    let mut docs_shard_paths = Vec::with_capacity(num_shards);
    let mut doc_writers = Vec::with_capacity(num_shards);
    for shard_id in 0..num_shards {
        let p = stage_dir.join(format!("docs_shard_{shard_id:02}.bin"));
        let f = File::create(&p).map_err(map_io)?;
        docs_shard_paths.push(p);
        doc_writers.push(DocShardWriter {
            shard_id: shard_id as u32,
            offset: 0,
            writer: BufWriter::with_capacity(256 * 1024, f),
        });
    }
    let doc_writer_pool = Mutex::new(doc_writers);

    let sorter = ExternalSorter::<DocRowPair>::new(
        "doc_row",
        stage_dir.join("runs_doc_row"),
        memory_budget_bytes / 2,
        num_shards,
    )?;

    pool.install(|| -> Result<()> {
        tasks.par_iter().try_for_each(|task| -> Result<()> {
            let path = &parquet_files[task.file_idx];
            let file = File::open(path).map_err(|e| {
                TreeError::Storage(format!("open parquet {}: {e}", path.display()))
            })?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
                TreeError::CorruptedData(format!("parquet reader {}: {e}", path.display()))
            })?;
            let mask =
                ProjectionMask::roots(builder.parquet_schema(), task.projected_root_indices.iter().copied());
            let reader = builder
                .with_row_groups(vec![task.rg_idx])
                .with_projection(mask)
                .with_batch_size(8192)
                .build()
                .map_err(|e| {
                    TreeError::CorruptedData(format!(
                        "build parquet reader {} rg {}: {e}",
                        path.display(),
                        task.rg_idx
                    ))
                })?;

            let mut doc_writer = doc_writer_pool.lock().unwrap().pop().unwrap();
            let mut sorter_shard = sorter.shard();

            let mut rg_row_offset = 0usize;
            for batch_res in reader {
                let batch = batch_res.map_err(|e| {
                    TreeError::CorruptedData(format!(
                        "read parquet batch {} rg {}: {e}",
                        path.display(),
                        task.rg_idx
                    ))
                })?;
                let batch_rows = batch.num_rows();
                process_record_batch(
                    path,
                    task.rg_idx,
                    task.row_in_file_start + rg_row_offset,
                    task.global_row_start + rg_row_offset as u32,
                    &batch,
                    schema,
                    dim,
                    metric,
                    id_column,
                    vector_column,
                    &task.attr_bindings,
                    &vectors_file,
                    &doc_ids_file,
                    &docs_idx_file,
                    &mut doc_writer,
                    &mut sorter_shard,
                )?;
                rg_row_offset += batch_rows;
            }

            sorter_shard.finish()?;
            doc_writer_pool.lock().unwrap().push(doc_writer);
            Ok(())
        })
    })?;

    for mut dw in doc_writer_pool.into_inner().unwrap() {
        dw.writer.flush().map_err(map_io)?;
    }
    drop(vectors_file);
    drop(doc_ids_file);
    drop(docs_idx_file);

    // Phase 1.4: Merge sorted (doc_id, row) pairs and resolve duplicates.
    let mut sorted_iter = sorter.into_sorted_iter()?;
    let rows_out = File::create(&rows_by_doc_id_path).map_err(map_io)?;
    let mut rows_writer = BufWriter::with_capacity(256 * 1024, rows_out);

    let mut docs_imported = 0u64;
    let mut duplicates_dropped = 0u64;
    let mut live_rows = Vec::with_capacity(total_rows as usize);

    let mut pending: Option<DocRowPair> = sorted_iter.next_item()?;
    while let Some(mut cur) = pending {
        loop {
            match sorted_iter.next_item()? {
                Some(next) if next.doc_id == cur.doc_id => {
                    match on_duplicate_id {
                        DuplicatePolicy::Error => {
                            return Err(TreeError::InvalidConfig(format!(
                                "duplicate doc_id {} encountered during bulk import",
                                cur.doc_id
                            )));
                        }
                        DuplicatePolicy::LastWins => {
                            duplicates_dropped += 1;
                            // Pairs are ordered by (doc_id, row); higher row == later (file_idx, row_in_file).
                            cur = next;
                        }
                    }
                }
                other => {
                    pending = other;
                    break;
                }
            }
        }
        rows_writer.write_all(&cur.row.to_le_bytes()).map_err(map_io)?;
        live_rows.push(cur.row);
        docs_imported += 1;
    }
    rows_writer.flush().map_err(map_io)?;

    Ok(StagedData {
        rows_read: total_rows,
        docs_imported,
        duplicates_dropped,
        vectors_path,
        doc_ids_by_row_path,
        docs_idx_path,
        docs_shard_paths,
        rows_by_doc_id_path,
        live_rows,
    })
}

fn build_column_mapping<'a>(
    schema: &'a CollectionSchema,
    column_map: Option<&[(String, String)]>,
) -> Result<HashMap<String, &'a SchemaEntry>> {
    let mut map = HashMap::new();
    match column_map {
        None => {
            for entry in schema.entries() {
                map.insert(entry.name.clone(), entry);
            }
        }
        Some(pairs) => {
            let mut seen_targets = HashSet::new();
            for (pq_col, schema_attr) in pairs {
                let entry = schema.get_by_name(schema_attr).ok_or_else(|| {
                    TreeError::InvalidConfig(format!(
                        "column_map references unknown schema attribute '{schema_attr}'"
                    ))
                })?;
                if !seen_targets.insert(schema_attr.as_str()) {
                    return Err(TreeError::InvalidConfig(format!(
                        "duplicate target schema attribute '{schema_attr}' in column_map"
                    )));
                }
                if map.insert(pq_col.clone(), entry).is_some() {
                    return Err(TreeError::InvalidConfig(format!(
                        "duplicate parquet column '{pq_col}' in column_map"
                    )));
                }
            }
        }
    }
    Ok(map)
}

fn validate_and_bind_file_schema(
    path: &Path,
    arrow_schema: &SchemaRef,
    id_column: &str,
    vector_column: &str,
    dim: usize,
    col_to_schema: &HashMap<String, &SchemaEntry>,
) -> Result<(Vec<usize>, Vec<AttrColumnBinding>)> {
    let id_idx = arrow_schema.index_of(id_column).map_err(|_| {
        TreeError::InvalidConfig(format!(
            "parquet file {} missing id column '{id_column}'",
            path.display()
        ))
    })?;
    let id_dt = arrow_schema.field(id_idx).data_type();
    if !matches!(
        id_dt,
        DataType::UInt64 | DataType::Int64 | DataType::UInt32 | DataType::Int32
    ) {
        return Err(TreeError::InvalidConfig(format!(
            "parquet file {} id column '{id_column}' has unsupported Arrow type {id_dt:?}",
            path.display()
        )));
    }

    let vec_idx = arrow_schema.index_of(vector_column).map_err(|_| {
        TreeError::InvalidConfig(format!(
            "parquet file {} missing vector column '{vector_column}'",
            path.display()
        ))
    })?;
    let vec_dt = arrow_schema.field(vec_idx).data_type();
    validate_vector_datatype(path, vector_column, vec_dt, dim)?;

    let mut projected_indices = vec![id_idx, vec_idx];
    let mut bindings = Vec::new();

    for (idx, field) in arrow_schema.fields().iter().enumerate() {
        let name = field.name();
        if name == id_column || name == vector_column {
            continue;
        }
        if let Some(entry) = col_to_schema.get(name) {
            validate_attr_datatype(path, name, field.data_type(), entry.attr_type)?;
            projected_indices.push(idx);
            bindings.push(AttrColumnBinding {
                parquet_col: name.clone(),
                schema_attr_name: entry.name.clone(),
                attr_type: entry.attr_type,
            });
        }
    }

    projected_indices.sort_unstable();
    projected_indices.dedup();
    Ok((projected_indices, bindings))
}

fn validate_vector_datatype(
    path: &Path,
    col: &str,
    dt: &DataType,
    expected_dim: usize,
) -> Result<()> {
    match dt {
        DataType::FixedSizeList(inner, size) => {
            if *size as usize != expected_dim {
                return Err(TreeError::DimensionMismatch {
                    expected: expected_dim,
                    actual: *size as usize,
                });
            }
            if !matches!(inner.data_type(), DataType::Float32 | DataType::Float64) {
                return Err(TreeError::InvalidConfig(format!(
                    "parquet file {} vector column '{col}' inner type must be Float32/Float64, got {:?}",
                    path.display(),
                    inner.data_type()
                )));
            }
            Ok(())
        }
        DataType::List(inner) | DataType::LargeList(inner) => {
            if !matches!(inner.data_type(), DataType::Float32 | DataType::Float64) {
                return Err(TreeError::InvalidConfig(format!(
                    "parquet file {} vector column '{col}' inner type must be Float32/Float64, got {:?}",
                    path.display(),
                    inner.data_type()
                )));
            }
            Ok(())
        }
        other => Err(TreeError::InvalidConfig(format!(
            "parquet file {} vector column '{col}' must be FixedSizeList or List, got {other:?}",
            path.display()
        ))),
    }
}

fn validate_attr_datatype(
    path: &Path,
    col: &str,
    dt: &DataType,
    attr_type: AttributeType,
) -> Result<()> {
    let ok = match attr_type {
        AttributeType::String => matches!(
            dt,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ),
        AttributeType::Int => matches!(
            dt,
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
        ),
        AttributeType::Uint => matches!(
            dt,
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
        ),
        AttributeType::Float => matches!(dt, DataType::Float32 | DataType::Float64),
        AttributeType::Bool => matches!(dt, DataType::Boolean),
        AttributeType::StringArray => matches!(
            dt,
            DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _)
                if matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
        ),
        AttributeType::IntArray => matches!(
            dt,
            DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _)
                if matches!(f.data_type(), DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64)
        ),
        AttributeType::UintArray => matches!(
            dt,
            DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _)
                if matches!(f.data_type(), DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64)
        ),
    };
    if !ok {
        return Err(TreeError::InvalidConfig(format!(
            "parquet file {} column '{col}' Arrow type {dt:?} incompatible with schema type {attr_type:?}",
            path.display()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_record_batch(
    path: &Path,
    rg_idx: usize,
    file_row_base: usize,
    global_row_base: u32,
    batch: &RecordBatch,
    schema: &CollectionSchema,
    dim: usize,
    metric: DistanceMetric,
    id_column: &str,
    vector_column: &str,
    attr_bindings: &[AttrColumnBinding],
    vectors_file: &File,
    doc_ids_file: &File,
    docs_idx_file: &File,
    doc_writer: &mut DocShardWriter,
    sorter_shard: &mut crate::sorter::SorterShard<DocRowPair>,
) -> Result<()> {
    let n = batch.num_rows();
    if n == 0 {
        return Ok(());
    }

    let id_col = batch.column_by_name(id_column).ok_or_else(|| {
        TreeError::CorruptedData(format!(
            "missing projected id column '{id_column}' in {}",
            path.display()
        ))
    })?;
    let vec_col = batch.column_by_name(vector_column).ok_or_else(|| {
        TreeError::CorruptedData(format!(
            "missing projected vector column '{vector_column}' in {}",
            path.display()
        ))
    })?;

    let attr_cols: Vec<(&AttrColumnBinding, &ArrayRef)> = attr_bindings
        .iter()
        .map(|b| {
            let arr = batch.column_by_name(&b.parquet_col).unwrap();
            (b, arr)
        })
        .collect();

    let mut vec_buf = vec![0.0f32; dim];
    let mut batch_vecs_bytes = Vec::with_capacity(n * dim * 4);
    let mut batch_doc_ids_bytes = Vec::with_capacity(n * 8);
    let mut batch_idx_bytes = Vec::with_capacity(n * DocIndexEntry::SIZE);

    for row in 0..n {
        let file_row = file_row_base + row;
        let global_row = global_row_base + row as u32;

        let doc_id = decode_doc_id(path, rg_idx, file_row, id_col, row)?;
        decode_vector_into(path, rg_idx, file_row, vec_col, row, dim, &mut vec_buf)?;
        let prepared = prepare_vector(&vec_buf, dim, metric).map_err(|e| {
            TreeError::InvalidAttribute {
                name: vector_column.to_owned(),
                reason: format!("{}:{rg_idx}:{file_row}: {e}", path.display()),
            }
        })?;

        let mut attrs = Attributes::new();
        for &(binding, arr) in &attr_cols {
            if let Some(val) = decode_attr_value(
                path,
                rg_idx,
                file_row,
                &binding.parquet_col,
                arr,
                row,
                binding.attr_type,
            )? {
                attrs.insert(binding.schema_attr_name.clone(), val);
            }
        }
        schema.validate_attributes(&attrs).map_err(|e| {
            TreeError::InvalidAttribute {
                name: String::new(),
                reason: format!("{}:{rg_idx}:{file_row}: {e}", path.display()),
            }
        })?;

        let doc_bytes = encode_attributes(schema, &attrs)?;
        let offset = doc_writer.offset;
        let len = doc_bytes.len() as u32;
        doc_writer.writer.write_all(&doc_bytes).map_err(map_io)?;
        doc_writer.offset += len as u64;

        for &f in &prepared {
            batch_vecs_bytes.extend_from_slice(&f.to_le_bytes());
        }
        batch_doc_ids_bytes.extend_from_slice(&doc_id.to_le_bytes());
        batch_idx_bytes.extend_from_slice(
            &DocIndexEntry {
                offset,
                len,
                shard: doc_writer.shard_id,
            }
            .encode(),
        );

        sorter_shard.push(DocRowPair {
            doc_id,
            row: global_row,
        })?;
    }

    vectors_file
        .write_all_at(&batch_vecs_bytes, (global_row_base as u64) * (dim as u64) * 4)
        .map_err(map_io)?;
    doc_ids_file
        .write_all_at(&batch_doc_ids_bytes, (global_row_base as u64) * 8)
        .map_err(map_io)?;
    docs_idx_file
        .write_all_at(
            &batch_idx_bytes,
            (global_row_base as u64) * DocIndexEntry::SIZE as u64,
        )
        .map_err(map_io)?;

    Ok(())
}

fn row_err(path: &Path, rg_idx: usize, row: usize, msg: impl std::fmt::Display) -> TreeError {
    TreeError::InvalidConfig(format!("{}:rg={rg_idx}:row={row}: {msg}", path.display()))
}

fn decode_doc_id(
    path: &Path,
    rg_idx: usize,
    file_row: usize,
    arr: &ArrayRef,
    row: usize,
) -> Result<DocId> {
    if arr.is_null(row) {
        return Err(row_err(path, rg_idx, file_row, "null id value"));
    }
    match arr.data_type() {
        DataType::UInt64 => Ok(arr.as_primitive::<UInt64Type>().value(row)),
        DataType::Int64 => {
            let v = arr.as_primitive::<Int64Type>().value(row);
            if v < 0 {
                return Err(row_err(path, rg_idx, file_row, format!("negative id {v}")));
            }
            Ok(v as u64)
        }
        DataType::UInt32 => Ok(arr.as_primitive::<UInt32Type>().value(row) as u64),
        DataType::Int32 => {
            let v = arr.as_primitive::<Int32Type>().value(row);
            if v < 0 {
                return Err(row_err(path, rg_idx, file_row, format!("negative id {v}")));
            }
            Ok(v as u64)
        }
        other => Err(row_err(
            path,
            rg_idx,
            file_row,
            format!("unsupported id type {other:?}"),
        )),
    }
}

fn decode_vector_into(
    path: &Path,
    rg_idx: usize,
    file_row: usize,
    arr: &ArrayRef,
    row: usize,
    dim: usize,
    dst: &mut [f32],
) -> Result<()> {
    if arr.is_null(row) {
        return Err(row_err(path, rg_idx, file_row, "null vector value"));
    }
    let slice_arr: ArrayRef = match arr.data_type() {
        DataType::FixedSizeList(_, _) => arr.as_fixed_size_list().value(row),
        DataType::List(_) => arr.as_list::<i32>().value(row),
        DataType::LargeList(_) => arr.as_list::<i64>().value(row),
        other => {
            return Err(row_err(
                path,
                rg_idx,
                file_row,
                format!("unsupported vector column type {other:?}"),
            ));
        }
    };
    if slice_arr.len() != dim {
        return Err(TreeError::DimensionMismatch {
            expected: dim,
            actual: slice_arr.len(),
        });
    }
    if slice_arr.null_count() > 0 {
        return Err(row_err(path, rg_idx, file_row, "vector contains null element"));
    }
    match slice_arr.data_type() {
        DataType::Float32 => {
            let vals = slice_arr.as_primitive::<Float32Type>().values();
            dst.copy_from_slice(vals);
        }
        DataType::Float64 => {
            let vals = slice_arr.as_primitive::<Float64Type>().values();
            for (out, &v) in dst.iter_mut().zip(vals.iter()) {
                if !v.is_finite() {
                    return Err(TreeError::NonFiniteVector);
                }
                *out = v as f32;
            }
        }
        other => {
            return Err(row_err(
                path,
                rg_idx,
                file_row,
                format!("unsupported vector element type {other:?}"),
            ));
        }
    }
    Ok(())
}

fn decode_attr_value(
    path: &Path,
    rg_idx: usize,
    file_row: usize,
    col_name: &str,
    arr: &ArrayRef,
    row: usize,
    expected_type: AttributeType,
) -> Result<Option<AttributeValue>> {
    if arr.is_null(row) {
        return Ok(None);
    }
    let val = match expected_type {
        AttributeType::String => AttributeValue::String(match arr.data_type() {
            DataType::Utf8 => arr.as_string::<i32>().value(row).to_owned(),
            DataType::LargeUtf8 => arr.as_string::<i64>().value(row).to_owned(),
            DataType::Utf8View => arr.as_string_view().value(row).to_owned(),
            other => {
                return Err(row_err(
                    path,
                    rg_idx,
                    file_row,
                    format!("column '{col_name}' type mismatch: {other:?}"),
                ));
            }
        }),
        AttributeType::Int => AttributeValue::Int(match arr.data_type() {
            DataType::Int8 => arr.as_primitive::<Int8Type>().value(row) as i64,
            DataType::Int16 => arr.as_primitive::<Int16Type>().value(row) as i64,
            DataType::Int32 => arr.as_primitive::<Int32Type>().value(row) as i64,
            DataType::Int64 => arr.as_primitive::<Int64Type>().value(row),
            other => {
                return Err(row_err(
                    path,
                    rg_idx,
                    file_row,
                    format!("column '{col_name}' type mismatch: {other:?}"),
                ));
            }
        }),
        AttributeType::Uint => AttributeValue::Uint(match arr.data_type() {
            DataType::UInt8 => arr.as_primitive::<UInt8Type>().value(row) as u64,
            DataType::UInt16 => arr.as_primitive::<UInt16Type>().value(row) as u64,
            DataType::UInt32 => arr.as_primitive::<UInt32Type>().value(row) as u64,
            DataType::UInt64 => arr.as_primitive::<UInt64Type>().value(row),
            other => {
                return Err(row_err(
                    path,
                    rg_idx,
                    file_row,
                    format!("column '{col_name}' type mismatch: {other:?}"),
                ));
            }
        }),
        AttributeType::Float => {
            let v = match arr.data_type() {
                DataType::Float32 => arr.as_primitive::<Float32Type>().value(row) as f64,
                DataType::Float64 => arr.as_primitive::<Float64Type>().value(row),
                other => {
                    return Err(row_err(
                        path,
                        rg_idx,
                        file_row,
                        format!("column '{col_name}' type mismatch: {other:?}"),
                    ));
                }
            };
            if v.is_nan() {
                return Err(row_err(
                    path,
                    rg_idx,
                    file_row,
                    format!("NaN float in column '{col_name}'"),
                ));
            }
            AttributeValue::Float(v)
        }
        AttributeType::Bool => AttributeValue::Bool(arr.as_boolean().value(row)),
        AttributeType::StringArray => {
            let list_elem = extract_list_elem(path, rg_idx, file_row, col_name, arr, row)?;
            if list_elem.null_count() > 0 {
                return Err(row_err(
                    path,
                    rg_idx,
                    file_row,
                    format!("null element in list column '{col_name}'"),
                ));
            }
            let items = match list_elem.data_type() {
                DataType::Utf8 => {
                    let s = list_elem.as_string::<i32>();
                    (0..s.len()).map(|i| s.value(i).to_owned()).collect()
                }
                DataType::LargeUtf8 => {
                    let s = list_elem.as_string::<i64>();
                    (0..s.len()).map(|i| s.value(i).to_owned()).collect()
                }
                DataType::Utf8View => {
                    let s = list_elem.as_string_view();
                    (0..s.len()).map(|i| s.value(i).to_owned()).collect()
                }
                other => {
                    return Err(row_err(
                        path,
                        rg_idx,
                        file_row,
                        format!("list column '{col_name}' element type mismatch: {other:?}"),
                    ));
                }
            };
            AttributeValue::StringArray(items)
        }
        AttributeType::IntArray => {
            let list_elem = extract_list_elem(path, rg_idx, file_row, col_name, arr, row)?;
            if list_elem.null_count() > 0 {
                return Err(row_err(
                    path,
                    rg_idx,
                    file_row,
                    format!("null element in list column '{col_name}'"),
                ));
            }
            let items = match list_elem.data_type() {
                DataType::Int8 => list_elem
                    .as_primitive::<Int8Type>()
                    .values()
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
                DataType::Int16 => list_elem
                    .as_primitive::<Int16Type>()
                    .values()
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
                DataType::Int32 => list_elem
                    .as_primitive::<Int32Type>()
                    .values()
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
                DataType::Int64 => list_elem.as_primitive::<Int64Type>().values().to_vec(),
                other => {
                    return Err(row_err(
                        path,
                        rg_idx,
                        file_row,
                        format!("list column '{col_name}' element type mismatch: {other:?}"),
                    ));
                }
            };
            AttributeValue::IntArray(items)
        }
        AttributeType::UintArray => {
            let list_elem = extract_list_elem(path, rg_idx, file_row, col_name, arr, row)?;
            if list_elem.null_count() > 0 {
                return Err(row_err(
                    path,
                    rg_idx,
                    file_row,
                    format!("null element in list column '{col_name}'"),
                ));
            }
            let items = match list_elem.data_type() {
                DataType::UInt8 => list_elem
                    .as_primitive::<UInt8Type>()
                    .values()
                    .iter()
                    .map(|&v| v as u64)
                    .collect(),
                DataType::UInt16 => list_elem
                    .as_primitive::<UInt16Type>()
                    .values()
                    .iter()
                    .map(|&v| v as u64)
                    .collect(),
                DataType::UInt32 => list_elem
                    .as_primitive::<UInt32Type>()
                    .values()
                    .iter()
                    .map(|&v| v as u64)
                    .collect(),
                DataType::UInt64 => list_elem.as_primitive::<UInt64Type>().values().to_vec(),
                other => {
                    return Err(row_err(
                        path,
                        rg_idx,
                        file_row,
                        format!("list column '{col_name}' element type mismatch: {other:?}"),
                    ));
                }
            };
            AttributeValue::UintArray(items)
        }
    };
    Ok(Some(val))
}

fn extract_list_elem(
    path: &Path,
    rg_idx: usize,
    file_row: usize,
    col_name: &str,
    arr: &ArrayRef,
    row: usize,
) -> Result<ArrayRef> {
    match arr.data_type() {
        DataType::List(_) => Ok(arr.as_list::<i32>().value(row)),
        DataType::LargeList(_) => Ok(arr.as_list::<i64>().value(row)),
        DataType::FixedSizeList(_, _) => Ok(arr.as_fixed_size_list().value(row)),
        other => Err(row_err(
            path,
            rg_idx,
            file_row,
            format!("expected list column '{col_name}', got {other:?}"),
        )),
    }
}

#[inline]
fn map_io(e: std::io::Error) -> TreeError {
    TreeError::Storage(e.to_string())
}
