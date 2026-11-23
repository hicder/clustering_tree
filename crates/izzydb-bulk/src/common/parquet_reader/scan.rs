//! Parquet scanning and staging of documents/detached columns.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use arrow_array::{ArrayRef, RecordBatch};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;

use izzydb::config::{AttrId, DistanceMetric, DocId};
use izzydb::error::{Result, TreeError};
use izzydb::metric::prepare_vector;
use izzydb::schema::{
    AttributeType, AttributeValue, Attributes, CollectionSchema, ColumnStorage, SchemaEntry,
};

use crate::DuplicatePolicy;
use crate::common::sorter::ExternalSorter;

use super::*;

pub(super) struct RowGroupTask {
    pub(super) file_idx: usize,
    pub(super) rg_idx: usize,
    pub(super) global_row_start: u32,
    pub(super) row_in_file_start: usize,
    pub(super) projected_root_indices: Vec<usize>,
    pub(super) attr_bindings: Vec<AttrColumnBinding>,
}

#[derive(Clone)]
pub(super) struct AttrColumnBinding {
    pub(super) parquet_col: String,
    pub(super) schema_attr_name: String,
    pub(super) attr_id: AttrId,
    pub(super) attr_type: AttributeType,
    pub(super) storage: ColumnStorage,
}

#[allow(clippy::too_many_arguments)]
pub fn scan_and_stage(
    parquet_files: &[PathBuf],
    schema: &CollectionSchema,
    vec_cfg: Option<(usize, DistanceMetric, &str)>,
    id_column: &str,
    column_map: Option<&[(String, String)]>,
    on_duplicate_id: DuplicatePolicy,
    stage_dir: &Path,
    memory_budget_bytes: usize,
    pool: &rayon::ThreadPool,
) -> Result<StagedData> {
    fs::create_dir_all(stage_dir).map_err(|e| {
        TreeError::Storage(format!("create staging dir {}: {e}", stage_dir.display()))
    })?;

    let schema_has_detached = schema
        .entries()
        .iter()
        .any(|e| e.storage == ColumnStorage::Detached);
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
            vec_cfg.map(|(dim, _, col)| (dim, col)),
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
            has_detached_staging: schema_has_detached,
        });
    }

    let vectors_file = if let Some((dim, _, _)) = vec_cfg {
        let f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&vectors_path)
            .map_err(map_io)?;
        f.set_len(total_rows * (dim as u64) * 4).map_err(map_io)?;
        Some(f)
    } else {
        File::create(&vectors_path).map_err(map_io)?;
        None
    };

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
            let mask = ProjectionMask::roots(
                builder.parquet_schema(),
                task.projected_root_indices.iter().copied(),
            );
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
                    schema_has_detached,
                    vec_cfg,
                    id_column,
                    &task.attr_bindings,
                    vectors_file.as_ref(),
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
                Some(next) if next.doc_id == cur.doc_id => match on_duplicate_id {
                    DuplicatePolicy::Error => {
                        return Err(TreeError::InvalidConfig(format!(
                            "duplicate doc_id {} encountered during bulk import",
                            cur.doc_id
                        )));
                    }
                    DuplicatePolicy::LastWins => {
                        duplicates_dropped += 1;
                        cur = next;
                    }
                },
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
        has_detached_staging: schema_has_detached,
    })
}

/// Read selected detached columns from `parquet_files` as `(DocId, Vec<AttributeValue>)` pairs
/// aligned with `target_columns` order (substituting `AttributeValue::Null` for missing/null cells).
pub fn scan_detached_from_parquet(
    parquet_files: &[PathBuf],
    schema: &CollectionSchema,
    id_column: &str,
    target_columns: &[String],
    column_map: Option<&[(String, String)]>,
) -> Result<Vec<(DocId, Vec<AttributeValue>)>> {
    if target_columns.is_empty() {
        return Err(TreeError::InvalidSchema(
            "detached column import requires at least one column".into(),
        ));
    }

    let mut target_entries: Vec<&SchemaEntry> = Vec::with_capacity(target_columns.len());
    let mut target_set: HashSet<&str> = HashSet::new();
    for name in target_columns {
        let entry = schema.get_by_name(name).ok_or_else(|| {
            TreeError::InvalidSchema(format!("unknown detached attribute '{name}'"))
        })?;
        if entry.storage != ColumnStorage::Detached {
            return Err(TreeError::InvalidSchema(format!(
                "attribute '{name}' is Inline, not Detached"
            )));
        }
        if !target_set.insert(name.as_str()) {
            return Err(TreeError::InvalidSchema(format!(
                "duplicate detached column '{name}'"
            )));
        }
        target_entries.push(entry);
    }

    // Build reverse lookup: schema_attr -> parquet_col name (defaults to identity).
    let mut schema_to_parquet: HashMap<&str, &str> = HashMap::new();
    if let Some(pairs) = column_map {
        for (pq_col, schema_attr) in pairs {
            if target_set.contains(schema_attr.as_str()) {
                schema_to_parquet.insert(schema_attr.as_str(), pq_col.as_str());
            }
        }
    }

    let mut out = Vec::new();

    for path in parquet_files {
        let file = File::open(path).map_err(|e| {
            TreeError::Storage(format!("open parquet file {}: {e}", path.display()))
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            TreeError::CorruptedData(format!("read parquet metadata {}: {e}", path.display()))
        })?;
        let arrow_schema = builder.schema().clone();

        let id_idx = arrow_schema.index_of(id_column).map_err(|_| {
            TreeError::InvalidConfig(format!(
                "parquet file {} missing id column '{id_column}'",
                path.display()
            ))
        })?;
        validate_id_datatype(path, id_column, arrow_schema.field(id_idx).data_type())?;

        let mut projected_indices = vec![id_idx];
        let mut col_pq_names: Vec<Option<String>> = Vec::with_capacity(target_entries.len());
        for entry in &target_entries {
            let pq_name = schema_to_parquet
                .get(entry.name.as_str())
                .copied()
                .unwrap_or(entry.name.as_str());
            if let Ok(idx) = arrow_schema.index_of(pq_name) {
                validate_attr_datatype(
                    path,
                    pq_name,
                    arrow_schema.field(idx).data_type(),
                    entry.attr_type,
                )?;
                projected_indices.push(idx);
                col_pq_names.push(Some(pq_name.to_owned()));
            } else {
                col_pq_names.push(None);
            }
        }
        projected_indices.sort_unstable();
        projected_indices.dedup();

        let mask = ProjectionMask::roots(builder.parquet_schema(), projected_indices);
        let reader = builder.with_projection(mask).with_batch_size(8192).build().map_err(|e| {
            TreeError::CorruptedData(format!("build parquet reader {}: {e}", path.display()))
        })?;

        let mut file_row = 0usize;
        for batch_res in reader {
            let batch = batch_res.map_err(|e| {
                TreeError::CorruptedData(format!("read parquet batch {}: {e}", path.display()))
            })?;
            let id_col = batch.column_by_name(id_column).unwrap();
            let col_arrs: Vec<Option<(&String, &ArrayRef)>> = col_pq_names
                .iter()
                .map(|opt| opt.as_ref().map(|n| (n, batch.column_by_name(n).unwrap())))
                .collect();

            for row in 0..batch.num_rows() {
                let doc_id = decode_doc_id(path, 0, file_row + row, id_col, row)?;
                let mut vals = Vec::with_capacity(target_entries.len());
                for (entry, arr_opt) in target_entries.iter().zip(&col_arrs) {
                    let val = match arr_opt {
                        Some((pq_name, arr)) => decode_attr_value(
                            path,
                            0,
                            file_row + row,
                            pq_name,
                            arr,
                            row,
                            entry.attr_type,
                        )?
                        .unwrap_or(AttributeValue::Null),
                        None => AttributeValue::Null,
                    };
                    vals.push(val);
                }
                out.push((doc_id, vals));
            }
            file_row += batch.num_rows();
        }
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn process_record_batch(
    path: &Path,
    rg_idx: usize,
    file_row_base: usize,
    global_row_base: u32,
    batch: &RecordBatch,
    schema: &CollectionSchema,
    schema_has_detached: bool,
    vec_cfg: Option<(usize, DistanceMetric, &str)>,
    id_column: &str,
    attr_bindings: &[AttrColumnBinding],
    vectors_file: Option<&File>,
    doc_ids_file: &File,
    docs_idx_file: &File,
    doc_writer: &mut DocShardWriter,
    sorter_shard: &mut crate::common::sorter::SorterShard<DocRowPair>,
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
    let vec_col = match vec_cfg {
        Some((_, _, vector_column)) => Some(
            batch.column_by_name(vector_column).ok_or_else(|| {
                TreeError::CorruptedData(format!(
                    "missing projected vector column '{vector_column}' in {}",
                    path.display()
                ))
            })?,
        ),
        None => None,
    };

    let attr_cols: Vec<(&AttrColumnBinding, &ArrayRef)> = attr_bindings
        .iter()
        .map(|b| {
            let arr = batch.column_by_name(&b.parquet_col).unwrap();
            (b, arr)
        })
        .collect();

    let dim = vec_cfg.map_or(0, |(d, _, _)| d);
    let mut vec_buf = vec![0.0f32; dim];
    let mut batch_vecs_bytes = Vec::with_capacity(n * dim * 4);
    let mut batch_doc_ids_bytes = Vec::with_capacity(n * 8);
    let mut batch_idx_bytes = Vec::with_capacity(n * DocIndexEntry::SIZE);

    for row in 0..n {
        let file_row = file_row_base + row;
        let global_row = global_row_base + row as u32;

        let doc_id = decode_doc_id(path, rg_idx, file_row, id_col, row)?;
        if let (Some((dim, metric, vector_column)), Some(vc)) = (vec_cfg, vec_col) {
            decode_vector_into(path, rg_idx, file_row, vc, row, dim, &mut vec_buf)?;
            let prepared = prepare_vector(&vec_buf, dim, metric).map_err(|e| {
                TreeError::InvalidAttribute {
                    name: vector_column.to_owned(),
                    reason: format!("{}:{rg_idx}:{file_row}: {e}", path.display()),
                }
            })?;
            for &f in &prepared {
                batch_vecs_bytes.extend_from_slice(&f.to_le_bytes());
            }
        }

        let mut attrs = Attributes::new();
        for &(binding, arr) in &attr_cols {
            let _ = (binding.attr_id, binding.storage);
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

        let doc_bytes = encode_staged_doc(
            schema,
            &attrs,
            vec_cfg.is_some(),
            schema_has_detached,
        )?;
        let offset = doc_writer.offset;
        let len = doc_bytes.len() as u32;
        doc_writer.writer.write_all(&doc_bytes).map_err(map_io)?;
        doc_writer.offset += len as u64;

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

    if let Some(vf) = vectors_file {
        vf.write_all_at(&batch_vecs_bytes, (global_row_base as u64) * (dim as u64) * 4)
            .map_err(map_io)?;
    }
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
