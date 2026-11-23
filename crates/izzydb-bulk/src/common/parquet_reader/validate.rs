//! Parquet schema validation and column binding.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use arrow_schema::{DataType, SchemaRef};

use izzydb::error::{Result, TreeError};
use izzydb::schema::{
    AttributeType, CollectionSchema, SchemaEntry,
};


use super::*;

pub(super) fn build_column_mapping<'a>(
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

pub(super) fn validate_id_datatype(path: &Path, id_column: &str, id_dt: &DataType) -> Result<()> {
    if !matches!(
        id_dt,
        DataType::UInt64 | DataType::Int64 | DataType::UInt32 | DataType::Int32
    ) {
        return Err(TreeError::InvalidConfig(format!(
            "parquet file {} id column '{id_column}' has unsupported Arrow type {id_dt:?}",
            path.display()
        )));
    }
    Ok(())
}

pub(super) fn validate_and_bind_file_schema(
    path: &Path,
    arrow_schema: &SchemaRef,
    id_column: &str,
    vec_cfg: Option<(usize, &str)>,
    col_to_schema: &HashMap<String, &SchemaEntry>,
) -> Result<(Vec<usize>, Vec<AttrColumnBinding>)> {
    let id_idx = arrow_schema.index_of(id_column).map_err(|_| {
        TreeError::InvalidConfig(format!(
            "parquet file {} missing id column '{id_column}'",
            path.display()
        ))
    })?;
    validate_id_datatype(path, id_column, arrow_schema.field(id_idx).data_type())?;

    let mut projected_indices = vec![id_idx];
    let vec_col_name = if let Some((dim, vector_column)) = vec_cfg {
        let vec_idx = arrow_schema.index_of(vector_column).map_err(|_| {
            TreeError::InvalidConfig(format!(
                "parquet file {} missing vector column '{vector_column}'",
                path.display()
            ))
        })?;
        let vec_dt = arrow_schema.field(vec_idx).data_type();
        validate_vector_datatype(path, vector_column, vec_dt, dim)?;
        projected_indices.push(vec_idx);
        Some(vector_column)
    } else {
        None
    };

    let mut bindings = Vec::new();

    for (idx, field) in arrow_schema.fields().iter().enumerate() {
        let name = field.name();
        if name == id_column || vec_col_name == Some(name.as_str()) {
            continue;
        }
        if let Some(entry) = col_to_schema.get(name) {
            validate_attr_datatype(path, name, field.data_type(), entry.attr_type)?;
            projected_indices.push(idx);
            bindings.push(AttrColumnBinding {
                parquet_col: name.clone(),
                schema_attr_name: entry.name.clone(),
                attr_id: entry.id,
                attr_type: entry.attr_type,
                storage: entry.storage,
            });
        }
    }

    projected_indices.sort_unstable();
    projected_indices.dedup();
    Ok((projected_indices, bindings))
}

pub(super) fn validate_vector_datatype(
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

pub(super) fn validate_attr_datatype(
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
