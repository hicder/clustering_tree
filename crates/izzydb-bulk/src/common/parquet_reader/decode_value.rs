//! Arrow value decoding into documents, vectors, and attributes.

use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;

use izzydb::config::DocId;
use izzydb::error::{Result, TreeError};
use izzydb::schema::{
    AttributeType, AttributeValue,
};



pub(super) fn row_err(path: &Path, rg_idx: usize, row: usize, msg: impl std::fmt::Display) -> TreeError {
    TreeError::InvalidConfig(format!("{}:rg={rg_idx}:row={row}: {msg}", path.display()))
}

pub(super) fn decode_doc_id(
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

pub(super) fn decode_vector_into(
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

pub(super) fn decode_attr_value(
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

pub(super) fn extract_list_elem(
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
pub fn read_doc_id_at(doc_ids_bytes: &[u8], row: u32) -> DocId {
    let off = (row as usize) * 8;
    DocId::from_le_bytes(doc_ids_bytes[off..off + 8].try_into().unwrap())
}

#[inline]
pub(super) fn map_io(e: std::io::Error) -> TreeError {
    TreeError::Storage(e.to_string())
}
