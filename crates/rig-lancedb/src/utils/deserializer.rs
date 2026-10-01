use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, RecordBatch, RunArray, UnionArray,
    cast::AsArray,
    types::{
        ArrowDictionaryKeyType, BinaryType, ByteArrayType, Date32Type, Date64Type, Decimal128Type,
        DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
        IntervalDayTime, IntervalDayTimeType, IntervalMonthDayNano, IntervalMonthDayNanoType,
        IntervalYearMonthType, LargeBinaryType, LargeUtf8Type, RunEndIndexType,
        Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType,
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type, Utf8Type,
    },
};
use lancedb::arrow::arrow_schema::{ArrowError, DataType, IntervalUnit, TimeUnit};
use rig_core::vector_store::VectorStoreError;
use serde::Serialize;
use serde_json::{Value, json};

/// Converts an Arrow record batch returned by LanceDB into JSON rows.
pub(super) fn record_batch_to_json(batch: &RecordBatch) -> Result<Vec<Value>, VectorStoreError> {
    let schema = batch.schema();
    let column_names = schema
        .fields()
        .iter()
        .map(|field| field.name().as_str())
        .collect::<Vec<_>>();

    let columns = batch
        .columns()
        .iter()
        .map(type_matcher)
        .collect::<Result<Vec<_>, _>>()?;

    // A record batch is assembled exactly like a nested struct column: one
    // JSON object per row, keyed by column name.
    Ok(build_struct(&columns, batch.num_rows(), &column_names))
}

/// Converts one Arrow column into a JSON value per row, recursing through nested
/// types. Unsupported types are rejected.
fn type_matcher(column: &ArrayRef) -> Result<Vec<Value>, VectorStoreError> {
    /// Expands to a `type_matcher` arm body converting `column` to JSON values
    /// via the given helper function (e.g. `primitive_values::<Float32Type>`).
    macro_rules! json_arm {
        ($function:ident, $t:ty) => {
            $function::<$t>(column).map_err(VectorStoreError::JsonError)
        };
    }

    match column.data_type() {
        DataType::Null => Ok(vec![serde_json::Value::Null]),
        DataType::Float32 => json_arm!(primitive_values, Float32Type),
        DataType::Float64 => json_arm!(primitive_values, Float64Type),
        DataType::Int8 => json_arm!(primitive_values, Int8Type),
        DataType::Int16 => json_arm!(primitive_values, Int16Type),
        DataType::Int32 => json_arm!(primitive_values, Int32Type),
        DataType::Int64 => json_arm!(primitive_values, Int64Type),
        DataType::UInt8 => json_arm!(primitive_values, UInt8Type),
        DataType::UInt16 => json_arm!(primitive_values, UInt16Type),
        DataType::UInt32 => json_arm!(primitive_values, UInt32Type),
        DataType::UInt64 => json_arm!(primitive_values, UInt64Type),
        DataType::Date32 => json_arm!(primitive_values, Date32Type),
        DataType::Date64 => json_arm!(primitive_values, Date64Type),
        DataType::Decimal128(..) => json_arm!(primitive_values, Decimal128Type),
        DataType::Time32(TimeUnit::Second) => json_arm!(primitive_values, Time32SecondType),
        DataType::Time32(TimeUnit::Millisecond) => {
            json_arm!(primitive_values, Time32MillisecondType)
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            json_arm!(primitive_values, Time64MicrosecondType)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            json_arm!(primitive_values, Time64NanosecondType)
        }
        DataType::Timestamp(TimeUnit::Microsecond, ..) => {
            json_arm!(primitive_values, TimestampMicrosecondType)
        }
        DataType::Timestamp(TimeUnit::Millisecond, ..) => {
            json_arm!(primitive_values, TimestampMillisecondType)
        }
        DataType::Timestamp(TimeUnit::Second, ..) => {
            json_arm!(primitive_values, TimestampSecondType)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, ..) => {
            json_arm!(primitive_values, TimestampNanosecondType)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            json_arm!(primitive_values, DurationMicrosecondType)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            json_arm!(primitive_values, DurationMillisecondType)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            json_arm!(primitive_values, DurationNanosecondType)
        }
        DataType::Duration(TimeUnit::Second) => json_arm!(primitive_values, DurationSecondType),
        DataType::Interval(IntervalUnit::YearMonth) => {
            json_arm!(primitive_values, IntervalYearMonthType)
        }
        DataType::Interval(IntervalUnit::DayTime) => Ok(column
            .as_primitive::<IntervalDayTimeType>()
            .values()
            .iter()
            .map(|IntervalDayTime { days, milliseconds }| {
                json!({
                    "days": days,
                    "milliseconds": milliseconds,
                })
            })
            .collect()),
        DataType::Interval(IntervalUnit::MonthDayNano) => Ok(column
            .as_primitive::<IntervalMonthDayNanoType>()
            .values()
            .iter()
            .map(
                |IntervalMonthDayNano {
                     months,
                     days,
                     nanoseconds,
                 }| {
                    json!({
                        "months": months,
                        "days": days,
                        "nanoseconds": nanoseconds,
                    })
                },
            )
            .collect()),
        DataType::Utf8 => json_arm!(byte_values, Utf8Type),
        DataType::LargeUtf8 => json_arm!(byte_values, LargeUtf8Type),
        DataType::Binary => json_arm!(byte_values, BinaryType),
        DataType::LargeBinary => json_arm!(byte_values, LargeBinaryType),
        DataType::FixedSizeBinary(n) => (0..*n)
            .map(|i| serde_json::to_value(column.as_fixed_size_binary().value(i as usize)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(VectorStoreError::JsonError),
        DataType::Boolean => {
            let bool_array = column.as_boolean();
            (0..bool_array.len())
                .map(|i| bool_array.value(i))
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(VectorStoreError::JsonError)
        }
        DataType::FixedSizeList(..) => {
            let list_array = column.as_fixed_size_list();
            nested_values((0..list_array.len()).map(|i| list_array.value(i)))
        }
        DataType::List(..) => {
            let list_array = column.as_list::<i32>();
            nested_values((0..list_array.len()).map(|i| list_array.value(i)))
        }
        DataType::LargeList(..) => {
            let list_array = column.as_list::<i64>();
            nested_values((0..list_array.len()).map(|i| list_array.value(i)))
        }
        DataType::Struct(..) => {
            let struct_array = column.as_struct();
            let struct_columns = struct_array
                .columns()
                .iter()
                .map(type_matcher)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(build_struct(
                &struct_columns,
                struct_array.len(),
                &struct_array.column_names(),
            ))
        }
        DataType::Map(..) => {
            let map_columns = column
                .as_map()
                .entries()
                .columns()
                .iter()
                .map(type_matcher)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(build_map(&map_columns))
        }
        DataType::Dictionary(keys_type, ..) => {
            let (keys, v) = match **keys_type {
                DataType::Int8 => dictionary_keys::<Int8Type>(column)?,
                DataType::Int16 => dictionary_keys::<Int16Type>(column)?,
                DataType::Int32 => dictionary_keys::<Int32Type>(column)?,
                DataType::Int64 => dictionary_keys::<Int64Type>(column)?,
                DataType::UInt8 => dictionary_keys::<UInt8Type>(column)?,
                DataType::UInt16 => dictionary_keys::<UInt16Type>(column)?,
                DataType::UInt32 => dictionary_keys::<UInt32Type>(column)?,
                DataType::UInt64 => dictionary_keys::<UInt64Type>(column)?,
                _ => {
                    return Err(VectorStoreError::datastore(ArrowError::CastError(format!(
                        "Dictionary keys type is not accepted: {keys_type:?}"
                    ))));
                }
            };

            let values = type_matcher(v)?;

            Ok(keys
                .iter()
                .zip(values)
                .map(|(k, v)| {
                    let mut map = serde_json::Map::new();
                    map.insert(k.clone(), v);
                    map
                })
                .map(Value::Object)
                .collect())
        }
        DataType::Union(..) => match column.as_any().downcast_ref::<UnionArray>() {
            Some(union_array) => {
                nested_values((0..union_array.len()).map(|i| union_array.value(i)))
            }
            None => Err(VectorStoreError::datastore(ArrowError::CastError(format!(
                "Can't cast column {column:?} to union array"
            )))),
        },
        DataType::RunEndEncoded(index_type, ..) => match index_type.data_type() {
            DataType::Int16 => run_end_values::<Int16Type>(column),
            DataType::Int32 => run_end_values::<Int32Type>(column),
            DataType::Int64 => run_end_values::<Int64Type>(column),
            _ => Err(VectorStoreError::datastore(ArrowError::CastError(format!(
                "RunEndEncoded index type is not accepted: {index_type:?}"
            )))),
        },
        DataType::BinaryView
        | DataType::Utf8View
        | DataType::ListView(..)
        | DataType::LargeListView(..) => Err(VectorStoreError::datastore(ArrowError::CastError(
            format!("Data type: {} not yet fully supported", column.data_type()),
        ))),
        DataType::Float16 | DataType::Decimal256(..) => {
            Err(VectorStoreError::datastore(ArrowError::CastError(format!(
                "Data type: {} currently unstable",
                column.data_type()
            ))))
        }
        _ => Err(VectorStoreError::datastore(ArrowError::CastError(format!(
            "Unsupported data type: {}",
            column.data_type()
        )))),
    }
}

/// Expands a run-end encoded column into one JSON value per logical row.
fn run_end_values<T: RunEndIndexType>(column: &ArrayRef) -> Result<Vec<Value>, VectorStoreError>
where
    T::Native: Into<i64>,
{
    let Some(run_array) = column.as_any().downcast_ref::<RunArray<T>>() else {
        return Err(VectorStoreError::datastore(ArrowError::CastError(format!(
            "Can't cast array: {column:?} to list array"
        ))));
    };
    let indexes = run_array
        .run_ends()
        .values()
        .iter()
        .map(|&index| index.into())
        .collect::<Vec<i64>>();

    let mut prev = vec![0];
    prev.extend(indexes.clone());

    Ok(prev
        .iter()
        .zip(indexes)
        .map(|(prev, cur)| cur - prev)
        .zip(type_matcher(run_array.values())?)
        .flat_map(|(n, value)| vec![value; n as usize])
        .collect())
}

/// Returns a primitive Arrow column's values as JSON.
fn primitive_values<T: ArrowPrimitiveType>(
    column: &ArrayRef,
) -> Result<Vec<Value>, serde_json::Error>
where
    T::Native: Serialize,
{
    column
        .as_primitive::<T>()
        .values()
        .iter()
        .map(serde_json::to_value)
        .collect()
}

/// Returns a byte-backed Arrow column's values as JSON.
fn byte_values<T: ByteArrayType>(column: &ArrayRef) -> Result<Vec<Value>, serde_json::Error>
where
    T::Native: Serialize,
{
    let byte_array = column.as_bytes::<T>();
    (0..byte_array.len())
        .map(|i| serde_json::to_value(byte_array.value(i)))
        .collect()
}

/// Returns a dictionary-encoded column's keys, serialized as strings, alongside
/// its value array.
fn dictionary_keys<T: ArrowDictionaryKeyType>(
    column: &ArrayRef,
) -> Result<(Vec<String>, &ArrayRef), serde_json::Error>
where
    T::Native: Serialize,
{
    let dict_array = column.as_dictionary::<T>();
    let keys = dict_array
        .keys()
        .values()
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((keys, dict_array.values()))
}

/// Converts each nested array to one JSON array value.
fn nested_values(arrays: impl Iterator<Item = ArrayRef>) -> Result<Vec<Value>, VectorStoreError> {
    arrays
        .map(|array| {
            serde_json::to_value(type_matcher(&array)?).map_err(VectorStoreError::JsonError)
        })
        .collect()
}

/// Assembles one JSON object per row from per-column values, keyed by column name.
fn build_struct(columns: &[Vec<Value>], num_rows: usize, col_names: &[&str]) -> Vec<Value> {
    (0..num_rows)
        .map(|row_i| {
            columns
                .iter()
                .enumerate()
                .fold(serde_json::Map::new(), |mut acc, (col_i, col)| {
                    if let (Some(name), Some(value)) = (col_names.get(col_i), col.get(row_i)) {
                        acc.insert((*name).to_string(), value.clone());
                    }
                    acc
                })
        })
        .map(Value::Object)
        .collect()
}

/// Assembles one single-entry JSON object per map entry from its key and value
/// columns.
fn build_map(columns: &[Vec<Value>]) -> Vec<Value> {
    let (Some(keys), Some(values)) = (columns.first(), columns.get(1)) else {
        return Vec::new();
    };

    keys.iter()
        .zip(values)
        .map(|(k, v)| {
            let mut map = serde_json::Map::new();
            map.insert(
                match k {
                    serde_json::Value::String(s) => s.clone(),
                    _ => k.to_string(),
                },
                v.clone(),
            );
            map
        })
        .map(Value::Object)
        .collect()
}

#[cfg(test)]
mod tests;
