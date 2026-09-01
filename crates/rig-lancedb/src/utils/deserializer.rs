use std::sync::Arc;

use arrow_array::{
    Array, ArrowPrimitiveType, OffsetSizeTrait, RecordBatch, RunArray, StructArray, UnionArray,
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

/// Trait used to deserialize data returned from LanceDB queries into a serde_json::Value vector.
/// Data returned by LanceDB is a vector of `RecordBatch` items.
pub(crate) trait RecordBatchDeserializer {
    fn deserialize(&self) -> Result<Vec<serde_json::Value>, VectorStoreError>;
}

impl RecordBatchDeserializer for Vec<RecordBatch> {
    fn deserialize(&self) -> Result<Vec<serde_json::Value>, VectorStoreError> {
        Ok(self
            .iter()
            .map(RecordBatchDeserializer::deserialize)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect())
    }
}

/// Trait used to deserialize data returned from LanceDB queries into a serde_json::Value vector.
impl RecordBatchDeserializer for RecordBatch {
    fn deserialize(&self) -> Result<Vec<serde_json::Value>, VectorStoreError> {
        let schema = self.schema();
        let column_names = schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>();

        let columns = self
            .columns()
            .iter()
            .map(type_matcher)
            .collect::<Result<Vec<_>, _>>()?;

        // A record batch is assembled exactly like a nested struct column: one
        // JSON object per row, keyed by column name.
        Ok(columns.build_struct(self.num_rows(), column_names))
    }
}

/// Recursive function that matches all possible data types store in LanceDB and converts them to serde_json::Value vector.
fn type_matcher(column: &Arc<dyn Array>) -> Result<Vec<Value>, VectorStoreError> {
    /// Expands to a `type_matcher` arm body converting `column` to JSON values
    /// via the given helper method (e.g. `to_primitive_value::<Float32Type>`).
    macro_rules! json_arm {
        ($method:ident, $t:ty) => {
            column.$method::<$t>().map_err(VectorStoreError::JsonError)
        };
    }

    match column.data_type() {
        DataType::Null => Ok(vec![serde_json::Value::Null]),
        DataType::Float32 => json_arm!(to_primitive_value, Float32Type),
        DataType::Float64 => json_arm!(to_primitive_value, Float64Type),
        DataType::Int8 => json_arm!(to_primitive_value, Int8Type),
        DataType::Int16 => json_arm!(to_primitive_value, Int16Type),
        DataType::Int32 => json_arm!(to_primitive_value, Int32Type),
        DataType::Int64 => json_arm!(to_primitive_value, Int64Type),
        DataType::UInt8 => json_arm!(to_primitive_value, UInt8Type),
        DataType::UInt16 => json_arm!(to_primitive_value, UInt16Type),
        DataType::UInt32 => json_arm!(to_primitive_value, UInt32Type),
        DataType::UInt64 => json_arm!(to_primitive_value, UInt64Type),
        DataType::Date32 => json_arm!(to_primitive_value, Date32Type),
        DataType::Date64 => json_arm!(to_primitive_value, Date64Type),
        DataType::Decimal128(..) => json_arm!(to_primitive_value, Decimal128Type),
        DataType::Time32(TimeUnit::Second) => json_arm!(to_primitive_value, Time32SecondType),
        DataType::Time32(TimeUnit::Millisecond) => {
            json_arm!(to_primitive_value, Time32MillisecondType)
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            json_arm!(to_primitive_value, Time64MicrosecondType)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            json_arm!(to_primitive_value, Time64NanosecondType)
        }
        DataType::Timestamp(TimeUnit::Microsecond, ..) => {
            json_arm!(to_primitive_value, TimestampMicrosecondType)
        }
        DataType::Timestamp(TimeUnit::Millisecond, ..) => {
            json_arm!(to_primitive_value, TimestampMillisecondType)
        }
        DataType::Timestamp(TimeUnit::Second, ..) => {
            json_arm!(to_primitive_value, TimestampSecondType)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, ..) => {
            json_arm!(to_primitive_value, TimestampNanosecondType)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            json_arm!(to_primitive_value, DurationMicrosecondType)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            json_arm!(to_primitive_value, DurationMillisecondType)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            json_arm!(to_primitive_value, DurationNanosecondType)
        }
        DataType::Duration(TimeUnit::Second) => json_arm!(to_primitive_value, DurationSecondType),
        DataType::Interval(IntervalUnit::YearMonth) => {
            json_arm!(to_primitive_value, IntervalYearMonthType)
        }
        DataType::Interval(IntervalUnit::DayTime) => Ok(column
            .to_primitive::<IntervalDayTimeType>()
            .iter()
            .map(|IntervalDayTime { days, milliseconds }| {
                json!({
                    "days": days,
                    "milliseconds": milliseconds,
                })
            })
            .collect()),
        DataType::Interval(IntervalUnit::MonthDayNano) => Ok(column
            .to_primitive::<IntervalMonthDayNanoType>()
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
        DataType::Utf8 => json_arm!(to_str_value, Utf8Type),
        DataType::LargeUtf8 => json_arm!(to_str_value, LargeUtf8Type),
        DataType::Binary => json_arm!(to_str_value, BinaryType),
        DataType::LargeBinary => json_arm!(to_str_value, LargeBinaryType),
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
        DataType::FixedSizeList(..) => column.to_fixed_lists().iter().map(type_matcher).map_ok(),
        DataType::List(..) => column.to_list::<i32>().iter().map(type_matcher).map_ok(),
        DataType::LargeList(..) => column.to_list::<i64>().iter().map(type_matcher).map_ok(),
        DataType::Struct(..) => {
            let struct_array = column.as_struct();
            let struct_columns = struct_array
                .inner_lists()
                .iter()
                .map(type_matcher)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(struct_columns.build_struct(struct_array.num_rows(), struct_array.column_names()))
        }
        DataType::Map(..) => {
            let map_columns = column
                .as_map()
                .entries()
                .inner_lists()
                .iter()
                .map(type_matcher)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(map_columns.build_map())
        }
        DataType::Dictionary(keys_type, ..) => {
            let (keys, v) = match **keys_type {
                DataType::Int8 => column.to_dict_values::<Int8Type>()?,
                DataType::Int16 => column.to_dict_values::<Int16Type>()?,
                DataType::Int32 => column.to_dict_values::<Int32Type>()?,
                DataType::Int64 => column.to_dict_values::<Int64Type>()?,
                DataType::UInt8 => column.to_dict_values::<UInt8Type>()?,
                DataType::UInt16 => column.to_dict_values::<UInt16Type>()?,
                DataType::UInt32 => column.to_dict_values::<UInt32Type>()?,
                DataType::UInt64 => column.to_dict_values::<UInt64Type>()?,
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
            Some(union_array) => (0..union_array.len())
                .map(|i| union_array.value(i))
                .collect::<Vec<_>>()
                .iter()
                .map(type_matcher)
                .map_ok(),
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

// ================================================================
// Everything below includes helpers for the recursive function `type_matcher`
// ================================================================

/// Expands a run-end encoded column into one JSON value per logical row.
fn run_end_values<T: RunEndIndexType>(
    column: &Arc<dyn Array>,
) -> Result<Vec<Value>, VectorStoreError>
where
    T::Native: Into<i64>,
{
    let (indexes, v) = column
        .to_run_end::<T>()
        .map_err(VectorStoreError::datastore)?;
    let indexes = indexes.into_iter().map(Into::into).collect::<Vec<i64>>();

    let mut prev = vec![0];
    prev.extend(indexes.clone());

    Ok(prev
        .iter()
        .zip(indexes)
        .map(|(prev, cur)| cur - prev)
        .zip(type_matcher(&v)?)
        .flat_map(|(n, value)| vec![value; n as usize])
        .collect())
}

/// Trait used to "deserialize" an arrow_array::Array as as list of primitive objects.
trait DeserializePrimitiveArray {
    /// Downcast arrow Array into a `PrimitiveArray` with items that implement trait `ArrowPrimitiveType`.
    /// Return the primitive array values.
    fn to_primitive<T: ArrowPrimitiveType>(&self) -> Vec<<T as ArrowPrimitiveType>::Native>;

    /// Same as above but convert the resulting array values into serde_json::Value.
    fn to_primitive_value<T: ArrowPrimitiveType>(&self) -> Result<Vec<Value>, serde_json::Error>
    where
        <T as ArrowPrimitiveType>::Native: Serialize;
}

impl DeserializePrimitiveArray for &Arc<dyn Array> {
    fn to_primitive<T: ArrowPrimitiveType>(&self) -> Vec<<T as ArrowPrimitiveType>::Native> {
        let primitive_array = self.as_primitive::<T>();

        (0..primitive_array.len())
            .map(|i| primitive_array.value(i))
            .collect()
    }

    fn to_primitive_value<T: ArrowPrimitiveType>(&self) -> Result<Vec<Value>, serde_json::Error>
    where
        <T as ArrowPrimitiveType>::Native: Serialize,
    {
        self.to_primitive::<T>()
            .iter()
            .map(serde_json::to_value)
            .collect()
    }
}

/// Trait used to "deserialize" an arrow_array::Array as as list of str objects.
trait DeserializeByteArray {
    /// Downcast arrow Array into a `GenericByteArray` with items that implement trait `ByteArrayType`.
    /// Return the generic byte array values.
    fn to_str<T: ByteArrayType>(&self) -> Vec<&<T as ByteArrayType>::Native>;

    /// Same as above but convert the resulting array values into serde_json::Value.
    fn to_str_value<T: ByteArrayType>(&self) -> Result<Vec<Value>, serde_json::Error>
    where
        <T as ByteArrayType>::Native: Serialize;
}

impl DeserializeByteArray for &Arc<dyn Array> {
    fn to_str<T: ByteArrayType>(&self) -> Vec<&<T as ByteArrayType>::Native> {
        let byte_array = self.as_bytes::<T>();
        (0..byte_array.len()).map(|j| byte_array.value(j)).collect()
    }

    fn to_str_value<T: ByteArrayType>(&self) -> Result<Vec<Value>, serde_json::Error>
    where
        <T as ByteArrayType>::Native: Serialize,
    {
        self.to_str::<T>()
            .iter()
            .map(serde_json::to_value)
            .collect()
    }
}

/// Trait used to "deserialize" an arrow_array::Array as a list of list objects.
trait DeserializeListArray {
    /// Downcast arrow Array into a `GenericListArray` with items that implement trait `OffsetSizeTrait`.
    /// Return the generic list array values.
    fn to_list<T: OffsetSizeTrait>(&self) -> Vec<Arc<dyn arrow_array::Array>>;
}

impl DeserializeListArray for &Arc<dyn Array> {
    fn to_list<T: OffsetSizeTrait>(&self) -> Vec<Arc<dyn arrow_array::Array>> {
        (0..self.as_list::<T>().len())
            .map(|j| self.as_list::<T>().value(j))
            .collect()
    }
}

/// Trait used to "deserialize" an arrow_array::Array as a list of dict objects.
trait DeserializeDictArray {
    /// Downcast arrow Array into a `DictionaryArray` with items that implement trait `ArrowDictionaryKeyType`.
    /// Return the dictionary keys and values as a tuple.
    fn to_dict<T: ArrowDictionaryKeyType>(
        &self,
    ) -> (
        Vec<<T as ArrowPrimitiveType>::Native>,
        &Arc<dyn arrow_array::Array>,
    );

    fn to_dict_values<T: ArrowDictionaryKeyType>(
        &self,
    ) -> Result<(Vec<String>, &Arc<dyn arrow_array::Array>), serde_json::Error>
    where
        <T as ArrowPrimitiveType>::Native: Serialize;
}

impl DeserializeDictArray for &Arc<dyn Array> {
    fn to_dict<T: ArrowDictionaryKeyType>(
        &self,
    ) -> (
        Vec<<T as ArrowPrimitiveType>::Native>,
        &Arc<dyn arrow_array::Array>,
    ) {
        let dict_array = self.as_dictionary::<T>();
        (
            (0..dict_array.keys().len())
                .map(|i| dict_array.keys().value(i))
                .collect(),
            dict_array.values(),
        )
    }

    fn to_dict_values<T: ArrowDictionaryKeyType>(
        &self,
    ) -> Result<(Vec<String>, &Arc<dyn arrow_array::Array>), serde_json::Error>
    where
        <T as ArrowPrimitiveType>::Native: Serialize,
    {
        let (k, v) = self.to_dict::<T>();

        Ok((
            k.iter()
                .map(serde_json::to_string)
                .collect::<Result<Vec<_>, _>>()?,
            v,
        ))
    }
}

/// Trait used to "deserialize" an arrow_array::Array as as list of fixed size list objects.
trait DeserializeArray {
    /// Downcast arrow Array into a `FixedSizeListArray`.
    /// Return the fixed size list array values.
    fn to_fixed_lists(&self) -> Vec<Arc<dyn arrow_array::Array>>;
}

impl DeserializeArray for &Arc<dyn Array> {
    fn to_fixed_lists(&self) -> Vec<Arc<dyn arrow_array::Array>> {
        let list_array = self.as_fixed_size_list();

        (0..list_array.len()).map(|i| list_array.value(i)).collect()
    }
}

type RunArrayParts<T> = (
    Vec<<T as ArrowPrimitiveType>::Native>,
    Arc<dyn arrow_array::Array>,
);

/// Trait used to "deserialize" an arrow_array::Array as a list of list objects.
trait DeserializeRunArray {
    /// Downcast arrow Array into a `GenericListArray` with items that implement trait `RunEndIndexType`.
    /// Return the generic list array values.
    fn to_run_end<T: RunEndIndexType>(&self) -> Result<RunArrayParts<T>, ArrowError>;
}

impl DeserializeRunArray for &Arc<dyn Array> {
    fn to_run_end<T: RunEndIndexType>(&self) -> Result<RunArrayParts<T>, ArrowError> {
        if let Some(run_array) = self.as_any().downcast_ref::<RunArray<T>>() {
            return Ok((
                run_array.run_ends().values().to_vec(),
                run_array.values().clone(),
            ));
        }
        Err(ArrowError::CastError(format!(
            "Can't cast array: {self:?} to list array"
        )))
    }
}

trait DeserializeStructArray {
    fn inner_lists(&self) -> Vec<Arc<dyn arrow_array::Array>>;

    fn num_rows(&self) -> usize;
}

impl DeserializeStructArray for StructArray {
    fn inner_lists(&self) -> Vec<Arc<dyn arrow_array::Array>> {
        (0..self.num_columns())
            .map(|j| self.column(j).clone())
            .collect::<Vec<_>>()
    }

    fn num_rows(&self) -> usize {
        self.column(0).into_data().len()
    }
}

trait MapOk {
    fn map_ok(self) -> Result<Vec<Value>, VectorStoreError>;
}

impl<I> MapOk for I
where
    I: Iterator<Item = Result<Vec<Value>, VectorStoreError>>,
{
    fn map_ok(self) -> Result<Vec<Value>, VectorStoreError> {
        self.map(|maybe_list| match maybe_list {
            Ok(list) => serde_json::to_value(list).map_err(VectorStoreError::JsonError),
            Err(e) => Err(e),
        })
        .collect::<Result<Vec<_>, _>>()
    }
}

trait RebuildObject {
    fn build_struct(&self, num_rows: usize, col_names: Vec<&str>) -> Vec<Value>;

    fn build_map(&self) -> Vec<Value>;
}

impl RebuildObject for Vec<Vec<Value>> {
    fn build_struct(&self, num_rows: usize, col_names: Vec<&str>) -> Vec<Value> {
        (0..num_rows)
            .map(|row_i| {
                self.iter()
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

    fn build_map(&self) -> Vec<Value> {
        let Some(keys) = self.first() else {
            return Vec::new();
        };
        let Some(values) = self.get(1) else {
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
}

#[cfg(test)]
mod tests;
