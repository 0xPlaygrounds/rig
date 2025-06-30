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
use rig::vector_store::VectorStoreError;
use serde::Serialize;
use serde_json::{Value, json};

use crate::serde_to_rig_error;

fn arrow_to_rig_error(e: ArrowError) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

/// Trait used to deserialize data returned from LanceDB queries into a serde_json::Value vector.
/// Data returned by LanceDB is a vector of `RecordBatch` items.
pub(crate) trait RecordBatchDeserializer {
    fn deserialize(&self) -> Result<Vec<serde_json::Value>, VectorStoreError>;
}

impl RecordBatchDeserializer for Vec<RecordBatch> {
    fn deserialize(&self) -> Result<Vec<serde_json::Value>, VectorStoreError> {
        Ok(self
            .iter()
            .map(|record_batch| record_batch.deserialize())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect())
    }
}

/// Trait used to deserialize data returned from LanceDB queries into a serde_json::Value vector.
impl RecordBatchDeserializer for RecordBatch {
    fn deserialize(&self) -> Result<Vec<serde_json::Value>, VectorStoreError> {
        let binding = self.schema();
        let column_names = binding
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>();

        let columns = self
            .columns()
            .iter()
            .map(type_matcher)
            .collect::<Result<Vec<_>, _>>()?;

        Ok((0..self.num_rows())
            .map(|row_i| {
                columns
                    .iter()
                    .enumerate()
                    .fold(serde_json::Map::new(), |mut acc, (col_i, col)| {
                        acc.insert(column_names[col_i].to_string(), col[row_i].clone());
                        acc
                    })
            })
            .map(Value::Object)
            .collect())
    }
}

/// Recursive function that matches all possible data types store in LanceDB and converts them to serde_json::Value vector.
fn type_matcher(column: &Arc<dyn Array>) -> Result<Vec<Value>, VectorStoreError> {
    match column.data_type() {
        DataType::Null => Ok(vec![serde_json::Value::Null]),
        DataType::Float32 => column
            .to_primitive_value::<Float32Type>()
            .map_err(serde_to_rig_error),
        DataType::Float64 => column
            .to_primitive_value::<Float64Type>()
            .map_err(serde_to_rig_error),
        DataType::Int8 => column
            .to_primitive_value::<Int8Type>()
            .map_err(serde_to_rig_error),
        DataType::Int16 => column
            .to_primitive_value::<Int16Type>()
            .map_err(serde_to_rig_error),
        DataType::Int32 => column
            .to_primitive_value::<Int32Type>()
            .map_err(serde_to_rig_error),
        DataType::Int64 => column
            .to_primitive_value::<Int64Type>()
            .map_err(serde_to_rig_error),
        DataType::UInt8 => column
            .to_primitive_value::<UInt8Type>()
            .map_err(serde_to_rig_error),
        DataType::UInt16 => column
            .to_primitive_value::<UInt16Type>()
            .map_err(serde_to_rig_error),
        DataType::UInt32 => column
            .to_primitive_value::<UInt32Type>()
            .map_err(serde_to_rig_error),
        DataType::UInt64 => column
            .to_primitive_value::<UInt64Type>()
            .map_err(serde_to_rig_error),
        DataType::Date32 => column
            .to_primitive_value::<Date32Type>()
            .map_err(serde_to_rig_error),
        DataType::Date64 => column
            .to_primitive_value::<Date64Type>()
            .map_err(serde_to_rig_error),
        DataType::Decimal128(..) => column
            .to_primitive_value::<Decimal128Type>()
            .map_err(serde_to_rig_error),
        DataType::Time32(TimeUnit::Second) => column
            .to_primitive_value::<Time32SecondType>()
            .map_err(serde_to_rig_error),
        DataType::Time32(TimeUnit::Millisecond) => column
            .to_primitive_value::<Time32MillisecondType>()
            .map_err(serde_to_rig_error),
        DataType::Time64(TimeUnit::Microsecond) => column
            .to_primitive_value::<Time64MicrosecondType>()
            .map_err(serde_to_rig_error),
        DataType::Time64(TimeUnit::Nanosecond) => column
            .to_primitive_value::<Time64NanosecondType>()
            .map_err(serde_to_rig_error),
        DataType::Timestamp(TimeUnit::Microsecond, ..) => column
            .to_primitive_value::<TimestampMicrosecondType>()
            .map_err(serde_to_rig_error),
        DataType::Timestamp(TimeUnit::Millisecond, ..) => column
            .to_primitive_value::<TimestampMillisecondType>()
            .map_err(serde_to_rig_error),
        DataType::Timestamp(TimeUnit::Second, ..) => column
            .to_primitive_value::<TimestampSecondType>()
            .map_err(serde_to_rig_error),
        DataType::Timestamp(TimeUnit::Nanosecond, ..) => column
            .to_primitive_value::<TimestampNanosecondType>()
            .map_err(serde_to_rig_error),
        DataType::Duration(TimeUnit::Microsecond) => column
            .to_primitive_value::<DurationMicrosecondType>()
            .map_err(serde_to_rig_error),
        DataType::Duration(TimeUnit::Millisecond) => column
            .to_primitive_value::<DurationMillisecondType>()
            .map_err(serde_to_rig_error),
        DataType::Duration(TimeUnit::Nanosecond) => column
            .to_primitive_value::<DurationNanosecondType>()
            .map_err(serde_to_rig_error),
        DataType::Duration(TimeUnit::Second) => column
            .to_primitive_value::<DurationSecondType>()
            .map_err(serde_to_rig_error),
        DataType::Interval(IntervalUnit::YearMonth) => column
            .to_primitive_value::<IntervalYearMonthType>()
            .map_err(serde_to_rig_error),
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
        DataType::Utf8 => column
            .to_str_value::<Utf8Type>()
            .map_err(serde_to_rig_error),
        DataType::LargeUtf8 => column
            .to_str_value::<LargeUtf8Type>()
            .map_err(serde_to_rig_error),
        DataType::Binary => column
            .to_str_value::<BinaryType>()
            .map_err(serde_to_rig_error),
        DataType::LargeBinary => column
            .to_str_value::<LargeBinaryType>()
            .map_err(serde_to_rig_error),
        DataType::FixedSizeBinary(n) => (0..*n)
            .map(|i| serde_json::to_value(column.as_fixed_size_binary().value(i as usize)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde_to_rig_error),
        DataType::Boolean => {
            let bool_array = column.as_boolean();
            (0..bool_array.len())
                .map(|i| bool_array.value(i))
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
                .map_err(serde_to_rig_error)
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
                    return Err(VectorStoreError::DatastoreError(Box::new(
                        ArrowError::CastError(format!(
                            "Dictionary keys type is not accepted: {keys_type:?}"
                        )),
                    )));
                }
            };

            let values = type_matcher(v)?;

            Ok(keys
                .iter()
                .zip(values)
                .map(|(k, v)| {
                    let mut map = serde_json::Map::new();
                    map.insert(k.to_string(), v);
                    map
                })
                .map(Value::Object)
                .collect())
        }
        DataType::Union(..) => match column.as_any().downcast_ref::<UnionArray>() {
            Some(union_array) => (0..union_array.len())
                .map(|i| union_array.value(i).clone())
                .collect::<Vec<_>>()
                .iter()
                .map(type_matcher)
                .map_ok(),
            None => Err(VectorStoreError::DatastoreError(Box::new(
                ArrowError::CastError(format!("Can't cast column {column:?} to union array")),
            ))),
        },
        DataType::RunEndEncoded(index_type, ..) => {
            let items = match index_type.data_type() {
                DataType::Int16 => {
                    let (indexes, v) = column
                        .to_run_end::<Int16Type>()
                        .map_err(arrow_to_rig_error)?;

                    let mut prev = vec![0];
                    prev.extend(indexes.clone());

                    prev.iter()
                        .zip(indexes)
                        .map(|(prev, cur)| cur - prev)
                        .zip(type_matcher(&v)?)
                        .flat_map(|(n, value)| vec![value; n as usize])
                        .collect::<Vec<_>>()
                }
                DataType::Int32 => {
                    let (indexes, v) = column
                        .to_run_end::<Int32Type>()
                        .map_err(arrow_to_rig_error)?;

                    let mut prev = vec![0];
                    prev.extend(indexes.clone());

                    prev.iter()
                        .zip(indexes)
                        .map(|(prev, cur)| cur - prev)
                        .zip(type_matcher(&v)?)
                        .flat_map(|(n, value)| vec![value; n as usize])
                        .collect::<Vec<_>>()
                }
                DataType::Int64 => {
                    let (indexes, v) = column
                        .to_run_end::<Int64Type>()
                        .map_err(arrow_to_rig_error)?;

                    let mut prev = vec![0];
                    prev.extend(indexes.clone());

                    prev.iter()
                        .zip(indexes)
                        .map(|(prev, cur)| cur - prev)
                        .zip(type_matcher(&v)?)
                        .flat_map(|(n, value)| vec![value; n as usize])
                        .collect::<Vec<_>>()
                }
                _ => {
                    return Err(VectorStoreError::DatastoreError(Box::new(
                        ArrowError::CastError(format!(
                            "RunEndEncoded index type is not accepted: {index_type:?}"
                        )),
                    )));
                }
            };

            items
                .iter()
                .map(|item| serde_json::to_value(item).map_err(serde_to_rig_error))
                .collect()
        }
        DataType::BinaryView
        | DataType::Utf8View
        | DataType::ListView(..)
        | DataType::LargeListView(..) => Err(VectorStoreError::DatastoreError(Box::new(
            ArrowError::CastError(format!(
                "Data type: {} not yet fully supported",
                column.data_type()
            )),
        ))),
        DataType::Float16 | DataType::Decimal256(..) => Err(VectorStoreError::DatastoreError(
            Box::new(ArrowError::CastError(format!(
                "Data type: {} currently unstable",
                column.data_type()
            ))),
        )),
        _ => Err(VectorStoreError::DatastoreError(Box::new(
            ArrowError::CastError(format!("Unsupported data type: {}", column.data_type())),
        ))),
    }
}

// ================================================================
// Everything below includes helpers for the recursive function `type_matcher`
// ================================================================

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
            Ok(list) => serde_json::to_value(list).map_err(serde_to_rig_error),
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
                        acc.insert(col_names[col_i].to_string(), col[row_i].clone());
                        acc
                    })
            })
            .map(Value::Object)
            .collect()
    }

    fn build_map(&self) -> Vec<Value> {
        let keys = &self[0];
        let values = &self[1];

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
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, Float64Array, GenericListArray,
        Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray,
        MapArray, RecordBatch, StringArray, StructArray, UInt8Array, UInt16Array, UInt32Array,
        UInt64Array,
        builder::{
            FixedSizeListBuilder, ListBuilder, StringBuilder, StringDictionaryBuilder,
            StringRunBuilder, UnionBuilder,
        },
        types::{Float64Type, Int8Type, Int16Type, Int32Type},
    };
    use lancedb::arrow::arrow_schema::{DataType, Field, Fields};
    use serde_json::json;

    use crate::utils::deserializer::RecordBatchDeserializer;

    fn fixed_list_actors() -> FixedSizeListArray {
        let mut builder = FixedSizeListBuilder::new(StringBuilder::new(), 2);
        builder.values().append_value("Johnny Depp");
        builder.values().append_value("Cate Blanchet");
        builder.append(true);
        builder.values().append_value("Meryl Streep");
        builder.values().append_value("Scarlett Johansson");
        builder.append(true);
        builder.values().append_value("Brad Pitt");
        builder.values().append_value("Natalie Portman");
        builder.append(true);

        builder.finish()
    }

    fn name_list() -> GenericListArray<i32> {
        let mut builder = ListBuilder::new(StringBuilder::new());
        builder.values().append_value("Alice");
        builder.values().append_value("Bob");
        builder.append(true);
        builder.values().append_value("Charlie");
        builder.append(true);
        builder.values().append_value("David");
        builder.values().append_value("Eve");
        builder.values().append_value("Frank");
        builder.append(true);
        builder.finish()
    }

    fn nested_list_of_animals() -> GenericListArray<i32> {
        // [ [ [ "Dog", "Cat" ], ["Mouse"] ], [ [ "Giraffe" ], ["Cow", "Pig"] ], [ [ "Sloth" ], ["Ant", "Monkey"] ] ]
        let mut builder = ListBuilder::new(ListBuilder::new(StringBuilder::new()));
        builder
            .values()
            .append_value(vec![Some("Dog"), Some("Cat")]);
        builder.values().append_value(vec![Some("Mouse")]);
        builder.append(true);
        builder.values().append_value(vec![Some("Giraffe")]);
        builder
            .values()
            .append_value(vec![Some("Cow"), Some("Pig")]);
        builder.append(true);
        builder.values().append_value(vec![Some("Sloth")]);
        builder
            .values()
            .append_value(vec![Some("Ant"), Some("Monkey")]);
        builder.append(true);
        builder.finish()
    }

    fn movie_struct() -> StructArray {
        StructArray::from(vec![
            (
                Arc::new(Field::new("name", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec![
                    "Pulp Fiction",
                    "The Shawshank Redemption",
                    "La La Land",
                ])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("year", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![1999, 2026, 1745])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(
                    "actors",
                    DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Utf8, true)), 2),
                    false,
                )),
                Arc::new(fixed_list_actors()) as ArrayRef,
            ),
        ])
    }

    #[tokio::test]
    async fn test_primitive_deserialization() {
        let string = Arc::new(StringArray::from_iter_values(vec!["Marty", "Tony"])) as ArrayRef;
        let large_string =
            Arc::new(LargeStringArray::from_iter_values(vec!["Jerry", "Freddy"])) as ArrayRef;
        let binary = Arc::new(BinaryArray::from_iter_values(vec![b"hello", b"world"])) as ArrayRef;
        let large_binary =
            Arc::new(LargeBinaryArray::from_iter_values(vec![b"abc", b"def"])) as ArrayRef;
        let float_32 = Arc::new(Float32Array::from_iter_values(vec![0.0, 1.0])) as ArrayRef;
        let float_64 = Arc::new(Float64Array::from_iter_values(vec![0.0, 1.0])) as ArrayRef;
        let int_8 = Arc::new(Int8Array::from_iter_values(vec![0, -1])) as ArrayRef;
        let int_16 = Arc::new(Int16Array::from_iter_values(vec![-0, 1])) as ArrayRef;
        let int_32 = Arc::new(Int32Array::from_iter_values(vec![0, -1])) as ArrayRef;
        let int_64 = Arc::new(Int64Array::from_iter_values(vec![-0, 1])) as ArrayRef;
        let uint_8 = Arc::new(UInt8Array::from_iter_values(vec![0, 1])) as ArrayRef;
        let uint_16 = Arc::new(UInt16Array::from_iter_values(vec![0, 1])) as ArrayRef;
        let uint_32 = Arc::new(UInt32Array::from_iter_values(vec![0, 1])) as ArrayRef;
        let uint_64 = Arc::new(UInt64Array::from_iter_values(vec![0, 1])) as ArrayRef;

        let record_batch = RecordBatch::try_from_iter(vec![
            ("float_32", float_32),
            ("float_64", float_64),
            ("int_8", int_8),
            ("int_16", int_16),
            ("int_32", int_32),
            ("int_64", int_64),
            ("uint_8", uint_8),
            ("uint_16", uint_16),
            ("uint_32", uint_32),
            ("uint_64", uint_64),
            ("string", string),
            ("large_string", large_string),
            ("large_binary", large_binary),
            ("binary", binary),
        ])
        .unwrap();

        assert_eq!(
            record_batch.deserialize().unwrap(),
            vec![
                json!({
                    "binary": [
                        104,
                        101,
                        108,
                        108,
                        111
                    ],
                    "float_32": 0.0,
                    "float_64": 0.0,
                    "int_16": 0,
                    "int_32": 0,
                    "int_64": 0,
                    "int_8": 0,
                    "large_binary": [
                        97,
                        98,
                        99
                    ],
                    "large_string": "Jerry",
                    "string": "Marty",
                    "uint_16": 0,
                    "uint_32": 0,
                    "uint_64": 0,
                    "uint_8": 0
                }),
                json!({
                    "binary": [
                        119,
                        111,
                        114,
                        108,
                        100
                    ],
                    "float_32": 1.0,
                    "float_64": 1.0,
                    "int_16": 1,
                    "int_32": -1,
                    "int_64": 1,
                    "int_8": -1,
                    "large_binary": [
                        100,
                        101,
                        102
                    ],
                    "large_string": "Freddy",
                    "string": "Tony",
                    "uint_16": 1,
                    "uint_32": 1,
                    "uint_64": 1,
                    "uint_8": 1
                })
            ]
        )
    }

    #[tokio::test]
    async fn test_dictionary_deserialization() {
        let dictionary_values = StringArray::from(vec![None, Some("abc"), Some("def")]);

        let mut builder =
            StringDictionaryBuilder::<Int8Type>::new_with_dictionary(3, &dictionary_values)
                .unwrap();
        builder.append("def").unwrap();
        builder.append_null();
        builder.append("abc").unwrap();

        let dictionary_array = builder.finish();

        let record_batch =
            RecordBatch::try_from_iter(vec![("some_dict", Arc::new(dictionary_array) as ArrayRef)])
                .unwrap();

        assert_eq!(
            record_batch.deserialize().unwrap(),
            vec![
                json!({
                    "some_dict": {
                        "2": ""
                    }
                }),
                json!({
                    "some_dict": {
                        "0": "abc"
                    }
                }),
                json!({
                    "some_dict": {
                        "1": "def"
                    }
                })
            ]
        )
    }

    #[tokio::test]
    async fn test_union_deserialization() {
        let mut builder = UnionBuilder::new_dense();
        builder.append::<Int32Type>("type_a", 1).unwrap();
        builder.append::<Float64Type>("type_b", 3.0).unwrap();
        builder.append::<Int32Type>("type_a", 4).unwrap();
        let union = builder.build().unwrap();

        let record_batch =
            RecordBatch::try_from_iter(vec![("some_union", Arc::new(union) as ArrayRef)]).unwrap();

        assert_eq!(
            record_batch.deserialize().unwrap(),
            vec![
                json!({
                    "some_union": [
                        1
                    ]
                }),
                json!({
                    "some_union": [
                        3.0
                    ]
                }),
                json!({
                    "some_union": [
                        4
                    ]
                })
            ]
        )
    }

    #[tokio::test]
    async fn test_run_end_deserialization() {
        let mut builder = StringRunBuilder::<Int16Type>::new();

        // The builder builds the dictionary value by value
        builder.append_value("abc");
        builder.append_null();
        builder.extend([Some("def"), Some("def"), Some("abc")]);
        let array = builder.finish();

        let record_batch =
            RecordBatch::try_from_iter(vec![("some_run_end", Arc::new(array) as ArrayRef)])
                .unwrap();

        assert_eq!(
            record_batch.deserialize().unwrap(),
            vec![
                json!({
                    "some_run_end": "abc"
                }),
                json!({
                    "some_run_end": ""
                }),
                json!({
                    "some_run_end": "def"
                }),
                json!({
                    "some_run_end": "def"
                }),
                json!({
                    "some_run_end": "abc"
                })
            ]
        )
    }

    #[tokio::test]
    async fn test_map_deserialization() {
        let record_batch = RecordBatch::try_from_iter(vec![(
            "map_col",
            Arc::new(
                MapArray::new_from_strings(
                    vec!["tarentino", "darabont", "chazelle"].into_iter(),
                    &movie_struct(),
                    &[0, 1, 2],
                )
                .unwrap(),
            ) as ArrayRef,
        )])
        .unwrap();

        assert_eq!(
            record_batch.deserialize().unwrap(),
            vec![
                json!({
                    "map_col": {
                        "tarentino": {
                            "actors": [
                                "Johnny Depp",
                                "Cate Blanchet"
                            ],
                            "name": "Pulp Fiction",
                            "year": 1999
                        }
                    }
                }),
                json!({
                    "map_col": {
                        "darabont": {
                            "actors": [
                                "Meryl Streep",
                                "Scarlett Johansson"
                            ],
                            "name": "The Shawshank Redemption",
                            "year": 2026
                        }
                    }
                })
            ]
        )
    }

    #[tokio::test]
    async fn test_recursion() {
        let id_values = StringArray::from(vec!["id1", "id2", "id3"]);
        let age_values = Float32Array::from(vec![25.0, 30.5, 22.1]);
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("id", DataType::Utf8, false)),
                Arc::new(id_values) as ArrayRef,
            ),
            (
                Arc::new(Field::new("age", DataType::Float32, false)),
                Arc::new(age_values) as ArrayRef,
            ),
            (
                Arc::new(Field::new(
                    "names",
                    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                    false,
                )),
                Arc::new(name_list()) as ArrayRef,
            ),
            (
                Arc::new(Field::new(
                    "favorite_animals",
                    DataType::List(Arc::new(Field::new(
                        "item",
                        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                        true,
                    ))),
                    false,
                )),
                Arc::new(nested_list_of_animals()) as ArrayRef,
            ),
            (
                Arc::new(Field::new(
                    "favorite_movie",
                    DataType::Struct(Fields::from_iter(vec![
                        Field::new("name", DataType::Utf8, false),
                        Field::new("year", DataType::UInt32, false),
                        Field::new(
                            "actors",
                            DataType::FixedSizeList(
                                Arc::new(Field::new("item", DataType::Utf8, true)),
                                2,
                            ),
                            false,
                        ),
                    ])),
                    false,
                )),
                Arc::new(movie_struct()) as ArrayRef,
            ),
        ]);

        let record_batch =
            RecordBatch::try_from_iter(vec![("employees", Arc::new(struct_array) as ArrayRef)])
                .unwrap();

        assert_eq!(
            record_batch.deserialize().unwrap(),
            vec![
                json!({
                    "employees": {
                        "age": 25.0,
                        "favorite_animals": [
                            [
                                "Dog",
                                "Cat"
                            ],
                            [
                                "Mouse"
                            ]
                        ],
                        "favorite_movie": {
                            "actors": [
                                "Johnny Depp",
                                "Cate Blanchet"
                            ],
                            "name": "Pulp Fiction",
                            "year": 1999
                        },
                        "id": "id1",
                        "names": [
                            "Alice",
                            "Bob"
                        ]
                    }
                }),
                json!({
                    "employees": {
                        "age": 30.5,
                        "favorite_animals": [
                            [
                                "Giraffe"
                            ],
                            [
                                "Cow",
                                "Pig"
                            ]
                        ],
                        "favorite_movie": {
                            "actors": [
                                "Meryl Streep",
                                "Scarlett Johansson"
                            ],
                            "name": "The Shawshank Redemption",
                            "year": 2026
                        },
                        "id": "id2",
                        "names": [
                            "Charlie"
                        ]
                    }
                }),
                json!({
                    "employees": {
                        "age": 22.100000381469727,
                        "favorite_animals": [
                            [
                                "Sloth"
                            ],
                            [
                                "Ant",
                                "Monkey"
                            ]
                        ],
                        "favorite_movie": {
                            "actors": [
                                "Brad Pitt",
                                "Natalie Portman"
                            ],
                            "name": "La La Land",
                            "year": 1745
                        },
                        "id": "id3",
                        "names": [
                            "David",
                            "Eve",
                            "Frank"
                        ]
                    }
                })
            ]
        )
    }
}
