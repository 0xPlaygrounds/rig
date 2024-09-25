use std::sync::Arc;

use arrow_array::{
    types::{
        BinaryType, ByteArrayType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
        DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
        Int8Type, IntervalDayTimeType, IntervalMonthDayNanoType, IntervalYearMonthType,
        LargeBinaryType, LargeUtf8Type, Time32MillisecondType, Time32SecondType,
        Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt16Type,
        UInt32Type, UInt64Type, UInt8Type, Utf8Type,
    },
    Array, ArrowPrimitiveType, FixedSizeBinaryArray, FixedSizeListArray, GenericByteArray,
    GenericListArray, OffsetSizeTrait, PrimitiveArray, RecordBatch, StructArray,
};
use lancedb::arrow::arrow_schema::{ArrowError, DataType, IntervalUnit, TimeUnit};
use rig::vector_store::VectorStoreError;
use serde::Serialize;
use serde_json::{json, Value};

use crate::serde_to_rig_error;

fn arrow_to_rig_error(e: ArrowError) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

trait Test {
    fn deserialize(&self) -> Result<serde_json::Value, VectorStoreError>;
}

impl Test for RecordBatch {
    fn deserialize(&self) -> Result<serde_json::Value, VectorStoreError> {
        fn type_matcher(column: &Arc<dyn Array>) -> Result<Vec<Value>, VectorStoreError> {
            match column.data_type() {
                DataType::Null => Ok(vec![serde_json::Value::Null]),
                // f16 does not implement serde_json::Deserialize. Need to cast to f32.
                DataType::Float16 => column
                    .to_primitive::<Float16Type>()
                    .map_err(arrow_to_rig_error)?
                    .iter()
                    .map(|float_16| serde_json::to_value(float_16.to_f32()))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(serde_to_rig_error),
                DataType::Float32 => column.to_primitive_value::<Float32Type>(),
                DataType::Float64 => column.to_primitive_value::<Float64Type>(),
                DataType::Int8 => column.to_primitive_value::<Int8Type>(),
                DataType::Int16 => column.to_primitive_value::<Int16Type>(),
                DataType::Int32 => column.to_primitive_value::<Int32Type>(),
                DataType::Int64 => column.to_primitive_value::<Int64Type>(),
                DataType::UInt8 => column.to_primitive_value::<UInt8Type>(),
                DataType::UInt16 => column.to_primitive_value::<UInt16Type>(),
                DataType::UInt32 => column.to_primitive_value::<UInt32Type>(),
                DataType::UInt64 => column.to_primitive_value::<UInt64Type>(),
                DataType::Date32 => column.to_primitive_value::<Date32Type>(),
                DataType::Date64 => column.to_primitive_value::<Date64Type>(),
                DataType::Decimal128(..) => column.to_primitive_value::<Decimal128Type>(),
                // i256 does not implement serde_json::Deserialize. Need to cast to i128.
                DataType::Decimal256(..) => column
                    .to_primitive::<Decimal256Type>()
                    .map_err(arrow_to_rig_error)?
                    .iter()
                    .map(|dec_256| serde_json::to_value(dec_256.as_i128()))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(serde_to_rig_error),
                DataType::Time32(TimeUnit::Second) => {
                    column.to_primitive_value::<Time32SecondType>()
                }
                DataType::Time32(TimeUnit::Millisecond) => {
                    column.to_primitive_value::<Time32MillisecondType>()
                }
                DataType::Time64(TimeUnit::Microsecond) => {
                    column.to_primitive_value::<Time64MicrosecondType>()
                }
                DataType::Time64(TimeUnit::Nanosecond) => {
                    column.to_primitive_value::<Time64NanosecondType>()
                }
                DataType::Timestamp(TimeUnit::Microsecond, ..) => {
                    column.to_primitive_value::<TimestampMicrosecondType>()
                }
                DataType::Timestamp(TimeUnit::Millisecond, ..) => {
                    column.to_primitive_value::<TimestampMillisecondType>()
                }
                DataType::Timestamp(TimeUnit::Second, ..) => {
                    column.to_primitive_value::<TimestampSecondType>()
                }
                DataType::Timestamp(TimeUnit::Nanosecond, ..) => {
                    column.to_primitive_value::<TimestampNanosecondType>()
                }
                DataType::Duration(TimeUnit::Microsecond) => {
                    column.to_primitive_value::<DurationMicrosecondType>()
                }
                DataType::Duration(TimeUnit::Millisecond) => {
                    column.to_primitive_value::<DurationMillisecondType>()
                }
                DataType::Duration(TimeUnit::Nanosecond) => {
                    column.to_primitive_value::<DurationNanosecondType>()
                }
                DataType::Duration(TimeUnit::Second) => {
                    column.to_primitive_value::<DurationSecondType>()
                }
                DataType::Interval(IntervalUnit::DayTime) => Ok(column
                    .to_primitive::<IntervalDayTimeType>()
                    .map_err(arrow_to_rig_error)?
                    .iter()
                    .map(|interval| {
                        json!({
                            "days": interval.days,
                            "milliseconds": interval.milliseconds,
                        })
                    })
                    .collect()),
                DataType::Interval(IntervalUnit::MonthDayNano) => Ok(column
                    .to_primitive::<IntervalMonthDayNanoType>()
                    .map_err(arrow_to_rig_error)?
                    .iter()
                    .map(|interval| {
                        json!({
                            "months": interval.months,
                            "days": interval.days,
                            "nanoseconds": interval.nanoseconds,
                        })
                    })
                    .collect()),
                DataType::Interval(IntervalUnit::YearMonth) => {
                    column.to_primitive_value::<IntervalYearMonthType>()
                }
                DataType::Utf8 | DataType::Utf8View => column.to_str_value::<Utf8Type>(),
                DataType::LargeUtf8 => column.to_str_value::<LargeUtf8Type>(),
                DataType::Binary => column.to_str_value::<BinaryType>(),
                DataType::LargeBinary => column.to_str_value::<LargeBinaryType>(),
                DataType::FixedSizeBinary(n) => {
                    match column.as_any().downcast_ref::<FixedSizeBinaryArray>() {
                        Some(list_array) => (0..*n)
                            .map(|j| serde_json::to_value(list_array.value(j as usize)))
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(serde_to_rig_error),
                        None => Err(VectorStoreError::DatastoreError(Box::new(
                            ArrowError::CastError(format!(
                                "Can't cast column {column:?} to fixed size list array"
                            )),
                        ))),
                    }
                }
                DataType::FixedSizeList(..) => column
                    .fixed_nested_lists()
                    .map_err(arrow_to_rig_error)?
                    .iter()
                    .map(|nested_list| type_matcher(nested_list))
                    .map_ok(),
                DataType::List(..) | DataType::ListView(..) => column
                    .nested_lists::<i32>()
                    .map_err(arrow_to_rig_error)?
                    .iter()
                    .map(|nested_list| type_matcher(nested_list))
                    .map_ok(),
                DataType::LargeList(..) | DataType::LargeListView(..) => column
                    .nested_lists::<i64>()
                    .map_err(arrow_to_rig_error)?
                    .iter()
                    .map(|nested_list| type_matcher(nested_list))
                    .map_ok(),
                DataType::Struct(..) => match column.as_any().downcast_ref::<StructArray>() {
                    Some(struct_array) => struct_array
                    .nested_lists()
                    .iter()
                    .map(|nested_list| type_matcher(nested_list))
                    .map_ok(),
                    None => Err(VectorStoreError::DatastoreError(Box::new(
                        ArrowError::CastError(format!(
                            "Can't cast array: {column:?} to struct array"
                        )),
                    ))),
                },
                // DataType::Map(..) => {
                //     let item = match column.as_any().downcast_ref::<MapArray>() {
                //         Some(map_array) => map_array
                //             .entries()
                //             .nested_lists()
                //             .iter()
                //             .map(|nested_list| type_matcher(nested_list, nested_list.data_type()))
                //             .collect::<Result<Vec<_>, _>>(),
                //         None => Err(VectorStoreError::DatastoreError(Box::new(
                //             ArrowError::CastError(format!(
                //                 "Can't cast array: {column:?} to map array"
                //             )),
                //         ))),
                //     }?;
                // }
                // DataType::Dictionary(key_data_type, value_data_type) => {
                //     let item = match column.as_any().downcast_ref::<AnyDictionaryArray>() {
                //         Some(map_array) => {
                //             let keys = &Arc::new(map_array.keys());
                //             type_matcher(keys, keys.data_type())
                //         }
                //         None => Err(ArrowError::CastError(format!(
                //             "Can't cast array: {column:?} to map array"
                //         ))),
                //     }?;
                // },
                _ => {
                    println!("Unsupported data type");
                    Ok(vec![serde_json::Value::Null])
                }
            }
        }

        let columns = self
            .columns()
            .iter()
            .map(type_matcher)
            .collect::<Result<Vec<_>, _>>()?;

        println!("{:?}", serde_json::to_string(&columns).unwrap());

        Ok(json!({}))
    }
}

/// Trait used to "deserialize" an arrow_array::Array as as list of primitive objects.
pub trait DeserializePrimitiveArray {
    fn to_primitive<T: ArrowPrimitiveType>(
        &self,
    ) -> Result<Vec<<T as ArrowPrimitiveType>::Native>, ArrowError>;

    fn to_primitive_value<T: ArrowPrimitiveType>(&self) -> Result<Vec<Value>, VectorStoreError>
    where
        <T as ArrowPrimitiveType>::Native: Serialize;
}

impl DeserializePrimitiveArray for &Arc<dyn Array> {
    fn to_primitive<T: ArrowPrimitiveType>(
        &self,
    ) -> Result<Vec<<T as ArrowPrimitiveType>::Native>, ArrowError> {
        match self.as_any().downcast_ref::<PrimitiveArray<T>>() {
            Some(array) => Ok((0..array.len()).map(|j| array.value(j)).collect::<Vec<_>>()),
            None => Err(ArrowError::CastError(format!(
                "Can't cast array: {self:?} to float array"
            ))),
        }
    }

    fn to_primitive_value<T: ArrowPrimitiveType>(&self) -> Result<Vec<Value>, VectorStoreError>
    where
        <T as ArrowPrimitiveType>::Native: Serialize,
    {
        self.to_primitive::<T>()
            .map_err(arrow_to_rig_error)?
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde_to_rig_error)
    }
}

/// Trait used to "deserialize" an arrow_array::Array as as list of byte objects.
pub trait DeserializeByteArray {
    fn to_str<T: ByteArrayType>(&self) -> Result<Vec<&<T as ByteArrayType>::Native>, ArrowError>;

    fn to_str_value<T: ByteArrayType>(&self) -> Result<Vec<Value>, VectorStoreError>
    where
        <T as ByteArrayType>::Native: Serialize;
}

impl DeserializeByteArray for &Arc<dyn Array> {
    fn to_str<T: ByteArrayType>(&self) -> Result<Vec<&<T as ByteArrayType>::Native>, ArrowError> {
        match self.as_any().downcast_ref::<GenericByteArray<T>>() {
            Some(array) => Ok((0..array.len()).map(|j| array.value(j)).collect::<Vec<_>>()),
            None => Err(ArrowError::CastError(format!(
                "Can't cast array: {self:?} to float array"
            ))),
        }
    }

    fn to_str_value<T: ByteArrayType>(&self) -> Result<Vec<Value>, VectorStoreError>
    where
        <T as ByteArrayType>::Native: Serialize,
    {
        self.to_str::<T>()
            .map_err(arrow_to_rig_error)?
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde_to_rig_error)
    }
}

/// Trait used to "deserialize" an arrow_array::Array as as list of list objects.
trait DeserializeListArray {
    fn nested_lists<T: OffsetSizeTrait>(
        &self,
    ) -> Result<Vec<Arc<dyn arrow_array::Array>>, ArrowError>;
}

impl DeserializeListArray for &Arc<dyn Array> {
    fn nested_lists<T: OffsetSizeTrait>(
        &self,
    ) -> Result<Vec<Arc<dyn arrow_array::Array>>, ArrowError> {
        match self.as_any().downcast_ref::<GenericListArray<T>>() {
            Some(array) => Ok((0..array.len()).map(|j| array.value(j)).collect::<Vec<_>>()),
            None => Err(ArrowError::CastError(format!(
                "Can't cast array: {self:?} to float array"
            ))),
        }
    }
}

/// Trait used to "deserialize" an arrow_array::Array as as list of list objects.
trait DeserializeArray {
    fn fixed_nested_lists(&self) -> Result<Vec<Arc<dyn arrow_array::Array>>, ArrowError>;
}

impl DeserializeArray for &Arc<dyn Array> {
    fn fixed_nested_lists(&self) -> Result<Vec<Arc<dyn arrow_array::Array>>, ArrowError> {
        match self.as_any().downcast_ref::<FixedSizeListArray>() {
            Some(list_array) => Ok((0..list_array.len())
                .map(|j| list_array.value(j as usize))
                .collect::<Vec<_>>()),
            None => {
                return Err(ArrowError::CastError(format!(
                    "Can't cast column {self:?} to fixed size list array"
                )));
            }
        }
    }
}

trait DeserializeStructArray {
    fn nested_lists(&self) -> Vec<Arc<dyn arrow_array::Array>>;
}

impl DeserializeStructArray for StructArray {
    fn nested_lists(&self) -> Vec<Arc<dyn arrow_array::Array>> {
        (0..self.len())
            .map(|j| self.column(j).clone())
            .collect::<Vec<_>>()
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        builder::{FixedSizeListBuilder, ListBuilder, StringBuilder, StructBuilder}, ArrayRef, BinaryArray, FixedSizeBinaryArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray, LargeStringArray, ListArray, RecordBatch, StringArray, StructArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array
    };
    use lancedb::arrow::arrow_schema::{DataType, Field};

    use crate::utils::deserializer::Test;

    #[tokio::test]
    async fn test_primitive_deserialization() {
        let string = Arc::new(StringArray::from_iter_values(vec!["Marty", "Tony"])) as ArrayRef;
        let large_string =
            Arc::new(LargeStringArray::from_iter_values(vec!["Jerry", "Freddy"])) as ArrayRef;
        let binary = Arc::new(BinaryArray::from_iter_values(vec![b"hello", b"world"])) as ArrayRef;
        let large_binary = Arc::new(LargeBinaryArray::from_iter_values(vec![
            b"The bright sun sets behind the mountains, casting gold",
            b"A gentle breeze rustles through the trees at twilight.",
        ])) as ArrayRef;
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

        let _t = record_batch.deserialize().unwrap();

        assert!(false)
    }

    #[tokio::test]
    async fn test_list_recursion() {
        let mut builder = FixedSizeListBuilder::new(StringBuilder::new(), 3);
        builder.values().append_value("Hi");
        builder.values().append_value("Hey");
        builder.values().append_value("What's up");
        builder.append(true);
        builder.values().append_value("Bye");
        builder.values().append_value("Seeya");
        builder.values().append_value("Later");
        builder.append(true);

        let record_batch = RecordBatch::try_from_iter(vec![(
            "salutations",
            Arc::new(builder.finish()) as ArrayRef,
        )])
        .unwrap();

        let _t = record_batch.deserialize().unwrap();

        assert!(false)
    }

    #[tokio::test]
    async fn test_list_recursion_2() {
        let mut builder = ListBuilder::new(ListBuilder::new(StringBuilder::new()));
        builder
            .values()
            .append_value(vec![Some("Dog"), Some("Cat")]);
        builder
            .values()
            .append_value(vec![Some("Mouse"), Some("Bird")]);
        builder.append(true);
        builder
            .values()
            .append_value(vec![Some("Giraffe"), Some("Mammoth")]);
        builder
            .values()
            .append_value(vec![Some("Cow"), Some("Pig")]);

        let record_batch =
            RecordBatch::try_from_iter(vec![("animals", Arc::new(builder.finish()) as ArrayRef)])
                .unwrap();

        let _t = record_batch.deserialize().unwrap();

        assert!(false)
    }

    #[tokio::test]
    async fn test_struct() {
        let id_values = StringArray::from(vec!["id1", "id2", "id3"]);

        let age_values = Float32Array::from(vec![25.0, 30.5, 22.1]);

        let mut names_builder = ListBuilder::new(StringBuilder::new());
        names_builder.values().append_value("Alice");
        names_builder.values().append_value("Bob");
        names_builder.append(true);
        names_builder.values().append_value("Charlie");
        names_builder.append(true);
        names_builder.values().append_value("David");
        names_builder.values().append_value("Eve");
        names_builder.values().append_value("Frank");
        names_builder.append(true);

        let names_array = names_builder.finish();

        // Step 4: Combine into a StructArray
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
                Arc::new(names_array) as ArrayRef,
            ),
        ]);

        let record_batch =
            RecordBatch::try_from_iter(vec![("employees", Arc::new(struct_array) as ArrayRef)])
                .unwrap();

        let _t = record_batch.deserialize().unwrap();

        assert!(false)
    }
}
