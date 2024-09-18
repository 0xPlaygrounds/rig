use arrow_array::{Array, Float64Array, ListArray, RecordBatch, StringArray};
use futures::TryStreamExt;
use lancedb::{arrow::arrow_schema::ArrowError, query::ExecutableQuery};
use rig::vector_store::VectorStoreError;

use crate::lancedb_to_rig_error;

pub trait DeserializeArrow {
    fn deserialize_str_column(&self, i: usize) -> Result<Vec<&str>, ArrowError>;
    fn deserialize_list_column(&self, i: usize) -> Result<Vec<Vec<f64>>, ArrowError>;
}

impl DeserializeArrow for RecordBatch {
    fn deserialize_str_column(&self, i: usize) -> Result<Vec<&str>, ArrowError> {
        let column = self.column(i);
        match column.as_any().downcast_ref::<StringArray>() {
            Some(str_array) => Ok((0..str_array.len())
                .map(|j| str_array.value(j))
                .collect::<Vec<_>>()),
            None => Err(ArrowError::CastError(format!(
                "Can't cast column {i} to string array"
            ))),
        }
    }

    fn deserialize_list_column(&self, i: usize) -> Result<Vec<Vec<f64>>, ArrowError> {
        let column = self.column(i);
        match column.as_any().downcast_ref::<ListArray>() {
            Some(list_array) => (0..list_array.len())
                .map(
                    |j| match list_array.value(j).as_any().downcast_ref::<Float64Array>() {
                        Some(float_array) => Ok((0..float_array.len())
                            .map(|k| float_array.value(k))
                            .collect::<Vec<_>>()),
                        None => Err(ArrowError::CastError(format!(
                            "Can't cast value at index {j} to float array"
                        ))),
                    },
                )
                .collect::<Result<Vec<_>, _>>(),
            None => Err(ArrowError::CastError(format!(
                "Can't cast column {i} to list array"
            ))),
        }
    }
}

pub trait Query<T>
where
    T: TryFrom<Vec<RecordBatch>, Error = VectorStoreError>,
{
    async fn execute_query(&self) -> Result<T, VectorStoreError>;
}

impl<T> Query<T> for lancedb::query::Query
where
    T: TryFrom<Vec<RecordBatch>, Error = VectorStoreError>,
{
    async fn execute_query(&self) -> Result<T, VectorStoreError> {
        let record_batches = self
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?
            .try_collect::<Vec<_>>()
            .await
            .map_err(lancedb_to_rig_error)?;

        T::try_from(record_batches)
    }
}
