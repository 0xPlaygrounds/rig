use std::sync::Arc;

use arrow_array::{Array, Float64Array, ListArray, RecordBatch, RecordBatchIterator, StringArray};
use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{ArrowError, Schema},
    query::ExecutableQuery,
};
use rig::vector_store::VectorStoreError;

use crate::lancedb_to_rig_error;

/// Trait used to "deserialize" a column of a RecordBatch object into a list o primitive types
pub trait DeserializeArrow {
    /// Define the column number that contains strings, i.
    /// For each item in the column, convert it to a string and collect the result in a vector of strings.
    fn deserialize_str_column(&self, i: usize) -> Result<Vec<&str>, ArrowError>;
    /// Define the column number that contains the list of floats, i.
    /// For each item in the column, convert it to a list and for each item in the list, convert it to a float.
    /// Collect the result as a vector of vectors of floats.
    fn deserialize_float_list_column(&self, i: usize) -> Result<Vec<Vec<f64>>, ArrowError>;
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

    fn deserialize_float_list_column(&self, i: usize) -> Result<Vec<Vec<f64>>, ArrowError> {
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

/// Trait that facilitates the conversion of columnar data returned by a lanceDb query to the desired struct.
/// Used whenever a lanceDb table is queried.
/// First, execute the query and get the result as a list of RecordBatches (columnar data).
/// Then, convert the record batches to the desired type using the try_from trait.
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

/// Same as the above trait but for the VectorQuery type.
/// Used whenever a lanceDb table vector search is executed.
impl<T> Query<T> for lancedb::query::VectorQuery
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

/// Trait that facilitate inserting data defined as Rust structs into lanceDB table which contains columnar data.
pub trait Insert<T> {
    async fn insert(&self, data: T, schema: Schema) -> Result<(), lancedb::Error>;
}

impl<T: Into<Vec<Result<RecordBatch, ArrowError>>>> Insert<T> for lancedb::Table {
    async fn insert(&self, data: T, schema: Schema) -> Result<(), lancedb::Error> {
        self.add(RecordBatchIterator::new(data.into(), Arc::new(schema)))
            .execute()
            .await?;

        Ok(())
    }
}
