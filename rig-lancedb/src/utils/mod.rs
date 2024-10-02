pub mod deserializer;
use std::sync::Arc;

use arrow_array::{
    types::ByteArrayType, Array, ArrowPrimitiveType, FixedSizeListArray, GenericByteArray,
    PrimitiveArray, RecordBatch, RecordBatchIterator,
};
use deserializer::RecordBatchDeserializer;
use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{ArrowError, Schema},
    query::ExecutableQuery,
};
use rig::vector_store::VectorStoreError;

use crate::lancedb_to_rig_error;

/// Trait used to "deserialize" an arrow_array::Array as as list of primitive objects.
pub trait DeserializePrimitiveArray {
    fn to_float<T: ArrowPrimitiveType>(
        &self,
    ) -> Result<Vec<<T as ArrowPrimitiveType>::Native>, ArrowError>;
}

impl DeserializePrimitiveArray for &Arc<dyn Array> {
    fn to_float<T: ArrowPrimitiveType>(
        &self,
    ) -> Result<Vec<<T as ArrowPrimitiveType>::Native>, ArrowError> {
        match self.as_any().downcast_ref::<PrimitiveArray<T>>() {
            Some(array) => Ok((0..array.len()).map(|j| array.value(j)).collect::<Vec<_>>()),
            None => Err(ArrowError::CastError(format!(
                "Can't cast array: {self:?} to float array"
            ))),
        }
    }
}

/// Trait used to "deserialize" an arrow_array::Array as as list of byte objects.
pub trait DeserializeByteArray {
    fn to_str<T: ByteArrayType>(&self) -> Result<Vec<&<T as ByteArrayType>::Native>, ArrowError>;
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
}

/// Trait used to "deserialize" an arrow_array::Array as as list of lists of primitive objects.
pub trait DeserializeListArray {
    fn to_float_list<T: ArrowPrimitiveType>(
        &self,
    ) -> Result<Vec<Vec<<T as ArrowPrimitiveType>::Native>>, ArrowError>;
}

impl DeserializeListArray for &Arc<dyn Array> {
    fn to_float_list<T: ArrowPrimitiveType>(
        &self,
    ) -> Result<Vec<Vec<<T as ArrowPrimitiveType>::Native>>, ArrowError> {
        match self.as_any().downcast_ref::<FixedSizeListArray>() {
            Some(list_array) => (0..list_array.len())
                .map(|j| (&list_array.value(j)).to_float::<T>())
                .collect::<Result<Vec<_>, _>>(),
            None => Err(ArrowError::CastError(format!(
                "Can't cast column {self:?} to fixed size list array"
            ))),
        }
    }
}

/// Trait that facilitates the conversion of columnar data returned by a lanceDb query to the desired struct.
/// Used whenever a lanceDb table is queried.
/// First, execute the query and get the result as a list of RecordBatches (columnar data).
/// Then, convert the record batches to the desired type using the try_from trait.
pub trait Query {
    async fn execute_query(&self) -> Result<Vec<serde_json::Value>, VectorStoreError>;
}

/// Same as the above trait but for the VectorQuery type.
/// Used whenever a lanceDb table vector search is executed.
impl Query for lancedb::query::VectorQuery {
    async fn execute_query(&self) -> Result<Vec<serde_json::Value>, VectorStoreError> {
        let record_batches = self
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?
            .try_collect::<Vec<_>>()
            .await
            .map_err(lancedb_to_rig_error)?;

        record_batches.deserialize()
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
