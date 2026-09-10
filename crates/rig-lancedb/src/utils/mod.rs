mod deserializer;

use std::sync::Arc;

use deserializer::RecordBatchDeserializer;
use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{DataType, Schema},
    query::ExecutableQuery,
};
use rig_core::vector_store::VectorStoreError;

/// Trait that facilitates the conversion of columnar data returned by a lanceDb query to serde_json::Value.
/// Used whenever a lanceDb table is queried.
pub(crate) trait QueryToJson {
    async fn execute_query(&self) -> Result<Vec<serde_json::Value>, VectorStoreError>;
}

impl QueryToJson for lancedb::query::VectorQuery {
    async fn execute_query(&self) -> Result<Vec<serde_json::Value>, VectorStoreError> {
        let record_batches = self
            .execute()
            .await
            .map_err(VectorStoreError::datastore)?
            .try_collect::<Vec<_>>()
            .await
            .map_err(VectorStoreError::datastore)?;

        record_batches.deserialize()
    }
}

/// Filter out the columns from a table that do not include embeddings. Return the vector of column names.
pub(crate) trait FilterTableColumns {
    fn filter_embeddings(self) -> Vec<String>;
}

impl FilterTableColumns for Arc<Schema> {
    fn filter_embeddings(self) -> Vec<String> {
        self.fields()
            .iter()
            .filter_map(|field| match field.data_type() {
                DataType::FixedSizeList(inner, ..) => match inner.data_type() {
                    DataType::Float64 => None,
                    _ => Some(field.name().clone()),
                },
                _ => Some(field.name().clone()),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
