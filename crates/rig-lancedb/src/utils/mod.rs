mod deserializer;

use std::sync::Arc;

use deserializer::RecordBatchDeserializer;
use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{DataType, Schema},
    query::ExecutableQuery,
};
use rig_core::vector_store::VectorStoreError;

/// Runs a LanceDB query and converts its columnar result into JSON rows.
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

/// Selects the column names to project, dropping fixed-size lists of `f64`,
/// which is how embeddings are stored.
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
