pub mod deserializer;

use deserializer::RecordBatchDeserializer;
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use rig::vector_store::VectorStoreError;

use crate::lancedb_to_rig_error;

/// Trait that facilitates the conversion of columnar data returned by a lanceDb query to serde_json::Value.
/// Used whenever a lanceDb table is queried.
pub trait Query {
    async fn execute_query(&self) -> Result<Vec<serde_json::Value>, VectorStoreError>;
}

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
