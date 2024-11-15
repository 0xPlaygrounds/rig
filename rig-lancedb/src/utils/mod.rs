mod deserializer;

use deserializer::RecordBatchDeserializer;
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use rig::vector_store::VectorStoreError;
use serde::de::Error;

use crate::lancedb_to_rig_error;

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
            .map_err(lancedb_to_rig_error)?
            .try_collect::<Vec<_>>()
            .await
            .map_err(lancedb_to_rig_error)?;

        record_batches.deserialize()
    }
}

pub(crate) trait FilterEmbeddings {
    fn filter(self, embeddings_col: Option<String>) -> serde_json::Result<serde_json::Value>;
}

impl FilterEmbeddings for serde_json::Value {
    fn filter(mut self, embeddings_col: Option<String>) -> serde_json::Result<serde_json::Value> {
        match self.as_object_mut() {
            Some(obj) => {
                obj.remove(&embeddings_col.unwrap_or("embedding".to_string()));
                serde_json::to_value(obj)
            }
            None => Err(serde_json::Error::custom(format!(
                "{} is not an object",
                self
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::FilterEmbeddings;

    #[test]
    fn test_filter_default() {
        let json = serde_json::json!({
            "id": "doc0",
            "text": "Hello world",
            "embedding": vec![0.3889, 0.6987, 0.7758, 0.7750, 0.7289, 0.3380, 0.1165, 0.1551, 0.3783, 0.1458,
            0.3060, 0.2155, 0.8966, 0.5498, 0.7419, 0.8120, 0.2306, 0.5155, 0.9947, 0.0805]
        });

        let filtered_json = json.filter(None).unwrap();

        assert_eq!(
            filtered_json,
            serde_json::json!({"id": "doc0", "text": "Hello world"})
        );
    }

    #[test]
    fn test_filter_non_default() {
        let json = serde_json::json!({
            "id": "doc0",
            "text": "Hello world",
            "vectors": vec![0.3889, 0.6987, 0.7758, 0.7750, 0.7289, 0.3380, 0.1165, 0.1551, 0.3783, 0.1458,
            0.3060, 0.2155, 0.8966, 0.5498, 0.7419, 0.8120, 0.2306, 0.5155, 0.9947, 0.0805]
        });

        let filtered_json = json.filter(Some("vectors".to_string())).unwrap();

        assert_eq!(
            filtered_json,
            serde_json::json!({"id": "doc0", "text": "Hello world"})
        );
    }
}
