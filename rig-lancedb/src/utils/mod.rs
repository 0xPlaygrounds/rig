mod deserializer;

use std::sync::Arc;

use deserializer::RecordBatchDeserializer;
use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{DataType, Schema},
    query::ExecutableQuery,
};
use rig::vector_store::VectorStoreError;

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
                    _ => Some(field.name().to_string()),
                },
                _ => Some(field.name().to_string()),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lancedb::arrow::arrow_schema::{DataType, Field, Schema};

    use super::FilterTableColumns;

    #[tokio::test]
    async fn test_column_filtering() {
        let field_a = Field::new("id", DataType::Int64, false);
        let field_b = Field::new("my_bool", DataType::Boolean, false);
        let field_c = Field::new(
            "my_embeddings",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 10),
            false,
        );
        let field_d = Field::new(
            "my_list",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 10),
            false,
        );

        let schema = Schema::new(vec![field_a, field_b, field_c, field_d]);

        let columns = Arc::new(schema).filter_embeddings();

        assert_eq!(
            columns,
            vec![
                "id".to_string(),
                "my_bool".to_string(),
                "my_list".to_string()
            ]
        )
    }

    #[tokio::test]
    async fn test_column_filtering_2() {
        let field_a = Field::new("id", DataType::Int64, false);
        let field_b = Field::new("my_bool", DataType::Boolean, false);
        let field_c = Field::new(
            "my_embeddings",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 10),
            false,
        );
        let field_d = Field::new(
            "my_other_embeddings",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 10),
            false,
        );

        let schema = Schema::new(vec![field_a, field_b, field_c, field_d]);

        let columns = Arc::new(schema).filter_embeddings();

        assert_eq!(columns, vec!["id".to_string(), "my_bool".to_string()])
    }
}
