mod deserializer;

use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{DataType, Schema},
    query::{ExecutableQuery, VectorQuery},
};
use rig_core::vector_store::VectorStoreError;

/// Runs a LanceDB query and converts its columnar result into JSON rows.
pub(crate) async fn execute_query(
    query: &VectorQuery,
) -> Result<Vec<serde_json::Value>, VectorStoreError> {
    let record_batches = query
        .execute()
        .await
        .map_err(VectorStoreError::datastore)?
        .try_collect::<Vec<_>>()
        .await
        .map_err(VectorStoreError::datastore)?;

    let mut rows = Vec::new();
    for batch in &record_batches {
        rows.extend(deserializer::record_batch_to_json(batch)?);
    }
    Ok(rows)
}

/// Selects the column names to project, dropping fixed-size lists of `f64`,
/// which is how embeddings are stored.
pub(crate) fn filter_embeddings(schema: &Schema) -> Vec<String> {
    schema
        .fields()
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

#[cfg(test)]
mod tests;
