mod deserializer;

use arrow_array::RecordBatch;
use arrow_json::ReaderBuilder;
use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef},
    query::{ExecutableQuery, VectorQuery},
};
use rig_core::vector_store::VectorStoreError;
use serde_json::{Map, Value};

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

/// Names the column embeddings are written to: the schema's only fixed-size
/// list of `f32` or `f64`, or the one named `configured`.
pub(crate) fn embedding_column(
    schema: &Schema,
    configured: Option<&str>,
) -> Result<String, VectorStoreError> {
    let mut candidates = schema
        .fields()
        .iter()
        .filter(|field| is_float_list(field) && configured.is_none_or(|name| field.name() == name));
    match (candidates.next(), candidates.next(), configured) {
        (Some(field), None, _) => Ok(field.name().clone()),
        (None, _, Some(name)) => Err(schema_error(format!(
            "no fixed-size float list column named `{name}`"
        ))),
        (None, _, None) => Err(schema_error(
            "no fixed-size float list column to hold embeddings".to_string(),
        )),
        (Some(_), Some(_), _) => Err(schema_error(
            "several fixed-size float list columns; name one with `SearchParams::column`"
                .to_string(),
        )),
    }
}

fn is_float_list(field: &Field) -> bool {
    matches!(
        field.data_type(),
        DataType::FixedSizeList(item, _)
            if matches!(item.data_type(), DataType::Float32 | DataType::Float64)
    )
}

/// A table row for one embedding of `document`: an object's own fields, or any
/// other value under `document`, with the embedding under `column`.
pub(crate) fn embedded_row(document: &Value, column: &str, embedding: Vec<f64>) -> Value {
    let mut row = match document {
        Value::Object(fields) => fields.clone(),
        other => Map::from_iter([("document".to_string(), other.clone())]),
    };
    row.insert(column.to_string(), embedding.into());
    Value::Object(row)
}

/// Decodes JSON object rows into one batch of `schema`, or `None` without rows.
/// Fields the schema lacks are dropped and absent columns are null.
pub(crate) fn rows_to_batch(
    schema: SchemaRef,
    rows: &[Value],
) -> Result<Option<RecordBatch>, VectorStoreError> {
    let mut decoder = ReaderBuilder::new(schema)
        .build_decoder()
        .map_err(VectorStoreError::datastore)?;
    decoder
        .serialize(rows)
        .map_err(VectorStoreError::datastore)?;
    decoder.flush().map_err(VectorStoreError::datastore)
}

fn schema_error(message: String) -> VectorStoreError {
    VectorStoreError::datastore(ArrowError::SchemaError(message))
}

#[cfg(test)]
mod tests;
