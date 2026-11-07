mod deserializer;

use arrow_array::RecordBatch;
use arrow_json::ReaderBuilder;
use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef},
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

/// Wraps a refused input as a datastore error.
pub(crate) fn invalid_input(message: String) -> VectorStoreError {
    VectorStoreError::datastore(lancedb::Error::InvalidInput { message })
}

/// Selects the column embeddings are inserted into: `requested` if given, else
/// the only fixed-size list of floats. Errors when `requested` is not such a
/// column, or when there is no such column or more than one to choose from.
pub(crate) fn embedding_column<'a>(
    schema: &'a Schema,
    requested: Option<&str>,
) -> Result<&'a Field, VectorStoreError> {
    let is_embedding = |field: &Field| {
        matches!(field.data_type(), DataType::FixedSizeList(inner, _)
            if matches!(inner.data_type(), DataType::Float32 | DataType::Float64))
    };
    if let Some(column) = requested {
        return match schema.field_with_name(column) {
            Ok(field) if is_embedding(field) => Ok(field),
            _ => Err(invalid_input(format!(
                "column `{column}` is not a fixed-size list of floats in this table"
            ))),
        };
    }
    let columns = schema
        .fields()
        .iter()
        .filter(|field| is_embedding(field))
        .collect::<Vec<_>>();
    match columns.as_slice() {
        &[column] => Ok(column.as_ref()),
        [] => Err(invalid_input(
            "the table has no fixed-size list of floats to hold embeddings".to_string(),
        )),
        _ => {
            let names = columns.iter().map(|c| c.name()).collect::<Vec<_>>();
            Err(invalid_input(format!(
                "the table has several embedding columns, {names:?}: name one with SearchParams::column"
            )))
        }
    }
}

/// Whether every value of `embedding` stays finite in the float type of `column`.
pub(crate) fn fits_column(column: &Field, embedding: &[f64]) -> bool {
    let narrowed = matches!(column.data_type(), DataType::FixedSizeList(item, _)
        if *item.data_type() == DataType::Float32);
    embedding.iter().all(|&value| {
        if narrowed {
            (value as f32).is_finite()
        } else {
            value.is_finite()
        }
    })
}

/// Whether `value` holds a non-string where `data_type` expects binary. The JSON
/// decoder panics on this for `FixedSizeBinary` and errors for the other binary
/// types, so refusing all of them here gives one error and no panic.
fn has_non_string_binary(data_type: &DataType, value: &serde_json::Value) -> bool {
    use serde_json::Value;

    match (data_type, value) {
        (_, Value::Null) => false,
        (
            DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_)
            | DataType::BinaryView,
            value,
        ) => !value.is_string(),
        (
            DataType::List(item)
            | DataType::LargeList(item)
            | DataType::ListView(item)
            | DataType::LargeListView(item)
            | DataType::FixedSizeList(item, _),
            Value::Array(items),
        ) => items
            .iter()
            .any(|value| has_non_string_binary(item.data_type(), value)),
        (DataType::Struct(fields), Value::Object(object)) => fields.iter().any(|field| {
            object
                .get(field.name())
                .is_some_and(|value| has_non_string_binary(field.data_type(), value))
        }),
        (DataType::Map(entries, _), Value::Object(object)) => match entries.data_type() {
            DataType::Struct(fields) => fields.last().is_some_and(|field| {
                object
                    .values()
                    .any(|value| has_non_string_binary(field.data_type(), value))
            }),
            _ => false,
        },
        (DataType::RunEndEncoded(_, values), value) => {
            has_non_string_binary(values.data_type(), value)
        }
        _ => false,
    }
}

/// Decodes JSON rows against `schema`. Returns `None` when there are no rows.
///
/// Errors on a field the schema lacks, a missing non-nullable column, a list of
/// the wrong length, a binary value that is not a hex string, and a JSON type
/// its column does not accept. Other values convert as the decoder chooses: `1.5`
/// in an integer column stores `1`, `1e300` in a Float32 column stores infinity,
/// and the odd-length hex string `abc` stores the bytes `ab0c`.
pub(crate) fn rows_to_batch(
    schema: SchemaRef,
    rows: &[serde_json::Value],
) -> Result<Option<RecordBatch>, ArrowError> {
    for field in schema.fields() {
        let refused = rows.iter().any(|row| {
            row.get(field.name())
                .is_some_and(|value| has_non_string_binary(field.data_type(), value))
        });
        if refused {
            return Err(ArrowError::JsonError(format!(
                "column `{}` takes binary values as hex strings",
                field.name()
            )));
        }
    }

    let mut decoder = ReaderBuilder::new(schema)
        .with_strict_mode(true)
        .build_decoder()?;
    decoder.serialize(rows)?;
    decoder.flush()
}

#[cfg(test)]
mod tests;
