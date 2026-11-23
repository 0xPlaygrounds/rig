mod deserializer;

use arrow_array::{ArrowPrimitiveType, RecordBatch, types::Float16Type};
use arrow_json::ReaderBuilder;
use futures::TryStreamExt;
use lancedb::{
    arrow::arrow_schema::{DataType, Field, Schema, SchemaRef},
    query::{ExecutableQuery, VectorQuery},
};
use rig_core::vector_store::VectorStoreError;
use serde_json::Value;

/// Runs a LanceDB query and converts its columnar result into JSON rows.
pub(crate) async fn execute_query(query: &VectorQuery) -> Result<Vec<Value>, VectorStoreError> {
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

/// The value to name when `embedding` cannot be stored in `column` and
/// scored, or `None` when it can: a value that is not finite in the column's
/// float type, or else the largest value when the squared length is not finite.
/// Whether a finite squared length gives a finite distance depends on the
/// distance type each query picks, so search refuses the rest.
pub(crate) fn unscorable_value(column: &Field, embedding: &[f64]) -> Option<f64> {
    let narrowed = matches!(column.data_type(), DataType::FixedSizeList(item, _)
        if *item.data_type() == DataType::Float32);
    let values = || embedding.iter().copied();
    values()
        .find(|&value| !value.is_finite() || (narrowed && !(value as f32).is_finite()))
        .or_else(|| {
            let squared_length: f64 = values().map(|value| value * value).sum();
            let largest = values().max_by(|a, b| a.abs().total_cmp(&b.abs()));
            largest.filter(|_| !squared_length.is_finite())
        })
}

/// The distance LanceDB computed for `row`. Errors, naming the row's
/// `id_field` value, when it is missing or not finite, which JSON holds as null.
pub(crate) fn distance(row: &Value, id_field: &str) -> Result<f64, VectorStoreError> {
    row.get("_distance").and_then(Value::as_f64).ok_or_else(|| {
        let id = row.get(id_field).unwrap_or(&Value::Null);
        VectorStoreError::datastore(lancedb::Error::Runtime {
            message: format!("row {id} has no finite distance to the query"),
        })
    })
}

/// Why the JSON decoder cannot store `value` in a `data_type` column unchanged,
/// or `None` when it can or will refuse it itself.
///
/// The decoder panics on a non-string for `FixedSizeBinary`, truncates a
/// fraction for integer types, rounds to Float32 or, through Float32, to
/// Float16 with overflow to infinity, and reads an odd-length hex string as
/// if its last digit had a leading zero.
fn changed_value(data_type: &DataType, value: &Value) -> Option<String> {
    match (data_type, value) {
        (_, Value::Null) => None,
        (
            DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_)
            | DataType::BinaryView,
            value,
        ) => match value.as_str() {
            None => Some("binary values are hex strings".to_string()),
            Some(hex) if hex.len() % 2 == 1 => {
                Some(format!("hex string `{hex}` has an odd length"))
            }
            Some(_) => None,
        },
        (
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Duration(_),
            Value::Number(number),
        ) => number
            .as_f64()
            .filter(|n| number.is_f64() && n.fract() != 0.0)
            .map(|n| format!("{n} is not an integer")),
        (DataType::Float16 | DataType::Float32, Value::Number(number)) => {
            let n = number.as_f64()?;
            let narrowed = if *data_type == DataType::Float16 {
                <Float16Type as ArrowPrimitiveType>::Native::from_f32(n as f32).is_infinite()
            } else {
                (n as f32).is_infinite()
            };
            narrowed.then(|| format!("{n} is not finite in {data_type}"))
        }
        (
            DataType::List(item)
            | DataType::LargeList(item)
            | DataType::ListView(item)
            | DataType::LargeListView(item)
            | DataType::FixedSizeList(item, _),
            Value::Array(items),
        ) => items
            .iter()
            .find_map(|value| changed_value(item.data_type(), value)),
        (DataType::Struct(fields), Value::Object(object)) => fields.iter().find_map(|field| {
            object
                .get(field.name())
                .and_then(|value| changed_value(field.data_type(), value))
        }),
        (DataType::Map(entries, _), Value::Object(object)) => match entries.data_type() {
            DataType::Struct(fields) => fields.last().and_then(|field| {
                object
                    .values()
                    .find_map(|value| changed_value(field.data_type(), value))
            }),
            _ => None,
        },
        (DataType::RunEndEncoded(_, values), value) => changed_value(values.data_type(), value),
        _ => None,
    }
}

/// Decodes JSON rows against `schema`. Returns `Ok(None)` when there are no rows.
///
/// Errors, naming the row's `id_field` value and the column, on a value the
/// decoder would store as a different value: a number with a fraction for an
/// integer column, a finite number that overflows a Float16 or Float32 column,
/// and a binary value that is not an even-length hex string. Also errors on a
/// field the schema lacks, a missing non-nullable column, a list of the wrong
/// length, an integer out of its column's range, and a JSON type its column
/// does not accept.
pub(crate) fn rows_to_batch(
    schema: SchemaRef,
    rows: &[Value],
    id_field: &str,
) -> Result<Option<RecordBatch>, VectorStoreError> {
    for row in rows {
        for field in schema.fields() {
            let Some(reason) = row
                .get(field.name())
                .and_then(|value| changed_value(field.data_type(), value))
            else {
                continue;
            };
            let id = row.get(id_field).unwrap_or(&Value::Null);
            return Err(invalid_input(format!(
                "document {id} field `{}` would not be stored as given: {reason}",
                field.name()
            )));
        }
    }

    let decode = || {
        let mut decoder = ReaderBuilder::new(schema)
            .with_strict_mode(true)
            .build_decoder()?;
        decoder.serialize(rows)?;
        decoder.flush()
    };
    decode().map_err(VectorStoreError::datastore)
}

#[cfg(test)]
mod tests;
