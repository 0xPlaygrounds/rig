use std::sync::Arc;

use arrow_array::{types::Float64Type, ArrayRef, FixedSizeListArray, RecordBatch, StringArray};
use lancedb::arrow::arrow_schema::{DataType, Field, Fields, Schema};
use rig::embeddings::builder::DocumentEmbeddings;

// Schema of table in LanceDB.
pub fn schema(dims: usize) -> Schema {
    Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float64, true)),
                dims as i32,
            ),
            false,
        ),
    ]))
}

// Convert DocumentEmbeddings objects to a RecordBatch.
pub fn as_record_batch(
    records: Vec<DocumentEmbeddings>,
    dims: usize,
) -> Result<RecordBatch, lancedb::arrow::arrow_schema::ArrowError> {
    let id = StringArray::from_iter_values(
        records
            .iter()
            .flat_map(|record| (0..record.embeddings.len()).map(|i| format!("{}-{i}", record.id)))
            .collect::<Vec<_>>(),
    );

    let content = StringArray::from_iter_values(
        records
            .iter()
            .flat_map(|record| {
                record
                    .embeddings
                    .iter()
                    .map(|embedding| embedding.document.clone())
            })
            .collect::<Vec<_>>(),
    );

    let embedding = FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
        records
            .into_iter()
            .flat_map(|record| {
                record
                    .embeddings
                    .into_iter()
                    .map(|embedding| embedding.vec.into_iter().map(Some).collect::<Vec<_>>())
                    .map(Some)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        dims as i32,
    );

    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(id) as ArrayRef),
        ("content", Arc::new(content) as ArrayRef),
        ("embedding", Arc::new(embedding) as ArrayRef),
    ])
}
