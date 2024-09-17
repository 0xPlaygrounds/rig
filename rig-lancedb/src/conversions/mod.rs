use std::sync::Arc;

use arrow_array::{
    builder::{Float64Builder, ListBuilder},
    RecordBatch, StringArray,
};
use lancedb::arrow::arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
use rig::embeddings::DocumentEmbeddings;

pub fn document_schema() -> Schema {
    Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("document", DataType::Utf8, false),
    ]))
}

pub fn embedding_schema() -> Schema {
    Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("document_id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::List(Arc::new(Field::new("float", DataType::Float64, false))),
            false,
        ),
    ]))
}

pub fn document_records(documents: &Vec<DocumentEmbeddings>) -> Result<RecordBatch, ArrowError> {
    let id = StringArray::from_iter_values(documents.iter().map(|doc| doc.id.clone()));
    let document = StringArray::from_iter_values(
        documents
            .iter()
            .map(|doc| serde_json::to_string(&doc.document.clone()).unwrap()),
    );

    RecordBatch::try_new(
        Arc::new(document_schema()),
        vec![Arc::new(id), Arc::new(document)],
    )
}

struct EmbeddingRecord {
    id: String,
    document_id: String,
    content: String,
    embedding: Vec<f64>,
}

pub fn embedding_records(documents: &Vec<DocumentEmbeddings>) -> Result<RecordBatch, ArrowError> {
    let embedding_records = documents.into_iter().flat_map(|document| {
        document
            .embeddings.clone()
            .into_iter()
            .map(move |embedding| EmbeddingRecord {
                id: "".to_string(),
                document_id: document.id.clone(),
                content: embedding.document,
                embedding: embedding.vec,
            })
    });

    let id = StringArray::from_iter_values(embedding_records.clone().map(|record| record.id));
    let document_id =
        StringArray::from_iter_values(embedding_records.clone().map(|record| record.document_id));
    let content =
        StringArray::from_iter_values(embedding_records.clone().map(|record| record.content));

    let mut builder = ListBuilder::new(Float64Builder::new());
    embedding_records.for_each(|record| {
        record
            .embedding
            .iter()
            .for_each(|value| builder.values().append_value(*value));
        builder.append(true);
    });

    RecordBatch::try_new(
        Arc::new(document_schema()),
        vec![
            Arc::new(id),
            Arc::new(document_id),
            Arc::new(content),
            Arc::new(builder.finish()),
        ],
    )
}
