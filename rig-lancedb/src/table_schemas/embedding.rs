use std::{collections::HashMap, sync::Arc};

use arrow_array::{
    builder::{Float64Builder, ListBuilder},
    RecordBatch, StringArray,
};
use lancedb::arrow::arrow_schema::{ArrowError, DataType, Field, Fields, Schema};
use rig::{embeddings::DocumentEmbeddings, vector_store::VectorStoreError};

use crate::utils::DeserializeArrow;

// Data format in the LanceDB table `embeddings`
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingRecord {
    pub id: String,
    pub document_id: String,
    pub content: String,
    pub embedding: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct EmbeddingRecords(Vec<EmbeddingRecord>);

impl EmbeddingRecords {
    fn new(records: Vec<EmbeddingRecord>) -> Self {
        EmbeddingRecords(records)
    }

    pub fn as_iter(&self) -> impl Iterator<Item = &EmbeddingRecord> {
        self.0.iter()
    }

    fn add_record(&mut self, record: EmbeddingRecord) {
        self.0.push(record);
    }
}

impl From<DocumentEmbeddings> for EmbeddingRecords {
    fn from(document: DocumentEmbeddings) -> Self {
        EmbeddingRecords(
            document
                .embeddings
                .clone()
                .into_iter()
                .map(move |embedding| EmbeddingRecord {
                    id: "".to_string(),
                    document_id: document.id.clone(),
                    content: embedding.document,
                    embedding: embedding.vec,
                })
                .collect(),
        )
    }
}

impl From<Vec<DocumentEmbeddings>> for EmbeddingRecordsBatch {
    fn from(documents: Vec<DocumentEmbeddings>) -> Self {
        EmbeddingRecordsBatch(
            documents
                .into_iter()
                .fold(HashMap::new(), |mut acc, document| {
                    acc.insert(document.id.clone(), EmbeddingRecords::from(document));
                    acc
                }),
        )
    }
}

impl TryFrom<EmbeddingRecords> for RecordBatch {
    fn try_from(embedding_records: EmbeddingRecords) -> Result<Self, Self::Error> {
        let id = StringArray::from_iter_values(
            embedding_records.as_iter().map(|record| record.id.clone()),
        );
        let document_id = StringArray::from_iter_values(
            embedding_records
                .as_iter()
                .map(|record| record.document_id.clone()),
        );
        let content = StringArray::from_iter_values(
            embedding_records
                .as_iter()
                .map(|record| record.content.clone()),
        );

        let mut builder = ListBuilder::new(Float64Builder::new());
        embedding_records.as_iter().for_each(|record| {
            record
                .embedding
                .iter()
                .for_each(|value| builder.values().append_value(*value));
            builder.append(true);
        });

        RecordBatch::try_new(
            Arc::new(embedding_schema()),
            vec![
                Arc::new(id),
                Arc::new(document_id),
                Arc::new(content),
                Arc::new(builder.finish()),
            ],
        )
    }

    type Error = ArrowError;
}

pub struct EmbeddingRecordsBatch(HashMap<String, EmbeddingRecords>);
impl EmbeddingRecordsBatch {
    fn as_iter(&self) -> impl Iterator<Item = EmbeddingRecords> {
        self.0.clone().into_values().collect::<Vec<_>>().into_iter()
    }

    pub fn get_by_id(&self, id: &str) -> Option<EmbeddingRecords> {
        self.0.get(id).cloned()
    }

    pub fn document_ids(&self) -> Vec<String> {
        self.0.clone().into_keys().collect()
    }
}

impl From<EmbeddingRecordsBatch> for Vec<Result<RecordBatch, ArrowError>> {
    fn from(embeddings: EmbeddingRecordsBatch) -> Self {
        embeddings.as_iter().map(RecordBatch::try_from).collect()
    }
}

pub fn embedding_schema() -> Schema {
    Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("document_id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            false,
        ),
    ]))
}

impl TryFrom<RecordBatch> for EmbeddingRecords {
    type Error = ArrowError;

    fn try_from(record_batch: RecordBatch) -> Result<Self, Self::Error> {
        let ids = record_batch.deserialize_str_column(0)?;
        let document_ids = record_batch.deserialize_str_column(1)?;
        let contents = record_batch.deserialize_str_column(2)?;
        let embeddings = record_batch.deserialize_float_list_column(3)?;

        Ok(EmbeddingRecords(
            ids.into_iter()
                .zip(document_ids)
                .zip(contents)
                .zip(embeddings)
                .map(
                    |(((id, document_id), content), embedding)| EmbeddingRecord {
                        id: id.to_string(),
                        document_id: document_id.to_string(),
                        content: content.to_string(),
                        embedding,
                    },
                )
                .collect(),
        ))
    }
}

impl TryFrom<Vec<RecordBatch>> for EmbeddingRecordsBatch {
    type Error = VectorStoreError;

    fn try_from(record_batches: Vec<RecordBatch>) -> Result<Self, Self::Error> {
        let embedding_records = record_batches
            .into_iter()
            .map(EmbeddingRecords::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))?;

        let grouped_records =
            embedding_records
                .into_iter()
                .fold(HashMap::new(), |mut acc, records| {
                    records.as_iter().for_each(|record| {
                        acc.entry(record.document_id.clone())
                            .and_modify(|item: &mut EmbeddingRecords| {
                                item.add_record(record.clone())
                            })
                            .or_insert(EmbeddingRecords::new(vec![record.clone()]));
                    });
                    acc
                });

        Ok(EmbeddingRecordsBatch(grouped_records))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::RecordBatch;

    use crate::table_schemas::embedding::{EmbeddingRecord, EmbeddingRecords};

    #[tokio::test]
    async fn test_record_batch_deserialize() {
        let embedding_records = EmbeddingRecords(vec![
            EmbeddingRecord {
                id: "some_id".to_string(),
                document_id: "ABC".to_string(),
                content: serde_json::json!({
                    "title": "Hello world",
                    "body": "Greetings",
                })
                .to_string(),
                embedding: vec![1.0, 2.0, 3.0],
            },
            EmbeddingRecord {
                id: "another_id".to_string(),
                document_id: "DEF".to_string(),
                content: serde_json::json!({
                    "title": "Sup dog",
                    "body": "Greetings",
                })
                .to_string(),
                embedding: vec![4.0, 5.0, 6.0],
            },
        ]);

        let record_batch = RecordBatch::try_from(embedding_records).unwrap();

        let deserialized_record_batch = EmbeddingRecords::try_from(record_batch).unwrap();

        assert_eq!(deserialized_record_batch.as_iter().count(), 2);
        assert_eq!(
            deserialized_record_batch.as_iter().nth(0).unwrap().clone(),
            EmbeddingRecord {
                id: "some_id".to_string(),
                document_id: "ABC".to_string(),
                content: serde_json::json!({
                    "title": "Hello world",
                    "body": "Greetings",
                })
                .to_string(),
                embedding: vec![1.0, 2.0, 3.0],
            }
        );

        assert!(false)
    }
}
