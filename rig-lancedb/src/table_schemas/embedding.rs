use std::{collections::HashMap, sync::Arc};

use arrow_array::{
    builder::{FixedSizeListBuilder, Float64Builder},
    types::{Float32Type, Float64Type, Utf8Type},
    ArrayRef, RecordBatch, StringArray,
};
use lancedb::arrow::arrow_schema::ArrowError;
use rig::{embeddings::DocumentEmbeddings, vector_store::VectorStoreError};

use crate::utils::{DeserializeByteArray, DeserializeListArray, DeserializePrimitiveArray};

/// Data format in the LanceDB table `embeddings`
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingRecord {
    pub id: String,
    pub document_id: String,
    pub content: String,
    pub embedding: Vec<f64>,
    /// Distance from prompt.
    /// This value is only present after vector search executes and determines the distance
    pub distance: Option<f32>,
}

/// Group of EmbeddingRecord objects. This represents the list of embedding objects in a `DocumentEmbeddings` object.
#[derive(Clone, Debug)]
pub struct EmbeddingRecords {
    records: Vec<EmbeddingRecord>,
    dimension: i32,
}

impl EmbeddingRecords {
    fn new(records: Vec<EmbeddingRecord>, dimension: i32) -> Self {
        EmbeddingRecords { records, dimension }
    }

    fn add_record(&mut self, record: EmbeddingRecord) {
        self.records.push(record);
    }

    pub fn as_iter(&self) -> impl Iterator<Item = &EmbeddingRecord> {
        self.records.iter()
    }
}

/// HashMap where the key is the `DocumentEmbeddings` id
/// and the value is the`EmbeddingRecords` object that corresponds to the document.
#[derive(Debug)]
pub struct EmbeddingRecordsBatch(HashMap<String, EmbeddingRecords>);

impl EmbeddingRecordsBatch {
    fn as_iter(&self) -> impl Iterator<Item = EmbeddingRecords> {
        self.0.clone().into_values().collect::<Vec<_>>().into_iter()
    }

    pub fn get_by_id(&self, id: &str) -> Option<EmbeddingRecords> {
        self.0.get(id).cloned()
    }

    pub fn document_ids(&self) -> String {
        self.0
            .clone()
            .into_keys()
            .map(|id| format!("'{id}'"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Convert from a `DocumentEmbeddings` to an `EmbeddingRecords` object (a list of `EmbeddingRecord` objects)
impl From<DocumentEmbeddings> for EmbeddingRecords {
    fn from(document: DocumentEmbeddings) -> Self {
        EmbeddingRecords::new(
            document
                .embeddings
                .clone()
                .into_iter()
                .enumerate()
                .map(move |(i, embedding)| EmbeddingRecord {
                    id: format!("{}-{i}", document.id),
                    document_id: document.id.clone(),
                    content: embedding.document,
                    embedding: embedding.vec,
                    distance: None,
                })
                .collect(),
            document
                .embeddings
                .first()
                .map(|embedding| embedding.vec.len() as i32)
                .unwrap_or(0),
        )
    }
}

/// Convert from a list of `DocumentEmbeddings` to an `EmbeddingRecordsBatch` object
/// For each `DocumentEmbeddings`, we create an `EmbeddingRecords` and add it to the
/// hashmap with its corresponding `DocumentEmbeddings` id.
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

/// Convert a list of embeddings (`EmbeddingRecords`) to a `RecordBatch`, the data structure that needs ot be written to LanceDB.
/// All embeddings related to a document will be written to the database as part of the same batch.
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

        let mut builder =
            FixedSizeListBuilder::new(Float64Builder::new(), embedding_records.dimension);
        embedding_records.as_iter().for_each(|record| {
            record
                .embedding
                .iter()
                .for_each(|value| builder.values().append_value(*value));
            builder.append(true);
        });

        RecordBatch::try_from_iter(vec![
            ("id", Arc::new(id) as ArrayRef),
            ("document_id", Arc::new(document_id) as ArrayRef),
            ("content", Arc::new(content) as ArrayRef),
            ("embedding", Arc::new(builder.finish()) as ArrayRef),
        ])
    }

    type Error = ArrowError;
}

impl From<EmbeddingRecordsBatch> for Vec<Result<RecordBatch, ArrowError>> {
    fn from(embeddings: EmbeddingRecordsBatch) -> Self {
        embeddings.as_iter().map(RecordBatch::try_from).collect()
    }
}

impl TryFrom<RecordBatch> for EmbeddingRecords {
    type Error = ArrowError;

    fn try_from(record_batch: RecordBatch) -> Result<Self, Self::Error> {
        let binding_0 = record_batch.column(0);
        let ids = binding_0.to_str::<Utf8Type>()?;

        let binding_1 = record_batch.column(1);
        let document_ids = binding_1.to_str::<Utf8Type>()?;

        let binding_2 = record_batch.column(2);
        let contents = binding_2.to_str::<Utf8Type>()?;

        let embeddings = record_batch.column(3).to_float_list::<Float64Type>()?;

        // There is a `_distance` field in the response if the executed query was a VectorQuery
        // Otherwise, for normal queries, the `_distance` field is not present in the response.
        let distances = if record_batch.num_columns() == 5 {
            record_batch
                .column(4)
                .to_float::<Float32Type>()?
                .into_iter()
                .map(Some)
                .collect()
        } else {
            vec![None; record_batch.num_rows()]
        };

        Ok(EmbeddingRecords::new(
            ids.into_iter()
                .zip(document_ids)
                .zip(contents)
                .zip(embeddings.clone())
                .zip(distances)
                .map(
                    |((((id, document_id), content), embedding), distance)| EmbeddingRecord {
                        id: id.to_string(),
                        document_id: document_id.to_string(),
                        content: content.to_string(),
                        embedding,
                        distance,
                    },
                )
                .collect(),
            embeddings
                .iter()
                .map(|embedding| embedding.len() as i32)
                .next()
                .unwrap_or(0),
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
                            .or_insert(EmbeddingRecords::new(
                                vec![record.clone()],
                                record.embedding.len() as i32,
                            ));
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
    async fn test_record_batch_conversion() {
        let embedding_records = EmbeddingRecords::new(
            vec![
                EmbeddingRecord {
                    id: "some_id".to_string(),
                    document_id: "ABC".to_string(),
                    content: serde_json::json!({
                        "title": "Hello world",
                        "body": "Greetings",
                    })
                    .to_string(),
                    embedding: vec![1.0, 2.0, 3.0],
                    distance: None,
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
                    distance: None,
                },
            ],
            3,
        );

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
                distance: None
            }
        );

        assert!(false)
    }
}
