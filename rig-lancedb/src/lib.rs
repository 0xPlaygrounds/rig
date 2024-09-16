use std::sync::Arc;

use arrow_array::{
    builder::{Float64Builder, ListBuilder, StringBuilder, StructBuilder},
    cast::AsArray,
    RecordBatch, RecordBatchIterator, StringArray,
};
use futures::StreamExt;
use lancedb::{
    arrow::arrow_schema::{DataType, Field, Fields, Schema},
    query::{ExecutableQuery, QueryBase},
};
use rig::{
    embeddings::DocumentEmbeddings,
    vector_store::{VectorStore, VectorStoreError},
};

pub struct LanceDbVectorStore {
    table: lancedb::Table,
}

fn lancedb_to_rig_error(e: lancedb::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

impl VectorStore for LanceDbVectorStore {
    type Q = lancedb::query::Query;

    async fn add_documents(
        &mut self,
        documents: Vec<DocumentEmbeddings>,
    ) -> Result<(), VectorStoreError> {
        let id = StringArray::from_iter_values(documents.clone().into_iter().map(|doc| doc.id));
        let document = StringArray::from_iter_values(
            documents
                .clone()
                .into_iter()
                .map(|doc| serde_json::to_string(&doc.document).unwrap()),
        );

        let mut list_builder = ListBuilder::new(StructBuilder::from_fields(self.schema(), 0));
        documents.into_iter().map(|doc| {
            let struct_builder = list_builder.values();

            doc.embeddings.into_iter().for_each(|embedding| {
                struct_builder
                    .field_builder::<StringBuilder>(0)
                    .unwrap()
                    .append_value(embedding.document);
                struct_builder
                    .field_builder::<ListBuilder<Float64Builder>>(1)
                    .unwrap()
                    .append_value(embedding.vec.into_iter().map(Some).collect::<Vec<_>>());
                struct_builder.append(true); // Append the first struct
            });

            list_builder.append(true)
        });
        let embeddings = list_builder.finish();

        let batches = RecordBatchIterator::new(
            vec![RecordBatch::try_new(
                Arc::new(Schema::new(self.schema())),
                vec![Arc::new(id), Arc::new(document), Arc::new(embeddings)],
            )
            .unwrap()]
            .into_iter()
            .map(Ok),
            Arc::new(Schema::new(self.schema())),
        );

        self.table
            .add(batches)
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?;

        Ok(())
    }

    async fn get_document_embeddings(
        &self,
        id: &str,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        let mut stream = self
            .table
            .query()
            .only_if(format!("id = {id}"))
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?;

        // let record_batches = stream.try_collect::<Vec<_>>().await.map_err(lancedb_to_rig_error)?;

        stream.next().await.map(|maybe_record_batch| {
            let record_batch = maybe_record_batch?;

            Ok::<(), lancedb::Error>(())
        });

        todo!()
    }

    async fn get_document<T: for<'a> serde::Deserialize<'a>>(
        &self,
        id: &str,
    ) -> Result<Option<T>, VectorStoreError> {
        todo!()
    }

    async fn get_document_by_query(
        &self,
        query: Self::Q,
    ) -> Result<Option<DocumentEmbeddings>, VectorStoreError> {
        query.execute().await.map_err(lancedb_to_rig_error)?;

        todo!()
    }
}

pub fn to_document_embeddings(record_batch: arrow_array::RecordBatch) -> Vec<DocumentEmbeddings> {
    let columns = record_batch.columns().into_iter();

    let ids = match columns.next() {
        Some(column) => match column.data_type() {
            DataType::Utf8 => column.as_string::<i32>().into_iter().collect(),
            _ => vec![],
        },
        None => vec![],
    };
    let documents = match columns.next() {
        Some(column) => match column.data_type() {
            DataType::Utf8 => column.as_string::<i32>().into_iter().collect(),
            _ => vec![],
        },
        None => vec![],
    };

    let embeddings = match columns.next() {
        Some(column) => match column.data_type() {
            DataType::List(embeddings_list) => match embeddings_list.data_type() {
                DataType::Struct(embedding_fields) => match embedding_fields.into_iter().next() {
                    Some(field) => match field.data_type() {
                        DataType::Utf8 => {}
                        _ => vec![],
                    },
                    None => vec![],
                },
                _ => vec![],
            },
            _ => vec![],
        },
        None => vec![],
    };

    todo!()
}

impl LanceDbVectorStore {
    pub fn schema(&self) -> Vec<Field> {
        vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("document", DataType::Utf8, false),
            Field::new(
                "embeddings",
                DataType::List(Arc::new(Field::new(
                    "embedding_item",
                    DataType::Struct(Fields::from(vec![
                        Arc::new(Field::new("document", DataType::Utf8, false)),
                        Arc::new(Field::new(
                            "vec",
                            DataType::List(Arc::new(Field::new("float", DataType::Float64, false))),
                            false,
                        )),
                    ])),
                    false,
                ))),
                false,
            ),
        ]
    }
}
