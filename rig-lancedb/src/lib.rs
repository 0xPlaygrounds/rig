use std::sync::Arc;

use arrow_array::{cast::AsArray, RecordBatch, RecordBatchIterator};
use conversions::{record_batch::arrow_to_rig_error, DocumentEmbeddings};
use futures::StreamExt;
use lancedb::{
    arrow::arrow_schema::{DataType, Schema},
    query::{ExecutableQuery, QueryBase},
};
use rig::vector_store::{VectorStore, VectorStoreError};

mod conversions;

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
        documents: Vec<rig::embeddings::DocumentEmbeddings>,
    ) -> Result<(), VectorStoreError> {
        let document_embeddings = DocumentEmbeddings::new(documents);

        let record_batch = document_embeddings
            .clone()
            .try_into()
            .map_err(arrow_to_rig_error)?;

        let batches = RecordBatchIterator::new(
            vec![record_batch].into_iter().map(Ok),
            Arc::new(Schema::new(document_embeddings.schema())),
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
    ) -> Result<Option<rig::embeddings::DocumentEmbeddings>, VectorStoreError> {
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
    ) -> Result<Option<rig::embeddings::DocumentEmbeddings>, VectorStoreError> {
        query.execute().await.map_err(lancedb_to_rig_error)?;

        todo!()
    }
}
