use std::sync::Arc;

use arrow_array::RecordBatchIterator;
use conversions::{document_records, document_schema, embedding_records, embedding_schema};
use lancedb::{arrow::arrow_schema::{ArrowError, Schema}, query::ExecutableQuery};
use rig::vector_store::{VectorStore, VectorStoreError};

mod conversions;

pub struct LanceDbVectorStore {
    document_table: lancedb::Table,
    embedding_table: lancedb::Table,
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
        let document_batches = RecordBatchIterator::new(
            vec![document_records(&documents)],
            Arc::new(document_schema()),
        );

        let embedding_batches = RecordBatchIterator::new(
            vec![embedding_records(&documents)],
            Arc::new(embedding_schema()),
        );

        self.document_table
            .add(document_batches)
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?;

        self.embedding_table
            .add(embedding_batches)
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?;

        Ok(())
    }

    async fn get_document_embeddings(
        &self,
        id: &str,
    ) -> Result<Option<rig::embeddings::DocumentEmbeddings>, VectorStoreError> {
        // let mut stream = self
        //     .table
        //     .query()
        //     .only_if(format!("id = {id}"))
        //     .execute()
        //     .await
        //     .map_err(lancedb_to_rig_error)?;

        // // let record_batches = stream.try_collect::<Vec<_>>().await.map_err(lancedb_to_rig_error)?;

        // stream.next().await.map(|maybe_record_batch| {
        //     let record_batch = maybe_record_batch?;

        //     Ok::<(), lancedb::Error>(())
        // });

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
