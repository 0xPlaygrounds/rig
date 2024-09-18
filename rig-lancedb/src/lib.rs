use std::sync::Arc;

use arrow_array::RecordBatchIterator;
use lancedb::query::QueryBase;
use rig::vector_store::{VectorStore, VectorStoreError};
use table_schemas::{
    document::{document_schema, DocumentRecords},
    embedding::{embedding_schema, EmbeddingRecordsBatch},
    merge,
};
use utils::Query;

mod table_schemas;
mod utils;

pub struct LanceDbVectorStore {
    document_table: lancedb::Table,
    embedding_table: lancedb::Table,
}

fn lancedb_to_rig_error(e: lancedb::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

fn serde_to_rig_error(e: serde_json::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

impl VectorStore for LanceDbVectorStore {
    type Q = lancedb::query::Query;

    async fn add_documents(
        &mut self,
        documents: Vec<rig::embeddings::DocumentEmbeddings>,
    ) -> Result<(), VectorStoreError> {
        let document_records =
            DocumentRecords::try_from(documents.clone()).map_err(serde_to_rig_error)?;

        self.document_table
            .add(RecordBatchIterator::new(
                vec![document_records.try_into()],
                Arc::new(document_schema()),
            ))
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?;

        let embedding_records = EmbeddingRecordsBatch::from(documents);

        self.embedding_table
            .add(RecordBatchIterator::new(
                embedding_records.record_batch_iter(),
                Arc::new(embedding_schema()),
            ))
            .execute()
            .await
            .map_err(lancedb_to_rig_error)?;

        Ok(())
    }

    async fn get_document_embeddings(
        &self,
        id: &str,
    ) -> Result<Option<rig::embeddings::DocumentEmbeddings>, VectorStoreError> {
        let documents: DocumentRecords = self
            .document_table
            .query()
            .only_if(format!("id = {id}"))
            .execute_query()
            .await?;

        let embeddings: EmbeddingRecordsBatch = self
            .embedding_table
            .query()
            .only_if(format!("document_id = {id}"))
            .execute_query()
            .await?;

        Ok(merge(documents, embeddings)?.into_iter().next())
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
        let documents: DocumentRecords = query.execute_query().await?;

        let embeddings: EmbeddingRecordsBatch = self
            .embedding_table
            .query()
            .only_if(format!("document_id IN [{}]", documents.ids().join(",")))
            .execute_query()
            .await?;

        Ok(merge(documents, embeddings)?.into_iter().next())
    }
}
