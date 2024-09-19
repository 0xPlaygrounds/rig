use lancedb::{arrow::arrow_schema::Schema, query::QueryBase};
use rig::vector_store::{VectorStore, VectorStoreError, VectorStoreIndex};
use table_schemas::{document::DocumentRecords, embedding::EmbeddingRecordsBatch, merge};
use utils::{Insert, Query};

mod table_schemas;
mod utils;

pub struct LanceDbVectorStore {
    document_table: lancedb::Table,
    document_schema: Schema,

    embedding_table: lancedb::Table,
    embedding_schema: Schema,
}

fn lancedb_to_rig_error(e: lancedb::Error) -> VectorStoreError {
    VectorStoreError::DatastoreError(Box::new(e))
}

fn serde_to_rig_error(e: serde_json::Error) -> VectorStoreError {
    VectorStoreError::JsonError(e)
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
            .insert(document_records, self.document_schema.clone())
            .await
            .map_err(lancedb_to_rig_error)?;

        let embedding_records = EmbeddingRecordsBatch::from(documents);

        self.embedding_table
            .insert(embedding_records, self.embedding_schema.clone())
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
        let documents: DocumentRecords = self
            .document_table
            .query()
            .only_if(format!("id = {id}"))
            .execute_query()
            .await?;

        let document = documents
            .as_iter()
            .next()
            .map(|document| serde_json::from_str(&document.document).map_err(serde_to_rig_error))
            .transpose();

        document
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

impl VectorStoreIndex for LanceDbVectorStore {
    fn top_n_from_query(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, rig::embeddings::DocumentEmbeddings)>, VectorStoreError>> + Send {
        todo!()
    }

    fn top_n_from_embedding(
        &self,
        prompt_embedding: &rig::embeddings::Embedding,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, rig::embeddings::DocumentEmbeddings)>, VectorStoreError>> + Send {
        todo!()
    }
}