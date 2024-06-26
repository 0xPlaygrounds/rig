use serde::Deserialize;

use crate::embeddings::{DocumentEmbeddings, Embedding, EmbeddingError};

pub mod in_memory_store;

#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError),

    /// Json error (e.g.: serialization, deserialization, etc.)
    #[error("Json error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Datastore error: {0}")]
    DatastoreError(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Trait for vector stores
pub trait VectorStore: Send + Sync {
    /// Query type for the vector store
    type Q;

    /// Add a list of documents to the vector store
    fn add_documents(
        &mut self,
        documents: Vec<DocumentEmbeddings>,
    ) -> impl std::future::Future<Output = Result<(), VectorStoreError>> + Send;

    /// Get the embeddings of a document by its id
    fn get_document_embeddings(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Option<DocumentEmbeddings>, VectorStoreError>> + Send;

    /// Get the document by its id and deserialize it into the given type
    fn get_document<T: for<'a> Deserialize<'a>>(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Option<T>, VectorStoreError>> + Send;

    /// Get the document by a query and deserialize it into the given type
    fn get_document_by_query(
        &self,
        query: Self::Q,
    ) -> impl std::future::Future<Output = Result<Option<DocumentEmbeddings>, VectorStoreError>> + Send;
}

/// Trait for vector store indexes
pub trait VectorStoreIndex: Send + Sync {
    /// Get the top n documents based on the distance to the given embedding.
    /// The distance is calculated as the cosine distance between the prompt and
    /// the document embedding.
    /// The result is a list of tuples with the distance and the document.
    fn top_n_from_query(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, DocumentEmbeddings)>, VectorStoreError>> + Send;

    /// Same as `top_n_from_query` but returns the documents without its embeddings.
    /// The documents are deserialized into the given type.
    fn top_n_documents_from_query<T: for<'a> Deserialize<'a>>(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, T)>, VectorStoreError>> + Send {
        async move {
            let documents = self.top_n_from_query(query, n).await?;
            Ok(documents
                .into_iter()
                .map(|(distance, doc)| (distance, serde_json::from_value(doc.document).unwrap()))
                .collect())
        }
    }

    /// Same as `top_n_from_query` but returns the document ids only.
    fn top_n_ids_from_query(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String)>, VectorStoreError>> + Send
    {
        async move {
            let documents = self.top_n_from_query(query, n).await?;
            Ok(documents
                .into_iter()
                .map(|(distance, doc)| (distance, doc.id))
                .collect())
        }
    }

    /// Get the top n documents based on the distance to the given embedding.
    /// The distance is calculated as the cosine distance between the prompt and
    /// the document embedding.
    /// The result is a list of tuples with the distance and the document.
    fn top_n_from_embedding(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, DocumentEmbeddings)>, VectorStoreError>> + Send;

    /// Same as `top_n_from_embedding` but returns the documents without its embeddings.
    /// The documents are deserialized into the given type.
    fn top_n_documents_from_embedding<T: for<'a> Deserialize<'a>>(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, T)>, VectorStoreError>> + Send {
        async move {
            let documents = self.top_n_from_embedding(prompt_embedding, n).await?;
            Ok(documents
                .into_iter()
                .map(|(distance, doc)| (distance, serde_json::from_value(doc.document).unwrap()))
                .collect())
        }
    }

    /// Same as `top_n_from_embedding` but returns the document ids only.
    fn top_n_ids_from_embedding(
        &self,
        prompt_embedding: &Embedding,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String)>, VectorStoreError>> + Send
    {
        async move {
            let documents = self.top_n_from_embedding(prompt_embedding, n).await?;
            Ok(documents
                .into_iter()
                .map(|(distance, doc)| (distance, doc.id))
                .collect())
        }
    }
}

pub struct NoIndex;

impl VectorStoreIndex for NoIndex {
    async fn top_n_from_query(
        &self,
        _query: &str,
        _n: usize,
    ) -> Result<Vec<(f64, DocumentEmbeddings)>, VectorStoreError> {
        Ok(vec![])
    }

    async fn top_n_from_embedding(
        &self,
        _prompt_embedding: &Embedding,
        _n: usize,
    ) -> Result<Vec<(f64, DocumentEmbeddings)>, VectorStoreError> {
        Ok(vec![])
    }
}
