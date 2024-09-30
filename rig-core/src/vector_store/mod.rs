use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::Value;

use crate::embeddings::{DocumentEmbeddings, EmbeddingError};

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
    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples with the distance and the document.
    fn top_n<T: for<'a> Deserialize<'a> + std::marker::Send>(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String, T)>, VectorStoreError>> + Send;

    /// Same as `top_n_from_query` but returns the document ids only.
    fn top_n_ids<T: for<'a> Deserialize<'a> + std::marker::Send>(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String)>, VectorStoreError>> + Send
    {
        async move {
            Ok(self
                .top_n::<T>(query, n)
                .await?
                .into_iter()
                .map(|(distance, id, _)| (distance, id))
                .collect())
        }
    }
}

pub type TopNResults = Result<Vec<(f64, String, Value)>, VectorStoreError>;

pub trait VectorStoreIndexDyn: Send + Sync {
    fn top_n<'a>(&'a self, query: &'a str, n: usize) -> BoxFuture<'a, TopNResults>;

    fn top_n_ids<'a>(
        &'a self,
        query: &'a str,
        n: usize,
    ) -> BoxFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>>;
}

impl<I: VectorStoreIndex> VectorStoreIndexDyn for I {
    fn top_n<'a>(
        &'a self,
        query: &'a str,
        n: usize,
    ) -> BoxFuture<'a, Result<Vec<(f64, String, Value)>, VectorStoreError>> {
        Box::pin(self.top_n(query, n))
    }

    fn top_n_ids<'a>(
        &'a self,
        query: &'a str,
        n: usize,
    ) -> BoxFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>> {
        Box::pin(self.top_n_ids::<String>(query, n))
    }
}

pub struct NoIndex;

impl VectorStoreIndex for NoIndex {
    async fn top_n<T: for<'a> Deserialize<'a>>(
        &self,
        _query: &str,
        _n: usize,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        Ok(vec![])
    }

    async fn top_n_ids<T: for<'a> Deserialize<'a>>(
        &self,
        _query: &str,
        _n: usize,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(vec![])
    }
}
