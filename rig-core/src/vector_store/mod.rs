use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::embeddings::{Embedding, EmbeddingError};

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
pub trait VectorStore<D: Serialize>: Send + Sync {
    /// Query type for the vector store
    type Q;

    /// Add a list of documents to the vector store
    fn add_documents(
        &mut self,
        documents: Vec<(String, D, Vec<Embedding>)>,
    ) -> impl std::future::Future<Output = Result<(), VectorStoreError>> + Send;

    /// Get the embeddings of a document by its id
    fn get_document_embeddings(
        &self,
        id: &str,
    ) -> impl std::future::Future<Output = Result<Option<D>, VectorStoreError>> + Send;

    /// Get the document by a query and deserialize it into the given type
    fn get_document_by_query(
        &self,
        query: Self::Q,
    ) -> impl std::future::Future<Output = Result<Option<D>, VectorStoreError>> + Send;
}

/// Trait for vector store indexes
pub trait VectorStoreIndex: Send + Sync {
    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    fn top_n<T: for<'a> Deserialize<'a> + std::marker::Send>(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String, T)>, VectorStoreError>> + Send;

    /// Same as `top_n` but returns the document ids only.
    fn top_n_ids(
        &self,
        query: &str,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String)>, VectorStoreError>> + Send;
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
        Box::pin(self.top_n_ids(query, n))
    }
}
