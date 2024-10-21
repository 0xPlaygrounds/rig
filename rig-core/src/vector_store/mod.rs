use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::Value;

use crate::embeddings::EmbeddingError;

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

/// Trait for vector store indexes
pub trait VectorStoreIndex<T>: Send + Sync
where
    T: for<'a> Deserialize<'a> + Send,
{
    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    fn top_n(
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

impl<I: VectorStoreIndex<Value>> VectorStoreIndexDyn for I {
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
