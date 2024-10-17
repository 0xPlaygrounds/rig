//! The module defines the [EmbeddingModel] trait, which represents an embedding model that can
//! generate embeddings for documents. It also provides an implementation of the [embeddings::EmbeddingsBuilder]
//! struct, which allows users to build collections of document embeddings using different embedding
//! models and document sources.
//!
//! The module also defines the [Embedding] struct, which represents a single document embedding.
//!
//! Finally, the module defines the [EmbeddingError] enum, which represents various errors that
//! can occur during embedding generation or processing.

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] reqwest::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Error processing the document for embedding
    #[error("DocumentError: {0}")]
    DocumentError(String),

    /// Error parsing the completion response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the embedding model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),
}

/// Trait for embedding models that can generate embeddings for documents.
pub trait EmbeddingModel: Clone + Sync + Send {
    /// The maximum number of documents that can be embedded in a single request.
    const MAX_DOCUMENTS: usize;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed a single document
    fn embed_document(
        &self,
        document: &str,
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + Send
    where
        Self: Sync,
    {
        async {
            Ok(self
                .embed_documents(vec![document.to_string()])
                .await?
                .first()
                .cloned()
                .expect("One embedding should be present"))
        }
    }

    /// Embed multiple documents in a single request
    fn embed_documents(
        &self,
        documents: Vec<String>,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send;
}

/// Struct that holds a single document and its embedding.
#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct Embedding {
    /// The document that was embedded. Used for debugging.
    pub document: String,
    /// The embedding vector
    pub vec: Vec<f64>,
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.document == other.document
    }
}

impl Eq for Embedding {}

impl Embedding {
    pub fn distance(&self, other: &Self) -> f64 {
        let dot_product: f64 = self
            .vec
            .iter()
            .zip(other.vec.iter())
            .map(|(x, y)| x * y)
            .sum();

        let product_of_lengths = (self.vec.len() * other.vec.len()) as f64;

        dot_product / product_of_lengths
    }
}
