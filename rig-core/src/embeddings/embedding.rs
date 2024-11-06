//! The module defines the [EmbeddingModel] trait, which represents an embedding model that can
//! generate embeddings for documents. It also provides an implementation of the [crate::embeddings::EmbeddingsBuilder]
//! struct, which allows users to build collections of document embeddings using different embedding
//! models and document sources.
//!
//! The module also defines the [Embedding] struct, which represents a single document embedding.
//!
//! Finally, the module defines the [EmbeddingError] enum, which represents various errors that
//! can occur during embedding generation or processing.

use serde::{Deserialize, Serialize};

use crate::OneOrMany;

use super::{Embed, EmbeddingsBuilder};

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
    DocumentError(Box<dyn std::error::Error + Send + Sync + 'static>),

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

    /// Embed multiple text documents in a single request
    fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send;

    /// Embed a single text document
    fn embed_text(
        &self,
        document: &str,
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + Send {
        async {
            Ok(self
                .embed_texts(vec![document.to_string()])
                .await?
                .pop()
                .expect("There should be at least one embedding"))
        }
    }

    /// Embed a single document
    fn embed<T: Embed + Send>(
        &self,
        document: T,
    ) -> impl std::future::Future<Output = Result<OneOrMany<Embedding>, EmbeddingError>> + Send
    {
        async {
            Ok(self
                .embed_many(vec![document])
                .await?
                .pop()
                .map(|(_, embedding)| embedding)
                .expect("There should be at least one embedding"))
        }
    }

    /// Embed multiple documents in a single request
    fn embed_many<T: Embed + Send, I: IntoIterator<Item = T> + Send>(
        &self,
        documents: I,
    ) -> impl std::future::Future<Output = Result<Vec<(T, OneOrMany<Embedding>)>, EmbeddingError>> + Send
    where
        <I as IntoIterator>::IntoIter: std::marker::Send,
    {
        async {
            let builder = EmbeddingsBuilder::new(self.clone());
            builder
                .documents(documents)
                .map_err(|err| EmbeddingError::DocumentError(Box::new(err)))?
                .build()
                .await
        }
    }
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
