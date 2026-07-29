//! Vector store data vocabulary for semantic search and retrieval.
//!
//! This module defines the shared *data* types that vector store backends speak:
//!
//! - [`VectorSearchRequest`]: A pre-embedded similarity query (see [`request`] for filtering).
//! - [`SearchHit`]: One search result: id, score, and JSON payload.
//! - [`StoreRecord`]: One record to insert: id, JSON payload, and embeddings.
//!
//! There is no shared store trait. Each store crate exposes concrete inherent
//! async methods (`top_n`, `top_n_ids`, `top_n_as`, `insert`, `insert_as`) over
//! these types, with backend-specific filter types kept per store.

pub use request::VectorSearchRequest;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    OneOrMany,
    embeddings::{Embedding, EmbeddingError},
    vector_store::request::FilterError,
};

pub mod builder;
pub mod in_memory_store;
pub mod lsh;
pub mod request;

/// Errors from vector store operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VectorStoreError {
    /// Embedding generation failed while preparing a vector query or insert.
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError),

    /// JSON serialization or deserialization failed.
    #[error("Json error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[cfg(not(target_family = "wasm"))]
    /// Backend-specific datastore error.
    #[error("Datastore error: {0}")]
    DatastoreError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    /// Filter construction or translation failed.
    #[error("Filter error: {0}")]
    FilterError(#[from] FilterError),

    #[cfg(target_family = "wasm")]
    /// Backend-specific datastore error.
    #[error("Datastore error: {0}")]
    DatastoreError(#[from] Box<dyn std::error::Error + 'static>),

    /// A document was missing an ID required by the backend.
    #[error("Missing Id: {0}")]
    MissingIdError(String),

    /// HTTP request failed for an external vector store service.
    #[error("HTTP request error: {0}")]
    ReqwestError(#[from] reqwest::Error),

    /// External vector store service returned an error response.
    #[error("External call to API returned an error. Error code: {0} Message: {1}")]
    ExternalAPIError(StatusCode, String),

    /// A vector search request builder received invalid input.
    #[error("Error while building VectorSearchRequest: {0}")]
    BuilderError(String),
}

/// One vector search result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    /// Document ID as stored in the backend.
    pub id: String,
    /// Similarity score for this hit.
    pub score: f64,
    /// Serialized document payload.
    pub payload: serde_json::Value,
}

impl SearchHit {
    /// Deserializes the payload into `T`.
    pub fn payload_as<T: DeserializeOwned>(&self) -> Result<T, VectorStoreError> {
        Ok(serde_json::from_value(self.payload.clone())?)
    }
}

/// One record to insert into a vector store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoreRecord {
    /// Document ID.
    pub id: String,
    /// Serialized document payload.
    pub payload: serde_json::Value,
    /// Precomputed embeddings for the document.
    pub embeddings: OneOrMany<Embedding>,
}

impl StoreRecord {
    /// Builds a record from an id, a serializable payload, and its embeddings.
    pub fn new<T: Serialize>(
        id: impl Into<String>,
        payload: &T,
        embeddings: OneOrMany<Embedding>,
    ) -> Result<Self, VectorStoreError> {
        Ok(Self {
            id: id.into(),
            payload: serde_json::to_value(payload)?,
            embeddings,
        })
    }
}

/// Index strategy for the [`in_memory_store::InMemoryVectorStore`].
#[derive(Clone, Debug, Default)]
pub enum IndexStrategy {
    /// Checks all documents in the vector store to find the most relevant documents.
    #[default]
    BruteForce,

    /// Uses LSH to find candidates then computes exact distances.
    LSH {
        /// Number of tables to use for LSH.
        num_tables: usize,
        /// Number of hyperplanes to use for LSH.
        num_hyperplanes: usize,
    },
}
