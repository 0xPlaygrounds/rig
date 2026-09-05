//! Vector store abstractions for semantic search and retrieval.
//!
//! # Core Traits
//!
//! - [`VectorStoreIndex`]: Query a vector store for similar documents.
//! - [`InsertDocuments`]: Insert documents and their embeddings.
//!
//! Use [`VectorSearchRequest`] to build queries. See [`request`] for filtering.
//!
//! Types implementing [`VectorStoreIndex`] automatically implement [`PortableTool`].

use http::StatusCode;
pub use request::VectorSearchRequest;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{
    Embed,
    embeddings::{Embedding, EmbeddingError},
    tool::PortableTool,
    vector_store::request::{FilterError, SearchFilter},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

pub mod builder;
pub mod in_memory_store;
pub mod lsh;
pub mod request;

/// Errors from vector store operations.
#[derive(Debug, thiserror::Error)]
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
    ///
    /// Non-success responses arrive as the status-bearing
    /// [`crate::http_client::Error`] variants, so callers can still inspect
    /// the status code; response-less transport failures arrive as
    /// [`crate::http_client::Error::Instance`].
    #[error("HTTP request error: {0}")]
    Http(#[from] crate::http_client::Error),

    /// External vector store service returned an error response.
    #[error("External call to API returned an error. Error code: {0} Message: {1}")]
    ExternalAPIError(StatusCode, String),

    /// A vector search request builder received invalid input.
    #[error("Error while building VectorSearchRequest: {0}")]
    BuilderError(String),
}

impl VectorStoreError {
    /// Wraps a backend error as [`VectorStoreError::DatastoreError`].
    ///
    /// Handles the wasm/non-wasm trait-bound split in one place; use as
    /// `.map_err(VectorStoreError::datastore)`.
    #[cfg(not(target_family = "wasm"))]
    pub fn datastore(e: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::DatastoreError(Box::new(e))
    }

    /// Wraps a backend error as [`VectorStoreError::DatastoreError`].
    #[cfg(target_family = "wasm")]
    pub fn datastore(e: impl std::error::Error + 'static) -> Self {
        Self::DatastoreError(Box::new(e))
    }
}

/// Serializes each document to JSON once, then applies `f` to every
/// `(document, embedding)` pair, flattening the results into a single vector.
///
/// This is the shared shape of most [`InsertDocuments`] implementations:
/// build one backend record per embedding, carrying the owning document's
/// serialized form.
pub fn flatten_embedded<Doc: Serialize, R>(
    documents: Vec<(Doc, Vec<Embedding>)>,
    mut f: impl FnMut(&Value, Embedding) -> Result<R, VectorStoreError>,
) -> Result<Vec<R>, VectorStoreError> {
    let mut records = Vec::new();
    for (document, embeddings) in documents {
        let json_document = serde_json::to_value(&document)?;
        for embedding in embeddings {
            records.push(f(&json_document, embedding)?);
        }
    }
    Ok(records)
}

/// Trait for inserting documents and embeddings into a vector store.
pub trait InsertDocuments: WasmCompatSend + WasmCompatSync {
    /// Insert precomputed embeddings for each document.
    ///
    /// **Every document must carry at least one embedding.** The embedding
    /// list was non-empty by construction until it became a `Vec`; the
    /// requirement did not go away, it moved to the caller. Implementors do
    /// not guard it, and what an empty list does varies by store — some
    /// silently insert nothing, some store a document no similarity search
    /// can ever return, some surface a confusing driver error. Embeddings
    /// produced by `EmbeddingsBuilder` always satisfy this; only hand-built
    /// tuples can violate it.
    fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> impl std::future::Future<Output = Result<(), VectorStoreError>> + WasmCompatSend;
}

/// Trait for querying a vector store by similarity.
pub trait VectorStoreIndex: WasmCompatSend + WasmCompatSync {
    /// The filter type for this backend.
    type Filter: SearchFilter + WasmCompatSend + WasmCompatSync;

    /// Returns the top N most similar documents as `(score, id, document)` tuples.
    fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String, T)>, VectorStoreError>>
    + WasmCompatSend;

    /// Returns the top N most similar document IDs as `(score, id)` tuples.
    fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String)>, VectorStoreError>> + WasmCompatSend;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VectorStoreOutput {
    /// Similarity score returned by the vector store.
    pub score: f64,
    /// Document ID returned by the vector store.
    pub id: String,
    /// Serialized document payload.
    pub document: Value,
}

impl<T, F> PortableTool for T
where
    F: SearchFilter<Value = serde_json::Value>
        + WasmCompatSend
        + WasmCompatSync
        + serde::de::DeserializeOwned,
    T: VectorStoreIndex<Filter = F>,
{
    const NAME: &'static str = "search_vector_store";
    type Error = VectorStoreError;
    type Args = VectorSearchRequest<F>;
    type Output = Vec<VectorStoreOutput>;

    fn description(&self) -> String {
        "Retrieves the most relevant documents from a vector store based on a query.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query string to search for relevant documents in the vector store."
                },
                "samples": {
                    "type": "integer",
                    "description": "The maximum number of samples / documents to retrieve.",
                    "default": 5,
                    "minimum": 1
                },
                "threshold": {
                    "type": "number",
                    "description": "Similarity search threshold. If present, any result with a distance less than this may be omitted from the final result."
                }
            },
            "required": ["query", "samples"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let results = self.top_n(args).await?;
        Ok(results
            .into_iter()
            .map(|(score, id, document)| VectorStoreOutput {
                score,
                id,
                document,
            })
            .collect())
    }
}

/// Index strategy for the super::InMemoryVectorStore
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

#[cfg(test)]
mod tests;
