//! Vector store abstractions for semantic search and retrieval.
//!
//! # Core Traits
//!
//! - [`VectorStoreIndex`]: Query a vector store for similar documents.
//! - [`InsertDocuments`]: Insert documents and their embeddings.
//! - [`VectorStoreIndexDyn`]: Type-erased vector queries for runtime-defined retrieval policies.
//!
//! Use [`VectorSearchRequest`] to build queries. See [`request`] for filtering.
//!
//! Types implementing [`VectorStoreIndex`] automatically implement [`PortableTool`].

pub use request::VectorSearchRequest;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    Embed,
    embeddings::{Embedding, EmbeddingError},
    tool::PortableTool,
    vector_store::request::{DynamicSearchFilter, Filter, FilterError, SearchFilter},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
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
    #[error("HTTP request error: {0}")]
    ReqwestError(#[from] reqwest::Error),

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
    fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
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

/// Type-erased `top_n` result: `(score, id, document)` tuples as JSON values.
pub type TopNResults = Result<Vec<(f64, String, Value)>, VectorStoreError>;

/// Type-erased [`VectorStoreIndex`] for dynamic dispatch.
pub trait VectorStoreIndexDyn: WasmCompatSend + WasmCompatSync {
    /// Returns the top N documents for a JSON-serializable request.
    fn top_n<'a>(
        &'a self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, TopNResults>;

    /// Returns only the top N document IDs for a JSON-serializable request.
    fn top_n_ids<'a>(
        &'a self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>>;
}

impl<I, F> VectorStoreIndexDyn for I
where
    I: VectorStoreIndex<Filter = F>,
    F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
{
    fn top_n<'a>(
        &'a self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, TopNResults> {
        Box::pin(async move {
            let req = req.try_map_filter(F::from_dynamic_filter)?;
            Ok(self
                .top_n::<serde_json::Value>(req)
                .await?
                .into_iter()
                .map(|(score, id, doc)| (score, id, F::normalize_dynamic_document(doc)))
                .collect::<Vec<_>>())
        })
    }

    fn top_n_ids<'a>(
        &'a self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>> {
        Box::pin(async move {
            let req = req.try_map_filter(F::from_dynamic_filter)?;
            self.top_n_ids(req).await
        })
    }
}

/// The output of vector store queries invoked via [`PortableTool`]
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
        + for<'de> Deserialize<'de>,
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
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::vector_store::request::Filter;

    struct TestIndex {
        queries: Arc<Mutex<Vec<String>>>,
    }

    #[derive(Clone)]
    struct NativeFilter;

    impl SearchFilter for NativeFilter {
        type Value = String;

        fn eq(_key: impl AsRef<str>, _value: Self::Value) -> Self {
            Self
        }

        fn gt(_key: impl AsRef<str>, _value: Self::Value) -> Self {
            Self
        }

        fn lt(_key: impl AsRef<str>, _value: Self::Value) -> Self {
            Self
        }

        fn and(self, _rhs: Self) -> Self {
            self
        }

        fn or(self, _rhs: Self) -> Self {
            self
        }
    }

    impl DynamicSearchFilter for NativeFilter {
        fn from_dynamic_filter(filter: Filter<serde_json::Value>) -> Result<Self, FilterError> {
            filter.try_interpret(|value| match value {
                Value::String(value) => Ok(value),
                other => Err(FilterError::Expected {
                    expected: "string".to_owned(),
                    got: other.to_string(),
                }),
            })
        }
    }

    struct NativeIndex;

    impl VectorStoreIndex for NativeIndex {
        type Filter = NativeFilter;

        async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
            &self,
            _req: VectorSearchRequest<Self::Filter>,
        ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
            let document = serde_json::from_value(Value::Array(vec![Value::Null; 401]))?;
            Ok(vec![(0.9, "doc-1".to_owned(), document)])
        }

        async fn top_n_ids(
            &self,
            _req: VectorSearchRequest<Self::Filter>,
        ) -> Result<Vec<(f64, String)>, VectorStoreError> {
            Ok(vec![(0.9, "doc-1".to_owned())])
        }
    }

    impl VectorStoreIndex for TestIndex {
        type Filter = Filter<serde_json::Value>;

        async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
            &self,
            req: VectorSearchRequest,
        ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
            self.queries
                .lock()
                .expect("query recorder lock")
                .push(req.query().to_string());
            let document = serde_json::from_value(json!({ "answer": 42 }))?;
            Ok(vec![(0.9, "doc-1".to_string(), document)])
        }

        async fn top_n_ids(
            &self,
            _req: VectorSearchRequest,
        ) -> Result<Vec<(f64, String)>, VectorStoreError> {
            Ok(vec![(0.9, "doc-1".to_string())])
        }
    }

    #[tokio::test]
    async fn vector_store_index_remains_a_tool() {
        let queries = Arc::new(Mutex::new(Vec::new()));
        let index = TestIndex {
            queries: queries.clone(),
        };
        let request = VectorSearchRequest::builder()
            .query("answer")
            .samples(1)
            .build();
        let output = <TestIndex as PortableTool>::call(&index, request)
            .await
            .expect("vector tool call should succeed");

        assert_eq!(<TestIndex as PortableTool>::NAME, "search_vector_store");
        assert_eq!(
            *queries.lock().expect("query recorder lock"),
            vec!["answer"]
        );
        assert_eq!(output.len(), 1);
        let result = output.first().expect("one vector result");
        assert_eq!(result.score, 0.9);
        assert_eq!(result.id, "doc-1");
        assert_eq!(result.document, json!({ "answer": 42 }));
    }

    #[tokio::test]
    async fn dynamic_native_filter_preserves_backend_documents() {
        let request = VectorSearchRequest::builder()
            .query("answer")
            .samples(1)
            .filter(Filter::eq("tag", json!("example")))
            .build();

        let results = VectorStoreIndexDyn::top_n(&NativeIndex, request)
            .await
            .expect("dynamic vector search should succeed");

        assert_eq!(results[0].2.as_array().map(Vec::len), Some(401));
    }

    #[test]
    fn datastore_wraps_backend_errors() {
        let err = VectorStoreError::datastore(std::io::Error::other("db down"));
        assert!(matches!(err, VectorStoreError::DatastoreError(_)));
        assert_eq!(err.to_string(), "Datastore error: db down");
    }
}
