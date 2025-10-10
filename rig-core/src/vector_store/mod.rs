pub use request::VectorSearchRequest;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    Embed, OneOrMany,
    completion::ToolDefinition,
    embeddings::{Embedding, EmbeddingError},
    tool::Tool,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

pub mod in_memory_store;
pub mod request;

#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError),

    /// Json error (e.g.: serialization, deserialization, etc.)
    #[error("Json error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[cfg(not(target_family = "wasm"))]
    #[error("Datastore error: {0}")]
    DatastoreError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    #[error("Datastore error: {0}")]
    DatastoreError(#[from] Box<dyn std::error::Error + 'static>),

    #[error("Missing Id: {0}")]
    MissingIdError(String),

    #[error("HTTP request error: {0}")]
    ReqwestError(#[from] reqwest::Error),

    #[error("External call to API returned an error. Error code: {0} Message: {1}")]
    ExternalAPIError(StatusCode, String),

    #[error("Error while building VectorSearchRequest: {0}")]
    BuilderError(String),
}

/// Trait for inserting documents into a vector store.
pub trait InsertDocuments: WasmCompatSend + WasmCompatSync {
    /// Insert documents into the vector store.
    ///
    fn insert_documents<Doc: Serialize + Embed + WasmCompatSend>(
        &self,
        documents: Vec<(Doc, OneOrMany<Embedding>)>,
    ) -> impl std::future::Future<Output = Result<(), VectorStoreError>> + WasmCompatSend;
}

/// Trait for vector store indexes
pub trait VectorStoreIndex: WasmCompatSend + WasmCompatSync {
    /// Get the top n documents based on the distance to the given query.
    /// The result is a list of tuples of the form (score, id, document)
    fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
        &self,
        req: VectorSearchRequest,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String, T)>, VectorStoreError>>
    + WasmCompatSend;

    /// Same as `top_n` but returns the document ids only.
    fn top_n_ids(
        &self,
        req: VectorSearchRequest,
    ) -> impl std::future::Future<Output = Result<Vec<(f64, String)>, VectorStoreError>> + WasmCompatSend;
}

pub type TopNResults = Result<Vec<(f64, String, Value)>, VectorStoreError>;

pub trait VectorStoreIndexDyn: WasmCompatSend + WasmCompatSync {
    fn top_n<'a>(&'a self, req: VectorSearchRequest) -> WasmBoxedFuture<'a, TopNResults>;

    fn top_n_ids<'a>(
        &'a self,
        req: VectorSearchRequest,
    ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>>;
}

impl<I: VectorStoreIndex> VectorStoreIndexDyn for I {
    fn top_n<'a>(&'a self, req: VectorSearchRequest) -> WasmBoxedFuture<'a, TopNResults> {
        Box::pin(async move {
            Ok(self
                .top_n::<serde_json::Value>(req)
                .await?
                .into_iter()
                .map(|(score, id, doc)| (score, id, prune_document(doc).unwrap_or_default()))
                .collect::<Vec<_>>())
        })
    }

    fn top_n_ids<'a>(
        &'a self,
        req: VectorSearchRequest,
    ) -> WasmBoxedFuture<'a, Result<Vec<(f64, String)>, VectorStoreError>> {
        Box::pin(self.top_n_ids(req))
    }
}

fn prune_document(document: serde_json::Value) -> Option<serde_json::Value> {
    match document {
        Value::Object(mut map) => {
            let new_map = map
                .iter_mut()
                .filter_map(|(key, value)| {
                    prune_document(value.take()).map(|value| (key.clone(), value))
                })
                .collect::<serde_json::Map<_, _>>();

            Some(Value::Object(new_map))
        }
        Value::Array(vec) if vec.len() > 400 => None,
        Value::Array(vec) => Some(Value::Array(
            vec.into_iter().filter_map(prune_document).collect(),
        )),
        Value::Number(num) => Some(Value::Number(num)),
        Value::String(s) => Some(Value::String(s)),
        Value::Bool(b) => Some(Value::Bool(b)),
        Value::Null => Some(Value::Null),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VectorStoreOutput {
    pub score: f64,
    pub id: String,
    pub document: Value,
}

impl<T> Tool for T
where
    T: VectorStoreIndex,
{
    const NAME: &'static str = "search_vector_store";

    type Error = VectorStoreError;
    type Args = VectorSearchRequest;
    type Output = Vec<VectorStoreOutput>;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Retrieves the most relevant documents from a vector store based on a query."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query string to search for relevant documents in the vector store."
                    },
                    "samples": {
                        "type": "integer",
                        "description": "The maxinum number of samples / documents to retrieve.",
                        "default": 5,
                        "minimum": 1
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Similarity search threshold. If present, any result with a distance less than this may be omitted from the final result."
                    }
                },
                "required": ["query", "samples"]
            }),
        }
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
