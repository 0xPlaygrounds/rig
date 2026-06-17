//! Provider-agnostic reranking abstractions.
//!
//! Reranking models reorder a list of documents by relevance to a query.
//! The [`RerankModel`] trait defines the interface, and [`RerankResponse`]
//! carries both the scored results and token usage.

use crate::{
    completion::Usage,
    http_client,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum RerankError {
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("UrlError: {0}")]
    UrlError(#[from] url::ParseError),

    #[error("ResponseError: {0}")]
    ResponseError(String),

    #[error("ProviderError: {0}")]
    ProviderError(String),
}

/// Trait for reranking models that score documents by relevance to a query.
pub trait RerankModel: WasmCompatSend + WasmCompatSync {
    /// The maximum number of documents that can be reranked in a single request.
    const MAX_DOCUMENTS: usize;

    /// Provider client type used to construct this rerank model.
    type Client;

    /// Construct a model handle from a provider client and model identifier.
    fn make(client: &Self::Client, model: impl Into<String>) -> Self;

    /// Rerank a list of documents against a query.
    fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> impl std::future::Future<Output = Result<RerankResponse, RerankError>> + WasmCompatSend;
}

/// A single reranked document result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Index of the document in the original input list.
    pub index: usize,
    /// The document text, if requested via `return_documents`.
    pub document: Option<String>,
    /// Relevance score between 0 and 1 (higher is more relevant).
    pub relevance_score: f64,
}

/// Response from a reranking request.
#[derive(Debug, Clone)]
pub struct RerankResponse {
    /// Reranked results sorted by relevance (highest first).
    pub results: Vec<RerankResult>,
    /// Model identifier used for this request.
    pub model: String,
    /// Token usage for this rerank request.
    pub usage: Usage,
}
