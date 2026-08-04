//! Provider-agnostic reranking abstractions.
//!
//! Reranking models reorder a list of documents by relevance to a query.
//! Each provider exposes the call as a `functions::rerank` free function, and
//! [`RerankResponse`] carries both the scored results and token usage.

use crate::{completion::Usage, http_client, provider_response};
use serde::{Deserialize, Serialize};

/// Errors returned by reranking models.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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

    /// Raw error response preserved from the reranking model provider
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
}

crate::provider_response::impl_provider_response_helpers!(RerankError);

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
