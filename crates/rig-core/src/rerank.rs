//! Provider-agnostic reranking abstractions.
//!
//! Reranking models reorder a list of documents by relevance to a query.
//! The [`RerankModel`] trait defines the interface, and [`RerankResponse`]
//! carries both the scored results and token usage.

use crate::{
    completion::Usage,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};

crate::provider_response::provider_error_enum!(
    RerankError, "reranking" {
        /// URL construction or parsing failed while preparing a provider request.
        #[error("UrlError: {0}")]
        UrlError(#[from] url::ParseError),
    }
);

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
    /// How relevant this document is to the query — **higher is more
    /// relevant, and that is the only guarantee.**
    ///
    /// Deliberately not "between 0 and 1". The range is the provider's
    /// business and the two implementations in tree disagree: Voyage AI
    /// returns a normalized 0..1 score, while llama.cpp returns the
    /// cross-encoder's raw logit — measured against `llama-server`
    /// b10499-6d05498 with `bge-reranker-v2-m3`, ranking three documents
    /// against "What is a panda?" gives `0.8225`, `-4.7583` and `-8.3761`.
    /// Negative values are normal there, and code that treated this as a
    /// probability (thresholding at 0.5, or multiplying scores) would silently
    /// reorder or discard results on any logit-scoring provider.
    ///
    /// Use it to *order* documents, and compare only within one response.
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
