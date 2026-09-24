//! Provider-agnostic reranking abstractions.
//!
//! Reranking models reorder a list of documents by relevance to a query.
//! A [`Model`](crate::driver::Model) over a rerank wire calls one, and
//! [`RerankResponse`] carries both the scored results and token usage.
//!
//! ```no_run
//! use rig_core::driver::{Model, Transport};
//! use rig_core::operation::Rerank;
//! use rig_core::wire::Wire;
//!
//! # async fn example<W, T>(model: &Model<W, T>) -> Result<(), Box<dyn std::error::Error>>
//! # where W: Wire<Op = Rerank> + Clone, T: Transport<W> {
//! let response = model.rerank("Rust", vec!["A systems programming language".into()]).await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```

use crate::completion::{ResponseIdentity, Usage};
use crate::error::ProviderError;
use serde::{Deserialize, Serialize};

impl<W, T> crate::driver::Model<W, T>
where
    W: crate::wire::Wire<Op = crate::operation::Rerank> + Clone,
    T: crate::driver::Transport<W>,
{
    /// Rerank `documents` against `query`.
    pub async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> Result<RerankResponse, ProviderError> {
        let request = crate::operation::RerankRequest {
            query: query.to_owned(),
            documents,
        };
        self.call(request, None).await
    }
}

/// A single reranked document result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Index of the document in the original input list.
    pub index: usize,
    /// The document text, if requested via `return_documents`.
    pub document: Option<String>,
    /// Relevance score, with higher values more relevant within this response.
    /// The range is provider-specific and may include negative values. Do not
    /// interpret it as a probability or compare scores across responses.
    pub relevance_score: f64,
}

/// Ranked documents and normalized provider metadata.
/// Provider-specific response data is available through [`Self::raw`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Reranked results sorted by relevance (highest first).
    pub results: Vec<RerankResult>,
    /// Provider-reported model identifier, or `None` when omitted.
    #[serde(default)]
    pub model: Option<String>,
    /// Token usage for this rerank request; every counter is `None` when the
    /// provider reported none (see [`Usage`]).
    #[serde(default)]
    pub usage: Usage,
    /// Stable descriptor name of the provider that produced this response,
    /// for example `"voyageai"`. Always populated.
    pub provider: String,
    /// Provider-assigned response-scoped identifier, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Transport request ID from HTTP headers, or `None` when unreported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Provider response document. Defaults to null until populated.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub raw: serde_json::Value,
}

impl RerankResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers.
    pub fn new(results: Vec<RerankResult>, provider: impl Into<String>) -> Self {
        Self {
            results,
            model: None,
            usage: Usage::default(),
            provider: provider.into(),
            response_id: None,
            provider_request_id: None,
            raw: serde_json::Value::Null,
        }
    }

    /// This response's identity metadata as one [`ResponseIdentity`] carrier.
    /// `message_id` is always `None`: nothing here is replayed as an
    /// assistant message.
    pub fn identity(&self) -> ResponseIdentity {
        ResponseIdentity {
            message_id: None,
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }
}

crate::provider_response::modality_response_metadata_setters!(RerankResponse);
