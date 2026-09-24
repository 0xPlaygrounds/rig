//! Provider-agnostic reranking abstractions.
//!
//! Reranking models reorder a list of documents by relevance to a query.
//! The [`RerankModel`] trait defines the interface, and [`RerankResponse`]
//! carries both the scored results and token usage.
//!
//! ```no_run
//! use rig_core::rerank::RerankModel;
//!
//! # async fn example(model: &impl RerankModel) -> Result<(), Box<dyn std::error::Error>> {
//! let response = model.rerank("Rust", vec!["A systems programming language".into()]).await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```

use crate::error::ProviderError;
use crate::{
    completion::{ResponseIdentity, Usage},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};

/// Trait for reranking models that score documents by relevance to a query.
pub trait RerankModel: WasmCompatSend + WasmCompatSync {
    /// Maximum documents accepted in one request.
    fn max_documents(&self) -> usize;

    /// Rerank a list of documents against a query.
    fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> impl std::future::Future<Output = Result<RerankResponse, ProviderError>> + WasmCompatSend;
}

/// Forwards reranking through shared ownership.
impl<M: RerankModel + ?Sized> RerankModel for std::sync::Arc<M> {
    fn max_documents(&self) -> usize {
        (**self).max_documents()
    }

    fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> impl std::future::Future<Output = Result<RerankResponse, ProviderError>> + WasmCompatSend
    {
        (**self).rerank(query, documents)
    }
}

/// A [`RerankModel`] of any type, for tables of models or for naming a model without
/// its type. Clones share the model.
#[derive(Clone)]
pub struct BoxedRerankModel(std::sync::Arc<dyn DynRerankModel>);

impl BoxedRerankModel {
    /// Erase `model`.
    pub fn new(model: impl RerankModel + 'static) -> Self {
        Self(std::sync::Arc::new(model))
    }
}

impl std::fmt::Debug for BoxedRerankModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoxedRerankModel").finish_non_exhaustive()
    }
}

impl RerankModel for BoxedRerankModel {
    fn max_documents(&self) -> usize {
        self.0.max_documents()
    }

    fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> impl std::future::Future<Output = Result<RerankResponse, ProviderError>> + WasmCompatSend
    {
        self.0.rerank(query.to_owned(), documents)
    }
}

/// The object-safe form of [`RerankModel`] behind [`BoxedRerankModel`].
trait DynRerankModel: WasmCompatSend + WasmCompatSync {
    fn max_documents(&self) -> usize;

    fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> crate::wasm_compat::WasmBoxedFuture<'_, Result<RerankResponse, ProviderError>>;
}

impl<M: RerankModel> DynRerankModel for M {
    fn max_documents(&self) -> usize {
        RerankModel::max_documents(self)
    }

    fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> crate::wasm_compat::WasmBoxedFuture<'_, Result<RerankResponse, ProviderError>> {
        Box::pin(async move { RerankModel::rerank(self, &query, documents).await })
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
