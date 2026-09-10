//! Provider-agnostic reranking abstractions.
//!
//! Reranking models reorder a list of documents by relevance to a query.
//! The [`RerankModel`] trait defines the interface, and [`RerankResponse`]
//! carries both the scored results and token usage.

use std::{fmt, sync::Arc};

use crate::{
    completion::{ResponseIdentity, Usage},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
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
    /// The maximum number of documents that can be reranked in a single
    /// request.
    ///
    /// A method rather than an associated constant so the value survives type
    /// erasure: [`RerankModelHandle`] captures it by value at construction.
    fn max_documents(&self) -> usize;

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

/// The normalized reranking response: the scored results plus the metadata
/// every provider can report, attributed to the provider that produced it.
///
/// Concrete and provider-neutral, so it survives type erasure through a
/// [`RerankModelHandle`] unchanged. The provider's own payload stays reachable
/// through a model's inherent `raw_rerank` method, which performs the same
/// request and returns the provider's native type, and through [`Self::raw`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Reranked results sorted by relevance (highest first).
    pub results: Vec<RerankResult>,
    /// Provider-reported model identifier, when the wire response named one.
    /// `None` when the provider omitted it — a server that omits `model`
    /// still produced a ranking.
    #[serde(default)]
    pub model: Option<String>,
    /// Token usage for this rerank request. Zero-valued when the provider
    /// reported none — the sentinel [`Usage`] documents.
    #[serde(default)]
    pub usage: Usage,
    /// Stable descriptor name of the provider that produced this response,
    /// for example `"voyageai"`. Always populated.
    pub provider: String,
    /// Provider-assigned response-scoped identifier, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The provider's transport-level request identifier, taken from the HTTP
    /// response headers — the id provider support asks for. `None` means the
    /// provider reported none; that is a documented outcome, never an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// The provider's own response for this call: the value the model's
    /// inherent `raw_rerank` would have returned, serialized. Every provider
    /// seam populates it. `Value::Null` means the value was built without a
    /// provider behind it (a test double), never that the provider sent
    /// nothing.
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
            usage: Usage::new(),
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

/// Convert a provider's own rerank payload into the normalized
/// [`RerankResponse`].
///
/// The provider descriptor name is an *input*, never something the conversion
/// knows — the Jina-shaped wire is shared by several servers, and a hardcoded
/// name would mislabel every provider but one. A trait rather than
/// `TryFrom<(&str, T)>` so that out-of-tree provider extensions can implement
/// it on their own response type without tripping the orphan rule.
pub trait NormalizeRerankResponse {
    /// Normalize this payload, attributing it to `provider`.
    fn normalize(self, provider: &str) -> Result<RerankResponse, RerankError>;
}

/// Private object-safe mirror of [`RerankModel`]: the public trait stays
/// generic (RPITIT future); this dyn-safe twin exists only so
/// [`RerankModelHandle`] can store one vtable. `max_documents` is deliberately
/// absent: it is construction-time data captured alongside the erased model.
trait ErasedRerankModel: WasmCompatSend + WasmCompatSync {
    fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> WasmBoxedFuture<'_, Result<RerankResponse, RerankError>>;
}

impl<M> ErasedRerankModel for M
where
    M: RerankModel + 'static,
{
    fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
    ) -> WasmBoxedFuture<'_, Result<RerankResponse, RerankError>> {
        Box::pin(async move { RerankModel::rerank(self, &query, documents).await })
    }
}

/// The handle's single allocation: snapshot data first, the unsized erased
/// model last, so `Arc<RerankDriver<M>>` unsize-coerces to
/// `Arc<RerankDriver<dyn ErasedRerankModel>>` without a second box.
struct RerankDriver<M: ?Sized> {
    max_documents: usize,
    label: Option<String>,
    model: M,
}

/// A cloneable, opaque handle to live reranking-model behavior — the
/// [`RerankModel`] twin of `EmbeddingModelHandle`, same shape and same
/// guarantees: one `Arc`, `max_documents` captured by value at erasure, the
/// model never cloned, and no way to replace it. It exists for type
/// ergonomics and dyn-storability, not for swapping.
#[derive(Clone)]
pub struct RerankModelHandle {
    inner: Arc<RerankDriver<dyn ErasedRerankModel>>,
}

impl RerankModelHandle {
    /// Erase a typed rerank model into a runtime handle.
    pub fn new<M>(model: M) -> Self
    where
        M: RerankModel + 'static,
    {
        Self::from_parts(None, model)
    }

    /// Erase a typed rerank model and attach a diagnostic label.
    pub fn named<M>(label: impl Into<String>, model: M) -> Self
    where
        M: RerankModel + 'static,
    {
        Self::from_parts(Some(label.into()), model)
    }

    fn from_parts<M>(label: Option<String>, model: M) -> Self
    where
        M: RerankModel + 'static,
    {
        let max_documents = model.max_documents();
        Self {
            inner: Arc::new(RerankDriver {
                max_documents,
                label,
                model,
            }),
        }
    }

    /// Returns the optional diagnostic label attached to this handle.
    pub fn label(&self) -> Option<&str> {
        self.inner.label.as_deref()
    }
}

impl RerankModel for RerankModelHandle {
    fn max_documents(&self) -> usize {
        self.inner.max_documents
    }

    fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
    ) -> impl std::future::Future<Output = Result<RerankResponse, RerankError>> + WasmCompatSend
    {
        self.inner.model.rerank(query.to_owned(), documents)
    }
}

impl fmt::Debug for RerankModelHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RerankModelHandle")
            .field("label", &self.label())
            .field("max_documents", &self.inner.max_documents)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod handle_tests;
