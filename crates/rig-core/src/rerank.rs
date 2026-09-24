//! Provider-agnostic reranking abstractions.
//!
//! Reranking models reorder a list of documents by relevance to a query.
//! The [`RerankModel`] trait defines the interface, and [`RerankResponse`]
//! carries the scored results and the provider's metadata.
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
    response::Response,
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

/// A single reranked document result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Index of the document in the original input list.
    pub index: usize,
    /// The document text, if requested via `return_documents`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
    /// Relevance score, with higher values more relevant within this response.
    /// The range is provider-specific and may include negative values. Do not
    /// interpret it as a probability or compare scores across responses.
    pub relevance_score: f64,
}

/// Reranked results, highest relevance first, and the metadata the provider
/// reported.
pub type RerankResponse = Response<Vec<RerankResult>>;
