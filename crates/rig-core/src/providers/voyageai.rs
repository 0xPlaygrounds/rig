//! Voyage AI API integration
//!
//! # Example
//! ```no_run
//! use rig_core::providers::voyageai;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = voyageai::VoyageAi::from_env()?;
//!
//! let embeddings = provider.embeddings(voyageai::VOYAGE_3_5, None);
//! let reranker = provider.rerank(voyageai::RERANK_2_5);
//! # Ok(())
//! # }
//! ```
//!
//! A wire says what to send and how to read the reply; `.bind(transport)`
//! joins it to a socket and yields the [`Bound`](crate::driver::Bound) that
//! implements the consumer-facing model traits.

use serde::{Deserialize, Serialize};

pub mod wire;

pub use wire::{Embeddings, Rerank, VoyageAi};

/// Voyage AI's API root.
const VOYAGEAI_API_BASE_URL: &str = "https://api.voyageai.com/v1";

// ================================================================
// Voyage AI Embedding API
// ================================================================

/// `voyage-3-large` embedding model (Voyage AI)
pub const VOYAGE_3_LARGE: &str = "voyage-3-large";
/// `voyage-3.5` embedding model (Voyage AI)
pub const VOYAGE_3_5: &str = "voyage-3.5";
/// `voyage-3.5-lite` embedding model (Voyage AI)
pub const VOYAGE_3_5_LITE: &str = "voyage.3-5.lite";
/// `voyage-code-3` embedding model (Voyage AI)
pub const VOYAGE_CODE_3: &str = "voyage-code-3";
/// `voyage-finance-2` embedding model (Voyage AI)
pub const VOYAGE_FINANCE_2: &str = "voyage-finance-2";
/// `voyage-law-2` embedding model (Voyage AI)
pub const VOYAGE_LAW_2: &str = "voyage-law-2";
/// `voyage-code-2` embedding model (Voyage AI)
pub const VOYAGE_CODE_2: &str = "voyage-code-2";

pub fn model_dimensions_from_identifier(model_identifier: &str) -> Option<usize> {
    match model_identifier {
        "voyage-code-2" => Some(1536),
        "voyage-3-large" | "voyage-3.5" | "voyage.3-5.lite" | "voyage-code-3"
        | "voyage-finance-2" | "voyage-law-2" => Some(1024),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Usage {
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

// ================================================================
// Voyage AI Rerank API
// ================================================================

/// `rerank-2.5` reranker model (Voyage AI)
pub const RERANK_2_5: &str = "rerank-2.5";
/// `rerank-2.5-lite` reranker model (Voyage AI)
pub const RERANK_2_5_LITE: &str = "rerank-2.5-lite";
/// `rerank-2` reranker model (Voyage AI)
pub const RERANK_2: &str = "rerank-2";
/// `rerank-2-lite` reranker model (Voyage AI)
pub const RERANK_2_LITE: &str = "rerank-2-lite";
/// `rerank-1` reranker model (Voyage AI)
pub const RERANK_1: &str = "rerank-1";
/// `rerank-lite-1` reranker model (Voyage AI)
pub const RERANK_LITE_1: &str = "rerank-lite-1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankApiResponse {
    pub data: Vec<RerankApiData>,
    pub model: String,
    pub usage: RerankApiUsage,
}

/// The error envelope Voyage can answer a `/rerank` **200** with instead of
/// an ordering: `{"message":"…"}`.
///
/// Decoding it is the whole point — it proves the body is the envelope and
/// nothing else — but the error the consumer sees is built from the raw
/// body, so the provider's payload rides out verbatim.
#[derive(Debug, Deserialize)]
pub struct RerankErrorEnvelope {
    #[allow(dead_code)]
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankApiUsage {
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankApiData {
    pub index: usize,
    pub relevance_score: f64,
    #[serde(default)]
    pub document: Option<String>,
}

#[cfg(test)]
mod tests;
