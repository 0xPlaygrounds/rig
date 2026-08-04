// ================================================================
//! Doubleword Embeddings Integration
//! From [Doubleword Inference API](https://docs.doubleword.ai/inference-api/models)
// ================================================================

use serde::Deserialize;
use serde_json::json;

use crate::embeddings::{self, EmbeddingError};

use super::completion::doubleword_api_types::ApiResponse;

// ================================================================
// Doubleword Embedding API
// ================================================================
pub const QWEN3_EMBEDDING_8B: &str = "Qwen/Qwen3-Embedding-8B";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub object: String,
    pub data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<serde_json::Number>,
    pub index: usize,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// Build the serialized `/embeddings` request body. Pure; used by
/// [`super::functions::embed`].
pub(crate) fn build_embedding_body(
    model: &str,
    texts: &[String],
) -> Result<Vec<u8>, EmbeddingError> {
    Ok(serde_json::to_vec(&json!({
        "model": model,
        "input": texts,
    }))?)
}

/// Parse an `/embeddings` response into the normalized
/// [`embeddings::EmbeddingResponse`], zipping vectors back onto
/// `documents`. Pure; used by [`super::functions::embed`]. Doubleword
/// reports no usage.
pub(crate) fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
    if !status.is_success() {
        return Err(EmbeddingError::from_http_response(status, body.to_string()));
    }
    let parsed: ApiResponse<EmbeddingResponse> = serde_json::from_str(body)?;
    match parsed {
        ApiResponse::Ok(response) => {
            if response.data.len() != documents.len() {
                return Err(EmbeddingError::ResponseError(
                    "Response data length does not match input length".into(),
                ));
            }

            let embeddings = response
                .data
                .into_iter()
                .zip(documents)
                .map(|(embedding, document)| embeddings::Embedding {
                    document,
                    vec: embedding
                        .embedding
                        .into_iter()
                        .filter_map(|n| n.as_f64())
                        .collect(),
                })
                .collect();
            Ok(embeddings::EmbeddingResponse {
                embeddings,
                usage: crate::completion::Usage::new(),
            })
        }
        ApiResponse::Error(err) => {
            tracing::warn!(
                message = %err.message(),
                "provider returned an error response"
            );
            Err(EmbeddingError::from_http_response(status, body.to_string()))
        }
    }
}
