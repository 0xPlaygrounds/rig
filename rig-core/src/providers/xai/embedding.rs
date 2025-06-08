// ================================================================
//! xAI Embeddings Integration
//! From [xAI Reference](https://api.x.ai/docs/#/v1/handle_embedding_request)
// ================================================================

use serde::Deserialize;
use serde_json::json;

use crate::embeddings::{self, EmbeddingError};

use super::{
    client::xai_api_types::{ApiErrorResponse, ApiResponse},
    Client,
};

// ================================================================
// xAI Embedding API
// ================================================================
/// `v1` embedding model
pub const EMBEDDING_V1: &str = "v1";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    token_ids: Vec<EmbeddingToken>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingToken {
    pub token_id: usize,
    pub string_token: String,
    token_bytes: Vec<f64>,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message())
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Error(err) => Err(EmbeddingError::ProviderError(err.message())),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
    ndims: usize,
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        // Instead of the endpoint being called "embeddings",
        // in xAI it's actually called "tokenize-text"
        // ref: <https://docs.x.ai/docs/api-reference#tokenize-text>
        let response = self
            .client
            .post("/v1/tokenize-text")
            .json(&json!({
                "model": self.model,
                "input": documents,
            }))
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<EmbeddingResponse>>().await? {
                ApiResponse::Ok(response) => {
                    if response.token_ids.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    Ok(response
                        .token_ids
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding.token_bytes,
                        })
                        .collect())
                }
                ApiResponse::Error(err) => Err(EmbeddingError::ProviderError(err.message())),
            }
        } else {
            Err(EmbeddingError::ProviderError(response.text().await?))
        }
    }
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}
