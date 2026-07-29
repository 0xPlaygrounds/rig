// ================================================================
//! Doubleword Embeddings Integration
//! From [Doubleword Inference API](https://docs.doubleword.ai/inference-api/models)
// ================================================================

use serde::Deserialize;
use serde_json::json;

use crate::{
    embeddings::{self, EmbeddingError},
    http_client::{self, HttpClientExt},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::{Client, client::doubleword_api_types::ApiResponse};

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

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

/// Build the serialized `/embeddings` request body. Pure; shared by the
/// trait path and [`super::functions::embed`].
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
/// `documents`. Pure; shared by the trait path and
/// [`super::functions::embed`]. Doubleword reports no usage.
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

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Default + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    // Conservative default; adjust to Doubleword's documented limit if it differs.
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model, dims.unwrap_or_default())
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let body = build_embedding_body(&self.model, &documents)?;

        let req = self
            .client
            .post("/embeddings")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        let status = response.status();
        let body = if status.is_success() {
            let response_body: Vec<u8> = response.into_body().await?;
            String::from_utf8_lossy(&response_body).into_owned()
        } else {
            http_client::text(response).await?
        };
        parse_embedding_response(status, &body, documents).map(|response| response.embeddings)
    }
}

impl<T> EmbeddingModel<T>
where
    T: Default,
{
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }
}
