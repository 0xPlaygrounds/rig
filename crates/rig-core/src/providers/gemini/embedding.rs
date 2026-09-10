// ================================================================
//! Google Gemini Embeddings Integration
//! From [Gemini API Reference](https://ai.google.dev/api/embeddings)
// ================================================================

use serde_json::json;

use super::{Client, client::ApiResponse};
use crate::{
    embeddings::{self, EmbeddingError},
    http_client::HttpClientExt,
    wasm_compat::WasmCompatSend,
};

/// `gemini-embedding-001` embedding model (3072 dimensions by default)
pub const EMBEDDING_001: &str = "gemini-embedding-001";
/// `text-embedding-004` embedding model (768 dimensions by default)
pub const EMBEDDING_004: &str = "text-embedding-004";

/// Returns the default output dimensionality for known Gemini embedding models.
///
/// See <https://ai.google.dev/gemini-api/docs/models#gemini-embedding>
fn model_default_ndims(model: &str) -> Option<usize> {
    match model {
        EMBEDDING_001 => Some(3072),
        EMBEDDING_004 => Some(768),
        _ => None,
    }
}

#[derive(Clone)]
pub struct EmbeddingModel<T = crate::http_client::BoxedHttpClient> {
    client: Client<T>,
    model: String,
    ndims: usize,
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }

    pub fn with_model(client: Client<T>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

impl<T> EmbeddingModel<T>
where
    T: Clone + HttpClientExt + 'static,
{
    /// Perform the request and return Gemini's native `batchEmbedContents`
    /// response instead of the normalized [`embeddings::EmbeddingResponse`].
    /// Same request, transport, parser, and error path as
    /// [`embeddings::EmbeddingModel::embed_texts_response`].
    ///
    /// <https://ai.google.dev/api/embeddings#batch_embed_contents-SHELL>
    pub async fn raw_embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<gemini_api_types::EmbeddingResponse, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();
        self.raw_embed_texts_slice(&documents).await
    }

    /// Borrow-shaped twin of [`Self::raw_embed_texts`]: the batch is only
    /// serialized into the request body, so callers that keep their documents
    /// (the normalize path) can lend them instead of cloning the batch.
    async fn raw_embed_texts_slice(
        &self,
        documents: &[String],
    ) -> Result<gemini_api_types::EmbeddingResponse, EmbeddingError> {
        // Google batch embed requests. See docstrings for API ref link.
        let requests: Vec<_> = documents
            .iter()
            .map(|doc| {
                json!({
                    "model": format!("models/{}", self.model),
                    "content": json!({
                        "parts": [json!({
                            "text": doc
                        })]
                    }),
                    "output_dimensionality": self.ndims,
                })
            })
            .collect();

        let request_body = json!({ "requests": requests  });

        if let Ok(pretty_body) = serde_json::to_string_pretty(&request_body) {
            tracing::trace!(
                target: "rig::embedding",
                "Sending embedding request to Gemini API {pretty_body}"
            );
        }

        let request_body = serde_json::to_vec(&request_body)?;
        let path = format!("/v1beta/models/{}:batchEmbedContents", self.model);
        let req = self
            .client
            .post(path.as_str())?
            .body(request_body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;
        let response = self.client.send::<_, Vec<u8>>(req).await?;

        let status = response.status();
        let body = response.into_body().await?;

        // Preserve non-success bodies before deserialization because providers
        // may return empty, non-JSON, or otherwise unexpected error payloads.
        if !status.is_success() {
            return Err(EmbeddingError::from_http_response(
                status,
                String::from_utf8_lossy(&body),
            ));
        }

        match serde_json::from_slice::<ApiResponse<gemini_api_types::EmbeddingResponse>>(&body)? {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => {
                tracing::warn!(message = %err.error.message, "provider returned an error response");
                Err(EmbeddingError::from_http_response(
                    status,
                    String::from_utf8_lossy(&body),
                ))
            }
        }
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: Clone + HttpClientExt + 'static,
{
    fn max_documents(&self) -> usize {
        1024
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        crate::telemetry::instrument_modality(
            super::completion::PROVIDER_NAME,
            &self.model,
            crate::telemetry::ModalityOperation::Embeddings,
            async {
                use embeddings::NormalizeEmbeddingResponse as _;

                let documents: Vec<String> = documents.into_iter().collect();
                // Gemini sends no transport request-id header.
                let response = self.raw_embed_texts_slice(&documents).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response
                    .normalize(super::completion::PROVIDER_NAME, documents)?
                    .with_raw(captured))
            },
        )
        .await
    }
}

impl<T> EmbeddingModel<T>
where
    T: Clone + HttpClientExt,
{
    /// Build the model, defaulting `ndims` from the model identifier when the
    /// caller gave none — the body behind `EmbeddingsClient::embedding_model`.
    pub fn make(client: &Client<T>, model: String, dims: Option<usize>) -> Self {
        let ndims = dims.or_else(|| model_default_ndims(&model)).unwrap_or(768);
        Self::new(client.clone(), model, ndims)
    }
}

// =================================================================
// Gemini API Types
// =================================================================
/// Rust Implementation of the Gemini Types from [Gemini API Reference](https://ai.google.dev/api/embeddings)
pub mod gemini_api_types {
    use serde::{Deserialize, Serialize};

    use crate::embeddings::{self, EmbeddingError, NormalizeEmbeddingResponse};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingResponse {
        pub embeddings: Vec<EmbeddingValues>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EmbeddingValues {
        #[serde(default)]
        pub values: Vec<serde_json::Number>,
    }

    impl NormalizeEmbeddingResponse for EmbeddingResponse {
        fn normalize(
            self,
            provider: &str,
            documents: Vec<String>,
        ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
            if self.embeddings.len() != documents.len() {
                return Err(EmbeddingError::ResponseError(
                    "Number of returned embeddings does not match input".into(),
                ));
            }
            let docs = documents
                .into_iter()
                .zip(self.embeddings)
                .map(|(document, embedding)| embeddings::Embedding {
                    document,
                    vec: embedding
                        .values
                        .into_iter()
                        .filter_map(|n| n.as_f64())
                        .collect(),
                })
                .collect();
            // batchEmbedContents reports neither usage nor a response id.
            Ok(embeddings::EmbeddingResponse::new(docs, provider))
        }
    }
}

#[cfg(test)]
mod tests;
