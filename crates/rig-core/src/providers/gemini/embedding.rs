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
pub struct EmbeddingModel<T = reqwest::Client> {
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

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: Clone + HttpClientExt + 'static,
{
    type Client = Client<T>;

    const MAX_DOCUMENTS: usize = 1024;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        let model = model.into();
        let ndims = dims.or_else(|| model_default_ndims(&model)).unwrap_or(768);
        Self::new(client.clone(), model, ndims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    /// <https://ai.google.dev/api/embeddings#batch_embed_contents-SHELL>
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();

        // Google batch embed requests. See docstrings for API ref link.
        let requests: Vec<_> = documents
            .iter()
            .map(|doc| {
                json!({
                    "model": format!("models/{}", self.model),
                    "content": json!({
                        "parts": [json!({
                            "text": doc.to_string()
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

        // A non-success status may carry an error body that does not match the
        // `ApiResponse` shape (Gemini nests its error under `error`, with no
        // top-level `message`), so preserve it verbatim before trying to
        // deserialize the success envelope — otherwise it would surface as a
        // `JsonError` and the provider_response_* helpers would lose it.
        if !status.is_success() {
            return Err(EmbeddingError::from_http_response(
                status,
                String::from_utf8_lossy(&body),
            ));
        }

        match serde_json::from_slice::<ApiResponse<gemini_api_types::EmbeddingResponse>>(&body)? {
            ApiResponse::Ok(response) => {
                let docs = documents
                    .into_iter()
                    .zip(response.embeddings)
                    .map(|(document, embedding)| embeddings::Embedding {
                        document,
                        vec: embedding
                            .values
                            .into_iter()
                            .filter_map(|n| n.as_f64())
                            .collect(),
                    })
                    .collect();

                Ok(docs)
            }
            ApiResponse::Err(err) => {
                tracing::warn!(message = %err.message, "provider returned an error response");
                Err(EmbeddingError::from_http_response(
                    status,
                    String::from_utf8_lossy(&body),
                ))
            }
        }
    }
}

// =================================================================
// Gemini API Types
// =================================================================
/// Rust Implementation of the Gemini Types from [Gemini API Reference](https://ai.google.dev/api/embeddings)
mod gemini_api_types {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    pub struct EmbeddingResponse {
        pub embeddings: Vec<EmbeddingValues>,
    }

    #[derive(Debug, Deserialize)]
    pub struct EmbeddingValues {
        #[serde(default)]
        pub values: Vec<serde_json::Number>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_values_deserializes_without_empty_values_field() {
        let values: gemini_api_types::EmbeddingValues =
            serde_json::from_str("{}").expect("empty embedding values should deserialize");
        assert!(values.values.is_empty());
    }

    #[test]
    fn test_model_default_ndims_lookup() {
        assert_eq!(model_default_ndims(EMBEDDING_001), Some(3072));
        assert_eq!(model_default_ndims(EMBEDDING_004), Some(768));
        assert_eq!(model_default_ndims("unknown-model"), None);
    }

    #[test]
    fn test_make_resolves_default_dims() {
        let client = Client::new("test_key").unwrap();

        // EMBEDDING_001 defaults to 3072
        let model =
            <EmbeddingModel as embeddings::EmbeddingModel>::make(&client, EMBEDDING_001, None);
        assert_eq!(embeddings::EmbeddingModel::ndims(&model), 3072);

        // EMBEDDING_004 defaults to 768
        let model =
            <EmbeddingModel as embeddings::EmbeddingModel>::make(&client, EMBEDDING_004, None);
        assert_eq!(embeddings::EmbeddingModel::ndims(&model), 768);

        // Unknown model falls back to 768
        let model = <EmbeddingModel as embeddings::EmbeddingModel>::make(
            &client,
            "some-future-model",
            None,
        );
        assert_eq!(embeddings::EmbeddingModel::ndims(&model), 768);
    }

    #[test]
    fn test_make_respects_explicit_dims() {
        let client = Client::new("test_key").unwrap();

        let model =
            <EmbeddingModel as embeddings::EmbeddingModel>::make(&client, EMBEDDING_001, Some(256));
        assert_eq!(embeddings::EmbeddingModel::ndims(&model), 256);
    }

    #[test]
    fn test_new_uses_provided_ndims() {
        let client = Client::new("test_key").unwrap();

        let model = EmbeddingModel::new(client, EMBEDDING_001, 512);
        assert_eq!(embeddings::EmbeddingModel::ndims(&model), 512);
    }

    #[tokio::test]
    async fn embedding_non_success_preserves_status_and_body() {
        use crate::client::embeddings::EmbeddingsClient;
        use crate::embeddings::EmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        // A realistic Gemini error body: the error is nested under `error` with no
        // top-level `message`, so it does NOT match the `ApiResponse` envelope. The
        // non-success status guard must preserve it verbatim (otherwise it would
        // surface as a `JsonError`).
        let body =
            r#"{"error":{"code":503,"message":"service unavailable","status":"UNAVAILABLE"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.embedding_model(EMBEDDING_001);

        let error = model
            .embed_texts(vec!["hello".to_string()])
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, EmbeddingError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn embedding_2xx_error_envelope_preserves_status_and_body() {
        use crate::client::embeddings::EmbeddingsClient;
        use crate::embeddings::EmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        // 200 OK whose body deserializes to `ApiResponse::Err` (requires a top-level
        // `message`; absence of `embeddings` keeps it off the success variant).
        let body =
            r#"{"message":"boom","error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
        let http_client = RecordingHttpClient::new(body); // 200 OK
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.embedding_model(EMBEDDING_001);

        let error = model
            .embed_texts(vec!["hello".to_string()])
            .await
            .expect_err("should fail with provider error envelope");

        match &error {
            EmbeddingError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
