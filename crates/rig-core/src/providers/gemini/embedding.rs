// ================================================================
//! Google Gemini Embeddings Integration
//! From [Gemini API Reference](https://ai.google.dev/api/embeddings)
// ================================================================

use serde_json::json;

use super::ApiResponse;
use crate::embeddings::{self, EmbeddingError};

/// `gemini-embedding-001` embedding model (3072 dimensions by default)
pub const EMBEDDING_001: &str = "gemini-embedding-001";
/// `text-embedding-004` embedding model (768 dimensions by default)
pub const EMBEDDING_004: &str = "text-embedding-004";

/// Returns the default output dimensionality for known Gemini embedding models.
///
/// Pure lookup table. Callers that want the provider's documented default
/// dimensionality can feed it to
/// [`EmbeddingConfig::with_dimensions`](super::functions::EmbeddingConfig::with_dimensions);
/// leaving it unset lets Gemini apply the same default server-side.
///
/// See <https://ai.google.dev/gemini-api/docs/models#gemini-embedding>
pub fn model_default_ndims(model: &str) -> Option<usize> {
    match model {
        EMBEDDING_001 => Some(3072),
        EMBEDDING_004 => Some(768),
        _ => None,
    }
}

/// Build the serialized `batchEmbedContents` request body. Pure; used by
/// [`super::functions::embed`].
///
/// `output_dimensionality` is included per entry when `Some`.
pub(crate) fn build_embedding_body(
    model: &str,
    texts: &[String],
    output_dimensionality: Option<usize>,
) -> Result<Vec<u8>, EmbeddingError> {
    let requests: Vec<_> = texts
        .iter()
        .map(|doc| {
            let mut entry = json!({
                "model": format!("models/{model}"),
                "content": json!({
                    "parts": [json!({
                        "text": doc.to_string()
                    })]
                }),
            });
            if let (Some(ndims), Some(object)) = (output_dimensionality, entry.as_object_mut()) {
                object.insert("output_dimensionality".to_string(), json!(ndims));
            }
            entry
        })
        .collect();

    let request_body = json!({ "requests": requests });

    if let Ok(pretty_body) = serde_json::to_string_pretty(&request_body) {
        tracing::trace!(
            target: "rig::embedding",
            "Sending embedding request to Gemini API {pretty_body}"
        );
    }

    Ok(serde_json::to_vec(&request_body)?)
}

/// Parse a `batchEmbedContents` response into the normalized
/// [`embeddings::EmbeddingResponse`], zipping vectors back onto
/// `documents`. Pure; used by [`super::functions::embed`].
/// Gemini reports no embedding usage.
pub(crate) fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
    // Preserve non-success bodies before deserialization because providers
    // may return empty, non-JSON, or otherwise unexpected error payloads.
    if !status.is_success() {
        return Err(EmbeddingError::from_http_response(status, body.to_string()));
    }

    match serde_json::from_str::<ApiResponse<gemini_api_types::EmbeddingResponse>>(body)? {
        ApiResponse::Ok(response) => {
            let embeddings = documents
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
            Ok(embeddings::EmbeddingResponse {
                embeddings,
                usage: crate::completion::Usage::new(),
            })
        }
        ApiResponse::Err(err) => {
            tracing::warn!(message = %err.error.message, "provider returned an error response");
            Err(EmbeddingError::from_http_response(status, body.to_string()))
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

    #[tokio::test]
    async fn embedding_non_success_preserves_status_and_body() {
        use crate::http_runtime::HttpRuntime;
        use crate::providers::gemini::functions;
        use crate::test_utils::RecordingHttpClient;

        // The non-success status guard preserves the raw provider body without
        // depending on its envelope shape.
        let body =
            r#"{"error":{"code":503,"message":"service unavailable","status":"UNAVAILABLE"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let rt = HttpRuntime::recording(http_client);
        let cfg = functions::EmbeddingConfig::new(EMBEDDING_001).with_api_key("test-key");

        let error = functions::embed(&cfg, &rt, vec!["hello".to_string()])
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
        use crate::http_runtime::HttpRuntime;
        use crate::providers::gemini::functions;
        use crate::test_utils::RecordingHttpClient;

        // 200 OK carrying Gemini's standard nested error envelope.
        let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
        let http_client = RecordingHttpClient::new(body); // 200 OK
        let rt = HttpRuntime::recording(http_client);
        let cfg = functions::EmbeddingConfig::new(EMBEDDING_001).with_api_key("test-key");

        let error = functions::embed(&cfg, &rt, vec!["hello".to_string()])
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
