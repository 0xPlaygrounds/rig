use crate::embeddings::{self, EmbeddingError};
use serde::Deserialize;
use serde_json::json;

/// Cohere's error envelope, which can arrive with a 2xx status.
#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub message: String,
}

/// Either a successful payload or Cohere's error envelope.
///
/// Moved here from the deleted `cohere::client` module; it is a wire type the
/// [`super::functions`] embedding path parses through.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default)]
    pub response_type: Option<String>,
    pub id: String,
    pub embeddings: Vec<Vec<serde_json::Number>>,
    pub texts: Vec<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
}

#[derive(Deserialize)]
pub struct Meta {
    pub api_version: ApiVersion,
    pub billed_units: BilledUnits,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Deserialize)]
pub struct ApiVersion {
    pub version: String,
    #[serde(default)]
    pub is_deprecated: Option<bool>,
    #[serde(default)]
    pub is_experimental: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct BilledUnits {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub search_units: u32,
    #[serde(default)]
    pub classifications: u32,
}

impl std::fmt::Display for BilledUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input tokens: {}\nOutput tokens: {}\nSearch units: {}\nClassifications: {}",
            self.input_tokens, self.output_tokens, self.search_units, self.classifications
        )
    }
}

/// Build the serialized `/v1/embed` request body. Pure; shared by the trait
/// path and [`super::functions::embed`].
pub(crate) fn build_embedding_body(
    model: &str,
    input_type: &str,
    texts: &[String],
) -> Result<Vec<u8>, EmbeddingError> {
    let body = json!({
        "model": model,
        "texts": texts,
        "input_type": input_type
    });
    Ok(serde_json::to_vec(&body)?)
}

/// Parse a `/v1/embed` response into the normalized
/// [`embeddings::EmbeddingResponse`], zipping vectors back onto
/// `documents`. Pure; shared by the trait path and
/// [`super::functions::embed`]. Usage is taken from `meta.billed_units`
/// (input tokens; Cohere reports no total).
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
            let mut usage = crate::completion::Usage::new();
            match &response.meta {
                Some(meta) => {
                    tracing::info!(target: "rig",
                        "Cohere embeddings billed units: {}",
                        meta.billed_units,
                    );
                    usage.input_tokens = meta.billed_units.input_tokens as u64;
                    usage.total_tokens = meta.billed_units.input_tokens as u64;
                }
                None => tracing::info!(target: "rig",
                    "Cohere embeddings billed units: n/a",
                ),
            };

            if response.embeddings.len() != documents.len() {
                return Err(EmbeddingError::DocumentError(
                    format!(
                        "Expected {} embeddings, got {}",
                        documents.len(),
                        response.embeddings.len()
                    )
                    .into(),
                ));
            }

            let embeddings = response
                .embeddings
                .into_iter()
                .zip(documents)
                .map(|(embedding, document)| embeddings::Embedding {
                    document,
                    vec: embedding.into_iter().filter_map(|n| n.as_f64()).collect(),
                })
                .collect();
            Ok(embeddings::EmbeddingResponse { embeddings, usage })
        }
        ApiResponse::Err(error) => {
            tracing::warn!(
                message = %error.message,
                "Cohere returned an error response"
            );
            Err(EmbeddingError::from_http_response(status, body.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embed_config() -> crate::providers::cohere::functions::EmbeddingConfig {
        crate::providers::cohere::functions::EmbeddingConfig::new(
            crate::providers::cohere::EMBED_ENGLISH_V3,
        )
        .with_api_key("test-key")
    }

    #[tokio::test]
    async fn embeddings_non_success_preserves_status_and_body() {
        use crate::http_runtime::HttpRuntime;
        use crate::providers::cohere::functions;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let rt = HttpRuntime::recording(http_client);

        let error = functions::embed(&embed_config(), &rt, vec!["hello".to_string()])
            .await
            .expect_err("should fail with non-success status");

        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn embeddings_2xx_error_envelope_preserves_status_and_body() {
        use crate::http_runtime::HttpRuntime;
        use crate::providers::cohere::functions;
        use crate::test_utils::RecordingHttpClient;

        // Deserializes to `ApiResponse::Err(ApiErrorResponse { message })` on a 200 OK.
        let body = r#"{"message":"boom"}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::new(body));

        let error = functions::embed(&embed_config(), &rt, vec!["hello".to_string()])
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
