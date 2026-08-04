use super::api::ApiResponse;
use super::completion::Usage;
use crate::embeddings;
use crate::embeddings::EmbeddingError;
use serde::{Deserialize, Serialize};

// ================================================================
// OpenAI Embedding API
// ================================================================
/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
struct CompatibleEmbeddingResponse {
    #[serde(rename = "object")]
    _object: String,
    pub data: Vec<EmbeddingData>,
    #[serde(rename = "model")]
    _model: String,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Provider-specific spelling for an embedding dimension request field.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDimensions {
    /// Serialize the value as the OpenAI-compatible `dimensions` field.
    Dimensions(usize),
    /// Serialize the value as Mistral's `output_dimension` field.
    OutputDimension(usize),
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[derive(Debug, Serialize)]
struct CompatibleEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<serde_json::Number>,
    pub index: usize,
}

/// Build the serialized OpenAI-compatible embeddings request body. Pure.
///
/// The single source of truth for OpenAI-compatible embeddings request bytes;
/// [`super::functions::build_embedding_request`] wraps it with transport
/// concerns.
pub(crate) fn build_embedding_body(
    model: &str,
    texts: &[String],
    dimensions: Option<EmbeddingDimensions>,
    encoding_format: Option<EncodingFormat>,
    user: Option<&str>,
) -> Result<Vec<u8>, EmbeddingError> {
    let (dimensions, output_dimension) = match dimensions {
        Some(EmbeddingDimensions::Dimensions(value)) => (Some(value), None),
        Some(EmbeddingDimensions::OutputDimension(value)) => (None, Some(value)),
        None => (None, None),
    };
    Ok(serde_json::to_vec(&CompatibleEmbeddingRequest {
        model,
        input: texts,
        dimensions,
        output_dimension,
        encoding_format,
        user,
    })?)
}

/// Parse an OpenAI-compatible embeddings response into the normalized
/// [`embeddings::EmbeddingResponse`], zipping vectors back onto
/// `documents`. Pure.
///
/// `provider` names the provider in usage/parse errors; `requires_usage`
/// rejects success payloads that omit the `usage` object.
pub(crate) fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
    provider: &'static str,
    requires_usage: bool,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
    if !status.is_success() {
        return Err(EmbeddingError::from_http_response(status, body.to_string()));
    }
    let parsed: ApiResponse<CompatibleEmbeddingResponse> = serde_json::from_str(body)?;
    match parsed {
        ApiResponse::Ok(response) => {
            tracing::info!(target: "rig",
                "embedding token usage: {:?}",
                response.usage
            );

            if response.data.len() != documents.len() {
                return Err(EmbeddingError::ResponseError(
                    "Response data length does not match input length".into(),
                ));
            }

            let usage = match response.usage {
                Some(usage) => crate::completion::Usage {
                    input_tokens: usage.prompt_tokens as u64,
                    output_tokens: 0,
                    total_tokens: usage.total_tokens as u64,
                    cached_input_tokens: usage
                        .prompt_tokens_details
                        .as_ref()
                        .map_or(0, |details| details.cached_tokens as u64),
                    cache_creation_input_tokens: 0,
                    tool_use_prompt_tokens: 0,
                    reasoning_tokens: 0,
                },
                None if requires_usage => {
                    return Err(EmbeddingError::MissingUsage { provider });
                }
                None => crate::completion::Usage::new(),
            };

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

            Ok(embeddings::EmbeddingResponse { embeddings, usage })
        }
        ApiResponse::Err(err) => {
            tracing::warn!(message = %err.message, "provider returned an error response");
            Err(EmbeddingError::from_http_response(status, body.to_string()))
        }
    }
}

/// The native embedding width of a known OpenAI embedding model, if any.
///
/// OpenAI's `dimensions` request field defaults to the model's native width, so
/// this is only needed by callers that want to state the width explicitly (or
/// validate a requested one). `text-embedding-ada-002` rejects the field
/// entirely.
pub fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        TEXT_EMBEDDING_3_LARGE => Some(3_072),
        TEXT_EMBEDDING_3_SMALL | TEXT_EMBEDDING_ADA_002 => Some(1_536),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESPONSE_BODY: &str = r#"{
        "object": "list",
        "model": "text-embedding-3-small",
        "usage": { "prompt_tokens": 4, "total_tokens": 4 },
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
    }"#;

    #[test]
    fn known_models_report_their_native_width() {
        assert_eq!(
            model_dimensions_from_identifier(TEXT_EMBEDDING_3_LARGE),
            Some(3_072)
        );
        assert_eq!(
            model_dimensions_from_identifier(TEXT_EMBEDDING_3_SMALL),
            Some(1_536)
        );
        assert_eq!(model_dimensions_from_identifier("unknown-model"), None);
    }

    #[test]
    fn embedding_body_carries_model_input_and_optional_fields() {
        let texts = vec!["hello".to_string()];
        let body = build_embedding_body(
            TEXT_EMBEDDING_3_SMALL,
            &texts,
            Some(EmbeddingDimensions::Dimensions(1_536)),
            Some(EncodingFormat::Float),
            Some("user-123"),
        )
        .expect("body should build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], "text-embedding-3-small");
        assert_eq!(value["input"], serde_json::json!(["hello"]));
        assert_eq!(value["dimensions"], serde_json::json!(1_536));
        assert_eq!(value["encoding_format"], serde_json::json!("float"));
        assert_eq!(value["user"], serde_json::json!("user-123"));
        assert!(value.get("output_dimension").is_none());
    }

    #[test]
    fn mistral_dimension_spelling_uses_output_dimension() {
        let texts = vec!["hello".to_string()];
        let body = build_embedding_body(
            "mistral-embed",
            &texts,
            Some(EmbeddingDimensions::OutputDimension(512)),
            None,
            None,
        )
        .expect("body should build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["output_dimension"], serde_json::json!(512));
        assert!(value.get("dimensions").is_none());
    }

    #[test]
    fn parse_response_zips_vectors_onto_documents_and_carries_usage() {
        let response = parse_embedding_response(
            http::StatusCode::OK,
            RESPONSE_BODY,
            vec!["hello".to_string()],
            "openai",
            true,
        )
        .expect("response should parse");

        assert_eq!(response.usage.input_tokens, 4);
        assert_eq!(response.usage.total_tokens, 4);
        assert_eq!(response.embeddings.len(), 1);
        let first = response.embeddings.first().expect("one embedding");
        assert_eq!(first.document, "hello");
        assert_eq!(first.vec, vec![0.1, 0.2]);
    }

    #[test]
    fn parse_response_rejects_length_mismatch() {
        let error = parse_embedding_response(
            http::StatusCode::OK,
            RESPONSE_BODY,
            vec!["hello".to_string(), "world".to_string()],
            "openai",
            true,
        )
        .expect_err("length mismatch should error");
        assert!(matches!(error, EmbeddingError::ResponseError(_)));
    }

    #[test]
    fn parse_response_requires_usage_when_the_provider_promises_it() {
        let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1] }]
        }"#;

        let error = parse_embedding_response(
            http::StatusCode::OK,
            body,
            vec!["hello".to_string()],
            "openai",
            true,
        )
        .expect_err("missing usage should error");
        assert!(matches!(
            error,
            EmbeddingError::MissingUsage { provider: "openai" }
        ));

        // Providers that do not promise usage get a zeroed usage instead.
        let response = parse_embedding_response(
            http::StatusCode::OK,
            body,
            vec!["hello".to_string()],
            "openai",
            false,
        )
        .expect("optional usage should parse");
        assert_eq!(response.usage.total_tokens, 0);
    }

    #[test]
    fn public_openai_embedding_response_requires_usage() {
        let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1] }]
        }"#;

        assert!(serde_json::from_str::<EmbeddingResponse>(body).is_err());
    }

    #[test]
    fn parse_response_preserves_raw_provider_error_json_on_api_error_envelope() {
        let body = r#"{"message":"embedding quota exceeded","type":"insufficient_quota"}"#;

        let error = parse_embedding_response(
            http::StatusCode::ACCEPTED,
            body,
            vec!["hello".to_string()],
            "openai",
            true,
        )
        .expect_err("error envelope should fail");

        match &error {
            EmbeddingError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::ACCEPTED));
                let json = error
                    .provider_response_json()
                    .expect("raw body should be valid JSON")
                    .expect("parsed JSON should be present");
                assert_eq!(json["type"], "insufficient_quota");
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"invalid api key","type":"invalid_request_error"}}"#;

        let error = parse_embedding_response(
            http::StatusCode::UNAUTHORIZED,
            body,
            vec!["hello".to_string()],
            "openai",
            true,
        )
        .expect_err("non-success status should fail");

        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::UNAUTHORIZED)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
