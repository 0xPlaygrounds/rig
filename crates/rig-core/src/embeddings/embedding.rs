//! The module defines the [Embedding] struct, which represents a single document embedding,
//! and [EmbeddingResponse], which pairs a batch of embeddings with provider token usage.
//!
//! Finally, the module defines the [EmbeddingError] enum, which represents various errors that
//! can occur during embedding generation or processing.

use crate::{completion::Usage, http_client, provider_response};
use serde::{Deserialize, Serialize};

/// Errors returned by embedding models.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EmbeddingError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// URL construction or parsing failed while preparing a provider request.
    #[error("UrlError: {0}")]
    UrlError(#[from] url::ParseError),

    #[cfg(not(target_family = "wasm"))]
    /// Error processing the document for embedding
    #[error("DocumentError: {0}")]
    DocumentError(Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error processing the document for embedding
    #[error("DocumentError: {0}")]
    DocumentError(Box<dyn std::error::Error + 'static>),

    /// Error parsing the completion response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// The provider does not support an embedding request parameter configured on the model.
    #[error("{provider} embeddings do not support the `{parameter}` parameter")]
    UnsupportedParameter {
        /// Provider whose embedding API rejected the parameter.
        provider: &'static str,
        /// Unsupported request parameter.
        parameter: &'static str,
    },

    /// A provider request parameter was configured with a value outside the
    /// provider's supported range.
    #[error("{provider} embeddings require `{parameter}` {requirement}")]
    InvalidParameterValue {
        /// Provider whose embedding API constrains the parameter.
        provider: &'static str,
        /// Request parameter with the invalid value.
        parameter: &'static str,
        /// Concise description of the accepted values.
        requirement: &'static str,
    },

    /// Rig cannot decode the requested provider response encoding.
    #[error("Rig cannot decode {provider} embedding responses encoded as `{encoding_format}`")]
    UnsupportedResponseEncoding {
        /// Provider whose response encoding was requested.
        provider: &'static str,
        /// Response encoding that Rig cannot decode.
        encoding_format: &'static str,
    },

    /// A provider that guarantees embedding usage omitted it from the response.
    #[error("{provider} embedding response omitted required usage")]
    MissingUsage {
        /// Provider whose response omitted usage.
        provider: &'static str,
    },

    /// Error returned by the embedding model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),

    /// Raw error response preserved from the embedding model provider
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
}

crate::provider_response::impl_provider_response_helpers!(EmbeddingError);

/// Response from an embedding request containing the embeddings and token usage.
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// The embeddings returned by the provider, one per input text.
    pub embeddings: Vec<Embedding>,
    /// Token usage for this embedding request.
    pub usage: Usage,
}

/// Struct that holds a single document and its embedding.
#[derive(Clone, Default, Deserialize, Serialize, Debug)]
pub struct Embedding {
    /// The document that was embedded. Used for debugging.
    pub document: String,
    /// The embedding vector
    pub vec: Vec<f64>,
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.document == other.document
    }
}

impl Eq for Embedding {}

#[cfg(test)]
mod provider_response_tests {
    use super::*;
    use http::StatusCode;

    #[test]
    fn embedding_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"message":"rate limited"}}"#;
        let error = EmbeddingError::ProviderResponse(provider_response::ProviderResponseError {
            status: None,
            body: body.to_string(),
        });

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "rate limited" } }))
        );
    }

    #[test]
    fn embedding_error_provider_error_is_not_a_provider_response() {
        let error = EmbeddingError::ProviderError("internal diagnostic".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }

    #[test]
    fn embedding_error_provider_response_helpers_with_http_non_success() {
        let body = r#"{"error":{"message":"bad request"}}"#;
        let error = EmbeddingError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
            StatusCode::BAD_REQUEST,
            body.to_string(),
        ));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            error.provider_response_json().expect("valid JSON"),
            Some(serde_json::json!({ "error": { "message": "bad request" } }))
        );
    }

    #[test]
    fn embedding_error_provider_response_helpers_with_preserved_plain_text_body() {
        let error = EmbeddingError::ProviderResponse(provider_response::ProviderResponseError {
            status: None,
            body: "not json".to_string(),
        });

        assert_eq!(error.provider_response_body(), Some("not json"));
        assert!(error.provider_response_json().is_err());
    }

    #[test]
    fn embedding_error_provider_response_helpers_with_unrelated_variant() {
        let error = EmbeddingError::ResponseError("parse failed".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(error.provider_response_json().expect("no body"), None);
    }
}
