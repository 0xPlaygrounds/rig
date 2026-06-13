//! The module defines the [EmbeddingModel] trait, which represents an embedding model that can
//! generate embeddings for documents.
//!
//! The module also defines the [Embedding] struct, which represents a single document embedding.
//!
//! Finally, the module defines the [EmbeddingError] enum, which represents various errors that
//! can occur during embedding generation or processing.

use crate::{
    completion::Usage,
    http_client, provider_response,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
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

    /// Error returned by the embedding model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),

    /// Raw error response preserved from the embedding model provider
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
}

crate::provider_response::impl_provider_response_helpers!(EmbeddingError);

/// Trait for embedding models that can generate embeddings for documents.
pub trait EmbeddingModel: WasmCompatSend + WasmCompatSync {
    /// The maximum number of documents that can be embedded in a single request.
    const MAX_DOCUMENTS: usize;

    /// Provider client type used to construct this embedding model.
    type Client;

    /// Construct a model handle from a provider client, model identifier, and optional dimensions.
    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed multiple text documents in a single request
    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + WasmCompatSend;

    /// Embed a single text document.
    fn embed_text(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + WasmCompatSend {
        async {
            let mut embeddings = self.embed_texts(vec![text.to_string()]).await?;
            embeddings.pop().ok_or_else(|| {
                EmbeddingError::ResponseError(
                    "embedding provider returned an empty response for embed_text".to_string(),
                )
            })
        }
    }

    /// Embed multiple text documents in a single request and return token usage.
    ///
    /// The default implementation delegates to [`EmbeddingModel::embed_texts`] and returns
    /// zero-valued usage. Providers that expose usage information from their embedding API
    /// should override this method.
    fn embed_texts_with_usage(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, EmbeddingError>> + WasmCompatSend
    {
        async {
            let embeddings = self.embed_texts(texts).await?;
            Ok(EmbeddingResponse {
                embeddings,
                usage: Usage::default(),
            })
        }
    }

    /// Embed a single text document and return token usage.
    ///
    /// The default implementation delegates to
    /// [`EmbeddingModel::embed_texts_with_usage`].
    fn embed_text_with_usage(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, EmbeddingError>> + WasmCompatSend
    {
        async {
            let response = self.embed_texts_with_usage(vec![text.to_string()]).await?;
            if response.embeddings.is_empty() {
                return Err(EmbeddingError::ResponseError(
                    "embedding provider returned an empty response for embed_text_with_usage"
                        .to_string(),
                ));
            }
            Ok(response)
        }
    }
}

/// Response from an embedding request containing the embeddings and token usage.
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// The embeddings returned by the provider, one per input text.
    pub embeddings: Vec<Embedding>,
    /// Token usage for this embedding request.
    pub usage: Usage,
}

/// Trait for embedding models that can generate embeddings for images.
pub trait ImageEmbeddingModel: Clone + WasmCompatSend + WasmCompatSync {
    /// The maximum number of images that can be embedded in a single request.
    const MAX_DOCUMENTS: usize;

    /// The number of dimensions in the embedding vector.
    fn ndims(&self) -> usize;

    /// Embed multiple images in a single request from bytes.
    ///
    /// Implementations should preserve input order in the returned embeddings.
    fn embed_images(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + Send;

    /// Embed a single image from bytes.
    fn embed_image<'a>(
        &'a self,
        bytes: &'a [u8],
    ) -> impl std::future::Future<Output = Result<Embedding, EmbeddingError>> + WasmCompatSend {
        async move {
            let mut embeddings = self.embed_images(vec![bytes.to_owned()]).await?;
            embeddings.pop().ok_or_else(|| {
                EmbeddingError::ResponseError(
                    "embedding provider returned an empty response for embed_image".to_string(),
                )
            })
        }
    }
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
