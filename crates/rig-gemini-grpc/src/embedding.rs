// ================================================================
//! Google Gemini gRPC embedding wire conversions.
//!
//! Embedding model identifiers plus the single-document `EmbedContent` RPC
//! that [`crate::functions::embed`] loops over. Drive embeddings through
//! [`crate::functions`], not this module.
// ================================================================

/// `text-embedding-004` embedding model
pub const EMBEDDING_004: &str = "text-embedding-004";

/// Default `output_dimensionality` for [`EMBEDDING_004`] when a config leaves
/// [`EmbeddingConfig::ndims`](crate::functions::EmbeddingConfig::ndims) unset.
pub const DEFAULT_NDIMS: usize = 768;

use rig_core::embeddings::{self, EmbeddingError};

use super::Client;
use super::proto::{self, EmbedContentRequest};

/// Embed one document over the `EmbedContent` RPC.
///
/// The single source of truth is [`crate::functions::embed`], which calls
/// this once per document (the gRPC embedding API is single-document).
pub(crate) async fn embed_one(
    client: &Client,
    model: &str,
    output_dimensionality: Option<usize>,
    doc: String,
) -> Result<embeddings::Embedding, EmbeddingError> {
    let mut grpc_client = client
        .grpc_client()
        .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
    let request = EmbedContentRequest {
        model: format!("models/{model}"),
        content: Some(proto::Content {
            parts: vec![proto::Part {
                data: Some(proto::part::Data::Text(doc.clone())),
                thought: false,
                thought_signature: Vec::new(),
                part_metadata: None,
            }],
            role: String::new(),
        }),
        task_type: None,
        title: None,
        output_dimensionality: output_dimensionality.map(|n| n as i32),
    };

    let response = grpc_client
        .embed_content(request)
        .await
        .map_err(rpc_error)?
        .into_inner();

    match response.embedding {
        Some(embedding) => Ok(embeddings::Embedding {
            document: doc,
            vec: embedding.values.into_iter().map(|v| v as f64).collect(),
        }),
        None => Err(EmbeddingError::ResponseError(
            "No embedding in response".to_string(),
        )),
    }
}

// Map a failed gRPC call into an `EmbeddingError` that preserves the provider's
// error payload verbatim. gRPC is a non-HTTP transport, so there is no
// `http::StatusCode`; the body is preserved via `from_provider_body` (status:
// None) rather than a Rig-prefixed `ProviderError` diagnostic. Note: tonic does
// not distinguish a server-returned gRPC error from a transport/connection
// failure, so a pure connection error is also preserved here rather than gated
// out as a Rig diagnostic the way Bedrock's typed service errors are.
fn rpc_error(status: tonic::Status) -> EmbeddingError {
    EmbeddingError::from_provider_body(status.to_string())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn rpc_error_preserves_status_text_without_http_status() {
        let status = tonic::Status::unavailable("boom");
        let expected = status.to_string();

        let err = rpc_error(status);

        // The raw provider error text is preserved verbatim, and there is no
        // HTTP status because gRPC is a non-HTTP transport.
        assert_eq!(err.provider_response_body(), Some(expected.as_str()));
        assert_eq!(err.provider_response_status(), None);
    }
}
