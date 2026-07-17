// ================================================================
//! Google Gemini gRPC Embedding Integration
// ================================================================

/// `text-embedding-004` embedding model
pub const EMBEDDING_004: &str = "text-embedding-004";

use rig_core::embeddings::{self, EmbeddingError};

use super::Client;
use super::proto::{self, EmbedContentRequest};

#[derive(Clone, Debug)]
pub struct EmbeddingModel {
    client: Client,
    model: String,
    ndims: usize,
}

impl EmbeddingModel {
    pub fn new(client: Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self {
            client,
            model: model.into(),
            ndims: dims.unwrap_or(768), // Default embedding size for text-embedding-004
        }
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 100;

    type Client = super::Client;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + rig_core::wasm_compat::WasmCompatSend,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents_vec: Vec<String> = documents.into_iter().collect();
        let mut embeddings = Vec::new();

        let mut grpc_client = self
            .client
            .grpc_client()
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;

        for doc in documents_vec {
            let request = EmbedContentRequest {
                model: format!("models/{}", self.model),
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
                output_dimensionality: Some(self.ndims as i32),
            };

            let response = grpc_client
                .embed_content(request)
                .await
                .map_err(rpc_error)?
                .into_inner();

            if let Some(embedding) = response.embedding {
                embeddings.push(embeddings::Embedding {
                    document: doc,
                    vec: embedding.values.into_iter().map(|v| v as f64).collect(),
                });
            } else {
                return Err(EmbeddingError::ResponseError(
                    "No embedding in response".to_string(),
                ));
            }
        }

        Ok(embeddings)
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
