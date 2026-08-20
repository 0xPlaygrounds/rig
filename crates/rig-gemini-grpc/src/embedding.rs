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

impl EmbeddingModel {
    /// Perform the requests and return Gemini's native gRPC answers — one
    /// `EmbedContentResponse` per input text, in input order, because
    /// `EmbedContent` takes one content per call — instead of the normalized
    /// [`embeddings::EmbeddingResponse`]. Same requests, transport, and error
    /// path as [`embeddings::EmbeddingModel::embed_texts_response`].
    pub async fn raw_embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + rig_core::wasm_compat::WasmCompatSend,
    ) -> Result<Vec<proto::EmbedContentResponse>, EmbeddingError> {
        let documents_vec: Vec<String> = documents.into_iter().collect();
        let mut responses = Vec::with_capacity(documents_vec.len());

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
                .map_err(|status| rpc_error(&status))?
                .into_inner();

            responses.push(response);
        }

        Ok(responses)
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    fn max_documents(&self) -> usize {
        100
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String> + rig_core::wasm_compat::WasmCompatSend,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        rig_core::telemetry::instrument_modality(
            super::completion::PROVIDER_NAME,
            &self.model,
            rig_core::telemetry::ModalityOperation::Embeddings,
            async {
                let documents_vec: Vec<String> = documents.into_iter().collect();
                let responses = self.raw_embed_texts(documents_vec.clone()).await?;
                let mut embeddings = Vec::with_capacity(responses.len());
                for (response, doc) in responses.into_iter().zip(documents_vec) {
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

                // gRPC: the native answers are prost messages, not JSON, and
                // `EmbedContent` reports no usage or response id. `raw` stays `Null`;
                // `raw_embed_texts` is the typed route.
                Ok(embeddings::EmbeddingResponse::new(
                    embeddings,
                    super::completion::PROVIDER_NAME,
                ))
            },
        )
        .await
    }
}

impl rig_core::client::ConstructEmbeddingModel<super::Client> for EmbeddingModel {
    fn construct(client: &super::Client, model: String, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model, dims)
    }
}

// Map a failed gRPC call into an `EmbeddingError` that preserves the provider's
// error payload verbatim. gRPC is a non-HTTP transport, so there is no
// `http::StatusCode`; the body is preserved via `from_provider_body` (status:
// None) rather than a Rig-prefixed `ProviderError` diagnostic. Note: tonic does
// not distinguish a server-returned gRPC error from a transport/connection
// failure, so a pure connection error is also preserved here rather than gated
// out as a Rig diagnostic the way Bedrock's typed service errors are.
fn rpc_error(status: &tonic::Status) -> EmbeddingError {
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

        let err = rpc_error(&status);

        // The raw provider error text is preserved verbatim, and there is no
        // HTTP status because gRPC is a non-HTTP transport.
        assert_eq!(err.provider_response_body(), Some(expected.as_str()));
        assert_eq!(err.provider_response_status(), None);
    }
}
