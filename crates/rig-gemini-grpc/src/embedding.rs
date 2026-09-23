//! Text embeddings through Gemini's gRPC API.
//!
//! ```no_run
//! use rig_gemini_grpc::{Client, embedding::EMBEDDING_004};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let model = Client::new("API_KEY").await?.embedding(EMBEDDING_004, None);
//! # Ok(())
//! # }
//! ```

/// `text-embedding-004` embedding model
pub const EMBEDDING_004: &str = "text-embedding-004";

use rig_core::embeddings;
use rig_core::error::ProviderError;

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
    /// Embeds texts sequentially and returns native responses in input order.
    /// Stops at the first client or RPC error without returning partial results.
    pub async fn raw_embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + rig_core::wasm_compat::WasmCompatSend,
    ) -> Result<Vec<proto::EmbedContentResponse>, ProviderError> {
        let documents_vec: Vec<String> = documents.into_iter().collect();
        let mut responses = Vec::with_capacity(documents_vec.len());

        let mut grpc_client = self
            .client
            .grpc_client()
            .map_err(|e| ProviderError::Provider(e.to_string()))?;

        for doc in documents_vec {
            let request = EmbedContentRequest {
                model: format!("models/{}", self.model),
                content: Some(proto::Content {
                    parts: vec![proto::Part {
                        data: Some(proto::part::Data::Text(doc)),
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
    ) -> Result<embeddings::EmbeddingResponse, ProviderError> {
        rig_core::telemetry::instrument_modality::<rig_core::operation::Embedding, _>(
            super::completion::PROVIDER_NAME,
            &self.model,
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
                        return Err(ProviderError::Response(
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

/// Preserves tonic status display text as an error body with RPC code and retry
/// classification. Transport failures use the same representation.
fn rpc_error(status: &tonic::Status) -> ProviderError {
    ProviderError::from_provider_body(status.to_string())
        .with_provider_code(Some(super::completion::grpc_code_name(status.code())))
        .with_transient(Some(super::completion::transient_grpc_code(status.code())))
}

#[cfg(test)]
mod tests;
