// ================================================================
//! Google Gemini gRPC Embedding Integration
// ================================================================

/// `text-embedding-004` embedding model
pub const EMBEDDING_004: &str = "text-embedding-004";

use crate::embeddings::{self, EmbeddingError};

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
        documents: impl IntoIterator<Item = String> + crate::wasm_compat::WasmCompatSend,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents_vec: Vec<String> = documents.into_iter().collect();
        let mut embeddings = Vec::new();

        let mut grpc_client = self
            .client
            .ext()
            .grpc_client()
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;

        for doc in documents_vec {
            let request = EmbedContentRequest {
                model: format!("models/{}", self.model),
                content: Some(proto::Content {
                    parts: vec![proto::Part {
                        thought: None,
                        thought_signature: None,
                        data: Some(proto::part::Data::Text(doc.clone())),
                    }],
                    role: None,
                }),
                task_type: None,
                title: None,
                output_dimensionality: Some(self.ndims as i32),
            };

            let response = grpc_client
                .embed_content(request)
                .await
                .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
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
