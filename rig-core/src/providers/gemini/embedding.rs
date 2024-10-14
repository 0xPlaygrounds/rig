// ================================================================
// Google Gemini Embeddings
// ================================================================

use serde::Deserialize;
use serde_json::json;

use crate::embeddings::{self, EmbeddingError};

use super::{client::ApiResponse, Client};

/// `embedding-gecko-001` embedding model
pub const EMBEDDING_GECKO_001: &str = "embedding-gecko-001";
/// `embedding-001` embedding model
pub const EMBEDDING_001: &str = "embedding-001";
/// `text-embedding-004` embedding model
pub const EMBEDDING_004: &str = "text-embedding-004";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub embedding: EmbeddingValues,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingValues {
    pub values: Vec<f64>,
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    model: String,
    ndims: Option<usize>,
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: Option<usize>) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        match self.model.as_str() {
            EMBEDDING_GECKO_001 | EMBEDDING_001 => 768,
            EMBEDDING_004 => 1024,
            _ => 0, // Default to 0 for unknown models
        }
    }

    async fn embed_documents(
        &self,
        documents: Vec<String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let mut request_body = json!({
            "model": format!("models/{}", self.model),
            "content": {
                "parts": documents.iter().map(|doc| json!({ "text": doc })).collect::<Vec<_>>(),
            },
        });

        if let Some(ndims) = self.ndims {
            request_body["output_dimensionality"] = json!(ndims);
        }

        let response = self
            .client
            .post(&format!("/v1beta/models/{}:embedContent", self.model))
            .json(&request_body)
            .send()
            .await?
            .error_for_status()?
            .json::<ApiResponse<EmbeddingResponse>>()
            .await?;

        match response {
            ApiResponse::Ok(response) => {
                let chunk_size = self.ndims.unwrap_or_else(|| self.ndims());
                Ok(documents
                    .into_iter()
                    .zip(response.embedding.values.chunks(chunk_size))
                    .map(|(document, embedding)| embeddings::Embedding {
                        document,
                        vec: embedding.to_vec(),
                    })
                    .collect())
            }
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}
