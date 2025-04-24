// ================================================================
//! Aliyun Embedding API Integration
//! Implementation of Aliyun embedding models for text vectorization
//! From https://help.aliyun.com/zh/model-studio/developer-reference/text-embedding-synchronous-api
// ================================================================

use serde_json::json;

use crate::embeddings::{self, EmbeddingError};

use super::client::{ApiResponse, Client};

// Available embedding models provided by Aliyun
pub const EMBEDDING_V1: &str = "text-embedding-v1";
pub const EMBEDDING_V2: &str = "text-embedding-v2";
pub const EMBEDDING_V3: &str = "text-embedding-v3";

/// Aliyun embedding model implementation
#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    model: String,
    ndims: Option<usize>,
}

impl EmbeddingModel {
    /// Creates a new instance of the Aliyun embedding model
    ///
    /// # Arguments
    /// * `client` - The Aliyun API client
    /// * `model` - The model identifier (e.g., "text-embedding-v1")
    /// * `ndims` - Optional custom dimension size for the embedding output
    pub fn new(client: Client, model: &str, ndims: Option<usize>) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}

impl EmbeddingModel {
    /// Returns the maximum number of documents supported by the model
    ///
    /// # Returns
    /// * For EMBEDDING_V3 model: 10 documents maximum
    /// * For other models: 25 documents maximum
    fn max_documents(&self) -> usize {
        match self.model.as_str() {
            EMBEDDING_V3 => 10,
            _ => 25,
        }
    }

    /// Returns the maximum number of tokens per document supported by the model
    ///
    /// # Returns
    /// * For EMBEDDING_V3 model: 8192 tokens per document
    /// * For other models: 2048 tokens per document
    fn max_tokens(&self) -> usize {
        match self.model.as_str() {
            EMBEDDING_V3 => 8192,
            _ => 2048,
        }
    }

    /// Validates if the document list meets the model's constraints
    ///
    /// # Validation Checks
    /// 1. Number of documents doesn't exceed model's maximum capacity
    /// 2. Each document's token count is within the model's token limit
    ///
    /// # Returns
    /// * `Ok(())` if validation passes
    /// * `Err(EmbeddingError)` with appropriate error message if validation fails
    fn validate_documents(&self, documents: &[String]) -> Result<(), EmbeddingError> {
        const AVG_CHARS_PER_TOKEN: usize = 4;

        if documents.len() > self.max_documents() {
            return Err(EmbeddingError::ProviderError(format!(
                "Model {} supports maximum {} documents",
                self.model,
                self.max_documents()
            )));
        }

        for (i, doc) in documents.iter().enumerate() {
            let estimated_tokens = doc.len() / AVG_CHARS_PER_TOKEN;
            if estimated_tokens > self.max_tokens() {
                return Err(EmbeddingError::ProviderError(format!(
                    "Document #{} exceeds maximum token limit of {}",
                    i + 1,
                    self.max_tokens()
                )));
            }
        }

        Ok(())
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 25;

    /// Returns the default embedding dimension for the current model
    ///
    /// # Returns
    /// * EMBEDDING_V1: 1536 dimensions
    /// * EMBEDDING_V2: 1536 dimensions
    /// * EMBEDDING_V3: 1024 dimensions (can be customized)
    /// * Unknown models: 0 dimensions
    fn ndims(&self) -> usize {
        match self.model.as_str() {
            EMBEDDING_V1 => 1536,
            EMBEDDING_V2 => 1536,

            // V3 model defaults to 1024 dimensions
            // Can be customized to [128, 256, 384, 512, 768, 1024]
            EMBEDDING_V3 => 1024,
            _ => 0, // Default to 0 for unknown models
        }
    }

    /// Generates embeddings for the provided text documents
    ///
    /// # Arguments
    /// * `documents` - Collection of text documents to embed
    ///
    /// # Returns
    /// * `Result<Vec<Embedding>, EmbeddingError>` - Vector of embeddings or error
    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();

        self.validate_documents(&documents)?;

        let request = json!({
            "model": self.model,
            "input": documents,
            "dimension": self.ndims.unwrap_or(self.ndims()),
            "encoding_format": "float",
        });

        tracing::info!("{}", serde_json::to_string_pretty(&request).unwrap());

        let response = self
            .client
            .post("/compatible-mode/v1/embeddings")
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<ApiResponse<aliyun_api_types::EmbeddingResponse>>()
            .await?;

        match response {
            ApiResponse::Ok(response) => {
                let docs = documents
                    .into_iter()
                    .zip(response.data)
                    .map(|(document, embedding)| embeddings::Embedding {
                        document,
                        vec: embedding.embedding,
                    })
                    .collect();

                Ok(docs)
            }
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

// =================================================================
// Aliyun API Types
// =================================================================
/// Type definitions for Aliyun Embedding API responses
/// Follows OpenAI-compatible API structure
#[allow(dead_code)]
mod aliyun_api_types {
    use serde::Deserialize;

    /// Response structure for embedding requests
    #[derive(Debug, Deserialize)]
    pub struct EmbeddingResponse {
        pub data: Vec<EmbeddingData>,
        pub model: String,
        pub object: String,
        pub usage: Usage,
        pub id: String,
    }

    /// Individual embedding data for a single input document
    #[derive(Debug, Deserialize)]
    pub struct EmbeddingData {
        pub embedding: Vec<f64>,
        pub index: usize,
        pub object: String,
    }

    /// Token usage statistics for the embedding request
    #[derive(Debug, Deserialize)]
    pub struct Usage {
        pub prompt_tokens: usize,
        pub total_tokens: usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::embedding::EmbeddingModel as EmbeddingModelTrait;

    #[tokio::test]
    async fn test_embed_texts() {
        let client = Client::from_env();
        let model = EmbeddingModel::new(client, EMBEDDING_V1, None);

        // Test embedding for a single document
        let documents = vec!["Hello, world!".to_string()];
        let embeddings = model.embed_texts(documents).await.unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].vec.len(), 1536);

        // Test embedding for multiple documents
        let documents = vec!["Hello, world!".to_string(), "This is a test".to_string()];
        let embeddings = model.embed_texts(documents).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].vec.len(), 1536);
        assert_eq!(embeddings[0].document, "Hello, world!");
        assert_eq!(embeddings[1].vec.len(), 1536);
        assert_eq!(embeddings[1].document, "This is a test");
    }
}
