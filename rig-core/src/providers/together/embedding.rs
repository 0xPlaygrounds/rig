// ================================================================
//! Together AI Embeddings Integration
//! From [Together AI Reference](https://docs.together.ai/docs/embeddings-overview)
// ================================================================

use serde::Deserialize;
use serde_json::json;

use crate::embeddings::{self, EmbeddingError};

use super::{
    client::together_ai_api_types::{ApiErrorResponse, ApiResponse},
    Client,
};

// ================================================================
// Together AI Embedding API
// ================================================================

pub const BGE_BASE_EN_V1_5: &str = "BAAI/bge-base-en-v1.5";
pub const BGE_LARGE_EN_V1_5: &str = "BAAI/bge-large-en-v1.5";
pub const BERT_BASE_UNCASED: &str = "bert-base-uncased";
pub const M2_BERT_2K_RETRIEVAL_ENCODER_V1: &str = "hazyresearch/M2-BERT-2k-Retrieval-Encoder-V1";
pub const M2_BERT_80M_32K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-32k-retrieval";
pub const M2_BERT_80M_2K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-2k-retrieval";
pub const M2_BERT_80M_8K_RETRIEVAL: &str = "togethercomputer/m2-bert-80M-8k-retrieval";
pub const SENTENCE_BERT: &str = "sentence-transformers/msmarco-bert-base-dot-v5";
pub const UAE_LARGE_V1: &str = "WhereIsAI/UAE-Large-V1";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub object: String,
    pub data: Vec<EmbeddingData>,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message())
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Error(err) => Err(EmbeddingError::ProviderError(err.message())),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
    ndims: usize,
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024; // This might need to be adjusted based on Together AI's actual limit

    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let response = self
            .client
            .post("/v1/embeddings")
            .json(&json!({
                "model": self.model,
                "input": documents,
            }))
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<EmbeddingResponse>>().await? {
                ApiResponse::Ok(response) => {
                    if response.data.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    Ok(response
                        .data
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding.embedding,
                        })
                        .collect())
                }
                ApiResponse::Error(err) => Err(EmbeddingError::ProviderError(err.message())),
            }
        } else {
            Err(EmbeddingError::ProviderError(response.text().await?))
        }
    }
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}
