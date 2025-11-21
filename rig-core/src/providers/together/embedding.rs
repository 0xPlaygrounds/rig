// ================================================================
//! Together AI Embeddings Integration
//! From [Together AI Reference](https://docs.together.ai/docs/embeddings-overview)
// ================================================================

use serde::Deserialize;
use serde_json::json;

use crate::{
    embeddings::{self, EmbeddingError},
    http_client::{self, HttpClientExt},
    models,
};

use super::{
    Client,
    client::together_ai_api_types::{ApiErrorResponse, ApiResponse},
};

// ================================================================
// Together AI Embedding API
// ================================================================

models! {
    #[allow(non_camel_case_types)]
    pub enum EmbeddingModels {
        BGE_BASE_EN_V1_5=> "BAAI/bge-base-en-v1.5",
        BGE_LARGE_EN_V1_5=> "BAAI/bge-large-en-v1.5",
        BERT_BASE_UNCASED=> "bert-base-uncased",
        M2_BERT_2K_RETRIEVAL_ENCODER_V1=> "hazyresearch/M2-BERT-2k-Retrieval-Encoder-V1",
        M2_BERT_80M_32K_RETRIEVAL=> "togethercomputer/m2-bert-80M-32k-retrieval",
        M2_BERT_80M_2K_RETRIEVAL=> "togethercomputer/m2-bert-80M-2k-retrieval",
        M2_BERT_80M_8K_RETRIEVAL=> "togethercomputer/m2-bert-80M-8k-retrieval",
        SENTENCE_BERT=> "sentence-transformers/msmarco-bert-base-dot-v5",
        UAE_LARGE_V1=> "WhereIsAI/UAE-Large-V1",
    }
}
pub use EmbeddingModels::*;

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
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Default + Clone + Send + 'static,
{
    const MAX_DOCUMENTS: usize = 1024; // This might need to be adjusted based on Together AI's actual limit

    type Client = Client<T>;
    type Models = EmbeddingModels;

    fn make(client: &Self::Client, model: Self::Models, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model, dims.unwrap_or_default())
    }

    fn make_custom(client: &Self::Client, model: &str, dims: Option<usize>) -> Self {
        Self::with_model(client.clone(), model, dims.unwrap_or_default())
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let body = serde_json::to_vec(&json!({
            "model": self.model,
            "input": documents,
        }))?;

        let req = self
            .client
            .post("/v1/embeddings")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        if response.status().is_success() {
            let body: Vec<u8> = response.into_body().await?;
            let body: ApiResponse<EmbeddingResponse> = serde_json::from_slice(&body)?;

            match body {
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
            let text = http_client::text(response).await?;
            Err(EmbeddingError::ProviderError(text))
        }
    }
}

impl<T> EmbeddingModel<T>
where
    T: Default,
{
    pub fn new(client: Client<T>, model: EmbeddingModels, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }

    pub fn with_model(client: Client<T>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }
}
