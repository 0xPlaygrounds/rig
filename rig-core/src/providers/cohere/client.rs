use crate::{
    agent::AgentBuilder, embeddings::EmbeddingsBuilder, extractor::ExtractorBuilder, Embed,
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{CompletionModel, EmbeddingModel};

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// Main Cohere Client
// ================================================================
const COHERE_API_BASE_URL: &str = "https://api.cohere.ai";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, COHERE_API_BASE_URL)
    }

    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        "Authorization",
                        format!("Bearer {}", api_key)
                            .parse()
                            .expect("Bearer token should parse"),
                    );
                    headers
                })
                .build()
                .expect("Cohere reqwest client should build"),
        }
    }

    /// Create a new Cohere client from the `COHERE_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
        Self::new(&api_key)
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Note: default embedding dimension of 0 will be used if model is not known.
    /// If this is the case, it's better to use function `embedding_model_with_ndims`
    pub fn embedding_model(&self, model: &str, input_type: &str) -> EmbeddingModel {
        let ndims = match model {
            super::EMBED_ENGLISH_V3
            | super::EMBED_MULTILINGUAL_V3
            | super::EMBED_ENGLISH_LIGHT_V2 => 1024,
            super::EMBED_ENGLISH_LIGHT_V3 | super::EMBED_MULTILINGUAL_LIGHT_V3 => 384,
            super::EMBED_ENGLISH_V2 => 4096,
            super::EMBED_MULTILINGUAL_V2 => 768,
            _ => 0,
        };
        EmbeddingModel::new(self.clone(), model, input_type, ndims)
    }

    /// Create an embedding model with the given name and the number of dimensions in the embedding generated by the model.
    pub fn embedding_model_with_ndims(
        &self,
        model: &str,
        input_type: &str,
        ndims: usize,
    ) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), model, input_type, ndims)
    }

    pub fn embeddings<D: Embed>(
        &self,
        model: &str,
        input_type: &str,
    ) -> EmbeddingsBuilder<EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model, input_type))
    }

    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}
