use crate::prelude::CompletionClient;
use crate::prelude::EmbeddingsClient;
use crate::prelude::ProviderClient;
use crate::prelude::TranscriptionClient;
use crate::providers::openai;
pub use openai::completion::*;

const LMSTUDIO_API_BASE_URL: &str = "http://localhost:8080/v1";

/// A client for the LM Studio API.
#[derive(Clone, Debug)]
pub struct Client {
    inner: openai::Client,
}

impl ProviderClient for Client {
    fn from_env() -> Self {
        let base_url = std::env::var("LMSTUDIO_API_BASE")
            .unwrap_or_else(|_| LMSTUDIO_API_BASE_URL.to_string());
        let api_key = std::env::var("LMSTUDIO_API_KEY").unwrap_or_else(|_| "lmstudio".to_string());

        let inner = openai::Client::builder(&api_key)
            .base_url(&base_url)
            .build()
            .expect("Failed to build LM Studio client");

        Self { inner }
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        let base_url = std::env::var("LMSTUDIO_API_BASE")
            .unwrap_or_else(|_| LMSTUDIO_API_BASE_URL.to_string());

        let inner = openai::Client::builder(&api_key)
            .base_url(&base_url)
            .build()
            .expect("Failed to build LM Studio client");

        Self { inner }
    }
}

impl CompletionClient for Client {
    type CompletionModel = openai::responses_api::ResponsesCompletionModel;

    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        self.inner.completion_model(model)
    }
}

impl EmbeddingsClient for Client {
    type EmbeddingModel = openai::embedding::EmbeddingModel;

    fn embedding_model(&self, model: &str) -> Self::EmbeddingModel {
        self.inner.embedding_model(model)
    }

    fn embedding_model_with_ndims(&self, model: &str, ndims: usize) -> Self::EmbeddingModel {
        self.inner.embedding_model_with_ndims(model, ndims)
    }
}

impl TranscriptionClient for Client {
    type TranscriptionModel = openai::transcription::TranscriptionModel;

    fn transcription_model(&self, model: &str) -> Self::TranscriptionModel {
        self.inner.transcription_model(model)
    }
}
