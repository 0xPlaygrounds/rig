//! Anthropic client api implementation

use crate::{
    agent::AgentBuilder,
    extractor::ExtractorBuilder,
    model::ModelBuilder,
    rag::RagAgentBuilder,
    vector_store::{NoIndex, VectorStoreIndex},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::completion::{CompletionModel, ANTHROPIC_VERSION_LATEST};

// ================================================================
// Main Cohere Client
// ================================================================
const COHERE_API_BASE_URL: &str = "https://api.anthropic.com";

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
    anthropic_version: &'a str,
    anthropic_betas: Option<Vec<&'a str>>,
}

impl<'a> ClientBuilder<'a> {
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: COHERE_API_BASE_URL,
            anthropic_version: ANTHROPIC_VERSION_LATEST,
            anthropic_betas: None,
        }
    }

    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn anthropic_version(mut self, anthropic_version: &'a str) -> Self {
        self.anthropic_version = anthropic_version;
        self
    }

    pub fn anthropic_betas(mut self, anthropic_betas: Vec<&'a str>) -> Self {
        self.anthropic_betas = Some(anthropic_betas);
        self
    }

    pub fn build(self) -> Client {
        Client::new(
            self.api_key,
            self.base_url,
            self.anthropic_betas,
            self.anthropic_version,
        )
    }
}

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    pub fn new(api_key: &str, base_url: &str, betas: Option<Vec<&str>>, version: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http_client: reqwest::Client::builder()
                .default_headers({
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert("x-api-key", api_key.parse().expect("API key should parse"));
                    headers.insert(
                        "anthropic-version",
                        version.parse().expect("Anthropic version should parse"),
                    );
                    if let Some(betas) = betas {
                        headers.insert(
                            "anthropic-beta",
                            betas
                                .join(",")
                                .parse()
                                .expect("Anthropic betas should parse"),
                        );
                    }
                    headers
                })
                .build()
                .expect("Cohere reqwest client should build"),
        }
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    pub fn model(&self, model: &str) -> ModelBuilder<CompletionModel> {
        ModelBuilder::new(self.completion_model(model))
    }

    /// Create an embedding builder with the given embedding model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let embeddings = openai.embeddings(openai::TEXT_EMBEDDING_3_LARGE)
    ///     .simple_document("doc0", "Hello, world!")
    ///     .simple_document("doc1", "Goodbye, world!")
    ///     .build()
    ///     .await
    ///     .expect("Failed to embed documents");
    /// ```
    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    pub fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }

    pub fn rag_agent<C: VectorStoreIndex, T: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, C, T> {
        RagAgentBuilder::new(self.completion_model(model))
    }

    pub fn tool_rag_agent<T: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, NoIndex, T> {
        RagAgentBuilder::new(self.completion_model(model))
    }

    pub fn context_rag_agent<C: VectorStoreIndex>(
        &self,
        model: &str,
    ) -> RagAgentBuilder<CompletionModel, C, NoIndex> {
        RagAgentBuilder::new(self.completion_model(model))
    }
}
