//! Anthropic client api implementation

use crate::{agent::AgentBuilder, extractor::ExtractorBuilder};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::completion::{CompletionModel, ANTHROPIC_VERSION_LATEST};

// ================================================================
// Main Anthropic Client
// ================================================================
const ANTHROPIC_API_BASE_URL: &str = "https://api.anthropic.com";

#[derive(Clone)]
pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
    anthropic_version: &'a str,
    anthropic_betas: Option<Vec<&'a str>>,
}

/// Create a new anthropic client using the builder
///
/// # Example
/// ```
/// use rig::providers::anthropic::{ClientBuilder, self};
///
/// // Initialize the Anthropic client
/// let anthropic_client = ClientBuilder::new("your-claude-api-key")
///    .anthropic_version(ANTHROPIC_VERSION_LATEST)
///    .anthropic_beta("prompt-caching-2024-07-31")
///    .build()
/// ```
impl<'a> ClientBuilder<'a> {
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: ANTHROPIC_API_BASE_URL,
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

    pub fn anthropic_beta(mut self, anthropic_beta: &'a str) -> Self {
        if let Some(mut betas) = self.anthropic_betas {
            betas.push(anthropic_beta);
            self.anthropic_betas = Some(betas);
        } else {
            self.anthropic_betas = Some(vec![anthropic_beta]);
        }
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
    /// Create a new Anthropic client with the given API key, base URL, betas, and version.
    /// Note, you probably want to use the `ClientBuilder` instead.
    ///
    /// Panics:
    /// - If the API key or version cannot be parsed as a Json value from a String.
    ///   - This should really never happen.
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
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
                .expect("Anthropic reqwest client should build"),
        }
    }

    /// Create a new Anthropic client from the `ANTHROPIC_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
        ClientBuilder::new(&api_key).build()
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an agent builder with the given completion model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::anthropic::{ClientBuilder, self};
    ///
    /// // Initialize the Anthropic client
    /// let anthropic = ClientBuilder::new("your-claude-api-key").build();
    ///
    /// let agent = anthropic.agent(anthropic::CLAUDE_3_5_SONNET)
    ///    .preamble("You are comedian AI with a mission to make people laugh.")
    ///    .temperature(0.0)
    ///    .build();
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
}
