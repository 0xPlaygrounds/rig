use std::fmt::Display;

use crate::agent::AgentBuilder;

use super::completion::CompletionModel;

// ================================================================
// Main Huggingface Client
// ================================================================
const HUGGINGFACE_API_BASE_URL: &str = "https://router.huggingface.co/";

#[derive(Debug, Clone, PartialEq)]
pub enum SubProvider {
    HFInference,
    Together,
    SambaNova,
    Custom(String),
}

impl From<&str> for SubProvider {
    fn from(s: &str) -> Self {
        SubProvider::Custom(s.to_string())
    }
}

impl Display for SubProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let route = match self {
            SubProvider::HFInference => "hf-inference/models".to_string(),
            SubProvider::Together => "together".to_string(),
            SubProvider::SambaNova => "sambanova".to_string(),
            SubProvider::Custom(route) => route.clone(),
        };

        write!(f, "{}", route)
    }
}

impl From<String> for SubProvider {
    fn from(value: String) -> Self {
        SubProvider::Custom(value)
    }
}

pub struct ClientBuilder {
    api_key: String,
    base_url: String,
    sub_provider: SubProvider,
}

impl ClientBuilder {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: HUGGINGFACE_API_BASE_URL.to_string(),
            sub_provider: SubProvider::HFInference,
        }
    }

    pub fn base_url(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_string();
        self
    }

    pub fn sub_provider(mut self, provider: impl Into<SubProvider>) -> Self {
        self.sub_provider = provider.into();
        self
    }

    pub fn build(self) -> Client {
        let route = self.sub_provider.to_string();

        let base_url = format!("{}/{}", self.base_url, route).replace("//", "/");

        Client::from_url(self.api_key.as_str(), base_url.as_str(), self.sub_provider)
    }
}

#[derive(Clone)]
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
    pub(crate) sub_provider: SubProvider,
}

impl Client {
    /// Create a new Huggingface client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, HUGGINGFACE_API_BASE_URL, SubProvider::HFInference)
    }

    /// Create a new Client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str, sub_provider: SubProvider) -> Self {
        let http_client = reqwest::Client::builder()
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    "Authorization",
                    format!("Bearer {api_key}")
                        .parse()
                        .expect("Failed to parse API key"),
                );
                headers.insert(
                    "Content-Type",
                    "application/json"
                        .parse()
                        .expect("Failed to parse Content-Type"),
                );
                headers
            })
            .build()
            .expect("Failed to build HTTP client");

        Self {
            base_url: base_url.to_owned(),
            http_client,
            sub_provider,
        }
    }
    /// Create a new Huggingface client from the `HUGGINGFACE_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    pub fn from_env() -> Self {
        let api_key = std::env::var("HUGGINGFACE_API_KEY").expect("HUGGINGFACE_API_KEY is not set");
        Self::new(&api_key)
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url)
    }

    /// Create a new completion model with the given name
    ///
    /// # Example
    /// ```
    /// use rig::providers::huggingface::{Client, self}
    ///
    /// // Initialize the Huggingface client
    /// let client = Client::new("your-huggingface-api-key");
    ///
    /// let completion_model = client.completion_model(huggingface::GEMMA_2);
    /// ```
    pub fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }

    /// Create an agent builder with the given completion model.
    ///
    /// # Example
    /// ```
    /// use rig::providers::huggingface::{Client, self};
    ///
    /// // Initialize the Anthropic client
    /// let client = Client::new("your-huggingface-api-key");
    ///
    /// let agent = client.agent(huggingface::GEMMA_2)
    ///    .preamble("You are comedian AI with a mission to make people laugh.")
    ///    .temperature(0.0)
    ///    .build();
    /// ```
    pub fn agent(&self, model: &str) -> AgentBuilder<CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }
}
