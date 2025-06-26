//! Anthropic client api implementation
use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::client::{CompletionClient, ProviderClient, impl_conversion_traits};

// ================================================================
// Main Anthropic Client
// ================================================================
const ANTHROPIC_API_BASE_URL: &str = "https://api.anthropic.com";

#[derive(Clone, Debug)]
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
    /// The base URL
    base_url: String,
    /// The API key
    api_key: String,
    /// The underlying HTTP client
    http_client: reqwest::Client,
    /// Default headers that will be automatically added to any given request with this client (API key, Anthropic Version and any betas that have been added)
    default_headers: reqwest::header::HeaderMap,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .field("default_headers", &self.default_headers)
            .finish()
    }
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
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            "anthropic-version",
            version.parse().expect("Anthropic version should parse"),
        );
        if let Some(betas) = betas {
            default_headers.insert(
                "anthropic-beta",
                betas
                    .join(",")
                    .parse()
                    .expect("Anthropic betas should parse"),
            );
        };

        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            default_headers,
            http_client: reqwest::Client::builder()
                .build()
                .expect("Anthropic reqwest client should build"),
        }
    }

    /// Use your own `reqwest::Client`.
    /// The default headers will be automatically attached upon trying to make a request.
    pub fn with_custom_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;

        self
    }

    pub fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client
            .post(url)
            .header("X-Api-Key", &self.api_key)
            .headers(self.default_headers.clone())
    }
}

impl ProviderClient for Client {
    /// Create a new Anthropic client from the `ANTHROPIC_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
        ClientBuilder::new(&api_key).build()
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;
    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsEmbeddings,
    AsImageGeneration,
    AsAudioGeneration for Client
);
