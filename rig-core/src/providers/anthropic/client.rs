//! Anthropic client api implementation
use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::client::{
    ClientBuilderError, CompletionClient, ProviderClient, ProviderValue, VerifyClient, VerifyError,
    impl_conversion_traits,
};

// ================================================================
// Main Anthropic Client
// ================================================================
const ANTHROPIC_API_BASE_URL: &str = "https://api.anthropic.com";

pub struct ClientBuilder<'a> {
    api_key: &'a str,
    base_url: &'a str,
    anthropic_version: &'a str,
    anthropic_betas: Option<Vec<&'a str>>,
    http_client: Option<reqwest::Client>,
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
            http_client: None,
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

    pub fn custom_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = Some(client);
        self
    }

    pub fn build(self) -> Result<Client, ClientBuilderError> {
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            "anthropic-version",
            self.anthropic_version
                .parse()
                .map_err(|_| ClientBuilderError::InvalidProperty("anthropic-version"))?,
        );
        if let Some(betas) = self.anthropic_betas {
            default_headers.insert(
                "anthropic-beta",
                betas
                    .join(",")
                    .parse()
                    .map_err(|_| ClientBuilderError::InvalidProperty("anthropic-beta"))?,
            );
        };

        let http_client = if let Some(http_client) = self.http_client {
            http_client
        } else {
            reqwest::Client::builder().build()?
        };

        Ok(Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            default_headers,
            http_client,
        })
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
    /// Create a new Anthropic client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::anthropic::{ClientBuilder, self};
    ///
    /// // Initialize the Anthropic client
    /// let anthropic_client = Client::builder("your-claude-api-key")
    ///    .anthropic_version(ANTHROPIC_VERSION_LATEST)
    ///    .anthropic_beta("prompt-caching-2024-07-31")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<'_> {
        ClientBuilder::new(api_key)
    }

    /// Create a new Anthropic client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the API key or version cannot be parsed as a Json value from a String.
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key)
            .build()
            .expect("Anthropic client should build")
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client
            .post(url)
            .header("X-Api-Key", &self.api_key)
            .headers(self.default_headers.clone())
    }

    pub(crate) fn get(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client
            .get(url)
            .header("X-Api-Key", &self.api_key)
            .headers(self.default_headers.clone())
    }
}

impl ProviderClient for Client {
    /// Create a new Anthropic client from the `ANTHROPIC_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
        Client::new(&api_key)
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        Client::new(&api_key)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;
    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl VerifyClient for Client {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let response = self.get("/v1/models").send().await?;
        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                Err(VerifyError::InvalidAuthentication)
            }
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                Err(VerifyError::ProviderError(response.text().await?))
            }
            status if status.as_u16() == 529 => {
                Err(VerifyError::ProviderError(response.text().await?))
            }
            _ => {
                response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsEmbeddings,
    AsImageGeneration,
    AsAudioGeneration for Client
);
