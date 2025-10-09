//! Anthropic client api implementation
use bytes::Bytes;
use http_client::{Method, Request};

use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::{
    client::{
        ClientBuilderError, CompletionClient, ProviderClient, ProviderValue, VerifyClient,
        VerifyError, impl_conversion_traits,
    },
    http_client::{self, HttpClientExt},
};

// ================================================================
// Main Anthropic Client
// ================================================================
const ANTHROPIC_API_BASE_URL: &str = "https://api.anthropic.com";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    api_key: &'a str,
    base_url: &'a str,
    anthropic_version: &'a str,
    anthropic_betas: Option<Vec<&'a str>>,
    http_client: T,
}

impl<'a> ClientBuilder<'a, reqwest::Client> {
    pub fn new(api_key: &'a str) -> Self {
        ClientBuilder {
            api_key,
            base_url: ANTHROPIC_API_BASE_URL,
            anthropic_version: ANTHROPIC_VERSION_LATEST,
            anthropic_betas: None,
            http_client: Default::default(),
        }
    }
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
impl<'a, T> ClientBuilder<'a, T>
where
    T: HttpClientExt,
{
    pub fn new_with_client(api_key: &'a str, http_client: T) -> Self {
        Self {
            api_key,
            base_url: ANTHROPIC_API_BASE_URL,
            anthropic_version: ANTHROPIC_VERSION_LATEST,
            anthropic_betas: None,
            http_client,
        }
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
        ClientBuilder {
            api_key: self.api_key,
            base_url: self.base_url,
            anthropic_version: self.anthropic_version,
            anthropic_betas: self.anthropic_betas,
            http_client,
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

    pub fn build(self) -> Result<Client<T>, ClientBuilderError> {
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

        Ok(Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            default_headers,
            http_client: self.http_client,
        })
    }
}

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    /// The base URL
    base_url: String,
    /// The API key
    api_key: String,
    /// The underlying HTTP client
    http_client: T,
    /// Default headers that will be automatically added to any given request with this client (API key, Anthropic Version and any betas that have been added)
    default_headers: reqwest::header::HeaderMap,
}

impl<T> std::fmt::Debug for Client<T>
where
    T: HttpClientExt + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .field("default_headers", &self.default_headers)
            .finish()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt + Clone + Default,
{
    pub async fn send<U, V>(
        &self,
        req: http_client::Request<U>,
    ) -> Result<http_client::Response<http_client::LazyBody<V>>, http_client::Error>
    where
        U: Into<Bytes> + Send,
        V: From<Bytes> + Send + 'static,
    {
        self.http_client.send(req).await
    }

    pub async fn send_streaming<U>(
        &self,
        req: Request<U>,
    ) -> Result<http_client::StreamingResponse, http_client::Error>
    where
        U: Into<Bytes>,
    {
        self.http_client.send_streaming(req).await
    }

    pub(crate) fn post(&self, path: &str) -> http_client::Builder {
        let uri = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        let mut headers = self.default_headers.clone();

        headers.insert(
            "X-Api-Key",
            http_client::HeaderValue::from_str(&self.api_key).unwrap(),
        );

        let mut req = http_client::Request::builder()
            .method(Method::POST)
            .uri(uri);

        if let Some(hs) = req.headers_mut() {
            *hs = headers;
        }

        req
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Builder {
        let uri = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        let mut headers = self.default_headers.clone();
        headers.insert(
            "X-Api-Key",
            http_client::HeaderValue::from_str(&self.api_key).unwrap(),
        );

        let mut req = http_client::Request::builder().method(Method::GET).uri(uri);

        if let Some(hs) = req.headers_mut() {
            *hs = headers;
        }

        req
    }
}

impl Client<reqwest::Client> {
    /// Create a new Anthropic client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the API key or version cannot be parsed as a Json value from a String.
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        ClientBuilder::new(api_key)
            .build()
            .expect("Anthropic client should build")
    }
}

impl ProviderClient for Client<reqwest::Client> {
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

impl CompletionClient for Client<reqwest::Client> {
    type CompletionModel = CompletionModel<reqwest::Client>;

    fn completion_model(&self, model: &str) -> CompletionModel<reqwest::Client> {
        CompletionModel::new(self.clone(), model)
    }
}

impl VerifyClient for Client<reqwest::Client> {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/v1/models")
            .body(http_client::NoBody)
            .map_err(http_client::Error::from)?;

        let response = HttpClientExt::send(&self.http_client, req).await?;

        match response.status() {
            http::StatusCode::OK => Ok(()),
            http::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                Err(VerifyError::InvalidAuthentication)
            }
            http::StatusCode::INTERNAL_SERVER_ERROR => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            status if status.as_u16() == 529 => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                let status = response.status();

                if status.is_success() {
                    Ok(())
                } else {
                    let text: String = String::from_utf8_lossy(&response.into_body().await?).into();
                    Err(VerifyError::HttpError(http_client::Error::Instance(
                        format!("Failed with '{status}': {text}").into(),
                    )))
                }
            }
        }
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsEmbeddings,
    AsImageGeneration,
    AsAudioGeneration
    for Client<T>
);
