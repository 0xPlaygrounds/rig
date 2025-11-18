//! Anthropic client api implementation
use bytes::Bytes;
use http_client::Method;

use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::{
    client::{
        ClientBuilderError, CompletionClient, ProviderClient, ProviderValue, StandardClientBuilder,
        VerifyClient, VerifyError, impl_conversion_traits,
    },
    http_client::{self, HttpClientExt},
    wasm_compat::WasmCompatSend,
};

// ================================================================
// Main Anthropic Client
// ================================================================
const ANTHROPIC_API_BASE_URL: &str = "https://api.anthropic.com";

/// Extension data for Anthropic client builder
#[derive(Default, Clone)]
pub struct BuilderExtension {
    pub anthropic_version: Option<String>,
    pub anthropic_betas: Option<Vec<String>>,
}

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    /// The base URL
    base_url: String,
    /// The API key
    api_key: String,
    /// The underlying HTTP client
    pub http_client: T,
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

impl<T> StandardClientBuilder<T> for Client<T>
where
    T: HttpClientExt,
{
    fn build_from_builder<Ext>(
        builder: crate::client::Builder<'_, Self, T, Ext>,
    ) -> Result<Self, crate::client::ClientBuilderError>
    where
        Ext: Default + 'static,
        T: Default + Clone,
    {
        let api_key = builder.get_api_key();
        let base_url = builder.get_base_url(ANTHROPIC_API_BASE_URL);
        let http_client = builder.get_http_client();
        let mut default_headers = builder.get_headers(false);

        // Extract extension data using try_get_extension
        let (anthropic_version, anthropic_betas) =
            if let Some(ext) = builder.try_get_extension::<BuilderExtension>() {
                (
                    ext.anthropic_version
                        .unwrap_or_else(|| ANTHROPIC_VERSION_LATEST.to_string()),
                    ext.anthropic_betas,
                )
            } else {
                (ANTHROPIC_VERSION_LATEST.to_string(), None)
            };

        default_headers.insert(
            "anthropic-version",
            anthropic_version
                .parse()
                .map_err(|_| ClientBuilderError::InvalidProperty("anthropic-version"))?,
        );

        if let Some(betas) = anthropic_betas {
            default_headers.insert(
                "anthropic-beta",
                betas
                    .join(",")
                    .parse()
                    .map_err(|_| ClientBuilderError::InvalidProperty("anthropic-beta"))?,
            );
        }

        Ok(Client {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            default_headers,
            http_client,
        })
    }
}

// Helper implementation for Builder with AnthropicBuilderExtension
impl<'a, T> crate::client::Builder<'a, Client<T>, T, BuilderExtension>
where
    T: crate::http_client::HttpClientExt + Default,
{
    pub fn anthropic_version(self, version: &str) -> Self {
        self.update_extension(|mut ext| {
            ext.anthropic_version = Some(version.to_string());
            ext
        })
    }

    pub fn anthropic_beta(self, beta: &str) -> Self {
        self.update_extension(|mut ext| {
            match &mut ext.anthropic_betas {
                Some(betas) => betas.push(beta.to_string()),
                None => ext.anthropic_betas = Some(vec![beta.to_string()]),
            }
            ext
        })
    }
}

impl Client<reqwest::Client> {
    pub fn builder(
        api_key: &str,
    ) -> crate::client::Builder<'_, Self, reqwest::Client, BuilderExtension> {
        <Self as StandardClientBuilder<reqwest::Client>>::builder(api_key)
            .with_extension(BuilderExtension::default())
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
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
}

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    /// Create a new Anthropic client from the `ANTHROPIC_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");

        Self::builder(&api_key)
            .build()
            .expect("Anthropic client should build")
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };

        Self::builder(&api_key)
            .build()
            .expect("Anthropic client should build")
    }
}

impl<T> CompletionClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + WasmCompatSend + 'static,
{
    type CompletionModel = CompletionModel<T>;

    fn completion_model(&self, model: &str) -> CompletionModel<T> {
        CompletionModel::new(self.clone(), model)
    }
}

impl<T> VerifyClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
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
