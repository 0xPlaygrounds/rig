use http::Method;

use super::completion::CompletionModel;
use crate::{
    client::{CompletionClient, ProviderClient, VerifyClient, VerifyError, impl_conversion_traits},
    http_client::{self, HttpClientExt, NoBody, Result as HttpResult, with_bearer_auth},
};

// ================================================================
// xAI Client
// ================================================================
const XAI_BASE_URL: &str = "https://api.x.ai";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: T,
}

impl<'a, T> ClientBuilder<'a, T>
where
    T: Default,
{
    pub fn new(api_key: &'a str) -> Self {
        Self {
            api_key,
            base_url: XAI_BASE_URL,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn new_with_client(api_key: &'a str, http_client: T) -> Self {
        Self {
            api_key,
            base_url: XAI_BASE_URL,
            http_client,
        }
    }

    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
        ClientBuilder {
            api_key: self.api_key,
            base_url: self.base_url,
            http_client,
        }
    }

    pub fn build(self) -> Client<T> {
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            default_headers,
            http_client: self.http_client,
        }
    }
}

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    api_key: String,
    default_headers: http_client::HeaderMap,
    pub http_client: T,
}

impl<T> std::fmt::Debug for Client<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("default_headers", &self.default_headers)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl Client<reqwest::Client> {
    /// Create a new xAI client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::xai::{ClientBuilder, self};
    ///
    /// // Initialize the xAI client
    /// let xai = Client::builder("your-xai-api-key")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<'_, reqwest::Client> {
        ClientBuilder::new(api_key)
    }

    /// Create a new xAI client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key).build()
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    pub(crate) fn req(&self, method: Method, path: &str) -> HttpResult<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        let mut builder = http_client::Builder::new().uri(url).method(method);
        for (header, value) in &self.default_headers {
            builder = builder.header(header, value);
        }

        with_bearer_auth(builder, &self.api_key)
    }

    pub(crate) fn post(&self, path: &str) -> HttpResult<http_client::Builder> {
        self.req(Method::POST, path)
    }

    pub(crate) fn get(&self, path: &str) -> HttpResult<http_client::Builder> {
        self.req(Method::GET, path)
    }
}

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    /// Create a new xAI client from the `XAI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("XAI_API_KEY").expect("XAI_API_KEY not set");
        ClientBuilder::<T>::new(&api_key).build()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        ClientBuilder::<T>::new(&api_key).build()
    }
}

impl<T> CompletionClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type CompletionModel = CompletionModel<T>;

    /// Create a completion model with the given name.
    fn completion_model(&self, model: &str) -> CompletionModel<T> {
        CompletionModel::new(self.clone(), model)
    }
}

impl<T> VerifyClient for Client<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self.get("/v1/api-key").unwrap().body(NoBody).unwrap();

        let response = self.http_client.send::<_, Vec<u8>>(req).await.unwrap();
        let status = response.status();

        match status {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                Err(VerifyError::InvalidAuthentication)
            }
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => Err(VerifyError::ProviderError(
                http_client::text(response).await?,
            )),
            _ => Err(VerifyError::HttpError(http_client::Error::Instance(
                http_client::text(response).await?.into(),
            ))),
        }
    }
}

impl_conversion_traits!(
    AsEmbeddings,
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client<T>
);

pub mod xai_api_types {
    use serde::Deserialize;

    impl ApiErrorResponse {
        pub fn message(&self) -> String {
            format!("Code `{}`: {}", self.code, self.error)
        }
    }

    #[derive(Debug, Deserialize)]
    pub struct ApiErrorResponse {
        pub error: String,
        pub code: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    pub enum ApiResponse<T> {
        Ok(T),
        Error(ApiErrorResponse),
    }
}
