use http::Method;

use super::completion::CompletionModel;
use crate::{
    client::{
        CompletionClient, ProviderClient, StandardClientBuilder, VerifyClient, VerifyError,
        impl_conversion_traits,
    },
    http_client::{self, HttpClientExt, NoBody, Result as HttpResult, with_bearer_auth},
};

// ================================================================
// xAI Client
// ================================================================
const XAI_BASE_URL: &str = "https://api.x.ai";

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
    /// Create a new xAI client. For more control, use the `builder` method.
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key)
            .build()
            .expect("xAI client should build")
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }

    /// Create a new xAI client builder
    pub fn builder(api_key: &str) -> crate::client::Builder<'_, Self, reqwest::Client> {
        <Self as StandardClientBuilder<reqwest::Client>>::builder(api_key)
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

impl<T> StandardClientBuilder<T> for Client<T>
where
    T: HttpClientExt,
{
    fn build_from_builder<Ext>(
        builder: crate::client::Builder<'_, Self, T, Ext>,
    ) -> Result<Self, crate::client::ClientBuilderError>
    where
        Ext: Default,
        T: Default + Clone,
    {
        let api_key = builder.get_api_key();
        let base_url = builder.get_base_url(XAI_BASE_URL);
        let http_client = builder.get_http_client();
        let default_headers = builder.get_headers(true);
        Ok(Client {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            default_headers,
            http_client,
        })
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
        Self::builder(&api_key)
            .build()
            .expect("xAI client should build")
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        Self::builder(&api_key)
            .build()
            .expect("xAI client should build")
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
