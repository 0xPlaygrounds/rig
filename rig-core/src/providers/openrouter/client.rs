use crate::{
    client::{CompletionClient, ProviderClient, StandardClientBuilder, VerifyClient, VerifyError},
    completion::GetTokenUsage,
    http_client::{self, HttpClientExt},
    impl_conversion_traits,
};
use http::Method;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::completion::CompletionModel;

// ================================================================
// Main openrouter Client
// ================================================================
const OPENROUTER_API_BASE_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    api_key: String,
    pub http_client: T,
}

impl<T> Debug for Client<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl<T> Client<T> {
    pub(crate) fn req(
        &self,
        method: Method,
        path: &str,
    ) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        let req = http_client::Request::builder().uri(url).method(method);

        http_client::with_bearer_auth(req, &self.api_key)
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        self.req(Method::GET, path)
    }

    pub(crate) fn post(&self, path: &str) -> http_client::Result<http_client::Builder> {
        self.req(Method::POST, path)
    }
}

impl Client<reqwest::Client> {
    /// Create a new OpenRouter client. For more control, use the `builder` method.
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key)
            .build()
            .expect("OpenRouter client should build")
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }

    /// Create a new OpenRouter client builder
    pub fn builder(api_key: &str) -> crate::client::Builder<'_, Self, reqwest::Client> {
        <Self as StandardClientBuilder<reqwest::Client>>::builder(api_key)
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
        let base_url = builder.get_base_url(OPENROUTER_API_BASE_URL);
        let http_client = builder.get_http_client();
        Ok(Client {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            http_client,
        })
    }
}

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    /// Create a new openrouter client from the `OPENROUTER_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");
        Self::builder(&api_key)
            .build()
            .expect("OpenRouter client should build")
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        Self::builder(&api_key)
            .build()
            .expect("OpenRouter client should build")
    }
}

impl<T> CompletionClient for Client<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    type CompletionModel = CompletionModel<T>;

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::openrouter::{Client, self};
    ///
    /// // Initialize the openrouter client
    /// let openrouter = Client::new("your-openrouter-api-key");
    ///
    /// let llama_3_1_8b = openrouter.completion_model(openrouter::LLAMA_3_1_8B);
    /// ```
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
            .get("/key")?
            .body(http_client::NoBody)
            .map_err(|e| VerifyError::HttpError(e.into()))?;

        let response = HttpClientExt::send(&self.http_client, req).await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                //response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(
    AsEmbeddings,
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client<T>
);

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.prompt_tokens as u64;
        usage.output_tokens = self.completion_tokens as u64;
        usage.total_tokens = self.total_tokens as u64;

        Some(usage)
    }
}
