use super::completion::CompletionModel;
#[cfg(feature = "image")]
use crate::client::ImageGenerationClient;
use crate::client::{
    ClientBuilderError, CompletionClient, ProviderClient, TranscriptionClient, VerifyClient,
    VerifyError,
};
use crate::http_client::{self, HttpClientExt};
#[cfg(feature = "image")]
use crate::image_generation::ImageGenerationError;
#[cfg(feature = "image")]
use crate::providers::huggingface::image_generation::ImageGenerationModel;
use crate::providers::huggingface::transcription::TranscriptionModel;
use crate::transcription::TranscriptionError;
use bytes::Bytes;
use rig::client::impl_conversion_traits;
use std::fmt::Debug;
use std::fmt::Display;

// ================================================================
// Main Huggingface Client
// ================================================================
const HUGGINGFACE_API_BASE_URL: &str = "https://router.huggingface.co/";

#[derive(Debug, Clone, PartialEq, Default)]
pub enum SubProvider {
    #[default]
    HFInference,
    Together,
    SambaNova,
    Fireworks,
    Hyperbolic,
    Nebius,
    Novita,
    Custom(String),
}

impl SubProvider {
    /// Get the chat completion endpoint for the SubProvider
    /// Required because Huggingface Inference requires the model
    /// in the url and in the request body.
    pub fn completion_endpoint(&self, model: &str) -> String {
        match self {
            SubProvider::HFInference => format!("/{model}/v1/chat/completions"),
            _ => "/v1/chat/completions".to_string(),
        }
    }

    /// Get the transcription endpoint for the SubProvider
    /// Required because Huggingface Inference requires the model
    /// in the url and in the request body.
    pub fn transcription_endpoint(&self, model: &str) -> Result<String, TranscriptionError> {
        match self {
            SubProvider::HFInference => Ok(format!("/{model}")),
            _ => Err(TranscriptionError::ProviderError(format!(
                "transcription endpoint is not supported yet for {self}"
            ))),
        }
    }

    /// Get the image generation endpoint for the SubProvider
    /// Required because Huggingface Inference requires the model
    /// in the url and in the request body.
    #[cfg(feature = "image")]
    pub fn image_generation_endpoint(&self, model: &str) -> Result<String, ImageGenerationError> {
        match self {
            SubProvider::HFInference => Ok(format!("/{model}")),
            _ => Err(ImageGenerationError::ProviderError(format!(
                "image generation endpoint is not supported yet for {self}"
            ))),
        }
    }

    pub fn model_identifier(&self, model: &str) -> String {
        match self {
            SubProvider::Fireworks => format!("accounts/fireworks/models/{model}"),
            _ => model.to_string(),
        }
    }
}

impl From<&str> for SubProvider {
    fn from(s: &str) -> Self {
        SubProvider::Custom(s.to_string())
    }
}

impl From<String> for SubProvider {
    fn from(value: String) -> Self {
        SubProvider::Custom(value)
    }
}

impl Display for SubProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let route = match self {
            SubProvider::HFInference => "hf-inference/models".to_string(),
            SubProvider::Together => "together".to_string(),
            SubProvider::SambaNova => "sambanova".to_string(),
            SubProvider::Fireworks => "fireworks-ai".to_string(),
            SubProvider::Hyperbolic => "hyperbolic".to_string(),
            SubProvider::Nebius => "nebius".to_string(),
            SubProvider::Novita => "novita".to_string(),
            SubProvider::Custom(route) => route.clone(),
        };

        write!(f, "{route}")
    }
}

pub struct ClientBuilder<T = reqwest::Client> {
    api_key: String,
    base_url: String,
    sub_provider: SubProvider,
    http_client: T,
}

impl<T> ClientBuilder<T>
where
    T: Default,
{
    pub fn new(api_key: &str) -> ClientBuilder<T> {
        ClientBuilder {
            api_key: api_key.to_string(),
            base_url: HUGGINGFACE_API_BASE_URL.to_string(),
            sub_provider: SubProvider::default(),
            http_client: Default::default(),
        }
    }
}

impl<T> ClientBuilder<T> {
    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<U> {
        ClientBuilder {
            api_key: self.api_key,
            base_url: self.base_url,
            sub_provider: self.sub_provider,
            http_client,
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

    pub fn build(self) -> Result<Client<T>, ClientBuilderError> {
        let route = self.sub_provider.to_string();
        let base_url = format!("{}/{}", self.base_url, route).replace("//", "/");

        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            "Content-Type",
            "application/json"
                .parse()
                .expect("Failed to parse Content-Type"),
        );

        Ok(Client {
            base_url,
            default_headers,
            api_key: self.api_key,
            http_client: self.http_client,
            sub_provider: self.sub_provider,
        })
    }
}

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    default_headers: reqwest::header::HeaderMap,
    api_key: String,
    http_client: T,
    pub(crate) sub_provider: SubProvider,
}

impl<T> Debug for Client<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("default_headers", &self.default_headers)
            .field("sub_provider", &self.sub_provider)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl<T> Client<T>
where
    T: Default,
{
    /// Create a new Huggingface client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::huggingface::{ClientBuilder, self};
    ///
    /// // Initialize the Huggingface client
    /// let client = Client::builder("your-huggingface-api-key")
    ///    .build()
    /// ```
    pub fn builder(api_key: &str) -> ClientBuilder<T> {
        ClientBuilder::new(api_key)
    }

    /// Create a new Huggingface client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new(api_key: &str) -> Self {
        Self::builder(api_key)
            .build()
            .expect("Huggingface client should build")
    }
}

impl Client<reqwest::Client> {
    pub(crate) fn post_reqwest(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

        self.http_client
            .post(url)
            .headers(self.default_headers.clone())
            .bearer_auth(&self.api_key)
    }
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    pub(crate) fn post(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

        let mut req = http_client::Request::post(url);

        if let Some(hs) = req.headers_mut() {
            *hs = self.default_headers.clone();
        }

        http_client::with_bearer_auth(req, &self.api_key)
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");

        let mut req = http_client::Request::get(url);

        if let Some(hs) = req.headers_mut() {
            *hs = self.default_headers.clone();
        }

        http_client::with_bearer_auth(req, &self.api_key)
    }

    pub(crate) async fn send<U, V>(
        &self,
        req: http_client::Request<U>,
    ) -> http_client::Result<http_client::Response<http_client::LazyBody<V>>>
    where
        U: Into<Bytes> + Send,
        V: From<Bytes> + Send + 'static,
    {
        self.http_client.send(req).await
    }
}

impl ProviderClient for Client<reqwest::Client> {
    /// Create a new Huggingface client from the `HUGGINGFACE_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("HUGGINGFACE_API_KEY").expect("HUGGINGFACE_API_KEY is not set");
        Self::new(&api_key)
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!("Incorrect provider value type")
        };
        Self::new(&api_key)
    }
}

impl CompletionClient for Client<reqwest::Client> {
    type CompletionModel = CompletionModel<reqwest::Client>;

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
    fn completion_model(&self, model: &str) -> CompletionModel<reqwest::Client> {
        CompletionModel::new(self.clone(), model)
    }
}

impl TranscriptionClient for Client<reqwest::Client> {
    type TranscriptionModel = TranscriptionModel<reqwest::Client>;

    /// Create a new transcription model with the given name
    ///
    /// # Example
    /// ```
    /// use rig::providers::huggingface::{Client, self}
    ///
    /// // Initialize the Huggingface client
    /// let client = Client::new("your-huggingface-api-key");
    ///
    /// let completion_model = client.transcription_model(huggingface::WHISPER_LARGE_V3);
    /// ```
    ///
    fn transcription_model(&self, model: &str) -> TranscriptionModel<reqwest::Client> {
        TranscriptionModel::new(self.clone(), model)
    }
}

#[cfg(feature = "image")]
impl ImageGenerationClient for Client<reqwest::Client> {
    type ImageGenerationModel = ImageGenerationModel<reqwest::Client>;

    /// Create a new image generation model with the given name
    ///
    /// # Example
    /// ```
    /// use rig::providers::huggingface::{Client, self}
    ///
    /// // Initialize the Huggingface client
    /// let client = Client::new("your-huggingface-api-key");
    ///
    /// let completion_model = client.image_generation_model(huggingface::WHISPER_LARGE_V3);
    /// ```
    fn image_generation_model(&self, model: &str) -> ImageGenerationModel<reqwest::Client> {
        ImageGenerationModel::new(self.clone(), model)
    }
}

impl VerifyClient for Client<reqwest::Client> {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/api/whoami-v2")?
            .body(http_client::NoBody)
            .map_err(|e| VerifyError::HttpError(e.into()))?;

        let req = reqwest::Request::try_from(req)
            .map_err(|e| VerifyError::HttpError(http_client::Error::Instance(e.into())))?;

        let response: reqwest::Response = self
            .http_client
            .execute(req)
            .await
            .map_err(|e| VerifyError::HttpError(http_client::Error::Instance(e.into())))?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                let text = response
                    .text()
                    .await
                    .map_err(|e| VerifyError::HttpError(http_client::Error::Instance(e.into())))?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                response
                    .error_for_status()
                    .map_err(|e| VerifyError::HttpError(http_client::Error::Instance(e.into())))?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(AsEmbeddings, AsAudioGeneration for Client<T>);
