use super::completion::CompletionModel;
#[cfg(feature = "image")]
use crate::client::ImageGenerationClient;
use crate::client::{CompletionClient, ProviderClient, TranscriptionClient};
#[cfg(feature = "image")]
use crate::image_generation::ImageGenerationError;
#[cfg(feature = "image")]
use crate::providers::huggingface::image_generation::ImageGenerationModel;
use crate::providers::huggingface::transcription::TranscriptionModel;
use crate::transcription::TranscriptionError;
use rig::client::impl_conversion_traits;
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
            sub_provider: SubProvider::default(),
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
    default_headers: reqwest::header::HeaderMap,
    api_key: String,
    http_client: reqwest::Client,
    pub(crate) sub_provider: SubProvider,
}

impl std::fmt::Debug for Client {
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

impl Client {
    /// Create a new Huggingface client with the given API key.
    pub fn new(api_key: &str) -> Self {
        let base_url =
            format!("{}/{}", HUGGINGFACE_API_BASE_URL, SubProvider::HFInference).replace("//", "/");
        Self::from_url(api_key, &base_url, SubProvider::HFInference)
    }

    /// Create a new Client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str, sub_provider: SubProvider) -> Self {
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            "Content-Type",
            "application/json"
                .parse()
                .expect("Failed to parse Content-Type"),
        );
        let http_client = reqwest::Client::builder()
            .build()
            .expect("Failed to build HTTP client");

        Self {
            base_url: base_url.to_owned(),
            api_key: api_key.to_string(),
            default_headers,
            http_client,
            sub_provider,
        }
    }

    /// Use your own `reqwest::Client`.
    /// The API key will be automatically attached upon trying to make a request, so you shouldn't need to add it as a default header.
    pub fn with_custom_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;

        self
    }

    pub(crate) fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client
            .post(url)
            .bearer_auth(&self.api_key)
            .headers(self.default_headers.clone())
    }
}

impl ProviderClient for Client {
    /// Create a new Huggingface client from the `HUGGINGFACE_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("HUGGINGFACE_API_KEY").expect("HUGGINGFACE_API_KEY is not set");
        Self::new(&api_key)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

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
    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl TranscriptionClient for Client {
    type TranscriptionModel = TranscriptionModel;

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
    fn transcription_model(&self, model: &str) -> TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
    }
}

#[cfg(feature = "image")]
impl ImageGenerationClient for Client {
    type ImageGenerationModel = ImageGenerationModel;

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
    fn image_generation_model(&self, model: &str) -> ImageGenerationModel {
        ImageGenerationModel::new(self.clone(), model)
    }
}

impl_conversion_traits!(AsEmbeddings, AsAudioGeneration for Client);
