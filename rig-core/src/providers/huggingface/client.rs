use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client;
#[cfg(feature = "image")]
use crate::image_generation::ImageGenerationError;
use crate::transcription::TranscriptionError;
use std::fmt::Debug;
use std::fmt::Display;

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
    pub fn completion_endpoint(&self, _model: &str) -> String {
        "v1/chat/completions".to_string()
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

// ================================================================
// Main Huggingface Client
// ================================================================
const HUGGINGFACE_API_BASE_URL: &str = "https://router.huggingface.co";

#[derive(Debug, Default, Clone)]
pub struct HuggingFaceExt {
    subprovider: SubProvider,
}

#[derive(Debug, Default, Clone)]
pub struct HuggingFaceBuilder {
    subprovider: SubProvider,
}

type HuggingFaceApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<HuggingFaceExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<HuggingFaceBuilder, HuggingFaceApiKey, H>;

impl Provider for HuggingFaceExt {
    type Builder = HuggingFaceBuilder;

    const VERIFY_PATH: &'static str = "/api/whoami-v2";

    fn build<H>(
        builder: &client::ClientBuilder<Self::Builder, HuggingFaceApiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self {
            subprovider: builder.ext().subprovider.clone(),
        })
    }
}

impl<H> Capabilities<H> for HuggingFaceExt {
    type Completion = Capable<super::completion::CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Capable<super::transcription::TranscriptionModel<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Capable<super::image_generation::ImageGenerationModel<H>>;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for HuggingFaceExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn Debug)> {
        std::iter::once(("subprovider", (&self.subprovider as &dyn Debug)))
    }
}

impl ProviderBuilder for HuggingFaceBuilder {
    type Output = HuggingFaceExt;
    type ApiKey = HuggingFaceApiKey;

    const BASE_URL: &'static str = HUGGINGFACE_API_BASE_URL;
}

impl ProviderClient for Client {
    type Input = String;

    /// Create a new Huggingface client from the `HUGGINGFACE_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("HUGGINGFACE_API_KEY").expect("HUGGINGFACE_API_KEY is not set");

        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
    }
}

impl<H> ClientBuilder<H> {
    pub fn subprovider(mut self, subprovider: SubProvider) -> Self {
        *self.ext_mut() = HuggingFaceBuilder { subprovider };
        self
    }
}

impl<H> Client<H> {
    pub(crate) fn subprovider(&self) -> &SubProvider {
        &self.ext().subprovider
    }
}
