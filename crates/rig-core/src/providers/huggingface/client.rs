#[cfg(feature = "image")]
use crate::client::HasImageGeneration;
use crate::client::{
    self, BearerAuth, HasCompletion, HasTranscription, ModelTransport, Provider,
    ProviderClientResult,
};
use crate::http_client::{self, HttpClientExt};
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
            // Fireworks addresses models by a fully-qualified id. Guard against
            // re-prefixing an already-qualified id (e.g. a per-request model
            // override that is already fully qualified) — the generic path
            // applies this to the resolved request model unconditionally, so
            // without the guard a qualified override would become an invalid
            // `accounts/fireworks/models/accounts/fireworks/models/...` id.
            SubProvider::Fireworks => {
                const FIREWORKS_PREFIX: &str = "accounts/fireworks/models/";
                if model.starts_with(FIREWORKS_PREFIX) {
                    model.to_string()
                } else {
                    format!("{FIREWORKS_PREFIX}{model}")
                }
            }
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

/// The Hugging Face Inference Providers router, routed to one sub-provider.
#[derive(Debug, Default, Clone)]
pub struct HuggingFace {
    subprovider: SubProvider,
}

/// Builder settings for [`HuggingFace`]: which sub-provider the router sends to.
#[derive(Debug, Default, Clone)]
pub struct HuggingFaceConfig {
    subprovider: SubProvider,
}

type HuggingFaceApiKey = BearerAuth;

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<HuggingFace, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<HuggingFace, H>;

impl Provider for HuggingFace {
    const NAME: &'static str = "huggingface";
    const BASE_URL: &'static str = HUGGINGFACE_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/api/whoami-v2";
    type ApiKey = HuggingFaceApiKey;
    type Config = HuggingFaceConfig;
    type EnvInput = String;

    fn build(config: HuggingFaceConfig, _: &HuggingFaceApiKey) -> http_client::Result<Self> {
        Ok(HuggingFace {
            subprovider: config.subprovider,
        })
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("HUGGINGFACE_API_KEY", None, http)
    }

    fn from_val<H: HttpClientExt>(input: String, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for HuggingFace {
    type Model<H>
        = super::completion::CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::completion::CompletionModel::new(client.clone(), model)
    }
}

impl HasTranscription for HuggingFace {
    type Model<H>
        = super::transcription::TranscriptionModel<H>
    where
        H: ModelTransport;

    fn transcription_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::transcription::TranscriptionModel::new(client.clone(), model)
    }
}

#[cfg(feature = "image")]
impl HasImageGeneration for HuggingFace {
    type Model<H>
        = super::image_generation::ImageGenerationModel<H>
    where
        H: ModelTransport;

    fn image_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::image_generation::ImageGenerationModel::new(client.clone(), model)
    }
}

impl crate::providers::openai::completion::OpenAICompatibleProvider for HuggingFace {
    const PROVIDER_NAME: &'static str = "huggingface";

    type StreamingUsage = crate::providers::openai::Usage;

    // Structured-output support varies by sub-provider; keep the
    // pre-migration behavior of dropping `output_schema` with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = crate::providers::openai::CompletionResponse;

    // Chat completions live under the router's `/v1` while verification,
    // transcription, and image generation use root-relative paths, so the
    // prefix cannot live in the client base URL.
    fn completion_path(&self, _model: &str) -> String {
        self.subprovider.completion_endpoint(_model)
    }

    fn prepare_request(
        &self,
        request: &mut crate::providers::openai::completion::CompletionRequest,
    ) -> Result<(), crate::completion::CompletionError> {
        // Some sub-providers (Fireworks) address models through a qualified
        // identifier in the request body.
        request.model = self.subprovider.model_identifier(&request.model);
        Ok(())
    }
}

impl<H> ClientBuilder<H> {
    pub fn subprovider(mut self, subprovider: SubProvider) -> Self {
        self.config_mut().subprovider = subprovider;
        self
    }
}

impl<H> Client<H> {
    pub(crate) fn subprovider(&self) -> &SubProvider {
        &self.provider().subprovider
    }
}
#[cfg(test)]
mod tests;
