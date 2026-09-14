use super::responses_api::{
    ConfigurableSystemInstructionsPlacement, ResponsesProviderExt, SystemInstructionsPlacement,
};
#[cfg(feature = "audio")]
use crate::client::HasAudioGeneration;
#[cfg(feature = "image")]
use crate::client::HasImageGeneration;
use crate::{
    client::{
        self, BearerAuth, HasCompletion, HasEmbeddings, HasModelListing, HasTranscription,
        ModelTransport, Provider, ProviderClientResult,
    },
    http_client::{self, HttpClientExt},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::Deserialize;
use std::fmt::Debug;

// ================================================================
// Main OpenAI Client
// ================================================================
const OPENAI_API_BASE_URL: &str = "https://api.openai.com/v1";

// ================================================================
// OpenAI Responses API Extension
// ================================================================
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAIResponses {
    pub(crate) system_instructions_placement: SystemInstructionsPlacement,
}

// ================================================================
// OpenAI Completions API Extension
// ================================================================
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAICompletions {
    /// Carried through API switches so that a placement configured on a
    /// Responses client survives `completions_api()` → `responses_api()`
    /// round trips. Not used by Chat Completions requests themselves.
    pub(crate) system_instructions_placement: SystemInstructionsPlacement,
}

type OpenAIApiKey = BearerAuth;

// Responses API client (default)
pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<OpenAIResponses, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<OpenAIResponses, H>;

// Completions API client
pub type CompletionsClient<H = crate::http_client::BoxedHttpClient> =
    client::Client<OpenAICompletions, H>;
pub type CompletionsClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<OpenAICompletions, H>;

impl Provider for OpenAIResponses {
    const NAME: &'static str = "openai";
    const BASE_URL: &'static str = OPENAI_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/models";
    type ApiKey = OpenAIApiKey;
    type Config = ();
    type EnvInput = OpenAIApiKey;

    fn build(_: (), _: &OpenAIApiKey) -> http_client::Result<Self> {
        Ok(OpenAIResponses::default())
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("OPENAI_API_KEY", Some("OPENAI_BASE_URL"), http)
    }

    fn from_val<H: HttpClientExt>(input: OpenAIApiKey, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for OpenAIResponses {
    type Model<H>
        = super::responses_api::ResponsesCompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::responses_api::ResponsesCompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for OpenAIResponses {
    type Model<H>
        = super::EmbeddingModel<H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        super::EmbeddingModel::make(client, model, ndims)
    }
}

impl HasTranscription for OpenAIResponses {
    type Model<H>
        = super::TranscriptionModel<H>
    where
        H: ModelTransport;

    fn transcription_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::TranscriptionModel::new(client.clone(), model)
    }
}

impl HasModelListing for OpenAIResponses {
    type Lister<H>
        = super::OpenAIModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &Client<H>) -> Self::Lister<H> {
        super::OpenAIModelLister::new(client.clone())
    }
}

#[cfg(feature = "image")]
impl HasImageGeneration for OpenAIResponses {
    type Model<H>
        = super::ImageGenerationModel<H>
    where
        H: ModelTransport;

    fn image_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::ImageGenerationModel::new(client.clone(), model)
    }
}

#[cfg(feature = "audio")]
impl HasAudioGeneration for OpenAIResponses {
    type Model<H>
        = super::audio_generation::AudioGenerationModel<H>
    where
        H: ModelTransport;

    fn audio_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::audio_generation::AudioGenerationModel::new(client.clone(), model)
    }
}

impl ResponsesProviderExt for OpenAIResponses {
    fn system_instructions_placement(&self) -> SystemInstructionsPlacement {
        self.system_instructions_placement
    }
}

impl ConfigurableSystemInstructionsPlacement for OpenAIResponses {}

impl Provider for OpenAICompletions {
    const NAME: &'static str = "openai";
    const BASE_URL: &'static str = OPENAI_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/models";
    type ApiKey = OpenAIApiKey;
    type Config = ();
    type EnvInput = OpenAIApiKey;

    fn build(_: (), _: &OpenAIApiKey) -> http_client::Result<Self> {
        Ok(OpenAICompletions::default())
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<CompletionsClient<H>> {
        CompletionsClient::from_env_api_key("OPENAI_API_KEY", Some("OPENAI_BASE_URL"), http)
    }

    fn from_val<H: HttpClientExt>(
        input: OpenAIApiKey,
        http: H,
    ) -> ProviderClientResult<CompletionsClient<H>> {
        CompletionsClient::new_with(input, http)
    }
}

impl HasCompletion for OpenAICompletions {
    type Model<H>
        = super::completion::CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
    ) -> Self::Model<H> {
        super::completion::CompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for OpenAICompletions {
    type Model<H>
        = super::GenericEmbeddingModel<OpenAICompletions, H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        super::GenericEmbeddingModel::make(client, model, ndims)
    }
}

impl HasTranscription for OpenAICompletions {
    type Model<H>
        = super::CompletionsTranscriptionModel<H>
    where
        H: ModelTransport;

    fn transcription_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
    ) -> Self::Model<H> {
        super::CompletionsTranscriptionModel::new(client.clone(), model)
    }
}

impl HasModelListing for OpenAICompletions {
    type Lister<H>
        = super::OpenAICompletionsModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &CompletionsClient<H>) -> Self::Lister<H> {
        super::OpenAICompletionsModelLister::new(client.clone())
    }
}

#[cfg(feature = "image")]
impl HasImageGeneration for OpenAICompletions {
    type Model<H>
        = super::CompletionsImageGenerationModel<H>
    where
        H: ModelTransport;

    fn image_generation_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
    ) -> Self::Model<H> {
        super::CompletionsImageGenerationModel::new(client.clone(), model)
    }
}

#[cfg(feature = "audio")]
impl HasAudioGeneration for OpenAICompletions {
    type Model<H>
        = super::audio_generation::CompletionsAudioGenerationModel<H>
    where
        H: ModelTransport;

    fn audio_generation_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
    ) -> Self::Model<H> {
        super::audio_generation::CompletionsAudioGenerationModel::new(client.clone(), model)
    }
}

impl<H> Client<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Sets where Rig system instructions are placed in Responses requests for
    /// every completion model created from this client. Models capture the
    /// placement when they are created, so models built before this call are
    /// unaffected. See [`SystemInstructionsPlacement`] for when each placement applies.
    pub fn with_system_instructions_placement(
        self,
        placement: SystemInstructionsPlacement,
    ) -> Self {
        let mut ext = *self.provider();
        ext.system_instructions_placement = placement;
        self.with_provider(ext)
    }

    /// Sends Rig system instructions as `system` messages in `input` instead of
    /// as top-level Responses API `instructions` for every completion model
    /// created from this client. Models built before this call are unaffected.
    ///
    /// OpenAI's Responses API supports `instructions`, and Rig uses it by
    /// default. Use this compatibility fallback for OpenAI-compatible providers
    /// that reject or ignore top-level `instructions`.
    pub fn with_system_instructions_as_messages(self) -> Self {
        self.with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
    }

    /// Create a Completions API client from this Responses API client.
    /// Useful for switching to the traditional Chat Completions API.
    pub fn completions_api(self) -> CompletionsClient<H> {
        let system_instructions_placement = self.provider().system_instructions_placement;
        self.with_provider(OpenAICompletions {
            system_instructions_placement,
        })
    }
}

impl<H> CompletionsClient<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Create a Responses API client from this Completions API client.
    /// Useful for switching to the newer Responses API. A system-instructions
    /// placement configured before switching to the Completions API is
    /// restored.
    pub fn responses_api(self) -> Client<H> {
        let system_instructions_placement = self.provider().system_instructions_placement;
        self.with_provider(OpenAIResponses {
            system_instructions_placement,
        })
    }
}

/// Error envelope returned by OpenAI-compatible providers alongside 2xx
/// statuses. Providers spell the message field differently (`message`,
/// `error`, nested objects), so anything that isn't a valid success payload
/// is treated as an error envelope and the raw body is preserved for the
/// caller; `message` is only used for logging.
#[derive(Debug)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

// Manual impl (not a field-level `alias = "error"`): the alias makes serde
// treat `message` and `error` as one field, so a body carrying both keys
// fails as a duplicate field instead of classifying as this envelope.
impl<'de> Deserialize<'de> for ApiErrorResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            message: crate::providers::internal::envelope::error_message(deserializer)?,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[cfg(test)]
mod tests;
