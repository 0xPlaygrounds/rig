use super::responses_api::{
    ConfigurableSystemInstructionsPlacement, ResponsesProviderExt, SystemInstructionsPlacement,
};
use crate::{
    client::{self, BearerAuth, DebugExt, Provider},
    http_client::HttpClientExt,
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
pub struct OpenAIResponsesExt {
    pub(crate) system_instructions_placement: SystemInstructionsPlacement,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAIResponsesExtBuilder;

// ================================================================
// OpenAI Completions API Extension
// ================================================================
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAICompletionsExt {
    /// Carried through API switches so that a placement configured on a
    /// Responses client survives `completions_api()` → `responses_api()`
    /// round trips. Not used by Chat Completions requests themselves.
    pub(crate) system_instructions_placement: SystemInstructionsPlacement,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAICompletionsExtBuilder;

type OpenAIApiKey = BearerAuth;

// Responses API client (default)
pub type Client<H> = client::Client<OpenAIResponsesExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<OpenAIResponsesExtBuilder, OpenAIApiKey, H>;

// Completions API client
pub type CompletionsClient<H> = client::Client<OpenAICompletionsExt, H>;
pub type CompletionsClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<OpenAICompletionsExtBuilder, OpenAIApiKey, H>;

impl Provider for OpenAIResponsesExt {
    type Builder = OpenAIResponsesExtBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl ResponsesProviderExt for OpenAIResponsesExt {
    fn system_instructions_placement(&self) -> SystemInstructionsPlacement {
        self.system_instructions_placement
    }
}

impl ConfigurableSystemInstructionsPlacement for OpenAIResponsesExt {}

impl Provider for OpenAICompletionsExt {
    type Builder = OpenAICompletionsExtBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

client::impl_capabilities!(
    OpenAIResponsesExt,
    completion = super::responses_api::ResponsesCompletionModel<H>,
    embeddings = super::EmbeddingModel<H>,
    transcription = super::TranscriptionModel<H>,
    model_listing = super::OpenAIModelLister<H>,
    image_generation = super::ImageGenerationModel<H>,
    audio_generation = super::audio_generation::AudioGenerationModel<H>,
);

client::impl_capabilities!(
    OpenAICompletionsExt,
    completion = super::completion::CompletionModel<H>,
    embeddings = super::GenericEmbeddingModel<OpenAICompletionsExt, H>,
    transcription = super::CompletionsTranscriptionModel<H>,
    model_listing = super::OpenAICompletionsModelLister<H>,
    image_generation = super::CompletionsImageGenerationModel<H>,
    audio_generation = super::audio_generation::CompletionsAudioGenerationModel<H>,
);

impl DebugExt for OpenAIResponsesExt {}

impl DebugExt for OpenAICompletionsExt {}

client::impl_default_provider_builder!(
    OpenAIResponsesExtBuilder => OpenAIResponsesExt,
    api_key = OpenAIApiKey,
    base_url = OPENAI_API_BASE_URL,
);
client::impl_default_provider_builder!(
    OpenAICompletionsExtBuilder => OpenAICompletionsExt,
    api_key = OpenAIApiKey,
    base_url = OPENAI_API_BASE_URL,
);

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
        let mut ext = *self.ext();
        ext.system_instructions_placement = placement;
        self.with_ext(ext)
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
        let system_instructions_placement = self.ext().system_instructions_placement;
        self.with_ext(OpenAICompletionsExt {
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
        let system_instructions_placement = self.ext().system_instructions_placement;
        self.with_ext(OpenAIResponsesExt {
            system_instructions_placement,
        })
    }
}

client::impl_provider_from_env!(
    OpenAIResponsesExt,
    input = OpenAIApiKey,
    api_key_env = "OPENAI_API_KEY",
    base_url_env_first = "OPENAI_BASE_URL",
);
client::impl_provider_from_env!(
    OpenAICompletionsExt,
    input = OpenAIApiKey,
    api_key_env = "OPENAI_API_KEY",
    base_url_env_first = "OPENAI_BASE_URL",
);

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
