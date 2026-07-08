use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
    providers::mistral::MistralModelLister,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

const MISTRAL_API_BASE_URL: &str = "https://api.mistral.ai";

#[derive(Debug, Default, Clone, Copy)]
pub struct MistralExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct MistralBuilder;

type MistralApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<MistralExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<MistralBuilder, MistralApiKey, H>;

impl Provider for MistralExt {
    type Builder = MistralBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl crate::providers::openai::completion::OpenAICompatibleProvider for MistralExt {
    const PROVIDER_NAME: &'static str = "mistral";

    type StreamingUsage = Usage;

    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = true;

    // Mistral is strict about unknown parameters and reports usage on the
    // final stream chunk without `stream_options`.
    const STREAM_INCLUDE_USAGE: bool = false;

    type Response = super::CompletionResponse;

    // The client base URL is the bare host; other Mistral capabilities
    // (embeddings, transcription, model listing) build their own v1 paths.
    fn completion_path(&self, _model: &str) -> String {
        "/v1/chat/completions".to_string()
    }

    fn finalize_request_body(
        &self,
        body: &mut serde_json::Value,
    ) -> Result<(), crate::completion::CompletionError> {
        let Some(map) = body.as_object_mut() else {
            return Ok(());
        };

        // Mistral spells the "must call some tool" mode `any`, not `required`.
        if let Some(tool_choice) = map.get_mut("tool_choice")
            && tool_choice.as_str() == Some("required")
        {
            *tool_choice = serde_json::Value::String("any".to_string());
        }

        if let Some(messages) = map
            .get_mut("messages")
            .and_then(serde_json::Value::as_array_mut)
        {
            for message in messages {
                let Some(message) = message.as_object_mut() else {
                    continue;
                };
                let is_assistant =
                    message.get("role").and_then(serde_json::Value::as_str) == Some("assistant");

                // Mistral takes message `content` as a plain string.
                if let Some(content) = message.get_mut("content") {
                    crate::providers::openai::completion::flatten_text_content_parts(
                        content, "", false,
                    );
                }

                if is_assistant {
                    if !message.contains_key("content") {
                        message.insert(
                            "content".to_string(),
                            serde_json::Value::String(String::new()),
                        );
                    }
                    // `prefix` is part of Mistral's assistant message schema.
                    message
                        .entry("prefix")
                        .or_insert(serde_json::Value::Bool(false));
                    // Mistral rejects unknown assistant fields; hidden
                    // reasoning cannot be echoed back.
                    message.remove("reasoning_content");
                }
            }
        }

        Ok(())
    }
}

impl<H> Capabilities<H> for MistralExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;

    type Transcription = Capable<super::TranscriptionModel<H>>;
    type ModelListing = Capable<MistralModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for MistralExt {}

impl ProviderBuilder for MistralBuilder {
    type Extension<H>
        = MistralExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = MistralApiKey;

    const BASE_URL: &'static str = MISTRAL_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(MistralExt)
    }
}

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new Mistral client from the `MISTRAL_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        let api_key = crate::client::required_env_var("MISTRAL_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

/// In-depth details on prompt tokens.
///
/// Mirrors Mistral's `PromptTokensDetails` schema. The Mistral API also exposes
/// the same shape under the singular field name `prompt_token_details`; the
/// `Usage` field accepts either form via `serde(alias = ...)`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PromptTokensDetails {
    /// Number of tokens served from the prompt cache.
    #[serde(default)]
    pub cached_tokens: u64,
}

/// Token usage returned by Mistral's chat completions and embeddings endpoints.
///
/// See <https://docs.mistral.ai/api/> (`UsageInfo` schema). The three counts are
/// always present; the remaining fields are populated by Mistral on a best-effort
/// basis (e.g. cached-token information appears once a prompt is large enough to
/// be cached).
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    /// Duration in seconds of audio tokens in the prompt (audio-input models only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_audio_seconds: Option<u64>,
    /// Total cached prompt tokens reported at the top level. Some Mistral
    /// responses populate this in addition to (or instead of)
    /// `prompt_tokens_details.cached_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_cached_tokens: Option<u64>,
    /// In-depth breakdown of prompt token usage (currently only cached tokens).
    #[serde(
        default,
        alias = "prompt_token_details",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl Usage {
    /// Returns the number of cached prompt tokens, preferring the structured
    /// `prompt_tokens_details.cached_tokens` field and falling back to the
    /// top-level `num_cached_tokens`. Returns 0 when neither is present.
    pub fn cached_tokens(&self) -> u64 {
        self.prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .or(self.num_cached_tokens)
            .unwrap_or(0)
    }
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

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::mistral::Client::new("dummy-key").expect("Client::new() failed");
        let builder: crate::providers::mistral::ClientBuilder =
            crate::providers::mistral::Client::builder().api_key("dummy-key");
        let _client_from_builder = builder.build().expect("Client::builder() failed");
    }
}
