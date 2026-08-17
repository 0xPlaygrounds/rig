use crate::{
    client::{self, BearerAuth, DebugExt, Provider},
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
    // The client base URL is the bare host, so every Mistral path carries its
    // own `/v1` — as `completion_path` and `MistralModelLister` already do.
    // `/models` is a gateway 404 ("no Route matched with those values"), which
    // made `verify()` fail for every key, valid or not.
    const VERIFY_PATH: &'static str = "/v1/models";
}

impl crate::providers::openai::completion::OpenAICompatibleProvider for MistralExt {
    const PROVIDER_NAME: &'static str = "mistral";

    /// Mistral labels its transport request id `mistral-correlation-id`, and
    /// sends it on every response — success and error alike. It also mirrors
    /// the same value under the gateway's `x-kong-request-id`; the
    /// provider-branded spelling is the one rig reads.
    const REQUEST_ID_HEADER: Option<&'static str> = Some("mistral-correlation-id");

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

        // Mistral accepts a *structured* response format beside tools only
        // under `tool_choice: auto` (or `none`): anything that forces a call
        // is a 400, "`json_schema` response type with tools is only compatible
        // with `tool_choice: auto`". Rig reaches that combination on its own —
        // a structured-output agent defers `response_format` until a tool
        // result exists, then emits it beside the caller's standing
        // `tool_choice`, so the turn after the first tool call dies. Relaxing
        // the choice keeps both features working; dropping the response format
        // instead would silently discard the schema the caller asked for.
        //
        // Keyed on the format's *type* rather than its presence: the
        // constraint is specific to `json_schema` and `json_object`, and
        // `{"type": "text"}` — the API default, which a caller can still pass
        // explicitly — rides beside a forced choice happily.
        let forces_a_tool_call = map
            .get("tool_choice")
            .is_some_and(|choice| !matches!(choice.as_str(), Some("auto" | "none")));
        let has_tools = map
            .get("tools")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|tools| !tools.is_empty());
        let has_structured_format = map
            .get("response_format")
            .and_then(|format| format.get("type"))
            .and_then(serde_json::Value::as_str)
            .is_some_and(|kind| matches!(kind, "json_schema" | "json_object"));
        if forces_a_tool_call && has_tools && has_structured_format {
            tracing::debug!(
                "relaxing tool_choice to `auto`: Mistral rejects a forced tool choice \
                 alongside a response format"
            );
            map.insert(
                "tool_choice".to_string(),
                serde_json::Value::String("auto".to_string()),
            );
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

                // Mistral takes text-only message `content` as a plain string
                // and carries images, audio and documents as its own chunk
                // array. Content it has no chunk for fails here rather than
                // reaching the API with the part removed.
                if let Some(content) = message.get_mut("content") {
                    super::completion::normalize_request_content(content)?;
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

client::impl_capabilities!(
    MistralExt,
    completion = super::CompletionModel<H>,
    embeddings = super::EmbeddingModel<H>,
    transcription = super::TranscriptionModel<H>,
    model_listing = MistralModelLister<H>,
);

impl DebugExt for MistralExt {}

client::impl_default_provider_builder!(
    MistralBuilder => MistralExt,
    api_key = MistralApiKey,
    base_url = MISTRAL_API_BASE_URL,
);

client::impl_provider_client!(Client, input = String, api_key_env = "MISTRAL_API_KEY");

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
    /// Tokens the audio-input models charge for the prompt's audio. Reported
    /// *alongside* `prompt_tokens` rather than inside it — the two plus
    /// `completion_tokens` are what add up to `total_tokens`.
    #[serde(default)]
    pub audio_tokens: u64,
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
    /// Capacity tier that served the request, when Mistral reports it.
    ///
    /// Although the generated `UsageInfo` reference currently omits this
    /// field, the live chat-completions wire includes values such as
    /// `"standard"` in both blocking responses and terminal stream chunks.
    /// Keeping it here prevents the provider-native `raw_completion` and
    /// `raw_stream` surfaces from silently discarding that wire metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
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

    /// Tokens charged for audio in the prompt. 0 for every non-audio turn.
    pub fn audio_tokens(&self) -> u64 {
        self.prompt_tokens_details
            .as_ref()
            .map_or(0, |details| details.audio_tokens)
    }

    /// Every token charged against the prompt.
    ///
    /// Mistral reports audio outside `prompt_tokens`: a Voxtral turn answering
    /// a 375-audio-token clip reports `prompt_tokens: 6`, `audio_tokens: 375`,
    /// `completion_tokens: 2` and `total_tokens: 383`. Counting only
    /// `prompt_tokens` as input leaves `input + output` short of `total` by the
    /// whole audio payload.
    pub fn input_tokens(&self) -> u64 {
        self.prompt_tokens as u64 + self.audio_tokens()
    }
}

impl From<&Usage> for crate::completion::Usage {
    fn from(usage: &Usage) -> Self {
        crate::providers::internal::completion_usage(
            usage.input_tokens(),
            usage.completion_tokens as u64,
            usage.total_tokens as u64,
            usage.cached_tokens(),
        )
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(usage: Usage) -> Self {
        Self::from(&usage)
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

#[cfg(test)]
mod tests {
    use super::Usage;

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::mistral::Client::new("dummy-key").expect("Client::new() failed");
        let builder: crate::providers::mistral::ClientBuilder =
            crate::providers::mistral::Client::builder().api_key("dummy-key");
        let _client_from_builder = builder.build().expect("Client::builder() failed");
    }

    #[test]
    fn usage_retains_live_service_tier() {
        let usage: Usage = serde_json::from_value(serde_json::json!({
            "completion_tokens": 4,
            "prompt_tokens": 20,
            "total_tokens": 24,
            "prompt_tokens_details": { "cached_tokens": 0 },
            "service_tier": "standard"
        }))
        .expect("live Mistral usage should deserialize");

        assert_eq!(usage.service_tier.as_deref(), Some("standard"));
        assert_eq!(
            serde_json::to_value(usage).expect("Mistral usage should serialize")["service_tier"],
            "standard"
        );
    }
}
