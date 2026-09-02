//! DeepSeek API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::{client::CompletionClient, providers::deepseek};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = deepseek::Client::new("DEEPSEEK_API_KEY")?;
//!
//! let deepseek_chat = client.completion_model(deepseek::DEEPSEEK_V4_FLASH);
//! # Ok(())
//! # }
//! ```

use serde_json::Value;

use crate::client::{self, BearerAuth, DebugExt, Provider};
use crate::providers::openai;
use crate::telemetry::ProviderResponseExt;
use crate::{
    completion::{self, CompletionError},
    json_utils,
};
use serde::{Deserialize, Serialize};

// ================================================================
// Main DeepSeek Client
// ================================================================
const DEEPSEEK_API_BASE_URL: &str = "https://api.deepseek.com";

#[derive(Debug, Default, Clone, Copy)]
pub struct DeepSeekExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct DeepSeekExtBuilder;

type DeepSeekApiKey = BearerAuth;

impl Provider for DeepSeekExt {
    type Builder = DeepSeekExtBuilder;
    const VERIFY_PATH: &'static str = "/user/balance";
}

impl openai::completion::OpenAICompatibleProvider for DeepSeekExt {
    const PROVIDER_NAME: &'static str = "deepseek";

    type StreamingUsage = Usage;

    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = true;

    // DeepSeek's API only supports `json_object` response formats (passed via
    // `additional_params`), not the `json_schema` mapping of `output_schema`.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = CompletionResponse;

    fn finalize_request_body(&self, body: &mut Value) -> Result<(), CompletionError> {
        let Some(map) = body.as_object_mut() else {
            return Ok(());
        };

        // DeepSeek takes message `content` as a plain string, not an array of
        // content parts, and echoes tool calls back with an `index` field.
        if let Some(messages) = map.get_mut("messages").and_then(Value::as_array_mut) {
            for message in messages {
                let Some(message) = message.as_object_mut() else {
                    continue;
                };
                let is_assistant = message.get("role").and_then(Value::as_str) == Some("assistant");

                if let Some(content) = message.get_mut("content") {
                    let separator = if is_assistant { "" } else { "\n" };
                    // Text-only arrays flatten; an array carrying an image,
                    // audio, video or file part is left alone so DeepSeek's
                    // own rejection reaches the caller ("unknown variant
                    // `image_url`, expected `text`", verified live). Dropping
                    // those parts here answered the question from the text
                    // alone and never told anyone the attachment was gone.
                    openai::completion::flatten_text_content_parts(content, separator, true);
                } else if is_assistant && !message.contains_key("content") {
                    // Tool-call-only assistant turns must still carry an
                    // (empty) string content field.
                    message.insert("content".to_string(), Value::String(String::new()));
                }

                if is_assistant
                    && let Some(tool_calls) =
                        message.get_mut("tool_calls").and_then(Value::as_array_mut)
                {
                    for tool_call in tool_calls {
                        if let Some(tool_call) = tool_call.as_object_mut() {
                            tool_call
                                .entry("index")
                                .or_insert_with(|| serde_json::json!(0));
                        }
                    }
                }
            }
        }

        // DeepSeek rejects forced tool choices (`required` or a specific
        // function) unless thinking is explicitly disabled; suppress them to
        // an explicit `null` otherwise.
        let thinking_disabled = map
            .get("thinking")
            .and_then(|thinking| thinking.get("type"))
            .and_then(Value::as_str)
            .is_some_and(|mode| mode.eq_ignore_ascii_case("disabled"));
        if !thinking_disabled && let Some(tool_choice) = map.get_mut("tool_choice") {
            let forced = tool_choice.is_object() || tool_choice.as_str() == Some("required");
            if forced {
                *tool_choice = Value::Null;
            }
        }

        Ok(())
    }
}

client::impl_capabilities!(
    DeepSeekExt,
    completion = CompletionModel<H>,
    model_listing = DeepSeekModelLister<H>,
);

impl DebugExt for DeepSeekExt {}

client::impl_default_provider_builder!(
    DeepSeekExtBuilder => DeepSeekExt,
    api_key = DeepSeekApiKey,
    base_url = DEEPSEEK_API_BASE_URL,
);

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<DeepSeekExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<DeepSeekExtBuilder, DeepSeekApiKey, H>;

/// DeepSeek completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    openai::completion::GenericCompletionModel<DeepSeekExt, H>;

/// DeepSeek's provider-native terminal streaming record: the value carried by
/// the final item of the stream returned by `CompletionModel::raw_stream`.
/// Shared with the OpenAI Chat Completions path but carrying DeepSeek's own
/// usage payload (cache hit/miss counters).
pub type StreamingCompletionResponse = openai::StreamingCompletionResponse<Usage>;

impl crate::client::ProviderFromEnv for DeepSeekExt {
    type Input = DeepSeekApiKey;
    // If you prefer the environment variable approach:
    fn from_env_with<H>(
        http: H,
    ) -> Result<crate::client::Client<Self, H>, crate::client::ProviderClientError>
    where
        H: crate::http_client::HttpClientExt,
        Self::Builder: crate::client::ProviderBuilder<Extension<H> = Self>,
    {
        let api_key = crate::client::required_env_var("DEEPSEEK_API_KEY")?;
        let mut client_builder = crate::client::Client::<Self, crate::markers::Missing>::builder();
        client_builder.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        let client_builder = client_builder.api_key(&api_key);
        client_builder.http_client(http).build().map_err(Into::into)
    }

    fn from_val_with<H>(
        input: Self::Input,
        http: H,
    ) -> Result<crate::client::Client<Self, H>, crate::client::ProviderClientError>
    where
        H: crate::http_client::HttpClientExt,
        Self::Builder: crate::client::ProviderBuilder<Extension<H> = Self>,
    {
        crate::client::Client::new_with(input, http).map_err(Into::into)
    }
}

/// The response shape from the DeepSeek API
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    #[serde(
        deserialize_with = "crate::providers::internal::openai_chat_completions_compatible::deserialize_choices_dropping_incomplete_tool_calls"
    )]
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

impl ProviderResponseExt for CompletionResponse {
    type Usage = Usage;

    fn response_id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    fn response_model_name(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn text_response(&self) -> Option<String> {
        self.choices
            .iter()
            .find_map(|choice| match &choice.message {
                Message::Assistant { content, .. } if !content.is_empty() => Some(content.clone()),
                _ => None,
            })
    }

    fn usage(&self) -> Option<Self::Usage> {
        Some(self.usage)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    pub prompt_cache_hit_tokens: u32,
    pub prompt_cache_miss_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl From<&Usage> for crate::completion::Usage {
    fn from(usage: &Usage) -> Self {
        let mut normalized = crate::providers::internal::completion_usage(
            usage.prompt_tokens as u64,
            usage.completion_tokens as u64,
            usage.total_tokens as u64,
            usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|details| details.cached_tokens)
                .map_or(u64::from(usage.prompt_cache_hit_tokens), u64::from),
        );
        normalized.reasoning_tokens = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens)
            .map_or(0, u64::from);
        normalized
    }
}

impl From<Usage> for crate::completion::Usage {
    fn from(usage: Usage) -> Self {
        Self::from(&usage)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

/// DeepSeek's provider-native message shape, as it appears in responses.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_default",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<ToolCall>,
        /// only exists on `deepseek-reasoner` model at time of addition
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    pub id: String,
    pub index: usize,
    #[serde(default)]
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    #[serde(with = "json_utils::stringified_json")]
    pub arguments: serde_json::Value,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}

/// Normalize a DeepSeek chat completion response.
///
/// The provider descriptor name is an *input* rather than a constant so the
/// shared OpenAI-compatible completion path labels the response with the
/// descriptor that actually produced it, exactly as it does for the OpenAI
/// wire type.
impl crate::completion::NormalizeCompletionResponse for CompletionResponse {
    fn normalize(self, provider: &str) -> Result<completion::CompletionResponse, CompletionError> {
        use crate::providers::internal::openai_chat_completions_compatible as compat;

        let usage = crate::completion::Usage::from(&self.usage);
        compat::normalize_openai_response(
            provider,
            &self.choices,
            self.id.as_deref(),
            self.model.as_deref(),
            usage,
            |choice| choice.finish_reason.as_str(),
            |choice| {
                let Message::Assistant {
                    content: text,
                    tool_calls,
                    reasoning_content,
                    ..
                } = &choice.message;
                // Reasoning leads the turn, as it does on the streaming
                // path: DeepSeek's stream emits every `reasoning_content`
                // delta before the first `content` delta and before the tool
                // call, and the shared canonical chunk order is the same
                // (reasoning, then text, then tool events). Appending it last
                // made the two transports disagree about identical bytes.
                let mut content = match reasoning_content {
                    Some(reasoning_content) => {
                        vec![completion::AssistantContent::reasoning(reasoning_content)]
                    }
                    None => Vec::new(),
                };

                content.extend(compat::text_then_tool_calls(
                    text,
                    text.trim().is_empty(),
                    tool_calls.iter().map(|call| {
                        (
                            call.id.as_str(),
                            call.function.name.as_str(),
                            call.function.arguments.clone(),
                        )
                    }),
                ));

                Some(content)
            },
        )
    }
}

crate::providers::internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// DeepSeek API (`GET /models`).
    DeepSeekModelLister,
    Client<H>,
    crate::providers::internal::model_listing::ListModelEntry,
    "DeepSeek",
    "/models"
);

// ================================================================
// DeepSeek Completion API
// ================================================================
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";

// Tests
#[cfg(test)]
mod tests;
