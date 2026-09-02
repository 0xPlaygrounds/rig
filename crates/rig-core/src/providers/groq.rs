//! Groq API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::{client::CompletionClient, providers::groq};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = groq::Client::new("YOUR_API_KEY")?;
//!
//! let llama = client.completion_model(groq::LLAMA_3_1_8B_INSTANT);
//! # Ok(())
//! # }
//! ```
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::openai;
use crate::client::{self, BearerAuth, DebugExt, Provider};
use crate::completion::CompletionError;
use crate::http_client::HttpClientExt;
use crate::providers::internal::transcription::OpenAiTranscriptionClient;

// ================================================================
// Main Groq Client
// ================================================================
const GROQ_API_BASE_URL: &str = "https://api.groq.com/openai/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct GroqExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct GroqBuilder;

type GroqApiKey = BearerAuth;

impl Provider for GroqExt {
    type Builder = GroqBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl openai::completion::OpenAICompatibleProvider for GroqExt {
    const PROVIDER_NAME: &'static str = "groq";

    /// Groq reports its transport request id on the same `x-request-id`
    /// header OpenAI uses (verified live; see the recorded
    /// `response_identity_edge` fixture, where the header arrives scrubbed).
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");

    type StreamingUsage = openai::Usage;

    const EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS: bool = true;

    type Response = openai::CompletionResponse;

    fn prepare_request(
        &self,
        request: &mut openai::completion::CompletionRequest,
    ) -> Result<(), CompletionError> {
        // Groq's provider-native tools (`browser_search`, `code_interpreter`,
        // ...) arrive via `additional_params.tools`. Left in place they would
        // clobber the function-tool array on serialization, so fold them into
        // `compound_custom.enabled_tools` (deduplicated by tool type).
        let Some(map) = request
            .additional_params
            .as_mut()
            .and_then(Value::as_object_mut)
        else {
            return Ok(());
        };
        let Some(raw_tools) = map.remove("tools") else {
            return Ok(());
        };
        let native_tools = serde_json::from_value::<Vec<Value>>(raw_tools).map_err(|err| {
            CompletionError::RequestError(
                format!("Invalid Groq `additional_params.tools` payload: {err}").into(),
            )
        })?;
        apply_native_tools_to_additional_params(map, native_tools);

        Ok(())
    }
}

client::impl_capabilities!(
    GroqExt,
    completion = CompletionModel<H>,
    transcription = TranscriptionModel<H>,
    model_listing = GroqModelLister<H>,
);

/// A Groq listing entry.
///
/// Groq reports its context window and output ceiling on every entry, and
/// [`Model`](crate::model::Model) has fields for both — `max_output_tokens`
/// exists precisely because rig used to drop a provider-reported output
/// ceiling on the floor (rig#2322). The shared `ListModelEntry` decodes
/// neither (Groq spells them `context_window` / `max_completion_tokens`, and
/// the spellings differ across providers), so Groq keeps its own DTO rather
/// than losing them.
#[derive(Debug, serde::Deserialize)]
struct GroqModelEntry {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    owned_by: Option<String>,
    #[serde(default)]
    context_window: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
}

impl From<GroqModelEntry> for crate::model::Model {
    fn from(value: GroqModelEntry) -> Self {
        let mut model = crate::model::Model::from_id(value.id);
        model.name = value.name;
        model.created_at = value.created;
        model.owned_by = value.owned_by;
        model.context_length = value.context_window;
        model.max_output_tokens = value.max_completion_tokens;
        model
    }
}

crate::providers::internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the Groq
    /// API (`GET /models`), the same path [`GroqExt::VERIFY_PATH`] already
    /// uses.
    GroqModelLister,
    Client<H>,
    GroqModelEntry,
    "Groq",
    "/models"
);

impl DebugExt for GroqExt {}

client::impl_default_provider_builder!(
    GroqBuilder => GroqExt,
    api_key = GroqApiKey,
    base_url = GROQ_API_BASE_URL,
);

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<GroqExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<GroqBuilder, GroqApiKey, H>;

/// Groq completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    openai::completion::GenericCompletionModel<GroqExt, H>;

/// Groq's provider-native terminal streaming record: the value carried by the
/// final item of the stream returned by `CompletionModel::raw_stream`. Shared
/// with the OpenAI Chat Completions path, usage payload included.
pub type StreamingCompletionResponse = openai::StreamingCompletionResponse;

client::impl_provider_from_env!(GroqExt, input = String, api_key_env = "GROQ_API_KEY");

#[cfg(test)]
use crate::providers::openai::client::ApiResponse;

fn apply_native_tools_to_additional_params(
    extra: &mut Map<String, Value>,
    native_tools: Vec<Value>,
) {
    if native_tools.is_empty() {
        return;
    }

    let mut compound_custom = match extra.remove("compound_custom") {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };

    let mut enabled_tools = match compound_custom.remove("enabled_tools") {
        Some(Value::Array(values)) => values,
        _ => Vec::new(),
    };

    for native_tool in native_tools {
        let already_enabled = enabled_tools
            .iter()
            .any(|existing| native_tools_match(existing, &native_tool));
        if !already_enabled {
            enabled_tools.push(native_tool);
        }
    }

    compound_custom.insert("enabled_tools".to_string(), Value::Array(enabled_tools));
    extra.insert(
        "compound_custom".to_string(),
        Value::Object(compound_custom),
    );
}

fn native_tools_match(lhs: &Value, rhs: &Value) -> bool {
    if let (Some(lhs_type), Some(rhs_type)) = (native_tool_kind(lhs), native_tool_kind(rhs)) {
        return lhs_type == rhs_type;
    }

    lhs == rhs
}

fn native_tool_kind(value: &Value) -> Option<&str> {
    match value {
        Value::String(kind) => Some(kind),
        Value::Object(map) => map.get("type").and_then(Value::as_str),
        _ => None,
    }
}

// ================================================================
// Groq Completion API
// ================================================================

/// The `openai/gpt-oss-120b` model. Used for chat completion.
pub const GPT_OSS_120B: &str = "openai/gpt-oss-120b";
/// The `openai/gpt-oss-20b` model. Used for chat completion.
pub const GPT_OSS_20B: &str = "openai/gpt-oss-20b";
/// The `qwen/qwen3.6-27b` model. Used for chat completion.
pub const QWEN_3_6_27B: &str = "qwen/qwen3.6-27b";
/// The `qwen/qwen3.8-27b` model. Used for chat completion.
pub const QWEN_3_8_27B: &str = "qwen/qwen3.8-27b";
/// The `groq/compound` model. Used for compound inference.
pub const GROQ_COMPOUND: &str = "groq/compound";
/// The `groq/compound-mini` model. Used for compound inference.
pub const GROQ_COMPOUND_MINI: &str = "groq/compound-mini";
/// The `openai/gpt-oss-safeguard-20b` model. Used for content moderation.
pub const GPT_OSS_SAFEGUARD_20B: &str = "openai/gpt-oss-safeguard-20b";
/// The `meta-llama/llama-prompt-guard-2-22m` model. Used for content moderation.
pub const LLAMA_PROMPT_GUARD_2_22M: &str = "meta-llama/llama-prompt-guard-2-22m";
/// The `meta-llama/llama-prompt-guard-2-86m` model. Used for content moderation.
pub const LLAMA_PROMPT_GUARD_2_86M: &str = "meta-llama/llama-prompt-guard-2-86m";
/// The `canopylabs/orpheus-v1-english` model. Used for text to speech.
pub const ORPHEUS_V1_ENGLISH: &str = "canopylabs/orpheus-v1-english";
/// The `canopylabs/orpheus-arabic-saudi` model. Used for text to speech.
pub const ORPHEUS_ARABIC_SAUDI: &str = "canopylabs/orpheus-arabic-saudi";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningFormat {
    Parsed,
    Raw,
    Hidden,
}

/// Additional parameters to send to the Groq API. Serialize this into the
/// request's `additional_params` to set Groq's reasoning options.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroqAdditionalParameters {
    /// The reasoning format. See Groq's API docs for more details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_format: Option<ReasoningFormat>,
    /// Whether or not to include reasoning. See Groq's API docs for more details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    /// Any other properties not included by default on this struct (that you want to send)
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Map<String, serde_json::Value>>,
}

// ================================================================
// Groq Transcription API
// ================================================================

/// The `whisper-large-v3` model. Used for transcription.
pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
/// The `whisper-large-v3-turbo` model. Used for transcription.
pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";

/// Groq transcription model using the shared OpenAI-style implementation.
pub type TranscriptionModel<T = crate::http_client::BoxedHttpClient> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<Client<T>>;

impl<T> OpenAiTranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + 'static,
{
    const MODEL_IN_FORM: bool = true;
    const PROVIDER_NAME: &'static str = "groq";
    const REQUEST_ID_HEADER: Option<&'static str> = Some("x-request-id");

    fn transcription_request(
        &self,
        _model: &str,
    ) -> crate::http_client::Result<crate::http_client::Builder> {
        self.post("/audio/transcriptions")
    }
}

#[cfg(test)]
mod tests;
