//! Completion wires and types for the [Gemini GenerateContent API](https://ai.google.dev/api/generate-content).
//!
//! ```no_run
//! use rig_core::providers::gemini::{Gemini, completion::GEMINI_2_5_FLASH};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wire = Gemini::from_env()?.generate_content(GEMINI_2_5_FLASH);
//! # Ok(())
//! # }
//! ```
/// `gemini-3.1-flash-lite-preview` completion model
pub const GEMINI_3_1_FLASH_LITE_PREVIEW: &str = "gemini-3.1-flash-lite-preview";
/// `gemini-3-flash-preview` completion model
pub const GEMINI_3_FLASH_PREVIEW: &str = "gemini-3-flash-preview";
/// `gemini-2.5-pro-preview-06-05` completion model
pub const GEMINI_2_5_PRO_PREVIEW_06_05: &str = "gemini-2.5-pro-preview-06-05";
/// `gemini-2.5-pro-preview-05-06` completion model
pub const GEMINI_2_5_PRO_PREVIEW_05_06: &str = "gemini-2.5-pro-preview-05-06";
/// `gemini-2.5-pro-preview-03-25` completion model
pub const GEMINI_2_5_PRO_PREVIEW_03_25: &str = "gemini-2.5-pro-preview-03-25";
/// `gemini-2.5-flash-preview-04-17` completion model
pub const GEMINI_2_5_FLASH_PREVIEW_04_17: &str = "gemini-2.5-flash-preview-04-17";
/// `gemini-2.5-pro-exp-03-25` experimental completion model
pub const GEMINI_2_5_PRO_EXP_03_25: &str = "gemini-2.5-pro-exp-03-25";
/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.5-flash-image` image generation model, commonly referred to as Nano Banana.
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use self::gemini_api_types::tool_parameters_to_schema;
use crate::completion::{self, CompletionRequest};
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, FunctionCallingMode, ToolConfig,
};
use crate::telemetry::GenAiOperation;
use crate::wire::{Body, Encoded, Framing, Mode, Wire};
use gemini_api_types::{
    Content, FinishReason, FunctionDeclaration, GenerateContentRequest, GenerationConfig, Part,
    PartKind, Role, Tool,
};
use serde_json::{Map, Value};
use std::convert::TryFrom;

/// Provider name used in normalized responses, streams, and telemetry.
pub const PROVIDER_NAME: &str = "gcp.gemini";

/// Completion wire for unary `generateContent` and SSE `streamGenerateContent`.
/// Both modes use [`GenerateContentDecoder`](super::streaming::GenerateContentDecoder).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GenerateContent {
    /// The key and the API root.
    pub provider: super::Gemini,
    /// The model to address, e.g. [`GEMINI_2_5_FLASH`].
    pub model: String,
    /// Handle of a `cachedContents` resource every request reads its prefix
    /// from. See [`Self::with_cached_content`].
    pub cached_content: Option<String>,
}

impl GenerateContent {
    /// The wire for `model`.
    pub fn new(provider: super::Gemini, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            cached_content: None,
        }
    }

    /// Use an explicit `cachedContents/<id>` handle as every request's prefix.
    /// Encoding rejects requests with their own system instruction, tools, or
    /// tool choice. See [`crate::providers::gemini::cached_content`] for cache ownership.
    pub fn with_cached_content(mut self, name: impl Into<String>) -> Self {
        self.cached_content = Some(name.into());
        self
    }
}

impl Wire for GenerateContent {
    type Op = Completion;
    type Decoder = super::streaming::GenerateContentDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    /// Select the telemetry operation for unary or streamed completion.
    fn telemetry(&self, streaming: bool) -> GenAiOperation {
        if streaming {
            GenAiOperation::ChatStreaming
        } else {
            GenAiOperation::GenerateContent
        }
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        // The request may name a model of its own; the wire's is the default.
        let model = resolve_request_model(&self.model, &request);
        let mut body = create_request_body(request)?;
        if let Some(name) = self.cached_content.as_deref() {
            body.with_cached_content(name)?;
        }
        let (path, framing, target) = match mode {
            Mode::Unary => (
                completion_endpoint(&model),
                Framing::Whole,
                crate::providers::internal::LogTarget::Completions,
            ),
            // `alt=sse` is what makes the streamed reply an event stream
            // rather than a JSON array of the same chunks.
            Mode::Streaming => (
                format!("{}?alt=sse", streaming_endpoint(&model)),
                Framing::Sse,
                crate::providers::internal::LogTarget::Streaming,
            ),
        };
        crate::providers::internal::trace_json(target, "Gemini completion request", &body);
        let request = http::Request::post(self.provider.uri(&path))
            .header("Content-Type", "application/json")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        // Gemini supplies no transport request-id response header.
        Ok(Encoded::new(request, framing))
    }

    fn decoder(&self, mode: Mode) -> Self::Decoder {
        super::streaming::GenerateContentDecoder::new(mode)
    }
}

pub(crate) fn create_request_body(
    completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, EncodeError> {
    let chat_history = completion_request.chat_history_with_documents();

    let CompletionRequest {
        model: _,
        chat_history: _,
        documents: _,
        tools: function_tools,
        temperature,
        max_tokens,
        tool_choice,
        mut additional_params,
        output_schema,
        record_telemetry_content: _,
    } = completion_request;

    let mut full_history = Vec::new();
    full_history.extend(chat_history);
    // functionResponse.name keys the replay: cross-provider ingested
    // results arrive with an empty name and their call carries it.
    crate::providers::internal::resolve_empty_tool_result_names(&mut full_history);
    let (history_system, full_history) = split_system_messages_from_history(full_history);

    let mut additional_params_payload = additional_params
        .take()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mut additional_tools =
        extract_tools_from_additional_params(&mut additional_params_payload)?;
    // Validate both proto JSON spellings through the typed cache field so
    // flattened parameters cannot bypass conflicts or specify competing handles.
    let mut smuggled_cached_content = Vec::new();
    for spelling in CACHED_CONTENT {
        let Some(value) = additional_params_payload
            .as_object_mut()
            .and_then(|object| object.remove(spelling))
        else {
            continue;
        };
        match value {
            Value::String(name) => smuggled_cached_content.push(name),
            other => {
                return Err(EncodeError::request(format!(
                    "additional_params.{spelling} should be a string, got {other}"
                )));
            }
        }
    }
    // Preserve untyped instructions and tool configuration verbatim; typed
    // conversion could discard restrictions or reject unmodeled provider values.
    let smuggled_system_instruction =
        smuggled_field(&additional_params_payload, &SYSTEM_INSTRUCTION);
    let smuggled_tool_config = smuggled_field(&additional_params_payload, &TOOL_CONFIG);

    let AdditionalParameters {
        mut generation_config,
        additional_params,
    } = serde_json::from_value::<AdditionalParameters>(additional_params_payload)?;

    if let Some(schema) = output_schema {
        let cfg = generation_config.get_or_insert_with(GenerationConfig::default);
        cfg.response_mime_type = Some("application/json".to_string());
        cfg.response_json_schema = Some(schema.to_value());
    }

    // Explicit limits must work without additional generation parameters.
    // Unset fields remain absent so model defaults still apply.
    if temperature.is_some() || max_tokens.is_some() {
        let cfg = generation_config.get_or_insert_with(GenerationConfig::default);

        if let Some(temp) = temperature {
            cfg.temperature = Some(temp);
        }

        if let Some(max_tokens) = max_tokens {
            cfg.max_output_tokens = Some(max_tokens);
        }
    }

    let mut system_parts: Vec<Part> = Vec::new();
    for content in history_system {
        if !content.is_empty() {
            system_parts.push(content.into());
        }
    }
    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(Content {
            parts: system_parts,
            role: Some(Role::Model),
        })
    };
    // Gemini rejects duplicate system-instruction fields rather than selecting one.
    if let (Some(typed), Some(spelling)) = (&system_instruction, smuggled_system_instruction) {
        return Err(EncodeError::request(format!(
            "a Gemini request set the system instruction twice — once as a preamble or \
                 system message ({} part(s)) and once through `additional_params.{spelling}`. \
                 Both would reach the wire, and Gemini rejects that outright: \
                 `system_instruction` is an optional proto field, so a second one is `oneof \
                 field '_system_instruction' is already set`. Set it one way or the other",
            typed.parts.len()
        )));
    }

    let mut tools = if function_tools.is_empty() {
        Vec::new()
    } else {
        vec![serde_json::to_value(Tool::try_from(function_tools)?)?]
    };
    tools.append(&mut additional_tools);
    let tools = if tools.is_empty() { None } else { Some(tools) };

    let tool_config = if let Some(cfg) = tool_choice {
        Some(ToolConfig {
            function_calling_config: Some(FunctionCallingMode::try_from(cfg)?),
        })
    } else {
        None
    };
    // Same rule as the system instruction above: `tool_choice` and
    // `additional_params.toolConfig` are one field reached two ways.
    if tool_config.is_some()
        && let Some(spelling) = smuggled_tool_config
    {
        return Err(EncodeError::request(format!(
            "a Gemini request set the tool choice twice — once as `tool_choice` and once \
                 through `additional_params.{spelling}`. Both would reach the wire, and Gemini \
                 does not take the last — it *merges* them, so the two allowed-function lists \
                 are unioned and the narrower `tool_choice` silently stops restricting anything. \
                 Set it one way or the other"
        )));
    }

    let mut request = GenerateContentRequest {
        contents: full_history
            .into_iter()
            .map(|msg| msg.try_into().map_err(EncodeError::request))
            .collect::<Result<Vec<_>, _>>()?,
        generation_config,
        safety_settings: None,
        tools,
        tool_config,
        system_instruction,
        cached_content: None,
        additional_params,
    };

    for name in smuggled_cached_content {
        request.with_cached_content(&name)?;
    }

    Ok(request)
}

/// Split system messages out of a chat history, keeping their contents in
/// order. Shared with sibling Gemini transports (e.g. `rig-gemini-grpc`).
pub fn split_system_messages_from_history(
    history: Vec<completion::Message>,
) -> (Vec<String>, Vec<completion::Message>) {
    let mut system = Vec::new();
    let mut remaining = Vec::new();

    for message in history {
        match message {
            completion::Message::System { content } => system.push(content),
            other => remaining.push(other),
        }
    }

    (system, remaining)
}

/// Proto3 JSON accepts both lowerCamelCase and original proto field names.
const SYSTEM_INSTRUCTION: [&str; 2] = ["systemInstruction", "system_instruction"];

/// Both spellings Gemini accepts for `toolConfig`. See [`SYSTEM_INSTRUCTION`].
const TOOL_CONFIG: [&str; 2] = ["toolConfig", "tool_config"];

/// Both spellings Gemini accepts for `cachedContent`. See [`SYSTEM_INSTRUCTION`].
const CACHED_CONTENT: [&str; 2] = ["cachedContent", "cached_content"];

/// The shared proto and JSON spelling of the tools field.
const TOOLS: [&str; 1] = ["tools"];

/// Return the first spelling present with a non-null value in `payload`.
/// Inspect presence without narrowing untyped provider fields.
fn smuggled_field<'a>(payload: &Value, spellings: &[&'a str]) -> Option<&'a str> {
    let object = payload.as_object()?;
    spellings
        .iter()
        .find(|spelling| object.get(**spelling).is_some_and(|value| !value.is_null()))
        .copied()
}

fn extract_tools_from_additional_params(
    additional_params: &mut Value,
) -> Result<Vec<Value>, EncodeError> {
    if let Some(map) = additional_params.as_object_mut()
        && let Some(raw_tools) = map.remove("tools")
    {
        return serde_json::from_value::<Vec<Value>>(raw_tools).map_err(|err| {
            EncodeError::request(format!(
                "Invalid Gemini `additional_params.tools` payload: {err}"
            ))
        });
    }

    Ok(Vec::new())
}

pub(crate) fn resolve_request_model(
    default_model: &str,
    completion_request: &CompletionRequest,
) -> String {
    completion_request
        .model
        .clone()
        .unwrap_or_else(|| default_model.to_string())
}

pub(crate) fn completion_endpoint(model: &str) -> String {
    format!("/v1beta/models/{model}:generateContent")
}

pub(crate) fn streaming_endpoint(model: &str) -> String {
    format!("/v1beta/models/{model}:streamGenerateContent")
}

impl TryFrom<Vec<completion::ToolDefinition>> for Tool {
    type Error = EncodeError;

    fn try_from(tools: Vec<completion::ToolDefinition>) -> Result<Self, Self::Error> {
        let mut function_declarations = Vec::new();

        for tool in tools {
            let parameters = tool_parameters_to_schema(tool.parameters).map_err(|error| {
                // The reason without the inner error's own `RequestError:` prefix.
                let reason = std::error::Error::source(&error)
                    .map_or_else(|| error.to_string(), ToString::to_string);
                EncodeError::request(format!(
                    "Tool '{}' could not be converted to a schema: {reason}",
                    tool.name
                ))
            })?;

            function_declarations.push(FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters,
            });
        }

        Ok(Self {
            function_declarations,
            code_execution: None,
        })
    }
}

/// The wire spelling of a serde enum (`SCREAMING_SNAKE_CASE`, or the raw
/// string of an `Unknown` variant), for messages that quote the provider.
mod erased_wire {
    pub(super) trait Wire {
        fn wire_name(&self) -> String;
    }
    impl<T: serde::Serialize> Wire for T {
        fn wire_name(&self) -> String {
            match serde_json::to_value(self) {
                Ok(serde_json::Value::String(name)) => name,
                Ok(other) => other.to_string(),
                Err(_) => "<unserializable>".to_owned(),
            }
        }
    }
}

/// Convert a specified prompt block reason into a provider error with safety ratings.
/// Content refusals are final; unknown and `OTHER` reasons are transient.
pub(crate) fn blocked_prompt_error(
    feedback: &gemini_api_types::PromptFeedback,
) -> Option<ProviderError> {
    let reason = match feedback.block_reason.as_ref()? {
        // Documented as unused: the zero value is never sent, and it names
        // no block if it ever were.
        gemini_api_types::BlockReason::BlockReasonUnspecified => return None,
        reason => reason,
    };
    let wire = |value: &dyn erased_wire::Wire| value.wire_name();
    let ratings = feedback
        .safety_ratings
        .as_ref()
        .filter(|ratings| !ratings.is_empty())
        .map(|ratings| {
            ratings
                .iter()
                .map(|rating| format!("{}={}", wire(&rating.category), wire(&rating.probability)))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .map(|ratings| format!(", safety_ratings=[{ratings}]"))
        .unwrap_or_default();
    let message = format!(
        "Gemini blocked the prompt: block_reason={}{ratings}",
        reason.as_wire_str()
    );
    Some(match reason {
        gemini_api_types::BlockReason::Safety
        | gemini_api_types::BlockReason::Blocklist
        | gemini_api_types::BlockReason::ProhibitedContent
        | gemini_api_types::BlockReason::BlockReasonUnspecified => ProviderError::ProviderResponse(
            crate::provider_response::ProviderResponseError::without_status(message)
                .with_code(Some(reason.as_wire_str().to_owned()))
                .with_refusal(true),
        ),
        gemini_api_types::BlockReason::Other | gemini_api_types::BlockReason::Unknown(_) => {
            ProviderError::ProviderResponse(
                crate::provider_response::ProviderResponseError::without_status(message)
                    .with_code(Some(reason.as_wire_str().to_owned()))
                    .with_transient(Some(true)),
            )
        }
    })
}

pub(crate) fn function_call_finish_reason_error(
    reason: &FinishReason,
    finish_message: Option<&str>,
) -> Option<ProviderError> {
    match reason {
        FinishReason::MalformedFunctionCall
        | FinishReason::UnexpectedToolCall
        | FinishReason::MissingThoughtSignature
        | FinishReason::TooManyToolCalls
        | FinishReason::MalformedResponse => {
            let message = finish_message.unwrap_or("no finish message provided");
            Some(ProviderError::Response(format!(
                "Gemini stopped with finish_reason={reason:?}: {message}"
            )))
        }
        _ => None,
    }
}

/// The wire name of a part kind, for error messages.
pub(crate) fn part_kind_name(part: &PartKind) -> &'static str {
    match part {
        PartKind::Text(_) => "text",
        PartKind::InlineData(_) => "inlineData",
        PartKind::FunctionCall(_) => "functionCall",
        PartKind::FunctionResponse(_) => "functionResponse",
        PartKind::FileData(_) => "fileData",
        PartKind::ExecutableCode(_) => "executableCode",
        PartKind::CodeExecutionResult(_) => "codeExecutionResult",
    }
}

pub mod gemini_api_types {
    use crate::error::EncodeError;
    use std::{collections::HashMap, convert::Infallible, str::FromStr};

    use serde::{Deserialize, Serialize};
    use serde_json::{Value, json};

    use crate::message::{DocumentSourceKind, ImageMediaType, MessageError, MimeType};
    use crate::{
        message,
        providers::gemini::gemini_api_types::{CodeExecutionResult, ExecutableCode},
    };

    #[derive(Debug, Deserialize, Serialize, Default)]
    #[serde(rename_all = "camelCase")]
    pub struct AdditionalParameters {
        /// Change your Gemini request configuration.
        pub generation_config: Option<GenerationConfig>,
        /// Any additional parameters that you want.
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pub additional_params: Option<serde_json::Value>,
    }

    impl AdditionalParameters {
        pub fn with_config(mut self, cfg: GenerationConfig) -> Self {
            self.generation_config = Some(cfg);
            self
        }

        pub fn with_params(mut self, params: serde_json::Value) -> Self {
            self.additional_params = Some(params);
            self
        }
    }

    /// The GenerateContent reply document: the whole `generateContent`
    /// body, and equally one `streamGenerateContent` chunk, which is the
    /// same document delivered in pieces.
    ///
    /// Safety ratings and content filtering are reported for the prompt in
    /// `prompt_feedback` and for each candidate in `finish_reason` and
    /// `safety_ratings`. The API returns either all requested candidates or
    /// none of them, and none at all only when something was wrong with the
    /// prompt.
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerateContentResponse {
        #[serde(default)]
        pub response_id: String,
        /// Candidate responses from the model.
        #[serde(default)]
        pub candidates: Vec<ContentCandidate>,
        /// The prompt's content-filter verdict. A set `blockReason` means the
        /// prompt was refused and no candidate follows.
        pub prompt_feedback: Option<PromptFeedback>,
        /// Output only. Metadata on the generation requests' token usage.
        pub usage_metadata: Option<UsageMetadata>,
        pub model_version: Option<String>,
        /// Gemini's error envelope, sent as a frame of its own when the
        /// service aborts a stream in-band (`{"error":{"code":500,"message":
        /// …,"status":"INTERNAL"}}`). Kept raw so every field (code, status,
        /// message, details) survives into the report.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub error: Option<Value>,
    }

    /// Iterate over visible text in wire order, excluding thought and non-text parts.
    /// Callers choose how to join part boundaries.
    pub(crate) fn visible_text_parts(content: &Content) -> impl Iterator<Item = &str> {
        content.parts.iter().filter_map(|part| match &part.part {
            PartKind::Text(text) if !part.thought.unwrap_or(false) => Some(text.as_str()),
            _ => None,
        })
    }

    /// A response candidate generated from the model.
    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ContentCandidate {
        /// Output only. Generated content returned from the model.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<Content>,
        /// Optional. Output only. The reason why the model stopped generating tokens.
        /// If empty, the model has not stopped generating tokens.
        pub finish_reason: Option<FinishReason>,
        /// List of ratings for the safety of a response candidate.
        /// There is at most one rating per category.
        pub safety_ratings: Option<Vec<SafetyRating>>,
        /// Output only. Citation information for model-generated candidate.
        /// This field may be populated with recitation information for any text included in the content.
        /// These are passages that are "recited" from copyrighted material in the foundational LLM's training data.
        pub citation_metadata: Option<CitationMetadata>,
        /// Output only. Token count for this candidate.
        pub token_count: Option<i32>,
        /// Output only.
        pub avg_logprobs: Option<f64>,
        /// Output only. Log-likelihood scores for the response tokens and top tokens
        pub logprobs_result: Option<LogprobsResult>,
        /// Output only. Index of the candidate in the list of response candidates.
        pub index: Option<i32>,
        /// Output only. Additional information about why the model stopped generating tokens.
        pub finish_message: Option<String>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct Content {
        /// Ordered Parts that constitute a single message. Parts may have different MIME types.
        #[serde(default)]
        pub parts: Vec<Part>,
        /// The producer of the content. Must be either 'user' or 'model'.
        /// Useful to set for multi-turn conversations, otherwise can be left blank or unset.
        pub role: Option<Role>,
    }

    impl TryFrom<message::Message> for Content {
        type Error = message::MessageError;

        fn try_from(msg: message::Message) -> Result<Self, Self::Error> {
            Ok(match msg {
                message::Message::System { content } => Content {
                    parts: vec![content.into()],
                    role: Some(Role::User),
                },
                message::Message::User { content } => Content {
                    parts: content
                        .into_iter()
                        .map(std::convert::TryInto::try_into)
                        .collect::<Result<Vec<_>, _>>()?,
                    role: Some(Role::User),
                },
                message::Message::Assistant { content, .. } => Content {
                    role: Some(Role::Model),
                    parts: content
                        .into_iter()
                        .map(std::convert::TryInto::try_into)
                        .collect::<Result<Vec<_>, _>>()?,
                },
            })
        }
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "lowercase")]
    pub enum Role {
        User,
        Model,
    }

    #[derive(Debug, Default, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub struct Part {
        /// Whether this part contains reasoning rather than visible output.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thought: Option<bool>,
        /// Opaque base64 signature required to replay the thought.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thought_signature: Option<String>,
        #[serde(flatten)]
        pub part: PartKind,
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pub additional_params: Option<Value>,
    }

    /// One content payload in a multipart [`Content`] message.
    /// Inline media requires an IANA MIME type.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub enum PartKind {
        Text(String),
        InlineData(Blob),
        FunctionCall(FunctionCall),
        FunctionResponse(FunctionResponse),
        FileData(FileData),
        ExecutableCode(ExecutableCode),
        CodeExecutionResult(CodeExecutionResult),
    }

    impl Default for PartKind {
        fn default() -> Self {
            Self::Text(String::new())
        }
    }

    impl From<String> for Part {
        fn from(text: String) -> Self {
            Self {
                thought: Some(false),
                thought_signature: None,
                part: PartKind::Text(text),
                additional_params: None,
            }
        }
    }

    impl From<&str> for Part {
        fn from(text: &str) -> Self {
            Self::from(text.to_string())
        }
    }

    impl FromStr for Part {
        type Err = Infallible;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(s.into())
        }
    }

    /// Convert a URL or base64 media source into a Gemini part.
    /// Accept untagged strings as base64 only when `string_is_data` is true.
    /// Reject other sources with a conversion error naming `kind`.
    fn media_source_to_part_kind(
        kind: &str,
        mime_type: String,
        source: DocumentSourceKind,
        string_is_data: bool,
    ) -> Result<PartKind, message::MessageError> {
        match source {
            DocumentSourceKind::Url(file_uri) => Ok(PartKind::FileData(FileData {
                mime_type: Some(mime_type),
                file_uri,
            })),
            DocumentSourceKind::Base64(data) => Ok(PartKind::InlineData(Blob { mime_type, data })),
            DocumentSourceKind::String(data) if string_is_data => {
                Ok(PartKind::InlineData(Blob { mime_type, data }))
            }
            DocumentSourceKind::String(_) => Err(message::MessageError::ConversionError(format!(
                "Strings cannot be used as Gemini {kind} inputs"
            ))),
            DocumentSourceKind::Raw(_) => Err(message::MessageError::ConversionError(
                "Raw files not supported, encode as base64 first".to_string(),
            )),
            DocumentSourceKind::FileId(_) => Err(message::MessageError::ConversionError(format!(
                "Provider file IDs are not supported for Gemini {kind} inputs"
            ))),
            DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(format!(
                "Gemini {kind} input has no body"
            ))),
        }
    }

    impl TryFrom<(ImageMediaType, DocumentSourceKind)> for PartKind {
        type Error = message::MessageError;
        fn try_from(
            (mime_type, doc_src): (ImageMediaType, DocumentSourceKind),
        ) -> Result<Self, Self::Error> {
            media_source_to_part_kind("image", mime_type.to_mime_type().to_string(), doc_src, true)
        }
    }

    /// Convert a message image into a Gemini part.
    ///
    /// Gemini takes images identically in either role, so the user and
    /// assistant conversions share this.
    fn image_to_part(image: message::Image) -> Result<Part, message::MessageError> {
        let message::Image {
            data, media_type, ..
        } = image;

        let Some(media_type) = media_type else {
            return Err(message::MessageError::ConversionError(
                "Media type for image is required for Gemini".to_string(),
            ));
        };

        match media_type {
            message::ImageMediaType::JPEG
            | message::ImageMediaType::PNG
            | message::ImageMediaType::WEBP
            | message::ImageMediaType::HEIC
            | message::ImageMediaType::HEIF => Ok(Part {
                thought: Some(false),
                thought_signature: None,
                part: PartKind::try_from((media_type, data))?,
                additional_params: None,
            }),
            _ => Err(message::MessageError::ConversionError(format!(
                "Unsupported image media type {media_type:?}"
            ))),
        }
    }

    fn gemini_tool_result_image_mime_type(
        media_type: Option<&ImageMediaType>,
    ) -> Result<&'static str, MessageError> {
        let media_type = media_type.ok_or_else(|| {
            MessageError::ConversionError(
                "Image media type is required for Gemini tool results".to_string(),
            )
        })?;

        match media_type {
            ImageMediaType::JPEG | ImageMediaType::PNG | ImageMediaType::WEBP => {
                Ok(media_type.to_mime_type())
            }
            _ => Err(MessageError::ConversionError(format!(
                "Unsupported image media type {media_type:?} for Gemini tool results; supported types are JPEG, PNG, and WEBP"
            ))),
        }
    }

    impl TryFrom<message::UserContent> for Part {
        type Error = message::MessageError;

        fn try_from(content: message::UserContent) -> Result<Self, Self::Error> {
            match content {
                message::UserContent::Text(message::Text { text, .. }) => Ok(Part {
                    thought: Some(false),
                    thought_signature: None,
                    part: PartKind::Text(text),
                    additional_params: None,
                }),
                message::UserContent::ToolResult(message::ToolResult {
                    call: _,
                    provider,
                    name,
                    content,
                }) => {
                    let function_name = name;
                    let mut response_values = Vec::new();
                    let mut parts: Vec<FunctionResponsePart> = Vec::new();

                    for item in content.iter() {
                        match item {
                            message::ToolResultContent::Text(text) => {
                                response_values.push(json!(&text.text));
                            }
                            message::ToolResultContent::Json { value } => {
                                response_values.push(value.clone());
                            }
                            message::ToolResultContent::Image(image) => {
                                let part = match &image.data {
                                    DocumentSourceKind::Base64(b64) => {
                                        let mime_type = gemini_tool_result_image_mime_type(
                                            image.media_type.as_ref(),
                                        )?;

                                        // Gemini rejects synthetic `$ref` links for inline
                                        // function-response media, so preserve ordered parts directly.
                                        FunctionResponsePart {
                                            inline_data: Some(FunctionResponseInlineData {
                                                mime_type: mime_type.to_string(),
                                                data: b64.clone(),
                                                display_name: None,
                                            }),
                                            file_data: None,
                                        }
                                    }
                                    DocumentSourceKind::Url(_) => {
                                        return Err(message::MessageError::ConversionError(
                                            "Gemini tool result images must use base64 inline data; URL-backed images are not supported"
                                                .to_string(),
                                        ));
                                    }
                                    _ => {
                                        return Err(message::MessageError::ConversionError(
                                            "Unsupported image source kind for tool results"
                                                .to_string(),
                                        ));
                                    }
                                };
                                parts.push(part);
                            }
                        }
                    }

                    let response_json = if response_values.is_empty() {
                        None
                    } else {
                        let result = if response_values.len() == 1 {
                            response_values.remove(0)
                        } else {
                            serde_json::Value::Array(response_values)
                        };
                        Some(json!({ "result": result }))
                    };

                    Ok(Part {
                        thought: Some(false),
                        thought_signature: None,
                        part: PartKind::FunctionResponse(FunctionResponse {
                            name: function_name,
                            id: provider.map(|provider| provider.call_id),
                            response: response_json,
                            parts: if parts.is_empty() { None } else { Some(parts) },
                        }),
                        additional_params: None,
                    })
                }
                message::UserContent::Image(image) => image_to_part(image),
                message::UserContent::Document(message::Document {
                    data, media_type, ..
                }) => {
                    let Some(media_type) = media_type else {
                        return Err(MessageError::ConversionError(
                            "A mime type is required for document inputs to Gemini".to_string(),
                        ));
                    };

                    // For text-like documents (RAG context), convert inline content to plain text.
                    // URL-backed files should stay as file_data references so Gemini can fetch them.
                    if matches!(
                        media_type,
                        message::DocumentMediaType::TXT
                            | message::DocumentMediaType::RTF
                            | message::DocumentMediaType::HTML
                            | message::DocumentMediaType::CSS
                            | message::DocumentMediaType::MARKDOWN
                            | message::DocumentMediaType::CSV
                            | message::DocumentMediaType::XML
                            | message::DocumentMediaType::Javascript
                            | message::DocumentMediaType::Python
                    ) {
                        use base64::Engine;
                        let part = match data {
                            DocumentSourceKind::String(text) => PartKind::Text(text),
                            DocumentSourceKind::Base64(data) => {
                                let text = String::from_utf8(
                                    base64::engine::general_purpose::STANDARD
                                        .decode(&data)
                                        .map_err(|e| {
                                            MessageError::ConversionError(format!(
                                                "Failed to decode base64: {e}"
                                            ))
                                        })?,
                                )
                                .map_err(|e| {
                                    MessageError::ConversionError(format!(
                                        "Invalid UTF-8 in document: {e}"
                                    ))
                                })?;
                                PartKind::Text(text)
                            }
                            DocumentSourceKind::Url(file_uri) => PartKind::FileData(FileData {
                                mime_type: Some(media_type.to_mime_type().to_string()),
                                file_uri,
                            }),
                            DocumentSourceKind::Raw(_) => {
                                return Err(MessageError::ConversionError(
                                    "Raw files not supported, encode as base64 first".to_string(),
                                ));
                            }
                            DocumentSourceKind::FileId(_) => {
                                return Err(MessageError::ConversionError(
                                    "Provider file IDs are not supported for Gemini documents"
                                        .to_string(),
                                ));
                            }
                            DocumentSourceKind::Unknown => {
                                return Err(MessageError::ConversionError(
                                    "Document has no body".to_string(),
                                ));
                            }
                        };

                        Ok(Part {
                            thought: Some(false),
                            part,
                            ..Default::default()
                        })
                    } else if !media_type.is_code() {
                        let part = media_source_to_part_kind(
                            "document",
                            media_type.to_mime_type().to_string(),
                            data,
                            true,
                        )?;

                        Ok(Part {
                            thought: Some(false),
                            part,
                            ..Default::default()
                        })
                    } else {
                        Err(message::MessageError::ConversionError(format!(
                            "Unsupported document media type {media_type:?}"
                        )))
                    }
                }

                message::UserContent::Audio(message::Audio {
                    data, media_type, ..
                }) => {
                    let Some(media_type) = media_type else {
                        return Err(MessageError::ConversionError(
                            "A mime type is required for audio inputs to Gemini".to_string(),
                        ));
                    };

                    let part = media_source_to_part_kind(
                        "audio",
                        media_type.to_mime_type().to_string(),
                        data,
                        false,
                    )?;

                    Ok(Part {
                        thought: Some(false),
                        part,
                        ..Default::default()
                    })
                }
                message::UserContent::Video(message::Video {
                    data,
                    media_type,
                    additional_params,
                    ..
                }) => {
                    let mime_type = media_type.map(|media_ty| media_ty.to_mime_type().to_string());

                    let part = match data {
                        // YouTube links are the one Gemini video source that
                        // needs no MIME type: the service resolves the media
                        // itself. Every other source must declare one.
                        DocumentSourceKind::Url(file_uri)
                            if file_uri.starts_with("https://www.youtube.com") =>
                        {
                            PartKind::FileData(FileData {
                                mime_type,
                                file_uri,
                            })
                        }
                        data => {
                            let mime_type = mime_type.ok_or_else(|| {
                                MessageError::ConversionError(
                                    "A mime type is required for non-Youtube video inputs to Gemini"
                                        .to_string(),
                                )
                            })?;

                            media_source_to_part_kind("video", mime_type, data, false)?
                        }
                    };

                    Ok(Part {
                        thought: Some(false),
                        thought_signature: None,
                        part,
                        additional_params: additional_params.map(Into::into),
                    })
                }
            }
        }
    }

    impl TryFrom<message::AssistantContent> for Part {
        type Error = message::MessageError;

        fn try_from(content: message::AssistantContent) -> Result<Self, Self::Error> {
            match content {
                message::AssistantContent::Text(text) => {
                    // A signed answer part returns with its signature.
                    let thought_signature =
                        super::super::text_thought_signature(&text).map(str::to_owned);
                    Ok(Part {
                        thought_signature,
                        ..text.text.into()
                    })
                }
                message::AssistantContent::Image(image) => image_to_part(image),
                message::AssistantContent::ToolCall(tool_call) => Ok(tool_call.into()),
                message::AssistantContent::Reasoning(reasoning) => Ok(Part {
                    thought: Some(true),
                    thought_signature: reasoning.first_signature().map(str::to_owned),
                    part: PartKind::Text(reasoning.display_text()),
                    additional_params: None,
                }),
            }
        }
    }

    impl From<message::ToolCall> for Part {
        fn from(tool_call: message::ToolCall) -> Self {
            Self {
                thought: Some(false),
                thought_signature: tool_call.signature,
                part: PartKind::FunctionCall(FunctionCall {
                    name: tool_call.function.name,
                    args: tool_call.function.arguments,
                    // Only a provider-issued id may travel back on the wire;
                    // minted correlation handles stay internal.
                    id: tool_call.provider.map(|provider| provider.call_id),
                }),
                additional_params: None,
            }
        }
    }

    /// Raw media bytes.
    /// Text should not be sent as raw bytes, use the 'text' field.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub struct Blob {
        /// The IANA standard MIME type of the source data. Examples: - image/png - image/jpeg
        /// If an unsupported MIME type is provided, an error will be returned.
        pub mime_type: String,
        /// Raw bytes for media formats. A base64-encoded string.
        pub data: String,
    }

    /// A model-requested function call with its name, arguments, and optional identifier.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct FunctionCall {
        /// Required. The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores
        /// and dashes, with a maximum length of 63.
        pub name: String,
        /// Optional. The function parameters and values in JSON object format.
        pub args: serde_json::Value,
        /// Provider-supplied identifier used to correlate the function response.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
    }

    impl From<message::ToolCall> for FunctionCall {
        fn from(tool_call: message::ToolCall) -> Self {
            Self {
                name: tool_call.function.name,
                args: tool_call.function.arguments,
                id: tool_call.provider.map(|provider| provider.call_id),
            }
        }
    }

    /// Result of a model-requested function call, returned as context to the model.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct FunctionResponse {
        /// The name of the function to call. Must be a-z, A-Z, 0-9, or contain underscores and dashes,
        /// with a maximum length of 63.
        pub name: String,
        /// Provider-supplied identifier from the corresponding function call.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub id: Option<String>,
        /// The function response in JSON object format.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response: Option<serde_json::Value>,
        /// Multimodal parts for the function response (e.g., images).
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parts: Option<Vec<FunctionResponsePart>>,
    }

    /// A part of a multimodal function response.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub struct FunctionResponsePart {
        /// Inline data containing base64-encoded media content.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub inline_data: Option<FunctionResponseInlineData>,
        /// File data containing a URI reference.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub file_data: Option<FileData>,
    }

    /// Inline data for function response parts.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub struct FunctionResponseInlineData {
        /// The IANA standard MIME type of the source data.
        pub mime_type: String,
        /// Raw bytes for media formats. A base64-encoded string.
        pub data: String,
        /// Optional display name for the content.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub display_name: Option<String>,
    }

    /// URI based data.
    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "camelCase")]
    pub struct FileData {
        /// Optional. The IANA standard MIME type of the source data.
        pub mime_type: Option<String>,
        /// Required. URI.
        pub file_uri: String,
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    pub struct SafetyRating {
        pub category: HarmCategory,
        pub probability: HarmProbability,
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum HarmProbability {
        HarmProbabilityUnspecified,
        Negligible,
        Low,
        Medium,
        High,
        /// A probability this crate does not know yet, carried verbatim so
        /// a rating (and the chunk that carries it) stays deserializable.
        #[serde(untagged)]
        Unknown(String),
    }

    #[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum HarmCategory {
        HarmCategoryUnspecified,
        HarmCategoryDerogatory,
        HarmCategoryToxicity,
        HarmCategoryViolence,
        HarmCategorySexually,
        HarmCategoryMedical,
        HarmCategoryDangerous,
        HarmCategoryHarassment,
        HarmCategoryHateSpeech,
        HarmCategorySexuallyExplicit,
        HarmCategoryDangerousContent,
        HarmCategoryCivicIntegrity,
        /// An unrecognized category, preserved verbatim without rejecting the rating.
        #[serde(untagged)]
        Unknown(String),
    }

    #[derive(Debug, Deserialize, Clone, Default, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct UsageMetadata {
        #[serde(default)]
        pub prompt_token_count: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cached_content_token_count: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub candidates_token_count: Option<i32>,
        #[serde(default)]
        pub total_token_count: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thoughts_token_count: Option<i32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub cache_tokens_details: Option<Vec<ModalityTokenCount>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub candidates_tokens_details: Option<Vec<ModalityTokenCount>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub tool_use_prompt_token_count: Option<i32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub tool_use_prompt_tokens_details: Option<Vec<ModalityTokenCount>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub traffic_type: Option<TrafficType>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ModalityTokenCount {
        pub modality: Modality,
        #[serde(default)]
        pub token_count: i32,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum Modality {
        ModalityUnspecified,
        Text,
        Image,
        Video,
        Audio,
        Document,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum TrafficType {
        TrafficTypeUnspecified,
        OnDemand,
        ProvisionedThroughput,
    }

    impl From<&UsageMetadata> for crate::completion::Usage {
        fn from(value: &UsageMetadata) -> crate::completion::Usage {
            let count = |count: i32| count as u64;
            crate::completion::Usage {
                input_tokens: Some(count(value.prompt_token_count)),
                output_tokens: value.candidates_token_count.map(count),
                cached_input_tokens: value.cached_content_token_count.map(count),
                reasoning_tokens: value.thoughts_token_count.map(count),
                tool_use_prompt_tokens: value.tool_use_prompt_token_count.map(count),
                total_tokens: Some(count(value.total_token_count)),
                cache_creation_input_tokens: None,
            }
        }
    }

    /// A set of the feedback metadata the prompt specified in [GenerateContentRequest.contents](GenerateContentRequest).
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PromptFeedback {
        /// Optional. If set, the prompt was blocked and no candidates are returned. Rephrase the prompt.
        pub block_reason: Option<BlockReason>,
        /// Ratings for safety of the prompt. There is at most one rating per category.
        pub safety_ratings: Option<Vec<SafetyRating>>,
    }

    /// Reason why a prompt was blocked by the model
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum BlockReason {
        /// Default value. This value is unused.
        BlockReasonUnspecified,
        /// Prompt was blocked due to safety reasons. Inspect safetyRatings to understand which safety category blocked it.
        Safety,
        /// Prompt was blocked due to unknown reasons.
        Other,
        /// Prompt was blocked due to the terms which are included from the terminology blocklist.
        Blocklist,
        /// Prompt was blocked due to prohibited content.
        ProhibitedContent,
        /// A block reason this crate does not know yet. Google adds wire
        /// values without notice; carrying the spelling verbatim keeps the
        /// whole payload deserializable instead of failing on the new value.
        #[serde(untagged)]
        Unknown(String),
    }

    impl BlockReason {
        /// The exact spelling Gemini uses for this reason on the wire (see
        /// [`FinishReason::as_wire_str`] for why it is spelled out).
        pub fn as_wire_str(&self) -> &str {
            match self {
                Self::BlockReasonUnspecified => "BLOCK_REASON_UNSPECIFIED",
                Self::Safety => "SAFETY",
                Self::Other => "OTHER",
                Self::Blocklist => "BLOCKLIST",
                Self::ProhibitedContent => "PROHIBITED_CONTENT",
                Self::Unknown(raw) => raw.as_str(),
            }
        }
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum FinishReason {
        /// Default value. This value is unused.
        FinishReasonUnspecified,
        /// Natural stop point of the model or provided stop sequence.
        Stop,
        /// The maximum number of tokens as specified in the request was reached.
        MaxTokens,
        /// The response candidate content was flagged for safety reasons.
        Safety,
        /// The response candidate content was flagged for recitation reasons.
        Recitation,
        /// The response candidate content was flagged for using an unsupported language.
        Language,
        /// Unknown reason.
        Other,
        /// Token generation stopped because the content contains forbidden terms.
        Blocklist,
        /// Token generation stopped for potentially containing prohibited content.
        ProhibitedContent,
        /// Token generation stopped because the content potentially contains Sensitive Personally Identifiable Information (SPII).
        Spii,
        /// The function call generated by the model is invalid.
        MalformedFunctionCall,
        /// The model emitted a tool call that was not expected by the request.
        UnexpectedToolCall,
        /// The response omitted a thought signature required for a tool-calling turn.
        MissingThoughtSignature,
        /// The model emitted more tool calls than the provider allows for the request.
        TooManyToolCalls,
        /// The provider could not parse the generated response into a valid protocol shape.
        MalformedResponse,
        /// An unrecognized finish reason, preserved verbatim without rejecting the response.
        #[serde(untagged)]
        Unknown(String),
    }

    impl FinishReason {
        /// The exact spelling Gemini uses for this reason on the wire.
        ///
        /// Spelled out rather than derived from `Debug` (which would yield
        /// `MaxTokens`, not `MAX_TOKENS`) so the string that reaches
        /// [`crate::completion::FinishReason::Other`] is the provider's own.
        pub fn as_wire_str(&self) -> &str {
            match self {
                Self::FinishReasonUnspecified => "FINISH_REASON_UNSPECIFIED",
                Self::Stop => "STOP",
                Self::MaxTokens => "MAX_TOKENS",
                Self::Safety => "SAFETY",
                Self::Recitation => "RECITATION",
                Self::Language => "LANGUAGE",
                Self::Other => "OTHER",
                Self::Blocklist => "BLOCKLIST",
                Self::ProhibitedContent => "PROHIBITED_CONTENT",
                Self::Spii => "SPII",
                Self::MalformedFunctionCall => "MALFORMED_FUNCTION_CALL",
                Self::UnexpectedToolCall => "UNEXPECTED_TOOL_CALL",
                Self::MissingThoughtSignature => "MISSING_THOUGHT_SIGNATURE",
                Self::TooManyToolCalls => "TOO_MANY_TOOL_CALLS",
                Self::MalformedResponse => "MALFORMED_RESPONSE",
                Self::Unknown(reason) => reason,
            }
        }
    }

    /// Normalize a Google `finishReason` in its SCREAMING_SNAKE_CASE wire spelling.
    /// Return `None` for `FINISH_REASON_UNSPECIFIED`. Preserve reasons without a
    /// normalized counterpart as `Other`, including tool-protocol failures.
    pub fn map_google_finish_reason(wire_name: &str) -> Option<crate::completion::FinishReason> {
        Some(match wire_name {
            "FINISH_REASON_UNSPECIFIED" => return None,
            "STOP" => crate::completion::FinishReason::Stop,
            "MAX_TOKENS" => crate::completion::FinishReason::Length,
            "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
                crate::completion::FinishReason::ContentFilter
            }
            other => crate::completion::FinishReason::Other(other.to_owned()),
        })
    }

    /// Map a Gemini REST `finishReason` onto rig's normalized vocabulary.
    ///
    /// Shared by the unary and streaming paths so both agree.
    pub(crate) fn map_finish_reason(
        reason: &FinishReason,
    ) -> Option<crate::completion::FinishReason> {
        map_google_finish_reason(reason.as_wire_str())
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CitationMetadata {
        #[serde(default)]
        pub citation_sources: Vec<CitationSource>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct CitationSource {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub uri: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub start_index: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub end_index: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub license: Option<String>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct LogprobsResult {
        #[serde(default)]
        pub top_candidates: Vec<TopCandidate>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub log_probability_sum: Option<f64>,
        #[serde(default)]
        pub chosen_candidates: Vec<LogProbCandidate>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct TopCandidate {
        #[serde(default)]
        pub candidates: Vec<LogProbCandidate>,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct LogProbCandidate {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub token: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub token_id: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub log_probability: Option<f64>,
    }

    /// Model generation options from the [Gemini API](https://ai.google.dev/api/generate-content#generationconfig).
    /// Supported fields depend on the model. All fields default to `None` and
    /// are omitted when unset, preserving provider defaults.
    #[derive(Debug, Default, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerationConfig {
        /// Up to five stop sequences. Generation ends at the first match,
        /// excluding that sequence from the response.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop_sequences: Option<Vec<String>>,
        /// Output MIME type: `text/plain` by default, `application/json` for JSON,
        /// or `text/x.enum` for a string enum.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_mime_type: Option<String>,
        /// OpenAPI-subset output schema for objects, primitives, or arrays.
        /// Requires `response_mime_type` to be `application/json`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_schema: Option<Schema>,
        /// Optional. The output schema of the generated response.
        /// This is an alternative to responseSchema that accepts a standard JSON Schema.
        /// If this is set, responseSchema must be omitted.
        /// Compatible MIME type: application/json.
        /// Supported properties: $id, $defs, $ref, type, properties, etc.
        #[serde(
            skip_serializing_if = "Option::is_none",
            rename = "_responseJsonSchema"
        )]
        pub _response_json_schema: Option<Value>,
        /// Internal or alternative representation for `response_json_schema`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_json_schema: Option<Value>,
        /// Number of generated responses to return. Currently, this value can only be set to 1. If
        /// unset, this will default to 1.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub candidate_count: Option<i32>,
        /// Maximum output tokens per candidate. The provider default depends on the model.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_output_tokens: Option<u64>,
        /// Sampling temperature in `[0.0, 2.0]`. The provider default depends on the model.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f64>,
        /// Maximum cumulative token probability for nucleus sampling.
        /// The provider default depends on the model.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f64>,
        /// Maximum number of likely tokens considered for sampling.
        /// Set only for models whose metadata reports top-k support.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_k: Option<i32>,
        /// Penalty for tokens already present in the response, independent of frequency.
        /// Positive values discourage reuse; negative values encourage it.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub presence_penalty: Option<f64>,
        /// Penalty scaled by each token's frequency in the response.
        /// Positive values discourage repetition; negative values encourage it
        /// and may produce repetition until the output limit.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub frequency_penalty: Option<f64>,
        /// If true, export the logprobs results in response.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_logprobs: Option<bool>,
        /// Only valid if responseLogprobs=True. This sets the number of top logprobs to return at each decoding step in
        /// [Candidate.logprobs_result].
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<i32>,
        /// Configuration for thinking/reasoning.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_config: Option<ThinkingConfig>,
        /// Response modalities requested from models that support multimodal output.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_modalities: Option<Vec<ResponseModality>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub image_config: Option<ImageConfig>,
    }

    /// Response modalities supported by Gemini multimodal output models.
    #[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum ResponseModality {
        Text,
        Image,
        Audio,
    }

    /// Thinking depth level for Gemini 3 models.
    #[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    pub enum ThinkingLevel {
        Minimal,
        Low,
        Medium,
        High,
    }

    /// Configuration for the model's thinking/reasoning process.
    /// Note: `thinking_budget` (Gemini 2.5) and `thinking_level` (Gemini 3) are mutually exclusive
    /// and cannot be set in the same request.
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ThinkingConfig {
        /// Token budget for thinking. Used by Gemini 2.5 models. Range: 0 to 32768.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_budget: Option<u32>,
        /// Thinking depth level. Used by Gemini 3 models.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking_level: Option<ThinkingLevel>,
        /// When true, includes summarized versions of the model's reasoning in the response.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub include_thoughts: Option<bool>,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ImageConfig {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub aspect_ratio: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub image_size: Option<String>,
    }

    /// The Schema object allows the definition of input and output data types. These types can be objects, but also
    /// primitives and arrays. Represents a select subset of an OpenAPI 3.0 schema object.
    /// From [Gemini API Reference](https://ai.google.dev/api/caching#Schema)
    #[derive(Debug, Deserialize, Serialize, Clone)]
    pub struct Schema {
        pub r#type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub format: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub nullable: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub r#enum: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_items: Option<i32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub min_items: Option<i32>,
        /// Argument properties, serialized in sorted key order to stabilize cache prefixes.
        #[serde(
            skip_serializing_if = "Option::is_none",
            serialize_with = "crate::json_utils::serialize_optional_map_sorted"
        )]
        pub properties: Option<HashMap<String, Schema>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub required: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub items: Option<Box<Schema>>,
    }

    /// Converts Rig tool parameters into Gemini's schema representation.
    ///
    /// Gemini does not need a `parameters` object for no-argument tools, and it
    /// does not support JSON Schema references, so this helper keeps those
    /// conventions centralized for all Gemini transports.
    pub fn tool_parameters_to_schema(parameters: Value) -> Result<Option<Schema>, EncodeError> {
        if parameters.is_null() || parameters == json!({"type": "object", "properties": {}}) {
            Ok(None)
        } else {
            parameters.try_into().map(Some)
        }
    }

    /// Inline references from `$defs` or `definitions` and remove those sections.
    /// Return unchanged input if neither section exists. Return an error for a
    /// non-object definitions section, unsupported reference paths, or missing definitions.
    /// Callers must supply acyclic references.
    pub fn flatten_schema(mut schema: Value) -> Result<Value, EncodeError> {
        let defs = schema
            .as_object()
            .and_then(|obj| obj.get("$defs").or_else(|| obj.get("definitions")))
            .cloned();

        let Some(defs_value) = defs else {
            return Ok(schema);
        };

        let Some(defs_obj) = defs_value.as_object() else {
            return Err(EncodeError::request("$defs must be an object"));
        };

        resolve_refs(&mut schema, defs_obj)?;

        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$defs");
            obj.remove("definitions");
        }

        Ok(schema)
    }

    /// Recursively resolves all `$ref` references in a JSON value by
    /// replacing them with their definitions.
    fn resolve_refs(
        value: &mut Value,
        defs: &serde_json::Map<String, Value>,
    ) -> Result<(), EncodeError> {
        match value {
            Value::Object(obj) => {
                if let Some(ref_value) = obj.get("$ref")
                    && let Some(ref_str) = ref_value.as_str()
                {
                    let def_name = parse_ref_path(ref_str)?;

                    let def = defs.get(&def_name).ok_or_else(|| {
                        EncodeError::request(format!("Reference not found: {ref_str}"))
                    })?;

                    let mut resolved = def.clone();
                    resolve_refs(&mut resolved, defs)?;
                    *value = resolved;
                    return Ok(());
                }

                for (_, v) in obj.iter_mut() {
                    resolve_refs(v, defs)?;
                }
            }
            Value::Array(arr) => {
                for item in arr.iter_mut() {
                    resolve_refs(item, defs)?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Extract the suffix of `#/$defs/` or `#/definitions/`.
    /// Return a request error for any other reference prefix.
    fn parse_ref_path(ref_str: &str) -> Result<String, EncodeError> {
        if let Some(fragment) = ref_str.strip_prefix('#') {
            if let Some(name) = fragment.strip_prefix("/$defs/") {
                Ok(name.to_string())
            } else if let Some(name) = fragment.strip_prefix("/definitions/") {
                Ok(name.to_string())
            } else {
                Err(EncodeError::request(format!(
                    "Unsupported reference format: {ref_str}"
                )))
            }
        } else {
            Err(EncodeError::request(format!(
                "Only fragment references (#/...) are supported: {ref_str}"
            )))
        }
    }

    /// Helper function to extract the type string from a JSON value.
    /// Handles both direct string types and array types.
    fn extract_type(type_value: &Value) -> Option<String> {
        if let Some(t) = type_value.as_str() {
            return Some(t.to_string());
        }

        type_value.as_array().and_then(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .find(|t| *t != "null")
                .or_else(|| arr.iter().find_map(|v| v.as_str()))
                .map(str::to_owned)
        })
    }

    fn schema_is_null(obj: &serde_json::Map<String, Value>) -> bool {
        obj.get("type")
            .and_then(extract_type)
            .as_deref()
            .is_some_and(|t| t == "null")
    }

    fn schema_is_nullable(obj: &serde_json::Map<String, Value>) -> bool {
        obj.get("nullable")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
            || obj
                .get("type")
                .and_then(|v| v.as_array())
                .is_some_and(|arr| arr.iter().any(|v| v.as_str() == Some("null")))
            || ["anyOf", "oneOf", "allOf"].iter().any(|key| {
                obj.get(*key).and_then(|v| v.as_array()).is_some_and(|arr| {
                    arr.iter()
                        .filter_map(|schema| schema.as_object())
                        .any(schema_is_null)
                })
            })
    }

    /// Helper function to extract type from anyOf, oneOf, or allOf schemas.
    /// Returns the type of the first non-null schema found.
    fn extract_type_from_composition(composition: &Value) -> Option<String> {
        composition.as_array().and_then(|arr| {
            arr.iter().find_map(|schema| {
                let obj = schema.as_object()?;
                if schema_is_null(obj) {
                    return None;
                }

                obj.get("type").and_then(extract_type).or_else(|| {
                    if obj.contains_key("properties") {
                        Some("object".to_string())
                    } else if obj.contains_key("enum") {
                        // Enum schemas without explicit type are string-backed
                        Some("string".to_string())
                    } else {
                        None
                    }
                })
            })
        })
    }

    /// Helper function to extract the first non-null schema from anyOf, oneOf, or allOf.
    /// Returns the schema object that should be used for properties, required, etc.
    fn extract_schema_from_composition(
        composition: &Value,
    ) -> Option<serde_json::Map<String, Value>> {
        composition.as_array().and_then(|arr| {
            arr.iter().find_map(|schema| {
                let obj = schema.as_object()?;
                if schema_is_null(obj) {
                    None
                } else {
                    Some(obj.clone())
                }
            })
        })
    }

    fn extract_schema_from_composition_obj(
        obj: &serde_json::Map<String, Value>,
    ) -> Option<serde_json::Map<String, Value>> {
        obj.get("anyOf")
            .and_then(extract_schema_from_composition)
            .or_else(|| obj.get("oneOf").and_then(extract_schema_from_composition))
            .or_else(|| obj.get("allOf").and_then(extract_schema_from_composition))
    }

    /// Helper function to infer the type of a schema object.
    /// Checks for explicit type, then anyOf/oneOf/allOf, then infers from properties.
    fn infer_type(obj: &serde_json::Map<String, Value>) -> String {
        if let Some(type_val) = obj.get("type")
            && let Some(type_str) = extract_type(type_val)
        {
            return type_str;
        }

        if let Some(any_of) = obj.get("anyOf")
            && let Some(type_str) = extract_type_from_composition(any_of)
        {
            return type_str;
        }

        if let Some(one_of) = obj.get("oneOf")
            && let Some(type_str) = extract_type_from_composition(one_of)
        {
            return type_str;
        }

        if let Some(all_of) = obj.get("allOf")
            && let Some(type_str) = extract_type_from_composition(all_of)
        {
            return type_str;
        }

        if obj.contains_key("properties") {
            "object".to_string()
        } else if obj.contains_key("enum") {
            "string".to_string()
        } else {
            String::new()
        }
    }

    impl TryFrom<Value> for Schema {
        type Error = EncodeError;

        fn try_from(value: Value) -> Result<Self, Self::Error> {
            let flattened_val = flatten_schema(value)?;
            if let Some(obj) = flattened_val.as_object() {
                let composition_source = extract_schema_from_composition_obj(obj);
                let props_source = if obj.get("properties").is_none() {
                    composition_source.clone().unwrap_or_else(|| obj.clone())
                } else {
                    obj.clone()
                };

                let schema_type = infer_type(obj);
                let items = obj
                    .get("items")
                    .or_else(|| props_source.get("items"))
                    .and_then(|v| v.clone().try_into().ok())
                    .map(Box::new);

                // Gemini requires `items` on array-typed schemas; default to
                // string items when the source schema omits it.
                let items = if schema_type == "array" && items.is_none() {
                    Some(Box::new(Schema {
                        r#type: "string".to_string(),
                        format: None,
                        description: None,
                        nullable: None,
                        r#enum: None,
                        max_items: None,
                        min_items: None,
                        properties: None,
                        required: None,
                        items: None,
                    }))
                } else {
                    items
                };

                Ok(Schema {
                    r#type: schema_type,
                    format: obj
                        .get("format")
                        .or_else(|| props_source.get("format"))
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    description: obj
                        .get("description")
                        .or_else(|| props_source.get("description"))
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    nullable: if schema_is_nullable(obj)
                        || composition_source.as_ref().is_some_and(schema_is_nullable)
                    {
                        Some(true)
                    } else {
                        None
                    },
                    r#enum: obj
                        .get("enum")
                        .or_else(|| props_source.get("enum"))
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        }),
                    max_items: obj
                        .get("maxItems")
                        .and_then(serde_json::Value::as_i64)
                        .map(|v| v as i32),
                    min_items: obj
                        .get("minItems")
                        .and_then(serde_json::Value::as_i64)
                        .map(|v| v as i32),
                    properties: props_source
                        .get("properties")
                        .and_then(|v| v.as_object())
                        .map(|map| {
                            map.iter()
                                .filter_map(|(k, v)| {
                                    v.clone().try_into().ok().map(|schema| (k.clone(), schema))
                                })
                                .collect()
                        }),
                    required: props_source
                        .get("required")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        }),
                    items,
                })
            } else {
                Err(EncodeError::request("Expected a JSON object for Schema"))
            }
        }
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct GenerateContentRequest {
        pub contents: Vec<Content>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<Vec<Value>>,
        pub tool_config: Option<ToolConfig>,
        /// Optional. Configuration options for model generation and outputs.
        pub generation_config: Option<GenerationConfig>,
        /// Safety thresholds applied to request content and response candidates.
        /// Supply at most one setting per category; omitted categories retain defaults.
        /// Supported categories are hate speech, sexually explicit content,
        /// dangerous content, and harassment.
        pub safety_settings: Option<Vec<SafetySetting>>,
        /// Optional. Developer set system instruction(s). Currently, text only.
        /// From [Gemini API Reference](https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest)
        pub system_instruction: Option<Content>,
        /// Cache handle whose content prefixes this request (`cachedContents/<id>`).
        /// Use [`Self::with_cached_content`] to validate conflicts with system
        /// instructions, tools, and tool configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cached_content: Option<String>,
        /// Additional parameters.
        #[serde(flatten, skip_serializing_if = "Option::is_none")]
        pub additional_params: Option<serde_json::Value>,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Tool {
        pub function_declarations: Vec<FunctionDeclaration>,
        pub code_execution: Option<CodeExecution>,
    }

    #[derive(Debug, Serialize, Clone)]
    #[serde(rename_all = "camelCase")]
    pub struct FunctionDeclaration {
        pub name: String,
        pub description: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parameters: Option<Schema>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct ToolConfig {
        pub function_calling_config: Option<FunctionCallingMode>,
    }

    #[derive(Debug, Serialize, Deserialize, Default)]
    #[serde(tag = "mode", rename_all = "UPPERCASE")]
    pub enum FunctionCallingMode {
        #[default]
        Auto,
        None,
        Any {
            #[serde(skip_serializing_if = "Option::is_none")]
            allowed_function_names: Option<Vec<String>>,
        },
    }

    impl TryFrom<message::ToolChoice> for FunctionCallingMode {
        type Error = EncodeError;
        fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
            let res = match value {
                message::ToolChoice::Auto => Self::Auto,
                message::ToolChoice::None => Self::None,
                message::ToolChoice::Required => Self::Any {
                    allowed_function_names: None,
                },
                message::ToolChoice::Specific { function_names } => Self::Any {
                    allowed_function_names: Some(function_names),
                },
            };

            Ok(res)
        }
    }

    #[derive(Debug, Serialize)]
    pub struct CodeExecution {}

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct SafetySetting {
        pub category: HarmCategory,
        pub threshold: HarmBlockThreshold,
    }

    #[derive(Debug, Serialize)]
    #[serde(rename_all = "SCREAMING_SNAKE_CASE")]
    pub enum HarmBlockThreshold {
        HarmBlockThresholdUnspecified,
        BlockLowAndAbove,
        BlockMediumAndAbove,
        BlockOnlyHigh,
        BlockNone,
        Off,
    }
}

impl gemini_api_types::GenerateContentRequest {
    /// Set an explicit cache handle, requiring the `cachedContents/` prefix.
    /// Return a request error for a different existing handle or for system
    /// instructions, tools, or tool configuration in typed or additional fields.
    pub fn with_cached_content(&mut self, name: &str) -> Result<(), EncodeError> {
        if !name.starts_with("cachedContents/") {
            return Err(EncodeError::request(format!(
                "gemini cached content handle should look like `cachedContents/<id>`, got \
                     `{name}`"
            )));
        }

        // Reject competing handles rather than silently choose a cache.
        if let Some(existing) = self.cached_content.as_deref()
            && existing != name
        {
            return Err(EncodeError::request(format!(
                "a Gemini request set cached content twice, to `{existing}` and `{name}` — \
                     set it one way or the other"
            )));
        }

        // Hand-built requests can retain conflicting fields in flattened parameters,
        // so validation must inspect both typed and untyped fields.
        let blob = self.additional_params.as_ref();
        let smuggled = |spellings: &'static [&'static str]| {
            blob.and_then(|payload| smuggled_field(payload, spellings))
        };

        let mut conflicts = Vec::new();
        if self.system_instruction.is_some() || smuggled(&SYSTEM_INSTRUCTION).is_some() {
            conflicts.push("a system instruction (preamble)");
        }
        let smuggled_tools = smuggled(&TOOLS);
        if self.tools.is_some() || smuggled_tools.is_some() {
            conflicts.push("tools");
        }
        if self.tool_config.is_some() || smuggled(&TOOL_CONFIG).is_some() {
            conflicts.push("a tool choice");
        }
        if !conflicts.is_empty() {
            // Cached function declarations need caller-side dispatch, whereas
            // provider-hosted tools remain executable without an agent tool registry.
            let declares_function = |tool: &Value| {
                ["functionDeclarations", "function_declarations"]
                    .iter()
                    .any(|spelling| {
                        tool.get(spelling)
                            .and_then(Value::as_array)
                            .is_some_and(|declarations| !declarations.is_empty())
                    })
            };
            let declares_functions = self
                .tools
                .iter()
                .flatten()
                .chain(
                    smuggled_tools
                        .and_then(|spelling| blob?.get(spelling))
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten(),
                )
                .any(declares_function);
            let tool_caveat = if declares_functions {
                " Note that function declarations in a cache are declarations only — rig's \
                 `Agent` can only dispatch tools it advertised, so a cached function tool set is \
                 never executable from an agent; it is usable only when you drive \
                 `GenerateContent` yourself and run the tool loop. (Provider-hosted tools such \
                 as `codeExecution` run on Gemini's side and are fine to keep in the cache.)"
            } else {
                ""
            };
            return Err(EncodeError::request(format!(
                "a Gemini request using cached content `{name}` also set {}. The cached \
                     content already owns the system instruction, tools and tool choice for every \
                     request that uses it — move them into the cache, or drop the cache \
                     handle.{tool_caveat}",
                conflicts.join(" and ")
            )));
        }

        self.cached_content = Some(name.to_owned());
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod cached_content_conflict_matrix;
#[cfg(test)]
mod cached_content_request_tests;
