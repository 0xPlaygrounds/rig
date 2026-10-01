//! Chat Completions request encoding, and one decoder for a whole reply and
//! a stream of chunks.
//!
//! ```
//! use rig_core::providers::openai::OpenAI;
//! let wire = OpenAI::new("key").chat("gpt-5.2");
//! ```

use serde::{Deserialize, Serialize};

use crate::completion::{CompletionRequest, FinishReason, ProviderCapabilities};
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::observe::ObservedError;
use crate::providers::internal::openai_chat_completions_compatible::{
    drop_tool_calls_cut_by_budget, map_native_finish_reason, map_openai_finish_reason,
    provider_error_envelope,
};
use crate::providers::internal::wire::classify_chat_completions_frame;
use crate::providers::openai::completion::{
    self as unary, AssistantContent, Message, ToolChoice, assistant_refusal_fallback,
    is_openai_reasoning_model, request_body,
};
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Body, Capabilities, Decoder, Descriptor, Encoded,
    Framing, Mode, ObservationSink, Out, Wire, WireEvent, WireFrame,
};

use super::dto::{
    ChatChoice, ChatFrame, ChatUsage, StreamingCompletionResponse, StreamingDelta, delta_text,
};
use super::{BodyRewrite, OpenAIConfig, OutputCap};

/// The chat-completions wire: a provider configuration, a model, and the
/// per-turn options the endpoint takes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chat {
    /// Which provider, and how to reach it.
    pub provider: OpenAIConfig,
    /// The model this wire addresses.
    pub model: String,
    /// Whether tool schemas are sanitized for OpenAI's strict mode:
    /// `additionalProperties: false` on every object, every property
    /// required, and `strict: true` on each function definition.
    pub strict_tools: bool,
    /// Whether tool-result messages serialize their content as arrays.
    pub tool_result_array_content: bool,
    /// Whether the request asks for provider-side prompt caching
    /// (OpenRouter's ephemeral `cache_control` on the system prompt).
    pub prompt_caching: bool,
}

impl Chat {
    pub(crate) fn encode_with_headers(
        &self,
        request: CompletionRequest,
        mode: Mode,
        headers: impl FnOnce(
            &OpenAIConfig,
            &CompletionRequest,
            http::request::Builder,
        ) -> http::request::Builder,
    ) -> Result<Encoded, EncodeError> {
        let (request, issuers) =
            super::scope_reasoning(&self.provider.dialect, &self.model, request)?;
        let quirks = &self.provider.dialect.quirks;
        // Azure's deployment URL remains pinned to the handle, not a request override.
        let uri = self.provider.uri(
            quirks.completion_path,
            self.provider.deployment(&self.model),
        );
        let builder = headers(
            &self.provider,
            &request,
            http::Request::post(uri).header("Content-Type", "application/json"),
        );
        if !quirks.accepts_file_ids {
            refuse_file_ids(&request)?;
        }
        let mut typed = unary::CompletionRequest::try_from(unary::OpenAIRequestParams {
            model: self.model.clone(),
            request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
            supports_response_format: quirks.supports_response_format,
            response_format_with_tools: quirks.response_format_with_tools,
            supports_tools: quirks.supports_tools,
            supports_image_tool_results: quirks.supports_image_tool_results,
            reasoning_details: quirks.reasoning_details,
            issuers,
        })?;
        self.prepare(&mut typed)?;

        // The resolved model, not the handle's: a per-request override
        // changes which endpoint answers, so it decides the spelling too.
        let modern_output_cap = match quirks.output_cap {
            OutputCap::Legacy => false,
            OutputCap::OpenAiReasoningFamilies => is_openai_reasoning_model(&typed.model),
        };
        let mut body = request_body(&typed, modern_output_cap)?;

        if mode == Mode::Streaming {
            if quirks.stream_include_usage {
                // Preserve caller stream options, including an explicit include_usage value.
                match body.get_mut("stream_options") {
                    Some(serde_json::Value::Object(options)) => {
                        options
                            .entry("include_usage")
                            .or_insert(serde_json::Value::Bool(true));
                    }
                    Some(_) => {}
                    None => {
                        body = crate::json_utils::merge(
                            body,
                            serde_json::json!({"stream_options": {"include_usage": true}}),
                        );
                    }
                }
            }
            body = crate::json_utils::merge(body, serde_json::json!({"stream": true}));
        }
        self.finalize(&mut body)?;

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "OpenAI Chat Completions request",
            &body,
        );

        let request = builder.body(Body::Bytes(serde_json::to_vec(&body)?))?;

        let framing = match mode {
            Mode::Streaming => Framing::Sse,
            Mode::Unary => Framing::Whole,
        };
        Ok(Encoded::new(request, framing)
            .with_request_id_header(self.provider.dialect.request_id_header)
            .with_projection(ChatDecoder::project)
            .with_route(Some(self.provider.dialect.quirks.completion_path)))
    }

    /// The wire for `model` on `provider`, with every option off.
    pub fn new(provider: OpenAIConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            strict_tools: false,
            tool_result_array_content: false,
            prompt_caching: false,
        }
    }

    /// Sanitize tool schemas for OpenAI's strict mode, so the provider can
    /// guarantee a tool call matches its schema exactly.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// Serialize tool-result content as arrays.
    pub fn with_tool_result_array_content(mut self) -> Self {
        self.tool_result_array_content = true;
        self
    }

    /// Ask the provider to cache the prompt.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Apply typed dialect rewrites, rejecting unsupported tool choices or parameters.
    fn prepare(&self, request: &mut unary::CompletionRequest) -> Result<(), EncodeError> {
        // Only OpenAI's own endpoint serves its reasoning families.
        if matches!(
            self.provider.dialect.quirks.output_cap,
            OutputCap::OpenAiReasoningFamilies
        ) {
            refuse_tools_while_reasoning(request)?;
        }
        match self.provider.dialect.quirks.rewrite {
            BodyRewrite::GroqCompoundTools => {
                fold_groq_native_tools(request)?;
                strip_assistant_reasoning(request);
            }
            BodyRewrite::LlamaCpp => {
                if let Some(ToolChoice::Function { name }) = &request.tool_choice {
                    return Err(EncodeError::request(format!(
                        "llama.cpp cannot force a specific tool: `llama-server` accepts only \
                         `auto`, `none` or `required` for tool_choice and silently treats \
                         anything else as `auto`, so requesting `{name}` would return whichever \
                         tool the model picked. Use `ToolChoice::Required` to force a call, or \
                         advertise only `{name}` in `tools`."
                    )));
                }
            }
            BodyRewrite::Moonshot => steer_moonshot_tool_choice(request)?,
            BodyRewrite::Mira => {
                // The gateway rejects pass-through parameters.
                if request.additional_params.take().is_some() {
                    tracing::warn!(
                        "Additional parameters are not supported by Mira and will be ignored"
                    );
                }
            }
            BodyRewrite::HuggingFaceRouter => {
                // Some sub-providers (Fireworks) address models through a
                // qualified identifier in the request body.
                request.model = self.provider.route().model_identifier(&request.model);
            }
            BodyRewrite::None
            | BodyRewrite::DeepSeek
            | BodyRewrite::Perplexity
            | BodyRewrite::Hyperbolic
            | BodyRewrite::Mistral
            | BodyRewrite::OpenRouter => {}
        }
        Ok(())
    }

    /// Apply dialect body rewrites after merging streaming parameters.
    /// Return conversion errors for unsupported content.
    fn finalize(&self, body: &mut serde_json::Value) -> Result<(), EncodeError> {
        let Some(map) = body.as_object_mut() else {
            return Ok(());
        };
        match self.provider.dialect.quirks.rewrite {
            BodyRewrite::Perplexity => {
                // Perplexity accepts only system/user/assistant roles with
                // strict user/assistant alternation. Text-only content-part
                // arrays flatten; arrays with non-text parts are left for
                // its own multimodal handling on the sonar models.
                if let Some(messages) = map.get_mut("messages").and_then(as_array_mut) {
                    unary::sanitize_plain_text_history(messages, Some(("\n", true)), false, true);
                }
            }
            BodyRewrite::Hyperbolic => {
                // Strip tool-exchange remnants a shared history may carry;
                // content-part arrays stay as-is for the vision models.
                if let Some(messages) = map.get_mut("messages").and_then(as_array_mut) {
                    unary::sanitize_plain_text_history(messages, None, false, false);
                }
            }
            BodyRewrite::Mira => {
                if let Some(messages) = map.get_mut("messages").and_then(as_array_mut) {
                    unary::sanitize_plain_text_history(messages, Some(("\n", false)), true, false);
                }
            }
            BodyRewrite::DeepSeek => finalize_deepseek(map),
            BodyRewrite::Mistral => finalize_mistral(map)?,
            BodyRewrite::OpenRouter => finalize_openrouter(map, self.prompt_caching),
            BodyRewrite::None
            | BodyRewrite::HuggingFaceRouter
            | BodyRewrite::GroqCompoundTools
            | BodyRewrite::LlamaCpp
            | BodyRewrite::Moonshot => {}
        }
        Ok(())
    }
}

/// GPT-6 models that call function tools on Chat Completions only at
/// `reasoning_effort: "none"` (their model pages), and those that do not
/// support `"none"` at all, so never call tools there.
const TOOLS_ONLY_WITHOUT_REASONING: [&str; 2] = [unary::GPT_6_SOL, unary::GPT_6_LUNA];
const NO_TOOLS_ON_CHAT: [&str; 2] = [unary::GPT_6_ASTRA, unary::GPT_6_1_SOL];

/// Whether `model` is `id` or one of its dated snapshots (`<id>-YYYY-MM-DD`).
fn is_model(model: &str, id: &str) -> bool {
    model
        .strip_prefix(id)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with("-20"))
}

/// Refuse a Chat Completions request with function tools on a GPT-6 model
/// that would answer it with a 400, naming the fix. A caller who already
/// sends `reasoning_effort: "none"` where the model takes it is not refused.
fn refuse_tools_while_reasoning(request: &unary::CompletionRequest) -> Result<(), EncodeError> {
    if request.tools.is_empty() {
        return Ok(());
    }
    let model = request.model.as_str();
    if NO_TOOLS_ON_CHAT.iter().any(|id| is_model(model, id)) {
        return Err(EncodeError::request(format!(
            "{model} cannot call function tools on Chat Completions: it takes them there only \
             at reasoning_effort \"none\", which it does not support. Use the Responses wire."
        )));
    }
    let effort_none = request
        .additional_params
        .as_ref()
        .and_then(|params| params.get("reasoning_effort"))
        .and_then(serde_json::Value::as_str)
        == Some("none");
    if TOOLS_ONLY_WITHOUT_REASONING
        .iter()
        .any(|id| is_model(model, id))
        && !effort_none
    {
        return Err(EncodeError::request(format!(
            "{model} calls function tools on Chat Completions only at reasoning_effort \"none\": \
             send `\"reasoning_effort\": \"none\"` in additional_params, or use the Responses wire."
        )));
    }
    Ok(())
}

fn as_array_mut(value: &mut serde_json::Value) -> Option<&mut Vec<serde_json::Value>> {
    value.as_array_mut()
}

/// Groq delivers hidden reasoning under `reasoning` and rejects an assistant
/// message that carries `reasoning_content` with a 400, so a reasoning turn
/// replays without it. The reasoning is dropped from the replay rather than
/// respelled: whether Groq accepts it back under `reasoning` is unverified.
fn strip_assistant_reasoning(request: &mut unary::CompletionRequest) {
    for message in &mut request.messages {
        if let Message::Assistant { reasoning, .. } = message {
            *reasoning = None;
        }
    }
}

/// Groq's compound-system native tools (`browser_search`, `code_interpreter`,
/// …) arrive through `additional_params.tools`. Left there they would clobber
/// the function-tool array on serialization, because `additional_params` is
/// flattened into the body and the flattened key wins.
fn fold_groq_native_tools(request: &mut unary::CompletionRequest) -> Result<(), EncodeError> {
    let Some(map) = request
        .additional_params
        .as_mut()
        .and_then(serde_json::Value::as_object_mut)
    else {
        return Ok(());
    };
    let Some(raw_tools) = map.remove("tools") else {
        return Ok(());
    };
    let serde_json::Value::Array(native_tools) = raw_tools else {
        return Err(EncodeError::request(
            "Groq `additional_params.tools` must be an array of native tool objects",
        ));
    };

    // `compound_custom.enabled_tools` is a set keyed by tool type, so a
    // caller who names the same tool twice enables it once.
    let enabled = map
        .entry("compound_custom")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .map(|custom| {
            custom
                .entry("enabled_tools")
                .or_insert_with(|| serde_json::Value::Array(Vec::new()))
        });
    let Some(serde_json::Value::Array(enabled)) = enabled else {
        return Ok(());
    };
    for tool in native_tools {
        let kind = tool.get("type").and_then(serde_json::Value::as_str);
        let already_enabled = enabled
            .iter()
            .any(|existing| existing.get("type").and_then(serde_json::Value::as_str) == kind);
        if !already_enabled {
            enabled.push(tool);
        }
    }
    Ok(())
}

/// Moonshot supports only `auto`/`none`: forcing one specific tool has no
/// workaround, and `required` is steered with an extra user message.
fn steer_moonshot_tool_choice(request: &mut unary::CompletionRequest) -> Result<(), EncodeError> {
    if matches!(request.tool_choice, Some(ToolChoice::Function { .. })) {
        return Err(EncodeError::request(
            "Moonshot does not support forcing a specific tool".to_owned(),
        ));
    }
    if matches!(request.tool_choice, Some(ToolChoice::Required)) {
        tracing::warn!(
            "Moonshot does not support tool_choice=required; coercing to auto with an \
             additional steering message"
        );
        request.tool_choice = Some(ToolChoice::Auto);
        request.messages.push(Message::User {
            content: vec![unary::UserContent::Text {
                text: "Please select a tool to handle the current issue.".to_owned(),
            }],
            name: None,
        });
    }
    Ok(())
}

/// DeepSeek takes message `content` as a plain string, echoes tool calls back
/// with an `index`, and needs an explicit empty `content` on a tool-call-only
/// assistant turn.
fn finalize_deepseek(map: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(messages) = map.get_mut("messages").and_then(as_array_mut) {
        for message in messages {
            let Some(message) = message.as_object_mut() else {
                continue;
            };
            let is_assistant =
                message.get("role").and_then(serde_json::Value::as_str) == Some("assistant");

            if let Some(content) = message.get_mut("content") {
                let separator = if is_assistant { "" } else { "\n" };
                // Preserve nontext parts so unsupported attachments are rejected
                // rather than silently omitted from the prompt.
                unary::flatten_text_content_parts(content, separator, true);
            } else if is_assistant {
                message.insert(
                    "content".to_owned(),
                    serde_json::Value::String(String::new()),
                );
            }

            if is_assistant
                && let Some(tool_calls) = message.get_mut("tool_calls").and_then(as_array_mut)
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

    // DeepSeek rejects forced tool choices unless thinking is explicitly
    // disabled; suppress them to an explicit `null` otherwise.
    let thinking_disabled = map
        .get("thinking")
        .and_then(|thinking| thinking.get("type"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|mode| mode.eq_ignore_ascii_case("disabled"));
    if !thinking_disabled
        && let Some(tool_choice) = map.get_mut("tool_choice")
        && (tool_choice.is_object() || tool_choice.as_str() == Some("required"))
    {
        *tool_choice = serde_json::Value::Null;
    }
}

/// Mistral's wire-level differences: its own spelling for a forced tool
/// choice, the relaxation that lets a structured format ride beside tools,
/// and its assistant-message schema.
///
/// Its multimodal content mapping is [`mistral_content`].
fn finalize_mistral(
    map: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<(), EncodeError> {
    // Mistral spells the "must call some tool" mode `any`, not `required`.
    if let Some(tool_choice) = map.get_mut("tool_choice")
        && tool_choice.as_str() == Some("required")
    {
        *tool_choice = serde_json::Value::String("any".to_owned());
    }

    // Mistral rejects forced tool calls beside JSON response formats.
    // Relax the choice rather than discard the requested schema; text formats are exempt.
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
            "tool_choice".to_owned(),
            serde_json::Value::String("auto".to_owned()),
        );
    }

    let Some(messages) = map.get_mut("messages").and_then(as_array_mut) else {
        return Ok(());
    };
    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        let is_assistant =
            message.get("role").and_then(serde_json::Value::as_str) == Some("assistant");

        // Mistral takes text-only message `content` as a plain string and
        // carries images, audio and documents as its own chunk array.
        // Content it has no chunk for fails here rather than reaching the API
        // with the part removed.
        if let Some(content) = message.get_mut("content") {
            mistral_content(content)?;
        }

        if is_assistant {
            if !message.contains_key("content") {
                message.insert(
                    "content".to_owned(),
                    serde_json::Value::String(String::new()),
                );
            }
            // `prefix` is part of Mistral's assistant message schema.
            message
                .entry("prefix")
                .or_insert(serde_json::Value::Bool(false));
            // Mistral rejects unknown assistant fields; hidden reasoning
            // cannot be echoed back.
            message.remove("reasoning_content");
        }
    }
    Ok(())
}

/// Mistral's text chunk tag.
const MISTRAL_TEXT: &str = "text";
/// Mistral's image chunk tag.
const MISTRAL_IMAGE: &str = "image_url";
/// Mistral's audio chunk tag.
const MISTRAL_AUDIO: &str = "input_audio";
/// Mistral's document chunk tag.
const MISTRAL_DOCUMENT: &str = "document_url";
/// Mistral's uploaded-file chunk tag.
const MISTRAL_FILE: &str = "file";
/// OpenAI's refusal part: textual, but under a key Mistral's chunk schema
/// has no field for, so it is re-tagged rather than forwarded.
const MISTRAL_REFUSAL: &str = "refusal";

/// The text a part carries, under either key the shared conversion uses.
fn mistral_part_text(part: &serde_json::Value) -> Option<&str> {
    part.get(MISTRAL_TEXT)
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            part.get(MISTRAL_REFUSAL)
                .and_then(serde_json::Value::as_str)
        })
}

/// Identify text and refusal parts by tag, falling back to payload keys without a tag.
/// An explicit nontext tag prevents flattening even when a text key is present.
fn is_mistral_text_part(part: &serde_json::Value) -> bool {
    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(MISTRAL_TEXT | MISTRAL_REFUSAL) => true,
        Some(_) => false,
        None => mistral_part_text(part).is_some(),
    }
}

fn mistral_unsupported(what: &str) -> EncodeError {
    crate::message::MessageError::ConversionError(format!(
        "Mistral cannot carry {what}. Mistral messages accept text, `{MISTRAL_IMAGE}`, \
         `{MISTRAL_AUDIO}`, `{MISTRAL_DOCUMENT}` and `{MISTRAL_FILE}` content; convert the \
         content to one of those before sending it."
    ))
    .into()
}

/// Convert file data to a document URL or a file reference to a top-level file ID.
/// Preserve optional filenames for inline documents. Return a conversion error
/// when neither file data nor a file ID is present.
fn mistral_file_chunk(part: &serde_json::Value) -> Result<serde_json::Value, EncodeError> {
    let file = part.get(MISTRAL_FILE);
    let field = |name: &str| {
        file.and_then(|file| file.get(name))
            .and_then(serde_json::Value::as_str)
    };

    // Accept already-converted file references to keep finalization idempotent.
    if let Some(file_id) = part.get("file_id").and_then(serde_json::Value::as_str) {
        return Ok(serde_json::json!({"type": MISTRAL_FILE, "file_id": file_id}));
    }

    if let Some(data) = field("file_data") {
        // `document_name` is left out entirely rather than sent as null when
        // the part has no filename.
        Ok(match field("filename") {
            Some(filename) => serde_json::json!({
                "type": MISTRAL_DOCUMENT,
                MISTRAL_DOCUMENT: data,
                "document_name": filename,
            }),
            None => serde_json::json!({"type": MISTRAL_DOCUMENT, MISTRAL_DOCUMENT: data}),
        })
    } else if let Some(file_id) = field("file_id") {
        Ok(serde_json::json!({"type": MISTRAL_FILE, "file_id": file_id}))
    } else {
        Err(mistral_unsupported(
            "a file content part carrying neither `file_data` nor `file_id`",
        ))
    }
}

/// Convert string or object audio payloads to Mistral's base64-string form.
/// Return a conversion error for missing or nonstring audio data.
fn mistral_audio_chunk(part: &serde_json::Value) -> Result<serde_json::Value, EncodeError> {
    let payload = part.get(MISTRAL_AUDIO).ok_or_else(|| {
        mistral_unsupported("an audio content part carrying no `input_audio` payload")
    })?;

    let data = match payload {
        serde_json::Value::String(data) => data.as_str(),
        payload => payload
            .get("data")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                mistral_unsupported(
                    "an audio content part whose `input_audio` payload is not base64 data",
                )
            })?,
    };

    Ok(serde_json::json!({"type": MISTRAL_AUDIO, MISTRAL_AUDIO: data}))
}

/// One content part as the Mistral chunk that carries it.
///
/// Dispatched on the `type` tag, which the shared conversion always emits, so
/// a part naming a chunk kind is converted as that kind regardless of what
/// other keys it carries.
fn mistral_chunk(part: &serde_json::Value) -> Result<serde_json::Value, EncodeError> {
    /// Text and refusal parts are both re-tagged `text`: Mistral's schema has
    /// no `refusal` field and every chunk forbids unknown keys.
    fn text_chunk(part: &serde_json::Value) -> Result<serde_json::Value, EncodeError> {
        let text = mistral_part_text(part)
            .ok_or_else(|| mistral_unsupported("a text content part carrying no text"))?;
        Ok(serde_json::json!({"type": MISTRAL_TEXT, MISTRAL_TEXT: text}))
    }

    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(MISTRAL_TEXT | MISTRAL_REFUSAL) => text_chunk(part),
        // Rebuild the envelope because Mistral rejects unknown sibling fields.
        Some(MISTRAL_IMAGE) => {
            let image = part.get(MISTRAL_IMAGE).ok_or_else(|| {
                mistral_unsupported("an image content part carrying no `image_url` payload")
            })?;
            Ok(serde_json::json!({"type": MISTRAL_IMAGE, MISTRAL_IMAGE: image}))
        }
        Some(MISTRAL_AUDIO) => mistral_audio_chunk(part),
        Some(MISTRAL_FILE) => mistral_file_chunk(part),
        // Accept already-converted document parts to keep finalization idempotent.
        Some(MISTRAL_DOCUMENT) => {
            let url = part.get(MISTRAL_DOCUMENT).ok_or_else(|| {
                mistral_unsupported("a document content part carrying no `document_url`")
            })?;
            Ok(match part.get("document_name") {
                Some(name) => serde_json::json!({
                    "type": MISTRAL_DOCUMENT, MISTRAL_DOCUMENT: url, "document_name": name,
                }),
                None => serde_json::json!({"type": MISTRAL_DOCUMENT, MISTRAL_DOCUMENT: url}),
            })
        }
        Some(kind) => Err(mistral_unsupported(&format!("`{kind}` message content"))),
        // Untagged, but textual: the flattening would have taken it, so it
        // converts rather than failing.
        None if mistral_part_text(part).is_some() => text_chunk(part),
        None => Err(mistral_unsupported("untyped message content")),
    }
}

/// Flatten text-only arrays and convert mixed arrays to Mistral content chunks.
/// Nonarrays remain unchanged. Unsupported mixed content returns a conversion
/// error; text-only parts without string payloads are omitted.
fn mistral_content(content: &mut serde_json::Value) -> Result<(), EncodeError> {
    let Some(parts) = content.as_array() else {
        return Ok(());
    };

    if parts.iter().all(is_mistral_text_part) {
        // The tag-based guard is authoritative even for text parts missing their payload.
        unary::flatten_text_content_parts(content, "", false);
        return Ok(());
    }

    if let Some(parts) = content.as_array_mut() {
        for part in parts {
            *part = mistral_chunk(part)?;
        }
    }

    Ok(())
}

/// OpenRouter's body rewrites.
///
/// OpenRouter's routing preferences (`ProviderPreferences`) need no rewrite:
/// they reach the body through the request's `additional_params` as
/// `{"provider": …}`.
fn finalize_openrouter(map: &mut serde_json::Map<String, serde_json::Value>, prompt_caching: bool) {
    if prompt_caching {
        apply_openrouter_prompt_caching(map);
    }

    let Some(messages) = map.get_mut("messages").and_then(as_array_mut) else {
        return;
    };
    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        // The shared assistant message serializes hidden reasoning under the
        // llama.cpp/DeepSeek key `reasoning_content`; OpenRouter's documented
        // assistant field is `reasoning`.
        if message.get("role").and_then(serde_json::Value::as_str) == Some("assistant")
            && let Some(reasoning) = message.remove("reasoning_content")
        {
            message.insert("reasoning".to_owned(), reasoning);
        }

        // OpenRouter image parts omit the shared fidelity hint.
        for part in message
            .get_mut("content")
            .and_then(as_array_mut)
            .into_iter()
            .flatten()
        {
            if let Some(image) = part
                .get_mut("image_url")
                .and_then(serde_json::Value::as_object_mut)
            {
                image.remove("detail");
            }
        }
    }
}

fn apply_openrouter_prompt_caching(map: &mut serde_json::Map<String, serde_json::Value>) {
    let Some(messages) = map.get_mut("messages").and_then(as_array_mut) else {
        return;
    };
    let Some(system) = messages
        .iter_mut()
        .find(|message| message.get("role").and_then(serde_json::Value::as_str) == Some("system"))
    else {
        return;
    };
    match system.get("content").cloned() {
        Some(serde_json::Value::String(text)) => {
            if let Some(object) = system.as_object_mut() {
                object.insert(
                    "content".to_owned(),
                    serde_json::json!([{
                        "type": "text",
                        "text": text,
                        "cache_control": { "type": "ephemeral" }
                    }]),
                );
            }
        }
        Some(serde_json::Value::Array(mut parts)) => {
            // The last block marks the cache boundary without altering earlier content.
            if let Some(last) = parts.last_mut()
                && let Some(object) = last.as_object_mut()
            {
                object.insert(
                    "cache_control".to_owned(),
                    serde_json::json!({ "type": "ephemeral" }),
                );
            }
            if let Some(object) = system.as_object_mut() {
                object.insert("content".to_owned(), serde_json::Value::Array(parts));
            }
        }
        _ => {}
    }
}

/// Return a request error for document or image inputs using provider file IDs.
fn refuse_file_ids(request: &CompletionRequest) -> Result<(), EncodeError> {
    use crate::message::{DocumentSourceKind, Message, UserContent};

    let refusal = || {
        EncodeError::request("Provider file IDs are not supported for OpenRouter document inputs")
    };
    for message in &request.chat_history {
        let Message::User { content, .. } = message else {
            continue;
        };
        for part in content {
            match part {
                UserContent::Document(document) => {
                    if matches!(document.data, DocumentSourceKind::FileId(_)) {
                        return Err(refusal());
                    }
                }
                UserContent::Image(image) => {
                    if matches!(image.data, DocumentSourceKind::FileId(_)) {
                        return Err(refusal());
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

impl Wire for Chat {
    type Op = crate::operation::Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ChatDecoder<'id>;

    /// Format deferral permits tool composition; dialects without schema
    /// support require the agent's tool-mode enforcement instead.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
            .model(self.model.as_str())
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default().with_native_output_tool_composition(
                    self.provider.dialect.quirks.supports_response_format,
                ),
            ))
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        self.encode_with_headers(request, mode, OpenAIConfig::completion_headers)
    }

    fn decoder<'id>(&self) -> ChatDecoder<'id> {
        ChatDecoder::new(self.provider.dialect.name, self.provider.dialect.quirks)
    }
}

/// Classified Chat Completions frame, including whole replies and terminal signals.
pub enum ChatEvent {
    /// A `chat.completion.chunk`: one step of a streamed turn.
    Chunk(ChatFrame),
    /// A `chat.completion`: the whole turn in one frame.
    Whole(ChatFrame),
    /// The `[DONE]` sentinel: the provider ended the stream.
    Done,
    /// The wire's in-band error envelope, delivered with a 200 status.
    Failure(ProviderError),
    /// A bare JSON string where an envelope belongs: the whole answer, with
    /// no metadata and no terminal reason. Mira's gateway sends this.
    BareText(String),
}

/// The chat-completions decoder: one state machine for a whole reply and a
/// stream of chunks.
pub struct ChatDecoder<'id> {
    /// Descriptor name the reply is attributed to.
    provider: &'static str,
    quirks: super::Quirks,
    /// The text part bare text extends; a reasoning fragment or a tool call
    /// closes it.
    text: Option<TextPart<'id>>,
    /// `reasoning_content` carries no ids or boundaries.
    thoughts: Thoughts<'id>,
    final_usage: Option<ChatUsage>,
    final_finish_reason: Option<FinishReason>,
    response_id: Option<String>,
    response_model: Option<String>,
    /// Accumulated primary-choice token metadata. `AdditionalParams::merge`
    /// concatenates nested arrays, which is the wire's token order.
    logprobs: Option<crate::message::AdditionalParams>,
    /// Accumulated provider-specific top-level chunk metadata.
    additional_params: Option<crate::message::AdditionalParams>,
    /// Whether a finish reason established the turn complete.
    saw_terminal: bool,
    /// Whether any frame decoded successfully. A bare `[DONE]` after only
    /// parse failures must not dress the failure up as a default-usage
    /// success.
    saw_any_valid_frame: bool,
}

impl<'id> ChatDecoder<'id> {
    fn new(provider: &'static str, quirks: super::Quirks) -> Self {
        Self {
            provider,
            quirks,
            text: None,
            thoughts: Thoughts::new(),
            final_usage: None,
            final_finish_reason: None,
            response_id: None,
            response_model: None,
            logprobs: None,
            additional_params: None,
            saw_terminal: false,
            saw_any_valid_frame: false,
        }
    }

    /// The normalized finish reason a choice reported, `None` when the
    /// chunk carried no `finish_reason`.
    ///
    /// A gateway's upstream-native reason is consulted only when the
    /// normalized field is absent or empty, which is OpenRouter's documented
    /// precedence; a direct provider has no native field to consult.
    fn finish_reason(&self, choice: &ChatChoice) -> Option<FinishReason> {
        if let Some(reason) = choice
            .finish_reason
            .as_ref()
            .map(super::dto::FinishReason::as_wire)
            .filter(|reason| !reason.is_empty())
        {
            return Some(map_openai_finish_reason(reason));
        }
        if self.quirks.native_finish_reason
            && let Some(native) = choice
                .native_finish_reason
                .as_deref()
                .filter(|reason| !reason.is_empty())
        {
            return Some(map_native_finish_reason(native));
        }
        None
    }

    /// Absorb the metadata every frame carries, whichever shape it is.
    fn absorb_metadata(&mut self, frame: &mut ChatFrame) {
        if let Some(id) = frame.id.take() {
            self.response_id = Some(id);
        }
        if let Some(model) = frame.model.take() {
            self.response_model = Some(model);
        }
        if let Some(usage) = frame.usage.take() {
            self.final_usage = Some(usage);
        }
        if let Some(additional_params) =
            crate::message::AdditionalParams::new(std::mem::take(&mut frame.additional_params))
        {
            match self.additional_params.as_mut() {
                Some(accumulated) => accumulated.merge(additional_params),
                None => self.additional_params = Some(additional_params),
            }
        }
    }

    /// One chunk's parts, in the order the wire implies: reasoning, its
    /// signature or the boundary that stops it, text, then tool calls.
    fn emit_parts(
        &mut self,
        out: &mut Out<'id, Completion>,
        reasoning: Option<String>,
        signature: Option<String>,
        text: Option<String>,
        calls: bool,
    ) {
        if let Some(reasoning) = reasoning.filter(|reasoning| !reasoning.is_empty()) {
            out.close_open_text(&mut self.text);
            self.thoughts.fragment(out, &reasoning);
        }
        if let Some(signature) = signature {
            self.thoughts.signature(out, signature);
        }
        let text = text.filter(|text| !text.is_empty());
        if text.is_some() || calls {
            // These wires omit the boundary before interleaving output.
            self.thoughts.boundary();
        }
        if let Some(text) = text {
            out.extend_text(&mut self.text, &text);
        }
    }

    /// One `chat.completion.chunk`.
    fn interpret_chunk(
        &mut self,
        mut frame: ChatFrame,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        self.saw_any_valid_frame = true;
        self.absorb_metadata(&mut frame);
        let Some(choice) = frame.into_primary() else {
            return Ok(());
        };
        let finish_reason = self.finish_reason(&choice);
        let text = delta_text(&choice.delta);
        let StreamingDelta {
            reasoning_content,
            reasoning,
            tool_calls,
            reasoning_details,
            ..
        } = choice.delta;
        let reasoning = reasoning_content.or(reasoning);
        let details: Vec<unary::ReasoningDetails> =
            reasoning_details.iter().filter_map(typed_detail).collect();

        if let Some(reason) = &finish_reason {
            self.final_finish_reason = Some(reason.clone());
            self.saw_terminal = true;
        }

        if let Some(logprobs) = choice.logprobs {
            match self.logprobs.as_mut() {
                Some(accumulated) => accumulated.merge(logprobs),
                None => self.logprobs = Some(logprobs),
            }
        }

        // Replayable reasoning must precede the tool calls it accompanies.
        if self.quirks.reasoning_details {
            for detail in &details {
                if let Some(reasoning) = detail_reasoning(detail) {
                    out.close_open_text(&mut self.text);
                    out.reasoning_block(reasoning);
                }
            }
        }

        let reasoning_signature = self
            .quirks
            .reasoning_details
            .then(|| details.iter().find_map(reasoning_signature))
            .flatten();

        self.emit_parts(
            out,
            reasoning,
            reasoning_signature,
            text,
            !tool_calls.is_empty(),
        );

        for incoming in tool_calls {
            out.close_open_text(&mut self.text);
            if let Some(existing) = out.pending_id(incoming.index)
                && incoming.evicts(&existing, &out.pending_name(incoming.index))
            {
                // The wire reused this call's index: the call it held is
                // delivered even when its arguments never parse.
                out.close_pending(incoming.index, IfMalformed::EmptyObject)?;
            }
            out.call_fragment(
                incoming.index,
                CallFragment {
                    id: incoming.id.as_deref(),
                    name: incoming.function.name.as_deref(),
                    arguments: incoming.function.arguments.as_deref(),
                    ..CallFragment::default()
                },
            )?;
            if self.quirks.emits_complete_single_chunk_tool_calls
                && incoming.is_complete_single_chunk()
            {
                // A probe: the call closes if its input parses, and stays
                // open for more fragments otherwise.
                out.close_pending(incoming.index, IfMalformed::KeepOpen)?;
            }
        }

        if matches!(finish_reason, Some(FinishReason::ToolCalls)) {
            for index in out.pending_calls() {
                // Completed calls with malformed arguments must fail, not
                // disappear. Empty arguments remain valid for zero-argument
                // tools.
                out.close_pending(index, IfMalformed::Fail)?;
            }
        }
        Ok(())
    }

    /// Whether a length-truncated unary choice contains tool calls needing raw inspection.
    /// Empty argument strings normalize to `{}`, so typed arguments alone cannot
    /// distinguish truncation before the first token from a zero-argument call.
    fn is_budget_cut_tool_turn(&self, frame: &ChatFrame) -> bool {
        let Some(choice) = frame.primary() else {
            return false;
        };
        if !matches!(self.finish_reason(choice), Some(FinishReason::Length)) {
            return false;
        }
        matches!(
            &choice.message,
            Some(Message::Assistant { tool_calls, .. }) if !tool_calls.is_empty()
        )
    }

    /// Whether a raw choice blames the output-token budget, under the same
    /// precedence [`Self::finish_reason`] applies to a decoded one.
    fn reports_output_length(&self, choice: &serde_json::Value) -> bool {
        let reason = |key: &str| {
            choice
                .get(key)
                .and_then(serde_json::Value::as_str)
                .filter(|reason| !reason.is_empty())
        };
        if let Some(normalized) = reason("finish_reason") {
            return matches!(map_openai_finish_reason(normalized), FinishReason::Length);
        }
        self.quirks.native_finish_reason
            && reason("native_finish_reason").is_some_and(|native| {
                matches!(map_native_finish_reason(native), FinishReason::Length)
            })
    }

    /// Decode a unary body after dropping incomplete calls from length-truncated choices.
    /// Preserve valid arguments and require the shared compound-defect check before
    /// dropping calls. Return `None` if nothing is dropped or decoding still fails.
    fn body_without_calls_cut_by_the_budget(&self, data: &str) -> Option<ChatFrame> {
        let mut body = serde_json::from_str::<serde_json::Value>(data).ok()?;
        let mut dropped = 0;
        for choice in body.get_mut("choices").and_then(as_array_mut)? {
            if self.reports_output_length(choice) {
                dropped += drop_tool_calls_cut_by_budget::<ChatChoice>(choice);
            }
        }
        if dropped == 0 {
            return None;
        }
        let frame = serde_json::from_value::<ChatFrame>(body).ok()?;
        tracing::debug!(
            provider = self.provider,
            dropped,
            "dropping unary tool calls whose arguments the output-token budget cut short"
        );
        Some(frame)
    }

    /// The `chat.completion` body: the parts a stream of the same turn
    /// would have written, then the end.
    fn interpret_whole(
        &mut self,
        mut frame: ChatFrame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        self.saw_any_valid_frame = true;
        let Some(choice) = frame.primary() else {
            return Err(ProviderError::Response(
                "Response contained no choices".to_owned(),
            ));
        };
        let finish_reason = self.finish_reason(choice);
        let Some(Message::Assistant {
            content,
            reasoning,
            refusal,
            tool_calls,
            reasoning_details,
            ..
        }) = choice.message.clone()
        else {
            return Err(ProviderError::Response(
                "Response did not contain a valid message or tool call".to_owned(),
            ));
        };
        let logprobs = choice.logprobs.clone();
        self.absorb_metadata(&mut frame);
        self.logprobs = logprobs;
        self.final_finish_reason = finish_reason;
        self.saw_terminal = true;

        // Response IDs are not replayable message IDs; retain them only as terminal metadata.
        let text = {
            // The streamed path concatenates a turn's text fragments into one
            // part, so the unary body's parts join the same way rather than
            // producing a different number of parts for the same turn.
            let mut text = String::new();
            for part in &content {
                let part = match part {
                    AssistantContent::Text { text } => text,
                    AssistantContent::Refusal { refusal } => refusal,
                };
                text.push_str(part);
            }
            // This wire spells a refusal as a *sibling* of `content`
            // (`{"content": null, "refusal": "…"}`), so a path reading
            // `content` alone would drop it entirely.
            if let Some(refusal) = assistant_refusal_fallback(&content, refusal.as_deref()) {
                text.push_str(refusal);
            }
            text
        };

        let reasoning = reasoning.filter(|reasoning| !reasoning.is_empty());
        // Structured details retain signatures needed to replay reasoning.
        let details: Vec<&unary::ReasoningDetails> = if self.quirks.reasoning_details {
            reasoning_details.iter().collect()
        } else {
            Vec::new()
        };
        let blocks: Vec<_> = details
            .iter()
            .copied()
            .filter_map(whole_detail_reasoning)
            .collect();
        // Prefer replayable structured blocks to avoid duplicating their plaintext display.
        // Without those blocks, preserve plaintext and any signature-only detail.
        let (reasoning, reasoning_signature) = if blocks.is_empty() {
            (
                reasoning,
                details.iter().copied().find_map(reasoning_signature),
            )
        } else {
            (None, None)
        };
        // Truncation or filtering can leave no visible content; retain its reason and usage.
        // Empty replies without such a reason are response errors.
        let cut_short = self
            .final_finish_reason
            .as_ref()
            .is_some_and(FinishReason::truncated_output);
        if text.is_empty()
            && tool_calls.is_empty()
            && reasoning.is_none()
            && blocks.is_empty()
            && !cut_short
        {
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }

        // Reasoning details are the turn's own output, so they are written
        // before the text and tool calls, exactly as the streamed path orders
        // them.
        for reasoning in blocks {
            out.reasoning_block(reasoning);
        }

        self.emit_parts(
            &mut out,
            reasoning,
            reasoning_signature,
            (!text.is_empty()).then_some(text),
            !tool_calls.is_empty(),
        );
        out.close_open_text(&mut self.text);

        // Each call is buffered at its own position, so separate id-less
        // calls stay distinct.
        for (index, call) in tool_calls.iter().enumerate() {
            out.call_fragment(
                index,
                CallFragment {
                    id: Some(call.id.as_str()),
                    name: Some(call.function.name.as_str()),
                    ..CallFragment::default()
                },
            )?;
            out.announce_pending(index, call.function.arguments.clone());
            out.close_pending(index, IfMalformed::Fail)?;
        }

        self.end(out, false)
    }

    /// Write the provider's end of the reply. A stream's `raw` is the
    /// native terminal record the chunks built; a whole body's is the body
    /// itself, which the transport keeps.
    fn end(
        &mut self,
        mut out: Out<'id, Completion>,
        streamed: bool,
    ) -> Result<Flow, ProviderError> {
        out.close_open_text(&mut self.text);
        self.thoughts.close(&mut out, None);
        // A gateway's reasoning belongs to the upstream model that produced it.
        if self.quirks.upstream_reasoning_issuer
            && let Some(model) = self.response_model.as_deref()
        {
            out.issued_by(super::upstream_reasoning_issuer(self.provider, model));
        }
        let usage = self
            .final_usage
            .as_ref()
            .map(|usage| usage.to_normalized_for(&self.quirks));
        let native = StreamingCompletionResponse {
            usage: self.final_usage.take(),
            finish_reason: self.final_finish_reason.take(),
            response_id: self.response_id.take(),
            model: self.response_model.take(),
            logprobs: self.logprobs.take().map(Into::into),
            additional_params: self.additional_params.take(),
        };
        if streamed {
            out.raw(serde_json::to_value(&native)?);
        }
        let mut finish = native.into_finish();
        finish.usage = usage.unwrap_or_default();
        Ok(out.end(finish))
    }

    /// The stream ended: flush the calls the provider delivered, then end
    /// the reply. Tool calls the provider fully delivered are content, so
    /// a length cut still flushes them.
    fn finish(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        let output_length_truncation = matches!(
            self.final_finish_reason.as_ref(),
            Some(FinishReason::Length)
        );
        for index in out.pending_calls() {
            if output_length_truncation && !out.pending_has_arguments(index) {
                tracing::debug!(
                    "dropping streamed tool call cut off before its first argument token"
                );
                out.drop_pending(index);
                continue;
            }
            // Only an explicit length finish permits dropping malformed arguments.
            let if_malformed = if output_length_truncation {
                IfMalformed::Drop
            } else {
                IfMalformed::Fail
            };
            out.close_pending(index, if_malformed)?;
        }
        // A bare terminator without valid content cannot establish success.
        if !self.saw_any_valid_frame {
            return Err(ProviderError::Truncated);
        }
        self.end(out, true)
    }
}

use crate::operation::{CallFragment, Completion, IfMalformed, TextPart};
use crate::providers::internal::thoughts::Thoughts;
use crate::wire::Flow;

impl<'id> Decoder<'id, Completion> for ChatDecoder<'id> {
    type Event = ChatEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<ChatEvent> {
        let data = frame.as_str();
        // `[DONE]` is the wire's terminal sentinel, not JSON; it is Known by
        // definition and its decode ends the reply.
        if data == "[DONE]" {
            return WireEvent::Known(ChatEvent::Done);
        }
        // The wire's in-band error envelope arrives with a 200 status and is
        // this wire's own terminal failure, so it is a modeled event rather
        // than a transport-level filter.
        if let Some(error) = provider_error_envelope(&data) {
            return WireEvent::Known(ChatEvent::Failure(error));
        }
        // Supported bare-string replies must be recognized before object classification.
        if self.quirks.accepts_bare_string_reply
            && let Ok(serde_json::Value::String(text)) =
                serde_json::from_str::<serde_json::Value>(&data)
        {
            return WireEvent::Known(ChatEvent::BareText(text));
        }
        let classified = classify_chat_completions_frame::<ChatFrame>(&data);
        // Inspect raw arguments for length cuts, including empty strings normalized to {}.
        let may_be_budget_cut = match &classified {
            WireEvent::Corrupt(_) => true,
            WireEvent::Known(frame) => self.is_budget_cut_tool_turn(frame),
            WireEvent::Unknown { .. } => false,
        };
        if may_be_budget_cut && let Some(frame) = self.body_without_calls_cut_by_the_budget(&data) {
            return WireEvent::Known(ChatEvent::Whole(frame));
        }
        classified.map(|frame| {
            if frame.is_whole() {
                ChatEvent::Whole(frame)
            } else {
                ChatEvent::Chunk(frame)
            }
        })
    }

    fn decode(
        &mut self,
        event: ChatEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            ChatEvent::Chunk(frame) => {
                self.interpret_chunk(frame, &mut out)?;
                Ok(Flow::More)
            }
            ChatEvent::Whole(frame) => self.interpret_whole(frame, out),
            ChatEvent::Done => {
                if !self.saw_terminal {
                    // `[DONE]` without a finish reason still ends the turn.
                    self.saw_terminal = true;
                }
                self.finish(out)
            }
            ChatEvent::BareText(text) => {
                self.saw_any_valid_frame = true;
                self.saw_terminal = true;
                if !text.is_empty() {
                    let part = out.text();
                    out.push_text(&part, &text);
                    out.close_text(part);
                }
                self.end(out, false)
            }
            ChatEvent::Failure(error) => Err(error),
        }
    }

    /// A stream that stops after a finish reason without `[DONE]` still
    /// ended: some dialects (Perplexity) never send the sentinel.
    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        if !self.saw_terminal {
            return Err(ProviderError::Truncated);
        }
        self.finish(out)
    }
}

impl ChatDecoder<'_> {
    /// Verdict, model, response id, usage and error envelope, read off a raw
    /// payload before normalization discards them. The driver calls it for
    /// the unary reply and for every stream frame without anyone having to
    /// attach it.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(payload) = serde_json::from_slice::<ObservedPayload>(payload) else {
            return;
        };
        if let Some(usage) = payload.usage {
            sink.emit(AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                    total_tokens: usage.total_tokens,
                    cached_input_tokens: usage
                        .prompt_tokens_details
                        .and_then(|details| details.cached_tokens),
                    reasoning_tokens: usage
                        .completion_tokens_details
                        .and_then(|details| details.reasoning_tokens),
                    tool_input_tokens: None,
                },
            });
        }
        // Every chunk names the model; only the chunk that carries the finish
        // reason is a verdict, so the model rides with it rather than on each
        // delta. The id still lands on the terminal verdict or the closure.
        let choice = payload.choices.into_iter().next().unwrap_or_default();
        let verdict = match choice.finish_reason {
            Some(reason) => AdapterVerdict {
                finish_reason: Some(sink.scrub(&reason)),
                block_reason: None,
                detail: None,
                model: payload.model.map(|value| sink.scrub(&value)),
            },
            None => AdapterVerdict::default(),
        };
        let response_id = payload.id.map(|value| sink.scrub(&value));
        sink.provider(verdict, response_id);
        if let Some(error) = payload.error {
            error.emit(sink);
        }
    }
}

/// One object for the unary reply and each stream chunk `project` above
/// sees. Every field is optional: a chunk carries a delta, the last chunk
/// (or the reply) carries the usage, and `[DONE]` is not JSON at all.
#[derive(Deserialize)]
struct ObservedPayload {
    id: Option<String>,
    model: Option<String>,
    usage: Option<ObservedUsage>,
    #[serde(default)]
    choices: Vec<ObservedChoice>,
    error: Option<ObservedError>,
}

#[derive(Deserialize)]
struct ObservedUsage {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    prompt_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    completion_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    total_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<ObservedTokenDetails>,
    #[serde(default)]
    completion_tokens_details: Option<ObservedTokenDetails>,
}

#[derive(Default, Deserialize)]
struct ObservedTokenDetails {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    cached_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    reasoning_tokens: Option<u64>,
}

#[derive(Default, Deserialize)]
struct ObservedChoice {
    finish_reason: Option<String>,
}

/// A gateway's encrypted-reasoning detail as a whole reasoning block.
///
/// Encrypted reasoning (`{"type":"reasoning.encrypted"}`) is the turn's own
/// output, not tool-call metadata: it arrives with `reasoning: null` and an
/// `rs_*` id of its own, which never matches a `call_*` tool-call id, and it
/// arrives before any tool call opens. Writing it as a reasoning part is
/// what lets the blob reach the aggregated choice and be replayed next turn.
fn detail_reasoning(detail: &unary::ReasoningDetails) -> Option<crate::message::Reasoning> {
    let unary::ReasoningDetails::Encrypted { id, data, .. } = detail else {
        return None;
    };
    Some(crate::message::Reasoning {
        id: id.clone().filter(|id| !id.is_empty()),
        content: vec![crate::message::ReasoningContent::Encrypted(data.clone())],
    })
}

/// A nonempty unary reasoning detail as one replayable part. Empty and
/// signature-only entries return `None`.
fn whole_detail_reasoning(detail: &unary::ReasoningDetails) -> Option<crate::message::Reasoning> {
    let (id, content) = match detail {
        unary::ReasoningDetails::Summary { id, summary, .. } if !summary.is_empty() => (
            id,
            crate::message::ReasoningContent::Summary(summary.clone()),
        ),
        unary::ReasoningDetails::Encrypted { id, data, .. } if !data.is_empty() => (
            id,
            crate::message::ReasoningContent::Encrypted(data.clone()),
        ),
        unary::ReasoningDetails::Text {
            id,
            text: Some(text),
            signature,
            ..
        } if !text.is_empty() => (
            id,
            crate::message::ReasoningContent::Text {
                text: text.clone(),
                signature: signature.clone().filter(|signature| !signature.is_empty()),
            },
        ),
        // Signature-only details attach to separately supplied plaintext.
        _ => return None,
    };
    Some(crate::message::Reasoning {
        id: id.clone().filter(|id| !id.is_empty()),
        content: vec![content],
    })
}

/// Return a nonempty signature from a text reasoning detail.
fn reasoning_signature(detail: &unary::ReasoningDetails) -> Option<String> {
    let unary::ReasoningDetails::Text {
        signature: Some(signature),
        ..
    } = detail
    else {
        return None;
    };
    (!signature.is_empty()).then(|| signature.clone())
}

/// Decode a modeled reasoning detail, returning `None` for unrecognized or invalid shapes.
fn typed_detail(detail: &serde_json::Value) -> Option<unary::ReasoningDetails> {
    serde_json::from_value(detail.clone()).ok()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod hard_case_tests;
