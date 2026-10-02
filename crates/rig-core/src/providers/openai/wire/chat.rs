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
use crate::operation::{Block, CallFragment, Completion, IfMalformed};
use crate::providers::internal::openai_chat_completions_compatible::{
    map_native_finish_reason, map_openai_finish_reason, provider_error_envelope,
};
use crate::providers::internal::wire::classify_chat_completions_frame;
use crate::providers::openai::completion::{
    self as unary, Message, ToolChoice, is_openai_reasoning_model, request_body,
};
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Body, Capabilities, Decoder, Descriptor, Encoded,
    Flow, Framing, Mode, ObservationSink, Out, Wire, WireEvent, WireFrame,
};

use super::dto::{
    ChatChoice, ChatFrame, ChatUsage, StreamingCompletionResponse, StreamingToolCall, delta_text,
    merge_fields,
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
            BodyRewrite::GroqCompoundTools => fold_groq_native_tools(request)?,
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
            BodyRewrite::GroqCompoundTools => finalize_groq(map),
            BodyRewrite::DeepSeek => finalize_deepseek(map),
            BodyRewrite::Mistral => finalize_mistral(map)?,
            BodyRewrite::OpenRouter => finalize_openrouter(map, self.prompt_caching),
            BodyRewrite::None
            | BodyRewrite::HuggingFaceRouter
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

/// Groq's streamed gpt-oss messages carry a `channel`, which it answers with
/// a 400 when the message comes back ("'messages.2' : for 'role:assistant'
/// the following must be satisfied[('messages.2' : property 'channel' is
/// unsupported)]", recorded 2026-10-01), so the field is left out.
fn finalize_groq(map: &mut serde_json::Map<String, serde_json::Value>) {
    for message in map
        .get_mut("messages")
        .and_then(as_array_mut)
        .into_iter()
        .flatten()
    {
        if let Some(message) = message.as_object_mut()
            && message.get("role").and_then(serde_json::Value::as_str) == Some("assistant")
        {
            message.remove("channel");
        }
    }
}

/// DeepSeek takes message `content` as a plain string, echoes tool calls back
/// with an `index`, and needs an explicit empty `content` on a tool-call-only
/// assistant turn. Its thinking models take `reasoning_content` back on every
/// assistant message, so a turn that carries none sends it empty (pi's rule).
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
            if is_assistant {
                message
                    .entry("reasoning_content")
                    .or_insert_with(|| serde_json::Value::String(String::new()));
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
/// its content chunks, and `content` on every assistant message.
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
        // An assistant message is Mistral's own reply or one rebuilt with
        // string content; only its missing `content` needs filling in.
        if message.get("role").and_then(serde_json::Value::as_str) == Some("assistant") {
            message
                .entry("content")
                .or_insert_with(|| serde_json::Value::String(String::new()));
            continue;
        }
        // Mistral takes text-only message `content` as a plain string and
        // carries images, audio and documents as its own chunk array.
        // Content it has no chunk for fails here rather than reaching the API
        // with the part removed.
        if let Some(content) = message.get_mut("content") {
            mistral_content(content)?;
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
    type Decoder<'id> = ChatDecoder;

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
            .replay(self)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        self.encode_with_headers(request, mode, OpenAIConfig::completion_headers)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ChatDecoder::new(self.provider.dialect.quirks)
    }
}

impl crate::completion::ReplayTarget for Chat {
    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("openai.chat")
    }

    fn provider(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// pi's rule for this wire: an id keeps the characters Chat Completions
    /// accepts and at most 40 of them, a longer one ending in a hash of the
    /// whole id. Mistral takes exactly nine alphanumerics.
    fn normalize_tool_call_id(&self, id: &str, _: Option<&crate::message::Origin>) -> String {
        if self.provider.dialect.quirks.rewrite == BodyRewrite::Mistral {
            return mistral_call_id(id);
        }
        let sanitized: String = id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if sanitized.len() <= 40 {
            return sanitized;
        }
        let hash: String = short_hash(id).chars().take(8).collect();
        let prefix: String = sanitized.chars().take(40 - hash.len() - 1).collect();
        format!("{prefix}_{hash}")
    }
}

/// pi's derivation of a Mistral call id: the id's alphanumerics when there
/// are exactly nine, otherwise nine characters of their hash.
fn mistral_call_id(id: &str) -> String {
    const LENGTH: usize = 9;
    let normalized: String = id.chars().filter(char::is_ascii_alphanumeric).collect();
    if normalized.len() == LENGTH {
        return normalized;
    }
    let seed = if normalized.is_empty() {
        id
    } else {
        &normalized
    };
    short_hash(seed)
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(LENGTH)
        .collect()
}

/// pi's `shortHash`: two 32-bit multiplicative hashes of the UTF-16 code
/// units, written in base 36.
fn short_hash(text: &str) -> String {
    fn base36(mut value: u32) -> String {
        let mut digits = Vec::new();
        loop {
            digits.push(char::from_digit(value % 36, 36).unwrap_or('0'));
            value /= 36;
            if value == 0 {
                break;
            }
        }
        digits.iter().rev().collect()
    }
    let (mut h1, mut h2) = (0xdead_beef_u32, 0x41c6_ce57_u32);
    for unit in text.encode_utf16().map(u32::from) {
        h1 = (h1 ^ unit).wrapping_mul(2_654_435_761);
        h2 = (h2 ^ unit).wrapping_mul(1_597_334_677);
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h2 ^ (h2 >> 13)).wrapping_mul(3_266_489_909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h1 ^ (h1 >> 13)).wrapping_mul(3_266_489_909);
    format!("{}{}", base36(h2), base36(h1))
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

/// The keys an assistant message carries reasoning under, in the order
/// their text is read: compatible servers that send several send the same
/// text under each.
const REASONING_TEXT_KEYS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];

/// The structured reasoning key, kept verbatim with the reasoning text.
const REASONING_DETAILS: &str = "reasoning_details";

/// The chat-completions decoder: one state machine for a whole reply and a
/// stream of chunks. A whole reply is restated as the one chunk carrying
/// its message. The assistant message is assembled from the deltas and
/// becomes the turn's native; its reasoning, its text and each tool call
/// are the turn's blocks, each holding its own fields of the message.
pub struct ChatDecoder {
    quirks: super::Quirks,
    /// The assistant message as assembled so far, without its tool calls.
    message: serde_json::Map<String, serde_json::Value>,
    /// Each tool call as assembled so far, in the order they opened; `None`
    /// once dropped.
    calls: Vec<Option<serde_json::Value>>,
    /// The position in `calls` of the call open at each wire index.
    open_calls: std::collections::BTreeMap<usize, usize>,
    /// The writer index of the reply's text.
    text: Option<usize>,
    /// The writer index of the reply's reasoning.
    reasoning: Option<usize>,
    final_usage: Option<ChatUsage>,
    final_finish_reason: Option<FinishReason>,
    response_id: Option<String>,
    response_model: Option<String>,
    /// Accumulated primary-choice token metadata, in the wire's token order.
    logprobs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Accumulated provider-specific top-level chunk metadata.
    additional_params: serde_json::Map<String, serde_json::Value>,
    /// Whether a finish reason established the turn complete.
    saw_terminal: bool,
    /// Whether any frame decoded successfully. A bare `[DONE]` after only
    /// parse failures must not dress the failure up as a default-usage
    /// success.
    saw_any_valid_frame: bool,
}

impl ChatDecoder {
    fn new(quirks: super::Quirks) -> Self {
        Self {
            quirks,
            message: serde_json::Map::new(),
            calls: Vec::new(),
            open_calls: std::collections::BTreeMap::new(),
            text: None,
            reasoning: None,
            final_usage: None,
            final_finish_reason: None,
            response_id: None,
            response_model: None,
            logprobs: None,
            additional_params: serde_json::Map::new(),
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

    /// Absorb the metadata every frame carries, and take its primary choice.
    fn absorb(&mut self, frame: ChatFrame) -> Option<ChatChoice> {
        self.saw_any_valid_frame = true;
        let ChatFrame {
            id,
            model,
            choices,
            usage,
            additional_params,
        } = frame;
        self.response_id = id.or(self.response_id.take());
        self.response_model = model.or(self.response_model.take());
        self.final_usage = usage.or(self.final_usage.take());
        merge_fields(&mut self.additional_params, &additional_params);
        // `n > 1` streams interleave candidates told apart only by
        // `choices[].index`; candidate 0 is the turn, as in a whole reply.
        let choice = choices
            .into_iter()
            .find(|choice| choice.index.is_none_or(|index| index == 0))?;
        if let Some(reason) = self.finish_reason(&choice) {
            self.final_finish_reason = Some(reason);
            self.saw_terminal = true;
        }
        if let Some(serde_json::Value::Object(logprobs)) = &choice.logprobs {
            merge_fields(self.logprobs.get_or_insert_default(), logprobs);
        }
        Some(choice)
    }

    /// One delta of the assistant message: its reasoning, its text and its
    /// tool calls reach their blocks, and every field reaches the message.
    fn delta(
        &mut self,
        mut delta: serde_json::Map<String, serde_json::Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let reasoning = REASONING_TEXT_KEYS.iter().find_map(|key| {
            delta
                .get(*key)
                .and_then(serde_json::Value::as_str)
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        });
        let details = match delta.remove(REASONING_DETAILS) {
            Some(serde_json::Value::Array(details)) => details,
            _ => Vec::new(),
        };
        if reasoning.is_some() || !details.is_empty() {
            let index = open_once(
                &mut self.reasoning,
                Block::Reasoning { redacted: false },
                out,
            )?;
            out.push(index, reasoning.as_deref().unwrap_or_default())?;
        }
        merge_details(&mut self.message, details);
        if let Some(text) = delta_text(&delta) {
            let index = open_once(&mut self.text, Block::Text, out)?;
            out.push(index, &text)?;
        }
        if let Some(serde_json::Value::Array(calls)) = delta.remove("tool_calls") {
            for call in calls {
                self.call(call, out)?;
            }
        }
        delta.remove("tool_calls");
        merge_fields(&mut self.message, &delta);
        Ok(())
    }

    /// One tool-call fragment, buffered at its wire index.
    fn call(
        &mut self,
        mut call: serde_json::Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let incoming = serde_json::from_value::<StreamingToolCall>(call.clone())
            .map_err(|error| ProviderError::Response(format!("malformed tool call: {error}")))?;
        let index = incoming.index;
        if let Some(existing) = self.open_call(index) {
            let id = existing.get("id").and_then(serde_json::Value::as_str);
            let name = existing
                .pointer("/function/name")
                .and_then(serde_json::Value::as_str);
            if incoming.evicts(id.unwrap_or_default(), name.unwrap_or_default()) {
                // The wire reused this call's index: the call it held is
                // delivered even when its arguments never parse.
                self.close_call(index, IfMalformed::EmptyObject, out)?;
            }
        }
        let at = *self.open_calls.entry(index).or_insert_with(|| {
            self.calls
                .push(Some(serde_json::Value::Object(serde_json::Map::new())));
            self.calls.len() - 1
        });
        // The index orders the stream; the call it assembles has none, as
        // in a whole reply.
        if let (
            Some(serde_json::Value::Object(fields)),
            Some(Some(serde_json::Value::Object(existing))),
        ) = (
            call.as_object_mut().map(|fields| {
                fields.remove("index");
                serde_json::Value::Object(std::mem::take(fields))
            }),
            self.calls.get_mut(at),
        ) {
            merge_fields(existing, &fields);
        }
        out.fragment(
            index,
            CallFragment {
                id: incoming.id.as_deref(),
                name: incoming.function.name.as_deref(),
                arguments: incoming.function.arguments.as_deref(),
            },
        )?;
        if self.quirks.emits_complete_single_chunk_tool_calls && incoming.is_complete_single_chunk()
        {
            // A probe: the call closes if its input parses, and stays open
            // for more fragments otherwise.
            self.close_call(index, IfMalformed::KeepOpen, out)?;
        }
        Ok(())
    }

    /// The call assembled at wire `index`, while it is open.
    fn open_call(&self, index: usize) -> Option<&serde_json::Value> {
        self.calls.get(*self.open_calls.get(&index)?)?.as_ref()
    }

    /// Close the call at wire `index` with the call as its native.
    fn close_call(
        &mut self,
        index: usize,
        if_malformed: IfMalformed,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if let Some(call) = self.open_call(index).cloned() {
            out.edit(index, |item| *item = call)?;
        }
        out.close(index, if_malformed)?;
        if !out.is_open(index) {
            self.open_calls.remove(&index);
        }
        Ok(())
    }

    /// Drop the call at wire `index`: it never reaches the turn.
    fn drop_call(&mut self, index: usize, out: &mut Out<'_, Completion>) {
        out.discard(index);
        if let Some(at) = self.open_calls.remove(&index)
            && let Some(slot) = self.calls.get_mut(at)
        {
            *slot = None;
        }
    }

    /// Close every open call. A call the output-token budget cut short is
    /// dropped; any other call whose arguments do not parse fails the reply,
    /// and so does one whose arguments the provider said were complete.
    fn close_calls(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let cut_short = matches!(self.final_finish_reason, Some(FinishReason::Length));
        let open: Vec<usize> = self.open_calls.keys().copied().collect();
        for index in open {
            let arguments = self
                .open_call(index)
                .and_then(|call| call.pointer("/function/arguments"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let named = self
                .open_call(index)
                .and_then(|call| call.pointer("/function/name"))
                .and_then(serde_json::Value::as_str)
                .is_some_and(|name| !name.is_empty());
            // A cut before the first argument token leaves no call at all.
            if !named
                || (cut_short
                    && (arguments.trim().is_empty()
                        || crate::json_utils::parse_tool_arguments(arguments).is_err()))
            {
                tracing::debug!("dropping a streamed tool call cut off before it completed");
                self.drop_call(index, out);
                continue;
            }
            self.close_call(index, IfMalformed::Fail, out)?;
        }
        Ok(())
    }

    /// One `chat.completion.chunk`.
    fn interpret_chunk(
        &mut self,
        frame: ChatFrame,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let Some(choice) = self.absorb(frame) else {
            return Ok(());
        };
        self.delta(choice.delta, out)?;
        if matches!(self.final_finish_reason, Some(FinishReason::ToolCalls)) {
            // Completed calls with malformed arguments must fail, not
            // disappear. Empty arguments remain valid for zero-argument
            // tools.
            self.close_calls(out)?;
        }
        Ok(())
    }

    /// The `chat.completion` body, restated as the one chunk whose delta is
    /// its whole message, then the end.
    fn interpret_whole(
        &mut self,
        frame: ChatFrame,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let Some(choice) = self.absorb(frame) else {
            return Err(ProviderError::Response(
                "Response contained no choices".to_owned(),
            ));
        };
        let Some(mut message) = choice.message else {
            return Err(ProviderError::Response(
                "Response did not contain a valid message or tool call".to_owned(),
            ));
        };
        self.saw_terminal = true;
        // Truncation or filtering can leave no visible content; retain its
        // reason and usage. Empty replies without such a reason are errors.
        let cut_short = self
            .final_finish_reason
            .as_ref()
            .is_some_and(FinishReason::truncated_output);
        let has_calls = message
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|calls| !calls.is_empty());
        let has_reasoning = REASONING_TEXT_KEYS.iter().any(|key| {
            message
                .get(*key)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|text| !text.is_empty())
        }) || message
            .get(REASONING_DETAILS)
            .and_then(serde_json::Value::as_array)
            .is_some_and(|details| !details.is_empty());
        if delta_text(&message).is_none() && !has_calls && !has_reasoning && !cut_short {
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        // Each call is buffered at its own index, so separate id-less calls
        // stay distinct.
        if let Some(serde_json::Value::Array(calls)) = message.get_mut("tool_calls") {
            for (index, call) in calls.iter_mut().enumerate() {
                if let Some(call) = call.as_object_mut() {
                    call.entry("index").or_insert(index.into());
                }
            }
        }
        self.delta(message, &mut out)?;
        self.close_calls(&mut out)?;
        self.end(out, false)
    }

    /// Write the provider's end of the reply: the reasoning and text blocks
    /// close holding their fields of the message, which becomes the turn's
    /// native. A stream's `raw` is the native terminal record the chunks
    /// built; a whole body's is the body itself, which the transport keeps.
    fn end(&mut self, mut out: Out<'_, Completion>, streamed: bool) -> Result<Flow, ProviderError> {
        let mut message = std::mem::take(&mut self.message);
        for (index, reasoning) in [(self.reasoning.take(), true), (self.text.take(), false)] {
            let Some(index) = index else {
                continue;
            };
            let fields: serde_json::Map<_, _> = message
                .iter()
                .filter(|(key, _)| {
                    let key = key.as_str();
                    key != "role"
                        && (REASONING_TEXT_KEYS.contains(&key) || key == REASONING_DETAILS)
                            == reasoning
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            out.edit(index, |item| *item = serde_json::Value::Object(fields))?;
            out.close(index, IfMalformed::Fail)?;
        }
        let calls: Vec<_> = self.calls.drain(..).flatten().collect();
        if !calls.is_empty() {
            message.insert("tool_calls".to_owned(), serde_json::Value::Array(calls));
        }
        out.message_native(serde_json::Value::Object(message));
        let usage = self
            .final_usage
            .as_ref()
            .map(|usage| usage.to_normalized_for(&self.quirks));
        let native = StreamingCompletionResponse {
            usage: self.final_usage.take(),
            finish_reason: self.final_finish_reason.take(),
            response_id: self.response_id.take(),
            model: self.response_model.take(),
            logprobs: self.logprobs.take().map(serde_json::Value::Object),
            additional_params: Some(std::mem::take(&mut self.additional_params))
                .filter(|params| !params.is_empty()),
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
    fn finish(&mut self, mut out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
        self.close_calls(&mut out)?;
        // A bare terminator without valid content cannot establish success.
        if !self.saw_any_valid_frame {
            return Err(ProviderError::Truncated);
        }
        self.end(out, true)
    }
}

/// Open the block `slot` names, once: its writer index.
fn open_once(
    slot: &mut Option<usize>,
    block: Block,
    out: &mut Out<'_, Completion>,
) -> Result<usize, ProviderError> {
    if let Some(index) = *slot {
        return Ok(index);
    }
    let index = out.fresh_index();
    out.open(index, block, serde_json::Value::Null)?;
    *slot = Some(index);
    Ok(index)
}

/// Append streamed reasoning details to the message's, by pi's merge: a
/// text or summary fragment continues the last entry of its type and
/// index, appending its text and filling the fields that entry lacks, and an
/// encrypted entry stays whole.
fn merge_details(
    message: &mut serde_json::Map<String, serde_json::Value>,
    details: Vec<serde_json::Value>,
) {
    if details.is_empty() {
        return;
    }
    let serde_json::Value::Array(merged) = message
        .entry(REASONING_DETAILS)
        .or_insert_with(|| serde_json::Value::Array(Vec::new()))
    else {
        return;
    };
    for detail in details {
        let kind = detail.get("type").and_then(serde_json::Value::as_str);
        let continues = matches!(kind, Some("reasoning.text" | "reasoning.summary"))
            && merged.last().is_some_and(|last| {
                last.get("type").and_then(serde_json::Value::as_str) == kind
                    && last.get("index") == detail.get("index")
            });
        match (continues, merged.last_mut(), detail) {
            (true, Some(serde_json::Value::Object(last)), serde_json::Value::Object(fields)) => {
                for (key, value) in fields {
                    let missing = last
                        .get(&key)
                        .is_none_or(|existing| existing.is_null() || existing.as_str() == Some(""));
                    match (last.get_mut(&key), value) {
                        (
                            Some(serde_json::Value::String(text)),
                            serde_json::Value::String(more),
                        ) if matches!(key.as_str(), "text" | "summary") => {
                            text.push_str(&more);
                        }
                        (_, value) if missing => {
                            last.insert(key, value);
                        }
                        _ => {}
                    }
                }
            }
            (_, _, detail) => merged.push(detail),
        }
    }
}

#[deny(clippy::wildcard_enum_match_arm)]
impl<'id> Decoder<'id, Completion> for ChatDecoder {
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
        classify_chat_completions_frame::<ChatFrame>(&data).map(|frame| {
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
            // `[DONE]` without a finish reason still ends the turn.
            ChatEvent::Done => {
                self.saw_terminal = true;
                self.finish(out)
            }
            ChatEvent::BareText(text) => {
                self.saw_any_valid_frame = true;
                self.saw_terminal = true;
                let mut message = serde_json::Map::new();
                message.insert("role".to_owned(), "assistant".into());
                message.insert("content".to_owned(), text.into());
                self.delta(message, &mut out)?;
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

impl ChatDecoder {
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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod hard_case_tests;
