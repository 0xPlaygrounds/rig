//! The one chat-completions wire.
//!
//! [`Chat`] encodes both replies — a unary `chat.completion` body and an SSE
//! stream of `chat.completion.chunk` frames — and [`ChatDecoder`] folds
//! both, because the unary body is one more [`WireEvent`] variant whose
//! `interpret` *synthesizes the stream's events*. There is no second content
//! mapping, so the two paths cannot disagree about a turn.

use serde::{Deserialize, Serialize};

use crate::completion::{CompletionError, CompletionRequest, FinishReason, ProviderCapabilities};
use crate::providers::internal::chunk_lifecycle::{ChunkParts, MintedReasoningLifecycle};
use crate::providers::internal::openai_chat_completions_compatible::{
    CompatibleFinishReason, CompatibleTerminal, CompatibleToolCallChunk, map_native_finish_reason,
    map_openai_finish_reason, provider_error_envelope, should_evict_distinct_named_tool_call,
};
use crate::providers::internal::tool_call_bridge::ToolCallBridge;
use crate::providers::internal::wire::classify_chat_completions_frame;
use crate::providers::openai::completion::{
    self as unary, AssistantContent, Message, ToolChoice, assistant_refusal_fallback,
    is_openai_reasoning_model, request_body,
};
use crate::streaming::{BlockId, Delta, MintKind, StreamEvent, ToolCallEnd, UnparseableToolInput};
use crate::wire::{
    AdapterErrorEnvelope, AdapterEvent, AdapterUsage, AdapterVerdict, Body, Decoder, Encoded,
    Framing, Mode, ObservationSink, Output, Wire, WireEvent, WireFrame,
};

use super::dto::{ChatFrame, ChatUsage, StreamingCompletionResponse, delta_text};
use super::{BodyRewrite, OpenAI, OutputCap};

/// The chat-completions wire: a provider configuration, a model, and the
/// per-turn options the endpoint takes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chat {
    /// Which provider, and how to reach it.
    pub provider: OpenAI,
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
    /// The wire for `model` on `provider`, with every option off.
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
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

    /// Adjust the typed request before serialization.
    ///
    /// One `match` over the dialect's rewrite, in one place — the
    /// `prepare_request` overrides that used to be one per provider.
    fn prepare(&self, request: &mut unary::CompletionRequest) -> Result<(), CompletionError> {
        match self.provider.dialect.quirks.rewrite {
            BodyRewrite::GroqCompoundTools => fold_groq_native_tools(request)?,
            BodyRewrite::LlamaCpp => {
                if let Some(ToolChoice::Function { name }) = &request.tool_choice {
                    return Err(CompletionError::ProviderError(format!(
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

    /// Adjust the serialized body immediately before it is sent — after the
    /// streaming parameters are merged, so a rewrite sees them.
    fn finalize(&self, body: &mut serde_json::Value) -> Result<(), CompletionError> {
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

fn as_array_mut(value: &mut serde_json::Value) -> Option<&mut Vec<serde_json::Value>> {
    value.as_array_mut()
}

/// The raw tool-call array of a unary choice's assistant message.
///
/// Reached on the body rather than on a decoded frame because the policy
/// below turns on the *verbatim* `arguments` string, which the typed decode
/// has already normalized away.
fn message_tool_calls_mut(choice: &mut serde_json::Value) -> Option<&mut Vec<serde_json::Value>> {
    choice
        .get_mut("message")
        .and_then(|message| message.get_mut("tool_calls"))
        .and_then(as_array_mut)
}

/// Whether a raw tool call's `arguments` string is unusable as tool input.
///
/// Empty counts as unusable alongside unparseable: `parse_tool_arguments`
/// maps an empty string onto `{}` so a genuine zero-argument tool works, and
/// a call cut *before* its first argument token is exactly what that
/// normalization would disguise as a zero-argument call. `"{}"` itself is
/// neither empty nor unparseable and stays.
///
/// Arguments the dialect sent as a raw JSON value rather than a string
/// (llama.cpp and Hugging Face both do) are never unusable: there is no
/// half-written string to fail on.
fn arguments_are_unusable(call: &serde_json::Value) -> bool {
    call.get("function")
        .and_then(|function| function.get("arguments"))
        .and_then(serde_json::Value::as_str)
        .is_some_and(|raw| {
            raw.trim().is_empty() || crate::json_utils::parse_tool_arguments(raw).is_err()
        })
}

/// Stub every unusable `arguments` string to `{}`, reporting whether any
/// call needed it. The input to the compound-defect probe below.
fn stub_unusable_arguments(choice: &mut serde_json::Value) -> bool {
    let Some(calls) = message_tool_calls_mut(choice) else {
        return false;
    };
    let mut stubbed = false;
    for call in calls {
        if !arguments_are_unusable(call) {
            continue;
        }
        let Some(arguments) = call
            .get_mut("function")
            .and_then(|function| function.get_mut("arguments"))
        else {
            continue;
        };
        *arguments = serde_json::Value::String("{}".to_owned());
        stubbed = true;
    }
    stubbed
}

/// Remove every call with unusable arguments, reporting how many went.
fn drop_unusable_calls(choice: &mut serde_json::Value) -> usize {
    let Some(calls) = message_tool_calls_mut(choice) else {
        return 0;
    };
    let before = calls.len();
    calls.retain(|call| !arguments_are_unusable(call));
    before - calls.len()
}

/// Groq's compound-system native tools (`browser_search`, `code_interpreter`,
/// …) arrive through `additional_params.tools`. Left there they would clobber
/// the function-tool array on serialization, because `additional_params` is
/// flattened into the body and the flattened key wins.
fn fold_groq_native_tools(request: &mut unary::CompletionRequest) -> Result<(), CompletionError> {
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
    // Taken as an array directly rather than through serde: this is the
    // *request* being shaped, and the only thing to learn about the value is
    // whether the caller gave an array at all.
    let serde_json::Value::Array(native_tools) = raw_tools else {
        return Err(CompletionError::RequestError(
            "Groq `additional_params.tools` must be an array of native tool objects".into(),
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
fn steer_moonshot_tool_choice(
    request: &mut unary::CompletionRequest,
) -> Result<(), CompletionError> {
    if matches!(request.tool_choice, Some(ToolChoice::Function { .. })) {
        return Err(CompletionError::ProviderError(
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
                // Text-only arrays flatten; an array carrying an image,
                // audio, video or file part is left alone so DeepSeek's own
                // rejection reaches the caller ("unknown variant
                // `image_url`, expected `text`", verified live). Dropping
                // those parts here answered the question from the text alone
                // and never told anyone the attachment was gone.
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
) -> Result<(), CompletionError> {
    // Mistral spells the "must call some tool" mode `any`, not `required`.
    if let Some(tool_choice) = map.get_mut("tool_choice")
        && tool_choice.as_str() == Some("required")
    {
        *tool_choice = serde_json::Value::String("any".to_owned());
    }

    // Mistral accepts a *structured* response format beside tools only under
    // `tool_choice: auto` (or `none`): anything that forces a call is a 400,
    // "`json_schema` response type with tools is only compatible with
    // `tool_choice: auto`". Rig reaches that combination on its own — a
    // structured-output agent defers `response_format` until a tool result
    // exists, then emits it beside the caller's standing `tool_choice`, so
    // the turn after the first tool call dies. Relaxing the choice keeps
    // both features working; dropping the response format instead would
    // silently discard the schema the caller asked for.
    //
    // Keyed on the format's *type* rather than its presence: the constraint
    // is specific to `json_schema` and `json_object`, and `{"type": "text"}`
    // rides beside a forced choice happily.
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

// ── Mistral's message content schema ────────────────────────────────────
//
// Mistral validates message content as a tagged union, not as OpenAI's
// content parts, so a part has to be rebuilt as the chunk its schema names.
// The five below are the kinds the shared OpenAI-compatible conversion can
// produce.

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

/// Whether a part is purely textual, and so belongs in the plain-string form.
///
/// Decided on the `type` tag first, and only on the keys for a part carrying
/// no tag. Deciding on the keys alone would let a part that names a chunk
/// kind *and* happens to carry a `text` key be flattened away, which is the
/// silent drop this whole path exists to prevent.
fn is_mistral_text_part(part: &serde_json::Value) -> bool {
    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(MISTRAL_TEXT | MISTRAL_REFUSAL) => true,
        Some(_) => false,
        None => mistral_part_text(part).is_some(),
    }
}

fn mistral_unsupported(what: &str) -> CompletionError {
    crate::message::MessageError::ConversionError(format!(
        "Mistral cannot carry {what}. Mistral messages accept text, `{MISTRAL_IMAGE}`, \
         `{MISTRAL_AUDIO}`, `{MISTRAL_DOCUMENT}` and `{MISTRAL_FILE}` content; convert the \
         content to one of those before sending it."
    ))
    .into()
}

/// OpenAI's `{"type": "file", "file": {…}}` as the Mistral chunk carrying the
/// same document.
///
/// Inline bytes become `document_url`, which reads the base64 `data:` URI the
/// shared conversion already built for `file_data` and carries the filename
/// in its own optional `document_name`. An uploaded-file reference becomes
/// Mistral's `file` chunk, which names the id at the top level rather than
/// nesting it under `file` as OpenAI does — sending OpenAI's nesting is
/// rejected twice over, for a missing `file_id` and for a forbidden extra
/// `file`, since every Mistral chunk forbids unknown fields.
fn mistral_file_chunk(part: &serde_json::Value) -> Result<serde_json::Value, CompletionError> {
    let file = part.get(MISTRAL_FILE);
    let field = |name: &str| {
        file.and_then(|file| file.get(name))
            .and_then(serde_json::Value::as_str)
    };

    // Already a Mistral file chunk (`file_id` at the top level, as this
    // emits): pass it through, so finalizing an already-finalized body is a
    // no-op rather than an error about content rig itself built.
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

/// An `input_audio` part as Mistral's canonical audio chunk, whose payload is
/// the base64 string itself.
///
/// Mistral currently also accepts the `{data, format}` object the shared
/// conversion produces — its schema flattens the object and discards
/// `format` — but the bare string is the form its published schema
/// documents. Nothing is lost: a deliberately wrong `format` changes no
/// result, and a `format` placed as a *sibling* of `input_audio` is rejected
/// outright.
fn mistral_audio_chunk(part: &serde_json::Value) -> Result<serde_json::Value, CompletionError> {
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
fn mistral_chunk(part: &serde_json::Value) -> Result<serde_json::Value, CompletionError> {
    /// Text and refusal parts are both re-tagged `text`: Mistral's schema has
    /// no `refusal` field and every chunk forbids unknown keys.
    fn text_chunk(part: &serde_json::Value) -> Result<serde_json::Value, CompletionError> {
        let text = mistral_part_text(part)
            .ok_or_else(|| mistral_unsupported("a text content part carrying no text"))?;
        Ok(serde_json::json!({"type": MISTRAL_TEXT, MISTRAL_TEXT: text}))
    }

    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(MISTRAL_TEXT | MISTRAL_REFUSAL) => text_chunk(part),
        // The payload needs no reshaping — Mistral's image chunk takes the
        // `{url, detail}` object rig sends as readily as a bare URL string,
        // and reads a base64 `data:` URI in either. It is still rebuilt
        // rather than forwarded, because every Mistral chunk forbids unknown
        // fields: a stray sibling key riding on the part would 422 the whole
        // request.
        Some(MISTRAL_IMAGE) => {
            let image = part.get(MISTRAL_IMAGE).ok_or_else(|| {
                mistral_unsupported("an image content part carrying no `image_url` payload")
            })?;
            Ok(serde_json::json!({"type": MISTRAL_IMAGE, MISTRAL_IMAGE: image}))
        }
        Some(MISTRAL_AUDIO) => mistral_audio_chunk(part),
        Some(MISTRAL_FILE) => mistral_file_chunk(part),
        // Already a Mistral document chunk — see `mistral_file_chunk` on why
        // an already-converted part passes through.
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

/// Rewrite one serialized message `content` into Mistral's content schema.
///
/// Mistral accepts content as either a plain string or an array of typed
/// chunks. Text-only content keeps the plain-string form it has always taken.
/// Content carrying anything else keeps the array, with each part rendered
/// the way Mistral's schema names it, instead of being flattened away: a
/// text-only flattening keeps only parts with a `text`/`refusal` key, so an
/// attached image, document or audio clip was dropped from the request and
/// the caller got an ordinary completion answering a prompt it never sent
/// (rig#2290).
///
/// Content Mistral has no chunk for — video, and any part type a future
/// conversion adds — fails here rather than being silently removed. The one
/// exception is content whose parts are *all* tagged `text`/`refusal`: that
/// takes the flattening path, which drops a part carrying no string payload
/// exactly as it always has, rather than inventing a new failure for a shape
/// rig's own conversion cannot produce.
fn mistral_content(content: &mut serde_json::Value) -> Result<(), CompletionError> {
    let Some(parts) = content.as_array() else {
        return Ok(());
    };

    if parts.iter().all(is_mistral_text_part) {
        // Flattened unconditionally rather than under `only_if_all_text`, so
        // the helper does not re-decide: it judges per key while the guard
        // above judges on the type tag, and the two disagree for a malformed
        // part such as `{"type": "text"}` carrying no `text`. Letting the
        // helper decline would leave that content as an array of chunks
        // Mistral cannot read.
        unary::flatten_text_content_parts(content, "", false);
        return Ok(());
    }

    // Re-borrowed rather than held across the branch above, which needs
    // `content` itself. The array-ness was just established, so the `else` is
    // unreachable — expressed as a no-op instead of an unwrap.
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

        // OpenRouter's image part is `{"image_url": {"url": …}}` and nothing
        // else. The shared part carries OpenAI's `detail` hint, which rig
        // defaults to `"auto"` — i.e. "no preference" — so sending it states
        // a fidelity choice the caller never made to a gateway whose own
        // conversion never had the field. The recorded requests carry no
        // `detail`.
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
            // Mark the last block as the cache boundary; every other block —
            // images included — is preserved unchanged.
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

/// Refuse a document or file part that carries only a provider file id.
///
/// The message is the one `openrouter::completion`'s conversion returned, so
/// a caller who hit this before hits the same wording now. Checked on the
/// normalized request rather than during conversion because that is where the
/// dialect is known.
fn refuse_file_ids(request: &CompletionRequest) -> Result<(), CompletionError> {
    use crate::message::{DocumentSourceKind, Message, UserContent};

    let refusal = || {
        CompletionError::RequestError(
            "Provider file IDs are not supported for OpenRouter document inputs".into(),
        )
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
                // An OpenAI `file` part converts into a rig document, and
                // that conversion prefers `file_data` over `file_id` — so a
                // part reaching here with a bare id genuinely carries only
                // the id.
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
    type Decoder = ChatDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn route(&self) -> Option<&str> {
        Some(self.provider.dialect.quirks.completion_path)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, CompletionError> {
        let quirks = &self.provider.dialect.quirks;
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
                // Shallow, so `include_usage` is inserted *into* any
                // caller-supplied `stream_options` rather than merged over
                // it: the caller's keys survive and the usage chunk is still
                // requested.
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

        // Deliberately the configured model, not the per-request override:
        // Azure's deployment URL is pinned to the model handle.
        let uri = self.provider.uri(
            quirks.completion_path,
            self.provider.deployment(&self.model),
        );
        let builder = http::Request::post(uri).header("Content-Type", "application/json");
        let request = self
            .provider
            .headers(builder)
            .body(Body::Bytes(serde_json::to_vec(&body)?))
            .map_err(|error| CompletionError::ResponseError(error.to_string()))?;

        // A streamed reply is SSE; the unary reply is one whole JSON body.
        let framing = match mode {
            Mode::Streaming => Framing::Sse,
            Mode::Unary => Framing::Whole,
        };
        Ok(Encoded::new(request, framing)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder(&self, mode: Mode) -> ChatDecoder {
        ChatDecoder::new(
            self.provider.dialect.name,
            self.provider.dialect.quirks,
            mode,
        )
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Chat Completions *defers* `response_format` while tools are present
        // and no tool result exists yet, then applies it once a tool result
        // is in the history — so the native constraint does not suppress tool
        // calls; they compose. A dialect measured to honour both at once
        // (`Quirks::response_format_with_tools`) composes them from the first
        // turn. A dialect that drops `output_schema` cannot compose them at
        // all, and the agent falls back to tool-mode enforcement.
        ProviderCapabilities::default().with_native_output_tool_composition(
            self.provider.dialect.quirks.supports_response_format,
        )
    }
}

/// One classified frame of the chat-completions wire.
///
/// `Whole` is the unary reply's shape. It is a *modeled event*, not a second
/// decode path: its `interpret` synthesizes the message id, one block per
/// content part and tool call, and the terminal record — exactly what a
/// stream would have pushed — and the shared accumulator does the rest.
pub enum ChatEvent {
    /// A `chat.completion.chunk`: one step of a streamed turn.
    Chunk(ChatFrame),
    /// A `chat.completion`: the whole turn in one frame.
    Whole(ChatFrame),
    /// The wire's `[DONE]` sentinel — the deferred terminal, modeled rather
    /// than filtered out by the transport.
    Done,
    /// The wire's in-band error envelope, delivered with a 200 status.
    Failure(CompletionError),
    /// A bare JSON string where an envelope belongs: the whole answer, with
    /// no metadata and no terminal reason. Mira's gateway sends this.
    BareText(String),
}

/// The chat-completions decoder: one state machine for both replies.
pub struct ChatDecoder {
    /// Descriptor name the reply is attributed to.
    provider: &'static str,
    quirks: super::Quirks,
    /// Owns the constant-key `reasoning_content` lifecycle: those deltas
    /// carry no wire id or block boundaries, so the shared derivation
    /// synthesizes the end this wire never announces.
    reasoning: MintedReasoningLifecycle,
    /// Index-to-identity bridge only: this wire keys tool-call fragments by
    /// chunk index, so the decoder must correlate.
    open_tool_calls: ToolCallBridge<usize>,
    final_usage: Option<ChatUsage>,
    final_finish_reason: Option<FinishReason>,
    response_id: Option<String>,
    response_model: Option<String>,
    /// Accumulated primary-choice token metadata. `AdditionalParams::merge`
    /// concatenates nested arrays, which is the wire's token order.
    logprobs: Option<crate::message::AdditionalParams>,
    /// Accumulated provider-specific top-level chunk metadata.
    additional_params: Option<crate::message::AdditionalParams>,
    /// Whether `[DONE]` or a frame carrying a finish reason arrived — the
    /// only signals that count as the provider completing the turn.
    saw_terminal: bool,
    /// Whether any frame decoded successfully. A bare `[DONE]` after only
    /// parse failures must not dress the failure up as a default-usage
    /// success.
    saw_any_valid_frame: bool,
    /// Whether the wire's own in-band failure was consumed.
    failed: bool,
    /// Whether this reply arrives whole rather than as a stream — the
    /// [`Mode`] this decoder was built for: a buffered reply's EOF is the
    /// end of the answer, a stream's may be truncation.
    whole: bool,
}

impl ChatDecoder {
    fn new(provider: &'static str, quirks: super::Quirks, mode: Mode) -> Self {
        Self {
            provider,
            quirks,
            reasoning: MintedReasoningLifecycle::new(MintKind::Reasoning),
            open_tool_calls: ToolCallBridge::new(),
            final_usage: None,
            final_finish_reason: None,
            response_id: None,
            response_model: None,
            logprobs: None,
            additional_params: None,
            saw_terminal: false,
            saw_any_valid_frame: false,
            failed: false,
            whole: mode == Mode::Unary,
        }
    }

    /// The normalized finish reason a choice reported.
    ///
    /// A gateway's upstream-native reason is consulted only when the
    /// normalized field is absent or empty, which is OpenRouter's documented
    /// precedence; a direct provider has no native field to consult.
    fn finish_reason(&self, choice: &super::dto::ChatChoice) -> CompatibleFinishReason {
        if let Some(reason) = choice
            .finish_reason
            .as_ref()
            .map(super::dto::FinishReason::as_wire)
            .filter(|reason| !reason.is_empty())
        {
            return CompatibleFinishReason::Reported(map_openai_finish_reason(reason));
        }
        if self.quirks.native_finish_reason
            && let Some(native) = choice
                .native_finish_reason
                .as_deref()
                .filter(|reason| !reason.is_empty())
        {
            return CompatibleFinishReason::Reported(map_native_finish_reason(native));
        }
        CompatibleFinishReason::Absent
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

    /// One `chat.completion.chunk`.
    fn interpret_chunk(&mut self, mut frame: ChatFrame, out: &mut Output<Completion>) {
        self.saw_any_valid_frame = true;
        let choice = frame.primary().map(|choice| ChunkChoice {
            finish_reason: self.finish_reason(choice),
            text: delta_text(&choice.delta),
            reasoning: choice
                .delta
                .reasoning_content
                .clone()
                .or_else(|| choice.delta.reasoning.clone()),
            tool_calls: choice
                .delta
                .tool_calls
                .iter()
                .map(CompatibleToolCallChunk::from)
                .collect(),
            details: choice
                .delta
                .reasoning_details
                .iter()
                .filter_map(typed_detail)
                .collect(),
            logprobs: choice.logprobs.clone(),
        });
        self.absorb_metadata(&mut frame);

        let Some(choice) = choice else {
            return;
        };

        if let Some(reason) = choice.finish_reason.reported() {
            self.final_finish_reason = Some(reason);
            self.saw_terminal = true;
        }

        if let Some(logprobs) = choice.logprobs {
            match self.logprobs.as_mut() {
                Some(accumulated) => accumulated.merge(logprobs),
                None => self.logprobs = Some(logprobs),
            }
        }

        // Reasoning details are the turn's own output, so they are emitted
        // before this chunk's tool-call events: on the wire the detail that
        // carries a reasoning block arrives before (or with) the tool call it
        // precedes, and a reasoning block never depends on an open slot.
        if self.quirks.reasoning_details {
            for detail in &choice.details {
                if let Some((id, provider_id, content)) = detail_reasoning(detail) {
                    out.reasoning_block(id, provider_id, content);
                }
            }
        }

        // The tool-call events are built before they are emitted: the shared
        // lifecycle emits this chunk's classes in canonical order (reasoning,
        // its derived boundary end, text, then tool calls), so a chunk
        // carrying several at once keeps the wire's logical order — the model
        // reasons, speaks, then acts.
        let mut tool_events = Vec::new();
        for incoming in choice.tool_calls {
            if let Some(evicted) = self.open_tool_calls.evict_if(incoming.index, |existing| {
                should_evict_distinct_named_tool_call(existing, &incoming)
            }) {
                // The wire reused this call's slot: the evicted call is
                // delivered even when its arguments never parse.
                tool_events.push(evicted.end_event(UnparseableToolInput::EmptyObject));
            }

            // The bridge fixes the assembly key at open — the wire id, or a
            // provenance-gated mint when the wire omits one — and updates the
            // established id/name from later fragments.
            let slot = self.open_tool_calls.open(
                incoming.index,
                incoming.id.as_deref(),
                incoming.name.as_deref(),
            );

            if let Some(name) = incoming.name.as_ref().filter(|name| !name.is_empty()) {
                tool_events.push(StreamEvent::BlockDelta {
                    id: slot.key().clone(),
                    delta: Delta::ToolName { name: name.clone() },
                });
            }

            if let Some(arguments) = incoming
                .arguments
                .as_ref()
                .filter(|arguments| !arguments.is_empty())
            {
                slot.observe_arguments_delta(arguments);
                tool_events.push(StreamEvent::BlockDelta {
                    id: slot.key().clone(),
                    delta: Delta::ToolArguments {
                        arguments: arguments.clone(),
                    },
                });
            }

            if self.quirks.emits_complete_single_chunk_tool_calls
                && incoming.is_complete_single_chunk()
            {
                // Completion probe: the accumulator finalizes the call only
                // if its input parses, and keeps it open otherwise (`Keep`).
                tool_events.push(slot.end_event(UnparseableToolInput::Keep));
            }
        }

        let reasoning_signature = self
            .quirks
            .reasoning_details
            .then(|| choice.details.iter().find_map(reasoning_signature))
            .flatten();

        self.reasoning.emit_chunk(
            ChunkParts {
                reasoning: choice.reasoning,
                reasoning_signature,
                text: choice.text,
                tool_events,
            },
            out,
        );

        if choice.finish_reason.is_tool_calls() {
            for slot in self.open_tool_calls.drain_ordered() {
                // `tool_calls` says the provider completed the call. Invalid
                // JSON in that state is a provider defect, not evidence that
                // the output-token cap cut the payload short, and must remain
                // loud. Empty arguments still normalize to `{}` for genuine
                // zero-argument tools.
                out.push(Ok(slot.end_event(UnparseableToolInput::Error)));
            }
        }
    }

    /// Whether a decoded unary body is an output-length-truncated turn that
    /// still carries tool calls — the one state whose verbatim `arguments`
    /// strings are worth re-reading off the body.
    ///
    /// An over-approximation on purpose: a cut that landed before the first
    /// argument token decodes as `{}`, indistinguishable at this level from a
    /// genuine zero-argument call, so the raw-body pass decides.
    fn is_budget_cut_tool_turn(&self, frame: &ChatFrame) -> bool {
        let Some(choice) = frame.primary() else {
            return false;
        };
        if !matches!(
            self.finish_reason(choice).reported(),
            Some(FinishReason::Length)
        ) {
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

    /// The unary body with every tool call the output-token budget cut out of
    /// it dropped, or `None` when nothing was dropped and the ordinary
    /// classification stands.
    ///
    /// # The scenario
    ///
    /// A turn that runs out of output budget mid-arguments comes back with
    /// `finish_reason: "length"` and `tool_calls[].function.arguments` cut
    /// partway through the JSON object — `{"note": "The` on llama.cpp at a
    /// 20-token cap. The strict argument decode then fails, and failing it
    /// takes the *whole frame* down: this is the defect rig#2359 fixed, and
    /// the shared policy DeepSeek, Mistral and OpenRouter all carry.
    ///
    /// # Why the call is dropped rather than raised
    ///
    /// The turn happened. It has usage the caller is billed for, an id, a
    /// finish reason saying precisely what went wrong, and often text beside
    /// the call — erroring throws all of that away to report one unusable
    /// fragment. And a mid-arguments cut is not a protocol violation: the
    /// provider did what it was told and stopped at the cap the caller set,
    /// so the honest normalization is a `Length` turn minus the call that
    /// never finished. A half-parsed call must not reach the caller either,
    /// because invoking a tool on truncated input is worse than not invoking
    /// it — hence dropped, not repaired.
    ///
    /// # What stays loud
    ///
    /// Only an output-length choice is eligible, so an ordinary completed
    /// `tool_calls` turn carrying malformed JSON is still a decode error —
    /// there the provider claims it finished, and malformed arguments are its
    /// own defect. Valid-JSON arguments are never touched, whatever they
    /// contain: unexpected *content* is a schema problem, not a cut. And
    /// before any call is dropped, a copy with its arguments stubbed to `{}`
    /// must decode, so a choice that is *also* broken elsewhere (a call
    /// missing its id, an unknown tool type) keeps its original error rather
    /// than having the evidence deleted underneath it.
    fn body_without_calls_cut_by_the_budget(&self, data: &str) -> Option<ChatFrame> {
        let mut body = serde_json::from_str::<serde_json::Value>(data).ok()?;
        let mut dropped = 0;
        for choice in body.get_mut("choices").and_then(as_array_mut)? {
            if !self.reports_output_length(choice) {
                continue;
            }
            let mut probe = choice.clone();
            if !stub_unusable_arguments(&mut probe)
                || serde_json::from_value::<super::dto::ChatChoice>(probe).is_err()
            {
                continue;
            }
            dropped += drop_unusable_calls(choice);
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

    /// The unary `chat.completion` body: synthesize the events a stream of
    /// the same turn would have pushed.
    fn interpret_whole(&mut self, mut frame: ChatFrame, out: &mut Output<Completion>) {
        self.saw_any_valid_frame = true;
        let Some(choice) = frame.primary() else {
            out.error(CompletionError::ResponseError(
                "Response contained no choices".to_owned(),
            ));
            self.failed = true;
            return;
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
            out.error(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".to_owned(),
            ));
            self.failed = true;
            return;
        };
        let logprobs = choice.logprobs.clone();
        self.absorb_metadata(&mut frame);
        self.logprobs = logprobs;
        self.final_finish_reason = finish_reason.reported();
        self.saw_terminal = true;

        // No message-id block: `chatcmpl-…` is a *response*-scoped id, not
        // an id this wire would recognize on a replayed assistant message,
        // so it rides on the terminal record's `response_id` — which is also
        // the only place the streamed reply could put it.

        // The whole turn is declared as one chunk and emitted through the
        // SAME lifecycle the streamed path uses. Open-coding the emission
        // here is what made a `reasoning_content` turn panic the sequence
        // law: `out.reasoning` mints a reasoning part and `out.text` then
        // interleaved text into it without the derived boundary end. It is
        // also exactly the unary/stream drift this model exists to remove,
        // so there is one emitter and not two.
        let text = {
            // The streamed path concatenates a turn's text deltas into one
            // block, so the unary body's parts join the same way rather than
            // producing a different number of blocks for the same turn.
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

        let mut tool_events = Vec::with_capacity(tool_calls.len());
        for call in &tool_calls {
            // Every id-less call needs its OWN key. A shared constant made
            // each one restate the block the previous one closed, so a turn
            // calling the same tool three times without ids folded to one
            // call — the streamed path mints per call through the same
            // namespace, so the unary body does too.
            let key = crate::streaming::non_empty_id(call.id.clone())
                .map_or_else(|| self.open_tool_calls.minted_ids().mint(), BlockId::wire);
            tool_events.push(StreamEvent::BlockEnd {
                id: key,
                end: crate::streaming::BlockClose::ToolCall(
                    ToolCallEnd::whole(&call.function.name, call.function.arguments.clone())
                        .with_tool_id(call.id.clone()),
                ),
                block: None,
            });
        }

        let reasoning = reasoning.filter(|reasoning| !reasoning.is_empty());
        // The unary body carries the same `reasoning_details` array the
        // streamed path reads off its deltas — an OpenRouter tool-call turn
        // answers with the plaintext in `message.reasoning` and its
        // replay-required signature in `message.reasoning_details`. Reading
        // only `reasoning` dropped the signature, and a reasoning block
        // replayed unsigned is one the upstream rejects on the next turn.
        let details: Vec<&unary::ReasoningDetails> = if self.quirks.reasoning_details {
            reasoning_details.iter().collect()
        } else {
            Vec::new()
        };
        let blocks: Vec<_> = details
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(position, detail)| whole_detail_reasoning(position as u64, detail))
            .collect();
        // `message.reasoning` is the DISPLAY of the same chain of thought
        // those entries state structurally (an OpenRouter OpenAI route
        // answers with the summary in both), so publishing it beside them
        // would carry one chain twice and replay it twice — the precedence
        // `replay_whole_response` already gives a Responses body's
        // structured reasoning items over its top-level `reasoning` string.
        // Entries that state nothing replayable leave the plaintext as the
        // turn's only statement, and a signature-only entry rides onto it.
        let (reasoning, reasoning_signature) = if blocks.is_empty() {
            (
                reasoning,
                details.iter().copied().find_map(reasoning_signature),
            )
        } else {
            (None, None)
        };
        // An empty turn is legal exactly where the reply named a terminal
        // that CUT IT SHORT — `FinishReason::truncated_output`, the one
        // statement of that set (`completion::request`). A cap consumed
        // entirely by hidden reasoning, or a filter that removed
        // everything, leaves nothing to deliver and the reason is then the
        // caller's only diagnostic; rejecting it would also discard the
        // usage the caller is billed for (the recorded contract in
        // `deepseek_long_loop_output_cap_midway`, record 4: an empty
        // choice, `Length`, and 900 in / 32 out / 766 cached). `stop` and
        // `tool_calls` describe a turn that RAN TO COMPLETION, so an empty
        // one is the provider defect `EMPTY_RESPONSE_ERROR` names, and so
        // is a body that named no terminal at all.
        let cut_short = self
            .final_finish_reason
            .as_ref()
            .is_some_and(FinishReason::truncated_output);
        if text.is_empty()
            && tool_events.is_empty()
            && reasoning.is_none()
            && blocks.is_empty()
            && !cut_short
        {
            out.error(CompletionError::ResponseError(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
            self.failed = true;
            return;
        }

        // Reasoning details are the turn's own output, so they are emitted
        // before the chunk's text and tool calls, exactly as the streamed
        // path orders them.
        for (id, provider_id, content) in blocks {
            out.reasoning_block(id, provider_id, content);
        }

        self.reasoning.emit_chunk(
            ChunkParts {
                reasoning,
                reasoning_signature,
                text: (!text.is_empty()).then_some(text),
                tool_events,
            },
            out,
        );

        out.close_active_blocks();
        self.emit_terminal(out);
    }

    /// Build and push the provider's terminal record.
    fn emit_terminal(&mut self, out: &mut Output<Completion>) {
        let terminal = CompatibleTerminal {
            usage: self.final_usage.take(),
            finish_reason: self.final_finish_reason.take(),
            response_id: self.response_id.take(),
            model: self.response_model.take(),
            logprobs: self.logprobs.take(),
            additional_params: self.additional_params.take(),
        };
        let native = StreamingCompletionResponse::from_terminal(terminal);
        // The provider's own terminal record rides along serialized — the
        // same capture the unary path performed before normalizing.
        match serde_json::to_value(&native) {
            Ok(raw) => out.final_record(native.into_stream_final(self.provider).with_raw(raw)),
            Err(error) => out.error(CompletionError::from(error)),
        }
    }
}

/// One chunk's primary choice, in the shape the state machine consumes.
struct ChunkChoice {
    finish_reason: CompatibleFinishReason,
    text: Option<String>,
    reasoning: Option<String>,
    tool_calls: Vec<CompatibleToolCallChunk>,
    details: Vec<unary::ReasoningDetails>,
    logprobs: Option<crate::message::AdditionalParams>,
}

use crate::operation::Completion;

impl Decoder<Completion> for ChatDecoder {
    type Event = ChatEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<ChatEvent> {
        let data = frame.as_str();
        // `[DONE]` is the wire's terminal sentinel, not JSON; it is Known by
        // definition and its `interpret` emits the deferred terminal.
        if data == "[DONE]" {
            return WireEvent::Known(ChatEvent::Done);
        }
        // The wire's in-band error envelope arrives with a 200 status and is
        // this wire's own terminal failure, so it is a modeled event rather
        // than a transport-level filter.
        if let Some(error) = provider_error_envelope(&data) {
            return WireEvent::Known(ChatEvent::Failure(error));
        }
        // A gateway that answers with a bare JSON string rather than an
        // envelope. Modeled as its own event for the dialects measured to do
        // it, because the shared classifier reads a non-object frame as
        // `Unknown` and the turn would then produce no answer at all. This
        // pre-empts the classifier exactly as `[DONE]` and the error
        // envelope above do: a frame that is not the wire's object shape is
        // not the classifier's business.
        if self.quirks.accepts_bare_string_reply
            && let Ok(serde_json::Value::String(text)) =
                serde_json::from_str::<serde_json::Value>(&data)
        {
            return WireEvent::Known(ChatEvent::BareText(text));
        }
        let classified = classify_chat_completions_frame::<ChatFrame>(&data);
        // A tool call the output-token budget cut mid-arguments is not a
        // corrupt frame. It reaches here two ways — the strict argument
        // decode fails outright, or the cut landed before the first argument
        // token and decoded as `{}` — and both are settled on the raw body.
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

    fn interpret(&mut self, event: ChatEvent, out: &mut Output<Completion>) {
        match event {
            ChatEvent::Chunk(frame) => self.interpret_chunk(frame, out),
            ChatEvent::Whole(frame) => self.interpret_whole(frame, out),
            ChatEvent::Done => self.saw_terminal = true,
            ChatEvent::BareText(text) => {
                // The whole reply: no id, no model, no usage and no finish
                // reason, which is what the gateway sent and what the
                // deleted `Simple(String)` branch normalized to.
                self.saw_any_valid_frame = true;
                self.saw_terminal = true;
                if !text.is_empty() {
                    out.text(text);
                }
                out.close_active_blocks();
                self.emit_terminal(out);
            }
            ChatEvent::Failure(error) => {
                // Content the provider fully delivered reaches the consumer
                // before the failure, and no terminal record follows.
                self.flush_before_terminal_error(out);
                out.error(error);
                self.failed = true;
            }
        }
    }

    fn finish(&mut self, out: &mut Output<Completion>) {
        // Tool calls the provider fully delivered are content, so a truncated
        // reply still flushes them. Partial calls drop in the accumulator.
        let output_length_truncation = matches!(
            self.final_finish_reason.as_ref(),
            Some(FinishReason::Length)
        );
        for slot in self.open_tool_calls.drain_ordered() {
            if output_length_truncation && !slot.has_substantive_arguments() {
                tracing::debug!(
                    tool = %slot.name,
                    "dropping streamed tool call cut off before its first argument token"
                );
                continue;
            }
            // Only a provider-declared output-length truncation authorizes
            // discarding malformed partial arguments. `stop`, an unknown
            // reason, and a bare `[DONE]` all claim completion; treating
            // their malformed calls as truncation would silently erase
            // provider output and could hide compound wire defects.
            let on_unparseable = if output_length_truncation {
                UnparseableToolInput::Drop
            } else {
                UnparseableToolInput::Error
            };
            out.push(Ok(slot.end_event(on_unparseable)));
        }

        // A WHOLE reply in which this wire recognized nothing — no chunk,
        // no completion body, no `[DONE]`, no error envelope — delivered
        // no turn and reported no defect either: the classifier
        // warn-skips an unmodeled frame (a gateway answering with a bare
        // JSON string on a dialect without that quirk) so nothing else
        // will speak. That is the nothing-delivered state
        // `EMPTY_RESPONSE_ERROR` names, and reporting it keeps such a
        // reply a failed call rather than a silent, contentless success.
        // Scoped to a whole reply because the same EOF on a *stream* is
        // truncation, reported by the missing terminal record. A corrupt
        // frame never reaches here as silence: its parse error was
        // already yielded and is what the caller sees.
        if self.whole && !self.saw_any_valid_frame && !self.saw_terminal {
            out.error(CompletionError::ResponseError(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
            return;
        }

        // Only `[DONE]` or a frame carrying a finish reason counts as the
        // provider completing the turn. A reply that reached EOF without
        // either signal (truncation) gets no terminal record — synthesizing
        // one would present the partial turn as a successful, default-usage
        // completion. A bare `[DONE]` with no successfully decoded frame at
        // all is treated the same way: the parse errors were already yielded.
        if !self.saw_terminal || !self.saw_any_valid_frame {
            return;
        }
        self.emit_terminal(out);
    }

    fn flush_before_terminal_error(&mut self, out: &mut Output<Completion>) {
        // Fully-delivered tool calls flush before the terminal error reaches
        // the consumer, so a first-`Err`-stop consumer sees them too.
        for slot in self.open_tool_calls.drain_ordered() {
            out.push(Ok(slot.end_event(UnparseableToolInput::Drop)));
        }
    }

    fn is_finished(&self) -> bool {
        self.failed
    }

    /// Verdict, model, response id, usage and error envelope, read off a raw
    /// payload before normalization discards them. The driver calls it for
    /// the unary reply and for every stream frame without anyone having to
    /// attach it.
    fn project(&self, payload: &[u8], sink: &mut dyn ObservationSink) {
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
            let code = error.code.map(|code| match code {
                serde_json::Value::String(code) => sink.scrub(&code),
                serde_json::Value::Number(code) => code.to_string(),
                _ => "[invalid]".to_owned(),
            });
            sink.emit(AdapterEvent::ErrorEnvelope {
                error: AdapterErrorEnvelope {
                    code,
                    status: error.kind.map(|value| sink.scrub(&value)),
                    message: error.message.map(|value| sink.scrub(&value)),
                },
            });
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

/// The error envelope this wire sends: `{"error": {code, message, type}}`.
#[derive(Deserialize)]
struct ObservedError {
    code: Option<serde_json::Value>,
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
}

/// A gateway's encrypted-reasoning detail as a whole reasoning block.
///
/// Encrypted reasoning (`{"type":"reasoning.encrypted"}`) is the turn's own
/// output, not tool-call metadata: it arrives with `reasoning: null` and an
/// `rs_*` id of its own, which never matches a `call_*` tool-call id, and it
/// arrives before any tool call opens. Emitting it as a reasoning block is
/// what lets the blob reach the aggregated choice and be replayed next turn.
fn detail_reasoning(
    detail: &unary::ReasoningDetails,
) -> Option<(BlockId, Option<String>, crate::message::ReasoningContent)> {
    let unary::ReasoningDetails::Encrypted { id, data, .. } = detail else {
        return None;
    };
    // The durable handle exists only when the wire issued one; an id-less
    // detail keys accumulation by a minted key and replays with the id
    // absent — no fabricated empty "wire" id. The mint kind is
    // `EncryptedReasoning`, NOT `Reasoning`: plaintext `reasoning` text
    // accumulates under `Minted { Reasoning, 0 }`, and a whole block under
    // that same key would restate — i.e. replace — the open text part.
    let provider_id = id.clone().and_then(crate::streaming::non_empty_id);
    let key = provider_id
        .as_ref()
        .map_or(BlockId::minted(MintKind::EncryptedReasoning, 0), |id| {
            BlockId::wire(id.as_str())
        });
    Some((
        key,
        provider_id,
        crate::message::ReasoningContent::Encrypted(data.clone()),
    ))
}

/// A unary body's reasoning detail as a whole reasoning block.
///
/// The streamed path sees this array as fragments — one `summary`/`text`
/// token per chunk — so it can lift only a self-contained entry out of one
/// ([`detail_reasoning`]). A unary body states every entry COMPLETE, and the
/// gateway requires the array back entry for entry on the next turn:
/// `crates/rig-cassette/fixtures/cassettes/openrouter/reasoning_roundtrip/nonstreaming.yaml`
/// record 2 replays the summary AND the encrypted blob, in the order the
/// reply sent them and nothing besides.
///
/// One block per entry, because each carries its own id — the summary none,
/// the blob its `rs_*` — and a block replays under a single id.
fn whole_detail_reasoning(
    position: u64,
    detail: &unary::ReasoningDetails,
) -> Option<(BlockId, Option<String>, crate::message::ReasoningContent)> {
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
        // Everything else states nothing replayable: an empty entry, or the
        // signature-only `reasoning.text` an Anthropic route sends to sign
        // the plaintext it states separately — which is what
        // [`reasoning_signature`] reads it for.
        _ => return None,
    };
    let provider_id = id.clone().and_then(crate::streaming::non_empty_id);
    // Keyed by the wire id when the entry has one, else by the entry's
    // POSITION under the structured-detail mint kind: two id-less entries
    // are then two blocks rather than one restating — replacing — the
    // other, and neither can restate the plaintext reasoning accumulating
    // under `Minted { Reasoning, 0 }`.
    let key = provider_id.as_ref().map_or_else(
        || BlockId::minted(MintKind::EncryptedReasoning, position),
        |id| BlockId::wire(id.as_str()),
    );
    Some((key, provider_id, content))
}

/// A gateway's signature-only reasoning detail.
///
/// Anthropic routes stream the plaintext in `delta.reasoning`, then send its
/// replay-required signature as a final signature-only `reasoning.text`
/// detail immediately before the tool call. The unary body carries the same
/// detail on its assistant message. Feeding that authoritative close into the
/// shared lifecycle signs the normalized reasoning block on either path.
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

/// One reasoning detail, typed.
///
/// A detail type this wire does not model is not an error: the gateways
/// extend the vocabulary independently, and an unknown entry simply carries
/// nothing this decoder acts on.
fn typed_detail(detail: &serde_json::Value) -> Option<unary::ReasoningDetails> {
    serde_json::from_value(detail.clone()).ok()
}

#[cfg(test)]
mod tests;
