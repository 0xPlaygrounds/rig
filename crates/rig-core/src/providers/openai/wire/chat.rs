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
use crate::operation::{Block, CallFragment, Completion};
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
    ChatChoice, ChatFrame, StreamingCompletionResponse, StreamingToolCall, UsageCounts, delta_text,
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
        let mut typed = unary::CompletionRequest::try_from(unary::OpenAIRequestParams {
            model: self.model.clone(),
            request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
            supports_response_format: quirks.supports_response_format,
            response_format_with_tools: quirks.response_format_with_tools,
            supports_tools: quirks.supports_tools,
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
        self.finalize(&mut body);

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
            | BodyRewrite::Mistral
            | BodyRewrite::OpenRouter => {}
        }
        Ok(())
    }

    /// Apply dialect body rewrites after merging streaming parameters.
    fn finalize(&self, body: &mut serde_json::Value) {
        let Some(map) = body.as_object_mut() else {
            return;
        };
        match self.provider.dialect.quirks.rewrite {
            BodyRewrite::Perplexity => {
                if let Some(messages) = map.get_mut("messages").and_then(as_array_mut) {
                    finalize_perplexity(messages);
                }
            }
            BodyRewrite::Mira => {
                // The gateway takes every message's content as one string.
                for message in map
                    .get_mut("messages")
                    .and_then(as_array_mut)
                    .into_iter()
                    .flatten()
                {
                    if let Some(content) = message.get_mut("content") {
                        unary::flatten_text_content_parts(content, "\n", false);
                    }
                }
            }
            BodyRewrite::DeepSeek => finalize_deepseek(map),
            BodyRewrite::Mistral => finalize_mistral(map),
            BodyRewrite::OpenRouter => {
                if self.prompt_caching {
                    apply_openrouter_prompt_caching(map);
                }
            }
            BodyRewrite::None
            | BodyRewrite::GroqCompoundTools
            | BodyRewrite::HuggingFaceRouter
            | BodyRewrite::LlamaCpp
            | BodyRewrite::Moonshot => {}
        }
    }
}

/// Perplexity accepts only system, user and assistant roles in strict
/// user/assistant alternation: text-only content-part arrays flatten (arrays
/// with other parts are left for its sonar models), and adjacent text
/// messages of one role become one.
fn finalize_perplexity(messages: &mut Vec<serde_json::Value>) {
    let mut merged: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
    for mut message in std::mem::take(messages) {
        if let Some(content) = message.get_mut("content") {
            unary::flatten_text_content_parts(content, "\n", true);
        }
        let role = message.get("role").and_then(serde_json::Value::as_str);
        let text = message.get("content").and_then(serde_json::Value::as_str);
        if let (Some(role @ ("user" | "assistant")), Some(text)) = (role, text)
            && let Some(previous) = merged.last_mut()
            && previous.get("role").and_then(serde_json::Value::as_str) == Some(role)
            && let Some(serde_json::Value::String(previous)) = previous.get_mut("content")
        {
            previous.push('\n');
            previous.push_str(text);
            continue;
        }
        merged.push(message);
    }
    *messages = merged;
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
    let Some(raw_tools) = map.shift_remove("tools") else {
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

/// DeepSeek takes message `content` as a plain string and needs an explicit
/// empty `content` on a tool-call-only assistant turn. Its thinking models take `reasoning_content` back on every
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
fn finalize_mistral(map: &mut serde_json::Map<String, serde_json::Value>) {
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
        return;
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
        if let Some(content) = message.get_mut("content") {
            mistral_content(content);
        }
    }
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

/// One content part as the Mistral chunk that carries it, by its `type`:
/// text and refusal parts are `text` chunks, a file's data is a
/// `document_url` and its id a `file` chunk, audio is its base64 string,
/// and an image keeps only its `image_url`, for every chunk forbids unknown
/// keys. A part the shared conversion does not emit stays as it is.
fn mistral_chunk(part: &serde_json::Value) -> serde_json::Value {
    use serde_json::json;
    let field = |pointer: &str| part.pointer(pointer).and_then(serde_json::Value::as_str);
    match part.get("type").and_then(serde_json::Value::as_str) {
        Some(MISTRAL_TEXT | MISTRAL_REFUSAL) | None => {
            json!({"type": MISTRAL_TEXT, MISTRAL_TEXT: mistral_part_text(part).unwrap_or_default()})
        }
        Some(MISTRAL_IMAGE) => {
            json!({"type": MISTRAL_IMAGE, MISTRAL_IMAGE: part.get(MISTRAL_IMAGE)})
        }
        Some(MISTRAL_AUDIO) => json!({"type": MISTRAL_AUDIO,
            MISTRAL_AUDIO: field("/input_audio/data").or(field("/input_audio"))}),
        Some(MISTRAL_FILE) => match (field("/file/file_data"), field("/file/filename")) {
            (Some(data), Some(name)) => {
                json!({"type": MISTRAL_DOCUMENT, MISTRAL_DOCUMENT: data, "document_name": name})
            }
            (Some(data), None) => json!({"type": MISTRAL_DOCUMENT, MISTRAL_DOCUMENT: data}),
            (None, _) => json!({"type": MISTRAL_FILE,
                "file_id": field("/file/file_id").or(field("/file_id"))}),
        },
        Some(_) => part.clone(),
    }
}

/// Flatten a text-only part array to a string, and convert a mixed one to
/// Mistral content chunks. Anything else stays as it is.
fn mistral_content(content: &mut serde_json::Value) {
    let Some(parts) = content.as_array_mut() else {
        return;
    };
    if parts.iter().all(is_mistral_text_part) {
        unary::flatten_text_content_parts(content, "", false);
        return;
    }
    for part in parts {
        *part = mistral_chunk(part);
    }
}

/// OpenRouter's ephemeral `cache_control` on the system prompt.
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

    /// Chat reads no images in assistant messages, images in tool results
    /// only on a dialect that says so, and tools where the dialect takes
    /// them. Which models read user images follows each provider's
    /// documented model rules (`reads_images`).
    fn accepts(&self, model: &str) -> crate::completion::Accepts {
        let quirks = &self.provider.dialect.quirks;
        let user_images = reads_images(&self.provider.dialect, model);
        crate::completion::Accepts {
            user_images,
            assistant_images: false,
            tool_result_images: user_images && quirks.supports_image_tool_results,
            tools: quirks.supports_tools,
        }
    }

    /// What the Chat encoder carries, by dialect: an image as a URL or
    /// typed data; audio as data; video as a URL or typed data, except to
    /// OpenAI, Azure and Mistral; a PDF as data, or as a URL OpenRouter or
    /// Mistral fetches; a file id where the dialect takes one; and a string
    /// document as text. DeepSeek and Mira take text only, and Perplexity
    /// takes no media but images.
    fn encodes(&self, _model: &str, media: crate::completion::Media<'_>) -> bool {
        use crate::completion::Media;
        use crate::message::{DocumentMediaType, DocumentSourceKind as Source};
        let dialect = &self.provider.dialect;
        let rewrite = dialect.quirks.rewrite;
        let parts = !matches!(rewrite, BodyRewrite::DeepSeek | BodyRewrite::Mira);
        let files = parts && rewrite != BodyRewrite::Perplexity;
        let linked = |source: &Source, typed: bool| match source {
            Source::Url(_) => true,
            Source::Base64(_) => typed,
            Source::Raw(_) | Source::FileId(_) | Source::String(_) | Source::Unknown => false,
        };
        match media {
            Media::Image(image, place) => {
                parts
                    && place != crate::completion::Place::Assistant
                    && linked(&image.data, image.media_type.is_some())
            }
            Media::Audio(audio) => files && matches!(audio.data, Source::Base64(_)),
            Media::Video(video) => {
                files
                    && rewrite != BodyRewrite::Mistral
                    && ![super::dialects::OPENAI.name, super::dialects::AZURE.name]
                        .contains(&dialect.name)
                    && linked(&video.data, video.media_type.is_some())
            }
            Media::Document(document) => {
                let pdf = document.media_type == Some(DocumentMediaType::PDF);
                match &document.data {
                    Source::String(_) => !pdf,
                    Source::FileId(_) => files && dialect.quirks.accepts_file_ids,
                    Source::Base64(_) => files && pdf,
                    Source::Url(_) => {
                        pdf && matches!(rewrite, BodyRewrite::OpenRouter | BodyRewrite::Mistral)
                    }
                    Source::Raw(_) | Source::Unknown => false,
                }
            }
        }
    }

    /// pi's rule for this wire. A `call|item` id joins its sanitized halves
    /// with `_`, ending a result over 40 characters in a hash of the whole
    /// id; OpenAI's own ids are cut to 40; any other id is kept. Mistral
    /// takes exactly nine alphanumerics.
    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _: Option<&crate::message::Origin>,
    ) -> String {
        if self.provider.dialect.quirks.rewrite == BodyRewrite::Mistral {
            return mistral_call_id(id);
        }
        let sanitized = |part: &str| -> String {
            part.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect()
        };
        if let Some((call, item)) = id.split_once('|') {
            let call = sanitized(call);
            let item = sanitized(item);
            let combined = if item.is_empty() {
                call.clone()
            } else {
                format!("{call}_{item}")
            };
            if combined.len() <= 40 {
                return combined;
            }
            let hash: String = short_hash(id).chars().take(8).collect();
            let prefix: String = call.chars().take((40 - hash.len() - 1).max(1)).collect();
            return format!("{prefix}_{hash}");
        }
        if self.provider.dialect.name == super::dialects::OPENAI.name {
            return id.chars().take(40).collect();
        }
        id.to_owned()
    }
}

/// OpenAI's Chat models that read no images.
const OPENAI_TEXT_ONLY: [&str; 10] = [
    unary::GPT_4,
    unary::GPT_4_0613,
    unary::GPT_4_32K,
    unary::GPT_4_32K_0613,
    unary::GPT_4_0125_PREVIEW,
    unary::GPT_4_1106_PREVIEW,
    unary::GPT_4_TURBO_PREVIEW,
    unary::O1_MINI,
    unary::O1_PREVIEW,
    unary::O3_MINI,
];

/// Whether `model` reads user images from `vendor`, a dialect's name or the
/// vendor an OpenRouter model id starts with, by the provider's documented
/// model rules. A model the rules do not name reads them. DeepSeek's API
/// takes text content only (it answers an image part with a 400) and
/// Mira's gateway takes text; Groq reads images on its Llama 4 models;
/// Mistral's Codestral and Devstral are text models; OpenAI's are GPT-3.5
/// and [`OPENAI_TEXT_ONLY`] with their dated snapshots; Z.AI, Moonshot,
/// MiniMax and MiMo apply the rules their Messages wires share.
fn vendor_reads_images(vendor: &str, model: &str) -> bool {
    use crate::providers::{minimax, moonshot, xiaomimimo, zai};
    match vendor {
        "deepseek" | "mira" => false,
        "groq" => model.contains("llama-4"),
        "mistral" | "mistralai" => {
            !(model.starts_with("codestral") || model.starts_with("devstral"))
        }
        "openai" | "azure.openai" => {
            !(model.starts_with("gpt-3.5") || OPENAI_TEXT_ONLY.iter().any(|id| is_model(model, id)))
        }
        "zai" | "z-ai" => zai::reads_images(model),
        "moonshot" | "moonshotai" => moonshot::reads_images(model),
        "minimax" => minimax::reads_images(model),
        "xiaomimimo" | "xiaomi" => xiaomimimo::reads_images(model),
        _ => true,
    }
}

/// Whether `model` reads user images on `dialect`. OpenRouter names a
/// model `vendor/model`, and the vendor's rule applies.
fn reads_images(dialect: &super::Dialect, model: &str) -> bool {
    match model.split_once('/') {
        Some((vendor, model)) if dialect.quirks.rewrite == BodyRewrite::OpenRouter => {
            vendor_reads_images(vendor, model)
        }
        _ => vendor_reads_images(dialect.name, model),
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
/// text under each, so the first that carries text is the one replayed.
const REASONING_TEXT_KEYS: [&str; 3] = ["reasoning_content", "reasoning", "reasoning_text"];

/// The structured reasoning key, kept verbatim with the reasoning text.
const REASONING_DETAILS: &str = "reasoning_details";

/// What one content part of a Chat message is. Every part becomes a block.
#[derive(Debug, PartialEq)]
pub(crate) enum Part {
    /// Answer text: a `text` part, or a `refusal` part's text.
    Text(String),
    /// Mistral and Magistral reasoning: a `thinking` part's text.
    Thinking(String),
    /// An image the model produced.
    Image,
    /// A part rig has no canonical meaning for, kept as it came.
    Unknown,
}

impl Part {
    /// The part `part` is, by its `type`.
    pub(crate) fn of(part: &serde_json::Value) -> Self {
        let text = |key: &str| {
            part.get(key)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        match part.get("type").and_then(serde_json::Value::as_str) {
            Some("text") => Self::Text(text("text")),
            Some("refusal") => Self::Text(text("refusal")),
            Some("thinking") => Self::Thinking(match part.get("thinking") {
                Some(serde_json::Value::Array(chunks)) => chunks
                    .iter()
                    .filter_map(|chunk| chunk.get("text").and_then(serde_json::Value::as_str))
                    .collect(),
                _ => text("thinking"),
            }),
            Some("image_url") => Self::Image,
            _ => Self::Unknown,
        }
    }
}

/// What one tool call of a Chat message is.
#[derive(Debug, PartialEq)]
pub(crate) enum CallKind {
    /// A function call: its arguments are JSON text.
    Function,
    /// A custom tool call: its input is free text, the call's `{"input"}`.
    Custom,
    /// A call kind rig cannot answer, kept but never sent back.
    Unknown,
}

impl CallKind {
    /// The kind of the assembled `call`, by its `type`; a call without one
    /// is a function call.
    pub(crate) fn of(call: &serde_json::Value) -> Self {
        match call.get("type").and_then(serde_json::Value::as_str) {
            None | Some("function") => Self::Function,
            Some("custom") => Self::Custom,
            Some(_) => Self::Unknown,
        }
    }
}

/// The kind of block a decoder is writing.
#[derive(Clone, Copy, PartialEq)]
enum Writing {
    Text,
    Reasoning,
}

/// The chat-completions decoder: one state machine for a whole reply and a
/// stream of chunks. A whole reply is restated as the one chunk carrying
/// its message. As pi's `openai-completions` decodes a message, a run of
/// reasoning is a block, a run of text a block, and each tool call, image
/// and unknown content part a block of its own. A block closes when the
/// next one starts. An answer's audio transcript is its text when it has
/// no other, and its first text block holds the audio's id. Each block
/// holds only its own item, and the assembled message is the turn's native,
/// for display.
pub struct ChatDecoder {
    quirks: super::Quirks,
    /// The assistant message as assembled so far, without its tool calls.
    message: serde_json::Map<String, serde_json::Value>,
    /// Each tool call as assembled so far, in the order they opened; `None`
    /// once it left the message.
    calls: Vec<Option<serde_json::Value>>,
    /// The position in `calls` of the call open at each wire index.
    open_calls: std::collections::BTreeMap<usize, usize>,
    /// The block being written, and its writer index.
    writing: Option<(Writing, usize)>,
    /// The reasoning block's item as it will hold it: the field its text
    /// arrived in, or Mistral's thinking part, its text, and its own
    /// `reasoning_details`.
    reasoning_field: Option<&'static str>,
    thinking_part: bool,
    reasoning_text: String,
    reasoning_details: serde_json::Map<String, serde_json::Value>,
    /// The id of the answer's audio.
    audio_id: Option<serde_json::Value>,
    /// Whether any block opened.
    wrote: bool,
    final_usage: Option<serde_json::Value>,
    final_finish_reason: Option<FinishReason>,
    response_id: Option<String>,
    response_model: Option<String>,
    /// Accumulated primary-choice token metadata, in the wire's token order.
    logprobs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Accumulated provider-specific top-level chunk metadata.
    additional_params: serde_json::Map<String, serde_json::Value>,
    /// Whether a finish reason established the turn complete.
    saw_terminal: bool,
}

impl ChatDecoder {
    fn new(quirks: super::Quirks) -> Self {
        Self {
            quirks,
            message: serde_json::Map::new(),
            calls: Vec::new(),
            open_calls: std::collections::BTreeMap::new(),
            writing: None,
            reasoning_field: None,
            thinking_part: false,
            reasoning_text: String::new(),
            reasoning_details: serde_json::Map::new(),
            audio_id: None,
            wrote: false,
            final_usage: None,
            final_finish_reason: None,
            response_id: None,
            response_model: None,
            logprobs: None,
            additional_params: serde_json::Map::new(),
            saw_terminal: false,
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
        let ChatFrame {
            id,
            model,
            choices,
            usage,
            additional_params,
        } = frame;
        self.response_id = id.or(self.response_id.take());
        self.response_model = model.or(self.response_model.take());
        self.final_usage = usage
            .filter(|usage| !usage.is_null())
            .or(self.final_usage.take());
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

    /// One delta of the assistant message: its reasoning, its text, its
    /// content parts, its audio transcript, images and tool calls reach
    /// their blocks, and every field reaches the message.
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
                .map(|text| (*key, text.to_owned()))
        });
        if let Some((key, text)) = reasoning {
            self.write(Writing::Reasoning, &text, out)?;
            self.reasoning_field.get_or_insert(key);
        }
        if let Some(serde_json::Value::Array(details)) = delta.shift_remove(REASONING_DETAILS)
            && !details.is_empty()
        {
            self.write(Writing::Reasoning, "", out)?;
            merge_details(&mut self.reasoning_details, details.clone());
            merge_details(&mut self.message, details);
        }
        let audio = delta.get("audio");
        if let Some(id) = audio
            .and_then(|audio| audio.get("id"))
            .filter(|id| !id.is_null())
        {
            self.audio_id = Some(id.clone());
        }
        let transcript = audio
            .and_then(|audio| audio.get("transcript"))
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.is_empty())
            .map(str::to_owned);
        match delta.get("content") {
            Some(serde_json::Value::Array(parts)) => {
                for part in parts.clone() {
                    self.part(part, out)?;
                }
            }
            Some(part @ serde_json::Value::Object(_)) => self.part(part.clone(), out)?,
            _ => {
                if let Some(text) = delta_text(&delta).or(transcript) {
                    self.write(Writing::Text, &text, out)?;
                }
            }
        }
        if let Some(serde_json::Value::Array(images)) = delta.get("images") {
            for image in images.clone() {
                self.image(image, out)?;
            }
        }
        if let Some(serde_json::Value::Array(calls)) = delta.shift_remove("tool_calls") {
            for call in calls {
                self.call(call, out)?;
            }
        }
        merge_fields(&mut self.message, &delta);
        Ok(())
    }

    /// Append `text` to a block of kind `writing`, first closing the block
    /// being written and opening a new one when that is of another kind.
    fn write(
        &mut self,
        writing: Writing,
        text: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let index = match self.writing {
            Some((current, index)) if current == writing => index,
            _ => {
                self.close_writing(out)?;
                let index = out.fresh_index();
                let block = match writing {
                    Writing::Reasoning => Block::Reasoning { redacted: false },
                    Writing::Text => Block::Text,
                };
                out.open(index, block, serde_json::Value::Null)?;
                self.writing = Some((writing, index));
                index
            }
        };
        self.wrote = true;
        if writing == Writing::Reasoning {
            self.reasoning_text.push_str(text);
        }
        out.push(index, text)
    }

    /// Close the block being written, holding its item: the reasoning's
    /// field or thinking part and its details, or the audio's id.
    fn close_writing(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        match self.writing.take() {
            None => Ok(()),
            Some((Writing::Text, index)) => match self.audio_id.take() {
                Some(id) => out.finish_with(index, serde_json::json!({ "audio": { "id": id } })),
                None => out.finish(index),
            },
            Some((Writing::Reasoning, index)) => {
                let text = std::mem::take(&mut self.reasoning_text);
                let mut item = serde_json::Map::new();
                if std::mem::take(&mut self.thinking_part) {
                    item.insert("type".to_owned(), "thinking".into());
                    item.insert(
                        "thinking".to_owned(),
                        serde_json::json!([{"type": "text", "text": text}]),
                    );
                } else if let Some(field) = self.reasoning_field.take() {
                    item.insert(field.to_owned(), text.into());
                }
                item.append(&mut self.reasoning_details);
                out.finish_with(index, serde_json::Value::Object(item))
            }
        }
    }

    /// One content part, through [`Part::of`].
    #[deny(clippy::wildcard_enum_match_arm)]
    fn part(
        &mut self,
        part: serde_json::Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        match Part::of(&part) {
            Part::Text(text) => self.write(Writing::Text, &text, out),
            Part::Thinking(text) => {
                self.write(Writing::Reasoning, &text, out)?;
                self.thinking_part = true;
                Ok(())
            }
            Part::Image => self.image(part, out),
            Part::Unknown => {
                self.close_writing(out)?;
                self.wrote = true;
                let index = out.fresh_index();
                out.whole(index, Block::Opaque { replay: true }, part, "")
            }
        }
    }

    /// One image the model produced, an `image_url` part, whole: a data URL
    /// becomes base64 data, any other URL stays a URL.
    fn image(
        &mut self,
        part: serde_json::Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        use crate::message::{DocumentSourceKind, Image, ImageMediaType, MimeType};
        let url = part
            .pointer("/image_url/url")
            .or_else(|| part.get("image_url"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let (image, data) = match url
            .strip_prefix("data:")
            .and_then(|rest| rest.split_once(";base64,"))
        {
            Some((mime, data)) => (
                Image {
                    data: DocumentSourceKind::Base64(String::new()),
                    media_type: ImageMediaType::from_mime_type(mime),
                    ..Image::default()
                },
                data.to_owned(),
            ),
            None => (
                Image {
                    data: DocumentSourceKind::Url(url),
                    ..Image::default()
                },
                String::new(),
            ),
        };
        self.close_writing(out)?;
        self.wrote = true;
        let index = out.fresh_index();
        out.whole(index, Block::Image(image), part, &data)
    }

    /// One tool-call fragment, buffered at its wire index.
    fn call(
        &mut self,
        mut call: serde_json::Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.close_writing(out)?;
        let incoming = StreamingToolCall::read(&call);
        let index = incoming.index;
        let custom = call
            .pointer("/custom/name")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        if let Some(existing) = self.open_call(index) {
            let id = existing.get("id").and_then(serde_json::Value::as_str);
            let name = existing
                .pointer("/function/name")
                .and_then(serde_json::Value::as_str);
            if incoming.evicts(id.unwrap_or_default(), name.unwrap_or_default()) {
                // The wire reused this call's index: the call it held is
                // delivered even when its arguments never parse.
                self.close_call(index, false, out)?;
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
                fields.shift_remove("index");
                serde_json::Value::Object(std::mem::take(fields))
            }),
            self.calls.get_mut(at),
        ) {
            merge_fields(existing, &fields);
        }
        self.wrote = true;
        out.fragment(
            Some(index),
            CallFragment {
                id: incoming.id.as_deref(),
                name: incoming.function.name.as_deref().or(custom.as_deref()),
                arguments: incoming.function.arguments.as_deref(),
            },
        )?;
        if self.quirks.emits_complete_single_chunk_tool_calls && incoming.is_complete_single_chunk()
        {
            // A probe: the call closes if its input parses, and stays open
            // for more fragments otherwise.
            self.close_call(index, true, out)?;
        }
        Ok(())
    }

    /// The call assembled at wire `index`, while it is open.
    fn open_call(&self, index: usize) -> Option<&serde_json::Value> {
        self.calls.get(*self.open_calls.get(&index)?)?.as_ref()
    }

    /// Finish the call at wire `index`, through [`CallKind::of`], with the
    /// call as its native; with `probe`, only when its arguments are already
    /// a complete object. A call the output budget cut short before its
    /// arguments were a whole object closes with no native: the provider
    /// never stated it complete. A custom call's
    /// arguments are its `{"input"}`; a call of a kind rig cannot answer is
    /// kept as an item that is never sent back.
    #[deny(clippy::wildcard_enum_match_arm)]
    fn close_call(
        &mut self,
        index: usize,
        probe: bool,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let cut = matches!(self.final_finish_reason, Some(FinishReason::Length));
        let call = self.open_call(index).cloned().unwrap_or_default();
        match CallKind::of(&call) {
            CallKind::Function => {}
            CallKind::Custom => {
                let input = call
                    .pointer("/custom/input")
                    .cloned()
                    .unwrap_or_else(|| serde_json::Value::String(String::new()));
                out.announce(index, serde_json::json!({ "input": input }))?;
            }
            CallKind::Unknown => {
                out.discard(index);
                if let Some(at) = self.open_calls.remove(&index)
                    && let Some(slot) = self.calls.get_mut(at)
                {
                    *slot = None;
                }
                let at = out.fresh_index();
                return out.whole(at, Block::Opaque { replay: false }, call, "");
            }
        }
        out.edit(index, |item| *item = call)?;
        let closed = if probe {
            out.close_if_complete(index)?
        } else if cut {
            // Arguments that are already a whole object were stated whole,
            // budget or not; only a call the cut left unfinished has no item.
            if !out.close_if_complete(index)? {
                out.close(index)?;
            }
            true
        } else {
            out.finish(index)?;
            true
        };
        if closed {
            self.open_calls.remove(&index);
        }
        Ok(())
    }

    /// Close every open call. Arguments the output-token budget cut short
    /// keep what they state.
    fn close_calls(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let open: Vec<usize> = self.open_calls.keys().copied().collect();
        for index in open {
            self.close_call(index, false, out)?;
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
        // A finish states the message complete.
        if self.final_finish_reason.is_some() {
            self.close_writing(out)?;
        }
        if matches!(self.final_finish_reason, Some(FinishReason::ToolCalls)) {
            self.close_calls(out)?;
        }
        Ok(())
    }

    /// The `chat.completion` body, restated as the one chunk whose delta is
    /// its whole message, then the end.
    fn restate_whole(
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
        // Truncation or filtering can leave no visible content; retain its
        // reason and usage. Empty replies without such a reason are errors.
        let cut_short = self
            .final_finish_reason
            .as_ref()
            .is_some_and(FinishReason::truncated_output);
        if !self.wrote && !cut_short {
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        self.end(out, false)
    }

    /// Write the provider's end of the reply: the block being written
    /// closes, and the message becomes the turn's native. A stream's
    /// `raw` is the native terminal record the chunks built; a whole body's
    /// is the body itself, which the transport keeps.
    fn end(&mut self, mut out: Out<'_, Completion>, streamed: bool) -> Result<Flow, ProviderError> {
        self.close_writing(&mut out)?;
        let mut message = std::mem::take(&mut self.message);
        let calls: Vec<_> = self.calls.drain(..).flatten().collect();
        if !calls.is_empty() {
            message.insert("tool_calls".to_owned(), serde_json::Value::Array(calls));
        }
        let usage = self
            .final_usage
            .as_ref()
            .map(|usage| UsageCounts::read(usage).normalized(&self.quirks))
            .unwrap_or_default();
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
        Ok(out.end(crate::operation::Finish {
            usage,
            reason: native.finish_reason,
            response_id: native.response_id,
            model: native.model,
            ..crate::operation::Finish::default()
        }))
    }

    /// The stream ended: flush the calls the provider delivered, then end
    /// the reply. Tool calls the provider fully delivered are content, so
    /// a length cut still flushes them.
    fn finish(&mut self, mut out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
        // A stream no finish reason ended was cut short.
        if !self.saw_terminal {
            return Err(ProviderError::Truncated);
        }
        self.close_calls(&mut out)?;
        self.end(out, true)
    }
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
            ChatEvent::Whole(frame) => self.restate_whole(frame, out),
            ChatEvent::Done => self.finish(out),
            ChatEvent::BareText(text) => {
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
        if let Some(usage) = payload.usage.filter(|usage| !usage.is_null()) {
            let counts = UsageCounts::read(&usage);
            sink.emit(AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: counts.prompt,
                    output_tokens: counts.completion,
                    total_tokens: counts.total,
                    cached_input_tokens: counts.cached,
                    reasoning_tokens: counts.reasoning,
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
    usage: Option<serde_json::Value>,
    #[serde(default)]
    choices: Vec<ObservedChoice>,
    error: Option<ObservedError>,
}

#[derive(Default, Deserialize)]
struct ObservedChoice {
    finish_reason: Option<String>,
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod hard_case_tests;

#[cfg(test)]
mod history_tests;
