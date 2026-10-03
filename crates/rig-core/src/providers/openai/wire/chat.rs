//! Chat Completions: the request built as JSON, and one lenient decoder for
//! a whole reply and a stream of chunks.
//!
//! ```
//! use rig_core::providers::openai::OpenAI;
//! let wire = OpenAI::new("key").chat("gpt-5.2");
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::completion::{CompletionRequest, FinishReason, ProviderCapabilities, Replay};
use crate::error::{EncodeError, ProviderError};
use crate::json_utils::Lenient;
use crate::message::{
    AssistantContent, AssistantMessage, DocumentMediaType, DocumentSourceKind as Source, Message,
    MimeType, ToolCall, ToolResult, ToolResultContent, UserContent,
};
use crate::observe::ObservedError;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::openai_chat_completions_compatible::{
    finish_reason, native_finish_reason, provider_error_envelope,
};
use crate::providers::internal::wire::classify_chat_completions_frame;
use crate::providers::internal::wire_ids::WireIds;
use crate::providers::openai::completion as models;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Body, Capabilities, Decoder, Descriptor, Encoded,
    Flow, Framing, Mode, ObservationSink, Out, Wire, WireEvent, WireFrame,
};

use super::dto::merge_fields;
use super::{BodyRewrite, OpenAIConfig, OutputCap, Quirks};

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

/// The error for media in a form Chat Completions cannot carry, which the
/// adapter replaces before a request is encoded
/// ([`ReplayTarget::encodes`](crate::completion::ReplayTarget::encodes)).
fn unsendable(what: &str) -> EncodeError {
    EncodeError::request(format!("Chat Completions cannot carry {what} in this form"))
}

/// `data` as a URL a content part names: a URL, or typed base64 data as a
/// data URI.
fn media_url(data: &Source, mime: Option<&str>) -> Option<String> {
    match (data, mime) {
        (Source::Url(url), _) => Some(url.clone()),
        (Source::Base64(data), Some(mime)) => Some(format!("data:{mime};base64,{data}")),
        _ => None,
    }
}

/// The `image_url` part an image is sent as; a user image always names a
/// detail level, `auto` by default.
fn image_url(image: &crate::message::Image, user: bool) -> Result<Value, EncodeError> {
    let mime = image.media_type.as_ref().map(MimeType::to_mime_type);
    let url = media_url(&image.data, mime).ok_or_else(|| unsendable("an image"))?;
    let mut part = Map::from_iter([("url".to_owned(), Value::String(url))]);
    let detail = match &image.detail {
        Some(detail) => Some(detail.clone()),
        None => user.then(Default::default),
    };
    if let Some(detail) = detail {
        part.insert("detail".to_owned(), serde_json::to_value(detail)?);
    }
    Ok(json!({"type": "image_url", "image_url": part}))
}

/// One user content part as Chat Completions carries it.
fn user_part(part: &UserContent) -> Result<Value, EncodeError> {
    let file = |file: Value| json!({"type": "file", "file": file});
    Ok(match part {
        UserContent::Text(text) => json!({"type": "text", "text": text.text}),
        UserContent::Image(image) => image_url(image, true)?,
        UserContent::Document(document) => {
            let pdf = document.media_type == Some(DocumentMediaType::PDF);
            match &document.data {
                Source::FileId(id) => file(json!({ "file_id": id })),
                Source::Base64(data) if pdf => file(json!({
                    "file_data": format!("data:application/pdf;base64,{data}"),
                    "filename": "document.pdf",
                })),
                // OpenRouter and Mistral fetch a PDF a URL names.
                Source::Url(url) if pdf => {
                    file(json!({"file_data": url, "filename": "document.pdf"}))
                }
                Source::String(text) if !pdf => json!({"type": "text", "text": text}),
                _ => return Err(unsendable("a document")),
            }
        }
        UserContent::Audio(audio) => match &audio.data {
            Source::Base64(data) => json!({"type": "input_audio", "input_audio": {
                "data": data,
                "format": audio.media_type.clone().unwrap_or(crate::message::AudioMediaType::MP3),
            }}),
            _ => return Err(unsendable("audio")),
        },
        UserContent::Video(video) => {
            let mime = video.media_type.as_ref().map(MimeType::to_mime_type);
            let url = media_url(&video.data, mime).ok_or_else(|| unsendable("a video"))?;
            json!({"type": "video_url", "video_url": {"url": url}})
        }
        UserContent::ToolResult(_) => return Err(unsendable("a tool result as a content part")),
    })
}

/// A result as the `tool` message that answers its call: its text joined
/// into one string, or its parts as an array when the wire asks for one or
/// it holds an image, which has no string form.
fn tool_message(result: &ToolResult, ids: &WireIds, array: bool) -> Result<Value, EncodeError> {
    let parts = result
        .content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => Ok(json!({"type": "text", "text": text.text})),
            ToolResultContent::Json { value } => {
                Ok(json!({"type": "text", "text": value.to_string()}))
            }
            ToolResultContent::Image(image) => image_url(image, false),
        })
        .collect::<Result<Vec<_>, EncodeError>>()?;
    let content = if array
        || parts
            .iter()
            .any(|part| part.str("type") == Some("image_url"))
    {
        Value::Array(parts)
    } else {
        let texts: Vec<&str> = parts.iter().filter_map(|part| part.str("text")).collect();
        Value::String(texts.join("\n"))
    };
    let id = ids.spell(&result.call);
    Ok(json!({"role": "tool", "tool_call_id": id, "content": content}))
}

/// Push the user parts gathered so far as one message: a lone text part as
/// a plain string.
fn user_message(messages: &mut Vec<Value>, parts: &mut Vec<Value>) {
    let content = match parts.as_slice() {
        [] => return,
        [part] if part.str("type") == Some("text") => part.get("text").cloned().unwrap_or_default(),
        _ => Value::Array(std::mem::take(parts)),
    };
    parts.clear();
    messages.push(json!({"role": "user", "content": content}));
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
        let mut body = self.body(request)?;
        if mode == Mode::Streaming {
            // A caller's own stream options, `include_usage` among them, stand.
            if quirks.stream_include_usage
                && let Some(options) = body
                    .entry("stream_options")
                    .or_insert_with(|| json!({}))
                    .as_object_mut()
            {
                options.entry("include_usage").or_insert(Value::Bool(true));
            }
            body.insert("stream".to_owned(), Value::Bool(true));
        }
        self.rewrite(&mut body)?;
        let body = Value::Object(body);
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
            .with_route(Some(quirks.completion_path)))
    }

    /// The field every assistant message carries its reasoning under,
    /// empty when the turn has none (pi's
    /// `requiresReasoningContentOnAssistantMessages`): the dialect's own, and
    /// `reasoning_content` for DeepSeek at any base URL, for Kimi K3 under
    /// any gateway's path, and for OpenRouter's Kimi K2.6, as pi's catalogue
    /// marks them.
    fn reasoning_field(&self, model: &str) -> Option<&'static str> {
        let deepseek = self
            .provider
            .base_url
            .to_ascii_lowercase()
            .contains("deepseek.com");
        let name = model.rsplit('/').next().unwrap_or(model);
        let listed = is_model(name, crate::providers::moonshot::KIMI_K3)
            || is_model(model, "moonshotai/kimi-k2.6");
        self.provider
            .dialect
            .quirks
            .reasoning_field
            .or((deepseek || listed).then_some("reasoning_content"))
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

    /// The request body, before the stream options and the dialect's final
    /// rewrite. `additional_params` is flattened in last, so its keys
    /// override the typed ones.
    fn body(&self, request: CompletionRequest) -> Result<Map<String, Value>, EncodeError> {
        let quirks = &self.provider.dialect.quirks;
        let mut model = request.model.clone().unwrap_or_else(|| self.model.clone());
        if quirks.rewrite == BodyRewrite::HuggingFaceRouter {
            // Some sub-providers (Fireworks) address models by a qualified id.
            model = self.provider.route().model_identifier(&model);
        }
        let mut params = match request.additional_params {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(params)) => params,
            Some(_) => {
                return Err(EncodeError::request(
                    "Chat Completions `additional_params` must be a JSON object",
                ));
            }
        };
        if quirks.rewrite == BodyRewrite::Mira && !params.is_empty() {
            // The gateway rejects pass-through parameters.
            tracing::warn!("Additional parameters are not supported by Mira and will be ignored");
            params.clear();
        }
        // A call to a custom tool the request declares is a custom call (pi).
        let custom: Vec<String> = params
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|tool| tool.str("type") == Some("custom"))
            .filter_map(|tool| tool.at("/custom/name").and_then(Value::as_str))
            .map(str::to_owned)
            .collect();
        let messages = self.messages(&request.chat_history, &model, &custom)?;

        let mut tools: Vec<Value> = Vec::new();
        let mut tool_choice = None;
        if quirks.supports_tools {
            for tool in &request.tools {
                let mut parameters = tool.parameters.clone();
                let mut function = json!({"name": tool.name, "description": tool.description});
                if self.strict_tools {
                    crate::providers::openai::sanitize_schema(&mut parameters);
                }
                if let Some(function) = function.as_object_mut() {
                    function.insert("parameters".to_owned(), parameters);
                    if self.strict_tools {
                        function.insert("strict".to_owned(), Value::Bool(true));
                    }
                }
                tools.push(json!({"type": "function", "function": function}));
            }
            // Passthrough tools, function or native (Groq's `browser_search`),
            // join the typed ones in one array, which a flattened `tools` key
            // would otherwise replace.
            if let Some(passthrough) = params.shift_remove("tools") {
                let Value::Array(passthrough) = passthrough else {
                    return Err(EncodeError::request(
                        "Invalid OpenAI Chat Completions `additional_params.tools` payload: \
                         expected an array",
                    ));
                };
                tools.extend(passthrough);
            }
            tool_choice = request
                .tool_choice
                .clone()
                .map(tool_choice_value)
                .transpose()?
                .filter(|_| !tools.is_empty());
        } else {
            if !request.tools.is_empty() {
                tracing::warn!("Tool use is not supported by this provider; tools will be ignored");
            }
            if request.tool_choice.is_some() {
                tracing::warn!("Tool choice is not supported by this provider and will be ignored");
            }
        }

        if request.output_schema.is_some() && !quirks.supports_response_format {
            tracing::warn!(
                "Structured outputs are not supported by this provider; ignoring output_schema"
            );
        }
        // Defer schemas until a tool result exists unless the dialect supports both at once.
        let answered = messages
            .iter()
            .any(|message| message.str("role") == Some("tool"));
        if let Some(schema) = request.output_schema
            && quirks.supports_response_format
            && (quirks.response_format_with_tools || tools.is_empty() || answered)
        {
            let (name, schema) = crate::providers::openai::structured_output_schema(schema);
            params.insert(
                "response_format".to_owned(),
                json!({"type": "json_schema",
                    "json_schema": {"name": name, "strict": true, "schema": schema}}),
            );
        }

        let fields = [
            ("model", Some(Value::String(model))),
            ("messages", Some(Value::Array(messages))),
            ("tools", (!tools.is_empty()).then_some(Value::Array(tools))),
            ("tool_choice", tool_choice),
            ("temperature", request.temperature.map(Value::from)),
            ("max_tokens", request.max_tokens.map(Value::from)),
        ];
        let mut body: Map<String, Value> = fields
            .into_iter()
            .filter_map(|(key, value)| Some((key.to_owned(), value?)))
            .collect();
        body.extend(params);
        // The resolved model, not the handle's: a per-request override
        // changes which endpoint answers, so it decides the spelling too.
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if quirks.output_cap == OutputCap::OpenAiReasoningFamilies
            && models::is_openai_reasoning_model(model)
            && let Some(max_tokens) = body.shift_remove("max_tokens")
        {
            body.entry("max_completion_tokens").or_insert(max_tokens);
        }
        Ok(body)
    }

    /// The history as Chat messages, each call and result spelled by one
    /// [`WireIds`].
    fn messages(
        &self,
        history: &[Message],
        model: &str,
        custom: &[String],
    ) -> Result<Vec<Value>, EncodeError> {
        let ids = WireIds::for_target(history, self, model);
        let mut messages = Vec::new();
        for message in history {
            match message {
                Message::System { content } => messages.push(json!({"role": "system",
                    "content": [{"type": "text", "text": content}]})),
                Message::User { content } => {
                    let mut parts = Vec::new();
                    for part in content {
                        if let UserContent::ToolResult(result) = part {
                            user_message(&mut messages, &mut parts);
                            let array = self.tool_result_array_content;
                            messages.push(tool_message(result, &ids, array)?);
                        } else {
                            parts.push(user_part(part)?);
                        }
                    }
                    user_message(&mut messages, &mut parts);
                }
                Message::Assistant(turn) => {
                    messages.extend(self.assistant(turn, &ids, custom, model));
                }
            }
        }
        if messages.is_empty() {
            return Err(EncodeError::request(
                "OpenAI Chat Completions request has no provider-compatible messages after \
                 conversion",
            ));
        }
        Ok(messages)
    }

    /// One assistant turn as pi's `openai-completions` rebuilds it: text
    /// joined into a `content` string, or a part array when a block is a
    /// content part (Mistral's thinking); reasoning under the field it
    /// arrived in, or the dialect's; an item's other fields beside it; and
    /// each call with its canonical name and arguments. A message with
    /// neither content nor calls is `None`, as pi skips it.
    fn assistant(
        &self,
        turn: &AssistantMessage,
        ids: &WireIds,
        custom: &[String],
        model: &str,
    ) -> Option<Value> {
        let reasoning_field = self.reasoning_field(model);
        let (mut text, mut parts, mut has_parts) = (String::new(), Vec::new(), false);
        let mut reasoning: Vec<(String, String)> = Vec::new();
        let (mut fields, mut calls) = (Map::new(), Vec::new());
        for block in &turn.content {
            let replay = block.replay(self, ids);
            match block {
                AssistantContent::Text(block) => {
                    text.push_str(&block.text);
                    parts.push(json!({"type": "text", "text": block.text}));
                    // A text item is message fields: an answer's audio.
                    if let Replay::Item(item) = replay
                        && let Value::Object(item) = item.into_owned()
                    {
                        fields.extend(item);
                    }
                }
                AssistantContent::Reasoning(block) => {
                    let field = match replay {
                        Replay::Item(item) if item.get("type").is_some() => {
                            has_parts = true;
                            parts.push(item.into_owned());
                            None
                        }
                        Replay::Item(item) => {
                            let mut field = None;
                            for (key, value) in item.as_object().into_iter().flatten() {
                                if value.is_string() {
                                    field.get_or_insert_with(|| key.clone());
                                } else {
                                    let detail = Map::from_iter([(key.clone(), value.clone())]);
                                    merge_fields(&mut fields, &detail);
                                }
                            }
                            field
                        }
                        Replay::Identity(identity)
                            if identity.get("type").and_then(Value::as_str) == Some("thinking") =>
                        {
                            has_parts = true;
                            parts.push(json!({"type": "thinking",
                                "thinking": [{"type": "text", "text": block.text}]}));
                            None
                        }
                        Replay::Identity(identity) => identity.keys().next().cloned(),
                        Replay::Rebuild => reasoning_field.map(str::to_owned),
                    };
                    if let Some(field) = field {
                        match reasoning.iter_mut().find(|(name, _)| *name == field) {
                            Some((_, joined)) => {
                                joined.push('\n');
                                joined.push_str(&block.text);
                            }
                            None => reasoning.push((field, block.text.clone())),
                        }
                    }
                }
                AssistantContent::ToolCall(call) => {
                    calls.push(call_item(call, replay, ids, custom));
                }
                // An opaque is a content part; one with no `type` is not
                // sent, so its keys never land on the message.
                AssistantContent::Opaque(opaque) if opaque.replay => {
                    if opaque.item.get("type").is_some() {
                        has_parts = true;
                        parts.push(opaque.item.clone());
                    }
                }
                AssistantContent::Opaque(_) | AssistantContent::Image(_) => {}
            }
        }
        let mut message = Map::from_iter([("role".to_owned(), Value::from("assistant"))]);
        if has_parts {
            message.insert("content".to_owned(), Value::Array(parts));
        } else if !text.is_empty() {
            message.insert("content".to_owned(), Value::String(text));
        }
        for (field, text) in reasoning {
            message.insert(field, Value::String(text));
        }
        message.extend(fields);
        if !calls.is_empty() {
            message.insert("tool_calls".to_owned(), Value::Array(calls));
        }
        let has_content = message.contains_key("audio")
            || message.contains_key("tool_calls")
            || match message.get("content") {
                Some(Value::String(text)) => !text.is_empty(),
                Some(Value::Array(parts)) => !parts.is_empty(),
                _ => false,
            };
        if let Some(field) = reasoning_field {
            message
                .entry(field)
                .or_insert_with(|| Value::String(String::new()));
        }
        has_content.then_some(Value::Object(message))
    }

    /// The dialect's rewrite of the finished body ([`BodyRewrite`]): a
    /// refusal where the provider would answer with an error or silently do
    /// something else, Moonshot's steering for `required`, and each
    /// dialect's spellings and content shapes.
    fn rewrite(&self, map: &mut Map<String, Value>) -> Result<(), EncodeError> {
        let forced = map
            .get("tool_choice")
            .and_then(|choice| choice.at("/function/name"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        match self.provider.dialect.quirks.rewrite {
            BodyRewrite::LlamaCpp => {
                if let Some(name) = forced {
                    return Err(EncodeError::request(format!(
                        "llama.cpp cannot force a specific tool: `llama-server` accepts only \
                         `auto`, `none` or `required` for tool_choice and silently treats \
                         anything else as `auto`, so requesting `{name}` would return whichever \
                         tool the model picked. Use `ToolChoice::Required` to force a call, or \
                         advertise only `{name}` in `tools`."
                    )));
                }
            }
            BodyRewrite::Moonshot => {
                if forced.is_some() {
                    return Err(EncodeError::request(
                        "Moonshot does not support forcing a specific tool",
                    ));
                }
                if map.get("tool_choice").and_then(Value::as_str) == Some("required") {
                    tracing::warn!(
                        "Moonshot does not support tool_choice=required; coercing to auto with an \
                         additional steering message"
                    );
                    map.insert("tool_choice".to_owned(), Value::from("auto"));
                    if let Some(Value::Array(messages)) = map.get_mut("messages") {
                        messages.push(json!({"role": "user",
                            "content": "Please select a tool to handle the current issue."}));
                    }
                }
            }
            // The gateway takes every message's content as one string.
            // Perplexity spends output budget on a text-part array (a short
            // `max_tokens` comes back empty at `length`, checked live), so
            // text-only arrays go as one string; the gateway takes every
            // content as one string.
            BodyRewrite::Perplexity | BodyRewrite::Mira => {
                let all = self.provider.dialect.quirks.rewrite == BodyRewrite::Mira;
                for content in messages_mut(map).filter_map(|message| message.get_mut("content")) {
                    if let Value::Array(parts) = content
                        && (all || parts.iter().all(|part| part.str("type") == Some("text")))
                    {
                        let texts: Vec<&str> =
                            parts.iter().filter_map(|part| part.str("text")).collect();
                        *content = Value::String(texts.join("\n"));
                    }
                }
            }
            BodyRewrite::DeepSeek => finalize_deepseek(map),
            BodyRewrite::Mistral => finalize_mistral(map),
            BodyRewrite::OpenRouter if self.prompt_caching => apply_openrouter_prompt_caching(map),
            BodyRewrite::None | BodyRewrite::OpenRouter | BodyRewrite::HuggingFaceRouter => {}
        }
        Ok(())
    }
}

/// `call` as the item the wire sends, from what replay hands the encoder:
/// the current item, an edited one's kind, or nothing. A call is `custom`
/// when its item or kept kind says so, or when it is rebuilt and the request
/// declares a custom tool by its name (pi); its arguments' `input` is then
/// its input. Any other is a function call with its arguments as JSON text.
fn call_item(call: &ToolCall, replay: Replay<'_>, ids: &WireIds, custom: &[String]) -> Value {
    let mut item = match replay {
        Replay::Item(item) => match item.into_owned() {
            Value::Object(item) => item,
            _ => Map::new(),
        },
        Replay::Identity(identity) => identity,
        Replay::Rebuild => Map::new(),
    };
    let name = call.function.name.as_str();
    let custom = match item.get("type").and_then(Value::as_str) {
        Some(kind) => kind == "custom",
        None => {
            let declared = custom.iter().any(|tool| tool == name);
            let kind = if declared { "custom" } else { "function" };
            item.insert("type".to_owned(), Value::from(kind));
            declared
        }
    };
    let id = ids.spell(&call.id);
    item.insert("id".to_owned(), Value::String(id));
    let (slot, key, value) = if custom {
        let input = match call.function.arguments.get("input") {
            Some(Value::String(input)) => input.clone(),
            Some(input) => input.to_string(),
            None => String::new(),
        };
        ("custom", "input", input)
    } else {
        let arguments = Value::Object(call.function.arguments.clone()).to_string();
        ("function", "arguments", arguments)
    };
    let slot = item
        .entry(slot)
        .or_insert_with(|| Value::Object(Map::new()));
    if !slot.is_object() {
        *slot = Value::Object(Map::new());
    }
    if let Value::Object(fields) = slot {
        fields.insert("name".to_owned(), Value::from(name));
        fields.insert(key.to_owned(), Value::String(value));
    }
    Value::Object(item)
}

/// The Chat `tool_choice` for a canonical one.
fn tool_choice_value(choice: crate::message::ToolChoice) -> Result<Value, EncodeError> {
    use crate::message::ToolChoice;
    Ok(match choice {
        ToolChoice::Auto => Value::from("auto"),
        ToolChoice::None => Value::from("none"),
        ToolChoice::Required => Value::from("required"),
        ToolChoice::Specific { function_names } => {
            let [name] = function_names.as_slice() else {
                return Err(EncodeError::request(
                    "Provider only supports forcing exactly one specific tool",
                ));
            };
            json!({"type": "function", "function": {"name": name}})
        }
    })
}

/// The body's messages, each as its object.
fn messages_mut(map: &mut Map<String, Value>) -> impl Iterator<Item = &mut Map<String, Value>> {
    let messages = map.get_mut("messages").and_then(Value::as_array_mut);
    messages
        .into_iter()
        .flatten()
        .filter_map(Value::as_object_mut)
}

/// Whether `model` is `id` or one of its dated snapshots (`<id>-YYYY-MM-DD`).
fn is_model(model: &str, id: &str) -> bool {
    model
        .strip_prefix(id)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with("-20"))
}

/// DeepSeek rejects a forced tool choice while thinking, and every model
/// but `deepseek-chat` thinks unless `thinking` says `disabled` (checked
/// live), so the choice is relaxed to the default there.
fn finalize_deepseek(map: &mut Map<String, Value>) {
    let thinking = match map
        .get("thinking")
        .and_then(|thinking| thinking.str("type"))
    {
        Some(mode) => !mode.eq_ignore_ascii_case("disabled"),
        None => map.get("model").and_then(Value::as_str) != Some("deepseek-chat"),
    };
    let forced = map
        .get("tool_choice")
        .is_some_and(|choice| choice.is_object() || choice.as_str() == Some("required"));
    if thinking && forced {
        tracing::debug!(
            "dropping tool_choice: DeepSeek rejects a forced tool choice while thinking"
        );
        map.shift_remove("tool_choice");
    }
}

/// Mistral's wire-level differences: a forced tool choice relaxed beside a
/// structured format, and its content chunks.
fn finalize_mistral(map: &mut Map<String, Value>) {
    // Mistral rejects forced tool calls beside JSON response formats: the
    // choice is relaxed rather than the schema discarded.
    let forces_a_tool_call = map
        .get("tool_choice")
        .is_some_and(|choice| !matches!(choice.as_str(), Some("auto" | "none")));
    let has_tools = map
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    let structured = map
        .get("response_format")
        .and_then(|format| format.str("type"))
        .is_some_and(|kind| matches!(kind, "json_schema" | "json_object"));
    if forces_a_tool_call && has_tools && structured {
        tracing::debug!(
            "relaxing tool_choice to `auto`: Mistral rejects a forced tool choice \
             alongside a response format"
        );
        map.insert("tool_choice".to_owned(), Value::from("auto"));
    }
    for content in messages_mut(map).filter_map(|message| message.get_mut("content")) {
        for part in content.as_array_mut().into_iter().flatten() {
            *part = mistral_chunk(part);
        }
    }
}

/// One content part as the Mistral chunk that carries it, by its `type`:
/// text and refusal parts are `text` chunks, a file's data is a
/// `document_url` and its id a `file` chunk, audio is its base64 string,
/// and an image keeps only its `image_url`, for every chunk forbids unknown
/// keys. A part the conversion does not emit stays as it is.
fn mistral_chunk(part: &Value) -> Value {
    let field = |pointer: &str| part.at(pointer).and_then(Value::as_str);
    match part.str("type") {
        Some("image_url") => {
            json!({"type": "image_url", "image_url": part.get("image_url")})
        }
        Some("input_audio") => json!({"type": "input_audio",
            "input_audio": field("/input_audio/data")}),
        Some("file") => match field("/file/file_data") {
            Some(data) => json!({"type": "document_url", "document_url": data,
                "document_name": field("/file/filename")}),
            None => json!({"type": "file", "file_id": field("/file/file_id")}),
        },
        _ => part.clone(),
    }
}

/// OpenRouter's ephemeral `cache_control` on the system prompt: on its
/// text, or on the last of its parts.
fn apply_openrouter_prompt_caching(map: &mut Map<String, Value>) {
    let Some(system) = map
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|messages| {
            messages
                .iter_mut()
                .find(|message| message.str("role") == Some("system"))
        })
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    let ephemeral = json!({"type": "ephemeral"});
    let content = match system.get_mut("content") {
        Some(Value::String(text)) => {
            json!([{"type": "text", "text": text, "cache_control": ephemeral}])
        }
        Some(Value::Array(parts)) => {
            if let Some(Value::Object(last)) = parts.last_mut() {
                last.insert("cache_control".to_owned(), ephemeral);
            }
            return;
        }
        _ => return,
    };
    system.insert("content".to_owned(), content);
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
    /// document as text. DeepSeek and Mira take text only; Perplexity and
    /// Cohere take no media but images, and Ollama only images as data.
    fn encodes(&self, _model: &str, media: crate::completion::Media<'_>) -> bool {
        use crate::completion::Media;
        let dialect = &self.provider.dialect;
        let rewrite = dialect.quirks.rewrite;
        let parts = !matches!(rewrite, BodyRewrite::DeepSeek | BodyRewrite::Mira);
        // Cohere and Ollama read images and text, and no other media.
        let files = parts
            && rewrite != BodyRewrite::Perplexity
            && ![super::dialects::COHERE.name, super::dialects::OLLAMA.name]
                .contains(&dialect.name);
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
                    // Ollama reads an image as data and never fetches a URL.
                    && !(dialect.name == super::dialects::OLLAMA.name
                        && matches!(image.data, Source::Url(_)))
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
    /// takes any such id (checked live on eight models), where pi derives
    /// nine alphanumerics.
    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _: Option<&crate::message::Origin>,
    ) -> String {
        let sanitized =
            |part: &str| crate::providers::internal::wire_ids::legal_call_id(part, usize::MAX);
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
            let hash: String = crate::providers::internal::wire_ids::short_hash(id)
                .chars()
                .take(8)
                .collect();
            let prefix: String = call.chars().take((40 - hash.len() - 1).max(1)).collect();
            return format!("{prefix}_{hash}");
        }
        if self.provider.dialect.name == super::dialects::OPENAI.name {
            return id.chars().take(40).collect();
        }
        id.to_owned()
    }

    /// What an edited block keeps of its item: a custom call's kind and
    /// Mistral's thinking part (`type`), or the field reasoning arrived in,
    /// as pi keeps `thinkingSignature`. Signed data such as
    /// `reasoning_details` never survives an edit.
    fn identity(&self, item: &Value) -> Map<String, Value> {
        if let Some(kind @ ("custom" | "thinking")) = item.str("type") {
            return Map::from_iter([("type".to_owned(), Value::from(kind))]);
        }
        REASONING_TEXT_KEYS
            .iter()
            .find(|key| item.get(**key).is_some_and(Value::is_string))
            .map(|key| Map::from_iter([((*key).to_owned(), Value::String(String::new()))]))
            .unwrap_or_default()
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/id")
    }

    fn later_system(&self, _model: &str) -> crate::completion::LaterSystem {
        self.provider.dialect.quirks.later_system
    }

    /// In array mode a result's parts go as an array.
    fn result_parts(&self, _model: &str) -> bool {
        self.tool_result_array_content
    }

    /// A Chat message needs content or calls, as pi's rebuild does: text
    /// only when it is not empty or its item carries audio, reasoning only
    /// as a content part (Mistral's thinking), an opaque item only as a
    /// typed part, and never an image.
    fn sends_alone(&self, block: &AssistantContent) -> bool {
        let ids = crate::providers::internal::wire_ids::WireIds::default();
        match (block, block.replay(self, &ids)) {
            (AssistantContent::ToolCall(_), _) => true,
            (AssistantContent::Text(text), Replay::Item(item)) => {
                !text.text.is_empty() || item.get("audio").is_some()
            }
            (AssistantContent::Text(text), _) => !text.text.is_empty(),
            (AssistantContent::Reasoning(_), Replay::Item(item)) => item.get("type").is_some(),
            (AssistantContent::Reasoning(_), Replay::Identity(identity)) => {
                identity.get("type").and_then(Value::as_str) == Some("thinking")
            }
            (AssistantContent::Opaque(opaque), _) => opaque.item.get("type").is_some(),
            (AssistantContent::Reasoning(_), Replay::Rebuild) | (AssistantContent::Image(_), _) => {
                false
            }
        }
    }
}

/// Whether `model` reads user images from `vendor`, a dialect's name or the
/// vendor an OpenRouter model id starts with, by the provider's documented
/// model rules. A model the rules do not name reads them. DeepSeek's API
/// takes text content only (it answers an image part with a 400) and
/// Mira's gateway takes text; Groq reads images on its Llama 4 models;
/// Mistral's Codestral and Devstral are text models; OpenAI, xAI, Z.AI,
/// Moonshot, MiniMax and MiMo apply the rules their other wires share.
fn vendor_reads_images(vendor: &str, model: &str) -> bool {
    use crate::providers::{minimax, moonshot, xiaomimimo, zai};
    match vendor {
        "deepseek" | "mira" => false,
        "groq" => model.contains("llama-4"),
        "mistral" | "mistralai" => {
            !(model.starts_with("codestral") || model.starts_with("devstral"))
        }
        "openai" | "azure.openai" => crate::providers::openai::reads_images(model),
        "xai" | "x-ai" => crate::providers::xai::reads_images(model),
        "cohere" => crate::providers::cohere::reads_images(model),
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

/// Classified Chat Completions frame, including whole replies and terminal signals.
pub enum ChatEvent {
    /// A `chat.completion.chunk`: one step of a streamed turn.
    Chunk(Value),
    /// A `chat.completion`: the whole turn in one frame.
    Whole(Value),
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
    /// The part `part` is, by its `type`. A part with no `type` and a
    /// string `text` is text.
    pub(crate) fn of(part: &Value) -> Self {
        let text = |key: &str| part.str(key).unwrap_or_default().to_owned();
        match part.str("type") {
            Some("text") => Self::Text(text("text")),
            None if part.str("text").is_some() => Self::Text(text("text")),
            Some("refusal") => Self::Text(text("refusal")),
            Some("thinking") => Self::Thinking(match part.get("thinking") {
                Some(Value::Array(chunks)) => chunks
                    .iter()
                    .filter_map(|chunk| chunk.str("text"))
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
    /// The kind of `call`, by its `type`; a call without one is a function
    /// call.
    pub(crate) fn of(call: &Value) -> Self {
        match call.str("type") {
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

/// A tool call the reply has open: its writer index, the id it states,
/// whether its kind is one rig cannot answer (an opaque item), and the
/// function arguments streamed so far.
struct OpenCall {
    at: usize,
    id: Option<String>,
    opaque: bool,
    arguments: String,
}

impl OpenCall {
    /// Whether its function arguments are already a whole object.
    fn complete(&self) -> bool {
        matches!(
            crate::json_utils::parse_tool_arguments(&self.arguments),
            Ok(Value::Object(_))
        )
    }
}

/// The chat-completions decoder: one state machine for a whole reply and a
/// stream of chunks, reading every frame leniently as JSON. A whole reply
/// is the one chunk carrying its message. As pi's `openai-completions`
/// decodes a message, a run of reasoning is a block, a run of text a block,
/// and each tool call, image and unknown content part a block of its own;
/// a block closes when the next one starts. An answer's audio transcript
/// is its text when it has no other, and its first text block holds the
/// audio's id.
#[derive(Default)]
pub struct ChatDecoder {
    quirks: Quirks,
    /// The block being written, and its writer index.
    writing: Option<(Writing, usize)>,
    /// The reasoning block's item as it will hold it: the field its text
    /// arrived in, or Mistral's thinking part, its text, and its own
    /// `reasoning_details`.
    reasoning_field: Option<&'static str>,
    thinking_part: bool,
    reasoning_text: String,
    reasoning_details: Map<String, Value>,
    /// The id of the answer's audio.
    audio_id: Option<Value>,
    /// The calls still open, in the order they opened.
    calls: Vec<OpenCall>,
    usage: Option<Value>,
    finish: Option<FinishReason>,
    response_id: Option<String>,
    response_model: Option<String>,
    /// Accumulated primary-choice token metadata, in the wire's token order.
    logprobs: Option<Map<String, Value>>,
    /// Provider-specific top-level fields of every frame.
    fields: Map<String, Value>,
    /// Whether a finish reason or a whole reply ended the turn.
    ended: bool,
}

impl ChatDecoder {
    fn new(quirks: Quirks) -> Self {
        Self {
            quirks,
            ..Self::default()
        }
    }

    /// Absorb the metadata every frame carries, and take its primary choice.
    /// Usage is the frame's, or its choice's where Moonshot reports it (pi).
    fn absorb(&mut self, frame: &Value) -> Option<Value> {
        if let Some(id) = frame.str("id") {
            self.response_id = Some(id.to_owned());
        }
        if let Some(model) = frame.str("model") {
            self.response_model = Some(model.to_owned());
        }
        for (key, value) in frame.as_object().into_iter().flatten() {
            if !matches!(key.as_str(), "id" | "model" | "choices" | "usage") {
                let field = Map::from_iter([(key.clone(), value.clone())]);
                merge_fields(&mut self.fields, &field);
            }
        }
        // `n > 1` streams interleave candidates told apart only by
        // `choices[].index`; candidate 0 is the turn, as in a whole reply.
        let choice = frame
            .arr("choices")
            .iter()
            .find(|choice| {
                let index = choice.get("index").and_then(Value::as_u64);
                index.is_none_or(|index| index == 0)
            })
            .cloned();
        if let Some(usage) = frame
            .at("/usage")
            .or_else(|| choice.as_ref().and_then(|choice| choice.at("/usage")))
        {
            self.usage = Some(usage.clone());
        }
        let choice = choice?;
        // A gateway's upstream-native reason is consulted only when the
        // normalized field is absent or empty (OpenRouter's precedence).
        let reason = match choice
            .str("finish_reason")
            .filter(|reason| !reason.is_empty())
        {
            Some(reason) => Some(finish_reason(reason, &self.quirks)),
            None => choice
                .str("native_finish_reason")
                .filter(|reason| self.quirks.native_finish_reason && !reason.is_empty())
                .map(native_finish_reason),
        };
        if let Some(reason) = reason {
            self.finish = Some(reason);
            self.ended = true;
        }
        if let Some(logprobs) = choice.obj("logprobs") {
            merge_fields(self.logprobs.get_or_insert_default(), logprobs);
        }
        Some(choice)
    }

    /// One delta of the assistant message, in the order a stream sends it:
    /// reasoning and the details that come with it, the content, the
    /// details that sign one of the delta's calls (Gemini's, which arrive
    /// with the call), images and calls.
    fn delta(
        &mut self,
        delta: &Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let reasoning = REASONING_TEXT_KEYS.iter().find_map(|key| {
            delta
                .get(*key)
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(|text| (*key, text))
        });
        if let Some((key, text)) = reasoning {
            self.write(Writing::Reasoning, text, out)?;
            self.reasoning_field.get_or_insert(key);
        }
        let calls = delta.get("tool_calls").and_then(Value::as_array);
        let signs = |detail: &Value| {
            detail.str("id").is_some_and(|id| {
                calls
                    .into_iter()
                    .flatten()
                    .any(|call| call.str("id") == Some(id))
            })
        };
        let (signing, leading): (Vec<Value>, Vec<Value>) = delta
            .get(REASONING_DETAILS)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .cloned()
            .partition(signs);
        self.details(leading, out)?;
        let audio = delta.get("audio");
        if let Some(id) = audio.and_then(|audio| audio.at("/id")) {
            self.audio_id = Some(id.clone());
        }
        match delta.get("content") {
            Some(Value::Array(parts)) => {
                for part in parts {
                    self.part(part, out)?;
                }
            }
            Some(part @ Value::Object(_)) => self.part(part, out)?,
            _ => {
                let text = ["content", "refusal"]
                    .iter()
                    .find_map(|key| {
                        delta
                            .get(*key)
                            .and_then(Value::as_str)
                            .filter(|text| !text.is_empty())
                    })
                    .or_else(|| audio.and_then(|audio| audio.str("transcript")))
                    .filter(|text| !text.is_empty());
                if let Some(text) = text {
                    self.write(Writing::Text, text, out)?;
                }
            }
        }
        self.details(signing, out)?;
        for image in delta
            .get("images")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            self.image(image, out)?;
        }
        for call in calls.into_iter().flatten() {
            self.call(call, out)?;
        }
        Ok(())
    }

    /// Reasoning `details`, into the reasoning block.
    fn details(
        &mut self,
        details: Vec<Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        if !details.is_empty() {
            self.write(Writing::Reasoning, "", out)?;
            merge_details(&mut self.reasoning_details, details);
        }
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
                out.open(index, block, Value::Null)?;
                self.writing = Some((writing, index));
                index
            }
        };
        if writing == Writing::Reasoning {
            self.reasoning_text.push_str(text);
        }
        out.push(index, text)
    }

    /// Close the block being written, holding its item: the reasoning's
    /// field or thinking part and its details, or the audio's id.
    fn close_writing(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let (index, item) = match self.writing.take() {
            None => return Ok(()),
            Some((Writing::Text, index)) => (
                index,
                self.audio_id
                    .take()
                    .map(|id| json!({ "audio": { "id": id } })),
            ),
            Some((Writing::Reasoning, index)) => {
                let text = std::mem::take(&mut self.reasoning_text);
                let mut item = Map::new();
                if std::mem::take(&mut self.thinking_part) {
                    item.insert("type".to_owned(), "thinking".into());
                    item.insert(
                        "thinking".to_owned(),
                        json!([{"type": "text", "text": text}]),
                    );
                } else if let Some(field) = self.reasoning_field.take() {
                    item.insert(field.to_owned(), text.into());
                }
                item.append(&mut self.reasoning_details);
                (index, Some(Value::Object(item)))
            }
        };
        if let Some(item) = item {
            out.edit(index, |slot| *slot = item)?;
        }
        out.finish(index)
    }

    /// One content part, through [`Part::of`].
    #[deny(clippy::wildcard_enum_match_arm)]
    fn part(&mut self, part: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        match Part::of(part) {
            Part::Text(text) => self.write(Writing::Text, &text, out),
            Part::Thinking(text) => {
                self.write(Writing::Reasoning, &text, out)?;
                self.thinking_part = true;
                Ok(())
            }
            Part::Image => self.image(part, out),
            Part::Unknown => {
                self.close_writing(out)?;
                let index = out.fresh_index();
                out.whole(index, Block::Opaque { replay: true }, part.clone(), "")
            }
        }
    }

    /// One image the model produced, an `image_url` part, whole: a data URL
    /// becomes base64 data, any other URL stays a URL.
    fn image(&mut self, part: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        use crate::message::{Image, ImageMediaType};
        let url = part
            .at("/image_url/url")
            .or_else(|| part.get("image_url"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let inline = url
            .strip_prefix("data:")
            .and_then(|rest| rest.split_once(";base64,"));
        let mut image = Image::default();
        let data = match inline {
            Some((mime, data)) => {
                image.data = Source::Base64(String::new());
                image.media_type = ImageMediaType::from_mime_type(mime);
                data
            }
            None => {
                image.data = Source::Url(url.to_owned());
                ""
            }
        };
        self.close_writing(out)?;
        let index = out.fresh_index();
        out.whole(index, Block::Image(image), part.clone(), data)
    }

    /// One tool-call fragment. It goes to the call at its wire index, where
    /// the writer tells a new id under it from the call it held. Without an
    /// index (or with `null`) it goes to the open call that states its id;
    /// one that states no id continues the latest call while that call's
    /// arguments are not yet a whole object, and otherwise opens a new call
    /// when it brings a name or arguments. Its fields merge into the call's
    /// item.
    fn call(&mut self, call: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        self.close_writing(out)?;
        let id = match call.get("id") {
            Some(Value::String(id)) if !id.is_empty() && id != "null" => Some(id.clone()),
            Some(id @ Value::Number(_)) => Some(id.to_string()),
            _ => None,
        };
        let index = call
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok());
        let arguments = call
            .at("/function/arguments")
            .map(crate::json_utils::value_to_json_string);
        let name = call
            .at("/function/name")
            .or_else(|| call.at("/custom/name"))
            .and_then(Value::as_str);
        let starts = name.is_some_and(|name| !name.is_empty())
            || arguments.as_deref().is_some_and(|text| !text.is_empty());
        let at = match index {
            Some(index) => index,
            None => match id.as_deref() {
                Some(id) => self
                    .calls
                    .iter()
                    .find(|open| open.id.as_deref() == Some(id)),
                None => self
                    .calls
                    .last()
                    .filter(|open| open.opaque || !open.complete() || !starts),
            }
            .map_or_else(|| out.fresh_index(), |open| open.at),
        };
        let opaque = match self.calls.iter_mut().find(|open| open.at == at) {
            Some(open) => {
                if id.is_some() {
                    open.id.clone_from(&id);
                }
                open.arguments
                    .push_str(arguments.as_deref().unwrap_or_default());
                open.opaque
            }
            None => {
                let opaque = CallKind::of(call) == CallKind::Unknown;
                if opaque {
                    out.open(at, Block::Opaque { replay: false }, Value::Null)?;
                }
                self.calls.push(OpenCall {
                    at,
                    id: id.clone(),
                    opaque,
                    arguments: arguments.clone().unwrap_or_default(),
                });
                opaque
            }
        };
        if !opaque {
            out.fragment(
                Some(at),
                CallFragment {
                    id: id.as_deref(),
                    name,
                    arguments: arguments.as_deref(),
                },
            )?;
        }
        let mut fields = call.as_object().cloned().unwrap_or_default();
        // The index orders the stream; the call it assembles has none.
        fields.shift_remove("index");
        out.edit(at, |item| {
            if !item.is_object() {
                *item = Value::Object(Map::new());
            }
            if let Value::Object(item) = item {
                merge_fields(item, &fields);
            }
        })
    }

    /// Close the open calls, with their items as natives: every one at the
    /// end (`all`), only those whose arguments are whole at a finish chunk,
    /// since a provider may still send fragments after it. A custom call's
    /// arguments are its `{"input"}`. A call the output budget cut before
    /// its arguments were a whole object closes with no native: the
    /// provider never stated it complete.
    fn close_calls(
        &mut self,
        out: &mut Out<'_, Completion>,
        all: bool,
    ) -> Result<(), ProviderError> {
        let cut = self.finish == Some(FinishReason::Length);
        let mut waiting = Vec::new();
        for call in std::mem::take(&mut self.calls) {
            if call.opaque {
                out.finish(call.at)?;
                continue;
            }
            let (mut input, mut complete) = (None, false);
            out.edit(call.at, |item| {
                if CallKind::of(item) == CallKind::Custom {
                    let custom = item.at("/custom/input").cloned();
                    input = Some(custom.unwrap_or_else(|| Value::from("")));
                }
                complete = match item.at("/function/arguments") {
                    Some(Value::String(text)) => matches!(
                        crate::json_utils::parse_tool_arguments(text),
                        Ok(Value::Object(_))
                    ),
                    Some(arguments) => arguments.is_object(),
                    None => input.is_some(),
                };
            })?;
            if !all && !complete {
                waiting.push(call);
                continue;
            }
            if let Some(input) = input {
                out.announce(call.at, json!({ "input": input }))?;
            }
            if cut && !complete {
                out.close(call.at)?;
            } else {
                out.finish(call.at)?;
            }
        }
        self.calls = waiting;
        Ok(())
    }

    /// One `chat.completion.chunk`.
    fn chunk(&mut self, frame: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Some(choice) = self.absorb(frame) else {
            return Ok(());
        };
        if let Some(delta) = choice.obj("delta") {
            self.delta(delta, out)?;
        }
        // A finish states the message complete.
        if self.finish.is_some() {
            self.close_writing(out)?;
        }
        if self.finish == Some(FinishReason::ToolCalls) {
            self.close_calls(out, false)?;
        }
        Ok(())
    }

    /// The `chat.completion` body: the one chunk whose delta is its whole
    /// message, each call at its position in the list, then the end.
    fn whole(
        &mut self,
        frame: &Value,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let Some(choice) = self.absorb(frame) else {
            return Err(ProviderError::Response(
                "Response contained no choices".to_owned(),
            ));
        };
        let Some(mut message) = choice.obj("message").cloned() else {
            return Err(ProviderError::Response(
                "Response did not contain a valid message or tool call".to_owned(),
            ));
        };
        self.ended = true;
        if let Some(Value::Array(calls)) = message.get_mut("tool_calls") {
            for (index, call) in calls.iter_mut().enumerate() {
                if let Some(call) = call.as_object_mut() {
                    call.insert("index".to_owned(), index.into());
                }
            }
        }
        self.delta(&message, &mut out)?;
        self.close_calls(&mut out, true)?;
        self.end(out, false)
    }

    /// Write the provider's end of the reply. A stream's `raw` is the
    /// terminal record the chunks built; a whole body's is the body itself,
    /// which the transport keeps.
    fn end(&mut self, mut out: Out<'_, Completion>, streamed: bool) -> Result<Flow, ProviderError> {
        self.close_writing(&mut out)?;
        let usage = self
            .usage
            .as_ref()
            .map(|usage| normalized_usage(usage, &self.quirks))
            .unwrap_or_default();
        if streamed {
            let fields = std::mem::take(&mut self.fields);
            let record = [
                ("usage", self.usage.clone()),
                (
                    "finish_reason",
                    self.finish.as_ref().map(serde_json::to_value).transpose()?,
                ),
                ("response_id", self.response_id.clone().map(Value::String)),
                ("model", self.response_model.clone().map(Value::String)),
                ("logprobs", self.logprobs.take().map(Value::Object)),
                (
                    "additional_params",
                    (!fields.is_empty()).then_some(Value::Object(fields)),
                ),
            ];
            let record = record
                .into_iter()
                .filter_map(|(key, value)| Some((key.to_owned(), value?)))
                .collect();
            out.raw(Value::Object(record));
        }
        Ok(out.end(Finish {
            usage,
            reason: self.finish.take(),
            response_id: self.response_id.take(),
            model: self.response_model.take(),
            ..Finish::default()
        }))
    }

    /// The stream ended: flush the calls the provider delivered, then end
    /// the reply. A stream no finish reason ended was cut short.
    fn finish(&mut self, mut out: Out<'_, Completion>) -> Result<Flow, ProviderError> {
        if !self.ended {
            return Err(ProviderError::Truncated);
        }
        self.close_calls(&mut out, true)?;
        self.end(out, true)
    }
}

/// Append streamed reasoning details to the block's, by pi's merge: a
/// text or summary fragment continues the last entry of its type and
/// index, appending its text and filling the fields that entry lacks, and an
/// encrypted entry stays whole.
fn merge_details(block: &mut Map<String, Value>, details: Vec<Value>) {
    let Value::Array(merged) = block
        .entry(REASONING_DETAILS)
        .or_insert_with(|| Value::Array(Vec::new()))
    else {
        return;
    };
    for detail in details {
        let kind = detail.str("type");
        let continues = matches!(kind, Some("reasoning.text" | "reasoning.summary"))
            && merged.last().is_some_and(|last| {
                last.str("type") == kind && last.get("index") == detail.get("index")
            });
        match (continues, merged.last_mut(), detail) {
            (true, Some(Value::Object(last)), Value::Object(fields)) => {
                for (key, value) in fields {
                    let missing = last
                        .get(&key)
                        .is_none_or(|existing| existing.is_null() || existing.as_str() == Some(""));
                    match (last.get_mut(&key), value) {
                        (Some(Value::String(text)), Value::String(more))
                            if matches!(key.as_str(), "text" | "summary") =>
                        {
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
        // `[DONE]` is the wire's terminal sentinel, not JSON.
        if data == "[DONE]" {
            return WireEvent::Known(ChatEvent::Done);
        }
        // The in-band error envelope arrives with a 200 status and is this
        // wire's own terminal failure.
        if let Some(error) = provider_error_envelope(&data) {
            return WireEvent::Known(ChatEvent::Failure(error));
        }
        if self.quirks.accepts_bare_string_reply
            && let Ok(Value::String(text)) = serde_json::from_str::<serde_json::Value>(&data)
        {
            return WireEvent::Known(ChatEvent::BareText(text));
        }
        classify_chat_completions_frame::<Value>(&data).map(|frame| {
            // The `object` tag decides when the dialect sends one: stream
            // chunks may carry whole messages. Several gateways omit it.
            let whole = match frame.str("object") {
                Some(object) => object == "chat.completion",
                None => frame
                    .arr("choices")
                    .iter()
                    .any(|choice| choice.obj("message").is_some()),
            };
            if whole {
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
                self.chunk(&frame, &mut out)?;
                Ok(Flow::More)
            }
            ChatEvent::Whole(frame) => self.whole(&frame, out),
            ChatEvent::Done => self.finish(out),
            // A bare string is the whole answer, ended: pi's stop for a
            // reply that names no reason.
            ChatEvent::BareText(text) => {
                self.ended = true;
                self.finish = Some(FinishReason::Stop);
                self.write(Writing::Text, &text, &mut out)?;
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

/// A Chat usage object, normalized for a dialect with `quirks` and read
/// leniently, so no usage shape fails a reply: a counter that is absent,
/// `null` or not a non-negative integer is unreported, and one inside a
/// details object the usage has is zero. Cached input falls back to
/// Mistral's `num_cached_tokens` and then DeepSeek's
/// `prompt_cache_hit_tokens`; audio input counts as input when the total
/// says it sits beside the prompt; output is the remainder of the total when
/// unreported; and the reasoning count is left out where the dialect's
/// cannot be trusted ([`Quirks::reliable_reasoning_count`]).
fn normalized_usage(usage: &Value, quirks: &Quirks) -> crate::completion::Usage {
    let count = |pointer: &str| usage.at(pointer).and_then(Value::as_u64);
    let detail = |object: &str, key: &str| {
        count(&format!("/{object}/{key}")).or(usage.obj(object).map(|_| 0))
    };
    let (prompt, completion, total) = (
        count("/prompt_tokens"),
        count("/completion_tokens"),
        count("/total_tokens"),
    );
    let audio = count("/prompt_tokens_details/audio_tokens").unwrap_or(0);
    let input = prompt.map(|prompt| {
        let beside = prompt.saturating_add(audio);
        let accounted = beside.saturating_add(completion.unwrap_or(0));
        if audio != 0 && Some(accounted) == total {
            beside
        } else {
            prompt
        }
    });
    crate::completion::Usage {
        input_tokens: input,
        output_tokens: completion
            .or_else(|| total.map(|total| total.saturating_sub(input.unwrap_or(0)))),
        total_tokens: total,
        cached_input_tokens: detail("prompt_tokens_details", "cached_tokens")
            .or(count("/num_cached_tokens"))
            .or(count("/prompt_cache_hit_tokens")),
        cache_creation_input_tokens: count("/prompt_tokens_details/cache_write_tokens"),
        reasoning_tokens: detail("completion_tokens_details", "reasoning_tokens")
            .filter(|_| quirks.reliable_reasoning_count),
        ..Default::default()
    }
}

impl ChatDecoder {
    /// Verdict, model, response id, usage and error envelope, read off a raw
    /// payload before normalization discards them. The driver calls it for
    /// the unary reply and for every stream frame.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(payload) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        if let Some(usage) = payload.at("/usage") {
            let count = |pointer: &str| usage.at(pointer).and_then(Value::as_u64);
            sink.emit(AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: count("/prompt_tokens"),
                    output_tokens: count("/completion_tokens"),
                    total_tokens: count("/total_tokens"),
                    cached_input_tokens: count("/prompt_tokens_details/cached_tokens"),
                    reasoning_tokens: count("/completion_tokens_details/reasoning_tokens"),
                    tool_input_tokens: None,
                },
            });
        }
        // Only the chunk that carries the finish reason is a verdict, so the
        // model rides with it rather than on each delta.
        let verdict = match payload
            .at("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            Some(reason) => AdapterVerdict {
                finish_reason: Some(sink.scrub(reason)),
                block_reason: None,
                detail: None,
                model: payload.str("model").map(|value| sink.scrub(value)),
            },
            None => AdapterVerdict::default(),
        };
        let response_id = payload.str("id").map(|value| sink.scrub(value));
        sink.provider(verdict, response_id);
        if let Some(error) = payload
            .get("error")
            .and_then(|error| ObservedError::deserialize(error).ok())
        {
            error.emit(sink);
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod hard_case_tests;

#[cfg(test)]
mod history_tests;

#[cfg(test)]
mod request_tests;
