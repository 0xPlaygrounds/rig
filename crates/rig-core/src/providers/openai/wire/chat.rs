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
    CompatibleFinishReason, CompatibleTerminal, CompatibleToolCallChunk, map_openai_finish_reason,
    provider_error_envelope, should_evict_distinct_named_tool_call,
};
use crate::providers::internal::tool_call_bridge::ToolCallBridge;
use crate::providers::internal::wire::classify_chat_completions_frame;
use crate::providers::openai::completion::{
    self as unary, AssistantContent, Message, ToolChoice, assistant_refusal_fallback,
    is_openai_reasoning_model, request_body,
};
use crate::streaming::{
    BlockId, Delta, MintKind, StreamEvent, ToolCallEnd, UnparseableToolInput,
};
use crate::wire::{
    Body, Decoder, Encoded, Framing, Mode, ObservationSink, Output, Wire, WireEvent, WireFrame,
};

use super::dto::{
    ChatFrame, ChatUsage, StreamingCompletionResponse, delta_text,
};
use super::{BodyRewrite, OpenAI, OutputCap, Routing};

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
            BodyRewrite::None
            | BodyRewrite::HuggingFaceRouter
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
            BodyRewrite::Mistral => finalize_mistral(map),
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
                message.insert("content".to_owned(), serde_json::Value::String(String::new()));
            }

            if is_assistant
                && let Some(tool_calls) = message.get_mut("tool_calls").and_then(as_array_mut)
            {
                for tool_call in tool_calls {
                    if let Some(tool_call) = tool_call.as_object_mut() {
                        tool_call.entry("index").or_insert_with(|| serde_json::json!(0));
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
/// Mistral's multimodal *content* mapping — text-only arrays to a string,
/// images/audio/documents to its own chunk shapes — still lives in
/// `providers::mistral::completion::normalize_request_content` and must move
/// here when that module is collapsed onto this dialect.
fn finalize_mistral(map: &mut serde_json::Map<String, serde_json::Value>) {
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
        return;
    };
    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        if message.get("role").and_then(serde_json::Value::as_str) != Some("assistant") {
            continue;
        }
        if !message.contains_key("content") {
            message.insert("content".to_owned(), serde_json::Value::String(String::new()));
        }
        // `prefix` is part of Mistral's assistant message schema.
        message.entry("prefix").or_insert(serde_json::Value::Bool(false));
        // Mistral rejects unknown assistant fields; hidden reasoning cannot
        // be echoed back.
        message.remove("reasoning_content");
    }
}

/// OpenRouter's two body rewrites.
///
/// OpenRouter's routing preferences (`ProviderPreferences`) are still built
/// by `providers::openrouter::completion`'s own request type and must move
/// here when that module is collapsed onto this dialect.
fn finalize_openrouter(map: &mut serde_json::Map<String, serde_json::Value>, prompt_caching: bool) {
    if prompt_caching {
        apply_openrouter_prompt_caching(map);
    }

    // The shared assistant message serializes hidden reasoning under the
    // llama.cpp/DeepSeek key `reasoning_content`; OpenRouter's documented
    // assistant field is `reasoning`.
    if let Some(messages) = map.get_mut("messages").and_then(as_array_mut) {
        for message in messages {
            if let Some(message) = message.as_object_mut()
                && message.get("role").and_then(serde_json::Value::as_str) == Some("assistant")
                && let Some(reasoning) = message.remove("reasoning_content")
            {
                message.insert("reasoning".to_owned(), reasoning);
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

impl Wire for Chat {
    type Op = crate::operation::Completion;
    type Decoder = ChatDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, CompletionError> {
        let quirks = &self.provider.dialect.quirks;
        let mut typed = unary::CompletionRequest::try_from(unary::OpenAIRequestParams {
            model: self.model.clone(),
            request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
            supports_response_format: quirks.supports_response_format,
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
        let deployment = match quirks.routing {
            Routing::AzureDeployment => Some(self.model.as_str()),
            Routing::Path => None,
        };
        let builder = http::Request::post(self.provider.uri(quirks.completion_path, deployment))
            .header("Content-Type", "application/json");
        let request = self
            .provider
            .authenticate(builder)
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

    fn decoder(&self) -> ChatDecoder {
        ChatDecoder::new(self.provider.dialect.name, self.provider.dialect.quirks)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Chat Completions *defers* `response_format` while tools are present
        // and no tool result exists yet, then applies it once a tool result
        // is in the history — so the native constraint does not suppress tool
        // calls; they compose. A dialect that drops `output_schema` cannot
        // compose them, and the agent falls back to tool-mode enforcement.
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
}

impl ChatDecoder {
    fn new(provider: &'static str, quirks: super::Quirks) -> Self {
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
            details: choice.delta.reasoning_details.clone(),
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

        // Reasoning first: the compatible providers that expose hidden
        // reasoning on this non-standard field stream it before any text, and
        // the unary body must produce the same block order.
        if let Some(reasoning) = reasoning.filter(|reasoning| !reasoning.is_empty()) {
            out.reasoning(reasoning);
        }

        for part in &content {
            let text = match part {
                AssistantContent::Text { text } => text,
                AssistantContent::Refusal { refusal } => refusal,
            };
            if !text.is_empty() {
                out.text(text);
            }
        }
        // This wire spells a refusal as a *sibling* of `content`
        // (`{"content": null, "refusal": "…"}`), so a path reading `content`
        // alone would drop it entirely.
        if let Some(refusal) = assistant_refusal_fallback(&content, refusal.as_deref()) {
            out.text(refusal);
        }

        for call in &tool_calls {
            let key = crate::streaming::non_empty_id(call.id.clone())
                .map_or_else(|| BlockId::minted(MintKind::Tool, 0), BlockId::wire);
            out.tool_end(
                key,
                ToolCallEnd::whole(&call.function.name, call.function.arguments.clone())
                    .with_tool_id(call.id.clone()),
            );
        }

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
    details: Vec<serde_json::Value>,
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
        classify_chat_completions_frame::<ChatFrame>(&data).map(|frame| {
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
        let output_length_truncation =
            matches!(self.final_finish_reason.as_ref(), Some(FinishReason::Length));
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

    fn project(&self, payload: &[u8], sink: &mut dyn ObservationSink) {
        super::observation::project_chat(payload, sink);
    }
}

/// Map a gateway's upstream-native finish reason.
///
/// Its vocabulary is the union of its upstreams' — Anthropic's `end_turn`,
/// Gemini's `STOP`, the OpenAI-compatible spellings — so it is wider than
/// the normalized one and cannot be read through `map_openai_finish_reason`.
fn map_native_finish_reason(reason: &str) -> FinishReason {
    match reason.to_ascii_lowercase().as_str() {
        "stop" | "end_turn" | "stop_sequence" | "complete" | "completed" => FinishReason::Stop,
        "length" | "max_tokens" | "max_output_tokens" | "model_length" => FinishReason::Length,
        "tool_calls" | "function_call" | "tool_use" => FinishReason::ToolCalls,
        "content_filter" | "safety" | "blocklist" | "prohibited_content" | "spii" => {
            FinishReason::ContentFilter
        }
        other => FinishReason::Other(other.to_owned()),
    }
}

/// A gateway's encrypted-reasoning detail as a whole reasoning block.
///
/// Encrypted reasoning (`{"type":"reasoning.encrypted"}`) is the turn's own
/// output, not tool-call metadata: it arrives with `reasoning: null` and an
/// `rs_*` id of its own, which never matches a `call_*` tool-call id, and it
/// arrives before any tool call opens. Emitting it as a reasoning block is
/// what lets the blob reach the aggregated choice and be replayed next turn.
fn detail_reasoning(
    detail: &serde_json::Value,
) -> Option<(BlockId, Option<String>, crate::message::ReasoningContent)> {
    let Ok(unary::ReasoningDetails::Encrypted { id, data, .. }) =
        serde_json::from_value::<unary::ReasoningDetails>(detail.clone())
    else {
        return None;
    };
    // The durable handle exists only when the wire issued one; an id-less
    // detail keys accumulation by a minted key and replays with the id
    // absent — no fabricated empty "wire" id. The mint kind is
    // `EncryptedReasoning`, NOT `Reasoning`: plaintext `reasoning` text
    // accumulates under `Minted { Reasoning, 0 }`, and a whole block under
    // that same key would restate — i.e. replace — the open text part.
    let provider_id = id.and_then(crate::streaming::non_empty_id);
    let key = provider_id.as_ref().map_or(
        BlockId::minted(MintKind::EncryptedReasoning, 0),
        |id| BlockId::wire(id.as_str()),
    );
    Some((
        key,
        provider_id,
        crate::message::ReasoningContent::Encrypted(data),
    ))
}

/// A gateway's signature-only reasoning detail.
///
/// Anthropic routes stream the plaintext in `delta.reasoning`, then send its
/// replay-required signature as a final signature-only `reasoning.text`
/// detail immediately before the tool call. Feeding that authoritative close
/// into the shared lifecycle signs the normalized reasoning block just as the
/// unary body does.
fn reasoning_signature(detail: &serde_json::Value) -> Option<String> {
    let Ok(unary::ReasoningDetails::Text {
        signature: Some(signature),
        ..
    }) = serde_json::from_value::<unary::ReasoningDetails>(detail.clone())
    else {
        return None;
    };
    (!signature.is_empty()).then_some(signature)
}

#[cfg(test)]
mod tests;
