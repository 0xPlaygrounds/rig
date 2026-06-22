use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use super::completion::{
    AnthropicCompatibleProvider, CacheTtl, Content, GenericCompletionModel, Message, SystemContent,
    ToolChoice, Usage, apply_prompt_cache_control, build_tool_definitions,
    resolve_top_level_cache_control, split_system_messages_from_history,
    supports_mid_conversation_system_messages,
};
use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::sse::{Event, GenericEventSource};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge_inplace;
use crate::message::ReasoningContent;
use crate::streaming::{
    self, RawStreamingChoice, RawStreamingToolCall, StreamingResult, ToolCallDeltaContent,
};
use crate::telemetry::SpanCombinator;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::collections::HashMap;

fn create_streaming_request_body(
    request_model: String,
    completion_request: &mut CompletionRequest,
    max_tokens: u64,
    prompt_caching: bool,
    automatic_caching: bool,
    automatic_caching_ttl: Option<CacheTtl>,
) -> Result<Value, CompletionError> {
    let chat_history = completion_request.chat_history_with_documents();
    let (history_system, chat_history) = split_system_messages_from_history(
        chat_history,
        supports_mid_conversation_system_messages(&request_model),
    );
    let mut full_history = vec![];
    full_history.extend(chat_history);

    let mut messages = full_history
        .into_iter()
        .map(Message::try_from)
        .collect::<Result<Vec<Message>, _>>()?;

    // Convert system prompt to array format for cache_control support.
    let mut system: Vec<SystemContent> =
        if let Some(preamble) = completion_request.preamble.as_ref() {
            if preamble.is_empty() {
                vec![]
            } else {
                vec![SystemContent::Text {
                    text: preamble.clone(),
                    cache_control: None,
                }]
            }
        } else {
            vec![]
        };
    system.extend(history_system);

    let mut additional_params_payload = completion_request
        .additional_params
        .take()
        .unwrap_or(Value::Null);
    let top_level_cache_control = resolve_top_level_cache_control(
        automatic_caching,
        automatic_caching_ttl,
        &mut additional_params_payload,
    )?;
    let mut tools = build_tool_definitions(
        std::mem::take(&mut completion_request.tools),
        &mut additional_params_payload,
    )?;

    apply_prompt_cache_control(
        &mut system,
        &mut messages,
        &mut tools,
        prompt_caching,
        top_level_cache_control.as_ref(),
    )?;

    let mut body = json!({
        "model": request_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": true,
    });

    // Automatic caching: one top-level field; the API moves the breakpoint automatically.
    // No beta header is required.
    if let Some(cache_control) = top_level_cache_control {
        merge_inplace(
            &mut body,
            json!({ "cache_control": serde_json::to_value(&cache_control)? }),
        );
    }

    // Add system prompt if non-empty.
    if !system.is_empty() {
        merge_inplace(&mut body, json!({ "system": system }));
    }

    if let Some(temperature) = completion_request.temperature {
        merge_inplace(&mut body, json!({ "temperature": temperature }));
    }

    if !tools.is_empty() {
        merge_inplace(
            &mut body,
            json!({
                "tools": tools,
                "tool_choice": ToolChoice::Auto,
            }),
        );
    }

    if !additional_params_payload.is_null() {
        merge_inplace(&mut body, additional_params_payload)
    }

    Ok(body)
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamingEvent {
    MessageStart {
        message: MessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: Content,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: PartialUsage,
    },
    MessageStop,
    Ping,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct MessageStart {
    pub id: String,
    pub role: String,
    pub content: Vec<Content>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    CitationsDelta {
        citation: super::completion::Citation,
    },
    /// Forward-compatibility fallback. Any delta type Anthropic adds in the
    /// future that this crate does not yet model deserializes here so the
    /// surrounding [`StreamingEvent`] still parses.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct PartialUsage {
    pub output_tokens: usize,
    #[serde(default)]
    pub input_tokens: Option<usize>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
}

impl GetTokenUsage for PartialUsage {
    fn token_usage(&self) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.input_tokens.unwrap_or_default() as u64;
        usage.output_tokens = self.output_tokens as u64;
        usage.cached_input_tokens = self.cache_read_input_tokens.unwrap_or(0);
        usage.cache_creation_input_tokens = self.cache_creation_input_tokens.unwrap_or(0);
        usage.total_tokens = usage.input_tokens
            + usage.cached_input_tokens
            + usage.cache_creation_input_tokens
            + usage.output_tokens;
        usage
    }
}

#[derive(Default)]
struct ToolCallState {
    name: String,
    id: String,
    internal_call_id: String,
    input_json: String,
}

struct ServerToolUseState {
    name: String,
    id: String,
    initial_input: Value,
    input_json: String,
}

#[derive(Default)]
struct ThinkingState {
    thinking: String,
    signature: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StreamingCompletionResponse {
    pub usage: PartialUsage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.input_tokens.unwrap_or(0) as u64;
        usage.output_tokens = self.usage.output_tokens as u64;
        usage.cached_input_tokens = self.usage.cache_read_input_tokens.unwrap_or(0);
        usage.cache_creation_input_tokens = self.usage.cache_creation_input_tokens.unwrap_or(0);
        usage.total_tokens = usage.input_tokens
            + usage.cached_input_tokens
            + usage.cache_creation_input_tokens
            + usage.output_tokens;

        usage
    }
}

impl<Ext, T> GenericCompletionModel<Ext, T>
where
    T: HttpClientExt + Clone + Default + 'static,
    Ext: AnthropicCompatibleProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    pub(crate) async fn stream(
        &self,
        mut completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request_model = completion_request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = Ext::PROVIDER_NAME,
                gen_ai.request.model = &request_model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = &request_model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };
        let max_tokens = if let Some(tokens) = completion_request.max_tokens {
            tokens
        } else if let Some(tokens) = self.default_max_tokens {
            tokens
        } else {
            return Err(CompletionError::RequestError(
                "`max_tokens` must be set for Anthropic".into(),
            ));
        };

        let body = create_streaming_request_body(
            request_model,
            &mut completion_request,
            max_tokens,
            self.prompt_caching,
            self.automatic_caching,
            self.automatic_caching_ttl.clone(),
        )?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Anthropic completion request: {}",
                serde_json::to_string_pretty(&body)?
            );
        }

        let body: Vec<u8> = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/v1/messages")?
            .body(body)
            .map_err(http_client::Error::Protocol)?;

        let stream = GenericEventSource::new(self.client.clone(), req);

        // Use our SSE decoder to directly handle Server-Sent Events format
        let stream: StreamingResult<StreamingCompletionResponse> = Box::pin(stream! {
            let mut current_tool_call: Option<ToolCallState> = None;
            let mut server_tool_uses: HashMap<usize, ServerToolUseState> = HashMap::new();
            let mut current_thinking: Option<ThinkingState> = None;
            let mut sse_stream = Box::pin(stream);
            let mut input_tokens = 0;
            let mut final_usage = None;

            let mut text_content = String::new();

            while let Some(sse_result) = sse_stream.next().await {
                match sse_result {
                    Ok(Event::Open) => {}
                    Ok(Event::Message(sse)) => {
                        // Parse the SSE data as a StreamingEvent
                        match serde_json::from_str::<StreamingEvent>(&sse.data) {
                            Ok(event) => {
                                match &event {
                                    StreamingEvent::MessageStart { message } => {
                                        input_tokens = message.usage.input_tokens;

                                        let span = tracing::Span::current();
                                        span.record("gen_ai.response.id", &message.id);
                                        span.record("gen_ai.response.model", &message.model);
                                    },
                                    StreamingEvent::MessageDelta { delta, usage } => {
                                        if delta.stop_reason.is_some() {
                                            // cache_creation_input_tokens and cache_read_input_tokens
                                            // are cumulative totals on message_delta.usage per the
                                            // Anthropic streaming API spec — use them directly.
                                            let usage = PartialUsage {
                                                 output_tokens: usage.output_tokens,
                                                 input_tokens: usize::try_from(input_tokens).ok(),
                                                 cache_creation_input_tokens: usage.cache_creation_input_tokens,
                                                 cache_read_input_tokens: usage.cache_read_input_tokens
                                            };

                                            let span = tracing::Span::current();
                                            span.record_token_usage(&usage);
                                            final_usage = Some(usage);
                                            break;
                                        }
                                    }
                                    _ => {}
                                }

                                if let Some(result) = handle_event(
                                    &event,
                                    &mut current_tool_call,
                                    &mut server_tool_uses,
                                    &mut current_thinking,
                                ) {
                                    if let Ok(RawStreamingChoice::Message(ref text)) = result {
                                        text_content += text;
                                    }
                                    yield result;
                                }
                            },
                            Err(e) => {
                                if !sse.data.trim().is_empty() {
                                    yield Err(CompletionError::ResponseError(
                                        format!("Failed to parse JSON: {} (Data: {})", e, sse.data)
                                    ));
                                }
                            }
                        }
                    },
                    Err(e) => {
                        yield Err(CompletionError::ProviderError(format!("SSE Error: {e}")));
                        break;
                    }
                }
            }

            // Ensure event source is closed when stream ends
            sse_stream.close();

            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                usage: final_usage.unwrap_or_default()
            }))
        }.instrument(span));

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}

fn handle_event(
    event: &StreamingEvent,
    current_tool_call: &mut Option<ToolCallState>,
    server_tool_uses: &mut HashMap<usize, ServerToolUseState>,
    current_thinking: &mut Option<ThinkingState>,
) -> Option<Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>> {
    match event {
        StreamingEvent::ContentBlockDelta { index, delta } => match delta {
            ContentDelta::TextDelta { text } => {
                if current_tool_call.is_none() {
                    return Some(Ok(RawStreamingChoice::Message(text.clone())));
                }
                None
            }
            ContentDelta::InputJsonDelta { partial_json } => {
                if let Some(server_tool_use) = server_tool_uses.get_mut(index) {
                    server_tool_use.input_json.push_str(partial_json);
                    return None;
                }

                if let Some(tool_call) = current_tool_call {
                    tool_call.input_json.push_str(partial_json);
                    // Emit the delta so UI can show progress
                    return Some(Ok(RawStreamingChoice::ToolCallDelta {
                        id: tool_call.id.clone(),
                        internal_call_id: tool_call.internal_call_id.clone(),
                        content: ToolCallDeltaContent::Delta(partial_json.clone()),
                    }));
                }
                None
            }
            ContentDelta::ThinkingDelta { thinking } => {
                current_thinking
                    .get_or_insert_with(ThinkingState::default)
                    .thinking
                    .push_str(thinking);

                Some(Ok(RawStreamingChoice::ReasoningDelta {
                    id: None,
                    reasoning: thinking.clone(),
                }))
            }
            ContentDelta::SignatureDelta { signature } => {
                current_thinking
                    .get_or_insert_with(ThinkingState::default)
                    .signature
                    .push_str(signature);

                // Don't yield signature chunks, they will be included in the final Reasoning
                None
            }
            ContentDelta::CitationsDelta { citation } => {
                Some(Ok(RawStreamingChoice::TextAdditionalParams(json!({
                    "citations": [citation]
                }))))
            }
            ContentDelta::Unknown => None,
        },
        StreamingEvent::ContentBlockStart {
            index,
            content_block,
        } => match content_block {
            Content::Text { citations, .. } => {
                let additional_params = (!citations.is_empty()).then(|| {
                    json!({
                        "citations": citations
                    })
                });
                Some(Ok(RawStreamingChoice::TextStart { additional_params }))
            }
            Content::ServerToolUse { id, name, input } => {
                server_tool_uses.insert(
                    *index,
                    ServerToolUseState {
                        name: name.clone(),
                        id: id.clone(),
                        initial_input: input.clone(),
                        input_json: String::new(),
                    },
                );
                None
            }
            raw @ Content::WebSearchToolResult { .. } => Some(Ok(RawStreamingChoice::TextStart {
                additional_params: Some(json!({
                    super::completion::ANTHROPIC_RAW_CONTENT_KEY: raw
                })),
            })),
            Content::ToolUse { id, name, .. } => {
                let internal_call_id = crate::id::generate();
                *current_tool_call = Some(ToolCallState {
                    name: name.clone(),
                    id: id.clone(),
                    internal_call_id: internal_call_id.clone(),
                    input_json: String::new(),
                });
                Some(Ok(RawStreamingChoice::ToolCallDelta {
                    id: id.clone(),
                    internal_call_id,
                    content: ToolCallDeltaContent::Name(name.clone()),
                }))
            }
            Content::Thinking { .. } => {
                *current_thinking = Some(ThinkingState::default());
                None
            }
            Content::RedactedThinking { data } => Some(Ok(RawStreamingChoice::Reasoning {
                id: None,
                content: ReasoningContent::Redacted { data: data.clone() },
            })),
            // Handle other content types - they don't need special handling
            _ => None,
        },
        StreamingEvent::ContentBlockStop { index } => {
            if let Some(thinking_state) = Option::take(current_thinking)
                && !thinking_state.thinking.is_empty()
            {
                let signature = if thinking_state.signature.is_empty() {
                    None
                } else {
                    Some(thinking_state.signature)
                };

                return Some(Ok(RawStreamingChoice::Reasoning {
                    id: None,
                    content: ReasoningContent::Text {
                        text: thinking_state.thinking,
                        signature,
                    },
                }));
            }

            if let Some(server_tool_use) = server_tool_uses.remove(index) {
                let input = if server_tool_use.input_json.is_empty() {
                    if server_tool_use.initial_input.is_null() {
                        json!({})
                    } else {
                        server_tool_use.initial_input
                    }
                } else {
                    match serde_json::from_str(&server_tool_use.input_json) {
                        Ok(json_value) => json_value,
                        Err(e) => return Some(Err(CompletionError::from(e))),
                    }
                };

                return Some(Ok(RawStreamingChoice::TextStart {
                    additional_params: Some(json!({
                        super::completion::ANTHROPIC_RAW_CONTENT_KEY: Content::ServerToolUse {
                            id: server_tool_use.id,
                            name: server_tool_use.name,
                            input,
                        }
                    })),
                }));
            }

            if let Some(tool_call) = Option::take(current_tool_call) {
                let json_str = if tool_call.input_json.is_empty() {
                    "{}"
                } else {
                    &tool_call.input_json
                };
                match serde_json::from_str(json_str) {
                    Ok(json_value) => {
                        let raw_tool_call =
                            RawStreamingToolCall::new(tool_call.id, tool_call.name, json_value)
                                .with_internal_call_id(tool_call.internal_call_id);
                        Some(Ok(RawStreamingChoice::ToolCall(raw_tool_call)))
                    }
                    Err(e) => Some(Err(CompletionError::from(e))),
                }
            } else {
                None
            }
        }
        // Ignore other event types or handle as needed
        StreamingEvent::MessageStart { .. }
        | StreamingEvent::MessageDelta { .. }
        | StreamingEvent::MessageStop
        | StreamingEvent::Ping
        | StreamingEvent::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::completion::{CLAUDE_OPUS_4_8, CacheControl, CacheTtl};
    use super::*;
    use crate::OneOrMany;
    use crate::completion::Message as RigMessage;
    use crate::completion::request::Document as RigDocument;
    use async_stream::stream;
    use futures::StreamExt;

    #[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
    fn to_stream_result(
        stream: impl futures::Stream<
            Item = Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>,
        > + Send
        + 'static,
    ) -> crate::streaming::StreamingResult<StreamingCompletionResponse> {
        Box::pin(stream)
    }

    #[cfg(all(feature = "wasm", target_arch = "wasm32"))]
    fn to_stream_result(
        stream: impl futures::Stream<
            Item = Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>,
        > + 'static,
    ) -> crate::streaming::StreamingResult<StreamingCompletionResponse> {
        Box::pin(stream)
    }

    #[test]
    fn test_streaming_tool_build_marks_final_combined_tool() {
        let mut additional_params = json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"}
            }]
        });

        let mut tools = build_tool_definitions(
            vec![crate::completion::ToolDefinition {
                name: "rig_tool".to_string(),
                description: "Rig tool".to_string(),
                parameters: json!({"type": "object", "properties": {}}),
            }],
            &mut additional_params,
        )
        .unwrap();
        let mut system: Vec<SystemContent> = Vec::new();
        let mut messages: Vec<Message> = Vec::new();
        apply_prompt_cache_control(&mut system, &mut messages, &mut tools, true, None).unwrap();

        assert_eq!(tools.len(), 2);
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["name"], "provider_tool");
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn streaming_request_keeps_documents_after_leading_system_messages() {
        let mut request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                RigMessage::system("System prompt"),
                RigMessage::assistant("Earlier assistant turn"),
                RigMessage::system("Mid-conversation instruction"),
                RigMessage::user("Prompt"),
            ])
            .unwrap(),
            documents: vec![RigDocument {
                id: "doc1".to_string(),
                text: "Document text.".to_string(),
                additional_props: Default::default(),
            }],
            tools: vec![],
            temperature: None,
            max_tokens: Some(64),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let body = create_streaming_request_body(
            CLAUDE_OPUS_4_8.to_string(),
            &mut request,
            64,
            false,
            false,
            None,
        )
        .expect("streaming request body should build");

        assert_eq!(body["system"][0]["text"], "System prompt");
        assert_eq!(body["system"][1]["text"], "Mid-conversation instruction");
        let messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert!(
            messages[0].to_string().contains("<file id: doc1>"),
            "document message should follow top-level system: {messages:?}"
        );
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(
            messages
                .iter()
                .filter(|message| message.to_string().contains("<file id: doc1>"))
                .count(),
            1,
            "document message should appear exactly once: {messages:?}"
        );
    }

    #[test]
    fn test_streaming_prompt_cache_control_uses_raw_top_level_ttl() {
        let mut additional_params = json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        });
        let top_level_cache_control =
            resolve_top_level_cache_control(false, None, &mut additional_params).unwrap();
        let mut tools = build_tool_definitions(
            vec![crate::completion::ToolDefinition {
                name: "rig_tool".to_string(),
                description: "Rig tool".to_string(),
                parameters: json!({"type": "object", "properties": {}}),
            }],
            &mut additional_params,
        )
        .unwrap();
        let mut system = vec![SystemContent::Text {
            text: "System prompt".to_string(),
            cache_control: None,
        }];
        let mut messages: Vec<Message> = Vec::new();

        apply_prompt_cache_control(
            &mut system,
            &mut messages,
            &mut tools,
            true,
            top_level_cache_control.as_ref(),
        )
        .unwrap();

        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        match &system[0] {
            SystemContent::Text {
                cache_control: Some(CacheControl::Ephemeral { ttl }),
                ..
            } => assert_eq!(ttl.as_ref(), Some(&CacheTtl::OneHour)),
            other => panic!("expected system cache_control, got {other:?}"),
        }
        assert!(additional_params.get("cache_control").is_none());
    }

    fn handle_event(
        event: &StreamingEvent,
        current_tool_call: &mut Option<ToolCallState>,
        current_thinking: &mut Option<ThinkingState>,
    ) -> Option<Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>> {
        let mut server_tool_uses = HashMap::new();
        super::handle_event(
            event,
            current_tool_call,
            &mut server_tool_uses,
            current_thinking,
        )
    }

    #[test]
    fn test_thinking_delta_deserialization() {
        let json = r#"{"type": "thinking_delta", "thinking": "Let me think about this..."}"#;
        let delta: ContentDelta = serde_json::from_str(json).unwrap();

        match delta {
            ContentDelta::ThinkingDelta { thinking } => {
                assert_eq!(thinking, "Let me think about this...");
            }
            _ => panic!("Expected ThinkingDelta variant"),
        }
    }

    #[test]
    fn test_signature_delta_deserialization() {
        let json = r#"{"type": "signature_delta", "signature": "abc123def456"}"#;
        let delta: ContentDelta = serde_json::from_str(json).unwrap();

        match delta {
            ContentDelta::SignatureDelta { signature } => {
                assert_eq!(signature, "abc123def456");
            }
            _ => panic!("Expected SignatureDelta variant"),
        }
    }

    #[test]
    fn test_thinking_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "First, I need to understand the problem."
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamingEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentDelta::ThinkingDelta { thinking } => {
                        assert_eq!(thinking, "First, I need to understand the problem.");
                    }
                    _ => panic!("Expected ThinkingDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_signature_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "signature_delta",
                "signature": "ErUBCkYICBgCIkCaGbqC85F4"
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamingEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentDelta::SignatureDelta { signature } => {
                        assert_eq!(signature, "ErUBCkYICBgCIkCaGbqC85F4");
                    }
                    _ => panic!("Expected SignatureDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_handle_thinking_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::ThinkingDelta {
                thinking: "Analyzing the request...".to_string(),
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::ReasoningDelta { id, reasoning, .. } => {
                assert_eq!(id, None);
                assert_eq!(reasoning, "Analyzing the request...");
            }
            _ => panic!("Expected ReasoningDelta choice"),
        }

        // Verify thinking state was updated
        assert!(thinking_state.is_some());
        assert_eq!(thinking_state.unwrap().thinking, "Analyzing the request...");
    }

    #[test]
    fn test_handle_signature_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::SignatureDelta {
                signature: "test_signature".to_string(),
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        // SignatureDelta should not yield anything (returns None)
        assert!(result.is_none());

        // But signature should be captured in thinking state
        assert!(thinking_state.is_some());
        assert_eq!(thinking_state.unwrap().signature, "test_signature");
    }

    #[test]
    fn test_handle_redacted_thinking_content_block_start_event() {
        let event = StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::RedactedThinking {
                data: "redacted_blob".to_string(),
            },
        };
        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        match result.unwrap().unwrap() {
            RawStreamingChoice::Reasoning {
                content: ReasoningContent::Redacted { data },
                ..
            } => {
                assert_eq!(data, "redacted_blob");
            }
            _ => panic!("Expected Redacted reasoning chunk"),
        }
    }

    #[test]
    fn test_handle_text_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta {
                text: "Hello, world!".to_string(),
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::Message(text) => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected Message choice"),
        }
    }

    #[test]
    fn test_handle_text_block_start_event() {
        let event = StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::Text {
                text: String::new(),
                citations: Vec::new(),
                cache_control: None,
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();
        assert!(matches!(
            choice,
            RawStreamingChoice::TextStart {
                additional_params: None
            }
        ));
    }

    #[test]
    fn test_thinking_delta_does_not_interfere_with_tool_calls() {
        // Thinking deltas should still be processed even if a tool call is in progress
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::ThinkingDelta {
                thinking: "Thinking while tool is active...".to_string(),
            },
        };

        let mut tool_call_state = Some(ToolCallState {
            name: "test_tool".to_string(),
            id: "tool_123".to_string(),
            internal_call_id: crate::id::generate(),
            input_json: String::new(),
        });
        let mut thinking_state = None;

        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::ReasoningDelta { reasoning, .. } => {
                assert_eq!(reasoning, "Thinking while tool is active...");
            }
            _ => panic!("Expected ReasoningDelta choice"),
        }

        // Tool call state should remain unchanged
        assert!(tool_call_state.is_some());
    }

    #[test]
    fn test_handle_input_json_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::InputJsonDelta {
                partial_json: "{\"arg\":\"value".to_string(),
            },
        };

        let mut tool_call_state = Some(ToolCallState {
            name: "test_tool".to_string(),
            id: "tool_123".to_string(),
            internal_call_id: crate::id::generate(),
            input_json: String::new(),
        });
        let mut thinking_state = None;

        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        // Should emit a ToolCallDelta
        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::ToolCallDelta {
                id,
                internal_call_id: _,
                content,
            } => {
                assert_eq!(id, "tool_123");
                match content {
                    ToolCallDeltaContent::Delta(delta) => assert_eq!(delta, "{\"arg\":\"value"),
                    _ => panic!("Expected Delta content"),
                }
            }
            _ => panic!("Expected ToolCallDelta choice, got {:?}", choice),
        }

        // Verify the input_json was accumulated
        assert!(tool_call_state.is_some());
        let state = tool_call_state.unwrap();
        assert_eq!(state.input_json, "{\"arg\":\"value");
    }

    #[test]
    fn test_tool_call_accumulation_with_multiple_deltas() {
        let mut tool_call_state = Some(ToolCallState {
            name: "test_tool".to_string(),
            id: "tool_123".to_string(),
            internal_call_id: crate::id::generate(),
            input_json: String::new(),
        });
        let mut thinking_state = None;

        // First delta
        let event1 = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::InputJsonDelta {
                partial_json: "{\"location\":".to_string(),
            },
        };
        let result1 = handle_event(&event1, &mut tool_call_state, &mut thinking_state);
        assert!(result1.is_some());

        // Second delta
        let event2 = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::InputJsonDelta {
                partial_json: "\"Paris\",".to_string(),
            },
        };
        let result2 = handle_event(&event2, &mut tool_call_state, &mut thinking_state);
        assert!(result2.is_some());

        // Third delta
        let event3 = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::InputJsonDelta {
                partial_json: "\"temp\":\"20C\"}".to_string(),
            },
        };
        let result3 = handle_event(&event3, &mut tool_call_state, &mut thinking_state);
        assert!(result3.is_some());

        // Verify accumulated JSON
        assert!(tool_call_state.is_some());
        let state = tool_call_state.as_ref().unwrap();
        assert_eq!(
            state.input_json,
            "{\"location\":\"Paris\",\"temp\":\"20C\"}"
        );

        // Final ContentBlockStop should emit complete tool call
        let stop_event = StreamingEvent::ContentBlockStop { index: 0 };
        let final_result = handle_event(&stop_event, &mut tool_call_state, &mut thinking_state);
        assert!(final_result.is_some());

        match final_result.unwrap().unwrap() {
            RawStreamingChoice::ToolCall(RawStreamingToolCall {
                id,
                name,
                arguments,
                ..
            }) => {
                assert_eq!(id, "tool_123");
                assert_eq!(name, "test_tool");
                assert_eq!(
                    arguments.get("location").unwrap().as_str().unwrap(),
                    "Paris"
                );
                assert_eq!(arguments.get("temp").unwrap().as_str().unwrap(), "20C");
            }
            other => panic!("Expected ToolCall, got {:?}", other),
        }

        // Tool call state should be taken
        assert!(tool_call_state.is_none());
    }

    #[test]
    fn test_citations_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "char_location",
                    "cited_text": "The grass is green.",
                    "document_index": 0,
                    "document_title": "Example",
                    "start_char_index": 0,
                    "end_char_index": 20
                }
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        let StreamingEvent::ContentBlockDelta { index, delta } = event else {
            panic!("expected ContentBlockDelta");
        };
        assert_eq!(index, 0);
        let ContentDelta::CitationsDelta { citation } = delta else {
            panic!("expected CitationsDelta");
        };
        let crate::providers::anthropic::completion::Citation::CharLocation {
            start_char_index,
            end_char_index,
            ..
        } = citation
        else {
            panic!("expected CharLocation");
        };
        assert_eq!(start_char_index, 0);
        assert_eq!(end_char_index, 20);
    }

    #[test]
    fn test_search_result_citations_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "search_result_location",
                    "cited_text": "API requests require a key.",
                    "source": "https://docs.example.com/api-reference",
                    "title": "API Reference",
                    "search_result_index": 0,
                    "start_block_index": 0,
                    "end_block_index": 1
                }
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        let StreamingEvent::ContentBlockDelta { delta, .. } = event else {
            panic!("expected ContentBlockDelta");
        };
        let ContentDelta::CitationsDelta { citation } = delta else {
            panic!("expected CitationsDelta");
        };
        assert!(matches!(
            citation,
            crate::providers::anthropic::completion::Citation::SearchResultLocation {
                search_result_index: 0,
                start_block_index: 0,
                end_block_index: 1,
                ..
            }
        ));
    }

    #[test]
    fn test_web_search_result_citations_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was a mathematician.",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_index": "encrypted-reference"
                }
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        let StreamingEvent::ContentBlockDelta { delta, .. } = event else {
            panic!("expected ContentBlockDelta");
        };
        let ContentDelta::CitationsDelta { citation } = delta else {
            panic!("expected CitationsDelta");
        };
        assert!(matches!(
            citation,
            crate::providers::anthropic::completion::Citation::WebSearchResultLocation {
                ref url,
                ref encrypted_index,
                ..
            } if url == "https://example.com/shannon"
                && encrypted_index == "encrypted-reference"
        ));
    }

    #[test]
    fn test_web_search_result_citations_delta_allows_null_title() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was a mathematician.",
                    "url": "https://example.com/shannon",
                    "title": null,
                    "encrypted_index": "encrypted-reference"
                }
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        let StreamingEvent::ContentBlockDelta { delta, .. } = event else {
            panic!("expected ContentBlockDelta");
        };
        let ContentDelta::CitationsDelta { citation } = delta else {
            panic!("expected CitationsDelta");
        };
        assert!(matches!(
            citation,
            crate::providers::anthropic::completion::Citation::WebSearchResultLocation {
                title: None,
                ..
            }
        ));
    }

    #[test]
    fn test_web_search_content_block_start_events_deserialize() {
        let server_tool_use = r#"{
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {
                    "query": "claude shannon birth date"
                }
            }
        }"#;
        let event: StreamingEvent = serde_json::from_str(server_tool_use).unwrap();
        assert!(matches!(
            event,
            StreamingEvent::ContentBlockStart {
                content_block: Content::ServerToolUse {
                    ref id,
                    ref name,
                    ref input
                },
                ..
            } if id == "srvtoolu_01"
                && name == "web_search"
                && input["query"] == "claude shannon birth date"
        ));

        let web_search_tool_result = r#"{
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [{
                    "type": "web_search_result",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_content": "encrypted-content"
                }]
            }
        }"#;
        let event: StreamingEvent = serde_json::from_str(web_search_tool_result).unwrap();
        assert!(matches!(
            event,
            StreamingEvent::ContentBlockStart {
                content_block: Content::WebSearchToolResult {
                    ref tool_use_id,
                    ref content
                },
                ..
            } if tool_use_id == "srvtoolu_01"
                && content[0]["encrypted_content"] == "encrypted-content"
        ));
    }

    #[tokio::test]
    async fn test_streaming_web_search_blocks_are_preserved_on_final_choice() {
        let raw_stream = stream! {
            let mut tool_call_state = None;
            let mut server_tool_uses = HashMap::new();
            let mut thinking_state = None;

            let server_tool_use_start = super::handle_event(
                &StreamingEvent::ContentBlockStart {
                    index: 0,
                    content_block: Content::ServerToolUse {
                        id: "srvtoolu_01".to_string(),
                        name: "web_search".to_string(),
                        input: serde_json::Value::Null,
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            );
            assert!(
                server_tool_use_start.is_none(),
                "server_tool_use start should be accumulated until its input JSON is complete"
            );

            let server_tool_use_delta = super::handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 0,
                    delta: ContentDelta::InputJsonDelta {
                        partial_json: r#"{"query":"claude shannon birth date"}"#.to_string(),
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            );
            assert!(
                server_tool_use_delta.is_none(),
                "server_tool_use input JSON should not be emitted as a Rig tool-call delta"
            );

            yield super::handle_event(
                &StreamingEvent::ContentBlockStop { index: 0 },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("server_tool_use stop should produce completed raw metadata");

            yield super::handle_event(
                &StreamingEvent::ContentBlockStart {
                    index: 1,
                    content_block: Content::WebSearchToolResult {
                        tool_use_id: "srvtoolu_01".to_string(),
                        content: serde_json::json!([{
                            "type": "web_search_result",
                            "url": "https://example.com/shannon",
                            "title": "Claude Shannon",
                            "encrypted_content": "encrypted-content"
                        }]),
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("web_search_tool_result block should produce raw metadata");

            yield super::handle_event(
                &StreamingEvent::ContentBlockStart {
                    index: 2,
                    content_block: Content::Text {
                        text: String::new(),
                        citations: Vec::new(),
                        cache_control: None,
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("text block start should produce a raw choice");

            yield super::handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 2,
                    delta: ContentDelta::TextDelta {
                        text: "Claude Shannon was born on April 30, 1916.".to_string(),
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("text delta should produce a raw choice");

            yield super::handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 2,
                    delta: ContentDelta::CitationsDelta {
                        citation: crate::providers::anthropic::completion::Citation::WebSearchResultLocation {
                            cited_text: "Claude Shannon was born on April 30, 1916.".to_string(),
                            url: "https://example.com/shannon".to_string(),
                            title: Some("Claude Shannon".to_string()),
                            encrypted_index: "encrypted-index".to_string(),
                        },
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("citation delta should produce a raw choice");

            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                usage: PartialUsage::default(),
            }));
        };

        let mut stream =
            crate::streaming::StreamingCompletionResponse::stream(to_stream_result(raw_stream));
        while stream.next().await.is_some() {}

        let choice_items: Vec<crate::message::AssistantContent> =
            stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 3);
        assert!(
            choice_items
                .iter()
                .all(|item| !matches!(item, crate::message::AssistantContent::ToolCall(_))),
            "provider-owned web-search blocks must not become Rig client tool calls"
        );

        let Some(crate::message::AssistantContent::Text(server_tool_use)) = choice_items.first()
        else {
            panic!("expected raw server_tool_use metadata");
        };
        assert_eq!(
            server_tool_use.additional_params.as_ref().unwrap()
                [crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["type"],
            "server_tool_use"
        );
        assert_eq!(
            server_tool_use.additional_params.as_ref().unwrap()
                [crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["input"]["query"],
            "claude shannon birth date"
        );

        let Some(crate::message::AssistantContent::Text(web_search_result)) = choice_items.get(1)
        else {
            panic!("expected raw web_search_tool_result metadata");
        };
        assert_eq!(
            web_search_result.additional_params.as_ref().unwrap()
                [crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["content"][0]
                ["encrypted_content"],
            "encrypted-content"
        );

        let Some(crate::message::AssistantContent::Text(answer)) = choice_items.get(2) else {
            panic!("expected answer text");
        };
        assert_eq!(answer.text, "Claude Shannon was born on April 30, 1916.");
        let citations = crate::providers::anthropic::completion::anthropic_citations(answer)
            .expect("expected preserved citations");
        assert!(matches!(
            citations.first(),
            Some(crate::providers::anthropic::completion::Citation::WebSearchResultLocation {
                encrypted_index,
                ..
            }) if encrypted_index == "encrypted-index"
        ));
    }

    #[test]
    fn test_handle_citations_delta_event_preserves_metadata() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::CitationsDelta {
                citation: crate::providers::anthropic::completion::Citation::CharLocation {
                    cited_text: "The grass is green.".to_string(),
                    document_index: 0,
                    document_title: Some("Example".to_string()),
                    start_char_index: 0,
                    end_char_index: 20,
                },
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();
        let RawStreamingChoice::TextAdditionalParams(additional_params) = choice else {
            panic!("expected TextAdditionalParams choice");
        };
        assert_eq!(additional_params["citations"][0]["type"], "char_location");
    }

    #[tokio::test]
    async fn test_streaming_citation_deltas_are_preserved_on_final_text() {
        let citation = crate::providers::anthropic::completion::Citation::CharLocation {
            cited_text: "The grass is green.".to_string(),
            document_index: 0,
            document_title: Some("Example".to_string()),
            start_char_index: 0,
            end_char_index: 20,
        };

        let raw_stream = stream! {
            let mut tool_call_state = None;
            let mut thinking_state = None;

            yield handle_event(
                &StreamingEvent::ContentBlockStart {
                    index: 0,
                    content_block: Content::Text {
                        text: String::new(),
                        citations: Vec::new(),
                        cache_control: None,
                    },
                },
                &mut tool_call_state,
                &mut thinking_state,
            )
            .expect("text block start should produce a raw choice");

            yield handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 0,
                    delta: ContentDelta::TextDelta {
                        text: "the grass is green".to_string(),
                    },
                },
                &mut tool_call_state,
                &mut thinking_state,
            )
            .expect("text delta should produce a raw choice");

            yield handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 0,
                    delta: ContentDelta::CitationsDelta {
                        citation: crate::providers::anthropic::completion::Citation::CharLocation {
                            cited_text: "The grass is green.".to_string(),
                            document_index: 0,
                            document_title: Some("Example".to_string()),
                            start_char_index: 0,
                            end_char_index: 20,
                        },
                    },
                },
                &mut tool_call_state,
                &mut thinking_state,
            )
            .expect("citation delta should produce a raw choice");

            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                usage: PartialUsage::default(),
            }));
        };

        let mut stream =
            crate::streaming::StreamingCompletionResponse::stream(to_stream_result(raw_stream));
        while stream.next().await.is_some() {}

        let choice_items: Vec<crate::message::AssistantContent> =
            stream.choice.clone().into_iter().collect();
        let Some(crate::message::AssistantContent::Text(text)) = choice_items.first() else {
            panic!("expected accumulated text item");
        };

        assert_eq!(text.text, "the grass is green");
        let citations = crate::providers::anthropic::completion::anthropic_citations(text).unwrap();
        assert_eq!(citations, vec![citation]);
    }

    #[test]
    fn test_unknown_content_delta_falls_back() {
        let json = r#"{"type": "something_new_from_anthropic", "field": "x"}"#;
        let delta: ContentDelta = serde_json::from_str(json).unwrap();
        assert!(matches!(delta, ContentDelta::Unknown));
    }
}
