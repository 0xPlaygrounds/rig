use async_stream::stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use super::InteractionsCompletionModel;
use super::create_request_body;
use super::interactions_api_types::{
    ContentDelta, FunctionCallContent, FunctionCallDelta, Interaction, InteractionSseEvent,
    InteractionUsage, StreamingContentStart, StreamingTextStart, TextDelta, ThoughtContent,
    ThoughtSignatureDelta, ThoughtSummaryContent, ThoughtSummaryDelta,
};
use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::Request;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::message::ReasoningContent;
use crate::streaming::{self, ToolCallDeltaContent};
use crate::telemetry::SpanCombinator;
use serde_json::{Map, Value};
use std::fmt;

#[derive(Debug, Default)]
struct ThoughtState {
    text: String,
    signature: Option<String>,
}

#[derive(Debug)]
struct FunctionCallState {
    internal_call_id: String,
    provider_call_id: Option<String>,
    name: Option<String>,
    arguments: Option<Value>,
    emitted_name: bool,
}

impl Default for FunctionCallState {
    fn default() -> Self {
        Self {
            internal_call_id: nanoid::nanoid!(),
            provider_call_id: None,
            name: None,
            arguments: None,
            emitted_name: false,
        }
    }
}

#[derive(Debug, Default)]
struct InteractionStreamState {
    thoughts_by_index: HashMap<i32, ThoughtState>,
    function_calls_by_index: HashMap<i32, FunctionCallState>,
}

/// Final metadata yielded by an Interactions streaming response.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StreamingCompletionResponse {
    pub usage: Option<InteractionUsage>,
    pub interaction: Option<Interaction>,
}

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>> + Send>>;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>>>>;

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        self.usage.as_ref().and_then(|usage| usage.token_usage())
    }
}

impl<T> InteractionsCompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "interactions_streaming",
                gen_ai.operation.name = "interactions_streaming",
                gen_ai.provider.name = "gcp.gemini",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request = create_request_body(self.model.clone(), completion_request, Some(true))?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::streaming",
                "Gemini interactions streaming request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post_sse("/v1beta/interactions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let mut event_source = GenericEventSource::new(self.client.clone(), req);

        let stream = stream! {
            let mut final_interaction: Option<Interaction> = None;
            let mut final_usage: Option<InteractionUsage> = None;
            let mut stream_state = InteractionStreamState::default();

            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::debug!("SSE connection opened");
                        continue;
                    }
                    Ok(Event::Message(message)) => {
                        let data = match parse_interaction_sse_event(&message.data) {
                            Ok(Some(data)) => data,
                            Ok(None) => continue,
                            Err(err) => {
                                yield Err(CompletionError::ProviderError(format!(
                                    "Failed to parse interactions SSE event: {err}"
                                )));
                                break;
                            }
                        };

                        match data {
                            InteractionSseEvent::ContentDelta { index, delta, .. } => {
                                for choice in stream_state.handle_content_delta(index, delta) {
                                    yield Ok(choice);
                                }
                            }
                            InteractionSseEvent::ContentStart { index, content, .. } => {
                                for choice in stream_state.handle_content_start(index, content) {
                                    yield Ok(choice);
                                }
                            }
                            InteractionSseEvent::ContentStop { index, .. } => {
                                if let Some(choice) = stream_state.handle_content_stop(index) {
                                    yield Ok(choice);
                                }
                            }
                            InteractionSseEvent::InteractionComplete { interaction, .. } => {
                                let span = tracing::Span::current();
                                span.record("gen_ai.response.id", &interaction.id);
                                if let Some(model) = interaction.model.clone() {
                                    span.record("gen_ai.response.model", model);
                                }

                                if let Some(usage) = interaction.usage.clone() {
                                    span.record_token_usage(&usage);
                                    final_usage = Some(usage);
                                }
                                final_interaction = Some(interaction);
                            }
                            InteractionSseEvent::Error { error, .. } => {
                                yield Err(CompletionError::ProviderError(error.message));
                                break;
                            }
                            _ => continue,
                        }
                    }
                    Err(crate::http_client::Error::StreamEnded) => {
                        break;
                    }
                    Err(error) => {
                        tracing::error!(?error, "SSE error");
                        yield Err(CompletionError::ProviderError(error.to_string()));
                        break;
                    }
                }
            }

            event_source.close();

            yield Ok(streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                usage: final_usage.or_else(|| final_interaction.as_ref().and_then(|i| i.usage.clone())),
                interaction: final_interaction,
            }));
        }
        .instrument(span);

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
    }
}

pub(crate) fn stream_interaction_events<T>(
    client: super::InteractionsClient<T>,
    request: Request<Vec<u8>>,
) -> InteractionEventStream
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + 'static,
{
    let mut event_source = GenericEventSource::new(client.clone(), request);

    let stream = stream! {
        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => continue,
                Ok(Event::Message(message)) => {
                    let data = match parse_interaction_sse_event(&message.data) {
                        Ok(Some(data)) => data,
                        Ok(None) => continue,
                        Err(err) => {
                            yield Err(CompletionError::ProviderError(format!(
                                "Failed to parse interactions SSE event: {err}"
                            )));
                            break;
                        }
                    };

                    yield Ok(data);
                }
                Err(crate::http_client::Error::StreamEnded) => break,
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    yield Err(CompletionError::ProviderError(error.to_string()));
                    break;
                }
            }
        }

        event_source.close();
    };

    Box::pin(stream)
}

fn parse_interaction_sse_event(
    data: &str,
) -> Result<Option<InteractionSseEvent>, InteractionSseParseError> {
    let trimmed = data.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Ok(None);
    }
    if !trimmed.starts_with('{') {
        return Err(InteractionSseParseError::UnexpectedPayload(
            trimmed.to_owned(),
        ));
    }

    let value: Value = serde_json::from_str(trimmed).map_err(InteractionSseParseError::Json)?;
    if !value.is_object() {
        return Err(InteractionSseParseError::UnexpectedPayload(
            trimmed.to_owned(),
        ));
    }

    serde_json::from_value::<InteractionSseEvent>(value)
        .map(Some)
        .map_err(InteractionSseParseError::Json)
}

#[derive(Debug)]
enum InteractionSseParseError {
    Json(serde_json::Error),
    UnexpectedPayload(String),
}

impl fmt::Display for InteractionSseParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "{error}"),
            Self::UnexpectedPayload(payload) => {
                write!(f, "unexpected non-object SSE payload: {payload}")
            }
        }
    }
}

impl ThoughtState {
    fn seed_from_start(
        &mut self,
        summary: Option<Vec<ThoughtSummaryContent>>,
        signature: Option<String>,
    ) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        self.signature = signature;
        self.push_summary_items(summary.unwrap_or_default())
    }

    fn push_summary(
        &mut self,
        content: ThoughtSummaryContent,
    ) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        self.push_summary_items([content])
    }

    fn push_summary_items(
        &mut self,
        summary: impl IntoIterator<Item = ThoughtSummaryContent>,
    ) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        summary
            .into_iter()
            .filter_map(|content| match content {
                ThoughtSummaryContent::Text(text) if !text.text.is_empty() => Some(text.text),
                _ => None,
            })
            .map(|text| {
                self.text.push_str(&text);
                streaming::RawStreamingChoice::ReasoningDelta {
                    id: None,
                    reasoning: text,
                }
            })
            .collect()
    }

    fn push_signature(&mut self, signature: String) {
        merge_chunked_string(&mut self.signature, signature);
    }

    fn finish(self) -> Option<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        let mut content = self
            .signature
            .into_iter()
            .map(ReasoningContent::Signature)
            .collect::<Vec<_>>();
        if !self.text.is_empty() {
            content.push(ReasoningContent::Summary(self.text));
        }

        if content.is_empty() {
            return None;
        }

        Some(streaming::RawStreamingChoice::Reasoning { id: None, content })
    }
}

impl FunctionCallState {
    fn seed_from_start(
        &mut self,
        FunctionCallContent {
            name,
            arguments,
            id,
        }: FunctionCallContent,
    ) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        self.apply_update(name, arguments, id)
    }

    fn push_delta(
        &mut self,
        FunctionCallDelta {
            name,
            arguments,
            id,
        }: FunctionCallDelta,
    ) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        self.apply_update(name, arguments, id)
    }

    fn apply_update(
        &mut self,
        name: Option<String>,
        arguments: Option<Value>,
        id: Option<String>,
    ) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        if let Some(id) = id {
            merge_chunked_string(&mut self.provider_call_id, id);
        }

        let mut choices = Vec::new();

        if let Some(name) = name {
            merge_chunked_string(&mut self.name, name);
            if !self.emitted_name {
                if let (Some(id), Some(name)) = (self.delta_identifier(), self.name.clone()) {
                    choices.push(streaming::RawStreamingChoice::ToolCallDelta {
                        id,
                        internal_call_id: self.internal_call_id.clone(),
                        content: ToolCallDeltaContent::Name(name),
                    });
                }
                self.emitted_name = true;
            }
        }

        if let Some(arguments) = arguments {
            let delta = arguments.to_string();
            merge_json_chunk(&mut self.arguments, arguments);
            if !delta.is_empty()
                && let Some(id) = self.delta_identifier()
            {
                choices.push(streaming::RawStreamingChoice::ToolCallDelta {
                    id,
                    internal_call_id: self.internal_call_id.clone(),
                    content: ToolCallDeltaContent::Delta(delta),
                });
            }
        }

        choices
    }

    fn delta_identifier(&self) -> Option<String> {
        self.provider_call_id.clone().or_else(|| self.name.clone())
    }

    fn finish(self) -> Option<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        let name = self.name?;
        let call_id = self.provider_call_id.unwrap_or_else(|| name.clone());

        Some(streaming::RawStreamingChoice::ToolCall(
            streaming::RawStreamingToolCall::new(
                name.clone(),
                name,
                self.arguments.unwrap_or(Value::Object(Map::new())),
            )
            .with_internal_call_id(self.internal_call_id)
            .with_call_id(call_id),
        ))
    }
}

impl InteractionStreamState {
    fn handle_content_start(
        &mut self,
        index: i32,
        content: StreamingContentStart,
    ) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        self.clear_index(index);

        match content {
            StreamingContentStart::Text(StreamingTextStart { text, .. }) => {
                if text.as_ref().is_none_or(String::is_empty) {
                    Vec::new()
                } else {
                    vec![streaming::RawStreamingChoice::Message(
                        text.expect("checked is_some"),
                    )]
                }
            }
            StreamingContentStart::FunctionCall(function_call) => {
                let mut state = FunctionCallState::default();
                let choices = state.seed_from_start(function_call);
                self.function_calls_by_index.insert(index, state);
                choices
            }
            StreamingContentStart::Thought(ThoughtContent { summary, signature }) => {
                let mut state = ThoughtState::default();
                let choices = state.seed_from_start(summary, signature);
                self.thoughts_by_index.insert(index, state);
                choices
            }
            _ => Vec::new(),
        }
    }

    fn handle_content_delta(
        &mut self,
        index: i32,
        delta: ContentDelta,
    ) -> Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        match delta {
            ContentDelta::Text(TextDelta {
                text: Some(text), ..
            }) => vec![streaming::RawStreamingChoice::Message(text)],
            ContentDelta::Text(TextDelta { text: None, .. }) => Vec::new(),
            ContentDelta::FunctionCall(function_call) => self
                .function_calls_by_index
                .entry(index)
                .or_default()
                .push_delta(function_call),
            ContentDelta::ThoughtSummary(ThoughtSummaryDelta { content }) => self
                .thoughts_by_index
                .entry(index)
                .or_default()
                .push_summary(content),
            ContentDelta::ThoughtSignature(ThoughtSignatureDelta { signature }) => {
                self.thoughts_by_index
                    .entry(index)
                    .or_default()
                    .push_signature(signature);
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    fn handle_content_stop(
        &mut self,
        index: i32,
    ) -> Option<streaming::RawStreamingChoice<StreamingCompletionResponse>> {
        if let Some(thought_state) = self.thoughts_by_index.remove(&index) {
            return thought_state.finish();
        }

        self.function_calls_by_index
            .remove(&index)
            .and_then(FunctionCallState::finish)
    }

    fn clear_index(&mut self, index: i32) {
        self.thoughts_by_index.remove(&index);
        self.function_calls_by_index.remove(&index);
    }
}

fn merge_chunked_string(target: &mut Option<String>, incoming: String) {
    if incoming.is_empty() {
        return;
    }

    match target {
        Some(existing) if incoming.starts_with(existing.as_str()) => *existing = incoming,
        Some(existing) if !existing.ends_with(&incoming) => existing.push_str(&incoming),
        Some(_) => {}
        None => *target = Some(incoming),
    }
}

fn merge_json_chunk(target: &mut Option<Value>, incoming: Value) {
    if incoming.is_null() {
        return;
    }

    match target {
        Some(existing) => merge_json_value(existing, incoming),
        None => *target = Some(incoming),
    }
}

fn merge_json_value(existing: &mut Value, incoming: Value) {
    match (existing, incoming) {
        (Value::String(existing), Value::String(incoming)) => existing.push_str(&incoming),
        (Value::Array(existing), Value::Array(mut incoming)) => existing.append(&mut incoming),
        (Value::Object(existing), Value::Object(incoming)) => {
            for (key, value) in incoming {
                if let Some(existing_value) = existing.get_mut(&key) {
                    merge_json_value(existing_value, value);
                } else {
                    existing.insert(key, value);
                }
            }
        }
        (existing, incoming) => *existing = incoming,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::gemini::interactions_api::{
        FunctionCallContent, InteractionSseEvent, StreamingContentStart, TextContent,
        ThoughtContent, ThoughtSummaryContent,
    };
    use crate::streaming::RawStreamingChoice;
    use serde_json::json;

    #[test]
    fn test_content_delta_text_event() {
        let event_json = json!({
            "event_type": "content.delta",
            "index": 0,
            "delta": {
                "type": "text",
                "text": "Hello"
            }
        });

        let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
        let InteractionSseEvent::ContentDelta { delta, .. } = event else {
            panic!("expected content delta");
        };

        let mut stream_state = InteractionStreamState::default();
        let mut choices = stream_state.handle_content_delta(0, delta);
        assert_eq!(choices.len(), 1);

        match choices.remove(0) {
            crate::streaming::RawStreamingChoice::Message(text) => {
                assert_eq!(text, "Hello");
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }

    #[test]
    fn test_parse_interaction_sse_event_ignores_done_payload() {
        let parsed =
            parse_interaction_sse_event("[DONE]").expect("non-json payload should be ignored");
        assert!(parsed.is_none());
    }

    #[test]
    fn test_parse_interaction_sse_event_rejects_unexpected_non_object_payload() {
        let error = parse_interaction_sse_event("keepalive")
            .expect_err("unexpected non-object payload should fail");
        assert_eq!(
            error.to_string(),
            "unexpected non-object SSE payload: keepalive"
        );
    }

    #[test]
    fn test_content_start_text_without_text_field_deserializes_via_streaming_type() {
        let event_json = json!({
            "event_type": "content.start",
            "index": 0,
            "content": {
                "type": "text"
            }
        });

        let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
        let InteractionSseEvent::ContentStart { content, .. } = event else {
            panic!("expected content start");
        };

        let mut stream_state = InteractionStreamState::default();
        let choices = stream_state.handle_content_start(0, content);
        assert!(choices.is_empty());
    }

    #[test]
    fn test_content_start_function_call_waits_for_stop_before_emitting_tool_call() {
        let mut stream_state = InteractionStreamState::default();

        let choices = stream_state.handle_content_start(
            0,
            StreamingContentStart::FunctionCall(FunctionCallContent {
                name: Some("get_weather".to_string()),
                arguments: Some(json!({"location": "Paris"})),
                id: Some("call-1".to_string()),
            }),
        );

        assert_eq!(choices.len(), 2);
        match &choices[0] {
            RawStreamingChoice::ToolCallDelta {
                id,
                internal_call_id: _,
                content: ToolCallDeltaContent::Name(name),
            } => {
                assert_eq!(id, "call-1");
                assert_eq!(name, "get_weather");
            }
            other => panic!("unexpected start choice: {other:?}"),
        }
        match &choices[1] {
            RawStreamingChoice::ToolCallDelta {
                id,
                internal_call_id: _,
                content: ToolCallDeltaContent::Delta(arguments),
            } => {
                assert_eq!(id, "call-1");
                assert_eq!(arguments, "{\"location\":\"Paris\"}");
            }
            other => panic!("unexpected start choice: {other:?}"),
        }

        let stop_choice = stream_state
            .handle_content_stop(0)
            .expect("stop should emit final tool call");
        match stop_choice {
            RawStreamingChoice::ToolCall(call) => {
                assert_eq!(call.name, "get_weather");
                assert_eq!(call.id, "get_weather");
                assert_eq!(call.call_id.as_deref(), Some("call-1"));
                assert_eq!(call.arguments, json!({"location": "Paris"}));
            }
            other => panic!("unexpected stop choice: {other:?}"),
        }
    }

    #[test]
    fn test_function_call_fragments_merge_across_deltas() {
        let mut stream_state = InteractionStreamState::default();

        let first_choices = stream_state.handle_content_delta(
            0,
            ContentDelta::FunctionCall(FunctionCallDelta {
                name: Some("lookup_".to_string()),
                arguments: None,
                id: Some("call_".to_string()),
            }),
        );
        assert_eq!(first_choices.len(), 1);
        match &first_choices[0] {
            RawStreamingChoice::ToolCallDelta {
                id,
                internal_call_id: _,
                content: ToolCallDeltaContent::Name(name),
            } => {
                assert_eq!(id, "call_");
                assert_eq!(name, "lookup_");
            }
            other => panic!("unexpected first choice: {other:?}"),
        }

        let second_choices = stream_state.handle_content_delta(
            0,
            ContentDelta::FunctionCall(FunctionCallDelta {
                name: Some("order".to_string()),
                arguments: Some(json!({"order_id": "A-17"})),
                id: Some("1".to_string()),
            }),
        );
        assert_eq!(second_choices.len(), 1);
        match &second_choices[0] {
            RawStreamingChoice::ToolCallDelta {
                id,
                internal_call_id: _,
                content: ToolCallDeltaContent::Delta(arguments),
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(arguments, "{\"order_id\":\"A-17\"}");
            }
            other => panic!("unexpected second choice: {other:?}"),
        }

        let third_choices = stream_state.handle_content_delta(
            0,
            ContentDelta::FunctionCall(FunctionCallDelta {
                name: None,
                arguments: Some(json!({"status": "backordered"})),
                id: None,
            }),
        );
        assert_eq!(third_choices.len(), 1);
        match &third_choices[0] {
            RawStreamingChoice::ToolCallDelta {
                id,
                internal_call_id: _,
                content: ToolCallDeltaContent::Delta(arguments),
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(arguments, "{\"status\":\"backordered\"}");
            }
            other => panic!("unexpected third choice: {other:?}"),
        }

        let stop_choice = stream_state
            .handle_content_stop(0)
            .expect("stop should emit final tool call");
        match stop_choice {
            RawStreamingChoice::ToolCall(call) => {
                assert_eq!(call.name, "lookup_order");
                assert_eq!(call.id, "lookup_order");
                assert_eq!(call.call_id.as_deref(), Some("call_1"));
                assert_eq!(
                    call.arguments,
                    json!({
                        "order_id": "A-17",
                        "status": "backordered"
                    })
                );
            }
            other => panic!("unexpected stop choice: {other:?}"),
        }
    }

    #[test]
    fn test_thought_summary_and_signature_emit_final_signed_reasoning() {
        let mut stream_state = InteractionStreamState::default();

        let start_choices = stream_state.handle_content_start(
            0,
            StreamingContentStart::Thought(ThoughtContent {
                summary: None,
                signature: None,
            }),
        );
        assert!(start_choices.is_empty());

        let mut summary_choices = stream_state.handle_content_delta(
            0,
            ContentDelta::ThoughtSummary(ThoughtSummaryDelta {
                content: ThoughtSummaryContent::Text(TextContent {
                    text: "thinking...".to_string(),
                    annotations: None,
                }),
            }),
        );
        assert_eq!(summary_choices.len(), 1);
        match summary_choices.remove(0) {
            RawStreamingChoice::ReasoningDelta { id, reasoning } => {
                assert_eq!(id, None);
                assert_eq!(reasoning, "thinking...");
            }
            other => panic!("unexpected choice: {other:?}"),
        }

        let signature_choices = stream_state.handle_content_delta(
            0,
            ContentDelta::ThoughtSignature(ThoughtSignatureDelta {
                signature: "sig-123".to_string(),
            }),
        );
        assert!(signature_choices.is_empty());

        let stop_choice = stream_state
            .handle_content_stop(0)
            .expect("stop should emit reasoning");
        match stop_choice {
            RawStreamingChoice::Reasoning { id, content } => {
                assert_eq!(id, None);
                assert_eq!(
                    content,
                    vec![
                        ReasoningContent::Signature("sig-123".to_string()),
                        ReasoningContent::Summary("thinking...".to_string()),
                    ]
                );
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }

    #[test]
    fn test_signature_only_thought_emits_final_signed_reasoning() {
        let mut stream_state = InteractionStreamState::default();

        let signature_choices = stream_state.handle_content_delta(
            0,
            ContentDelta::ThoughtSignature(ThoughtSignatureDelta {
                signature: "sig-only".to_string(),
            }),
        );
        assert!(signature_choices.is_empty());

        let stop_choice = stream_state
            .handle_content_stop(0)
            .expect("stop should emit reasoning");
        match stop_choice {
            RawStreamingChoice::Reasoning { id, content } => {
                assert_eq!(id, None);
                assert_eq!(
                    content,
                    vec![ReasoningContent::Signature("sig-only".to_string())]
                );
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }

    #[test]
    fn test_content_stop_without_tracked_thought_emits_nothing() {
        let mut stream_state = InteractionStreamState::default();
        assert!(stream_state.handle_content_stop(0).is_none());
    }

    #[test]
    fn test_content_start_thought_seeds_summary_and_signature() {
        let mut stream_state = InteractionStreamState::default();

        let start_choices = stream_state.handle_content_start(
            0,
            StreamingContentStart::Thought(ThoughtContent {
                summary: Some(vec![ThoughtSummaryContent::Text(TextContent {
                    text: "seeded".to_string(),
                    annotations: None,
                })]),
                signature: Some("sig-seeded".to_string()),
            }),
        );

        assert_eq!(start_choices.len(), 1);
        match &start_choices[0] {
            RawStreamingChoice::ReasoningDelta { id, reasoning } => {
                assert_eq!(id, &None);
                assert_eq!(reasoning, "seeded");
            }
            other => panic!("unexpected choice: {other:?}"),
        }

        let stop_choice = stream_state
            .handle_content_stop(0)
            .expect("stop should emit reasoning");
        match stop_choice {
            RawStreamingChoice::Reasoning { id, content } => {
                assert_eq!(id, None);
                assert_eq!(
                    content,
                    vec![
                        ReasoningContent::Signature("sig-seeded".to_string()),
                        ReasoningContent::Summary("seeded".to_string()),
                    ]
                );
            }
            other => panic!("unexpected choice: {other:?}"),
        }
    }
}
