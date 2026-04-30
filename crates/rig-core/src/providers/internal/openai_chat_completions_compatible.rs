//! Shared helpers for OpenAI Chat Completions-compatible streaming providers.
//!
//! Several providers expose an SSE stream that looks like OpenAI Chat
//! Completions: text arrives in deltas, tool calls are streamed piecemeal, and
//! a trailing event may carry usage. This module centralizes the common stream
//! state machine while leaving request parsing and provider-specific metadata to
//! small profile hooks.

use std::collections::HashMap;

use async_stream::stream;
use futures::StreamExt;
use http::Request;
use tracing_futures::Instrument;

use crate::completion::{CompletionError, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::json_utils;
use crate::streaming::{self, RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent};
use crate::wasm_compat::WasmCompatSend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompatibleFinishReason {
    ToolCalls,
    Other,
}

#[derive(Debug, Clone)]
pub(crate) struct CompatibleToolCallChunk {
    pub(crate) index: usize,
    pub(crate) id: Option<String>,
    pub(crate) name: Option<String>,
    pub(crate) arguments: Option<String>,
}

impl CompatibleToolCallChunk {
    fn has_nonempty_name(&self) -> bool {
        self.name.as_ref().is_some_and(|name| !name.is_empty())
    }

    fn has_nonempty_arguments(&self) -> bool {
        self.arguments
            .as_ref()
            .is_some_and(|arguments| !arguments.is_empty())
    }

    fn starts_new_tool_call(&self) -> bool {
        self.has_nonempty_name()
            && self
                .arguments
                .as_ref()
                .map(|arguments| arguments.is_empty())
                .unwrap_or(true)
    }

    fn is_complete_single_chunk(&self) -> bool {
        self.has_nonempty_name() && self.has_nonempty_arguments()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CompatibleChoice<D> {
    pub(crate) finish_reason: CompatibleFinishReason,
    pub(crate) text: Option<String>,
    pub(crate) reasoning: Option<String>,
    pub(crate) tool_calls: Vec<CompatibleToolCallChunk>,
    pub(crate) details: Vec<D>,
}

#[derive(Debug, Clone)]
pub(crate) struct CompatibleChoiceData<T, D> {
    pub(crate) finish_reason: CompatibleFinishReason,
    pub(crate) text: Option<String>,
    pub(crate) reasoning: Option<String>,
    pub(crate) tool_calls: Vec<T>,
    pub(crate) details: Vec<D>,
}

#[derive(Debug, Clone)]
pub(crate) struct CompatibleChunk<U, D> {
    pub(crate) response_id: Option<String>,
    pub(crate) response_model: Option<String>,
    pub(crate) choice: Option<CompatibleChoice<D>>,
    pub(crate) usage: Option<U>,
}

pub(crate) type NormalizedCompatibleChunk<U, D> =
    Result<Option<CompatibleChunk<U, D>>, CompletionError>;

impl<T, D> From<CompatibleChoiceData<T, D>> for CompatibleChoice<D>
where
    T: Into<CompatibleToolCallChunk>,
{
    fn from(value: CompatibleChoiceData<T, D>) -> Self {
        Self {
            finish_reason: value.finish_reason,
            text: value.text,
            reasoning: value.reasoning,
            tool_calls: value.tool_calls.into_iter().map(Into::into).collect(),
            details: value.details,
        }
    }
}

pub(crate) fn normalize_first_choice_chunk<U, D, Choice, ToolCall, F>(
    response_id: Option<String>,
    response_model: Option<String>,
    usage: Option<U>,
    choices: &[Choice],
    map_choice: F,
) -> CompatibleChunk<U, D>
where
    ToolCall: Into<CompatibleToolCallChunk>,
    F: FnOnce(&Choice) -> CompatibleChoiceData<ToolCall, D>,
{
    let choice = choices.first().map(|choice| map_choice(choice).into());

    CompatibleChunk {
        response_id,
        response_model,
        choice,
        usage,
    }
}

pub(crate) fn tool_call_chunks<T>(tool_calls: &[T]) -> Vec<CompatibleToolCallChunk>
where
    for<'a> CompatibleToolCallChunk: From<&'a T>,
{
    tool_calls
        .iter()
        .map(CompatibleToolCallChunk::from)
        .collect()
}

pub(crate) trait CompatibleStreamProfile: WasmCompatSend {
    type Usage: Clone + Default + GetTokenUsage + WasmCompatSend + 'static;
    type Detail: WasmCompatSend + 'static;
    type FinalResponse: Clone + Unpin + GetTokenUsage + WasmCompatSend + 'static;

    fn normalize_chunk(&self, data: &str) -> NormalizedCompatibleChunk<Self::Usage, Self::Detail>;

    fn build_final_response(&self, usage: Self::Usage) -> Self::FinalResponse;

    fn uses_distinct_tool_call_eviction(&self) -> bool {
        false
    }

    fn should_evict(
        &self,
        existing: &RawStreamingToolCall,
        incoming: &CompatibleToolCallChunk,
    ) -> bool {
        self.uses_distinct_tool_call_eviction()
            && should_evict_distinct_named_tool_call(existing, incoming)
    }

    fn decorate_tool_call(
        &self,
        _detail: &Self::Detail,
        _tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
    ) {
    }

    fn emits_complete_single_chunk_tool_calls(&self) -> bool {
        false
    }

    fn should_emit_completed_tool_call_immediately(
        &self,
        _tool_call: &RawStreamingToolCall,
        incoming: &CompatibleToolCallChunk,
    ) -> bool {
        self.emits_complete_single_chunk_tool_calls() && incoming.is_complete_single_chunk()
    }
}

pub(crate) fn should_evict_distinct_named_tool_call(
    existing: &RawStreamingToolCall,
    incoming: &CompatibleToolCallChunk,
) -> bool {
    if let Some(new_id) = &incoming.id
        && !new_id.is_empty()
        && let Some(new_name) = &incoming.name
        && incoming.has_nonempty_name()
        && !existing.id.is_empty()
        && existing.id != *new_id
        && !existing.name.is_empty()
    {
        return existing.name != *new_name || incoming.starts_new_tool_call();
    }

    false
}

pub(crate) async fn send_compatible_streaming_request<T, P>(
    http_client: T,
    req: Request<Vec<u8>>,
    profile: P,
) -> Result<streaming::StreamingCompletionResponse<P::FinalResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
    P: CompatibleStreamProfile + 'static,
{
    let span = tracing::Span::current();
    let instrument_span = span.clone();
    let mut event_source = GenericEventSource::new(http_client, req);

    let stream = stream! {
        let mut tool_calls: HashMap<usize, RawStreamingToolCall> = HashMap::new();
        let mut final_usage = None;
        let mut terminated_with_error = false;

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }
                Ok(Event::Message(message)) => {
                    if message.data.trim().is_empty() || message.data == "[DONE]" {
                        continue;
                    }

                    let chunk = match profile.normalize_chunk(&message.data) {
                        Ok(Some(chunk)) => chunk,
                        Ok(None) => continue,
                        Err(error) => {
                            terminated_with_error = true;
                            yield Err(error);
                            break;
                        }
                    };

                    record_response_metadata(
                        &span,
                        chunk.response_id.as_deref(),
                        chunk.response_model.as_deref(),
                    );

                    if let Some(usage) = chunk.usage {
                        final_usage = Some(usage);
                    }

                    let Some(choice) = chunk.choice else {
                        continue;
                    };

                    for incoming in choice.tool_calls {
                        if let Some(existing) = tool_calls.get(&incoming.index)
                            && profile.should_evict(existing, &incoming)
                            && let Some(evicted) = tool_calls.remove(&incoming.index)
                        {
                            yield Ok(RawStreamingChoice::ToolCall(
                                finalize_completed_streaming_tool_call(evicted),
                            ));
                        }

                        let existing_tool_call = tool_calls
                            .entry(incoming.index)
                            .or_insert_with(RawStreamingToolCall::empty);

                        if let Some(id) = incoming.id.as_ref()
                            && !id.is_empty()
                        {
                            existing_tool_call.id = id.clone();
                        }

                        if let Some(name) = incoming.name.as_ref()
                            && !name.is_empty()
                        {
                            existing_tool_call.name = name.clone();
                            yield Ok(RawStreamingChoice::ToolCallDelta {
                                id: existing_tool_call.id.clone(),
                                internal_call_id: existing_tool_call.internal_call_id.clone(),
                                content: ToolCallDeltaContent::Name(name.clone()),
                            });
                        }

                        if let Some(arguments) = incoming.arguments.as_ref()
                            && !arguments.is_empty()
                        {
                            append_tool_call_arguments(existing_tool_call, arguments);
                            yield Ok(RawStreamingChoice::ToolCallDelta {
                                id: existing_tool_call.id.clone(),
                                internal_call_id: existing_tool_call.internal_call_id.clone(),
                                content: ToolCallDeltaContent::Delta(arguments.clone()),
                            });
                        }

                        let emit_completed_tool_call_immediately = profile
                            .should_emit_completed_tool_call_immediately(
                                existing_tool_call,
                                &incoming,
                            );
                        let finalized_tool_call = emit_completed_tool_call_immediately
                            .then(|| tool_calls.get(&incoming.index).cloned())
                            .flatten()
                            .and_then(finalize_pending_tool_call);

                        if let Some(tool_call) = finalized_tool_call {
                            tool_calls.remove(&incoming.index);
                            yield Ok(RawStreamingChoice::ToolCall(tool_call));
                        }
                    }

                    for detail in &choice.details {
                        profile.decorate_tool_call(detail, &mut tool_calls);
                    }

                    if let Some(reasoning) = choice.reasoning
                        && !reasoning.is_empty()
                    {
                        yield Ok(RawStreamingChoice::ReasoningDelta {
                            id: None,
                            reasoning,
                        });
                    }

                    if let Some(content) = choice.text
                        && !content.is_empty()
                    {
                        yield Ok(RawStreamingChoice::Message(content));
                    }

                    if choice.finish_reason == CompatibleFinishReason::ToolCalls {
                        for tool_call in take_finalized_tool_calls(
                            &mut tool_calls,
                            DroppedToolCallContext::ToolCallsFinishReason,
                        ) {
                            yield Ok(RawStreamingChoice::ToolCall(tool_call));
                        }
                    }
                }
                Err(crate::http_client::Error::StreamEnded) => {
                    break;
                }
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    terminated_with_error = true;
                    yield Err(CompletionError::ProviderError(error.to_string()));
                    break;
                }
            }
        }

        event_source.close();

        if terminated_with_error {
            return;
        }

        for tool_call in
            take_finalized_tool_calls(&mut tool_calls, DroppedToolCallContext::EndOfStream)
        {
            yield Ok(RawStreamingChoice::ToolCall(tool_call));
        }

        let final_usage = final_usage.unwrap_or_default();
        record_usage(&span, &final_usage);
        yield Ok(RawStreamingChoice::FinalResponse(
            profile.build_final_response(final_usage),
        ));
    }
    .instrument(instrument_span);

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream,
    )))
}

fn record_usage<T>(span: &tracing::Span, usage: &T)
where
    T: GetTokenUsage,
{
    if span.is_disabled() {
        return;
    }

    let Some(usage) = usage.token_usage() else {
        return;
    };

    span.record("gen_ai.usage.input_tokens", usage.input_tokens);
    span.record("gen_ai.usage.output_tokens", usage.output_tokens);
    span.record(
        "gen_ai.usage.cache_read.input_tokens",
        usage.cached_input_tokens,
    );
}

fn record_response_metadata(
    span: &tracing::Span,
    response_id: Option<&str>,
    response_model: Option<&str>,
) {
    if span.is_disabled() {
        return;
    }

    if let Some(response_id) = response_id
        && !response_id.is_empty()
    {
        span.record("gen_ai.response.id", response_id);
    }

    if let Some(response_model) = response_model
        && !response_model.is_empty()
    {
        span.record("gen_ai.response.model", response_model);
    }
}

fn append_tool_call_arguments(tool_call: &mut RawStreamingToolCall, chunk: &str) {
    let current_args = match &tool_call.arguments {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(existing) => {
            // Some OpenAI-compatible gateways emit a literal `null` placeholder
            // before streaming the real JSON argument fragments. Once a later
            // fragment arrives, treat that placeholder as empty so it doesn't
            // poison the accumulated payload.
            if existing.trim() == "null" && !chunk.trim().is_empty() {
                String::new()
            } else {
                existing.clone()
            }
        }
        value => value.to_string(),
    };

    let combined = format!("{current_args}{chunk}");

    if combined.trim_start().starts_with('{') && combined.trim_end().ends_with('}') {
        match serde_json::from_str(&combined) {
            Ok(parsed) => tool_call.arguments = parsed,
            Err(_) => tool_call.arguments = serde_json::Value::String(combined),
        }
    } else {
        tool_call.arguments = serde_json::Value::String(combined);
    }
}

pub(crate) fn finalize_completed_streaming_tool_call(
    mut tool_call: RawStreamingToolCall,
) -> RawStreamingToolCall {
    if tool_call.arguments.is_null() {
        tool_call.arguments = serde_json::Value::Object(serde_json::Map::new());
    }

    tool_call
}

fn finalize_pending_tool_call(mut tool_call: RawStreamingToolCall) -> Option<RawStreamingToolCall> {
    // Canonical cleanup for OpenAI Chat Completions-compatible providers:
    // a pending tool call with an established name but no streamed arguments is
    // treated as a valid parameterless invocation and normalized to `{}`.
    // Only nameless entries or syntactically partial argument payloads are dropped.
    if tool_call.name.is_empty() {
        return None;
    }

    match &tool_call.arguments {
        serde_json::Value::Null => Some(finalize_completed_streaming_tool_call(tool_call)),
        serde_json::Value::String(arguments) => {
            if arguments.trim().is_empty() {
                tool_call.arguments = serde_json::Value::Object(serde_json::Map::new());
                return Some(tool_call);
            }

            let parsed = json_utils::parse_tool_arguments(arguments).ok()?;
            tool_call.arguments = parsed;
            Some(tool_call)
        }
        _ => Some(tool_call),
    }
}

#[derive(Clone, Copy)]
enum DroppedToolCallContext {
    ToolCallsFinishReason,
    EndOfStream,
}

fn drain_finalized_tool_calls(
    tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
) -> (Vec<RawStreamingToolCall>, usize) {
    let mut completed_tool_calls = Vec::new();
    let mut dropped_tool_calls = 0;

    for (_, tool_call) in tool_calls.drain() {
        if let Some(tool_call) = finalize_pending_tool_call(tool_call) {
            completed_tool_calls.push(tool_call);
        } else {
            dropped_tool_calls += 1;
        }
    }

    (completed_tool_calls, dropped_tool_calls)
}

fn take_finalized_tool_calls(
    tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
    context: DroppedToolCallContext,
) -> Vec<RawStreamingToolCall> {
    let (completed_tool_calls, dropped_tool_calls) = drain_finalized_tool_calls(tool_calls);

    if dropped_tool_calls > 0 {
        match context {
            DroppedToolCallContext::ToolCallsFinishReason => tracing::debug!(
                dropped_tool_calls,
                "Dropping incomplete tool calls on tool_calls finish reason"
            ),
            DroppedToolCallContext::EndOfStream => {
                tracing::debug!(
                    dropped_tool_calls,
                    "Dropping incomplete tool calls at stream end"
                )
            }
        }
    }

    completed_tool_calls
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::completion::GetTokenUsage;
    use crate::streaming::{self, StreamedAssistantContent};
    use bytes::Bytes;
    use futures::StreamExt;

    pub(crate) fn sse_bytes_from_data_lines<T>(events: impl IntoIterator<Item = T>) -> Bytes
    where
        T: AsRef<str>,
    {
        Bytes::from(
            events
                .into_iter()
                .map(|event| format!("data: {}\n\n", event.as_ref()))
                .collect::<String>(),
        )
    }

    pub(crate) fn sse_bytes_from_json_events(events: &[serde_json::Value]) -> Bytes {
        Bytes::from(
            events
                .iter()
                .map(|event| {
                    format!(
                        "data: {}\n\n",
                        serde_json::to_string(event).expect("event should serialize")
                    )
                })
                .collect::<String>(),
        )
    }

    pub(crate) async fn assert_zero_arg_tool_call_is_emitted<R>(
        mut stream: streaming::StreamingCompletionResponse<R>,
        expected_id: &str,
        expected_name: &str,
        expect_final_response: bool,
    ) where
        R: Clone + Unpin + GetTokenUsage,
    {
        let mut saw_final = false;
        let mut collected_tool_calls = Vec::new();

        while let Some(chunk) = stream.next().await {
            match chunk.expect("stream item should be ok") {
                StreamedAssistantContent::ToolCallDelta { .. } => {}
                StreamedAssistantContent::Final(_) => saw_final = true,
                StreamedAssistantContent::ToolCall { tool_call, .. } => {
                    collected_tool_calls.push(tool_call);
                }
                _ => panic!("unexpected stream item while asserting zero-arg tool call"),
            }
        }

        if expect_final_response {
            assert!(saw_final, "stream should still yield a final response");
        }

        assert_eq!(collected_tool_calls.len(), 1);
        assert_eq!(collected_tool_calls[0].id, expected_id);
        assert_eq!(collected_tool_calls[0].function.name, expected_name);
        assert_eq!(
            collected_tool_calls[0].function.arguments,
            serde_json::json!({})
        );
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::sse_bytes_from_data_lines;
    use super::{
        CompatibleChoice, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
        CompatibleToolCallChunk, finalize_pending_tool_call, send_compatible_streaming_request,
    };
    use crate::completion::{CompletionError, GetTokenUsage};
    use crate::http_client::mock::MockStreamingClient;
    use crate::streaming::RawStreamingToolCall;
    use crate::streaming::StreamedAssistantContent;
    use futures::StreamExt;

    #[derive(Clone, Default)]
    struct TestUsage;

    impl GetTokenUsage for TestUsage {
        fn token_usage(&self) -> Option<crate::completion::Usage> {
            None
        }
    }

    #[derive(Clone, Default, Debug)]
    struct TestFinalResponse;

    impl GetTokenUsage for TestFinalResponse {
        fn token_usage(&self) -> Option<crate::completion::Usage> {
            None
        }
    }

    fn test_chunk(choice: CompatibleChoice<()>) -> CompatibleChunk<TestUsage, ()> {
        CompatibleChunk {
            response_id: None,
            response_model: None,
            choice: Some(choice),
            usage: None,
        }
    }

    fn tool_call_choice(
        finish_reason: CompatibleFinishReason,
        tool_calls: Vec<CompatibleToolCallChunk>,
    ) -> CompatibleChoice<()> {
        CompatibleChoice {
            finish_reason,
            text: None,
            reasoning: None,
            tool_calls,
            details: Vec::new(),
        }
    }

    fn tool_call_chunk(
        index: usize,
        id: Option<&str>,
        name: Option<&str>,
        arguments: Option<&str>,
    ) -> CompatibleToolCallChunk {
        CompatibleToolCallChunk {
            index,
            id: id.map(ToOwned::to_owned),
            name: name.map(ToOwned::to_owned),
            arguments: arguments.map(ToOwned::to_owned),
        }
    }

    #[derive(Clone, Copy)]
    struct ErrorAfterPendingToolCallProfile;

    impl CompatibleStreamProfile for ErrorAfterPendingToolCallProfile {
        type Usage = TestUsage;
        type Detail = ();
        type FinalResponse = TestFinalResponse;

        fn normalize_chunk(
            &self,
            data: &str,
        ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
            match data {
                "start" => Ok(Some(test_chunk(tool_call_choice(
                    CompatibleFinishReason::Other,
                    vec![tool_call_chunk(0, Some("call_123"), Some("ping"), Some(""))],
                )))),
                "bad" => Err(CompletionError::ProviderError(
                    "normalize failed".to_owned(),
                )),
                _ => Ok(None),
            }
        }

        fn build_final_response(&self, _usage: Self::Usage) -> Self::FinalResponse {
            TestFinalResponse
        }
    }

    #[derive(Clone, Copy)]
    struct DistinctToolCallEvictionProfile;

    impl CompatibleStreamProfile for DistinctToolCallEvictionProfile {
        type Usage = TestUsage;
        type Detail = ();
        type FinalResponse = TestFinalResponse;

        fn normalize_chunk(
            &self,
            data: &str,
        ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
            let choice = match data {
                "first_start" => Some(tool_call_choice(
                    CompatibleFinishReason::Other,
                    vec![tool_call_chunk(
                        0,
                        Some("call_aaa"),
                        Some("search"),
                        Some(""),
                    )],
                )),
                "first_args" => Some(tool_call_choice(
                    CompatibleFinishReason::Other,
                    vec![tool_call_chunk(0, None, None, Some("{\"query\":\"one\"}"))],
                )),
                "second_start" => Some(tool_call_choice(
                    CompatibleFinishReason::Other,
                    vec![tool_call_chunk(
                        0,
                        Some("call_bbb"),
                        Some("search"),
                        Some(""),
                    )],
                )),
                "second_args" => Some(tool_call_choice(
                    CompatibleFinishReason::Other,
                    vec![tool_call_chunk(0, None, None, Some("{\"query\":\"two\"}"))],
                )),
                "finish" => Some(tool_call_choice(
                    CompatibleFinishReason::ToolCalls,
                    Vec::new(),
                )),
                _ => None,
            };

            Ok(choice.map(test_chunk))
        }

        fn build_final_response(&self, _usage: Self::Usage) -> Self::FinalResponse {
            TestFinalResponse
        }

        fn uses_distinct_tool_call_eviction(&self) -> bool {
            true
        }
    }

    #[derive(Clone, Copy)]
    struct FinishReasonCleanupProfile;

    impl CompatibleStreamProfile for FinishReasonCleanupProfile {
        type Usage = TestUsage;
        type Detail = ();
        type FinalResponse = TestFinalResponse;

        fn normalize_chunk(
            &self,
            data: &str,
        ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
            let choice = match data {
                "start" => Some(tool_call_choice(
                    CompatibleFinishReason::Other,
                    vec![tool_call_chunk(
                        0,
                        Some("call_123"),
                        Some("ping"),
                        Some("{\"x\":"),
                    )],
                )),
                "finish" => Some(tool_call_choice(
                    CompatibleFinishReason::ToolCalls,
                    Vec::new(),
                )),
                _ => None,
            };

            Ok(choice.map(test_chunk))
        }

        fn build_final_response(&self, _usage: Self::Usage) -> Self::FinalResponse {
            TestFinalResponse
        }
    }

    #[test]
    fn eof_cleanup_preserves_parameterless_tool_calls() {
        let tool_call = RawStreamingToolCall::new(
            "call_123".to_owned(),
            "ping".to_owned(),
            serde_json::Value::Null,
        );

        let finalized =
            finalize_pending_tool_call(tool_call).expect("tool call should be preserved");

        assert_eq!(finalized.id, "call_123");
        assert_eq!(finalized.name, "ping");
        assert_eq!(finalized.arguments, serde_json::json!({}));
    }

    #[test]
    fn eof_cleanup_preserves_empty_argument_chunks_as_empty_object() {
        let tool_call = RawStreamingToolCall::new(
            "call_123".to_owned(),
            "ping".to_owned(),
            serde_json::Value::String(String::new()),
        );

        let finalized =
            finalize_pending_tool_call(tool_call).expect("tool call should be preserved");

        assert_eq!(finalized.arguments, serde_json::json!({}));
    }

    #[test]
    fn eof_cleanup_drops_nameless_pending_entries() {
        let tool_call = RawStreamingToolCall::empty();

        assert!(finalize_pending_tool_call(tool_call).is_none());
    }

    #[test]
    fn eof_cleanup_drops_partial_argument_payloads() {
        let tool_call = RawStreamingToolCall::new(
            "call_123".to_owned(),
            "ping".to_owned(),
            serde_json::Value::String("{\"x\":".to_owned()),
        );

        assert!(finalize_pending_tool_call(tool_call).is_none());
    }

    #[test]
    fn null_placeholder_is_replaced_by_following_json_fragments() {
        let mut tool_call = RawStreamingToolCall::new(
            "call_123".to_owned(),
            "web_search".to_owned(),
            serde_json::Value::String("null".to_owned()),
        );

        super::append_tool_call_arguments(&mut tool_call, "{\"query\": \"META");
        super::append_tool_call_arguments(&mut tool_call, " Platforms news\"}");

        let finalized =
            finalize_pending_tool_call(tool_call).expect("tool call should be preserved");

        assert_eq!(
            finalized.arguments,
            serde_json::json!({"query": "META Platforms news"})
        );
    }

    #[tokio::test]
    async fn normalize_chunk_errors_terminate_without_flushing_or_finalizing() {
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines(["start", "bad"]),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream =
            send_compatible_streaming_request(client, req, ErrorAfterPendingToolCallProfile)
                .await
                .expect("stream should start");

        match stream
            .next()
            .await
            .expect("expected tool call delta before normalize error")
            .expect("first item should be ok")
        {
            StreamedAssistantContent::ToolCallDelta { id, content, .. } => {
                assert_eq!(id, "call_123");
                assert_eq!(
                    content,
                    crate::streaming::ToolCallDeltaContent::Name("ping".to_owned())
                );
            }
            other => panic!("expected tool call delta, got {other:?}"),
        }

        let err = stream
            .next()
            .await
            .expect("expected normalize error")
            .expect_err("second item should be the normalize error");
        assert_eq!(err.to_string(), "ProviderError: normalize failed");

        assert!(
            stream.next().await.is_none(),
            "stream should terminate immediately after normalize_chunk error"
        );
    }

    #[tokio::test]
    async fn distinct_same_name_tool_calls_evict_by_id_when_a_new_call_starts() {
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "first_start",
                "first_args",
                "second_start",
                "second_args",
                "finish",
            ]),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream =
            send_compatible_streaming_request(client, req, DistinctToolCallEvictionProfile)
                .await
                .expect("stream should start");

        let mut collected_tool_calls = Vec::new();
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::ToolCall { tool_call, .. } =
                item.expect("stream item should be ok")
            {
                collected_tool_calls.push(tool_call);
            }
        }

        assert_eq!(collected_tool_calls.len(), 2);
        assert_eq!(collected_tool_calls[0].id, "call_aaa");
        assert_eq!(collected_tool_calls[0].function.name, "search");
        assert_eq!(
            collected_tool_calls[0].function.arguments,
            serde_json::json!({"query":"one"})
        );
        assert_eq!(collected_tool_calls[1].id, "call_bbb");
        assert_eq!(collected_tool_calls[1].function.name, "search");
        assert_eq!(
            collected_tool_calls[1].function.arguments,
            serde_json::json!({"query":"two"})
        );
    }

    #[tokio::test]
    async fn tool_calls_finish_reason_drops_partial_argument_payloads() {
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines(["start", "finish"]),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_compatible_streaming_request(client, req, FinishReasonCleanupProfile)
            .await
            .expect("stream should start");

        let mut saw_final = false;
        let mut saw_tool_call = false;

        while let Some(item) = stream.next().await {
            match item.expect("stream item should be ok") {
                StreamedAssistantContent::ToolCallDelta { .. } => {}
                StreamedAssistantContent::Final(_) => saw_final = true,
                StreamedAssistantContent::ToolCall { .. } => saw_tool_call = true,
                other => panic!(
                    "unexpected stream item while asserting finish-reason cleanup: {other:?}"
                ),
            }
        }

        assert!(
            saw_final,
            "stream should still yield a final response after dropping the partial tool call"
        );
        assert!(
            !saw_tool_call,
            "finish_reason cleanup should drop partial tool calls instead of emitting them"
        );
    }
}
