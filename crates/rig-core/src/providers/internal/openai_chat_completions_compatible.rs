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

use crate::completion::{CompletionError, FinishReason, Usage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::streaming::{
    self, RawStreamingChoice, ToolCallDecoration, ToolCallDeltaContent, ToolInputEnd,
    UnparseableToolInput,
};
use crate::wasm_compat::WasmCompatSend;

fn provider_response_from_compatible_sse_data(data: &str) -> Option<CompletionError> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    // Treat the chunk as an error only when `error` is present AND carries a
    // payload: either an object (`{"error":{...}}`, the canonical OpenAI-compatible
    // error event) or a non-empty string (`{"error":"oops"}`, used by some
    // gateways). A `{"error":null}` or `{"error":""}` chunk — which some providers
    // send alongside the terminal usage event — must not terminate the stream.
    let error = value
        .get("error")
        .filter(|error| error.is_object() || error.as_str().is_some_and(|s| !s.is_empty()))?;
    if value.get("choices").is_some() {
        return None;
    }

    if let Some(message) = error.get("message").and_then(serde_json::Value::as_str) {
        tracing::warn!(message, "provider returned a streaming error event");
    }

    Some(crate::provider_response::completion_error_from_body(data))
}

/// Map an OpenAI Chat Completions-style `finish_reason` string onto the
/// normalized vocabulary, preserving anything unrecognized verbatim.
///
/// Shared by the unary and streaming paths so both agree, and so a gateway
/// inventing a new reason surfaces it rather than reading as a natural stop.
pub(crate) fn map_openai_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" | "max_tokens" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// A chunk's terminal reason, as reported by an OpenAI-compatible provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompatibleFinishReason {
    /// The chunk reported a terminal reason, normalized.
    Reported(FinishReason),
    /// The chunk carried no `finish_reason` field.
    Absent,
}

impl CompatibleFinishReason {
    /// Normalize a wire `finish_reason` field.
    pub(crate) fn from_wire(reason: Option<&str>) -> Self {
        match reason.filter(|reason| !reason.is_empty()) {
            Some(reason) => Self::Reported(map_openai_finish_reason(reason)),
            None => Self::Absent,
        }
    }

    /// Whether the provider explicitly ended the turn to call tools.
    pub(crate) fn is_tool_calls(&self) -> bool {
        matches!(self, Self::Reported(FinishReason::ToolCalls))
    }

    /// The normalized reason, when the provider reported one.
    pub(crate) fn reported(&self) -> Option<FinishReason> {
        match self {
            Self::Reported(reason) => Some(reason.clone()),
            Self::Absent => None,
        }
    }
}

/// The terminal state a compatible stream reached, handed to a profile so it
/// can build its own provider-native terminal record.
#[derive(Debug, Clone)]
pub(crate) struct CompatibleTerminal<U> {
    /// Provider-native usage payload from the terminal event.
    pub(crate) usage: U,
    /// Normalized finish reason, when the stream reported one.
    pub(crate) finish_reason: Option<FinishReason>,
    /// Provider-assigned response identifier, when emitted.
    pub(crate) response_id: Option<String>,
    /// Provider-reported model identifier, when emitted.
    pub(crate) model: Option<String>,
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
    type Usage: Clone + Default + Into<Usage> + WasmCompatSend + 'static;
    type Detail: WasmCompatSend + 'static;
    type FinalResponse: Clone + WasmCompatSend + 'static;

    fn normalize_chunk(&self, data: &str) -> NormalizedCompatibleChunk<Self::Usage, Self::Detail>;

    /// Build the provider's own terminal record from the stream's terminal
    /// state. The record stays provider-native for `raw_stream`; the normalized
    /// path maps it once through [`crate::streaming::normalize_stream`].
    fn build_final_response(
        &self,
        terminal: CompatibleTerminal<Self::Usage>,
    ) -> Self::FinalResponse;

    fn uses_distinct_tool_call_eviction(&self) -> bool {
        false
    }

    fn should_evict(
        &self,
        existing: &OpenToolCallSlot,
        incoming: &CompatibleToolCallChunk,
    ) -> bool {
        self.uses_distinct_tool_call_eviction()
            && should_evict_distinct_named_tool_call(existing, incoming)
    }

    /// Map a provider-specific per-chunk detail onto a decoration for an
    /// in-flight tool call (matched by its established provider id). This is
    /// the adapter-level event rewrite that replaced the old hook mutating
    /// the assembler state directly — assembly lives in the shared
    /// accumulator now.
    fn decorate_tool_call(&self, _detail: &Self::Detail) -> Option<ToolCallDecoration> {
        None
    }

    fn emits_complete_single_chunk_tool_calls(&self) -> bool {
        false
    }

    fn should_emit_completed_tool_call_immediately(
        &self,
        incoming: &CompatibleToolCallChunk,
    ) -> bool {
        self.emits_complete_single_chunk_tool_calls() && incoming.is_complete_single_chunk()
    }
}

/// Wire identity of a tool call whose input is streaming, as tracked by the
/// adapter. Argument assembly, internal-id minting, and finalize policy live
/// in the shared accumulator; the adapter keeps only what the *wire* keys
/// by — the chunk index — mapped to the identity it established.
#[derive(Debug, Clone)]
pub(crate) struct OpenToolCallSlot {
    /// Assembly id: the id under which this call's fragments are emitted.
    /// Fixed at open (the first-seen provider id, possibly empty).
    key: String,
    /// Established provider id: updated when a later chunk carries one.
    pub(crate) id: String,
    /// Established tool name: the last non-empty value seen.
    pub(crate) name: String,
    signature: Option<String>,
    additional_params: Option<serde_json::Value>,
}

impl OpenToolCallSlot {
    fn end_event(&self, on_unparseable: UnparseableToolInput) -> ToolInputEnd {
        let mut end = ToolInputEnd::new(self.key.clone(), on_unparseable);
        end.tool_id = Some(self.id.clone());
        end.signature = self.signature.clone();
        end.additional_params = self.additional_params.clone();
        end
    }
}

pub(crate) fn should_evict_distinct_named_tool_call(
    existing: &OpenToolCallSlot,
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

pub(crate) async fn send_compatible_raw_streaming_request<T, P>(
    http_client: T,
    req: Request<Vec<u8>>,
    profile: P,
) -> Result<streaming::RawStreamingResult<P::FinalResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
    P: CompatibleStreamProfile + 'static,
{
    let span = tracing::Span::current();
    let instrument_span = span.clone();
    let mut event_source = GenericEventSource::new(http_client, req);

    let stream = stream! {
        // Index-to-identity map only: the Chat Completions wire keys tool
        // call fragments by chunk index, so the adapter must correlate.
        // Fragment assembly itself lives in the shared accumulator.
        let mut open_tool_calls: HashMap<usize, OpenToolCallSlot> = HashMap::new();
        let mut final_usage = None;
        let mut final_finish_reason = None;
        let mut response_id = None;
        let mut response_model = None;
        let mut pending_error = None;
        let mut saw_terminal = false;
        let mut saw_any_valid_frame = false;

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }
                Ok(Event::Message(message)) => {
                    if message.data == "[DONE]" {
                        saw_terminal = true;
                        continue;
                    }
                    if message.data.trim().is_empty() {
                        continue;
                    }

                    if let Some(error) = provider_response_from_compatible_sse_data(&message.data) {
                        // Buffered rather than yielded here: fully-delivered
                        // tool calls flush before the terminal error reaches
                        // the consumer.
                        pending_error = Some(error);
                        break;
                    }

                    let chunk = match profile.normalize_chunk(&message.data) {
                        Ok(Some(chunk)) => {
                            saw_any_valid_frame = true;
                            chunk
                        }
                        Ok(None) => continue,
                        Err(error) => {
                            // Surface the malformed frame but keep consuming:
                            // a later genuine terminal signal can still
                            // complete the stream, and without one the
                            // missing `saw_terminal` suppresses the terminal
                            // record below.
                            yield Err(error);
                            continue;
                        }
                    };

                    record_response_metadata(
                        &span,
                        chunk.response_id.as_deref(),
                        chunk.response_model.as_deref(),
                    );

                    if let Some(id) = chunk.response_id.as_ref() {
                        response_id = Some(id.clone());
                    }

                    if let Some(model) = chunk.response_model.as_ref() {
                        response_model = Some(model.clone());
                    }

                    if let Some(usage) = chunk.usage {
                        final_usage = Some(usage);
                    }

                    let Some(choice) = chunk.choice else {
                        continue;
                    };

                    if let Some(reason) = choice.finish_reason.reported() {
                        final_finish_reason = Some(reason);
                        saw_terminal = true;
                    }

                    for incoming in choice.tool_calls {
                        if let Some(existing) = open_tool_calls.get(&incoming.index)
                            && profile.should_evict(existing, &incoming)
                            && let Some(evicted) = open_tool_calls.remove(&incoming.index)
                        {
                            // The wire reused this call's slot: the evicted
                            // call is delivered even when its arguments never
                            // parse (empty-object fallback).
                            yield Ok(RawStreamingChoice::ToolInputEnd(
                                evicted.end_event(UnparseableToolInput::EmptyObject),
                            ));
                        }

                        let slot = open_tool_calls
                            .entry(incoming.index)
                            .or_insert_with(|| OpenToolCallSlot {
                                key: incoming.id.clone().unwrap_or_default(),
                                id: String::new(),
                                name: String::new(),
                                signature: None,
                                additional_params: None,
                            });

                        if let Some(id) = incoming.id.as_ref()
                            && !id.is_empty()
                        {
                            slot.id = id.clone();
                        }

                        if let Some(name) = incoming.name.as_ref()
                            && !name.is_empty()
                        {
                            slot.name = name.clone();
                            yield Ok(RawStreamingChoice::ToolCallDelta {
                                id: slot.key.clone(),
                                content: ToolCallDeltaContent::Name(name.clone()),
                            });
                        }

                        if let Some(arguments) = incoming.arguments.as_ref()
                            && !arguments.is_empty()
                        {
                            yield Ok(RawStreamingChoice::ToolCallDelta {
                                id: slot.key.clone(),
                                content: ToolCallDeltaContent::Delta(arguments.clone()),
                            });
                        }

                        if profile.should_emit_completed_tool_call_immediately(&incoming) {
                            // Completion probe: the accumulator finalizes the
                            // call only if its input parses, and keeps it
                            // open otherwise (`Keep`). The slot stays in the
                            // map either way — a later flush of an already
                            // finalized key is a no-op downstream.
                            yield Ok(RawStreamingChoice::ToolInputEnd(
                                slot.end_event(UnparseableToolInput::Keep),
                            ));
                        }
                    }

                    for detail in &choice.details {
                        if let Some(decoration) = profile.decorate_tool_call(detail)
                            && let Some(slot) = open_tool_calls
                                .values_mut()
                                .find(|slot| slot.id == decoration.tool_id)
                        {
                            slot.signature = decoration.signature;
                            slot.additional_params = decoration.additional_params;
                        }
                    }

                    if let Some(reasoning) = choice.reasoning
                        && !reasoning.is_empty()
                    {
                        yield Ok(RawStreamingChoice::ReasoningDelta {
                            // `reasoning_content` deltas carry no wire id and
                            // never interleave; per-stream constant.
                            id: "reasoning-0".to_string(),
                            reasoning,
                        });
                    }

                    if let Some(content) = choice.text
                        && !content.is_empty()
                    {
                        yield Ok(RawStreamingChoice::Message(content));
                    }

                    if choice.finish_reason.is_tool_calls() {
                        for end in drain_open_tool_calls(&mut open_tool_calls) {
                            yield Ok(RawStreamingChoice::ToolInputEnd(end));
                        }
                    }
                }
                Err(crate::http_client::Error::StreamEnded) => {
                    break;
                }
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    pending_error = Some(CompletionError::from_stream_transport(error));
                    break;
                }
            }
        }

        event_source.close();

        // Tool calls the provider fully delivered are content, so a truncated
        // or errored stream still flushes them to the consumer — before any
        // terminal error, so a first-`Err`-stop consumer sees them too.
        // Partial calls (arguments that never parse) drop in the accumulator.
        for end in drain_open_tool_calls(&mut open_tool_calls) {
            yield Ok(RawStreamingChoice::ToolInputEnd(end));
        }

        // A terminal failure ends the stream after the flush, with no
        // terminal record, preserving the truncation signal.
        if let Some(error) = pending_error {
            yield Err(error);
            return;
        }

        // But only `[DONE]` or a chunk carrying a finish reason counts as the
        // provider completing the turn. A stream that reached EOF without
        // either signal (truncation) gets no terminal record — synthesizing
        // one would present the partial turn as a successful, default-usage
        // completion. A bare `[DONE]` with no successfully decoded frame at
        // all is treated the same way: the parse errors were already yielded,
        // and a default-usage terminal would dress the failure up as success.
        if !saw_terminal || !saw_any_valid_frame {
            return;
        }

        let final_usage = final_usage.unwrap_or_default();
        record_usage(&span, &final_usage.clone().into());
        yield Ok(RawStreamingChoice::FinalResponse(
            profile.build_final_response(CompatibleTerminal {
                usage: final_usage,
                finish_reason: final_finish_reason,
                response_id,
                model: response_model,
            }),
        ));
    }
    .instrument(instrument_span);

    Ok(Box::pin(stream))
}

fn record_usage(span: &tracing::Span, usage: &Usage) {
    if span.is_disabled() {
        return;
    }

    if !usage.has_values() {
        // Zero-valued usage is the documented sentinel for missing provider
        // usage metrics; leave the span fields unset.
        return;
    }

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

fn drain_open_tool_calls(
    open_tool_calls: &mut HashMap<usize, OpenToolCallSlot>,
) -> Vec<ToolInputEnd> {
    // The wire keys tool calls by chunk index; flush in index order so a
    // multi-call turn keeps its wire ordering. Nameless or partial calls
    // drop in the accumulator (`UnparseableToolInput::Drop`).
    let mut slots: Vec<(usize, OpenToolCallSlot)> = open_tool_calls.drain().collect();
    slots.sort_by_key(|(index, _)| *index);
    slots
        .into_iter()
        .map(|(_, slot)| slot.end_event(UnparseableToolInput::Drop))
        .collect()
}

#[cfg(test)]
pub(crate) mod test_support {
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

    pub(crate) async fn assert_zero_arg_tool_call_is_emitted(
        mut stream: streaming::StreamingCompletionResponse,
        expected_id: &str,
        expected_name: &str,
        expect_final_response: bool,
    ) {
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
        } else {
            assert!(
                !saw_final,
                "a truncated stream must not synthesize a terminal record"
            );
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
    use super::{CompatibleStreamProfile, send_compatible_raw_streaming_request};
    use crate::completion::CompletionError;
    use crate::http_client;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use crate::test_utils::internal_streaming_profiles::{
        DistinctToolCallEvictionProfile, ErrorAfterPendingToolCallProfile,
        FinishReasonCleanupProfile,
    };
    use futures::StreamExt;

    /// Wrap a profile-driven raw stream into the normalized carrier, so these
    /// tests exercise the same path providers use.
    async fn send_compatible_streaming_request<T, P>(
        http_client: T,
        req: http::Request<Vec<u8>>,
        profile: P,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError>
    where
        T: crate::http_client::HttpClientExt + Clone + 'static,
        P: CompatibleStreamProfile<FinalResponse = crate::streaming::StreamFinal> + 'static,
    {
        let raw = send_compatible_raw_streaming_request(http_client, req, profile).await?;
        Ok(crate::streaming::StreamingCompletionResponse::stream(
            "test-compatible",
            crate::streaming::normalize_stream(raw, Ok),
        ))
    }

    #[test]
    fn sse_error_detector_handles_null_empty_and_object_or_string_errors() {
        use super::provider_response_from_compatible_sse_data as detect;

        // An empty `error` (`null` or `""`) with no choices must NOT terminate the
        // stream — some providers send one with the terminal usage event. Each of
        // these should be treated as "not an error chunk".
        assert!(detect(r#"{"error":null}"#).is_none());
        assert!(detect(r#"{"error":null,"usage":{"total_tokens":3}}"#).is_none());
        assert!(detect(r#"{"error":""}"#).is_none());
        // A normal content chunk (no `error` key) is also not an error.
        assert!(detect(r#"{"choices":[{"delta":{"content":"hi"}}]}"#).is_none());
        // A live content chunk that ALSO carries an `error` field must NOT terminate
        // the stream — the `choices` guard wins regardless of the error value.
        assert!(detect(r#"{"error":"metadata","choices":[{"delta":{"content":"hi"}}]}"#).is_none());
        assert!(
            detect(r#"{"error":{"message":"x"},"choices":[{"delta":{"content":"hi"}}]}"#).is_none()
        );

        // A non-empty string `error` IS detected, preserving the raw body.
        let string_body = r#"{"error":"oops"}"#;
        let string_error = detect(string_body).expect("string error should be detected");
        assert_eq!(string_error.provider_response_body(), Some(string_body));
        assert_eq!(string_error.provider_response_status(), None);

        // A real provider error envelope IS detected, preserving the raw body.
        let body = r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#;
        let error = detect(body).expect("object error envelope should be detected");
        assert_eq!(error.provider_response_body(), Some(body));
        // It arrives mid-stream with no HTTP status attached.
        assert_eq!(error.provider_response_status(), None);
    }

    #[tokio::test]
    async fn evicted_tool_call_emits_object_input_end_to_end() {
        // Regression guard for #1958, end-to-end through the streaming aggregator.
        //
        // The first tool call is evicted (a distinct second call starts at the
        // same index) **while its arguments are still a partial, non-object
        // string** (`first_args_partial` streams `{"query":` — a fragment the
        // accumulator holds as a bare `Value::String`). Before the fix,
        // `finalize_completed_streaming_tool_call` forwarded that string verbatim,
        // so the evicted call emerged with a string `function.arguments`; a
        // downstream object-typed serializer (e.g. Anthropic's `tool_use.input`)
        // then sent a bare string and strict providers rejected it.
        //
        // This sequence is what makes the test load-bearing: with the fix
        // reverted the evicted call's arguments are `String("{\"query\":")` and
        // the `is_object()` assertion below fails; the sibling
        // `distinct_same_name_tool_calls_evict_by_id_when_a_new_call_starts` test
        // (which lets the first call's args *complete* before eviction) does not
        // exercise this path.
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "first_start",
                "first_args_partial",
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
        for tc in &collected_tool_calls {
            assert!(
                tc.function.arguments.is_object(),
                "tool_use input must be an object, got {:?} for {}",
                tc.function.arguments,
                tc.function.name
            );
        }
        // Pin the evicted call specifically: its unparseable partial string is
        // normalized to `{}` (not forwarded as a string, not dropped).
        let evicted = &collected_tool_calls[0];
        assert_eq!(evicted.id, "call_aaa");
        assert_eq!(evicted.function.arguments, serde_json::json!({}));
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

        // The malformed frame does not abort the stream; consumption continues
        // to EOF. The fully-delivered zero-arg tool call still flushes as
        // content, but with no `[DONE]` or finish reason the truncated stream
        // must not synthesize a terminal record.
        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            match item.expect("post-error items should be ok") {
                StreamedAssistantContent::Final(_) => saw_final = true,
                StreamedAssistantContent::ToolCall { .. } => {}
                other => panic!("unexpected post-error stream item: {other:?}"),
            }
        }
        assert!(
            !saw_final,
            "a truncated stream must not synthesize a terminal record"
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
    async fn streaming_http_non_success_preserves_status_and_body() {
        use crate::test_utils::HttpErrorStreamingClient;

        let body = r#"{"error":{"type":"rate_limit","message":"slow down"}}"#;
        let client = HttpErrorStreamingClient::new(http::StatusCode::TOO_MANY_REQUESTS, body);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_compatible_streaming_request(client, req, FinishReasonCleanupProfile)
            .await
            .expect("stream should start");

        let err = stream
            .next()
            .await
            .expect("stream should yield transport error")
            .expect_err("HTTP non-success should surface as a stream error");
        assert_eq!(
            err.to_string(),
            format!(
                "HttpError: Invalid status code {} with message: {}",
                http::StatusCode::TOO_MANY_REQUESTS,
                body
            )
        );
        assert_eq!(
            err.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(err.provider_response_body(), Some(body));
        assert_eq!(
            err.provider_response_json().expect("valid JSON body"),
            Some(serde_json::json!({
                "error": {
                    "type": "rate_limit",
                    "message": "slow down"
                }
            }))
        );
        assert!(
            stream.next().await.is_none(),
            "stream should terminate after HTTP non-success"
        );
    }

    #[tokio::test]
    async fn streaming_in_band_error_envelope_preserves_full_payload() {
        use crate::providers::openai::send_compatible_streaming_request;
        use crate::test_utils::MockStreamingClient;

        let body = r#"{"error":{"message":"upstream unavailable","type":"server_error"}}"#;
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"content\":\"partial\",\"tool_calls\":[]}}],\"usage\":null}",
                body,
            ]),
        };
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_compatible_streaming_request(client, req, "openai")
            .await
            .expect("stream should start");

        let first = stream
            .next()
            .await
            .expect("stream should yield partial content")
            .expect("partial content should be ok");
        assert!(matches!(
            first,
            StreamedAssistantContent::Text(text) if text.text == "partial"
        ));

        let err = match stream.next().await {
            Some(Err(err)) => err,
            Some(Ok(_)) => panic!("expected in-band provider error after partial content"),
            None => panic!("stream ended before in-band provider error"),
        };
        assert!(matches!(err, CompletionError::ProviderResponse(_)));
        assert_eq!(err.provider_response_status(), None);
        assert_eq!(err.provider_response_body(), Some(body));
        assert!(
            stream.next().await.is_none(),
            "stream should terminate after in-band provider error"
        );
    }

    #[tokio::test]
    async fn streaming_mid_stream_http_non_success_preserves_status_and_body() {
        use crate::providers::openai::send_compatible_streaming_request;
        use crate::test_utils::SequencedStreamingHttpClient;

        let body = r#"{"error":{"message":"upstream unavailable"}}"#;
        let chunks = vec![
            Ok(sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"content\":\"partial\",\"tool_calls\":[]}}],\"usage\":null}",
            ])),
            Err(http_client::Error::InvalidStatusCodeWithMessage(
                http::StatusCode::BAD_GATEWAY,
                body.to_string(),
            )),
        ];
        let client = SequencedStreamingHttpClient::new(chunks);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_compatible_streaming_request(client, req, "openai")
            .await
            .expect("stream should start");

        let first = stream
            .next()
            .await
            .expect("stream should yield partial content")
            .expect("partial content should be ok");
        assert!(matches!(
            first,
            StreamedAssistantContent::Text(text) if text.text == "partial"
        ));

        let err = match stream.next().await {
            Some(Err(err)) => err,
            Some(Ok(_)) => panic!("expected HTTP transport error after partial content"),
            None => panic!("stream ended before HTTP transport error"),
        };
        assert_eq!(
            err.provider_response_status(),
            Some(http::StatusCode::BAD_GATEWAY)
        );
        assert_eq!(err.provider_response_body(), Some(body));
        assert!(
            stream.next().await.is_none(),
            "stream should terminate after mid-stream HTTP non-success"
        );
    }

    #[tokio::test]
    async fn streaming_http_non_success_json_parse_error_is_visible() {
        use crate::test_utils::HttpErrorStreamingClient;

        let client = HttpErrorStreamingClient::new(http::StatusCode::BAD_REQUEST, "not json");
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_compatible_streaming_request(client, req, FinishReasonCleanupProfile)
            .await
            .expect("stream should start");

        let err = match stream.next().await {
            Some(Err(err)) => err,
            _ => panic!("expected HTTP transport error"),
        };
        assert_eq!(err.provider_response_body(), Some("not json"));
        assert!(err.provider_response_json().is_err());
    }

    #[tokio::test]
    async fn streaming_non_http_transport_error_stays_provider_error() {
        use crate::test_utils::SequencedStreamingHttpClient;

        use crate::providers::openai::send_compatible_streaming_request;

        let chunks = vec![Err(http_client::Error::InvalidContentType(
            http::HeaderValue::from_static("application/json"),
        ))];
        let client = SequencedStreamingHttpClient::new(chunks);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_compatible_streaming_request(client, req, "openai")
            .await
            .expect("stream should start");

        let err = match stream.next().await {
            Some(Err(err)) => err,
            Some(Ok(_)) => panic!("expected non-HTTP transport error"),
            None => panic!("stream ended before transport error"),
        };
        assert_eq!(
            err.to_string(),
            "ProviderError: Invalid content type was returned: \"application/json\""
        );
        assert!(matches!(err, CompletionError::ProviderError(_)));
        // Rig-generated transport diagnostics are not provider response bodies.
        assert_eq!(err.provider_response_body(), None);
        assert_eq!(err.provider_response_status(), None);
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

    #[tokio::test]
    async fn transport_error_still_flushes_fully_delivered_tool_calls() {
        use crate::providers::openai::send_compatible_streaming_request;
        use crate::test_utils::SequencedStreamingHttpClient;

        // A fully-delivered tool call followed by a transport error: the tool
        // call is content and must flush BEFORE the error surfaces (so a
        // first-`Err`-stop consumer sees it), and the stream must end without
        // a terminal record.
        let chunks = vec![
            Ok(sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"{\\\"x\\\":1}\"}}]}}],\"usage\":null}",
            ])),
            Err(http_client::Error::InvalidStatusCodeWithMessage(
                http::StatusCode::BAD_GATEWAY,
                r#"{"error":{"message":"upstream unavailable"}}"#.to_string(),
            )),
        ];
        let client = SequencedStreamingHttpClient::new(chunks);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_compatible_streaming_request(client, req, "openai")
            .await
            .expect("stream should start");

        let mut saw_error = false;
        let mut saw_final = false;
        let mut collected_tool_calls = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::ToolCallDelta { .. }) => {}
                Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                    assert!(
                        !saw_error,
                        "the flushed tool call must arrive before the terminal error"
                    );
                    collected_tool_calls.push(tool_call);
                }
                Ok(StreamedAssistantContent::Final(_)) => saw_final = true,
                Ok(other) => panic!("unexpected stream item: {other:?}"),
                Err(_) => saw_error = true,
            }
            if saw_error {
                break;
            }
        }

        assert!(saw_error, "the transport failure must reach the consumer");
        assert_eq!(
            collected_tool_calls.len(),
            1,
            "the fully-delivered tool call must flush despite the transport error"
        );
        assert_eq!(collected_tool_calls[0].id, "call_123");
        assert_eq!(collected_tool_calls[0].function.name, "ping");
        assert_eq!(
            collected_tool_calls[0].function.arguments,
            serde_json::json!({"x": 1})
        );
        assert!(
            stream.next().await.is_none(),
            "nothing may follow the terminal error"
        );
        assert!(
            !saw_final,
            "an errored stream must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn bare_done_after_only_unparseable_frames_emits_no_terminal() {
        // Every frame fails to decode; the trailing `[DONE]` must not dress
        // the failure up as a successful, default-usage completion.
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines(["bad", "bad", "[DONE]"]),
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

        let mut error_count = 0;
        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Final(_)) => saw_final = true,
                Ok(other) => panic!("unexpected stream item: {other:?}"),
                Err(_) => error_count += 1,
            }
        }

        assert_eq!(
            error_count, 2,
            "each unparseable frame must surface as an error item"
        );
        assert!(
            !saw_final,
            "a stream with no successfully decoded frame must not emit a terminal record"
        );
        assert!(stream.response.is_none());
    }
}
