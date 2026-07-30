//! The sans-IO stream state machine for OpenAI Chat Completions-compatible
//! providers.
//!
//! Several providers expose an SSE stream that looks like OpenAI Chat
//! Completions: text arrives in deltas, tool calls are streamed piecemeal, and
//! a trailing event may carry usage. This module owns the shared state machine.
//! Dialect variation is **plain data**, not a trait: [`ChunkNormalizer`] is a
//! concrete enum whose arms name the per-dialect pure chunk parsers
//! (`&str -> NormalizedCompatibleChunk`), and the boolean knobs come from each
//! provider's [`ProviderDescriptor`](crate::providers::descriptor::ProviderDescriptor).
//! The transport is the type-erased
//! [`BoxedEventSource`](crate::http_client::sse::BoxedEventSource), so nothing
//! here is generic over the HTTP client.

use std::collections::HashMap;

use async_stream::stream;
use futures::StreamExt;
use tracing_futures::Instrument;

use crate::completion::CompletionError;
use crate::http_client::sse::{BoxedEventSource, Event};
use crate::json_utils;
pub(crate) use crate::providers::descriptor::{
    ChatCompletionsDialect, ChatCompletionsUsageDialect,
};
use crate::streaming::{self, RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent};

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

/// One normalized choice from a compatible streaming chunk.
///
/// `details` carries provider-specific side payloads (OpenRouter's
/// `reasoning_details`) that decorate already-accumulated tool calls; dialects
/// without them leave it empty.
#[derive(Debug, Clone)]
pub(crate) struct CompatibleChoice {
    pub(crate) finish_reason: CompatibleFinishReason,
    pub(crate) text: Option<String>,
    pub(crate) reasoning: Option<String>,
    pub(crate) tool_calls: Vec<CompatibleToolCallChunk>,
    pub(crate) details: Vec<serde_json::Value>,
}

/// One normalized compatible streaming chunk.
///
/// `usage` is already normalized to [`crate::completion::Usage`]: each dialect
/// parses its own wire usage type and converts at the parse site, so the state
/// machine carries no provider usage generic.
#[derive(Debug, Clone)]
pub(crate) struct CompatibleChunk {
    pub(crate) response_id: Option<String>,
    pub(crate) response_model: Option<String>,
    pub(crate) choice: Option<CompatibleChoice>,
    pub(crate) usage: Option<crate::completion::Usage>,
}

/// The result of a pure per-dialect chunk parser: `Ok(None)` skips the chunk
/// (unparseable payloads are logged and skipped, as before), `Err` terminates
/// the stream.
pub(crate) type NormalizedCompatibleChunk = Result<Option<CompatibleChunk>, CompletionError>;

/// Assemble a normalized chunk from a dialect's first choice.
pub(crate) fn first_choice_chunk<Choice, F>(
    response_id: Option<String>,
    response_model: Option<String>,
    usage: Option<crate::completion::Usage>,
    choices: &[Choice],
    map_choice: F,
) -> CompatibleChunk
where
    F: FnOnce(&Choice) -> CompatibleChoice,
{
    CompatibleChunk {
        response_id,
        response_model,
        choice: choices.first().map(map_choice),
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

/// The per-dialect chunk parser the shared state machine runs, as a concrete
/// enum rather than a trait object.
///
/// Each arm names a pure `&str -> NormalizedCompatibleChunk` function living
/// with the wire types it parses.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ChunkNormalizer {
    /// The OpenAI Chat Completions wire dialect (17 providers).
    ChatCompletions(ChatCompletionsDialect),
    /// GitHub Copilot's chat completions dialect (a narrower schema: required
    /// `index`, no `reasoning`/`reasoning_details` keys, no deprecated
    /// `function_call` finish reason).
    CopilotChat,
    /// Scripted normalizers that drive the state machine directly in tests.
    #[cfg(test)]
    Test(crate::test_utils::internal_streaming_profiles::TestNormalizer),
}

impl ChunkNormalizer {
    /// Parse one SSE `data` payload into a normalized chunk.
    fn normalize(&self, data: &str) -> NormalizedCompatibleChunk {
        match self {
            Self::ChatCompletions(dialect) => {
                crate::providers::openai::completion::streaming::normalize_chat_completions_chunk(
                    data, *dialect,
                )
            }
            Self::CopilotChat => crate::providers::copilot::normalize_copilot_chat_chunk(data),
            #[cfg(test)]
            Self::Test(normalizer) => normalizer.normalize(data),
        }
    }

    /// The provider name stamped on the terminal record.
    fn provider_name(&self) -> &'static str {
        match self {
            Self::ChatCompletions(dialect) => dialect.provider,
            Self::CopilotChat => "copilot",
            #[cfg(test)]
            Self::Test(normalizer) => normalizer.provider_name(),
        }
    }

    /// The usage carried on the terminal record when the provider never sent a
    /// usage payload — the dialect's wire default, converted.
    fn default_usage(&self) -> crate::completion::Usage {
        match self {
            Self::ChatCompletions(dialect) => default_usage_for(dialect.usage),
            Self::CopilotChat => crate::providers::openai::completion::Usage::default().into(),
            #[cfg(test)]
            Self::Test(_) => crate::completion::Usage::default(),
        }
    }

    /// Whether same-index tool calls with distinct ids start a new call
    /// (gateways that reuse `index: 0` for parallel calls).
    fn uses_distinct_tool_call_eviction(&self) -> bool {
        match self {
            Self::ChatCompletions(_) | Self::CopilotChat => true,
            #[cfg(test)]
            Self::Test(normalizer) => normalizer.uses_distinct_tool_call_eviction(),
        }
    }

    fn should_evict(
        &self,
        existing: &RawStreamingToolCall,
        incoming: &CompatibleToolCallChunk,
    ) -> bool {
        self.uses_distinct_tool_call_eviction()
            && should_evict_distinct_named_tool_call(existing, incoming)
    }

    /// Whether a tool call that arrives complete in one chunk is emitted
    /// immediately instead of being held until the stream ends.
    fn should_emit_completed_tool_call_immediately(
        &self,
        incoming: &CompatibleToolCallChunk,
    ) -> bool {
        let emits = match self {
            Self::ChatCompletions(dialect) => dialect.emits_complete_single_chunk_tool_calls,
            Self::CopilotChat => false,
            #[cfg(test)]
            Self::Test(_) => false,
        };
        emits && incoming.is_complete_single_chunk()
    }

    /// Apply a dialect's side payloads to the accumulated tool calls.
    fn decorate_tool_call(
        &self,
        detail: &serde_json::Value,
        tool_calls: &mut HashMap<usize, RawStreamingToolCall>,
    ) {
        if let Self::ChatCompletions(dialect) = self
            && dialect.decorates_reasoning_details
        {
            crate::providers::openrouter::decorate_streaming_tool_call(detail, tool_calls);
        }
    }
}

/// The normalized value of a dialect's default (all-zero) wire usage.
fn default_usage_for(usage: ChatCompletionsUsageDialect) -> crate::completion::Usage {
    match usage {
        ChatCompletionsUsageDialect::OpenAi => {
            crate::providers::openai::completion::Usage::default().into()
        }
        ChatCompletionsUsageDialect::DeepSeek => {
            crate::providers::deepseek::Usage::default().into()
        }
        ChatCompletionsUsageDialect::Mistral => crate::providers::mistral::Usage::default().into(),
        ChatCompletionsUsageDialect::OpenRouter => {
            crate::providers::openrouter::Usage::default().into()
        }
    }
}

/// Parse a dialect's wire usage payload from an already-decoded JSON value,
/// normalizing it at the parse site.
pub(crate) fn normalize_wire_usage(
    usage: Option<serde_json::Value>,
    dialect: ChatCompletionsUsageDialect,
) -> Result<Option<crate::completion::Usage>, serde_json::Error> {
    let Some(usage) = usage else {
        return Ok(None);
    };
    Ok(Some(match dialect {
        ChatCompletionsUsageDialect::OpenAi => {
            serde_json::from_value::<crate::providers::openai::completion::Usage>(usage)?.into()
        }
        ChatCompletionsUsageDialect::DeepSeek => {
            serde_json::from_value::<crate::providers::deepseek::Usage>(usage)?.into()
        }
        ChatCompletionsUsageDialect::Mistral => {
            serde_json::from_value::<crate::providers::mistral::Usage>(usage)?.into()
        }
        ChatCompletionsUsageDialect::OpenRouter => {
            serde_json::from_value::<crate::providers::openrouter::Usage>(usage)?.into()
        }
    }))
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

/// Drive `event_source` through the shared compatible stream state machine.
///
/// Sans-IO apart from consuming the already-opened transport-edge event stream:
/// every dialect decision is a match on `normalizer`.
pub(crate) fn drive_compatible_stream(
    event_source: BoxedEventSource,
    normalizer: ChunkNormalizer,
) -> streaming::StreamingCompletionResponse {
    let span = tracing::Span::current();
    let instrument_span = span.clone();
    let mut event_source = event_source;
    let profile = normalizer;

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

                    if let Some(error) = provider_response_from_compatible_sse_data(&message.data) {
                        terminated_with_error = true;
                        yield Err(error);
                        break;
                    }

                    let chunk = match profile.normalize(&message.data) {
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

                        let emit_completed_tool_call_immediately =
                            profile.should_emit_completed_tool_call_immediately(&incoming);
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
                    yield Err(CompletionError::from_stream_transport(error));
                    break;
                }
            }
        }

        if terminated_with_error {
            return;
        }

        for tool_call in
            take_finalized_tool_calls(&mut tool_calls, DroppedToolCallContext::EndOfStream)
        {
            yield Ok(RawStreamingChoice::ToolCall(tool_call));
        }

        let final_usage = final_usage.unwrap_or_else(|| profile.default_usage());
        record_usage(&span, &final_usage);
        yield Ok(RawStreamingChoice::FinalResponse(
            crate::streaming::StreamFinal::new(profile.provider_name(), final_usage),
        ));
    }
    .instrument(instrument_span);

    streaming::StreamingCompletionResponse::stream(Box::pin(stream))
}

fn record_usage(span: &tracing::Span, usage: &crate::completion::Usage) {
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
    // The eviction path (distinct-named tool calls within one assistant turn)
    // previously only normalized a `null` arguments value to `{}` and otherwise
    // passed the value through verbatim. Streamed OpenAI-compatible tool-call
    // arguments accumulate as a JSON *string* (`Value::String`), so an evicted
    // tool call leaked a bare string into `ToolCall.function.arguments`. A
    // downstream serializer that expects an object (e.g. the Anthropic protocol's
    // `tool_use.input`) then emitted a string, which strict providers reject with
    // `tool_use.input: Input should be a valid dictionary`. Mirror
    // `finalize_pending_tool_call`: parse a string payload into the underlying
    // JSON value (empty string -> `{}`, unparseable -> `{}` rather than a bare
    // string). Valid scalar and array arguments are canonical JSON too and must
    // not be changed only because this call was evicted before end-of-stream.
    match &tool_call.arguments {
        serde_json::Value::Null => {
            tool_call.arguments = serde_json::Value::Object(serde_json::Map::new());
        }
        serde_json::Value::String(arguments) => {
            tool_call.arguments = json_utils::parse_tool_arguments(arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
        }
        _ => {}
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
    let mut pending_tool_calls = tool_calls.drain().collect::<Vec<_>>();
    pending_tool_calls.sort_by_key(|(index, _)| *index);

    for (_, tool_call) in pending_tool_calls {
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
        ChunkNormalizer, drive_compatible_stream, finalize_completed_streaming_tool_call,
        finalize_pending_tool_call,
    };
    use crate::completion::CompletionError;
    use crate::http_client;
    use crate::http_client::Backend;
    use crate::streaming::RawStreamingToolCall;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use crate::test_utils::internal_streaming_profiles::TestNormalizer;
    use futures::StreamExt;

    /// Drive a scripted normalizer over a test HTTP client.
    fn send_compatible_streaming_request<T>(
        client: T,
        req: http::Request<Vec<u8>>,
        normalizer: TestNormalizer,
    ) -> crate::streaming::StreamingCompletionResponse
    where
        T: Backend + Clone + 'static,
    {
        drive_compatible_stream(
            crate::http_client::sse::boxed_event_source(client, req, false),
            ChunkNormalizer::Test(normalizer),
        )
    }

    /// Drive the OpenAI chat-completions dialect over a test HTTP client.
    fn send_openai_streaming_request<T>(
        client: T,
        req: http::Request<Vec<u8>>,
    ) -> crate::streaming::StreamingCompletionResponse
    where
        T: Backend + Clone + 'static,
    {
        drive_compatible_stream(
            crate::http_client::sse::boxed_event_source(client, req, false),
            ChunkNormalizer::ChatCompletions(crate::providers::openai::functions::STREAM_DIALECT),
        )
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

    // Regression guard: the eviction finalizer must parse a JSON-string
    // arguments payload into the underlying object, exactly like
    // `finalize_pending_tool_call`. Before the fix it only normalized `null` and
    // passed a `Value::String` through verbatim, so an evicted tool call leaked a
    // string into `function.arguments`. A downstream serializer expecting an
    // object (e.g. Anthropic's `tool_use.input`) then emitted a bare string,
    // which strict providers reject with
    // `tool_use.input: Input should be a valid dictionary`.
    #[test]
    fn eviction_finalizer_parses_string_arguments_into_object() {
        let tool_call = RawStreamingToolCall::new(
            "call_evicted".to_owned(),
            "memory_search".to_owned(),
            // The accumulated state when eviction fires before the args were
            // recognized as a complete `{...}` object (e.g. whitespace/fragment).
            serde_json::Value::String("{\"query\":\"one\"}".to_owned()),
        );

        let finalized = finalize_completed_streaming_tool_call(tool_call);

        assert!(
            finalized.arguments.is_object(),
            "evicted tool_use input must be a JSON object, got: {:?}",
            finalized.arguments
        );
        assert_eq!(finalized.arguments, serde_json::json!({"query": "one"}));
    }

    #[test]
    fn eviction_finalizer_normalizes_empty_and_unparseable_strings_to_object() {
        // Empty string -> {}.
        let empty = finalize_completed_streaming_tool_call(RawStreamingToolCall::new(
            "c1".to_owned(),
            "ping".to_owned(),
            serde_json::Value::String(String::new()),
        ));
        assert_eq!(empty.arguments, serde_json::json!({}));

        // Null -> {} (pre-existing behavior, kept).
        let null = finalize_completed_streaming_tool_call(RawStreamingToolCall::new(
            "c2".to_owned(),
            "ping".to_owned(),
            serde_json::Value::Null,
        ));
        assert_eq!(null.arguments, serde_json::json!({}));

        // Unparseable JSON -> {} rather than leaking a partial wire fragment.
        let malformed = finalize_completed_streaming_tool_call(RawStreamingToolCall::new(
            "c3".to_owned(),
            "ping".to_owned(),
            serde_json::Value::String("[1,".to_owned()),
        ));
        assert!(malformed.arguments.is_object());
        assert_eq!(malformed.arguments, serde_json::json!({}));
    }

    #[test]
    fn eviction_and_eof_preserve_the_same_valid_scalar_and_array_json() {
        for (encoded, expected) in [
            ("5", serde_json::json!(5)),
            (r#""value""#, serde_json::json!("value")),
            ("[1,2]", serde_json::json!([1, 2])),
        ] {
            let evicted = finalize_completed_streaming_tool_call(RawStreamingToolCall::new(
                "evicted".to_owned(),
                "tool".to_owned(),
                serde_json::Value::String(encoded.to_owned()),
            ));
            let eof = finalize_pending_tool_call(RawStreamingToolCall::new(
                "eof".to_owned(),
                "tool".to_owned(),
                serde_json::Value::String(encoded.to_owned()),
            ))
            .expect("valid JSON must survive EOF finalization");

            assert_eq!(evicted.arguments, expected, "eviction changed {encoded}");
            assert_eq!(eof.arguments, expected, "EOF changed {encoded}");
            assert_eq!(evicted.arguments, eof.arguments);
        }
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

        let mut stream = send_compatible_streaming_request(
            client,
            req,
            TestNormalizer::DistinctToolCallEviction,
        );

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

        let mut stream = send_compatible_streaming_request(
            client,
            req,
            TestNormalizer::ErrorAfterPendingToolCall,
        );

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

        let mut stream = send_compatible_streaming_request(
            client,
            req,
            TestNormalizer::DistinctToolCallEviction,
        );

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

        let mut stream =
            send_compatible_streaming_request(client, req, TestNormalizer::FinishReasonCleanup);

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

        let mut stream = send_openai_streaming_request(client, req);

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

        let mut stream = send_openai_streaming_request(client, req);

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

        let mut stream =
            send_compatible_streaming_request(client, req, TestNormalizer::FinishReasonCleanup);

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

        let chunks = vec![Err(http_client::Error::InvalidContentType(
            http::HeaderValue::from_static("application/json"),
        ))];
        let client = SequencedStreamingHttpClient::new(chunks);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_openai_streaming_request(client, req);

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

        let mut stream =
            send_compatible_streaming_request(client, req, TestNormalizer::FinishReasonCleanup);

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
