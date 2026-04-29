//! The streaming module for the OpenAI Responses API.
//! Please see the `openai_streaming` or `openai_streaming_with_tools` example for more practical usage.
use crate::completion::{self, CompletionError, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::message::ReasoningContent;
use crate::providers::openai::responses_api::{ReasoningSummary, ResponsesUsage};
use crate::streaming;
use crate::streaming::RawStreamingChoice;
use crate::wasm_compat::WasmCompatSend;
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, enabled, info_span};
use tracing_futures::Instrument as _;

use super::{CompletionResponse, GenericResponsesCompletionModel, Output};

type StreamingRawChoice = RawStreamingChoice<StreamingCompletionResponse>;

// ================================================================
// OpenAI Responses Streaming API
// ================================================================

/// A streaming completion chunk.
/// Streaming chunks can come in one of two forms:
/// - A response chunk (where the completed response will have the total token usage)
/// - An item chunk commonly referred to as a delta. In the completions API this would be referred to as the message delta.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum StreamingCompletionChunk {
    Response(Box<ResponseChunk>),
    Delta(ItemChunk),
}

/// The final streaming response from the OpenAI Responses API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingCompletionResponse {
    /// Token usage
    pub usage: ResponsesUsage,
}

pub(crate) fn reasoning_choices_from_done_item(
    id: &str,
    summary: &[ReasoningSummary],
    encrypted_content: Option<&str>,
) -> Vec<RawStreamingChoice<StreamingCompletionResponse>> {
    let mut choices = summary
        .iter()
        .map(|reasoning_summary| match reasoning_summary {
            ReasoningSummary::SummaryText { text } => RawStreamingChoice::Reasoning {
                id: Some(id.to_owned()),
                content: ReasoningContent::Summary(text.to_owned()),
            },
        })
        .collect::<Vec<_>>();

    if let Some(encrypted_content) = encrypted_content {
        choices.push(RawStreamingChoice::Reasoning {
            id: Some(id.to_owned()),
            content: ReasoningContent::Encrypted(encrypted_content.to_owned()),
        });
    }

    choices
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        self.usage.token_usage()
    }
}

/// A response chunk from OpenAI's response API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseChunk {
    /// The response chunk type
    #[serde(rename = "type")]
    pub kind: ResponseChunkKind,
    /// The response itself
    pub response: CompletionResponse,
    /// The item sequence
    pub sequence_number: u64,
}

/// Response chunk type.
/// Renames are used to ensure that this type gets (de)serialized properly.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ResponseChunkKind {
    #[serde(rename = "response.created")]
    ResponseCreated,
    #[serde(rename = "response.in_progress")]
    ResponseInProgress,
    #[serde(rename = "response.completed")]
    ResponseCompleted,
    #[serde(rename = "response.failed")]
    ResponseFailed,
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete,
}

fn response_chunk_error_message(
    kind: &ResponseChunkKind,
    response: &CompletionResponse,
    provider_name: &str,
) -> Option<String> {
    match kind {
        ResponseChunkKind::ResponseFailed => Some(response_error_message(
            response.error.as_ref(),
            &format!("{provider_name} response stream returned a failed response"),
        )),
        ResponseChunkKind::ResponseIncomplete => {
            let reason = response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str())
                .unwrap_or("unknown reason");

            Some(format!(
                "{provider_name} response stream was incomplete: {reason}"
            ))
        }
        _ => None,
    }
}

fn response_error_message(error: Option<&super::ResponseError>, fallback: &str) -> String {
    if let Some(error) = error {
        if error.code.is_empty() {
            error.message.clone()
        } else {
            format!("{}: {}", error.code, error.message)
        }
    } else {
        fallback.to_string()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ResponsesStreamOptions {
    Strict,
    StrictWithImmediateToolCalls,
}

impl ResponsesStreamOptions {
    pub(crate) const fn strict() -> Self {
        Self::Strict
    }

    pub(crate) const fn strict_with_immediate_tool_calls() -> Self {
        Self::StrictWithImmediateToolCalls
    }

    const fn errors_on_terminal_response(self) -> bool {
        true
    }

    const fn emits_completed_tool_calls_immediately(self) -> bool {
        matches!(self, Self::StrictWithImmediateToolCalls)
    }
}

pub(crate) fn parse_sse_completion_body(
    body: &str,
    provider_name: &str,
) -> Result<CompletionResponse, CompletionError> {
    let mut completed = None;
    let mut provider_error = None;

    for line in body.lines() {
        let data = line
            .strip_prefix("data:")
            .map(str::trim)
            .unwrap_or_default();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        if let Ok(chunk) = serde_json::from_str::<StreamingCompletionChunk>(data) {
            if let StreamingCompletionChunk::Response(chunk) = chunk {
                let ResponseChunk { kind, response, .. } = *chunk;
                match kind {
                    ResponseChunkKind::ResponseCompleted => {
                        completed = Some(response);
                        break;
                    }
                    ResponseChunkKind::ResponseFailed | ResponseChunkKind::ResponseIncomplete => {
                        provider_error =
                            response_chunk_error_message(&kind, &response, provider_name);
                    }
                    _ => {}
                }
            }
            continue;
        }

        let value = match serde_json::from_str::<serde_json::Value>(data) {
            Ok(value) => value,
            Err(_) => continue,
        };

        match value.get("type").and_then(serde_json::Value::as_str) {
            Some("response.completed") => {
                if let Some(response) = value.get("response") {
                    completed = Some(serde_json::from_value(response.clone())?);
                    break;
                }
            }
            Some("response.failed") | Some("response.incomplete") => {
                provider_error = value
                    .get("response")
                    .cloned()
                    .and_then(|response| {
                        serde_json::from_value::<CompletionResponse>(response).ok()
                    })
                    .and_then(|response| {
                        let kind = if value.get("type").and_then(serde_json::Value::as_str)
                            == Some("response.failed")
                        {
                            ResponseChunkKind::ResponseFailed
                        } else {
                            ResponseChunkKind::ResponseIncomplete
                        };
                        response_chunk_error_message(&kind, &response, provider_name)
                    })
                    .or_else(|| {
                        value
                            .get("error")
                            .and_then(|error| error.get("message"))
                            .and_then(serde_json::Value::as_str)
                            .map(ToOwned::to_owned)
                    })
                    .or_else(|| Some(data.to_string()));
            }
            Some("error") => {
                provider_error = value
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(serde_json::Value::as_str)
                    .map(ToOwned::to_owned)
                    .or_else(|| Some(data.to_string()));
            }
            _ => {}
        }
    }

    completed.ok_or_else(|| {
        CompletionError::ProviderError(
            provider_error.unwrap_or_else(|| {
                format!("{provider_name} stream did not yield response.completed")
            }),
        )
    })
}

struct RawChoiceAccumulator {
    final_usage: ResponsesUsage,
    tool_calls: Vec<StreamingRawChoice>,
    tool_call_internal_ids: std::collections::HashMap<String, String>,
}

impl RawChoiceAccumulator {
    fn new(initial_usage: ResponsesUsage) -> Self {
        Self {
            final_usage: initial_usage,
            tool_calls: Vec::new(),
            tool_call_internal_ids: std::collections::HashMap::new(),
        }
    }

    fn decode_item_chunk(
        &mut self,
        item: ItemChunkKind,
        options: ResponsesStreamOptions,
    ) -> Vec<StreamingRawChoice> {
        let mut immediate = Vec::new();

        match item {
            ItemChunkKind::OutputItemAdded(StreamingItemDoneOutput {
                item: Output::FunctionCall(func),
                ..
            }) => {
                let internal_call_id = self
                    .tool_call_internal_ids
                    .entry(func.id.clone())
                    .or_insert_with(|| nanoid::nanoid!())
                    .clone();
                immediate.push(streaming::RawStreamingChoice::ToolCallDelta {
                    id: func.id,
                    internal_call_id,
                    content: streaming::ToolCallDeltaContent::Name(func.name),
                });
            }
            ItemChunkKind::OutputItemDone(message) => {
                self.push_output_item_done(
                    message.item,
                    &mut immediate,
                    options.emits_completed_tool_calls_immediately(),
                );
            }
            ItemChunkKind::OutputTextDelta(delta) => {
                immediate.push(streaming::RawStreamingChoice::Message(delta.delta));
            }
            ItemChunkKind::ReasoningSummaryTextDelta(delta) => {
                immediate.push(streaming::RawStreamingChoice::ReasoningDelta {
                    id: None,
                    reasoning: delta.delta,
                });
            }
            ItemChunkKind::RefusalDelta(delta) => {
                immediate.push(streaming::RawStreamingChoice::Message(delta.delta));
            }
            ItemChunkKind::FunctionCallArgsDelta(delta) => {
                let internal_call_id = self
                    .tool_call_internal_ids
                    .entry(delta.item_id.clone())
                    .or_insert_with(|| nanoid::nanoid!())
                    .clone();
                immediate.push(streaming::RawStreamingChoice::ToolCallDelta {
                    id: delta.item_id,
                    internal_call_id,
                    content: streaming::ToolCallDeltaContent::Delta(delta.delta),
                });
            }
            _ => {}
        }

        immediate
    }

    fn record_response_chunk(
        &mut self,
        kind: ResponseChunkKind,
        response: CompletionResponse,
        provider_name: &str,
        options: ResponsesStreamOptions,
    ) -> Result<(), CompletionError> {
        match kind {
            ResponseChunkKind::ResponseCompleted => {
                if let Some(usage) = response.usage {
                    self.final_usage = usage;
                }
                Ok(())
            }
            ResponseChunkKind::ResponseFailed | ResponseChunkKind::ResponseIncomplete
                if options.errors_on_terminal_response() =>
            {
                let error_message = response_chunk_error_message(&kind, &response, provider_name)
                    .unwrap_or_else(|| {
                        format!(
                            "{provider_name} returned terminal response {:?} without an error message",
                            kind
                        )
                    });
                Err(CompletionError::ProviderError(error_message))
            }
            _ => Ok(()),
        }
    }

    fn push_output_item_done(
        &mut self,
        item: Output,
        immediate: &mut Vec<StreamingRawChoice>,
        emit_completed_tool_calls_immediately: bool,
    ) {
        match item {
            Output::FunctionCall(func) => {
                let internal_call_id = self
                    .tool_call_internal_ids
                    .entry(func.id.clone())
                    .or_insert_with(|| nanoid::nanoid!())
                    .clone();
                let tool_call =
                    streaming::RawStreamingToolCall::new(func.id, func.name, func.arguments)
                        .with_internal_call_id(internal_call_id)
                        .with_call_id(func.call_id);

                if emit_completed_tool_calls_immediately {
                    immediate.push(streaming::RawStreamingChoice::ToolCall(tool_call));
                } else {
                    self.tool_calls
                        .push(streaming::RawStreamingChoice::ToolCall(tool_call));
                }
            }
            Output::Reasoning {
                id,
                summary,
                encrypted_content,
                ..
            } => {
                immediate.extend(reasoning_choices_from_done_item(
                    &id,
                    &summary,
                    encrypted_content.as_deref(),
                ));
            }
            Output::Message(message) => {
                immediate.push(streaming::RawStreamingChoice::MessageId(message.id));
            }
            Output::Unknown => {}
        }
    }

    fn finish(mut self) -> Vec<StreamingRawChoice> {
        let mut choices = Vec::new();
        choices.append(&mut self.tool_calls);
        choices.push(RawStreamingChoice::FinalResponse(
            StreamingCompletionResponse {
                usage: self.final_usage,
            },
        ));
        choices
    }
}

pub(crate) fn raw_choices_from_sse_body(
    body: &str,
    initial_usage: ResponsesUsage,
    provider_name: &str,
) -> Result<Vec<StreamingRawChoice>, CompletionError> {
    let mut raw_choices = Vec::new();
    let mut accumulator = RawChoiceAccumulator::new(initial_usage);
    let options = ResponsesStreamOptions::strict();

    for line in body.lines() {
        let data = line
            .strip_prefix("data:")
            .map(str::trim)
            .unwrap_or_default();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        if let Ok(chunk) = serde_json::from_str::<StreamingCompletionChunk>(data) {
            match chunk {
                StreamingCompletionChunk::Delta(chunk) => {
                    raw_choices.extend(accumulator.decode_item_chunk(chunk.data, options));
                }
                StreamingCompletionChunk::Response(chunk) => {
                    let ResponseChunk { kind, response, .. } = *chunk;
                    accumulator.record_response_chunk(kind, response, provider_name, options)?;
                }
            }
            continue;
        }

        let value = match serde_json::from_str::<serde_json::Value>(data) {
            Ok(value) => value,
            Err(_) => continue,
        };

        match value.get("type").and_then(serde_json::Value::as_str) {
            Some("response.output_text.delta") | Some("response.refusal.delta") => {
                if let Some(delta) = value.get("delta").and_then(serde_json::Value::as_str) {
                    raw_choices.push(streaming::RawStreamingChoice::Message(delta.to_owned()));
                }
            }
            Some("response.reasoning_summary_text.delta") => {
                if let Some(delta) = value.get("delta").and_then(serde_json::Value::as_str) {
                    raw_choices.push(streaming::RawStreamingChoice::ReasoningDelta {
                        id: None,
                        reasoning: delta.to_owned(),
                    });
                }
            }
            Some("response.output_item.added") => {
                if let Some(item) = value
                    .get("item")
                    .cloned()
                    .and_then(|item| serde_json::from_value::<Output>(item).ok())
                    && let Output::FunctionCall(func) = item
                {
                    let internal_call_id = accumulator
                        .tool_call_internal_ids
                        .entry(func.id.clone())
                        .or_insert_with(|| nanoid::nanoid!())
                        .clone();
                    raw_choices.push(streaming::RawStreamingChoice::ToolCallDelta {
                        id: func.id,
                        internal_call_id,
                        content: streaming::ToolCallDeltaContent::Name(func.name),
                    });
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = value
                    .get("item")
                    .cloned()
                    .and_then(|item| serde_json::from_value::<Output>(item).ok())
                {
                    accumulator.push_output_item_done(item, &mut raw_choices, false);
                }
            }
            Some("response.function_call_arguments.delta") => {
                if let (Some(item_id), Some(delta)) = (
                    value.get("item_id").and_then(serde_json::Value::as_str),
                    value.get("delta").and_then(serde_json::Value::as_str),
                ) {
                    let internal_call_id = accumulator
                        .tool_call_internal_ids
                        .entry(item_id.to_owned())
                        .or_insert_with(|| nanoid::nanoid!())
                        .clone();
                    raw_choices.push(streaming::RawStreamingChoice::ToolCallDelta {
                        id: item_id.to_owned(),
                        internal_call_id,
                        content: streaming::ToolCallDeltaContent::Delta(delta.to_owned()),
                    });
                }
            }
            Some("response.completed") | Some("response.failed") | Some("response.incomplete") => {
                if let Some(response) = value.get("response").cloned() {
                    let response = serde_json::from_value::<CompletionResponse>(response)?;
                    let Some(kind) = (match value.get("type").and_then(serde_json::Value::as_str) {
                        Some("response.completed") => Some(ResponseChunkKind::ResponseCompleted),
                        Some("response.failed") => Some(ResponseChunkKind::ResponseFailed),
                        Some("response.incomplete") => Some(ResponseChunkKind::ResponseIncomplete),
                        _ => None,
                    }) else {
                        continue;
                    };
                    accumulator.record_response_chunk(kind, response, provider_name, options)?;
                }
            }
            Some("error") => {
                let message = value
                    .get("error")
                    .and_then(|error| error.get("message"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(data);
                return Err(CompletionError::ProviderError(message.to_owned()));
            }
            _ => {}
        }
    }

    raw_choices.extend(accumulator.finish());
    Ok(raw_choices)
}

pub(crate) async fn completion_response_from_sse_body(
    body: &str,
    raw_response: CompletionResponse,
    provider_name: &str,
) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
    let raw_choices = raw_choices_from_sse_body(
        body,
        raw_response
            .usage
            .clone()
            .unwrap_or_else(ResponsesUsage::new),
        provider_name,
    )?;
    let stream = futures::stream::iter(
        raw_choices
            .into_iter()
            .map(Ok::<_, CompletionError>)
            .collect::<Vec<_>>(),
    );
    let mut stream = crate::streaming::StreamingCompletionResponse::stream(Box::pin(stream));

    while let Some(item) = stream.next().await {
        item?;
    }

    if choice_is_empty(&stream.choice) {
        return Err(CompletionError::ResponseError(
            "Response contained no parts".to_owned(),
        ));
    }

    Ok(completion::CompletionResponse {
        usage: stream
            .response
            .as_ref()
            .and_then(GetTokenUsage::token_usage)
            .unwrap_or_else(|| usage_from_raw_response(&raw_response)),
        message_id: stream
            .message_id
            .clone()
            .or_else(|| message_id_from_response(&raw_response)),
        choice: stream.choice,
        raw_response,
    })
}

fn choice_is_empty(choice: &crate::OneOrMany<completion::AssistantContent>) -> bool {
    choice.iter().all(|content| match content {
        completion::AssistantContent::Text(text) => text.text.trim().is_empty(),
        completion::AssistantContent::Reasoning(reasoning) => reasoning.content.is_empty(),
        completion::AssistantContent::Image(_) => false,
        completion::AssistantContent::ToolCall(_) => false,
    })
}

fn message_id_from_response(response: &CompletionResponse) -> Option<String> {
    response.output.iter().find_map(|item| match item {
        Output::Message(message) => Some(message.id.clone()),
        _ => None,
    })
}

fn usage_from_raw_response(response: &CompletionResponse) -> completion::Usage {
    response
        .usage
        .as_ref()
        .and_then(GetTokenUsage::token_usage)
        .unwrap_or_default()
}

pub(crate) fn stream_from_event_source<HttpClient, RequestBody>(
    event_source: GenericEventSource<HttpClient, RequestBody>,
    span: tracing::Span,
    provider_name: &'static str,
) -> streaming::StreamingCompletionResponse<StreamingCompletionResponse>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
{
    stream_from_event_source_with_options(
        event_source,
        span,
        provider_name,
        ResponsesStreamOptions::strict(),
    )
}

pub(crate) fn stream_from_event_source_with_options<HttpClient, RequestBody>(
    mut event_source: GenericEventSource<HttpClient, RequestBody>,
    span: tracing::Span,
    provider_name: &'static str,
    options: ResponsesStreamOptions,
) -> streaming::StreamingCompletionResponse<StreamingCompletionResponse>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
{
    let stream = stream! {
        let mut accumulator = RawChoiceAccumulator::new(ResponsesUsage::new());
        let span = tracing::Span::current();

        let mut terminated_with_error = false;

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }
                Ok(Event::Message(evt)) => {
                    if evt.data.trim().is_empty() || evt.data == "[DONE]" {
                        continue;
                    }

                    let data = serde_json::from_str::<StreamingCompletionChunk>(&evt.data);

                    let Ok(data) = data else {
                        let Err(err) = data else {
                            continue;
                        };
                        debug!(
                            "Couldn't deserialize SSE data as StreamingCompletionChunk: {:?}",
                            err
                        );
                        continue;
                    };

                    match data {
                        StreamingCompletionChunk::Delta(chunk) => {
                            for choice in accumulator.decode_item_chunk(chunk.data, options) {
                                yield Ok(choice);
                            }
                        }
                        StreamingCompletionChunk::Response(chunk) => {
                            let ResponseChunk { kind, response, .. } = *chunk;
                            if matches!(kind, ResponseChunkKind::ResponseCompleted) {
                                span.record("gen_ai.response.id", response.id.as_str());
                                span.record("gen_ai.response.model", response.model.as_str());
                            }
                            if let Err(error) =
                                accumulator.record_response_chunk(kind, response, provider_name, options)
                            {
                                terminated_with_error = true;
                                yield Err(error);
                                break;
                            }
                        }
                    }
                }
                Err(crate::http_client::Error::StreamEnded) => {
                    event_source.close();
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

        let final_usage = accumulator.final_usage.clone();

        for tool_call in accumulator.finish() {
            yield Ok(tool_call)
        }

        span.record("gen_ai.usage.input_tokens", final_usage.input_tokens);
        span.record("gen_ai.usage.output_tokens", final_usage.output_tokens);
        let cached_tokens = final_usage
            .input_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .unwrap_or(0);
        span.record("gen_ai.usage.cache_read.input_tokens", cached_tokens);

    }
    .instrument(span);

    streaming::StreamingCompletionResponse::stream(Box::pin(stream))
}

/// An item message chunk from OpenAI's Responses API.
/// See
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ItemChunk {
    /// Item ID. Optional.
    pub item_id: Option<String>,
    /// The output index of the item from a given streamed response.
    pub output_index: u64,
    /// The item type chunk, as well as the inner data.
    #[serde(flatten)]
    pub data: ItemChunkKind,
}

/// The item chunk type from OpenAI's Responses API.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ItemChunkKind {
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded(StreamingItemDoneOutput),
    #[serde(rename = "response.output_item.done")]
    OutputItemDone(StreamingItemDoneOutput),
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded(ContentPartChunk),
    #[serde(rename = "response.content_part.done")]
    ContentPartDone(ContentPartChunk),
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta(DeltaTextChunk),
    #[serde(rename = "response.output_text.done")]
    OutputTextDone(OutputTextChunk),
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta(DeltaTextChunk),
    #[serde(rename = "response.refusal.done")]
    RefusalDone(RefusalTextChunk),
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgsDelta(DeltaTextChunkWithItemId),
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgsDone(ArgsTextChunk),
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta(SummaryTextChunk),
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone(SummaryTextChunk),
    /// Catch-all for unknown item chunk types (e.g., `web_search_call` events).
    /// This prevents unknown streaming events from breaking deserialization.
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingItemDoneOutput {
    pub sequence_number: u64,
    pub item: Output,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContentPartChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub part: ContentPartChunkPart,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPartChunkPart {
    OutputText { text: String },
    SummaryText { text: String },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DeltaTextChunkWithItemId {
    pub item_id: String,
    pub content_index: u64,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RefusalTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub refusal: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArgsTextChunk {
    pub content_index: u64,
    pub sequence_number: u64,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryPartChunk {
    pub summary_index: u64,
    pub sequence_number: u64,
    pub part: SummaryPartChunkPart,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SummaryTextChunk {
    pub summary_index: u64,
    pub sequence_number: u64,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SummaryPartChunkPart {
    SummaryText { text: String },
}

impl<Ext, H> GenericResponsesCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>:
        HttpClientExt + Clone + std::fmt::Debug + WasmCompatSend + 'static,
    Ext: crate::client::Provider + Clone + 'static,
    H: Clone + Default + std::fmt::Debug + WasmCompatSend + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let mut request = self.create_completion_request(completion_request)?;
        request.stream = Some(true);

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenAI Responses streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/responses")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        // let request_builder = self.client.post_reqwest("/responses").json(&request);

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = tracing::field::Empty,
                gen_ai.request.model = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };
        span.record("gen_ai.provider.name", "openai");
        span.record("gen_ai.request.model", &self.model);
        let client = self.client.clone();
        let event_source = GenericEventSource::new(client, req);

        Ok(stream_from_event_source(event_source, span, "OpenAI"))
    }
}

#[cfg(test)]
mod tests {
    use super::{ItemChunkKind, StreamingCompletionChunk, reasoning_choices_from_done_item};
    use crate::completion::CompletionModel;
    use crate::http_client::mock::MockStreamingClient;
    use crate::message::ReasoningContent;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
    use crate::providers::openai::responses_api::{
        AdditionalParameters, CompletionResponse, IncompleteDetailsReason, OutputTokensDetails,
        ReasoningSummary, ResponseError, ResponseObject, ResponseStatus, ResponsesUsage,
    };
    use crate::streaming::{RawStreamingChoice, StreamedAssistantContent};
    use bytes::Bytes;
    use futures::StreamExt;
    use serde_json::{self, json};

    use crate::{
        client::CompletionClient,
        completion::{Message, ToolDefinition},
        providers::openai,
        streaming::StreamingChat,
        tool::{Tool, ToolError},
    };

    struct ExampleTool;

    impl Default for MockStreamingClient {
        fn default() -> Self {
            Self {
                sse_bytes: Bytes::new(),
            }
        }
    }

    impl std::fmt::Debug for MockStreamingClient {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockStreamingClient")
                .finish_non_exhaustive()
        }
    }

    fn sample_response(status: ResponseStatus) -> CompletionResponse {
        CompletionResponse {
            id: "resp_123".to_string(),
            object: ResponseObject::Response,
            created_at: 0,
            status,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: "gpt-5.4".to_string(),
            usage: None,
            output: Vec::new(),
            tools: Vec::new(),
            additional_parameters: AdditionalParameters::default(),
        }
    }

    async fn first_error_from_event(
        event: serde_json::Value,
    ) -> crate::completion::CompletionError {
        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[event]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        stream
            .next()
            .await
            .expect("stream should yield an item")
            .expect_err("stream should surface a provider error")
    }

    async fn final_usage_from_event(event: serde_json::Value) -> ResponsesUsage {
        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[event]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        while let Some(item) = stream.next().await {
            match item.expect("completed stream should not error") {
                StreamedAssistantContent::Final(res) => return res.usage,
                _ => continue,
            }
        }

        panic!("stream should yield a final response");
    }

    impl Tool for ExampleTool {
        type Args = ();
        type Error = ToolError;
        type Output = String;
        const NAME: &'static str = "example_tool";

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: self.name(),
                description: "A tool that returns some example text.".to_string(),
                parameters: serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                }),
            }
        }

        async fn call(&self, _input: Self::Args) -> Result<Self::Output, Self::Error> {
            let result = "Example answer".to_string();
            Ok(result)
        }
    }

    #[test]
    fn reasoning_done_item_emits_summary_then_encrypted() {
        let summary = vec![
            ReasoningSummary::SummaryText {
                text: "step 1".to_string(),
            },
            ReasoningSummary::SummaryText {
                text: "step 2".to_string(),
            },
        ];
        let choices = reasoning_choices_from_done_item("rs_1", &summary, Some("enc_blob"));

        assert_eq!(choices.len(), 3);
        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Summary(text),
            }) if id == "rs_1" && text == "step 1"
        ));
        assert!(matches!(
            choices.get(1),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Summary(text),
            }) if id == "rs_1" && text == "step 2"
        ));
        assert!(matches!(
            choices.get(2),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Encrypted(data),
            }) if id == "rs_1" && data == "enc_blob"
        ));
    }

    #[test]
    fn reasoning_done_item_without_encrypted_emits_summary_only() {
        let summary = vec![ReasoningSummary::SummaryText {
            text: "only summary".to_string(),
        }];
        let choices = reasoning_choices_from_done_item("rs_2", &summary, None);

        assert_eq!(choices.len(), 1);
        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Summary(text),
            }) if id == "rs_2" && text == "only summary"
        ));
    }

    #[test]
    fn content_part_added_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
            "part": {
                "type": "output_text",
                "text": "hello"
            }
        }))
        .expect("content part event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ContentPartAdded(_)
                )
        ));
    }

    #[test]
    fn content_part_done_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 4,
            "part": {
                "type": "summary_text",
                "text": "done"
            }
        }))
        .expect("content part done event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ContentPartDone(_)
                )
        ));
    }

    #[test]
    fn reasoning_summary_part_added_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.reasoning_summary_part.added",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 5,
            "part": {
                "type": "summary_text",
                "text": "step 1"
            }
        }))
        .expect("reasoning summary part event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ReasoningSummaryPartAdded(_)
                )
        ));
    }

    #[test]
    fn reasoning_summary_part_done_deserializes_snake_case_part_type() {
        let chunk: StreamingCompletionChunk = serde_json::from_value(json!({
            "type": "response.reasoning_summary_part.done",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 6,
            "part": {
                "type": "summary_text",
                "text": "step 2"
            }
        }))
        .expect("reasoning summary part done event should deserialize");

        assert!(matches!(
            chunk,
            StreamingCompletionChunk::Delta(chunk)
                if matches!(
                    chunk.data,
                    ItemChunkKind::ReasoningSummaryPartDone(_)
                )
        ));
    }

    #[tokio::test]
    async fn response_failed_chunk_surfaces_provider_error_without_empty_code_prefix() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: String::new(),
            message: "maximum context length exceeded".to_string(),
        });

        let event = json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": response,
        });

        let err = first_error_from_event(event).await;

        assert_eq!(
            err.to_string(),
            "ProviderError: maximum context length exceeded"
        );
    }

    #[tokio::test]
    async fn response_failed_chunk_surfaces_provider_error_with_code_prefix() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: "context_length_exceeded".to_string(),
            message: "maximum context length exceeded".to_string(),
        });

        let event = json!({
            "type": "response.failed",
            "sequence_number": 1,
            "response": response,
        });

        let err = first_error_from_event(event).await;

        assert_eq!(
            err.to_string(),
            "ProviderError: context_length_exceeded: maximum context length exceeded"
        );
    }

    #[tokio::test]
    async fn response_incomplete_chunk_uses_incomplete_details_reason() {
        let mut response = sample_response(ResponseStatus::Incomplete);
        response.incomplete_details = Some(IncompleteDetailsReason {
            reason: "max_output_tokens".to_string(),
        });

        let event = json!({
            "type": "response.incomplete",
            "sequence_number": 1,
            "response": response,
        });

        let err = first_error_from_event(event).await;

        assert_eq!(
            err.to_string(),
            "ProviderError: OpenAI response stream was incomplete: max_output_tokens"
        );
    }

    #[tokio::test]
    async fn response_failed_chunk_terminates_stream_without_followup_items() {
        let tool_call_done = json!({
            "type": "response.output_item.done",
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "arguments": "{}",
                "call_id": "call_123",
                "name": "example_tool",
                "status": "completed"
            }
        });

        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: "server_error".to_string(),
            message: "response stream failed".to_string(),
        });

        let failed = json!({
            "type": "response.failed",
            "sequence_number": 2,
            "response": response,
        });

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[tool_call_done, failed]),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let err = stream
            .next()
            .await
            .expect("stream should yield an item")
            .expect_err("stream should surface a provider error");
        assert_eq!(
            err.to_string(),
            "ProviderError: server_error: response stream failed"
        );
        assert!(
            stream.next().await.is_none(),
            "stream should terminate immediately after the first terminal error"
        );
    }

    #[tokio::test]
    async fn response_completed_chunk_populates_final_usage() {
        let mut response = sample_response(ResponseStatus::Completed);
        response.usage = Some(ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: None,
            output_tokens: 5,
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 0,
            },
            total_tokens: 15,
        });

        let event = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": response,
        });

        let usage = final_usage_from_event(event).await;
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn done_sentinel_is_ignored_without_debug_parse_noise() {
        use std::io::{self, Write};
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct SharedWriter(Arc<Mutex<Vec<u8>>>);

        impl Write for SharedWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.0
                    .lock()
                    .expect("log buffer mutex should not be poisoned")
                    .extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        let mut response = sample_response(ResponseStatus::Completed);
        response.usage = Some(ResponsesUsage {
            input_tokens: 4,
            input_tokens_details: None,
            output_tokens: 2,
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: 0,
            },
            total_tokens: 6,
        });

        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .without_time()
            .with_writer({
                let captured = captured.clone();
                move || SharedWriter(captured.clone())
            })
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: bytes::Bytes::from(format!(
                    "data: {}\n\ndata: [DONE]\n\n",
                    serde_json::to_string(&json!({
                        "type": "response.completed",
                        "sequence_number": 1,
                        "response": response,
                    }))
                    .expect("response event should serialize")
                )),
            })
            .api_key("test-key")
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");
        let request = model.completion_request("hello").build();
        let mut stream = model.stream(request).await.expect("stream should start");

        let mut final_usage = None;
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Final(response) =
                item.expect("stream should complete successfully")
            {
                final_usage = Some(response.usage);
            }
        }

        let usage = final_usage.expect("expected final response");
        assert_eq!(usage.input_tokens, 4);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.total_tokens, 6);

        let logs = String::from_utf8(
            captured
                .lock()
                .expect("log buffer mutex should not be poisoned")
                .clone(),
        )
        .expect("captured logs should be valid UTF-8");
        assert!(
            !logs.contains("Couldn't deserialize SSE data as StreamingCompletionChunk"),
            "expected [DONE] to bypass the parse-failure debug path, logs were: {logs}"
        );
    }

    // requires `derive` rig-core feature due to using tool macro
    #[tokio::test]
    #[ignore = "requires API key"]
    async fn test_openai_streaming_tools_reasoning() {
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var should exist");
        let client = openai::Client::new(&api_key).expect("Failed to build client");
        let agent = client
            .agent("gpt-5.2")
            .max_tokens(8192)
            .tool(ExampleTool)
            .additional_params(serde_json::json!({
                "reasoning": {"effort": "high"}
            }))
            .build();

        let chat_history: Vec<Message> = Vec::new();
        let mut stream = agent
            .stream_chat("Call my example tool", &chat_history)
            .multi_turn(5)
            .await;

        while let Some(item) = stream.next().await {
            println!("Got item: {item:?}");
        }
    }
}
