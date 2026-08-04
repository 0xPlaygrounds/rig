//! The streaming module for the OpenAI Responses API.
//! Please see the `openai_streaming` or `openai_streaming_with_tools` example for more practical usage.
use crate::completion::{self, CompletionError};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::message::ReasoningContent;
use crate::providers::openai::responses_api::{ReasoningSummary, ResponsesUsage};
use crate::streaming;
use crate::streaming::RawStreamingChoice;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};
use crate::wasm_compat::WasmCompatSend;
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, enabled};
use tracing_futures::Instrument as _;

use super::{CompletionResponse, GenericResponsesCompletionModel, Output};

type StreamingRawChoice = RawStreamingChoice;

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

/// Wire-level aggregate of the terminal Responses API stream event (usage plus
/// reasoning metadata). The normalized stream terminates with
/// [`crate::streaming::StreamFinal`]; this type remains for parsing by
/// downstream Responses-compatible providers (e.g. Copilot).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingCompletionResponse {
    /// Token usage
    pub usage: ResponsesUsage,
    /// The complete object-shaped reasoning metadata from the terminal response event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// The effective reasoning context from the terminal response event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_context: Option<String>,
    /// Provider-assigned response identifier, when emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Provider-assigned assistant message identifier, when emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Provider-reported model identifier, when emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Provider-native terminal response status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<super::ResponseStatus>,
    /// Provider-native incomplete-response details, when emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<super::IncompleteDetailsReason>,
    /// Whether the stream emitted at least one function call.
    pub has_tool_calls: bool,
}

/// Map a Responses-family raw terminal into Rig's normalized terminal record.
///
/// `response_id` deliberately remains raw-only: a top-level `resp_...` ID is
/// not an assistant `msg_...` item ID and cannot be replayed as one.
pub(crate) fn normalize_terminal_response(
    provider: &str,
    response: StreamingCompletionResponse,
) -> streaming::StreamFinal {
    let mut final_response = streaming::StreamFinal::new(provider, response.usage.into());
    final_response.finish_reason = response.status.as_ref().and_then(|status| {
        let reason = super::finish_reason_from_status(status, response.incomplete_details.as_ref());
        match reason {
            Some(completion::FinishReason::Stop) if response.has_tool_calls => {
                Some(completion::FinishReason::ToolCalls)
            }
            other => other,
        }
    });
    final_response.message_id = response.message_id;
    final_response.model = response.model;
    final_response
}

pub(crate) fn reasoning_choices_from_done_item(
    id: &str,
    summary: &[ReasoningSummary],
    content: &[String],
    encrypted_content: Option<&str>,
) -> Vec<RawStreamingChoice> {
    let mut choices = summary
        .iter()
        .map(|reasoning_summary| match reasoning_summary {
            ReasoningSummary::SummaryText { text } => RawStreamingChoice::Reasoning {
                id: Some(id.to_owned()),
                content: ReasoningContent::Summary(text.to_owned()),
            },
        })
        .collect::<Vec<_>>();

    choices.extend(content.iter().map(|text| RawStreamingChoice::Reasoning {
        id: Some(id.to_owned()),
        content: ReasoningContent::Text {
            text: text.to_owned(),
            signature: None,
        },
    }));

    if let Some(encrypted_content) = encrypted_content.filter(|s| !s.is_empty()) {
        choices.push(RawStreamingChoice::Reasoning {
            id: Some(id.to_owned()),
            content: ReasoningContent::Encrypted(encrypted_content.to_owned()),
        });
    }

    choices
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

fn provider_response_from_responses_error_value(
    value: &serde_json::Value,
    data: &str,
) -> CompletionError {
    if let Some(message) = value
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
    {
        tracing::warn!(message, "provider returned a streaming error event");
    }

    crate::provider_response::completion_error_from_body(data)
}

fn provider_response_from_responses_sse_data(data: &str) -> Option<CompletionError> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    (value.get("type").and_then(serde_json::Value::as_str) == Some("error"))
        .then(|| provider_response_from_responses_error_value(&value, data))
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

    const fn emits_completed_tool_calls_immediately(self) -> bool {
        matches!(self, Self::StrictWithImmediateToolCalls)
    }
}

pub(crate) fn parse_sse_completion_body(
    body: &str,
    provider_name: &str,
) -> Result<CompletionResponse, CompletionError> {
    let mut completed = None;

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
                        return Err(crate::provider_response::completion_error_from_body(data));
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
                return Err(crate::provider_response::completion_error_from_body(data));
            }
            Some("error") => {
                return Err(provider_response_from_responses_error_value(&value, data));
            }
            _ => {}
        }
    }

    completed.ok_or_else(|| {
        CompletionError::ProviderError(format!(
            "{provider_name} stream did not yield response.completed"
        ))
    })
}

struct RawChoiceAccumulator {
    final_usage: ResponsesUsage,
    finish_reason: Option<completion::FinishReason>,
    model: Option<String>,
    response_id: Option<String>,
    message_id: Option<String>,
    status: Option<super::ResponseStatus>,
    incomplete_details: Option<super::IncompleteDetailsReason>,
    reasoning_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    reasoning_context: Option<String>,
    tool_calls: Vec<StreamingRawChoice>,
    tool_call_internal_ids: std::collections::HashMap<String, String>,
}

impl RawChoiceAccumulator {
    fn new(initial_usage: ResponsesUsage) -> Self {
        Self {
            final_usage: initial_usage,
            finish_reason: None,
            model: None,
            response_id: None,
            message_id: None,
            status: None,
            incomplete_details: None,
            reasoning_metadata: None,
            reasoning_context: None,
            tool_calls: Vec::new(),
            tool_call_internal_ids: std::collections::HashMap::new(),
        }
    }

    fn decode_item_chunk(
        &mut self,
        chunk: ItemChunk,
        options: ResponsesStreamOptions,
    ) -> Vec<StreamingRawChoice> {
        let mut immediate = Vec::new();

        let ItemChunk {
            item_id: outer_item_id,
            data: item,
            ..
        } = chunk;

        match item {
            ItemChunkKind::OutputItemAdded(StreamingItemDoneOutput {
                item: Output::FunctionCall(func),
                ..
            }) => {
                let internal_call_id = self
                    .tool_call_internal_ids
                    .entry(func.id.clone())
                    .or_insert_with(crate::id::generate)
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
            ItemChunkKind::ReasoningTextDelta(delta) => {
                immediate.push(streaming::RawStreamingChoice::ReasoningDelta {
                    id: outer_item_id,
                    reasoning: delta.delta,
                });
            }
            ItemChunkKind::RefusalDelta(delta) => {
                immediate.push(streaming::RawStreamingChoice::Message(delta.delta));
            }
            ItemChunkKind::FunctionCallArgsDelta(delta) => {
                if let Some(item_id) = outer_item_id {
                    let internal_call_id = self
                        .tool_call_internal_ids
                        .entry(item_id.clone())
                        .or_insert_with(crate::id::generate)
                        .clone();
                    immediate.push(streaming::RawStreamingChoice::ToolCallDelta {
                        id: item_id,
                        internal_call_id,
                        content: streaming::ToolCallDeltaContent::Delta(delta.delta),
                    });
                }
            }
            _ => {}
        }

        immediate
    }

    fn record_response_chunk(
        &mut self,
        kind: ResponseChunkKind,
        response: CompletionResponse,
        raw_event_data: &str,
    ) -> Result<(), CompletionError> {
        match kind {
            ResponseChunkKind::ResponseCompleted => {
                self.response_id = Some(response.id.clone());
                if let Some(usage) = response.usage {
                    self.final_usage = usage;
                }
                self.model = Some(response.model);
                self.status = Some(response.status.clone());
                self.incomplete_details = response.incomplete_details.clone();
                self.reasoning_metadata = response.reasoning_metadata.clone();
                self.reasoning_context = response.reasoning_context.clone();
                self.finish_reason = super::finish_reason_from_status(
                    &response.status,
                    response.incomplete_details.as_ref(),
                );
                Ok(())
            }
            ResponseChunkKind::ResponseFailed | ResponseChunkKind::ResponseIncomplete => Err(
                crate::provider_response::completion_error_from_body(raw_event_data),
            ),
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
                    .or_insert_with(crate::id::generate)
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
                content,
                encrypted_content,
                ..
            } => {
                immediate.extend(reasoning_choices_from_done_item(
                    &id,
                    &summary,
                    &content,
                    encrypted_content.as_deref(),
                ));
            }
            Output::Message(message) => {
                self.message_id = Some(message.id.clone());
                immediate.push(streaming::RawStreamingChoice::MessageId(message.id));
            }
            // An unmodeled output item (e.g. a hosted-tool result such as
            // `web_search_call`) arriving on `response.output_item.done`. Surface
            // the raw item to stream consumers, mirroring how the non-streaming
            // decode preserves it on `CompletionResponse.output`.
            Output::Unknown(value) => {
                immediate.push(streaming::RawStreamingChoice::Unknown(value));
            }
        }
    }

    fn finish(mut self) -> Vec<StreamingRawChoice> {
        let mut choices = Vec::new();
        // Any tool call seen during the stream (held back or emitted
        // immediately) registers an internal call ID.
        let emitted_tool_calls = !self.tool_call_internal_ids.is_empty();
        choices.append(&mut self.tool_calls);

        let finish_reason = match self.finish_reason {
            Some(completion::FinishReason::Stop) if emitted_tool_calls => {
                Some(completion::FinishReason::ToolCalls)
            }
            other => other,
        };

        let mut final_response =
            crate::streaming::StreamFinal::new("openai", self.final_usage.into());
        final_response.finish_reason = finish_reason;
        final_response.model = self.model;

        choices.push(RawStreamingChoice::FinalResponse(final_response));
        choices
    }

    fn finish_raw(
        mut self,
    ) -> Result<Vec<streaming::RawStreamingChoice<StreamingCompletionResponse>>, CompletionError>
    {
        let emitted_tool_calls = !self.tool_call_internal_ids.is_empty();
        let mut choices = self
            .tool_calls
            .drain(..)
            .map(|choice| {
                choice.try_map_final(|_| {
                    Err(CompletionError::ResponseError(
                        "OpenAI queued an unexpected terminal response".to_owned(),
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        choices.push(RawStreamingChoice::FinalResponse(
            StreamingCompletionResponse {
                usage: self.final_usage,
                reasoning_metadata: self.reasoning_metadata,
                reasoning_context: self.reasoning_context,
                response_id: self.response_id,
                message_id: self.message_id,
                model: self.model,
                status: self.status,
                incomplete_details: self.incomplete_details,
                has_tool_calls: emitted_tool_calls,
            },
        ));
        Ok(choices)
    }
}

pub(crate) fn raw_choices_from_sse_body(
    body: &str,
    initial_usage: ResponsesUsage,
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
                    raw_choices.extend(accumulator.decode_item_chunk(chunk, options));
                }
                StreamingCompletionChunk::Response(chunk) => {
                    let ResponseChunk { kind, response, .. } = *chunk;
                    accumulator.record_response_chunk(kind, response, data)?;
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
            Some("response.reasoning_text.delta") => {
                if let Some(delta) = value.get("delta").and_then(serde_json::Value::as_str) {
                    raw_choices.push(streaming::RawStreamingChoice::ReasoningDelta {
                        id: value
                            .get("item_id")
                            .and_then(serde_json::Value::as_str)
                            .map(ToOwned::to_owned),
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
                        .or_insert_with(crate::id::generate)
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
                        .or_insert_with(crate::id::generate)
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
                    accumulator.record_response_chunk(kind, response, data)?;
                }
            }
            Some("error") => {
                return Err(provider_response_from_responses_error_value(&value, data));
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
) -> Result<completion::CompletionResponse, CompletionError> {
    let raw_choices = raw_choices_from_sse_body(
        body,
        raw_response
            .usage
            .clone()
            .unwrap_or_else(ResponsesUsage::new),
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

    let usage = stream
        .response
        .as_ref()
        .map(|final_response| final_response.usage)
        .unwrap_or_else(|| usage_from_raw_response(&raw_response));
    let message_id = stream
        .message_id
        .clone()
        .or_else(|| message_id_from_response(&raw_response));
    let finish_reason = stream
        .response
        .as_ref()
        .and_then(|final_response| final_response.finish_reason.clone())
        .or_else(|| {
            super::finish_reason_from_status(
                &raw_response.status,
                raw_response.incomplete_details.as_ref(),
            )
        });

    let mut normalized = completion::CompletionResponse::new(stream.choice, usage, "openai")
        .with_model(raw_response.model.clone());
    normalized.message_id = message_id;
    normalized.finish_reason = finish_reason;
    Ok(normalized)
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
        .cloned()
        .map(Into::into)
        .unwrap_or_default()
}

#[cfg(test)]
pub(crate) fn stream_from_event_source<HttpClient, RequestBody>(
    event_source: GenericEventSource<HttpClient, RequestBody>,
    span: tracing::Span,
) -> streaming::StreamingCompletionResponse
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
{
    stream_from_event_source_with_options(event_source, span, ResponsesStreamOptions::strict())
}

#[cfg(test)]
pub(crate) fn stream_from_event_source_with_options<HttpClient, RequestBody>(
    event_source: GenericEventSource<HttpClient, RequestBody>,
    span: tracing::Span,
    options: ResponsesStreamOptions,
) -> streaming::StreamingCompletionResponse
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
{
    let raw = raw_stream_from_event_source_with_options(event_source, span, options);
    let normalized = streaming::normalize_stream(raw, |response| {
        Ok(normalize_terminal_response("openai", response))
    });
    streaming::StreamingCompletionResponse::stream(normalized)
}

pub(crate) fn raw_stream_from_event_source_with_options<HttpClient, RequestBody>(
    mut event_source: GenericEventSource<HttpClient, RequestBody>,
    span: tracing::Span,
    options: ResponsesStreamOptions,
) -> streaming::RawStreamingResult<StreamingCompletionResponse>
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

                    if let Some(error) = provider_response_from_responses_sse_data(&evt.data) {
                        terminated_with_error = true;
                        yield Err(error);
                        break;
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
                            for choice in accumulator.decode_item_chunk(chunk, options) {
                                yield choice.try_map_final(|_| {
                                    Err(CompletionError::ResponseError(
                                        "OpenAI emitted an unexpected terminal delta chunk".to_owned(),
                                    ))
                                });
                            }
                        }
                        StreamingCompletionChunk::Response(chunk) => {
                            let ResponseChunk { kind, response, .. } = *chunk;
                            if matches!(kind, ResponseChunkKind::ResponseCompleted) {
                                span.record("gen_ai.response.id", response.id.as_str());
                                span.record("gen_ai.response.model", response.model.as_str());
                            }
                            if let Err(error) = accumulator.record_response_chunk(
                                kind,
                                response,
                                &evt.data,
                            ) {
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
                    yield Err(CompletionError::from_stream_transport(error));
                    break;
                }
            }
        }

        event_source.close();

        if terminated_with_error {
            return;
        }

        let final_usage = accumulator.final_usage.clone();

        for tool_call in accumulator.finish_raw()? {
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

    Box::pin(stream)
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
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta(DeltaTextChunkWithItemId),
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u64>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u64>,
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
    Ext: crate::client::Provider + super::ResponsesProviderExt + Clone + 'static,
    H: Clone + Default + std::fmt::Debug + WasmCompatSend + 'static,
{
    /// Execute a streaming Responses API completion and preserve the native
    /// terminal response record.
    pub async fn raw_stream(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<streaming::RawStreamingResult<StreamingCompletionResponse>, CompletionError> {
        let system_instructions = completion_request.preamble.clone();
        let record_telemetry_content = completion_request.record_telemetry_content;
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

        let span = CompletionSpanBuilder::new(
            "openai",
            &request.model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();
        let client = self.client.clone();
        let event_source = GenericEventSource::new(client, req);

        Ok(raw_stream_from_event_source_with_options(
            event_source,
            span,
            ResponsesStreamOptions::strict(),
        ))
    }

    pub(crate) async fn stream(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let raw = self.raw_stream(completion_request).await?;
        let normalized = streaming::normalize_stream(raw, |response| {
            Ok(normalize_terminal_response("openai", response))
        });
        Ok(streaming::StreamingCompletionResponse::stream(normalized))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ItemChunkKind, StreamingCompletionChunk, StreamingCompletionResponse,
        normalize_terminal_response, raw_choices_from_sse_body, reasoning_choices_from_done_item,
    };
    use crate::completion::CompletionModel;
    use crate::message::ReasoningContent;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
    use crate::providers::openai::responses_api::{
        AdditionalParameters, CompletionResponse, IncompleteDetailsReason, OutputTokensDetails,
        ReasoningSummary, ResponseError, ResponseObject, ResponseStatus, ResponsesUsage,
    };
    use crate::streaming::{RawStreamingChoice, StreamedAssistantContent};
    use crate::test_utils::MockStreamingClient;
    use crate::{client::CompletionClient, providers::openai};
    use futures::StreamExt;
    use serde_json::{self, json};

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
            provider_reasoning: None,
            reasoning_metadata: None,
            reasoning_context: None,
            usage: None,
            output: Vec::new(),
            tools: Vec::new(),
            additional_parameters: AdditionalParameters::default(),
        }
    }

    fn sample_stream_terminal(
        response_id: Option<&str>,
        message_id: Option<&str>,
        has_tool_calls: bool,
    ) -> StreamingCompletionResponse {
        StreamingCompletionResponse {
            usage: ResponsesUsage::new(),
            reasoning_metadata: None,
            reasoning_context: None,
            response_id: response_id.map(str::to_owned),
            message_id: message_id.map(str::to_owned),
            model: Some("gpt-5.4".to_owned()),
            status: Some(ResponseStatus::Completed),
            incomplete_details: None,
            has_tool_calls,
        }
    }

    #[test]
    fn normalized_terminal_never_uses_response_id_as_message_id() {
        let normalized = normalize_terminal_response(
            "openai",
            sample_stream_terminal(Some("resp_123"), None, true),
        );

        assert_eq!(normalized.message_id, None);
        assert_eq!(
            normalized.finish_reason,
            Some(crate::completion::FinishReason::ToolCalls)
        );
    }

    #[test]
    fn normalized_terminal_preserves_actual_assistant_message_id() {
        let normalized = normalize_terminal_response(
            "openai",
            sample_stream_terminal(Some("resp_123"), Some("msg_123"), false),
        );

        assert_eq!(normalized.message_id.as_deref(), Some("msg_123"));
        assert_eq!(
            normalized.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
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

    async fn final_response_from_event(event: serde_json::Value) -> crate::streaming::StreamFinal {
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
                StreamedAssistantContent::Final(response) => return response,
                _ => continue,
            }
        }

        panic!("stream should yield a final response");
    }

    #[test]
    fn parse_sse_completion_body_preserves_error_payloads() {
        let mut response = sample_response(ResponseStatus::Failed);
        response.error = Some(ResponseError {
            code: "server_error".to_string(),
            message: "response failed".to_string(),
        });
        let events = [
            json!({
                "type": "response.failed",
                "sequence_number": 1,
                "response": response,
            }),
            json!({
                "type": "error",
                "error": {
                    "message": "boom",
                    "code": "server_error",
                    "type": "server_error"
                }
            }),
        ];

        for event in events {
            let payload = serde_json::to_string(&event).expect("event should serialize");
            let body = format!("data: {payload}\n");
            let err = super::parse_sse_completion_body(&body, "ChatGPT")
                .expect_err("error payload should surface as provider response");

            assert!(matches!(
                err,
                crate::completion::CompletionError::ProviderResponse(_)
            ));
            assert_eq!(err.provider_response_status(), None);
            assert_eq!(err.provider_response_body(), Some(payload.as_str()));
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
        let content = vec!["private reasoning".to_string()];
        let choices =
            reasoning_choices_from_done_item("rs_1", &summary, &content, Some("enc_blob"));

        assert_eq!(choices.len(), 4);
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
                content: ReasoningContent::Text { text, signature: None },
            }) if id == "rs_1" && text == "private reasoning"
        ));
        assert!(matches!(
            choices.get(3),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Encrypted(data),
            }) if id == "rs_1" && data == "enc_blob"
        ));
    }

    #[test]
    fn reasoning_output_item_done_emits_reasoning_text_content() {
        let body = format!(
            "data: {}\n",
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 1,
                "item": {
                    "type": "reasoning",
                    "id": "rs_text_1",
                    "summary": [],
                    "content": [{ "type": "reasoning_text", "text": "visible reasoning" }],
                    "status": "completed"
                },
            })
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("sse body should decode");

        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Text { text, signature: None },
            }) if id == "rs_text_1" && text == "visible reasoning"
        ));
    }

    #[test]
    fn reasoning_text_delta_emits_reasoning_delta() {
        let body = format!(
            "data: {}\n",
            json!({
                "type": "response.reasoning_text.delta",
                "item_id": "rs_delta_1",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 1,
                "delta": "thinking",
            })
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("sse body should decode");

        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::ReasoningDelta { id: Some(id), reasoning })
                if id == "rs_delta_1" && reasoning == "thinking"
        ));
    }

    #[test]
    fn unknown_output_item_surfaces_as_raw_unknown_choice() {
        // A hosted-tool item (web_search_call) arriving on
        // `response.output_item.done` must surface to stream consumers as
        // `RawStreamingChoice::Unknown` carrying the verbatim item, mirroring how
        // the non-streaming decode preserves it on `CompletionResponse.output`.
        let item = json!({
            "type": "web_search_call",
            "id": "ws_001",
            "status": "completed",
            "action": { "type": "search", "queries": ["rig framework"] },
        });
        let body = format!(
            "data: {}\n",
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 1,
                "item": item,
            })
        );

        let choices = raw_choices_from_sse_body(&body, ResponsesUsage::new())
            .expect("sse body should decode");

        let unknown = choices.iter().find_map(|choice| match choice {
            RawStreamingChoice::Unknown(value) => Some(value),
            _ => None,
        });
        assert_eq!(
            unknown,
            Some(&item),
            "the raw web_search_call item should reach the consumer verbatim",
        );
    }

    #[test]
    fn reasoning_done_item_without_encrypted_emits_summary_only() {
        let summary = vec![ReasoningSummary::SummaryText {
            text: "only summary".to_string(),
        }];
        let choices = reasoning_choices_from_done_item("rs_2", &summary, &[], None);

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
    fn empty_encrypted_reasoning_is_not_emitted() {
        let content = vec!["visible reasoning".to_string()];

        let choices = reasoning_choices_from_done_item("rs_1", &[], &content, Some(""));

        assert_eq!(choices.len(), 1);
        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::Reasoning {
                content: ReasoningContent::Text { text, .. },
                ..
            }) if text == "visible reasoning"
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

        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        assert!(err.provider_response_body().is_some_and(|body| {
            body.contains("response.failed") && body.contains("maximum context length exceeded")
        }));
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

        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        assert!(err.provider_response_body().is_some_and(|body| {
            body.contains("response.failed")
                && body.contains("context_length_exceeded")
                && body.contains("maximum context length exceeded")
        }));
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

        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        assert!(err.provider_response_body().is_some_and(|body| {
            body.contains("response.incomplete") && body.contains("max_output_tokens")
        }));
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
        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        assert!(err.provider_response_body().is_some_and(|body| {
            body.contains("response.failed") && body.contains("response stream failed")
        }));
        assert!(
            stream.next().await.is_none(),
            "stream should terminate immediately after the first terminal error"
        );
    }

    #[tokio::test]
    async fn streaming_error_event_preserves_full_payload_in_live_loop() {
        use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
        use crate::test_utils::MockStreamingClient;

        let payload = json!({
            "type": "error",
            "error": {
                "message": "boom",
                "code": "server_error",
                "type": "server_error"
            }
        });

        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_bytes_from_json_events(&[payload]),
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
            .expect_err("stream should surface a provider response error");
        assert_eq!(err.provider_response_status(), None);
        assert!(
            err.provider_response_body().is_some_and(|body| {
                body.contains("\"type\":\"error\"") && body.contains("boom")
            })
        );
        assert!(
            stream.next().await.is_none(),
            "stream should terminate after error event"
        );
    }

    #[tokio::test]
    async fn streaming_http_non_success_preserves_status_and_body() {
        use crate::http_client::sse::GenericEventSource;
        use crate::test_utils::HttpErrorStreamingClient;

        let body = r#"{"error":{"message":"quota exceeded"}}"#;
        let client = HttpErrorStreamingClient::new(http::StatusCode::TOO_MANY_REQUESTS, body);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/responses")
            .body(Vec::new())
            .expect("request should build");
        let event_source = GenericEventSource::new(client, req);
        let span = tracing::Span::none();
        let mut stream = super::stream_from_event_source(event_source, span);

        let err = stream
            .next()
            .await
            .expect("stream should yield transport error")
            .expect_err("HTTP non-success should surface as a stream error");
        assert_eq!(
            err.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(err.provider_response_body(), Some(body));
        assert_eq!(
            err.provider_response_json().expect("valid JSON body"),
            Some(serde_json::json!({"error": {"message": "quota exceeded"}}))
        );
        assert!(
            stream.next().await.is_none(),
            "stream should terminate after HTTP non-success"
        );
    }

    #[test]
    fn streaming_error_event_preserves_full_payload() {
        let payload = r#"{"type":"error","error":{"message":"boom","code":"server_error","type":"server_error"}}"#;
        let body = format!("data: {payload}\n");

        let err = super::raw_choices_from_sse_body(&body, super::ResponsesUsage::new())
            .expect_err("error event should surface as a provider response error");

        assert_eq!(err.provider_response_status(), None);
        assert_eq!(err.provider_response_body(), Some(payload));
        let json = err
            .provider_response_json()
            .expect("raw body should be valid JSON")
            .expect("parsed JSON should be present");
        assert_eq!(json["error"]["code"], "server_error");
    }

    #[tokio::test]
    async fn streaming_non_http_transport_error_stays_provider_error() {
        use crate::http_client::sse::GenericEventSource;
        use crate::test_utils::SequencedStreamingHttpClient;

        let chunks = vec![Err(crate::http_client::Error::InvalidContentType(
            http::HeaderValue::from_static("application/json"),
        ))];
        let client = SequencedStreamingHttpClient::new(chunks);
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/responses")
            .body(Vec::new())
            .expect("request should build");
        let event_source = GenericEventSource::new(client, req);
        let span = tracing::Span::none();
        let mut stream = super::stream_from_event_source(event_source, span);

        let err = stream
            .next()
            .await
            .expect("stream should yield transport error")
            .expect_err("non-HTTP transport failure should surface as provider error");
        assert_eq!(
            err.to_string(),
            "ProviderError: Invalid content type was returned: \"application/json\""
        );
        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderError(_)
        ));
        // Rig-generated transport diagnostics are not provider response bodies.
        assert_eq!(err.provider_response_body(), None);
        assert_eq!(err.provider_response_status(), None);
    }

    #[tokio::test]
    async fn response_completed_chunk_populates_final_usage() {
        let mut response = sample_response(ResponseStatus::Completed);
        response.usage = Some(ResponsesUsage {
            input_tokens: 10,
            input_tokens_details: None,
            output_tokens: 5,
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 0,
            }),
            total_tokens: 15,
        });

        let event = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": response,
        });

        let usage = final_response_from_event(event).await.usage;
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn response_completed_chunk_populates_reasoning_metadata_and_context() {
        let response = sample_response(ResponseStatus::Completed);
        let mut event = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": response,
        });
        let metadata = json!({
            "context": "all_turns",
            "effort": "ultra",
            "summary": null,
            "future_control": true
        });
        event["response"]["reasoning"] = metadata.clone();

        // Reasoning metadata is preserved at the wire-parsing layer (still
        // consumed by Responses-compatible providers such as Copilot); the
        // normalized final record carries the usage and provider identity.
        let chunk: StreamingCompletionChunk =
            serde_json::from_value(event.clone()).expect("terminal event should parse");
        let StreamingCompletionChunk::Response(chunk) = chunk else {
            panic!("terminal event should parse as a response chunk");
        };
        assert_eq!(
            chunk.response.reasoning_context.as_deref(),
            Some("all_turns")
        );
        assert_eq!(
            chunk.response.reasoning_metadata.as_ref(),
            metadata.as_object()
        );

        let final_response = final_response_from_event(event).await;
        assert_eq!(final_response.provider, "openai");
        assert_eq!(final_response.model.as_deref(), Some("gpt-5.4"));
        assert_eq!(
            final_response.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
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
            output_tokens_details: Some(OutputTokensDetails {
                reasoning_tokens: 0,
            }),
            total_tokens: 6,
        });

        // Scoped-subscriber tests must not run concurrently; see
        // `test_utils::scoped_tracing_subscriber_guard`.
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
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
}
