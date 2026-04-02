//! The streaming module for the OpenAI Responses API.
//! Please see the `openai_streaming` or `openai_streaming_with_tools` example for more practical usage.
use crate::completion::{CompletionError, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::message::ReasoningContent;
use crate::providers::openai::responses_api::{
    ReasoningSummary, ResponsesCompletionModel, ResponsesUsage,
};
use crate::streaming;
use crate::streaming::RawStreamingChoice;
use crate::wasm_compat::WasmCompatSend;
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, enabled, info_span};
use tracing_futures::Instrument as _;

use super::{CompletionResponse, Output};

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
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.input_tokens;
        usage.output_tokens = self.usage.output_tokens;
        usage.total_tokens = self.usage.total_tokens;
        usage.cached_input_tokens = self
            .usage
            .input_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .unwrap_or(0);
        Some(usage)
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

impl<T> ResponsesCompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + WasmCompatSend + 'static,
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
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };
        span.record("gen_ai.provider.name", "openai");
        span.record("gen_ai.request.model", &self.model);
        // Build the request with proper headers for SSE
        let client = self.client.clone();

        let mut event_source = GenericEventSource::new(client, req);

        let stream = stream! {
            let mut final_usage = ResponsesUsage::new();

            let mut tool_calls: Vec<RawStreamingChoice<StreamingCompletionResponse>> = Vec::new();
            let mut tool_call_internal_ids: std::collections::HashMap<String, String> = std::collections::HashMap::new();
            let span = tracing::Span::current();

            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::trace!("SSE connection opened");
                        tracing::info!("OpenAI stream started");
                        continue;
                    }
                    Ok(Event::Message(evt)) => {
                        // Skip heartbeat messages or empty data
                        if evt.data.trim().is_empty() {
                            continue;
                        }

                        let data = serde_json::from_str::<StreamingCompletionChunk>(&evt.data);

                        let Ok(data) = data else {
                            let err = data.unwrap_err();
                            debug!("Couldn't serialize data as StreamingCompletionResponse: {:?}", err);
                            continue;
                        };

                        if let StreamingCompletionChunk::Delta(chunk) = &data {
                            match &chunk.data {
                                ItemChunkKind::OutputItemAdded(message) => {
                                    if let StreamingItemDoneOutput { item: Output::FunctionCall(func), .. } = message {
                                        let internal_call_id = tool_call_internal_ids
                                            .entry(func.id.clone())
                                            .or_insert_with(|| nanoid::nanoid!())
                                            .clone();
                                        yield Ok(streaming::RawStreamingChoice::ToolCallDelta {
                                            id: func.id.clone(),
                                            internal_call_id,
                                            content: streaming::ToolCallDeltaContent::Name(func.name.clone()),
                                        });
                                    }
                                }
                                ItemChunkKind::OutputItemDone(message) => {
                                    match message {
                                        StreamingItemDoneOutput {  item: Output::FunctionCall(func), .. } => {
                                            let internal_id = tool_call_internal_ids
                                                .entry(func.id.clone())
                                                .or_insert_with(|| nanoid::nanoid!())
                                                .clone();
                                            let raw_tool_call = streaming::RawStreamingToolCall::new(
                                                func.id.clone(),
                                                func.name.clone(),
                                                func.arguments.clone(),
                                            )
                                                .with_internal_call_id(internal_id)
                                                .with_call_id(func.call_id.clone());
                                            tool_calls.push(streaming::RawStreamingChoice::ToolCall(raw_tool_call));
                                        }

                                        StreamingItemDoneOutput {  item: Output::Reasoning {  summary, id, encrypted_content, .. }, .. } => {
                                            for reasoning_choice in reasoning_choices_from_done_item(
                                                id,
                                                summary,
                                                encrypted_content.as_deref(),
                                            ) {
                                                yield Ok(reasoning_choice);
                                            }
                                        }
                                        StreamingItemDoneOutput { item: Output::Message(msg), .. } => {
                                            yield Ok(streaming::RawStreamingChoice::MessageId(msg.id.clone()));
                                        }
                                    }
                                }
                                ItemChunkKind::OutputTextDelta(delta) => {
                                    yield Ok(streaming::RawStreamingChoice::Message(delta.delta.clone()))
                                }
                                ItemChunkKind::ReasoningSummaryTextDelta(delta) => {
                                    yield Ok(streaming::RawStreamingChoice::ReasoningDelta { id: None, reasoning: delta.delta.clone() })
                                }
                                ItemChunkKind::RefusalDelta(delta) => {
                                    yield Ok(streaming::RawStreamingChoice::Message(delta.delta.clone()))
                                }
                                ItemChunkKind::FunctionCallArgsDelta(delta) => {
                                    let internal_call_id = tool_call_internal_ids
                                        .entry(delta.item_id.clone())
                                        .or_insert_with(|| nanoid::nanoid!())
                                        .clone();
                                    yield Ok(streaming::RawStreamingChoice::ToolCallDelta {
                                        id: delta.item_id.clone(),
                                        internal_call_id,
                                        content: streaming::ToolCallDeltaContent::Delta(delta.delta.clone())
                                    })
                                }

                                _ => { continue }
                            }
                        }

                        if let StreamingCompletionChunk::Response(chunk) = data {
                            if let ResponseChunk { kind: ResponseChunkKind::ResponseCompleted, response, .. } = *chunk {
                                span.record("gen_ai.response.id", response.id);
                                span.record("gen_ai.response.model", response.model);
                                if let Some(usage) = response.usage {
                                    final_usage = usage;
                                }
                            } else {
                                continue;
                            }
                        }
                    }
                    Err(crate::http_client::Error::StreamEnded) => {
                        event_source.close();
                    }
                    Err(error) => {
                        tracing::error!(?error, "SSE error");
                        yield Err(CompletionError::ProviderError(error.to_string()));
                        break;
                    }
                }
            }

            // Ensure event source is closed when stream ends
            event_source.close();

            for tool_call in &tool_calls {
                yield Ok(tool_call.to_owned())
            }

            span.record("gen_ai.usage.input_tokens", final_usage.input_tokens);
            span.record("gen_ai.usage.output_tokens", final_usage.output_tokens);
            span.record(
                "gen_ai.usage.cached_tokens",
                final_usage
                    .input_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0),
            );
            tracing::info!("OpenAI stream finished");

            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                usage: final_usage
            }));
        }.instrument(span);

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{ItemChunkKind, StreamingCompletionChunk, reasoning_choices_from_done_item};
    use crate::http_client::{self, HttpClientExt, LazyBody, MultipartForm, Request, Response};
    use crate::message::{ReasoningContent, Text};
    use crate::providers::openai::responses_api::{
        AdditionalParameters, AssistantContent as ResponsesAssistantContent,
        CompletionResponse as ResponsesCompletionResponse, InputTokensDetails,
        Message as ResponsesMessage, Output, OutputFunctionCall, OutputMessage, OutputRole,
        OutputTokensDetails, ReasoningSummary, ResponseObject, ResponseStatus, ResponsesUsage,
        ToolResult as ResponsesToolResult, ToolStatus, UserContent as ResponsesUserContent,
    };
    use crate::streaming::{RawStreamingChoice, StreamedAssistantContent, StreamingPrompt};
    use bytes::Bytes;
    use futures::StreamExt;
    use rig::{
        agent::MultiTurnStreamItem, client::CompletionClient, providers::openai,
        streaming::StreamingChat,
    };
    use serde_json::{self, Value, json};
    use std::sync::{Arc, Mutex};

    use crate::{
        completion::{Message, ToolDefinition},
        tool::{Tool, ToolError},
    };

    #[derive(Clone, Debug, Default)]
    struct RecordingResponsesStreamingClient {
        requests: Arc<Mutex<Vec<Bytes>>>,
    }

    impl RecordingResponsesStreamingClient {
        fn recorded_requests(&self) -> Vec<Bytes> {
            self.requests
                .lock()
                .expect("request capture should not be poisoned")
                .clone()
        }
    }

    impl HttpClientExt for RecordingResponsesStreamingClient {
        fn send<T, U>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>>
        + crate::wasm_compat::WasmCompatSend
        + 'static
        where
            T: Into<Bytes>,
            T: crate::wasm_compat::WasmCompatSend,
            U: From<Bytes>,
            U: crate::wasm_compat::WasmCompatSend + 'static,
        {
            std::future::ready(Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::NOT_IMPLEMENTED,
            )))
        }

        fn send_multipart<U>(
            &self,
            _req: Request<MultipartForm>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>>
        + crate::wasm_compat::WasmCompatSend
        + 'static
        where
            U: From<Bytes>,
            U: crate::wasm_compat::WasmCompatSend + 'static,
        {
            std::future::ready(Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::NOT_IMPLEMENTED,
            )))
        }

        fn send_streaming<T>(
            &self,
            req: Request<T>,
        ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>>
        + crate::wasm_compat::WasmCompatSend
        where
            T: Into<Bytes>,
        {
            let client = self.clone();
            let body: Bytes = req.into_body().into();
            async move {
                let turn = {
                    let mut requests = client
                        .requests
                        .lock()
                        .expect("request capture should not be poisoned");
                    requests.push(body.clone());
                    requests.len() - 1
                };

                match turn {
                    0 => sse_response(first_turn_sse_bytes()),
                    1 => {
                        validate_openai_follow_up_request(body.as_ref()).map_err(|message| {
                            http_client::Error::InvalidStatusCodeWithMessage(
                                http::StatusCode::BAD_REQUEST,
                                message,
                            )
                        })?;
                        sse_response(second_turn_sse_bytes())
                    }
                    _ => Err(http_client::Error::InvalidStatusCodeWithMessage(
                        http::StatusCode::BAD_REQUEST,
                        format!("unexpected extra streaming turn {turn}"),
                    )),
                }
            }
        }
    }

    struct ExampleTool;

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

    fn sse_response(bytes: Bytes) -> http_client::Result<http_client::StreamingResponse> {
        let byte_stream = futures::stream::iter(vec![Ok::<Bytes, http_client::Error>(bytes)]);
        let boxed_stream: crate::http_client::sse::BoxedStream = Box::pin(byte_stream);

        Response::builder()
            .status(http::StatusCode::OK)
            .header(http::header::CONTENT_TYPE, "text/event-stream")
            .body(boxed_stream)
            .map_err(http_client::Error::Protocol)
    }

    fn sse_event_bytes(events: &[Value]) -> Bytes {
        let payload = events
            .iter()
            .map(|event| {
                format!(
                    "data: {}\n\n",
                    serde_json::to_string(event).expect("event should serialize")
                )
            })
            .collect::<String>();

        Bytes::from(payload)
    }

    fn sample_response(response_id: &str, model: &str) -> ResponsesCompletionResponse {
        ResponsesCompletionResponse {
            id: response_id.to_string(),
            object: ResponseObject::Response,
            created_at: 0,
            status: ResponseStatus::Completed,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: model.to_string(),
            usage: Some(ResponsesUsage {
                input_tokens: 10,
                input_tokens_details: Some(InputTokensDetails { cached_tokens: 0 }),
                output_tokens: 5,
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                },
                total_tokens: 15,
            }),
            output: Vec::new(),
            tools: Vec::new(),
            additional_parameters: AdditionalParameters::default(),
        }
    }

    fn response_completed_event(
        sequence_number: u64,
        response: ResponsesCompletionResponse,
    ) -> Value {
        json!({
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": response
        })
    }

    fn output_item_done_event(
        item_id: &str,
        output_index: u64,
        sequence_number: u64,
        item: Output,
    ) -> Value {
        json!({
            "type": "response.output_item.done",
            "item_id": item_id,
            "output_index": output_index,
            "sequence_number": sequence_number,
            "item": item
        })
    }

    fn output_text_delta_event(
        item_id: &str,
        output_index: u64,
        content_index: u64,
        sequence_number: u64,
        delta: &str,
    ) -> Value {
        json!({
            "type": "response.output_text.delta",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": content_index,
            "sequence_number": sequence_number,
            "delta": delta
        })
    }

    fn first_turn_sse_bytes() -> Bytes {
        sse_event_bytes(&[
            output_item_done_event(
                "tool_call_1",
                0,
                1,
                Output::FunctionCall(OutputFunctionCall {
                    id: "tool_call_1".to_string(),
                    arguments: Value::Null,
                    call_id: "call_1".to_string(),
                    name: "example_tool".to_string(),
                    status: ToolStatus::Completed,
                }),
            ),
            response_completed_event(2, sample_response("resp_turn_1", "gpt-5.4")),
        ])
    }

    fn second_turn_sse_bytes() -> Bytes {
        sse_event_bytes(&[
            output_text_delta_event("msg_1", 0, 0, 1, "done"),
            output_item_done_event(
                "msg_1",
                0,
                2,
                Output::Message(OutputMessage {
                    id: "msg_1".to_string(),
                    role: OutputRole::Assistant,
                    status: ResponseStatus::Completed,
                    content: vec![ResponsesAssistantContent::OutputText(Text {
                        text: "done".to_string(),
                    })],
                }),
            ),
            response_completed_event(3, sample_response("resp_turn_2", "gpt-5.4")),
        ])
    }

    fn validate_openai_follow_up_request(body: &[u8]) -> Result<(), String> {
        let body_json: Value = serde_json::from_slice(body)
            .map_err(|err| format!("expected valid OpenAI Responses request JSON: {err}"))?;
        let input = body_json
            .get("input")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("expected OpenAI request input array, got {body_json:?}"))?;

        if input.len() != 3 {
            return Err(format!(
                "expected second turn input to contain [user prompt, function_call, function_call_output], got {input:?}"
            ));
        }

        let user_prompt: ResponsesMessage = serde_json::from_value(input[0].clone())
            .map_err(|err| format!("expected typed OpenAI user prompt, got {input:?}: {err}"))?;
        if !matches!(
            user_prompt,
            ResponsesMessage::User { content, .. }
                if matches!(
                    content.first(),
                    ResponsesUserContent::InputText { text } if text == "Call my example tool"
                )
        ) {
            return Err(format!(
                "expected first second-turn input item to be the original user prompt, got {input:?}"
            ));
        }

        let function_call: OutputFunctionCall =
            serde_json::from_value(input[1].clone()).map_err(|err| {
                format!("expected typed OpenAI function_call item, got {input:?}: {err}")
            })?;
        if !(function_call.id == "tool_call_1"
            && function_call.call_id == "call_1"
            && function_call.name == "example_tool")
        {
            return Err(format!(
                "expected assistant function_call to be preserved before the tool result, got {input:?}"
            ));
        }

        let function_call_output: ResponsesToolResult = serde_json::from_value(input[2].clone())
            .map_err(|err| {
                format!("expected typed OpenAI function_call_output item, got {input:?}: {err}")
            })?;
        if !(function_call_output.call_id == "call_1"
            && function_call_output.output.contains("Example answer"))
        {
            return Err(format!(
                "expected tool result to be serialized as the third second-turn input item, got {input:?}"
            ));
        }

        Ok(())
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
    async fn openai_stream_prompt_preserves_tool_call_history_between_turns() {
        let http_client = RecordingResponsesStreamingClient::default();
        let client = openai::Client::builder()
            .http_client(http_client.clone())
            .api_key("test-key")
            .build()
            .expect("client should build");
        let agent = client.agent("gpt-5.4").tool(ExampleTool).build();

        let mut stream = agent
            .stream_prompt("Call my example tool")
            .multi_turn(3)
            .await;
        let mut final_text = String::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => final_text.push_str(&text.text),
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Err(err) => panic!("unexpected OpenAI streaming error: {err:?}"),
                _ => {}
            }
        }

        assert_eq!(final_text, "done");

        let requests = http_client.recorded_requests();
        assert_eq!(
            requests.len(),
            2,
            "expected exactly two OpenAI streaming turns"
        );

        // Intentionally redundant to keep the expected follow-up request shape explicit here.
        validate_openai_follow_up_request(requests[1].as_ref())
            .expect("second OpenAI request should preserve tool call history");
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
