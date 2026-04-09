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

fn response_error_message(error: Option<&super::ResponseError>, fallback: &str) -> String {
    if let Some(error) = error {
        if error.code.is_empty() {
            error.message.clone()
        } else {
            format!("{}: {}", error.code, error.message)
        }
    } else {
        format!("OpenAI response stream returned a {fallback}")
    }
}

fn response_chunk_error_message(
    kind: &ResponseChunkKind,
    response: &CompletionResponse,
) -> Option<String> {
    match kind {
        ResponseChunkKind::ResponseFailed => Some(response_error_message(
            response.error.as_ref(),
            "failed response",
        )),
        ResponseChunkKind::ResponseIncomplete => {
            let reason = response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str())
                .unwrap_or("unknown reason");

            Some(format!("OpenAI response stream was incomplete: {reason}"))
        }
        _ => None,
    }
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
                            let ResponseChunk { kind, response, .. } = *chunk;

                            match kind {
                                ResponseChunkKind::ResponseCompleted => {
                                    span.record("gen_ai.response.id", response.id.as_str());
                                    span.record("gen_ai.response.model", response.model.as_str());
                                    if let Some(usage) = response.usage {
                                        final_usage = usage;
                                    }
                                }
                                ResponseChunkKind::ResponseFailed | ResponseChunkKind::ResponseIncomplete => {
                                    let error_message = response_chunk_error_message(&kind, &response)
                                        .expect("terminal response should have an error message");
                                    yield Err(CompletionError::ProviderError(error_message));
                                    break;
                                }
                                _ => continue,
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
    use crate::completion::CompletionModel;
    use crate::http_client::mock::MockStreamingClient;
    use crate::message::ReasoningContent;
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

    fn sse_event_bytes(event: serde_json::Value) -> Bytes {
        Bytes::from(format!(
            "data: {}\n\n",
            serde_json::to_string(&event).expect("event should serialize")
        ))
    }

    async fn first_error_from_event(
        event: serde_json::Value,
    ) -> crate::completion::CompletionError {
        let client = openai::Client::builder()
            .http_client(MockStreamingClient {
                sse_bytes: sse_event_bytes(event),
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
                sse_bytes: sse_event_bytes(event),
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
