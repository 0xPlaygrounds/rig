//! The streaming module for the OpenAI Responses API.
//! Please see the `openai_streaming` or `openai_streaming_with_tools` example for more practical usage.
use crate::completion::{CompletionError, GetTokenUsage};
use crate::providers::openai::responses_api::{
    ReasoningSummary, ResponsesCompletionModel, ResponsesUsage,
};
use crate::streaming;
use crate::streaming::RawStreamingChoice;
use async_stream::stream;
use futures::StreamExt;
use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};
use tracing::debug;

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

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.input_tokens;
        usage.output_tokens = self.usage.output_tokens;
        usage.total_tokens = self.usage.total_tokens;
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
    FunctionCallArgsDelta(DeltaTextChunk),
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgsDone(ArgsTextChunk),
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone(SummaryPartChunk),
    #[serde(rename = "response.reasoning_summary_text.added")]
    ReasoningSummaryTextAdded(SummaryTextChunk),
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
#[serde(tag = "type")]
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
#[serde(tag = "type")]
pub enum SummaryPartChunkPart {
    SummaryText { text: String },
}

impl ResponsesCompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: crate::completion::CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let mut request = self.create_completion_request(completion_request)?;
        request.stream = Some(true);

        tracing::debug!("Input: {}", serde_json::to_string_pretty(&request)?);

        let builder = self.client.post("/responses").json(&request);
        send_compatible_streaming_request(builder).await
    }
}

pub async fn send_compatible_streaming_request(
    request_builder: RequestBuilder,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError> {
    let response = request_builder.send().await?;

    if !response.status().is_success() {
        return Err(CompletionError::ProviderError(format!(
            "{}: {}",
            response.status(),
            response.text().await?
        )));
    }

    // Handle OpenAI Compatible SSE chunks
    let inner = Box::pin(stream! {
        let mut stream = response.bytes_stream();

        let mut final_usage = ResponsesUsage::new();

        let mut partial_data = None;

        let mut tool_calls: Vec<RawStreamingChoice<StreamingCompletionResponse>> = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(CompletionError::from(e));
                    break;
                }
            };

            let text = match String::from_utf8(chunk.to_vec()) {
                Ok(t) => t,
                Err(e) => {
                    yield Err(CompletionError::ResponseError(e.to_string()));
                    break;
                }
            };

            for line in text.lines() {
                let mut line = line.to_string();

                // If there was a remaining part, concat with current line
                if partial_data.is_some() {
                    line = format!("{}{}", partial_data.unwrap(), line);
                    partial_data = None;
                }
                // Otherwise full data line
                else {
                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };

                    // Partial data, split somewhere in the middle
                    if !line.ends_with("}") {
                        partial_data = Some(data.to_string());
                    } else {
                        line = data.to_string();
                    }
                }

                let data = serde_json::from_str::<StreamingCompletionChunk>(&line);

                let Ok(data) = data else {
                    let err = data.unwrap_err();
                    debug!("Couldn't serialize data as StreamingCompletionResponse: {:?}", err);
                    continue;
                };

                debug!("Data get: {data:?}");


                if let StreamingCompletionChunk::Delta(chunk) = &data {
                    match &chunk.data {
                        ItemChunkKind::OutputItemDone(message) => {
                            match message {
                                StreamingItemDoneOutput {  item: Output::FunctionCall(func), .. } => {
                                    tracing::debug!("Function call received: {func:?}");
                                    tool_calls.push(streaming::RawStreamingChoice::ToolCall { id: func.id.clone(), call_id: Some(func.call_id.clone()), name: func.name.clone(), arguments: func.arguments.clone() });
                                }

                                StreamingItemDoneOutput {  item: Output::Reasoning {  summary, id }, .. } => {
                                    let reasoning = summary
                                        .iter()
                                        .map(|x| {
                                            let ReasoningSummary::SummaryText { text } = x;
                                            text.to_owned()
                                        })
                                        .collect::<Vec<String>>()
                                        .join("\n");
                                    yield Ok(streaming::RawStreamingChoice::Reasoning { reasoning, id: Some(id.to_string()) })
                                }
                                _ => continue
                            }
                        }
                        ItemChunkKind::OutputTextDelta(delta) => {
                            yield Ok(streaming::RawStreamingChoice::Message(delta.delta.clone()))
                        }
                        ItemChunkKind::RefusalDelta(delta) => {
                            yield Ok(streaming::RawStreamingChoice::Message(delta.delta.clone()))
                        }

                        _ => { continue }
                    }
                }

                    if let StreamingCompletionChunk::Response(chunk) = data && let Some(usage) = chunk.response.usage {
                        final_usage = usage;
                    }
            }
        }

        for tool_call in tool_calls {
            yield Ok(tool_call)
        }

        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: final_usage.clone()
        }))
    });

    Ok(streaming::StreamingCompletionResponse::stream(inner))
}
