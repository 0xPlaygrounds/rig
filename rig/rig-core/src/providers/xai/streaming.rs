//! xAI Responses API Streaming
//!
//! This module reuses OpenAI's Responses API streaming types since xAI's API
//! is designed to be compatible with OpenAI's format.

use async_stream::stream;
use futures::StreamExt;
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::json_utils;
use crate::message::ReasoningContent;
use crate::providers::openai::responses_api::streaming::{
    ItemChunkKind, ResponseChunk, ResponseChunkKind, StreamingCompletionChunk,
    StreamingCompletionResponse, StreamingItemDoneOutput,
};
use crate::providers::openai::responses_api::{Output, ReasoningSummary, ResponsesUsage};
use crate::providers::xai::completion::{CompletionModel, XAICompletionRequest};
use crate::streaming::{self, RawStreamingChoice};

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let preamble = completion_request.preamble.clone();
        let mut request =
            XAICompletionRequest::try_from((self.model.to_string().as_ref(), completion_request))?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true}),
        );

        request.additional_params = Some(params);

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "xAI streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/v1/responses")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "xai",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        send_xai_streaming_request(self.client.clone(), req)
            .instrument(span)
            .await
    }
}

/// Send a streaming request
pub(crate) async fn send_xai_streaming_request<T>(
    http_client: T,
    req: http::Request<Vec<u8>>,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    let span = tracing::Span::current();
    let mut event_source = GenericEventSource::new(http_client, req);

    let stream = stream! {
        let span = tracing::Span::current();
        let mut final_usage = ResponsesUsage::new();
        let mut tool_call_internal_ids: std::collections::HashMap<String, String> = std::collections::HashMap::new();

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

                    let data = match serde_json::from_str::<StreamingCompletionChunk>(&evt.data) {
                        Ok(data) => data,
                        Err(err) => {
                            tracing::debug!(?err, data = evt.data, "Failed to parse SSE message");
                            continue;
                        }
                    };

                    if let StreamingCompletionChunk::Delta(chunk) = &data {
                        match &chunk.data {
                            ItemChunkKind::OutputItemAdded(StreamingItemDoneOutput {
                                item: Output::FunctionCall(func),
                                ..
                            }) => {
                                let internal_call_id = tool_call_internal_ids
                                    .entry(func.id.clone())
                                    .or_insert_with(|| nanoid::nanoid!())
                                    .clone();
                                yield Ok(RawStreamingChoice::ToolCallDelta {
                                    id: func.id.clone(),
                                    internal_call_id,
                                    content: streaming::ToolCallDeltaContent::Name(func.name.clone()),
                                });
                            }

                            ItemChunkKind::OutputItemDone(StreamingItemDoneOutput {
                                item: Output::FunctionCall(func),
                                ..
                            }) => {
                                let internal_id = tool_call_internal_ids
                                    .entry(func.id.clone())
                                    .or_insert_with(|| nanoid::nanoid!())
                                    .clone();
                                // Yield immediately so users can execute tools while stream continues
                                yield Ok(RawStreamingChoice::ToolCall(
                                    streaming::RawStreamingToolCall::new(
                                        func.id.clone(),
                                        func.name.clone(),
                                        func.arguments.clone(),
                                    )
                                    .with_internal_call_id(internal_id)
                                    .with_call_id(func.call_id.clone()),
                                ));
                            }

                            ItemChunkKind::OutputItemDone(StreamingItemDoneOutput {
                                item: Output::Reasoning { summary, id },
                                ..
                            }) => {
                                let reasoning = summary
                                    .iter()
                                    .map(|x| {
                                        let ReasoningSummary::SummaryText { text } = x;
                                        text.to_owned()
                                    })
                                    .collect::<Vec<String>>()
                                    .join("\n");
                                yield Ok(RawStreamingChoice::Reasoning {
                                    id: Some(id.to_string()),
                                    content: ReasoningContent::Summary(reasoning),
                                });
                            }

                            ItemChunkKind::OutputTextDelta(delta) => {
                                yield Ok(RawStreamingChoice::Message(delta.delta.clone()));
                            }

                            ItemChunkKind::ReasoningSummaryTextDelta(delta) => {
                                yield Ok(RawStreamingChoice::ReasoningDelta {
                                    id: None,
                                    reasoning: delta.delta.clone(),
                                });
                            }

                            ItemChunkKind::FunctionCallArgsDelta(delta) => {
                                let internal_call_id = tool_call_internal_ids
                                    .entry(delta.item_id.clone())
                                    .or_insert_with(|| nanoid::nanoid!())
                                    .clone();
                                yield Ok(RawStreamingChoice::ToolCallDelta {
                                    id: delta.item_id.clone(),
                                    internal_call_id,
                                    content: streaming::ToolCallDeltaContent::Delta(delta.delta.clone()),
                                });
                            }

                            ItemChunkKind::RefusalDelta(delta) => {
                                yield Ok(RawStreamingChoice::Message(delta.delta.clone()));
                            }

                            _ => continue,
                        }
                    }

                    if let StreamingCompletionChunk::Response(chunk) = data
                        && let ResponseChunk {
                            kind: ResponseChunkKind::ResponseCompleted,
                            response,
                            ..
                        } = *chunk
                    {
                            span.record("gen_ai.response.id", &response.id);
                            span.record("gen_ai.response.model", &response.model);
                            if let Some(usage) = response.usage {
                                final_usage = usage;
                            }
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

        if !span.is_disabled() {
            span.record("gen_ai.usage.input_tokens", final_usage.input_tokens);
            span.record("gen_ai.usage.output_tokens", final_usage.output_tokens);
        }

        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: final_usage,
        }));
    }
    .instrument(span);

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream,
    )))
}
