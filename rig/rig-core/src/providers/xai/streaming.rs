//! xAI Responses API Streaming
//!
//! This module reuses OpenAI's Responses API streaming types since xAI's API
//! is designed to be compatible with OpenAI's format.

use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::GenericEventSource;
use crate::json_utils;
use crate::providers::openai::responses_api::streaming::{
    ResponsesStreamOptions, StreamingCompletionResponse, stream_from_event_source_with_options,
};
use crate::providers::xai::completion::{CompletionModel, XAICompletionRequest};
use crate::streaming;

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
            XAICompletionRequest::try_from((self.model.as_str(), completion_request))?;

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
                gen_ai.usage.cached_tokens = tracing::field::Empty,
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
    let event_source = GenericEventSource::new(http_client, req);

    Ok(stream_from_event_source_with_options(
        event_source,
        span,
        "xAI",
        ResponsesStreamOptions::strict_with_immediate_tool_calls(),
    ))
}

#[cfg(test)]
mod tests {
    use super::send_xai_streaming_request;
    use crate::message::ReasoningContent;
    use crate::providers::internal::chat_compatible::test_support::sse_bytes_from_json_events;
    use crate::providers::openai::responses_api::ReasoningSummary;
    use crate::providers::openai::responses_api::streaming::reasoning_choices_from_done_item;
    use crate::streaming::{RawStreamingChoice, StreamedAssistantContent};

    #[test]
    fn reasoning_done_item_emits_summary_then_encrypted() {
        let summary = vec![
            ReasoningSummary::SummaryText {
                text: "s1".to_string(),
            },
            ReasoningSummary::SummaryText {
                text: "s2".to_string(),
            },
        ];
        let choices = reasoning_choices_from_done_item("xr_1", &summary, Some("enc"));

        assert_eq!(choices.len(), 3);
        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Summary(text),
            }) if id == "xr_1" && text == "s1"
        ));
        assert!(matches!(
            choices.get(1),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Summary(text),
            }) if id == "xr_1" && text == "s2"
        ));
        assert!(matches!(
            choices.get(2),
            Some(RawStreamingChoice::Reasoning {
                id: Some(id),
                content: ReasoningContent::Encrypted(data),
            }) if id == "xr_1" && data == "enc"
        ));
    }

    #[tokio::test]
    async fn xai_stream_uses_shared_responses_stream_behavior() {
        use crate::http_client::mock::MockStreamingClient;
        use futures::StreamExt;
        use serde_json::json;

        let events = vec![
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "sequence_number": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "arguments": "{}",
                    "call_id": "call_123",
                    "name": "example_tool",
                    "status": "in_progress"
                }
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_123",
                "content_index": 0,
                "sequence_number": 2,
                "delta": "{\"query\":\"rig\"}"
            }),
            json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 3,
                "item": {
                    "type": "function_call",
                    "id": "fc_123",
                    "arguments": "{\"query\":\"rig\"}",
                    "call_id": "call_123",
                    "name": "example_tool",
                    "status": "completed"
                }
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 0,
                "summary_index": 0,
                "sequence_number": 4,
                "delta": "thinking"
            }),
            json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 5,
                "delta": "done"
            }),
            json!({
                "type": "response.completed",
                "sequence_number": 6,
                "response": {
                    "id": "resp_123",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "error": null,
                    "incomplete_details": null,
                    "instructions": null,
                    "max_output_tokens": null,
                    "model": "grok-4-0709",
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": null,
                        "output_tokens": 5,
                        "output_tokens_details": { "reasoning_tokens": 0 },
                        "total_tokens": 15
                    },
                    "output": [],
                    "tools": []
                }
            }),
        ];

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&events),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/responses")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_xai_streaming_request(client, req)
            .await
            .expect("stream should start");

        let mut saw_name_delta = false;
        let mut reasoning = String::new();
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut final_usage = None;
        let mut saw_tool_call_before_reasoning_or_text = false;

        while let Some(item) = stream.next().await {
            match item.expect("remaining stream items should be ok") {
                StreamedAssistantContent::ToolCallDelta {
                    content: crate::streaming::ToolCallDeltaContent::Name(name),
                    ..
                } => {
                    saw_name_delta = true;
                    assert_eq!(name, "example_tool");
                }
                StreamedAssistantContent::ToolCallDelta {
                    content: crate::streaming::ToolCallDeltaContent::Delta(_),
                    ..
                } => {}
                StreamedAssistantContent::ToolCall { tool_call, .. } => {
                    tool_calls.push(tool_call);
                }
                StreamedAssistantContent::ReasoningDelta {
                    reasoning: chunk, ..
                } => {
                    if !tool_calls.is_empty() {
                        saw_tool_call_before_reasoning_or_text = true;
                    }
                    reasoning.push_str(&chunk);
                }
                StreamedAssistantContent::Text(chunk) => {
                    if !tool_calls.is_empty() {
                        saw_tool_call_before_reasoning_or_text = true;
                    }
                    text.push_str(&chunk.text);
                }
                StreamedAssistantContent::Final(response) => final_usage = Some(response.usage),
                _ => {}
            }
        }

        assert!(saw_name_delta, "expected tool name delta");
        assert_eq!(reasoning, "thinking");
        assert_eq!(text, "done");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "fc_123");
        assert_eq!(tool_calls[0].call_id.as_deref(), Some("call_123"));
        assert_eq!(tool_calls[0].function.name, "example_tool");
        assert_eq!(
            tool_calls[0].function.arguments,
            serde_json::json!({"query":"rig"})
        );
        assert!(
            saw_tool_call_before_reasoning_or_text,
            "expected completed tool call before later reasoning/text chunks"
        );
        let usage = final_usage.expect("expected final usage");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn xai_stream_surfaces_terminal_errors_after_completed_tool_calls() {
        use crate::http_client::mock::MockStreamingClient;
        use futures::StreamExt;
        use serde_json::json;

        let tool_call_done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
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

        let failed = json!({
            "type": "response.failed",
            "sequence_number": 2,
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 0,
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "message": "response stream failed"
                },
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "grok-4-0709",
                "usage": null,
                "output": [],
                "tools": []
            }
        });

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&[tool_call_done, failed]),
        };
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/responses")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_xai_streaming_request(client, req)
            .await
            .expect("stream should start");

        match stream
            .next()
            .await
            .expect("stream should yield an item")
            .expect("first stream item should be ok")
        {
            StreamedAssistantContent::ToolCall { tool_call, .. } => {
                assert_eq!(tool_call.id, "fc_123");
                assert_eq!(tool_call.call_id.as_deref(), Some("call_123"));
                assert_eq!(tool_call.function.name, "example_tool");
                assert_eq!(tool_call.function.arguments, serde_json::json!({}));
            }
            other => panic!("expected completed tool call before terminal error, got {other:?}"),
        }

        let err = stream
            .next()
            .await
            .expect("stream should yield terminal error")
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
}
