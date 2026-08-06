//! xAI Responses API Streaming
//!
//! This module reuses OpenAI's Responses API streaming types since xAI's API
//! is designed to be compatible with OpenAI's format.

use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};
use tracing::{Level, enabled};
use tracing_futures::Instrument;

use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::GenericEventSource;
use crate::json_utils;
use crate::providers::openai::responses_api::streaming::{
    ResponsesStreamOptions, StreamingCompletionResponse, normalize_responses_stream,
    raw_stream_from_event_source_with_options,
};
use crate::providers::xai::completion::{CompletionModel, PROVIDER_NAME, XAICompletionRequest};
use crate::streaming::{self, RawStreamingResult};

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    /// Open a stream whose terminal record stays xAI-native.
    ///
    /// This is the escape hatch for provider-specific streaming fields rig
    /// does not normalize. The normalized
    /// [`stream`](crate::completion::CompletionModel::stream) path calls it and
    /// maps the terminal record once, so both paths open exactly one stream.
    pub async fn raw_stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<RawStreamingResult<StreamingCompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let record_telemetry_content = completion_request.record_telemetry_content;
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

        let span = CompletionSpanBuilder::new(
            PROVIDER_NAME,
            &request.model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(preamble.as_deref(), record_telemetry_content)
        .build();

        send_xai_streaming_request(self.client.clone(), req)
            .instrument(span)
            .await
    }

    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let stream = self.raw_stream(completion_request).await?;

        Ok(normalize_responses_stream(PROVIDER_NAME, stream))
    }
}

/// Send a streaming request, keeping the terminal record provider-native.
pub(crate) async fn send_xai_streaming_request<T>(
    http_client: T,
    req: http::Request<Vec<u8>>,
) -> Result<RawStreamingResult<StreamingCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    let span = tracing::Span::current();
    let event_source = GenericEventSource::new(http_client, req);

    Ok(raw_stream_from_event_source_with_options(
        event_source,
        span,
        ResponsesStreamOptions::strict_with_immediate_tool_calls(),
    ))
}

#[cfg(test)]
mod tests {
    use super::send_xai_streaming_request;
    use crate::message::ReasoningContent;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_json_events;
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
        let choices = reasoning_choices_from_done_item(
            &crate::streaming::PartId::wire("xr_1"),
            &summary,
            &[],
            Some("enc"),
        );

        assert_eq!(choices.len(), 3);
        assert!(matches!(
            choices.first(),
            Some(RawStreamingChoice::Reasoning {
                id,
                content: ReasoningContent::Summary(text),
            }) if id.as_wire() == Some("xr_1") && text == "s1"
        ));
        assert!(matches!(
            choices.get(1),
            Some(RawStreamingChoice::Reasoning {
                id,
                content: ReasoningContent::Summary(text),
            }) if id.as_wire() == Some("xr_1") && text == "s2"
        ));
        assert!(matches!(
            choices.get(2),
            Some(RawStreamingChoice::Reasoning {
                id,
                content: ReasoningContent::Encrypted(data),
            }) if id.as_wire() == Some("xr_1") && data == "enc"
        ));
    }

    #[tokio::test]
    async fn truncated_stream_does_not_synthesize_a_terminal_record() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;
        use serde_json::json;

        // Deltas then EOF without `response.completed`: the truncated stream
        // must deliver its content but never a synthesized terminal record.
        let delta = json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "hi"
        });

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&[delta]),
        };
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/responses")
            .body(Vec::new())
            .expect("request should build");

        let raw = send_xai_streaming_request(client, req)
            .await
            .expect("stream should start");
        let mut stream = super::normalize_responses_stream(super::PROVIDER_NAME, raw);

        let mut texts = Vec::new();
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item.expect("stream item should be Ok") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(_) => saw_terminal = true,
                _ => {}
            }
        }

        assert_eq!(texts, ["hi"]);
        assert!(
            !saw_terminal,
            "EOF without response.completed must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn xai_stream_surfaces_terminal_errors_after_completed_tool_calls() {
        use crate::test_utils::MockStreamingClient;
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

        let raw = send_xai_streaming_request(client, req)
            .await
            .expect("stream should start");
        let mut stream = super::normalize_responses_stream(super::PROVIDER_NAME, raw);

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
}
