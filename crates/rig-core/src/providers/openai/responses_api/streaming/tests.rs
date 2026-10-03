use super::{ResponsesDecoder, ResponsesEvent, classify_responses_payload};
use crate::completion::CompletionRequest;
use crate::driver::{Decoded, feed_frames};
use crate::error::{ErrorKind, ErrorReport, ProviderError};
use crate::message::AssistantContent;
use crate::operation::Completion;
use crate::providers::internal::openai_chat_completions_compatible::test_support::{
    sse_bytes_from_data_lines, sse_bytes_from_json_events,
};
use crate::providers::openai::OpenAIConfig;
use crate::streaming::{Item, PartKind, StreamEvent};
use crate::test_utils::MockStreamingClient;
use crate::wire::{Mode, WireEvent, WireFrame};
use futures::StreamExt;
use serde_json::{self, json};

#[test]
fn classify_known_event_decodes() {
    let frame = json!({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": "hi",
    })
    .to_string();
    assert!(matches!(
        classify_responses_payload(&frame),
        WireEvent::Known(ResponsesEvent::Frame { .. })
    ));
}

#[test]
fn classify_unknown_event_type_is_unknown() {
    let frame = json!({
        "type": "response.web_search_call.searching",
        "output_index": 0,
        "sequence_number": 1,
    })
    .to_string();
    assert!(matches!(
        classify_responses_payload(&frame),
        WireEvent::Unknown { event_type, .. } if event_type == "response.web_search_call.searching"
    ));
}

/// #2258 G4: `response.reasoning_text.done` terminates every raw-reasoning
/// block on all three Responses surfaces. It used to be absent from the
/// known-event set, so each block logged a spurious "unknown event" warn
/// and passed through as `Unknown`.
///
/// The tag must be known (no `Unknown`) and its frame must classify (no
/// `Corrupt`).
///
/// No recorded cassette contains this event; the wire shape is the
/// Responses spec's, so this unit test is the pin.
#[test]
fn classify_reasoning_text_done_is_known_and_decodes() {
    let frame = json!({
        "type": "response.reasoning_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 7,
        "text": "the model's raw chain of thought",
    })
    .to_string();

    let event = classify_responses_payload(&frame);
    assert!(
        !matches!(event, WireEvent::Unknown { .. }),
        "the tag must be in the known-event set: {event:?}"
    );
    assert!(matches!(
        event,
        WireEvent::Known(ResponsesEvent::Frame { kind, .. })
            if kind == "response.reasoning_text.done"
    ));
}

#[test]
fn classify_invalid_json_is_corrupt() {
    assert!(matches!(
        classify_responses_payload("{not json"),
        WireEvent::Corrupt(_)
    ));
}

/// A known event whose fields a gateway retyped still classifies: only the
/// decoder reads its fields, and it reads only what it needs (#2668).
#[test]
fn a_known_event_with_a_retyped_field_still_classifies() {
    let frame = json!({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": 42,
    })
    .to_string();
    assert!(matches!(
        classify_responses_payload(&frame),
        WireEvent::Known(ResponsesEvent::Frame { .. })
    ));
}

fn sample_response(status: &str) -> serde_json::Value {
    json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "gpt-5.4",
        "output": [],
        "tools": [],
    })
}

#[tokio::test]
async fn response_failed_chunk_surfaces_provider_error_without_empty_code_prefix() {
    let mut response = sample_response("failed");
    response["error"] = json!({ "code": "", "message": "maximum context length exceeded" });

    let event = json!({
        "type": "response.failed",
        "sequence_number": 1,
        "response": response,
    });

    let err = first_error_from_event(event).await;

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(err.provider_response_body().is_some_and(|body| {
        body.contains("response.failed") && body.contains("maximum context length exceeded")
    }));
}

#[tokio::test]
async fn response_failed_chunk_surfaces_provider_error_with_code_prefix() {
    let mut response = sample_response("failed");
    response["error"] =
        json!({ "code": "context_length_exceeded", "message": "maximum context length exceeded" });

    let event = json!({
        "type": "response.failed",
        "sequence_number": 1,
        "response": response,
    });

    let err = first_error_from_event(event).await;

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(err.provider_response_body().is_some_and(|body| {
        body.contains("response.failed")
            && body.contains("context_length_exceeded")
            && body.contains("maximum context length exceeded")
    }));
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

    let mut stream = responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[payload]),
    })
    .await;

    let err = stream
        .next()
        .await
        .expect("stream should yield an item")
        .expect_err("stream should surface a provider response error");
    let err = ErrorReport::from(&err);
    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(
        err.provider_response_body()
            .is_some_and(|body| { body.contains("\"type\":\"error\"") && body.contains("boom") })
    );
    assert!(
        stream.next().await.is_none(),
        "stream should terminate after error event"
    );
}

#[tokio::test]
async fn streaming_http_non_success_preserves_status_and_body() {
    use crate::test_utils::HttpErrorStreamingClient;

    let body = r#"{"error":{"message":"quota exceeded"}}"#;
    let mut stream = responses_stream(HttpErrorStreamingClient::new(
        http::StatusCode::TOO_MANY_REQUESTS,
        body,
    ))
    .await;

    let err = stream
        .next()
        .await
        .expect("stream should yield transport error")
        .expect_err("HTTP non-success should surface as a stream error");
    let err = ErrorReport::from(&err);
    assert_eq!(
        err.http_status,
        Some(http::StatusCode::TOO_MANY_REQUESTS.as_u16())
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

#[tokio::test]
async fn streaming_non_http_transport_error_stays_a_transport_error() {
    use crate::test_utils::SequencedStreamingHttpClient;

    let chunks = vec![Err(crate::http_client::Error::InvalidContentType(
        http::HeaderValue::from_static("application/json"),
    ))];
    let mut stream = responses_stream(SequencedStreamingHttpClient::new(chunks)).await;

    let err = stream
        .next()
        .await
        .expect("stream should yield transport error")
        .expect_err("non-HTTP transport failure should surface as a transport error");
    let err = ErrorReport::from(&err);
    assert_eq!(
        err.to_string(),
        "HttpError: Invalid content type was returned: \"application/json\""
    );
    assert_eq!(err.kind, ErrorKind::Http);
    // A response-less transport failure has no provider response body.
    assert_eq!(err.provider_response_body(), None);
    assert_eq!(err.http_status, None);
}

#[tokio::test]
async fn response_completed_chunk_populates_final_usage() {
    let mut response = sample_response("completed");
    response["usage"] = json!({ "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "output_tokens_details": { "reasoning_tokens": 0 } });

    let event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": response,
    });

    let usage = stream_final_from_event(event).await.usage;
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(5));
    assert_eq!(usage.total_tokens, Some(15));
    assert_eq!(usage.reasoning_tokens, Some(0));
}

/// The terminal `response.completed` frame carries usage. An object-shaped
/// `top_p` echoed on that frame (MiniMax-style endpoints, rig#2483) or a
/// numeric one buffered under `serde_json/arbitrary_precision` (rig#2493)
/// must not turn the terminal into a parse error, which would fail the turn
/// and lose the usage.
#[tokio::test]
async fn response_completed_chunk_tolerates_object_shaped_top_p() {
    let mut response = sample_response("completed");
    response["top_p"] = json!({ "value": 0.95 });
    response["usage"] = json!({
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15
    });
    let event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": response,
    });

    let usage = stream_final_from_event(event).await.usage;
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.total_tokens, Some(15));
}

/// One `message` output item, as a terminal response body states it.
fn message_output_item(id: &str, text: &str) -> serde_json::Value {
    json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "content": [{ "type": "output_text", "annotations": [], "text": text }],
    })
}

/// The visible text parts of a folded choice, in order.
fn choice_text_parts(response: &crate::completion::CompletionResponse) -> Vec<String> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            crate::completion::AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

/// The OpenAI Responses stream one scripted transport yields: the live
/// loop, over a socket that answers from a script.
async fn responses_stream<H>(http: H) -> crate::streaming::CompletionStream
where
    H: crate::driver::Transport<crate::providers::openai::responses_api::wire::Responses>,
{
    let model = crate::driver::Model::new(OpenAIConfig::new("test-key").responses("gpt-5.4"), http);
    let request = CompletionRequest::new("hello");
    model.stream(request).expect("stream should start")
}

/// The same, for a body scripted as JSON events.
async fn responses_stream_of(events: &[serde_json::Value]) -> crate::streaming::CompletionStream {
    responses_stream(MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(events),
    })
    .await
}

/// Decode a Responses SSE body, its `data:` lines, through the decoder a
/// buffered replay runs.
fn decoded_body(body: &str) -> Decoded<Completion> {
    let frames: Vec<WireFrame> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim))
        .filter(|data| !data.is_empty() && *data != "[DONE]")
        .map(|data| WireFrame::Text(data.to_owned()))
        .collect();
    feed_frames!(ResponsesDecoder::new(), "openai", frames)
}

/// An SSE body of `events`, one `data:` line each.
fn body_of(events: &[serde_json::Value]) -> String {
    events
        .iter()
        .map(|event| format!("data: {event}\n"))
        .collect()
}

/// The text fragments among `events`.
fn texts_of(events: &[&StreamEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

/// The calls a response's choice carries.
fn calls_of(response: &crate::completion::CompletionResponse) -> Vec<crate::message::ToolCall> {
    response.tool_calls().cloned().collect()
}

async fn first_error_from_event(event: serde_json::Value) -> ErrorReport {
    let mut stream = responses_stream_of(&[event]).await;
    let error = stream
        .next()
        .await
        .expect("stream should yield an item")
        .expect_err("stream should surface a provider error");
    ErrorReport::from(&error)
}

/// What a stream of `event` finished with.
async fn stream_final_from_event(
    event: serde_json::Value,
) -> crate::completion::CompletionResponse {
    let mut stream = responses_stream_of(&[event]).await;
    while let Some(item) = stream.next().await {
        item.expect("completed stream should not error");
    }
    stream.finish().await.expect("the stream ended")
}

/// Drain a stream whose provider fully delivered one tool call before a
/// terminal error: the call's part comes first, then the error, then
/// nothing.
async fn flushed_tool_call_then_error(
    stream: &mut crate::streaming::CompletionStream,
) -> (crate::message::ToolCall, ErrorReport) {
    let mut tool_call = None;
    let err = loop {
        match stream
            .next()
            .await
            .expect("stream should yield the flushed tool call, then the error")
        {
            Ok(Item::Event(
                StreamEvent::Start {
                    kind: PartKind::ToolCall,
                    ..
                }
                | StreamEvent::Arguments { .. },
            )) => {}
            Ok(Item::Event(StreamEvent::End {
                content: AssistantContent::ToolCall(call),
                ..
            })) => tool_call = Some(call),
            Ok(other) => panic!("expected the flushed tool call first, got {other:?}"),
            Err(err) => break ErrorReport::from(&err),
        }
    };
    let tool_call = tool_call.expect("the flushed tool call must precede the terminal error");
    assert!(
        stream.next().await.is_none(),
        "nothing may follow the terminal error"
    );
    (tool_call, err)
}

/// The done event restates text the deltas already streamed, so it must be
/// a no-op: replaying it would double every raw-reasoning block.
#[test]
fn reasoning_text_done_emits_nothing() {
    let decoded = decoded_body(&body_of(&[json!({
        "type": "response.reasoning_text.done",
        "item_id": "rs_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 7,
        "text": "the model's raw chain of thought",
    })]));
    assert!(
        decoded.events().is_empty(),
        "the done restatement must not re-emit the reasoning text: {:?}",
        decoded.events()
    );
}

#[test]
fn a_buffered_body_preserves_its_error_payloads() {
    let mut response = sample_response("failed");
    response["error"] = json!({ "code": "server_error", "message": "response failed" });
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
        let err = decoded_body(&format!("data: {payload}\n"))
            .outcome
            .expect_err("error payload should surface as provider response");

        assert!(matches!(err, ProviderError::ProviderResponse(_)));
        assert_eq!(err.provider_response_status(), None);
        assert_eq!(err.provider_response_body(), Some(payload.as_str()));
    }
}

#[test]
fn reasoning_output_item_done_emits_reasoning_text_content() {
    let decoded = decoded_body(&body_of(&[
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
        }),
        completed_with(2, json!([])),
    ]));
    // The done item is one whole reasoning part holding the item.
    let ended = decoded.ended();
    let [AssistantContent::Reasoning(reasoning)] = ended.as_slice() else {
        panic!("one reasoning part: {:?}", decoded.events());
    };
    assert_eq!(reasoning.text, "visible reasoning");
    assert_eq!(
        ended[0].native_item().and_then(|item| item["id"].as_str()),
        Some("rs_text_1")
    );
}

/// Envelope-less replay shape (ChatGPT bodies): an id-less summary delta,
/// the done item restating the whole part, then visible text.
#[test]
fn envelope_less_reasoning_then_text_decodes() {
    let decoded = decoded_body(&body_of(&[
        json!({
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 1,
            "delta": "thinking",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "reasoning",
                "id": "",
                "summary": [{ "type": "summary_text", "text": "thinking, complete" }],
                "status": "completed"
            },
        }),
        json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 1,
            "content_index": 0,
            "sequence_number": 3,
            "delta": "the answer",
        }),
    ]));
    assert_eq!(texts_of(&decoded.events()), ["the answer"]);
}

#[test]
fn reasoning_text_delta_emits_reasoning_delta() {
    let decoded = decoded_body(&body_of(&[json!({
        "type": "response.reasoning_text.delta",
        "item_id": "rs_delta_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": "thinking",
    })]));
    // The first fragment of an unseen item opens its part.
    assert!(
        matches!(
            decoded.events().as_slice(),
            [
                StreamEvent::Start { kind: PartKind::Reasoning, .. },
                StreamEvent::Reasoning { text, .. },
            ] if text == "thinking"
        ),
        "{:?}",
        decoded.events()
    );
}

#[tokio::test]
async fn response_incomplete_chunk_is_a_successful_terminal_with_mapped_finish_reason() {
    let text_delta = json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "partial",
        "item_id": "msg_incomplete_1",
        "output_index": 0,
        "sequence_number": 1,
    });

    let mut response = sample_response("incomplete");
    response["incomplete_details"] = json!({ "reason": "max_output_tokens" });
    response["usage"] = json!({ "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "output_tokens_details": { "reasoning_tokens": 0 } });

    let incomplete = json!({
        "type": "response.incomplete",
        "sequence_number": 2,
        "response": response,
    });

    let mut stream = responses_stream_of(&[text_delta, incomplete]).await;
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text: delta, .. }) =
            item.expect("incomplete stream should not error")
        {
            text.push_str(&delta);
        }
    }

    // The partial output survives, and the end maps the incomplete status
    // to the same finish reason as the unary path.
    assert_eq!(text, "partial");
    let response = stream.finish().await.expect("the reply ended");
    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.total_tokens, Some(15));
}

/// A `response.failed` after a fully-delivered tool call: the call ended
/// when its done item arrived, the terminal error follows, and nothing
/// comes after it.
#[tokio::test]
async fn response_failed_follows_the_delivered_tool_call() {
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

    let mut response = sample_response("failed");
    response["error"] = json!({ "code": "server_error", "message": "response stream failed" });

    let failed = json!({
        "type": "response.failed",
        "sequence_number": 2,
        "response": response,
    });

    let mut stream = responses_stream_of(&[tool_call_done, failed]).await;
    let (tool_call, err) = flushed_tool_call_then_error(&mut stream).await;
    assert_eq!(tool_call.id, crate::message::CallId::from_wire("call_123"));
    assert_eq!(
        tool_call.native.as_ref().map(|native| &native.item["id"]),
        Some(&json!("fc_123"))
    );
    assert_eq!(tool_call.function.name, "example_tool");

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
    assert!(err.provider_response_body().is_some_and(|body| {
        body.contains("response.failed") && body.contains("response stream failed")
    }));
    assert!(stream.finish().await.is_err());
}

/// Same ordering for a transport failure: the fully-delivered tool call,
/// then the error, then the end — with no response.
#[tokio::test]
async fn a_transport_error_follows_the_delivered_tool_call() {
    use crate::test_utils::SequencedStreamingHttpClient;

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
    let chunks = vec![
        Ok(sse_bytes_from_data_lines([tool_call_done.to_string()])),
        Err(crate::http_client::Error::non_success_with_details(
            http::StatusCode::BAD_GATEWAY,
            http::HeaderMap::new(),
            r#"{"error":{"message":"upstream unavailable"}}"#.to_string(),
        )),
    ];
    let mut stream = responses_stream(SequencedStreamingHttpClient::new(chunks)).await;

    let (tool_call, err) = flushed_tool_call_then_error(&mut stream).await;
    assert_eq!(tool_call.id, crate::message::CallId::from_wire("call_123"));
    assert_eq!(
        tool_call.native.as_ref().map(|native| &native.item["id"]),
        Some(&json!("fc_123"))
    );
    assert_eq!(
        err.http_status,
        Some(http::StatusCode::BAD_GATEWAY.as_u16())
    );
    assert!(stream.finish().await.is_err());
}

/// A terminal whose `usage` is not an object still ends the reply: usage
/// is read leniently, and a counter that is not one is unreported.
#[tokio::test]
async fn a_terminal_with_malformed_usage_still_ends_the_reply() {
    let mut event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": sample_response("completed"),
    });
    event["response"]["usage"] = json!("banana");

    let response = stream_final_from_event(event).await;
    assert!(!response.usage.is_reported());
    assert_eq!(response.stop(), crate::message::StopReason::Stop);
}

/// An invented event type stays skippable for forward compatibility; a
/// later genuine terminal still ends the reply.
#[tokio::test]
async fn unknown_event_type_is_skipped_and_stream_completes() {
    let unknown = json!({
        "type": "response.rocket_launch",
        "payload": { "count": 3 }
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response("completed"),
    });
    let mut stream = responses_stream_of(&[unknown, completed]).await;
    while let Some(item) = stream.next().await {
        item.expect("unknown event types must not surface as errors");
    }
    stream
        .finish()
        .await
        .expect("the genuine terminal must still end the reply");
}

#[tokio::test]
async fn refusal_content_part_frames_are_no_ops_and_refusal_text_streams() {
    // A refusal turn emits `response.content_part.added/.done` with a
    // `refusal` part — a shape outside the modeled text parts — followed
    // by the refusal text via `response.refusal.delta`. The part frames
    // must parse as no-ops (never error items); the deltas carry the
    // content.
    let events = [
        json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "part": { "type": "refusal", "refusal": "" }
        }),
        json!({
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 2,
            "delta": "I can't help with that."
        }),
        json!({
            "type": "response.content_part.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
            "part": { "type": "refusal", "refusal": "I can't help with that." }
        }),
        json!({
            "type": "response.content_part.added",
            "item_id": "rs_1",
            "output_index": 1,
            "content_index": 0,
            "sequence_number": 4,
            "part": { "type": "reasoning_text", "text": "" }
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 5,
            "response": sample_response("completed"),
        }),
    ];
    let decoded = decoded_body(&body_of(&events));
    assert_eq!(texts_of(&decoded.events()), ["I can't help with that."]);
    assert!(decoded.outcome.is_ok(), "the terminal must still arrive");
}

#[tokio::test]
async fn truncated_stream_does_not_synthesize_an_end() {
    // Deltas then EOF without `response.completed`: the truncated turn is
    // never presented as a successful completion.
    let deltas = [
        json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "hel"
        }),
        json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 2,
            "delta": "lo"
        }),
    ];

    let mut stream = responses_stream_of(&deltas).await;
    let mut texts = Vec::new();
    while let Some(item) = stream.next().await {
        if let Ok(Item::Event(StreamEvent::Text { text, .. })) = item {
            texts.push(text);
        }
    }
    assert_eq!(texts, ["hel", "lo"]);
    assert!(matches!(
        stream.finish().await,
        Err(ProviderError::Truncated)
    ));
}

/// A corrupt known frame fails the reply — even when a valid terminal
/// follows — instead of returning a silently partial completion.
#[test]
fn only_invalid_json_fails_the_buffered_body() {
    // A known event whose field is retyped is read leniently.
    let retyped = json!({
        "type": "response.output_text.delta",
        "delta": 42
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response("completed"),
    });
    decoded_body(&body_of(&[retyped, completed.clone()]))
        .outcome
        .expect("a retyped field no block needs does not fail the reply");

    // Syntactically invalid JSON fails.
    let body = format!("data: {{not json\ndata: {completed}\n");
    assert!(decoded_body(&body).outcome.is_err());

    // Unknown event types stay skippable.
    let unknown = json!({ "type": "response.rocket_launch", "count": 3 });
    decoded_body(&body_of(&[unknown, completed]))
        .outcome
        .expect("unknown event types must stay skippable");
}

/// Envelope-less frames (ChatGPT's replayed bodies) are repaired and fed
/// through the same decoder as the live loop.
#[test]
fn envelope_less_frames_repair_onto_the_shared_interpreter() {
    let completed = json!({
        "type": "response.completed",
        "response": sample_response("completed"),
    });

    // A ChatGPT-style text delta with no envelope bookkeeping fields.
    let decoded = decoded_body(&body_of(&[
        json!({ "type": "response.output_text.delta", "delta": "hi" }),
        completed.clone(),
    ]));
    assert_eq!(texts_of(&decoded.events()), ["hi"]);

    // An id-less, nameless arguments delta repairs and decodes; it never
    // becomes a call.
    let decoded = decoded_body(&body_of(&[
        json!({ "type": "response.function_call_arguments.delta", "delta": "{}" }),
        completed.clone(),
    ]));
    assert!(calls_of(&decoded.outcome.expect("it decodes")).is_empty());

    // An envelope-less bookkeeping event whose data is intact (`.done`
    // events) is a no-op, not an error.
    decoded_body(&body_of(&[
        json!({ "type": "response.output_text.done", "text": "hi" }),
        completed.clone(),
    ]))
    .outcome
    .expect("an envelope-less done event must repair to the live no-op");

    // An envelope-less reasoning summary delta streams.
    let decoded = decoded_body(&body_of(&[
        json!({ "type": "response.reasoning_summary_text.delta", "delta": "think" }),
        completed,
    ]));
    assert!(decoded.events().into_iter().any(|event| matches!(
        event,
        StreamEvent::Reasoning { text, .. } if text == "think"
    )));
}

/// The `max_output_tokens`-mid-tool-call shape: `arguments_delta` frames
/// stream partial JSON and the done item restates the same truncated bytes.
/// Whether or not fragments preceded the restatement, the call keeps what
/// its arguments state, with the text they arrived as.
#[test]
fn a_cut_off_call_keeps_what_its_arguments_state() {
    let delta = json!({
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_1",
        "output_index": 0,
        "sequence_number": 1,
        "delta": "{\"x\":481",
    });
    let done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 2,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "add",
            "arguments": "{\"x\":481",
            "status": "incomplete"
        },
    });
    for events in [vec![delta, done.clone()], vec![done]] {
        let decoded = decoded_body(&body_of(&events));
        let calls: Vec<_> = decoded
            .ended()
            .into_iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some((
                    call.function.arguments_value(),
                    call.function.invalid_arguments,
                )),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            [(json!({"x": 481}), Some("{\"x\":481".to_owned()))],
            "{:?}",
            decoded.events()
        );
    }
}

/// #2258 P3: when the stream truncates before the call completes, partial
/// arguments never fabricate a call.
#[tokio::test]
async fn truncation_fabricates_no_call() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "call_id": "call_a",
                "name": "tool_a",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 2,
            "delta": "{\"loc\":"
        }),
    ];
    let decoded = decoded_body(&body_of(&events));
    assert!(matches!(decoded.outcome, Err(ProviderError::Truncated)));
    assert!(decoded.ended().is_empty(), "{:?}", decoded.events());
}

#[test]
fn refusal_content_part_frames_do_not_fail_the_buffered_body() {
    // The ChatGPT buffered route replays recorded SSE bodies; a refusal
    // turn's `content_part` frames (an unmodeled `refusal` part) must not
    // fail the whole completion — the refusal text arrives via the
    // modeled `response.refusal.delta`.
    let events = [
        json!({
            "type": "response.content_part.added",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "part": { "type": "refusal", "refusal": "" }
        }),
        json!({
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 2,
            "delta": "no"
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 3,
            "response": sample_response("completed"),
        }),
    ];
    let decoded = decoded_body(&body_of(&events));
    assert_eq!(texts_of(&decoded.events()), ["no"]);
    decoded
        .outcome
        .expect("refusal content-part frames must not fail the buffered decode");
}

/// The terminal restates the whole turn, and its message text IS the turn's
/// answer when nothing else stated it: a gateway that answers a unary call
/// with a replayed event stream can deliver a message only inside
/// `response.completed`'s `output` — no `output_text.delta`, no
/// `output_item.done` for it — so dropping that text loses the reply
/// entirely.
///
/// Driven on the plain `openai` provider: a body-only terminal is a shape
/// any Responses dialect can send, so the merge is no dialect's quirk.
#[test]
fn terminal_body_message_text_merges_when_no_delta_delivered_it() {
    let mut raw_response = sample_response("completed");
    raw_response["output"] = json!([message_output_item("msg_body_1", "from body")]);
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": raw_response,
    });
    let response = decoded_body(&body_of(&[completed]))
        .outcome
        .expect("a body-only terminal must decode");
    assert_eq!(
        choice_text_parts(&response),
        ["from body"],
        "text stated only in the terminal body must reach the choice once"
    );
    assert_eq!(
        response.choice[0]
            .native_item()
            .and_then(|item| item["id"].as_str()),
        Some("msg_body_1")
    );
}

/// The other half of that boundary: a terminal restating text the deltas
/// already delivered adds nothing.
#[test]
fn terminal_body_message_text_restating_a_delta_is_not_duplicated() {
    let text_delta = json!({
        "type": "response.output_text.delta",
        "item_id": "msg_body_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": "from body",
    });
    let mut raw_response = sample_response("completed");
    raw_response["output"] = json!([message_output_item("msg_body_1", "from body")]);
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": raw_response,
    });
    let response = decoded_body(&body_of(&[text_delta, completed]))
        .outcome
        .expect("a restating terminal must decode");
    assert_eq!(
        choice_text_parts(&response),
        ["from body"],
        "the terminal's restatement must not duplicate the delta-built text"
    );
}

#[test]
fn streaming_error_event_preserves_full_payload() {
    let payload = r#"{"type":"error","error":{"message":"boom","code":"server_error","type":"server_error"}}"#;
    let err = decoded_body(&format!("data: {payload}\n"))
        .outcome
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
async fn the_end_normalizes_the_terminal_record() {
    let mut response = sample_response("completed");
    response["usage"] = json!({ "input_tokens": 10, "output_tokens": 5, "total_tokens": 15 });

    let mut event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": response,
    });
    event["response"]["output"] = json!([{
        "type": "message",
        "id": "msg_stream_1",
        "status": "completed",
        "role": "assistant",
        "content": [{ "type": "output_text", "annotations": [], "text": "hi" }]
    }]);

    let response = stream_final_from_event(event).await;
    assert_eq!(response.provider(), "openai");
    assert_eq!(response.model(), Some("gpt-5.4"));
    assert_eq!(response.response_id(), Some("resp_123"));
    // The message item the terminal states is the turn's one block.
    assert_eq!(response.text(), "hi");
    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Stop)
    );
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.total_tokens, Some(15));
}

#[tokio::test]
async fn the_end_reports_tool_calls_when_the_stream_called_a_tool() {
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
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response("completed"),
    });
    let mut stream = responses_stream_of(&[tool_call_done, completed]).await;
    while let Some(item) = stream.next().await {
        item.expect("completed stream should not error");
    }
    // `completed` is reported as `ToolCalls`: the reply called a tool.
    assert_eq!(
        stream
            .finish()
            .await
            .expect("the reply ended")
            .finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
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

    let mut response = sample_response("completed");
    response["usage"] = json!({ "input_tokens": 4, "output_tokens": 2, "total_tokens": 6, "output_tokens_details": { "reasoning_tokens": 0 } });

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

    let mut stream = responses_stream(MockStreamingClient {
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
    .await;

    while let Some(item) = stream.next().await {
        item.expect("stream should complete successfully");
    }
    let usage = stream
        .finish()
        .await
        .expect("expected final response")
        .usage;
    assert_eq!(usage.input_tokens, Some(4));
    assert_eq!(usage.output_tokens, Some(2));
    assert_eq!(usage.total_tokens, Some(6));

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

/// A malformed frame ends the reply: the genuine terminal after it is never
/// read.
#[tokio::test]
async fn a_malformed_frame_ends_the_reply() {
    let delta = json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "hello",
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response("completed"),
    });
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            delta.to_string(),
            "{not valid json".to_string(),
            completed.to_string(),
        ]),
    };
    let mut stream = responses_stream(http_client).await;

    let mut text = String::new();
    let mut error = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text: chunk, .. })) => text.push_str(&chunk),
            Ok(_) => {}
            Err(err) => error = Some(err),
        }
    }
    assert_eq!(text, "hello");
    let error = error.expect("malformed frame should surface an error item");
    assert_eq!(error.kind(), ErrorKind::Json, "{error:?}");
    assert!(
        stream.finish().await.is_err(),
        "the corrupt frame ended the reply"
    );
}

/// An item id the wire left empty identifies nothing: a text delta under
/// `"item_id": ""` still streams, and a message item with `"id": ""` names
/// no message; neither panics.
#[test]
fn empty_item_ids_identify_nothing_and_do_not_panic() {
    let decoded = decoded_body(&body_of(&[
        json!({
            "type": "response.output_text.delta",
            "item_id": "",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "still text",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "message",
                "id": "",
                "role": "assistant",
                "status": "completed",
                "content": []
            },
        }),
    ]));
    assert_eq!(texts_of(&decoded.events()), ["still text"]);
}

/// The `output_item.done` restating `item` at `output_index`.
fn item_done(output_index: u64, sequence: u64, item: serde_json::Value) -> serde_json::Value {
    json!({
        "type": "response.output_item.done",
        "output_index": output_index,
        "sequence_number": sequence,
        "item": item,
    })
}

/// The terminal `response.completed` whose `output` is `output`.
fn completed_with(sequence: u64, output: serde_json::Value) -> serde_json::Value {
    let mut response = sample_response("completed");
    response["output"] = output;
    json!({
        "type": "response.completed",
        "sequence_number": sequence,
        "response": response,
    })
}

/// The OpenAI Responses wire every decode below runs.
fn wire() -> crate::providers::openai::responses_api::wire::Responses {
    OpenAIConfig::new("test-key").responses("gpt-5.4")
}

/// One frame per event.
fn frames(events: &[serde_json::Value]) -> Vec<WireFrame> {
    events
        .iter()
        .map(|event| WireFrame::Text(event.to_string()))
        .collect()
}

/// The unary body whose `output` is `output`, as its one frame.
fn whole(output: &[serde_json::Value]) -> Vec<WireFrame> {
    let mut body = sample_response("completed");
    body["output"] = json!(output);
    vec![WireFrame::Text(body.to_string())]
}

/// The response `frames` fold into in `mode`, from a request the wire's own
/// model sent.
fn decode(mode: Mode, frames: Vec<WireFrame>) -> crate::completion::CompletionResponse {
    crate::test_utils::decode_reply(
        &wire(),
        &CompletionRequest::new("hello"),
        mode,
        frames,
        serde_json::Value::Null,
    )
    .expect("the reply decodes")
}

/// `text` in two deltas, so a restatement is never one fragment.
fn halves(text: &str) -> [String; 2] {
    let cut = text
        .char_indices()
        .map(|(at, _)| at)
        .nth(text.chars().count() / 2)
        .unwrap_or(text.len());
    [text[..cut].to_owned(), text[cut..].to_owned()]
}

/// The stream a Responses endpoint sends for `output`: each item added with
/// no content, its text in deltas, then done; then `response.completed`
/// restating the output.
fn restated(output: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut body = sample_response("completed");
    body["output"] = json!(output);
    restated_body(&body)
}

/// [`restated`] for a whole response `body`, which the stream's terminal
/// restates.
fn restated_body(body: &serde_json::Value) -> Vec<serde_json::Value> {
    let mut created = body.clone();
    created["status"] = json!("in_progress");
    created["output"] = json!([]);
    let mut events = vec![json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": created,
    })];
    let output = body["output"].as_array().cloned().unwrap_or_default();
    for (index, item) in output.iter().enumerate() {
        let mut added = item.clone();
        match item["type"].as_str() {
            Some("message") => {
                added["content"] = json!([]);
                added["status"] = json!("in_progress");
            }
            Some("reasoning") => {
                added["summary"] = json!([]);
                if let Some(added) = added.as_object_mut() {
                    added.shift_remove("content");
                    added.shift_remove("encrypted_content");
                }
            }
            Some("function_call") => added["arguments"] = json!(""),
            _ => {}
        }
        events.push(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "sequence_number": events.len(),
            "item": added,
        }));
        let mut delta = |kind: &str, part: (&str, usize), text: &str| {
            for half in halves(text) {
                events.push(json!({
                    "type": kind,
                    "item_id": item["id"],
                    "output_index": index,
                    part.0: part.1,
                    "sequence_number": events.len(),
                    "delta": half,
                }));
            }
        };
        match item["type"].as_str() {
            Some("message") => {
                for (at, part) in item["content"].as_array().into_iter().flatten().enumerate() {
                    match part["type"].as_str() {
                        Some("refusal") => delta(
                            "response.refusal.delta",
                            ("content_index", at),
                            part["refusal"].as_str().unwrap_or_default(),
                        ),
                        _ => delta(
                            "response.output_text.delta",
                            ("content_index", at),
                            part["text"].as_str().unwrap_or_default(),
                        ),
                    }
                }
            }
            Some("reasoning") => {
                for (at, part) in item["summary"].as_array().into_iter().flatten().enumerate() {
                    delta(
                        "response.reasoning_summary_text.delta",
                        ("summary_index", at),
                        part["text"].as_str().unwrap_or_default(),
                    );
                }
            }
            Some("function_call") => delta(
                "response.function_call_arguments.delta",
                ("content_index", 0),
                item["arguments"].as_str().unwrap_or_default(),
            ),
            _ => {}
        }
        events.push(item_done(index as u64, events.len() as u64, item.clone()));
    }
    events.push(json!({
        "type": "response.completed",
        "sequence_number": events.len(),
        "response": body,
    }));
    events
}

fn message(id: &str, text: &str) -> serde_json::Value {
    json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "content": [{ "type": "output_text", "annotations": [], "logprobs": [], "text": text }],
    })
}

fn reasoning(id: &str, summaries: &[&str]) -> serde_json::Value {
    json!({
        "type": "reasoning",
        "id": id,
        "summary": summaries
            .iter()
            .map(|text| json!({ "type": "summary_text", "text": text }))
            .collect::<Vec<_>>(),
        "encrypted_content": format!("ciphertext-of-{id}"),
    })
}

fn function_call(id: &str, call_id: &str, arguments: &str) -> serde_json::Value {
    json!({
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": "lookup",
        "arguments": arguments,
        "status": "completed",
    })
}

/// A reply carrying every kind of output item the wire distinguishes, and
/// a kind and a field no version of rig has seen.
fn every_kind() -> Vec<serde_json::Value> {
    vec![
        reasoning("rs_1", &["Planning.", "Checking twice."]),
        message("msg_1", "Looking it up."),
        function_call("fc_1", "call_1", r#"{"q":"rig"}"#),
        json!({
            "type": "custom_tool_call",
            "id": "ctc_1",
            "call_id": "call_2",
            "name": "apply_patch",
            "input": "*** Begin Patch",
        }),
        json!({
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": { "type": "search", "query": "rig" },
        }),
        json!({ "type": "compaction", "id": "cmp_1", "encrypted_content": "compacted" }),
        json!({
            "type": "computer_call",
            "id": "cu_1",
            "call_id": "call_3",
            "action": { "type": "click", "x": 1, "y": 2 },
            "status": "completed",
        }),
        json!({
            "type": "message",
            "id": "msg_2",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "future_field": { "nested": [1, 2] },
            "content": [
                { "type": "output_text", "annotations": [{ "type": "url_citation", "url": "https://rig.rs" }], "text": "Rig is " },
                { "type": "refusal", "refusal": "all I can say." },
            ],
        }),
        json!({ "type": "future_item", "id": "fut_1", "payload": { "kept": true } }),
    ]
}

/// The provider item each block holds: its native, or an opaque block's
/// item.
fn natives(response: &crate::completion::CompletionResponse) -> Vec<serde_json::Value> {
    response
        .choice
        .iter()
        .map(|block| match block {
            AssistantContent::Opaque(opaque) => opaque.item.clone(),
            block => block.native_item().cloned().unwrap_or_default(),
        })
        .collect()
}

#[test]
fn each_output_item_is_one_block_holding_the_item_in_output_order() {
    let output = vec![
        reasoning("rs_1", &["First thought."]),
        message("msg_1", "Let me check."),
        function_call("fc_1", "call_1", r#"{"q":"a"}"#),
        reasoning("rs_2", &["Second thought."]),
        message("msg_2", "Done."),
    ];
    for response in [
        decode(Mode::Streaming, frames(&restated(&output))),
        decode(Mode::Unary, whole(&output)),
    ] {
        assert_eq!(
            natives(&response),
            output,
            "one block per item, nothing hoisted"
        );
        let [
            AssistantContent::Reasoning(first),
            AssistantContent::Text(check),
            AssistantContent::ToolCall(call),
            AssistantContent::Reasoning(second),
            AssistantContent::Text(done),
        ] = response.choice.as_slice()
        else {
            panic!("blocks follow the output: {:?}", response.choice);
        };
        assert_eq!(first.text, "First thought.");
        assert_eq!(check.text, "Let me check.");
        assert_eq!(call.id, crate::message::CallId::from_wire("call_1"));
        assert_eq!(call.function.arguments_value(), json!({ "q": "a" }));
        assert_eq!(second.text, "Second thought.");
        assert_eq!(done.text, "Done.");
        assert_eq!(response.stop(), crate::message::StopReason::ToolUse);
    }
}

#[test]
fn a_whole_reply_and_its_restatement_as_a_stream_agree() {
    let replies = [
        every_kind(),
        vec![message("msg_1", "Just text.")],
        vec![
            reasoning("rs_1", &[]),
            function_call("fc_1", "call_1", ""),
            function_call("fc_2", "call_2", r#"{"nested":{"deep":[1,2]}}"#),
        ],
    ];
    for output in replies {
        crate::test_utils::history::assert_restated_agrees(
            &wire(),
            whole(&output),
            frames(&restated(&output)),
        );
    }
}

/// The block each item type becomes, numbered without a wildcard: a new
/// kind fails the test until a sample decodes to it.
#[deny(clippy::wildcard_enum_match_arm)]
fn variant_index(block: &AssistantContent) -> usize {
    match block {
        AssistantContent::Text(_) => 0,
        AssistantContent::ToolCall(_) => 1,
        AssistantContent::Reasoning(_) => 2,
        AssistantContent::Opaque(_) => 3,
        AssistantContent::Image(_) => 4,
    }
}

#[test]
fn every_output_variant_decodes_to_a_block() {
    let output = every_kind();
    for response in [
        decode(Mode::Streaming, frames(&restated(&output))),
        decode(Mode::Unary, whole(&output)),
    ] {
        // Responses has no image block: hosted image generation is opaque.
        crate::test_utils::history::assert_every_variant(&response.choice, variant_index, 4);
        assert_eq!(natives(&response), output);
    }
}

#[test]
fn an_invented_item_and_an_invented_field_replay_to_the_same_model() {
    use crate::message::Message;
    let output = every_kind();
    for mode in [Mode::Streaming, Mode::Unary] {
        let frames = match mode {
            Mode::Streaming => frames(&restated(&output)),
            Mode::Unary => whole(&output),
        };
        let reply = decode(mode, frames);
        let mut history = vec![Message::user("hello")];
        history.extend(reply.message());
        history.push(Message::User {
            content: reply
                .tool_calls()
                .map(|call| {
                    crate::message::UserContent::ToolResult(
                        call.result(vec![crate::message::ToolResultContent::text("ok")]),
                    )
                })
                .collect(),
        });
        let input = encoded_input(&wire(), CompletionRequest::from(history));
        let replayed: Vec<&serde_json::Value> = input
            .iter()
            .filter(|item| item.get("id").is_some())
            .collect();
        // Everything but the client-executed computer call goes back as it
        // came, invented kind and field included.
        let expected: Vec<&serde_json::Value> = output
            .iter()
            .filter(|item| item["type"] != "computer_call")
            .collect();
        assert_eq!(replayed, expected, "{mode:?}");
        let results: Vec<(&str, &str)> = input
            .iter()
            .filter_map(|item| Some((item["type"].as_str()?, item["call_id"].as_str()?)))
            .filter(|(kind, _)| kind.ends_with("_output"))
            .collect();
        assert_eq!(
            results,
            [
                ("function_call_output", "call_1"),
                ("custom_tool_call_output", "call_2"),
            ]
        );
    }
}

/// The `input` a request sends through `wire`, history shaped by the
/// driver's own `prepare`.
fn encoded_input(
    wire: &crate::providers::openai::responses_api::wire::Responses,
    request: CompletionRequest,
) -> Vec<serde_json::Value> {
    use crate::wire::{Operation, Wire};
    // The request declares the tools its history calls, so the calls stay
    // calls.
    let mut request = request;
    let calls = request
        .chat_history
        .iter()
        .filter_map(|message| match message {
            crate::message::Message::Assistant(turn) => Some(turn.tool_calls()),
            _ => None,
        });
    let mut tools: Vec<crate::completion::ToolDefinition> = Vec::new();
    for call in calls.flatten() {
        if !tools.iter().any(|tool| tool.name == call.function.name) {
            tools.push(crate::completion::ToolDefinition {
                name: call.function.name.clone(),
                description: "A tool".to_owned(),
                parameters: json!({"type": "object"}),
            });
        }
    }
    request.tools.extend(tools);
    let request = Completion::prepare(request, &wire.describe()).expect("the history is valid");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let body = crate::test_utils::json_body(&encoded.request);
    body["input"].as_array().cloned().unwrap_or_default()
}

#[test]
fn hosted_steps_and_compaction_replay_while_client_executed_calls_stay_home() {
    let output = vec![
        json!({ "type": "web_search_call", "id": "ws_1", "status": "completed" }),
        json!({ "type": "compaction", "id": "cmp_1", "encrypted_content": "x" }),
        json!({ "type": "mcp_call", "id": "mcp_1", "name": "f", "server_label": "s" }),
        json!({ "type": "computer_call", "id": "cu_1", "call_id": "c1", "status": "completed" }),
        json!({ "type": "local_shell_call", "id": "ls_1", "call_id": "c2", "status": "completed" }),
        json!({ "type": "mcp_approval_request", "id": "ar_1", "name": "f", "server_label": "s" }),
    ];
    let response = decode(Mode::Unary, whole(&output));
    let replays: Vec<(Option<&str>, bool)> = response
        .choice
        .iter()
        .map(|block| match block {
            AssistantContent::Opaque(opaque) => (opaque.kind(), opaque.replay),
            other => panic!("every item is opaque: {other:?}"),
        })
        .collect();
    assert_eq!(
        replays,
        [
            (Some("web_search_call"), true),
            (Some("compaction"), true),
            (Some("mcp_call"), true),
            (Some("computer_call"), false),
            (Some("local_shell_call"), false),
            (Some("mcp_approval_request"), false),
        ]
    );
}

#[test]
fn a_refusal_is_the_message_text_and_its_item_keeps_the_part() {
    let refusal = json!({
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{ "type": "refusal", "refusal": "I can't help with that." }],
    });
    for response in [
        decode(
            Mode::Streaming,
            frames(&restated(std::slice::from_ref(&refusal))),
        ),
        decode(Mode::Unary, whole(std::slice::from_ref(&refusal))),
    ] {
        assert_eq!(response.text(), "I can't help with that.");
        assert_eq!(natives(&response), std::slice::from_ref(&refusal));
        assert_eq!(response.stop(), crate::message::StopReason::Stop);
    }
}

#[test]
fn a_custom_tool_call_is_a_call_whose_arguments_hold_its_input() {
    let call = json!({
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_9",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
    });
    let response = decode(Mode::Unary, whole(std::slice::from_ref(&call)));
    let [AssistantContent::ToolCall(decoded)] = response.choice.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(decoded.id, crate::message::CallId::from_wire("call_9"));
    assert_eq!(decoded.function.name, "apply_patch");
    assert_eq!(
        decoded.function.arguments_value(),
        json!({ "input": "*** Begin Patch\n*** End Patch" })
    );
    assert_eq!(natives(&response), [call]);
}

/// Azure states a reasoning item's `encrypted_content` only in the terminal
/// response: the block waits for it, so a stateless follow-up can replay
/// the reasoning.
#[test]
fn reasoning_takes_the_ciphertext_the_terminal_states() {
    let item = reasoning("rs_1", &["Thinking."]);
    let mut without = item.clone();
    if let Some(without) = without.as_object_mut() {
        without.shift_remove("encrypted_content");
    }
    let mut events = restated(std::slice::from_ref(&item));
    for event in &mut events {
        if event["type"] == "response.output_item.done" {
            event["item"] = without.clone();
        }
    }
    let response = decode(Mode::Streaming, frames(&events));
    assert_eq!(natives(&response), std::slice::from_ref(&item));
    assert_eq!(
        response.message(),
        decode(Mode::Unary, whole(&[item])).message(),
        "the stream and the body it ends with fold into one turn"
    );
}

#[test]
fn reasoning_parts_are_paragraphs_of_one_block() {
    let item = reasoning("rs_1", &["First part.", "Second part."]);
    for response in [
        decode(
            Mode::Streaming,
            frames(&restated(std::slice::from_ref(&item))),
        ),
        decode(Mode::Unary, whole(std::slice::from_ref(&item))),
    ] {
        assert_eq!(response.reasoning(), "First part.\n\nSecond part.");
        assert_eq!(response.choice.len(), 1);
    }
}

#[test]
fn an_incomplete_turn_reports_why_unless_the_token_limit_cut_it() {
    for (reason, stop) in [
        ("max_output_tokens", crate::message::StopReason::Length),
        (
            "content_filter",
            crate::message::StopReason::Error("Provider finish_reason: content_filter".into()),
        ),
        (
            "max_tool_calls",
            crate::message::StopReason::Error("Response incomplete: max_tool_calls".into()),
        ),
    ] {
        let mut body = sample_response("incomplete");
        body["incomplete_details"] = json!({ "reason": reason });
        body["output"] = json!([message("msg_1", "partial")]);
        let response = decode(Mode::Unary, vec![WireFrame::Text(body.to_string())]);
        assert_eq!(response.stop(), stop, "{reason}");
        assert_eq!(response.text(), "partial");
    }
}

#[test]
fn a_reply_names_the_model_and_response_it_came_from() {
    let response = decode(Mode::Unary, whole(&[message("msg_1", "hi")]));
    let Some(crate::message::Message::Assistant(turn)) = response.message() else {
        panic!("one assistant turn");
    };
    let origin = turn.origin.expect("a decoded turn has an origin");
    assert_eq!(origin.api.as_str(), "openai.responses");
    assert_eq!(origin.provider, "openai");
    assert_eq!(origin.model, "gpt-5.4");
    assert_eq!(origin.response_id.as_deref(), Some("resp_123"));
}

/// The whole Responses replies a provider's cassettes record: each unary
/// body, and the response object each recorded stream's terminal restates.
fn recorded_whole_replies(provider: &str) -> Vec<serde_json::Value> {
    let mut replies = Vec::new();
    let mut stack = vec![
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../rig-cassette/fixtures/cassettes")
            .join(provider),
    ];
    while let Some(dir) = stack.pop() {
        for path in std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
        {
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for reply in text.split("\nthen:").skip(1) {
                let Some(body) = reply.split_once("  body: ").map(|(_, body)| body) else {
                    continue;
                };
                let documents: Vec<serde_json::Value> = match body.strip_prefix("|+\n") {
                    Some(block) => block
                        .lines()
                        .take_while(|line| line.starts_with("    ") || line.trim().is_empty())
                        .filter_map(|line| line.trim().strip_prefix("data:"))
                        .filter_map(|data| {
                            serde_json::from_str::<serde_json::Value>(data.trim()).ok()
                        })
                        .filter(|event| event["type"] == "response.completed")
                        .map(|event| event["response"].clone())
                        .collect(),
                    None => body
                        .lines()
                        .next()
                        .map(|line| line.trim().trim_matches('\'').replace("''", "'"))
                        .and_then(|json| serde_json::from_str::<serde_json::Value>(&json).ok())
                        .into_iter()
                        .collect(),
                };
                replies.extend(documents.into_iter().filter(|document| {
                    document["object"] == "response"
                        && document["status"] == "completed"
                        && document["output"].is_array()
                }));
            }
        }
    }
    replies
}

/// Every whole reply the family's cassettes record folds into the same turn
/// restated as a stream, through the shared restate-as-stream harness.
#[test]
fn every_recorded_whole_reply_agrees_with_its_restatement_as_a_stream() {
    let mut checked = 0;
    for (provider, wire) in [
        ("openai", wire()),
        ("xai", wire()),
        ("copilot", wire()),
        (
            "chatgpt",
            OpenAIConfig::with_key(&crate::providers::chatgpt::DIALECT, "token")
                .responses("gpt-5.4"),
        ),
    ] {
        for body in recorded_whole_replies(provider) {
            crate::test_utils::history::assert_restated_agrees(
                &wire,
                [WireFrame::Text(body.to_string())],
                frames(&restated_body(&body)),
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "the family records whole replies: {checked}");
}

/// xAI gives every reasoning item of a reply the same id; each stays its
/// own block in both modes.
#[test]
fn items_sharing_an_id_stay_distinct_blocks() {
    let output = vec![
        reasoning("rs_1", &["First thought."]),
        message("msg_1", "Searching."),
        reasoning("rs_1", &["Second thought."]),
        message("msg_2", "Done."),
    ];
    for response in [
        decode(Mode::Streaming, frames(&restated(&output))),
        decode(Mode::Unary, whole(&output)),
    ] {
        assert_eq!(natives(&response), output);
    }
}

/// A failed or cancelled reply ends the turn in an error, so it never
/// replays.
#[test]
fn a_failed_or_cancelled_reply_ends_in_an_error() {
    for status in ["failed", "cancelled"] {
        let mut body = sample_response(status);
        body["output"] = json!([message("msg_1", "partial")]);
        let decoded = crate::test_utils::decode_reply(
            &wire(),
            &CompletionRequest::new("hello"),
            Mode::Unary,
            vec![WireFrame::Text(body.to_string())],
            serde_json::Value::Null,
        );
        let response = decoded.expect("the reply decodes");
        assert!(
            matches!(response.stop(), crate::message::StopReason::Error(_)),
            "{:?}",
            response.stop()
        );
    }
}

/// A content part, summary part or call status of a kind rig does not
/// model decodes and keeps its item, in both modes.
#[test]
fn new_nested_kinds_do_not_fail_the_reply() {
    let output = vec![
        json!({"type": "reasoning", "id": "rs_1", "summary": [
            {"type": "summary_text", "text": "Think."},
            {"type": "summary_image", "url": "u"},
        ]}),
        json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [
            {"type": "output_text", "text": "Done.", "annotations": []},
            {"type": "output_audio", "transcript": "Done."},
        ]}),
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
            "arguments": "{}", "status": "searching"}),
    ];
    for response in [
        decode(Mode::Streaming, frames(&restated(&output))),
        decode(Mode::Unary, whole(&output)),
    ] {
        assert_eq!(natives(&response), output);
    }
}

/// The history the next request sends for `response`, on the same model.
fn replayed_input(response: &crate::completion::CompletionResponse) -> Vec<serde_json::Value> {
    use crate::message::Message;
    let Some(Message::Assistant(turn)) = response.message() else {
        panic!("the reply is an assistant turn");
    };
    let mut request = CompletionRequest::new("next");
    request.chat_history = vec![
        Message::user("q"),
        Message::Assistant(turn),
        Message::user("next"),
    ];
    encoded_input(&wire(), request)
}

/// #2668: a gateway that omits `sequence_number`, `content_index` and
/// `summary_index` (LiteLLM) streams a reply on every dialect, not only
/// Codex.
#[test]
fn frames_without_bookkeeping_decode_on_every_dialect() {
    let events = [
        json!({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"}),
        json!({"type": "response.output_text.delta", "output_index": 0, "delta": "lo"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": message("msg_1", "Hello")}),
        json!({"type": "response.completed", "response": {"id": "resp_1", "status": "completed"}}),
    ];
    for dialect in [
        &crate::providers::openai::wire::OPENAI,
        &crate::providers::openai::wire::AZURE,
        &crate::providers::xai::DIALECT,
        &crate::providers::copilot::wire::DIALECT,
        &crate::providers::chatgpt::DIALECT,
    ] {
        let wire = crate::providers::openai::responses_api::wire::Responses::new(
            OpenAIConfig::with_key(dialect, "key"),
            "gpt-5.4",
        );
        let response = crate::test_utils::decode_reply(
            &wire,
            &CompletionRequest::new("hello"),
            Mode::Streaming,
            frames(&events),
            serde_json::Value::Null,
        )
        .unwrap_or_else(|error| panic!("{}: the reply decodes: {error}", dialect.name));
        assert_eq!(response.text(), "Hello", "{}", dialect.name);
    }
}

/// #1426, #1176, #1512, #2194: a body missing every field no block or the
/// finish reads, with its echoed request config retyped, still decodes.
#[test]
fn a_sparse_body_with_retyped_echoed_config_decodes() {
    let body = json!({
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
        ],
        "usage": {"input_tokens": 3, "output_tokens_details": {}},
        "tools": [{"type": "function", "name": "lookup", "description": null, "strict": null}],
        "reasoning": {"effort": null},
        "text": {"format": null},
        "instructions": ["not", "a", "string"],
    });
    let response = decode(Mode::Unary, vec![WireFrame::Text(body.to_string())]);
    assert_eq!(response.text(), "hi");
    assert_eq!(response.tool_calls().count(), 1);
    assert_eq!(response.reasoning(), "why");
    assert_eq!(response.usage.input_tokens, Some(3));
    assert_eq!(response.usage.reasoning_tokens, None);
    assert_eq!(response.stop(), crate::message::StopReason::ToolUse);
}

/// Every status the API documents, and every incomplete reason, has its own
/// finish; anything else is a failed turn.
#[test]
fn every_documented_status_maps_explicitly() {
    use crate::completion::FinishReason;
    use crate::message::StopReason;
    let incomplete =
        |reason: &str| json!({"status": "incomplete", "incomplete_details": {"reason": reason}});
    let failed = |status: &str| json!({"status": status, "error": {"code": "server_error", "message": "boom"}});
    let cases: [(serde_json::Value, Option<FinishReason>, bool); 11] = [
        (
            json!({"status": "completed"}),
            Some(FinishReason::Stop),
            false,
        ),
        (
            incomplete("max_output_tokens"),
            Some(FinishReason::Length),
            false,
        ),
        (
            incomplete("content_filter"),
            Some(FinishReason::ContentFilter),
            true,
        ),
        (
            incomplete("max_tool_calls"),
            Some(FinishReason::Other("incomplete: max_tool_calls".into())),
            true,
        ),
        (
            json!({"status": "incomplete"}),
            Some(FinishReason::Other("incomplete".into())),
            true,
        ),
        (
            failed("failed"),
            Some(FinishReason::Other("failed".into())),
            true,
        ),
        (
            failed("cancelled"),
            Some(FinishReason::Other("cancelled".into())),
            true,
        ),
        (
            json!({"status": "queued"}),
            Some(FinishReason::Other("queued".into())),
            true,
        ),
        (
            json!({"status": "in_progress"}),
            Some(FinishReason::Other("in_progress".into())),
            true,
        ),
        (
            json!({"status": "paused"}),
            Some(FinishReason::Other("paused".into())),
            true,
        ),
        (json!({}), Some(FinishReason::Stop), false),
    ];
    for (status, reason, fails) in cases {
        let mut body = status.clone();
        body["output"] = json!([message("msg_1", "Done.")]);
        let response = decode(Mode::Unary, vec![WireFrame::Text(body.to_string())]);
        assert_eq!(response.finish_reason(), reason, "{status}");
        assert_eq!(
            response.stop().is_failure(),
            fails,
            "{status}: {:?}",
            response.stop()
        );
        if status.get("error").is_some() {
            assert_eq!(
                response.stop(),
                StopReason::Error("server_error: boom".to_owned()),
                "the provider's own message"
            );
        }
    }
}

/// An item `output_item.added` announced and no done item finished keeps
/// no provider item: its snapshot is not the item, so the next request
/// rebuilds the text from its fields and sends no reasoning.
#[test]
fn an_item_never_done_replays_from_its_fields() {
    let mut incomplete = sample_response("incomplete");
    incomplete["incomplete_details"] = json!({"reason": "max_output_tokens"});
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "Thinking"}),
        json!({"type": "response.output_item.added", "output_index": 1,
               "item": {"type": "message", "id": "msg_1", "status": "in_progress", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 1, "delta": "Hello there"}),
        json!({"type": "response.output_item.added", "output_index": 2,
               "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"}}),
        json!({"type": "response.incomplete", "response": incomplete}),
    ];
    let response = decode(Mode::Streaming, frames(&events));
    assert_eq!(response.stop(), crate::message::StopReason::Length);
    assert!(
        response
            .choice
            .iter()
            .all(|block| block.native_item().is_none()),
        "no block holds a snapshot: {:?}",
        response.choice
    );
    let input = replayed_input(&response);
    let assistant: Vec<&serde_json::Value> = input
        .iter()
        .filter(|item| item.get("role").and_then(serde_json::Value::as_str) != Some("user"))
        .collect();
    assert_eq!(assistant.len(), 1, "only the text goes back: {input:?}");
    assert_eq!(assistant[0]["content"][0]["text"], "Hello there");
    assert!(
        !json!(input).to_string().contains("ws_1"),
        "an unfinished hosted item stays home"
    );
}

/// A call `output_item.added` announced and no done item finished fails
/// the turn, as pi refuses it: its arguments may be cut off, and without
/// its item it cannot replay beside the reasoning before it.
#[test]
fn a_call_added_but_never_done_fails_the_turn() {
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": ""}}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"q\":"}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "\"rig\"}"}),
        json!({"type": "response.function_call_arguments.done", "output_index": 0, "arguments": "{\"q\":\"rig\"}"}),
        completed_with(4, json!([])),
    ];
    let response = decode(Mode::Streaming, frames(&events));
    assert!(
        response.stop().is_failure(),
        "the call was never finished: {:?}",
        response.stop()
    );
    let restated = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": ""}}),
        completed_with(
            1,
            json!([function_call("fc_1", "call_1", r#"{"q":"rig"}"#)]),
        ),
    ];
    let response = decode(Mode::Streaming, frames(&restated));
    assert_eq!(response.stop(), crate::message::StopReason::ToolUse);
    assert_eq!(
        natives(&response),
        [function_call("fc_1", "call_1", r#"{"q":"rig"}"#)],
        "a call the terminal restates is finished by it"
    );
}

/// An item only the terminal response states follows the items the stream
/// carried, which keep their place.
#[test]
fn a_terminal_only_item_follows_the_streamed_items() {
    let events = [
        json!({"type": "response.output_text.delta", "output_index": 1, "delta": "Hi"}),
        completed_with(
            2,
            json!([reasoning("rs_1", &["Why."]), message("msg_1", "Hi")]),
        ),
    ];
    let response = decode(Mode::Streaming, frames(&events));
    let kinds: Vec<&str> = response
        .choice
        .iter()
        .map(|block| match block {
            AssistantContent::Reasoning(_) => "reasoning",
            AssistantContent::Text(_) => "text",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["text", "reasoning"]);
    assert_eq!(
        natives(&response),
        [message("msg_1", "Hi"), reasoning("rs_1", &["Why."])]
    );
}

/// pi's pairing rule: an edit of any one of `rs_1, msg_1, fc_1` keeps every
/// item id in the next request. An edited block is rebuilt under its
/// item's identity: reasoning keeps its id and ciphertext, text its id and
/// phase, a call its id.
#[test]
fn reasoning_message_and_call_ids_survive_an_edit_of_each_one() {
    let output = [
        reasoning("rs_1", &["Plan."]),
        json!({
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Looking.", "annotations": []}],
        }),
        function_call("fc_1", "call_1", r#"{"q":"rig"}"#),
    ];
    for edited in 0..3 {
        let mut response = decode(Mode::Unary, whole(&output));
        match response.choice.get_mut(edited) {
            Some(AssistantContent::Reasoning(reasoning)) => reasoning.text.push_str(" (edited)"),
            Some(AssistantContent::Text(text)) => text.text.push_str(" (edited)"),
            Some(AssistantContent::ToolCall(call)) => {
                call.function
                    .arguments
                    .insert("q".to_owned(), json!("edited"));
            }
            other => panic!("block {edited}: {other:?}"),
        }
        assert!(
            response.choice[edited].native_item().is_none(),
            "the edit is stale"
        );
        let input = replayed_input(&response);
        let ids: Vec<&str> = input
            .iter()
            .filter_map(|item| item.get("id").and_then(serde_json::Value::as_str))
            .collect();
        assert_eq!(
            ids,
            ["rs_1", "msg_1", "fc_1"],
            "edit of block {edited}: {input:?}"
        );
        let items: Vec<&serde_json::Value> = input
            .iter()
            .filter(|item| item.get("id").is_some())
            .collect();
        assert_eq!(items[1]["phase"], "commentary", "edit of block {edited}");
        if edited == 0 {
            assert_eq!(
                items[0]["encrypted_content"],
                output[0]["encrypted_content"]
            );
            assert_eq!(items[0]["summary"][0]["text"], "Plan. (edited)");
        }
    }
}

/// `events` as a gateway that names no `output_index` streams them.
fn without_indices(mut events: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    for event in &mut events {
        if let Some(event) = event.as_object_mut() {
            event.shift_remove("output_index");
        }
    }
    events
}

/// The ids of the items the next request sends back.
fn replayed_ids(response: &crate::completion::CompletionResponse) -> Vec<serde_json::Value> {
    replayed_input(response)
        .iter()
        .filter_map(|item| item.get("id").cloned())
        .collect()
}

/// A stream that names no output index folds into the turn its whole
/// reply does, every item in its place.
#[test]
fn a_stream_without_output_indices_agrees_with_its_whole_reply() {
    let replies = [
        every_kind(),
        vec![
            reasoning("rs_1", &["First."]),
            message("msg_1", "Between."),
            reasoning("rs_2", &["Second."]),
            function_call("fc_1", "call_1", r#"{"q":"rig"}"#),
        ],
    ];
    for output in replies {
        crate::test_utils::history::assert_restated_agrees(
            &wire(),
            whole(&output),
            frames(&without_indices(restated(&output))),
        );
    }
}

/// Reasoning done without its ciphertext (`store: true`) keeps its item
/// when the next item opens in a stream that names no output index, so the
/// message after it replays with it.
#[test]
fn reasoning_done_without_ciphertext_keeps_its_item_in_a_stream_without_indices() {
    let mut thought = reasoning("rs_1", &["Plan."]);
    if let Some(fields) = thought.as_object_mut() {
        fields.shift_remove("encrypted_content");
    }
    let output = [thought.clone(), message("msg_1", "Hello")];
    let response = decode(Mode::Streaming, frames(&without_indices(restated(&output))));
    assert_eq!(natives(&response), output);
    assert_eq!(replayed_ids(&response), [json!("rs_1"), json!("msg_1")]);
}

/// Text streamed with no index and no item events, then a terminal that
/// states reasoning before that message: the text is said once, and both
/// blocks hold the provider's items in the order they arrived.
#[test]
fn a_terminal_only_item_does_not_repeat_text_streamed_without_indices() {
    let events = [
        json!({"type": "response.output_text.delta", "delta": "Hel"}),
        json!({"type": "response.output_text.delta", "delta": "lo"}),
        completed_with(
            2,
            json!([reasoning("rs_1", &["Why."]), message("msg_1", "Hello")]),
        ),
    ];
    let response = decode(Mode::Streaming, frames(&events));
    assert_eq!(response.text(), "Hello");
    assert_eq!(
        natives(&response),
        [message("msg_1", "Hello"), reasoning("rs_1", &["Why."])]
    );
}

/// A done item that agrees with what streamed completes it; one that
/// contradicts it cannot be its native, so the streamed text stands and the
/// turn replays canonically from that block on.
#[test]
fn a_done_item_that_contradicts_its_deltas_is_not_kept() {
    let output = [reasoning("rs_1", &["Plan B."]), message("msg_1", "Hello")];
    let mut events = restated(&output);
    for event in &mut events {
        match event["type"].as_str() {
            Some("response.output_text.delta") => event["delta"] = json!("Bye"),
            Some("response.reasoning_summary_text.delta") => event["delta"] = json!("Plan A"),
            _ => {}
        }
    }
    let response = decode(Mode::Streaming, frames(&events));
    assert_eq!(response.text(), "ByeBye");
    assert_eq!(response.reasoning(), "Plan APlan A");
    assert!(
        response
            .choice
            .iter()
            .all(|block| block.native_item().is_none()),
        "{:?}",
        response.choice
    );
}

/// An item that names no `type` is kept in history and never sent back,
/// as pi drops it.
#[test]
fn an_item_without_a_type_never_replays() {
    let output = [
        json!({"id": "x_1", "payload": {"n": 1}}),
        message("msg_1", "Noted."),
    ];
    for response in [
        decode(Mode::Unary, whole(&output)),
        decode(Mode::Streaming, frames(&restated(&output))),
    ] {
        assert!(
            matches!(response.choice.first(), Some(AssistantContent::Opaque(opaque)) if !opaque.replay),
            "{:?}",
            response.choice
        );
        assert_eq!(replayed_ids(&response), [json!("msg_1")]);
    }
}

/// `response.incomplete` is an incomplete turn whatever its `status` says:
/// the output cap is `Length`, any other reason or none fails the turn.
#[test]
fn response_incomplete_is_incomplete_whatever_its_status_says() {
    use crate::message::StopReason;
    for (status, reason, stop) in [
        (None, Some("max_output_tokens"), Some(StopReason::Length)),
        (
            Some("completed"),
            Some("max_output_tokens"),
            Some(StopReason::Length),
        ),
        (None, None, None),
        (Some("completed"), None, None),
        (None, Some("content_filter"), None),
    ] {
        let mut response = sample_response("incomplete");
        match status {
            Some(status) => response["status"] = json!(status),
            None => {
                if let Some(fields) = response.as_object_mut() {
                    fields.shift_remove("status");
                }
            }
        }
        response["incomplete_details"] = json!(reason.map(|reason| json!({"reason": reason})));
        response["output"] = json!([message("msg_1", "partial")]);
        let events = [
            item_done(0, 1, message("msg_1", "partial")),
            json!({"type": "response.incomplete", "sequence_number": 2, "response": response}),
        ];
        let response = decode(Mode::Streaming, frames(&events));
        match stop {
            Some(stop) => assert_eq!(response.stop(), stop, "{status:?} {reason:?}"),
            None => assert!(
                response.stop().is_failure(),
                "{status:?} {reason:?}: {:?}",
                response.stop()
            ),
        }
    }
}

/// A gateway that names the output index on its item events but not on
/// its deltas streams each delta into the item it continues.
#[test]
fn deltas_without_an_index_continue_the_item_they_name() {
    let output = [reasoning("rs_1", &["Plan."]), message("msg_1", "Hello")];
    let mut events = restated(&output);
    for event in &mut events {
        if event["type"]
            .as_str()
            .is_some_and(|kind| kind.ends_with(".delta"))
            && let Some(fields) = event.as_object_mut()
        {
            fields.shift_remove("output_index");
        }
    }
    let response = decode(Mode::Streaming, frames(&events));
    assert_eq!(natives(&response), output);
    assert_eq!(response.text(), "Hello");
    assert_eq!(response.reasoning(), "Plan.");
}
