use super::{
    ContentPartChunkPart, ItemChunkKind, ResponsesDecoder, StreamingCompletionChunk,
    classify_responses_frame,
};
use crate::completion::CompletionRequest;
use crate::driver::{Decoded, feed_frames};
use crate::error::{ErrorKind, ErrorReport, ProviderError};
use crate::message::AssistantContent;
use crate::operation::Completion;
use crate::providers::internal::openai_chat_completions_compatible::test_support::{
    sse_bytes_from_data_lines, sse_bytes_from_json_events,
};
use crate::providers::openai::OpenAIConfig;
use crate::providers::openai::responses_api::{
    AdditionalParameters, CompletionResponse, IncompleteDetailsReason, OutputTokensDetails,
    ResponseError, ResponseObject, ResponseStatus, ResponsesUsage,
};
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
        classify_responses_frame(&frame),
        WireEvent::Known(StreamingCompletionChunk::Delta(_))
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
        classify_responses_frame(&frame),
        WireEvent::Unknown { event_type, .. } if event_type == "response.web_search_call.searching"
    ));
}

/// #2258 G4: `response.reasoning_text.done` terminates every raw-reasoning
/// block on all three Responses surfaces. It used to be absent from the
/// known-event set, so each block logged a spurious "unknown event" warn
/// and passed through as `Unknown`.
///
/// Both halves of the fix are asserted here, because either alone is a
/// regression: the tag must be KNOWN (no `Unknown`), and `ItemChunkKind`
/// must carry a variant for it (no `Corrupt`, which is what naming the tag
/// without the variant would have produced — strictly worse than the warn).
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

    let event = classify_responses_frame(&frame);
    assert!(
        !matches!(event, WireEvent::Unknown { .. }),
        "the tag must be in the known-event set: {event:?}"
    );
    assert!(
        !matches!(event, WireEvent::Corrupt(_)),
        "a known tag with no matching ItemChunkKind variant decodes to Corrupt, which the \
             driver surfaces as an in-band Err — worse than the warn it replaced: {event:?}"
    );
    assert!(matches!(
        event,
        WireEvent::Known(StreamingCompletionChunk::Delta(chunk))
            if matches!(chunk.data, ItemChunkKind::ReasoningTextDone(_))
    ));
}

#[test]
fn classify_invalid_json_is_corrupt() {
    assert!(matches!(
        classify_responses_frame("{not json"),
        WireEvent::Corrupt(_)
    ));
}

#[test]
fn classify_known_event_with_defective_payload_is_corrupt() {
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
        classify_responses_frame(&frame),
        WireEvent::Corrupt(_)
    ));
}

// The P2 probe shape from `rig-2257-code-review-findings-34ee8ba5.md`: a
// known part tag whose payload is schema-defective must classify as
// `Corrupt`, not slide into the unknown-part catch-all.
#[test]
fn classify_defective_known_content_part_is_corrupt() {
    let frame = json!({
        "type": "response.content_part.added",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "part": {"type": "output_text", "text": 42},
    })
    .to_string();
    assert!(matches!(
        classify_responses_frame(&frame),
        WireEvent::Corrupt(_)
    ));
}

#[test]
fn content_part_known_tag_decodes() {
    let part: ContentPartChunkPart =
        serde_json::from_value(json!({"type": "output_text", "text": "hi"})).unwrap();
    assert!(matches!(part, ContentPartChunkPart::OutputText { text } if text == "hi"));
}

#[test]
fn content_part_known_tag_with_defective_payload_errors() {
    let result =
        serde_json::from_value::<ContentPartChunkPart>(json!({"type": "output_text", "text": 42}));
    assert!(result.is_err());
    let result =
        serde_json::from_value::<ContentPartChunkPart>(json!({"type": "summary_text", "text": 42}));
    assert!(result.is_err());
}

// A non-string `type` is a data-level defect of the tagged shape, never a
// skippable unknown part (#2258 F8).
#[test]
fn content_part_non_string_type_errors() {
    let result = serde_json::from_value::<ContentPartChunkPart>(json!({"type": 42, "text": "hi"}));
    assert!(result.is_err());
    let result =
        serde_json::from_value::<ContentPartChunkPart>(json!({"type": null, "text": "hi"}));
    assert!(result.is_err());
}

// Pins the documented duplicate-key edge (#2258 F8): `serde_json::Value`
// keeps the last duplicate key, so the hand dispatch resolves on the LAST
// `type` — unlike a derived internally-tagged enum, which takes the first.
#[test]
fn content_part_duplicate_type_key_dispatches_on_the_last_occurrence() {
    let part: ContentPartChunkPart =
        serde_json::from_str(r#"{"type":"bogus","type":"output_text","text":"hi"}"#).unwrap();
    assert!(matches!(part, ContentPartChunkPart::OutputText { text } if text == "hi"));
}

// `refusal` and `reasoning_text` part tags are not in the modeled set:
// they must stay skippable no-ops (the content arrives via the
// corresponding delta events), round-tripping the value verbatim.
#[test]
fn content_part_unknown_tag_is_preserved_verbatim() {
    let wire = json!({"type": "refusal", "refusal": "no"});
    let part: ContentPartChunkPart = serde_json::from_value(wire.clone()).unwrap();
    let ContentPartChunkPart::Unknown(value) = &part else {
        panic!("unmodeled part tag must fall back to Unknown");
    };
    assert_eq!(value, &wire);
    assert_eq!(serde_json::to_value(&part).unwrap(), wire);
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
        provider_reasoning: None,
        reasoning_metadata: None,
        reasoning_context: None,
        usage: None,
        output: Vec::new(),
        tools: Vec::new(),
        additional_parameters: AdditionalParameters::default(),
    }
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

    assert_eq!(err.kind, ErrorKind::ProviderResponse);
    assert_eq!(err.http_status, None);
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

    let usage = final_response_from_event(event)
        .await
        .usage
        .expect("the terminal carries usage");
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

/// The terminal `response.completed` frame carries usage. An object-shaped
/// `top_p` echoed on that frame (MiniMax-style endpoints, rig#2483) or a
/// numeric one buffered under `serde_json/arbitrary_precision` (rig#2493)
/// must not turn the terminal into a parse error — that both fails the turn
/// and loses the usage. Contrast `known_terminal_with_malformed_usage_*`:
/// a defect in a field rig reads is still an error.
#[tokio::test]
async fn response_completed_chunk_tolerates_object_shaped_top_p() {
    let mut response = serde_json::to_value(sample_response(ResponseStatus::Completed))
        .expect("sample response serializes");
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

    let usage = final_response_from_event(event)
        .await
        .usage
        .expect("the terminal carries usage");
    assert_eq!(usage.input_tokens, 10);
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

    let response = final_response_from_event(event).await;
    assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
    assert_eq!(response.reasoning_metadata.as_ref(), metadata.as_object());
}

/// One `message` output item, as a terminal response body states it.
fn message_output_item(id: &str, text: &str) -> crate::providers::openai::responses_api::Output {
    serde_json::from_value(json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "content": [{ "type": "output_text", "annotations": [], "text": text }],
    }))
    .expect("output message should deserialize")
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

/// Decode a Responses SSE body — its `data:` lines — through the decoder a
/// buffered replay runs: envelope repair on.
fn decoded_body(body: &str) -> Decoded<Completion> {
    let frames: Vec<WireFrame> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim))
        .filter(|data| !data.is_empty() && *data != "[DONE]")
        .map(|data| WireFrame::Text(data.to_owned()))
        .collect();
    feed_frames!(
        ResponsesDecoder::new().with_envelope_repair(),
        "openai",
        frames
    )
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

/// The provider's own response object a stream of `event` carries on its
/// response's `raw`: the terminal event's document.
async fn final_response_from_event(event: serde_json::Value) -> CompletionResponse {
    serde_json::from_value(stream_final_from_event(event).await.raw)
        .expect("the raw document is the provider's own response object")
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

    let mut response = sample_response(ResponseStatus::Incomplete);
    response.incomplete_details = Some(IncompleteDetailsReason {
        reason: "max_output_tokens".to_string(),
    });
    response.usage = Some(ResponsesUsage {
        input_tokens: 10,
        input_tokens_details: None,
        output_tokens: 5,
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: 0,
        }),
        total_tokens: 15,
    });

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

/// A known terminal event with a data-level defect (malformed `usage`) is
/// a corrupt frame, not silent truncation: the error ends the reply.
#[tokio::test]
async fn known_terminal_with_malformed_usage_surfaces_error_without_an_end() {
    let mut event = json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": sample_response(ResponseStatus::Completed),
    });
    event["response"]["usage"] = json!("banana");

    let mut stream = responses_stream_of(&[event]).await;
    let items: Vec<_> = (&mut stream).collect().await;
    let [Err(err)] = items.as_slice() else {
        panic!("the corrupt terminal is the only item: {items:?}");
    };
    assert_eq!(err.kind(), ErrorKind::Json, "{err:?}");
    assert!(stream.finish().await.is_err());
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
        "response": sample_response(ResponseStatus::Completed),
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
            "response": sample_response(ResponseStatus::Completed),
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
fn corrupt_known_frame_fails_the_buffered_body() {
    let corrupt = json!({
        "type": "response.output_text.delta",
        "delta": 42
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response(ResponseStatus::Completed),
    });
    let err = decoded_body(&body_of(&[corrupt, completed.clone()]))
        .outcome
        .expect_err("a corrupt known frame must fail the buffered decode");
    assert!(
        err.to_string().contains("response.output_text.delta"),
        "the error should name the malformed event, got: {err}"
    );

    // Syntactically invalid JSON fails too.
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
        "response": sample_response(ResponseStatus::Completed),
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
            "response": sample_response(ResponseStatus::Completed),
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
    let mut raw_response = sample_response(ResponseStatus::Completed);
    raw_response.output = vec![message_output_item("msg_body_1", "from body")];
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
    let mut raw_response = sample_response(ResponseStatus::Completed);
    raw_response.output = vec![message_output_item("msg_body_1", "from body")];
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
    let mut response = sample_response(ResponseStatus::Completed);
    response.usage = Some(ResponsesUsage {
        input_tokens: 10,
        input_tokens_details: None,
        output_tokens: 5,
        output_tokens_details: None,
        total_tokens: 15,
    });

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
        "response": sample_response(ResponseStatus::Completed),
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
        "response": sample_response(ResponseStatus::Completed),
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
    let mut response = serde_json::to_value(sample_response(ResponseStatus::Completed))
        .expect("the sample response serializes");
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
    let mut body = serde_json::to_value(sample_response(ResponseStatus::Completed))
        .expect("the sample response serializes");
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
    let mut body = serde_json::to_value(sample_response(ResponseStatus::Completed))
        .expect("the sample response serializes");
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

/// The wire's output variants, numbered without a wildcard: a new variant
/// fails to compile here until it is numbered, and fails the test until a
/// sample decodes to it.
#[deny(clippy::wildcard_enum_match_arm)]
fn variant_index(item: &crate::providers::openai::responses_api::Output) -> usize {
    use crate::providers::openai::responses_api::Output;
    match item {
        Output::Message(_) => 0,
        Output::FunctionCall(_) => 1,
        Output::CustomToolCall(_) => 2,
        Output::Reasoning { .. } => 3,
        Output::Unknown(_) => 4,
    }
}

#[test]
fn every_output_variant_decodes_to_a_block() {
    let output = every_kind();
    let typed: Vec<crate::providers::openai::responses_api::Output> = output
        .iter()
        .map(|item| serde_json::from_value(item.clone()).expect("every sample is well formed"))
        .collect();
    crate::test_utils::history::assert_every_variant(&typed, variant_index, 5);
    for response in [
        decode(Mode::Streaming, frames(&restated(&output))),
        decode(Mode::Unary, whole(&output)),
    ] {
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
        let mut body = serde_json::to_value(sample_response(ResponseStatus::Incomplete))
            .expect("the sample response serializes");
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
    for status in [ResponseStatus::Failed, ResponseStatus::Cancelled] {
        let mut body =
            serde_json::to_value(sample_response(status)).expect("the sample response serializes");
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
