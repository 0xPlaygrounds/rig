use super::{
    ContentPartChunkPart, ItemChunkKind, ResponsesDecoder, StreamingCompletionChunk,
    classify_responses_frame, reasoning_from_done_item,
};
use crate::completion::CompletionRequest;
use crate::driver::{Decoded, feed_frames};
use crate::error::{ErrorKind, ErrorReport, ProviderError};
use crate::message::{AssistantContent, ReasoningContent};
use crate::operation::Completion;
use crate::providers::internal::openai_chat_completions_compatible::test_support::{
    sse_bytes_from_data_lines, sse_bytes_from_json_events,
};
use crate::providers::openai::OpenAIConfig;
use crate::providers::openai::responses_api::{
    AdditionalParameters, CompletionResponse, IncompleteDetailsReason, OutputTokensDetails,
    ReasoningSummary, ResponseError, ResponseObject, ResponseStatus, ResponsesUsage,
};
use crate::streaming::{Item, PartKind, StreamEvent};
use crate::test_utils::MockStreamingClient;
use crate::wire::WireEvent;
use crate::wire::WireFrame;
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
fn reasoning_done_item_fuses_summary_content_and_encrypted_into_one_end() {
    let summary = vec![
        ReasoningSummary::SummaryText {
            text: "step 1".to_string(),
        },
        ReasoningSummary::SummaryText {
            text: "step 2".to_string(),
        },
    ];
    let content = vec!["private reasoning".to_string()];
    let reasoning = reasoning_from_done_item(
        Some("rs_1"),
        summary,
        content,
        Some("enc_blob".to_string()),
        None,
    );

    // ONE restatement carrying every block in wire field order — never a
    // block per entry, which made siblings under one `rs_*` id.
    let Some(reasoning) = reasoning else {
        panic!("expected one wire-sent reasoning restatement");
    };
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert_eq!(
        reasoning.content,
        vec![
            ReasoningContent::Summary("step 1".to_string()),
            ReasoningContent::Summary("step 2".to_string()),
            ReasoningContent::Text {
                text: "private reasoning".to_string(),
                signature: None,
            },
            ReasoningContent::Encrypted("enc_blob".to_string()),
        ]
    );
}

#[test]
fn reasoning_done_item_without_encrypted_emits_summary_only() {
    let summary = vec![ReasoningSummary::SummaryText {
        text: "only summary".to_string(),
    }];
    let reasoning = reasoning_from_done_item(Some("rs_2"), summary, Vec::new(), None, None);

    let Some(reasoning) = reasoning else {
        panic!("expected one reasoning restatement");
    };
    assert_eq!(reasoning.id.as_deref(), Some("rs_2"));
    assert_eq!(
        reasoning.content,
        vec![ReasoningContent::Summary("only summary".to_string())]
    );
}

#[test]
fn empty_encrypted_reasoning_is_not_emitted() {
    let content = vec!["visible reasoning".to_string()];

    let reasoning =
        reasoning_from_done_item(Some("rs_1"), Vec::new(), content, Some(String::new()), None);

    let Some(reasoning) = reasoning else {
        panic!("expected one reasoning restatement");
    };
    assert_eq!(
        reasoning.content,
        vec![ReasoningContent::Text {
            text: "visible reasoning".to_string(),
            signature: None,
        }],
        "an empty encrypted payload contributes no block"
    );

    // An entirely empty done item says nothing at the boundary.
    assert!(
        reasoning_from_done_item(
            Some("rs_1"),
            Vec::new(),
            Vec::new(),
            Some(String::new()),
            None
        )
        .is_none()
    );
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
/// buffered replay runs: envelope repair on, seeded with `initial_usage`.
fn decoded_body(
    provider: &str,
    body: &str,
    initial_usage: Option<ResponsesUsage>,
) -> Decoded<Completion> {
    let frames: Vec<WireFrame> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim))
        .filter(|data| !data.is_empty() && *data != "[DONE]")
        .map(|data| WireFrame::Text(data.to_owned()))
        .collect();
    feed_frames!(
        ResponsesDecoder::new(provider)
            .with_envelope_repair()
            .with_initial_usage(initial_usage),
        provider,
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
    let decoded = decoded_body(
        "openai",
        &body_of(&[json!({
            "type": "response.reasoning_text.done",
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 7,
            "text": "the model's raw chain of thought",
        })]),
        None,
    );
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
        let err = decoded_body("ChatGPT", &format!("data: {payload}\n"), None)
            .outcome
            .expect_err("error payload should surface as provider response");

        assert!(matches!(err, ProviderError::ProviderResponse(_)));
        assert_eq!(err.provider_response_status(), None);
        assert_eq!(err.provider_response_body(), Some(payload.as_str()));
    }
}

#[test]
fn reasoning_output_item_done_emits_reasoning_text_content() {
    let decoded = decoded_body(
        "openai",
        &body_of(&[json!({
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
        })]),
        None,
    );
    // The done item is one whole reasoning part, under the item's id.
    let ended = decoded.ended();
    let [AssistantContent::Reasoning(reasoning)] = ended.as_slice() else {
        panic!("one reasoning part: {:?}", decoded.events());
    };
    let reasoning = reasoning.value();
    assert_eq!(reasoning.id.as_deref(), Some("rs_text_1"));
    assert_eq!(
        reasoning.content,
        vec![ReasoningContent::Text {
            text: "visible reasoning".to_string(),
            signature: None,
        }]
    );
}

/// Envelope-less replay shape (ChatGPT bodies): an id-less summary delta,
/// the done item restating the whole part, then visible text.
#[test]
fn envelope_less_reasoning_then_text_decodes() {
    let decoded = decoded_body(
        "openai",
        &body_of(&[
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
        ]),
        None,
    );
    assert_eq!(texts_of(&decoded.events()), ["the answer"]);
}

#[test]
fn reasoning_text_delta_emits_reasoning_delta() {
    let decoded = decoded_body(
        "openai",
        &body_of(&[json!({
            "type": "response.reasoning_text.delta",
            "item_id": "rs_delta_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "thinking",
        })]),
        None,
    );
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

#[test]
fn unknown_output_item_is_kept_in_the_turn_verbatim() {
    // A hosted-tool item (web_search_call) arriving on
    // `response.output_item.done` is an opaque part of the turn holding the
    // verbatim item, sealed to the reply's issuer, so history replays it.
    let item = json!({
        "type": "web_search_call",
        "id": "ws_001",
        "status": "completed",
        "action": { "type": "search", "queries": ["rig framework"] },
    });
    let decoded = decoded_body(
        "openai",
        &body_of(&[json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 1,
            "item": item,
        })]),
        None,
    );
    let ended = decoded.ended();
    let [AssistantContent::Opaque(part)] = ended.as_slice() else {
        panic!("expected one opaque part: {:?}", decoded.events());
    };
    assert_eq!(part.issuer().as_str(), "openai");
    assert_eq!(
        part.extension_for::<crate::providers::openai::responses_api::ResponsesItem>(&[
            crate::message::Issuer::from("openai")
        ])
        .expect("well-formed"),
        Some(crate::providers::openai::responses_api::ResponsesItem(item)),
        "the raw web_search_call item should reach the turn verbatim",
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

/// A multi-block reasoning done item (summaries + `encrypted_content`)
/// is exactly ONE reasoning part carrying every block in wire order —
/// never sibling parts sharing one `rs_*` id, which would replay as
/// duplicate reasoning input items carrying the identical id.
#[tokio::test]
async fn multi_block_reasoning_done_item_yields_one_part() {
    let reasoning_done = json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [
                {"type": "summary_text", "text": "step 1"},
                {"type": "summary_text", "text": "step 2"}
            ],
            "content": [],
            "encrypted_content": "enc_blob"
        }
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": sample_response(ResponseStatus::Completed),
    });

    let mut stream = responses_stream_of(&[reasoning_done, completed]).await;
    while let Some(item) = stream.next().await {
        item.expect("stream items should be ok");
    }
    let choice = stream.finish().await.expect("the reply ended").choice;
    let [AssistantContent::Reasoning(reasoning)] = choice.as_slice() else {
        panic!("one reasoning part per rs_* id, got {choice:?}");
    };
    assert_eq!(reasoning.value().id.as_deref(), Some("rs_1"));
    assert_eq!(
        reasoning.value().content,
        vec![
            ReasoningContent::Summary("step 1".to_string()),
            ReasoningContent::Summary("step 2".to_string()),
            ReasoningContent::Encrypted("enc_blob".to_string()),
        ],
        "every block survives, in wire order, inside the one part"
    );
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
    let provider = tool_call.id.provider().expect("provider ids are kept");
    assert_eq!(provider.call_id, "call_123");
    assert_eq!(provider.item_id.as_deref(), Some("fc_123"));
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
    let provider = tool_call.id.provider().expect("provider ids are kept");
    assert_eq!(provider.call_id, "call_123");
    assert_eq!(provider.item_id.as_deref(), Some("fc_123"));
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
    let decoded = decoded_body("openai", &body_of(&events), None);
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
    let err = decoded_body("openai", &body_of(&[corrupt, completed.clone()]), None)
        .outcome
        .expect_err("a corrupt known frame must fail the buffered decode");
    assert!(
        err.to_string().contains("response.output_text.delta"),
        "the error should name the malformed event, got: {err}"
    );

    // Syntactically invalid JSON fails too.
    let body = format!("data: {{not json\ndata: {completed}\n");
    assert!(decoded_body("openai", &body, None).outcome.is_err());

    // Unknown event types stay skippable.
    let unknown = json!({ "type": "response.rocket_launch", "count": 3 });
    decoded_body("openai", &body_of(&[unknown, completed]), None)
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
    let decoded = decoded_body(
        "openai",
        &body_of(&[
            json!({ "type": "response.output_text.delta", "delta": "hi" }),
            completed.clone(),
        ]),
        None,
    );
    assert_eq!(texts_of(&decoded.events()), ["hi"]);

    // An id-less, nameless arguments delta repairs and decodes; it never
    // becomes a call.
    let decoded = decoded_body(
        "openai",
        &body_of(&[
            json!({ "type": "response.function_call_arguments.delta", "delta": "{}" }),
            completed.clone(),
        ]),
        None,
    );
    assert!(calls_of(&decoded.outcome.expect("it decodes")).is_empty());

    // An envelope-less bookkeeping event whose data is intact (`.done`
    // events) is a no-op, not an error.
    decoded_body(
        "openai",
        &body_of(&[
            json!({ "type": "response.output_text.done", "text": "hi" }),
            completed.clone(),
        ]),
        None,
    )
    .outcome
    .expect("an envelope-less done event must repair to the live no-op");

    // An envelope-less reasoning summary delta streams.
    let decoded = decoded_body(
        "openai",
        &body_of(&[
            json!({ "type": "response.reasoning_summary_text.delta", "delta": "think" }),
            completed,
        ]),
        None,
    );
    assert!(decoded.events().into_iter().any(|event| matches!(
        event,
        StreamEvent::Reasoning { text, .. } if text == "think"
    )));
}

/// The `max_output_tokens`-mid-tool-call shape: `arguments_delta` frames
/// stream partial JSON and the done item restates the same truncated bytes
/// (unparseable). Whether or not fragments preceded the restatement, the
/// partial arguments never fabricate a call.
#[test]
fn an_unparseable_restatement_never_becomes_a_call() {
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
        let decoded = decoded_body("openai", &body_of(&events), None);
        assert!(
            decoded
                .ended()
                .iter()
                .all(|content| !matches!(content, AssistantContent::ToolCall(_))),
            "{:?}",
            decoded.events()
        );
    }
}

/// A slot mixing id-bearing and id-less reasoning frames (gateways and
/// ChatGPT's envelope-less replay bodies omit the id on a subset of a
/// slot's events) is ONE reasoning part: the done item closes the part
/// the fragments opened, and nothing orphans.
#[tokio::test]
async fn mixed_id_and_id_less_reasoning_frames_are_one_part() {
    let events = [
        json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 1,
            "delta": "s1 ",
        }),
        json!({
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "summary_index": 0,
            "sequence_number": 2,
            "delta": "s2",
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 3,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "s1 s2"}],
                "content": [],
                "status": "completed",
            },
        }),
        json!({
            "type": "response.completed",
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let response = decoded_body("openai", &body_of(&events), None)
        .outcome
        .expect("the mixed slot should normalize");
    let reasoning_parts = response
        .choice
        .iter()
        .filter(|content| matches!(content, AssistantContent::Reasoning(_)))
        .count();
    assert_eq!(reasoning_parts, 1, "{:?}", response.choice);
}

/// #2258 F3: an id-less reasoning delta and the slot's `output_item.done`
/// (which always carries the real `rs_*` id) are one part: the restated
/// summary supersedes the fragments instead of duplicating them. This is
/// the ChatGPT envelope-less replay shape.
#[tokio::test]
async fn envelope_less_reasoning_deltas_are_superseded_by_their_done_item() {
    let events = [
        json!({ "type": "response.reasoning_summary_text.delta", "delta": "think" }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "think"}],
                "content": [],
                "status": "completed",
            },
        }),
        json!({
            "type": "response.completed",
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let response = decoded_body("chatgpt", &body_of(&events), None)
        .outcome
        .expect("replay should normalize");

    let reasoning: Vec<_> = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasoning.len(),
        1,
        "deltas and their full block must collapse to one reasoning item: {reasoning:?}"
    );
    let occurrences = reasoning
        .iter()
        .flat_map(|item| item.value().content.iter())
        .filter(|content| match content {
            ReasoningContent::Summary(text) | ReasoningContent::Text { text, .. } => {
                text.contains("think")
            }
            _ => false,
        })
        .count();
    assert_eq!(
        occurrences, 1,
        "the restated summary must supersede its deltas, not duplicate them"
    );
}

/// #2258 P2: text deltas for one message item interleaved with reasoning
/// are ONE text part.
#[tokio::test]
async fn same_item_text_resumes_as_one_part_across_interleaved_reasoning() {
    let events = [
        json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "hello "
        }),
        json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_2",
            "output_index": 1,
            "summary_index": 0,
            "sequence_number": 2,
            "delta": "because"
        }),
        json!({
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 3,
            "delta": "world"
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let response = decoded_body("openai", &body_of(&events), None)
        .outcome
        .expect("replay should normalize");
    assert_eq!(
        choice_text_parts(&response),
        ["hello world"],
        "same-item text must aggregate as one part around the reasoning"
    );
    assert!(
        response
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::Reasoning(_))),
        "the interleaved reasoning must survive"
    );
}

/// A slot whose `added` event carries a real `fc_*` id but whose later
/// args delta arrives id-less is one call, reporting the wire id.
#[tokio::test]
async fn mixed_id_and_id_less_events_are_one_call() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_real",
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
            "delta": "{\"x\":1}"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 3,
            "item": {
                "type": "function_call",
                "id": "fc_real",
                "call_id": "call_a",
                "name": "tool_a",
                "arguments": "{\"x\":1}",
                "status": "completed",
            },
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 4,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let response = decoded_body("openai", &body_of(&events), None)
        .outcome
        .expect("replay should normalize");
    let [call] = calls_of(&response).try_into().expect("one call");
    assert_eq!(call.function.name, "tool_a");
    assert_eq!(call.function.arguments, json!({"x": 1}));
    let provider = call.id.provider().expect("the wire issued ids");
    assert_eq!(provider.call_id, "call_a");
    assert_eq!(provider.item_id.as_deref(), Some("fc_real"));
}

/// #2258 P3: two parallel function calls whose events all lack `fc_*` ids
/// assemble as two distinct calls.
#[tokio::test]
async fn parallel_id_less_function_calls_assemble_distinctly() {
    let call_item = |name: &str, call_id: &str, arguments: &str| {
        json!({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "status": "completed",
        })
    };
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": call_item("tool_a", "call_a", ""),
        }),
        json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "sequence_number": 2,
            "item": call_item("tool_b", "call_b", ""),
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 3,
            "delta": "{\"x\":1}"
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "sequence_number": 4,
            "delta": "{\"y\":2}"
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "sequence_number": 5,
            "item": call_item("tool_a", "call_a", "{\"x\":1}"),
        }),
        json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "sequence_number": 6,
            "item": call_item("tool_b", "call_b", "{\"y\":2}"),
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 7,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let response = decoded_body("openai", &body_of(&events), None)
        .outcome
        .expect("replay should normalize");
    let calls: Vec<_> = calls_of(&response)
        .into_iter()
        .map(|call| (call.function.name.to_string(), call.function.arguments))
        .collect();
    assert_eq!(
        calls,
        [
            ("tool_a".to_owned(), json!({"x": 1})),
            ("tool_b".to_owned(), json!({"y": 2})),
        ],
        "each id-less slot must assemble its own call"
    );
}

/// A lost `output_item.done` frame followed by a healthy
/// `response.completed` must not discard the call as truncation: the
/// provider proved the turn ended, so the still-open call closes at the
/// terminal from its streamed fragments, with the dual-wire identity the
/// added event announced.
#[tokio::test]
async fn a_lost_done_frame_does_not_discard_a_provider_completed_call() {
    let events = [
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": "",
                "status": "in_progress",
            },
        }),
        json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "sequence_number": 2,
            "delta": "{\"city\":\"Paris\"}"
        }),
        json!({
            "type": "response.completed",
            "sequence_number": 3,
            "response": sample_response(ResponseStatus::Completed),
        }),
    ];
    let response = decoded_body("openai", &body_of(&events), None)
        .outcome
        .expect("replay should normalize");
    let [call] = calls_of(&response).try_into().expect("the call survives");
    assert_eq!(call.function.name, "get_weather");
    assert_eq!(call.function.arguments, json!({"city": "Paris"}));
    let provider = call.id.provider().expect("the wire issued ids");
    assert_eq!(provider.call_id, "call_abc");
    assert_eq!(provider.item_id.as_deref(), Some("fc_1"));
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
    let decoded = decoded_body("openai", &body_of(&events), None);
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
    let decoded = decoded_body("openai", &body_of(&events), None);
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
    let response = decoded_body("openai", &body_of(&[completed]), None)
        .outcome
        .expect("a body-only terminal must decode");
    assert_eq!(
        choice_text_parts(&response),
        ["from body"],
        "text stated only in the terminal body must reach the choice once"
    );
    assert_eq!(response.message_id.as_deref(), Some("msg_body_1"));
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
    let response = decoded_body("openai", &body_of(&[text_delta, completed]), None)
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
    let err = decoded_body("openai", &format!("data: {payload}\n"), None)
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
    assert_eq!(response.provider, "openai");
    assert_eq!(response.model.as_deref(), Some("gpt-5.4"));
    // The assistant message ID (`msg_...`), never the response ID
    // (`resp_123`) that the same event carries.
    assert_eq!(response.message_id.as_deref(), Some("msg_stream_1"));
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

#[test]
fn the_end_preserves_an_unknown_incomplete_reason() {
    let response = super::StreamingCompletionResponse {
        status: Some(ResponseStatus::Incomplete),
        incomplete_details: Some(IncompleteDetailsReason {
            reason: "MAX_TOOL_CALLS".to_string(),
        }),
        model: Some("gpt-5.4".to_string()),
        message_id: Some("msg_1".to_string()),
        ..super::StreamingCompletionResponse::new(None)
    };

    let (finish, issuer) = super::finish_of("openai", false, response);
    assert_eq!(
        finish.reason,
        Some(crate::completion::FinishReason::Other(
            "MAX_TOOL_CALLS".to_string()
        ))
    );
    assert_eq!(finish.message_id.as_deref(), Some("msg_1"));
    assert_eq!(finish.model.as_deref(), Some("gpt-5.4"));
    assert_eq!(issuer, None);
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
    let decoded = decoded_body(
        "openai",
        &body_of(&[
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
        ]),
        None,
    );
    assert_eq!(texts_of(&decoded.events()), ["still text"]);
}

/// A `url_citation` annotation as the Responses API attaches it to an
/// `output_text` part.
fn url_citation(start: u64, end: u64, url: &str) -> serde_json::Value {
    json!({
        "type": "url_citation",
        "start_index": start,
        "end_index": end,
        "url": url,
        "title": "Source",
    })
}

/// A completed `message` item whose `output_text` parts are `(text, annotations)`.
fn annotated_message(id: &str, parts: &[(&str, serde_json::Value)]) -> serde_json::Value {
    let content: Vec<serde_json::Value> = parts
        .iter()
        .map(|(text, annotations)| {
            json!({ "type": "output_text", "text": text, "annotations": annotations })
        })
        .collect();
    json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "content": content,
    })
}

/// One `output_text.delta` of the message item `item_id`.
fn text_delta(
    item_id: &str,
    output_index: u64,
    content_index: u64,
    sequence: u64,
    delta: &str,
) -> serde_json::Value {
    json!({
        "type": "response.output_text.delta",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "sequence_number": sequence,
        "delta": delta,
    })
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

/// The text blocks a reply's parts ended with, in order.
fn ended_texts(decoded: &Decoded<Completion>) -> Vec<crate::message::Text> {
    decoded
        .ended()
        .into_iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text),
            _ => None,
        })
        .collect()
}

/// The Responses-owned annotations a text block carries.
fn annotations_of(text: &crate::message::Text) -> Option<&serde_json::Value> {
    text.additional_params
        .as_ref()
        .and_then(|params| {
            params.wire_extras(<crate::providers::openai::responses_api::ResponsesText as crate::message::Extension>::KEY)
        })
        .and_then(|extras| extras.get("annotations"))
}

/// The streamed text part carries the annotations its `output_item.done`
/// restates, once, and its text stays what the deltas delivered. Unit-level
/// because each snapshot combination below is a frame sequence one live
/// recording cannot select; the recorded web-search cassettes pin the live
/// shape.
#[test]
fn output_item_done_attaches_annotations_to_the_streamed_text() {
    let citation = url_citation(0, 4, "https://a.example");
    let decoded = decoded_body(
        "openai",
        &body_of(&[
            text_delta("msg_1", 0, 0, 1, "Rust "),
            text_delta("msg_1", 0, 0, 2, "is fast."),
            item_done(
                0,
                3,
                annotated_message("msg_1", &[("Rust is fast.", json!([citation]))]),
            ),
            completed_with(4, json!([])),
        ]),
        None,
    );
    let texts = ended_texts(&decoded);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert_eq!(texts[0].text, "Rust is fast.");
    assert_eq!(annotations_of(&texts[0]), Some(&json!([citation])));
    let response = decoded.outcome.expect("the stream decodes");
    assert_eq!(choice_text_parts(&response), ["Rust is fast."]);
}

/// A gateway that sends no `output_item.done` for a streamed message still
/// states its annotations in the terminal snapshot, which supplies them.
#[test]
fn terminal_supplies_annotations_no_item_done_covered() {
    let citation = url_citation(0, 4, "https://a.example");
    let decoded = decoded_body(
        "openai",
        &body_of(&[
            text_delta("msg_1", 0, 0, 1, "Rust is fast."),
            completed_with(
                2,
                json!([annotated_message(
                    "msg_1",
                    &[("Rust is fast.", json!([citation]))]
                )]),
            ),
        ]),
        None,
    );
    let texts = ended_texts(&decoded);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert_eq!(
        texts[0].text, "Rust is fast.",
        "the terminal restates no text"
    );
    assert_eq!(annotations_of(&texts[0]), Some(&json!([citation])));
}

/// Every snapshot restates the same annotations: `content_part.done`,
/// `output_item.done` and `response.completed`. They attach once, with
/// or without the incremental `annotation.added` event, which still
/// reaches the consumer as an unmodeled payload.
#[test]
fn repeated_snapshots_attach_annotations_once() {
    let first = url_citation(0, 4, "https://a.example");
    let second = url_citation(8, 12, "https://b.example");
    let message = annotated_message("msg_1", &[("Rust is fast.", json!([first, second]))]);
    for with_added in [false, true] {
        let mut events = vec![text_delta("msg_1", 0, 0, 1, "Rust is fast.")];
        if with_added {
            for (index, annotation) in [&first, &second].into_iter().enumerate() {
                events.push(json!({
                    "type": "response.output_text.annotation.added",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "annotation_index": index,
                    "sequence_number": 2 + index,
                    "annotation": annotation,
                }));
            }
        }
        events.extend([
            json!({
                "type": "response.content_part.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": 4,
                "part": message["content"][0],
            }),
            item_done(0, 5, message.clone()),
            completed_with(6, json!([message])),
        ]);
        let decoded = decoded_body("openai", &body_of(&events), None);
        let unknown_types: Vec<String> = decoded
            .items
            .iter()
            .filter_map(|item| match item {
                Ok(Item::Unknown(payload)) => serde_json::to_value(payload)
                    .ok()
                    .and_then(|value| value["type"].as_str().map(str::to_owned)),
                _ => None,
            })
            .collect();
        let expected_unknown = if with_added {
            vec!["response.output_text.annotation.added"; 2]
        } else {
            Vec::new()
        };
        assert_eq!(unknown_types, expected_unknown);
        let texts = ended_texts(&decoded);
        assert_eq!(texts.len(), 1, "{texts:?}");
        assert_eq!(
            annotations_of(&texts[0]),
            Some(&json!([first, second])),
            "with_added = {with_added}"
        );
    }
}

/// Empty annotations are no metadata: the text part ends without params,
/// as the unary path decodes the same content.
#[test]
fn empty_annotations_add_no_params() {
    let message = annotated_message("msg_1", &[("hi", json!([]))]);
    let decoded = decoded_body(
        "openai",
        &body_of(&[
            text_delta("msg_1", 0, 0, 1, "hi"),
            item_done(0, 2, message.clone()),
            completed_with(3, json!([message])),
        ]),
        None,
    );
    let texts = ended_texts(&decoded);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert_eq!(texts[0].additional_params, None);
}

/// A message interleaved with a function call keeps its annotations, and
/// the call is still decoded whole.
#[test]
fn annotations_attach_beside_an_interleaved_tool_call() {
    let citation = url_citation(0, 4, "https://a.example");
    let call = json!({
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{\"q\":\"rust\"}",
        "status": "completed",
    });
    let message = annotated_message("msg_1", &[("Rust is fast.", json!([citation]))]);
    let decoded = decoded_body(
        "openai",
        &body_of(&[
            text_delta("msg_1", 0, 0, 1, "Rust is fast."),
            json!({
                "type": "response.output_item.added",
                "output_index": 1,
                "sequence_number": 2,
                "item": call,
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 1,
                "sequence_number": 3,
                "delta": "{\"q\":\"rust\"}",
            }),
            item_done(0, 4, message.clone()),
            item_done(1, 5, call.clone()),
            completed_with(6, json!([message, call])),
        ]),
        None,
    );
    let texts = ended_texts(&decoded);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert_eq!(annotations_of(&texts[0]), Some(&json!([citation])));
    let response = decoded.outcome.expect("the stream decodes");
    let calls = calls_of(&response);
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].function.name, "lookup");
    assert_eq!(calls[0].function.arguments, json!({ "q": "rust" }));
}

/// Streaming and the unary body end a message's text with equal extras,
/// content parts in wire order, and a message stated only by snapshots
/// keeps its text and annotations once.
#[test]
fn streamed_and_unary_text_carry_equal_extras() {
    let first = url_citation(0, 4, "https://a.example");
    let second = url_citation(0, 4, "https://b.example");
    let message = annotated_message(
        "msg_1",
        &[
            ("Part one.", json!([first])),
            ("Part two.", json!([second])),
        ],
    );

    let streamed = decoded_body(
        "openai",
        &body_of(&[
            text_delta("msg_1", 0, 0, 1, "Part one."),
            text_delta("msg_1", 0, 1, 2, "Part two."),
            item_done(0, 3, message.clone()),
            completed_with(4, json!([message])),
        ]),
        None,
    );
    let snapshot_only = decoded_body(
        "openai",
        &body_of(&[
            item_done(0, 1, message.clone()),
            completed_with(2, json!([message])),
        ]),
        None,
    );
    let mut body = serde_json::to_value(sample_response(ResponseStatus::Completed))
        .expect("the sample response serializes");
    body["output"] = json!([message]);
    let unary = decoded_body("openai", &format!("data: {body}\n"), None);

    let unary_texts = ended_texts(&unary);
    assert_eq!(unary_texts.len(), 1, "{unary_texts:?}");
    assert_eq!(
        annotations_of(&unary_texts[0]),
        Some(&json!([first, second]))
    );
    for (label, decoded) in [("streamed", &streamed), ("snapshot only", &snapshot_only)] {
        let texts = ended_texts(decoded);
        assert_eq!(texts, unary_texts, "{label}");
    }
}

/// A completed `message` item with one `output_text` part and `phase`
/// spelled as given (`Value::Null` states it as `null`).
fn phased_message(id: &str, text: &str, phase: serde_json::Value) -> serde_json::Value {
    let mut message = annotated_message(id, &[(text, json!([]))]);
    if phase != json!("<absent>") {
        message["phase"] = phase;
    }
    message
}

/// The `output_item.added` opening `item` at `output_index`, with no content.
fn item_added(output_index: u64, sequence: u64, mut item: serde_json::Value) -> serde_json::Value {
    if item["type"] == "message" {
        item["content"] = json!([]);
        item["status"] = json!("in_progress");
    }
    json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "sequence_number": sequence,
        "item": item,
    })
}

/// The Responses-owned extras a text block carries.
fn own_extras(text: &crate::message::Text) -> Option<&serde_json::Map<String, serde_json::Value>> {
    text.additional_params.as_ref().and_then(|params| {
        params.wire_extras(<crate::providers::openai::responses_api::ResponsesText as crate::message::Extension>::KEY)
    })
}

/// The unary body whose `output` is `output`, decoded as a buffered replay.
fn unary_of(output: serde_json::Value) -> Decoded<Completion> {
    let mut body = serde_json::to_value(sample_response(ResponseStatus::Completed))
        .expect("the sample response serializes");
    body["output"] = output;
    decoded_body("openai", &format!("data: {body}\n"), None)
}

/// Streamed text carries its message's `phase` whichever restatement
/// states it: `output_item.added`, `output_item.done`, the terminal, or all
/// three. It lands once, on the text part, and the text is what the deltas
/// delivered. Unit-level because a live stream always states it in all
/// three places; the recorded streams pin that shape.
#[test]
fn streamed_text_carries_its_message_phase_from_any_restatement() {
    let phased = phased_message("msg_1", "Hello.", json!("commentary"));
    let bare = phased_message("msg_1", "Hello.", json!("<absent>"));
    let cases = [
        ("added", &phased, &bare, &bare),
        ("done", &bare, &phased, &bare),
        ("terminal", &bare, &bare, &phased),
        ("all", &phased, &phased, &phased),
    ];
    for (label, added, done, terminal) in cases {
        let decoded = decoded_body(
            "openai",
            &body_of(&[
                item_added(0, 1, added.clone()),
                text_delta("msg_1", 0, 0, 2, "Hel"),
                text_delta("msg_1", 0, 0, 3, "lo."),
                item_done(0, 4, done.clone()),
                completed_with(5, json!([terminal])),
            ]),
            None,
        );
        let texts = ended_texts(&decoded);
        assert_eq!(texts.len(), 1, "{label}: {texts:?}");
        assert_eq!(texts[0].text, "Hello.", "{label}");
        assert_eq!(
            own_extras(&texts[0]),
            json!({ "phase": "commentary" }).as_object(),
            "{label}"
        );
    }
}

/// A `null` or absent `phase` is no phase: the text ends without params,
/// streamed or unary.
#[test]
fn null_and_absent_phase_add_no_params() {
    for phase in [serde_json::Value::Null, json!("<absent>")] {
        let message = phased_message("msg_1", "Hello.", phase.clone());
        let streamed = decoded_body(
            "openai",
            &body_of(&[
                item_added(0, 1, message.clone()),
                text_delta("msg_1", 0, 0, 2, "Hello."),
                item_done(0, 3, message.clone()),
                completed_with(4, json!([message])),
            ]),
            None,
        );
        let unary = unary_of(json!([message]));
        for (label, decoded) in [("streamed", &streamed), ("unary", &unary)] {
            let texts = ended_texts(decoded);
            assert_eq!(texts.len(), 1, "{label} {phase}: {texts:?}");
            assert_eq!(texts[0].additional_params, None, "{label} {phase}");
        }
    }
}

/// A commentary message between reasoning and a function call keeps its
/// `phase`; the reasoning and the call decode as before.
#[test]
fn phase_attaches_beside_interleaved_reasoning_and_tool_call() {
    let reasoning = json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "opaque",
    });
    let message = phased_message("msg_1", "Checking the weather.", json!("commentary"));
    let call = json!({
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": "{\"city\":\"Paris\"}",
        "status": "completed",
    });
    let decoded = decoded_body(
        "openai",
        &body_of(&[
            item_added(0, 1, reasoning.clone()),
            item_done(0, 2, reasoning.clone()),
            item_added(1, 3, message.clone()),
            text_delta("msg_1", 1, 0, 4, "Checking the weather."),
            item_done(1, 5, message.clone()),
            item_added(2, 6, call.clone()),
            json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 2,
                "sequence_number": 7,
                "delta": "{\"city\":\"Paris\"}",
            }),
            item_done(2, 8, call.clone()),
            completed_with(9, json!([reasoning, message, call])),
        ]),
        None,
    );
    let texts = ended_texts(&decoded);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert_eq!(texts[0].text, "Checking the weather.");
    assert_eq!(
        own_extras(&texts[0]),
        json!({ "phase": "commentary" }).as_object()
    );
    let response = decoded.outcome.expect("the stream decodes");
    let calls = calls_of(&response);
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].function.arguments, json!({ "city": "Paris" }));
    assert!(
        response
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::Reasoning(reasoning) if reasoning.value().id.as_deref() == Some("rs_1"))),
        "{:?}",
        response.choice
    );
}

/// A reply with a commentary message and a final answer ends two text
/// blocks, each with its own item's `phase` and id, streamed and unary
/// alike. A reply with one message records no id: the assistant message's
/// own id names it.
#[test]
fn several_messages_each_keep_their_phase_and_id() {
    let commentary = phased_message("msg_1", "Let me think.", json!("commentary"));
    let answer = phased_message("msg_2", "Apple.", json!("final_answer"));
    let streamed = decoded_body(
        "openai",
        &body_of(&[
            item_added(0, 1, commentary.clone()),
            text_delta("msg_1", 0, 0, 2, "Let me think."),
            item_done(0, 3, commentary.clone()),
            item_added(1, 4, answer.clone()),
            text_delta("msg_2", 1, 0, 5, "Apple."),
            item_done(1, 6, answer.clone()),
            completed_with(7, json!([commentary, answer])),
        ]),
        None,
    );
    let unary = unary_of(json!([commentary, answer]));
    for (label, decoded) in [("streamed", &streamed), ("unary", &unary)] {
        let texts = ended_texts(decoded);
        let facts: Vec<(&str, Option<&serde_json::Map<String, serde_json::Value>>)> = texts
            .iter()
            .map(|text| (text.text.as_str(), own_extras(text)))
            .collect();
        assert_eq!(
            facts,
            [
                (
                    "Let me think.",
                    json!({ "phase": "commentary", "message_id": "msg_1" }).as_object()
                ),
                (
                    "Apple.",
                    json!({ "phase": "final_answer", "message_id": "msg_2" }).as_object()
                ),
            ],
            "{label}"
        );
    }
}

/// A gateway that names a message's deltas with another id than its item
/// events still gets the item's `phase` on the text: the output slot ties
/// them.
#[test]
fn phase_follows_the_output_slot_when_delta_ids_differ() {
    let message = phased_message("msg_1", "Hello.", json!("final_answer"));
    let decoded = decoded_body(
        "copilot",
        &body_of(&[
            item_added(0, 1, message.clone()),
            text_delta("item_a", 0, 0, 2, "Hello."),
            item_done(0, 3, message.clone()),
            completed_with(4, json!([message])),
        ]),
        None,
    );
    let texts = ended_texts(&decoded);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert_eq!(
        own_extras(&texts[0]),
        json!({ "phase": "final_answer" }).as_object()
    );
}

/// For equal content the streamed, snapshot-only and unary text blocks
/// carry equal `openai_responses` extras, `phase` among them.
#[test]
fn streamed_and_unary_text_carry_equal_phase() {
    let citation = url_citation(0, 4, "https://a.example");
    let mut message = annotated_message("msg_1", &[("Rust is fast.", json!([citation]))]);
    message["phase"] = json!("final_answer");
    let streamed = decoded_body(
        "openai",
        &body_of(&[
            item_added(0, 1, message.clone()),
            text_delta("msg_1", 0, 0, 2, "Rust is fast."),
            item_done(0, 3, message.clone()),
            completed_with(4, json!([message])),
        ]),
        None,
    );
    let snapshot_only = decoded_body(
        "openai",
        &body_of(&[
            item_done(0, 1, message.clone()),
            completed_with(2, json!([message])),
        ]),
        None,
    );
    let unary_texts = ended_texts(&unary_of(json!([message])));
    assert_eq!(unary_texts.len(), 1, "{unary_texts:?}");
    assert_eq!(
        own_extras(&unary_texts[0]),
        json!({
            "annotations": [citation],
            "phase": "final_answer",
            "text_fingerprint": crate::message::Fingerprint::of("Rust is fast."),
        })
        .as_object()
    );
    for (label, decoded) in [("streamed", &streamed), ("snapshot only", &snapshot_only)] {
        assert_eq!(ended_texts(decoded), unary_texts, "{label}");
    }
}

/// A terminal that restates a message at another position than its stream
/// did (it left an earlier item out) still gives that message its `phase`:
/// the item's id ties the restatement to it, and the part at the terminal's
/// position keeps its own.
#[test]
fn terminal_phase_follows_the_item_id_when_positions_shift() {
    let reasoning = json!({ "type": "reasoning", "id": "rs_1", "summary": [] });
    let commentary = phased_message("msg_1", "Let me think.", json!("<absent>"));
    let answer = phased_message("msg_2", "Apple.", json!("final_answer"));
    let decoded = decoded_body(
        "openai",
        &body_of(&[
            item_added(0, 1, answer.clone()),
            text_delta("msg_2", 0, 0, 2, "Apple."),
            item_done(0, 3, answer.clone()),
            item_added(1, 4, reasoning.clone()),
            item_done(1, 5, reasoning),
            item_added(2, 6, commentary.clone()),
            text_delta("msg_1", 2, 0, 7, "Let me think."),
            item_done(2, 8, commentary),
            completed_with(
                9,
                json!([
                    answer,
                    phased_message("msg_1", "Let me think.", json!("commentary"))
                ]),
            ),
        ]),
        None,
    );
    let phases: Vec<(String, Option<String>)> = ended_texts(&decoded)
        .iter()
        .map(|text| {
            let phase = own_extras(text)
                .and_then(|extras| extras.get("phase"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            (text.text.clone(), phase)
        })
        .collect();
    assert_eq!(
        phases,
        [
            ("Apple.".to_owned(), Some("final_answer".to_owned())),
            ("Let me think.".to_owned(), Some("commentary".to_owned())),
        ]
    );
}

/// Copilot names one message item differently in each event: the
/// `output_item.added`, every delta and the `output_item.done` carry their
/// own ids. The output slot still ties the text to its item, the done id
/// names it (as it names the reply's message id), and a reply with two
/// items replays as two, each with its own `phase`. The stream shape is a
/// live Copilot probe's, recorded outside the corpus.
#[test]
fn a_gateway_naming_each_event_differently_replays_each_message_once() {
    let with_id = |message: &serde_json::Value, id: &str| {
        let mut message = message.clone();
        message["id"] = json!(id);
        message
    };
    let commentary = phased_message("msg", "Let me think.", json!("commentary"));
    let answer = phased_message("msg", "Apple.", json!("final_answer"));
    let decoded = decoded_body(
        "copilot",
        &body_of(&[
            item_added(0, 1, with_id(&commentary, "added_1")),
            text_delta("delta_1", 0, 0, 2, "Let "),
            text_delta("delta_2", 0, 0, 3, "me think."),
            item_done(0, 4, with_id(&commentary, "done_1")),
            item_added(1, 5, with_id(&answer, "added_2")),
            text_delta("delta_3", 1, 0, 6, "Apple."),
            item_done(1, 7, with_id(&answer, "done_2")),
            // The terminal renames the items again and lists them in another
            // order; what `output_item.done` stated stands.
            completed_with(
                8,
                json!([with_id(&answer, "final_2"), with_id(&commentary, "final_1")]),
            ),
        ]),
        None,
    );
    let response = decoded.outcome.expect("the stream decodes");
    assert_eq!(response.message_id.as_deref(), Some("done_2"));
    let history = response.message().expect("the reply has content");
    let items: Vec<serde_json::Value> =
        Vec::<crate::providers::openai::responses_api::InputItem>::try_from(history)
            .expect("history converts")
            .iter()
            .map(|item| serde_json::to_value(item).expect("item serializes"))
            .collect();
    let replayed: Vec<(&str, &str, String)> = items
        .iter()
        .map(|item| {
            let text = item["content"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|block| block["text"].as_str())
                .collect::<String>();
            (
                item["id"].as_str().unwrap_or("-"),
                item["phase"].as_str().unwrap_or("-"),
                text,
            )
        })
        .collect();
    assert_eq!(
        replayed,
        [
            ("done_1", "commentary", "Let me think.".to_owned()),
            ("done_2", "final_answer", "Apple.".to_owned()),
        ]
    );
}

/// A stream whose output indices were repaired to zero puts every item in
/// one slot. Each message's id still ties its text to its own `phase`,
/// whether or not the stream opens its items with `output_item.added`.
#[test]
fn repaired_indices_keep_each_message_phase_by_id() {
    let strip = |mut event: serde_json::Value| {
        if let Some(event) = event.as_object_mut() {
            event.remove("output_index");
        }
        event
    };
    let commentary = phased_message("msg_1", "Let me think.", json!("commentary"));
    let answer = phased_message("msg_2", "Apple.", json!("final_answer"));
    for with_added in [true, false] {
        let mut events = Vec::new();
        for (index, (message, text)) in [(&commentary, "Let me think."), (&answer, "Apple.")]
            .into_iter()
            .enumerate()
        {
            let index = index as u64;
            let id = message["id"].as_str().unwrap_or_default();
            if with_added {
                events.push(item_added(index, 3 * index + 1, message.clone()));
            }
            events.push(text_delta(id, index, 0, 3 * index + 2, text));
            events.push(item_done(index, 3 * index + 3, message.clone()));
        }
        events.push(completed_with(7, json!([commentary, answer])));
        let decoded = decoded_body(
            "chatgpt",
            &body_of(&events.into_iter().map(strip).collect::<Vec<_>>()),
            None,
        );
        let texts = ended_texts(&decoded);
        let facts: Vec<(&str, Option<&serde_json::Map<String, serde_json::Value>>)> = texts
            .iter()
            .map(|text| (text.text.as_str(), own_extras(text)))
            .collect();
        assert_eq!(
            facts,
            [
                (
                    "Let me think.",
                    json!({ "phase": "commentary", "message_id": "msg_1" }).as_object()
                ),
                (
                    "Apple.",
                    json!({ "phase": "final_answer", "message_id": "msg_2" }).as_object()
                ),
            ],
            "with_added = {with_added}"
        );
    }
}

/// A reply whose second message item carries no text still names the
/// first text by its own item: the reply's message id is the second's.
#[test]
fn a_text_beside_an_empty_message_item_keeps_its_own_id() {
    let commentary = phased_message("msg_1", "Let me think.", json!("commentary"));
    let mut empty = phased_message("msg_2", "", json!("final_answer"));
    empty["content"] = json!([]);
    let decoded = unary_of(json!([commentary, empty]));
    let texts = ended_texts(&decoded);
    assert_eq!(texts.len(), 1, "{texts:?}");
    assert_eq!(
        own_extras(&texts[0]),
        json!({ "phase": "commentary", "message_id": "msg_1" }).as_object()
    );
    let response = decoded.outcome.expect("the body decodes");
    assert_eq!(response.message_id.as_deref(), Some("msg_2"));
}

/// Unmodelled items fold to the same parts, in the same order, streamed or
/// whole: both run through this decoder, which writes each one as an opaque
/// part when its `output_item.done` (or the unary body) states it.
#[test]
fn unmodelled_items_fold_the_same_streamed_and_whole() {
    let compaction = json!({"type": "compaction", "id": "cmp_1",
                            "encrypted_content": "opaque", "status": "completed"});
    let custom = json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_c1",
                        "name": "grammar", "input": "x = 1", "status": "completed"});
    let message = phased_message("msg_1", "Done.", json!("final_answer"));
    let output = json!([compaction, custom, message]);
    let streamed = decoded_body(
        "openai",
        &body_of(&[
            item_added(0, 1, compaction.clone()),
            item_done(0, 2, compaction.clone()),
            item_added(1, 3, custom.clone()),
            item_done(1, 4, custom.clone()),
            item_added(2, 5, message.clone()),
            text_delta("msg_1", 2, 0, 6, "Done."),
            item_done(2, 7, message.clone()),
            completed_with(8, output.clone()),
        ]),
        None,
    );
    let whole = unary_of(output);
    let streamed = streamed.ended();
    assert_eq!(streamed, whole.ended());
    assert!(matches!(
        streamed.as_slice(),
        [
            AssistantContent::Opaque(_),
            AssistantContent::Opaque(_),
            AssistantContent::Text(_)
        ]
    ));
}
