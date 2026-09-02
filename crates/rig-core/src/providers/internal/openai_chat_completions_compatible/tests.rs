use super::test_support::sse_bytes_from_data_lines;
use super::{
    CompatibleStreamProfile, map_openai_finish_reason, send_compatible_raw_streaming_request,
};
use crate::completion::{CompletionError, FinishReason};
use crate::http_client;
use crate::message::AssistantContent;
use crate::streaming::{BlockClose, BlockKind, Delta, StreamEvent};
use crate::test_utils::MockStreamingClient;
use crate::test_utils::internal_streaming_profiles::{
    DistinctToolCallEvictionProfile, ErrorAfterPendingToolCallProfile, FinishReasonCleanupProfile,
    ReasoningAroundToolCallProfile,
};
use futures::StreamExt;

/// Wrap a profile-driven raw stream into the normalized carrier, so these
/// tests exercise the same path providers use.
async fn send_compatible_streaming_request<T, P>(
    http_client: T,
    req: http::Request<Vec<u8>>,
    profile: P,
) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError>
where
    T: crate::http_client::HttpClientExt + Clone + 'static,
    P: CompatibleStreamProfile + 'static,
{
    let raw = send_compatible_raw_streaming_request(
        http_client,
        req,
        None,
        "test-compatible".to_owned(),
        profile,
    )
    .await?;
    Ok(crate::streaming::StreamingCompletionResponse::stream(
        "test-compatible",
        raw,
    ))
}

/// A tool call's open and its fragments — the lifecycle events around a
/// call that tests asserting on completed calls skip over.
fn is_tool_lifecycle(event: &StreamEvent) -> bool {
    matches!(
        event,
        StreamEvent::BlockStart {
            kind: BlockKind::ToolCall,
            ..
        } | StreamEvent::BlockDelta {
            delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
            ..
        }
    )
}

/// The completed tool call a `BlockEnd` published, if it did.
fn completed_tool_call(event: StreamEvent) -> Option<crate::message::ToolCall> {
    match event {
        StreamEvent::BlockEnd {
            block: Some(AssistantContent::ToolCall(tool_call)),
            ..
        } => Some(tool_call),
        _ => None,
    }
}

/// Normalize a turn that produced `content` under `finish_reason`.
fn normalize_one(
    finish_reason: &'static str,
    content: Vec<crate::completion::AssistantContent>,
) -> Result<crate::completion::CompletionResponse, CompletionError> {
    super::normalize_openai_response(
        "test-compatible",
        &[()],
        Some("chatcmpl-1"),
        Some("test-model"),
        crate::completion::Usage {
            input_tokens: 16,
            output_tokens: 16,
            total_tokens: 32,
            reasoning_tokens: 16,
            ..Default::default()
        },
        |(): &()| finish_reason,
        |()| Some(content),
    )
}

/// A cap spent entirely on hidden reasoning: the turn is empty and the
/// reason is the whole diagnostic, so it must reach the caller.
#[test]
fn empty_choice_survives_a_truncated_turn() {
    for (wire, expected) in [
        ("length", crate::completion::FinishReason::Length),
        (
            "content_filter",
            crate::completion::FinishReason::ContentFilter,
        ),
    ] {
        let response = normalize_one(wire, Vec::new())
            .unwrap_or_else(|error| panic!("{wire} should normalize: {error}"));

        assert_eq!(response.finish_reason(), Some(expected));
        assert!(response.choice.is_empty());
        assert_eq!(response.usage.reasoning_tokens, 16);
    }
}

/// A turn that ran to completion with nothing in it is still a provider
/// defect, and so is one whose reason rig could not classify.
#[test]
fn empty_choice_still_fails_a_completed_turn() {
    for wire in ["stop", "tool_calls", "GUARDRAIL_INTERVENED", ""] {
        assert!(
            normalize_one(wire, Vec::new()).is_err(),
            "an empty {wire:?} turn must stay an error"
        );
    }
}

#[test]
fn non_empty_truncated_turn_is_unchanged() {
    let response = normalize_one(
        "length",
        vec![crate::completion::AssistantContent::text("hi")],
    )
    .expect("partial text should normalize");

    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(response.choice.len(), 1);
}

#[test]
fn truncated_output_covers_only_the_cut_short_reasons() {
    use crate::completion::FinishReason;

    assert!(FinishReason::Length.truncated_output());
    assert!(FinishReason::ContentFilter.truncated_output());
    assert!(!FinishReason::Stop.truncated_output());
    assert!(!FinishReason::ToolCalls.truncated_output());
    assert!(!FinishReason::Other("whatever".to_owned()).truncated_output());
}

#[test]
fn sse_error_detector_handles_null_empty_and_object_or_string_errors() {
    use super::provider_response_from_compatible_sse_data as detect;

    // An empty `error` (`null` or `""`) with no choices must NOT terminate the
    // stream — some providers send one with the terminal usage event. Each of
    // these should be treated as "not an error chunk".
    assert!(detect(r#"{"error":null}"#).is_none());
    assert!(detect(r#"{"error":null,"usage":{"total_tokens":3}}"#).is_none());
    assert!(detect(r#"{"error":""}"#).is_none());
    // A normal content chunk (no `error` key) is also not an error.
    assert!(detect(r#"{"choices":[{"delta":{"content":"hi"}}]}"#).is_none());
    // A live content chunk that ALSO carries an `error` field must NOT terminate
    // the stream — the `choices` guard wins regardless of the error value.
    assert!(detect(r#"{"error":"metadata","choices":[{"delta":{"content":"hi"}}]}"#).is_none());
    assert!(
        detect(r#"{"error":{"message":"x"},"choices":[{"delta":{"content":"hi"}}]}"#).is_none()
    );

    // A non-empty string `error` IS detected, preserving the raw body.
    let string_body = r#"{"error":"oops"}"#;
    let string_error = detect(string_body).expect("string error should be detected");
    assert_eq!(string_error.provider_response_body(), Some(string_body));
    assert_eq!(string_error.provider_response_status(), None);

    // A real provider error envelope IS detected, preserving the raw body.
    let body = r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#;
    let error = detect(body).expect("object error envelope should be detected");
    assert_eq!(error.provider_response_body(), Some(body));
    // It arrives mid-stream with no HTTP status attached.
    assert_eq!(error.provider_response_status(), None);

    // The choices guard is narrowed to a NON-EMPTY array: an error body
    // that also carries `"choices":[]` (or `null`) is still an error —
    // pre-#2258-B6 it classified as a normal chunk, and a following
    // `[DONE]` committed the failed turn as a successful zero-usage
    // completion.
    let masked = r#"{"error":{"message":"rate limited"},"choices":[]}"#;
    let error = detect(masked).expect("an empty choices array must not mask the error");
    assert_eq!(error.provider_response_body(), Some(masked));
    assert!(
        detect(r#"{"error":{"message":"rate limited"},"choices":null}"#).is_some(),
        "a null choices value must not mask the error"
    );
}

/// A tool call starting is a reasoning boundary on this wire: reasoning
/// deltas straddling a complete tool call aggregate as TWO reasoning
/// parts, because the adapter synthesizes the end this wire never
/// announces before the first tool-call fragment (as it already did for
/// interleaved text).
#[tokio::test]
async fn tool_call_closes_the_open_reasoning_block() {
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines(["reasoning_a", "tool_call", "reasoning_b", "finish"]),
    };

    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, ReasoningAroundToolCallProfile)
        .await
        .expect("stream should start");
    while stream.next().await.is_some() {}

    let reasoning_texts: Vec<String> = stream
        .snapshot()
        .into_iter()
        .filter_map(|item| match item {
            crate::completion::AssistantContent::Reasoning(reasoning) => Some(
                reasoning
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        crate::message::ReasoningContent::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect();

    assert_eq!(
        reasoning_texts,
        vec!["thinking before".to_owned(), "thinking after".to_owned()],
        "the tool call must split the reasoning into two parts"
    );
}

/// One chunk carrying BOTH a reasoning delta and a complete tool call:
/// the adapter's within-chunk order is reasoning → text → tool calls
/// (the model reasons, speaks, then acts — the order every boundary-less
/// wire and this crate's ollama adapter use), so the reasoning part
/// completes BEFORE the tool call in the aggregated content.
#[tokio::test]
async fn a_combined_chunk_emits_reasoning_before_its_tool_call() {
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines(["combined", "finish"]),
    };

    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, ReasoningAroundToolCallProfile)
        .await
        .expect("stream should start");
    while stream.next().await.is_some() {}

    let kinds: Vec<&'static str> = stream
        .snapshot()
        .into_iter()
        .map(|item| match item {
            crate::completion::AssistantContent::Reasoning(_) => "reasoning",
            crate::completion::AssistantContent::ToolCall(_) => "tool_call",
            _ => "other",
        })
        .collect();

    assert_eq!(
        kinds,
        vec!["reasoning", "tool_call"],
        "the same-chunk reasoning must close before the tool call opens"
    );
}

#[tokio::test]
async fn evicted_tool_call_emits_object_input_end_to_end() {
    // Regression guard for #1958, end-to-end through the streaming aggregator.
    //
    // The first tool call is evicted (a distinct second call starts at the
    // same index) **while its arguments are still a partial, non-object
    // string** (`first_args_partial` streams `{"query":` — a fragment the
    // accumulator holds as a bare `Value::String`). Before the fix,
    // `finalize_completed_streaming_tool_call` forwarded that string verbatim,
    // so the evicted call emerged with a string `function.arguments`; a
    // downstream object-typed serializer (e.g. Anthropic's `tool_use.input`)
    // then sent a bare string and strict providers rejected it.
    //
    // This sequence is what makes the test load-bearing: with the fix
    // reverted the evicted call's arguments are `String("{\"query\":")` and
    // the `is_object()` assertion below fails; the sibling
    // `distinct_same_name_tool_calls_evict_by_id_when_a_new_call_starts` test
    // (which lets the first call's args *complete* before eviction) does not
    // exercise this path.
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            "first_start",
            "first_args_partial",
            "second_start",
            "second_args",
            "finish",
        ]),
    };

    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream =
        send_compatible_streaming_request(client, req, DistinctToolCallEvictionProfile)
            .await
            .expect("stream should start");

    let mut collected_tool_calls = Vec::new();
    while let Some(item) = stream.next().await {
        if let Some(tool_call) = completed_tool_call(item.expect("stream item should be ok")) {
            collected_tool_calls.push(tool_call);
        }
    }

    assert_eq!(collected_tool_calls.len(), 2);
    for tc in &collected_tool_calls {
        assert!(
            tc.function.arguments.is_object(),
            "tool_use input must be an object, got {:?} for {}",
            tc.function.arguments,
            tc.function.name
        );
    }
    // Pin the evicted call specifically: its unparseable partial string is
    // normalized to `{}` (not forwarded as a string, not dropped).
    let evicted = &collected_tool_calls[0];
    assert_eq!(evicted.id, "call_aaa");
    assert_eq!(evicted.function.arguments, serde_json::json!({}));
}

#[tokio::test]
async fn normalize_chunk_errors_terminate_without_flushing_or_finalizing() {
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines(["start", "bad"]),
    };

    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream =
        send_compatible_streaming_request(client, req, ErrorAfterPendingToolCallProfile)
            .await
            .expect("stream should start");

    // The call's name fragment (which opens its block leniently) arrives
    // before the malformed frame is reached.
    match stream
        .next()
        .await
        .expect("expected tool call delta before normalize error")
        .expect("first item should be ok")
    {
        StreamEvent::BlockDelta {
            delta: Delta::ToolName { name },
            ..
        } => assert_eq!(name, "ping"),
        other => panic!("expected tool call name delta, got {other:?}"),
    }

    let err = stream
        .next()
        .await
        .expect("expected normalize error")
        .expect_err("second item should be the normalize error");
    assert_eq!(err.to_string(), "JsonError: normalize failed");

    // The malformed frame does not abort the stream; consumption continues
    // to EOF. The fully-delivered zero-arg tool call still flushes as
    // content, but with no `[DONE]` or finish reason the truncated stream
    // must not synthesize a terminal record.
    let mut saw_final = false;
    while let Some(item) = stream.next().await {
        match item.expect("post-error items should be ok") {
            StreamEvent::Final(_) => saw_final = true,
            StreamEvent::BlockEnd {
                end: BlockClose::ToolCall(_),
                block: Some(AssistantContent::ToolCall(_)),
                ..
            } => {}
            other => panic!("unexpected post-error stream item: {other:?}"),
        }
    }
    assert!(
        !saw_final,
        "a truncated stream must not synthesize a terminal record"
    );
}

#[tokio::test]
async fn distinct_same_name_tool_calls_evict_by_id_when_a_new_call_starts() {
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            "first_start",
            "first_args",
            "second_start",
            "second_args",
            "finish",
        ]),
    };

    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream =
        send_compatible_streaming_request(client, req, DistinctToolCallEvictionProfile)
            .await
            .expect("stream should start");

    let mut collected_tool_calls = Vec::new();
    while let Some(item) = stream.next().await {
        if let Some(tool_call) = completed_tool_call(item.expect("stream item should be ok")) {
            collected_tool_calls.push(tool_call);
        }
    }

    assert_eq!(collected_tool_calls.len(), 2);
    assert_eq!(collected_tool_calls[0].id, "call_aaa");
    assert_eq!(collected_tool_calls[0].function.name, "search");
    assert_eq!(
        collected_tool_calls[0].function.arguments,
        serde_json::json!({"query":"one"})
    );
    assert_eq!(collected_tool_calls[1].id, "call_bbb");
    assert_eq!(collected_tool_calls[1].function.name, "search");
    assert_eq!(
        collected_tool_calls[1].function.arguments,
        serde_json::json!({"query":"two"})
    );
}

#[tokio::test]
async fn streaming_http_non_success_preserves_status_and_body() {
    use crate::test_utils::HttpErrorStreamingClient;

    let body = r#"{"error":{"type":"rate_limit","message":"slow down"}}"#;
    let client = HttpErrorStreamingClient::new(http::StatusCode::TOO_MANY_REQUESTS, body);
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, FinishReasonCleanupProfile)
        .await
        .expect("stream should start");

    let err = stream
        .next()
        .await
        .expect("stream should yield transport error")
        .expect_err("HTTP non-success should surface as a stream error");
    assert_eq!(
        err.to_string(),
        format!(
            "HttpError: Invalid status code {} with message: {}",
            http::StatusCode::TOO_MANY_REQUESTS,
            body
        )
    );
    assert_eq!(
        err.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(err.provider_response_body(), Some(body));
    assert_eq!(
        err.provider_response_json().expect("valid JSON body"),
        Some(serde_json::json!({
            "error": {
                "type": "rate_limit",
                "message": "slow down"
            }
        }))
    );
    assert!(
        stream.next().await.is_none(),
        "stream should terminate after HTTP non-success"
    );
}

#[tokio::test]
async fn streaming_in_band_error_envelope_preserves_full_payload() {
    use crate::providers::openai::send_compatible_streaming_request;
    use crate::test_utils::MockStreamingClient;

    let body = r#"{"error":{"message":"upstream unavailable","type":"server_error"}}"#;
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"content\":\"partial\",\"tool_calls\":[]}}],\"usage\":null}",
            body,
        ]),
    };
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, "openai")
        .await
        .expect("stream should start");

    // The text block opens, then the partial content arrives.
    let opened = stream
        .next()
        .await
        .expect("stream should open the text block")
        .expect("text block start should be ok");
    assert!(matches!(
        opened,
        StreamEvent::BlockStart {
            kind: BlockKind::Text { .. },
            ..
        }
    ));
    let first = stream
        .next()
        .await
        .expect("stream should yield partial content")
        .expect("partial content should be ok");
    assert!(matches!(
        first,
        StreamEvent::BlockDelta { delta: Delta::Text { text }, .. } if text == "partial"
    ));

    let err = match stream.next().await {
        Some(Err(err)) => err,
        Some(Ok(_)) => panic!("expected in-band provider error after partial content"),
        None => panic!("stream ended before in-band provider error"),
    };
    assert!(matches!(err, CompletionError::ProviderResponse(_)));
    assert_eq!(err.provider_response_status(), None);
    assert_eq!(err.provider_response_body(), Some(body));
    assert!(
        stream.next().await.is_none(),
        "stream should terminate after in-band provider error"
    );
}

#[tokio::test]
async fn streaming_mid_stream_http_non_success_preserves_status_and_body() {
    use crate::providers::openai::send_compatible_streaming_request;
    use crate::test_utils::SequencedStreamingHttpClient;

    let body = r#"{"error":{"message":"upstream unavailable"}}"#;
    let chunks = vec![
        Ok(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"content\":\"partial\",\"tool_calls\":[]}}],\"usage\":null}",
        ])),
        Err(http_client::Error::InvalidStatusCodeWithMessage(
            http::StatusCode::BAD_GATEWAY,
            body.to_string(),
        )),
    ];
    let client = SequencedStreamingHttpClient::new(chunks);
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, "openai")
        .await
        .expect("stream should start");

    // The text block opens, then the partial content arrives.
    let opened = stream
        .next()
        .await
        .expect("stream should open the text block")
        .expect("text block start should be ok");
    assert!(matches!(
        opened,
        StreamEvent::BlockStart {
            kind: BlockKind::Text { .. },
            ..
        }
    ));
    let first = stream
        .next()
        .await
        .expect("stream should yield partial content")
        .expect("partial content should be ok");
    assert!(matches!(
        first,
        StreamEvent::BlockDelta { delta: Delta::Text { text }, .. } if text == "partial"
    ));

    let err = match stream.next().await {
        Some(Err(err)) => err,
        Some(Ok(_)) => panic!("expected HTTP transport error after partial content"),
        None => panic!("stream ended before HTTP transport error"),
    };
    assert_eq!(
        err.provider_response_status(),
        Some(http::StatusCode::BAD_GATEWAY)
    );
    assert_eq!(err.provider_response_body(), Some(body));
    assert!(
        stream.next().await.is_none(),
        "stream should terminate after mid-stream HTTP non-success"
    );
}

#[tokio::test]
async fn streaming_http_non_success_json_parse_error_is_visible() {
    use crate::test_utils::HttpErrorStreamingClient;

    let client = HttpErrorStreamingClient::new(http::StatusCode::BAD_REQUEST, "not json");
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, FinishReasonCleanupProfile)
        .await
        .expect("stream should start");

    let Some(Err(err)) = stream.next().await else {
        panic!("expected HTTP transport error")
    };
    assert_eq!(err.provider_response_body(), Some("not json"));
    assert!(err.provider_response_json().is_err());
}

#[tokio::test]
async fn streaming_non_http_transport_error_stays_provider_error() {
    use crate::test_utils::SequencedStreamingHttpClient;

    use crate::providers::openai::send_compatible_streaming_request;

    let chunks = vec![Err(http_client::Error::InvalidContentType(
        http::HeaderValue::from_static("application/json"),
    ))];
    let client = SequencedStreamingHttpClient::new(chunks);
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, "openai")
        .await
        .expect("stream should start");

    let err = match stream.next().await {
        Some(Err(err)) => err,
        Some(Ok(_)) => panic!("expected non-HTTP transport error"),
        None => panic!("stream ended before transport error"),
    };
    assert_eq!(
        err.to_string(),
        "ProviderError: Invalid content type was returned: \"application/json\""
    );
    assert!(matches!(err, CompletionError::ProviderError(_)));
    // Rig-generated transport diagnostics are not provider response bodies.
    assert_eq!(err.provider_response_body(), None);
    assert_eq!(err.provider_response_status(), None);
}

#[tokio::test]
async fn tool_calls_finish_reason_surfaces_partial_argument_errors() {
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines(["start", "finish"]),
    };

    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, FinishReasonCleanupProfile)
        .await
        .expect("stream should start");

    let mut saw_final = false;
    let mut saw_tool_call = false;
    let mut errors = Vec::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(event) if is_tool_lifecycle(&event) => {}
            Ok(StreamEvent::Final(_)) => saw_final = true,
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(_)),
                ..
            }) => saw_tool_call = true,
            Ok(other) => {
                panic!("unexpected stream item while asserting finish-reason policy: {other:?}")
            }
            Err(error) => errors.push(error.to_string()),
        }
    }

    assert!(
        saw_final,
        "the malformed call error must not erase terminal metadata"
    );
    assert!(
        !saw_tool_call,
        "a malformed call must not be emitted as valid"
    );
    assert_eq!(errors.len(), 1, "the malformed completed call stays loud");
    assert!(
        errors[0].contains("tool call") && errors[0].contains("malformed JSON input"),
        "the error should identify malformed tool arguments: {}",
        errors[0]
    );
}

#[tokio::test]
async fn length_finish_reason_drops_partial_argument_payloads() {
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines(["start", "length_finish"]),
    };
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, FinishReasonCleanupProfile)
        .await
        .expect("stream should start");

    while let Some(item) = stream.next().await {
        match item.expect("length-truncated partial calls are tolerated") {
            event if is_tool_lifecycle(&event) => {}
            StreamEvent::Final(_) => {}
            // The dropped call's end finalized nothing.
            StreamEvent::BlockEnd {
                end: BlockClose::ToolCall(_),
                block: None,
                ..
            } => {}
            StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(_)),
                ..
            } => {
                panic!("a partial length-truncated call must not be emitted")
            }
            other => panic!("unexpected truncation stream item: {other:?}"),
        }
    }

    assert!(
        stream
            .snapshot()
            .iter()
            .all(|content| !matches!(content, crate::completion::AssistantContent::ToolCall(_)))
    );
    assert_eq!(
        stream
            .response
            .as_ref()
            .and_then(|response| response.finish_reason.clone()),
        Some(crate::completion::FinishReason::Length)
    );
}

#[tokio::test]
async fn length_finish_reason_drops_a_call_with_no_argument_tokens() {
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines(["empty_start", "length_finish"]),
    };
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, FinishReasonCleanupProfile)
        .await
        .expect("stream should start");

    while stream.next().await.is_some() {}

    assert!(
        stream
            .snapshot()
            .iter()
            .all(|content| !matches!(content, crate::completion::AssistantContent::ToolCall(_))),
        "a length-truncated empty argument slot must not become a tool invocation"
    );
    assert_eq!(
        stream
            .response
            .as_ref()
            .and_then(|response| response.finish_reason.clone()),
        Some(crate::completion::FinishReason::Length)
    );
}

#[tokio::test]
async fn tool_calls_finish_reason_keeps_a_deliberate_zero_argument_call() {
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines(["empty_start", "finish"]),
    };
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, FinishReasonCleanupProfile)
        .await
        .expect("stream should start");

    while stream.next().await.is_some() {}

    let snapshot = stream.snapshot();
    let calls = snapshot
        .iter()
        .filter_map(|content| match content {
            crate::completion::AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments, serde_json::json!({}));
}

#[tokio::test]
async fn transport_error_still_flushes_fully_delivered_tool_calls() {
    use crate::providers::openai::send_compatible_streaming_request;
    use crate::test_utils::SequencedStreamingHttpClient;

    // A fully-delivered tool call followed by a transport error: the tool
    // call is content and must flush BEFORE the error surfaces (so a
    // first-`Err`-stop consumer sees it), and the stream must end without
    // a terminal record.
    let chunks = vec![
        Ok(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"{\\\"x\\\":1}\"}}]}}],\"usage\":null}",
        ])),
        Err(http_client::Error::InvalidStatusCodeWithMessage(
            http::StatusCode::BAD_GATEWAY,
            r#"{"error":{"message":"upstream unavailable"}}"#.to_string(),
        )),
    ];
    let client = SequencedStreamingHttpClient::new(chunks);
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream = send_compatible_streaming_request(client, req, "openai")
        .await
        .expect("stream should start");

    let mut saw_error = false;
    let mut saw_final = false;
    let mut collected_tool_calls = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) if is_tool_lifecycle(&event) => {}
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            }) => {
                assert!(
                    !saw_error,
                    "the flushed tool call must arrive before the terminal error"
                );
                collected_tool_calls.push(tool_call);
            }
            Ok(StreamEvent::Final(_)) => saw_final = true,
            Ok(other) => panic!("unexpected stream item: {other:?}"),
            Err(_) => saw_error = true,
        }
        if saw_error {
            break;
        }
    }

    assert!(saw_error, "the transport failure must reach the consumer");
    assert_eq!(
        collected_tool_calls.len(),
        1,
        "the fully-delivered tool call must flush despite the transport error"
    );
    assert_eq!(collected_tool_calls[0].id, "call_123");
    assert_eq!(collected_tool_calls[0].function.name, "ping");
    assert_eq!(
        collected_tool_calls[0].function.arguments,
        serde_json::json!({"x": 1})
    );
    assert!(
        stream.next().await.is_none(),
        "nothing may follow the terminal error"
    );
    assert!(
        !saw_final,
        "an errored stream must not synthesize a terminal record"
    );
    assert!(stream.response.is_none());
}

#[tokio::test]
async fn bare_done_after_only_unparseable_frames_emits_no_terminal() {
    // Every frame fails to decode; the trailing `[DONE]` must not dress
    // the failure up as a successful, default-usage completion.
    let client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines(["bad", "bad", "[DONE]"]),
    };
    let req = http::Request::builder()
        .method("POST")
        .uri("http://localhost/v1/chat/completions")
        .body(Vec::new())
        .expect("request should build");

    let mut stream =
        send_compatible_streaming_request(client, req, ErrorAfterPendingToolCallProfile)
            .await
            .expect("stream should start");

    let mut error_count = 0;
    let mut saw_final = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::Final(_)) => saw_final = true,
            Ok(other) => panic!("unexpected stream item: {other:?}"),
            Err(_) => error_count += 1,
        }
    }

    assert_eq!(
        error_count, 2,
        "each unparseable frame must surface as an error item"
    );
    assert!(
        !saw_final,
        "a stream with no successfully decoded frame must not emit a terminal record"
    );
    assert!(stream.response.is_none());
}

/// Mistral truncates at its context ceiling with `model_length`, which is
/// the same truncation class as `length` — only the limit differs.
///
/// Not a cassette test: forcing the state needs a prompt padded to the
/// model's full context window, which would commit a ~145 KB fixture of
/// repeated filler to exercise one mapping arm. The shape below is the
/// live response recorded while confirming the bug against
/// `voxtral-small-latest` (`max_context_length` 32768):
/// `finish_reason: "model_length"` with
/// `usage {prompt_tokens: 32424, completion_tokens: 344, total_tokens: 32768}`
/// — generation stopped dead on the ceiling with 4096 output tokens still
/// budgeted.
#[test]
fn model_length_is_truncation_not_a_natural_stop() {
    assert_eq!(
        map_openai_finish_reason("model_length"),
        FinishReason::Length,
        "a turn cut off by the context window must be distinguishable from one that \
             simply had nothing more to say"
    );

    // The vocabulary it joins, and the fallback that still preserves an
    // unrecognized spelling verbatim.
    assert_eq!(map_openai_finish_reason("length"), FinishReason::Length);
    assert_eq!(map_openai_finish_reason("max_tokens"), FinishReason::Length);
    assert_eq!(map_openai_finish_reason("stop"), FinishReason::Stop);
    assert_eq!(
        map_openai_finish_reason("some_new_reason"),
        FinishReason::Other("some_new_reason".to_owned())
    );
}
