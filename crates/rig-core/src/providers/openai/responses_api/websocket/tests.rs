use super::*;
use crate::http_client::{HeaderMap, StatusCode};
use crate::providers::openai::OpenAIConfig;
use crate::providers::openai::responses_api::{
    IncompleteDetailsReason, ResponseError, ResponseObject, ResponsesUsage,
};
use crate::ws_client::{CloseFrame, WebSocketConnection};
use serde_json::json;

/// The wire a session is opened over.
fn test_wire(base_url: &str) -> Responses {
    OpenAIConfig::new("test-key")
        .with_base_url(base_url)
        .responses("gpt-5.4")
}

/// The shape a rejected upgrade reaches this module in: the backend has
/// already read the status, headers and body off the refusing HTTP
/// response.
fn rejection(
    status: u16,
    request_id: Option<&str>,
    body: Option<&str>,
    extra_headers: &[(&str, &str)],
) -> http_client::Error {
    let mut headers = HeaderMap::new();
    if let Some(request_id) = request_id {
        headers.insert(
            "x-request-id",
            request_id.parse().expect("header value should be valid"),
        );
    }
    for (name, value) in extra_headers {
        headers.insert(
            http::HeaderName::from_bytes(name.as_bytes()).expect("header name should be valid"),
            value.parse().expect("header value should be valid"),
        );
    }
    http_client::Error::non_success_with_details(
        StatusCode::from_u16(status).expect("status should be valid"),
        headers,
        body.unwrap_or_default().to_string(),
    )
}

/// The live shape, recorded in
/// `websocket_error_identity_matrix/handshake_rejection_carries_status_body_and_request_id`.
const REJECTION_BODY: &str = r#"{"error":{"message":"Incorrect API key provided: sk-inval***-key.","type":"invalid_request_error","code":"invalid_api_key","param":null},"status":401}"#;

#[test]
fn websocket_provider_error_preserves_status_body_and_request_id() {
    let error = websocket_provider_error(rejection(
        401,
        Some("req_websocket_1"),
        Some(REJECTION_BODY),
        &[],
    ));

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::UNAUTHORIZED)
    );
    assert_eq!(error.provider_response_body(), Some(REJECTION_BODY));
    assert_eq!(error.provider_request_id(), Some("req_websocket_1"));
    assert_eq!(
        error
            .provider_response_json()
            .expect("body should be valid JSON")
            .expect("parsed JSON should be present")["error"]["code"],
        "invalid_api_key"
    );
}

/// The id is optional everywhere else in this crate and is optional here:
/// its absence must not cost the status or the body.
#[test]
fn websocket_provider_error_without_a_request_id_keeps_the_rest() {
    let error = websocket_provider_error(rejection(401, None, Some(REJECTION_BODY), &[]));

    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::UNAUTHORIZED)
    );
    assert_eq!(error.provider_response_body(), Some(REJECTION_BODY));
    assert_eq!(error.provider_request_id(), None);
}

#[test]
fn websocket_provider_error_treats_an_empty_request_id_as_absent() {
    let error = websocket_provider_error(rejection(401, Some(""), Some(REJECTION_BODY), &[]));

    assert_eq!(error.provider_request_id(), None);
    assert_eq!(error.provider_response_body(), Some(REJECTION_BODY));
}

#[test]
fn websocket_provider_error_without_a_body_keeps_the_status() {
    let error = websocket_provider_error(rejection(503, Some("req_websocket_2"), None, &[]));

    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_request_id(), Some("req_websocket_2"));
    // An empty preserved body is `Some("")`, not `None`: the provider
    // answered, it just said nothing.
    assert_eq!(error.provider_response_body(), Some(""));
}

/// Every status a refused upgrade can carry, **including 2xx and 3xx**:
/// tungstenite raises `Error::Http` for any non-101 response, and
/// `connect_async` does not follow redirects, so a `200` or a `302` reaches
/// this mapping exactly as a `401` does. Classification here follows the
/// call path, not the status class, so those two must survive too.
#[test]
fn websocket_provider_error_preserves_every_rejection_status() {
    for status in [200u16, 302, 400, 401, 403, 404, 429, 500, 503] {
        let error = websocket_provider_error(rejection(status, None, Some("{}"), &[]));
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::from_u16(status).expect("status should be valid")),
            "status {status} should survive"
        );
    }
}

/// A `429` upgrade carries the same rate-limit metadata its HTTP twin
/// does, and a caller that has to back off needs it (rig#2210).
#[test]
fn websocket_provider_error_preserves_the_rejections_headers() {
    let error = websocket_provider_error(rejection(
        429,
        Some("req_websocket_3"),
        Some("{}"),
        &[("retry-after", "20"), ("x-ratelimit-remaining", "0")],
    ));

    let headers = error
        .provider_response_headers()
        .expect("headers should be preserved");
    assert_eq!(
        headers.get("retry-after").and_then(|v| v.to_str().ok()),
        Some("20")
    );
    assert_eq!(
        headers
            .get("x-ratelimit-remaining")
            .and_then(|v| v.to_str().ok()),
        Some("0")
    );
    // The id is read before the map is consumed.
    assert_eq!(error.provider_request_id(), Some("req_websocket_3"));
}

/// A failure that never reached the provider has no response to preserve
/// and stays a transport error, with the transport table's retryability.
#[test]
fn websocket_provider_error_leaves_a_transport_failure_alone() {
    let error = websocket_provider_error(http_client::Error::StreamEnded);

    assert!(matches!(error, ProviderError::Http(_)));
    assert!(error.is_retryable());
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_request_id(), None);
}

/// The regression this mapping exists for: a rejection must not flatten to
/// its display string (rig#2314, rig#2315).
#[test]
fn websocket_provider_error_no_longer_flattens_a_rejection_to_a_string() {
    let error = websocket_provider_error(rejection(
        401,
        Some("req_websocket_4"),
        Some(REJECTION_BODY),
        &[],
    ));

    assert!(
        error.provider_response_body().is_some(),
        "the provider's own body must survive, not just a display string"
    );
}

#[test]
fn websocket_error_event_preserves_provider_payload_as_json() {
    let mut extra = Map::new();
    extra.insert(
        "type".to_string(),
        Value::String("invalid_request_error".to_string()),
    );
    let event = ResponsesWebSocketErrorEvent {
        kind: ResponsesWebSocketErrorEventKind::Error,
        error: ResponsesWebSocketErrorPayload {
            code: Some("rate_limit_exceeded".to_string()),
            message: Some("slow down".to_string()),
            extra,
        },
    };

    let err = provider_error_from_event(&event);

    // No HTTP status on the websocket stream, and the raw payload round-trips
    // through provider_response_json() (code + message + extra all preserved).
    assert_eq!(err.provider_response_status(), None);
    let json = err
        .provider_response_json()
        .expect("preserved body should be valid JSON")
        .expect("provider response body should be present");
    assert_eq!(json["error"]["code"], "rate_limit_exceeded");
    assert_eq!(json["error"]["message"], "slow down");
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

fn sample_response(status: ResponseStatus) -> CompletionResponse {
    CompletionResponse {
        id: "resp_123".to_string(),
        object: ResponseObject::Response,
        provider_request_id: None,
        created_at: 0,
        status,
        error: None,
        incomplete_details: None,
        instructions: None,
        max_output_tokens: None,
        model: "gpt-5.4".to_string(),
        usage: Some(ResponsesUsage {
            input_tokens: 1,
            input_tokens_details: None,
            output_tokens: 2,
            output_tokens_details: Some(
                crate::providers::openai::responses_api::OutputTokensDetails {
                    reasoning_tokens: 0,
                },
            ),
            total_tokens: 3,
        }),
        output: Vec::new(),
        tools: Vec::new(),
        additional_parameters: Default::default(),
        provider_reasoning: None,
        reasoning_metadata: None,
        reasoning_context: None,
    }
}

/// A warmup wire says so on every event it encodes; no HTTP-only flag is
/// sent.
#[test]
fn a_warmup_wire_encodes_generate_false() {
    let wire = ResponsesSocket::new(test_wire("https://api.openai.com/v1")).warmup();
    let event = wire
        .encode(completion::CompletionRequest::new("hello"), Mode::Streaming)
        .expect("encodes");
    assert_eq!(event.generate(), Some(false));
    let json = serde_json::to_value(&event).expect("serializes");
    assert_eq!(json["type"], "response.create");
    assert_eq!(json["generate"], false);
    assert!(json.get("stream").is_none(), "{json}");
    assert!(json.get("route").is_none(), "{json}");
}

/// The handshake request carries the endpoint path and the wire's own auth
/// headers, on the websocket scheme.
#[test]
fn websocket_request_targets_the_responses_endpoint_with_the_wires_headers() {
    let request =
        websocket_request(&test_wire("https://api.openai.com/v1")).expect("request should build");

    assert_eq!(request.uri(), "wss://api.openai.com/v1/responses");
    assert_eq!(
        request
            .headers()
            .get(http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer test-key")
    );
}

#[test]
fn websocket_request_rejects_an_unsupported_base_url_scheme() {
    let error = websocket_request(&test_wire("ftp://api.openai.com/v1"))
        .expect_err("ftp is not a websocket base");
    assert!(
        error.to_string().contains("ftp"),
        "the error should name the scheme, got {error}"
    );
    // Building the handshake request is request building, not transport.
    let error = ProviderError::from(error);
    assert!(matches!(error, ProviderError::Request(_)), "{error:?}");
    assert!(!error.is_retryable());
}

/// A `response.done` for the response the last turn ended with is that
/// turn's trailer: skipped once, never read as this turn's end.
#[test]
fn the_previous_turns_trailing_done_is_skipped_once() {
    let mut chain = Chain::default();
    let completed = json!({
        "type": "response.completed",
        "response": serde_json::to_value(sample_response(ResponseStatus::Completed))
            .expect("serializes"),
    });
    assert!(matches!(
        chain.read(&completed.to_string()),
        Lifecycle::Last(_)
    ));
    let trailer = json!({
        "type": "response.done",
        "response": { "id": "resp_123", "status": "completed" },
    })
    .to_string();
    assert!(matches!(chain.read(&trailer), Lifecycle::Skip));
    // A second one is not a trailer: without a body it ends the turn.
    let Lifecycle::Fail(error) = chain.read(&trailer) else {
        panic!("a bodiless done that is not a trailer fails the turn");
    };
    assert!(
        error
            .to_string()
            .contains("before a terminal response body was available"),
        "{error}"
    );
}

fn completed_event(id: &str) -> Value {
    json!({
        "type": "response.completed",
        "sequence_number": 12,
        "response": {
            "id": id,
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.4",
            "usage": null,
            "output": [],
            "tools": []
        }
    })
}

fn text_delta_event() -> String {
    json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "Web",
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1
    })
    .to_string()
}

/// After streamed items the terminal event reaches the decoder as it
/// arrived, and continues the chain.
#[test]
fn a_completed_event_after_streamed_items_is_the_last_frame() {
    let mut chain = Chain::default();
    assert!(matches!(chain.read(&text_delta_event()), Lifecycle::Frame));
    assert!(matches!(
        chain.read(&completed_event("resp_completed_1").to_string()),
        Lifecycle::Last(None)
    ));
    assert_eq!(chain.previous.get().as_deref(), Some("resp_completed_1"));
    assert!(!chain.streamed, "the next turn starts unstreamed");
}

/// A turn that streamed no item states its output only in the terminal
/// body, which the decoder then reads whole.
#[test]
fn a_completed_event_after_no_items_is_lowered_to_its_body() {
    let mut chain = Chain::default();
    let created = json!({ "type": "response.created", "sequence_number": 0,
        "response": completed_event("resp_1")["response"].clone() });
    assert!(matches!(chain.read(&created.to_string()), Lifecycle::Frame));
    let Lifecycle::Last(Some(lowered)) = chain.read(&completed_event("resp_1").to_string()) else {
        panic!("a terminal after no items is lowered");
    };
    let lowered: Value = serde_json::from_str(&lowered).expect("the body is JSON");
    assert_eq!(lowered["id"], "resp_1");
    assert_eq!(lowered.get("type"), None);
    assert_eq!(chain.previous.get().as_deref(), Some("resp_1"));
}

/// Content events, from live traffic, pass to the decoder untouched and do
/// not touch the chain.
#[test]
fn content_events_are_frames_for_the_decoder() {
    let mut chain = chain_after("resp_before");
    for payload in [
        json!({
            "type": "response.output_item.added",
            "item": {
                "id": "msg_036471c3a72c147b0069ae7848d68881959773fd2d99e3d98a",
                "type": "message",
                "status": "in_progress",
                "content": [],
                "role": "assistant"
            },
            "output_index": 0,
            "sequence_number": 2
        }),
        json!({
            "type": "response.content_part.added",
            "content_index": 0,
            "item_id": "msg_036471c3a72c147b0069ae7848d68881959773fd2d99e3d98a",
            "output_index": 0,
            "part": { "type": "output_text", "annotations": [], "logprobs": [], "text": "" },
            "sequence_number": 3
        }),
        json!({
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": "Web",
            "item_id": "msg_023af0f0a91bc2a90069ae788612e881958345bb156915ba29",
            "logprobs": [],
            "obfuscation": "2YYErYq7jkqqM",
            "output_index": 0,
            "sequence_number": 4
        }),
        json!({
            "type": "response.reasoning_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "thinking",
        }),
        json!({ "type": "response.some_future_event", "data": "hello" }),
    ] {
        assert!(
            matches!(chain.read(&payload.to_string()), Lifecycle::Frame),
            "{payload}"
        );
    }
    assert_eq!(chain.previous.get().as_deref(), Some("resp_before"));
}

/// A terminal whose body does not decode still ends the turn: the decoder
/// then reports it, the connection stays in step, and nothing is chained.
#[test]
fn a_malformed_terminal_is_the_last_frame_and_ends_the_chain() {
    let mut chain = chain_after("resp_before");
    let payload = json!({ "type": "response.completed", "response": { "id": "resp_bad" } });
    assert!(matches!(
        chain.read(&payload.to_string()),
        Lifecycle::Last(None)
    ));
    assert_eq!(chain.previous.get(), None);
    let bodiless = json!({ "type": "response.completed" }).to_string();
    assert!(matches!(chain.read(&bodiless), Lifecycle::Last(None)));
    assert_eq!(chain.previous.get(), None);
}

/// Every websocket event names its type; a frame that does not is no part
/// of a turn and fails it rather than ending it as a whole body, leaving
/// the rest of the turn to be drained.
#[test]
fn a_frame_without_a_type_fails_the_turn_and_the_chain() {
    let mut chain = chain_after("resp_before");
    let body =
        serde_json::to_value(sample_response(ResponseStatus::Completed)).expect("serializes");
    assert!(matches!(
        chain.read(&body.to_string()),
        Lifecycle::Corrupt(_)
    ));
    assert!(matches!(chain.read("not json"), Lifecycle::Corrupt(_)));
    assert_eq!(chain.previous.get(), None);
}

/// A `response.completed` whose response failed fails the turn, and the
/// failed response is never chained.
#[test]
fn a_completed_event_with_a_failed_response_ends_the_chain() {
    let mut chain = chain_after("resp_before");
    let payload = json!({
        "type": "response.completed",
        "response": serde_json::to_value(sample_response(ResponseStatus::Failed))
            .expect("serializes"),
    });
    assert!(matches!(
        chain.read(&payload.to_string()),
        Lifecycle::Fail(_)
    ));
    assert_eq!(chain.previous.get(), None);
}

/// A chain whose last turn produced `previous`.
fn chain_after(previous: &str) -> Chain {
    let chain = Chain::default();
    chain.previous.set(Some(previous.to_string()));
    chain
}

fn created_event(id: &str) -> String {
    json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": { "id": id, "status": "in_progress" },
    })
    .to_string()
}

/// Once a turn has named its response, a terminal event for another one is
/// not its end: the turn fails and the rest of it is drained.
#[test]
fn a_terminal_for_another_response_fails_the_turn_for_draining() {
    for kind in [
        "response.completed",
        "response.incomplete",
        "response.failed",
    ] {
        let mut chain = chain_after("resp_before");
        assert!(matches!(
            chain.read(&created_event("resp_2")),
            Lifecycle::Frame
        ));
        let stale = json!({
            "type": kind,
            "sequence_number": 4,
            "response": serde_json::to_value(sample_response(ResponseStatus::Completed))
                .expect("serializes"),
        });
        assert!(
            matches!(chain.read(&stale.to_string()), Lifecycle::Corrupt(_)),
            "{kind}"
        );
        assert_eq!(chain.previous.get(), None, "{kind}");
        // The turn's own response is still the one it waits for.
        assert_eq!(chain.current_response_id.as_deref(), Some("resp_2"));
    }
}

/// An `error` after the turn's response opened may leave that response
/// sending: the turn fails and the rest of it is drained. Before it opened,
/// the error is the turn's whole answer.
#[test]
fn an_error_after_the_response_opened_leaves_it_to_be_drained() {
    let error = json!({ "type": "error", "error": { "code": "server_error", "message": "boom" } })
        .to_string();
    let mut chain = chain_after("resp_before");
    assert!(matches!(
        chain.read(&created_event("resp_1")),
        Lifecycle::Frame
    ));
    assert!(matches!(chain.read(&error), Lifecycle::Corrupt(_)));
    assert_eq!(chain.previous.get(), None);

    let mut chain = chain_after("resp_before");
    assert!(matches!(chain.read(&error), Lifecycle::Fail(_)));
    assert_eq!(chain.previous.get(), None);
}

/// A `response.done` for a response still in progress ends the turn in
/// failure, and ends the chain.
#[test]
fn a_done_event_still_in_progress_ends_the_chain() {
    let mut chain = chain_after("resp_before");
    let payload = json!({
        "type": "response.done",
        "response": serde_json::to_value(sample_response(ResponseStatus::InProgress))
            .expect("serializes"),
    });
    assert!(matches!(
        chain.read(&payload.to_string()),
        Lifecycle::Fail(_)
    ));
    assert_eq!(chain.previous.get(), None);
}

/// Once a turn has named its response, a `response.done` for another one
/// belongs to another turn, even after the trailer filter was cleared.
#[test]
fn a_done_event_for_another_response_is_not_this_turns_end() {
    let mut chain = Chain::default();
    let created = json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": { "id": "resp_2", "status": "in_progress" },
    });
    assert!(matches!(chain.read(&created.to_string()), Lifecycle::Frame));
    let stray = json!({
        "type": "response.done",
        "response": serde_json::to_value(sample_response(ResponseStatus::Completed))
            .expect("serializes"),
    });
    assert!(matches!(chain.read(&stray.to_string()), Lifecycle::Skip));
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 3,
        "response": serde_json::to_value(CompletionResponse {
            id: "resp_2".to_string(),
            ..sample_response(ResponseStatus::Completed)
        })
        .expect("serializes"),
    });
    assert!(matches!(
        chain.read(&completed.to_string()),
        Lifecycle::Last(_)
    ));
    assert_eq!(chain.previous.get().as_deref(), Some("resp_2"));
}

#[test]
fn a_failed_event_ends_the_turn_and_the_chain() {
    let mut chain = chain_after("resp_before");
    let payload = json!({
        "type": "response.failed",
        "response": serde_json::to_value(sample_response(ResponseStatus::Failed))
            .expect("serializes"),
    });
    let Lifecycle::Fail(error) = chain.read(&payload.to_string()) else {
        panic!("a failed event fails the turn");
    };
    assert!(error.to_string().contains("failed response"), "{error}");
    assert_eq!(chain.previous.get(), None);
    assert_eq!(chain.pending_done_response_id.as_deref(), Some("resp_123"));
}

/// After streamed items, a `response.done` carrying the response ends the
/// turn as `response.completed` would, so the items are not stated twice.
#[test]
fn a_done_event_after_streamed_items_ends_the_turn_as_completed() {
    let mut chain = Chain::default();
    assert!(matches!(chain.read(&text_delta_event()), Lifecycle::Frame));
    let body =
        serde_json::to_value(sample_response(ResponseStatus::Completed)).expect("serializes");
    let payload = json!({ "type": "response.done", "response": body });
    let Lifecycle::Last(Some(lowered)) = chain.read(&payload.to_string()) else {
        panic!("a done event with a body is the last frame");
    };
    let lowered: Value = serde_json::from_str(&lowered).expect("the event is JSON");
    assert_eq!(lowered["type"], "response.completed");
    assert_eq!(lowered["response"]["id"], "resp_123");
}

/// A `response.done` carrying the response is lowered to it: the decoder's
/// whole-body shape.
#[test]
fn a_done_event_with_a_body_is_lowered_to_the_body() {
    let mut chain = Chain::default();
    let body =
        serde_json::to_value(sample_response(ResponseStatus::Completed)).expect("serializes");
    let payload = json!({ "type": "response.done", "response": body });
    let Lifecycle::Last(Some(lowered)) = chain.read(&payload.to_string()) else {
        panic!("a done event with a body is the last frame, lowered");
    };
    let lowered: Value = serde_json::from_str(&lowered).expect("the body is JSON");
    assert_eq!(lowered["id"], "resp_123");
    assert_eq!(lowered.get("type"), None);
    assert_eq!(chain.previous.get().as_deref(), Some("resp_123"));
}

#[test]
fn terminal_response_requires_completed_status() {
    let completed = terminal_response_result(sample_response(ResponseStatus::Completed))
        .expect("completed response should succeed");
    assert_eq!(completed.id, "resp_123");

    let failed = terminal_response_result(sample_response(ResponseStatus::Failed))
        .expect_err("failed response should error");
    assert!(failed.to_string().contains("failed response"));
}

#[test]
fn terminal_failed_response_with_error_preserves_raw_payload() {
    let mut response = sample_response(ResponseStatus::Failed);
    response.error = Some(ResponseError {
        code: "server_error".to_string(),
        message: "the model failed to generate a response".to_string(),
    });

    let Err(err) = terminal_response_result(response) else {
        panic!("failed response with an error object should fail")
    };

    // The full failed-response envelope is preserved as a ProviderResponse with
    // no HTTP status (the websocket stream carries none), so the raw JSON parses
    // back with the provider error nested under `error` — proving the whole
    // envelope is kept, not just the error object.
    assert_eq!(err.provider_response_status(), None);

    let json = err
        .provider_response_json()
        .expect("preserved body should parse as JSON")
        .expect("preserved body should not be empty");
    assert_eq!(
        json["error"]["message"],
        "the model failed to generate a response"
    );
    assert_eq!(json["error"]["code"], "server_error");
}

#[test]
fn terminal_failed_response_without_error_is_rig_diagnostic() {
    let Err(err) = terminal_response_result(sample_response(ResponseStatus::Failed)) else {
        panic!("failed response should fail")
    };

    // No provider error object, so this is a Rig-authored diagnostic and exposes
    // no preserved provider response body.
    assert_eq!(err.provider_response_body(), None);
    assert!(err.to_string().contains("failed response"));
}

/// An incomplete terminal is a success, not a failure: the partial output
/// and usage are kept and normalization maps the status downstream.
#[test]
fn terminal_incomplete_response_is_a_terminal_success() {
    let mut response = sample_response(ResponseStatus::Incomplete);
    response.incomplete_details = Some(IncompleteDetailsReason {
        reason: "max_output_tokens".to_string(),
    });

    let response = terminal_response_result(response).expect("incomplete is a terminal");
    assert!(matches!(response.status, ResponseStatus::Incomplete));
}

/// A close frame mid-turn is an error naming the peer's reason; a keepalive
/// is skipped without ending the turn.
#[test]
fn websocket_frame_to_text_maps_control_frames() {
    assert_eq!(
        websocket_frame_to_text(Frame::Text("{}".to_string())).expect("text frame"),
        Some("{}".to_string())
    );
    assert_eq!(
        websocket_frame_to_text(Frame::Ping(bytes::Bytes::new())).expect("ping is skipped"),
        None
    );

    let error = websocket_frame_to_text(Frame::Close(Some(CloseFrame {
        code: 1011,
        reason: "server restarting".to_string(),
    })))
    .expect_err("a close frame ends the turn");
    assert!(
        error.to_string().contains("server restarting"),
        "the peer's reason should surface, got {error}"
    );

    let error = websocket_frame_to_text(Frame::Close(None))
        .expect_err("a reasonless close still ends the turn");
    assert!(error.to_string().contains("without a close reason"));
}

// Observation: a turn is an attempt on the bus.

/// A connection that answers the first write with `frames`, then stalls.
struct Scripted(std::collections::VecDeque<Frame>, bool);

impl WebSocketConnection for Scripted {
    fn send(
        &mut self,
        _frame: Frame,
    ) -> crate::wasm_compat::WasmBoxedFuture<'_, http_client::Result<()>> {
        self.1 = true;
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(
        &mut self,
    ) -> crate::wasm_compat::WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        match self.0.pop_front().filter(|_| self.1) {
            Some(frame) => Box::pin(std::future::ready(Ok(Some(frame)))),
            None => Box::pin(std::future::pending()),
        }
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> crate::wasm_compat::WasmBoxedFuture<'_, http_client::Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

/// A key no pattern would recognize: only the handshake names it secret.
const SECRET_KEY: &str = "observed-7f3a9c-credential";

/// A model over a scripted connection that scrubs the handshake's key.
fn observed_model(
    frames: impl IntoIterator<Item = String>,
    event_timeout: Option<Duration>,
) -> Model<ResponsesSocket, ResponsesWebSocket> {
    let wire = OpenAIConfig::new(SECRET_KEY)
        .with_base_url("https://api.openai.com/v1")
        .responses("gpt-5.4");
    let handshake = websocket_request(&wire).expect("handshake builds");
    let connection = Scripted(frames.into_iter().map(Frame::Text).collect(), false);
    let transport = ResponsesWebSocket::from_connection(Box::new(connection))
        .scrubbing(crate::observe::handshake_secrets(&handshake))
        .event_timeout(event_timeout);
    Model::new(ResponsesSocket::new(wire), transport)
}

/// Every adapter event `run` records, in order.
async fn adapter_events<F, Fut>(run: F) -> Vec<crate::observe::AdapterEvent>
where
    F: FnOnce(crate::observe::AdapterContext) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let log = Arc::new(crate::observe::ObservationLog::default());
    let context = crate::observe::AdapterContext::new(
        log.clone(),
        crate::observe::Subject::default(),
        "websocket",
    );
    run(context).await;
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            crate::observe::Action::Adapter { observation } => Some(observation.event.clone()),
            _ => None,
        })
        .collect()
}

/// Drain one observed stream of `model`.
async fn observe_turn<W, T>(model: &Model<W, T>) -> Vec<crate::observe::AdapterEvent>
where
    W: Wire<Op = Completion>,
    T: Transport<W>,
{
    adapter_events(|context| async move {
        let mut stream = model
            .stream_observed(completion::CompletionRequest::new("hello"), context)
            .expect("stream opens");
        while futures::StreamExt::next(&mut stream).await.is_some() {}
    })
    .await
}

/// The Responses conformance fixture's frames, as websocket messages.
fn fixture_messages() -> Vec<String> {
    let fixture = crate::test_utils::streaming_conformance::fixtures::openai_responses::fixture();
    fixture
        .text_frames
        .iter()
        .chain(&fixture.tool_call_frames)
        .chain(&fixture.terminal_frames)
        .filter_map(|frame| frame.as_bytes().cloned())
        .flat_map(|bytes| {
            String::from_utf8_lossy(&bytes)
                .lines()
                .filter_map(|line| line.strip_prefix("data:").map(str::trim))
                .filter(|data| !data.is_empty() && *data != "[DONE]")
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn an_observed_turn_has_the_shape_of_an_observed_http_stream() {
    use crate::observe::{AdapterEnding, AdapterEvent};

    let messages = fixture_messages();
    let body: String = messages
        .iter()
        .map(|message| format!("data: {message}\n\n"))
        .collect();
    let http = OpenAIConfig::new(SECRET_KEY)
        .connect(crate::test_utils::SequencedStreamingHttpClient::new(vec![
            Ok(bytes::Bytes::from(body)),
        ]))
        .responses("gpt-5.4");
    let over_http = observe_turn(&http).await;
    let over_websocket = observe_turn(&observed_model(messages, None)).await;

    // The same facts in the same order. Only the method and status differ:
    // a turn is a message on a connection a GET upgraded, not a POST.
    let shape = |events: &[AdapterEvent]| -> Vec<AdapterEvent> {
        events
            .iter()
            .cloned()
            .map(|event| match event {
                AdapterEvent::Started { route, .. } => AdapterEvent::Started {
                    method: String::new(),
                    route,
                },
                AdapterEvent::Response { .. } => AdapterEvent::Response { status: 0 },
                event => event,
            })
            .collect()
    };
    assert_eq!(shape(&over_websocket), shape(&over_http));
    assert!(
        matches!(
            over_websocket.first(),
            Some(AdapterEvent::Started { method, route }) if method == "GET" && route == "/responses"
        ),
        "{over_websocket:?}"
    );
    assert!(over_websocket.contains(&AdapterEvent::Response { status: 101 }));
    assert!(
        over_websocket
            .iter()
            .any(|event| matches!(event, AdapterEvent::Usage { .. })),
        "the projector read the terminal's usage: {over_websocket:?}"
    );
    assert_eq!(
        over_websocket.last(),
        Some(&AdapterEvent::Finished {
            ending: AdapterEnding::Terminal
        })
    );
}

#[tokio::test]
async fn an_observed_turn_scrubs_the_handshake_credentials() {
    use crate::observe::{AdapterEnding, AdapterEvent};

    let error = json!({
        "type": "error",
        "error": {
            "code": "invalid_api_key",
            "message": format!("Incorrect API key provided: {SECRET_KEY}"),
        },
    });
    let events = observe_turn(&observed_model([error.to_string()], None)).await;
    let envelope = events
        .iter()
        .find_map(|event| match event {
            AdapterEvent::ErrorEnvelope { error } => Some(error.clone()),
            _ => None,
        })
        .expect("the error envelope is projected");
    let recorded = serde_json::to_string(&envelope).expect("serializes");
    assert!(!recorded.contains(SECRET_KEY), "{recorded}");
    assert!(
        matches!(
            events.last(),
            Some(AdapterEvent::Finished {
                ending: AdapterEnding::Error { .. }
            })
        ),
        "{events:?}"
    );
}

#[tokio::test]
async fn a_connection_failure_closes_the_attempt_at_the_transport() {
    use crate::observe::{AdapterEnding, AdapterErrorBoundary, AdapterEvent};

    let events = observe_turn(&observed_model([], Some(Duration::from_millis(20)))).await;
    assert!(
        matches!(
            events.last(),
            Some(AdapterEvent::Finished {
                ending: AdapterEnding::Error {
                    boundary: AdapterErrorBoundary::Transport,
                    ..
                }
            })
        ),
        "{events:?}"
    );
}

#[tokio::test]
async fn an_unpolled_turn_reports_no_attempt() {
    let model = observed_model(fixture_messages(), None);
    let events = adapter_events(|context| async move {
        drop(
            model
                .stream_observed(completion::CompletionRequest::new("hello"), context)
                .expect("stream opens"),
        );
    })
    .await;
    assert!(events.is_empty(), "{events:?}");
}
