use super::*;
use crate::http_client::{HeaderMap, StatusCode};
use crate::providers::openai::responses_api::{
    IncompleteDetailsReason, ResponseError, ResponseObject,
};
use crate::ws_client::CloseFrame;
use serde_json::json;

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

    assert!(matches!(error, CompletionError::ProviderResponse(_)));
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
/// and stays a plain diagnostic.
#[test]
fn websocket_provider_error_leaves_a_transport_failure_alone() {
    let error = websocket_provider_error(http_client::Error::StreamEnded);

    assert!(matches!(error, CompletionError::ProviderError(_)));
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

#[test]
fn warmup_options_serialize_generate_false() {
    let options = ResponsesWebSocketCreateOptions::warmup();
    let json = serde_json::to_value(options).expect("options should serialize");

    assert_eq!(json, json!({ "generate": false }));
}

/// The handshake request carries the endpoint path and the client's auth
/// headers, on the websocket scheme.
#[test]
fn websocket_request_targets_the_responses_endpoint_with_the_clients_headers() {
    let mut headers = HeaderMap::new();
    headers.insert(
        http::header::AUTHORIZATION,
        "Bearer test-key".parse().expect("header should parse"),
    );

    let request =
        websocket_request("https://api.openai.com/v1", &headers).expect("request should build");

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
    let error = websocket_request("ftp://api.openai.com/v1", &HeaderMap::new())
        .expect_err("ftp is not a websocket base");
    assert!(
        error.to_string().contains("ftp"),
        "the error should name the scheme, got {error}"
    );
}

#[test]
fn parse_done_event_exposes_response_id() {
    let payload = json!({
        "type": "response.done",
        "response": {
            "id": "resp_done_1",
            "status": "completed"
        }
    });

    let event = parse_server_event(&payload.to_string())
        .expect("done event should deserialize")
        .expect("done event should not be skipped");

    assert!(matches!(
        event,
        ResponsesWebSocketEvent::Done(ResponsesWebSocketDoneEvent { .. })
    ));
    assert_eq!(event.response_id(), Some("resp_done_1"));
    assert!(event.is_terminal());
}

#[test]
fn parse_response_completed_event_is_terminal() {
    let payload = json!({
        "type": "response.completed",
        "sequence_number": 12,
        "response": {
            "id": "resp_completed_1",
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
    });

    let event = parse_server_event(&payload.to_string())
        .expect("response event should deserialize")
        .expect("response event should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Response(_)));
    assert!(event.is_terminal());
    assert_eq!(event.response_id(), Some("resp_completed_1"));
}

#[test]
fn parse_live_output_item_added_event() {
    let payload = json!({
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
    });

    let event = parse_server_event(&payload.to_string())
        .expect("output item event should parse")
        .expect("output item event should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
}

#[test]
fn parse_live_content_part_added_event() {
    let payload = json!({
        "type": "response.content_part.added",
        "content_index": 0,
        "item_id": "msg_036471c3a72c147b0069ae7848d68881959773fd2d99e3d98a",
        "output_index": 0,
        "part": {
            "type": "output_text",
            "annotations": [],
            "logprobs": [],
            "text": ""
        },
        "sequence_number": 3
    });

    let event = parse_server_event(&payload.to_string())
        .expect("content part event should parse")
        .expect("content part event should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
}

#[test]
fn parse_live_output_text_delta_event() {
    let payload = json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "Web",
        "item_id": "msg_023af0f0a91bc2a90069ae788612e881958345bb156915ba29",
        "logprobs": [],
        "obfuscation": "2YYErYq7jkqqM",
        "output_index": 0,
        "sequence_number": 4
    });

    let event = parse_server_event(&payload.to_string())
        .expect("output text delta event should parse")
        .expect("output text delta event should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
}

#[test]
fn parse_reasoning_text_delta_event_is_item() {
    let payload = json!({
        "type": "response.reasoning_text.delta",
        "item_id": "rs_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 1,
        "delta": "thinking",
    });

    let event = parse_server_event(&payload.to_string())
        .expect("reasoning delta should parse")
        .expect("reasoning delta should not be skipped");

    assert!(matches!(event, ResponsesWebSocketEvent::Item(_)));
    assert!(!event.is_terminal());
}

#[test]
fn unknown_event_type_is_forwarded_raw() {
    let payload = json!({
        "type": "response.some_future_event",
        "data": "hello"
    });

    let result = parse_server_event(&payload.to_string()).expect("unknown event should not error");
    // Semantically skipped, but carried verbatim so the streaming surface
    // can yield it on the `StreamEvent::Unknown` passthrough.
    match result {
        Some(ResponsesWebSocketEvent::Unknown(value)) => assert_eq!(value, payload.into()),
        other => panic!("expected the raw Unknown passthrough event, got {other:?}"),
    }
}

#[test]
fn malformed_known_event_returns_error() {
    let payload = json!({
        "type": "response.completed"
    });

    let error =
        parse_server_event(&payload.to_string()).expect_err("malformed known event should error");
    assert!(
        error.to_string().contains("StreamingCompletionChunk"),
        "expected strict decode failure, got {error}"
    );
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
