use super::*;
use crate::http_client::{HeaderMap, StatusCode};
use crate::providers::openai::OpenAIConfig;
use crate::ws_client::CloseFrame;
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

fn sample_response(status: &str) -> Value {
    json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "gpt-5.4",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 3
        },
        "output": [],
        "tools": []
    })
}

#[test]
fn warmup_options_serialize_generate_false() {
    let options = ResponsesWebSocketCreateOptions::warmup();
    let json = serde_json::to_value(options).expect("options should serialize");

    assert_eq!(json, json!({ "generate": false }));
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

/// A terminal event that states no response object still ends the turn:
/// nothing in it is needed to build a block (#2668).
#[test]
fn a_terminal_without_its_response_object_still_ends_the_turn() {
    let payload = json!({
        "type": "response.completed"
    });

    let event = parse_server_event(&payload.to_string())
        .expect("a sparse known event classifies")
        .expect("it is not skipped");
    assert!(event.is_terminal());
    assert_eq!(event.response_id(), None);
}

#[test]
fn terminal_response_requires_completed_status() {
    let completed = terminal_response_result(sample_response("completed"))
        .expect("completed response should succeed");
    assert_eq!(completed["id"], "resp_123");

    let failed = terminal_response_result(sample_response("failed"))
        .expect_err("failed response should error");
    assert!(failed.to_string().contains("failed response"));
}

#[test]
fn terminal_failed_response_with_error_preserves_raw_payload() {
    let mut response = sample_response("failed");
    response["error"] =
        json!({ "code": "server_error", "message": "the model failed to generate a response" });

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
    let Err(err) = terminal_response_result(sample_response("failed")) else {
        panic!("failed response should fail")
    };

    // No provider error object, so this is a Rig-authored diagnostic and exposes
    // no preserved provider response body.
    assert_eq!(err.provider_response_body(), None);
    assert!(err.to_string().contains("failed response"));
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
