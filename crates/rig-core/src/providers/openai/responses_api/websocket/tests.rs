use super::*;
use crate::http_client::{HeaderMap, StatusCode};
use crate::providers::openai::OpenAIConfig;
use crate::ws_client::CloseFrame;
use serde_json::json;

/// The wire a connection is opened for.
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

fn response(id: &str, status: &str) -> Value {
    json!({ "id": id, "object": "response", "status": status, "output": [] })
}

fn event(kind: &str, id: &str, status: &str) -> String {
    json!({ "type": kind, "response": response(id, status) }).to_string()
}

#[test]
fn a_warmup_wire_encodes_a_response_create_without_http_flags() {
    let wire = ResponsesSocket::new(test_wire("https://api.openai.com/v1")).warmup();
    let mut request = crate::completion::CompletionRequest::new("hi");
    request.additional_params = Some(json!({ "background": true, "stream": true }));
    let payload = wire.encode(request, Mode::Streaming).expect("encodes");
    let json = serde_json::to_value(&payload).expect("serializes");

    assert_eq!(json["type"], "response.create");
    assert_eq!(json["generate"], false);
    assert_eq!(json["model"], "gpt-5.4");
    assert!(json.get("stream").is_none(), "{json}");
    assert!(json.get("background").is_none(), "{json}");
}

/// A chaining wire shapes every turn's history as a stored continuation's.
#[test]
fn a_chaining_wire_continues_stored_state() {
    let wire = ResponsesSocket::new(test_wire("https://api.openai.com/v1"));
    let request = crate::completion::CompletionRequest::new("hi");
    assert!(!wire.continues_stored(&request));
    assert!(wire.clone().chaining().continues_stored(&request));
}

/// How the connection reads each event: what ends a turn, what is skipped,
/// what leaves the turn's response to be drained, and what chains.
#[test]
fn the_lifecycle_reader_ends_skips_and_chains() {
    let mut chain = Chain::default();
    let created = event("response.created", "resp_1", "in_progress");
    assert!(matches!(chain.read(&created), Lifecycle::Frame));
    let delta = json!({ "type": "response.output_text.delta", "delta": "x" }).to_string();
    assert!(matches!(chain.read(&delta), Lifecycle::Frame));
    // Another response's terminal is never this turn's end.
    let foreign = event("response.completed", "resp_0", "completed");
    assert!(matches!(chain.read(&foreign), Lifecycle::Corrupt(_)));
    let completed = event("response.completed", "resp_1", "completed");
    assert!(matches!(
        chain.read(&completed),
        Lifecycle::Last {
            frame: None,
            settled: true
        }
    ));
    assert_eq!(chain.previous.get().as_deref(), Some("resp_1"));
    // The previous turn's trailing `response.done` is skipped.
    let late = event("response.done", "resp_1", "completed");
    assert!(matches!(chain.read(&late), Lifecycle::Skip));

    // A `response.done` that ends a turn is lowered to its terminal event.
    let done = event("response.done", "resp_2", "incomplete");
    let Lifecycle::Last {
        frame: Some(frame),
        settled: true,
    } = chain.read(&done)
    else {
        panic!("a response.done ends its turn")
    };
    let frame: Value = serde_json::from_str(&frame).expect("json");
    assert_eq!(frame["type"], "response.incomplete");
    assert_eq!(frame["response"]["id"], "resp_2");
    assert_eq!(chain.previous.get().as_deref(), Some("resp_2"));

    // Before a response opened, an error is the turn's settled end.
    let error = json!({ "type": "error", "error": { "code": "server_error" } }).to_string();
    assert!(matches!(
        chain.read(&error),
        Lifecycle::Last { settled: true, .. }
    ));
    // After one opened, it leaves that response to be drained: another
    // response's events neither end nor reopen it, its own terminal does.
    assert!(matches!(chain.read(&created), Lifecycle::Frame));
    assert!(matches!(
        chain.read(&error),
        Lifecycle::Last { settled: false, .. }
    ));
    assert_eq!(chain.previous.get(), None);
    let other = event("response.in_progress", "resp_9", "in_progress");
    assert!(matches!(chain.read(&other), Lifecycle::Corrupt(_)));
    assert!(matches!(chain.read(&foreign), Lifecycle::Corrupt(_)));
    assert!(matches!(
        chain.read(&completed),
        Lifecycle::Last { settled: true, .. }
    ));

    // A frame with no type is no event of the turn.
    assert!(matches!(chain.read("{}"), Lifecycle::Corrupt(_)));
    let pending = event("response.done", "resp_3", "in_progress");
    assert!(matches!(chain.read(&pending), Lifecycle::Fail(_)));
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
