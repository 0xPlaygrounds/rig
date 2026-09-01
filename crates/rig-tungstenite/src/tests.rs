use super::*;
use rig_core::http_client::StatusCode;

/// The live shape of a rejected upgrade, recorded against the real
/// endpoint: status, request id, and the provider's error envelope.
const REJECTION_BODY: &str = r#"{"error":{"message":"Incorrect API key provided: sk-inval***-key.","type":"invalid_request_error","code":"invalid_api_key","param":null},"status":401}"#;

fn handshake_rejection(
    status: u16,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> tungstenite::Error {
    let mut response = http::Response::builder().status(status);
    for (name, value) in headers {
        response = response.header(*name, *value);
    }
    tungstenite::Error::Http(Box::new(
        response
            .body(body.map(|body| body.as_bytes().to_vec()))
            .expect("response should build"),
    ))
}

#[test]
fn a_rejected_upgrade_keeps_its_status_headers_and_body() {
    let error = from_tungstenite(handshake_rejection(
        401,
        &[("x-request-id", "req_websocket_1")],
        Some(REJECTION_BODY),
    ));

    assert!(
        matches!(&error, Error::InvalidStatusCodeWithDetails { status, body, headers }
            if *status == StatusCode::UNAUTHORIZED
                && body == REJECTION_BODY
                && headers.get("x-request-id").and_then(|v| v.to_str().ok())
                    == Some("req_websocket_1")),
        "the rejection must survive whole, got {error:?}"
    );
}

/// A `429` upgrade carries rate-limit metadata its caller needs (rig#2210).
#[test]
fn a_rejection_keeps_rate_limit_headers() {
    let error = from_tungstenite(handshake_rejection(
        429,
        &[("retry-after", "20"), ("x-ratelimit-remaining", "0")],
        Some("{}"),
    ));

    let headers = error
        .non_success_headers()
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
}

/// An upgrade refused without a body still carries its status, which is
/// more than the display string preserved.
#[test]
fn a_rejection_without_a_body_keeps_the_status() {
    let error = from_tungstenite(handshake_rejection(503, &[], None));

    assert_eq!(
        error.non_success_status(),
        Some(StatusCode::SERVICE_UNAVAILABLE)
    );
}

#[test]
fn every_rejection_status_survives() {
    for status in [400u16, 401, 403, 404, 429, 500, 503] {
        let error = from_tungstenite(handshake_rejection(status, &[], Some("{}")));
        assert_eq!(
            error.non_success_status(),
            Some(StatusCode::from_u16(status).expect("status should be valid")),
            "status {status} should survive"
        );
    }
}

/// A failure that never reached the provider has no response to preserve —
/// for **every** non-`Http` variant, not just the one that is easy to
/// construct. `Http` is the only variant carrying a provider response, so
/// this enumeration is what pins that boundary; a new variant mapped to a
/// non-success status by accident would invent a provider answer that
/// never existed.
///
/// `Tls` is absent because it cannot be constructed without a TLS backend
/// in scope, and it is a connect-time failure like `Io`.
#[test]
fn every_transport_failure_is_left_alone() {
    let cases: Vec<tungstenite::Error> = vec![
        tungstenite::Error::ConnectionClosed,
        tungstenite::Error::AlreadyClosed,
        tungstenite::Error::Io(std::io::Error::other("connection reset")),
        tungstenite::Error::Capacity(tungstenite::error::CapacityError::TooManyHeaders),
        tungstenite::Error::Protocol(tungstenite::error::ProtocolError::HandshakeIncomplete),
        tungstenite::Error::WriteBufferFull(Box::new(tungstenite::Message::Text("queued".into()))),
        tungstenite::Error::AttackAttempt,
        tungstenite::Error::Url(tungstenite::error::UrlError::NoPathOrQuery),
        tungstenite::Error::HttpFormat(
            http::header::HeaderName::from_bytes(b"not a header")
                .expect_err("an invalid header name should not parse")
                .into(),
        ),
    ];

    for error in cases {
        let expected = error.to_string();
        let mapped = from_tungstenite(error);

        assert!(
            matches!(mapped, Error::Instance(_)),
            "a failure with no provider response must stay an Instance: {mapped:?}"
        );
        assert_eq!(mapped.non_success_status(), None);
        assert_eq!(mapped.non_success_body(), None);
        assert!(
            mapped.to_string().contains(&expected),
            "the transport's own message must survive: {mapped}"
        );
    }
}

/// The connect timeout's message is part of the caller-visible surface: it
/// is what a host sees on a hung handshake, and it changed when the
/// timeout moved from the session to the backend (rig#2426, recorded in
/// MIGRATING.md). Pin it so a further change is deliberate.
#[test]
fn the_connect_timeout_names_itself_and_its_duration() {
    let error = Error::instance(ConnectTimeout(Duration::from_secs(30)));

    assert!(
        error
            .to_string()
            .contains("timed out connecting the websocket after 30s"),
        "the timeout should name itself and its duration, got {error}"
    );
    // A timeout never reached the provider, so it carries no response.
    assert_eq!(error.non_success_status(), None);
}

/// The handshake request must carry the caller's auth headers onto the
/// websocket scheme.
#[test]
fn the_client_request_keeps_the_callers_headers() {
    let request = Request::builder()
        .method(http::Method::GET)
        .uri("wss://api.openai.com/v1/responses")
        .header(http::header::AUTHORIZATION, "Bearer test-key")
        .body(NoBody)
        .expect("request should build");

    let request = client_request(request).expect("client request should build");

    assert_eq!(
        request
            .headers()
            .get(http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer test-key")
    );
    // `into_client_request` supplied the handshake headers.
    assert!(request.headers().contains_key("sec-websocket-key"));
}
