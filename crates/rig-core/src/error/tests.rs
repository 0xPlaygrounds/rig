use http::StatusCode;

use super::*;
use crate::{http_client, provider_response::ProviderResponseError};

fn http_error(status: u16) -> CompletionError {
    CompletionError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
        StatusCode::from_u16(status).expect("valid status"),
        "body".to_string(),
    ))
}

#[test]
fn retrieval_wrapping_preserves_embedding_error_classification() {
    for (status, retryable) in [(400, false), (429, true), (503, true)] {
        let inner = EmbeddingError::ProviderResponse(ProviderResponseError::new(
            StatusCode::from_u16(status).expect("valid status"),
            "embedding request failed",
        ));
        let direct = ErrorReport::from(&inner);
        assert_eq!(direct.retryable, retryable);
        let error = VectorStoreError::EmbeddingError(inner);
        let wrapped = ErrorReport::from(&error);
        assert_eq!(wrapped.kind, direct.kind);
        assert_eq!(wrapped.http_status, direct.http_status);
        assert_eq!(
            wrapped.retryable, direct.retryable,
            "wrapping status {status} must preserve retryability"
        );
        assert_eq!(wrapped.message, error.to_string());
        assert!(!wrapped.source_chain.is_empty());
    }
}

#[test]
fn retry_table_per_status() {
    // Each (status, decision) row is a sign-off entry: 408/425/429/5xx retry,
    // every other status does not, a response-less failure does.
    let rows = [
        (Some(400), false),
        (Some(401), false),
        (Some(403), false),
        (Some(404), false),
        (Some(408), true),
        (Some(409), false),
        (Some(422), false),
        (Some(425), true),
        (Some(429), true),
        (Some(500), true),
        (Some(502), true),
        (Some(503), true),
        (Some(504), true),
        (Some(599), true),
        (Some(600), false),
        (None, false),
    ];
    for (status, expected) in rows {
        assert_eq!(retryable_status(status), expected, "status {status:?}");
    }
}

#[test]
fn completion_http_error_reports_status_and_retryability() {
    let report = http_error(429).report();
    assert_eq!(report.kind, ErrorKind::Http { status: Some(429) });
    assert_eq!(report.http_status, Some(429));
    assert!(report.retryable);
    assert!(http_error(429).is_retryable());

    let report = http_error(400).report();
    assert!(!report.retryable);
    assert!(!http_error(400).is_retryable());
}

#[test]
fn transport_failures_without_a_status_classify_by_what_they_are() {
    // The sign-off table for response-less transport failures: transient
    // when the request never reached a decision, permanent when the client
    // could not form the request or read the response.
    let transient = [
        http_client::Error::StreamEnded,
        http_client::Error::Instance("connection reset by peer".into()),
    ];
    for error in transient {
        let error = CompletionError::HttpError(error);
        assert!(error.is_retryable(), "{error}");
        let report = error.report();
        assert_eq!(report.kind, ErrorKind::Http { status: None });
        assert_eq!(report.http_status, None);
        assert!(report.retryable, "{report:?}");
    }
    let permanent = [
        http_client::Error::NoHeaders,
        http_client::Error::InvalidContentType(http::HeaderValue::from_static("text/html")),
        http_client::Error::InvalidHeaderValue(
            http::HeaderValue::from_bytes(b"\x00").expect_err("illegal header value"),
        ),
    ];
    for error in permanent {
        let error = CompletionError::HttpError(error);
        assert!(!error.is_retryable(), "{error}");
        assert!(!error.report().retryable, "{error}");
    }
    // A status-carrying transport failure follows the status table.
    assert!(
        CompletionError::HttpError(http_client::Error::InvalidStatusCode(
            StatusCode::SERVICE_UNAVAILABLE
        ))
        .is_retryable()
    );
    // A provider response without a status decides nothing either.
    assert!(
        !CompletionError::ProviderResponse(ProviderResponseError::without_status("body"))
            .is_retryable()
    );
}

#[test]
fn tool_retryability_has_one_answer_on_every_surface() {
    // The kind default is the table: `retryable()`, `is_retryable()` and the
    // wire report agree for every kind, including the ones whose default is
    // `None` (not retryable on the wire).
    for kind in [
        ToolErrorKind::InvalidArgs,
        ToolErrorKind::Timeout,
        ToolErrorKind::Cancelled,
        ToolErrorKind::NotFound,
        ToolErrorKind::PermissionDenied,
        ToolErrorKind::RateLimited,
        ToolErrorKind::Provider,
        ToolErrorKind::Network,
        ToolErrorKind::Other,
    ] {
        let error = ToolExecutionError::new(kind, "x");
        let expected = kind.default_retryable().unwrap_or(false);
        assert_eq!(error.retryable(), kind.default_retryable(), "{kind:?}");
        assert_eq!(error.is_retryable(), expected, "{kind:?}");
        assert_eq!(error.report().retryable, expected, "{kind:?}");
    }
    let provider = ToolExecutionError::new(ToolErrorKind::Provider, "x");
    assert_eq!(provider.retryable(), None);
    assert!(
        !provider.is_retryable(),
        "a kind that leaves it to the tool is not retryable on the wire"
    );
    assert!(
        provider.with_retryable(true).is_retryable(),
        "the override wins"
    );
}

#[test]
fn completion_provider_response_classifies_by_status() {
    let error = CompletionError::ProviderResponse(ProviderResponseError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "down",
    ));
    let report = error.report();
    assert_eq!(report.kind, ErrorKind::ProviderResponse);
    assert_eq!(report.http_status, Some(503));
    assert!(report.retryable);
}

#[test]
fn completion_non_http_variants_are_not_retryable() {
    let cases = [
        (
            CompletionError::ResponseError("bad".into()),
            ErrorKind::Response,
        ),
        (
            CompletionError::ProviderError("bad".into()),
            ErrorKind::Provider,
        ),
        (
            CompletionError::UrlError(url::ParseError::EmptyHost),
            ErrorKind::Url,
        ),
    ];
    for (error, kind) in cases {
        let report = error.report();
        assert_eq!(report.kind, kind);
        assert!(!report.retryable);
        assert_eq!(report.message, error.to_string());
    }
}

#[test]
fn tool_error_uses_override_then_kind_default() {
    let timeout = ToolExecutionError::new(ToolErrorKind::Timeout, "slow");
    assert!(timeout.report().retryable);
    let pinned = ToolExecutionError::new(ToolErrorKind::Timeout, "slow").with_retryable(false);
    assert!(!pinned.report().retryable);
    let other = ToolExecutionError::new(ToolErrorKind::Other, "meh");
    assert!(!other.report().retryable);

    let report = ToolExecutionError::refused("no")
        .with_code("E1")
        .with_http_status(403)
        .report();
    assert_eq!(
        report.kind,
        ErrorKind::Tool(ToolErrorKind::PermissionDenied)
    );
    assert!(report.refusal);
    assert_eq!(report.code.as_deref(), Some("E1"));
    assert_eq!(report.http_status, Some(403));
}

#[test]
fn source_chain_is_outermost_first() {
    #[derive(Debug, thiserror::Error)]
    #[error("inner")]
    struct Inner;
    #[derive(Debug, thiserror::Error)]
    #[error("outer")]
    struct Outer(#[source] Inner);

    let error = ToolExecutionError::new(ToolErrorKind::Other, "tool").with_source(Outer(Inner));
    let report = error.report();
    assert_eq!(report.message, "tool");
    assert_eq!(
        report.source_chain,
        vec!["outer".to_string(), "inner".to_string()]
    );
}

#[test]
fn memory_error_kinds() {
    assert_eq!(
        MemoryError::Policy("p".into()).report().kind,
        ErrorKind::MemoryPolicy
    );
    assert_eq!(
        MemoryError::Internal("i".into()).report().kind,
        ErrorKind::Internal
    );
    let backend = MemoryError::backend(std::io::Error::other("disk"));
    let report = backend.report();
    assert_eq!(report.kind, ErrorKind::MemoryBackend);
    assert!(!report.retryable);
    assert!(report.message.contains("disk"));
}

#[test]
fn report_round_trips_through_serde() {
    let report = ErrorReport::new(ErrorKind::Tool(ToolErrorKind::RateLimited), "slow down")
        .with_retryable(true)
        .with_code("429")
        .with_http_status(429);
    let json = serde_json::to_string(&report).expect("serialize");
    let back: ErrorReport = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, report);
    assert_eq!(report.to_string(), "slow down");
}

#[test]
fn a_provider_response_travels_with_the_report() {
    // What a caller could read off the provider error — status, body,
    // headers, request id — is still readable after the failure crossed
    // the wire, and survives serde.
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("7"));
    let error = CompletionError::ProviderResponse(
        ProviderResponseError::new(StatusCode::TOO_MANY_REQUESTS, r#"{"error":"slow down"}"#)
            .with_provider_request_id(Some("req-9".to_owned()))
            .with_headers(Some(Box::new(headers))),
    );
    let report = ErrorReport::from(&error);
    assert_eq!(report.kind, ErrorKind::ProviderResponse);
    assert_eq!(
        report.provider_response_status(),
        Some(StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(
        report.provider_response_body(),
        Some(r#"{"error":"slow down"}"#)
    );
    assert_eq!(
        report
            .provider_response_json()
            .expect("json")
            .and_then(|json| json["error"].as_str().map(str::to_owned)),
        Some("slow down".to_owned())
    );
    assert_eq!(
        report
            .provider_response_headers()
            .and_then(|headers| headers.get("retry-after"))
            .and_then(|value| value.to_str().ok()),
        Some("7")
    );
    assert_eq!(report.provider_request_id(), Some("req-9"));
    assert!(report.retryable, "429 is retryable");

    // Through serde the report keeps the response's identity and drops
    // its headers, which are the transport's and never the same twice.
    let json = serde_json::to_string(&report).expect("serialize");
    let back: ErrorReport = serde_json::from_str(&json).expect("deserialize");
    assert!(back.provider_response_headers().is_none());
    let mut without_headers = report.clone();
    without_headers.provider_response = without_headers
        .provider_response
        .map(|response| Box::new(response.with_headers(None)));
    assert_eq!(back, without_headers);

    // A non-success HTTP failure carries its status and body the same way;
    // a diagnostic with no provider response carries nothing.
    let http = ErrorReport::from(&http_error(503));
    assert_eq!(
        http.provider_response_status(),
        Some(StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(http.provider_response_body(), Some("body"));
    let plain = ErrorReport::from(&CompletionError::ProviderError("oops".to_owned()));
    assert!(plain.provider_response.is_none());
    assert_eq!(plain.provider_response_body(), None);
}
