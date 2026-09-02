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
        (None, true),
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
fn completion_transport_failure_without_status_is_retryable() {
    let error = CompletionError::HttpError(http_client::Error::StreamEnded);
    assert!(error.is_retryable());
    let report = error.report();
    assert_eq!(report.kind, ErrorKind::Http { status: None });
    assert_eq!(report.http_status, None);
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
