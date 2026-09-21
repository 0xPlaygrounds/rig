use http::StatusCode;

use super::*;
use crate::{http_client, provider_response::ProviderResponseError};

/// A non-success reply as a transport reports it, routed like a `?` would.
fn http_error(status: u16) -> CompletionError {
    CompletionError::from_transport_error(http_client::Error::non_success_with_details(
        StatusCode::from_u16(status).expect("valid status"),
        http::HeaderMap::new(),
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

/// These are normalization contracts over captured parts, not provider wire behavior.
#[test]
fn vector_http_reports_preserve_response_details() {
    let body = " {\n  \"error\": {\"code\": \"quota_exceeded\", \"message\": \"café\"}\n}\n";
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("7"));
    headers.append("x-metadata", http::HeaderValue::from_static("first"));
    headers.append("x-metadata", http::HeaderValue::from_static("second"));
    let Ok(opaque) = http::HeaderValue::from_bytes(b"\x80") else {
        panic!("opaque header must be representable");
    };
    headers.insert("x-opaque", opaque);
    let mut request_id = http::HeaderValue::from_static("not-provider-identified");
    request_id.set_sensitive(true);
    headers.insert("x-request-id", request_id);
    let transport = http_client::Error::non_success_with_details(
        StatusCode::TOO_MANY_REQUESTS,
        headers.clone(),
        body.to_owned(),
    );
    let chain = vec![transport.to_string()];
    let error = VectorStoreError::from(transport);
    let report = ErrorReport::from(&error);
    assert_eq!(report.kind, ErrorKind::ProviderResponse);
    assert_eq!(report.http_status, Some(429));
    assert_eq!(
        report.provider_response_status(),
        Some(StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(report.provider_response_body(), Some(body));
    assert_eq!(report.provider_response_headers(), Some(&headers));
    assert!(
        report
            .provider_response_headers()
            .and_then(|headers| headers.get("x-request-id"))
            .is_some_and(http::HeaderValue::is_sensitive)
    );
    assert_eq!(report.code.as_deref(), Some("quota_exceeded"));
    assert!(report.retryable);
    assert!(!report.refusal);
    assert_eq!(report.provider_request_id(), None);
    assert_eq!(
        report.provider_response,
        Some(
            ProviderResponseError::new(StatusCode::TOO_MANY_REQUESTS, body)
                .with_headers(Some(headers))
        )
    );
    assert_eq!(report.message, error.to_string());
    assert_eq!(report.source_chain, chain);
    assert_eq!(ErrorReport::from(error), report);
}

/// Both store reply paths must use the existing response code and status policy.
#[test]
fn vector_reply_reports_preserve_machine_codes_and_status_retryability() {
    let bodies = [
        (
            r#"{"error":{"code":"quota_exceeded","status":"ignored","type":"ignored"}}"#,
            Some("quota_exceeded"),
        ),
        (
            r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED"}}"#,
            Some("RESOURCE_EXHAUSTED"),
        ),
        (
            r#"{"error":{"code":null,"type":"authentication_error"}}"#,
            Some("authentication_error"),
        ),
        (
            r#"{"error":{"code":"","status":"","type":"overloaded_error"}}"#,
            Some("overloaded_error"),
        ),
        (r#"{"error":{"code":429}}"#, None),
        (r#"{"error":{"code":""}}"#, None),
        (r#"{"error":"slow down"}"#, None),
        (" plain text\n", None),
        ("{malformed", None),
        ("", None),
    ];
    for (status, retryable) in [
        (100, false),
        (200, false),
        (302, false),
        (400, false),
        (401, false),
        (403, false),
        (408, true),
        (425, true),
        (429, true),
        (500, true),
        (503, true),
        (599, true),
        (600, false),
    ] {
        let Ok(status) = StatusCode::from_u16(status) else {
            panic!("invalid test status");
        };
        for (body, code) in bodies {
            let transport = http_client::Error::non_success_with_details(
                status,
                http::HeaderMap::new(),
                body.to_owned(),
            );
            let transport_chain = vec![transport.to_string()];
            for (error, headers, chain) in [
                (
                    VectorStoreError::ExternalAPIError(status, body.to_owned()),
                    None,
                    Vec::new(),
                ),
                (
                    VectorStoreError::from(transport),
                    Some(http::HeaderMap::new()),
                    transport_chain,
                ),
            ] {
                let report = ErrorReport::from(&error);
                assert_eq!(report.kind, ErrorKind::ProviderResponse, "{error}");
                assert_eq!(report.http_status, Some(status.as_u16()));
                assert_eq!(report.provider_response_status(), Some(status));
                assert_eq!(report.retryable, retryable, "{error}");
                assert_eq!(report.code.as_deref(), code, "{error}");
                assert_eq!(report.provider_response_body(), Some(body));
                assert_eq!(report.provider_response_headers(), headers.as_ref());
                assert_eq!(
                    report.provider_response,
                    Some(ProviderResponseError::new(status, body).with_headers(headers))
                );
                assert!(!report.refusal);
                assert_eq!(report.provider_request_id(), None);
                assert_eq!(report.message, error.to_string());
                assert_eq!(report.source_chain, chain);
                assert_eq!(ErrorReport::from(error), report);
            }
        }
    }
}

#[test]
fn vector_response_less_transport_reports_preserve_classification_and_sources() {
    #[derive(Debug, thiserror::Error)]
    #[error("socket reset")]
    struct Reset;
    #[derive(Debug, thiserror::Error)]
    #[error("transport backend")]
    struct Backend(#[source] Reset);

    let Err(invalid_header) = http::HeaderValue::from_bytes(b"\x00") else {
        panic!("NUL must be an invalid header value");
    };
    let Err(protocol) = http::Request::builder().header("x-invalid", "\n").body(()) else {
        panic!("a newline must be an invalid request header value");
    };
    let protocol_source = protocol.to_string();
    let header_source = invalid_header.to_string();
    for (transport, retryable, sources) in [
        (http_client::Error::StreamEnded, true, Vec::new()),
        (
            http_client::Error::instance(Backend(Reset)),
            true,
            vec!["transport backend".to_owned(), "socket reset".to_owned()],
        ),
        (
            http_client::Error::Protocol(protocol),
            false,
            vec![protocol_source],
        ),
        (
            http_client::Error::InvalidHeaderValue(invalid_header),
            false,
            vec![header_source],
        ),
        (http_client::Error::NoHeaders, false, Vec::new()),
        (
            http_client::Error::InvalidContentType(http::HeaderValue::from_static("text/html")),
            false,
            Vec::new(),
        ),
    ] {
        let mut chain = vec![transport.to_string()];
        chain.extend(sources);
        let error = VectorStoreError::from(transport);
        let report = ErrorReport::from(&error);
        assert_eq!(report.kind, ErrorKind::Http, "{error}");
        assert_eq!(report.retryable, retryable, "{error}");
        assert_eq!(report.http_status, None);
        assert_eq!(report.provider_response, None);
        assert_eq!(report.code, None);
        assert_eq!(report.provider_request_id(), None);
        assert!(!report.refusal);
        assert_eq!(report.message, error.to_string());
        assert_eq!(report.source_chain, chain);
        assert_eq!(ErrorReport::from(error), report);
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
    // The transport's rejection is the provider's reply: the kind says so,
    // and the status decides retryability.
    let report = http_error(429).report();
    assert_eq!(report.kind, ErrorKind::ProviderResponse);
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
        assert_eq!(report.kind, ErrorKind::Http);
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
    // A transport error never carries a status: a status-carrying
    // rejection routes to the provider's reply and follows the status table.
    assert!(http_error(503).is_retryable());
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
            .with_headers(Some(headers)),
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
        .map(|response| response.with_headers(None));
    assert_eq!(back, without_headers);

    // A non-success reply the transport rejected carries its status and body
    // the same way; a diagnostic with no provider response carries nothing.
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

#[test]
fn embedding_and_rerank_reports_retain_structured_provider_metadata() {
    let response = ProviderResponseError::new(StatusCode::TOO_MANY_REQUESTS, "retry later")
        .with_provider_request_id(Some("req-retained".into()));
    let embedding = EmbeddingError::ProviderResponse(response.clone());
    let rerank = RerankError::ProviderResponse(response.clone());
    let wrapped =
        VectorStoreError::EmbeddingError(EmbeddingError::ProviderResponse(response.clone()));
    for report in [
        ErrorReport::from(&embedding),
        ErrorReport::from(&rerank),
        ErrorReport::from(&wrapped),
    ] {
        assert_eq!(report.request_id.as_deref(), Some("req-retained"));
        assert_eq!(
            serde_json::to_value(report.provider_response.as_ref()).unwrap(),
            serde_json::to_value(Some(&response)).unwrap()
        );
        assert!(report.retryable);
        let restored: ErrorReport =
            serde_json::from_value(serde_json::to_value(&report).unwrap()).unwrap();
        assert_eq!(restored.request_id, report.request_id);
        assert!(restored.provider_response.is_some());
    }
}

/// A transport rejection routed through the `From` conversion is the
/// provider's reply on every capability: body, headers and status ride on
/// the report, and the kind says so. `HttpError` itself cannot carry any of
/// them, so a report from one has no provider response.
#[test]
fn embedding_and_rerank_http_reports_retain_body_and_headers() {
    for report in [
        ErrorReport::from(EmbeddingError::HttpError(http_client::Error::StreamEnded)),
        ErrorReport::from(RerankError::HttpError(http_client::Error::StreamEnded)),
    ] {
        assert_eq!(report.kind, ErrorKind::Http);
        assert_eq!(report.http_status, None);
        assert!(report.provider_response.is_none());
        assert!(report.retryable);
    }
    let make_error = || {
        let mut headers = http::HeaderMap::new();
        headers.insert("retry-after", http::HeaderValue::from_static("7"));
        http_client::Error::InvalidStatusCodeWithDetails {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: "temporary outage".into(),
            headers,
        }
    };
    for report in [
        ErrorReport::from(EmbeddingError::from(make_error())),
        ErrorReport::from(RerankError::from(make_error())),
    ] {
        assert_eq!(report.kind, ErrorKind::ProviderResponse);
        let response = report.provider_response.expect("structured HTTP response");
        assert_eq!(response.body, "temporary outage");
        assert_eq!(
            response
                .headers
                .as_ref()
                .expect("retained headers")
                .get("retry-after")
                .expect("retry header"),
            "7"
        );
        assert_eq!(response.status, Some(StatusCode::SERVICE_UNAVAILABLE));
        assert!(report.retryable);
        assert!(report.request_id.is_none());
    }
}

#[derive(Debug, thiserror::Error)]
#[error("backend failed")]
struct NestedBackendError(#[source] std::io::Error);

#[test]
fn wrapped_memory_error_retains_nested_sources() {
    let error = MemoryError::backend(NestedBackendError(std::io::Error::other("disk")));
    assert!(
        std::error::Error::source(&error).is_some_and(|source| source.is::<NestedBackendError>())
    );
    let report = ErrorReport::from(&error);
    assert_eq!(report.source_chain, vec!["backend failed", "disk"]);
    assert_eq!(report.message, error.to_string());
}

#[test]
fn wrapped_document_error_retains_nested_sources() {
    let error = EmbeddingError::DocumentError(Box::new(NestedBackendError(std::io::Error::other(
        "document",
    ))));
    assert!(
        std::error::Error::source(&error).is_some_and(|source| source.is::<NestedBackendError>())
    );
    let report = ErrorReport::from(&error);
    assert_eq!(report.source_chain, vec!["backend failed", "document"]);
    assert_eq!(report.message, error.to_string());
}
