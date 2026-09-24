use http::StatusCode;

use super::*;
use crate::{
    http_client::{self, Error as H},
    observe::AdapterErrorBoundary,
    provider_response::ProviderResponseError,
};

/// A non-success reply as a transport reports it, routed like a `?` would.
fn http_error(status: u16) -> ProviderError {
    ProviderError::from_transport_error(http_client::Error::non_success_with_details(
        StatusCode::from_u16(status).expect("valid status"),
        http::HeaderMap::new(),
        "body".to_string(),
    ))
}

#[test]
fn retrieval_wrapping_preserves_embedding_error_classification() {
    for (status, retryable) in [(400, false), (429, true), (503, true)] {
        let inner = ProviderError::ProviderResponse(ProviderResponseError::new(
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
        let error = ProviderError::Http(error);
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
        let error = ProviderError::Http(error);
        assert!(!error.is_retryable(), "{error}");
        assert!(!error.report().retryable, "{error}");
    }
    // A transport error never carries a status: a status-carrying
    // rejection routes to the provider's reply and follows the status table.
    assert!(http_error(503).is_retryable());
    // A provider response without a status decides nothing either.
    assert!(
        !ProviderError::ProviderResponse(ProviderResponseError::without_status("body"))
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
    let error = ProviderError::ProviderResponse(ProviderResponseError::new(
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
        (ProviderError::Response("bad".into()), ErrorKind::Response),
        (ProviderError::Provider("bad".into()), ErrorKind::Provider),
        (
            ProviderError::Url(url::ParseError::EmptyHost),
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
    let error = ProviderError::ProviderResponse(
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
    let plain = ErrorReport::from(&ProviderError::Provider("oops".to_owned()));
    assert!(plain.provider_response.is_none());
    assert_eq!(plain.provider_response_body(), None);
}

#[test]
fn provider_reports_retain_structured_provider_metadata() {
    let response = ProviderResponseError::new(StatusCode::TOO_MANY_REQUESTS, "retry later")
        .with_provider_request_id(Some("req-retained".into()));
    let direct = ProviderError::ProviderResponse(response.clone());
    let wrapped =
        VectorStoreError::EmbeddingError(ProviderError::ProviderResponse(response.clone()));
    for report in [ErrorReport::from(&direct), ErrorReport::from(&wrapped)] {
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
/// provider's reply: body, headers and status ride on the report, and the
/// kind says so. `Http` itself cannot carry any of them, so a report from
/// one has no provider response.
#[test]
fn http_reports_retain_body_and_headers() {
    let report = ErrorReport::from(ProviderError::Http(http_client::Error::StreamEnded));
    assert_eq!(report.kind, ErrorKind::Http);
    assert_eq!(report.http_status, None);
    assert!(report.provider_response.is_none());
    assert!(report.retryable);
    let error = || {
        let mut headers = http::HeaderMap::new();
        headers.insert("retry-after", http::HeaderValue::from_static("7"));
        http_client::Error::InvalidStatusCodeWithDetails {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: "temporary outage".into(),
            headers,
        }
    };
    let report = ErrorReport::from(ProviderError::from(error()));
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
    let error = ProviderError::Request(Box::new(NestedBackendError(std::io::Error::other(
        "document",
    ))));
    assert!(
        std::error::Error::source(&error).is_some_and(|source| source.is::<NestedBackendError>())
    );
    let report = ErrorReport::from(&error);
    assert_eq!(report.source_chain, vec!["backend failed", "document"]);
    assert_eq!(report.message, error.to_string());
}

/// Reports captured from the per-operation error enums `ProviderError`
/// replaced, one row per former variant whose report is unchanged. The
/// completion, embedding, rerank, transcription, image, audio and cached
/// content enums produced identical reports for identical inputs, so each
/// shared case is one row. Every field, the serialized reply included, and
/// the observation boundary must match.
#[test]
fn reports_match_the_replaced_error_enums() {
    let cases: Vec<(&str, ProviderError, &str, AdapterErrorBoundary)> = vec![
        (
            "*::HttpError(StreamEnded)",
            ProviderError::Http(H::StreamEnded),
            r#"{"code":null,"http_status":null,"kind":"http","message":"HttpError: Stream ended","refusal":false,"retryable":true,"source_chain":[]}"#,
            AdapterErrorBoundary::Transport,
        ),
        (
            "*::HttpError(Instance)",
            ProviderError::Http(H::instance(std::io::Error::other("conn reset"))),
            r#"{"code":null,"http_status":null,"kind":"http","message":"HttpError: Http client error: conn reset","refusal":false,"retryable":true,"source_chain":[]}"#,
            AdapterErrorBoundary::Unknown,
        ),
        (
            "*::HttpError(NoHeaders)",
            ProviderError::Http(H::NoHeaders),
            r#"{"code":null,"http_status":null,"kind":"http","message":"HttpError: Request in error state, cannot access headers","refusal":false,"retryable":false,"source_chain":[]}"#,
            AdapterErrorBoundary::Request,
        ),
        (
            "*::JsonError",
            ProviderError::Json(json_error()),
            r#"{"code":null,"http_status":null,"kind":"json","message":"JsonError: EOF while parsing an object at line 1 column 1","refusal":false,"retryable":false,"source_chain":["EOF while parsing an object at line 1 column 1"]}"#,
            AdapterErrorBoundary::Decode,
        ),
        (
            "*::ResponseError",
            ProviderError::Response("bad shape".into()),
            r#"{"code":null,"http_status":null,"kind":"response","message":"ResponseError: bad shape","refusal":false,"retryable":false,"source_chain":[]}"#,
            AdapterErrorBoundary::Decode,
        ),
        (
            "*::ProviderError",
            ProviderError::Provider("provider said no".into()),
            r#"{"code":null,"http_status":null,"kind":"provider","message":"ProviderError: provider said no","refusal":false,"retryable":false,"source_chain":[]}"#,
            AdapterErrorBoundary::ProviderResponse,
        ),
        (
            "*::from_http_response(429)+id+headers",
            ProviderError::from_http_response(
                StatusCode::TOO_MANY_REQUESTS,
                r#"{"error":{"type":"rate_limit"}}"#,
            )
            .with_provider_request_id(Some("req_1".into()))
            .with_response_headers(Some(headers())),
            r#"{"code":"rate_limit","http_status":429,"kind":"provider_response","message":"ProviderResponseError: status 429 Too Many Requests: {\"error\":{\"type\":\"rate_limit\"}} (request id: req_1)","provider_response":{"body":"{\"error\":{\"type\":\"rate_limit\"}}","provider_request_id":"req_1","status":429},"refusal":false,"request_id":"req_1","retryable":true,"source_chain":[]}"#,
            AdapterErrorBoundary::ProviderResponse,
        ),
        (
            "*::from_http_response(400)",
            ProviderError::from_http_response(StatusCode::BAD_REQUEST, "plain body"),
            r#"{"code":null,"http_status":400,"kind":"provider_response","message":"ProviderResponseError: status 400 Bad Request: plain body","provider_response":{"body":"plain body","provider_request_id":null,"status":400},"refusal":false,"retryable":false,"source_chain":[]}"#,
            AdapterErrorBoundary::ProviderResponse,
        ),
        (
            "*::from_transport_error(503)",
            ProviderError::from_transport_error(transport_503()),
            r#"{"code":"overloaded","http_status":503,"kind":"provider_response","message":"ProviderResponseError: status 503 Service Unavailable: {\"error\":{\"code\":\"overloaded\",\"message\":\"busy\"}}","provider_response":{"body":"{\"error\":{\"code\":\"overloaded\",\"message\":\"busy\"}}","provider_request_id":null,"status":503},"refusal":false,"retryable":true,"source_chain":[]}"#,
            AdapterErrorBoundary::ProviderResponse,
        ),
        (
            "*::from_provider_body+refusal",
            ProviderError::ProviderResponse(
                ProviderResponseError::without_status(r#"{"error":{"code":"content_filter"}}"#)
                    .with_refusal(true)
                    .with_code(Some("REFUSED".into())),
            ),
            r#"{"code":"REFUSED","http_status":null,"kind":"provider_response","message":"ProviderResponseError: {\"error\":{\"code\":\"content_filter\"}}","provider_response":{"body":"{\"error\":{\"code\":\"content_filter\"}}","code":"REFUSED","provider_request_id":null,"refusal":true,"status":null},"refusal":true,"retryable":false,"source_chain":[]}"#,
            AdapterErrorBoundary::ProviderResponse,
        ),
        (
            "*::from_provider_body+transient",
            ProviderError::from_provider_body("sdk said busy").with_transient(Some(true)),
            r#"{"code":null,"http_status":null,"kind":"provider_response","message":"ProviderResponseError: sdk said busy","provider_response":{"body":"sdk said busy","provider_request_id":null,"status":null,"transient":true},"refusal":false,"retryable":true,"source_chain":[]}"#,
            AdapterErrorBoundary::ProviderResponse,
        ),
        (
            "Completion/Embedding/Rerank::UrlError",
            ProviderError::Url(url_error()),
            r#"{"code":null,"http_status":null,"kind":"url","message":"UrlError: relative URL without a base","refusal":false,"retryable":false,"source_chain":["relative URL without a base"]}"#,
            AdapterErrorBoundary::Request,
        ),
        (
            "Completion/Transcription/ImageGeneration/AudioGeneration::RequestError",
            ProviderError::Request(boxed()),
            r#"{"code":null,"http_status":null,"kind":"request","message":"RequestError: io broke","refusal":false,"retryable":false,"source_chain":["io broke"]}"#,
            AdapterErrorBoundary::Request,
        ),
    ];
    for (case, error, expected, boundary) in cases {
        let expected: serde_json::Value = serde_json::from_str(expected).expect("expected report");
        assert_eq!(
            serde_json::to_value(error.report()).expect("report serializes"),
            expected,
            "{case}"
        );
        assert_eq!(error.boundary(), boundary, "{case}");
    }
}

fn headers() -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("7"));
    headers.insert("x-request-id", http::HeaderValue::from_static("req_hdr"));
    headers
}

fn json_error() -> serde_json::Error {
    serde_json::from_str::<serde_json::Value>("{").expect_err("malformed JSON")
}

fn url_error() -> url::ParseError {
    url::Url::parse("not a url").expect_err("relative URL")
}

fn boxed() -> BoxError {
    Box::new(std::io::Error::other("io broke"))
}

fn transport_503() -> H {
    H::InvalidStatusCodeWithDetails {
        status: StatusCode::SERVICE_UNAVAILABLE,
        body: r#"{"error":{"code":"overloaded","message":"busy"}}"#.to_owned(),
        headers: headers(),
    }
}

/// Response-shaped embedding faults keep their `response` report and report
/// the decode boundary that kind implies.
#[test]
fn response_shaped_faults_report_the_decode_boundary() {
    for error in [
        ProviderError::Response("openai embedding response omitted required usage".into()),
        ProviderError::MismatchedDimensions {
            provider: "llamacpp".into(),
            requested: 128,
            returned: 1024,
        },
    ] {
        assert_eq!(error.kind(), ErrorKind::Response, "{error}");
        assert_eq!(error.boundary(), AdapterErrorBoundary::Decode, "{error}");
    }
}

/// An expired cache and a rejected credential keep the provider's reply:
/// status, body, and request ID reach the report, which the replaced
/// `CachedContentError::Expired` and `VerifyError::InvalidAuthentication`
/// discarded.
#[test]
fn verdicts_on_a_handle_or_credential_keep_the_reply() {
    let expired = ProviderError::CacheExpired {
        name: "cachedContents/abc".into(),
        response: ProviderResponseError::new(StatusCode::NOT_FOUND, "not found body")
            .with_provider_request_id(Some("req_c".into())),
    };
    assert_eq!(
        expired.to_string(),
        "cached content `cachedContents/abc` is expired or was deleted: not found body"
    );
    let rejected = ProviderError::InvalidAuthentication(ProviderResponseError::new(
        StatusCode::UNAUTHORIZED,
        "bad key",
    ));
    assert_eq!(
        rejected.to_string(),
        "invalid authentication: status 401 Unauthorized: bad key"
    );
    for (error, status) in [(&expired, 404), (&rejected, 401)] {
        let report = error.report();
        assert_eq!(report.kind, ErrorKind::ProviderResponse);
        assert_eq!(report.http_status, Some(status));
        assert!(!report.retryable);
        assert!(report.provider_response.is_some());
        assert_eq!(error.boundary(), AdapterErrorBoundary::ProviderResponse);
    }
    assert_eq!(expired.provider_request_id(), Some("req_c"));
}

/// Request-building failures from every operation share `Request` and its
/// text, whatever the replaced enum called them.
#[test]
fn request_building_failures_share_one_shape() {
    let document = ProviderError::Request(boxed());
    let invalid = ProviderError::Request("bad name".into());
    let http = ProviderError::from(
        http::Request::builder()
            .method("bad method")
            .body(())
            .expect_err("invalid method"),
    );
    assert_eq!(document.to_string(), "RequestError: io broke");
    assert_eq!(invalid.to_string(), "RequestError: bad name");
    assert_eq!(http.to_string(), "RequestError: invalid HTTP method");
    for error in [document, invalid, http] {
        let report = error.report();
        assert_eq!(report.kind, ErrorKind::Request);
        assert!(!report.retryable);
        assert_eq!(report.source_chain.len(), 1, "{error}");
        assert_eq!(error.boundary(), AdapterErrorBoundary::Request);
    }
}
