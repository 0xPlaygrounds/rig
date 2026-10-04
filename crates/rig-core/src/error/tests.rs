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
fn samples_out_of_range_is_a_non_retryable_request_error() {
    let error = VectorStoreError::SamplesOutOfRange {
        requested: u64::MAX,
        max: i64::MAX as u64,
    };
    let report = ErrorReport::from(&error);
    assert_eq!(report.kind, ErrorKind::Request);
    assert!(!report.retryable);
    assert_eq!(report.http_status, None);
    assert_eq!(
        report.message,
        format!(
            "Requested {} samples, but this vector store returns at most {}",
            u64::MAX,
            i64::MAX
        )
    );
}

#[test]
fn tool_error_survives_a_report_round_trip() {
    let original = ToolExecutionError::rate_limited("slow down")
        .with_retryable(false)
        .with_code("quota")
        .with_http_status(429);
    let back = ToolExecutionError::from(original.report());
    assert_eq!(back.kind(), ToolErrorKind::RateLimited);
    assert_eq!(back.message(), "slow down");
    assert_eq!(back.model_feedback(), Some("slow down"));
    assert_eq!(back.retryable(), Some(false));
    assert_eq!(back.code(), Some("quota"));
    assert_eq!(back.http_status(), Some(429));
    assert!(!back.is_refusal());
    assert_eq!(
        back.downcast_ref::<ErrorReport>().map(|report| report.kind),
        Some(ErrorKind::Tool(ToolErrorKind::RateLimited))
    );

    let refused = ToolExecutionError::from(ToolExecutionError::refused("no").report());
    assert!(refused.is_refusal());
    assert_eq!(refused.kind(), ToolErrorKind::PermissionDenied);
}

#[test]
fn dispatch_report_kinds_map_to_tool_kinds() {
    for (kind, expected) in [
        (ErrorKind::Timeout, ToolErrorKind::Timeout),
        (ErrorKind::Cancelled, ToolErrorKind::Cancelled),
        (ErrorKind::Denied, ToolErrorKind::PermissionDenied),
        (ErrorKind::HandlerUnavailable, ToolErrorKind::NotFound),
        (ErrorKind::Http, ToolErrorKind::Network),
        (ErrorKind::ProviderResponse, ToolErrorKind::Provider),
        (ErrorKind::Internal, ToolErrorKind::Other),
    ] {
        let error = ToolExecutionError::from(ErrorReport::new(kind, "failed").with_retryable(true));
        assert_eq!(error.kind(), expected, "{kind:?}");
        assert_eq!(error.retryable(), Some(true), "{kind:?}");
    }
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
    let error = ProviderError::request(NestedBackendError(std::io::Error::other("document")));
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
            ProviderError::Http(H::StreamEnded.into()),
            r#"{"code":null,"http_status":null,"kind":"http","message":"HttpError: Stream ended","refusal":false,"retryable":true,"source_chain":[]}"#,
            AdapterErrorBoundary::Transport,
        ),
        (
            "*::HttpError(Instance)",
            ProviderError::Http(H::instance(std::io::Error::other("conn reset")).into()),
            r#"{"code":null,"http_status":null,"kind":"http","message":"HttpError: Http client error: conn reset","refusal":false,"retryable":true,"source_chain":[]}"#,
            AdapterErrorBoundary::Unknown,
        ),
        (
            "*::HttpError(NoHeaders)",
            ProviderError::Http(H::NoHeaders.into()),
            r#"{"code":null,"http_status":null,"kind":"http","message":"HttpError: Request in error state, cannot access headers","refusal":false,"retryable":false,"source_chain":[]}"#,
            AdapterErrorBoundary::Request,
        ),
        (
            "*::JsonError",
            ProviderError::Json(json_error().into()),
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
            ProviderError::Request(boxed().into()),
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
    let document = ProviderError::Request(boxed().into());
    let invalid = ProviderError::request("bad name");
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

/// A relayed failure keeps the provider's reply it was reported with: the
/// preserved response reads back through the error, and the error reports as
/// the relayed report unchanged.
#[test]
fn a_relayed_report_keeps_its_provider_response() {
    let report = ErrorReport::from(&http_error(503));
    let relayed = ProviderError::Relayed(Box::new(report.clone()));
    assert_eq!(
        relayed.provider_response(),
        report.provider_response.as_ref()
    );
    assert_eq!(relayed.provider_response_body(), Some("body"));
    assert_eq!(
        relayed.provider_response_status(),
        Some(StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(relayed.kind(), ErrorKind::ProviderResponse);
    assert!(relayed.is_retryable());
    assert_eq!(ErrorReport::from(&relayed), report);
}
