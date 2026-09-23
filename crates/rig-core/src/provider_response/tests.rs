use http::StatusCode;

/// The one funnel preserves a provider's status and body across every route:
/// a non-success HTTP response, a 2xx provider error envelope, a non-HTTP
/// (gRPC/SDK) transport, and a transport that reported the reply as an error.
#[test]
fn funnel_preserves_status_and_body() {
    type E = crate::error::ProviderError;
    let body = r#"{"error":{"message":"boom"}}"#;

    // Non-success status -> ProviderResponse, with status + body recoverable.
    let err = E::from_http_response(StatusCode::SERVICE_UNAVAILABLE, body);
    assert!(
        matches!(err, E::ProviderResponse(_)),
        "a provider's reply is a ProviderResponse",
    );
    assert_eq!(
        err.provider_response_status(),
        Some(StatusCode::SERVICE_UNAVAILABLE),
        "non-success status not preserved",
    );
    assert_eq!(
        err.provider_response_body(),
        Some(body),
        "non-success body not preserved",
    );
    assert_eq!(
        err.provider_response_json()
            .expect("valid json")
            .expect("present json")["error"]["message"],
        "boom",
    );
    assert_eq!(err.provider_request_id(), None);

    // A provider error envelope returned with a 2xx status -> ProviderResponse,
    // preserving the (success) status so callers can still see it.
    let err = E::from_http_response(StatusCode::OK, body);
    assert_eq!(
        err.provider_response_status(),
        Some(StatusCode::OK),
        "2xx envelope status not preserved",
    );
    assert_eq!(err.provider_response_body(), Some(body));

    // No HTTP status available (gRPC/SDK) -> ProviderResponse with status None.
    let err = E::from_provider_body(body);
    assert_eq!(
        err.provider_response_status(),
        None,
        "status should be None for provider body",
    );
    assert_eq!(err.provider_response_body(), Some(body));

    // Empty-body asymmetry: the body is `Some("")` but JSON parses to `Ok(None)`.
    let err = E::from_provider_body("");
    assert_eq!(err.provider_response_body(), Some(""));
    assert!(err.provider_response_json().expect("ok").is_none());

    // A transport that reported the reply as an error routes through the
    // same funnel: status, body and headers -> ProviderResponse; no
    // response at all stays a transport error, with no status.
    let err = E::from_transport_error(crate::http_client::Error::InvalidStatusCodeWithDetails {
        status: StatusCode::TOO_MANY_REQUESTS,
        body: body.to_string(),
        headers: retry_after_headers(),
    });
    assert!(matches!(err, E::ProviderResponse(_)));
    assert_eq!(err.provider_response_body(), Some(body));
    assert_eq!(
        err.provider_response_headers()
            .and_then(|headers| headers.get(http::header::RETRY_AFTER))
            .and_then(|value| value.to_str().ok()),
        Some("20"),
    );
    let err = E::from_transport_error(crate::http_client::Error::StreamEnded);
    assert!(matches!(err, E::Http(_)));
    assert_eq!(err.provider_response_status(), None);
    // The `?` conversion is the same route.
    let err: E = crate::http_client::Error::non_success_with_details(
        StatusCode::NOT_FOUND,
        http::HeaderMap::new(),
        body.to_string(),
    )
    .into();
    assert!(matches!(err, E::ProviderResponse(_)));

    // rig#2210 — headers are only present when a capture path
    // preserved them. The status+body funnels never have any...
    for err in [
        E::from_http_response(StatusCode::TOO_MANY_REQUESTS, body),
        E::from_http_response(StatusCode::OK, body),
        E::from_provider_body(body),
        E::from_http_response(StatusCode::TOO_MANY_REQUESTS, body)
            .with_provider_request_id(Some("req_abc".to_string())),
    ] {
        assert!(
            err.provider_response_headers().is_none(),
            "a funnel cannot invent headers",
        );
        // ...and attaching `None` must not disturb the error.
        let untouched = err.with_response_headers(None);
        assert!(untouched.provider_response_headers().is_none());
        assert_eq!(untouched.provider_response_body(), Some(body));
    }

    // ...but a preserved response carries headers once attached, so
    // `Retry-After` stays readable on a 429 whether or not the provider
    // reported a request id.
    let without_id = E::from_http_response(StatusCode::TOO_MANY_REQUESTS, body)
        .with_response_headers(Some(retry_after_headers()));
    let with_id = E::from_http_response(StatusCode::TOO_MANY_REQUESTS, body)
        .with_provider_request_id(Some("req_abc".to_string()))
        .with_response_headers(Some(retry_after_headers()));

    for (label, err) in [("without id", without_id), ("with id", with_id)] {
        assert_eq!(
            err.provider_response_headers()
                .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                .and_then(|value| value.to_str().ok()),
            Some("20"),
            "{label}: captured Retry-After not surfaced",
        );
        // Attaching headers must not disturb the status or body the
        // funnel already preserved.
        assert_eq!(
            err.provider_response_status(),
            Some(StatusCode::TOO_MANY_REQUESTS),
            "{label}: status lost when headers were attached",
        );
        assert_eq!(
            err.provider_response_body(),
            Some(body),
            "{label}: body lost when headers were attached",
        );
    }
}

/// A 429's rate-limit metadata, as a provider would send it.
fn retry_after_headers() -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::RETRY_AFTER,
        http::HeaderValue::from_static("20"),
    );
    headers.insert("x-ratelimit-remaining", http::HeaderValue::from_static("0"));
    headers
}

/// rig#2314: the transport id stamps onto the preserved response, so
/// status, body and id all stay recoverable and the id appears in the
/// logged message.
#[test]
fn stamping_a_request_id_keeps_status_body_and_names_the_id() {
    let error = crate::error::ProviderError::from_http_response(
        StatusCode::NOT_FOUND,
        r#"{"error":"nope"}"#,
    )
    .with_provider_request_id(Some("req_abc".to_string()));
    assert!(matches!(
        error,
        crate::error::ProviderError::ProviderResponse(_)
    ));
    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::NOT_FOUND)
    );
    assert_eq!(error.provider_response_body(), Some(r#"{"error":"nope"}"#));
    assert_eq!(error.provider_request_id(), Some("req_abc"));
    assert!(
        error.to_string().contains("request id: req_abc"),
        "the id support asks for appears in the message: {error}"
    );
}

/// A missing id is `None`, never a secondary failure, and leaves the
/// message unchanged; an empty header value is a missing id.
#[test]
fn an_absent_id_leaves_the_message_unchanged() {
    for id in [None, Some(String::new())] {
        let error = crate::error::ProviderError::from_http_response(StatusCode::BAD_REQUEST, "bad")
            .with_provider_request_id(id);
        assert_eq!(error.provider_request_id(), None);
        assert!(!error.to_string().contains("request id"));
    }
}

/// First capture wins: the site that saw the response is the authority on
/// its id, and a later stamp only fills a gap. Variants with no slot
/// absorb the call unchanged.
#[test]
fn stamping_never_overwrites_an_earlier_id_and_is_a_no_op_without_a_slot() {
    let error = crate::error::ProviderError::from_http_response(StatusCode::BAD_REQUEST, "bad")
        .with_provider_request_id(Some("first".to_string()))
        .with_provider_request_id(Some("second".to_string()));
    assert_eq!(error.provider_request_id(), Some("first"));

    let error = crate::error::ProviderError::Provider("rig diagnostic".to_string())
        .with_provider_request_id(Some("req_abc".to_string()));
    assert!(matches!(error, crate::error::ProviderError::Provider(_)));
    assert_eq!(error.provider_request_id(), None);
}

/// rig#2210 × rig#2314: the two pieces of transport metadata are captured
/// on the same path and must not evict each other.
#[test]
fn request_id_and_headers_coexist_on_one_error() {
    let error = crate::error::ProviderError::from_http_response(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":"slow down"}"#,
    )
    .with_provider_request_id(Some("req_abc".to_string()))
    .with_response_headers(Some(retry_after_headers()));

    assert_eq!(error.provider_request_id(), Some("req_abc"));
    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get("x-ratelimit-remaining"))
            .and_then(|value| value.to_str().ok()),
        Some("0"),
    );
}

/// First capture wins on headers too: the site that saw the response is the
/// authority, and a later attach only fills a gap. Without this, a wrapper
/// that re-attaches would silently replace the real response's headers.
#[test]
fn attaching_headers_never_overwrites_an_earlier_capture() {
    let mut later = http::HeaderMap::new();
    later.insert(http::header::RETRY_AFTER, "999".parse().expect("value"));

    for build in [
        crate::error::ProviderError::from_http_response,
        |status, body| {
            crate::error::ProviderError::from_http_response(status, body)
                .with_provider_request_id(Some("req_abc".to_string()))
        },
    ] {
        let error = build(StatusCode::TOO_MANY_REQUESTS, "slow down")
            .with_response_headers(Some(retry_after_headers()))
            .with_response_headers(Some(later.clone()));

        assert_eq!(
            error
                .provider_response_headers()
                .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                .and_then(|value| value.to_str().ok()),
            Some("20"),
            "the first capture must win",
        );
    }
}

/// Variants with no slot for a response absorb the call unchanged, so a
/// capture site can attach unconditionally. A response-less transport error
/// has no response to annotate either.
#[test]
fn attaching_headers_to_a_slotless_variant_is_a_no_op() {
    let error = crate::error::ProviderError::Provider("rig diagnostic".to_string())
        .with_response_headers(Some(retry_after_headers()));
    assert!(matches!(error, crate::error::ProviderError::Provider(_)));
    assert!(error.provider_response_headers().is_none());
    assert_eq!(error.to_string(), "ProviderError: rig diagnostic");

    let error = crate::error::ProviderError::Http(crate::http_client::Error::StreamEnded)
        .with_response_headers(Some(retry_after_headers()));
    assert!(matches!(
        error,
        crate::error::ProviderError::Http(crate::http_client::Error::StreamEnded)
    ));
    assert!(error.provider_response_headers().is_none());
}

/// Display goldens (rig#2315 error matrix): error strings are what
/// callers grep and alert on — message churn must be a reviewed diff.
#[test]
fn display_goldens_for_error_shapes() {
    let with_id = crate::error::ProviderError::from_http_response(
        StatusCode::NOT_FOUND,
        r#"{"error":"nope"}"#,
    )
    .with_provider_request_id(Some("req_abc".to_string()));
    assert_eq!(
        with_id.to_string(),
        r#"ProviderResponseError: status 404 Not Found: {"error":"nope"} (request id: req_abc)"#
    );

    let without_id = crate::error::ProviderError::from_http_response(
        StatusCode::NOT_FOUND,
        r#"{"error":"nope"}"#,
    );
    assert_eq!(
        without_id.to_string(),
        r#"ProviderResponseError: status 404 Not Found: {"error":"nope"}"#
    );

    // A response-less transport failure is a transport error, and says so.
    let dropped =
        crate::error::ProviderError::from_transport_error(crate::http_client::Error::StreamEnded);
    assert_eq!(dropped.to_string(), "HttpError: Stream ended");

    // The transport's own rejection text names the status and body.
    let details = crate::http_client::Error::InvalidStatusCodeWithDetails {
        status: StatusCode::NOT_FOUND,
        body: "x".to_string(),
        headers: http::HeaderMap::new(),
    };
    assert_eq!(
        details.to_string(),
        "Invalid status code 404 Not Found with message: x"
    );

    // rig#2210: capturing headers must never change the text a caller logs.
    for build in [
        crate::error::ProviderError::from_http_response,
        |status, body| {
            crate::error::ProviderError::from_http_response(status, body)
                .with_provider_request_id(Some("req_abc".to_string()))
        },
    ] {
        let bare = build(StatusCode::TOO_MANY_REQUESTS, r#"{"error":"slow down"}"#);
        let bare_text = bare.to_string();
        let with_headers = build(StatusCode::TOO_MANY_REQUESTS, r#"{"error":"slow down"}"#)
            .with_response_headers(Some(retry_after_headers()));
        assert_eq!(with_headers.to_string(), bare_text);
    }
}

/// The wire carries the error's identity — status, body, request id — and
/// not the response's headers: those are the transport's (`date`, the
/// rate-limit counters) and differ on every response to one request, so a
/// record that carried them was never the same twice. A decoded error has
/// `headers: None`: not captured, which is what a replayed error is.
#[test]
fn provider_response_error_round_trips_its_identity_and_not_its_headers() {
    use super::ProviderResponseError;
    let mut headers = http::HeaderMap::new();
    headers.append("retry-after", http::HeaderValue::from_static("7"));
    headers.append(
        "date",
        http::HeaderValue::from_static("Thu, 03 Sep 2026 21:39:09 GMT"),
    );
    let error = ProviderResponseError::new(http::StatusCode::TOO_MANY_REQUESTS, "slow")
        .with_provider_request_id(Some("req-1".into()))
        .with_headers(Some(headers));
    let json = serde_json::to_string(&error).unwrap();
    assert!(!json.contains("headers"), "{json}");
    let back: ProviderResponseError = serde_json::from_str(&json).unwrap();
    assert_eq!(back, error.clone().with_headers(None));
    assert!(
        back.headers.is_none(),
        "a decoded error captured no headers"
    );
    // A wire that names headers is another format, refused.
    assert!(
        serde_json::from_str::<ProviderResponseError>(
            r#"{"status":429,"body":"slow","provider_request_id":null,"headers":[]}"#
        )
        .is_err()
    );

    let bare = ProviderResponseError::without_status("gone");
    let back: ProviderResponseError =
        serde_json::from_str(&serde_json::to_string(&bare).unwrap()).unwrap();
    assert_eq!(back, bare);

    let bad = r#"{"status":99,"body":"","provider_request_id":null,"headers":null}"#;
    assert!(serde_json::from_str::<ProviderResponseError>(bad).is_err());
}

/// The machine code a reply names is the transport's own when it gave one,
/// else the body's: `error.code` as a string, else `error.status`, else
/// `error.type`; a numeric `code` is the status again and is skipped; a
/// prose envelope, a non-JSON body and an empty string name none.
#[test]
fn a_reply_names_its_machine_code_from_the_transport_then_the_body() {
    use super::{ProviderResponseError, body_code};
    assert_eq!(
        body_code(
            r#"{"error":{"code":"model_not_found","type":"invalid_request_error","message":"m"}}"#
        )
        .as_deref(),
        Some("model_not_found"),
        "the OpenAI shape: `error.code` first"
    );
    assert_eq!(
        body_code(r#"{"error":{"code":null,"type":"invalid_request_error","message":"m"}}"#)
            .as_deref(),
        Some("invalid_request_error"),
        "a null code falls through to `error.type`"
    );
    assert_eq!(
        body_code(r#"{"error":{"code":404,"status":"NOT_FOUND","message":"m"}}"#).as_deref(),
        Some("NOT_FOUND"),
        "Google's shape: the number is the status, `error.status` is the code"
    );
    assert_eq!(
        body_code(r#"{"type":"error","error":{"type":"authentication_error","message":"m"}}"#)
            .as_deref(),
        Some("authentication_error"),
        "Anthropic's shape: `error.type`"
    );
    assert_eq!(
        body_code(r#"{"error":"Specified model not found"}"#),
        None,
        "prose"
    );
    assert_eq!(
        body_code(r#"{"error":{"code":"","message":"m"}}"#),
        None,
        "empty"
    );
    assert_eq!(body_code("not json"), None);
    assert_eq!(body_code(""), None);

    let bodied = ProviderResponseError::new(
        StatusCode::NOT_FOUND,
        r#"{"error":{"code":"model_not_found","message":"m"}}"#,
    );
    assert_eq!(bodied.machine_code().as_deref(), Some("model_not_found"));
    assert_eq!(bodied.code, None, "the field stays the transport's own");
    let transport = bodied
        .clone()
        .with_code(Some("ResourceNotFoundException".into()));
    assert_eq!(
        transport.machine_code().as_deref(),
        Some("ResourceNotFoundException"),
        "the transport's code wins over the body's"
    );

    // The report a completion failure becomes carries it.
    let report =
        crate::error::ErrorReport::from(&crate::error::ProviderError::ProviderResponse(bodied));
    assert_eq!(report.code.as_deref(), Some("model_not_found"));
    assert_eq!(report.http_status, Some(404));
    let report = crate::error::ErrorReport::from(&crate::error::ProviderError::ProviderResponse(
        ProviderResponseError::new(StatusCode::NOT_FOUND, r#"{"error":"prose"}"#),
    ));
    assert_eq!(report.code, None);
}

#[test]
fn a_refusal_is_never_retryable_whatever_its_status_or_transport_verdict() {
    use super::ProviderResponseError;
    // The provider judged the content; the same call gets the same verdict,
    // so a refusal beats a retryable status and a transient transport verdict.
    let with_status =
        ProviderResponseError::new(http::StatusCode::SERVICE_UNAVAILABLE, "blocked".to_owned())
            .with_refusal(true);
    assert!(!with_status.is_retryable());

    let by_transport = ProviderResponseError::without_status("blocked".to_owned())
        .with_transient(Some(true))
        .with_refusal(true);
    assert!(!by_transport.is_retryable());

    let plain_block =
        ProviderResponseError::without_status("blocked".to_owned()).with_refusal(true);
    assert!(!plain_block.is_retryable());
}

/// An error envelope delivered under a 200 (Gemini's blocked prompt, a 2xx
/// error body): the driver stamps the transport's real status so callers
/// see it, but a success status is no retry verdict — the decoder's
/// `transient` decides, exactly as it does with no status at all. A
/// non-success status still classifies by the status table alone.
#[test]
fn a_success_status_defers_to_the_transport_verdict() {
    use super::ProviderResponseError;
    let transient_under_200 = ProviderResponseError::without_status("blocked".to_owned())
        .with_transient(Some(true))
        .with_status(Some(http::StatusCode::OK));
    assert!(transient_under_200.is_retryable());

    let silent_under_200 = ProviderResponseError::without_status("blocked".to_owned())
        .with_status(Some(http::StatusCode::OK));
    assert!(!silent_under_200.is_retryable());

    let transient_under_400 = ProviderResponseError::without_status("bad".to_owned())
        .with_transient(Some(true))
        .with_status(Some(http::StatusCode::BAD_REQUEST));
    assert!(!transient_under_400.is_retryable());
}
