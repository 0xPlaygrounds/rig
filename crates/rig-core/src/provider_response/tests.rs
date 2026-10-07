use super::ProviderResponseError;
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

    let error = crate::error::ProviderError::Http(crate::http_client::Error::StreamEnded.into())
        .with_response_headers(Some(retry_after_headers()));
    assert!(matches!(
        error,
        crate::error::ProviderError::Http(ref error)
            if matches!(**error, crate::http_client::Error::StreamEnded)
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

/// A provider error that arrives with no HTTP status (an error frame inside
/// a stream, a gRPC reply) is still retried when the provider's own code
/// says the condition is transient. Ported from #2524, with one deliberate
/// change: a known code now outranks the `transient` hint.
#[test]
fn a_body_borne_transient_code_is_retryable_without_a_status() {
    let overloaded = ProviderResponseError::without_status(
        r#"{"error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    );
    assert_eq!(
        overloaded.machine_code().as_deref(),
        Some("overloaded_error")
    );
    assert!(overloaded.is_retryable());

    let throttled = ProviderResponseError::without_status(
        r#"{"error":{"type":"rate_limit_error","message":"slow down"}}"#,
    );
    assert!(throttled.is_retryable());

    let grpc = ProviderResponseError::without_status(
        r#"{"error":{"status":"UNAVAILABLE","message":"backend unavailable"}}"#,
    );
    assert!(grpc.is_retryable(), "the same condition, spelled by gRPC");

    // A spent quota is not transient: the same call gets the same answer.
    let quota = ProviderResponseError::without_status(
        r#"{"error":{"type":"insufficient_quota","message":"billing"}}"#,
    );
    assert!(!quota.is_retryable());

    // A code the table does not know decides nothing.
    let unknown = ProviderResponseError::without_status(
        r#"{"error":{"type":"invalid_request_error","message":"bad"}}"#,
    );
    assert!(!unknown.is_retryable());

    // The provider's own code outranks a decoder's hint.
    let hinted = ProviderResponseError::without_status(
        r#"{"error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    )
    .with_transient(Some(false));
    assert!(hinted.is_retryable());

    // A refusal is never retried, whatever the code says.
    let refusal = ProviderResponseError::without_status(
        r#"{"error":{"type":"server_error","message":"blocked"}}"#,
    )
    .with_refusal(true);
    assert!(!refusal.is_retryable());
}

/// Every place a provider puts its error, as [`ProviderResponseError::from_body`]
/// reads it: (body, machine code, status, retryable).
#[test]
fn the_envelope_locator_reads_every_shape_once() {
    let cases: &[(&str, Option<&str>, Option<u16>, bool)] = &[
        // (a) Responses `response.failed`: the error nests in the response.
        (
            r#"{"type":"response.failed","response":{"error":{"code":"server_error","message":"x"}}}"#,
            Some("server_error"),
            None,
            true,
        ),
        (
            r#"{"type":"response.failed","response":{"error":{"code":"invalid_prompt","message":"x"}}}"#,
            Some("invalid_prompt"),
            None,
            false,
        ),
        // (b) A top-level `error` object: Anthropic, Chat-compatible, Gemini.
        (
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            Some("overloaded_error"),
            None,
            true,
        ),
        (
            r#"{"error":{"code":"rate_limit_exceeded","type":"requests"}}"#,
            Some("rate_limit_exceeded"),
            None,
            true,
        ),
        // A name says more than a number; the number is the status.
        (
            r#"{"error":{"code":503,"status":"UNAVAILABLE","message":"x"}}"#,
            Some("UNAVAILABLE"),
            Some(503),
            true,
        ),
        (
            r#"{"error":{"code":500,"status":"INTERNAL","message":"x"}}"#,
            Some("INTERNAL"),
            Some(500),
            true,
        ),
        (r#"{"error":{"code":429}}"#, Some("429"), Some(429), true),
        (r#"{"error":{"code":400}}"#, Some("400"), Some(400), false),
        // A numeric string is a name the table does not know, not a status.
        (r#"{"error":{"code":"503"}}"#, Some("503"), None, false),
        // A number outside 400..=599 is neither a status nor a verdict.
        (r#"{"error":{"code":14}}"#, Some("14"), None, false),
        (r#"{"error":{"code":200}}"#, Some("200"), None, false),
        // An `error` string is a message with no code.
        (r#"{"error":"model failed to load"}"#, None, None, false),
        // (c) The document is itself the error event: its code is `code`
        // alone (`type` is the event tag), and it names no status.
        (
            r#"{"type":"error","code":"server_error","message":"x"}"#,
            Some("server_error"),
            None,
            true,
        ),
        (
            r#"{"type":"error","code":503,"message":"x"}"#,
            Some("503"),
            None,
            true,
        ),
        (r#"{"type":"error","message":"x"}"#, None, None, false),
        // No envelope at all.
        (r#"{"error":null}"#, None, None, false),
        (r#"{"error":{}}"#, None, None, false),
        (r#"{"error":""}"#, None, None, false),
        (
            r#"{"type":"response.failed","response":{"error":null}}"#,
            None,
            None,
            false,
        ),
        ("plain text", None, None, false),
        ("", None, None, false),
    ];
    for (body, code, status, retryable) in cases {
        let reply = ProviderResponseError::from_body(*body);
        assert_eq!(reply.machine_code().as_deref(), *code, "{body}");
        assert_eq!(reply.status.map(|s| s.as_u16()), *status, "{body}");
        assert_eq!(reply.is_retryable(), *retryable, "{body}");
        let report = crate::error::ProviderError::from_provider_body(*body).report();
        assert_eq!(report.http_status, *status, "{body}");
        assert_eq!(report.retryable, *retryable, "{body}");
        assert_eq!(report.code.as_deref(), *code, "{body}");
    }
}

/// The table: every transport's spelling of a transient condition retries,
/// a spent quota and gRPC `INTERNAL` do not, and names match whatever their
/// case.
#[test]
fn the_code_table_decides_known_codes_in_any_case() {
    for (code, retryable) in [
        ("UNAVAILABLE", true),
        ("RESOURCE_EXHAUSTED", true),
        ("DEADLINE_EXCEEDED", true),
        ("ABORTED", true),
        ("INTERNAL", false),
        ("ThrottlingException", true),
        ("ModelStreamErrorException", true),
        ("ValidationException", false),
        ("insufficient_quota", false),
        ("Insufficient_Quota", false),
        ("server_is_overloaded", true),
        ("slow_down", true),
    ] {
        let reply = ProviderResponseError::without_status("opaque transport text")
            .with_code(Some(code.to_owned()));
        assert_eq!(reply.is_retryable(), retryable, "{code}");
    }
}

/// Precedence: refusal > non-success status > known code > `transient`.
#[test]
fn a_known_code_outranks_the_hint_and_a_status_outranks_the_code() {
    let overloaded = r#"{"error":{"type":"overloaded_error"}}"#;
    let quota = r#"{"error":{"type":"insufficient_quota"}}"#;
    let unknown = r#"{"error":{"type":"mystery"}}"#;
    // A decoder cannot stamp a verdict that contradicts the body.
    assert!(
        ProviderResponseError::without_status(overloaded)
            .with_transient(Some(false))
            .is_retryable()
    );
    assert!(
        !ProviderResponseError::without_status(quota)
            .with_transient(Some(true))
            .is_retryable()
    );
    // An unknown code leaves the hint to decide.
    assert!(
        ProviderResponseError::without_status(unknown)
            .with_transient(Some(true))
            .is_retryable()
    );
    assert!(!ProviderResponseError::without_status(unknown).is_retryable());
    // An explicit code is the code, even when the body names another.
    assert!(
        !ProviderResponseError::without_status(overloaded)
            .with_code(Some("mystery".to_owned()))
            .is_retryable()
    );
    // A non-success status decides before any code.
    assert!(!ProviderResponseError::new(StatusCode::BAD_REQUEST, overloaded).is_retryable());
    assert!(ProviderResponseError::new(StatusCode::SERVICE_UNAVAILABLE, quota).is_retryable());
    // A 2xx envelope is classified by its code, as in band.
    assert!(ProviderResponseError::new(StatusCode::OK, overloaded).is_retryable());
    // A refusal beats everything.
    assert!(
        !ProviderResponseError::new(StatusCode::SERVICE_UNAVAILABLE, overloaded)
            .with_refusal(true)
            .is_retryable()
    );
}

/// A record read back re-derives its verdict from its body, so the wire
/// needs no `transient` for an envelope with a known code.
#[test]
fn a_deserialized_record_re_derives_its_verdict_from_the_body() {
    let reply = ProviderResponseError::from_body(r#"{"error":{"type":"overloaded_error"}}"#);
    let Ok(wire) = serde_json::to_string(&reply) else {
        panic!("the record serializes");
    };
    assert!(!wire.contains("transient"), "{wire}");
    let Ok(back) = serde_json::from_str::<ProviderResponseError>(&wire) else {
        panic!("the record reads back: {wire}");
    };
    assert!(back.is_retryable());
    assert_eq!(back, reply);
}

/// The located error as JSON: nested errors under `error`, an error event
/// as itself.
#[test]
fn the_envelope_json_is_the_located_error() {
    use serde_json::json;
    for (body, error) in [
        (
            r#"{"type":"response.failed","response":{"error":{"code":"x"}}}"#,
            Some(json!({"error": {"code": "x"}})),
        ),
        (
            r#"{"type":"error","error":{"type":"overloaded_error"}}"#,
            Some(json!({"error": {"type": "overloaded_error"}})),
        ),
        (r#"{"error":"boom"}"#, Some(json!({"error": "boom"}))),
        (
            r#"{"type":"error","code":"x"}"#,
            Some(json!({"type": "error", "code": "x"})),
        ),
        (r#"{"error":null}"#, None),
        ("text", None),
    ] {
        let reply = ProviderResponseError::without_status(body);
        assert_eq!(reply.envelope_json(), error, "{body}");
    }
}
