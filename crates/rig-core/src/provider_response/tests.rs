use http::StatusCode;

/// Asserts the shared funnel preserves a provider's status + body across the
/// three routes every capability error exposes: a non-success HTTP response,
/// a 2xx provider error envelope, and a non-HTTP (gRPC/SDK) transport.
macro_rules! assert_funnel {
    ($err:ty) => {{
        let body = r#"{"error":{"message":"boom"}}"#;

        // Non-success status -> HttpError, with status + body recoverable.
        let err = <$err>::from_http_response(StatusCode::SERVICE_UNAVAILABLE, body);
        assert_eq!(
            err.provider_response_status(),
            Some(StatusCode::SERVICE_UNAVAILABLE),
            concat!(stringify!($err), ": non-success status not preserved"),
        );
        assert_eq!(
            err.provider_response_body(),
            Some(body),
            concat!(stringify!($err), ": non-success body not preserved"),
        );
        assert_eq!(
            err.provider_response_json()
                .expect("valid json")
                .expect("present json")["error"]["message"],
            "boom",
        );

        // A provider error envelope returned with a 2xx status -> ProviderResponse,
        // preserving the (success) status so callers can still see it.
        let err = <$err>::from_http_response(StatusCode::OK, body);
        assert_eq!(
            err.provider_response_status(),
            Some(StatusCode::OK),
            concat!(stringify!($err), ": 2xx envelope status not preserved"),
        );
        assert_eq!(err.provider_response_body(), Some(body));

        // No HTTP status available (gRPC/SDK) -> ProviderResponse with status None.
        let err = <$err>::from_provider_body(body);
        assert_eq!(
            err.provider_response_status(),
            None,
            concat!(
                stringify!($err),
                ": status should be None for provider body"
            ),
        );
        assert_eq!(err.provider_response_body(), Some(body));

        // Empty-body asymmetry: the body is `Some("")` but JSON parses to `Ok(None)`.
        let err = <$err>::from_provider_body("");
        assert_eq!(err.provider_response_body(), Some(""));
        assert!(err.provider_response_json().expect("ok").is_none());

        // rig#2210 — headers are only present when a capture path
        // preserved them. The status+body funnels never have any...
        for err in [
            <$err>::from_http_response(StatusCode::TOO_MANY_REQUESTS, body),
            <$err>::from_http_response(StatusCode::OK, body),
            <$err>::from_provider_body(body),
            <$err>::from_http_response_with_request_id(
                StatusCode::TOO_MANY_REQUESTS,
                body,
                Some("req_abc".to_string()),
            ),
        ] {
            assert!(
                err.provider_response_headers().is_none(),
                concat!(stringify!($err), ": a funnel cannot invent headers"),
            );
            // ...and attaching `None` must not disturb the error.
            let untouched = err.with_response_headers(None);
            assert!(untouched.provider_response_headers().is_none());
            assert_eq!(untouched.provider_response_body(), Some(body));
        }

        // ...but both classifications carry headers once attached, so
        // `Retry-After` stays readable on a 429 whether the provider has a
        // request-id contract (ProviderResponse) or not (HttpError).
        let contract_less = <$err>::from_http_response(StatusCode::TOO_MANY_REQUESTS, body)
            .with_response_headers(Some(retry_after_headers()));
        let contract = <$err>::from_http_response_with_request_id(
            StatusCode::TOO_MANY_REQUESTS,
            body,
            Some("req_abc".to_string()),
        )
        .with_response_headers(Some(retry_after_headers()));

        for (label, err) in [("contract-less", contract_less), ("contract", contract)] {
            let err_ty = stringify!($err);
            assert_eq!(
                err.provider_response_headers()
                    .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                    .and_then(|value| value.to_str().ok()),
                Some("20"),
                "{err_ty}/{label}: captured Retry-After not surfaced",
            );
            // Attaching headers must not disturb the status or body the
            // funnel already preserved.
            assert_eq!(
                err.provider_response_status(),
                Some(StatusCode::TOO_MANY_REQUESTS),
                "{err_ty}/{label}: status lost when headers were attached",
            );
            assert_eq!(
                err.provider_response_body(),
                Some(body),
                "{err_ty}/{label}: body lost when headers were attached",
            );
        }
    }};
}

/// A 429's rate-limit metadata, as a provider would send it.
fn retry_after_headers() -> Box<http::HeaderMap> {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::RETRY_AFTER,
        http::HeaderValue::from_static("20"),
    );
    headers.insert("x-ratelimit-remaining", http::HeaderValue::from_static("0"));
    Box::new(headers)
}

#[test]
fn funnel_preserves_status_and_body_for_every_capability_error() {
    assert_funnel!(crate::completion::CompletionError);
    assert_funnel!(crate::embeddings::embedding::EmbeddingError);
    assert_funnel!(crate::transcription::TranscriptionError);
    assert_funnel!(crate::client::verify::VerifyError);
    assert_funnel!(crate::rerank::RerankError);
    #[cfg(feature = "image")]
    assert_funnel!(crate::image_generation::ImageGenerationError);
    #[cfg(feature = "audio")]
    assert_funnel!(crate::audio_generation::AudioGenerationError);
}

/// rig#2314: the metadata-aware funnel preserves non-success statuses as
/// `ProviderResponse` so the transport id has a home; status, body, and
/// id all stay recoverable, and the id appears in the logged message.
#[test]
fn with_request_id_funnel_preserves_non_success_as_provider_response() {
    let error = crate::completion::CompletionError::from_http_response_with_request_id(
        StatusCode::NOT_FOUND,
        r#"{"error":"nope"}"#,
        Some("req_abc".to_string()),
    );
    assert!(matches!(
        error,
        crate::completion::CompletionError::ProviderResponse(_)
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
/// message unchanged.
#[test]
fn with_request_id_funnel_tolerates_absent_id() {
    let error = crate::completion::CompletionError::from_http_response_with_request_id(
        StatusCode::BAD_REQUEST,
        "bad",
        None,
    );
    assert_eq!(error.provider_request_id(), None);
    assert!(!error.to_string().contains("request id"));
}

/// The metadata-less funnel's classification is untouched: non-success
/// stays transport-shaped, and its accessor reports no id.
#[test]
fn metadata_less_funnel_classification_is_unchanged() {
    let error =
        crate::completion::CompletionError::from_http_response(StatusCode::BAD_REQUEST, "bad");
    assert!(matches!(
        error,
        crate::completion::CompletionError::HttpError(_)
    ));
    assert_eq!(error.provider_request_id(), None);
}

/// rig#2210 × rig#2314: the two pieces of transport metadata are captured
/// on the same path and must not evict each other.
#[test]
fn request_id_and_headers_coexist_on_one_error() {
    let error = crate::completion::CompletionError::from_http_response_with_request_id(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":"slow down"}"#,
        Some("req_abc".to_string()),
    )
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

/// Attaching headers to a contract-less non-success error upgrades the
/// transport variant in place, leaving the classification callers match on
/// (`HttpError`) and the preserved status/body untouched.
#[test]
fn attaching_headers_upgrades_the_transport_variant_in_place() {
    let error = crate::completion::CompletionError::from_http_response(
        StatusCode::TOO_MANY_REQUESTS,
        "slow down",
    )
    .with_response_headers(Some(retry_after_headers()));

    assert!(matches!(
        error,
        crate::completion::CompletionError::HttpError(
            crate::http_client::Error::InvalidStatusCodeWithDetails { .. }
        ),
    ));
    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(error.provider_response_body(), Some("slow down"));
    // The contract-less path reports no id whether or not headers rode along.
    assert_eq!(error.provider_request_id(), None);
}

/// First capture wins on both classifications: the site that saw the
/// response is the authority, and a later attach only fills a gap. Without
/// this, a wrapper that re-attaches would silently replace the real
/// response's headers.
#[test]
fn attaching_headers_never_overwrites_an_earlier_capture() {
    let mut later = http::HeaderMap::new();
    later.insert(http::header::RETRY_AFTER, "999".parse().expect("value"));

    for build in [
        crate::completion::CompletionError::from_http_response,
        |status, body| {
            crate::completion::CompletionError::from_http_response_with_request_id(
                status,
                body,
                Some("req_abc".to_string()),
            )
        },
    ] {
        let error = build(StatusCode::TOO_MANY_REQUESTS, "slow down")
            .with_response_headers(Some(retry_after_headers()))
            .with_response_headers(Some(Box::new(later.clone())));

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
/// capture site can attach unconditionally.
#[test]
fn attaching_headers_to_a_slotless_variant_is_a_no_op() {
    let error = crate::completion::CompletionError::ProviderError("rig diagnostic".to_string())
        .with_response_headers(Some(retry_after_headers()));

    assert!(matches!(
        error,
        crate::completion::CompletionError::ProviderError(_)
    ));
    assert!(error.provider_response_headers().is_none());
    assert_eq!(error.to_string(), "ProviderError: rig diagnostic");
}

/// Display goldens (rig#2315 error matrix): error strings are what
/// callers grep and alert on — message churn must be a reviewed diff.
#[test]
fn display_goldens_for_error_shapes() {
    let with_id = crate::completion::CompletionError::from_http_response_with_request_id(
        StatusCode::NOT_FOUND,
        r#"{"error":"nope"}"#,
        Some("req_abc".to_string()),
    );
    assert_eq!(
        with_id.to_string(),
        r#"ProviderResponseError: status 404 Not Found: {"error":"nope"} (request id: req_abc)"#
    );

    let without_id = crate::completion::CompletionError::from_http_response_with_request_id(
        StatusCode::NOT_FOUND,
        r#"{"error":"nope"}"#,
        None,
    );
    assert_eq!(
        without_id.to_string(),
        r#"ProviderResponseError: status 404 Not Found: {"error":"nope"}"#
    );

    let contract_less = crate::completion::CompletionError::from_http_response(
        StatusCode::NOT_FOUND,
        r#"{"error":"nope"}"#,
    );
    assert_eq!(
        contract_less.to_string(),
        r#"HttpError: Invalid status code 404 Not Found with message: {"error":"nope"}"#
    );

    // The two transport variants display identically.
    let details = crate::http_client::Error::InvalidStatusCodeWithDetails {
        status: StatusCode::NOT_FOUND,
        body: "x".to_string(),
        headers: Box::new(http::HeaderMap::new()),
    };
    let message = crate::http_client::Error::InvalidStatusCodeWithMessage(
        StatusCode::NOT_FOUND,
        "x".to_string(),
    );
    assert_eq!(details.to_string(), message.to_string());

    // rig#2210: capturing headers must never change the text a caller
    // logs, on either classification.
    for build in [
        crate::completion::CompletionError::from_http_response,
        |status, body| {
            crate::completion::CompletionError::from_http_response_with_request_id(
                status,
                body,
                Some("req_abc".to_string()),
            )
        },
    ] {
        let bare = build(StatusCode::TOO_MANY_REQUESTS, r#"{"error":"slow down"}"#);
        let bare_text = bare.to_string();
        let with_headers = build(StatusCode::TOO_MANY_REQUESTS, r#"{"error":"slow down"}"#)
            .with_response_headers(Some(retry_after_headers()));
        assert_eq!(with_headers.to_string(), bare_text);
    }
}
