//! Shared request driver for unary completion endpoints.
//!
//! Every provider builds its own request body and path — those are the real
//! wire differences — but the tail is identical: send the request, split
//! status and body, decode through the provider's success-or-error envelope,
//! record telemetry, trace-log the payload, and preserve raw error bodies via
//! [`CompletionError::from_http_response`]. This driver owns that tail.

use bytes::Bytes;
use serde::de::DeserializeOwned;

use super::envelope::ProviderEnvelope;
use crate::completion::CompletionError;
use crate::http_client::HttpClientExt;

/// Sends a unary completion request and decodes the provider's
/// success-or-error envelope.
///
/// `request` is the provider's fully built POST request; `A` is the
/// provider's own response envelope (use
/// [`DirectPayload`](super::envelope::DirectPayload) when the 2xx body IS the
/// payload); `record_telemetry` records response metadata and token usage on
/// the current span; `label` names the provider in trace/error logs (e.g.
/// `"Gemini completion"`).
///
/// `request_id_header` names the provider's transport request-id response
/// header (e.g. Anthropic `request-id`, OpenAI `x-request-id`); when present
/// and the response carries it, its value is returned alongside the payload.
/// `None` — as the parameter or in the returned pair — means "this provider
/// does not report one", never an error.
///
/// Error paths, preserved exactly:
/// - non-success status → `from_http_response(status, raw_body)`, with the
///   failed response's headers attached so rate-limit metadata such as
///   `Retry-After` stays readable (rig#2210);
/// - 2xx error envelope → warn-log the provider message, preserve raw body;
/// - undecodable 2xx body → error-log the body, surface the JSON error.
pub(crate) async fn send_completion<C, A, F>(
    client: &C,
    request: crate::http_client::Request<Vec<u8>>,
    label: &str,
    request_id_header: Option<&str>,
    record_telemetry: F,
) -> Result<(A::Payload, Option<String>), CompletionError>
where
    C: HttpClientExt,
    A: DeserializeOwned + ProviderEnvelope,
    A::Payload: serde::Serialize,
    F: FnOnce(&A::Payload),
{
    let response = match client.send::<_, Bytes>(request).await {
        Ok(response) => response,
        // The reqwest transport reports a non-success status as an error with
        // the failed response's headers preserved. A provider with a
        // request-id contract reads its header off them so the failed call's
        // transport id — the one support asks for — survives onto the error
        // (rig#2314); classification then follows the contract, so a given
        // provider's errors stay one shape. Either way the whole header map
        // rides along, so a caller can still read `Retry-After` off a 429
        // (rig#2210).
        Err(crate::http_client::Error::InvalidStatusCodeWithDetails {
            status,
            body,
            headers,
        }) => {
            return Err(match request_id_header {
                Some(header) => {
                    let provider_request_id = headers
                        .get(header)
                        .and_then(|value| value.to_str().ok())
                        .filter(|value| !value.is_empty())
                        .map(str::to_string);
                    CompletionError::from_http_response_with_request_id(
                        status,
                        body,
                        provider_request_id,
                    )
                    .with_response_headers(Some(headers))
                }
                // Contract-less providers keep the pre-#2314 transport shape;
                // the details variant is that shape plus the headers, and
                // displays identically.
                None => CompletionError::HttpError(
                    crate::http_client::Error::InvalidStatusCodeWithDetails {
                        status,
                        body,
                        headers,
                    },
                ),
            });
        }
        // A transport that reports non-success without preserved headers (a
        // custom `HttpClientExt`): a contract provider still classifies as
        // ProviderResponse — the shape follows the contract on every
        // transport — with no id to read.
        Err(crate::http_client::Error::InvalidStatusCodeWithMessage(status, body))
            if request_id_header.is_some() =>
        {
            return Err(CompletionError::from_http_response_with_request_id(
                status, body, None,
            ));
        }
        Err(other) => return Err(other.into()),
    };

    // Take the response apart before awaiting the body: that hands over the
    // headers already owned, so preserving them onto an error (rig#2210) costs
    // no clone and every error path below can afford them — including the 2xx
    // error envelope, which is a failure the caller may need to back off from
    // even though its status says success.
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let provider_request_id = request_id_header.and_then(|header| {
        parts
            .headers
            .get(header)
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    });
    let response_headers = Some(Box::new(parts.headers));
    let body = body.await.map_err(CompletionError::HttpError)?;

    if !status.is_success() {
        // A provider with a request-id contract routes through the
        // metadata-aware funnel so the failed call's transport id — the one
        // support asks for — survives onto the error (rig#2314).
        // Classification follows the contract, not the header's presence on
        // a particular response, so a given provider's errors stay one shape.
        return Err(match request_id_header {
            Some(_) => CompletionError::from_http_response_with_request_id(
                status,
                String::from_utf8_lossy(&body),
                provider_request_id,
            ),
            None => CompletionError::from_http_response(status, String::from_utf8_lossy(&body)),
        }
        .with_response_headers(response_headers));
    }

    let envelope: A = serde_json::from_slice(&body).map_err(|err| {
        tracing::error!(
            error = %err,
            body = %String::from_utf8_lossy(&body),
            "failed to deserialize {label} response"
        );
        CompletionError::JsonError(err)
    })?;

    match envelope.into_payload() {
        Ok(payload) => {
            record_telemetry(&payload);
            super::trace_json(
                crate::providers::internal::LogTarget::Completions,
                &format!("{label} response"),
                &payload,
            );
            Ok((payload, provider_request_id))
        }
        Err(message) => {
            tracing::warn!(message = %message, "provider returned an error response");
            // A 2xx error envelope preserves as ProviderResponse either way;
            // the metadata-aware funnel just adds the captured id. Its headers
            // matter as much as a non-success response's: gateways report rate
            // limits this way, with `Retry-After` alongside a 200 (rig#2210).
            Err(match request_id_header {
                Some(_) => CompletionError::from_http_response_with_request_id(
                    status,
                    String::from_utf8_lossy(&body),
                    provider_request_id,
                ),
                None => CompletionError::from_http_response(status, String::from_utf8_lossy(&body)),
            }
            .with_response_headers(response_headers))
        }
    }
}

/// rig#2210: the failed response's headers must survive the driver, so a
/// caller can read `Retry-After` off a 429 and back off correctly.
///
/// The driver sees two transport shapes (the bundled reqwest client reports
/// non-success as an *error* carrying the response's headers; a custom
/// `HttpClientExt` may hand the non-success *response* back) and classifies by
/// two contracts (a provider with a request-id header vs one without). All
/// four cells must preserve the headers.
#[cfg(test)]
mod header_preservation_tests {
    use super::*;
    use crate::test_utils::RecordingHttpClient;

    const CONTRACT: Option<&str> = Some("x-request-id");
    const NO_CONTRACT: Option<&str> = None;
    const BODY: &str = r#"{"error":{"message":"rate limited"}}"#;

    /// A 429's rate-limit metadata plus a request id, as a provider sends it.
    fn rate_limited_headers() -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::RETRY_AFTER,
            http::HeaderValue::from_static("20"),
        );
        headers.insert("x-ratelimit-remaining", http::HeaderValue::from_static("0"));
        headers.insert("x-request-id", http::HeaderValue::from_static("req_abc"));
        headers
    }

    /// A 2xx body that the provider's own envelope classifies as an error —
    /// the `into_payload` failure branch, which no `DirectPayload` can reach.
    struct RejectingEnvelope;

    impl<'de> serde::Deserialize<'de> for RejectingEnvelope {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            serde::de::IgnoredAny::deserialize(deserializer)?;
            Ok(Self)
        }
    }

    impl super::super::envelope::ProviderEnvelope for RejectingEnvelope {
        type Payload = serde_json::Value;

        fn into_payload(self) -> Result<Self::Payload, String> {
            Err("envelope".to_string())
        }
    }

    async fn drive(
        client: RecordingHttpClient,
        request_id_header: Option<&str>,
    ) -> CompletionError {
        drive_as::<super::super::envelope::DirectPayload<serde_json::Value>>(
            client,
            request_id_header,
        )
        .await
    }

    async fn drive_as<A>(
        client: RecordingHttpClient,
        request_id_header: Option<&str>,
    ) -> CompletionError
    where
        A: DeserializeOwned + ProviderEnvelope<Payload = serde_json::Value>,
    {
        let request = crate::http_client::Request::builder()
            .method(http::Method::POST)
            .uri("https://example.test/v1/chat")
            .body(Vec::new())
            .expect("valid request");
        send_completion::<_, A, _>(&client, request, "test provider", request_id_header, |_| {})
            .await
            .expect_err("the scripted response is a failure")
    }

    fn assert_rate_limit_metadata_survived(error: &CompletionError, cell: &str) {
        let headers = error
            .provider_response_headers()
            .unwrap_or_else(|| panic!("{cell}: response headers were dropped by the driver"));
        assert_eq!(
            headers
                .get(http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("20"),
            "{cell}: Retry-After not recoverable",
        );
        assert_eq!(
            headers
                .get("x-ratelimit-remaining")
                .and_then(|value| value.to_str().ok()),
            Some("0"),
            "{cell}: x-ratelimit-remaining not recoverable",
        );
        // The metadata #2314 already preserved must be untouched.
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS),
            "{cell}: status lost",
        );
        assert_eq!(
            error.provider_response_body(),
            Some(BODY),
            "{cell}: body lost"
        );
    }

    #[tokio::test]
    async fn transport_error_preserves_headers_for_a_contract_provider() {
        let client = RecordingHttpClient::with_error_headers(
            http::StatusCode::TOO_MANY_REQUESTS,
            BODY,
            rate_limited_headers(),
        );
        let error = drive(client, CONTRACT).await;

        assert_rate_limit_metadata_survived(&error, "transport-error/contract");
        // The id keeps its #2314 home and classification.
        assert!(matches!(error, CompletionError::ProviderResponse(_)));
        assert_eq!(error.provider_request_id(), Some("req_abc"));
    }

    #[tokio::test]
    async fn transport_error_preserves_headers_for_a_contract_less_provider() {
        let client = RecordingHttpClient::with_error_headers(
            http::StatusCode::TOO_MANY_REQUESTS,
            BODY,
            rate_limited_headers(),
        );
        let error = drive(client, NO_CONTRACT).await;

        assert_rate_limit_metadata_survived(&error, "transport-error/contract-less");
        // Contract-less providers keep the transport classification.
        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(error.provider_request_id(), None);
    }

    #[tokio::test]
    async fn non_success_response_preserves_headers_for_a_contract_provider() {
        let client = RecordingHttpClient::with_error_response_headers(
            http::StatusCode::TOO_MANY_REQUESTS,
            BODY,
            rate_limited_headers(),
        );
        let error = drive(client, CONTRACT).await;

        assert_rate_limit_metadata_survived(&error, "response/contract");
        assert!(matches!(error, CompletionError::ProviderResponse(_)));
        assert_eq!(error.provider_request_id(), Some("req_abc"));
    }

    #[tokio::test]
    async fn non_success_response_preserves_headers_for_a_contract_less_provider() {
        let client = RecordingHttpClient::with_error_response_headers(
            http::StatusCode::TOO_MANY_REQUESTS,
            BODY,
            rate_limited_headers(),
        );
        let error = drive(client, NO_CONTRACT).await;

        assert_rate_limit_metadata_survived(&error, "response/contract-less");
        assert!(matches!(error, CompletionError::HttpError(_)));
    }

    /// A transport that reports non-success *without* headers still classifies
    /// exactly as before, reporting `None` rather than an empty map — "not
    /// captured" must stay distinguishable from "the response had none".
    #[tokio::test]
    async fn header_less_transport_reports_no_headers() {
        for (contract, expect_provider_response) in [(CONTRACT, true), (NO_CONTRACT, false)] {
            let client = RecordingHttpClient::with_error(http::StatusCode::TOO_MANY_REQUESTS, BODY);
            let error = drive(client, contract).await;

            assert!(error.provider_response_headers().is_none());
            assert_eq!(
                matches!(error, CompletionError::ProviderResponse(_)),
                expect_provider_response,
                "classification must not depend on header capture",
            );
            assert_eq!(error.provider_response_body(), Some(BODY));
        }
    }

    /// A 2xx error envelope is still a failure the caller may need to back off
    /// from — gateways report rate limits this way, with `Retry-After` beside a
    /// 200 — so it carries the response's headers like any other error.
    #[tokio::test]
    async fn success_status_error_envelope_preserves_headers() {
        let client = RecordingHttpClient::with_error_response_headers(
            http::StatusCode::OK,
            r#"{"error":{"message":"envelope"}}"#,
            rate_limited_headers(),
        );
        let error = drive_as::<RejectingEnvelope>(client, CONTRACT).await;

        assert!(matches!(error, CompletionError::ProviderResponse(_)));
        assert_eq!(
            error
                .provider_response_headers()
                .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                .and_then(|value| value.to_str().ok()),
            Some("20"),
            "a 200-with-error-envelope must expose its Retry-After too",
        );
        assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
        assert_eq!(error.provider_request_id(), Some("req_abc"));
    }

    /// The success path must not be taxed for the error paths' benefit: the
    /// driver takes the response apart rather than cloning its header map, so a
    /// completed turn allocates nothing extra. Pinned behaviorally — a
    /// successful call still returns its payload and id.
    #[tokio::test]
    async fn successful_response_is_unaffected_by_header_capture() {
        let client = RecordingHttpClient::with_error_response_headers(
            http::StatusCode::OK,
            r#"{"ok":true}"#,
            rate_limited_headers(),
        );
        let request = crate::http_client::Request::builder()
            .method(http::Method::POST)
            .uri("https://example.test/v1/chat")
            .body(Vec::new())
            .expect("valid request");
        let (payload, request_id) = send_completion::<
            _,
            super::super::envelope::DirectPayload<serde_json::Value>,
            _,
        >(&client, request, "test provider", CONTRACT, |_| {})
        .await
        .expect("a 2xx payload should decode");

        assert_eq!(payload["ok"], true);
        assert_eq!(request_id.as_deref(), Some("req_abc"));
    }
}
