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

async fn drive(client: RecordingHttpClient, request_id_header: Option<&str>) -> CompletionError {
    drive_as::<super::super::envelope::DirectPayload<serde_json::Value>>(client, request_id_header)
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
