//! Non-success triage across every shape a transport can report one in.
//!
//! The recorded cassettes only ever exercise the bundled reqwest shape: the
//! transport rejects the reply as `InvalidStatusCodeWithDetails`. But the
//! socket is a public extension point — any [`HttpClientExt`] can be bound
//! to the provider — and a custom one may hand the same 404 back as an `Ok`
//! response carrying the status, or reject it with an empty body: shapes
//! rig's own test double produces. The driver funnels all of them through
//! [`ProviderError::from_transport_error`] / [`ProviderError::from_http_response`],
//! and these cells pin that the recovery this module documents
//! (`CacheExpired { .. } => recreate the cache`) fires on each.

use super::*;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

/// A `cachedContents` resource handle whose transport answers the next
/// request with `response` and nothing after it.
fn caches(response: MockHttpResponse) -> crate::driver::Bound<CachedContents, SequencedHttpClient> {
    crate::driver::Bound::new(
        crate::providers::gemini::Gemini::new("test-key"),
        SequencedHttpClient::new(vec![response]),
    )
    .cached_contents()
}

const GONE: &str =
    r#"{"error":{"code":404,"message":"CachedContent not found (or permission denied)."}}"#;

/// A transport that rejects the 404 without headers — the shape rig's test
/// double produces — must still reach `CacheExpired`.
///
/// Before the triage moved from the variant to the status this fell into
/// the catch-all `Err(error) => Http(error)` arm, so a caller matching
/// `Expired` to recreate the cache saw an opaque transport error instead.
#[tokio::test]
async fn a_status_error_without_captured_headers_still_reports_expired() {
    let error = caches(MockHttpResponse::error(http::StatusCode::NOT_FOUND, GONE))
        .get("cachedContents/abc123")
        .await
        .expect_err("a missing handle should not resolve");

    let ProviderError::CacheExpired { name, response } = &error else {
        panic!("a handle that is gone should report CacheExpired: {error:?}");
    };
    assert_eq!(name, "cachedContents/abc123");
    assert!(
        response.body.contains("permission denied"),
        "{}",
        response.body
    );
    assert_eq!(response.status, Some(http::StatusCode::NOT_FOUND));
}

/// A transport that hands back the 404 as an `Ok` response instead of an
/// error must reach `CacheExpired` too.
///
/// This is the worse half of the same bug: the error body reached
/// `serde_json::from_str::<CachedContent>` and failed there, so the call
/// reported a JSON error ("missing field `name`") for what is plainly a
/// 404 — a status-shaped failure disguised as a parse bug.
#[tokio::test]
async fn a_non_success_response_is_triaged_rather_than_deserialized() {
    let error = caches(MockHttpResponse::ErrorResponse(
        http::StatusCode::NOT_FOUND,
        GONE.into(),
    ))
    .get("cachedContents/abc123")
    .await
    .expect_err("a missing handle should not resolve");

    let ProviderError::CacheExpired { name, response } = &error else {
        panic!("an Ok-wrapped 404 should report CacheExpired, not a parse error: {error:?}");
    };
    assert_eq!(name, "cachedContents/abc123");
    assert!(
        response.body.contains("permission denied"),
        "{}",
        response.body
    );
    assert_eq!(response.status, Some(http::StatusCode::NOT_FOUND));
}

/// Gemini answers a handle that lapsed a while ago with 403 rather than
/// 404, and both mean the same thing to a caller.
#[tokio::test]
async fn a_403_on_an_existing_handle_reports_expired_like_a_404() {
    let error = caches(MockHttpResponse::error(
        http::StatusCode::FORBIDDEN,
        r#"{"error":{"code":403,"message":"You do not have permission to access the CachedContent."}}"#,
    ))
    .delete("cachedContents/abc123")
    .await
    .expect_err("a lapsed handle should not delete");

    assert!(
        matches!(&error, ProviderError::CacheExpired { name, .. } if name == "cachedContents/abc123"),
        "{error:?}"
    );
}

/// A 403 on `create` is not an expiry — there is no handle yet.
///
/// `create` passes `name: None` for exactly this reason: a disabled key or
/// a project without the API enabled answers 403, and calling that
/// `CacheExpired` would put a caller into a recreate loop against an API that
/// will keep refusing.
#[tokio::test]
async fn a_403_on_create_is_an_api_error_not_an_expiry() {
    let error = caches(MockHttpResponse::error(
        http::StatusCode::FORBIDDEN,
        r#"{"error":{"code":403,"message":"Generative Language API has not been used in project 1234 before or it is disabled."}}"#,
    ))
    .create(NewCachedContent::new("gemini-2.5-flash").content("corpus"))
    .await
    .expect_err("a refused create should not succeed");

    let ProviderError::ProviderResponse(response) = &error else {
        panic!("a create that never made a handle cannot be CacheExpired: {error:?}");
    };
    assert_eq!(response.status, Some(http::StatusCode::FORBIDDEN));
    assert!(
        response.body.contains("has not been used in project"),
        "{}",
        response.body
    );
}

/// Everything that is not a 403/404 on a named handle is a preserved
/// provider reply, carrying the status a caller needs to decide whether to
/// retry.
#[tokio::test]
async fn a_server_error_reports_the_status_rather_than_an_expiry() {
    let error = caches(MockHttpResponse::error(
        http::StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"error":{"code":500,"message":"Internal error encountered."}}"#,
    ))
    .get("cachedContents/abc123")
    .await
    .expect_err("a 500 should not resolve");

    let ProviderError::ProviderResponse(response) = &error else {
        panic!("a 500 is not an expiry: {error:?}");
    };
    assert_eq!(
        response.status,
        Some(http::StatusCode::INTERNAL_SERVER_ERROR)
    );
    assert!(
        response.body.contains("Internal error"),
        "{}",
        response.body
    );
}

/// A rejection with an empty body is still the provider's reply, with its
/// status and nothing to quote — the same shape every other operation
/// preserves, rather than a stand-in message of rig's own.
#[tokio::test]
async fn a_status_error_with_no_body_still_carries_its_status() {
    // `SequencedHttpClient` answers 501 with an empty body once its scripted
    // responses run out.
    let caches = crate::driver::Bound::new(
        crate::providers::gemini::Gemini::new("test-key"),
        SequencedHttpClient::new(Vec::new()),
    )
    .cached_contents();

    let error = caches
        .get("cachedContents/abc123")
        .await
        .expect_err("an unscripted request should not resolve");

    let ProviderError::ProviderResponse(response) = &error else {
        panic!("a 501 is not an expiry: {error:?}");
    };
    assert_eq!(response.status, Some(http::StatusCode::NOT_IMPLEMENTED));
    assert_eq!(response.body, "");
}
