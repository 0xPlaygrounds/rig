//! Non-success triage across every shape a transport can report one in.
//!
//! The recorded cassettes only ever exercise the bundled reqwest shape,
//! `http_client::Error::InvalidStatusCodeWithDetails`. But `H` is a public
//! extension point (`ClientBuilder::http_client`), and a custom
//! [`HttpClientExt`] may report the same 404 as a bare
//! `InvalidStatusCode`, as `InvalidStatusCodeWithMessage`, or as an `Ok`
//! response carrying the status — shapes rig's own test double produces.
//! On those the triage used to fall through to `CachedContentError::Http`
//! or to a bogus deserialization error, so the recovery this module
//! documents (`Expired { .. } => recreate the cache`) silently never fired.

use super::*;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

/// A `cachedContents` client whose transport answers the next request with
/// `response` and nothing after it.
fn caches(response: MockHttpResponse) -> CachedContentClient<SequencedHttpClient> {
    Client::builder()
        .api_key("test-key")
        .http_client(SequencedHttpClient::new(vec![response]))
        .build()
        .expect("client should build")
        .cached_contents()
}

const GONE: &str =
    r#"{"error":{"code":404,"message":"CachedContent not found (or permission denied)."}}"#;

/// A transport that reports the 404 as `InvalidStatusCodeWithMessage` —
/// the variant every non-bundled `HttpClientExt` in rig produces — must
/// still reach `Expired`.
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

    let CachedContentError::Expired { name, message } = &error else {
        panic!("a handle that is gone should report Expired: {error:?}");
    };
    assert_eq!(name, "cachedContents/abc123");
    assert!(message.contains("permission denied"), "{message}");
}

/// A transport that hands back the 404 as an `Ok` response instead of an
/// error must reach `Expired` too.
///
/// This is the worse half of the same bug: the error body reached
/// `serde_json::from_str::<CachedContent>` and failed there, so the call
/// reported `CachedContentError::Serde` ("missing field `name`") for what
/// is plainly a 404 — a status-shaped failure disguised as a parse bug.
#[tokio::test]
async fn a_non_success_response_is_triaged_rather_than_deserialized() {
    let error = caches(MockHttpResponse::ErrorResponse(
        http::StatusCode::NOT_FOUND,
        GONE.into(),
    ))
    .get("cachedContents/abc123")
    .await
    .expect_err("a missing handle should not resolve");

    let CachedContentError::Expired { name, message } = &error else {
        panic!("an Ok-wrapped 404 should report Expired, not a parse error: {error:?}");
    };
    assert_eq!(name, "cachedContents/abc123");
    assert!(message.contains("permission denied"), "{message}");
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
        matches!(&error, CachedContentError::Expired { name, .. } if name == "cachedContents/abc123"),
        "{error:?}"
    );
}

/// A 403 on `create` is not an expiry — there is no handle yet.
///
/// `create` passes `name: None` for exactly this reason: a disabled key or
/// a project without the API enabled answers 403, and calling that
/// `Expired` would put a caller into a recreate loop against an API that
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

    let CachedContentError::Api { status, message } = &error else {
        panic!("a create that never made a handle cannot be Expired: {error:?}");
    };
    assert_eq!(*status, 403);
    assert!(
        message.contains("has not been used in project"),
        "{message}"
    );
}

/// Everything that is not a 403/404 on a named handle is an `Api` failure,
/// carrying the status a caller needs to decide whether to retry.
#[tokio::test]
async fn a_server_error_reports_the_status_rather_than_an_expiry() {
    let error = caches(MockHttpResponse::error(
        http::StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"error":{"code":500,"message":"Internal error encountered."}}"#,
    ))
    .get("cachedContents/abc123")
    .await
    .expect_err("a 500 should not resolve");

    let CachedContentError::Api { status, message } = &error else {
        panic!("a 500 is not an expiry: {error:?}");
    };
    assert_eq!(*status, 500);
    assert!(message.contains("Internal error"), "{message}");
}

/// `InvalidStatusCode` carries no body, so there is no provider text to
/// quote. The message must still say something: an empty one leaves the
/// error Display ending in a bare colon, which reads as truncated output
/// rather than as a provider that said nothing.
#[tokio::test]
async fn a_status_error_with_no_body_still_names_why_it_has_no_message() {
    // `SequencedHttpClient` reports `InvalidStatusCode(501)` once its
    // scripted responses run out, which is the body-less shape.
    let caches = Client::builder()
        .api_key("test-key")
        .http_client(SequencedHttpClient::new(Vec::new()))
        .build()
        .expect("client should build")
        .cached_contents();

    let error = caches
        .get("cachedContents/abc123")
        .await
        .expect_err("an unscripted request should not resolve");

    let CachedContentError::Api { status, message } = &error else {
        panic!("a 501 is not an expiry: {error:?}");
    };
    assert_eq!(*status, 501);
    assert_eq!(message, NO_RESPONSE_BODY);
    assert!(
        !error.to_string().ends_with(": "),
        "the Display must not trail off: {error}"
    );
}
