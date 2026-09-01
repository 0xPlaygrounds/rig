//! What a paginated model listing *reports*, as distinct from what it returns
//! (rig#2339 review).
//!
//! The page ceiling is a bound on rig's own loop, so warning about it tells an
//! operator their catalog was truncated. Deciding that from the pagination
//! cursor rather than from loop exhaustion made every multi-page listing say
//! so: the loop breaks while still holding the *previous* page's cursor, so
//! `cursor.is_some()` is true for any listing past its first page. Gemini's
//! catalog paginates in production, so that fired on every real listing.
//!
//! Two of these cells assert that a warning is **absent**, which is only sound
//! while the capture is live — a silenced callsite would make them pass while
//! covering nothing. `common::tracing_capture` proves liveness on every call
//! rather than leaving it to this binary's isolation; see
//! `rig_core::test_utils::scoped_tracing_subscriber_guard` for why that is not
//! something the isolation guard alone can promise.

#![allow(clippy::expect_used)]

#[path = "common/tracing_capture.rs"]
mod tracing_capture;

use rig_core::client::ModelListingClient;
use rig_core::providers::anthropic;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

const CEILING_WARNING: &str = "hit its page ceiling";
const REPEATED_CURSOR_WARNING: &str = "repeated its pagination cursor";

/// One page of Anthropic's `/v1/models` envelope.
fn page(models: &[&str], has_more: bool, last_id: Option<&str>) -> MockHttpResponse {
    let data: Vec<_> = models
        .iter()
        .map(|id| serde_json::json!({"id": id, "display_name": id, "type": "model"}))
        .collect();
    MockHttpResponse::success(
        serde_json::json!({"data": data, "has_more": has_more, "last_id": last_id}).to_string(),
    )
}

/// Pages that exhaust the loop's budget: two cursors that alternate forever, so
/// every request differs from the one before, no repeat is ever observed, and
/// only the bound stops it.
fn budget_exhausting_pages() -> Vec<MockHttpResponse> {
    (0..1_010)
        .map(|i| {
            page(
                &["claude-a"],
                true,
                Some(if i % 2 == 0 { "ping" } else { "pong" }),
            )
        })
        .collect()
}

/// Pages that stall on one cursor.
fn repeated_cursor_pages() -> Vec<MockHttpResponse> {
    vec![
        page(&["claude-a"], true, Some("stuck")),
        page(&["claude-b"], true, Some("stuck")),
    ]
}

/// Drive one listing through a mock transport.
async fn list(pages: Vec<MockHttpResponse>) {
    let client = anthropic::Client::builder()
        .api_key("test-key")
        .http_client(SequencedHttpClient::new(pages))
        .build()
        .expect("client should build");
    client.list_models().await.expect("listing should succeed");
}

/// Everything `pages` logs at WARN, captured against an anchor that exercises
/// **both** warning callsites in the listing loop — so a cell asserting either
/// warning is absent has proof that it would have been captured if emitted.
async fn logs_from_listing(pages: Vec<MockHttpResponse>) -> String {
    tracing_capture::captured_logs(
        tracing::Level::WARN,
        || async {
            list(budget_exhausting_pages()).await;
            list(repeated_cursor_pages()).await;
        },
        &[CEILING_WARNING, REPEATED_CURSOR_WARNING],
        || list(pages.clone()),
    )
    .await
}

/// A listing that ends because the provider said so is not a ceiling, however
/// many pages it took.
#[tokio::test]
async fn a_completed_multi_page_listing_reports_no_page_ceiling() {
    let logs = logs_from_listing(vec![
        page(&["claude-a"], true, Some("claude-a")),
        page(&["claude-b"], true, Some("claude-b")),
        page(&["claude-c"], false, Some("claude-c")),
    ])
    .await;

    assert!(
        !logs.contains(CEILING_WARNING),
        "a completed three-page listing must not report a page ceiling; logged:\n{logs}"
    );
}

/// The counterpart, so the fix cannot be "stop warning at all": a listing that
/// genuinely runs out of its page budget still reports the ceiling.
#[tokio::test]
async fn a_listing_that_exhausts_the_page_budget_still_reports_the_ceiling() {
    let logs = logs_from_listing(budget_exhausting_pages()).await;

    assert!(
        logs.contains(CEILING_WARNING),
        "exhausting the page budget must still be reported; logged:\n{logs}"
    );
}

/// A repeated cursor is its own diagnosis and ends the listing, so it reports
/// that alone — not that *and* a page ceiling for one event.
#[tokio::test]
async fn a_repeated_cursor_reports_only_its_own_warning() {
    let logs = logs_from_listing(repeated_cursor_pages()).await;

    assert!(
        logs.contains(REPEATED_CURSOR_WARNING),
        "the repeated cursor is what ended the listing; logged:\n{logs}"
    );
    assert!(
        !logs.contains(CEILING_WARNING),
        "one event must not also be reported as a page ceiling; logged:\n{logs}"
    );
}
