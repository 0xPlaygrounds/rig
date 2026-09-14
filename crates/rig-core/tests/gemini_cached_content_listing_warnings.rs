//! What a paginated `cachedContents` listing *reports*, as distinct from what
//! it returns.
//!
//! `list_with_page_size` restates the three termination rules
//! `internal::model_listing::paginate_models` carries — absent cursor, repeated
//! cursor, page ceiling — because it cannot call it (that helper is typed on
//! `Model`/`ModelListingError` and fetches through `get_bytes`, which collapses
//! the 403/404 triage `CachedContentError::Expired` exists for). A restated
//! rule is a rule that can drift, and `crates/rig-core/src/providers/gemini/
//! cached_content.rs`'s unit tests pin only the half of it that is visible in
//! the returned `Vec`.
//!
//! The other half is invisible there by construction: whether rig stopped
//! because Gemini ended the listing or because rig ran out of pages, the caller
//! gets the same `Vec<CachedContent>` and the same `Ok`. Only the log
//! distinguishes a complete answer from a truncated one, so only a log
//! assertion can hold the distinction — which is the same reason
//! `model_listing_warnings.rs` exists for the model catalog (rig#2339: deciding
//! the ceiling from the cursor rather than from loop exhaustion made every
//! multi-page listing claim it had been truncated).
//!
//! Two of these cells assert a warning is **absent**, which is only sound while
//! the capture is live. `common::tracing_capture` proves liveness on every
//! call; see `rig_core::test_utils::scoped_tracing_subscriber_guard` for why
//! this binary's isolation cannot promise that on its own.

#![allow(clippy::expect_used)]

#[path = "common/tracing_capture.rs"]
mod tracing_capture;

use rig_core::providers::gemini;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

const CEILING_WARNING: &str = "hit its page ceiling";
const REPEATED_CURSOR_WARNING: &str = "repeated its pagination cursor";

/// `internal::model_listing::MAX_LISTING_PAGES`, which is `pub(crate)`. Pinned
/// rather than imported: a listing that needs more than a thousand pages of
/// caches does not exist, so this only has to be *at least* the real bound for
/// the ceiling cells to reach it.
const PAGE_BUDGET: usize = 1_000;

/// One page of Gemini's `cachedContents` list envelope.
fn page(names: &[&str], next_page_token: Option<&str>) -> MockHttpResponse {
    let cached_contents: Vec<_> = names
        .iter()
        .map(|name| serde_json::json!({ "name": format!("cachedContents/{name}") }))
        .collect();
    MockHttpResponse::success(
        serde_json::json!({
            "cachedContents": cached_contents,
            "nextPageToken": next_page_token,
        })
        .to_string(),
    )
}

/// Pages that exhaust the loop's budget: two cursors that alternate forever, so
/// every request differs from the one before, no repeat is ever observed, and
/// only the bound stops it.
fn budget_exhausting_pages() -> Vec<MockHttpResponse> {
    (0..PAGE_BUDGET + 10)
        .map(|i| page(&["a"], Some(if i % 2 == 0 { "ping" } else { "pong" })))
        .collect()
}

/// Pages that stall on one cursor.
fn repeated_cursor_pages() -> Vec<MockHttpResponse> {
    vec![page(&["a"], Some("stuck")), page(&["b"], Some("stuck"))]
}

/// Drive one listing through a mock transport.
///
/// Page size 1 because that is what makes the cursor loop reachable at all:
/// Gemini answers up to 1,000 caches per page, so a live listing is one page.
async fn list(pages: Vec<MockHttpResponse>) {
    let client = gemini::Client::builder()
        .api_key("test-key")
        .http_client(SequencedHttpClient::new(pages))
        .build()
        .expect("client should build");
    client
        .cached_contents()
        .list_with_page_size(1)
        .await
        .expect("listing should succeed");
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

/// A listing that ends because Gemini said so is not a ceiling, however many
/// pages it took.
///
/// This is the cell that fails if the ceiling is ever inferred from the
/// pagination cursor instead of from loop exhaustion: the loop breaks while
/// still holding the *previous* page's cursor, so `page_token.is_some()` is
/// true for every listing past its first page.
#[tokio::test]
async fn a_completed_multi_page_listing_reports_no_page_ceiling() {
    let logs = logs_from_listing(vec![
        page(&["a"], Some("p2")),
        page(&["b"], Some("p3")),
        page(&["c"], None),
    ])
    .await;

    assert!(
        !logs.contains(CEILING_WARNING),
        "a completed three-page listing must not report a page ceiling; logged:\n{logs}"
    );
}

/// A single-page listing — every real one — reports nothing at all.
#[tokio::test]
async fn an_ordinary_single_page_listing_reports_nothing() {
    let logs = logs_from_listing(vec![page(&["a", "b"], None)]).await;

    assert!(
        !logs.contains(CEILING_WARNING) && !logs.contains(REPEATED_CURSOR_WARNING),
        "the listing every caller actually makes must be silent; logged:\n{logs}"
    );
}

/// The counterpart, so the fix cannot be "stop warning at all": a listing that
/// genuinely runs out of its page budget still reports the ceiling.
///
/// Nothing else can tell the caller. `list` returns `Ok` with a `Vec` that
/// looks like a complete answer, so an operator who is silently handed the
/// first thousand pages of their caches has no way to know the rest exist.
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
