use super::*;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

fn page(models: &[&str], has_more: bool, last_id: Option<&str>) -> MockHttpResponse {
    let data: Vec<_> = models
        .iter()
        .map(|id| serde_json::json!({"id": id, "display_name": id, "type": "model"}))
        .collect();
    MockHttpResponse::success(
        serde_json::json!({
            "data": data,
            "has_more": has_more,
            "last_id": last_id,
        })
        .to_string(),
    )
}

fn lister(
    pages: Vec<MockHttpResponse>,
) -> (
    AnthropicModelLister<SequencedHttpClient>,
    SequencedHttpClient,
) {
    let http_client = SequencedHttpClient::new(pages);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("client should build");
    (
        <AnthropicModelLister<_> as crate::client::ConstructModelLister<_>>::construct(&client),
        http_client,
    )
}

/// The pagination loop follows the cursor across pages, sending `after_id`
/// on every request after the first. No fixture can cover this: Anthropic's
/// own catalog fits in one page, so the recorded `models` cassette always
/// answers `has_more: false` and the loop body never runs.
#[tokio::test]
async fn pagination_follows_the_cursor_across_pages() {
    let (lister, http_client) = lister(vec![
        page(&["claude-a"], true, Some("claude-a")),
        page(&["claude-b"], true, Some("claude-b")),
        page(&["claude-c"], false, Some("claude-c")),
    ]);

    let models = lister.list_all().await.expect("listing should succeed");

    let ids: Vec<_> = models.data.iter().map(|model| model.id.as_str()).collect();
    assert_eq!(ids, ["claude-a", "claude-b", "claude-c"]);

    let uris: Vec<_> = http_client
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    assert!(
        uris[0].ends_with("/v1/models"),
        "first page is uncursored: {uris:?}"
    );
    assert!(
        uris[1].ends_with("/v1/models?after_id=claude-a"),
        "second page must carry the first page's cursor: {uris:?}",
    );
    assert!(
        uris[2].ends_with("/v1/models?after_id=claude-b"),
        "third page must carry the second page's cursor: {uris:?}",
    );
}

/// A page claiming more pages while naming no cursor must end the loop.
///
/// Termination follows the cursor, not the flag: re-requesting the
/// uncursored first page cannot make progress, so the pre-fix loop
/// refetched page 1 forever and appended its models on every pass. The
/// scripted pages are finite, so on the unfixed code this test fails
/// (the loop drains them and errors) rather than hanging the suite.
///
/// Unit-tested rather than recorded because Anthropic pairs `has_more`
/// with `last_id`; no live request can produce the malformed page.
#[tokio::test]
async fn pagination_stops_when_a_page_claims_more_but_names_no_cursor() {
    let (lister, http_client) = lister(vec![
        page(&["claude-a"], true, None),
        page(&["claude-a"], true, None),
        page(&["claude-a"], true, None),
    ]);

    let models = lister
        .list_all()
        .await
        .expect("a cursor-less page ends the listing instead of looping");

    let ids: Vec<_> = models.data.iter().map(|model| model.id.as_str()).collect();
    assert_eq!(
        ids,
        ["claude-a"],
        "the page's models are returned exactly once"
    );
    assert_eq!(
        http_client.remaining_responses(),
        2,
        "the loop must stop after the first page rather than re-requesting it",
    );
}

/// An empty page that claims more is the same shape with nothing to return,
/// and must also terminate rather than spin.
#[tokio::test]
async fn pagination_stops_on_an_empty_page_claiming_more() {
    let (lister, http_client) = lister(vec![
        page(&[], true, None),
        page(&["claude-a"], false, None),
    ]);

    let models = lister.list_all().await.expect("listing should terminate");

    assert!(models.data.is_empty());
    assert_eq!(http_client.remaining_responses(), 1);
}

/// An empty cursor is as unusable as an absent one, and is read the same
/// way every other provider-reported identifier in rig is read.
#[tokio::test]
async fn pagination_stops_on_an_empty_cursor() {
    let (lister, http_client) = lister(vec![
        page(&["claude-a"], true, Some("")),
        page(&["claude-b"], false, None),
    ]);

    let models = lister.list_all().await.expect("listing should terminate");

    let ids: Vec<_> = models.data.iter().map(|model| model.id.as_str()).collect();
    assert_eq!(ids, ["claude-a"]);
    assert_eq!(http_client.remaining_responses(), 1);
}

/// A final page that names no cursor is the ordinary end of a listing:
/// the flag is authoritative for *stopping*, the cursor only for
/// *continuing*, so a missing cursor on a `has_more: false` page is not an
/// error.
#[tokio::test]
async fn stops_when_the_last_page_names_no_cursor() {
    let (lister, http_client) = lister(vec![page(&["claude-a"], false, None)]);

    let models = lister.list_all().await.expect("listing should succeed");

    assert_eq!(models.data.len(), 1);
    assert_eq!(http_client.remaining_responses(), 0);
}

/// A cursor-less page arriving mid-pagination keeps the pages already
/// fetched rather than discarding them or looping on the last one.
#[tokio::test]
async fn pagination_stops_midway_and_keeps_earlier_pages() {
    let (lister, http_client) = lister(vec![
        page(&["claude-a"], true, Some("claude-a")),
        page(&["claude-b"], true, None),
        page(&["claude-c"], false, None),
    ]);

    let models = lister.list_all().await.expect("listing should terminate");

    let ids: Vec<_> = models.data.iter().map(|model| model.id.as_str()).collect();
    assert_eq!(
        ids,
        ["claude-a", "claude-b"],
        "both fetched pages are kept, in order",
    );
    assert_eq!(
        http_client.remaining_responses(),
        1,
        "the loop stops at the cursor-less page",
    );
}

/// A server that keeps echoing the same cursor cannot advance the loop
/// either — the next request is byte-identical to the one just answered.
#[tokio::test]
async fn pagination_stops_on_a_cursor_that_does_not_advance() {
    let (lister, http_client) = lister(vec![
        page(&["claude-a"], true, Some("stuck")),
        page(&["claude-b"], true, Some("stuck")),
        page(&["claude-c"], true, Some("stuck")),
    ]);

    let models = lister.list_all().await.expect("listing should terminate");

    let ids: Vec<_> = models.data.iter().map(|model| model.id.as_str()).collect();
    assert_eq!(
        ids,
        ["claude-a", "claude-b"],
        "the repeat is only detectable on the second page, so both are kept",
    );
    assert_eq!(http_client.remaining_responses(), 1);
}

/// A cursor that keeps *changing* without making progress — a gateway
/// alternating between two values, or minting a fresh one per request —
/// defeats the repeat check, which only remembers the previous cursor.
/// Only the page ceiling stops it, and without one the listing never
/// returns while `all_models` grows without bound.
#[tokio::test]
async fn pagination_stops_at_the_page_ceiling_on_an_alternating_cursor() {
    use crate::providers::internal::model_listing::MAX_LISTING_PAGES;

    // Two cursors that alternate forever: every request differs from the
    // one before, so no repeat is ever observed.
    let pages: Vec<_> = (0..MAX_LISTING_PAGES + 10)
        .map(|i| {
            page(
                &["claude-a"],
                true,
                Some(if i % 2 == 0 { "ping" } else { "pong" }),
            )
        })
        .collect();
    let (lister, http_client) = lister(pages);

    let models = lister
        .list_all()
        .await
        .expect("the ceiling ends the listing instead of looping");

    assert_eq!(
        models.data.len(),
        MAX_LISTING_PAGES,
        "exactly the ceiling's worth of pages is fetched",
    );
    assert_eq!(
        http_client.remaining_responses(),
        10,
        "the loop stops at the ceiling rather than draining every page",
    );
}

/// A cursor carrying URL-significant characters is percent-encoded rather
/// than interpolated, so it cannot inject query parameters or truncate the
/// path. Anthropic's ids are URL-safe today; this pins that the code does
/// not depend on that.
#[tokio::test]
async fn pagination_percent_encodes_the_cursor() {
    let (lister, http_client) = lister(vec![
        page(&["a"], true, Some("weird id&x=1")),
        page(&["b"], false, None),
    ]);

    lister.list_all().await.expect("listing should succeed");

    let uris: Vec<_> = http_client
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    assert!(
        uris[1].ends_with("/v1/models?after_id=weird+id%26x%3D1"),
        "the cursor must be percent-encoded: {uris:?}",
    );
}

/// The ordinary single-page catalog is unchanged by the guard.
#[tokio::test]
async fn single_page_listing_is_unchanged() {
    let (lister, http_client) = lister(vec![page(
        &["claude-a", "claude-b"],
        false,
        Some("claude-b"),
    )]);

    let models = lister.list_all().await.expect("listing should succeed");

    let ids: Vec<_> = models.data.iter().map(|model| model.id.as_str()).collect();
    assert_eq!(ids, ["claude-a", "claude-b"]);
    assert_eq!(http_client.remaining_responses(), 0);
}
