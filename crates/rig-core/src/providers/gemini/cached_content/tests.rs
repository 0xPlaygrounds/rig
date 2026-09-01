use super::*;
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

#[test]
fn model_is_qualified_idempotently() {
    assert_eq!(qualify_model("gemini-2.5-flash"), "models/gemini-2.5-flash");
    assert_eq!(
        qualify_model("models/gemini-2.5-flash"),
        "models/gemini-2.5-flash"
    );
}

#[test]
fn resource_path_accepts_a_bare_id_or_a_full_handle() {
    assert_eq!(
        resource_path("abc123").expect("a bare id is a handle"),
        "/v1beta/cachedContents/abc123"
    );
    assert_eq!(
        resource_path("cachedContents/abc123").expect("a full handle is a handle"),
        "/v1beta/cachedContents/abc123"
    );
}

/// The destructive path, end to end: a handle that would mis-target must
/// not reach the socket at all.
///
/// `resource_path`'s unit tests prove the string is refused; this proves the
/// refusal happens *before* the request is built. It matters because the
/// URL these handles produce is not malformed — `GeminiExt::build_uri`
/// appends the API key with `&` once the path contains a `?`, so
/// `DELETE /v1beta/cachedContents/abc?stale&key=…` is a well-formed request
/// that deletes cache `abc` and returns 200.
#[tokio::test]
async fn a_mis_targeting_handle_never_reaches_the_socket() {
    for smuggled in ["abc?stale", "abc#frag", "abc/def", ""] {
        // No scripted responses: anything that does escape fails twice, once
        // on the error variant and once on the captured request.
        let http_client = SequencedHttpClient::default();
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("client should build");
        let caches = client.cached_contents();

        let outcomes = [
            ("get", caches.get(smuggled).await.err()),
            ("delete", caches.delete(smuggled).await.err()),
            (
                "update_expiry",
                caches
                    .update_expiry(smuggled, CacheExpiry::ttl(Duration::from_secs(60)))
                    .await
                    .err(),
            ),
        ];
        for (label, error) in outcomes {
            let error =
                error.unwrap_or_else(|| panic!("{label} should refuse the handle {smuggled:?}"));
            assert!(
                matches!(error, CachedContentError::Invalid(_)),
                "{label} on {smuggled:?}: {error:?}"
            );
        }

        assert!(
            http_client.requests().is_empty(),
            "handle {smuggled:?} escaped the process: {:?}",
            http_client.requests()
        );
    }
}

/// The exact URI `update_expiry` builds, so the ordering of its three
/// query-string writers is pinned in one place.
///
/// `resource_path` writes the path, the `format!` appends `?updateMask=`,
/// and `build_uri` follows with `&key=` because it now sees a `?`. That
/// layout is only stable while a handle cannot carry its own `?` — which is
/// what `resource_path` refuses, and what the cells above cover. This cell
/// pins the well-formed side: it passed before the validation existed and
/// exists to catch the mask being concatenated ahead of it, or the path
/// being escaped. The recorded PATCH in
/// `cached_content_matrix/edge_update_expiry_absolute` pins the same layout
/// against the live API; this one names it locally.
#[tokio::test]
async fn update_expiry_puts_its_update_mask_after_the_validated_path() {
    let http_client = SequencedHttpClient::new([MockHttpResponse::success(
        serde_json::json!({
            "name": "cachedContents/n3v1qk0nqz9k",
            "model": "models/gemini-2.5-flash"
        })
        .to_string(),
    )]);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("client should build");

    client
        .cached_contents()
        .update_expiry(
            "cachedContents/n3v1qk0nqz9k",
            CacheExpiry::ttl(Duration::from_secs(600)),
        )
        .await
        .expect("a well-formed handle should be patched");

    let requests = http_client.requests();
    let [request] = requests.as_slice() else {
        panic!("exactly one request should have been sent: {requests:?}");
    };
    assert!(
        request
            .uri
            .ends_with("/v1beta/cachedContents/n3v1qk0nqz9k?updateMask=ttl&key=test-key"),
        "{}",
        request.uri
    );
}

#[test]
fn ttl_serializes_in_geminis_duration_form() {
    assert_eq!(
        CacheExpiry::ttl_string(Duration::from_secs(600)),
        "600.000000000s"
    );
}

/// Expiry is one field on the wire, never two — Gemini rejects a body that
/// carries both, so the builder must replace rather than accumulate.
#[test]
fn setting_expiry_twice_replaces_rather_than_sending_both() {
    let request = NewCachedContent::new("gemini-2.5-flash")
        .content("corpus")
        .expiry(CacheExpiry::ttl(Duration::from_secs(60)))
        .expiry(CacheExpiry::expire_time("2030-01-01T00:00:00Z"));
    assert!(request.ttl.is_none());
    assert_eq!(request.expire_time.as_deref(), Some("2030-01-01T00:00:00Z"));

    let request = request.expiry(CacheExpiry::ttl(Duration::from_secs(60)));
    assert!(request.expire_time.is_none());
    assert!(request.ttl.is_some());
}

#[test]
fn an_empty_cache_is_rejected_before_it_bills_for_storage() {
    let error = NewCachedContent::new("gemini-2.5-flash")
        .display_name("empty")
        .validate()
        .expect_err("an empty cached content should be refused");
    assert!(matches!(error, CachedContentError::Invalid(_)), "{error:?}");
}

#[test]
fn create_body_omits_unset_fields() {
    let body = serde_json::to_value(
        NewCachedContent::new("gemini-2.5-flash")
            .content("corpus")
            .expiry(CacheExpiry::ttl(Duration::from_secs(600))),
    )
    .expect("serialize");
    let object = body.as_object().expect("object");
    assert!(!object.contains_key("expireTime"));
    assert!(!object.contains_key("tools"));
    assert!(!object.contains_key("systemInstruction"));
    assert_eq!(
        object.get("model").and_then(|m| m.as_str()),
        Some("models/gemini-2.5-flash")
    );
}

// The pagination loop, and every way its cursor can fail to advance:
// absent, empty, repeated, and alternating — the last of which only the
// page ceiling catches. `paginate_models` carries the same three rules for
// model listings, but this resource cannot call it (it is typed on
// `Model`/`ModelListingError` and fetches through `get_bytes`, which
// collapses the 403/404 triage `CachedContentError::Expired` exists for),
// so the rules are restated in `list_with_page_size` and pinned here.
//
// Only the malformed-cursor cells are unrecordable: no live response
// carries an empty, repeated or alternating cursor, and no live cursor
// carries URL-significant characters. Ordinary and multi-page listings are
// recorded — `prompt_caching/explicit_cache_lifecycle` for a single page,
// `cached_content_matrix/edge_list_pagination` for three pages at
// `pageSize=1`. These cells exist to pin the three termination guards,
// which a recording cannot exercise.

/// One page of Gemini's `cachedContents` list envelope.
fn cached_page(names: &[&str], next_page_token: Option<&str>) -> MockHttpResponse {
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

/// A `cachedContents` client whose transport answers the scripted pages in
/// order and `NOT_IMPLEMENTED` once they run out — so a loop that fails to
/// terminate ends its test with an error rather than hanging the suite.
fn caches(
    pages: Vec<MockHttpResponse>,
) -> (
    CachedContentClient<SequencedHttpClient>,
    SequencedHttpClient,
) {
    let http_client = SequencedHttpClient::new(pages);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("client should build");
    (client.cached_contents(), http_client)
}

/// The ordinary single-page listing — what `list()`'s default page size
/// returns for any realistic collection — is unchanged by the termination
/// guards. Recorded live in `prompt_caching/explicit_cache_lifecycle`;
/// repeated here so the guards have a no-cursor baseline on the same mock
/// transport as the cells below.
#[tokio::test]
async fn single_page_listing_is_unchanged() {
    let (caches, http_client) = caches(vec![cached_page(&["a", "b"], None)]);

    let listed = caches.list().await.expect("listing should succeed");

    let names: Vec<_> = listed.iter().map(|entry| entry.name.as_str()).collect();
    assert_eq!(names, ["cachedContents/a", "cachedContents/b"]);
    assert_eq!(http_client.remaining_responses(), 0);
}

/// An empty `nextPageToken` is as unusable as an absent one: re-sending an
/// empty `pageToken` returns the same page forever.
#[tokio::test]
async fn pagination_stops_on_an_empty_cursor() {
    let (caches, http_client) = caches(vec![
        cached_page(&["a"], Some("")),
        cached_page(&["b"], None),
    ]);

    let listed = caches
        .list_with_page_size(1)
        .await
        .expect("listing should terminate");

    let names: Vec<_> = listed.iter().map(|entry| entry.name.as_str()).collect();
    assert_eq!(names, ["cachedContents/a"]);
    assert_eq!(http_client.remaining_responses(), 1);
}

/// A server that keeps echoing the same cursor cannot advance the listing
/// either — the next request would be byte-identical to the one just
/// answered, so the same page would come back forever.
#[tokio::test]
async fn pagination_stops_on_a_cursor_that_does_not_advance() {
    let (caches, http_client) = caches(vec![
        cached_page(&["a"], Some("stuck")),
        cached_page(&["b"], Some("stuck")),
        cached_page(&["c"], Some("stuck")),
    ]);

    let listed = caches
        .list_with_page_size(1)
        .await
        .expect("listing should terminate");

    let names: Vec<_> = listed.iter().map(|entry| entry.name.as_str()).collect();
    assert_eq!(
        names,
        ["cachedContents/a", "cachedContents/b"],
        "the repeat is only detectable on the second page, so both are kept",
    );
    assert_eq!(http_client.remaining_responses(), 1);
}

/// A cursor that keeps *changing* without making progress — a gateway
/// alternating between two values, or minting a fresh one per request —
/// defeats the repeat check, which only remembers the previous cursor. Only
/// the page ceiling stops it, and without one `list` never returns while
/// `all` grows without bound (rig#2334).
#[tokio::test]
async fn pagination_stops_at_the_page_ceiling_on_an_alternating_cursor() {
    // Two cursors that alternate forever: every request differs from the
    // one before, so no repeat is ever observed.
    let pages: Vec<_> = (0..MAX_LISTING_PAGES + 10)
        .map(|i| cached_page(&["a"], Some(if i % 2 == 0 { "ping" } else { "pong" })))
        .collect();
    let (caches, http_client) = caches(pages);

    let listed = caches
        .list_with_page_size(1)
        .await
        .expect("the ceiling ends the listing instead of looping");

    assert_eq!(
        listed.len(),
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
/// than interpolated, so it cannot truncate the path or inject a query
/// parameter — Gemini appends `key=` to every URI, so a raw `&` in the
/// cursor would sit next to the credential.
#[tokio::test]
async fn pagination_percent_encodes_the_cursor() {
    let (caches, http_client) = caches(vec![
        cached_page(&["a"], Some("weird token&x=1")),
        cached_page(&["b"], None),
    ]);

    caches
        .list_with_page_size(1)
        .await
        .expect("listing should succeed");

    let uris: Vec<_> = http_client
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    assert!(
        uris[1].contains("pageSize=1&pageToken=weird+token%26x%3D1&key="),
        "the cursor must be percent-encoded: {uris:?}",
    );
}
