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
/// URL these handles produce is not malformed — `Gemini::uri`
/// appends the API key with `&` once the path contains a `?`, so
/// `DELETE /v1beta/cachedContents/abc?stale&key=…` is a well-formed request
/// that deletes cache `abc` and returns 200.
#[tokio::test]
async fn a_mis_targeting_handle_never_reaches_the_socket() {
    for smuggled in ["abc?stale", "abc#frag", "abc/def", ""] {
        // No scripted responses: anything that does escape fails twice, once
        // on the error variant and once on the captured request.
        let http_client = SequencedHttpClient::default();
        let caches = bound_caches(http_client.clone());

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
                matches!(error, ProviderError::Request(_)),
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
/// and `Gemini::uri` follows with `&key=` because it now sees a `?`. That
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
    bound_caches(http_client.clone())
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
    let error = ProviderError::from(error);
    assert!(matches!(error, ProviderError::Request(_)), "{error:?}");
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

// The pagination loop's request shape, and the two ways a cursor can fail
// to advance that the decoder decides: absent and empty. A repeated or
// alternating cursor is the driver's to stop (`driver/tests.rs`), since
// `Decoder::continuation` hands it the next request and the loop is its
// own.
//
// Only the malformed-cursor cells are unrecordable: no live response
// carries an empty cursor, and no live cursor carries URL-significant
// characters. Ordinary and multi-page listings are recorded —
// `prompt_caching/explicit_cache_lifecycle` for a single page,
// `cached_content_matrix/edge_list_pagination` for three pages at
// `pageSize=1`.

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

/// A `cachedContents` resource handle whose transport answers the scripted
/// pages in order and `NOT_IMPLEMENTED` once they run out — so a loop that
/// fails to terminate ends its test with an error rather than hanging the
/// suite.
fn caches(
    pages: Vec<MockHttpResponse>,
) -> (
    crate::driver::Bound<CachedContents, SequencedHttpClient>,
    SequencedHttpClient,
) {
    let http_client = SequencedHttpClient::new(pages);
    (bound_caches(http_client.clone()), http_client)
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

// ── the resource API on a bound provider ────────────────────────────────
//
// The cache lifecycle moved off the client layer onto
// `Bound<Gemini, H>::cached_contents()`. These two cells are the in-tree
// proof that it moved *without moving the bytes*: they pin the paths the
// recorded traffic and the axum-stub harness cells
// (`crates/rig-cassette/tests/providers/gemini/support.rs`, which asserts the literal
// `DELETE /v1beta/cachedContents/leaky`) match on.

fn bound_caches(
    http: SequencedHttpClient,
) -> crate::driver::Bound<CachedContents, SequencedHttpClient> {
    crate::driver::Bound::new(crate::providers::gemini::Gemini::new("test-key"), http)
        .cached_contents()
}

#[tokio::test]
async fn the_bound_cache_sends_each_recorded_path_with_the_key_in_the_query() {
    const HANDLE: &str = r#"{"name":"cachedContents/leaky","model":"models/gemini-2.5-flash"}"#;
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(HANDLE),
        MockHttpResponse::success(HANDLE),
        MockHttpResponse::success(HANDLE),
        MockHttpResponse::success("{}"),
    ]);
    let caches = bound_caches(http.clone());

    caches
        .create(NewCachedContent::new("gemini-2.5-flash").content("corpus"))
        .await
        .expect("create decodes");
    caches.get("leaky").await.expect("get decodes");
    caches
        .update_expiry("leaky", CacheExpiry::ttl(Duration::from_secs(600)))
        .await
        .expect("the patch decodes");
    // Storage bills until this is sent, and it is the one request whose
    // path a mistake would aim at another cache.
    caches.delete("cachedContents/leaky").await.expect("delete");

    let uris: Vec<_> = http
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    assert_eq!(
        uris,
        vec![
            "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=test-key",
            "https://generativelanguage.googleapis.com/v1beta/cachedContents/leaky?key=test-key",
            "https://generativelanguage.googleapis.com/v1beta/cachedContents/leaky?updateMask=ttl&key=test-key",
            "https://generativelanguage.googleapis.com/v1beta/cachedContents/leaky?key=test-key",
        ]
    );
}

#[tokio::test]
async fn the_bound_cache_follows_the_listing_cursor() {
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(
            r#"{"cachedContents":[{"name":"cachedContents/one"}],"nextPageToken":"two"}"#,
        ),
        MockHttpResponse::success(r#"{"cachedContents":[{"name":"cachedContents/two"}]}"#),
    ]);
    let caches = bound_caches(http.clone());

    let all = caches
        .list_with_page_size(1)
        .await
        .expect("both pages decode");
    assert_eq!(
        all.iter()
            .map(|cache| cache.name.as_str())
            .collect::<Vec<_>>(),
        vec!["cachedContents/one", "cachedContents/two"]
    );

    let uris: Vec<_> = http
        .requests()
        .into_iter()
        .map(|request| request.uri)
        .collect();
    assert_eq!(
        uris,
        vec![
            "https://generativelanguage.googleapis.com/v1beta/cachedContents?pageSize=1&key=test-key",
            "https://generativelanguage.googleapis.com/v1beta/cachedContents?pageSize=1&pageToken=two&key=test-key",
        ]
    );
}

/// A wire is data a host may serialize into a scene or a config file, and
/// the key is the one credential in it. Nothing serialized may carry it,
/// and what comes back must be the same wire.
#[test]
fn a_serialized_wire_carries_no_key_material_and_round_trips() {
    let wire = crate::providers::gemini::Gemini::new("AIzaSyNOTAREALKEY-0123456789")
        .cached_contents()
        .with_page_size(7);
    let json = serde_json::to_string(&wire).expect("the wire serializes");
    assert!(
        !json.contains("AIzaSyNOTAREALKEY"),
        "the serialized wire leaked the key: {json}"
    );
    assert!(!format!("{wire:?}").contains("AIzaSyNOTAREALKEY"));
    let restored: CachedContents = serde_json::from_str(&json).expect("the wire deserializes");
    assert_eq!(restored.page_size, 7);
    assert_eq!(restored.provider.base_url, wire.provider.base_url);
}

/// The one delete reply shape the cassettes never show — an empty body —
/// still acknowledges: the status already said yes. The same emptiness on a
/// verb that needs the resource is reported as such rather than as a
/// parse failure.
#[tokio::test]
async fn an_empty_body_acknowledges_a_delete_but_answers_no_get() {
    let caches = bound_caches(SequencedHttpClient::new([
        MockHttpResponse::success(""),
        MockHttpResponse::success(""),
    ]));
    caches
        .delete("cachedContents/leaky")
        .await
        .expect("an empty 200 acknowledges the delete");
    let error = caches
        .get("cachedContents/leaky")
        .await
        .expect_err("an empty 200 carries no resource");
    assert!(matches!(error, ProviderError::Response(_)), "{error:?}");
}

/// A body that names a resource and then fails to deserialize is the
/// defect it is.
///
/// The union this replaced read `{"name": 5}` through a flattened
/// `Option<CachedContent>`, which swallows the deserialization error: the
/// call reported a *missing* cached content for a body that carried one,
/// badly. The shape is now decided before the typed decode, so the decode
/// failure reaches the caller as one.
#[tokio::test]
async fn a_malformed_resource_body_is_a_decode_error() {
    let caches = bound_caches(SequencedHttpClient::new([MockHttpResponse::success(
        r#"{"name":5}"#,
    )]));

    let error = caches
        .get("leaky")
        .await
        .expect_err("a `name` that is not a string cannot decode");

    assert!(matches!(error, ProviderError::Json(_)), "{error:?}");
}

/// The `{}` every recorded delete is answered with is the whole reply —
/// [`CachedContentReply::Acknowledged`], not a resource that is absent.
#[tokio::test]
async fn a_deletes_empty_object_is_the_acknowledgement() {
    let wire = crate::providers::gemini::Gemini::new("test-key").cached_contents();
    let http = SequencedHttpClient::new([MockHttpResponse::success("{}")]);

    let reply = crate::driver::call(
        &wire,
        &http,
        CachedContentRequest::Delete("leaky".to_owned()),
        None,
    )
    .await
    .expect("the empty object acknowledges the delete");

    assert!(
        matches!(reply, CachedContentReply::Acknowledged),
        "{reply:?}"
    );
}

/// A reply of a shape the verb did not ask for names the shape it carried,
/// in both directions. The union could not: a page reaching a `get` left
/// its `Option<CachedContent>` empty, which read as "no cached content",
/// and a resource reaching a `list` left its `Vec` empty, which read as an
/// empty collection.
#[tokio::test]
async fn a_reply_of_the_other_shape_names_the_shape_it_carried() {
    let caches = bound_caches(SequencedHttpClient::new([
        cached_page(&["a"], None),
        MockHttpResponse::success(r#"{"name":"cachedContents/leaky"}"#),
    ]));

    let on_get = caches
        .get("leaky")
        .await
        .expect_err("a listing page is not one cached content");
    assert!(
        matches!(&on_get, ProviderError::Response(message) if message.contains("listing page")),
        "{on_get:?}"
    );

    let on_list = caches
        .list()
        .await
        .expect_err("one cached content is not a listing page");
    assert!(
        matches!(&on_list, ProviderError::Response(message) if message.contains("not a listing page")),
        "{on_list:?}"
    );
}
