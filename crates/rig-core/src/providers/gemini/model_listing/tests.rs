use super::*;
use crate::driver::WireDriver;
use crate::providers::gemini::Gemini;

#[test]
fn parse_models_page_accepts_omitted_empty_models_list() {
    let page =
        parse_models_page(br#"{}"#, "/v1beta/models?pageSize=1000").expect("page should parse");
    let (models, next_page_token) = (page.models, page.next_cursor);

    assert!(models.is_empty());
    assert_eq!(next_page_token, None);
}

/// The request path is what the recorded cassette matches on, so its exact
/// shape is load-bearing. Page 1 is covered by that replay; this pins the
/// cursored form, which no fixture exercises because Gemini's catalog fits
/// in one page.
#[test]
fn list_models_path_puts_page_size_first_and_encodes_the_cursor() {
    assert_eq!(list_models_path(None), "/v1beta/models?pageSize=1000");
    assert_eq!(
        list_models_path(Some("abc123")),
        "/v1beta/models?pageSize=1000&pageToken=abc123",
    );
    assert_eq!(
        list_models_path(Some("weird token&x=1")),
        "/v1beta/models?pageSize=1000&pageToken=weird+token%26x%3D1",
    );
}

/// An empty `nextPageToken` must read as "no more pages", not as a cursor.
///
/// Treated as a cursor it re-sends an empty `pageToken`, gets the same
/// page back, and loops forever — the listing never returns rather than
/// returning a short list, so the only observable symptom is a hang.
#[test]
fn parse_models_page_treats_an_empty_next_page_token_as_absent() {
    let next_page_token = parse_models_page(
        br#"{"models": [], "nextPageToken": ""}"#,
        "/v1beta/models?pageSize=1000",
    )
    .expect("page should parse")
    .next_cursor;

    assert_eq!(next_page_token, None);
}

/// A real cursor still advances the loop.
#[test]
fn parse_models_page_keeps_a_non_empty_next_page_token() {
    let next_page_token = parse_models_page(
        br#"{"models": [], "nextPageToken": "abc123"}"#,
        "/v1beta/models?pageSize=1000",
    )
    .expect("page should parse")
    .next_cursor;

    assert_eq!(next_page_token.as_deref(), Some("abc123"));
}

#[test]
fn parse_models_page_falls_back_to_name_when_base_model_id_is_missing() {
    let body = br#"{
            "models": [
                {
                    "name": "models/gemini-2.0-flash-001",
                    "displayName": "Gemini 2.0 Flash 001",
                    "description": "Stable Gemini 2.0 Flash",
                    "inputTokenLimit": 1048576
                }
            ]
        }"#;

    let page = parse_models_page(body, "/v1beta/models?pageSize=1000").expect("page should parse");
    let (models, next_page_token) = (page.models, page.next_cursor);

    assert_eq!(next_page_token, None);
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "gemini-2.0-flash-001");
    assert_eq!(models[0].name.as_deref(), Some("Gemini 2.0 Flash 001"));
    assert_eq!(
        models[0].description.as_deref(),
        Some("Stable Gemini 2.0 Flash")
    );
    assert_eq!(models[0].context_length, Some(1_048_576));
}

#[test]
fn parse_models_page_prefers_base_model_id_when_present() {
    let body = br#"{
            "models": [
                {
                    "name": "models/gemini-2.0-flash-001",
                    "baseModelId": "gemini-2.0-flash",
                    "displayName": "Gemini 2.0 Flash 001"
                }
            ]
        }"#;

    let models = parse_models_page(body, "/v1beta/models?pageSize=1000")
        .expect("page should parse")
        .models;

    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "gemini-2.0-flash");
}

#[test]
fn parse_models_page_reports_missing_model_id_when_name_is_omitted() {
    let error = parse_models_page(br#"{"models":[{}]}"#, "/v1beta/models?pageSize=1000")
        .expect_err("entry without name/baseModelId should fail with contextual error");

    match error {
        ProviderError::Response(message) => {
            assert!(message.contains("provider=Gemini"));
            assert!(message.contains("path=/v1beta/models?pageSize=1000"));
            assert!(message.contains(
                "parse_error=model entry missing usable `baseModelId` and `name` values"
            ));
        }
        _ => panic!("expected a response error"),
    }
}

#[test]
fn parse_models_page_returns_parse_error_when_entry_has_no_usable_id() {
    let body = br#"{
            "models": [
                {
                    "name": "models/",
                    "baseModelId": "   ",
                    "displayName": "Broken Gemini"
                }
            ]
        }"#;

    let error = parse_models_page(body, "/v1beta/models?pageSize=1000")
        .expect_err("page should fail when no usable ID is available");

    match error {
        ProviderError::Response(message) => {
            assert!(message.contains("provider=Gemini"));
            assert!(message.contains("path=/v1beta/models?pageSize=1000"));
            assert!(message.contains(
                "parse_error=model entry missing usable `baseModelId` and `name` values"
            ));
            assert!(message.contains(r#""name": "models/""#));
        }
        _ => panic!("expected a response error"),
    }
}

/// The path and query are what the recorded cassette
/// `crates/rig-cassette/fixtures/cassettes/gemini/models/list_models_smoke.yaml` matches on —
/// `pageSize=1000` then `key` — so their exact shape is load-bearing, the
/// same reason `list_models_path` is pinned above.
#[test]
fn models_sends_the_credential_as_the_last_query_pair() {
    let encoded = Models::new(Gemini::new("test-key"))
        .encode((), Mode::Unary)
        .expect("the request encodes");
    let request = encoded.requests.first().expect("one request");

    assert_eq!(encoded.framing, Framing::Whole);
    assert_eq!(encoded.request_id_header, None);
    assert_eq!(request.method(), http::Method::GET);
    assert_eq!(request.uri().path(), "/v1beta/models");
    assert_eq!(request.uri().query(), Some("pageSize=1000&key=test-key"));
    assert!(request.headers().get("x-goog-api-key").is_none());
}

/// The Interactions API reaches the same endpoint with the credential in a
/// header instead, and must not also leak it into the query.
#[test]
fn interactions_models_sends_the_credential_as_a_header_only() {
    let encoded = InteractionsModels::new(Gemini::new("test-key"))
        .encode((), Mode::Unary)
        .expect("the request encodes");
    let request = encoded.requests.first().expect("one request");

    assert_eq!(encoded.framing, Framing::Whole);
    assert_eq!(request.uri().path(), "/v1beta/models");
    assert_eq!(request.uri().query(), Some("pageSize=1000"));
    assert_eq!(
        request
            .headers()
            .get("x-goog-api-key")
            .and_then(|value| value.to_str().ok()),
        Some("test-key"),
    );
}

/// Two pages fold in arrival order and the cursor page one named is what
/// the next request asks for. The entries are verbatim from
/// `crates/rig-cassette/fixtures/cassettes/gemini/models/list_models_smoke.yaml`; the recorded
/// catalog fits in one page, so the `nextPageToken` is what this adds.
#[test]
fn a_paged_listing_folds_in_order_and_follows_the_cursor() {
    const PAGE_ONE: &str = r#"{"models":[{"description":"Stable version of Gemini 2.5 Flash, our mid-size multimodal model that supports up to 1 million tokens, released in June of 2025.","displayName":"Gemini 2.5 Flash","inputTokenLimit":1048576,"maxTemperature":2,"name":"models/gemini-2.5-flash","outputTokenLimit":65536,"supportedGenerationMethods":["generateContent","countTokens","createCachedContent","batchGenerateContent"],"temperature":1,"thinking":true,"topK":64,"topP":0.95,"version":"001"},{"description":"Stable release (June 17th, 2025) of Gemini 2.5 Pro","displayName":"Gemini 2.5 Pro","inputTokenLimit":1048576,"maxTemperature":2,"name":"models/gemini-2.5-pro","outputTokenLimit":65536,"supportedGenerationMethods":["generateContent","countTokens","createCachedContent","batchGenerateContent"],"temperature":1,"thinking":true,"topK":64,"topP":0.95,"version":"2.5"}],"nextPageToken":"page-two"}"#;
    const PAGE_TWO: &str = r#"{"models":[{"displayName":"Gemini 2.5 Flash-Lite","inputTokenLimit":1048576,"name":"models/gemini-2.5-flash-lite","outputTokenLimit":65536}]}"#;

    let wire = Models::new(Gemini::new("test-key"));
    let ids = |driver: &mut WireDriver<ModelListing, ModelsDecoder>, page: &str| {
        driver.push(WireFrame::Text(page.to_owned()));
        driver
            .drain()
            .flat_map(|item| {
                item.expect("the recorded page decodes")
                    .iter()
                    .map(|model| model.id.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };

    let mut first = WireDriver::<ModelListing, _>::new(wire.decoder(crate::wire::Mode::Unary));
    let mut listed = ids(&mut first, PAGE_ONE);
    let continuation = first.continuation().expect("page one named a cursor");
    assert_eq!(continuation.uri().path(), "/v1beta/models");
    assert_eq!(
        continuation.uri().query(),
        Some("pageSize=1000&pageToken=page-two&key=test-key"),
    );

    let mut second = WireDriver::<ModelListing, _>::new(wire.decoder(crate::wire::Mode::Unary));
    listed.extend(ids(&mut second, PAGE_TWO));

    assert_eq!(
        listed,
        vec![
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite"
        ],
    );
    assert!(
        second.continuation().is_none(),
        "a page naming no cursor ends the listing",
    );
}
