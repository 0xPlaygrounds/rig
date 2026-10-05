use super::*;
use crate::providers::gemini::GeminiConfig;

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
    let encoded = Models::new(GeminiConfig::new("test-key"))
        .encode(None, Mode::Unary)
        .expect("the request encodes");
    let request = &encoded.request;

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
    let encoded = InteractionsModels::new(GeminiConfig::new("test-key"))
        .encode(None, Mode::Unary)
        .expect("the request encodes");
    let request = &encoded.request;

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

    let wire = Models::new(GeminiConfig::new("test-key"));
    let page = |page: &str| {
        let page = crate::test_utils::decode_reply(
            &wire,
            &None,
            crate::wire::Mode::Unary,
            [WireFrame::Text(page.to_owned())],
            serde_json::Value::Null,
        )
        .expect("the recorded page decodes");
        let ids: Vec<_> = page.models.iter().map(|model| model.id.clone()).collect();
        (ids, page.next)
    };

    let (mut listed, cursor) = page(PAGE_ONE);
    let cursor = cursor.expect("page one named a cursor");
    let continuation = wire
        .encode(Some(cursor), crate::wire::Mode::Unary)
        .expect("the next page encodes");
    let continuation = &continuation.request;
    assert_eq!(continuation.uri().path(), "/v1beta/models");
    assert_eq!(
        continuation.uri().query(),
        Some("pageSize=1000&pageToken=page-two&key=test-key"),
    );

    let (second, next) = page(PAGE_TWO);
    listed.extend(second);

    assert_eq!(
        listed,
        vec![
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite"
        ],
    );
    assert!(next.is_none(), "a page naming no cursor ends the listing");
}
