use super::*;

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

/// Loop-level: a server that keeps echoing the same cursor cannot advance
/// the listing, so the loop must stop rather than fetch the same page
/// forever. The parser-level guard above only covers the *empty* cursor;
/// this covers the other way a cursor fails to move.
#[tokio::test]
async fn list_all_stops_on_a_cursor_that_does_not_advance() {
    use crate::client::ModelLister as _;
    use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

    let page = |id: &str, token: &str| {
        MockHttpResponse::success(
            serde_json::json!({
                "models": [{
                    "name": format!("models/{id}"),
                    "displayName": id,
                    "inputTokenLimit": 1024
                }],
                "nextPageToken": token
            })
            .to_string(),
        )
    };
    let http_client = SequencedHttpClient::new(vec![
        page("a", "stuck"),
        page("b", "stuck"),
        page("c", "stuck"),
    ]);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("client should build");

    let models = GeminiModelLister::new(client)
        .list_all()
        .await
        .expect("listing should terminate");

    assert_eq!(
        models.data.len(),
        2,
        "the repeat is only detectable on the second page, so both are kept",
    );
    assert_eq!(http_client.remaining_responses(), 1);
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
        ModelListingError::ParseError { message } => {
            assert!(message.contains("provider=Gemini"));
            assert!(message.contains("path=/v1beta/models?pageSize=1000"));
            assert!(message.contains(
                "parse_error=model entry missing usable `baseModelId` and `name` values"
            ));
        }
        _ => panic!("expected parse error"),
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
        ModelListingError::ParseError { message } => {
            assert!(message.contains("provider=Gemini"));
            assert!(message.contains("path=/v1beta/models?pageSize=1000"));
            assert!(message.contains(
                "parse_error=model entry missing usable `baseModelId` and `name` values"
            ));
            assert!(message.contains(r#""name": "models/""#));
        }
        _ => panic!("expected parse error"),
    }
}
