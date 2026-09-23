use super::*;

#[test]
fn test_model_from_id() {
    let model = Model::from_id("gpt-4");
    assert_eq!(model.id, "gpt-4");
    assert_eq!(model.name, None);
    assert_eq!(model.description, None);
    assert_eq!(model.r#type, None);
    assert_eq!(model.created_at, None);
    assert_eq!(model.owned_by, None);
    assert_eq!(model.context_length, None);
}

#[test]
fn test_model_new() {
    let model = Model::new("gpt-4", "GPT-4");
    assert_eq!(model.id, "gpt-4");
    assert_eq!(model.name, Some("GPT-4".to_string()));
}

#[test]
fn test_model_display_name() {
    let model_with_name = Model::new("gpt-4", "GPT-4");
    assert_eq!(model_with_name.display_name(), "GPT-4");

    let model_without_name = Model::from_id("gpt-4");
    assert_eq!(model_without_name.display_name(), "gpt-4");
}

#[test]
fn test_model_display() {
    let model = Model::new("gpt-4", "GPT-4");
    assert_eq!(format!("{model}"), "GPT-4");
}

#[test]
fn test_model_list_new() {
    let list = ModelList::new(vec![Model::from_id("gpt-4")]);
    assert_eq!(list.len(), 1);
}

#[test]
fn test_model_list_empty() {
    let list = ModelList::new(vec![]);
    assert!(list.is_empty());
    assert_eq!(list.len(), 0);
}

#[test]
fn test_model_list_iter() {
    let list = ModelList::new(vec![
        Model::from_id("gpt-4"),
        Model::from_id("gpt-3.5-turbo"),
    ]);
    let models: Vec<_> = list.iter().collect();
    assert_eq!(models.len(), 2);
}

#[test]
fn test_model_list_into_iter() {
    let list = ModelList::new(vec![
        Model::from_id("gpt-4"),
        Model::from_id("gpt-3.5-turbo"),
    ]);
    let models: Vec<_> = list.into_iter().collect();
    assert_eq!(models.len(), 2);
}

#[test]
fn a_rejected_listing_names_its_provider_and_path() {
    let error = with_route(
        ProviderError::from_http_response(http::StatusCode::NOT_FOUND, "Not found")
            .with_provider_request_id(Some("req_1".into())),
        "openai",
        "/v1/models",
    );

    let ProviderError::ProviderResponse(response) = &error else {
        panic!("a rejected listing keeps the provider's reply: {error:?}");
    };
    assert_eq!(response.status, Some(http::StatusCode::NOT_FOUND));
    assert_eq!(response.body, "Not found");
    assert_eq!(
        error.to_string(),
        "ProviderResponseError: status 404 Not Found: Not found (request id: req_1) \
         [provider=openai path=/v1/models]"
    );
}

#[test]
fn a_listing_that_did_not_parse_names_its_provider_and_path() {
    let json = serde_json::from_str::<serde_json::Value>("{").expect_err("malformed");
    for error in [
        ProviderError::Json(json),
        ProviderError::Response("Invalid JSON".to_owned()),
    ] {
        let ProviderError::Response(message) = with_route(error, "openai", "/v1/models") else {
            panic!("a listing decode failure reports as a response error");
        };
        assert!(message.starts_with("provider=openai\npath=/v1/models\nparse_error\n"));
    }
}

#[test]
fn a_listing_transport_failure_is_unchanged_by_its_route() {
    let error = with_route(
        ProviderError::Http(crate::http_client::Error::StreamEnded),
        "openai",
        "/v1/models",
    );
    assert!(matches!(
        error,
        ProviderError::Http(crate::http_client::Error::StreamEnded)
    ));
    assert!(error.is_retryable());
}

#[test]
fn a_listing_route_is_a_diagnostic_and_is_not_serialized() {
    let error = with_route(
        ProviderError::from_http_response(http::StatusCode::UNAUTHORIZED, "bad key"),
        "openai",
        "/v1/models",
    );
    let report = error.report();
    let restored: crate::error::ErrorReport =
        serde_json::from_str(&serde_json::to_string(&report).expect("serialize"))
            .expect("deserialize");
    let response = restored.provider_response.expect("reply is preserved");
    assert_eq!(response.route, None);
    assert_eq!(response.status, Some(http::StatusCode::UNAUTHORIZED));
    assert!(report.message.contains("[provider=openai path=/v1/models]"));
}

#[test]
fn test_model_serde() {
    let model = Model {
        id: "gpt-4".to_string(),
        name: Some("GPT-4".to_string()),
        description: None,
        r#type: Some("chat".to_string()),
        created_at: Some(1677610600),
        owned_by: Some("openai".to_string()),
        context_length: Some(8192),
        max_output_tokens: Some(4096),
    };

    let json = serde_json::to_string(&model).unwrap();
    assert!(json.contains("gpt-4"));
    assert!(json.contains("GPT-4"));

    let deserialized: Model = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.id, "gpt-4");
    assert_eq!(deserialized.name, Some("GPT-4".to_string()));
}

#[test]
fn test_model_list_serde() {
    let list = ModelList {
        data: vec![Model::from_id("gpt-4")],
    };

    let json = serde_json::to_string(&list).unwrap();
    assert!(json.contains("gpt-4"));

    let deserialized: ModelList = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.len(), 1);
}

#[test]
fn test_format_response_body_preview_without_truncation() {
    let preview = format_response_body_preview(br#"{"ok":true}"#);
    assert_eq!(preview, r#"{"ok":true}"#);
}

#[test]
fn test_format_response_body_preview_with_truncation() {
    let body = vec![b'a'; RESPONSE_BODY_PREVIEW_LIMIT + 3];
    let preview = format_response_body_preview(&body);

    assert!(preview.starts_with(&"a".repeat(RESPONSE_BODY_PREVIEW_LIMIT)));
    assert!(preview.ends_with("\n...<truncated 3 bytes>"));
}

#[test]
fn test_parse_error_with_context_includes_parse_error_and_preview() {
    let body = br#"{"models":[{"displayName":"broken"}]}"#;
    let json_error = serde_json::from_slice::<serde_json::Value>(b"{")
        .expect_err("expected malformed JSON to fail");
    let error = parse_error(
        "Gemini",
        "/v1beta/models?pageSize=1000",
        format_args!("parse_error={json_error}"),
        body,
    );

    match error {
        ProviderError::Response(message) => {
            assert!(message.contains("provider=Gemini"));
            assert!(message.contains("path=/v1beta/models?pageSize=1000"));
            assert!(message.contains("parse_error=EOF while parsing an object"));
            assert!(message.contains(r#"{"models":[{"displayName":"broken"}]}"#));
        }
        _ => panic!("Expected a response error"),
    }
}
