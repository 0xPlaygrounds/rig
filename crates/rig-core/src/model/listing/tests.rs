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
fn test_model_listing_error_display() {
    let error = ModelListingError::api_error(404, "Not found");
    assert_eq!(error.to_string(), "API error (status 404): Not found");

    let error = ModelListingError::request_error("Connection failed");
    assert_eq!(error.to_string(), "Request error: Connection failed");

    let error = ModelListingError::parse_error("Invalid JSON");
    assert_eq!(error.to_string(), "Parse error: Invalid JSON");

    let error = ModelListingError::AuthError {
        message: "Invalid API key".to_string(),
    };
    assert_eq!(error.to_string(), "Authentication error: Invalid API key");
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
fn test_model_listing_error_serde() {
    let error = ModelListingError::api_error(404, "Not found");

    let json = serde_json::to_string(&error).unwrap();
    assert!(json.contains("ApiError"));

    let deserialized: ModelListingError = serde_json::from_str(&json).unwrap();
    match deserialized {
        ModelListingError::ApiError {
            status_code,
            message,
        } => {
            assert_eq!(status_code, 404);
            assert_eq!(message, "Not found");
        }
        _ => panic!("Expected ApiError"),
    }
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
fn test_api_error_with_context_includes_provider_path_and_preview() {
    let error = ModelListingError::api_error_with_context(
        "Gemini",
        "/v1beta/models?pageSize=1000",
        500,
        br#"{"error":"boom"}"#,
    );

    match error {
        ModelListingError::ApiError {
            status_code,
            message,
        } => {
            assert_eq!(status_code, 500);
            assert!(message.contains("provider=Gemini"));
            assert!(message.contains("path=/v1beta/models?pageSize=1000"));
            assert!(message.contains("status=500"));
            assert!(message.contains(r#"{"error":"boom"}"#));
        }
        _ => panic!("Expected ApiError"),
    }
}

#[test]
fn test_parse_error_with_context_includes_parse_error_and_preview() {
    let body = br#"{"models":[{"displayName":"broken"}]}"#;
    let parse_error = serde_json::from_slice::<serde_json::Value>(b"{")
        .expect_err("expected malformed JSON to fail");
    let error = ModelListingError::parse_error_with_context(
        "Gemini",
        "/v1beta/models?pageSize=1000",
        &parse_error,
        body,
    );

    match error {
        ModelListingError::ParseError { message } => {
            assert!(message.contains("provider=Gemini"));
            assert!(message.contains("path=/v1beta/models?pageSize=1000"));
            assert!(message.contains("parse_error=EOF while parsing an object"));
            assert!(message.contains(r#"{"models":[{"displayName":"broken"}]}"#));
        }
        _ => panic!("Expected ParseError"),
    }
}
