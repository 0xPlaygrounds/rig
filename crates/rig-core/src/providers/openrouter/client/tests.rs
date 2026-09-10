use super::Usage;

#[test]
fn test_client_initialization() {
    let _client = crate::providers::openrouter::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::openrouter::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[test]
fn test_with_app_identity_sets_headers() {
    let client = crate::providers::openrouter::Client::builder()
        .with_app_identity("My App", "https://myapp.example.com")
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");

    let headers = client.headers();
    assert_eq!(
        headers
            .get("x-openrouter-title")
            .and_then(|v| v.to_str().ok()),
        Some("My App"),
    );
    assert_eq!(
        headers.get("http-referer").and_then(|v| v.to_str().ok()),
        Some("https://myapp.example.com"),
    );
}

#[test]
fn test_without_app_identity_no_extra_headers() {
    let client = crate::providers::openrouter::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");

    let headers = client.headers();
    assert!(headers.get("x-openrouter-title").is_none());
    assert!(headers.get("http-referer").is_none());
}

#[test]
fn test_with_app_categories_sets_header() {
    let client = crate::providers::openrouter::Client::builder()
        .with_app_categories(&["cli-agent", "ide-extension"])
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");

    assert_eq!(
        client
            .headers()
            .get("x-openrouter-categories")
            .and_then(|v| v.to_str().ok()),
        Some("cli-agent,ide-extension"),
    );
}

#[test]
fn test_with_app_categories_sends_at_most_two_categories() {
    let client = crate::providers::openrouter::Client::builder()
        .with_app_categories(&["cli-agent", "ide-extension", "chat"])
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");

    assert_eq!(
        client
            .headers()
            .get("x-openrouter-categories")
            .and_then(|v| v.to_str().ok()),
        Some("cli-agent,ide-extension"),
    );
}

#[test]
fn test_with_app_categories_empty_list_no_header() {
    let empty: [&str; 0] = [];
    let client = crate::providers::openrouter::Client::builder()
        .with_app_categories(&empty)
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");

    assert!(client.headers().get("x-openrouter-categories").is_none());
}

#[test]
fn test_without_app_categories_no_header() {
    let client = crate::providers::openrouter::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");

    assert!(client.headers().get("x-openrouter-categories").is_none());
}

/// A real usage object, copied verbatim out of
/// `tests/cassettes/openrouter/reasoning_usage_matrix/blocking_anthropic_routed_reports_reasoning_tokens.yaml`:
/// the breakdown must survive deserialization and reach the normalized
/// `reasoning_tokens` slot, unmodeled siblings and all.
#[test]
fn completion_tokens_details_reaches_normalized_usage() {
    let usage: Usage = serde_json::from_str(
        r#"{"completion_tokens":540,
                "completion_tokens_details":{"audio_tokens":0,"image_tokens":0,"reasoning_tokens":531},
                "cost":0.002794,
                "cost_details":{"upstream_inference_completions_cost":0.0027,"upstream_inference_cost":0.002794,"upstream_inference_prompt_cost":0.000094},
                "is_byok":false,
                "prompt_tokens":94,
                "prompt_tokens_details":{"audio_tokens":0,"cache_write_tokens":0,"cached_tokens":0,"video_tokens":0},
                "total_tokens":634}"#,
    )
    .expect("recorded usage should deserialize");

    let normalized = crate::completion::Usage::from(&usage);
    assert_eq!(normalized.reasoning_tokens, 531);
    assert_eq!(normalized.output_tokens, 540);
    assert_eq!(normalized.input_tokens, 94);
    assert_eq!(normalized.total_tokens, 634);
    // The reasoning share is counted *inside* the completion tokens.
    assert!(normalized.reasoning_tokens <= normalized.output_tokens);
}

/// A non-reasoning route reports the object with a zero share; a gateway
/// that omits it entirely, or sends it as `null`, must read the same.
#[test]
fn completion_tokens_details_absent_null_or_zero_all_read_zero() {
    for body in [
        r#"{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}"#,
        r#"{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":null}"#,
        r#"{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{"reasoning_tokens":0}}"#,
        r#"{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"completion_tokens_details":{}}"#,
    ] {
        let usage: Usage = serde_json::from_str(body).expect("usage should deserialize");
        let normalized = crate::completion::Usage::from(&usage);
        assert_eq!(normalized.reasoning_tokens, 0, "body: {body}");
        assert_eq!(normalized.output_tokens, 3, "body: {body}");
    }
}

/// Unknown siblings inside the breakdown (OpenRouter sends `audio_tokens`
/// and `image_tokens`) must not fail the decode — the object is read for
/// the one entry rig has a slot for.
#[test]
fn completion_tokens_details_tolerates_unmodeled_siblings() {
    let usage: Usage = serde_json::from_str(
        r#"{"prompt_tokens":9,"completion_tokens":1291,"total_tokens":1300,
                "completion_tokens_details":{"audio_tokens":0,"image_tokens":1290,"reasoning_tokens":7}}"#,
    )
    .expect("usage should deserialize");

    assert_eq!(crate::completion::Usage::from(&usage).reasoning_tokens, 7);
}

/// The completion-token fallback (`total - prompt` for gateways that omit
/// `completion_tokens`) must stay independent of the new field.
#[test]
fn completion_tokens_details_does_not_disturb_the_output_token_fallback() {
    let usage: Usage = serde_json::from_str(
        r#"{"prompt_tokens":10,"total_tokens":30,
                "completion_tokens_details":{"reasoning_tokens":12}}"#,
    )
    .expect("usage should deserialize");

    let normalized = crate::completion::Usage::from(&usage);
    assert_eq!(normalized.output_tokens, 20);
    assert_eq!(normalized.reasoning_tokens, 12);
}

/// Round-tripping the type must not start sending a breakdown rig never
/// received: the field is skipped when absent.
#[test]
fn completion_tokens_details_is_omitted_when_absent() {
    let usage: Usage =
        serde_json::from_str(r#"{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}"#)
            .expect("usage should deserialize");
    let encoded = serde_json::to_string(&usage).expect("usage should serialize");

    assert!(!encoded.contains("completion_tokens_details"), "{encoded}");
}
