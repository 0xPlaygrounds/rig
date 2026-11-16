use super::*;
use crate::completion::CompletionRequest;
use crate::providers::openai::OpenAIConfig;
use crate::wire::{Framing, Mode, Wire};

#[test]
fn official_and_legacy_dialects_have_separate_routes() {
    for (dialect, url, token) in [
        (
            &DIALECT,
            "https://api.openai.com/v1/responses",
            "official-token",
        ),
        (
            &super::super::DIALECT,
            "https://chatgpt.com/backend-api/codex/responses",
            "legacy-token",
        ),
    ] {
        let wire = OpenAIConfig::with_key(dialect, token).responses("gpt-6.1-sol");
        let encoded = wire
            .encode(CompletionRequest::new("Hello"), Mode::Unary)
            .unwrap_or_else(|error| panic!("{url} request should encode: {error}"));
        assert_eq!(encoded.request.uri(), url);
        assert_eq!(
            encoded
                .request
                .headers()
                .get("authorization")
                .and_then(|header| header.to_str().ok()),
            Some(format!("Bearer {token}").as_str())
        );
        assert_eq!(encoded.framing, Framing::Sse);
    }
    let config = OpenAIConfig::with_key(&DIALECT, "official-token");
    let saved = serde_json::to_value(&config)
        .unwrap_or_else(|error| panic!("SIWC config should serialize: {error}"));
    let restored: OpenAIConfig = serde_json::from_value(saved)
        .unwrap_or_else(|error| panic!("SIWC config should deserialize: {error}"));
    assert_eq!(restored.dialect, DIALECT);
}

#[test]
fn official_requests_apply_constraints_after_raw_options() {
    let wire = OpenAIConfig::with_key(&DIALECT, "official-token").responses("gpt-6.1-sol");
    let request = CompletionRequest {
        additional_params: Some(json!({
            "store": true, "stream": false, "previous_response_id": "old",
            "temperature": 0.5, "max_output_tokens": 100, "metadata": {"key": "value"},
            "background": true, "prompt_cache_retention": "24h"
        })),
        ..CompletionRequest::new("Hello")
    };
    for mode in [Mode::Unary, Mode::Streaming] {
        let encoded = wire
            .encode(request.clone(), mode)
            .unwrap_or_else(|error| panic!("SIWC request should encode: {error}"));
        let body = crate::test_utils::json_body(&encoded.request);
        assert_eq!(
            body,
            json!({
                "model": "gpt-6.1-sol", "store": false, "stream": true,
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
                "include": ["reasoning.encrypted_content"]
            })
        );
    }
}

#[test]
fn official_requests_lift_system_messages_and_namespace_tools() {
    let wire = OpenAIConfig::with_key(&DIALECT, "official-token").responses("gpt-6.1-sol");
    let request = CompletionRequest {
        chat_history: vec![
            crate::message::Message::system("Be brief").into(),
            crate::message::Message::user("Hello").into(),
        ],
        additional_params: Some(json!({
            "tools": [{
                "type": "function",
                "name": "read",
                "parameters": {"type": "object", "properties": {}}
            }]
        })),
        ..CompletionRequest::new("Hello")
    };
    let encoded = wire
        .encode(request, Mode::Streaming)
        .unwrap_or_else(|error| panic!("SIWC tool request should encode: {error}"));
    let body = crate::test_utils::json_body(&encoded.request);
    assert_eq!(body.get("instructions"), Some(&json!("Be brief")));
    assert_eq!(body.pointer("/tools/0/type"), Some(&json!("namespace")));
    assert_eq!(body.pointer("/tools/0/name"), Some(&json!("rig")));
    assert_eq!(body.pointer("/tools/0/tools/0/name"), Some(&json!("read")));
}

#[test]
fn official_requests_preserve_native_structured_output() {
    let wire = OpenAIConfig::with_key(&DIALECT, "official-token").responses("gpt-6.1-sol");
    let schema = schemars::json_schema!({
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
        "additionalProperties": false
    });
    let request = CompletionRequest {
        output_schema: Some(schema.clone()),
        ..CompletionRequest::new("Say hello")
    };
    let encoded = wire
        .encode(request, Mode::Unary)
        .unwrap_or_else(|error| panic!("SIWC structured output request should encode: {error}"));
    let body = crate::test_utils::json_body(&encoded.request);
    assert_eq!(
        body.pointer("/text/format/type"),
        Some(&json!("json_schema"))
    );
    assert_eq!(
        body.pointer("/text/format/schema"),
        Some(&schema.to_value())
    );
    assert_eq!(encoded.framing, Framing::Sse);
}

#[test]
fn official_catalogue_preserves_visible_model_order_and_names() {
    let models = catalogue(json!({"models": [
        {"slug": "second", "display_name": "Second", "visibility": "list"},
        {"slug": "hidden", "visibility": "hide"},
        {"slug": "first", "display_name": "First", "visibility": "list"}
    ]}));
    assert_eq!(
        models.data,
        vec![
            crate::model::ModelInfo::new("second", "Second"),
            crate::model::ModelInfo::new("first", "First")
        ]
    );
}
