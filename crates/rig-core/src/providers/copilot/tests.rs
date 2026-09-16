use super::{ChatApiErrorResponse, CopilotIntent, base_url_from_token, default_headers};
use crate::providers::openai;

/// Copilot's chat route relays OpenAI's wire, but not every field of it:
/// these are the reply shapes Copilot actually sends, and the shared type
/// has to read all of them.
#[test]
fn deserialize_standard_openai_response() {
    let json = r#"{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

    let response: openai::completion::CompletionResponse =
        serde_json::from_str(json).expect("standard OpenAI response should deserialize");
    assert_eq!(response.id, "chatcmpl-abc123");
    assert_eq!(response.object, "chat.completion");
    assert_eq!(response.created, 1700000000);
    assert_eq!(response.model, "gpt-4o");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].finish_reason, "stop");
}

/// Copilot omits `object` and `created` that OpenAI always sends.
#[test]
fn deserialize_copilot_response_without_object_and_created() {
    let json = r#"{
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 4,
                "total_tokens": 7
            }
        }"#;

    let response: openai::completion::CompletionResponse =
        serde_json::from_str(json).expect("Copilot response should deserialize");

    assert_eq!(response.id, "chatcmpl-123");
    assert_eq!(response.object, "");
    assert_eq!(response.created, 0);
    assert_eq!(response.model, "gpt-4o");
    assert_eq!(response.choices.len(), 1);
}

/// A vendor Copilot fronts can omit `finish_reason` and the choice `index`
/// as well.
#[test]
fn deserialize_copilot_response_without_finish_reason() {
    let json = r#"{
            "id": "chatcmpl-claude-001",
            "model": "claude-3.5-sonnet",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here is my analysis."
                }
            }],
            "usage": {
                "prompt_tokens": 50,
                "total_tokens": 80
            }
        }"#;

    let response: openai::completion::CompletionResponse =
        serde_json::from_str(json).expect("Claude-via-Copilot response should deserialize");

    assert_eq!(response.model, "claude-3.5-sonnet");
    assert_eq!(response.choices[0].finish_reason, "");
    assert_eq!(response.choices[0].index, 0);
}

#[test]
fn error_response_with_message_field() {
    let json = r#"{"message": "rate limit exceeded"}"#;
    let err: ChatApiErrorResponse = serde_json::from_str(json).expect("message-shaped error");

    assert_eq!(err.error_message(), "rate limit exceeded");
}

#[test]
fn error_response_with_error_field() {
    let json = r#"{"error": "model not found"}"#;
    let err: ChatApiErrorResponse = serde_json::from_str(json).expect("error-shaped error");

    assert_eq!(err.error_message(), "model not found");
}

/// The envelope declares the conversation intent, and the default is the
/// chat panel. Both routes stamp this same header set.
#[test]
fn copilot_intent_headers_use_panel_by_default_and_edits_when_requested() {
    let panel_headers = default_headers("token", "user", false, CopilotIntent::default());
    assert_eq!(
        panel_headers
            .iter()
            .find(|(name, _)| *name == "openai-intent")
            .map(|(_, value)| value.as_str()),
        Some("conversation-panel")
    );

    let edits_headers = default_headers("token", "user", false, CopilotIntent::Edits);
    assert_eq!(
        edits_headers
            .iter()
            .find(|(name, _)| *name == "openai-intent")
            .map(|(_, value)| value.as_str()),
        Some("conversation-edits")
    );
}

/// A vision request is gated behind a header the other requests must not
/// carry.
#[test]
fn only_a_vision_request_carries_the_vision_header() {
    let vision = |has_vision| {
        default_headers("token", "user", has_vision, CopilotIntent::default())
            .iter()
            .any(|(name, value)| *name == "copilot-vision-request" && value == "true")
    };
    assert!(vision(true));
    assert!(!vision(false));
}

#[test]
fn base_url_from_token_derives_api_endpoint() {
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2").as_deref(),
        Some("https://api.individual.githubcopilot.com")
    );
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=https://proxy.individual.githubcopilot.com;exp=2")
            .as_deref(),
        Some("https://api.individual.githubcopilot.com")
    );
    assert_eq!(base_url_from_token("tid=1;exp=2"), None);
}

#[test]
fn base_url_from_token_rejects_unsafe_or_non_copilot_endpoints() {
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=http://proxy.individual.githubcopilot.com;exp=2"),
        None
    );
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=https://evil.example.com;exp=2"),
        None
    );
    assert_eq!(base_url_from_token("tid=1;proxy-ep=://bad;exp=2"), None);
    assert_eq!(base_url_from_token("tid=1;proxy-ep=;exp=2"), None);
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=https://proxy.individual.githubcopilot.com/base;exp=2"),
        None
    );
}
