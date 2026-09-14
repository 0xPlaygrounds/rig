use super::*;
use crate::completion::NormalizeCompletionResponse as _;
use crate::telemetry::ProviderResponseExt as _;

/// A real b10499 chat-completions body, trimmed to the fields under test.
const BODY: &str = r#"{
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": { "role": "assistant", "content": "ok" }
        }],
        "created": 0,
        "id": "chatcmpl-abc",
        "model": "/models/Qwen3-1.7B-Q4_K_M.gguf",
        "object": "chat.completion",
        "system_fingerprint": "b10499-6d05498",
        "usage": { "completion_tokens": 2, "prompt_tokens": 9, "total_tokens": 11,
                   "prompt_tokens_details": { "cached_tokens": 8 } },
        "timings": { "cache_n": 8, "prompt_n": 1, "prompt_ms": 6.175,
                     "predicted_n": 2, "predicted_ms": 16.9,
                     "predicted_per_second": 118.3 }
    }"#;

#[test]
fn timings_survive_deserialization_and_normalization() {
    let response: CompletionResponse =
        serde_json::from_str(BODY).expect("a llama.cpp body should deserialize");

    let timings = response
        .timings
        .clone()
        .expect("llama.cpp reports timings on every chat completion");
    assert_eq!(timings.cache_n, Some(8));
    assert_eq!(timings.predicted_n, Some(2));
    assert_eq!(response.predicted_tokens_per_second(), Some(118.3));

    // The OpenAI half is intact and normalizes exactly as before.
    assert_eq!(response.response_id(), Some("chatcmpl-abc"));
    assert_eq!(
        response.openai.system_fingerprint.as_deref(),
        Some("b10499-6d05498")
    );
    let normalized = response
        .normalize("llamacpp")
        .expect("a llama.cpp body should normalize");
    assert_eq!(normalized.provider, "llamacpp");
    // `cache_n` and the normalized cached-token count are independently
    // populated and must agree.
    assert_eq!(normalized.usage.cached_input_tokens, 8);
}

/// A response with no `timings` is not an error.
///
/// llama.cpp always sends them today, but a `.llamafile` built from an
/// older core, or a proxy that strips unknown fields, must still decode.
#[test]
fn a_response_without_timings_still_decodes() {
    let mut body: serde_json::Value = serde_json::from_str(BODY).expect("fixture should parse");
    body.as_object_mut()
        .expect("body is an object")
        .remove("timings")
        .expect("the fixture carries timings to remove");

    let response: CompletionResponse = serde_json::from_value(body)
        .unwrap_or_else(|error| panic!("should decode without timings: {error}"));
    assert!(response.timings.is_none());
    assert!(response.predicted_tokens_per_second().is_none());
    assert_eq!(response.text_response().as_deref(), Some("ok"));
}

/// Round-tripping must not invent or lose the extra field.
#[test]
fn serialization_round_trips_the_extra_field() {
    let response: CompletionResponse = serde_json::from_str(BODY).expect("should deserialize");
    let value = serde_json::to_value(&response).expect("should serialize");
    assert_eq!(value["timings"]["cache_n"], serde_json::json!(8));
    assert_eq!(
        value["system_fingerprint"],
        serde_json::json!("b10499-6d05498"),
        "the flattened OpenAI half must stay at the top level, not nest under `openai`"
    );
    let again: CompletionResponse = serde_json::from_value(value).expect("should round-trip");
    assert_eq!(again.timings, response.timings);
}
