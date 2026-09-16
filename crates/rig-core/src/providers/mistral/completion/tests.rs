use super::*;

#[test]
fn deserializes_response_with_array_and_null_content() {
    let data = r#"{
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "mistral-small-latest",
            "system_fingerprint": null,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " world"}]
                    },
                    "logprobs": null,
                    "finish_reason": "stop"
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": "{\"x\":1,\"y\":2}"}
                        }]
                    },
                    "logprobs": null,
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }"#;

    let response: CompletionResponse =
        serde_json::from_str(data).expect("response should deserialize");
    match &response.choices[0].message {
        Message::Assistant { content, .. } => assert_eq!(content, "Hello world"),
        _ => panic!("expected assistant message"),
    }
    match &response.choices[1].message {
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            assert_eq!(content, "");
            assert_eq!(tool_calls[0].function.name, "add");
        }
        _ => panic!("expected assistant message"),
    }
}

#[test]
fn usage_prefers_structured_cached_tokens_and_falls_back() {
    let structured: Usage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "num_cached_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 7}
    }))
    .expect("usage should deserialize");
    assert_eq!(structured.cached_tokens(), Some(7));

    let fallback: Usage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "num_cached_tokens": 2
    }))
    .expect("usage should deserialize");
    assert_eq!(fallback.cached_tokens(), Some(2));

    // The singular alias form used by some Mistral responses.
    let aliased: Usage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_token_details": {"cached_tokens": 4}
    }))
    .expect("usage should deserialize");
    assert_eq!(aliased.cached_tokens(), Some(4));
}

/// Mistral reports audio outside `prompt_tokens`, so counting only that
/// field leaves `input + output` short of `total` by the audio payload.
/// The numbers are a live Voxtral turn's, quoted verbatim.
#[test]
fn usage_counts_audio_tokens_as_input() {
    let usage: Usage = serde_json::from_value(serde_json::json!({
        "prompt_audio_seconds": 0,
        "prompt_tokens": 6,
        "completion_tokens": 2,
        "total_tokens": 383,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 375}
    }))
    .expect("usage should deserialize");

    assert_eq!(usage.audio_tokens(), 375);
    assert_eq!(usage.input_tokens(), 381);

    let normalized = crate::completion::Usage::from(&usage);
    assert_eq!(normalized.input_tokens, Some(381));
    assert_eq!(normalized.output_tokens, Some(2));
    assert_eq!(
        normalized.total_tokens,
        Some(381 + 2),
        "the parts must add up to the total Mistral reported"
    );
}

/// A text turn carries no audio detail, and must be unaffected.
#[test]
fn usage_without_audio_is_unchanged() {
    let usage: Usage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 19, "completion_tokens": 2, "total_tokens": 21,
        "prompt_tokens_details": {"cached_tokens": 0}
    }))
    .expect("usage should deserialize");

    assert_eq!(usage.audio_tokens(), 0);
    assert_eq!(
        crate::completion::Usage::from(&usage).input_tokens,
        Some(19)
    );
}

/// Mistral emits the tool call anyway when `max_tokens` runs out mid
/// arguments — a live turn capped at 32 tokens returned
/// `finish_reason: "length"` with `arguments` cut off partway through the
/// object. Parsing strictly took the whole response down with it.
#[test]
fn truncated_tool_arguments_do_not_destroy_the_response() {
    let data = r#"{
            "id": "cmpl-1", "object": "chat.completion", "created": 1,
            "model": "mistral-small-latest", "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Recording that now.",
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "record", "arguments": "{\"note\": \"How to bake sour"}
                    }]
                },
                "logprobs": null,
                "finish_reason": "length"
            }],
            "usage": {"prompt_tokens": 30, "completion_tokens": 32, "total_tokens": 62}
        }"#;

    let response: CompletionResponse =
        serde_json::from_str(data).expect("a truncated tool call must not fail the response");
    let choice = response.choices.first().expect("the turn survives");
    assert_eq!(choice.finish_reason, "length");
    assert_eq!(
        response.usage.expect("usage survives").total_tokens,
        62,
        "the text and metadata of a truncated turn are kept"
    );
    // The unusable call is dropped, as the streaming path drops it.
    match &choice.message {
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            assert_eq!(content, "Recording that now.");
            assert!(
                tool_calls.is_empty(),
                "a call with truncated arguments must not be handed to a tool"
            );
        }
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

/// A complete tool call is unaffected by the tolerant parse.
#[test]
fn complete_tool_arguments_still_parse() {
    let data = r#"{
            "id": "cmpl-1", "object": "chat.completion", "created": 1,
            "model": "mistral-small-latest", "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "add", "arguments": "{\"x\":1,\"y\":2}"}
                }]},
                "logprobs": null, "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }"#;

    let response: CompletionResponse =
        serde_json::from_str(data).expect("response should deserialize");
    match &response.choices.first().expect("one choice").message {
        Message::Assistant { tool_calls, .. } => {
            assert_eq!(tool_calls.len(), 1, "a complete call must still be carried");
            assert_eq!(tool_calls[0].function.name, "add");
            assert_eq!(
                tool_calls[0].function.arguments,
                serde_json::json!({"x": 1, "y": 2}),
                "the stringified JSON on the wire reads back as an object"
            );
        }
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

/// The choice-level tolerance is gated by the truncation reason. Invalid
/// JSON on a completed tool turn remains a response error.
#[test]
fn malformed_completed_tool_arguments_still_fail() {
    let data = r#"{
            "id": "cmpl-1", "object": "chat.completion", "created": 1,
            "model": "mistral-small-latest", "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "add", "arguments": "{\"x\":"}
                }]},
                "logprobs": null, "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        }"#;

    assert!(
        serde_json::from_str::<CompletionResponse>(data).is_err(),
        "ordinary malformed tool output must remain loud"
    );
}

#[test]
fn usage_retains_live_service_tier() {
    let usage: Usage = serde_json::from_value(serde_json::json!({
        "completion_tokens": 4,
        "prompt_tokens": 20,
        "total_tokens": 24,
        "prompt_tokens_details": { "cached_tokens": 0 },
        "service_tier": "standard"
    }))
    .expect("live Mistral usage should deserialize");

    assert_eq!(usage.service_tier.as_deref(), Some("standard"));
    assert_eq!(
        serde_json::to_value(usage).expect("Mistral usage should serialize")["service_tier"],
        "standard"
    );
}
