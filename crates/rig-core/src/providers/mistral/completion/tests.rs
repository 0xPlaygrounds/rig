use super::*;
use crate::completion::NormalizeCompletionResponse as _;
use crate::providers::openai::completion::OpenAICompatibleProvider;

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
    assert_eq!(structured.cached_tokens(), 7);

    let fallback: Usage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "num_cached_tokens": 2
    }))
    .expect("usage should deserialize");
    assert_eq!(fallback.cached_tokens(), 2);

    // The singular alias form used by some Mistral responses.
    let aliased: Usage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_token_details": {"cached_tokens": 4}
    }))
    .expect("usage should deserialize");
    assert_eq!(aliased.cached_tokens(), 4);
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
    assert_eq!(normalized.input_tokens, 381);
    assert_eq!(normalized.output_tokens, 2);
    assert_eq!(
        normalized.input_tokens + normalized.output_tokens,
        normalized.total_tokens,
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
    assert_eq!(crate::completion::Usage::from(&usage).input_tokens, 19);
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

    let normalized = response
        .normalize("mistral")
        .expect("the turn must survive with its text and metadata");
    assert_eq!(
        normalized.finish_reason(),
        Some(crate::completion::FinishReason::Length),
        "the finish reason is what reports the truncation"
    );
    assert_eq!(normalized.usage.total_tokens, 62);
    // The unusable call is dropped, as the streaming path drops it.
    assert!(
        normalized
            .choice
            .iter()
            .all(|content| !matches!(content, crate::completion::AssistantContent::ToolCall(_))),
        "a call with truncated arguments must not be handed to a tool"
    );
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

    let normalized = serde_json::from_str::<CompletionResponse>(data)
        .expect("response should deserialize")
        .normalize("mistral")
        .expect("a complete call should normalize");
    assert!(
        normalized
            .choice
            .iter()
            .any(|content| matches!(content, crate::completion::AssistantContent::ToolCall(_))),
        "a complete call must still reach the caller"
    );
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

/// Mistral rejects a forced tool choice beside a response format with
/// "`json_schema` response type with tools is only compatible with
/// `tool_choice: auto`". Rig reaches that combination by itself on the
/// turn after a tool result, so finalization relaxes the choice.
#[test]
fn finalize_relaxes_a_forced_tool_choice_beside_a_response_format() {
    let mut body = serde_json::json!({
        "model": MISTRAL_SMALL,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "required",
        "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "Plan"}}
    });
    Mistral
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");
    assert_eq!(body["tool_choice"], "auto");
    assert!(
        body.get("response_format").is_some(),
        "the caller's schema must survive; relaxing the choice is what gives way"
    );

    // A specific function is forcing too, and equally rejected.
    let mut body = serde_json::json!({
        "model": MISTRAL_SMALL,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "function", "function": {"name": "add"}},
        "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}],
        "response_format": {"type": "json_object"}
    });
    Mistral
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");
    assert_eq!(body["tool_choice"], "auto");
}

/// The relaxation is narrow: without a response format, or without tools,
/// or when the choice is already compatible, nothing moves.
/// A `text` response format is not the constrained kind either — Mistral
/// takes it beside a forced choice, verified live.
#[test]
fn finalize_leaves_a_forced_tool_choice_alone_without_a_response_format() {
    let mut body = serde_json::json!({
        "model": MISTRAL_SMALL,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "required",
        "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}]
    });
    Mistral
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");
    assert_eq!(body["tool_choice"], "any", "still just the dialect rename");

    let mut body = serde_json::json!({
        "model": MISTRAL_SMALL,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "none",
        "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}],
        "response_format": {"type": "json_object"}
    });
    Mistral
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");
    assert_eq!(body["tool_choice"], "none", "`none` is already compatible");

    let mut body = serde_json::json!({
        "model": MISTRAL_SMALL,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "required",
        "tools": [{"type": "function", "function": {"name": "add", "parameters": {}}}],
        "response_format": {"type": "text"}
    });
    Mistral
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");
    assert_eq!(
        body["tool_choice"], "any",
        "a `text` response format is unconstrained; only the structured kinds conflict"
    );
}

#[test]
fn finalize_rewrites_required_tool_choice_to_any() {
    let mut body = serde_json::json!({
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "required"
    });

    Mistral
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");

    assert_eq!(body["tool_choice"], "any");
}

#[test]
fn finalize_preserves_specific_function_tool_choice() {
    let mut body = serde_json::json!({
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "function", "function": {"name": "beta"}}
    });

    Mistral
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");

    assert_eq!(
        body["tool_choice"],
        serde_json::json!({"type": "function", "function": {"name": "beta"}})
    );
}

#[test]
fn finalize_flattens_assistant_history_and_adds_prefix() {
    let mut body = serde_json::json!({
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello."}],
                "reasoning_content": "hidden thoughts"
            },
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{}"}
                }]
            }
        ]
    });

    Mistral
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");

    assert_eq!(body["messages"][0]["content"], "Be brief.");
    assert_eq!(body["messages"][2]["content"], "Hello.");
    assert_eq!(body["messages"][2]["prefix"], false);
    assert!(
        body["messages"][2].get("reasoning_content").is_none(),
        "Mistral rejects unknown assistant fields; reasoning must be stripped"
    );
    assert_eq!(body["messages"][3]["content"], "");
    assert_eq!(body["messages"][3]["prefix"], false);
}

/// Finalize a one-user-message body and return the message's `content`.
///
/// These cells are unit tests rather than cassettes because the behaviour
/// under test is that **no request is built** — there is no traffic to
/// record for content that is rejected before the wire. The cells covering
/// content that *is* sent are recorded, in
/// `tests/providers/mistral/multimodal_content.rs`.
fn finalized_content(parts: serde_json::Value) -> Result<serde_json::Value, CompletionError> {
    let mut body = serde_json::json!({
        "model": MISTRAL_SMALL,
        "messages": [{"role": "user", "content": parts}],
    });
    Mistral.finalize_request_body(&mut body)?;
    Ok(body["messages"][0]["content"].clone())
}

/// Video has no Mistral chunk — the API's own content discriminator does
/// not list one — so it must fail rather than be flattened away.
#[test]
fn finalize_rejects_video_content() {
    let error = finalized_content(serde_json::json!([
        {"type": "text", "text": "Describe this."},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
    ]))
    .expect_err("video content must not be dropped from the request");

    assert!(matches!(error, CompletionError::RequestError(_)));
    let rendered = error.to_string();
    assert!(rendered.contains("video_url"), "{rendered}");
}

/// An unrecognized part type fails closed, so a content kind added to the
/// shared conversion later cannot start disappearing silently.
#[test]
fn finalize_rejects_unrecognized_and_untyped_parts() {
    let error = finalized_content(serde_json::json!([
        {"type": "text", "text": "hi"},
        {"type": "some_future_part", "some_future_part": {}}
    ]))
    .expect_err("an unmodelled part must not be dropped");
    assert!(matches!(error, CompletionError::RequestError(_)));

    let error = finalized_content(serde_json::json!([
        {"type": "text", "text": "hi"},
        {"payload": "no type tag at all"}
    ]))
    .expect_err("an untyped part must not be dropped");
    assert!(error.to_string().contains("untyped"), "{error}");
}

/// OpenAI's file part must carry something convertible; a part with
/// neither inline bytes nor an id names no document at all.
#[test]
fn finalize_rejects_a_file_part_with_no_payload() {
    let error = finalized_content(serde_json::json!([
        {"type": "text", "text": "hi"},
        {"type": "file", "file": {"filename": "empty.pdf"}}
    ]))
    .expect_err("a file part naming no document must not be dropped");
    assert!(matches!(error, CompletionError::RequestError(_)));
}

/// An audio part whose payload is neither a base64 string nor an object
/// carrying one cannot be rendered as Mistral's audio chunk.
#[test]
fn finalize_rejects_an_audio_part_with_no_payload() {
    let error = finalized_content(serde_json::json!([
        {"type": "text", "text": "hi"},
        {"type": "input_audio", "input_audio": {"format": "mp3"}}
    ]))
    .expect_err("an audio part carrying no data must not be dropped");
    assert!(matches!(error, CompletionError::RequestError(_)));
}

/// The document conversions, pinned as exact wire shapes.
///
/// The `document_url` half is also proven live, by the recorded cells that
/// read `BANANA-7391` back out of an attached PDF. The `file`/`file_id`
/// half is pinned by shape only: exercising it end to end would mean
/// uploading a file and committing its account-scoped, expiring id to a
/// fixture, so this cell is the whole of its coverage.
#[test]
fn finalize_maps_openai_file_parts_onto_mistral_chunks() {
    let content = finalized_content(serde_json::json!([
        {"type": "text", "text": "Read these."},
        {"type": "file", "file": {
            "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
            "filename": "document.pdf"
        }},
        {"type": "file", "file": {"file_id": "00000000-0000-0000-0000-000000000000"}}
    ]))
    .expect("file parts should convert");

    assert_eq!(
        content,
        serde_json::json!([
            {"type": "text", "text": "Read these."},
            {
                "type": "document_url",
                "document_url": "data:application/pdf;base64,JVBERi0xLjQK",
                "document_name": "document.pdf"
            },
            // Mistral's file chunk names the id at the top level; OpenAI's
            // nesting under `file` is rejected as an extra field.
            {"type": "file", "file_id": "00000000-0000-0000-0000-000000000000"}
        ])
    );
}

/// Audio collapses to Mistral's documented bare-string payload, and the
/// image chunk forwards unchanged because Mistral accepts rig's object.
#[test]
fn finalize_maps_audio_and_image_parts_onto_mistral_chunks() {
    let content = finalized_content(serde_json::json!([
        {"type": "input_audio", "input_audio": {"data": "SUQzBAA=", "format": "mp3"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "auto"}}
    ]))
    .expect("audio and image parts should convert");

    assert_eq!(
        content,
        serde_json::json!([
            {"type": "input_audio", "input_audio": "SUQzBAA="},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "auto"}}
        ])
    );
}

/// A refusal travelling beside a chunk becomes a text chunk: Mistral's
/// schema has no `refusal` field, and every chunk forbids unknown keys.
#[test]
fn finalize_retags_a_refusal_beside_a_chunk_as_text() {
    let content = finalized_content(serde_json::json!([
        {"type": "refusal", "refusal": "I cannot help with that."},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
    ]))
    .expect("a refusal beside a chunk should convert");

    assert_eq!(
        content[0],
        serde_json::json!({"type": "text", "text": "I cannot help with that."})
    );
}

/// Text-only content keeps the plain-string form every existing Mistral
/// fixture pins — including a refusal-only message, which the shared
/// flattening has always treated as text.
#[test]
fn finalize_still_flattens_text_only_content() {
    assert_eq!(
        finalized_content(serde_json::json!([
            {"type": "text", "text": "First."},
            {"type": "text", "text": "Second."}
        ]))
        .expect("text-only content should flatten"),
        serde_json::json!("First.Second.")
    );

    assert_eq!(
        finalized_content(serde_json::json!([
            {"type": "text", "text": "Partly: "},
            {"type": "refusal", "refusal": "I cannot help with that."}
        ]))
        .expect("refusal content should flatten"),
        serde_json::json!("Partly: I cannot help with that.")
    );

    // Content that is already a plain string is left exactly as-is.
    assert_eq!(
        finalized_content(serde_json::json!("already a string"))
            .expect("string content should pass through"),
        serde_json::json!("already a string")
    );

    // An empty array still collapses to the empty string it always did.
    assert_eq!(
        finalized_content(serde_json::json!([])).expect("empty content should flatten"),
        serde_json::json!("")
    );
}

/// Textuality is decided on the `type` tag, not on the presence of a
/// `text` key. A part that names a chunk kind is that kind even if it also
/// carries text — deciding on the key alone would flatten the chunk away,
/// which is the silent drop this path exists to prevent.
///
/// The stray key is *dropped*, not forwarded: every Mistral chunk forbids
/// unknown fields, so carrying it through would 422 the whole request and
/// lose the image just as surely.
#[test]
fn finalize_renders_a_chunk_that_also_carries_text_as_its_own_kind() {
    let content = finalized_content(serde_json::json!([
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}, "text": "cat"}
    ]))
    .expect("a tagged image part should convert");

    assert_eq!(
        content,
        serde_json::json!([
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
        ]),
        "the image must reach the wire, in a chunk carrying only the fields Mistral names"
    );
}

/// Finalizing an already-finalized body is a no-op. `finalize_request_body`
/// is a public trait method, so a caller can reach it twice; the chunks
/// this code emits must not read as content Mistral cannot carry.
#[test]
fn finalize_is_idempotent_over_the_chunks_it_emits() {
    let parts = serde_json::json!([
        {"type": "text", "text": "Read these."},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        {"type": "input_audio", "input_audio": "SUQzBAA="},
        {"type": "document_url", "document_url": "data:application/pdf;base64,JVBERi0xLjQK",
         "document_name": "document.pdf"},
        {"type": "file", "file_id": "00000000-0000-0000-0000-000000000000"}
    ]);

    let once = finalized_content(parts).expect("emitted chunks should convert");
    let twice = finalized_content(once.clone()).expect("a second pass should be a no-op");

    assert_eq!(once, twice);
}

/// An image part with no payload names no image at all.
#[test]
fn finalize_rejects_an_image_part_with_no_payload() {
    let error = finalized_content(serde_json::json!([
        {"type": "text", "text": "hi"},
        {"type": "image_url"}
    ]))
    .expect_err("an image part carrying no payload must not be dropped");
    assert!(matches!(error, CompletionError::RequestError(_)));
}
