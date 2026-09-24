/// The shared chat-completions response type is deliberately lenient about
/// envelope metadata: `object`, `created`, `choices[].index`, and
/// `choices[].finish_reason` may be missing or explicit `null` (lossy
/// OpenAI-compatible gateways and Copilot's multi-vendor chat route both
/// rely on this). An empty `finish_reason` normalizes to `None` rather
/// than erroring. This pins that contract next to the type itself.
#[test]
fn completion_response_tolerates_null_or_missing_envelope_metadata() {
    let json = r#"{
            "id": "chatcmpl-1",
            "object": null,
            "created": null,
            "model": "some-model",
            "choices": [{
                "index": null,
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": null
            }]
        }"#;
    let response: super::CompletionResponse =
        serde_json::from_str(json).expect("null envelope metadata should deserialize");
    assert_eq!(response.object, "");
    assert_eq!(response.created, 0);
    assert_eq!(response.choices[0].index, 0);
    assert_eq!(response.choices[0].finish_reason, "");
}

/// Boundary-minted tool ids (`tool-{index}`, from id-less streamed calls)
/// replay to the chat wire as a self-consistent pair: the assistant
/// message's `tool_calls[].id` and the tool result's `tool_call_id` carry
/// the same minted value. The wire requires both fields, so gating minted
/// ids out (the Responses reasoning treatment) is impossible here — and
/// unnecessary: a gateway that omitted ids has no server-side id to
/// validate against, so the consistent pair is accepted. This pins the
/// per-wire upstream rule documented on `SyntheticIds`.
#[test]
fn minted_tool_ids_replay_as_a_consistent_pair() {
    let assistant = crate::message::Message::Assistant {
        id: None,
        content: vec![crate::message::AssistantContent::tool_call(
            "tool-0",
            "get_weather",
            serde_json::json!({"city": "Tokyo"}),
        )],
    };
    let tool_result = crate::message::Message::User {
        content: vec![crate::message::UserContent::tool_result(
            "tool-0",
            "get_weather",
            vec![crate::message::ToolResultContent::text("22C")],
        )],
    };

    let assistant_wire: Vec<super::Message> = assistant.try_into().expect("assistant converts");
    let result_wire: Vec<super::Message> = tool_result.try_into().expect("tool result converts");

    let call_id = assistant_wire
        .iter()
        .find_map(|message| match message {
            super::Message::Assistant { tool_calls, .. } => {
                tool_calls.first().map(|call| call.id.clone())
            }
            _ => None,
        })
        .expect("assistant message carries the tool call");
    let result_id = result_wire
        .iter()
        .find_map(|message| match message {
            super::Message::ToolResult { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .expect("tool result message present");

    assert_eq!(call_id, "tool-0");
    assert_eq!(
        result_id, call_id,
        "the minted pair must be self-consistent"
    );
}

use super::*;
use crate::completion::CompletionRequestBuilder;
use crate::error::ProviderError;
use crate::test_utils::MockCompletionModel;
use serde_json::{Value, json};
use std::collections::HashMap;

fn test_document(id: &str, text: &str) -> crate::completion::Document {
    crate::completion::Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: HashMap::new(),
    }
}

fn request_with_multi_block_tool_result() -> CoreCompletionRequest {
    let tool_result = message::ToolResult {
        call: message::ToolCallId::new_or_minted("call-id", 0),
        provider: message::ProviderCallId::new("call-id"),
        name: "tool".to_string(),
        content: vec![
            message::ToolResultContent::text("first"),
            message::ToolResultContent::text("second"),
        ],
    };

    CoreCompletionRequest {
        model: None,
        chat_history: vec![message::Message::User {
            content: vec![message::UserContent::ToolResult(tool_result)],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

#[test]
fn mixed_user_content_preserves_order_around_tool_results() {
    let content = vec![
        message::UserContent::text("before"),
        message::UserContent::tool_result_with_call_id(
            "result-id",
            "call-id".to_string(),
            "tool",
            vec![message::ToolResultContent::text("tool output")],
        ),
        message::UserContent::text("after"),
    ];

    let messages = user_content_to_messages(content).expect("message conversion");

    assert!(matches!(
        messages.as_slice(),
        [
            Message::User { content: before, .. },
            Message::ToolResult { tool_call_id, .. },
            Message::User { content: after, .. },
        ] if matches!(before.first(), Some(UserContent::Text { text }) if text == "before")
            && tool_call_id == "call-id"
            && matches!(after.first(), Some(UserContent::Text { text }) if text == "after")
    ));
}

#[test]
fn video_url_with_unrecognized_mime_stays_a_url_on_the_wire() {
    let original = "data:video/quicktime;base64,AAAA";
    let rig_content = message::UserContent::video_url(original.to_string(), None);

    let back = UserContent::try_from(rig_content).expect("video should convert back");
    assert!(matches!(
        back,
        UserContent::Video { video_url } if video_url.url == original
    ));
}

#[test]
fn sanitize_plain_text_history_strips_tool_exchange_and_keeps_alternation() {
    let mut messages = vec![
        serde_json::json!({"role": "user", "content": "Look up the label."}),
        serde_json::json!({"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ]}),
        serde_json::json!({"role": "tool", "tool_call_id": "call_1", "content": "crimson"}),
        serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "The label is crimson."}],
            "reasoning_content": "thinking"
        }),
        serde_json::json!({"role": "user", "content": "Thanks!"}),
    ];

    sanitize_plain_text_history(&mut messages, Some(("\n", true)), false, true);

    let roles = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    // tool message removed, tool-call-only assistant dropped, no
    // consecutive assistants left.
    assert_eq!(roles, ["user", "assistant", "user"]);
    assert_eq!(messages[1]["content"], "The label is crimson.");
    assert!(messages[1].get("reasoning_content").is_none());
    assert!(messages[1].get("tool_calls").is_none());
}

#[test]
fn sanitize_plain_text_history_merges_consecutive_user_messages() {
    // Dropping a tool exchange whose final assistant answer never made it
    // into history leaves user/user adjacency, which alternation-strict
    // APIs reject.
    let mut messages = vec![
        serde_json::json!({"role": "user", "content": "Look it up."}),
        serde_json::json!({"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
        ]}),
        serde_json::json!({"role": "tool", "tool_call_id": "call_1", "content": "crimson"}),
        serde_json::json!({"role": "user", "content": "Ask again."}),
    ];

    sanitize_plain_text_history(&mut messages, Some(("\n", true)), false, true);

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "Look it up.\nAsk again.");
}

#[test]
fn flatten_text_content_parts_treats_refusals_as_text() {
    let mut content = serde_json::json!([
        {"type": "text", "text": "Partly:"},
        {"type": "refusal", "refusal": "I cannot help with that."}
    ]);

    flatten_text_content_parts(&mut content, "\n", true);

    assert_eq!(content, "Partly:\nI cannot help with that.");
}

#[test]
fn sanitize_plain_text_history_merges_consecutive_assistant_messages() {
    let mut messages = vec![
        serde_json::json!({"role": "assistant", "content": "First."}),
        serde_json::json!({"role": "tool", "tool_call_id": "c", "content": "x"}),
        serde_json::json!({"role": "assistant", "content": "Second."}),
    ];

    sanitize_plain_text_history(&mut messages, Some(("\n", true)), false, true);

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["content"], "First.\nSecond.");
}

#[test]
fn tool_result_array_content_preserves_multiple_text_blocks() {
    let request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request: request_with_multi_block_tool_result(),
        strict_tools: false,
        tool_result_array_content: true,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");

    let wire = serde_json::to_value(&request.messages).expect("messages should serialize");

    assert_eq!(
        wire,
        serde_json::json!([
            {
                "role": "tool",
                "tool_call_id": "call-id",
                "content": [
                    {
                        "type": "text",
                        "text": "first"
                    },
                    {
                        "type": "text",
                        "text": "second"
                    }
                ]
            }
        ])
    );
}

#[test]
fn tool_result_string_content_flattens_multiple_text_blocks() {
    let request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request: request_with_multi_block_tool_result(),
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");

    let wire = serde_json::to_value(&request.messages).expect("messages should serialize");

    assert_eq!(
        wire,
        serde_json::json!([
            {
                "role": "tool",
                "tool_call_id": "call-id",
                "content": "first\nsecond"
            }
        ])
    );
}

#[test]
fn multiple_tool_result_blocks_convert_to_distinct_content_parts() {
    let result = message::ToolResult {
        call: message::ToolCallId::new_or_minted("call-id", 0),
        name: "tool".to_string(),
        provider: message::ProviderCallId::new("call-id"),
        content: vec![
            message::ToolResultContent::text("first"),
            message::ToolResultContent::json(serde_json::json!({
                "status": "ok"
            })),
            message::ToolResultContent::text("second"),
        ],
    };

    let converted = Message::try_from(result).expect("tool result should convert");

    assert_eq!(
        converted,
        Message::ToolResult {
            tool_call_id: "call-id".to_string(),
            content: ToolResultContentValue::Array(vec![
                ToolResultContent::from("first".to_string()),
                ToolResultContent::from(r#"{"status":"ok"}"#.to_string()),
                ToolResultContent::from("second".to_string()),
            ]),
        }
    );
}

#[test]
fn test_openai_request_uses_request_model_override() {
    let request = crate::completion::CompletionRequest {
        model: Some("gpt-4.1".to_string()),
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");
    let serialized = serde_json::to_value(openai_request).expect("serialization should succeed");

    assert_eq!(serialized["model"], "gpt-4.1");
}

/// The wire refuses a `tool_choice` beside an empty `tools`: a turn that
/// advertises none (an output tool degraded to native output under
/// `tool_choice: none`, an `active_tools: []` patch) carries no choice.
#[test]
fn tool_choice_is_dropped_when_no_tool_is_advertised() {
    let request =
        |tools: Vec<crate::completion::ToolDefinition>| crate::completion::CompletionRequest {
            model: None,
            chat_history: vec!["Hello".into()],
            documents: vec![],
            tools,
            temperature: None,
            max_tokens: None,
            tool_choice: Some(crate::message::ToolChoice::None),
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };
    let convert = |request| {
        CompletionRequest::try_from(OpenAIRequestParams {
            reasoning_details: false,
            model: "gpt-4o-mini".to_string(),
            request,
            strict_tools: false,
            tool_result_array_content: false,
            supports_response_format: true,
            response_format_with_tools: false,
            supports_image_tool_results: false,
            supports_tools: true,
        })
        .expect("request conversion should succeed")
    };

    let without_tools = serde_json::to_value(convert(request(vec![]))).expect("serializes");
    assert!(
        without_tools.get("tool_choice").is_none(),
        "no tool, no choice: {without_tools}"
    );

    let with_tool =
        serde_json::to_value(convert(request(vec![crate::completion::ToolDefinition {
            name: "add".to_string(),
            description: "add two numbers".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }])))
        .expect("serializes");
    assert_eq!(with_tool["tool_choice"], json!("none"), "{with_tool}");
}

#[test]
fn test_openai_request_uses_default_model_when_override_unset() {
    let request = crate::completion::CompletionRequest {
        model: None,
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");
    let serialized = serde_json::to_value(openai_request).expect("serialization should succeed");

    assert_eq!(serialized["model"], "gpt-4o-mini");
}

#[test]
fn openai_chat_request_keeps_documents_after_system_messages() {
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Prompt")
        .message(crate::completion::Message::system("System prompt"))
        .message(crate::completion::Message::user("Earlier user turn"))
        .message(crate::completion::Message::assistant(
            "Earlier assistant turn",
        ))
        .document(test_document("doc1", "Document text."))
        .build();

    let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");

    let serialized =
        serde_json::to_value(&openai_request.messages).expect("messages should serialize");
    let messages = serialized.as_array().expect("messages should be an array");

    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    assert!(
        messages[1].to_string().contains("<file id: doc1>"),
        "document message should follow system message: {messages:?}"
    );
    assert_eq!(messages[2]["role"], "user");
    assert!(
        messages[2].to_string().contains("Earlier user turn"),
        "prior user history should follow document message: {messages:?}"
    );
    assert_eq!(messages[3]["role"], "assistant");
    assert!(
        messages[3].to_string().contains("Earlier assistant turn"),
        "prior assistant history should follow prior user history: {messages:?}"
    );
    assert_eq!(messages[4]["role"], "user");
    assert!(
        messages[4].to_string().contains("Prompt"),
        "prompt should remain last: {messages:?}"
    );
}

#[test]
fn openai_chat_direct_request_keeps_documents_after_system_messages() {
    let request = CoreCompletionRequest {
        model: None,
        chat_history: vec![
            crate::completion::Message::system("System prompt"),
            crate::completion::Message::assistant("Earlier assistant turn"),
            crate::completion::Message::system("Mid-conversation instruction"),
            crate::completion::Message::user("Prompt"),
        ],
        documents: vec![test_document("doc1", "Document text.")],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");

    let serialized =
        serde_json::to_value(&openai_request.messages).expect("messages should serialize");
    let messages = serialized.as_array().expect("messages should be an array");

    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    assert!(
        messages[1].to_string().contains("<file id: doc1>"),
        "document message should follow leading system messages: {messages:?}"
    );
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[3]["role"], "system");
    assert_eq!(messages[4]["role"], "user");
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc1>"))
            .count(),
        1,
        "document message should appear exactly once: {messages:?}"
    );
}

#[test]
fn assistant_reasoning_alone_is_dropped() {
    let assistant_content = vec![message::AssistantContent::reasoning("hidden")];

    let converted: Vec<Message> =
        assistant_content_to_messages(assistant_content, false).expect("conversion should work");

    assert!(converted.is_empty());
}

// Regression test: providers that serve thinking models over the OpenAI
// Chat Completions schema (DeepSeek-R1, GLM-4.6, Qwen3-Thinking) return
// 400 "thinking is enabled but reasoning_content is missing" on the next
// turn if the prior assistant tool-call message didn't echo the reasoning.
#[test]
fn assistant_reasoning_is_attached_to_tool_call_message() {
    let assistant_content = vec![
        message::AssistantContent::reasoning("hidden"),
        message::AssistantContent::text("visible"),
        message::AssistantContent::tool_call(
            "call_1",
            "subtract",
            serde_json::json!({"x": 2, "y": 1}),
        ),
    ];

    let converted: Vec<Message> =
        assistant_content_to_messages(assistant_content, false).expect("conversion should work");
    assert_eq!(converted.len(), 1);

    match &converted[0] {
        Message::Assistant {
            content,
            tool_calls,
            reasoning,
            ..
        } => {
            assert_eq!(
                content,
                &vec![AssistantContent::Text {
                    text: "visible".to_string()
                }]
            );
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].id, "call_1");
            assert_eq!(tool_calls[0].function.name, "subtract");
            assert_eq!(
                tool_calls[0].function.arguments,
                serde_json::json!({"x": 2, "y": 1})
            );
            assert_eq!(reasoning.as_deref(), Some("hidden"));
        }
        _ => panic!("expected assistant message"),
    }

    let json = serde_json::to_value(&converted[0]).expect("serialize");
    assert_eq!(json["reasoning_content"], "hidden");
}

/// The raw text view of an assistant turn joins its non-empty parts in
/// order, refusal parts included. This is the helper every unary path on
/// this wire reads text through.
#[test]
fn assistant_message_text_joins_every_non_empty_part() {
    let message = Message::Assistant {
        content: vec![
            AssistantContent::Text {
                text: "first".to_owned(),
            },
            AssistantContent::Refusal {
                refusal: "second".to_owned(),
            },
            AssistantContent::Text {
                text: "third".to_owned(),
            },
        ],
        reasoning: Some("hidden".to_owned()),
        refusal: None,
        name: None,
        tool_calls: vec![],
        reasoning_details: vec![],
    };

    assert_eq!(
        assistant_message_text_response(&message),
        Some("first\nsecond\nthird".to_owned())
    );
}

#[test]
fn completion_response_retains_service_tier() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "chatcmpl-tier",
        "object": "chat.completion",
        "created": 0,
        "model": GPT_4O,
        "system_fingerprint": "fp_test",
        "service_tier": "priority",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }]
    }))
    .expect("live Chat Completions metadata should deserialize");

    assert_eq!(response.service_tier.as_deref(), Some("priority"));
}

fn usage(body: Value) -> crate::completion::Usage {
    serde_json::from_value::<Usage>(body)
        .expect("chat-completions usage should deserialize")
        .to_normalized()
}

/// A gateway fronting an upstream that bills cache writes (OpenRouter over
/// Anthropic) reports them as `prompt_tokens_details.cache_write_tokens`;
/// they are the normalized `cache_creation_input_tokens`.
#[test]
fn usage_maps_cache_token_accounting() {
    let converted = usage(json!({
        "prompt_tokens": 500,
        "completion_tokens": 10,
        "total_tokens": 510,
        "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 50}
    }));

    assert_eq!(converted.input_tokens, Some(500));
    assert_eq!(converted.output_tokens, Some(10));
    assert_eq!(converted.cached_input_tokens, Some(400));
    assert_eq!(converted.cache_creation_input_tokens, Some(50));
}

/// A counter the provider did not send stays `None`: OpenAI's own detail
/// block has no `cache_write_tokens`, and a reply with no detail block has
/// neither cache counter.
#[test]
fn usage_cache_counters_absent_are_unreported() {
    let openai = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}
    }));
    assert_eq!(openai.cached_input_tokens, Some(0));
    assert_eq!(openai.cache_creation_input_tokens, None);

    let bare = usage(json!({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110
    }));
    assert_eq!(bare.cached_input_tokens, None);
    assert_eq!(bare.cache_creation_input_tokens, None);
}

/// Mistral reports cache hits as the structured block and as a top-level
/// `num_cached_tokens`, sometimes on its own. The structured count wins when
/// both are present. Its embeddings reply also carries the singular
/// `prompt_token_details` (`null`) *beside* the plural key — the live shape
/// that rules out a serde alias — and must still decode.
#[test]
fn usage_prefers_structured_cached_tokens_and_falls_back() {
    let structured = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "num_cached_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 7}
    }));
    assert_eq!(structured.cached_input_tokens, Some(7));

    let fallback = usage(json!({
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "num_cached_tokens": 2
    }));
    assert_eq!(fallback.cached_input_tokens, Some(2));

    let both_spellings = usage(json!({
        "completion_tokens": 0,
        "prompt_token_details": null,
        "prompt_tokens": 42,
        "prompt_tokens_details": null,
        "total_tokens": 42
    }));
    assert_eq!(both_spellings.input_tokens, Some(42));
    assert_eq!(both_spellings.cached_input_tokens, None);
}

/// Mistral's Voxtral models report audio *beside* `prompt_tokens`, so the
/// input count is the sum; a text turn's detail block is unaffected. The
/// numbers are a live Voxtral turn's, quoted verbatim.
#[test]
fn usage_counts_audio_tokens_reported_beside_the_prompt_as_input() {
    let voxtral = usage(json!({
        "prompt_audio_seconds": 0,
        "prompt_tokens": 6,
        "completion_tokens": 2,
        "total_tokens": 383,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 375}
    }));
    assert_eq!(voxtral.input_tokens, Some(381));
    assert_eq!(voxtral.output_tokens, Some(2));
    assert_eq!(
        voxtral.total_tokens,
        Some(381 + 2),
        "the parts must add up to the total Mistral reported"
    );

    let text = usage(json!({
        "prompt_tokens": 19,
        "completion_tokens": 2,
        "total_tokens": 21,
        "prompt_tokens_details": {"cached_tokens": 0}
    }));
    assert_eq!(text.input_tokens, Some(19));
}

/// One chat-completions turn, built from the wire shape a structured-output
/// refusal actually has (`content: null` beside a top-level `refusal`).
fn refusal_message(body: Value) -> Message {
    serde_json::from_value(body).expect("the refusal wire shape must deserialize")
}

/// The fallback as every reader of this wire applies it: the message's own
/// parts decide, and the top-level `refusal` is the turn's text only when
/// they carry nothing.
fn fallback(message: &Message) -> Option<String> {
    let Message::Assistant {
        content, refusal, ..
    } = message
    else {
        panic!("assistant message expected");
    };
    assistant_refusal_fallback(content, refusal.as_deref()).map(str::to_owned)
}

#[test]
fn refusal_sibling_of_null_content_is_the_turns_text() {
    let message = refusal_message(json!({
        "role": "assistant",
        "content": null,
        "refusal": "I'm sorry, I can't help with that."
    }));

    assert_eq!(
        fallback(&message).as_deref(),
        Some("I'm sorry, I can't help with that.")
    );
    assert_eq!(
        assistant_message_text_response(&message).as_deref(),
        Some("I'm sorry, I can't help with that."),
        "the raw text view routes through the same rule"
    );
}

/// Content wins: the fallback only fires when the parts carry nothing, so a
/// turn with both never duplicates its text.
#[test]
fn refusal_beside_non_empty_content_does_not_duplicate() {
    let message = refusal_message(json!({
        "role": "assistant",
        "content": "here is the answer",
        "refusal": "I'm sorry, I can't help with that."
    }));

    assert_eq!(fallback(&message), None);
    assert_eq!(
        assistant_message_text_response(&message).as_deref(),
        Some("here is the answer")
    );
}

/// An empty `refusal` is not content: the turn stays empty rather than
/// gaining a fabricated empty text block.
#[test]
fn empty_refusal_is_not_content() {
    let message = refusal_message(json!({
        "role": "assistant",
        "content": null,
        "refusal": ""
    }));

    assert_eq!(fallback(&message), None);
    assert_eq!(assistant_message_text_response(&message), None);
}

/// The Responses-shaped `refusal` **content part** is not what chat
/// completions sends, but the model still accepts it — and it must not
/// also trigger the sibling fallback.
#[test]
fn refusal_content_part_suppresses_the_sibling_fallback() {
    let message = refusal_message(json!({
        "role": "assistant",
        "content": [{ "type": "refusal", "refusal": "part refusal" }],
        "refusal": "sibling refusal"
    }));

    assert_eq!(fallback(&message), None);
    assert_eq!(
        assistant_message_text_response(&message).as_deref(),
        Some("part refusal")
    );
}

#[test]
fn test_max_tokens_is_forwarded_to_request() {
    let request = crate::completion::CompletionRequest {
        model: None,
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(4096),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");
    let serialized = serde_json::to_value(openai_request).expect("serialization should succeed");

    assert_eq!(serialized["max_tokens"], 4096);
}

/// A chat-completions request whose only interesting property is the cap.
fn capped_request(max_tokens: Option<u64>, additional_params: Option<Value>) -> CompletionRequest {
    CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request: crate::completion::CompletionRequest {
            model: None,
            chat_history: vec!["Hello".into()],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens,
            tool_choice: None,
            additional_params,
            output_schema: None,
            record_telemetry_content: false,
        },
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed")
}

#[test]
fn request_body_keeps_the_legacy_cap_when_the_endpoint_wants_it() {
    let body =
        request_body(&capped_request(Some(4096), None), false).expect("body should serialize");

    assert_eq!(body["max_tokens"], 4096);
    assert!(body.get("max_completion_tokens").is_none());
}

#[test]
fn request_body_renames_the_cap_for_the_modern_endpoint() {
    let body =
        request_body(&capped_request(Some(4096), None), true).expect("body should serialize");

    assert_eq!(body["max_completion_tokens"], 4096);
    assert!(
        body.get("max_tokens").is_none(),
        "the legacy key must leave the body: reasoning models reject its presence"
    );
}

#[test]
fn request_body_without_a_cap_carries_neither_spelling() {
    let body = request_body(&capped_request(None, None), true).expect("body should serialize");

    assert!(body.get("max_tokens").is_none());
    assert!(body.get("max_completion_tokens").is_none());
}

#[test]
fn request_body_keeps_a_caller_supplied_modern_cap() {
    let body = request_body(
        &capped_request(Some(4096), Some(json!({ "max_completion_tokens": 48 }))),
        true,
    )
    .expect("body should serialize");

    assert_eq!(body["max_completion_tokens"], 48);
    assert!(body.get("max_tokens").is_none());
}

#[test]
fn request_body_upgrades_a_caller_supplied_legacy_cap() {
    let body = request_body(
        &capped_request(None, Some(json!({ "max_tokens": 48 }))),
        true,
    )
    .expect("body should serialize");

    assert_eq!(body["max_completion_tokens"], 48);
    assert!(body.get("max_tokens").is_none());
}

#[test]
fn request_body_moves_nothing_but_the_cap() {
    let request = capped_request(Some(4096), Some(json!({ "top_p": 0.5 })));
    let plain = serde_json::to_value(&request).expect("serialization should succeed");
    let mut renamed = request_body(&request, true).expect("body");

    let cap = renamed
        .as_object_mut()
        .expect("object body")
        .remove("max_completion_tokens")
        .expect("renamed cap");
    renamed["max_tokens"] = cap;

    assert_eq!(renamed, plain);
}

/// The gate itself, over every family whose behavior was measured against
/// the live endpoint: the reasoning models reject the legacy field, and
/// everything else — including OpenAI's own older models and any
/// compatible server's model names — still gets the bytes it always got.
#[test]
fn modern_output_cap_covers_exactly_the_reasoning_families() {
    for model in [
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5-nano",
        "gpt-5-2025-08-07",
        "gpt-6",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
        "o4-mini-2025-04-16",
    ] {
        assert!(
            is_openai_reasoning_model(model),
            "{model} rejects `max_tokens` and must get the modern spelling"
        );
    }

    for model in [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-nano",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "chatgpt-4o-latest",
        // Compatible-server model names reached through this extension.
        "Qwen/Qwen3-4B",
        "openai/gpt-oss-20b",
        "gpt-oss-120b",
        "llama-3.1-8b-instruct",
        // Near misses that must not be read as a family or a series.
        "gpt-45",
        "gpt-",
        "o",
        "opus",
        "o5x",
        "",
    ] {
        assert!(
            !is_openai_reasoning_model(model),
            "{model:?} still takes `max_tokens`; changing its request would be a regression"
        );
    }
}

#[test]
fn test_max_tokens_omitted_when_none() {
    let request = crate::completion::CompletionRequest {
        model: None,
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");
    let serialized = serde_json::to_value(openai_request).expect("serialization should succeed");

    assert!(serialized.get("max_tokens").is_none());
}

/// A mixed `additional_params.tools` array splits by shape: function tools
/// merge into the typed `tools` field (issue #1890 — left in the flattened
/// params they replace the typed field at serialization), while
/// non-function entries stay behind for the provider's `prepare_request`
/// hook (Groq folds its native tools into `compound_custom` from there).
/// Not a cassette test: OpenAI proper rejects non-function chat tools, so
/// the retained-entry half cannot be recorded against the live API.
#[test]
fn additional_params_function_tools_merge_and_native_tools_stay() {
    let request = CoreCompletionRequest {
        model: None,
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![crate::completion::ToolDefinition {
            name: "builder_tool".to_string(),
            description: "from the builder".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: Some(serde_json::json!({
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "params_tool",
                        "description": "from additional_params",
                        "parameters": {"type": "object", "properties": {}}
                    }
                },
                {"type": "browser_search"}
            ]
        })),
        output_schema: None,
        record_telemetry_content: false,
    };

    let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");

    let names: Vec<&str> = openai_request
        .tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect();
    assert_eq!(names, vec!["builder_tool", "params_tool"]);
    assert_eq!(
        openai_request.additional_params,
        Some(serde_json::json!({"tools": [{"type": "browser_search"}]}))
    );
}

#[test]
fn request_conversion_errors_when_all_messages_are_filtered() {
    let request = CoreCompletionRequest {
        model: None,
        chat_history: vec![message::Message::Assistant {
            id: None,
            content: vec![message::AssistantContent::reasoning("hidden")],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let result = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    });

    assert!(matches!(
        result.map_err(ProviderError::from),
        Err(ProviderError::Request(_))
    ));
}

#[test]
fn request_conversion_omits_response_format_on_initial_tool_turn() {
    let request = CoreCompletionRequest {
        model: None,
        chat_history: vec![message::Message::user(
            "Hello, whats the weather in London?",
        )],
        documents: vec![],
        tools: vec![completion::ToolDefinition {
            name: "weather".to_string(),
            description: "Get the weather".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        }],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: Some(
            serde_json::from_value(serde_json::json!({
                "title": "WeatherResponse",
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "weather": { "type": "string" }
                },
                "required": ["city", "weather"]
            }))
            .expect("schema should deserialize"),
        ),
        record_telemetry_content: false,
    };

    let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");

    let serialized = serde_json::to_value(openai_request).expect("serialization should succeed");

    assert!(
        serialized.get("response_format").is_none(),
        "initial tool turn should omit response_format: {serialized:?}"
    );
}

#[test]
fn request_conversion_restores_response_format_after_tool_result() {
    let request = CoreCompletionRequest {
        model: None,
        chat_history: vec![
            message::Message::user("Hello, whats the weather in London?"),
            message::Message::Assistant {
                id: None,
                content: vec![message::AssistantContent::tool_call(
                    "call_1",
                    "weather",
                    serde_json::json!({ "city": "London" }),
                )],
            },
            message::Message::tool_result(
                "call_1",
                "weather",
                "The weather in London is all fire and brimstone",
            ),
        ],
        documents: vec![],
        tools: vec![completion::ToolDefinition {
            name: "weather".to_string(),
            description: "Get the weather".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        }],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: Some(
            serde_json::from_value(serde_json::json!({
                "title": "WeatherResponse",
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "weather": { "type": "string" }
                },
                "required": ["city", "weather"]
            }))
            .expect("schema should deserialize"),
        ),
        record_telemetry_content: false,
    };

    let openai_request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");

    let serialized = serde_json::to_value(openai_request).expect("serialization should succeed");

    assert!(
        serialized.get("response_format").is_some(),
        "follow-up turn should restore response_format: {serialized:?}"
    );
}

#[test]
fn deserialize_llama_cpp_tool_call() {
    let request = r#"{
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{ "type": "function", "function": { "name": "hello_world", "arguments": { "city": "Paris" } }, "id": "xxx" }]
                }
            }],
            "created": 0,
            "model": "gpt-4o-mini",
            "system_fingerprint": "fp_xxx",
            "object": "chat.completion",
            "usage": { "completion_tokens": 13, "prompt_tokens": 255, "total_tokens": 268 },
            "id": "xxx"
        }
        "#;
    let response: CompletionResponse =
        serde_json::from_str(request).expect("the llama.cpp-shaped reply should decode");

    assert_eq!(response.choices.len(), 1);

    let Message::Assistant { tool_calls, .. } = &response.choices[0].message else {
        panic!("expected assistant message");
    };
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "xxx");
    assert_eq!(tool_calls[0].function.name, "hello_world");
    assert_eq!(
        tool_calls[0].function.arguments,
        serde_json::json!({"city": "Paris"})
    );
}

#[test]
fn deserialize_openai_stringified_tool_call() {
    let request = r#"{
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{ "type": "function", "function": { "name": "hello_world", "arguments": "{\"city\":\"Paris\"}" }, "id": "xxx" }]
                }
            }],
            "created": 0,
            "model": "gpt-4o-mini",
            "system_fingerprint": "fp_xxx",
            "object": "chat.completion",
            "usage": { "completion_tokens": 13, "prompt_tokens": 255, "total_tokens": 268 },
            "id": "xxx"
        }
        "#;
    let response: CompletionResponse =
        serde_json::from_str(request).expect("the OpenAI-shaped reply should decode");

    assert_eq!(response.choices.len(), 1);

    let Message::Assistant { tool_calls, .. } = &response.choices[0].message else {
        panic!("expected assistant message");
    };
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "xxx");
    assert_eq!(tool_calls[0].function.name, "hello_world");
    assert_eq!(
        tool_calls[0].function.arguments,
        serde_json::json!({"city": "Paris"})
    );
}

/// A `max_tokens`-capped turn still emits the tool call, with `arguments`
/// cut off partway through the JSON object. Parsing strictly failed the
/// *whole* response -- the text, usage, id and finish reason went with it
/// -- where the streaming path keeps the turn and drops the unusable call.
/// Reproduced live against DeepSeek (rig#2354) at 24/32/48/64-token
/// budgets; this wire type backs every other OpenAI-compatible provider in
/// the tree, so the same shape is pinned here.
#[test]
fn truncated_tool_arguments_do_not_destroy_the_response() {
    let request = r#"{
            "choices": [{
                "finish_reason": "length",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Acknowledged.",
                    "tool_calls": [
                        { "type": "function", "id": "call_1", "function": { "name": "page", "arguments": "{\"team\":\"platform\"}" } },
                        { "type": "function", "id": "call_2", "function": { "name": "file_report", "arguments": "{\"summary\": " } }
                    ]
                }
            }],
            "created": 0,
            "model": "gpt-4o-mini",
            "object": "chat.completion",
            "usage": { "completion_tokens": 24, "prompt_tokens": 372, "total_tokens": 396 },
            "id": "chatcmpl-truncated"
        }
        "#;

    let response: CompletionResponse =
        serde_json::from_str(request).expect("the truncated turn should survive decode");

    let Message::Assistant {
        content,
        tool_calls,
        ..
    } = &response.choices[0].message
    else {
        panic!("expected assistant message");
    };
    assert_eq!(
        tool_calls.len(),
        1,
        "the unusable call is dropped at decode; the complete one survives"
    );
    assert_eq!(tool_calls[0].function.name, "page");

    // The rest of the turn is untouched: the text, the usage, the id and
    // the terminal reason all survive the dropped call.
    assert!(matches!(
        content.as_slice(),
        [AssistantContent::Text { text }] if text == "Acknowledged."
    ));
    assert_eq!(response.choices[0].finish_reason, "length");
    assert_eq!(response.usage.expect("usage").total_tokens, 396);
    assert_eq!(response.id, "chatcmpl-truncated");
}

fn response_with_tool_call(finish_reason: &str, call: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "choices": [{
            "finish_reason": finish_reason,
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [call]
            },
            "logprobs": null
        }],
        "created": 0,
        "model": "gpt-4o-mini",
        "object": "chat.completion",
        "system_fingerprint": null,
        "usage": { "completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2 },
        "id": "chatcmpl-tool-call"
    })
}

/// Invalid JSON on a completed tool turn is a provider defect, not
/// truncation evidence. It must stay visible on the native raw surface.
#[test]
fn malformed_completed_tool_call_is_not_silently_dropped() {
    let response = response_with_tool_call(
        "tool_calls",
        serde_json::json!({
            "type": "function",
            "id": "call_1",
            "function": { "name": "page", "arguments": "{\"team\":" }
        }),
    );

    assert!(
        serde_json::from_value::<CompletionResponse>(response).is_err(),
        "ordinary malformed tool output must remain a loud response defect"
    );
}

/// Repairing the arguments in a validation copy must not hide an
/// independent defect on the same truncated call.
#[test]
fn truncated_tool_call_with_a_compound_defect_is_not_dropped() {
    let response = response_with_tool_call(
        "length",
        serde_json::json!({
            "type": "not_a_real_tool_type",
            "id": "call_1",
            "function": { "name": "page", "arguments": "{\"team\":" }
        }),
    );

    assert!(
        serde_json::from_value::<CompletionResponse>(response).is_err(),
        "the unknown type must remain loud even beside truncated arguments"
    );
}

/// Under `length`, an empty string means the turn ended before the first
/// argument token. Treating it as `{}` could dispatch a zero-argument
/// side-effect tool from an incomplete turn.
#[test]
fn output_length_drops_a_tool_call_with_no_argument_tokens() {
    let response = response_with_tool_call(
        "length",
        serde_json::json!({
            "type": "function",
            "id": "call_1",
            "function": { "name": "page", "arguments": "" }
        }),
    );
    let response: CompletionResponse =
        serde_json::from_value(response).expect("the truncated turn should survive");
    let Message::Assistant { tool_calls, .. } = &response.choices[0].message else {
        panic!("expected assistant message");
    };
    assert!(tool_calls.is_empty());
}

/// The choice-level truncation policy must not weaken a complete payload:
/// an empty string and Groq's literal `"null"` are both parameterless
/// invocations, and object-valued `arguments` (llama.cpp, Hugging Face)
/// still pass through untouched.
///
/// The `"null"` spelling is not hypothetical: every zero-argument call in
/// `crates/rig-cassette/fixtures/cassettes/groq/agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming.yaml`
/// carries it, so folding it to `{}` is what keeps the truncation sentinel
/// from swallowing a real call.
#[test]
fn tolerant_tool_arguments_leave_complete_payloads_alone() {
    let request = r#"{
            "choices": [{
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        { "type": "function", "id": "a", "function": { "name": "ping", "arguments": "" } },
                        { "type": "function", "id": "b", "function": { "name": "hello", "arguments": { "city": "Paris" } } },
                        { "type": "function", "id": "c", "function": { "name": "pong", "arguments": "null" } },
                        { "type": "function", "id": "d", "function": { "name": "pang", "arguments": null } }
                    ]
                }
            }],
            "created": 0,
            "model": "gpt-4o-mini",
            "object": "chat.completion",
            "usage": { "completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2 },
            "id": "chatcmpl-complete"
        }
        "#;

    let response: CompletionResponse =
        serde_json::from_str(request).expect("every complete call should survive decode");
    let Message::Assistant { tool_calls, .. } = &response.choices[0].message else {
        panic!("expected assistant message");
    };
    assert_eq!(tool_calls.len(), 4, "every completed call survives");
    assert_eq!(tool_calls[0].function.arguments, serde_json::json!({}));
    assert_eq!(
        tool_calls[1].function.arguments,
        serde_json::json!({"city": "Paris"})
    );
    assert_eq!(
        tool_calls[2].function.arguments,
        serde_json::Value::Null,
        "Groq's `\"null\"` spelling parses, so the call survives untouched — \
             which is exactly why `null` cannot be a truncation sentinel"
    );
    assert_eq!(
        tool_calls[3].function.arguments,
        serde_json::Value::Null,
        "and the same for a bare JSON null in the non-string branch"
    );
}

/// The Doubleword reasoning tool-loop cassette exposed text-before-reasoning
/// in the shared unary converter. The ordering itself is the chat wire's
/// rule; what this pins is the decode the wire reads it from — a
/// `reasoning_content` sibling of `content` lands on the assistant
/// message's `reasoning` field rather than being dropped.
#[test]
fn compatible_response_decodes_reasoning_content_beside_text() {
    let request = r#"
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer.",
                        "reasoning_content": "Now I understand the structure better. I need to: ..."
                    }
                }
            ],
            "created": 1776750378,
            "model": "unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0",
            "system_fingerprint": "fp_xxx",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 920,
                "prompt_tokens": 27806,
                "total_tokens": 28726,
                "prompt_tokens_details": { "cached_tokens": 18698 }
            },
            "id": "chatcmpl-xxxx",
            "timings": {
                "cache_n": 18698,
                "prompt_n": 9108,
                "prompt_ms": 226645.81,
                "prompt_per_token_ms": 24.884256697408873,
                "prompt_per_second": 40.186050648807495,
                "predicted_n": 920,
                "predicted_ms": 177167.955,
                "predicted_per_token_ms": 192.57386413043477,
                "predicted_per_second": 5.192812661860888
            }
        }
        "#;
    let response: CompletionResponse =
        serde_json::from_str(request).expect("the llama.cpp-shaped reply should decode");

    let Message::Assistant {
        content, reasoning, ..
    } = &response.choices[0].message
    else {
        panic!("expected assistant message");
    };
    assert_eq!(
        reasoning.as_deref(),
        Some("Now I understand the structure better. I need to: ...")
    );
    assert!(matches!(
        content.as_slice(),
        [AssistantContent::Text { text }] if text == "The answer."
    ));
}

#[test]
fn pdf_base64_document_serializes_as_file_content_part() {
    let doc = message::UserContent::Document(message::Document {
        data: DocumentSourceKind::Base64("JVBERi0xLjQK".into()),
        media_type: Some(message::DocumentMediaType::PDF),
        additional_params: None,
    });
    let converted: UserContent = doc.try_into().expect("conversion should succeed");
    let json = serde_json::to_value(&converted).expect("serialize");

    assert_eq!(json["type"], "file");
    assert_eq!(
        json["file"]["file_data"],
        "data:application/pdf;base64,JVBERi0xLjQK"
    );
    assert_eq!(json["file"]["filename"], "document.pdf");
    assert!(json["file"].get("file_id").is_none());
}

#[test]
fn file_id_document_serializes_as_file_content_part() {
    let doc = message::UserContent::Document(message::Document {
        data: DocumentSourceKind::FileId("file_abc".into()),
        media_type: None,
        additional_params: None,
    });
    let converted: UserContent = doc.try_into().expect("conversion should succeed");
    let json = serde_json::to_value(&converted).expect("serialize");

    assert_eq!(json["type"], "file");
    assert_eq!(json["file"]["file_id"], "file_abc");
    assert!(json["file"].get("file_data").is_none());
}

#[test]
fn base64_image_without_detail_defaults_to_auto() {
    let image = message::UserContent::Image(message::Image {
        data: DocumentSourceKind::Base64("iVBORw0KGgo=".into()),
        media_type: Some(message::ImageMediaType::PNG),
        detail: None,
        additional_params: None,
    });
    let converted: UserContent = image.try_into().expect("conversion should succeed");
    let UserContent::Image { image_url } = converted else {
        panic!("expected image content");
    };

    assert_eq!(image_url.url, "data:image/png;base64,iVBORw0KGgo=");
    assert_eq!(image_url.detail, Some(ImageDetail::Auto));
}

// Regression guard: callers passing markdown/plain text wrapped in
// `UserContent::Document` should keep getting flattened to `text`.
#[test]
fn non_pdf_document_still_serializes_as_text() {
    let doc = message::UserContent::Document(message::Document {
        data: DocumentSourceKind::String("# Markdown".into()),
        media_type: None,
        additional_params: None,
    });
    let converted: UserContent = doc.try_into().expect("conversion should succeed");
    let json = serde_json::to_value(&converted).expect("serialize");

    assert_eq!(json["type"], "text");
    assert_eq!(json["text"], "# Markdown");
}

#[test]
fn pdf_url_document_returns_conversion_error() {
    let doc = message::UserContent::Document(message::Document {
        data: DocumentSourceKind::Url("https://example.com/x.pdf".into()),
        media_type: Some(message::DocumentMediaType::PDF),
        additional_params: None,
    });
    let res: Result<UserContent, _> = doc.try_into();
    assert!(matches!(
        res,
        Err(message::MessageError::ConversionError(_))
    ));
}

#[test]
fn pdf_raw_document_returns_conversion_error() {
    let doc = message::UserContent::Document(message::Document {
        data: DocumentSourceKind::Raw(b"%PDF-1.4\n".to_vec()),
        media_type: Some(message::DocumentMediaType::PDF),
        additional_params: None,
    });
    let res: Result<UserContent, _> = doc.try_into();
    assert!(matches!(
        res,
        Err(message::MessageError::ConversionError(_))
    ));
}

#[test]
fn file_user_content_deserializes_from_wire_json() {
    let raw = r#"{"type":"file","file":{"file_data":"data:application/pdf;base64,AAAA","filename":"x.pdf"}}"#;
    let parsed: UserContent = serde_json::from_str(raw).expect("deserialize");
    let UserContent::File { file } = parsed else {
        panic!("expected File variant");
    };
    assert_eq!(
        file.file_data.as_deref(),
        Some("data:application/pdf;base64,AAAA")
    );
    assert_eq!(file.filename.as_deref(), Some("x.pdf"));
    assert!(file.file_id.is_none());
}

#[test]
fn document_file_id_serializes_as_a_file_part() {
    let doc = message::Document {
        data: DocumentSourceKind::FileId("file_abc".to_string()),
        media_type: None,
        additional_params: None,
    };

    let converted: UserContent = message::UserContent::Document(doc)
        .try_into()
        .expect("conversion should succeed");
    let json = serde_json::to_value(&converted).expect("serialize");

    assert_eq!(json["type"], "file");
    assert_eq!(json["file"]["file_id"], "file_abc");
    assert!(json["file"].get("file_data").is_none());
}

// A mixed text + PDF message must produce one User message carrying both
// parts, rather than being flattened or split at the User content site.
#[test]
fn mixed_text_and_pdf_user_message_produces_two_content_parts() {
    let user = message::Message::User {
        content: vec![
            message::UserContent::text("What is in this PDF?"),
            message::UserContent::Document(message::Document {
                data: DocumentSourceKind::Base64("JVBERi0K".into()),
                media_type: Some(message::DocumentMediaType::PDF),
                additional_params: None,
            }),
        ],
    };
    let converted: Vec<Message> = user.try_into().expect("conversion should succeed");
    assert_eq!(converted.len(), 1);
    let Message::User { content, .. } = &converted[0] else {
        panic!("expected user message");
    };
    let parts: Vec<&UserContent> = content.iter().collect();
    assert_eq!(parts.len(), 2);
    assert!(matches!(parts[0], UserContent::Text { .. }));
    assert!(matches!(parts[1], UserContent::File { .. }));
}

/// Synthetic history exercises a collision and result order that cannot be
/// reliably elicited from a live provider; assertions inspect the request wire.
#[test]
fn request_plans_tool_ids_across_namespaces_turns_and_split_user_content() {
    use crate::message::{AssistantContent, ToolCallId, ToolFunction, UserContent};
    let generated = message::ToolCall {
        id: ToolCallId::minted(0),
        provider: None,
        function: ToolFunction {
            name: "test".into(),
            arguments: serde_json::json!({}),
        },
        signature: None,
        additional_params: None,
    };
    let hint = generated.id.wire_hint().into_owned();
    let real = message::ToolCall::from_wire(&hint, generated.function.clone());
    let history = vec![
        message::Message::Assistant {
            id: None,
            content: vec![
                AssistantContent::ToolCall(generated.clone()),
                AssistantContent::ToolCall(real.clone()),
            ],
        },
        message::Message::User {
            content: vec![
                UserContent::tool_result_for(
                    real.id.clone(),
                    real.provider.clone(),
                    "test",
                    vec![],
                ),
                UserContent::text("between results"),
                UserContent::tool_result_for(
                    generated.id.clone(),
                    generated.provider.clone(),
                    "test",
                    vec![],
                ),
            ],
        },
        message::Message::Assistant {
            id: None,
            content: vec![AssistantContent::ToolCall(generated.clone())],
        },
        message::Message::User {
            content: vec![UserContent::tool_result_for(
                generated.id.clone(),
                generated.provider.clone(),
                "test",
                vec![],
            )],
        },
    ];
    let request = CoreCompletionRequest {
        model: None,
        chat_history: history,
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };
    let wire = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "test".into(),
        request,
        strict_tools: false,
        tool_result_array_content: true,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .unwrap();
    let wire = serde_json::to_value(wire).unwrap();
    let messages = wire["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 6);
    let first = &messages[0]["tool_calls"][0]["id"];
    let genuine = &messages[0]["tool_calls"][1]["id"];
    let later = &messages[4]["tool_calls"][0]["id"];
    assert_eq!(genuine, &serde_json::json!(hint));
    assert_ne!(first, genuine);
    assert_ne!(later, first);
    assert_ne!(later, genuine);
    assert_eq!(&messages[1]["tool_call_id"], genuine);
    assert_eq!(&messages[3]["tool_call_id"], first);
    assert_eq!(&messages[5]["tool_call_id"], later);
}

/// The documented precedence on the wire: `additional_params` is flattened
/// into the request body after the typed fields, so a passthrough key
/// overrides the typed value (`build` warns about it; it does not stop it).
#[test]
fn additional_params_override_typed_fields_on_the_wire() {
    let rig_request = crate::completion::CompletionRequestBuilder::unbound("hi")
        .temperature(0.1)
        .additional_params(serde_json::json!({"temperature": 0.9, "top_p": 0.5}))
        .build();
    let request = CompletionRequest::try_from(OpenAIRequestParams {
        reasoning_details: false,
        model: "gpt-4o-mini".to_string(),
        request: rig_request,
        strict_tools: false,
        tool_result_array_content: true,
        supports_response_format: true,
        response_format_with_tools: false,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request conversion should succeed");

    let body = request_body(&request, false).expect("body should serialize");
    assert_eq!(body["temperature"], 0.9, "{body}");
    assert_eq!(body["top_p"], 0.5, "{body}");
}
