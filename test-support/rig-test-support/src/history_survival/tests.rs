use super::*;
use serde_json::json;

fn kinds(tokens: &[Token]) -> Vec<(&'static str, &str)> {
    tokens
        .iter()
        .map(|token| (token.kind, token.value.as_str()))
        .collect()
}

#[test]
fn anthropic_blocks_deliver_signatures_redactions_and_tool_ids() {
    let body = json!({ "content": [
        { "type": "thinking", "thinking": "…", "signature": "sig-1" },
        { "type": "redacted_thinking", "data": "blob-1" },
        { "type": "tool_use", "id": "toolu_1", "name": "f", "input": {} },
        { "type": "text", "text": "answer" },
    ]})
    .to_string();
    let tokens = response_tokens(Dialect::AnthropicMessages, &body);
    assert_eq!(
        kinds(&tokens),
        [
            ("redacted_reasoning", "blob-1"),
            ("signature", "sig-1"),
            ("tool_call_id", "toolu_1"),
        ]
    );
    let next = json!({ "messages": [
        { "role": "assistant", "content": [
            { "type": "thinking", "thinking": "…", "signature": "sig-1" },
            { "type": "tool_use", "id": "toolu_1", "name": "f", "input": {} },
        ]},
        { "role": "user", "content": [
            { "type": "tool_result", "tool_use_id": "toolu_1", "content": "ok" },
        ]},
    ]});
    assert_eq!(
        kinds(&lost_tokens(Dialect::AnthropicMessages, &body, &next)),
        [("redacted_reasoning", "blob-1")]
    );
}

#[test]
fn anthropic_stream_signature_deltas_count_as_delivered() {
    let body = "event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-s\"}}\n\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_s\",\"name\":\"f\",\"input\":{}}}\n\n";
    assert_eq!(
        kinds(&response_tokens(Dialect::AnthropicMessages, body)),
        [("signature", "sig-s"), ("tool_call_id", "toolu_s")]
    );
}

#[test]
fn responses_streams_count_only_the_done_item() {
    let body = "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"added\"}}\n\n\
data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"done\"}}\n\n\
data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"f\",\"arguments\":\"{}\"}}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"reasoning\",\"id\":\"rs_1\",\"encrypted_content\":\"completed\"}]}}\n\n";
    assert_eq!(
        kinds(&response_tokens(Dialect::OpenAiResponses, body)),
        [
            ("encrypted_content", "done"),
            ("reasoning_id", "rs_1"),
            ("tool_call_id", "call_1"),
        ]
    );
    let whole = json!({ "object": "response", "output": [
        { "type": "reasoning", "id": "rs_2", "encrypted_content": "whole", "summary": [] },
    ]})
    .to_string();
    assert_eq!(
        kinds(&response_tokens(Dialect::OpenAiResponses, &whole)),
        [("encrypted_content", "whole"), ("reasoning_id", "rs_2")]
    );
}

#[test]
fn chat_completion_streams_deliver_tool_ids_once() {
    let body = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"function\":{\"name\":\"f\",\"arguments\":\"\"}}]}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n\
data: [DONE]\n\n";
    assert_eq!(
        kinds(&response_tokens(Dialect::ChatCompletions, body)),
        [("tool_call_id", "call_a")]
    );
}

#[test]
fn gemini_and_ollama_records_deliver_their_opaque_fields() {
    let gemini = json!({ "candidates": [{ "content": { "parts": [
        { "text": "t", "thoughtSignature": "ts-1" },
        { "functionCall": { "name": "f", "args": {}, "id": "call_g" } },
    ]}}]})
    .to_string();
    assert_eq!(
        kinds(&response_tokens(Dialect::GeminiGenerateContent, &gemini)),
        [("thought_signature", "ts-1"), ("tool_call_id", "call_g")]
    );
    let ollama = "{\"message\":{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_o\",\"function\":{\"name\":\"f\",\"arguments\":{}}}]},\"done\":false}\n\
{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n";
    assert_eq!(
        kinds(&response_tokens(Dialect::OllamaChat, ollama)),
        [("tool_call_id", "call_o")]
    );
}

#[test]
fn continuation_requires_the_same_opening_turn_and_growth() {
    let first = json!({ "messages": [
        { "role": "system", "content": "s" }, { "role": "user", "content": "u1" },
    ]});
    let grown = json!({ "messages": [
        { "role": "system", "content": "s" }, { "role": "user", "content": "u1" },
        { "role": "assistant", "content": "a" }, { "role": "user", "content": "u2" },
    ]});
    let other = json!({ "messages": [
        { "role": "system", "content": "s" }, { "role": "user", "content": "different" },
        { "role": "assistant", "content": "a" },
    ]});
    assert!(continues(Dialect::ChatCompletions, &first, &grown));
    assert!(!continues(Dialect::ChatCompletions, &first, &other));
    assert!(!continues(Dialect::ChatCompletions, &grown, &first));
}

#[test]
fn pairing_by_id_reports_dangling_calls_and_results() {
    let request = json!({ "messages": [
        { "role": "user", "content": "u" },
        { "role": "assistant", "tool_calls": [
            { "id": "call_1", "function": { "name": "f", "arguments": "{}" } },
            { "id": "call_2", "function": { "name": "g", "arguments": "{}" } },
        ]},
        { "role": "tool", "tool_call_id": "call_1", "content": "ok" },
        { "role": "tool", "tool_call_id": "call_9", "content": "stray" },
        { "role": "assistant", "content": "answer" },
    ]});
    let unpaired = unpaired_tool_calls(Dialect::ChatCompletions, &request);
    assert_eq!(
        unpaired
            .iter()
            .map(|entry| (entry.side, entry.label.as_str()))
            .collect::<Vec<_>>(),
        [("result", "call_9"), ("call", "call_2")]
    );
}

#[test]
fn responses_parallel_calls_form_one_assistant_run() {
    let request = json!({ "input": [
        { "type": "message", "role": "user", "content": "u" },
        { "type": "reasoning", "id": "rs_1", "summary": [] },
        { "type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}" },
        { "type": "function_call", "call_id": "call_2", "name": "g", "arguments": "{}" },
        { "type": "function_call_output", "call_id": "call_2", "output": "b" },
        { "type": "function_call_output", "call_id": "call_1", "output": "a" },
        { "type": "message", "role": "assistant", "content": "answer" },
    ]});
    assert!(unpaired_tool_calls(Dialect::OpenAiResponses, &request).is_empty());
}

#[test]
fn ollama_and_gemini_pair_by_name_in_order() {
    let ollama = json!({ "messages": [
        { "role": "assistant", "content": "", "tool_calls": [
            { "function": { "name": "add", "arguments": {} } },
            { "function": { "name": "multiply", "arguments": {} } },
        ]},
        { "role": "tool", "tool_name": "add", "content": "10" },
    ]});
    let unpaired = unpaired_tool_calls(Dialect::OllamaChat, &ollama);
    assert_eq!(unpaired.len(), 1);
    assert_eq!(unpaired[0].label, "multiply");

    let gemini = json!({ "contents": [
        { "role": "model", "parts": [{ "functionCall": { "name": "f", "args": {} } }] },
        { "role": "user", "parts": [{ "functionResponse": { "name": "f", "response": {} } }] },
    ]});
    assert!(unpaired_tool_calls(Dialect::GeminiGenerateContent, &gemini).is_empty());
}

#[test]
fn normalized_histories_pair_by_call_identity() {
    let call = json!({ "origin": "explicit", "id": "toolu_1" });
    let history = vec![
        json!({ "role": "user", "content": [{ "type": "text", "text": "u" }] }),
        json!({ "role": "assistant", "content": [
            { "type": "toolcall", "id": call, "function": { "name": "f", "arguments": {} } },
            { "type": "toolcall", "id": { "origin": "generated", "id": "tool-1" }, "function": { "name": "g", "arguments": {} } },
        ]}),
        json!({ "role": "user", "content": [
            { "type": "toolresult", "call": call, "name": "f", "content": [] },
        ]}),
        json!({ "role": "assistant", "content": [{ "type": "text", "text": "done" }] }),
    ];
    let unpaired = unpaired_normalized_tool_calls(&history);
    assert_eq!(unpaired.len(), 1);
    assert_eq!(
        (unpaired[0].side, unpaired[0].label.as_str()),
        ("call", "tool-1")
    );
}

/// Keeping a tool-call id on the result but dropping it from the call is a
/// loss, even though the value still appears in the request.
#[test]
fn a_tool_call_id_on_one_leg_only_is_lost() {
    let body = json!({ "choices": [{ "message": { "tool_calls": [
        { "id": "call_7", "function": { "name": "f", "arguments": "{}" } },
    ]}}]})
    .to_string();
    let half = json!({ "messages": [
        { "role": "assistant", "tool_calls": [{ "function": { "name": "f", "arguments": "{}" } }] },
        { "role": "tool", "tool_call_id": "call_7", "content": "ok" },
    ]});
    assert_eq!(
        kinds(&lost_tokens(Dialect::ChatCompletions, &body, &half)),
        [("tool_call_id", "call_7")]
    );
    let whole = json!({ "messages": [
        { "role": "assistant", "tool_calls": [{ "id": "call_7", "function": { "name": "f", "arguments": "{}" } }] },
        { "role": "tool", "tool_call_id": "call_7", "content": "ok" },
    ]});
    assert!(lost_tokens(Dialect::ChatCompletions, &body, &whole).is_empty());
}

/// A value in the wrong slot is lost: a signature carried as text is not a
/// signature carried back.
#[test]
fn a_value_outside_its_slot_is_lost() {
    let body =
        json!({ "content": [{ "type": "thinking", "thinking": "t", "signature": "sig-9" }] })
            .to_string();
    let misplaced = json!({ "messages": [{ "role": "assistant", "content": [
        { "type": "text", "text": "sig-9" },
    ]}]});
    assert_eq!(
        kinds(&lost_tokens(Dialect::AnthropicMessages, &body, &misplaced)),
        [("signature", "sig-9")]
    );
}

/// Legacy placeholders are counted apart, never passed or failed.
#[test]
fn legacy_placeholders_are_neither_lost_nor_proven() {
    let body = json!({ "content": [{ "type": "tool_use", "id": "toolu_REDACTED_1", "name": "f", "input": {} }] })
        .to_string();
    let next = json!({ "messages": [] });
    assert!(lost_tokens(Dialect::AnthropicMessages, &body, &next).is_empty());
    assert_eq!(
        kinds(&legacy_tokens(Dialect::AnthropicMessages, &body)),
        [("tool_call_id", "toolu_REDACTED_1")]
    );
}

#[test]
fn a_stored_chain_answers_a_server_held_call_only_before_its_own_turns() {
    let stored = json!({
        "previous_response_id": "resp_1",
        "input": [{ "type": "function_call_output", "call_id": "call_1", "output": "ok" }]
    });
    assert!(unpaired_tool_calls(Dialect::OpenAiResponses, &stored).is_empty());

    let stateless = json!({
        "input": [{ "type": "function_call_output", "call_id": "call_1", "output": "ok" }]
    });
    assert_eq!(
        unpaired_tool_calls(Dialect::OpenAiResponses, &stateless).len(),
        1
    );

    let late = json!({
        "previous_response_id": "resp_1",
        "input": [
            { "type": "function_call", "call_id": "call_2", "name": "f", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "call_2", "output": "ok" },
            { "type": "function_call_output", "call_id": "call_1", "output": "ok" }
        ]
    });
    assert_eq!(
        unpaired_tool_calls(Dialect::OpenAiResponses, &late).len(),
        1
    );
}

#[test]
fn swapped_parallel_call_ids_are_lost() {
    let body = json!({ "content": [
        { "type": "tool_use", "id": "toolu_a", "name": "lookup", "input": { "record": "alpha" } },
        { "type": "tool_use", "id": "toolu_b", "name": "lookup", "input": { "record": "beta" } },
    ]})
    .to_string();
    let next = |first: &str, second: &str| {
        json!({ "messages": [
            { "role": "assistant", "content": [
                { "type": "tool_use", "id": first, "name": "lookup", "input": { "record": "alpha" } },
                { "type": "tool_use", "id": second, "name": "lookup", "input": { "record": "beta" } },
            ]},
            { "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "toolu_a", "content": "a" },
                { "type": "tool_result", "tool_use_id": "toolu_b", "content": "b" },
            ]},
        ]})
    };
    assert!(
        lost_tokens(
            Dialect::AnthropicMessages,
            &body,
            &next("toolu_a", "toolu_b")
        )
        .is_empty()
    );
    assert_eq!(
        kinds(&lost_tokens(
            Dialect::AnthropicMessages,
            &body,
            &next("toolu_b", "toolu_a")
        )),
        [("tool_call_id", "toolu_a"), ("tool_call_id", "toolu_b")]
    );
}

#[test]
fn a_signature_moved_to_another_thinking_block_is_lost() {
    let body = json!({ "content": [
        { "type": "thinking", "thinking": "first", "signature": "sig-1" },
        { "type": "thinking", "thinking": "second", "signature": "sig-2" },
    ]})
    .to_string();
    let next = json!({ "messages": [
        { "role": "assistant", "content": [
            { "type": "thinking", "thinking": "first", "signature": "sig-2" },
            { "type": "thinking", "thinking": "second", "signature": "sig-1" },
        ]},
    ]});
    assert_eq!(
        kinds(&lost_tokens(Dialect::AnthropicMessages, &body, &next)),
        [("signature", "sig-1"), ("signature", "sig-2")]
    );
}

#[test]
fn a_call_signature_moved_to_a_text_part_is_lost() {
    let body = json!({ "candidates": [{ "content": { "role": "model", "parts": [
        { "functionCall": { "name": "lookup", "args": {} }, "thoughtSignature": "sig-call" },
    ]}}]})
    .to_string();
    let kept = json!({ "contents": [{ "role": "model", "parts": [
        { "functionCall": { "name": "renamed", "args": {} }, "thoughtSignature": "sig-call" },
    ]}]});
    assert!(lost_tokens(Dialect::GeminiGenerateContent, &body, &kept).is_empty());
    let moved = json!({ "contents": [{ "role": "model", "parts": [
        { "text": "", "thoughtSignature": "sig-call" },
        { "functionCall": { "name": "lookup", "args": {} } },
    ]}]});
    assert_eq!(
        kinds(&lost_tokens(Dialect::GeminiGenerateContent, &body, &moved)),
        [("thought_signature", "sig-call")]
    );
}

#[test]
fn openrouter_encrypted_details_must_return_on_their_own_item() {
    let body = json!({ "choices": [{ "message": { "role": "assistant", "reasoning_details": [
        { "type": "reasoning.encrypted", "id": "rd_1", "data": "cipher-1" },
    ]}}]})
    .to_string();
    assert_eq!(
        kinds(&response_tokens(Dialect::ChatCompletions, &body)),
        [("encrypted_content", "cipher-1")]
    );
    let next = |id: &str| {
        json!({ "messages": [{ "role": "assistant", "reasoning_details": [
            { "type": "reasoning.encrypted", "id": id, "data": "cipher-1" },
        ]}]})
    };
    assert!(lost_tokens(Dialect::ChatCompletions, &body, &next("rd_1")).is_empty());
    assert_eq!(
        kinds(&lost_tokens(Dialect::ChatCompletions, &body, &next("rd_2"))),
        [("encrypted_content", "cipher-1")]
    );
}

#[test]
fn a_responses_reasoning_signature_must_return_on_its_own_item() {
    let body = json!({ "object": "response", "output": [
        { "type": "reasoning", "id": "rs_1", "summary": [], "signature": "claude-sig",
          "content": [{ "type": "reasoning_text", "text": "t" }] },
    ]})
    .to_string();
    assert_eq!(
        kinds(&response_tokens(Dialect::OpenAiResponses, &body)),
        [("reasoning_id", "rs_1"), ("signature", "claude-sig")]
    );
    let next = |id: &str| json!({ "input": [{ "type": "reasoning", "id": id, "summary": [], "signature": "claude-sig" }] });
    assert!(lost_tokens(Dialect::OpenAiResponses, &body, &next("rs_1")).is_empty());
    assert_eq!(
        kinds(&lost_tokens(Dialect::OpenAiResponses, &body, &next("rs_2"))),
        [("reasoning_id", "rs_1"), ("signature", "claude-sig")]
    );
}

#[test]
fn a_gateway_model_switch_changes_the_reasoning_family() {
    let request = |model: &str| json!({ "model": model });
    let served = |model: &str| json!({ "model": model }).to_string();
    let switches = |earlier: &str, reply: &str, later: &str| {
        switches_reasoning_family(&request(earlier), reply, &request(later))
    };
    assert!(switches(
        "anthropic/claude-haiku-4.5",
        "",
        "google/gemini-3-flash-preview"
    ));
    assert!(!switches(
        "anthropic/claude-haiku-4.5",
        "",
        "anthropic/claude-sonnet-4.6"
    ));
    assert!(!switches("gpt-5-mini", "", "gpt-5.2"));
    assert!(!switches_reasoning_family(
        &json!({}),
        "",
        &request("openai/gpt-5-mini")
    ));
    for router in ["openrouter/auto", "@preset/fast"] {
        // A later router names no family and replays every one.
        assert!(!switches("anthropic/claude-haiku-4.5", "", router));
        // An earlier router's reasoning belongs to the model that answered.
        assert!(switches(
            router,
            &served("anthropic/claude-haiku-4.5"),
            "google/gemini-3-flash-preview"
        ));
        assert!(!switches(
            router,
            &served("google/gemini-3-flash-preview"),
            "google/gemini-3-flash-preview"
        ));
        assert!(!switches(router, "", "google/gemini-3-flash-preview"));
    }
}
