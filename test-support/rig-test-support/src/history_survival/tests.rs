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
