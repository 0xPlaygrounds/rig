//! Built here, not recorded: the recorded pairs in
//! `test_utils::raw_parity` cover text, thinking, tool use and a stop
//! sequence on Anthropic. No recording streams citations, server tools, a
//! `fallback` block, a container, a gateway's partial events, a failed or
//! cut reply, or any Messages dialect.

use serde_json::{Value, json};

use super::Message;
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::providers::anthropic::extension::Anthropic;
use crate::providers::anthropic::wire::{
    ANTHROPIC, AnthropicConfig, Dialect, MINIMAX, MOONSHOT, Messages, XIAOMIMIMO, ZAI,
};
use crate::wire::document::Reassemble;
use crate::wire::{Mode, WireFrame};

fn frame(event: &Value) -> WireFrame {
    WireFrame::Text(event.to_string())
}

/// The document `events` rebuild.
fn rebuilt(events: &[Value]) -> Value {
    let mut message = Message::default();
    for event in events {
        message.absorb(&frame(event));
    }
    message.finish()
}

fn start(index: usize, block: Value) -> Value {
    json!({"type": "content_block_start", "index": index, "content_block": block})
}

fn delta(index: usize, delta: Value) -> Value {
    json!({"type": "content_block_delta", "index": index, "delta": delta})
}

fn stop(index: usize) -> Value {
    json!({"type": "content_block_stop", "index": index})
}

fn message_start(model: &str, usage: Value) -> Value {
    json!({"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": null, "stop_sequence": null, "usage": usage
    }})
}

/// A turn that thinks, answers with a citation and calls a tool, streamed.
fn thinking_text_and_call(model: &str) -> Vec<Value> {
    let citation = json!({
        "type": "char_location", "cited_text": "Paris", "document_index": 0,
        "start_char_index": 0, "end_char_index": 5
    });
    vec![
        message_start(
            model,
            json!({"input_tokens": 20, "output_tokens": 1, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0, "service_tier": "standard"}),
        ),
        start(
            0,
            json!({"type": "thinking", "thinking": "", "signature": ""}),
        ),
        json!({"type": "ping"}),
        delta(0, json!({"type": "thinking_delta", "thinking": "Let me "})),
        delta(0, json!({"type": "thinking_delta", "thinking": "think."})),
        delta(0, json!({"type": "signature_delta", "signature": "sig"})),
        stop(0),
        start(1, json!({"type": "text", "text": ""})),
        delta(1, json!({"type": "text_delta", "text": "Paris"})),
        delta(1, json!({"type": "citations_delta", "citation": citation})),
        delta(1, json!({"type": "text_delta", "text": " it is."})),
        stop(1),
        start(
            2,
            json!({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}),
        ),
        delta(2, json!({"type": "input_json_delta", "partial_json": ""})),
        delta(
            2,
            json!({"type": "input_json_delta", "partial_json": "{\"q\": "}),
        ),
        delta(
            2,
            json!({"type": "input_json_delta", "partial_json": "\"x\"}"}),
        ),
        stop(2),
        json!({"type": "message_delta",
               "delta": {"stop_reason": "tool_use", "stop_sequence": null},
               "usage": {"output_tokens": 42}}),
        json!({"type": "message_stop"}),
    ]
}

/// The unary body of the turn [`thinking_text_and_call`] streams.
fn thinking_text_and_call_unary(model: &str) -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
        "content": [
            {"type": "thinking", "thinking": "Let me think.", "signature": "sig"},
            {"type": "text", "text": "Paris it is.", "citations": [{
                "type": "char_location", "cited_text": "Paris", "document_index": 0,
                "start_char_index": 0, "end_char_index": 5
            }]},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}
        ],
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 20, "output_tokens": 42, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0, "service_tier": "standard"}
    })
}

#[test]
fn a_stream_rebuilds_the_unary_message() {
    let model = "claude-sonnet-4-6";
    assert_eq!(
        rebuilt(&thinking_text_and_call(model)),
        thinking_text_and_call_unary(model)
    );
}

/// The reply `wire` folds from `frames` in `mode`, its `raw` the
/// reassembler's when `raw` is `Null`.
fn decoded(wire: &Messages, mode: Mode, frames: Vec<WireFrame>, raw: Value) -> CompletionResponse {
    crate::test_utils::decode_reply(wire, &CompletionRequest::new("hi"), mode, frames, raw)
        .expect("the reply decodes")
}

/// `raw` of the unary reply `unary` and of the stream `events` on `wire`.
fn both_ways(wire: &Messages, unary: &Value, events: &[Value]) -> (Value, Value) {
    let unary = decoded(
        wire,
        Mode::Unary,
        vec![WireFrame::Text(unary.to_string())],
        unary.clone(),
    );
    let streamed = decoded(
        wire,
        Mode::Streaming,
        events.iter().map(frame).collect(),
        Value::Null,
    );
    (unary.raw, streamed.raw)
}

/// Every Messages dialect shares the reassembler, so a stream's `raw` on
/// each is its unary document.
#[test]
fn every_dialect_streams_its_unary_document() {
    let dialects: [(&'static Dialect, &str); 5] = [
        (&ANTHROPIC, "claude-sonnet-4-6"),
        (&ZAI, "glm-5"),
        (&MINIMAX, "MiniMax-M2.7"),
        (&MOONSHOT, "kimi-k3"),
        (&XIAOMIMIMO, "mimo-v2.5"),
    ];
    for (dialect, model) in dialects {
        let wire = AnthropicConfig::with_key(dialect, "sk-test").completion(model);
        let (unary, streamed) = both_ways(
            &wire,
            &thinking_text_and_call_unary(model),
            &thinking_text_and_call(model),
        );
        assert_eq!(streamed, unary, "{}", dialect.name);
    }
}

/// A leading `fallback`, a server tool call with its streamed input, the
/// server tool's result, a cited answer, the container and the usage the
/// stream ends with.
fn server_tools() -> Vec<Value> {
    let location = json!({
        "type": "web_search_result_location", "url": "https://www.rust-lang.org",
        "title": "Rust", "encrypted_index": "idx", "cited_text": "Rust"
    });
    vec![
        message_start(
            "claude-opus-5-5",
            json!({"input_tokens": 5, "output_tokens": 1, "service_tier": "standard",
                   "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 0}}),
        ),
        start(
            0,
            json!({"type": "fallback", "from": {"model": "claude-opus-5-5"},
                   "to": {"model": "claude-opus-4-8"}}),
        ),
        stop(0),
        start(
            1,
            json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}),
        ),
        delta(
            1,
            json!({"type": "input_json_delta", "partial_json": "{\"query\": \"rust\"}"}),
        ),
        stop(1),
        start(
            2,
            json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
                {"type": "web_search_result", "url": "https://www.rust-lang.org",
                 "title": "Rust", "encrypted_content": "enc"}
            ]}),
        ),
        stop(2),
        start(3, json!({"type": "text", "text": ""})),
        delta(3, json!({"type": "citations_delta", "citation": location})),
        delta(3, json!({"type": "text_delta", "text": "Rust."})),
        stop(3),
        json!({"type": "message_delta",
               "delta": {"stop_reason": "end_turn", "stop_sequence": null, "stop_details": null,
                         "container": {"id": "cont_1", "expires_at": "2026-10-06T00:00:00Z"}},
               "usage": {"output_tokens": 9, "speed": "fast",
                         "server_tool_use": {"web_search_requests": 1}}}),
        json!({"type": "message_stop"}),
    ]
}

fn server_tools_unary() -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [
            {"type": "fallback", "from": {"model": "claude-opus-5-5"},
             "to": {"model": "claude-opus-4-8"}},
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
             "input": {"query": "rust"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
                {"type": "web_search_result", "url": "https://www.rust-lang.org",
                 "title": "Rust", "encrypted_content": "enc"}
            ]},
            {"type": "text", "text": "Rust.", "citations": [{
                "type": "web_search_result_location", "url": "https://www.rust-lang.org",
                "title": "Rust", "encrypted_index": "idx", "cited_text": "Rust"
            }]}
        ],
        "stop_reason": "end_turn", "stop_sequence": null, "stop_details": null,
        "container": {"id": "cont_1", "expires_at": "2026-10-06T00:00:00Z"},
        "usage": {"input_tokens": 5, "output_tokens": 9, "service_tier": "standard",
                  "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 0},
                  "speed": "fast", "server_tool_use": {"web_search_requests": 1}}
    })
}

/// Server-tool blocks and a `fallback` arrive whole and are kept, and every
/// field `AnthropicExtras` reads is read the same from both replies.
#[test]
fn server_tools_a_fallback_and_the_container_read_the_same_both_ways() {
    let wire = AnthropicConfig::new("sk-test").completion("claude-opus-5-5");
    assert_eq!(rebuilt(&server_tools()), server_tools_unary());
    let unary = server_tools_unary();
    let extras = |mode, frames, raw| {
        decoded(&wire, mode, frames, raw)
            .extras::<Anthropic>()
            .expect("an Anthropic reply")
            .expect("the extras read")
    };
    let from_body = extras(
        Mode::Unary,
        vec![WireFrame::Text(unary.to_string())],
        unary.clone(),
    );
    let from_stream = extras(
        Mode::Streaming,
        server_tools().iter().map(frame).collect(),
        Value::Null,
    );
    assert_eq!(from_stream, from_body);
    assert_eq!(
        from_stream.fallback_model.as_deref(),
        Some("claude-opus-4-8")
    );
    assert_eq!(from_stream.speed.as_deref(), Some("fast"));
    assert_eq!(from_stream.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(
        from_stream.container.map(|container| container.id),
        Some("cont_1".to_owned())
    );
    assert_eq!(
        from_stream
            .server_tool_use
            .map(|used| used.web_search_requests),
        Some(1)
    );
    assert_eq!(
        from_stream
            .cache_creation
            .map(|cache| cache.ephemeral_5m_input_tokens),
        Some(2)
    );
}

/// `message_delta`'s counters fold over `message_start`'s: a `null` does
/// not erase a count, and a zero input count (a gateway leaving it out)
/// does not erase the started one.
#[test]
fn the_terminal_usage_folds_over_the_start_usage() {
    let split = json!({"ephemeral_1h_input_tokens": 3, "ephemeral_5m_input_tokens": 1});
    let document = rebuilt(&[
        message_start(
            "claude-sonnet-4-6",
            json!({"input_tokens": 10, "output_tokens": 0, "cache_creation_input_tokens": 4,
                   "cache_read_input_tokens": 6, "cache_creation": split}),
        ),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
               "usage": {"input_tokens": 0, "output_tokens": 3, "cache_read_input_tokens": null}}),
    ]);
    assert_eq!(
        document["usage"],
        json!({"input_tokens": 10, "output_tokens": 3, "cache_creation_input_tokens": 4,
               "cache_read_input_tokens": 6, "cache_creation": split})
    );
    assert_eq!(document["stop_reason"], "end_turn");
}

/// A reply that fails in band, or is cut short, keeps the message so far;
/// a call cut short keeps its streamed input only when it parses.
#[test]
fn a_failed_or_cut_reply_keeps_the_message_so_far() {
    let opened = [
        message_start("claude-sonnet-4-6", json!({"input_tokens": 3})),
        start(0, json!({"type": "text", "text": ""})),
        delta(0, json!({"type": "text_delta", "text": "Par"})),
    ];
    let mut failed = opened.to_vec();
    failed.push(json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}));
    let document = rebuilt(&failed);
    assert_eq!(
        document["content"],
        json!([{"type": "text", "text": "Par"}])
    );
    assert_eq!(document["stop_reason"], Value::Null);
    assert!(document.get("error").is_none(), "{document}");

    let call = |input: &str| {
        rebuilt(&[
            message_start("claude-sonnet-4-6", json!({"input_tokens": 3})),
            start(
                0,
                json!({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}),
            ),
            delta(
                0,
                json!({"type": "input_json_delta", "partial_json": input}),
            ),
        ])
    };
    assert_eq!(
        call("{\"q\": \"x\"}")["content"][0]["input"],
        json!({"q": "x"})
    );
    assert_eq!(call("{\"q\": ")["content"][0]["input"], json!({}));

    assert_eq!(rebuilt(&[]), Value::Null);
    assert_eq!(rebuilt(&[json!({"type": "ping"})]), Value::Null);
}

/// A whole `message` frame is the document as sent.
#[test]
fn a_whole_message_frame_is_the_document() {
    let model = "claude-sonnet-4-6";
    let mut whole = thinking_text_and_call_unary(model);
    if let Some(whole) = whole.as_object_mut() {
        whole.insert("unknown_field".to_owned(), json!(true));
    }
    assert_eq!(rebuilt(&[whole.clone()]), whole);
}

/// A gateway that skips `content_block_start` and `message_start` still
/// rebuilds the text it streamed, as the decoder reads it.
#[test]
fn a_gateway_without_block_starts_still_rebuilds_its_text() {
    let document = rebuilt(&[
        delta(0, json!({"type": "thinking_delta", "thinking": "hm"})),
        delta(1, json!({"type": "text_delta", "text": "hi"})),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
               "usage": {"output_tokens": 1}}),
    ]);
    assert_eq!(
        document,
        json!({
            "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"output_tokens": 1}
        })
    );
}
