//! Provider items on the Messages wire: a block this crate does not model
//! is kept in place, a modelled block keeps what its canonical form loses,
//! both survive whole and streamed replies alike, replay verbatim to this
//! dialect, and follow the staleness and cross-dialect rules.

use serde_json::{Value, json};

use super::completion::CLAUDE_SONNET_4_6;
use super::wire::{AnthropicConfig, Messages};
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::message::{AssistantContent, Issuer, Message, NativeItem, Sealed, ToolName};
use crate::test_utils::{decode_reply, json_body};
use crate::wire::{Mode, Wire, WireFrame};

fn wire() -> Messages {
    AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6)
}

/// A whole reply whose content is `blocks`.
fn whole(blocks: &[Value]) -> Value {
    json!({
        "type": "message", "id": "msg_1", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 5},
        "content": blocks,
    })
}

/// The same reply as a stream states it: each block's start, its payload
/// by delta, and its stop.
fn streamed(blocks: &[Value]) -> Vec<Value> {
    let mut frames = vec![json!({"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": CLAUDE_SONNET_4_6,
        "content": [], "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 0},
    }})];
    for (index, block) in blocks.iter().enumerate() {
        let mut start = block.clone();
        let mut deltas = Vec::new();
        if let Some(text) = block.get("text").and_then(Value::as_str) {
            start["text"] = json!("");
            deltas.push(json!({"type": "text_delta", "text": text}));
        }
        if let Some(citations) = block.get("citations").and_then(Value::as_array) {
            start["citations"] = json!([]);
            for citation in citations {
                deltas.push(json!({"type": "citations_delta", "citation": citation}));
            }
        }
        if let Some(input) = block.get("input") {
            start["input"] = json!({});
            deltas.push(json!({"type": "input_json_delta", "partial_json": input.to_string()}));
        }
        frames.push(json!({"type": "content_block_start", "index": index, "content_block": start}));
        for delta in deltas {
            frames.push(json!({"type": "content_block_delta", "index": index, "delta": delta}));
        }
        frames.push(json!({"type": "content_block_stop", "index": index}));
    }
    frames.push(
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn",
        "stop_sequence": null}, "usage": {"output_tokens": 5}}),
    );
    frames.push(json!({"type": "message_stop"}));
    frames
}

fn decode(mode: Mode, frames: Vec<Value>) -> CompletionResponse {
    let wire = wire();
    let frames = frames
        .into_iter()
        .map(|frame| WireFrame::Text(frame.to_string()));
    decode_reply(
        &wire,
        &CompletionRequest::new("hello").max_tokens(64),
        mode,
        frames,
        Value::Null,
    )
    .expect("the reply decodes")
}

/// The assistant blocks a follow-up request to `wire` replays `content` as.
fn replayed(wire: &Messages, content: Vec<AssistantContent>) -> Vec<Value> {
    let request = CompletionRequest::new("Thanks.")
        .messages([
            Message::user("hello"),
            Message::Assistant { id: None, content },
        ])
        .max_tokens(64);
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the follow-up encodes");
    json_body(&encoded.request)["messages"][1]["content"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// Blocks a provider might add tomorrow, beside modelled ones that carry
/// members this crate does not model.
fn novel_reply() -> Vec<Value> {
    vec![
        json!({"type": "thinking", "thinking": "Let me look.", "signature": "sig_1"}),
        json!({"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup",
               "server_name": "records", "input": {"record": "alpha"}}),
        json!({"type": "hologram_tool_result", "tool_use_id": "mcptoolu_1",
               "content": {"frames": [1, 2, 3], "stderr": ""}}),
        json!({"type": "text", "text": "Alpha is 42.", "citations": [{
            "type": "char_location", "cited_text": "42", "document_index": 0,
            "start_char_index": 0, "end_char_index": 2}]}),
        json!({"type": "tool_use", "id": "toolu_1", "name": "record",
               "input": {"value": 42}, "caller": {"type": "code_execution_20260120",
               "tool_id": "srvtoolu_1"}}),
    ]
}

/// Loss 1, and bar 1 and 4: a block type no version of this crate knows
/// neither fails the reply nor is dropped. Whole and streamed replies of the
/// same content end with the same history, and replaying it to this dialect
/// sends every block back as stated, in order.
#[test]
fn novel_blocks_survive_both_modes_and_replay_verbatim() {
    let blocks = novel_reply();
    let unary = decode(Mode::Unary, vec![whole(&blocks)]);
    let stream = decode(Mode::Streaming, streamed(&blocks));
    assert_eq!(
        unary.choice, stream.choice,
        "both modes keep the same history"
    );

    let kinds: Vec<&str> = unary
        .choice
        .iter()
        .map(|part| match part {
            AssistantContent::Reasoning(_) => "reasoning",
            AssistantContent::Native(_) => "native",
            AssistantContent::Text(_) => "text",
            AssistantContent::ToolCall(_) => "tool_call",
            AssistantContent::Image(_) => "image",
        })
        .collect();
    assert_eq!(
        kinds,
        ["reasoning", "native", "native", "text", "tool_call"]
    );
    // The canonical blocks still say what they say.
    let Some(AssistantContent::ToolCall(call)) = unary.choice.last() else {
        panic!("the client call stays a call");
    };
    assert_eq!(call.function.arguments, json!({"value": 42}));
    assert!(call.native.is_some(), "its `caller` rides it");

    assert_eq!(replayed(&wire(), unary.choice), blocks);
}

/// Bar 1, unknown deltas aside: a block that round-trips through its
/// canonical form keeps no provider item, so ordinary history is unchanged.
#[test]
fn blocks_the_canonical_form_restates_keep_no_item() {
    let blocks = vec![
        json!({"type": "text", "text": "Hi.", "citations": null}),
        json!({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {},
               "caller": {"type": "direct"}}),
    ];
    let reply = decode(Mode::Unary, vec![whole(&blocks)]);
    for part in &reply.choice {
        match part {
            AssistantContent::Text(text) => assert!(text.native.is_none()),
            AssistantContent::ToolCall(call) => assert!(call.native.is_none()),
            other => panic!("unexpected {other:?}"),
        }
    }
}

/// The staleness rule under the agent's block edits: a repaired call is
/// renamed, so its item no longer projects to it and it replays canonically;
/// an ignored call takes its item with it; untouched blocks still replay
/// as stated.
#[test]
fn edited_blocks_replay_canonically() {
    let blocks = novel_reply();
    let mut choice = decode(Mode::Unary, vec![whole(&blocks)]).choice;
    let Some(AssistantContent::ToolCall(call)) = choice.last_mut() else {
        panic!("a call");
    };
    call.function.name = ToolName::new("record_v2").expect("a name");
    let replay = replayed(&wire(), choice.clone());
    assert_eq!(
        replay[..4],
        blocks[..4],
        "untouched blocks replay as stated"
    );
    assert_eq!(
        replay[4],
        json!({"type": "tool_use", "id": "toolu_1", "name": "record_v2", "input": {"value": 42}})
    );

    choice.retain(|part| !matches!(part, AssistantContent::ToolCall(_)));
    assert_eq!(
        replayed(&wire(), choice),
        blocks.iter().take(4).cloned().collect::<Vec<_>>()
    );
}

/// The cross-dialect rule: another dialect's items, even under an issuer
/// this request opens, are not sent; blocks they rode replay canonically.
#[test]
fn another_dialects_items_never_reach_this_wire() {
    let responses_item = |item: Value| {
        Sealed::new(
            Issuer::from("anthropic"),
            NativeItem::new("openai.responses", item),
        )
    };
    let content = vec![
        AssistantContent::Native(responses_item(json!({"type": "compaction", "id": "cmp_1"}))),
        AssistantContent::Text(crate::message::Text {
            native: Some(responses_item(
                json!({"type": "message", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Hi."}]}),
            )),
            ..crate::message::Text::new("Hi.")
        }),
    ];
    assert_eq!(
        replayed(&wire(), content),
        [json!({"type": "text", "text": "Hi."})]
    );
}
