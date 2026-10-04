//! Every recorded whole Messages reply, restated as the stream Anthropic
//! would have sent for it, folds into the same assistant turn: the same
//! blocks, provider items, origin and stop. One decoder serves both modes,
//! so a difference is a decoder defect, not a recording one.

use rig::providers::anthropic::{self, completion::CLAUDE_SONNET_4_6};
use rig::wire::WireFrame;
use rig_core::test_utils::history::assert_restated_agrees;
use serde_json::{Value, json};

/// `block` as its `content_block_start` states it, and the deltas that
/// grow it back to what the whole reply holds.
fn restate_block(block: &Value) -> (Value, Vec<Value>) {
    let mut start = block.clone();
    let mut deltas = Vec::new();
    match block["type"].as_str() {
        Some("text") => {
            start["text"] = json!("");
            if let Some(citations) = block["citations"].as_array() {
                start["citations"] = json!([]);
                deltas.extend(
                    citations
                        .iter()
                        .map(|citation| json!({"type": "citations_delta", "citation": citation})),
                );
            }
            deltas.push(json!({"type": "text_delta", "text": block["text"]}));
        }
        Some("thinking") => {
            start["thinking"] = json!("");
            start["signature"] = json!("");
            deltas.push(json!({"type": "thinking_delta", "thinking": block["thinking"]}));
            deltas.push(json!({"type": "signature_delta", "signature": block["signature"]}));
        }
        Some("tool_use" | "server_tool_use" | "mcp_tool_use") => {
            start["input"] = json!({});
            if block["input"] != json!({}) {
                deltas.push(json!({
                    "type": "input_json_delta",
                    "partial_json": block["input"].to_string(),
                }));
            }
        }
        _ => {}
    }
    (start, deltas)
}

/// The stream of events Anthropic sends for the whole reply `message`.
fn restate(message: &Value) -> Vec<WireFrame> {
    let mut start = message.clone();
    start["content"] = json!([]);
    start["stop_reason"] = Value::Null;
    start["stop_sequence"] = Value::Null;
    let mut frames = vec![json!({"type": "message_start", "message": start})];
    for (index, block) in message["content"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let (start, deltas) = restate_block(block);
        frames.push(json!({"type": "content_block_start", "index": index, "content_block": start}));
        frames.extend(
            deltas.into_iter().map(
                |delta| json!({"type": "content_block_delta", "index": index, "delta": delta}),
            ),
        );
        frames.push(json!({"type": "content_block_stop", "index": index}));
    }
    frames.push(json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": message["stop_reason"],
            "stop_sequence": message["stop_sequence"],
            "stop_details": message.get("stop_details").cloned().unwrap_or(Value::Null),
            "container": message.get("container").cloned().unwrap_or(Value::Null),
        },
        "usage": message["usage"],
    }));
    frames.push(json!({"type": "message_stop"}));
    frames
        .into_iter()
        .map(|frame| WireFrame::Text(frame.to_string()))
        .collect()
}

/// The scenario names of every recorded Anthropic fixture.
fn scenarios() -> Vec<String> {
    let root = crate::cassettes::cassette_root().join("anthropic");
    let mut pending = vec![root.clone()];
    let mut scenarios = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("the fixture tree is readable") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "yaml")
                && let Ok(relative) = path.with_extension("").strip_prefix(&root)
            {
                scenarios.push(relative.to_string_lossy().into_owned());
            }
        }
    }
    scenarios.sort();
    scenarios
}

#[test]
fn every_recorded_whole_reply_restates_as_the_same_turn() {
    let wire = anthropic::Anthropic::new("unused")
        .completion(CLAUDE_SONNET_4_6)
        .wire;
    let mut restated = 0;
    for scenario in scenarios() {
        for (status, body) in crate::cassettes::recorded_statuses_and_bodies("anthropic", &scenario)
        {
            let Ok(message) = serde_json::from_str::<Value>(&body) else {
                continue;
            };
            if status != 200 || message["type"] != "message" {
                continue;
            }
            assert_restated_agrees(&wire, [WireFrame::Text(body.clone())], restate(&message));
            restated += 1;
        }
    }
    assert!(
        restated > 100,
        "premise: the corpus holds whole replies, restated {restated}"
    );
}
