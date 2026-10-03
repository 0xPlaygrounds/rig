//! Doubleword: text and a tool call.

use rig_core::providers::openai::wire::DOUBLEWORD;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call, call_deltas};

fn rich() -> Value {
    json!({"role": "assistant", "content": "looking it up", "tool_calls": [call()]})
}

fn rich_deltas() -> Vec<Value> {
    let mut deltas = vec![
        json!({"role": "assistant", "content": "looking "}),
        json!({"content": "it up"}),
    ];
    deltas.extend(call_deltas());
    deltas
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &DOUBLEWORD,
    model: "Qwen/Qwen3.5-397B-A17B-FP8",
    other_model: "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    text_only_model: None,
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "doubleword",
    fixture: FIXTURE,
}
