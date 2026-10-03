//! Together AI: text and a tool call.

use rig_core::providers::openai::wire::TOGETHER;
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
    dialect: &TOGETHER,
    model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    other_model: "Qwen/Qwen2.5-72B-Instruct-Turbo",
    text_only_model: None,
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_core::history_conformance_suite! {
    wire: "together",
    fixture: FIXTURE,
}
