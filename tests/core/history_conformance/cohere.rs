//! Cohere's OpenAI Compatibility API: reasoning under `reasoning_content`,
//! text, and a tool call streamed in fragments. Only its vision models read
//! images.

use rig_core::providers::openai::wire::COHERE;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call, call_deltas};

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": "looking it up",
        "reasoning_content": "plan the lookup",
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut deltas = vec![
        json!({"role": "assistant", "reasoning_content": "plan "}),
        json!({"reasoning_content": "the lookup"}),
        json!({"content": "looking it up"}),
    ];
    deltas.extend(call_deltas());
    deltas
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &COHERE,
    model: "command-a-vision-07-2025",
    other_model: "command-a-reasoning-08-2025",
    text_only_model: Some("command-a-03-2025"),
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "cohere",
    fixture: FIXTURE,
}
