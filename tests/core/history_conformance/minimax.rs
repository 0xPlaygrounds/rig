//! MiniMax's Chat half: text and a tool call. M2 models read no images.

use rig_core::providers::openai::wire::MINIMAX;
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
    dialect: &MINIMAX,
    model: "MiniMax-M3",
    other_model: rig_core::providers::minimax::MINIMAX_M2_7,
    text_only_model: Some(rig_core::providers::minimax::MINIMAX_M2_7),
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_core::history_conformance_suite! {
    wire: "minimax",
    fixture: FIXTURE,
}
