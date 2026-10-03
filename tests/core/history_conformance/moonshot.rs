//! Moonshot's Chat half: Kimi reasoning under `reasoning_content`, text,
//! and a tool call. Kimi K2 before K2.5 reads no images.

use rig_core::providers::openai::wire::MOONSHOT;
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
    dialect: &MOONSHOT,
    model: rig_core::providers::moonshot::KIMI_K2_6,
    other_model: rig_core::providers::moonshot::KIMI_K3,
    text_only_model: Some("kimi-k2-0905-preview"),
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "moonshot",
    fixture: FIXTURE,
}
