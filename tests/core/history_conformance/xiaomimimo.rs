//! Xiaomi MiMo's Chat half: reasoning under `reasoning_content`, text,
//! and a tool call. MiMo V2 Flash and Pro read no images.

use rig_core::providers::openai::wire::XIAOMIMIMO;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call, call_deltas, interleaved_under};

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

fn interleaved() -> Vec<Value> {
    interleaved_under("reasoning_content")
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &XIAOMIMIMO,
    model: rig_core::providers::xiaomimimo::MIMO_V2_5,
    other_model: rig_core::providers::xiaomimimo::MIMO_V2_OMNI,
    text_only_model: Some(rig_core::providers::xiaomimimo::MIMO_V2_FLASH),
    rich,
    rich_deltas,
    interleaved: Some(interleaved),
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "xiaomimimo",
    fixture: FIXTURE,
}
