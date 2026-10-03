//! DeepSeek: reasoning under `reasoning_content`, sent back empty when a
//! turn has none, and calls it streams whole. Its models read no images.

use rig_core::providers::openai::wire::DEEPSEEK;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call, interleaved_under};

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": "looking it up",
        "reasoning_content": "plan the lookup",
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut whole = call();
    whole["index"] = json!(0);
    vec![
        json!({"role": "assistant", "content": null, "reasoning_content": "plan "}),
        json!({"reasoning_content": "the lookup"}),
        json!({"content": "looking it up"}),
        json!({"tool_calls": [whole]}),
    ]
}

fn interleaved() -> Vec<Value> {
    interleaved_under("reasoning_content")
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &DEEPSEEK,
    model: "deepseek-v4-flash",
    other_model: "deepseek-v4-pro",
    text_only_model: None,
    rich,
    rich_deltas,
    interleaved: Some(interleaved),
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "deepseek",
    fixture: FIXTURE,
}
