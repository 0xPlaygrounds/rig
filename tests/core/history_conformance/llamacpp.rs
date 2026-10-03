//! A local `llama-server`: reasoning under `reasoning_content`, text, and
//! a tool call. It reads images in tool results.

use rig_core::providers::openai::wire::LLAMACPP;
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
    dialect: &LLAMACPP,
    model: "Qwen3-VL-2B-Instruct-Q8_0",
    other_model: "Qwen3-4B-Q8_0",
    text_only_model: None,
    rich,
    rich_deltas,
    interleaved: Some(interleaved),
    has_items: true,
};

rig_core::history_conformance_suite! {
    wire: "llamacpp",
    fixture: FIXTURE,
}
