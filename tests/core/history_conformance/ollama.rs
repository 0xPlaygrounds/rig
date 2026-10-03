//! Ollama's OpenAI-compatible API: reasoning under `reasoning`, text, and
//! tool calls it streams whole.

use rig_core::providers::openai::wire::OLLAMA;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call, interleaved_under};

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": "looking it up",
        "reasoning": "plan the lookup",
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut whole = call();
    whole["index"] = json!(0);
    vec![
        json!({"role": "assistant", "content": "", "reasoning": "plan "}),
        json!({"role": "assistant", "content": "", "reasoning": "the lookup"}),
        json!({"role": "assistant", "content": "looking it up"}),
        json!({"role": "assistant", "content": "", "tool_calls": [whole]}),
    ]
}

fn interleaved() -> Vec<Value> {
    interleaved_under("reasoning")
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &OLLAMA,
    model: "qwen3:4b",
    other_model: "llama3.2",
    text_only_model: None,
    rich,
    rich_deltas,
    interleaved: Some(interleaved),
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "ollama",
    fixture: FIXTURE,
}
