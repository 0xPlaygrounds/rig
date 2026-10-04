//! Groq: gpt-oss reasoning under `reasoning`, the `channel` its messages
//! carry (which it refuses back), and calls it streams whole.

use rig_core::providers::openai::wire::GROQ;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call};

fn rich() -> Value {
    json!({
        "role": "assistant",
        "channel": "final",
        "content": "looking it up",
        "reasoning": "plan the lookup",
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut whole = call();
    whole["index"] = json!(0);
    vec![
        json!({"role": "assistant", "channel": "analysis", "reasoning": "plan "}),
        json!({"reasoning": "the lookup"}),
        json!({"channel": "final", "content": "looking it up"}),
        json!({"tool_calls": [whole]}),
    ]
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &GROQ,
    model: "openai/gpt-oss-120b",
    other_model: "llama-3.3-70b-versatile",
    text_only_model: Some("llama-3.3-70b-versatile"),
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "groq",
    fixture: FIXTURE,
}
