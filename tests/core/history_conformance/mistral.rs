//! Mistral: Magistral's thinking content parts, text, and calls it streams
//! whole. Thinking goes back as the part it came as.

use rig_core::providers::openai::wire::MISTRAL;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call};

fn thinking(text: &str) -> Value {
    json!({"type": "thinking", "thinking": [{"type": "text", "text": text}]})
}

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": [thinking("plan the lookup"), {"type": "text", "text": "looking it up"}],
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut whole = call();
    whole["index"] = json!(0);
    vec![
        json!({"role": "assistant", "content": [thinking("plan ")]}),
        json!({"content": [thinking("the lookup")]}),
        json!({"content": "looking it up"}),
        json!({"tool_calls": [whole]}),
    ]
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &MISTRAL,
    model: "magistral-medium-latest",
    other_model: "mistral-small-latest",
    text_only_model: Some("codestral-latest"),
    rich,
    rich_deltas,
    has_items: true,
};

rig_core::history_conformance_suite! {
    wire: "mistral",
    fixture: FIXTURE,
}
