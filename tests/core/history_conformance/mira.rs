//! Mira's gateway: text only, with every message's content one string.
//! It takes no tools and its models read no images.

use rig_core::providers::openai::wire::MIRA;
use serde_json::{Value, json};

use super::chat::ChatHistory;

fn rich() -> Value {
    json!({"role": "assistant", "content": "looking it up"})
}

fn rich_deltas() -> Vec<Value> {
    vec![
        json!({"role": "assistant", "content": "looking "}),
        json!({"content": "it up"}),
    ]
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &MIRA,
    model: "gpt-4o",
    other_model: "claude-3.5-sonnet",
    text_only_model: Some("llama-3.3-70b"),
    rich,
    rich_deltas,
    interleaved: None,
    has_items: false,
};

rig_core::history_conformance_suite! {
    wire: "mira",
    fixture: FIXTURE,
}
