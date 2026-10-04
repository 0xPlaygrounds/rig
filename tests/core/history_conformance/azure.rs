//! Azure OpenAI's Chat Completions: text with its annotations and a tool
//! call, by OpenAI's model rules.

use rig_core::providers::azure;
use rig_core::providers::openai::wire::AZURE;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call, call_deltas};

fn annotations() -> Value {
    json!([{"type": "url_citation",
        "url_citation": {"url": "https://rig.rs", "start_index": 0, "end_index": 7}}])
}

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": "looking it up",
        "annotations": annotations(),
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut deltas = vec![
        json!({"role": "assistant", "content": "looking "}),
        json!({"content": "it up"}),
        json!({"annotations": annotations()}),
    ];
    deltas.extend(call_deltas());
    deltas
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &AZURE,
    model: azure::GPT_4O,
    other_model: azure::GPT_4O_MINI,
    text_only_model: Some(azure::GPT_4),
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "azure",
    fixture: FIXTURE,
}
