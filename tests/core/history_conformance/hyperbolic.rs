//! Hyperbolic: text only, for it takes no tools, so calls and results
//! reach it as text.

use rig_core::providers::openai::wire::HYPERBOLIC;
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
    dialect: &HYPERBOLIC,
    model: rig_core::providers::hyperbolic::LLAMA_3_3_70B,
    other_model: rig_core::providers::hyperbolic::QWEN_2_5_72B,
    text_only_model: None,
    rich,
    rich_deltas,
    interleaved: None,
    has_items: false,
};

rig_core::history_conformance_suite! {
    wire: "hyperbolic",
    fixture: FIXTURE,
}
