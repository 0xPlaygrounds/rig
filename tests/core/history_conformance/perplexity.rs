//! Perplexity: text with the citations and search results its replies
//! carry beside the message. It takes no tools, so calls and results reach
//! it as text.

use rig_core::providers::openai::wire::PERPLEXITY;
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
    dialect: &PERPLEXITY,
    model: "sonar-pro",
    other_model: "sonar",
    text_only_model: None,
    rich,
    rich_deltas,
    interleaved: None,
    has_items: false,
};

rig_history_conformance::history_conformance_suite! {
    wire: "perplexity",
    fixture: FIXTURE,
}
