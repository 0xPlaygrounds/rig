//! OpenRouter: reasoning with its structured details, generated images,
//! and a tool call. A generated image is a block, and never goes back.

use rig_core::providers::openai::wire::OPENROUTER;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call, call_deltas};

fn encrypted() -> Value {
    json!({"type": "reasoning.encrypted", "data": "sig", "id": "rs_1",
        "format": "openai-responses-v1", "index": 0})
}

fn image() -> Value {
    json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}, "index": 0})
}

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": "looking it up",
        "reasoning": "plan the lookup",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "plan the lookup", "format": "unknown", "index": 0},
            encrypted(),
        ],
        "images": [image()],
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut deltas = vec![
        json!({"role": "assistant", "content": "", "reasoning": "plan ",
            "reasoning_details": [{"type": "reasoning.text", "text": "plan ", "format": "unknown", "index": 0}]}),
        json!({"reasoning": "the lookup",
            "reasoning_details": [{"type": "reasoning.text", "text": "the lookup", "index": 0}]}),
        json!({"reasoning_details": [encrypted()]}),
        json!({"content": "looking it up"}),
        json!({"images": [image()]}),
    ];
    deltas.extend(call_deltas());
    deltas
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &OPENROUTER,
    model: "google/gemini-2.5-flash-image",
    other_model: "anthropic/claude-sonnet-4.5",
    text_only_model: None,
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_history_conformance::history_conformance_suite! {
    wire: "openrouter",
    fixture: FIXTURE,
}
