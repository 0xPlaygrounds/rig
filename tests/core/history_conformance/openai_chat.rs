//! OpenAI's own Chat Completions: an answer's audio, whose transcript is
//! the text, and a tool call.

use rig_core::providers::openai::wire::OPENAI;
use serde_json::{Value, json};

use super::chat::{ChatHistory, call, call_deltas};

fn rich() -> Value {
    json!({
        "role": "assistant",
        "content": null,
        "audio": {"id": "audio_1", "transcript": "looking it up", "data": "UklG", "expires_at": 1},
        "tool_calls": [call()],
    })
}

fn rich_deltas() -> Vec<Value> {
    let mut deltas = vec![
        json!({"role": "assistant", "content": null,
            "audio": {"id": "audio_1", "transcript": "looking "}}),
        json!({"audio": {"transcript": "it up", "data": "UklG", "expires_at": 1}}),
    ];
    deltas.extend(call_deltas());
    deltas
}

pub const FIXTURE: ChatHistory = ChatHistory {
    dialect: &OPENAI,
    model: "gpt-4o-audio-preview",
    other_model: "gpt-4.1-nano",
    text_only_model: Some("o3-mini"),
    rich,
    rich_deltas,
    interleaved: None,
    has_items: true,
};

rig_core::history_conformance_suite! {
    wire: "openai_chat",
    fixture: FIXTURE,
}
