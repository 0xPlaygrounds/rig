//! An Ollama daemon's Messages route.

use rig_core::providers::anthropic::wire::OLLAMA;

use super::anthropic::MessagesHistory;

pub const OLLAMA_MESSAGES_HISTORY: MessagesHistory = MessagesHistory {
    dialect: &OLLAMA,
    model: "qwen3:4b",
    other_model: "llama3.2",
    text_only_model: None,
    signature: "sig_1",
    hosted: false,
};

rig_history_conformance::history_conformance_suite! {
    wire: "anthropic_ollama",
    fixture: OLLAMA_MESSAGES_HISTORY,
}
