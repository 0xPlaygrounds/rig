//! The Ollama dialect of the Messages wire's history suite. Ollama sends
//! its thinking unsigned and takes it back so.

use rig_core::providers::anthropic::wire::OLLAMA;
use rig_core::providers::ollama::{GPT_OSS, QWEN3};

use super::anthropic::MessagesHistory;

pub const OLLAMA_MESSAGES_HISTORY: MessagesHistory = MessagesHistory {
    dialect: &OLLAMA,
    model: QWEN3,
    other_model: GPT_OSS,
    text_only_model: None,
    signature: "",
    hosted: false,
};

rig_history_conformance::history_conformance_suite! {
    wire: "anthropic_ollama",
    fixture: OLLAMA_MESSAGES_HISTORY,
}
