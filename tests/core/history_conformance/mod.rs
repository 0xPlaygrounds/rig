//! The history conformance suites of the wires whose crates this binary
//! links: one module per wire, each expanding
//! `rig_core::history_conformance_suite!`. The registry
//! (`history_conformance_registry.rs`) reads [`SUITE_WIRES`], built from the
//! `HISTORY_WIRE` constant each expanded suite emits, so disabling a suite
//! is a compile error rather than a shrinking test count.

pub mod anthropic;
pub mod anthropic_moonshot;
pub mod chat;
pub mod chatgpt;
pub mod cohere;
pub mod copilot;
pub mod deepseek;
pub mod gemini_interactions;
pub mod gemini_rest;
pub mod groq;
pub mod mistral;
pub mod mock;
pub mod ollama;
pub mod openai_chat;
pub mod openai_responses;
pub mod openrouter;
pub mod perplexity;
pub mod xai;

/// The wires whose suites compiled into this binary.
pub const SUITE_WIRES: &[&str] = &[
    mock::HISTORY_WIRE,
    anthropic::HISTORY_WIRE,
    anthropic_moonshot::HISTORY_WIRE,
    chat::HISTORY_WIRE,
    chatgpt::HISTORY_WIRE,
    cohere::HISTORY_WIRE,
    copilot::HISTORY_WIRE,
    deepseek::HISTORY_WIRE,
    gemini_interactions::HISTORY_WIRE,
    gemini_rest::HISTORY_WIRE,
    groq::HISTORY_WIRE,
    mistral::HISTORY_WIRE,
    ollama::HISTORY_WIRE,
    openai_chat::HISTORY_WIRE,
    openai_responses::HISTORY_WIRE,
    openrouter::HISTORY_WIRE,
    perplexity::HISTORY_WIRE,
    xai::HISTORY_WIRE,
];
