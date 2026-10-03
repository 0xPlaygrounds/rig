//! The history conformance suites of the wires whose crates this binary
//! links: one module per wire, each expanding
//! `rig_core::history_conformance_suite!`. The registry
//! (`history_conformance_registry.rs`) reads [`SUITE_WIRES`], built from the
//! `HISTORY_WIRE` constant each expanded suite emits, so disabling a suite
//! is a compile error rather than a shrinking test count.

pub mod anthropic;
pub mod anthropic_moonshot;
pub mod azure;
pub mod chat;
pub mod chatgpt;
pub mod cohere;
pub mod copilot;
pub mod deepseek;
pub mod doubleword;
pub mod gemini_interactions;
pub mod gemini_rest;
pub mod groq;
pub mod huggingface;
pub mod hyperbolic;
pub mod llamacpp;
pub mod minimax;
pub mod mira;
pub mod mistral;
pub mod mock;
pub mod moonshot;
pub mod ollama;
pub mod openai_chat;
pub mod openai_responses;
pub mod openrouter;
pub mod perplexity;
pub mod together;
pub mod venice;
pub mod xai;
pub mod xiaomimimo;
pub mod zai;

/// The wires whose suites compiled into this binary.
pub const SUITE_WIRES: &[&str] = &[
    mock::HISTORY_WIRE,
    anthropic::HISTORY_WIRE,
    anthropic_moonshot::HISTORY_WIRE,
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
    azure::HISTORY_WIRE,
    hyperbolic::HISTORY_WIRE,
    mira::HISTORY_WIRE,
    together::HISTORY_WIRE,
    huggingface::HISTORY_WIRE,
    llamacpp::HISTORY_WIRE,
    venice::HISTORY_WIRE,
    doubleword::HISTORY_WIRE,
    zai::HISTORY_WIRE,
    minimax::HISTORY_WIRE,
    moonshot::HISTORY_WIRE,
    xiaomimimo::HISTORY_WIRE,
];
