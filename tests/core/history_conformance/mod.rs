//! The history conformance suites of the wires whose crates this binary
//! links: one module per wire, each expanding
//! `rig_core::history_conformance_suite!`. The registry
//! (`history_conformance_registry.rs`) reads [`SUITE_WIRES`], built from the
//! `HISTORY_WIRE` constant each expanded suite emits, so disabling a suite
//! is a compile error rather than a shrinking test count.

pub mod anthropic;
pub mod anthropic_moonshot;
pub mod chatgpt;
pub mod copilot;
pub mod gemini_rest;
pub mod mock;
pub mod openai_responses;
pub mod xai;

/// The wires whose suites compiled into this binary.
pub const SUITE_WIRES: &[&str] = &[
    mock::HISTORY_WIRE,
    anthropic::HISTORY_WIRE,
    anthropic_moonshot::HISTORY_WIRE,
    chatgpt::HISTORY_WIRE,
    copilot::HISTORY_WIRE,
    gemini_rest::HISTORY_WIRE,
    openai_responses::HISTORY_WIRE,
    xai::HISTORY_WIRE,
];
