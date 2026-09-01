//! Macro-expanded wire-conformance suites for the byte-transport wire
//! families whose fixtures live in
//! `rig_core::test_utils::streaming_conformance::fixtures`.
//!
//! Each invocation expands the full canonical scenario set plus the
//! anti-tamper tests (`suite_is_complete`,
//! `derived_capabilities_match_the_manifest`);
//! capability flags gate visible named skips, never absence. The workspace
//! registry test (`streaming_conformance_registry.rs`) fails CI when any wire
//! family lacks an invocation.

use rig_core::streaming_conformance_suite;
use rig_core::test_utils::streaming_conformance::fixtures::{
    anthropic, cohere, gemini_rest, interactions, ollama, openai_chat, openai_responses,
};

/// Compile-linked manifest of the wire families this file's suites cover.
///
/// Each entry is the `WIRE_FAMILY` const the macro emits inside the suite
/// module, so the registry
/// (`streaming_conformance_registry::all_wire_families_have_conformance_suites`)
/// reads the families that actually *compiled* rather than text it scraped
/// off disk. Commenting a suite out, or `#[cfg]`-disabling it, is therefore a
/// COMPILE error here instead of a silently shrinking test count (#2258 F3).
pub const SUITE_FAMILIES: &[&str] = &[
    openai_chat_suite::WIRE_FAMILY,
    openai_responses_suite::WIRE_FAMILY,
    gemini_rest_suite::WIRE_FAMILY,
    gemini_interactions_suite::WIRE_FAMILY,
    anthropic_suite::WIRE_FAMILY,
    cohere_suite::WIRE_FAMILY,
    ollama_suite::WIRE_FAMILY,
];

pub mod openai_chat_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "openai_chat",
        fixture: openai_chat::fixture(),
        manifest: [partial_tool_args, zero_usage_terminal, bare_terminal, malformed_frame, defective_known_frame, delta_less_prelude],
    }
}

pub mod openai_responses_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "openai_responses",
        fixture: openai_responses::fixture(),
        manifest: [partial_tool_args, zero_usage_terminal, malformed_frame, unknown_event_frame, defective_known_frame, refusal],
    }
}

pub mod gemini_rest_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "gemini_rest",
        fixture: gemini_rest::fixture(),
        manifest: [zero_usage_terminal, malformed_frame, unknown_event_frame, defective_known_frame, interleaved_reasoning],
    }
}

pub mod gemini_interactions_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "gemini_interactions",
        fixture: interactions::fixture(),
        manifest: [zero_usage_terminal, malformed_frame, unknown_event_frame, defective_known_frame, interleaved_reasoning],
    }
}

pub mod anthropic_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "anthropic",
        fixture: anthropic::fixture(),
        manifest: [partial_tool_args, bare_terminal, malformed_frame, unknown_event_frame, defective_known_frame],
    }
}

pub mod cohere_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "cohere",
        fixture: cohere::fixture(),
        manifest: [partial_tool_args, zero_usage_terminal, malformed_frame, unknown_event_frame, defective_known_frame, interleaved_reasoning],
    }
}

pub mod ollama_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "ollama",
        fixture: ollama::fixture(),
        manifest: [zero_usage_terminal, malformed_frame, defective_known_frame, interleaved_reasoning],
    }
}
