//! Macro-expanded wire-conformance suites for the byte-transport wire
//! families whose fixtures live in
//! `rig_core::test_utils::streaming_conformance::fixtures`.
//!
//! Each invocation expands the full canonical scenario set plus the
//! anti-tamper tests (`suite_is_complete`, `capabilities_match_fixture`);
//! capability flags gate visible named skips, never absence. The workspace
//! registry test (`streaming_conformance_registry.rs`) fails CI when any wire
//! family lacks an invocation.

use rig_core::streaming_conformance_suite;
use rig_core::test_utils::streaming_conformance::fixtures::{
    anthropic, cohere, gemini_rest, interactions, ollama, openai_chat, openai_responses,
};

mod openai_chat_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "openai_chat",
        fixture: openai_chat::fixture(),
        capabilities: {
            partial_tool_args: true,
            zero_usage_terminal: true,
            bare_terminal: true,
            malformed_frame: true,
            unknown_event_frame: false,
            defective_known_frame: true,
            delta_less_prelude: true,
            refusal: false,
        },
    }
}

mod openai_responses_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "openai_responses",
        fixture: openai_responses::fixture(),
        capabilities: {
            partial_tool_args: true,
            zero_usage_terminal: true,
            bare_terminal: false,
            malformed_frame: true,
            unknown_event_frame: true,
            defective_known_frame: true,
            delta_less_prelude: false,
            refusal: true,
        },
    }
}

mod gemini_rest_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "gemini_rest",
        fixture: gemini_rest::fixture(),
        capabilities: {
            partial_tool_args: false,
            zero_usage_terminal: true,
            bare_terminal: false,
            malformed_frame: true,
            unknown_event_frame: true,
            defective_known_frame: true,
            delta_less_prelude: false,
            refusal: false,
        },
    }
}

mod gemini_interactions_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "gemini_interactions",
        fixture: interactions::fixture(),
        capabilities: {
            partial_tool_args: false,
            zero_usage_terminal: true,
            bare_terminal: false,
            malformed_frame: true,
            unknown_event_frame: true,
            defective_known_frame: true,
            delta_less_prelude: false,
            refusal: false,
        },
    }
}

mod anthropic_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "anthropic",
        fixture: anthropic::fixture(),
        capabilities: {
            partial_tool_args: true,
            zero_usage_terminal: false,
            bare_terminal: true,
            malformed_frame: true,
            unknown_event_frame: true,
            defective_known_frame: true,
            delta_less_prelude: false,
            refusal: false,
        },
    }
}

mod cohere_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "cohere",
        fixture: cohere::fixture(),
        capabilities: {
            partial_tool_args: true,
            zero_usage_terminal: true,
            bare_terminal: false,
            malformed_frame: true,
            unknown_event_frame: true,
            defective_known_frame: true,
            delta_less_prelude: false,
            refusal: false,
        },
    }
}

mod ollama_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "ollama",
        fixture: ollama::fixture(),
        capabilities: {
            partial_tool_args: false,
            zero_usage_terminal: true,
            bare_terminal: false,
            malformed_frame: true,
            unknown_event_frame: false,
            defective_known_frame: true,
            delta_less_prelude: false,
            refusal: false,
        },
    }
}
