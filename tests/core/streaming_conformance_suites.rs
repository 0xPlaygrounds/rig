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
    }
}

mod openai_responses_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "openai_responses",
        fixture: openai_responses::fixture(),
    }
}

mod gemini_rest_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "gemini_rest",
        fixture: gemini_rest::fixture(),
    }
}

mod gemini_interactions_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "gemini_interactions",
        fixture: interactions::fixture(),
    }
}

mod anthropic_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "anthropic",
        fixture: anthropic::fixture(),
    }
}

mod cohere_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "cohere",
        fixture: cohere::fixture(),
    }
}

mod ollama_suite {
    use super::*;

    streaming_conformance_suite! {
        provider: "ollama",
        fixture: ollama::fixture(),
    }
}
