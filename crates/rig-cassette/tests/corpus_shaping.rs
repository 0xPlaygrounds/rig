//! Matrix M: per-turn shaping.
//!
//! A request patch from `on_completion_call` shapes one turn's request and
//! nothing else: the spec hash is the unpatched program's, the record's
//! request is the patched one, and a replay without the patching hook is
//! refused by the hook-stack check before any divergence. A model-selection
//! hook picks the turn's model, on the first turn only here, or a route the
//! agent registered after build.
//!
//! A committed `Tool`-mode run whose second turn is patched to
//! `tool_choice: None` does not stall: the model answers in text, output
//! validation reprompts, and the unpatched third turn calls the output tool.
//! A patched `history` on a later turn is not a cell, since it would pin a
//! run that forgets its own tool turn.
//!
//! A route registered after build with `register_model` is served and
//! recorded in the signature and handler table, but not in the required row,
//! which describes the builder's program. A replay must serve it the same
//! way, on the bus before the agent is built; registering it through the
//! builder is refused because the row would differ from the header's.

use crate::corpus;

use corpus::{Hook, LATE_ROUTE, Output, Program, ROUTE};
use rig_core::effect::HandlerKey;

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const CONTEXT_PROMPT: &str = "What is a glarb-glarb? Answer in one sentence.";
const NAME_PROMPT: &str = "What is my name? Reply with just the name.";
const SUM_EVENT_PROMPT: &str = "Use the add tool to add 17 and 25, then return a concise event object for a Rust meetup in Seattle whose summary states the sum.";
const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;

fn event_schema() -> serde_json::Value {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal parses")
}

const TOOLS: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};
const BASIC: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: CONTEXT_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};

const TOOL_CHOICE_REQUIRED_FIRST: Program = Program {
    fixture: "anthropic_shaping_tool_choice_required_first",
    hooks: &[Hook::PatchToolChoiceRequiredFirst],
    ..TOOLS
};
const TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT: Program = Program {
    fixture: "anthropic_shaping_tool_choice_none_on_committed_output",
    prompt: SUM_EVENT_PROMPT,
    output_schema: Some(event_schema),
    output_mode: Some(Output::Tool),
    hooks: &[Hook::PatchToolChoiceNoneSecond],
    ..TOOLS
};
const EXTRA_CONTEXT: Program = Program {
    fixture: "anthropic_shaping_extra_context",
    hooks: &[Hook::PatchExtraContext],
    ..BASIC
};
const EXTRA_CONTEXT_STREAMED: Program = Program {
    fixture: "anthropic_shaping_extra_context_streamed",
    hooks: &[Hook::PatchExtraContext],
    streamed: true,
    max_turns: None,
    ..BASIC
};
const MERGED_THREE: Program = Program {
    fixture: "anthropic_shaping_merged_three",
    hooks: &[
        Hook::PreambleOverride,
        Hook::PatchExtraContext,
        Hook::PatchToolChoiceRequiredFirst,
    ],
    ..TOOLS
};
const ROUTE_ON_FIRST_TURN: Program = Program {
    fixture: "anthropic_shaping_route_on_first_turn",
    route: Some(ROUTE),
    hooks: &[Hook::RouteOnFirstTurn],
    ..TOOLS
};
const LATE_ROUTE_PROGRAM: Program = Program {
    fixture: "anthropic_shaping_late_route",
    late_route: Some(LATE_ROUTE),
    hooks: &[Hook::SelectLate],
    ..TOOLS
};
const MAX_TOKENS_SECOND_TURN: Program = Program {
    fixture: "anthropic_shaping_max_tokens_second_turn",
    hooks: &[Hook::PatchMaxTokensSecond],
    ..TOOLS
};
const THINKING_SECOND_TURN: Program = Program {
    fixture: "anthropic_shaping_thinking_second_turn",
    hooks: &[Hook::PatchThinkingSecond],
    ..TOOLS
};
const PREAMBLE_SECOND_TURN: Program = Program {
    fixture: "anthropic_shaping_preamble_second_turn",
    hooks: &[Hook::PatchPreambleSecond],
    ..TOOLS
};
const ACTIVE_TOOLS_NONE_SECOND_TURN: Program = Program {
    fixture: "anthropic_shaping_active_tools_none_second_turn",
    hooks: &[Hook::PatchActiveToolsNoneSecond],
    ..TOOLS
};
const HISTORY_FIRST_TURN: Program = Program {
    fixture: "anthropic_shaping_history_first_turn",
    prompt: NAME_PROMPT,
    hooks: &[Hook::PatchHistoryFirst],
    ..BASIC
};

both_interpreters! {
    tool_choice_required_first: TOOL_CHOICE_REQUIRED_FIRST,
    tool_choice_none_on_committed_output: TOOL_CHOICE_NONE_ON_COMMITTED_OUTPUT,
    extra_context: EXTRA_CONTEXT,
    extra_context_streamed: EXTRA_CONTEXT_STREAMED,
    merged_three: MERGED_THREE,
    route_on_first_turn: ROUTE_ON_FIRST_TURN,
    late_route: LATE_ROUTE_PROGRAM,
    max_tokens_second_turn: MAX_TOKENS_SECOND_TURN,
    thinking_second_turn: THINKING_SECOND_TURN,
    preamble_second_turn: PREAMBLE_SECOND_TURN,
    active_tools_none_second_turn: ACTIVE_TOOLS_NONE_SECOND_TURN,
    history_first_turn: HISTORY_FIRST_TURN,
}

/// A patch is per turn and not program: a patched golden's spec hash is
/// the unpatched program's (Matrix B's `patch_tool_args` golden is the
/// same builder), and the golden replays only with the patching hook.
#[test]
fn a_patch_is_not_in_the_spec() {
    let patched = corpus::golden(PREAMBLE_SECOND_TURN.fixture);
    let unpatched = corpus::golden("anthropic_hooks_patch_tool_args");
    assert_eq!(patched.header.run_spec, unpatched.header.run_spec);
    assert_ne!(patched.header.hooks, unpatched.header.hooks);
}

/// A route registered after build is served, in the signature and the
/// handler table, and not in the required row (the builder's).
#[test]
fn a_late_route_is_served_but_not_required() {
    let log = corpus::golden(LATE_ROUTE_PROGRAM.fixture);
    let late = HandlerKey::from("golden/model:late");
    assert!(!log.header.required.contains_key(&late));
    assert!(log.header.signature.contains_key(&late));
    assert!(
        log.header
            .handlers
            .iter()
            .any(|handler| handler.key == late)
    );
}
