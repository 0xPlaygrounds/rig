//! Matrix M: per-turn shaping.
//!
//! A request patch from `on_completion_call` shapes one turn's request
//! and nothing else: the spec hash is the unpatched program's, the
//! record's request is the patched one, and a replay without the
//! patching hook is refused by the hook-stack check before any
//! divergence. A model-selection hook picks the turn's model, on the
//! first turn only here (the reverse of Matrix C's route after the first
//! turn), or a route the agent registered after build.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | patched field | `tool_choice` · `extra_context` · `preamble` · `max_tokens` · `additional_params` (with `temperature`) · `active_tools` · `history` |
//! | the turn | 1 only · 2 only · every turn |
//! | hooks merged | one · three (preamble, context, first-turn choice) |
//! | the run | one completion · a tool turn · a committed `Tool`-mode run |
//! | route | none · selected on turn 1 · registered after build |
//! | medium | unary · streamed with events |
//!
//! Full cross-product: 7 × 3 × 2 × 3 × 3 × 2 = 756. Recorded: the 12 cells
//! below. Pruned: every field on every turn (each field is pinned on one
//! turn of a two-turn program, where the other turn shows the unpatched
//! request; `extra_context` on every turn of a one-turn program is the
//! same shape); `history` on turn 2 (a patched history replaces the
//! run's own transcript mid-run: a shape no consumer asks for, refused
//! here as a cell — the pin would be a run that forgets its tool turn);
//! `active_tools` narrowing to a subset (Matrix E's `active_tools` cell
//! pins the filter; the patch pins the per-turn switch); a stream of the
//! tool-turn cells (the medium changes events, not the patch); a route
//! on turn 1 beside a patch (the two seams are independent).
//!
//! # Cells
//!
//! | golden | producer (anthropic `corpus_shaping.rs`) | shape |
//! |---|---|---|
//! | `anthropic_shaping_tool_choice_required_first` | `tool_choice_required_first_…` | `[Completion, Tool, Completion]`; `Required` on turn 1, unset on turn 2 |
//! | `anthropic_shaping_tool_choice_none_on_committed_output` | `tool_choice_none_on_committed_output_…` | a committed `Tool`-mode run whose turn 2 cannot call the output tool |
//! | `anthropic_shaping_extra_context` | `extra_context_…` | `[Completion]`; the document in the request |
//! | `anthropic_shaping_extra_context_streamed` | `extra_context_streamed_…` | the same, events kept |
//! | `anthropic_shaping_merged_three` | `merged_three_…` | `[Completion, Tool, Completion]`; preamble, document and choice from three hooks |
//! | `anthropic_shaping_route_on_first_turn` | `route_on_first_turn_…` | `[Completion(fast), Tool, Completion(default)]` |
//! | `anthropic_shaping_late_route` | `late_route_…` | `[Completion(late), Tool, Completion(late)]`; the route not in the row |
//! | `anthropic_shaping_max_tokens_second_turn` | `max_tokens_second_turn_…` | `[Completion, Tool, Completion]`; `max_tokens: 5` on turn 2 |
//! | `anthropic_shaping_thinking_second_turn` | `thinking_second_turn_…` | thinking and temperature 1.0 on turn 2 |
//! | `anthropic_shaping_preamble_second_turn` | `preamble_second_turn_…` | the pirate preamble on turn 2 |
//! | `anthropic_shaping_active_tools_none_second_turn` | `active_tools_none_second_turn_…` | no tools on turn 2 |
//! | `anthropic_shaping_history_first_turn` | `history_first_turn_…` | `[Completion]`; the patched exchange before the prompt |
//!
//! # What the matrix found
//!
//! - A committed `Tool`-mode run whose turn 2 is patched to
//!   `tool_choice: None` does not stall: the engine warns, the model
//!   answers in text, the run's output validation reprompts for the
//!   output tool's call, and turn 3 (unpatched) makes it —
//!   `[Completion, Tool, Completion, Completion]`, an answer. Pinned.
//! - A route registered after build (`register_model`) is served and
//!   recorded — in the signature and the handler table — and not in the
//!   required row, which is the builder's (#2450's stated limitation).
//!   A replay that registers it through the builder is refused (the row
//!   would differ from the header's); the replay serves it the way the
//!   producer did, on the bus before the agent is built, and the golden
//!   replays by both interpreters. Pinned, not fixed: the row is the
//!   program the builder describes, and a route registered on the
//!   built agent is the host's to serve.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

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
