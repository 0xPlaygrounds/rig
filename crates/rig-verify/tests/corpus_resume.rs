//! Matrix L: resumption under everything.
//!
//! A resumed run is a third interpretation of a program, not a row class
//! of its own: the hand driver takes the program up to and including its
//! first tool turn's results, serializes the `AgentRun`, and the bus
//! engine resumes it on the same replay bus to the golden's ending. #2450
//! resumed four plain tool programs; every matrix since added programs
//! the resume path had never seen. This matrix resumes eighteen existing
//! goldens across hooks, memory, a committed output tool, a route,
//! `Ignore`, the delta wire and retrieval, recording nothing new: the
//! golden is the oracle, the interpretation is the row.
//!
//! What a resumed run does and does not do (`runner.rs`, `run` and
//! `stream`: "a resumed run loads nothing and saves nothing"): it fires
//! no `on_run_start` (the run already started), loads no history (it
//! carries its own), appends nothing to memory — so the driver that
//! resumed it appends, as the head loaded — and it fires the turn, tool,
//! outcome and settled hooks of its own. The builder that resumes it
//! registers its own memory, route and context handlers, so the replay
//! re-registers them from the golden's tail; the tool server's replayers
//! are the head's and continue.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | hooks | none · observe-only · a dispatch patch · an outcome replace · two hooks · a run-start dispatch · a stop after the resume point · a per-call custom note |
//! | memory | none · loaded by the head, appended by the driver |
//! | output mode | `Auto` · `Tool` committed on turn 1 · `Prompted` |
//! | route | none · selected after the resume point |
//! | invalid call | none · ignored before the resume point |
//! | head medium | unary · streamed with events |
//! | wire | anthropic · mock delta wire · gemini (retrieval) |
//!
//! Full cross-product: 8 × 2 × 3 × 2 × 2 × 2 × 3 = 1152. Interpreted: the
//! 18 rows below. Pruned: a route selected before the resume point (the
//! head selects it; the tail's model is then the route's — no golden
//! has it, and recording one is Matrix M's first-turn route); a run
//! serialized twice (the resumed engine exposes no second suspension);
//! a resume after a streamed head on a live wire beyond the one streamed
//! patch row (the medium changes the head's assembly, which the streamed
//! rows pin); `Ignore` after the resume point (the ignored call is in
//! the same turn as the valid one the head dispatches; the tail answers).
//!
//! # Rows
//!
//! | golden | what the resume carries |
//! |---|---|
//! | `anthropic_hooks_observe_everything` | an observe-only hook and memory: the head loads, the tail observes, the driver appends |
//! | `anthropic_hooks_patch_tool_args` | the head patched the call; the tail answers |
//! | `anthropic_hooks_patch_tool_args_streamed` | the same, the head assembled from a stream |
//! | `anthropic_hooks_replace_tool_result` | the head replaced the result the tail reads |
//! | `anthropic_hooks_two_hooks` | both |
//! | `anthropic_hooks_lookup_before_run` | the head's run-start dispatch; the resumed run fires no run start |
//! | `anthropic_endings_answer_turn_stop` | the stop fires in the tail: `PromptCancelled` from the resumed engine |
//! | `anthropic_host_custom_at_outcome` | the head's per-call note; the tail dispatches none |
//! | `anthropic_host_custom_at_outcome_streamed` | the same, streamed head |
//! | `anthropic_serving_serial_memory_tools` | memory under serial serving |
//! | `anthropic_memory_serial_two_tools` | memory, a streamed head with two calls |
//! | `anthropic_output_tool_with_real_tool` | the output tool committed on turn 1; the tail finalizes on it |
//! | `anthropic_output_prompted_with_real_tool` | the prompted schema; the tail's answer is the JSON text |
//! | `anthropic_serving_model_route` | the route selected on the tail's turn |
//! | `mock_invalid_mixed_ignore` | the ignored call resolved by the head |
//! | `mock_invalid_mixed_ignore_streamed` | the same, streamed head |
//! | `mock_delta_repair` | a name-delta call repaired by the head |
//! | `gemini_retrieval_context_and_tools` | the head's retrievals; the tail retrieves again for its turn |
//!
//! # What the matrix found
//!
//! - A resumed run never appends to memory: `resume` skips the history
//!   and memory resolution, so the runner holds no memory handle and the
//!   `Done` append is not dispatched. The driver that suspended the run
//!   appends what the run produced; pinned, not changed (a driver that
//!   persists a run between steps owns its memory).
//! - A resumed run forgot the model the head asked: routing state lived
//!   in the driver only ("live routing state stays in the driver, not the
//!   serde `AgentRun`"), so a resumed engine's model-selection hook saw
//!   `previous_model: None` and a hook that routes after the first turn
//!   asked the default model again — a divergence from the golden the
//!   fresh run recorded. The run now carries `previous_model`, advanced
//!   where the driver advances it, and the resumed engine seeds its
//!   routing state from it (`rig-agent`, with a unit test; the hand
//!   driver sets it as the engine does).
//! - The builder that resumes a run registers its handlers as a fresh
//!   run's builder does, under the same keys the head's handlers hold;
//!   a second registration is refused ("a fresh key"), so the head's
//!   memory, route and context handlers are deregistered and the tail's
//!   registered from the golden's tail. Tools are the server's and
//!   continue.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{Ending, Hook, Output, Program, ROUTE, STOP_AT_ANSWER, Unhandled};

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const TWO_TOOL_STREAM_PREAMBLE: &str = "\
You are a precise assistant. When tools are available, you must use them instead of guessing. \
Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. \
Never call the same tool twice once you already have its result.";
const TWO_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
After both tool results are available, stop calling tools and respond in one short sentence that includes both exact tool outputs.";
const SUM_EVENT_PROMPT: &str = "Use the add tool to add 17 and 25, then return a concise event object for a Rust meetup in Seattle whose summary states the sum.";
const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;
const MOCK_PREAMBLE: &str = "Use the add tool.";
const MOCK_PROMPT: &str = "What is 2 + 3?";
const RETRIEVED_TOOLS_PREAMBLE: &str =
    "You are a calculator. You must use the provided tools for every arithmetic operation.";
const SUBTRACT_PROMPT: &str =
    "Subtract 8 from 50 with the subtract tool, then reply with just the number.";

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
const MOCK: Program = Program {
    preamble: Some(MOCK_PREAMBLE),
    prompt: MOCK_PROMPT,
    max_turns: Some(3),
    ..Program::DEFAULT
};

const OBSERVE_EVERYTHING: Program = Program {
    fixture: "anthropic_hooks_observe_everything",
    conversation: Some("golden-conversation"),
    hooks: &[Hook::ObserveEverything],
    ..TOOLS
};
const PATCH_TOOL_ARGS: Program = Program {
    fixture: "anthropic_hooks_patch_tool_args",
    hooks: &[Hook::PatchAddArgs],
    ..TOOLS
};
const PATCH_TOOL_ARGS_STREAMED: Program = Program {
    fixture: "anthropic_hooks_patch_tool_args_streamed",
    hooks: &[Hook::PatchAddArgs],
    streamed: true,
    ..TOOLS
};
const REPLACE_TOOL_RESULT: Program = Program {
    fixture: "anthropic_hooks_replace_tool_result",
    hooks: &[Hook::ReplaceAddResult],
    ..TOOLS
};
const TWO_HOOKS: Program = Program {
    fixture: "anthropic_hooks_two_hooks",
    hooks: &[Hook::PatchAddArgs, Hook::ReplaceAddResult],
    ..TOOLS
};
const LOOKUP_BEFORE_RUN: Program = Program {
    fixture: "anthropic_hooks_lookup_before_run",
    hooks: &[Hook::LookupBeforeRun],
    ..TOOLS
};
const ANSWER_TURN_STOP: Program = Program {
    fixture: "anthropic_endings_answer_turn_stop",
    hooks: &[Hook::StopAtAnswer, Hook::RecordSettled],
    ending: Ending::Cancelled(STOP_AT_ANSWER),
    ..TOOLS
};
const CUSTOM_AT_OUTCOME: Program = Program {
    fixture: "anthropic_host_custom_at_outcome",
    hooks: &[Hook::NoteAtOutcome],
    ..TOOLS
};
const CUSTOM_AT_OUTCOME_STREAMED: Program = Program {
    fixture: "anthropic_host_custom_at_outcome_streamed",
    hooks: &[Hook::NoteAtOutcome],
    streamed: true,
    ..TOOLS
};
const SERIAL_MEMORY_TOOLS: Program = Program {
    fixture: "anthropic_serving_serial_memory_tools",
    conversation: Some("golden-conversation"),
    ..TOOLS
};
const MEMORY_SERIAL_TWO_TOOLS: Program = Program {
    fixture: "anthropic_memory_serial_two_tools",
    preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
    prompt: TWO_TOOL_STREAM_PROMPT,
    max_turns: Some(8),
    streamed: true,
    conversation: Some("golden-conversation"),
    ..TOOLS
};
const OUTPUT_TOOL_WITH_REAL_TOOL: Program = Program {
    fixture: "anthropic_output_tool_with_real_tool",
    prompt: SUM_EVENT_PROMPT,
    output_schema: Some(event_schema),
    output_mode: Some(Output::Tool),
    ..TOOLS
};
const OUTPUT_PROMPTED_WITH_REAL_TOOL: Program = Program {
    fixture: "anthropic_output_prompted_with_real_tool",
    prompt: SUM_EVENT_PROMPT,
    output_schema: Some(event_schema),
    output_mode: Some(Output::Prompted),
    ..TOOLS
};
const MODEL_ROUTE: Program = Program {
    fixture: "anthropic_serving_model_route",
    route: Some(ROUTE),
    hooks: &[Hook::RouteAfterFirstTurn],
    ..TOOLS
};
const MIXED_IGNORE: Program = Program {
    fixture: "mock_invalid_mixed_ignore",
    unhandled: Unhandled::Ignore,
    ..MOCK
};
const MIXED_IGNORE_STREAMED: Program = Program {
    fixture: "mock_invalid_mixed_ignore_streamed",
    unhandled: Unhandled::Ignore,
    streamed: true,
    ..MOCK
};
const DELTA_REPAIR: Program = Program {
    fixture: "mock_delta_repair",
    hooks: &[Hook::RepairToAdd],
    streamed: true,
    ..MOCK
};
const GEMINI_CONTEXT_AND_TOOLS: Program = Program {
    fixture: "gemini_retrieval_context_and_tools",
    preamble: Some(RETRIEVED_TOOLS_PREAMBLE),
    prompt: SUBTRACT_PROMPT,
    retrieved_tools: Some(1),
    retrievable: &["add", "subtract"],
    dynamic_context: Some(1),
    ..TOOLS
};

macro_rules! resumed {
    ($($name:ident: $program:ident,)*) => {
        mod resumed {
            $(
                #[tokio::test]
                async fn $name() {
                    crate::corpus::resume_reproduces(&super::$program).await;
                }
            )*
        }
    };
}

resumed! {
    observe_everything: OBSERVE_EVERYTHING,
    patch_tool_args: PATCH_TOOL_ARGS,
    patch_tool_args_streamed: PATCH_TOOL_ARGS_STREAMED,
    replace_tool_result: REPLACE_TOOL_RESULT,
    two_hooks: TWO_HOOKS,
    lookup_before_run: LOOKUP_BEFORE_RUN,
    answer_turn_stop: ANSWER_TURN_STOP,
    custom_at_outcome: CUSTOM_AT_OUTCOME,
    custom_at_outcome_streamed: CUSTOM_AT_OUTCOME_STREAMED,
    serial_memory_tools: SERIAL_MEMORY_TOOLS,
    memory_serial_two_tools: MEMORY_SERIAL_TWO_TOOLS,
    output_tool_with_real_tool: OUTPUT_TOOL_WITH_REAL_TOOL,
    output_prompted_with_real_tool: OUTPUT_PROMPTED_WITH_REAL_TOOL,
    model_route: MODEL_ROUTE,
    mixed_ignore: MIXED_IGNORE,
    mixed_ignore_streamed: MIXED_IGNORE_STREAMED,
    delta_repair: DELTA_REPAIR,
    gemini_context_and_tools: GEMINI_CONTEXT_AND_TOOLS,
}

/// A resumed side whose bus lacks a required key is refused before any
/// dispatch: the route the golden's row names is not served.
#[tokio::test]
async fn a_resumed_side_missing_a_required_key_is_refused() {
    let replay = corpus::Replay::open(&MODEL_ROUTE);
    let server = replay.tool_server_for(&MODEL_ROUTE);
    server.attach(&replay.registrar);
    let without_route = Program {
        route: None,
        ..MODEL_ROUTE
    };
    let agent = corpus::build_agent_unchecked(&replay, &without_route, server, &replay.log);
    let refused = agent
        .check_replayable(&replay.log)
        .expect_err("the row names a route this bus does not serve");
    assert_eq!(
        refused.kind,
        rig_core::error::ErrorKind::HandlerUnavailable,
        "{refused:?}"
    );
    drop(agent);
    replay.close().await;
}
