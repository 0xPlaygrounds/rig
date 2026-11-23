//! Matrix L: resumption under everything.
//!
//! A resumed run is a third interpretation of a program, not a row class of
//! its own. The hand driver takes the program up to and including its first
//! tool turn's results and serializes the `AgentRun`. The bus engine then
//! resumes it on the same replay bus to the golden's ending. Each row resumes
//! an existing golden across hooks, a committed output tool, a route,
//! `Ignore`, the delta wire and retrieval, recording nothing new: the golden
//! is the oracle, the interpretation is the row.
//!
//! A resumed run fires no `on_run_start` (the run already started), takes no
//! history (it carries its own), and fires its own turn, tool, outcome and
//! settled hooks. It carries `previous_model`, so a hook that routes after the
//! first turn sees the model the head asked. The builder that resumes it
//! registers its own route and context handlers under the keys the head's
//! handlers hold, so the replay deregisters the head's and re-registers them
//! from the golden's tail; the tool server's replayers are the head's and
//! continue.

use crate::corpus;
use rig_cassette::agent::AgentReplayExt;

use corpus::{Ending, Hook, Output, Program, ROUTE, STOP_AT_ANSWER, Unhandled};

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
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
    server.attach(&replay.handlers);
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
