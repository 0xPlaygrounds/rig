//! Matrix D: continuation, cancellation and failure outcomes.
//!
//! The outcome axis, and the run beyond a single `prompt`. A record's
//! outcome can be a success, a `Cancelled` report (the consumer dropped
//! the stream), a failed tool result (the model sees it), or the
//! provider's own error (the run fails at the record). A run can end in
//! an answer, `MaxTurnsError`, `UnknownToolCall`, or the provider's error,
//! and a run can be suspended after its first tool result, serialized and
//! resumed on a fresh bus to the same answer.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | record outcome | success · `Cancelled` after a text delta (the corpus's) · `Cancelled` after a tool-call delta · a failed tool result · the provider's 401 |
//! | transport | unary · streamed with events |
//! | run ending | answer · `MaxTurnsError` with a tool pending · `UnknownToolCall` unhandled · the provider's error |
//! | turn budget | runner `max_turns` (not in the spec) · builder `default_max_turns` (in the spec) |
//! | continuation | one run · suspended after the first tool result, resumed |
//! | resume wire | anthropic · gemini (id-less) · openai (dual ids) · anthropic under serial serving with two calls |
//!
//! Full cross-product: 5 × 2 × 4 × 2 × 2 × 4 = 640. Recorded: the 8 goldens
//! below; replayed as resume rows: 4 existing goldens. Pruned: a cancel
//! after a `BlockEnd` (a text answer's only block ends with the stream, so
//! there is nothing left to cancel; a two-call stream cancelled between
//! calls is the tool-call-delta cell with more of the stream kept); a
//! unary cancel by timeout (whether a timeout fires before a cassette
//! answers is a race, so the record would not be deterministic); a handler
//! that panics (a panic in a handler is a bug, not an outcome); the
//! provider's error streamed with events dropped (an error has no events
//! to drop).
//!
//! # Cells
//!
//! | golden | producer | shape | ending |
//! |---|---|---|---|
//! | `anthropic_outcome_cancel_after_tool_call_delta` | `corpus_outcome.rs` `cancel_after_tool_call_delta_…` | `[Completion]`, outcome `Cancelled`, events kept; a `write_note` call whose long body streams past the drop | the run never finishes |
//! | `anthropic_outcome_tool_error` | `tool_error_…` | `[Completion, Tool (failed), Completion]` | an answer around the failure |
//! | `anthropic_outcome_tool_error_streamed` | `tool_error_streamed_…` | the same, events kept | an answer |
//! | `anthropic_outcome_model_error` | `model_error_…` | `[Completion]`, outcome the provider's 401 | `PromptError::Report(ProviderResponse)` |
//! | `anthropic_outcome_model_error_streamed` | `model_error_streamed_…` | the same, streamed | the stream's one item is the error |
//! | `anthropic_outcome_max_turns_exhausted` | `max_turns_exhausted_…` | `[Completion, Tool]` | `MaxTurnsError { max_turns: 1 }` |
//! | `anthropic_outcome_default_max_turns` | `default_max_turns_…` | `[Completion, Tool, Completion]` | an answer; the header refuses the runner-budget golden's program |
//! | `mock_outcome_invalid_call_unhandled` | `tests/core/golden_outcome.rs` `outcome_invalid_call_unhandled_…` | `[Completion]` | `UnknownToolCall` (mock-scripted: no live model calls an unadvertised tool) |
//!
//! Resume rows (no golden of their own: the hand driver takes the head,
//! the engine resumes the tail, the log is the golden):
//! `anthropic_tool_call_turn`, `gemini_tool_call_turns`,
//! `openai_tool_call_turns`, `anthropic_concurrent_tools_serial`.
//!
//! # What the matrix found
//!
//! - The hand driver shaped a failed tool result by unwrapping it, so a
//!   failed record could not be driven by hand; the driver now shapes the
//!   result's model-visible output as the engine does (`ToolResult::
//!   output`), failed or not.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{Ending, Program};
use rig_agent::AgentBuilder;

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const BASIC_PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const TWO_TOOL_STREAM_PREAMBLE: &str = "\
You are a precise assistant. When tools are available, you must use them instead of guessing. \
Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. \
Never call the same tool twice once you already have its result.";
const TWO_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
After both tool results are available, stop calling tools and respond in one short sentence that includes both exact tool outputs.";
const CHAIN_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for \
     every arithmetic operation instead of computing results yourself. Perform the steps in order, \
     using the result of each step as an input to the next. Once you have the final tool result, \
     reply with the final numeric answer in plain text.";
const CHAIN_PROMPT: &str = "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the \
     subtract tool. Report the final number.";

const TOOL_TURN: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};
const BASIC: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: BASIC_PROMPT,
    temperature: Some(0.0),
    ..Program::DEFAULT
};

const NOTE_PREAMBLE: &str =
    "You are a note-taking assistant. Use the write_note tool to save notes.";
const NOTE_PROMPT: &str = "Save a note titled 'Rust' whose body is a 400-word essay on the history of the Rust programming language, then reply with just the word saved.";

const CANCEL_AFTER_TOOL_CALL_DELTA: Program = Program {
    fixture: "anthropic_outcome_cancel_after_tool_call_delta",
    preamble: Some(NOTE_PREAMBLE),
    prompt: NOTE_PROMPT,
    streamed: true,
    cancel_after_first_delta: true,
    ..TOOL_TURN
};
const TOOL_ERROR: Program = Program {
    fixture: "anthropic_outcome_tool_error",
    ..TOOL_TURN
};
const TOOL_ERROR_STREAMED: Program = Program {
    fixture: "anthropic_outcome_tool_error_streamed",
    streamed: true,
    ..TOOL_TURN
};
const MODEL_ERROR: Program = Program {
    fixture: "anthropic_outcome_model_error",
    ending: Ending::ProviderError,
    ..BASIC
};
const MODEL_ERROR_STREAMED: Program = Program {
    fixture: "anthropic_outcome_model_error_streamed",
    streamed: true,
    ending: Ending::ProviderError,
    ..BASIC
};
const MAX_TURNS_EXHAUSTED: Program = Program {
    fixture: "anthropic_outcome_max_turns_exhausted",
    max_turns: Some(1),
    ending: Ending::MaxTurns,
    ..TOOL_TURN
};
const DEFAULT_MAX_TURNS: Program = Program {
    fixture: "anthropic_outcome_default_max_turns",
    default_max_turns: Some(3),
    max_turns: None,
    ..TOOL_TURN
};
const INVALID_CALL_UNHANDLED: Program = Program {
    fixture: "mock_outcome_invalid_call_unhandled",
    preamble: Some("Use the add tool."),
    prompt: "What is 2 * 3?",
    max_turns: Some(3),
    ending: Ending::UnknownToolCall,
    ..Program::DEFAULT
};

// The resume rows: the corpus's tool programs, verbatim
// (`golden_replay.rs`).
const RESUME_ANTHROPIC: Program = Program {
    fixture: "anthropic_tool_call_turn",
    ..TOOL_TURN
};
const RESUME_GEMINI: Program = Program {
    fixture: "gemini_tool_call_turns",
    owner: "stress-agent",
    preamble: Some(CHAIN_PREAMBLE),
    prompt: CHAIN_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(6),
    streamed: true,
    ..Program::DEFAULT
};
const RESUME_OPENAI: Program = Program {
    fixture: "openai_tool_call_turns",
    preamble: Some(CHAIN_PREAMBLE),
    prompt: CHAIN_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(6),
    ..Program::DEFAULT
};
const RESUME_SERIAL: Program = Program {
    fixture: "anthropic_concurrent_tools_serial",
    preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
    prompt: TWO_TOOL_STREAM_PROMPT,
    max_turns: Some(8),
    tool_concurrency: Some(2),
    streamed: true,
    ..Program::DEFAULT
};

both_interpreters! {
    cancel_after_tool_call_delta: CANCEL_AFTER_TOOL_CALL_DELTA,
    tool_error: TOOL_ERROR,
    tool_error_streamed: TOOL_ERROR_STREAMED,
    model_error: MODEL_ERROR,
    model_error_streamed: MODEL_ERROR_STREAMED,
    max_turns_exhausted: MAX_TURNS_EXHAUSTED,
    default_max_turns: DEFAULT_MAX_TURNS,
    invalid_call_unhandled: INVALID_CALL_UNHANDLED,
}

mod resumed {
    #[tokio::test]
    async fn anthropic_tool_call_turn() {
        crate::corpus::resume_reproduces(&super::RESUME_ANTHROPIC).await;
    }

    #[tokio::test]
    async fn gemini_tool_call_turns() {
        crate::corpus::resume_reproduces(&super::RESUME_GEMINI).await;
    }

    #[tokio::test]
    async fn openai_tool_call_turns() {
        crate::corpus::resume_reproduces(&super::RESUME_OPENAI).await;
    }

    #[tokio::test]
    async fn anthropic_concurrent_tools_serial() {
        crate::corpus::resume_reproduces(&super::RESUME_SERIAL).await;
    }
}

/// The builder's budget is in the spec, the runner's is not: the
/// default-budget golden's records are the runner-budget golden's, and
/// its header refuses the runner-budget program by spec hash.
#[tokio::test]
async fn the_default_budget_is_program_and_the_runner_budget_is_not() {
    let by_default = corpus::golden(DEFAULT_MAX_TURNS.fixture);
    let by_runner = corpus::golden(RESUME_ANTHROPIC.fixture);
    let records =
        |log: &rig_effect_log::EffectLog| log.iter().map(corpus::as_data).collect::<Vec<_>>();
    assert_eq!(records(&by_default), records(&by_runner), "the same run");
    assert_ne!(by_default.header.run_spec, by_runner.header.run_spec);

    let replay = corpus::Replay::open(&DEFAULT_MAX_TURNS);
    let server = replay.tool_server();
    let runner_budget = AgentBuilder::over_bus(
        replay.dispatcher.clone(),
        replay.registrar.clone(),
        "golden",
        replay.model_key.clone(),
    )
    .name("golden")
    .preamble(TOOLS_PREAMBLE)
    .temperature(0.0)
    .tool_server_handle(server)
    .build();
    let refusal = runner_budget
        .check_replayable(&replay.log)
        .expect_err("a default budget is another program")
        .to_string();
    assert!(refusal.contains("run spec"), "{refusal}");
    drop(runner_budget);
    replay.close().await;
}
