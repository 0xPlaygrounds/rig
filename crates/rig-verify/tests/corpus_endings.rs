//! Matrix F: hook-ended runs.
//!
//! Every `Stop` in the hook surface ends the run in
//! `PromptError::PromptCancelled` with the hook's reason. Where the stop
//! fires decides what the log holds: before any dispatch (nothing), before
//! a tool reaches the bus (the completion only), after a tool answered
//! (the completion and the tool, the tool's real result in the record),
//! after a model turn (the completion), or on a streamed delta (the
//! completion as a cancel: the engine drops the model's stream there).
//! The hand driver makes each decision at the point the engine does and
//! must reach the same reason with the same records.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | the stop | `on_run_start` · `on_model_select` · `on_completion_call` · `on_dispatch` Deny(Cancelled) · `on_outcome` Replace(Err) on a tool · on an answer · `on_model_turn_finished` first turn · answer turn of a tool program · `on_text_delta` · `on_tool_call_delta` · `on_reasoning_delta` |
//! | medium | unary · streamed with events |
//! | records before the stop | none · one completion · completion + tool · completion + tool + completion |
//!
//! Full cross-product: 11 × 2 × 4 = 88. Recorded: the 13 cells below.
//! Pruned: a stop that fires before any dispatch, streamed (the medium is
//! decided after those hooks fire, so the log is the same empty header —
//! the unary cell is the cell); the delta stops unary (a delta is a stream
//! event); `on_outcome` on an answer streamed and the answer-turn stop
//! streamed (the fold settles a streamed turn through the same
//! `settle_model_turn`; the streamed turn-finished cell covers the seam);
//! a stop with two records before it on every medium (one streamed
//! two-record cell, the outcome cancel, covers the tool-then-stop shape);
//! the reasoning-delta stop, recorded twice under extended thinking and
//! stopped: the recording proxy delivered the thinking block whole, the
//! stop landed after it, and the record was a complete completion that a
//! replay — which cancels at the delta — cannot reproduce (the
//! `StopOnReasoningDelta` hook stays in the corpus for a wire that
//! streams thinking deltas as deltas).
//!
//! # Cells
//!
//! | golden | producer | records | reason |
//! |---|---|---|---|
//! | `mock_endings_stop_at_start` | `tests/core/golden_endings.rs` `endings_stop_at_start_…` | none | `stopped at run start` |
//! | `mock_endings_stop_at_model_select` | `endings_stop_at_model_select_…` | none | `stopped at model selection` |
//! | `mock_endings_stop_at_completion_call` | `endings_stop_at_completion_call_…` | none | `stopped before the completion call` |
//! | `anthropic_endings_tool_dispatch_cancelled` | anthropic `corpus_endings.rs` `tool_dispatch_cancelled_…` | `[Completion]` | `add is cancelled before the bus` |
//! | `anthropic_endings_tool_outcome_cancelled` | `tool_outcome_cancelled_…` | `[Completion, Tool]`, the tool's real result | `add is cancelled after the bus` |
//! | `anthropic_endings_answer_outcome_cancelled` | `answer_outcome_cancelled_…` | `[Completion]`, the real answer | `the answer is cancelled` |
//! | `anthropic_endings_turn_finished_stop` | `turn_finished_stop_…` | `[Completion]` | `stopped after the model turn` |
//! | `anthropic_endings_answer_turn_stop` | `answer_turn_stop_…` | `[Completion, Tool, Completion]` | `stopped at the answer turn` |
//! | `anthropic_endings_text_delta_stop` | `text_delta_stop_…` | `[Completion]` as `Cancelled`, events kept | `stopped on the first text delta` |
//! | `anthropic_endings_tool_call_delta_stop` | `tool_call_delta_stop_…` | the same | `stopped on the first tool-call delta` |
//! | `anthropic_endings_tool_dispatch_cancelled_streamed` | `tool_dispatch_cancelled_streamed_…` | `[Completion]` whole, events kept | `add is cancelled before the bus` |
//! | `anthropic_endings_turn_finished_stop_streamed` | `turn_finished_stop_streamed_…` | `[Completion]` whole | `stopped after the model turn` |
//! | `anthropic_endings_tool_outcome_cancelled_streamed` | `tool_outcome_cancelled_streamed_…` | `[Completion, Tool]` | `add is cancelled after the bus` |
//!
//! # What the matrix found
//!
//! - `on_run_settled` never fired for a blocking run a hook stopped: the
//!   engine settles an error ending after yielding it, and the blocking
//!   fold returned at the yield, dropping the engine; the streaming
//!   surface's consumer, polling to the end, saw the hook fire. The fold
//!   now drains the engine before returning the error (`rig-agent`).
//! - A delta stop's record is the handler's timing: the tap records what
//!   the handler answered, so a stop lands as a cancel only if the model
//!   was still streaming when the run dropped its stream. The engine
//!   surfaced the stop and left the stream to be dropped with the run —
//!   later than it could be; it now drops the stream before surfacing the
//!   stop (`rig-agent`), the earliest cancel there is. What remains is
//!   the wire's: the cells stream long answers (an essay, a 400-word tool
//!   argument), as the corpus's consumer-cancel cells do, so the drop
//!   lands live and on replay alike.
//! - A dispatch a hook cancelled in flight left no record over an agent's
//!   own bus: the driver settles a cancel only when polled, and nothing
//!   polls an owned driver between runs, so the log missed the cancelled
//!   dispatch (a host-driven bus, polled by its host, recorded it). The
//!   run's drive now settles in-flight cancels before it finishes
//!   (`rig-agent`).
//!
//! Every producer's stack ends with `RecordSettled`, an observe-only hook
//! that records what `on_run_settled` saw, so the producer asserts the
//! settled outcome was the error; the replay's hook of that name observes
//! nothing (its state is not identity).

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{
    CANCEL_ADD_DISPATCH, CANCEL_ADD_OUTCOME, CANCEL_ANSWER, Ending, Hook, Program, STOP_AFTER_TURN,
    STOP_AT_ANSWER, STOP_AT_COMPLETION_CALL, STOP_AT_MODEL_SELECT, STOP_AT_START,
    STOP_ON_TEXT_DELTA, STOP_ON_TOOL_CALL_DELTA,
};

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const BASIC_PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const ESSAY_PROMPT: &str =
    "Write a 600-word essay on the history of the Rust programming language.";
const NOTE_PREAMBLE: &str =
    "You are a note-taking assistant. Use the write_note tool to save notes.";
const NOTE_PROMPT: &str = "Save a note titled 'Rust' whose body is a 400-word essay on the history of the Rust programming language, then reply with just the word saved.";

const MOCK: Program = Program {
    preamble: Some("Use the add tool."),
    prompt: "What is 2 + 3?",
    max_turns: Some(3),
    ..Program::DEFAULT
};
const TOOLS: Program = Program {
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

const STOP_AT_START_CELL: Program = Program {
    fixture: "mock_endings_stop_at_start",
    hooks: &[Hook::StopAtStart, Hook::RecordSettled],
    ending: Ending::Cancelled(STOP_AT_START),
    ..MOCK
};
const STOP_AT_MODEL_SELECT_CELL: Program = Program {
    fixture: "mock_endings_stop_at_model_select",
    hooks: &[Hook::StopAtModelSelect, Hook::RecordSettled],
    ending: Ending::Cancelled(STOP_AT_MODEL_SELECT),
    ..MOCK
};
const STOP_AT_COMPLETION_CALL_CELL: Program = Program {
    fixture: "mock_endings_stop_at_completion_call",
    hooks: &[Hook::StopAtCompletionCall, Hook::RecordSettled],
    ending: Ending::Cancelled(STOP_AT_COMPLETION_CALL),
    ..MOCK
};
const TOOL_DISPATCH_CANCELLED: Program = Program {
    fixture: "anthropic_endings_tool_dispatch_cancelled",
    hooks: &[Hook::CancelAddDispatch, Hook::RecordSettled],
    ending: Ending::Cancelled(CANCEL_ADD_DISPATCH),
    ..TOOLS
};
const TOOL_OUTCOME_CANCELLED: Program = Program {
    fixture: "anthropic_endings_tool_outcome_cancelled",
    hooks: &[Hook::CancelAddOutcome, Hook::RecordSettled],
    ending: Ending::Cancelled(CANCEL_ADD_OUTCOME),
    ..TOOLS
};
const ANSWER_OUTCOME_CANCELLED: Program = Program {
    fixture: "anthropic_endings_answer_outcome_cancelled",
    hooks: &[Hook::CancelAnswer, Hook::RecordSettled],
    ending: Ending::Cancelled(CANCEL_ANSWER),
    ..BASIC
};
const TURN_FINISHED_STOP: Program = Program {
    fixture: "anthropic_endings_turn_finished_stop",
    hooks: &[Hook::StopAfterTurn, Hook::RecordSettled],
    ending: Ending::Cancelled(STOP_AFTER_TURN),
    ..TOOLS
};
const ANSWER_TURN_STOP: Program = Program {
    fixture: "anthropic_endings_answer_turn_stop",
    hooks: &[Hook::StopAtAnswer, Hook::RecordSettled],
    ending: Ending::Cancelled(STOP_AT_ANSWER),
    ..TOOLS
};
const TEXT_DELTA_STOP: Program = Program {
    fixture: "anthropic_endings_text_delta_stop",
    prompt: ESSAY_PROMPT,
    hooks: &[Hook::StopOnTextDelta, Hook::RecordSettled],
    ending: Ending::Cancelled(STOP_ON_TEXT_DELTA),
    streamed: true,
    max_turns: Some(3),
    ..BASIC
};
const TOOL_CALL_DELTA_STOP: Program = Program {
    fixture: "anthropic_endings_tool_call_delta_stop",
    preamble: Some(NOTE_PREAMBLE),
    prompt: NOTE_PROMPT,
    hooks: &[Hook::StopOnToolCallDelta, Hook::RecordSettled],
    ending: Ending::Cancelled(STOP_ON_TOOL_CALL_DELTA),
    streamed: true,
    ..TOOLS
};
const TOOL_DISPATCH_CANCELLED_STREAMED: Program = Program {
    fixture: "anthropic_endings_tool_dispatch_cancelled_streamed",
    streamed: true,
    ..TOOL_DISPATCH_CANCELLED
};
const TURN_FINISHED_STOP_STREAMED: Program = Program {
    fixture: "anthropic_endings_turn_finished_stop_streamed",
    streamed: true,
    ..TURN_FINISHED_STOP
};
const TOOL_OUTCOME_CANCELLED_STREAMED: Program = Program {
    fixture: "anthropic_endings_tool_outcome_cancelled_streamed",
    streamed: true,
    ..TOOL_OUTCOME_CANCELLED
};

both_interpreters! {
    stop_at_start: STOP_AT_START_CELL,
    stop_at_model_select: STOP_AT_MODEL_SELECT_CELL,
    stop_at_completion_call: STOP_AT_COMPLETION_CALL_CELL,
    tool_dispatch_cancelled: TOOL_DISPATCH_CANCELLED,
    tool_outcome_cancelled: TOOL_OUTCOME_CANCELLED,
    answer_outcome_cancelled: ANSWER_OUTCOME_CANCELLED,
    turn_finished_stop: TURN_FINISHED_STOP,
    answer_turn_stop: ANSWER_TURN_STOP,
    text_delta_stop: TEXT_DELTA_STOP,
    tool_call_delta_stop: TOOL_CALL_DELTA_STOP,
    tool_dispatch_cancelled_streamed: TOOL_DISPATCH_CANCELLED_STREAMED,
    turn_finished_stop_streamed: TURN_FINISHED_STOP_STREAMED,
    tool_outcome_cancelled_streamed: TOOL_OUTCOME_CANCELLED_STREAMED,
}

/// Where the stop fires decides what the log holds: read off the goldens.
#[test]
fn a_stop_records_what_the_engine_dispatched_before_it() {
    let lengths = [
        (STOP_AT_START_CELL.fixture, 0),
        (STOP_AT_MODEL_SELECT_CELL.fixture, 0),
        (STOP_AT_COMPLETION_CALL_CELL.fixture, 0),
        (TOOL_DISPATCH_CANCELLED.fixture, 1),
        (TOOL_OUTCOME_CANCELLED.fixture, 2),
        (ANSWER_OUTCOME_CANCELLED.fixture, 1),
        (TURN_FINISHED_STOP.fixture, 1),
        (ANSWER_TURN_STOP.fixture, 3),
        (TEXT_DELTA_STOP.fixture, 1),
        (TOOL_CALL_DELTA_STOP.fixture, 1),
    ];
    for (fixture, records) in lengths {
        let log = corpus::golden(fixture);
        assert_eq!(log.len(), records, "{fixture}");
    }
    // A delta stop cancels the dispatch in flight: the completion is the
    // cancel. A dispatch or turn stop fires after the stream finished:
    // the completion is whole.
    for fixture in [TEXT_DELTA_STOP.fixture, TOOL_CALL_DELTA_STOP.fixture] {
        let log = corpus::golden(fixture);
        let report = log[0].outcome.as_ref().expect_err("a cancel");
        assert_eq!(
            report.kind,
            rig_core::error::ErrorKind::Cancelled,
            "{fixture}"
        );
        assert!(log[0].events.is_some(), "{fixture}");
    }
    for fixture in [
        TOOL_DISPATCH_CANCELLED_STREAMED.fixture,
        TURN_FINISHED_STOP_STREAMED.fixture,
    ] {
        let log = corpus::golden(fixture);
        assert!(log[0].outcome.is_ok(), "{fixture}");
    }
}
