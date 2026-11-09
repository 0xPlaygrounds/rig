//! Matrix G: invalid tool calls, streamed and ignored.
//!
//! An invalid call, a tool the request did not advertise, is resolved by a
//! hook (`Retry` with feedback, `Repair` to an allowed tool, `Skip` with a
//! reason) or, with no hook, by the runner's policy (`Fail`, the default, or
//! `Ignore`). This matrix records every resolution on both media and the
//! policy both ways. The two interpreters must agree on what each resolution
//! puts in the transcript: the hand driver resolves through `AgentRun`'s own
//! steps, the engine through its streamed path. An ignored call that was the
//! turn's only content leaves an empty turn, and the run settles on it as an
//! empty answer.
//!
//! Every cell is scripted from the mock model. No model in the corpus emits a
//! call to a tool that is not in its request, so no cassette can hold these
//! shapes.

use crate::corpus;

use corpus::{Choice, Ending, Hook, Program, Unhandled};

const PREAMBLE: &str = "Use the add tool.";
const PROMPT: &str = "What is 2 + 3?";

const MOCK: Program = Program {
    preamble: Some(PREAMBLE),
    prompt: PROMPT,
    max_turns: Some(3),
    ..Program::DEFAULT
};

const STREAMED_RETRY_ONCE: Program = Program {
    fixture: "mock_invalid_streamed_retry_once",
    max_turns: Some(4),
    hooks: &[Hook::RetryUnknownTool],
    invalid_retries: 1,
    streamed: true,
    ..MOCK
};
const STREAMED_RETRY_TWICE: Program = Program {
    fixture: "mock_invalid_streamed_retry_twice",
    max_turns: Some(5),
    hooks: &[Hook::RetryUnknownTool],
    invalid_retries: 2,
    streamed: true,
    ..MOCK
};
const IGNORE_UNARY: Program = Program {
    fixture: "mock_invalid_ignore_unary",
    unhandled: Unhandled::Ignore,
    ..MOCK
};
const IGNORE_STREAMED: Program = Program {
    fixture: "mock_invalid_ignore_streamed",
    unhandled: Unhandled::Ignore,
    streamed: true,
    ..MOCK
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
const MIXED_FAIL: Program = Program {
    fixture: "mock_invalid_mixed_fail",
    ending: Ending::UnknownToolCall,
    ..MOCK
};
const REPAIR_TO_ADD: Program = Program {
    fixture: "mock_invalid_repair_to_add",
    hooks: &[Hook::RepairToAdd],
    ..MOCK
};
const SKIP_UNDER_AUTO: Program = Program {
    fixture: "mock_invalid_skip_under_auto",
    hooks: &[Hook::SkipUnknown],
    ..MOCK
};
const SKIP_UNDER_NONE: Program = Program {
    fixture: "mock_invalid_skip_under_none",
    tool_choice: Some(Choice::None),
    hooks: &[Hook::SkipUnknown],
    ending: Ending::UnknownToolCall,
    ..MOCK
};
const RETRY_UNDER_REQUIRED: Program = Program {
    fixture: "mock_invalid_retry_under_required",
    tool_choice: Some(Choice::Required),
    max_turns: Some(2),
    hooks: &[Hook::RetryUnknownTool],
    invalid_retries: 1,
    ending: Ending::MaxTurns,
    ..MOCK
};
const STREAMED_REPAIR: Program = Program {
    fixture: "mock_invalid_streamed_repair",
    hooks: &[Hook::RepairToAdd],
    streamed: true,
    ..MOCK
};
const STREAMED_SKIP: Program = Program {
    fixture: "mock_invalid_streamed_skip",
    hooks: &[Hook::SkipUnknown],
    streamed: true,
    ..MOCK
};

both_interpreters! {
    streamed_retry_once: STREAMED_RETRY_ONCE,
    streamed_retry_twice: STREAMED_RETRY_TWICE,
    ignore_unary: IGNORE_UNARY,
    ignore_streamed: IGNORE_STREAMED,
    mixed_ignore: MIXED_IGNORE,
    mixed_ignore_streamed: MIXED_IGNORE_STREAMED,
    mixed_fail: MIXED_FAIL,
    repair_to_add: REPAIR_TO_ADD,
    skip_under_auto: SKIP_UNDER_AUTO,
    skip_under_none: SKIP_UNDER_NONE,
    retry_under_required: RETRY_UNDER_REQUIRED,
    streamed_repair: STREAMED_REPAIR,
    streamed_skip: STREAMED_SKIP,
}
