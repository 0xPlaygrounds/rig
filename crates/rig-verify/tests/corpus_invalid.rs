//! Matrix G: invalid tool calls, streamed and ignored.
//!
//! An invalid call — a tool the request did not advertise — is resolved
//! by a hook (`Retry` with feedback, `Repair` to an allowed tool, `Skip`
//! with a reason) or, with no hook, by the runner's policy (`Fail`, the
//! default, or `Ignore`). The corpus's recovery rows were unary retries;
//! this matrix records every resolution on both media and the policy
//! both ways, and the two interpreters must agree on what each resolution
//! puts in the transcript: the hand driver resolves through
//! `AgentRun`'s own steps, the engine through its streamed path.
//!
//! Every cell is scripted from the mock model: no model in the corpus
//! emits a call to a tool that is not in its request (#2449 tried Sonnet
//! 4.6 twice, unary; a stream is no different for a model that will not
//! name an unadvertised tool), so no cassette can hold these shapes.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | resolution | hook `Retry` · hook `Repair` · hook `Skip` · policy `Ignore` · policy `Fail` |
//! | medium | unary · streamed with events |
//! | the turn | one unknown call · an unknown beside a valid call · two unknown turns in a row |
//! | `tool_choice` | Auto · Required · None |
//!
//! Full cross-product: 5 × 2 × 3 × 3 = 90. Recorded: the 13 cells below.
//! Pruned: `Fail` streamed (the failure is the run's, at the completion
//! record, on either medium: the unary cell is the cell); `Retry` under
//! `None` (a forbidden tool set has nothing to retry into: the model must
//! not call at all, and the corpus's `tool_choice: none` cell shows what
//! it does); `Repair` and `Ignore` under `Required` and `None` (the
//! policy and the repair do not consult the choice; the skip does, and is
//! recorded under both); a valid call beside an unknown one under every
//! hook (the hook resolves the unknown one the same way whether or not a
//! valid one is beside it; the `Ignore` pair shows the valid one runs).
//!
//! # Cells
//!
//! | golden | producer (`tests/core/golden_invalid.rs`) | shape | ending |
//! |---|---|---|---|
//! | `mock_invalid_streamed_retry_once` | `invalid_streamed_retry_once_…` | `[Completion, Completion, Tool, Completion]`, events kept | answer |
//! | `mock_invalid_streamed_retry_twice` | `invalid_streamed_retry_twice_…` | `[Completion ×3, Tool, Completion]`, events kept | answer |
//! | `mock_invalid_ignore_unary` | `invalid_ignore_unary_…` | `[Completion]`: the ignored-only turn is the answer, empty | answer `""` |
//! | `mock_invalid_ignore_streamed` | `invalid_ignore_streamed_…` | the same, events kept | answer `""` |
//! | `mock_invalid_mixed_ignore` | `invalid_mixed_ignore_…` | `[Completion, Tool, Completion]`: `add` runs, `multiply` is dropped | answer |
//! | `mock_invalid_mixed_ignore_streamed` | `invalid_mixed_ignore_streamed_…` | the same, events kept | answer |
//! | `mock_invalid_mixed_fail` | `invalid_mixed_fail_…` | `[Completion]`; `add` does not run | `UnknownToolCall` |
//! | `mock_invalid_repair_to_add` | `invalid_repair_to_add_…` | `[Completion, Tool(add), Completion]` | answer |
//! | `mock_invalid_skip_under_auto` | `invalid_skip_under_auto_…` | `[Completion, Completion]`, the reason in the transcript | answer |
//! | `mock_invalid_skip_under_none` | `invalid_skip_under_none_…` | `[Completion]`; the skip is refused under `tool_choice: none` | `UnknownToolCall` |
//! | `mock_invalid_retry_under_required` | `invalid_retry_under_required_…` | `[Completion, Completion, Tool]` | `MaxTurnsError { max_turns: 2 }` |
//! | `mock_invalid_streamed_repair` | `invalid_streamed_repair_…` | `[Completion, Tool, Completion]`, events kept | answer |
//! | `mock_invalid_streamed_skip` | `invalid_streamed_skip_…` | `[Completion, Completion]`, events kept | answer |
//!
//! # What the matrix found
//!
//! - The streaming surface ignored the runner's `UnhandledInvalidToolCall`
//!   policy: with no hook resolving an invalid call it failed the run
//!   whatever the policy said, while the blocking surface honoured
//!   `Ignore`. The streamed path now applies the policy, with an
//!   `Ignored` resolution that drops the call from the turn (`rig-agent`).
//! - An ignored call that was the turn's only content leaves an empty
//!   turn, and the run settles on it as an empty answer; the next
//!   scripted turn is never asked for. Pinned on both media.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

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
