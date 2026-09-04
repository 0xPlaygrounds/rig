//! Matrix J: memory operations.
//!
//! A run with a conversation loads its history before the first
//! completion and appends the run's messages after the answer; a hook
//! can clear the conversation through the run's memory handle; explicit
//! runner history bypasses both; a store can refuse. The corpus's memory
//! rows were one load and one append; this matrix records the clear at
//! both ends, two runs over one conversation in one log, the bypass, the
//! store over a host's bus, serial serving beside two tool calls, and
//! each failure.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | clear | none · `on_run_start` (after `Load`, before the completion) · `on_run_settled` (after `Append`) |
//! | runs in the log | one · two |
//! | history | memory's · explicit (bypass) |
//! | bus | own · a host's |
//! | serving | concurrent · serial, with two tool calls |
//! | failure | none · `Load` · `Append` |
//! | medium | unary · streamed with events |
//!
//! Full cross-product: 3 × 2 × 2 × 2 × 2 × 3 × 2 = 288. Recorded: the 12
//! cells below. Pruned: a failing `Clear` (a hook's dispatch, the hook's
//! to handle; no run path consults it); the bypass under a clear or over
//! two runs (no memory op reaches the log to order); the host's bus under
//! every axis but the plain run (the bus does not change the ops, Matrix
//! C pins ownership); a failing `Load` on the wire (the run fails before
//! any completion, so the mock is the cell); the streamed twins of the
//! clears, the bypass and the host (the medium changes the completion's
//! events, not the memory ops around it).
//!
//! # Cells
//!
//! | golden | producer | shape | ending |
//! |---|---|---|---|
//! | `anthropic_memory_clear_at_start` | anthropic `corpus_memory.rs` `clear_at_start_…` | `[Load(0), Clear, Completion, Append]` | answer |
//! | `anthropic_memory_clear_at_settled` | `clear_at_settled_…` | `[Load, Completion, Append, Clear]` | answer |
//! | `anthropic_memory_two_runs` | `two_runs_…` | `[Load(0), Completion, Append, Load(2), Completion, Append]` | answer |
//! | `anthropic_memory_two_runs_streamed` | `two_runs_streamed_…` | the same, events kept | answer |
//! | `anthropic_memory_clear_at_settled_two_runs` | `clear_at_settled_two_runs_…` | `[Load(0), C, Append, Clear, Load(0), C, Append, Clear]` | answer |
//! | `anthropic_memory_clear_at_start_two_runs` | `clear_at_start_two_runs_…` | `[Load(0), Clear, C, Append, Load(2), Clear, C, Append]` | answer |
//! | `anthropic_memory_history_bypass` | `history_bypass_…` | `[Completion]`; memory in the row, untouched | answer |
//! | `anthropic_memory_host_bus` | `host_bus_memory_…` | `[Load, Completion, Append]`, `bus: None` | answer |
//! | `anthropic_memory_serial_two_tools` | `serial_two_tools_…` | `[Load, C, Tool, Tool, C, Append]`, events kept | answer |
//! | `anthropic_memory_failing_append` | `failing_append_…` | `[Load, Completion, Append(err)]` | answer |
//! | `anthropic_memory_failing_append_streamed` | `failing_append_streamed_…` | the same, events kept | answer |
//! | `mock_memory_failing_load` | `tests/core/golden_memory.rs` `memory_failing_load_…` | `[Load(err)]` | `MemoryError` |
//!
//! # What the matrix found
//!
//! - `on_run_start` fires after the memory load: the runner resolves the
//!   conversation's history before the engine starts the run, so a clear
//!   from that hook empties the store the run has already read, and the
//!   run appends onto the emptied store. A second run then loads the
//!   first run's append and clears it in turn. The hook's contract
//!   ("before the run's first model call") holds; "before the load" is
//!   not part of it, and the log pins where the clear lands.
//! - Only the builder names a run's memory: `memory` / `memory_handler`
//!   register the store under `<owner>/memory` and set the key the run
//!   loads through. A host that registers a store under that key itself
//!   leaves the agent without one (`conversation` alone dispatches
//!   nothing), so the host-bus cell registers through the builder, onto
//!   the host's registrar.
//! - Explicit history leaves memory in the required row: the builder
//!   registered it, the run did not touch it.
//! - A failed `Append` is a warning: the record holds the store's error
//!   and the answer stands. A failed `Load` fails the run at the memory
//!   record, before any completion.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{CONVERSATION, Ending, Hook, Program};
use rig_core::effect::{EffectKind, MemoryOp, MemoryOutcome, Outcome};
use rig_core::message::Message;

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const TWO_TOOL_STREAM_PREAMBLE: &str = "\
You are a precise assistant. When tools are available, you must use them instead of guessing. \
Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. \
Never call the same tool twice once you already have its result.";
const TWO_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
After both tool results are available, stop calling tools and respond in one short sentence that includes both exact tool outputs.";
const PROMPT: &str = "Reply with the single word: ready.";
const SECOND_PROMPT: &str = "Now reply with the single word: again.";
const NAME_PROMPT: &str = "What is my name? Reply with just the name.";

fn bypass_history() -> Vec<Message> {
    vec![
        Message::user("My name is Ada."),
        Message::assistant("Hello, Ada."),
    ]
}

const MEMORY: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    conversation: Some(CONVERSATION),
    ..Program::DEFAULT
};

const CLEAR_AT_START: Program = Program {
    fixture: "anthropic_memory_clear_at_start",
    hooks: &[Hook::ClearAtStart],
    ..MEMORY
};
const CLEAR_AT_SETTLED: Program = Program {
    fixture: "anthropic_memory_clear_at_settled",
    hooks: &[Hook::ClearAtSettled],
    ..MEMORY
};
const TWO_RUNS: Program = Program {
    fixture: "anthropic_memory_two_runs",
    second_prompt: Some(SECOND_PROMPT),
    ..MEMORY
};
const TWO_RUNS_STREAMED: Program = Program {
    fixture: "anthropic_memory_two_runs_streamed",
    second_prompt: Some(SECOND_PROMPT),
    max_turns: Some(8),
    streamed: true,
    ..MEMORY
};
const CLEAR_AT_SETTLED_TWO_RUNS: Program = Program {
    fixture: "anthropic_memory_clear_at_settled_two_runs",
    hooks: &[Hook::ClearAtSettled],
    second_prompt: Some(SECOND_PROMPT),
    ..MEMORY
};
const CLEAR_AT_START_TWO_RUNS: Program = Program {
    fixture: "anthropic_memory_clear_at_start_two_runs",
    hooks: &[Hook::ClearAtStart],
    second_prompt: Some(SECOND_PROMPT),
    ..MEMORY
};
const HISTORY_BYPASS: Program = Program {
    fixture: "anthropic_memory_history_bypass",
    prompt: NAME_PROMPT,
    history: Some(bypass_history),
    max_turns: None,
    ..MEMORY
};
const HOST_BUS: Program = Program {
    fixture: "anthropic_memory_host_bus",
    max_turns: None,
    ..MEMORY
};
const SERIAL_TWO_TOOLS: Program = Program {
    fixture: "anthropic_memory_serial_two_tools",
    preamble: Some(TWO_TOOL_STREAM_PREAMBLE),
    prompt: TWO_TOOL_STREAM_PROMPT,
    max_turns: Some(8),
    streamed: true,
    ..MEMORY
};
const FAILING_APPEND: Program = Program {
    fixture: "anthropic_memory_failing_append",
    ..MEMORY
};
const FAILING_APPEND_STREAMED: Program = Program {
    fixture: "anthropic_memory_failing_append_streamed",
    max_turns: Some(8),
    streamed: true,
    ..MEMORY
};
const FAILING_LOAD: Program = Program {
    fixture: "mock_memory_failing_load",
    temperature: None,
    max_turns: None,
    ending: Ending::MemoryError,
    ..MEMORY
};

both_interpreters! {
    clear_at_start: CLEAR_AT_START,
    clear_at_settled: CLEAR_AT_SETTLED,
    two_runs: TWO_RUNS,
    two_runs_streamed: TWO_RUNS_STREAMED,
    clear_at_settled_two_runs: CLEAR_AT_SETTLED_TWO_RUNS,
    clear_at_start_two_runs: CLEAR_AT_START_TWO_RUNS,
    history_bypass: HISTORY_BYPASS,
    host_bus: HOST_BUS,
    serial_two_tools: SERIAL_TWO_TOOLS,
    failing_append: FAILING_APPEND,
    failing_append_streamed: FAILING_APPEND_STREAMED,
    failing_load: FAILING_LOAD,
}

fn loaded_lengths(log: &rig_effect_log::EffectLog) -> Vec<usize> {
    log.iter()
        .filter_map(|record| match (&record.kind, &record.outcome) {
            (
                EffectKind::Memory {
                    op: MemoryOp::Load { .. },
                },
                Ok(Outcome::Memory(MemoryOutcome::Loaded { messages })),
            ) => Some(messages.len()),
            _ => None,
        })
        .collect()
}

/// The second run's load holds the first run's append; a clear after the
/// append leaves the next load empty, a clear at run start comes after
/// the load and does not.
#[test]
fn a_second_load_holds_the_first_append_unless_cleared_after_it() {
    assert_eq!(loaded_lengths(&corpus::golden(TWO_RUNS.fixture)), [0, 2]);
    assert_eq!(
        loaded_lengths(&corpus::golden(CLEAR_AT_SETTLED_TWO_RUNS.fixture)),
        [0, 0]
    );
    assert_eq!(
        loaded_lengths(&corpus::golden(CLEAR_AT_START_TWO_RUNS.fixture)),
        [0, 2]
    );
}
