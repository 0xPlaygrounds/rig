//! Matrix K: the delta wire.
//!
//! A tool call that arrives as a name delta and argument deltas — the
//! shape of the openai chat-completions and gemini interactions wires —
//! is not the same turn as a completed block (anthropic, openai
//! responses, gemini generate): the assembler validates the name when it
//! arrives, buffers arguments that arrive before it, surfaces an invalid
//! call at the name delta with the arguments seen so far, and after an
//! `Ignored` resolution keeps the block's state as a tombstone that
//! swallows the rest of the block. Matrix G recorded every resolution on
//! the block shape; this matrix records them on the delta shape, and the
//! two interpreters agree on what each leaves in the transcript: the
//! engine through the streamed path, the hand driver through the
//! assembler and `AgentRun`'s streamed seam.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | the call | valid · unknown name · unknown beside a valid call · the output tool |
//! | argument order | after the name · before the name (buffered) |
//! | resolution | none · `Retry` · `Repair` · `Skip` · policy `Ignore` · policy `Fail` |
//! | stop | none · on the name delta · on an arguments delta |
//! | wire | mock · openai chat-completions (dual id) · gemini interactions (id-less) |
//!
//! Full cross-product: 4 × 2 × 6 × 3 × 3 = 432. Recorded: the 13 cells
//! below. Pruned: every resolution on a live wire (no model in the corpus
//! emits an unadvertised call on request: #2450 tried twice, Matrix G
//! scripted); arguments-first under every resolution but `Retry` (the
//! buffering is the assembler's, decided before the resolution; one cell
//! pins it); the stops beside a resolution (a stop ends the turn before
//! any resolution); the output tool arriving as deltas on a live wire
//! (unrecorded: Matrix N's output-tool rows are on the block-shaped
//! Responses and generate wires; the mock cell pins the assembly);
//! `Skip` and `Repair` beside a valid call (Matrix G's mixed cells pin
//! the pairing; the delta shape changes the surfacing, which the single
//! cells pin).
//!
//! # Cells
//!
//! | golden | producer | shape | ending |
//! |---|---|---|---|
//! | `mock_delta_baseline` | `tests/core/golden_delta.rs` `delta_baseline_…` | `[Completion, Tool, Completion]`, deltas kept | answer |
//! | `mock_delta_retry` | `delta_retry_…` | `[Completion, Completion, Tool, Completion]` | answer |
//! | `mock_delta_retry_arguments_first` | `delta_retry_arguments_first_…` | the same; the arguments buffered before the name | answer |
//! | `mock_delta_repair` | `delta_repair_…` | `[Completion, Tool, Completion]` | answer |
//! | `mock_delta_skip` | `delta_skip_…` | `[Completion, Completion]` | answer |
//! | `mock_delta_ignore` | `delta_ignore_…` | `[Completion]`; the ignored-only turn is the empty answer | answer `""` |
//! | `mock_delta_ignore_beside_valid` | `delta_ignore_beside_valid_…` | `[Completion, Tool, Completion]`; the ignored block swallowed, `add` runs | answer |
//! | `mock_delta_fail` | `delta_fail_…` | `[Completion]` | `UnknownToolCall` |
//! | `mock_delta_output_tool` | `delta_output_tool_…` | `[Completion]`; the output tool assembled from deltas, no dispatch | answer |
//! | `mock_delta_stop_on_name` | `delta_stop_on_name_…` | `[Completion]` | `Cancelled` |
//! | `mock_delta_stop_on_arguments` | `delta_stop_on_arguments_…` | `[Completion]` | `Cancelled` |
//! | `openai_delta_chat_baseline` | openai `corpus_delta.rs` `chat_baseline_…` | `[Completion, Tool, Completion]`, the chat-completions wire's deltas | answer |
//! | `gemini_delta_interactions_baseline` | gemini `corpus_delta.rs` `interactions_baseline_…` | the same on the interactions wire | answer |
//!
//! # What the matrix found
//!
//! - The gemini interactions wire could not round-trip a tool call in
//!   client-managed history: it nested the call and its result inside
//!   `model_output` / `user_input` steps ("an invalid argument" — the API
//!   emits and accepts them as steps of their own), sent a thought
//!   summary untagged ("The 'type' parameter is required"), sent a
//!   signature-only thought with an empty summary item (invalid), and
//!   sent a scalar JSON result as a text block ("Multimodal function
//!   responses are not supported"). Only `previous_interaction_id`
//!   (server-side history) ever round-tripped, which is what the wire's
//!   existing round-trip test used. Four fixes in `rig-core`, each with a
//!   unit test; the gemini cell records the round trip.
//! - The mock could not script the delta wire to completion: the adapter
//!   closes text and reasoning blocks at the final event, never a tool
//!   block, so a delta-built call with no end event yields an empty
//!   choice. `MockStreamEvent::tool_call_end` ends it as a wire's
//!   step-stop does (test support, not a production change: every live
//!   wire ends its calls explicitly).

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{
    Ending, Hook, Output, Program, STOP_ON_TOOL_ARGUMENTS_DELTA, STOP_ON_TOOL_NAME_DELTA, Unhandled,
};

const PREAMBLE: &str = "Use the add tool.";
const PROMPT: &str = "What is 2 + 3?";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const STRUCTURED_OUTPUT_PROMPT: &str =
    "Return a concise event object for a local Rust meetup in Seattle.";
const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;

fn event_schema() -> serde_json::Value {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal parses")
}

const MOCK: Program = Program {
    preamble: Some(PREAMBLE),
    prompt: PROMPT,
    max_turns: Some(3),
    streamed: true,
    ..Program::DEFAULT
};

const BASELINE: Program = Program {
    fixture: "mock_delta_baseline",
    ..MOCK
};
const RETRY: Program = Program {
    fixture: "mock_delta_retry",
    max_turns: Some(4),
    hooks: &[Hook::RetryUnknownTool],
    invalid_retries: 1,
    ..MOCK
};
const RETRY_ARGUMENTS_FIRST: Program = Program {
    fixture: "mock_delta_retry_arguments_first",
    max_turns: Some(4),
    hooks: &[Hook::RetryUnknownTool],
    invalid_retries: 1,
    ..MOCK
};
const REPAIR: Program = Program {
    fixture: "mock_delta_repair",
    hooks: &[Hook::RepairToAdd],
    ..MOCK
};
const SKIP: Program = Program {
    fixture: "mock_delta_skip",
    hooks: &[Hook::SkipUnknown],
    ..MOCK
};
const IGNORE: Program = Program {
    fixture: "mock_delta_ignore",
    unhandled: Unhandled::Ignore,
    ..MOCK
};
const IGNORE_BESIDE_VALID: Program = Program {
    fixture: "mock_delta_ignore_beside_valid",
    unhandled: Unhandled::Ignore,
    ..MOCK
};
const FAIL: Program = Program {
    fixture: "mock_delta_fail",
    ending: Ending::UnknownToolCall,
    ..MOCK
};
const OUTPUT_TOOL: Program = Program {
    fixture: "mock_delta_output_tool",
    preamble: Some(BASIC_PREAMBLE),
    prompt: STRUCTURED_OUTPUT_PROMPT,
    max_turns: None,
    output_schema: Some(event_schema),
    output_mode: Some(Output::Tool),
    ..MOCK
};
const STOP_ON_NAME: Program = Program {
    fixture: "mock_delta_stop_on_name",
    hooks: &[Hook::StopOnToolNameDelta],
    ending: Ending::Cancelled(STOP_ON_TOOL_NAME_DELTA),
    ..MOCK
};
const STOP_ON_ARGUMENTS: Program = Program {
    fixture: "mock_delta_stop_on_arguments",
    hooks: &[Hook::StopOnToolArgumentsDelta],
    ending: Ending::Cancelled(STOP_ON_TOOL_ARGUMENTS_DELTA),
    ..MOCK
};
const OPENAI_BASELINE: Program = Program {
    fixture: "openai_delta_chat_baseline",
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    ..MOCK
};
const GEMINI_BASELINE: Program = Program {
    fixture: "gemini_delta_interactions_baseline",
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    ..MOCK
};

both_interpreters! {
    baseline: BASELINE,
    retry: RETRY,
    retry_arguments_first: RETRY_ARGUMENTS_FIRST,
    repair: REPAIR,
    skip: SKIP,
    ignore: IGNORE,
    ignore_beside_valid: IGNORE_BESIDE_VALID,
    fail: FAIL,
    output_tool: OUTPUT_TOOL,
    stop_on_name: STOP_ON_NAME,
    stop_on_arguments: STOP_ON_ARGUMENTS,
    openai_baseline: OPENAI_BASELINE,
    gemini_baseline: GEMINI_BASELINE,
}

/// The medium is in the record: every cell's first completion keeps a
/// tool-name delta among its events.
#[test]
fn every_cell_streams_the_tool_name_as_a_delta() {
    for fixture in [
        BASELINE.fixture,
        RETRY.fixture,
        RETRY_ARGUMENTS_FIRST.fixture,
        REPAIR.fixture,
        SKIP.fixture,
        IGNORE.fixture,
        IGNORE_BESIDE_VALID.fixture,
        FAIL.fixture,
        OUTPUT_TOOL.fixture,
        STOP_ON_NAME.fixture,
        STOP_ON_ARGUMENTS.fixture,
        OPENAI_BASELINE.fixture,
        GEMINI_BASELINE.fixture,
    ] {
        let log = corpus::golden(fixture);
        let events = log.records[0]
            .events
            .as_ref()
            .unwrap_or_else(|| panic!("{fixture}: events are kept"));
        assert!(
            events.iter().any(|event| matches!(
                event,
                rig_core::streaming::StreamEvent::BlockDelta {
                    delta: rig_core::streaming::Delta::ToolName { .. },
                    ..
                }
            )),
            "{fixture}: a tool name delta is in the record"
        );
    }
}
