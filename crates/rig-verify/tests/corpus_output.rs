//! Matrix H: output modes.
//!
//! `output_mode` lands in the spec, so each mode is its own program hash,
//! and each puts the schema somewhere else in the request: `Native` as
//! the provider's structured-output constraint (Matrix E's schema cells,
//! the baseline), `Tool` as a synthetic `final_result` tool whose call
//! the run settles without a dispatch (and whose absence or incomplete
//! arguments the run reprompts), `Prompted` as an instruction in the
//! preamble. The output tool's name is minted from the executable tool
//! set deterministically (`final_result`, or a numbered name on a
//! collision), so the record pins it.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | mode | `Native` (Matrix E) · `Tool` · `Prompted` |
//! | medium | unary · streamed with events |
//! | beside a real tool | no · yes (`add` first, then the answer) |
//! | `tool_choice` | Auto · Specific(output tool) · Required · None |
//! | reprompt | none · a text answer where the call was due · a call missing a required field |
//! | wire | anthropic · openai (dual-id call) · mock |
//! | `additional_params` | none · extended thinking |
//!
//! Full cross-product: 3 × 2 × 2 × 4 × 3 × 3 × 2 = 864. Recorded: the 14
//! cells below. Pruned: `Prompted` under any `tool_choice` (the mode adds
//! no tool, so the choice binds only the real tools Matrix E's choice
//! cells already pin); a `Prompted`
//! answer that violates the schema (the run does not validate a prompted
//! or native answer, the consumer's deserialization does — only `Tool`
//! mode reprompts); `Native` cells (Matrix E); the streamed twins of the
//! tool-choice and thinking cells (the stream changes the record's events,
//! not the mode's request shape, which the unary cell pins); openai beside
//! a real tool and streamed (the dual-id shape is pinned by the unary
//! cells and by Matrix C).
//!
//! # Cells
//!
//! | golden | producer | shape |
//! |---|---|---|
//! | `anthropic_output_tool_unary` | anthropic `corpus_output.rs` `tool_unary_…` | `[Completion]`; `final_result` advertised, preamble augmented, no native schema |
//! | `anthropic_output_tool_streamed` | `tool_streamed_…` | the same, events kept |
//! | `anthropic_output_prompted_unary` | `prompted_unary_…` | `[Completion]`; no tool, no native schema, the schema in the preamble |
//! | `anthropic_output_prompted_streamed` | `prompted_streamed_…` | the same, events kept |
//! | `anthropic_output_tool_with_real_tool` | `tool_with_real_tool_…` | `[Completion, Tool, Completion]`; `add` dispatched, `final_result` settled |
//! | `anthropic_output_prompted_with_real_tool` | `prompted_with_real_tool_…` | `[Completion, Tool, Completion]` |
//! | `anthropic_output_tool_choice_specific_output` | `tool_choice_specific_output_…` | `[Completion]`; the choice names the output tool |
//! | `anthropic_output_tool_choice_required` | `tool_choice_required_…` | `[Completion]`; the forced call is the output tool's and settles the run |
//! | `anthropic_output_tool_under_none_degrades` | `tool_under_none_degrades_…` | `[Completion]`; `Tool` resolves to `Native` under `tool_choice: none` |
//! | `anthropic_output_tool_thinking` | `tool_thinking_…` | `[Completion]`, a reasoning block and the call |
//! | `openai_output_tool_unary` | openai `corpus_output.rs` `tool_unary_…` | `[Completion]` |
//! | `openai_output_prompted_unary` | `prompted_unary_…` | `[Completion]` |
//! | `mock_output_tool_text_reprompt` | `tests/core/golden_output.rs` `output_tool_text_reprompt_…` | `[Completion, Completion]`: a text answer reprompted, then the call |
//! | `mock_output_tool_missing_field_reprompt` | `output_tool_missing_field_reprompt_…` | `[Completion, Completion]`: a call missing `summary` reprompted |
//!
//! # What the matrix found
//!
//! - `output_mode(Tool)` with `tool_choice: Specific(["final_result"])`
//!   refused the request: the mode resolution ran before the output
//!   tool's name was picked and treated every `Specific` set as
//!   forbidding the call, so the mode degraded to Native, no output tool
//!   was advertised, and the choice then named a tool that was not there
//!   ("requested tool names not advertised this turn"). The name is now
//!   picked first and the resolution asks whether the choice can call it
//!   (`rig-agent`); the cell records the choice honoured.
//! - `Prompted` mode quoted the schema into the preamble through
//!   `serde_json::to_string`, whose key order follows the building crate's
//!   `preserve_order` feature: the producer wrote one text and the
//!   interpreters another, so every prompted cell diverged at the system
//!   message — and, on the live wire, the same program's prompt-cache
//!   prefix would differ by build. The schema is now quoted in a
//!   canonical, key-sorted rendering (`rig-core::json_utils`).

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::{Choice, Output, Program};

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const STRUCTURED_OUTPUT_PROMPT: &str =
    "Return a concise event object for a local Rust meetup in Seattle.";
const SUM_EVENT_PROMPT: &str = "Use the add tool to add 17 and 25, then return a concise event object for a Rust meetup in Seattle whose summary states the sum.";
const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;

fn event_schema() -> serde_json::Value {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal parses")
}

fn thinking_params() -> serde_json::Value {
    serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } })
}

const SCHEMA: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: STRUCTURED_OUTPUT_PROMPT,
    temperature: Some(0.0),
    output_schema: Some(event_schema),
    ..Program::DEFAULT
};
const WITH_TOOL: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: SUM_EVENT_PROMPT,
    max_turns: Some(3),
    ..SCHEMA
};

const TOOL_UNARY: Program = Program {
    fixture: "anthropic_output_tool_unary",
    output_mode: Some(Output::Tool),
    ..SCHEMA
};
const TOOL_STREAMED: Program = Program {
    fixture: "anthropic_output_tool_streamed",
    output_mode: Some(Output::Tool),
    streamed: true,
    ..SCHEMA
};
const PROMPTED_UNARY: Program = Program {
    fixture: "anthropic_output_prompted_unary",
    output_mode: Some(Output::Prompted),
    ..SCHEMA
};
const PROMPTED_STREAMED: Program = Program {
    fixture: "anthropic_output_prompted_streamed",
    output_mode: Some(Output::Prompted),
    streamed: true,
    ..SCHEMA
};
const TOOL_WITH_REAL_TOOL: Program = Program {
    fixture: "anthropic_output_tool_with_real_tool",
    output_mode: Some(Output::Tool),
    ..WITH_TOOL
};
const PROMPTED_WITH_REAL_TOOL: Program = Program {
    fixture: "anthropic_output_prompted_with_real_tool",
    output_mode: Some(Output::Prompted),
    ..WITH_TOOL
};
const TOOL_CHOICE_SPECIFIC_OUTPUT: Program = Program {
    fixture: "anthropic_output_tool_choice_specific_output",
    output_mode: Some(Output::Tool),
    tool_choice: Some(Choice::Specific("final_result")),
    ..SCHEMA
};
const TOOL_CHOICE_REQUIRED: Program = Program {
    fixture: "anthropic_output_tool_choice_required",
    output_mode: Some(Output::Tool),
    tool_choice: Some(Choice::Required),
    max_turns: Some(2),
    ..SCHEMA
};
const TOOL_UNDER_NONE_DEGRADES: Program = Program {
    fixture: "anthropic_output_tool_under_none_degrades",
    output_mode: Some(Output::Tool),
    tool_choice: Some(Choice::None),
    ..SCHEMA
};
const TOOL_THINKING: Program = Program {
    fixture: "anthropic_output_tool_thinking",
    output_mode: Some(Output::Tool),
    additional_params: Some(thinking_params),
    temperature: None,
    ..SCHEMA
};
const OPENAI_TOOL_UNARY: Program = Program {
    fixture: "openai_output_tool_unary",
    output_mode: Some(Output::Tool),
    ..SCHEMA
};
const OPENAI_PROMPTED_UNARY: Program = Program {
    fixture: "openai_output_prompted_unary",
    output_mode: Some(Output::Prompted),
    ..SCHEMA
};
const MOCK_TEXT_REPROMPT: Program = Program {
    fixture: "mock_output_tool_text_reprompt",
    output_mode: Some(Output::Tool),
    temperature: None,
    max_turns: Some(3),
    ..SCHEMA
};
const MOCK_MISSING_FIELD_REPROMPT: Program = Program {
    fixture: "mock_output_tool_missing_field_reprompt",
    output_mode: Some(Output::Tool),
    temperature: None,
    max_turns: Some(3),
    ..SCHEMA
};

both_interpreters! {
    tool_unary: TOOL_UNARY,
    tool_streamed: TOOL_STREAMED,
    prompted_unary: PROMPTED_UNARY,
    prompted_streamed: PROMPTED_STREAMED,
    tool_with_real_tool: TOOL_WITH_REAL_TOOL,
    prompted_with_real_tool: PROMPTED_WITH_REAL_TOOL,
    tool_choice_specific_output: TOOL_CHOICE_SPECIFIC_OUTPUT,
    tool_choice_required: TOOL_CHOICE_REQUIRED,
    tool_under_none_degrades: TOOL_UNDER_NONE_DEGRADES,
    tool_thinking: TOOL_THINKING,
    openai_tool_unary: OPENAI_TOOL_UNARY,
    openai_prompted_unary: OPENAI_PROMPTED_UNARY,
    mock_text_reprompt: MOCK_TEXT_REPROMPT,
    mock_missing_field_reprompt: MOCK_MISSING_FIELD_REPROMPT,
}

/// Each mode is its own program: the three modes over one schema hash to
/// three spec values.
#[test]
fn every_mode_is_its_own_program() {
    let native = corpus::golden("anthropic_request_shape_output_schema_unary");
    let tool = corpus::golden(TOOL_UNARY.fixture);
    let prompted = corpus::golden(PROMPTED_UNARY.fixture);
    let hashes: std::collections::BTreeSet<_> = [native, tool, prompted]
        .iter()
        .map(|log| log.header.run_spec)
        .collect();
    assert_eq!(hashes.len(), 3, "{hashes:?}");
}
