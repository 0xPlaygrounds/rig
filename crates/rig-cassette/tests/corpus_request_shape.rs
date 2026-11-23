//! Matrix E: request-shape axes that change the spec hash.
//!
//! Everything on the builder or the runner that lands in a `CompletionRequest`, and so in every
//! record and in the header's `run_spec` hash. Each cell is the `anthropic_completion_smoke`
//! program with one axis changed: `tool_choice`, `max_tokens`, thinking via
//! `additional_params`, the preamble, static `context`, an output schema, or prior runner
//! history. The change is visible in the first record's request as data and refuses the
//! neighbour cell's golden by spec hash (`golden_refusal.rs` proves the refusal mechanism; here
//! the axis is the subject). Multi-axis combinations are not recorded: each axis lands in a
//! distinct request field and the hash covers the whole spec.
//!
//! Every cell is recorded on the real Anthropic wire (`CLAUDE_SONNET_4_6`, temperature 0 except
//! the thinking cells, where Anthropic allows only 1 or unset) into
//! `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_request_shape/`, with producers in
//! `tests/providers/anthropic/cassette/corpus_request_shape.rs`.
//!
//! A per-run `tool_choice` of `Required` or `Specific` applies to every turn, so those runs end
//! in `Ending::MaxTurns` with the model still calling the tool, and the corpus pins that ending.
//! Sonnet 4.6 answers `tool_choice: none` with empty content, which becomes the run's `""`
//! output. The spec hash is over canonical sorted-key JSON, because `serde_json/preserve_order`
//! differs between builds; the thinking and output-schema cells hold multi-key values that
//! would otherwise hash differently. The thinking cells set a budget because adaptive mode
//! skips thinking on a one-line prompt.

use crate::corpus;

use corpus::{Choice, Ending, Program};
use rig_core::message::Message;

/// The root suite's constants, verbatim (`test-support/rig-test-support/src/support.rs`,
/// `test-support/rig-test-support/src/goldens.rs`, the producer module).
const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const BASIC_PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";
const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const CONTEXT_DOCS: [&str; 3] = [
    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets.",
    "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
];
const CONTEXT_PROMPT: &str = "What does \"glarb-glarb\" mean?";
const STRUCTURED_OUTPUT_PROMPT: &str =
    "Return a concise event object for a local Rust meetup in Seattle.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const NO_TOOL_PROMPT: &str = "What is 17 + 25? Reply with just the number.";
const NAME_PROMPT: &str = "What is my name? Reply with just the name.";
const THINKING_PROMPT: &str =
    "Think briefly, then answer: what is 12 * 12? Reply with just the number.";
const APPENDED: &str = "Always end your answer with the word DONE.";
const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;

fn thinking_params() -> serde_json::Value {
    serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } })
}

fn event_schema() -> serde_json::Value {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal parses")
}

fn prior_history() -> Vec<Message> {
    vec![
        Message::user("My name is Ada."),
        Message::assistant("Nice to meet you, Ada."),
    ]
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
    prompt: BASIC_PROMPT,
    temperature: Some(0.0),
    ..Program::DEFAULT
};

const TOOL_CHOICE_AUTO: Program = Program {
    fixture: "anthropic_request_shape_tool_choice_auto",
    tool_choice: Some(Choice::Auto),
    ..TOOLS
};
const TOOL_CHOICE_REQUIRED: Program = Program {
    fixture: "anthropic_request_shape_tool_choice_required",
    tool_choice: Some(Choice::Required),
    max_turns: Some(2),
    ending: Ending::MaxTurns,
    ..TOOLS
};
const TOOL_CHOICE_SPECIFIC: Program = Program {
    fixture: "anthropic_request_shape_tool_choice_specific",
    tool_choice: Some(Choice::Specific("add")),
    max_turns: Some(2),
    ending: Ending::MaxTurns,
    ..TOOLS
};
const TOOL_CHOICE_NONE: Program = Program {
    fixture: "anthropic_request_shape_tool_choice_none",
    preamble: Some(BASIC_PREAMBLE),
    prompt: NO_TOOL_PROMPT,
    tool_choice: Some(Choice::None),
    ..TOOLS
};
const MAX_TOKENS: Program = Program {
    fixture: "anthropic_request_shape_max_tokens",
    max_tokens: Some(32),
    ..BASIC
};
// Anthropic refuses a temperature other than 1 in adaptive thinking mode;
// these two cells leave it unset.
const THINKING_UNARY: Program = Program {
    fixture: "anthropic_request_shape_thinking_unary",
    prompt: THINKING_PROMPT,
    additional_params: Some(thinking_params),
    temperature: None,
    ..BASIC
};
const THINKING_STREAMED: Program = Program {
    fixture: "anthropic_request_shape_thinking_streamed",
    prompt: THINKING_PROMPT,
    additional_params: Some(thinking_params),
    temperature: None,
    streamed: true,
    ..BASIC
};
const STATIC_CONTEXT: Program = Program {
    fixture: "anthropic_request_shape_static_context",
    prompt: CONTEXT_PROMPT,
    context: &[CONTEXT_DOCS[0], CONTEXT_DOCS[1]],
    ..BASIC
};
const APPEND_PREAMBLE: Program = Program {
    fixture: "anthropic_request_shape_append_preamble",
    append_preamble: Some(APPENDED),
    ..BASIC
};
const WITHOUT_PREAMBLE: Program = Program {
    fixture: "anthropic_request_shape_without_preamble",
    preamble: None,
    ..BASIC
};
const OUTPUT_SCHEMA_UNARY: Program = Program {
    fixture: "anthropic_request_shape_output_schema_unary",
    prompt: STRUCTURED_OUTPUT_PROMPT,
    output_schema: Some(event_schema),
    ..BASIC
};
const OUTPUT_SCHEMA_STREAMED: Program = Program {
    fixture: "anthropic_request_shape_output_schema_streamed",
    prompt: STRUCTURED_OUTPUT_PROMPT,
    output_schema: Some(event_schema),
    streamed: true,
    ..BASIC
};
const PRIOR_HISTORY: Program = Program {
    fixture: "anthropic_request_shape_prior_history",
    prompt: NAME_PROMPT,
    history: Some(prior_history),
    ..BASIC
};

both_interpreters! {
    tool_choice_auto: TOOL_CHOICE_AUTO,
    tool_choice_required: TOOL_CHOICE_REQUIRED,
    tool_choice_specific: TOOL_CHOICE_SPECIFIC,
    tool_choice_none: TOOL_CHOICE_NONE,
    max_tokens: MAX_TOKENS,
    thinking_unary: THINKING_UNARY,
    thinking_streamed: THINKING_STREAMED,
    static_context: STATIC_CONTEXT,
    append_preamble: APPEND_PREAMBLE,
    without_preamble: WITHOUT_PREAMBLE,
    output_schema_unary: OUTPUT_SCHEMA_UNARY,
    output_schema_streamed: OUTPUT_SCHEMA_STREAMED,
    prior_history: PRIOR_HISTORY,
}

/// Each cell's spec hash differs from the baseline's: the axis is in the
/// hash, so a golden of one cell refuses the program of another.
#[test]
fn every_cell_has_its_own_spec_hash() {
    let cells = [
        &TOOL_CHOICE_AUTO,
        &TOOL_CHOICE_REQUIRED,
        &TOOL_CHOICE_SPECIFIC,
        &TOOL_CHOICE_NONE,
        &MAX_TOKENS,
        &THINKING_UNARY,
        &THINKING_STREAMED,
        &STATIC_CONTEXT,
        &APPEND_PREAMBLE,
        &WITHOUT_PREAMBLE,
        &OUTPUT_SCHEMA_UNARY,
        &OUTPUT_SCHEMA_STREAMED,
        &PRIOR_HISTORY,
    ];
    let mut seen: std::collections::BTreeMap<u64, Vec<&str>> = Default::default();
    for cell in cells {
        let log = corpus::golden(cell.fixture);
        let hash = log.header.run_spec.expect("an agent recorded the golden");
        seen.entry(hash).or_default().push(cell.fixture);
    }
    // A streamed twin and a prior history share the spec with their unary
    // or history-free sibling: the stream is the runner's, the history is
    // the runner's, neither is in the spec. Every other cell is alone.
    let expected_shared = [
        [
            "anthropic_request_shape_thinking_unary",
            "anthropic_request_shape_thinking_streamed",
        ],
        [
            "anthropic_request_shape_output_schema_unary",
            "anthropic_request_shape_output_schema_streamed",
        ],
    ];
    for (hash, fixtures) in &seen {
        if fixtures.len() > 1 {
            assert!(
                expected_shared.iter().any(|pair| {
                    let mut pair: Vec<&str> = pair.to_vec();
                    pair.sort_unstable();
                    let mut got = fixtures.clone();
                    got.sort_unstable();
                    pair == got
                }),
                "cells share spec hash {hash:#018x} but are not a streamed twin: {fixtures:?}"
            );
        }
    }
}
