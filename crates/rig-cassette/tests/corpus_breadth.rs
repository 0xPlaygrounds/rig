//! Matrix N: provider breadth for the pass-2 shapes.
//!
//! Matrices F to J were recorded on the anthropic wire. A gemini or openai
//! row exists here only where the wire changes the record: the output
//! tool's call under gemini's minted `tool-<n>` ids and openai's dual ids, a
//! delta stop on each wire's stream, a cancelled tool dispatch, a host's
//! custom note beside each wire's tool-call ids, a host embed on gemini's
//! embedding model, and the prompted schema on the Responses wire.

use crate::corpus;

use corpus::{CANCEL_ADD_DISPATCH, Ending, Hook, Output, Program, STOP_ON_TEXT_DELTA};

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const STRUCTURED_OUTPUT_PROMPT: &str =
    "Return a concise event object for a local Rust meetup in Seattle.";
const ESSAY_PROMPT: &str =
    "Write four paragraphs about the history of the Rust programming language.";
const PROMPT: &str = "Reply with the single word: ready.";
const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;

fn event_schema() -> serde_json::Value {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal parses")
}

const BASIC: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: PROMPT,
    temperature: Some(0.0),
    ..Program::DEFAULT
};
const TOOLS: Program = Program {
    preamble: Some(TOOLS_PREAMBLE),
    prompt: ADD_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    ..Program::DEFAULT
};
const OUTPUT_TOOL: Program = Program {
    prompt: STRUCTURED_OUTPUT_PROMPT,
    output_schema: Some(event_schema),
    output_mode: Some(Output::Tool),
    ..BASIC
};
const TEXT_DELTA_STOP: Program = Program {
    prompt: ESSAY_PROMPT,
    streamed: true,
    hooks: &[Hook::StopOnTextDelta],
    ending: Ending::Cancelled(STOP_ON_TEXT_DELTA),
    ..BASIC
};
const DISPATCH_CANCELLED: Program = Program {
    hooks: &[Hook::CancelAddDispatch],
    ending: Ending::Cancelled(CANCEL_ADD_DISPATCH),
    ..TOOLS
};
const CUSTOM_AT_OUTCOME: Program = Program {
    hooks: &[Hook::NoteAtOutcome],
    ..TOOLS
};

const GEMINI_OUTPUT_TOOL_UNARY: Program = Program {
    fixture: "gemini_breadth_output_tool_unary",
    ..OUTPUT_TOOL
};
const GEMINI_OUTPUT_TOOL_STREAMED: Program = Program {
    fixture: "gemini_breadth_output_tool_streamed",
    streamed: true,
    ..OUTPUT_TOOL
};
const GEMINI_TEXT_DELTA_STOP: Program = Program {
    fixture: "gemini_breadth_text_delta_stop",
    ..TEXT_DELTA_STOP
};
const GEMINI_DISPATCH_CANCELLED: Program = Program {
    fixture: "gemini_breadth_tool_dispatch_cancelled",
    ..DISPATCH_CANCELLED
};
const GEMINI_CUSTOM_AT_OUTCOME: Program = Program {
    fixture: "gemini_breadth_custom_at_outcome",
    ..CUSTOM_AT_OUTCOME
};
const GEMINI_EMBED_PROMPT: Program = Program {
    fixture: "gemini_breadth_embed_prompt",
    hooks: &[Hook::EmbedPrompt],
    ..BASIC
};
const OPENAI_OUTPUT_TOOL_STREAMED: Program = Program {
    fixture: "openai_breadth_output_tool_streamed",
    streamed: true,
    ..OUTPUT_TOOL
};
const OPENAI_PROMPTED_STREAMED: Program = Program {
    fixture: "openai_breadth_prompted_streamed",
    streamed: true,
    output_mode: Some(Output::Prompted),
    ..OUTPUT_TOOL
};
const OPENAI_TEXT_DELTA_STOP: Program = Program {
    fixture: "openai_breadth_text_delta_stop",
    ..TEXT_DELTA_STOP
};
const OPENAI_DISPATCH_CANCELLED: Program = Program {
    fixture: "openai_breadth_tool_dispatch_cancelled",
    ..DISPATCH_CANCELLED
};
const OPENAI_CUSTOM_AT_OUTCOME: Program = Program {
    fixture: "openai_breadth_custom_at_outcome",
    ..CUSTOM_AT_OUTCOME
};
both_interpreters! {
    gemini_output_tool_unary: GEMINI_OUTPUT_TOOL_UNARY,
    gemini_output_tool_streamed: GEMINI_OUTPUT_TOOL_STREAMED,
    gemini_text_delta_stop: GEMINI_TEXT_DELTA_STOP,
    gemini_tool_dispatch_cancelled: GEMINI_DISPATCH_CANCELLED,
    gemini_custom_at_outcome: GEMINI_CUSTOM_AT_OUTCOME,
    gemini_embed_prompt: GEMINI_EMBED_PROMPT,
    openai_output_tool_streamed: OPENAI_OUTPUT_TOOL_STREAMED,
    openai_prompted_streamed: OPENAI_PROMPTED_STREAMED,
    openai_text_delta_stop: OPENAI_TEXT_DELTA_STOP,
    openai_tool_dispatch_cancelled: OPENAI_DISPATCH_CANCELLED,
    openai_custom_at_outcome: OPENAI_CUSTOM_AT_OUTCOME,
}

/// The output tool retains its identity origin: Gemini without a wire ID
/// generates a local identity; OpenAI retains an explicit provider correlator.
#[test]
fn the_output_tool_call_carries_the_wires_ids() {
    for (fixture, minted) in [
        (GEMINI_OUTPUT_TOOL_UNARY.fixture, true),
        (OPENAI_OUTPUT_TOOL_STREAMED.fixture, false),
    ] {
        let log = corpus::golden(fixture);
        let call = match &log.records[0].outcome {
            Ok(rig_core::effect::Answered::Completion(response)) => response
                .choice
                .iter()
                .find_map(|content| match content {
                    rig_core::message::AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .expect("the output tool's call"),
            other => panic!("{fixture}: a completion, not {other:?}"),
        };
        assert_eq!(call.function.name, "final_result", "{fixture}");
        assert_eq!(call.id.is_local(), minted, "{fixture}: {:?}", call.id);
    }
}
