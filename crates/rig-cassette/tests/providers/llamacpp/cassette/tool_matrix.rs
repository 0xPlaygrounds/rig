//! The tool-calling matrix.
//!
//! **Server**: the competent tier — `unsloth/Qwen3-8B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 8192`, `llama-server` b10964-b29c606e2 —
//! unless a cell says otherwise. The escalation is deliberate and is the rule
//! stated in `cassette_support`: a 1.7B model declines tool calls often enough
//! that a red cell would be a coin flip on the model rather than a finding
//! about rig, and "the model refused" and "rig dropped the tools" look
//! identical from the outside. Arity, `tool_choice` and result-shape cells all
//! need the model to actually make the call they are about, so they run here.
//!
//! | Cell | Dimension | Pinned |
//! | --- | --- | --- |
//! | [`tool_choice_specific_is_refused_on_the_streaming_path_too`] | the same, streaming | a guard covering one surface would let the old behaviour back |
//! | tool result carrying an image | result: image | `image_tool_result.rs`, on the vision server |
//!
//! # Two interesting rows
//!
//! `tool_choice` naming a specific function is **not representable** on this
//! wire. `llama-server` reads the field with
//! `json_value(body, "tool_choice", std::string("auto"))` — as a string — and
//! its vocabulary is exactly `auto | none | required`
//! (`common_chat_tool_choice_parse_oaicompat`). An OpenAI-shaped object
//! type-mismatches that read and falls through to the default, so the request
//! is served as `auto` and the caller silently gets whichever tool the model
//! felt like. The `BodyRewrite::LlamaCpp` arm of `Chat::encode` refuses it
//! locally instead.
//!
//! And `tool_choice: "none"`:
//!
//! OpenAI's contract for `none` is "the tools are still described to the model,
//! but do not call one". llama.cpp implements that by leaving the tool
//! definitions in the rendered prompt while switching off the tool-call
//! *parser*, so a model that decides to call one anyway emits its template's
//! raw syntax — `<tool_call>{"name": …}</tool_call>` for Qwen — as ordinary
//! assistant text. Rig surfaces exactly what the server sent, which is right;
//! the cell records the shape so nobody mistakes it for a rig defect later.

use rig::message::ToolChoice;
use rig::providers::openai::wire::{LLAMACPP, OpenAIConfig};

use crate::support::{Adder, OperationArgs, Subtract};

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

const NO_THINK: &str = "/no_think ";

// ---------------------------------------------------------------------------
// Arity
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// tool_choice
// ---------------------------------------------------------------------------

/// The refusal applies on the **streaming** transport too.
///
/// A guard that only covered one surface would be worse than none: a caller
/// who moved from `completion` to `stream` would silently get the old
/// behaviour back. `Chat::prepare` runs from `Chat::encode`, which both
/// `Mode::Unary` and `Mode::Stream` go through, and this cell is what makes
/// that a checked fact rather than a reading.
#[tokio::test]
async fn tool_choice_specific_is_refused_on_the_streaming_path_too() {
    let model = OpenAIConfig::with_key(&LLAMACPP, "")
        .with_base_url("http://127.0.0.1:1/v1")
        .connect(rig_test_support::cassettes::local_reqwest())
        .completion(CASSETTE_MODEL);

    let error = model
        .stream(
            CompletionRequest::new(format!("{NO_THINK}Compute 2 + 3."))
                .tool(rig::tool::tool_definition(&Adder))
                .tool(rig::tool::tool_definition(&Subtract))
                .tool_choice(ToolChoice::Specific {
                    function_names: vec![
                        rig_core::message::ToolName::new("subtract").expect("tool name"),
                    ],
                })
                .max_tokens(256),
        )
        .err()
        .expect("opening the stream must fail before anything is sent");

    assert!(
        matches!(error, rig::error::ProviderError::Request(_)),
        "the refusal must be rig's own, not the connection error the dead \
         address would produce: {error:?}"
    );
    assert!(error.to_string().contains("subtract"), "{error}");
}

// ---------------------------------------------------------------------------
// Tool result payloads
// ---------------------------------------------------------------------------

/// Keeps `OperationArgs` referenced from this file so the shared support type
/// cannot be removed without this matrix noticing.
#[allow(dead_code)]
fn _operation_args_is_the_shared_shape(args: OperationArgs) -> (f64, f64) {
    (args.x as f64, args.y as f64)
}
