//! The llama.cpp tool-calling matrix.
//!
//! Cells run on the competent tier, `unsloth/Qwen3-8B-GGUF` Q4_K_M with
//! `--jinja --seed 42 --temp 0 -c 8192`, unless a cell says otherwise. A
//! 1.7B model declines tool calls often enough that "the model refused" and
//! "rig dropped the tools" would look identical.
//!
//! A `tool_choice` naming a specific function is not representable on this
//! wire: `llama-server` reads the field as a string from the vocabulary
//! `auto | none | required`, so an OpenAI-shaped object falls through to
//! `auto`. The `BodyRewrite::LlamaCpp` arm of `Chat::encode` refuses it
//! locally instead, on the streaming path too
//! ([`tool_choice_specific_is_refused_on_the_streaming_path_too`]). Under `tool_choice: "none"` llama.cpp keeps the tool
//! definitions in the prompt but switches off its tool-call parser, so a
//! model that calls one anyway emits its template's raw syntax as assistant
//! text, and rig surfaces it unchanged.

use rig::message::ToolChoice;
use rig::providers::openai::wire::{LLAMACPP, OpenAIConfig};

use crate::support::{Adder, Subtract};

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

const NO_THINK: &str = "/no_think ";

// ---------------------------------------------------------------------------
// tool_choice
// ---------------------------------------------------------------------------

/// The refusal applies on the **streaming** transport too.
///
/// `Chat::prepare` runs from `Chat::encode`, which both `Mode::Unary` and
/// `Mode::Stream` go through, so a caller moving from `completion` to `stream`
/// is refused the same way.
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
