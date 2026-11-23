//! The output-token-cap spelling on OpenAI Chat Completions.
//!
//! OpenAI's reasoning-class models (`gpt-5` and up, the `o` series) reject
//! the legacy `max_tokens` field on `/v1/chat/completions`, so rig sends
//! `max_completion_tokens` for exactly the models
//! `OpenAICompatibleProvider::requires_modern_output_cap` names. The scope is
//! per model, not per endpoint, because the same extension reaches
//! OpenAI-compatible servers and OpenAI's older models, which still take
//! `max_tokens`.
//!
//! The cassette harness matches the recorded request body, so the reasoning
//! cell pins the modern spelling and the non-reasoning cell is a control
//! that still sends the legacy one. The spelling rule itself is unit-tested
//! by `request_body_*` and `modern_output_cap_*` in
//! `crates/rig-core/src/providers/openai/completion/mod.rs`.

use super::super::support::with_openai_cassette;
use crate::support::{
    Adder, assert_nonempty_response, assistant_text_response, collect_stream_observation,
};
use rig::completion::CompletionRequest;

/// A cap large enough that a reasoning model still has budget for visible
/// text after its hidden reasoning tokens.
const CAP: u64 = 256;
/// Tool-calling and extraction turns reason more before they emit anything, so
/// they need headroom the plain-answer cells do not — a cap these turns
/// exhaust would test truncation (see `truncated_turn_matrix`) rather than the
/// spelling of the field that carries it.
const TOOL_CAP: u64 = 4096;
const PROMPT: &str = "Reply with the single word OK.";

// ---------------------------------------------------------------------------
// Reasoning-class models: the cap is sent as `max_completion_tokens`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reasoning_gpt5_nano_tool_turn_streaming_cap() {
    with_openai_cassette(
        "max_completion_tokens_matrix/reasoning_gpt5_nano_tool_turn_streaming_cap",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.chat.completion("gpt-5-nano"))
                .preamble("Use the add tool to answer arithmetic questions.")
                .max_tokens(TOOL_CAP)
                .tool(Adder)
                .build();

            let mut stream = agent.prompt("What is 3 + 4?").max_turns(3).stream();
            let observed = collect_stream_observation(&mut stream).await;

            assert!(observed.errors.is_empty(), "{:?}", observed.errors);
            assert_eq!(observed.tool_calls, vec!["add".to_string()]);
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Non-reasoning models: controls. These still take the legacy `max_tokens`
// field.
// ---------------------------------------------------------------------------

/// The oldest chat model rig names: the far end of the untouched set.
#[tokio::test]
async fn legacy_gpt_3_5_turbo_blocking_cap() {
    with_openai_cassette(
        "max_completion_tokens_matrix/legacy_gpt_3_5_turbo_blocking_cap",
        |client| async move {
            let model = client.openai.chat("gpt-3.5-turbo");
            let request = CompletionRequest::new(PROMPT).max_tokens(CAP);

            let response = model
                .call(request)
                .await
                .expect("the oldest chat model rig names keeps the legacy field");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// Boundary: no cap at all, and caller-supplied spellings.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Control: the Responses surface has its own field and must not move.
// ---------------------------------------------------------------------------
