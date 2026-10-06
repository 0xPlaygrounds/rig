//! Edge matrix for the output-token-cap spelling on OpenAI Chat Completions.
//!
//! **Bug.** Rig always sent the cap as the legacy `max_tokens` field. OpenAI
//! deprecated that spelling for `/v1/chat/completions` and its reasoning-class
//! models reject it outright:
//!
//! ```text
//! Unsupported parameter: 'max_tokens' is not supported with this model.
//! Use 'max_completion_tokens' instead.
//! ```
//!
//! So `agent.max_tokens(n)` — or any capped request — could not succeed at all
//! against `gpt-5*` / `o*` through the Chat Completions surface. The fix sends
//! `max_completion_tokens` **only for the models that reject the legacy field**
//! (`OpenAICompatibleProvider::requires_modern_output_cap`, implemented for
//! OpenAI as the model catalog's reasoning entries: the `gpt-5`-and-up and
//! `o`-series families). Scoping it to those models rather than to the whole endpoint is
//! deliberate: this same extension is how rig reaches OpenAI-*compatible*
//! servers (mistral.rs, vLLM, llama.cpp, gateways), and OpenAI's own older
//! models still take `max_tokens` — so nothing that worked before changes a
//! byte, which the untouched `mistralrs` suite and cell 12 below prove.
//!
//! **How these cells fail on `origin/main`.** The cassette harness matches the
//! recorded *request body*, so every **reasoning-model** cell below is a mock
//! miss on `main` (which sends `max_tokens`) and passes with the fix. The
//! non-reasoning cells are controls: their recorded bodies are byte-identical
//! on both sides, which is exactly the claim they exist to make.
//!
//! | # | cell | model | class | transport | level | cap | status |
//! |---|------|-------|-------|-----------|-----|-----|--------|
//! | 8 | `reasoning_gpt5_nano_tool_turn_streaming_cap` | gpt-5-nano | reasoning | streaming | agent + tool | set | recorded |
//! | 12 | `legacy_gpt_3_5_turbo_blocking_cap` | gpt-3.5-turbo | oldest (control) | blocking | raw model | set | recorded |
//!
//! Unit cells (the field-spelling rule itself is definitory — no live turn can
//! observe a body rig did not send) live beside the fix in
//! `crates/rig-core/src/providers/openai/completion/mod.rs`:
//! `request_body_*` and `modern_output_cap_*`. They cover the legacy spelling,
//! the modern one, absent caps, caller-supplied spellings on both keys, that no
//! other request field moves, and the base-URL gate itself — including that an
//! OpenAI-compatible server reached through this extension keeps `max_tokens`,
//! which the untouched `mistralrs`, `vllm`, and `doubleword` suites replay as
//! their own proof.

use super::super::support::with_openai_max_tokens_cassette;
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
// Reasoning-class models: these could not be capped at all before the fix.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reasoning_gpt5_nano_tool_turn_streaming_cap() {
    with_openai_max_tokens_cassette(
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
// Non-reasoning models: controls. These still take the legacy field, so their
// recorded request bodies must be byte-identical to what `main` sends.
// ---------------------------------------------------------------------------

/// The oldest chat model rig names: the far end of the untouched set.
#[tokio::test]
async fn legacy_gpt_3_5_turbo_blocking_cap() {
    with_openai_max_tokens_cassette(
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
