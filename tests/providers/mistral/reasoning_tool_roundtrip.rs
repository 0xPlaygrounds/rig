//! Mistral reasoning-enabled tool roundtrip tests.
//!
//! The other half of the cross-provider reasoning contract: an agent loop on a
//! reasoning model, where every turn after the first replays the previous
//! turn's assistant message — trace included — back to the provider. Before
//! the fix Mistral had no entry here at all, because it produced no reasoning:
//! the non-streaming cell's history assertion requires a reasoning block and
//! fails outright on `origin/main`.
//!
//! Re-recording note: this prompt is **not reliably convergent** on
//! `mistral-small-latest` at `reasoning_effort: high`. Roughly half the live
//! runs observed here re-issued `get_weather("Tokyo")` until the turn budget
//! ran out — with the trace replayed, so it is the model and the prompt rather
//! than anything rig sends, and the streamed twin converged on an attempt the
//! blocking one did not. The fixtures pin converging runs; a re-record may
//! need a second attempt. Nothing in this file asserts that a *traceless*
//! replay behaves differently, because that direction did not reproduce.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::{Chat, Message};
use rig::prelude::*;
use rig::providers::mistral;
use rig::streaming::StreamingChat;

use super::support::with_mistral_cassette;
use crate::reasoning::{self, WeatherTool};

fn reasoning_params() -> serde_json::Value {
    serde_json::json!({ "reasoning_effort": "high" })
}

#[tokio::test]
async fn streaming() {
    with_mistral_cassette("reasoning_tool_roundtrip/streaming", |client| async move {
        let call_count = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent(mistral::MISTRAL_SMALL)
            .preamble(reasoning::TOOL_SYSTEM_PROMPT)
            .max_tokens(2048)
            .tool(WeatherTool::new(call_count.clone()))
            .additional_params(reasoning_params())
            .build();

        // Five, not the three the other providers' cells use: a reasoning turn
        // spends a turn thinking before it calls, and Mistral hit the limit at
        // three while converging normally at five.
        let stream = agent
            .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
            .max_turns(5)
            .await;

        let stats = reasoning::collect_stream_stats(stream, "mistral").await;
        reasoning::assert_universal(&stats, &call_count, "mistral");

        // Deltas, not a complete block: the chat-completions wire never signs
        // or closes a reasoning part, so `MintedReasoningLifecycle` emits
        // `ReasoningDelta`s and a bare end that `StreamingCompletionResponse`
        // suppresses — the same reason every other chat-completions provider's
        // cell guards this check rather than asserting a block outright.
        // Before the fix Mistral produced *neither*.
        assert!(
            stats.reasoning_delta_count > 0 || stats.reasoning_block_count > 0,
            "[mistral] a `reasoning_effort: high` turn streams a thinking chunk; \
             joining only the text parts of `delta.content` reported none. Events: {:?}",
            stats.events
        );
        if stats.reasoning_block_count > 0 {
            assert!(
                stats.reasoning_content_types.contains(&"Text"),
                "[mistral] Mistral's trace is plain text. Got: {:?}",
                stats.reasoning_content_types
            );
        }
    })
    .await;
}

#[tokio::test]
async fn nonstreaming() {
    with_mistral_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(mistral::MISTRAL_SMALL)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(2048)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(reasoning_params())
                .default_max_turns(5)
                .build();

            let mut chat_history = Vec::<Message>::new();
            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut chat_history)
                .await
                .expect("[mistral] Non-streaming chat failed - likely a rejected reasoning replay");

            reasoning::assert_nonstreaming_universal(&result, &call_count, "mistral");
            reasoning::assert_chat_history_preserves_reasoning_tool_roundtrip(
                &chat_history,
                &result,
                "mistral",
            );
        },
    )
    .await;
}
