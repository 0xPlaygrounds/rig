//! Live regression cassettes for shipped OpenAI fixes whose premise was
//! previously unrecorded.
//!
//! See `many_rigs/rig-regression-cassette-suite-proposal.md` for the catalogue.

use rig::completion::FinishReason;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::StreamingPrompt;

use super::super::support::with_openai_completions_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, collect_stream_final_response_and_provider_final,
};

/// B8 — Regression: chat-completions-compatible streaming surfaces
/// `finish_reason` (issue #2248).
///
/// The chat-compat adapter dropped the terminal's finish reason, so a truncated
/// turn and a completed one were indistinguishable downstream — the same defect
/// issue #2235 described on the Anthropic wire, on a different adapter. Recorded
/// against the chat-completions surface specifically, because the Responses
/// surface has its own terminal handling and would not witness this.
///
/// `max_tokens` is capped hard so the terminal must report truncation; a fix
/// that hardcoded `Stop`, or dropped the field again, fails here.
#[tokio::test]
async fn chat_completions_streaming_surfaces_finish_reason() {
    with_openai_completions_cassette(
        "regression/chat_compat_finish_reason",
        |client| async move {
            let agent = client
                .completion_model(openai::GPT_4O)
                .into_agent_builder()
                .preamble(STREAMING_PREAMBLE)
                .max_tokens(8)
                .build();

            let mut stream = agent
                .stream_prompt("Write a detailed five paragraph essay about the ocean.")
                .await;
            let (_response, provider_final): (_, rig::streaming::StreamFinal) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming prompt should succeed");

            assert_eq!(
                provider_final.finish_reason,
                Some(FinishReason::Length),
                "chat-completions streaming must surface the terminal's finish_reason"
            );
        },
    )
    .await;

    // The counterpart, in one test so the pair cannot drift apart: a turn that
    // ends on its own must NOT report truncation. Without this, an adapter that
    // hardcoded `Length` would keep the assertion above green.
    with_openai_completions_cassette(
        "regression/chat_compat_finish_reason_natural",
        |client| async move {
            let agent = client
                .completion_model(openai::GPT_4O)
                .into_agent_builder()
                .preamble(STREAMING_PREAMBLE)
                .max_tokens(512)
                .build();

            let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
            let (_response, provider_final): (_, rig::streaming::StreamFinal) =
                collect_stream_final_response_and_provider_final(&mut stream)
                    .await
                    .expect("streaming prompt should succeed");

            assert_eq!(
                provider_final.finish_reason,
                Some(FinishReason::Stop),
                "a turn that ended on its own must not report truncation"
            );
        },
    )
    .await;
}
