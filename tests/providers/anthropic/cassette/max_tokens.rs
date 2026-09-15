//! Claude 5 must run without the caller naming `max_tokens`.
//!
//! Anthropic refuses a Messages request that carries no `max_tokens`, so rig
//! defaults it per model family. The table knew only the 4-series, which made
//! every Claude 5 call fail before it left the process with
//! `RequestError("`max_tokens` must be set for Anthropic")` — a hard error, not
//! retryable, and indistinguishable from a caller mistake. The runtime in
//! `rig-ecs` never supplies one of its own (`MaxTokens` is an optional agent
//! setting), so this reached every ECS run against a Claude 5 model.
//!
//! The recording is the proof the wire agrees: the recorded request body
//! carries the default rig chose, and the model answered it.
//!
//! Recorded through OpenRouter's Anthropic Messages endpoint rather than
//! `api.anthropic.com` only because the recording key's Anthropic workspace is
//! over its usage limit; the code path under test is the same Anthropic client,
//! and the vendor-prefixed id (`anthropic/claude-opus-5`) additionally pins
//! that the family table sees through a gateway prefix.

use rig::prelude::*;

use super::super::support::with_anthropic_gateway_cassette;
use crate::support::assert_nonempty_response;

#[tokio::test]
async fn claude_5_answers_without_a_caller_supplied_max_tokens() {
    with_anthropic_gateway_cassette(
        "max_tokens/claude_5_default_max_tokens",
        |client| async move {
            // No `.max_tokens(..)`: the default for the family is the only
            // thing that can put the field on the wire.
            let agent = client.agent("anthropic/claude-opus-5").build();

            let response = agent
                .prompt("Reply with the single word: ok")
                .await
                .expect("Claude 5 should answer without a caller-supplied max_tokens");

            assert_nonempty_response(&response.to_string());
        },
    )
    .await;
}

/// The rig-ecs run that the defect actually broke: no `MaxTokens` on the
/// agent, a Claude 5 model, and the run must reach dispatch and settle. On
/// `main` this fails `Failure::Provider` with
/// `RequestError("`max_tokens` must be set for Anthropic")` before any
/// request leaves the process.
#[tokio::test]
async fn a_native_ecs_run_reaches_claude_5_without_a_max_tokens_setting() {
    with_anthropic_gateway_cassette("max_tokens/claude_5_ecs_run", |client| async move {
        let mut ecs = crate::ecs_agent::EcsAgent::new(
            client.completion_model("anthropic/claude-opus-5"),
            "You are a concise assistant. Answer directly.",
            1,
        );
        let answer = ecs.prompt("Reply with the single word: ok", false).await;
        assert_nonempty_response(&answer);
    })
    .await;
}
