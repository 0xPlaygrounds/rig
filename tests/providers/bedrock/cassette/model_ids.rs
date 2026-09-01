//! Bedrock model-identifier regression coverage, recorded from the real API.
//!
//! `rig-bedrock` shipped 39 model constants that Bedrock cannot invoke: 29
//! identifiers retired out of `ListFoundationModels` entirely, and 10 that
//! exist but are servable only through a cross-region inference profile. Both
//! failure modes are recorded here so a future edit that reintroduces a bare
//! or retired identifier fails against real provider responses rather than
//! against a hand-written expectation.

use rig::bedrock;
use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::support::with_bedrock_cassette;
use crate::support::assert_nonempty_response;

/// A retired identifier — every `anthropic.claude-*` constant the crate used
/// to ship was one of these — answers `ResourceNotFoundException`, and the
/// provider's own wording has to survive into the Rig error.
#[tokio::test]
async fn retired_model_id_preserves_provider_error() {
    with_bedrock_cassette(
        "model_ids/retired_model_id_preserves_provider_error",
        |client| async move {
            let model = client.completion_model("anthropic.claude-3-5-sonnet-20240620-v1:0");
            let request = model.completion_request("Say hi.").max_tokens(8).build();

            let error = model
                .completion(request)
                .await
                .expect_err("a retired model id should be a provider error");

            let body = error
                .provider_response_body()
                .expect("provider error body should be preserved");
            assert!(
                body.contains("end of its life"),
                "expected Bedrock's end-of-life wording, got {body:?}"
            );
        },
    )
    .await;
}

/// A model that exists but is profile-only rejects the *bare* identifier. This
/// is what made `DEEPSEEK_R1` and the Llama 3.3/4 constants unusable: the id
/// resolves, so the failure is a validation error naming on-demand throughput
/// rather than a missing resource.
#[tokio::test]
async fn bare_profile_only_model_id_is_rejected() {
    with_bedrock_cassette(
        "model_ids/bare_profile_only_model_id_is_rejected",
        |client| async move {
            let model = client.completion_model("deepseek.r1-v1:0");
            let request = model.completion_request("Say hi.").max_tokens(8).build();

            let error = model
                .completion(request)
                .await
                .expect_err("a bare profile-only model id should be a provider error");

            let body = error
                .provider_response_body()
                .expect("provider error body should be preserved");
            assert!(
                body.contains("on-demand throughput"),
                "expected Bedrock's on-demand-throughput wording, got {body:?}"
            );
        },
    )
    .await;
}

/// The replacement form works: the same model invoked through its cross-region
/// inference profile completes normally.
#[tokio::test]
async fn cross_region_profile_id_completes() {
    with_bedrock_cassette(
        "model_ids/cross_region_profile_id_completes",
        |client| async move {
            let model = client.completion_model(bedrock::completion::DEEPSEEK_R1);
            // DeepSeek R1 reasons before answering: recorded at 64 tokens the
            // whole budget went to `reasoningContent` and the turn stopped at
            // `max_tokens` with no text block at all.
            // DeepSeek R1 reasons before answering: recorded at 64 tokens the
            // whole budget went to `reasoningContent` and the turn stopped at
            // `max_tokens` with no text block at all.
            let request = model
                .completion_request("Reply with the single word: ready.")
                .max_tokens(512)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("cross-region profile completion should succeed");

            let text = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    rig::completion::AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            assert_nonempty_response(&text);
        },
    )
    .await;
}

/// Anthropic on Bedrock is the flagship pairing and the crate had no working
/// constant for it; this pins that the replacement Claude identifier is
/// invocable end-to-end.
#[tokio::test]
async fn claude_profile_constant_completes() {
    with_bedrock_cassette(
        "model_ids/claude_profile_constant_completes",
        |client| async move {
            let agent = client
                .agent(bedrock::completion::ANTHROPIC_CLAUDE_HAIKU_4_5)
                .preamble("You are concise.")
                .build();

            let response = agent
                .prompt("Reply with the single word: ready.")
                .await
                .expect("Claude completion should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
